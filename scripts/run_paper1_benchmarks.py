#!/usr/bin/env python3
"""Evaluate frozen Paper 1 checkpoints on Marmousi2 and SEG/EAGE Overthrust sections."""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from fwi_grad.data.dataset_builder import make_experiment
from fwi_grad.eval import plotting
from fwi_grad.eval.metrics import gradient_metrics, uncertainty_calibration_metrics, velocity_metrics
from fwi_grad.inversion.guided_fwi import GuidedFWIConfig, _predict
from fwi_grad.losses.physics_losses import directional_derivative_relative_error
from fwi_grad.physics.geometry import select_sparse_shots
from scripts.run_paper1_ablation import LOOP_METRICS, _benchmark_method, _load_model, _tex_number
from scripts.train_stage2 import _write_json


GUIDED = {"A3": "A3_guided", "A5": "A5_proposed_guided"}
METHODS = {"full_adjoint": "Full adjoint", "sparse_adjoint": "Sparse adjoint",
           "A3_guided": "A3 guided", "A5_proposed_guided": "A5 guided"}
FIGURES = ("A3_gradient_uncertainty", "A5_gradient_uncertainty", "closed_loop_convergence", "velocity_and_profile")
OPEN_METRICS = {"cosine_similarity": "Cosine similarity", "rel_l2_error": r"Relative $L_2$ error",
                "directional_derivative_relative_error": "Directional relative error", "spearman_r": "Spearman $r$"}
LOOP_COLUMNS = {**LOOP_METRICS, "grad_time_reduction_pct": r"Gradient time reduction (\%)",
                "iter_time_total": "Total iteration time (s)"}


def _open_loop(initial, obs, common, cfg, networks, shots, seed, directory):
    solver, wavelet, geom = common
    full = solver.adjoint_gradient(initial, obs, wavelet, geom, f_cut=cfg.f_cut)
    sparse = solver.adjoint_gradient(initial, obs, wavelet, geom, shots, f_cut=cfg.f_cut)
    full_shots = list(range(geom.n_shots))
    scores = {}
    for rung, method in GUIDED.items():
        model = networks[method]
        pred, sigma = _predict(model, model.normalizer, initial, initial, sparse, obs, solver, shots, cfg, 0)
        row = gradient_metrics(pred, full["grad"])
        rng = torch.Generator().manual_seed(seed)
        row["directional_derivative_relative_error"] = directional_derivative_relative_error(
            pred, initial, obs, solver, wavelet, geom, full_shots,
            f_cut=cfg.f_cut, eps_fd=0.05, n_probes=8, generator=rng,
        ).item()
        sparse_norm, scale = model.normalizer.normalize(sparse["grad"])
        ref_norm, _ = model.normalizer.normalize(full["grad"], scale)
        pred_norm, _ = model.normalizer.normalize(pred, scale)
        row["spearman_r"] = uncertainty_calibration_metrics(pred_norm, ref_norm, sigma)["spearman_r"]
        scores[rung] = row
        plotting.plot_gradient_and_uncertainty(
            initial.cpu(), ref_norm.cpu(), sparse_norm.cpu(), pred_norm.cpu(), sigma.cpu(),
            directory / f"{rung}_gradient_uncertainty", normalized=True, dx=geom.dx, dz=geom.dz)
    return scores, full["loss"]


def _case(benchmark, cut, experiment, obs, settings, networks, seed, directory):
    truth, initial, wavelet, geom, solver = experiment
    cfg = GuidedFWIConfig(**settings, v_bounds=benchmark["v_bounds"], f_cut=cut)
    common = (solver, wavelet, geom)
    shots = select_sparse_shots(geom.n_shots, float(cfg.sparse_ratio))
    scores, initial_loss = _open_loop(initial, obs, common, cfg, networks, shots, seed, directory)
    loop = {"initial_full_loss": initial_loss, "initial_metrics": velocity_metrics(initial, truth)}
    fields = {}
    for name, label in METHODS.items():
        final, row = _benchmark_method(name, truth, initial, obs, common, cfg, networks, seed,
                                       initial_full_loss=initial_loss)
        loop[name] = row
        baseline = loop["full_adjoint"]
        row["grad_time_reduction_pct"] = 100 * (1 - row["grad_time_mean"] / baseline["grad_time_mean"])
        row["time_reduction_pct"] = 100 * (1 - row["iter_time_mean"] / baseline["iter_time_mean"])
        row["iter_time_total"] = cfg.n_iters * row["iter_time_mean"]
        fields[label] = final.cpu()
    plotting.plot_closed_loop_convergence(loop, directory / "closed_loop_convergence", method_labels=METHODS)
    plotting.plot_velocity_and_profile(truth.cpu(), initial.cpu(), fields, [geom.nx // 3, 2 * geom.nx // 3],
                                       directory / "velocity_and_profile", dx=geom.dx, dz=geom.dz)
    return dict(benchmark=benchmark["name"], f_cut=cut, sparse_shots=shots, open_loop=scores, closed_loop=loop)


def _export_table(cases, path):
    sections = [r"\documentclass{article}", r"\usepackage[margin=2cm]{geometry}", r"\usepackage{booktabs}",
                r"\begin{document}", r"\small"]
    names = {"marmousi2": "Marmousi2", "overthrust": "SEG/EAGE Overthrust"}
    for i, case in enumerate(cases):
        name = names.get(case["benchmark"], case["benchmark"].replace("_", r"\_"))
        title = f"{name}, {case['f_cut']:g} Hz cutoff."
        tables = [
            ("open_loop", OPEN_METRICS, {key: key for key in GUIDED},
             "Zero-shot gradients. Directional error uses a full-shot central difference."),
            ("closed_loop", LOOP_COLUMNS, METHODS,
             "Closed-loop inversion. Descent is measured on the shots used to verify each update. "
             "Time reductions are relative to the full adjoint."),
        ]
        for group, metrics, labels, caption in tables:
            rows = []
            for key, label in metrics.items():
                values = [case[group][method][key] for method in labels]
                rows.append([label, *["--" if v is None else "$" + _tex_number(v, 5) + "$" for v in values]])
            sections.append(plotting._table(f"{title} {caption}", f"cross-{i}-{group}", "l" + "r" * len(labels),
                                            ["Metric", *labels.values()], rows))
        sections.append(r"\clearpage")
    path.write_text("\n".join([*sections, r"\end{document}", ""]), encoding="utf-8")


def _file_stamp(path):
    info = path.stat()
    return [str(path.resolve()), info.st_size, info.st_mtime_ns]


@torch.no_grad()
def run(config_path, device_name="auto", *, force=False):
    path = Path(config_path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    out = path.parent / config["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else device_name)
    settings = config["closed_loop"]
    if settings["n_iters"] < 1:
        raise ValueError("closed_loop.n_iters must be positive")
    seed = config.get("seed", 42)
    paths = {method: path.parent / config[f"{key.lower()}_checkpoint"] for key, method in GUIDED.items()}
    networks = {name: _load_model(file, device) for name, file in paths.items()}
    survey = {key: config[key] for key in ("grid", "acquisition", "physics")}
    protocol = dict(**survey, closed_loop=settings, seed=seed, device=str(device),
                    checkpoints={name: _file_stamp(file) for name, file in paths.items()})
    cases = []
    for bench in config["benchmarks"]:
        model_path = path.parent / bench["npy_path"]
        inputs = dict(protocol, benchmark=bench, velocity_file=_file_stamp(model_path))
        experiment = None
        for cut in config["f_cuts"]:
            directory = out / bench["name"] / f"f_{cut:g}Hz"
            marker = directory / "case.json"
            print(f"{bench['name']}, {cut:g} Hz", flush=True)
            if not force and marker.is_file():
                saved = json.loads(marker.read_text(encoding="utf-8"))
                if (saved["protocol"] == inputs and saved["case"]["f_cut"] == cut
                        and all((directory / f"{stem}.{ext}").is_file() for stem in FIGURES for ext in ("png", "pdf"))):
                    cases.append(saved["case"])
                    print("  Using cached case", flush=True)
                    continue
                marker.unlink()
            if experiment is None:
                model = dict(bench, npy_path=model_path)
                truth, initial, wavelet, geom, solver = make_experiment(dict(survey, model=model), 0, device)
                experiment = truth, initial.clamp(*bench["v_bounds"]), wavelet, geom, solver
                obs = solver.forward(truth, wavelet, geom, return_illumination=False)["seismograms"]
            case = _case(bench, cut, experiment, obs, settings, networks, seed, directory)
            _write_json(marker, {"protocol": inputs, "case": case})
            cases.append(case)

    report = dict(requested_config=config, device=str(device), seed=seed, cases=cases,
                  gradient_metric_space="physical", uncertainty_metric_space="normalized_with_sparse_scale")
    _export_table(cases, out / "paper1_benchmarks_table.tex")
    _write_json(out / "paper1_benchmarks_summary.json", report)
    print(f"Benchmark results: {out.resolve()}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default="auto", help="auto selects CUDA when available, otherwise CPU")
    parser.add_argument("--force", action="store_true", help="rerun cached benchmark cases")
    args = parser.parse_args()
    run(args.config, args.device, force=args.force)
