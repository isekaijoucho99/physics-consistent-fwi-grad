#!/usr/bin/env python3
"""Train the six Paper 1 rungs, benchmark held-out geologies and export the paper tables."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if __package__:
    from . import train_stage2 as stage2, train_stage3 as stage3
else:
    import train_stage2 as stage2
    import train_stage3 as stage3

from fwi_grad.data.dataset_builder import build_trajectory_samples, make_experiment
from fwi_grad.data.torch_dataset import collate_fwi_batch
from fwi_grad.eval import plotting
from fwi_grad.inversion.baseline_fwi import run_traditional_fwi
from fwi_grad.inversion.guided_fwi import GuidedFWIConfig, run_physics_guided_fwi
from fwi_grad.losses import Stage2GradientLoss
from fwi_grad.losses.physics_losses import (
    ProgressivePhysicsWeightSchedule, directional_derivative_relative_error, one_step_data_consistency_loss,
    structural_continuity_loss, subset_misfit,
)
from fwi_grad.models import MultiModalGradientNet, SignPreservingNormalizer
from fwi_grad.physics.geometry import select_sparse_shots


RUNGS = ("A0_sparse_only", "A1_no_data", "A2_multimodal_no_unc", "A3_stage2_full",
         "A4_stage3_dir_str", "A5_proposed_full")
GUIDED = {"A2_guided": RUNGS[2], "A3_guided": RUNGS[3], "A5_proposed_guided": RUNGS[5]}
METHODS = {"full_adjoint": "Full adjoint", "sparse_adjoint": "Sparse adjoint",
           "A2_guided": "A2 guided", "A3_guided": "A3 guided", "A5_proposed_guided": "A5 guided"}


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _build_dataset(experiment, config_dir, directory, device):
    manifest = directory / "manifest.json"
    settings = directory / "experiment.json"
    if manifest.is_file():
        if settings.is_file() and _read(settings) != experiment:
            raise ValueError(f"Dataset configuration differs from {settings}. Select a new output_dir.")
        print(f"[1/4] Reusing dataset {manifest}", flush=True)
        return manifest

    print(f"[1/4] Building trajectories in {directory}", flush=True)
    directory.mkdir(parents=True, exist_ok=True)
    stage2._write_json(settings, experiment)
    model = experiment["model"]
    sources = []
    rng = np.random.default_rng(model["seed"])
    for source in model["sources"]:
        if "smoke_relief_cells" in source:
            # Small layered models with independent contrasts; OOD uses fault and salt instead.
            nz, nx = experiment["grid"]["nz"], experiment["grid"]["nx"]
            z, x = np.ogrid[:nz, :nx]
            stack = []
            for _ in range(source["n_models"]):
                depth = z + source["smoke_relief_cells"] * np.sin(2 * np.pi * x / nx + rng.uniform(0, 2 * np.pi))
                stack.append(1650 + rng.uniform(0, 150) + 80 * depth
                             + rng.uniform(300, 600) * (depth > nz / 3)
                             + rng.uniform(450, 800) * (depth > 2 * nz / 3))
            path = directory / f"{source['name']}.npy"
            np.save(path, np.stack(stack).astype("float32"))
        else:
            path = config_dir / source["npy_path"]
        sources.append(dict(source, npy_path=path))
    build_trajectory_samples(dict(experiment, model=dict(model, sources=sources)), directory, device)
    return manifest


def _load_model(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = MultiModalGradientNet(**ckpt["config"]["model"])
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def _loader(dataset, settings, shuffle=False, seed=0):
    return DataLoader(dataset, batch_size=settings["batch_size"], shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), collate_fn=collate_fwi_batch,
                      num_workers=settings.get("num_workers", 0))


def _physics_sample(model, batch, inputs, output, i, survey, cfg, rng, *,
                    held_out=False, use_unc=True, need_step=True):
    v = inputs["v_k"][i, 0]
    if not held_out and "g_ref_scale" in batch:
        scale = batch["g_ref_scale"][i].to(v).reshape(())
    else:
        scale = model.normalizer.compute_scale(inputs["g_sparse"][i, 0])
    pred = model.normalizer.denormalize(output["g_pred"][i, 0], scale)
    obs = survey.observations(batch["model_id"][i], batch["scenario"][i], batch["v_true"][i, 0])
    common = (v, obs, survey.solver, survey.wavelet, survey.geom)
    full = list(range(survey.geom.n_shots))
    shots = full if held_out else torch.randperm(len(full), generator=rng)[:cfg["n_physics_shots"]].tolist()
    cut = float(batch["f_cut"][i]) if "f_cut" in batch else float(batch["q_k"][i, 0] * 10.0)
    n_probes = cfg.get("eval_probes", 8) if held_out else cfg.get("n_probes", 4)
    error = directional_derivative_relative_error(
        pred, *common, full, f_cut=cut, eps_fd=cfg.get("eps_fd", 0.05), n_probes=n_probes, generator=rng,
    )
    step = None
    if need_step:
        step = one_step_data_consistency_loss(
            pred, output["sigma"][i, 0], *common, shots, eta=cfg["eta"], rho=cfg.get("rho", 0.0),
            kappa=cfg.get("kappa", 2.0) if use_unc else 0.0, f_cut=cut,
            v_bounds=cfg.get("v_bounds", (1500.0, 5000.0)), max_update_mps=cfg.get("max_update_mps", 100.0))
    return error, step


def _physics_epoch(model, loader, objective, survey, cfg, weights, rng, optimizer=None):
    """Train on a physics subset, or score the fixed validation objective on every sample."""
    training = optimizer is not None
    model.train(training)
    w_dir, w_data, w_str = weights
    total = 0.
    with torch.set_grad_enabled(training):
        for batch in loader:
            inputs = stage2._inputs(batch, survey.wavelet.device)
            output = model(**inputs)
            pred = output["g_pred"]
            loss = objective(pred, output["sigma"], batch["g_ref_norm"].to(pred))["loss"]
            loss = loss + w_str * structural_continuity_loss(pred, inputs["v_k"], beta=cfg.get("beta", 5.))
            indices = (torch.randperm(len(pred), generator=rng)[:cfg["physics_subset_size"]].tolist()
                       if training else range(len(pred)))
            terms = []
            for i in indices:
                error, step = _physics_sample(model, batch, inputs, output, i, survey, cfg, rng, need_step=w_data != 0.)
                term = w_dir * error.clamp_max(10.)
                terms.append(term if step is None else term + w_data * step["loss_data"])
            loss = loss + torch.stack(terms).mean()
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
            total += loss.item() * len(pred)
    return total / len(loader.dataset)


@torch.no_grad()
def _evaluate(model, loader, objective, survey, physics, seed):
    device = survey.wavelet.device
    rng = torch.Generator().manual_seed(seed)
    families = {row["model_id"]: row["family"] for row in loader.dataset.entries}
    by_family, by_scenario, all_samples = defaultdict(list), defaultdict(list), []
    # Both held-out physics metrics score the full acquisition, even for sparse inputs.
    for batch in loader:
        inputs = stage2._inputs(batch, device)
        stage2._synchronize(device)
        start = perf_counter()
        output = model(**inputs)
        stage2._synchronize(device)
        seconds = (perf_counter() - start) / len(batch["model_id"])
        for i, model_id in enumerate(batch["model_id"]):
            error, step = _physics_sample(model, batch, inputs, output, i, survey, physics, rng,
                                          held_out=True, use_unc=objective.lambda_unc > 0.)
            fields = stage2._ValidationSample(output["g_pred"][i].cpu(), batch["g_ref_norm"][i],
                                              output["sigma"][i].cpu(), seconds)
            record = (fields, error.item(), step["descended"].item())
            all_samples.append(record)
            by_family[families[model_id]].append(record)
            by_scenario[batch["scenario"][i]].append(record)

    def summarize(records):
        fields, errors, descents = zip(*records)
        result = stage2._summarize(fields, objective.eps)
        result["inference_ms"] = 1000 * result.pop("inference_seconds_per_sample")
        result["directional_derivative_relative_error"] = float(np.mean(errors))
        result["one_step_descent_success_rate"] = float(np.mean(descents))
        return result

    return {"overall": summarize(all_samples),
            "per_family": {k: summarize(v) for k, v in by_family.items()},
            "per_scenario": {k: summarize(v) for k, v in by_scenario.items()}}


def _train_rung(rung, config, datasets, survey, out):
    seed, device = config["seed"], survey.wavelet.device
    torch.manual_seed(seed)
    tune = rung in RUNGS[4:]
    settings = config["stage3" if tune else "stage2"]
    physics = dict(config["physics_loss"])
    if rung == RUNGS[4]:
        physics["lambda_data"] = 0.
    schedule = ProgressivePhysicsWeightSchedule(physics["lambda_dir"], physics["lambda_data"],
                                               physics["lambda_str"], physics["warmup_epochs"])
    architecture = dict(config["network"], disable_model_branch=rung == RUNGS[0],
                        disable_data_branch=rung in RUNGS[:2])
    initial = f"{RUNGS[3]}/best.pt" if tune else None
    model = MultiModalGradientNet(**architecture).to(device)
    if tune:
        model.load_state_dict(torch.load(out / initial, map_location="cpu", weights_only=True)["model_state_dict"])
    lambda_unc = 0. if rung in RUNGS[:3] else .1
    objective = Stage2GradientLoss(lambda_unc=lambda_unc)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"])
    train = _loader(datasets["train"], settings, shuffle=True, seed=seed)
    val = _loader(datasets["val"], settings)
    epochs = settings["epochs"]
    scheduler = None if tune else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    directory = out / rung
    directory.mkdir(exist_ok=True)
    rng = torch.Generator().manual_seed(seed)
    best = math.inf
    history = []
    for epoch in range(1, epochs + 1):
        if tune:
            loss = _physics_epoch(model, train, objective, survey, physics, schedule(epoch), rng, optimizer)
            val_loss = _physics_epoch(model, val, objective, survey, physics, schedule(physics["warmup_epochs"]),
                                      torch.Generator().manual_seed(seed))
        else:
            loss = stage2._train_epoch(model, train, objective, optimizer, device)
            val_loss = stage2._validate(model, val, objective, device)["overall"]["loss"]
            scheduler.step()
        if not math.isfinite(val_loss):
            raise FloatingPointError(f"{rung} epoch {epoch}: non-finite validation loss")
        history.append(dict(epoch=epoch, train_loss=loss, val_loss=val_loss))
        if val_loss < best:
            best, best_epoch = val_loss, epoch
            stage2._save_checkpoint(directory / "best.pt", {
                "model_state_dict": model.state_dict(), "epoch": epoch, "val_loss": best,
                "config": {"model": architecture, "training": settings}, "initialization": initial,
            })
        print(f"  {rung} epoch {epoch}/{epochs}  train {loss:.5g}  val {val_loss:.5g}", flush=True)
    print(f"  {rung} evaluating best epoch {best_epoch} on {len(datasets['test'])} test samples", flush=True)
    model = _load_model(directory / "best.pt", device)
    report = _evaluate(model, _loader(datasets["test"], settings), objective, survey, physics, seed)
    report.update(rung=rung, split="test", seed=seed, best_epoch=best_epoch, history=history,
                  model=architecture, training=settings, device=str(device), lambda_unc=lambda_unc,
                  physics_loss=physics if tune else None, physical_prediction_scale="sparse_gradient",
                  initialization=initial, gradient_metric_space="normalized_gradient",
                  physics_evaluation_shots="full")
    stage2._write_json(directory / "eval.json", report)
    return report


def _benchmark_models(test, experiment, n_ood, survey):
    # Read one normal starting model per held-out geology directly from the manifest.
    seen = set()
    for i, entry in enumerate(test.entries):
        mid = entry["model_id"]
        if mid not in seen and entry["scenario"] == "normal" and entry["iter_k"] == 0:
            seen.add(mid)
            sample = test[i]
            truth, initial = (sample[key][0].to(survey.wavelet) for key in ("v_true", "v_init"))
            yield "test", mid, entry["family"], truth, initial
    if seen != {row["model_id"] for row in test.entries}:
        raise ValueError("benchmark requires the normal initial state of every test model")
    model = dict(experiment["model"])
    model.pop("sources")
    model.pop("npy_path", None)
    for i in range(n_ood):
        truth, initial, *_ = make_experiment(dict(experiment, model=model), i, survey.wavelet.device)
        yield "ood", f"ood_{i:04d}", "fault_salt", truth, initial


def _benchmark_method(name, truth, initial, obs, common, cfg, networks, seed, *, initial_full_loss=None):
    """Optionally score every iterate on all shots, outside the inversion timers."""
    _, _, geom = common
    full = list(range(geom.n_shots))
    sparse = select_sparse_shots(len(full), float(cfg.sparse_ratio))
    full_history = []

    def record_loss(velocity):
        full_history.append(subset_misfit(velocity, obs, *common, full, cfg.f_cut).item())

    observer = record_loss if initial_full_loss is not None and name != "full_adjoint" else None
    print(f"    {name} ({cfg.n_iters} iterations)", flush=True)
    if name in networks:
        control = replace(cfg, seed=seed, kappa=0.0, sigma_threshold=math.inf) if name == "A2_guided" else replace(cfg, seed=seed)
        run = run_physics_guided_fwi(initial, truth, obs, networks[name], *common, control, on_step=observer)
    else:
        shots = full if name == "full_adjoint" else sparse
        run = run_traditional_fwi(initial, truth, obs, *common, cfg.n_iters, cfg.lr, shot_indices=shots,
                                  f_cut=cfg.f_cut, optimizer=cfg.optimizer, v_bounds=cfg.v_bounds, on_step=observer)
    final = run["v_final"]
    full_loss = full_history[-1] if full_history else subset_misfit(final, obs, *common, full, cfg.f_cut).item()
    if initial_full_loss is not None:
        full_curve = [initial_full_loss, *full_history] if full_history else [*run["loss_history"], full_loss]
        descent = sum(b < a for a, b in zip(full_curve, full_curve[1:])) / cfg.n_iters
    elif name in networks:
        descent = run["descent_success_rate"]
    else:
        last = full_loss if shots == full else subset_misfit(final, obs, *common, shots, cfg.f_cut).item()
        losses = run["loss_history"] + [last]
        descent = sum(b < a for a, b in zip(losses, losses[1:])) / cfg.n_iters
    record = dict(run["metric_history"][-1], final_full_loss=full_loss,
                  predicted_accept_rate=run.get("predicted_accept_rate"), descent_success_rate=descent,
                  grad_time_mean=run["grad_time_mean"], iter_time_mean=run["iter_time_mean"])
    if initial_full_loss is not None:
        record["full_loss_history"] = full_curve
        record["metric_history"] = run["metric_history"]
    return final, record


def _benchmark_case(truth, initial, obs, survey, cfg, networks, seed):
    common = (survey.solver, survey.wavelet, survey.geom)
    full = list(range(survey.geom.n_shots))
    initial_full_loss = subset_misfit(initial, obs, *common, full, cfg.f_cut).item()
    methods = {}
    for name in METHODS:
        _, row = _benchmark_method(name, truth, initial, obs, common, cfg, networks, seed,
                                   initial_full_loss=initial_full_loss)
        methods[name] = {k: v for k, v in row.items() if k not in ("full_loss_history", "metric_history")}
        methods[name]["time_reduction_pct"] = 100.0 * (1.0 - row["iter_time_mean"] / methods["full_adjoint"]["iter_time_mean"])
    return methods


def _aggregate(cases):
    groups = defaultdict(list)
    for case in cases:
        groups[(case["cohort"], str(case["f_cut"]))].append(case["methods"])
    result = defaultdict(dict)
    for (cohort, cut), members in groups.items():
        stats = {}
        for name in METHODS:
            stats[name] = {}
            for key, value in members[0][name].items():
                values = [m[name][key] for m in members]
                stats[name][key] = (None if value is None else
                                    {"mean": float(np.mean(values)), "std": float(np.std(values))})
        result[cohort][cut] = stats
    return dict(result)


def _benchmark(config, test, survey, networks, cache):
    settings = dict(config["closed_loop"])
    n_ood = settings.pop("n_ood_models")
    cfg = GuidedFWIConfig(**settings)
    protocol = {"closed_loop": config["closed_loop"], "experiment": config["experiment"],
                "seed": config["seed"], "device": str(survey.wavelet.device)}
    saved = _read(cache) if cache.exists() else {}
    cases = saved["cases"] if saved.get("protocol") == protocol else []
    done = {(c["model_id"], c["f_cut"]) for c in cases}
    total = 2 * (len({row["model_id"] for row in test.entries}) + n_ood)
    print(f"[3/4] Closed-loop benchmark: {len(done)}/{total} model/condition pairs cached", flush=True)
    for cohort, mid, family, truth, initial in _benchmark_models(test, config["experiment"], n_ood, survey):
        for cut in (0., 5.):
            if (mid, cut) in done:
                continue
            print(f"  Case {len(cases)+1}/{total}: {cohort} {family} {mid}, {cut:g} Hz", flush=True)
            obs = survey.observations(mid, "normal", truth)
            methods = _benchmark_case(truth, initial, obs, survey, replace(cfg, f_cut=cut), networks, config["seed"])
            cases.append(dict(cohort=cohort, model_id=mid, family=family, f_cut=cut, methods=methods))
            stage2._write_json(cache, {"cases": cases, "protocol": protocol})
    return {"cases": cases, "aggregate": _aggregate(cases), "std_definition": "population across models"}


OPEN_METRICS = {"cosine_similarity": "Cosine similarity", "rel_l2_error": "Relative $L_2$ error",
                "directional_derivative_relative_error": "Directional rel. error",
                "one_step_descent_success_rate": "One-step descent rate",
                "spearman_r": "Spearman $r$", "inference_ms": "Inference (ms)"}
LOOP_METRICS = {"final_full_loss": "Final full-shot loss", "mae": "MAE (m/s)", "rmse": "RMSE (m/s)",
                "re": "Relative error", "ssim": "SSIM", "snr_db": "SNR (dB)",
                "predicted_accept_rate": "Prediction acceptance", "descent_success_rate": "Descent rate",
                "grad_time_mean": "Gradient time (s)", "iter_time_mean": "Iteration time (s)",
                "time_reduction_pct": r"Time reduction (\%)"}


def _tex_number(value, digits=3):
    number = f"{value:.{digits}g}"
    if "e" in number:
        mantissa, exponent = number.split("e")
        return rf"{mantissa}\times10^{{{int(exponent)}}}"
    return number


def _export_tables(out, ablation, loop):
    rows = [[rung.replace("_", " "), *[rf"${_tex_number(evaluation['overall'][key])}$" for key in OPEN_METRICS]]
            for rung, evaluation in ablation.items()]
    ladder = plotting._table("Layer-by-layer gradient ablation on the held-out test split. "
                             "Directional error and one-step descent use all shots. "
                             "A0--A2 report the untrained uncertainty head's rank correlation and use unweighted steps.",
                             "ladder", "lrrrrrr", ["Rung", "Cosine", r"Rel. $L_2$", "Dir. error",
                                                  "Descent", r"Spearman $r$", "ms/sample"], rows)
    rows = []
    for cohort, cuts in loop["aggregate"].items():
        for cut, group in cuts.items():
            n = sum(c["cohort"] == cohort and c["f_cut"] == float(cut) for c in loop["cases"])
            label = "Held-out test" if cohort == "test" else "OOD fault-salt"
            rows.append([rf"\multicolumn{{6}}{{l}}{{\textit{{{label}, {float(cut):g} Hz, $n={n}$}}}}"])
            for key, label in LOOP_METRICS.items():
                row = [label]
                for name in METHODS:
                    stat = group[name][key]
                    row.append("--" if stat is None else
                               rf"${_tex_number(stat['mean'])} \pm {_tex_number(stat['std'], 2)}$")
                rows.append(row)
    closed = plotting._table(
        r"Closed-loop results (mean $\pm$ population standard deviation across models). "
        r"Time reduction is $100(1-t/t_{\rm full})$ per paired model and cutoff. "
        "Descent uses each method's update objective. Adjoint methods propose no learned steps.",
        "benchmark", "lrrrrr", ["Metric", *METHODS.values()], rows)
    closed = closed.replace(r"\begin{tabular}", r"\scriptsize\setlength{\tabcolsep}{2pt}\begin{tabular}", 1)
    (out / "paper1_ablation_tables.tex").write_text("% Requires booktabs\n" + ladder + "\n\n" + closed + "\n",
                                                    encoding="utf-8")


@torch.no_grad()
def _export_figures(out, ablation, loop, sample, model, survey):
    labels = [rung[:2] for rung in ablation]
    with plotting._figure(out / "ablation_ladder", (7.2, 4.2)) as fig:
        for ax, (key, label) in zip(fig.subplots(2, 3).flat, OPEN_METRICS.items()):
            ax.bar(labels, [ev["overall"][key] for ev in ablation.values()], color=plotting._COLORS)
            ax.set_title(label)
            ax.axhline(0, linewidth=.5, color="0.4")

    groups = [(cohort, cut, stats) for cohort, cuts in loop["aggregate"].items() for cut, stats in cuts.items()]
    x = np.arange(len(groups))
    with plotting._figure(out / "closed_loop_summary", (7.2, 3.7)) as fig:
        grid = fig.add_gridspec(2, 2, height_ratios=[.15, 1])
        legend = fig.add_subplot(grid[0, :])
        legend.axis("off")
        axes = [fig.add_subplot(grid[1, i]) for i in range(2)]
        for ax, key, label in zip(axes, ("mae", "time_reduction_pct"), ("MAE (m/s)", "Time reduction (%)")):
            for i, (name, method) in enumerate(METHODS.items()):
                values = [stats[name][key] for _, _, stats in groups]
                ax.bar(x + (i - 2) * .16, [v["mean"] for v in values], width=.15,
                       yerr=[v["std"] for v in values], capsize=1.5, color=plotting._COLORS[i], label=method)
            ax.set_xticks(x, [f"{'Test' if cohort == 'test' else 'OOD'}\n{float(cut):g} Hz"
                              for cohort, cut, _ in groups])
            ax.axhline(0, linewidth=.5, color="0.4")
            ax.set_ylabel(label)
        handles, labels = axes[0].get_legend_handles_labels()
        legend.legend(handles, labels, loc="center", ncol=3)

    output = model(**stage2._inputs(collate_fwi_batch([sample]), survey.wavelet.device))
    sparse, _ = model.normalizer.normalize(sample["g_sparse"], scale=sample["g_ref_scale"])
    plotting.plot_gradient_and_uncertainty(sample["v_k"][0], sample["g_ref_norm"][0], sparse[0],
                                           output["g_pred"][0, 0].cpu(), output["sigma"][0, 0].cpu(),
                                           out / "best_gradient_uncertainty", normalized=True,
                                           dx=survey.geom.dx, dz=survey.geom.dz)


def run(config_path, device_name="auto", force=False):
    path = Path(config_path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else device_name)
    if device.type == "cpu":
        torch.set_num_threads(1)
    out = path.parent / config["output_dir"]
    experiment = config["experiment"]
    manifest = _build_dataset(experiment, path.parent, out / "dataset", device)
    splits = _read(manifest)["splits"]
    print("  Model counts: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()), flush=True)
    survey = stage3._Survey(experiment, device)
    normalizer = SignPreservingNormalizer(alpha=config["network"].get("alpha", 10.))
    datasets = {split: stage3._TrajectoryDataset(manifest, split=split, max_iters=experiment["dataset"]["n_iters"],
                                                 normalizer=normalizer)
                for split in splits}
    cache = out / "closed_loop.json"
    if force:
        # Invalidate all completion markers up front so an interrupted forced run can resume.
        for rung in RUNGS:
            (out / rung / "eval.json").unlink(missing_ok=True)
    if any(not (out / rung / "eval.json").is_file() for rung in GUIDED.values()):
        cache.unlink(missing_ok=True)

    print(f"[2/4] Six-rung ablation on {device}", flush=True)
    ablation = {}
    for i, rung in enumerate(RUNGS, 1):
        marker = out / rung / "eval.json"
        if marker.is_file():
            print(f"  {i}/6 {rung}: skip (eval.json exists)", flush=True)
            ablation[rung] = _read(marker)
        else:
            print(f"  {i}/6 {rung}: training", flush=True)
            ablation[rung] = _train_rung(rung, config, datasets, survey, out)
    networks = {name: _load_model(out / rung / "best.pt", device) for name, rung in GUIDED.items()}
    loop = _benchmark(config, datasets["test"], survey, networks, cache)
    print("[4/4] Exporting tables, figures and summary", flush=True)
    _export_tables(out, ablation, loop)
    # Fixed manifest order selects the example independently of its test performance.
    example = datasets["test"][0]
    _export_figures(out, ablation, loop, example, networks["A5_proposed_guided"], survey)
    report = dict(requested_config=config, device=str(device), splits=splits, ablation=ablation, closed_loop=loop,
                  gradient_figure_sample=datasets["test"].entries[0])
    stage2._write_json(out / "paper1_ablation_summary.json", report)
    print(f"Paper 1 results: {out.resolve()}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default="auto", help="auto selects CUDA when available, otherwise CPU")
    parser.add_argument("--force", action="store_true", help="rerun all rungs and closed-loop cases on the saved split")
    args = parser.parse_args()
    run(args.config, args.device, args.force)
