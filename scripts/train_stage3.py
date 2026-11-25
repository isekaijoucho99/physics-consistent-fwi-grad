#!/usr/bin/env python3
"""Fine-tune a Stage 2 checkpoint and compare its open- and closed-loop FWI performance."""

from __future__ import annotations

import argparse
import copy
import math
import sys
from dataclasses import replace
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if __package__:
    from . import train_stage2 as stage2
else:
    import train_stage2 as stage2

from fwi_grad.data.torch_dataset import FWIGradientDataset, collate_fwi_batch
from fwi_grad.data.velocity_models import add_gaussian_noise
from fwi_grad.eval.metrics import gradient_metrics, uncertainty_calibration_metrics, velocity_metrics
from fwi_grad.inversion.baseline_fwi import run_traditional_fwi
from fwi_grad.inversion.guided_fwi import GuidedFWIConfig, run_physics_guided_fwi
from fwi_grad.losses import Stage2GradientLoss
from fwi_grad.losses.physics_losses import (
    ProgressivePhysicsWeightSchedule, directional_derivative_loss, directional_derivative_relative_error,
    one_step_data_consistency_loss, structural_continuity_loss, subset_misfit,
)
from fwi_grad.models import MultiModalGradientNet
from fwi_grad.physics.acoustic2d import Acoustic2DSolver
from fwi_grad.physics.geometry import create_surface_geometry, select_sparse_shots
from fwi_grad.physics.wavelet import ricker_wavelet


class _TrajectoryDataset(FWIGradientDataset):
    def __getitem__(self, index):
        return dict(super().__getitem__(index), model_id=self.entries[index]["model_id"])


class _Survey:
    """Recreate raw Stage 1 observations once per geology and noise realization."""

    def __init__(self, config, device):
        physics = config.get("physics", {})
        grid = {"nz": 48, "nx": 64, "dz": 10., "dx": 10., **config.get("grid", {})}
        acquisition = {"n_shots": 4, "n_rec": 16, **config.get("acquisition", {})}
        self.geom = create_surface_geometry(**grid, **acquisition)
        self.solver = Acoustic2DSolver(
            physics.get("dt", 0.001), npml=physics.get("npml", 16),
            shot_batch_size=physics.get("shot_batch_size", 4), free_surface=physics.get("free_surface", False),
        )
        self.wavelet = ricker_wavelet(physics.get("nt", 180), self.solver.dt, physics.get("f0", 12.),
                                      device=device)
        self.seed = config.get("model", {}).get("seed", 0)
        self.snr = config.get("dataset", {}).get("snr_db", 15.)
        self.cache = {}

    @torch.no_grad()
    def observations(self, model_id, scenario, truth):
        clean_key = (model_id, "clean")
        if clean_key not in self.cache:
            syn = self.solver.forward(truth.to(self.wavelet), self.wavelet, self.geom, return_illumination=False)
            self.cache[clean_key] = syn["seismograms"].cpu()
        key = clean_key
        if scenario == "noisy":
            key = (model_id, "noisy")
            if key not in self.cache:
                index = int(model_id.removeprefix("model_"))
                self.cache[key] = add_gaussian_noise(self.cache[clean_key].to(self.wavelet), self.snr,
                                                      seed=self.seed + index).cpu()
        return self.cache[key].to(self.wavelet)


def _physics_sample(model, batch, inputs, output, i, survey, cfg, rng, training):
    v = inputs["v_k"][i, 0]
    if training and "g_ref_scale" in batch:
        scale = batch["g_ref_scale"][i].to(v).reshape(())
    else:
        scale = model.normalizer.compute_scale(inputs["g_sparse"][i, 0])
    pred = model.normalizer.denormalize(output["g_pred"][i, 0], scale)
    sigma = output["sigma"][i, 0]
    obs = survey.observations(batch["model_id"][i], batch["scenario"][i], batch["v_true"][i, 0])
    count = min(cfg["n_physics_shots"], survey.geom.n_shots)
    shots = torch.randperm(survey.geom.n_shots, generator=rng)[:count].tolist()
    args = (v, obs, survey.solver, survey.wavelet, survey.geom)
    cutoff = float(batch["f_cut"][i]) if "f_cut" in batch else float(batch["q_k"][i, 0] * 10.0)
    derivative = directional_derivative_loss if training else directional_derivative_relative_error
    full_shots = list(range(survey.geom.n_shots))
    n_probes = cfg.get("n_probes", 4) if training else cfg.get("eval_probes", 8)
    error = derivative(
        pred, *args, full_shots, f_cut=cutoff, eps_fd=cfg.get("eps_fd", 0.05),
        n_probes=n_probes, generator=rng,
    )
    step = one_step_data_consistency_loss(
        pred, sigma, *args, shots, eta=cfg["eta"], rho=cfg.get("rho", 0.0), kappa=cfg.get("kappa", 2.0),
        f_cut=cutoff, v_bounds=cfg.get("v_bounds", (1500.0, 5000.0)),
        max_update_mps=cfg.get("max_update_mps", 100.0),
    )
    return error, step


def _train_epoch(model, loader, objective, optimizer, survey, cfg, weights, rng):
    model.train()
    total = 0.
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        inputs = stage2._inputs(batch, survey.wavelet.device)
        output = model(**inputs)
        pred = output["g_pred"]
        supervised = objective(pred, output["sigma"], batch["g_ref_norm"].to(pred))["loss"]
        structural = structural_continuity_loss(pred, inputs["v_k"], beta=cfg.get("beta", 5.))
        selected = torch.randperm(len(pred), generator=rng)[:cfg["physics_subset_size"]].tolist()
        physics = []
        for i in selected:
            error, step = _physics_sample(model, batch, inputs, output, i, survey, cfg, rng, True)
            physics.append(weights[0] * error + weights[1] * step["loss_data"])
        loss = supervised + weights[2] * structural + torch.stack(physics).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        total += loss.item() * len(pred)
    return total / len(loader.dataset)


@torch.no_grad()
def _validate(model, loader, objective, survey, cfg, weights, seed):
    model.eval()
    # Restart this private stream so every model sees identical shots and directions.
    rng = torch.Generator().manual_seed(seed)
    totals = dict.fromkeys(("loss", "cosine_similarity", "rel_l2_error",
                            "directional_derivative_relative_error", "one_step_descent_success_rate"), 0.)
    preds, refs, sigmas = [], [], []
    for batch in loader:
        inputs = stage2._inputs(batch, survey.wavelet.device)
        output = model(**inputs)
        pred, sigma = output["g_pred"], output["sigma"]
        ref = batch["g_ref_norm"].to(pred)
        supervised = objective(pred, sigma, ref)["loss"]
        structural = structural_continuity_loss(pred, inputs["v_k"], beta=cfg.get("beta", 5.))
        totals["loss"] += (supervised + weights[2] * structural).item() * len(pred)
        for i in range(len(pred)):
            for name, value in gradient_metrics(pred[i], ref[i]).items():
                totals[name] += value
            error, step = _physics_sample(model, batch, inputs, output, i, survey, cfg, rng, False)
            totals["directional_derivative_relative_error"] += error.item()
            totals["one_step_descent_success_rate"] += step["descended"].item()
            totals["loss"] += (weights[0] * error.clamp_max(10.) + weights[1] * step["loss_data"]).item()
        preds.append(pred.flatten().cpu())
        refs.append(ref.flatten().cpu())
        sigmas.append(sigma.flatten().cpu())
    n = len(loader.dataset)
    return {"samples": n, **{key: value / n for key, value in totals.items()},
            **uncertainty_calibration_metrics(torch.cat(preds), torch.cat(refs), torch.cat(sigmas),
                                              eps=objective.eps)}


def _closed_loop(dataset, baseline, model, survey, cfg, seed):
    # Prefer the normal test trajectory. Each scenario stores its own starting velocity.
    index = next((i for i, row in enumerate(dataset.entries) if row["scenario"] == "normal"), 0)
    sample = dataset[index]
    truth, initial = (sample[key][0].to(survey.wavelet) for key in ("v_true", "v_init"))
    obs = survey.observations(sample["model_id"], sample["scenario"], truth)
    common = (survey.solver, survey.wavelet, survey.geom)
    cfg = replace(cfg, f_cut=cfg.f_cut or float(sample["q_k"][0] * 10))
    full = list(range(survey.geom.n_shots))
    sparse = select_sparse_shots(len(full), float(cfg.sparse_ratio))
    report = {"model_id": sample["model_id"], "scenario": sample["scenario"],
              "initial_full_loss": subset_misfit(initial, obs, *common, full, cfg.f_cut).item()}

    def record(name, result):
        final = result.pop("v_final")
        result["final_full_loss"] = subset_misfit(final, obs, *common, full, cfg.f_cut).item()
        result["final_metrics"] = velocity_metrics(final, truth)
        report[name] = result

    for name, shots in (("full_adjoint", full), ("sparse_adjoint", sparse)):
        result = run_traditional_fwi(initial, truth, obs, *common, cfg.n_iters, cfg.lr,
                                     shot_indices=shots, f_cut=cfg.f_cut, v_bounds=cfg.v_bounds)
        record(name, result)
    for name, net in (("stage2_guided", baseline), ("stage3_guided", model)):
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)  # check-shot selection uses the CPU RNG
            result = run_physics_guided_fwi(initial, truth, obs, net, *common, cfg)
        record(name, result)
    for name in ("sparse_adjoint", "stage2_guided", "stage3_guided"):
        report[name]["speedup_vs_full"] = report["full_adjoint"]["iter_time_mean"] / report[name]["iter_time_mean"]
    return report


def train(config_path: Path, device_name: str = "auto") -> dict:
    path = Path(config_path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    settings, cfg, data = config["training"], config["physics_loss"], config["dataset"]
    guided = GuidedFWIConfig(**config.get("guided_fwi", {}))
    if min(settings["epochs"], cfg["physics_subset_size"], cfg["n_physics_shots"], guided.n_iters) < 1:
        raise ValueError("training requires positive epochs, physics subset/shot counts and closed-loop iterations")
    seed = settings.get("seed", 42)
    torch.manual_seed(seed)
    checkpoint_path = path.parent / config["pretrained_checkpoint"]
    pretrained = None
    if checkpoint_path.is_file():
        pretrained = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    elif not config.get("allow_scratch", False):
        raise FileNotFoundError(f"Missing Stage 2 checkpoint {checkpoint_path}. Run train_stage2.py "
                                "with the matching config first, or set allow_scratch: true.")
    architecture = config if pretrained is None else pretrained["config"]
    model_config = architecture.get("model", {})
    model = MultiModalGradientNet(**model_config).to(device)
    if pretrained is not None:
        model.load_state_dict(pretrained["model_state_dict"])
    baseline = copy.deepcopy(model).eval()

    manifest = stage2._prepare_manifest(data, path.parent, device)
    source = data["stage1_config"]
    if isinstance(source, str):
        source = yaml.safe_load((path.parent / source).read_text(encoding="utf-8"))
    survey = _Survey(source, device)
    datasets = {name: _TrajectoryDataset(manifest, split=name, scenarios=data.get("scenarios"),
                                         max_iters=data.get("max_iters", 20), normalizer=model.normalizer)
                for name in ("train", "val", "test")}
    for name, dataset in datasets.items():
        if not len(dataset):
            raise ValueError(f"no {name} samples selected from {manifest}")
    loaders = {name: DataLoader(datasets[name], batch_size=settings["batch_size"], shuffle=name == "train",
                                num_workers=settings.get("num_workers", 0), pin_memory=device.type == "cuda",
                                collate_fn=collate_fwi_batch) for name in ("train", "val")}
    objective = Stage2GradientLoss(**config.get("loss", {}))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"],
                                  weight_decay=settings.get("weight_decay", 0.01))
    schedule = ProgressivePhysicsWeightSchedule(cfg["lambda_dir"], cfg["lambda_data"], cfg["lambda_str"],
                                               cfg["warmup_epochs"])
    # Use fixed target weights to compare validation losses across the warmup.
    targets = schedule(cfg["warmup_epochs"])
    val_args = (loaders["val"], objective, survey, cfg, targets, seed)
    report = {"device": str(device), "initialization": "stage2_checkpoint" if pretrained else "scratch",
              "pretrained_checkpoint": str(checkpoint_path) if pretrained else None,
              "gradient_metric_space": "normalized_gradient",
              "stage2_baseline": _validate(baseline, *val_args), "history": []}
    out = path.parent / config["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    rng = torch.Generator().manual_seed(seed)
    for epoch in range(1, settings["epochs"] + 1):
        weights = schedule(epoch)
        loss = _train_epoch(model, loaders["train"], objective, optimizer, survey, cfg, weights, rng)
        val = _validate(model, *val_args)
        if not math.isfinite(val["loss"]):
            raise FloatingPointError(f"non-finite Stage 3 validation loss at epoch {epoch}")
        report["history"].append({"epoch": epoch, "physics_weights": weights, "train_loss": loss, "val": val})
        if val["loss"] < best:
            best = val["loss"]
            report.update(best_epoch=epoch, stage3_physics=val)
            stage2._save_checkpoint(out / "best_stage3.pt", {
                "epoch": epoch, "val_loss": best, "config": dict(config, model=model_config),
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            })
        print(f"Epoch {epoch}/{settings['epochs']}  train {loss:.5f}  val {val['loss']:.5f}  "
              f"one-step descent {val['one_step_descent_success_rate']:.1%}", flush=True)
    model.load_state_dict(torch.load(out / "best_stage3.pt", map_location="cpu",
                                     weights_only=True)["model_state_dict"])
    report["closed_loop"] = _closed_loop(datasets["test"], baseline, model, survey, guided, seed)
    stage2._write_json(out / "stage3_metrics.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage3_smoke.yaml")
    parser.add_argument("--device", default="auto", help="auto selects CUDA when available, otherwise CPU")
    args = parser.parse_args()
    train(args.config, args.device)
