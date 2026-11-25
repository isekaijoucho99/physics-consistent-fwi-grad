#!/usr/bin/env python3
"""Compare grid and neural FWI with full-adjoint and Stage 3 gradients."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fwi_grad.data.dataset_builder import make_experiment
from fwi_grad.eval.metrics import velocity_metrics
from fwi_grad.eval.plotting import plot_closed_loop_convergence, plot_velocity_and_profile
from fwi_grad.inversion.baseline_fwi import run_traditional_fwi
from fwi_grad.inversion.guided_fwi import GuidedFWIConfig, run_physics_guided_fwi
from fwi_grad.inversion.reparam_fwi import ReparamFWIConfig, run_reparam_guided_fwi
from fwi_grad.losses.physics_losses import subset_misfit
from fwi_grad.models import MultiModalGradientNet


METHODS = {"full_grid_fwi": "Full adjoint (grid)", "stage3_grid_fwi": "Stage 3 (grid)",
           "full_reparam_fwi": "Full adjoint (neural)", "stage3_reparam_fwi": "Stage 3 (neural)"}


class _GridHistory:
    """Keep the Stage 3 pre-update velocities for full-shot scoring outside its timers."""

    def __init__(self, solver):
        self.solver = solver
        self.velocities = []

    def __getattr__(self, name):
        return getattr(self.solver, name)

    def adjoint_gradient(self, v, obs, wave, geom, shot_indices=None, **kwargs):
        if shot_indices is not None:  # A full fallback revisits the same model.
            self.velocities.append(v.detach())
        return self.solver.adjoint_gradient(v, obs, wave, geom, shot_indices=shot_indices, **kwargs)


def _load_predictor(config, directory, device):
    checkpoint = config.get("stage3_checkpoint")
    path = directory / checkpoint if checkpoint else None
    if path is not None and path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = MultiModalGradientNet(**saved["config"].get("model", {}))
        model.load_state_dict(saved["model_state_dict"])
        return model.to(device).eval(), "stage3_checkpoint"
    if not config.get("allow_scratch", False):
        raise FileNotFoundError(f"Missing Stage 3 checkpoint: {path}")
    return MultiModalGradientNet(**config.get("grad_model", {})).to(device).eval(), "scratch"


@torch.no_grad()
def run(config_path, device_name="auto", output_dir=None):
    path = Path(config_path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    common = config["inversion"]
    neural = ReparamFWIConfig(**common, **config.get("reparam", {}))
    guided = GuidedFWIConfig(**common, lr=config.get("grid_lr", 5.))
    torch.manual_seed(neural.seed)
    predictor, initialization = _load_predictor(config, path.parent, device)
    print(f"Gradient model: {initialization}", flush=True)
    model = config["model"]
    if model.get("npy_path"):
        model["npy_path"] = str(path.parent / model["npy_path"])
    truth, initial, wave, geom, solver = make_experiment(config, 0, device)
    # All four methods start inside the same normalized logit-clamp interval.
    lower, upper = neural.v_bounds
    margin = (upper - lower) * 1e-4
    initial = initial.clamp(lower + margin, upper - margin)
    obs = solver.forward(truth, wave, geom, return_illumination=False)["seismograms"]
    full = list(range(geom.n_shots))

    def full_loss(v):
        return subset_misfit(v, obs, solver, wave, geom, full, neural.f_cut).item()

    loop = {"initial_full_loss": full_loss(initial), "initial_metrics": velocity_metrics(initial, truth)}
    fields = {}

    def record(name, result, history):
        final = result.pop("v_final")
        result.pop("v_background", None)
        result.update(full_loss_history=history, final_full_loss=history[-1],
                      final_metrics=velocity_metrics(final, truth))
        loop[name] = result
        fields[METHODS[name]] = final.cpu().numpy()
        print(f"{name}: final full-shot misfit {history[-1]:.6g}", flush=True)

    grid = run_traditional_fwi(initial, truth, obs, solver, wave, geom, neural.n_iters, guided.lr,
                               f_cut=neural.f_cut, v_bounds=neural.v_bounds)
    record("full_grid_fwi", grid, [*grid["loss_history"], full_loss(grid["v_final"])])
    recorder = _GridHistory(solver)
    grid = run_physics_guided_fwi(initial, truth, obs, predictor, recorder, wave, geom, guided)
    recorder.velocities.append(grid["v_final"])
    history = [loop["initial_full_loss"]] + [full_loss(v) for v in recorder.velocities[1:]]
    record("stage3_grid_fwi", grid, history)
    for name, model_net in (("full_reparam_fwi", None), ("stage3_reparam_fwi", predictor)):
        result = run_reparam_guided_fwi(initial, truth, obs, solver, wave, geom, neural, model_net)
        record(name, result, result["full_loss_history"])

    out = Path(output_dir) if output_dir is not None else path.parent / config["output_dir"]
    plot_velocity_and_profile(truth.cpu().numpy(), initial.cpu().numpy(), fields,
                              [geom.nx // 3, 2 * geom.nx // 3], out / "velocity_and_profile",
                              dx=geom.dx, dz=geom.dz)
    plot_closed_loop_convergence(loop, out / "closed_loop_convergence", method_labels=METHODS)
    report = {"initialization": initialization, "stage3_checkpoint": config.get("stage3_checkpoint"),
              "model": model, "device": str(device), "f_cut": neural.f_cut, "closed_loop": loop}
    destination = out / "stage4_metrics.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(destination)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage4_smoke.yaml")
    parser.add_argument("--device", default="auto", help="auto selects CUDA when available, otherwise CPU")
    parser.add_argument("--output-dir", type=Path, help="override the configured figure and metrics directory")
    args = parser.parse_args()
    run(args.config, args.device, args.output_dir)
