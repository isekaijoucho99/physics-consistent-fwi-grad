#!/usr/bin/env python3
"""Run the gradient check, full/sparse FWI baselines and trajectory generator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch import Tensor

# Allow the documented command to run from a source checkout before installation.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fwi_grad.data.dataset_builder import build_trajectory_samples, make_experiment
from fwi_grad.eval.metrics import velocity_metrics
from fwi_grad.inversion.baseline_fwi import run_traditional_fwi
from fwi_grad.physics.acoustic2d import Acoustic2DSolver
from fwi_grad.physics.geometry import AcquisitionGeometry, select_sparse_shots
from fwi_grad.physics.wavelet import highpass_filter


def gradient_check(
    velocity: Tensor, observed: Tensor, wavelet: Tensor, geom: AcquisitionGeometry,
    solver: Acoustic2DSolver, f_cut: float, seed: int,
) -> dict:
    """Cross-check an unnormalised random direction on one representative shot."""
    # MPS lacks float64. Use the identical discrete operator on CPU for this check.
    check_device = velocity.device if velocity.device.type != "mps" else torch.device("cpu")
    model, data, source = [x.to(check_device, torch.float64) for x in (velocity, observed, wavelet)]
    shots = [geom.src_x.numel() // 2]
    direction = torch.randn(model.shape, generator=torch.Generator().manual_seed(seed),
                            dtype=model.dtype).to(check_device)
    grad = solver.adjoint_gradient(model, data, source, geom, shot_indices=shots, f_cut=f_cut)["grad"]
    projection = (grad * direction).sum().item()
    epsilon = 0.05  # m/s for a direction of unit expected RMS
    fd = solver.directional_derivative_fd(model, direction, data, source, geom, eps=epsilon,
                                          shot_indices=shots, f_cut=f_cut)
    relative = abs(projection - fd) / max(abs(projection), abs(fd), 1e-30)
    return {"adjoint_dot": projection, "central_fd": fd, "relative_error": relative,
            "epsilon_mps": epsilon, "dtype": "float64", "device": str(check_device),
            "shot_indices": shots, "passed": relative < 1e-4}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage1_smoke.yaml")
    parser.add_argument("--device", default="auto", help="auto selects CUDA if available, otherwise CPU")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device = torch.device(device_name)
    if device.type == "cpu":
        # Small stencil operations often cost more to schedule than to execute.
        torch.set_num_threads(min(4, torch.get_num_threads()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    truth, initial, wavelet, geom, solver = make_experiment(config, 0, device)
    with torch.no_grad():
        observed = solver.forward(truth, wavelet, geom, return_illumination=False)["seismograms"]

    inv = config["inversion"]
    cutoff = inv.get("f_cut", 0.0)
    summary = {"config": str(args.config), "device": str(device), "torch_version": str(torch.__version__),
               "initial_metrics": velocity_metrics(initial, truth)}
    summary["gradient_check"] = gradient_check(initial, observed, wavelet, geom, solver, cutoff,
                                                config["model"]["seed"])
    if not summary["gradient_check"]["passed"]:
        raise RuntimeError(f"Directional derivative check failed: {summary['gradient_check']}")
    print(f"FD relative error {summary['gradient_check']['relative_error']:.3g}", flush=True)

    selections = {"full": list(range(geom.src_x.numel())),
                  "sparse": select_sparse_shots(geom.src_x.numel(), inv["sparse_ratio"])}
    for name, shots in selections.items():
        result = run_traditional_fwi(initial, truth, observed, solver, wavelet, geom,
                                     n_iters=inv["n_iters"], lr=inv["lr"], shot_indices=shots,
                                     f_cut=cutoff, optimizer=inv.get("optimizer", "adam"),
                                     v_bounds=inv.get("v_bounds", (1500., 5000.)))
        final = result.pop("v_final")
        with torch.no_grad():
            traces = solver.forward(final, wavelet, geom, shots, return_illumination=False)["seismograms"]
            target = observed[shots]
            if cutoff > 0:
                traces = highpass_filter(traces, solver.dt, cutoff)
                target = highpass_filter(target, solver.dt, cutoff)
            final_loss = (0.5 * (traces - target).square().mean()).item()
        result.update(shot_indices=shots, final_loss=final_loss, final_metrics=velocity_metrics(final, truth))
        summary[name] = result
        print(f"{name}: {len(shots)} shots, final loss {final_loss:.6g}, "
              f"MAE {result['final_metrics']['mae']:.3f} m/s", flush=True)
    manifest = build_trajectory_samples(config, args.output_dir / "dataset", device=str(device))
    summary["dataset"] = {"manifest": "dataset/manifest.json", "n_samples": len(manifest["samples"]),
                          "models_per_split": {k: len(v) for k, v in manifest["splits"].items()}}
    # A reader sees either the previous complete summary or this complete run.
    destination = args.output_dir / "summary.json"
    pending = destination.with_suffix(".json.tmp")
    pending.write_text(json.dumps(summary, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(destination)
    print(destination)


if __name__ == "__main__":
    main()
