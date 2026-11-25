#!/usr/bin/env python3
"""Pretrain the multimodal gradient predictor on model-disjoint Stage 1 splits."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fwi_grad.data.dataset_builder import build_trajectory_samples
from fwi_grad.data.torch_dataset import FWIGradientDataset, collate_fwi_batch
from fwi_grad.eval.metrics import gradient_metrics, uncertainty_calibration_metrics
from fwi_grad.losses import Stage2GradientLoss
from fwi_grad.models import MultiModalGradientNet


_INPUTS = ("v_k", "v_init", "r_wave", "r_env", "r_phase", "shot_mask",
           "illumination", "g_sparse", "q_k")


def _prepare_manifest(data: dict, config_dir: Path, device: torch.device) -> Path:
    manifest = config_dir / data["manifest"]
    if manifest.is_file():
        return manifest
    if not data.get("auto_build", False):
        raise FileNotFoundError(f"Stage 1 manifest does not exist: {manifest}")

    stage1 = data["stage1_config"]
    if isinstance(stage1, str):
        path = config_dir / stage1
        stage1 = yaml.safe_load(path.read_text(encoding="utf-8"))
        config_dir = path.parent
    stage1 = dict(stage1)
    model = dict(stage1.get("model", {}))
    if model.get("npy_path"):
        model["npy_path"] = str(config_dir / model["npy_path"])
    stage1["model"] = model
    print(f"Building Stage 1 trajectories for {manifest}", flush=True)
    if manifest.name == "manifest.json":
        build_trajectory_samples(stage1, manifest.parent, device=device)
    else:
        # Stage 1 publishes a fixed basename. Isolate a custom-named dataset
        # so building it cannot overwrite a sibling manifest.json.
        manifest.parent.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="trajectory-", dir=manifest.parent))
        index = build_trajectory_samples(stage1, directory, device=device)
        for row in index["samples"]:
            row["path"] = (Path(directory.name) / row["path"]).as_posix()
        _write_json(manifest, index)
    return manifest


def _write_json(path: Path, data: dict) -> None:
    encoded = json.dumps(data, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, mode="w", encoding="utf-8") as f:
        f.write(encoded)
    Path(f.name).replace(path)


def _save_checkpoint(path: Path, checkpoint: dict) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        torch.save(checkpoint, f)
    Path(f.name).replace(path)


def _inputs(batch: dict, device: torch.device) -> dict[str, Tensor]:
    return {key: batch[key].to(device, non_blocking=True) for key in _INPUTS}


def _train_epoch(
    model: MultiModalGradientNet, loader: DataLoader, objective: Stage2GradientLoss,
    optimizer: torch.optim.Optimizer, device: torch.device,
) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        target = batch["g_ref_norm"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        output = model(**_inputs(batch, device))
        loss = objective(output["g_pred"], output["sigma"], target)["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
        total += loss.item() * len(target)
    return total / len(loader.dataset)


class _ValidationSample(NamedTuple):
    pred: Tensor
    ref: Tensor
    sigma: Tensor
    seconds: float


def _summarize(samples: list[_ValidationSample], eps: float) -> dict:
    directions = [gradient_metrics(s.pred, s.ref) for s in samples]
    pred = torch.cat([s.pred.flatten() for s in samples])
    ref = torch.cat([s.ref.flatten() for s in samples])
    sigma = torch.cat([s.sigma.flatten() for s in samples])
    return {
        "samples": len(samples),
        "cosine_similarity": sum(d["cosine_similarity"] for d in directions) / len(samples),
        "rel_l2_error": sum(d["rel_l2_error"] for d in directions) / len(samples),
        "inference_seconds_per_sample": sum(s.seconds for s in samples) / len(samples),
        **uncertainty_calibration_metrics(pred, ref, sigma, eps=eps),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def _validate(
    model: MultiModalGradientNet, loader: DataLoader, objective: Stage2GradientLoss, device: torch.device,
) -> dict:
    model.eval()
    by_scenario = defaultdict(list)
    total = 0.0
    for batch in loader:
        inputs = _inputs(batch, device)
        target = batch["g_ref_norm"].to(device, non_blocking=True)
        _synchronize(device)
        start = perf_counter()
        output = model(**inputs)
        _synchronize(device)
        seconds = (perf_counter() - start) / len(target)
        pred, sigma = output["g_pred"], output["sigma"]
        total += objective(pred, sigma, target)["loss"].item() * len(target)
        for name, p, r, s in zip(batch["scenario"], pred.cpu(), batch["g_ref_norm"], sigma.cpu()):
            by_scenario[name].append(_ValidationSample(p, r, s, seconds))
    samples = [sample for group in by_scenario.values() for sample in group]
    overall = _summarize(samples, objective.eps)
    overall["loss"] = total / len(samples)
    return {"overall": overall, "per_scenario": {name: _summarize(group, objective.eps)
                                                for name, group in by_scenario.items()}}


def train(config_path: Path, device_name: str = "auto") -> dict:
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    settings = config["training"]
    epochs = settings["epochs"]
    if epochs < 1:
        raise ValueError("training.epochs must be positive")
    torch.manual_seed(settings.get("seed", 42))
    data = config["dataset"]
    manifest = _prepare_manifest(data, config_path.parent, device)
    model = MultiModalGradientNet(**config.get("model", {})).to(device)
    loaders = {}
    for split in ("train", "val"):
        dataset = FWIGradientDataset(manifest, split=split, scenarios=data.get("scenarios"),
                                     max_iters=data.get("max_iters", 20), normalizer=model.normalizer)
        if not len(dataset):
            raise ValueError(f"no {split} samples selected from {manifest}")
        loaders[split] = DataLoader(
            dataset, batch_size=settings["batch_size"], shuffle=(split == "train"),
            num_workers=settings.get("num_workers", 0), pin_memory=(device.type == "cuda"),
            collate_fn=collate_fwi_batch,
        )
    objective = Stage2GradientLoss(**config.get("loss", {}))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"],
                                  weight_decay=settings.get("weight_decay", 0.01))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    out = config_path.parent / config["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    report = {"device": str(device), "metric_space": "normalized_gradient", "history": []}
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        train_loss = _train_epoch(model, loaders["train"], objective, optimizer, device)
        val = _validate(model, loaders["val"], objective, device)
        val_loss = val["overall"]["loss"]
        scheduler.step()
        checkpoint = {"epoch": epoch, "config": config, "val_loss": val_loss,
                      "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                      "scheduler_state_dict": scheduler.state_dict()}
        if val_loss < best_loss:
            best_loss = val_loss
            report["best_epoch"] = epoch
            _save_checkpoint(out / "best.pt", checkpoint)
        _save_checkpoint(out / "last.pt", checkpoint)
        report["history"].append({"epoch": epoch, "lr": lr, "train_loss": train_loss, "val": val})
        _write_json(out / "stage2_metrics.json", report)
        print(f"Epoch {epoch}/{epochs}  train {train_loss:.5f}  val {val_loss:.5f}  "
              f"cosine {val['overall']['cosine_similarity']:.4f}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage2_smoke.yaml")
    parser.add_argument("--device", default="auto", help="auto selects CUDA when available, otherwise CPU")
    args = parser.parse_args()
    train(args.config, args.device)


if __name__ == "__main__":
    main()
