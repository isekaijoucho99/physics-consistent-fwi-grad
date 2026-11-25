"""Generate model-disjoint FWI trajectories and their supervised gradient targets."""

from __future__ import annotations

import json
import math
import random
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch import Tensor

from ..inversion.baseline_fwi import BoundedVelocityOptimizer
from ..physics.acoustic2d import Acoustic2DSolver
from ..physics.geometry import AcquisitionGeometry, create_surface_geometry, select_sparse_shots
from ..physics.residuals import compute_multimodal_residuals
from ..physics.wavelet import highpass_filter, ricker_wavelet
from .velocity_models import (
    add_gaussian_noise,
    build_initial_model,
    generate_synthetic_complex_model,
    load_openfwi_velocity,
)


SCENARIOS = ("normal", "low_freq_cut", "noisy", "sparse", "biased_init")
SPLITS = ("train", "val", "test")


def make_experiment(
    config: Mapping, model_index: int, device: str | torch.device,
) -> tuple[Tensor, Tensor, Tensor, AcquisitionGeometry, Acoustic2DSolver]:
    """Construct one experiment. model_index offsets the seed or OpenFWI array index."""
    grid = config.get("grid", {})
    physics = config.get("physics", {})
    acquisition = config.get("acquisition", {})
    model = config.get("model", {})
    nz, nx = grid.get("nz", 48), grid.get("nx", 64)
    dz, dx = grid.get("dz", 10.0), grid.get("dx", 10.0)
    dt = physics.get("dt", 0.001)
    if model_index < 0:
        raise ValueError("model_index must be nonnegative")
    if model.get("npy_path"):
        truth = load_openfwi_velocity(
            model["npy_path"], index=model.get("index", 0) + model_index, target_shape=(nz, nx),
        )
    else:
        truth = generate_synthetic_complex_model(nz, nx, seed=model.get("seed", 0) + model_index)
    initial = build_initial_model(
        truth, method=model.get("method", "gaussian"), sigma=model.get("sigma", 8.0),
    )
    geom = create_surface_geometry(
        nz, nx, dz, dx, acquisition.get("n_shots", 4), acquisition.get("n_rec", 16),
        src_depth_idx=acquisition.get("src_depth_idx", 2),
        rec_depth_idx=acquisition.get("rec_depth_idx", 2),
        margin=acquisition.get("margin", 4),
    )
    wavelet = ricker_wavelet(
        physics.get("nt", 180), dt, physics.get("f0", 12.0), device=device,
    )
    solver = Acoustic2DSolver(
        dt, npml=physics.get("npml", 16), shot_batch_size=physics.get("shot_batch_size", 4),
        free_surface=physics.get("free_surface", False),
    )
    return (
        torch.as_tensor(truth, dtype=wavelet.dtype, device=device),
        torch.as_tensor(initial, dtype=wavelet.dtype, device=device),
        wavelet, geom, solver,
    )


def _split_models(ids: list[str], fractions: Sequence[float], seed: int) -> dict[str, list[str]]:
    if len(fractions) != len(SPLITS) or any(f < 0 or not math.isfinite(f) for f in fractions):
        raise ValueError("split_fractions must contain three finite nonnegative values")
    if not math.isclose(sum(fractions), 1.0, abs_tol=1e-6):
        raise ValueError("split_fractions must sum to one")
    targets = [len(ids) * f / sum(fractions) for f in fractions]
    counts = [math.floor(target) for target in targets]
    remainder_order = sorted(range(3), key=lambda i: targets[i] - counts[i], reverse=True)
    for i in remainder_order[:len(ids) - sum(counts)]:
        counts[i] += 1
    # For a small experiment, populate every requested partition whenever possible.
    if len(ids) >= sum(f > 0 for f in fractions):
        for i in range(3):
            if fractions[i] and counts[i] == 0:
                donor = max((j for j in range(3) if counts[j] > 1), key=lambda j: counts[j] - targets[j])
                counts[donor] -= 1
                counts[i] += 1
    shuffled = ids.copy()
    random.Random(seed).shuffle(shuffled)
    start = 0
    splits = {}
    for split, count in zip(SPLITS, counts):
        splits[split] = shuffled[start:start + count]
        start += count
    return splits


def _write_model_samples(
    config: Mapping, model_index: int, model_id: str, split: str, directory: Path,
    device: str | torch.device, scenarios: Sequence[str], n_iters: int, family: str | None = None,
) -> list[dict]:
    data = config.get("dataset", {})
    inversion = config.get("inversion", {})
    truth, initial, wavelet, geom, solver = make_experiment(config, model_index, device)
    with torch.no_grad():
        clean_obs = solver.forward(truth, wavelet, geom, return_illumination=False)["seismograms"]
    n_shots = clean_obs.shape[0]
    ratio = float(data.get("sparse_ratio", inversion.get("sparse_ratio", 0.5)))
    if not 0 < ratio <= 1:
        raise ValueError("sparse_ratio must be in (0, 1]")
    sparse_shots = select_sparse_shots(n_shots, ratio)
    full_shots = list(range(n_shots))
    bounds = inversion.get("v_bounds", (1500.0, 5000.0))
    entries = []
    for scenario in scenarios:
        f_cut = float(data.get("f_cut", 5.0)) if scenario == "low_freq_cut" else 0.0
        snr_db = float(data.get("snr_db", 15.0)) if scenario == "noisy" else None
        observations = clean_obs
        if snr_db is not None:
            seed = config.get("model", {}).get("seed", 0) + model_index
            observations = add_gaussian_noise(clean_obs, snr_db, seed=seed)
        start = initial
        if scenario == "biased_init":
            start = (initial + data.get("bias_mps", 200.0)).clamp(*bounds)
        update = BoundedVelocityOptimizer(
            start, inversion.get("lr", 5.0), inversion.get("optimizer", "adam"), bounds,
        )
        is_sparse = scenario == "sparse"
        input_shots = sparse_shots
        input_obs = observations[sparse_shots]
        if f_cut > 0.0:
            input_obs = highpass_filter(input_obs, solver.dt, f_cut)
        shot_fraction = len(sparse_shots) / n_shots
        raw_steps = data.get("sample_steps")
        sample_set = set(int(s) for s in raw_steps) if raw_steps else set(range(n_iters))
        total_steps = (max(sample_set) + 1) if raw_steps else n_iters
        for k in range(total_steps):
            velocity = update.velocity.detach()
            if k not in sample_set:
                step_shots = sparse_shots if is_sparse else None
                step_res = solver.adjoint_gradient(
                    velocity, observations, wavelet, geom, shot_indices=step_shots, f_cut=f_cut,
                )
                update.step(step_res["grad"])
                continue
            full = solver.adjoint_gradient(velocity, observations, wavelet, geom, f_cut=f_cut)
            sparse = solver.adjoint_gradient(
                velocity, observations, wavelet, geom, shot_indices=sparse_shots, f_cut=f_cut,
            )
            synthetic = sparse["d_syn"]
            if f_cut > 0.0:
                synthetic = highpass_filter(synthetic, solver.dt, f_cut)
            residuals = compute_multimodal_residuals(synthetic, input_obs)
            filename = f"{model_id}_{scenario}_{k:04d}.pt"
            entry = {
                "path": (Path(directory.name) / filename).as_posix(),
                "model_id": model_id,
                "split": split,
                "scenario": scenario,
                "iter_k": k,
            }
            sample = {
                "v_k": velocity,
                "v_init": start,
                "v_true": truth,
                **residuals,
                "illumination": sparse["illumination"],
                "g_sparse": sparse["grad"],
                "g_ref": full["grad"],
            }
            payload = {name: value.detach().cpu() for name, value in sample.items()}
            payload["meta"] = {
                "model_id": model_id,
                "split": split,
                "scenario": scenario,
                "iter_k": k,
                "f_cut": f_cut,
                "snr_db": snr_db,
                "sparse_ratio": ratio,
                "shot_fraction": shot_fraction,
                "shot_indices": input_shots,
                "sparse_shot_indices": sparse_shots,
                "loss_k": full["loss"],
                "t_full_grad": full["elapsed_sec"],
                "t_sparse_grad": sparse["elapsed_sec"],
            }
            if family is not None:
                entry["family"] = payload["meta"]["family"] = family
            torch.save(payload, directory / filename)
            entries.append(entry)
            if k + 1 < total_steps:
                update.step(sparse["grad"] if is_sparse else full["grad"])
    return entries


def _source_models(config, fractions):
    if any(f <= 0 for f in fractions):
        raise ValueError("multi-family datasets need positive split_fractions for train, val and test")
    model = config["model"]
    seed = model.get("seed", 0)
    jobs, families, origins = {}, defaultdict(list), set()
    for source in model["sources"]:
        family, path = source["name"], Path(source["npy_path"]).resolve()
        start, count = source.get("start_index", 0), source["n_models"]
        if start < 0 or count < 1:
            raise ValueError("sources require nonnegative start_index and positive n_models")
        # Local array indices and global model IDs use different origins.
        local = dict(model, npy_path=path, index=start, seed=seed + len(jobs))
        experiment = dict(config, model=local)
        for i in range(count):
            origin = (path, start + i)
            if origin in origins:
                raise ValueError(f"overlapping OpenFWI source: {origin}")
            origins.add(origin)
            model_id = f"model_{len(jobs):04d}"
            families[family].append(model_id)
            jobs[model_id] = (experiment, i, family)

    if not jobs:
        raise ValueError("model.sources must contain at least one source")
    targets = _split_models(list(jobs), fractions, seed)
    # Reserve one member per family in every requested split, then fill the
    # largest fractional deficits while respecting the global rounded totals.
    remaining = [len(targets[s]) - len(families) for s in SPLITS]
    rng = random.Random(seed)
    splits = {s: [] for s in SPLITS}
    for family, ids in families.items():
        sizes = [1, 1, 1]
        if len(ids) < sum(sizes):
            raise ValueError(f"{family} has too few models to populate all requested splits")
        for _ in range(len(ids) - sum(sizes)):
            j = max((j for j in range(3) if remaining[j] > 0),
                    key=lambda j: (len(ids) * fractions[j] - sizes[j], remaining[j]))
            sizes[j] += 1
            remaining[j] -= 1
        rng.shuffle(ids)
        offset = 0
        for split, size in zip(SPLITS, sizes):
            splits[split].extend(ids[offset:offset + size])
            offset += size
    return jobs, splits


def build_trajectory_samples(
    config: Mapping, output_dir: str | Path, device: str | torch.device = "cuda",
) -> dict:
    """Publish a manifest only after all model trajectories have been written.

    All scenarios use the same shots for their reference gradient. Only the
    sparse scenario advances with the subset gradient. Its saved residuals and
    illumination use those same shots. Clean observations are shared across
    scenarios of each model and noise is held fixed along each noisy trajectory.
    """
    data = config.get("dataset", {})
    n_models = data.get("n_models", 3)
    n_iters = data.get("n_iters", config.get("inversion", {}).get("n_iters", 3))
    scenarios = data.get("scenarios", SCENARIOS)
    model = config.get("model", {})
    if ("sources" not in model and n_models <= 0) or n_iters <= 0:
        raise ValueError("dataset n_models and n_iters must be positive")
    if isinstance(scenarios, str) or not scenarios or set(scenarios).difference(SCENARIOS):
        raise ValueError(f"scenarios must be a nonempty sequence selected from {SCENARIOS}")
    if len(scenarios) != len(set(scenarios)):
        raise ValueError("scenario names must be unique")
    fractions = data.get("split_fractions", (0.8, 0.1, 0.1))
    if "sources" in model:
        jobs, splits = _source_models(config, fractions)
    else:
        jobs = {f"model_{i:04d}": (config, i, None) for i in range(n_models)}
        splits = _split_models(list(jobs), fractions, model.get("seed", 0))
    assignments = {model_id: split for split, members in splits.items() for model_id in members}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(dir=output, prefix="samples-"))
    try:
        entries = []
        for number, (model_id, (source, i, family)) in enumerate(jobs.items(), 1):
            if family is not None:
                print(f"  Dataset {number}/{len(jobs)}: {family} {model_id} ({assignments[model_id]})", flush=True)
            entries.extend(_write_model_samples(
                source, i, model_id, assignments[model_id], directory, device, scenarios, n_iters, family,
            ))
        manifest = {"splits": splits, "samples": entries}
        pending = directory / "manifest.tmp"
        pending.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        pending.replace(output / "manifest.json")
    except BaseException:
        shutil.rmtree(directory)
        raise
    return manifest
