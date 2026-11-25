"""Shared conditioning encoding and velocity-update proposal rules."""

from __future__ import annotations

import torch
from torch import Tensor

F_CUT_SCALE = 10.0
ITER_REF = 25.0


def encode_condition(
    f_cut: float,
    iter_k: int,
    shot_fraction: float,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Encode frequency cutoff, iteration index, and active shot fraction into [3]."""
    iter_scaled = min(float(iter_k) / ITER_REF, 1.0)
    return torch.tensor(
        [float(f_cut) / F_CUT_SCALE, iter_scaled, float(shot_fraction)],
        dtype=dtype,
        device=device,
    )


def propose_update(
    grad: Tensor,
    confidence: Tensor | float,
    lr: float,
    max_update_mps: float = 150.0,
) -> Tensor:
    """Build a bounded, confidence-weighted velocity update in m/s.

    Normalizing by the 99th percentile of |grad| aligns the step scale across
    iterations and geological models so ``lr`` corresponds roughly to the peak
    update amplitude in m/s before clipping.
    """
    scale = torch.quantile(grad.abs().flatten(), 0.99).clamp_min(1e-12)
    step = lr * confidence * (grad / scale)
    peak = step.abs().amax().clamp_min(max_update_mps)
    return step * (max_update_mps / peak)
