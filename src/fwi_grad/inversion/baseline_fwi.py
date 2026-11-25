"""Conventional velocity-space full-waveform inversion in physical units."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from time import perf_counter

import torch
from torch import Tensor

from ..eval.metrics import velocity_metrics
from ..physics.acoustic2d import Acoustic2DSolver
from ..physics.geometry import AcquisitionGeometry


class BoundedVelocityOptimizer:
    """Keep optimizer moments across FWI updates and project velocity into bounds."""

    def __init__(
        self,
        v_init: Tensor,
        lr: float,
        optimizer: str = "adam",
        v_bounds: Sequence[float] = (1500.0, 5000.0),
    ) -> None:
        lower, upper = map(float, v_bounds)
        if not (math.isfinite(lower) and math.isfinite(upper) and 0 < lower < upper):
            raise ValueError("v_bounds must be finite, positive and increasing")
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be finite and positive")
        self.bounds = (lower, upper)
        self.velocity = torch.nn.Parameter(v_init.detach().clone())
        if optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam([self.velocity], lr=lr)
        elif optimizer.lower() == "sgd":
            self.optimizer = torch.optim.SGD([self.velocity], lr=lr)
        else:
            raise ValueError("optimizer must be 'adam' or 'sgd'")

    @torch.no_grad()
    def step(self, gradient: Tensor) -> None:
        self.velocity.grad = gradient.detach()
        self.optimizer.step()
        self.velocity.clamp_(*self.bounds)
        self.optimizer.zero_grad(set_to_none=True)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def run_traditional_fwi(
    v_init: Tensor,
    v_true: Tensor,
    d_obs: Tensor,
    solver: Acoustic2DSolver,
    wavelet: Tensor,
    geom: AcquisitionGeometry,
    n_iters: int,
    lr: float,
    shot_indices: Sequence[int] | None = None,
    f_cut: float = 0.0,
    optimizer: str = "adam",
    v_bounds: Sequence[float] = (1500.0, 5000.0),
    *, on_step: Callable[[Tensor], None] | None = None,
) -> dict:
    """Optimize the selected-shot objective using raw (unfiltered) observations.

    Entry k of loss_history measures J(v_k) before its update. Entry k of
    metric_history describes the resulting v_{k+1}, with the last entry
    scoring v_final. The optional observer sees the updated velocity after timing.
    """
    if n_iters < 0:
        raise ValueError("n_iters must be nonnegative")
    device = v_init.device if isinstance(v_init, Tensor) else wavelet.device
    initial = torch.as_tensor(v_init, device=device)
    truth = torch.as_tensor(v_true, device=device, dtype=initial.dtype)
    update = BoundedVelocityOptimizer(initial, lr, optimizer, v_bounds)
    losses, metrics = [], []
    grad_seconds = iter_seconds = 0.0
    _synchronize(initial.device)
    for _ in range(n_iters):
        start = perf_counter()
        result = solver.adjoint_gradient(
            update.velocity.detach(), d_obs, wavelet, geom,
            shot_indices=shot_indices, f_cut=f_cut,
        )
        grad_seconds += result["elapsed_sec"]
        losses.append(result["loss"])
        update.step(result["grad"])
        metrics.append(velocity_metrics(update.velocity.detach(), truth))
        _synchronize(initial.device)
        iter_seconds += perf_counter() - start
        if on_step is not None:
            on_step(update.velocity.detach())
    return {
        "v_final": update.velocity.detach(),
        "loss_history": losses,
        "metric_history": metrics,
        "grad_time_mean": grad_seconds / n_iters if n_iters else 0.0,
        "iter_time_mean": iter_seconds / n_iters if n_iters else 0.0,
    }
