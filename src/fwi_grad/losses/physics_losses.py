"""Wave-equation physics-consistent losses for FWI gradient prediction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from ..models.conditioning import propose_update
from ..physics.acoustic2d import Acoustic2DSolver, _objective_sum
from ..physics.geometry import AcquisitionGeometry


def confidence_weight(sigma: Tensor, kappa: float = 2.0, min_weight: float = 0.1) -> Tensor:
    """Map positive pixelwise uncertainty ``sigma`` to step weights in ``[min_weight, 1]``."""
    return (1.0 + kappa * sigma).reciprocal().clamp(min_weight, 1.0)


def structural_continuity_loss(
    g_pred_norm: Tensor,
    v_k: Tensor,
    beta: float = 5.0,
    eps: float = 1e-6,
) -> Tensor:
    """Boundary-aware total variation encouraging smooth gradients inside strata."""
    loss = g_pred_norm.new_zeros(())
    for axis in (-2, -1):
        if g_pred_norm.shape[axis] > 1:
            dv = v_k.detach().diff(dim=axis).abs()
            weights = torch.exp(-beta * dv / (dv.mean(dim=(-2, -1), keepdim=True) + eps))
            loss = loss + (weights * g_pred_norm.diff(dim=axis).abs()).mean()
    return loss


def subset_misfit(
    v: Tensor,
    d_obs: Tensor,
    solver: Acoustic2DSolver,
    wavelet: Tensor,
    geom: AcquisitionGeometry,
    shot_indices: Sequence[int],
    f_cut: float = 0.0,
) -> Tensor:
    """Differentiable half-MSE waveform misfit over the selected shots."""
    syn = solver.forward(
        v, wavelet, geom, shot_indices=shot_indices, return_illumination=False
    )["seismograms"]
    shots = torch.as_tensor(shot_indices, device=v.device, dtype=torch.long)
    obs = solver._observations(d_obs, shots, geom, wavelet.to(v))
    return _objective_sum(syn, obs, solver.dt, f_cut, "l2") / obs.numel()


def directional_derivative_relative_error(
    g_pred_phys: Tensor,
    v_k: Tensor,
    d_obs: Tensor,
    solver: Acoustic2DSolver,
    wavelet: Tensor,
    geom: AcquisitionGeometry,
    shot_indices: Sequence[int],
    f_cut: float = 0.0,
    eps_fd: float = 0.05,
    eps_denom: float = 1e-8,
    p: Tensor | None = None,
    n_probes: int = 1,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Multi-probe sum-ratio directional derivative discrepancy against central FD.

    When ``p`` is supplied explicitly, evaluates that single probe. Otherwise draws
    ``n_probes`` unit random directions and computes ``sum(|pred·p_m - fd_m|) / sum(|fd_m|)``
    to prevent near-zero single-probe denominators from dominating the statistic.
    """
    pred_flat = g_pred_phys.reshape_as(v_k)
    if p is not None:
        directions = [p.detach().to(v_k)]
    else:
        directions = []
        for _ in range(max(1, n_probes)):
            d = torch.randn(v_k.shape, generator=generator).to(v_k)
            directions.append(d / d.norm().clamp_min(1e-12))

    num = pred_flat.new_zeros(())
    den = 0.0
    for direction in directions:
        with torch.no_grad():
            ref = solver.directional_derivative_fd(
                v_k,
                direction,
                d_obs,
                wavelet,
                geom,
                eps=eps_fd,
                shot_indices=shot_indices,
                f_cut=f_cut,
            )
        pred_dot = (pred_flat * direction).sum()
        num = num + (pred_dot - ref).abs()
        den += abs(ref)
    return num / (den + eps_denom)


def directional_derivative_loss(
    g_pred_phys: Tensor,
    v_k: Tensor,
    d_obs: Tensor,
    solver: Acoustic2DSolver,
    wavelet: Tensor,
    geom: AcquisitionGeometry,
    shot_indices: Sequence[int],
    f_cut: float = 0.0,
    eps_fd: float = 0.05,
    eps_denom: float = 1e-8,
    p: Tensor | None = None,
    n_probes: int = 4,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Clamped multi-probe directional derivative consistency loss."""
    return directional_derivative_relative_error(
        g_pred_phys,
        v_k,
        d_obs,
        solver,
        wavelet,
        geom,
        shot_indices,
        f_cut=f_cut,
        eps_fd=eps_fd,
        eps_denom=eps_denom,
        p=p,
        n_probes=n_probes,
        generator=generator,
    ).clamp_max(10.0)


def one_step_data_consistency_loss(
    g_pred_phys: Tensor,
    sigma: Tensor,
    v_k: Tensor,
    d_obs: Tensor,
    solver: Acoustic2DSolver,
    wavelet: Tensor,
    geom: AcquisitionGeometry,
    shot_indices: Sequence[int],
    eta: float = 5.0,
    rho: float = 0.0,
    kappa: float = 2.0,
    f_cut: float = 0.0,
    v_bounds: tuple[float, float] = (1500.0, 5000.0),
    max_update_mps: float = 100.0,
) -> dict[str, Tensor]:
    """Differentiable one-step PDE descent loss using the deployed update rule.

    Applies ``propose_update`` (99th-percentile normalized, confidence-weighted,
    and capped) so the training step rule matches closed-loop ``GuidedFWI``.
    """
    with torch.no_grad():
        before = subset_misfit(v_k, d_obs, solver, wavelet, geom, shot_indices, f_cut)
    conf = confidence_weight(sigma, kappa).reshape_as(v_k)
    step = propose_update(g_pred_phys.reshape_as(v_k), conf, lr=eta, max_update_mps=max_update_mps)
    trial = (v_k - step).clamp(*v_bounds)
    after = subset_misfit(trial, d_obs, solver, wavelet, geom, shot_indices, f_cut)
    loss_data = torch.relu((after - before) / (before + 1e-8) + rho)
    return {
        "loss_data": loss_data,
        "loss_before": before,
        "loss_after": after,
        "descended": after < before,
    }


@dataclass
class ProgressivePhysicsWeightSchedule:
    """Linearly ramp physics loss weights from zero to target over ``warmup_epochs``."""

    lambda_dir: float
    lambda_data: float
    lambda_str: float
    warmup_epochs: int

    def __post_init__(self) -> None:
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be nonnegative")

    def __call__(self, epoch: int) -> tuple[float, float, float]:
        if self.warmup_epochs == 0:
            fraction = 1.0
        else:
            fraction = min(max(epoch, 0) / self.warmup_epochs, 1.0)
        return (
            self.lambda_dir * fraction,
            self.lambda_data * fraction,
            self.lambda_str * fraction,
        )
