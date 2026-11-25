"""Verified multi-scale neural FWI with full-adjoint or learned velocity gradients."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter

import torch
from torch import Tensor, nn

from ..eval.metrics import velocity_metrics
from ..losses.physics_losses import confidence_weight, subset_misfit
from ..models.normalization import SignPreservingNormalizer
from ..models.reparam_net import MultiScaleVelocityReparam
from ..physics.acoustic2d import Acoustic2DSolver, _objective_sum, check_cfl
from ..physics.geometry import AcquisitionGeometry, select_sparse_shots
from .baseline_fwi import _synchronize
from .guided_fwi import _predict


@dataclass
class ReparamFWIConfig:
    n_iters: int = 30
    lr_low: float = .008
    lr_high: float = .004
    lr_grid: float = .005
    v_bounds: tuple[float, float] = (1500., 5000.)
    kappa: float = 2.
    sigma_threshold: float = .5
    n_check_shots: int = 2
    max_backtracks: int = 3
    sparse_ratio: float = .25
    f_cut: float = 5.
    detail_unlock_frac: float = .3
    grid_unlock_frac: float = .75
    illumination_precond: bool = True
    enable_fallback: bool = True
    low_hidden: int = 32
    high_hidden: int = 64
    n_fourier: int = 32
    fourier_scale: float = 2.
    seed: int = 0
    max_step_mps: float = 20.0

    def __post_init__(self):
        if self.n_iters < 0 or self.n_check_shots < 1 or self.max_backtracks < 0:
            raise ValueError("invalid iteration, verification shot or backtrack count")
        if not self.max_step_mps > 0:
            raise ValueError("max_step_mps must be positive")


def _check_loss_before(adjoint, shots, check, obs, dt, cutoff):
    if check == shots:
        return adjoint["loss"]
    # Reuse the same forward recordings even when checking fewer shots than the adjoint.
    syn = adjoint["d_syn"][[shots.index(i) for i in check]]
    return (_objective_sum(syn, obs[check], dt, cutoff, "l2") / syn.numel()).item()


def _normalize_gradient(grad):
    return grad / torch.quantile(grad.abs().flatten(), .99).clamp_min(1e-12)


@torch.no_grad()
def _verified_step(net, optimizer, weights, step_scale, before, evaluate, max_backtracks,
                   v_prev, max_step_mps):
    """Backtrack one Adam proposal. Its moments advance exactly once on acceptance."""
    params = list(net.parameters())
    saved = [p.detach().clone() for p in params]
    state = deepcopy(optimizer.state_dict())
    # Dormant groups must not accumulate Adam steps or move under stored momentum.
    for group, weight in zip(optimizer.param_groups, (1., *weights)):
        if weight == 0.:
            for p in group["params"]:
                p.grad = None
    optimizer.step()
    if step_scale != 1.:
        for p, old in zip(params, saved):
            p.lerp_(old, 1. - step_scale)
    for attempt in range(max_backtracks + 1):
        trial = net(*weights)
        max_delta = (trial - v_prev).abs().amax().item()
        # If the nonlinear projection overshoots, allow a second contraction before loss verification.
        for safety in (1., .95):
            if max_delta <= max_step_mps:
                break
            ratio = safety * max_step_mps / max_delta
            for p, old in zip(params, saved):
                p.lerp_(old, 1. - ratio)
            trial = net(*weights)
            max_delta = (trial - v_prev).abs().amax().item()
        if max_delta <= max_step_mps:
            after = evaluate(trial)
            if after < before:
                return trial, after, attempt
        if attempt < max_backtracks:
            for p, old in zip(params, saved):
                p.lerp_(old, .5)
    for p, old in zip(params, saved):
        p.copy_(old)
    optimizer.load_state_dict(state)
    return None


@torch.no_grad()
def run_reparam_guided_fwi(
    v_init: Tensor, v_true: Tensor, d_obs: Tensor, solver: Acoustic2DSolver,
    wavelet: Tensor, geom: AcquisitionGeometry, config: ReparamFWIConfig,
    grad_model: nn.Module | None = None, normalizer: SignPreservingNormalizer | None = None,
    device: torch.device | str | None = None,
) -> dict:
    """Record verified iterates; full-shot history includes the initial objective.

    The adjoint/predictor and VJP use the same proposed branch weights.
    Verification compares their combined effect and the Adam update with the
    last accepted velocity on fixed check shots.
    Independent full-shot reporting is excluded from the update timings.
    """
    cfg = config
    initial = v_init.detach().to(device=device)
    truth, obs, wave = v_true.to(initial), d_obs.detach().to(initial), wavelet.to(initial)
    net = MultiScaleVelocityReparam(initial, cfg.v_bounds, cfg.low_hidden, cfg.high_hidden,
                                    cfg.n_fourier, cfg.fourier_scale, seed=cfg.seed)
    optimizer = torch.optim.Adam([{"params": net.low_branch.parameters(), "lr": cfg.lr_low},
                                  {"params": net.high_branch.parameters(), "lr": cfg.lr_high},
                                  {"params": [net.grid_residual], "lr": cfg.lr_grid}])
    full = list(range(geom.n_shots))
    sparse_shots = select_sparse_shots(len(full), float(cfg.sparse_ratio))
    shots = full if grad_model is None else sparse_shots
    count = min(cfg.n_check_shots, len(sparse_shots))
    check = [sparse_shots[i] for i in select_sparse_shots(len(sparse_shots), count)]

    def loss(velocity, selected):
        return subset_misfit(velocity, obs, solver, wave, geom, selected, cfg.f_cut).item()

    def verify(trial):
        try:
            check_cfl(trial.max().item(), geom.dz, geom.dx, solver.dt)
        except ValueError:
            return float("inf")
        return loss(trial, check)

    weights = (0., 0.)
    v = net(*weights)
    full_losses = [loss(v, full)]
    records = []
    pause_ramp = False
    current_scale = 1.
    if grad_model is not None:
        training = grad_model.training
        grad_model.to(v.device).eval()
        if normalizer is None:
            normalizer = getattr(grad_model, "normalizer", None) or SignPreservingNormalizer()
    try:
        for k in range(cfg.n_iters):
            _synchronize(v.device)
            start = perf_counter()
            fraction = k / cfg.n_iters
            detail = max(0., min(1., (fraction - cfg.detail_unlock_frac) / .2,
                                 weights[0] + 1 / (.2 * cfg.n_iters)))
            proposal = weights if pause_ramp else (detail, float(fraction >= cfg.grid_unlock_frac))
            evaluation_v = v if proposal == weights else net(*proposal)
            adj = solver.adjoint_gradient(evaluation_v, obs, wave, geom, shots, f_cut=cfg.f_cut)
            grad = adj["grad"]
            mean_sigma = 0.
            decision = "full_adjoint"
            if grad_model is not None:
                pred, sigma = _predict(grad_model, normalizer, evaluation_v, initial, adj, obs,
                                       solver, shots, cfg, k)
                mean_sigma = sigma.mean().item()
                grad = pred * confidence_weight(sigma, cfg.kappa)
                decision = "predicted"
            if cfg.illumination_precond:
                illum = adj["illumination"]
                grad = grad / (illum / torch.quantile(illum, .9).clamp_min(1e-12) + .1)

            uncertain = grad_model is not None and mean_sigma > cfg.sigma_threshold
            net.project_gradient(_normalize_gradient(grad), *proposal)
            _synchronize(v.device)
            grad_end = perf_counter()
            before = (records[-1]["loss"] if records and proposal != weights else
                      _check_loss_before(adj, shots, check, obs, solver.dt, cfg.f_cut))
            step_scale = current_scale * (.5 if uncertain else 1.)
            trial = _verified_step(net, optimizer, proposal, step_scale, before, verify,
                                   cfg.max_backtracks, v, cfg.max_step_mps)
            pause_ramp = uncertain or trial is None
            if trial is None and cfg.enable_fallback and (grad_model is not None or cfg.illumination_precond):
                # Retry the measured, unweighted adjoint from the same linearization point.
                net.project_gradient(_normalize_gradient(adj["grad"]), *proposal)
                trial = _verified_step(net, optimizer, proposal, current_scale, before, verify,
                                       cfg.max_backtracks, v, cfg.max_step_mps)
                decision = "full_fallback" if grad_model is None else "sparse_fallback"
            if trial is None:
                after, decision = before, "rejected"
                current_scale *= .5
            else:
                v, after, attempt = trial
                weights = proposal
                if attempt:
                    current_scale = max(.1, current_scale * .5**attempt)
                else:
                    current_scale = min(1., current_scale * 1.25)
                if attempt and decision in ("predicted", "full_adjoint"):
                    decision = "backtracked"
            _synchronize(v.device)
            verified = perf_counter()
            metrics = velocity_metrics(v, truth)
            _synchronize(v.device)
            records.append({"loss_before": before, "loss": after, "check_shots": check,
                            "decision": decision, "detail_weight": weights[0], "grid_weight": weights[1],
                            "mean_sigma": mean_sigma, "velocity_metrics": metrics,
                            "grad_time": grad_end - start, "verify_time": verified - grad_end,
                            "iter_time": perf_counter() - start})
            if trial is None:
                full_losses.append(full_losses[-1])
            else:
                full_losses.append(after if check == full else loss(v, full))
    finally:
        if grad_model is not None:
            grad_model.train(training)

    return {"v_final": v, "v_background": net.components(*weights)["v_background"],
            "iterations": records, "loss_history": [r["loss"] for r in records],
            "full_loss_history": full_losses, "metric_history": [r["velocity_metrics"] for r in records],
            "decisions": [r["decision"] for r in records], "detail_weights": [r["detail_weight"] for r in records],
            **{key + "_mean": sum(r[key] for r in records) / max(len(records), 1)
               for key in ("grad_time", "verify_time", "iter_time")},
            "final_metrics": records[-1]["velocity_metrics"] if records else velocity_metrics(v, truth)}
