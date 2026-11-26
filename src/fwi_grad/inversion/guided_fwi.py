"""Uncertainty-gated closed-loop FWI with out-of-sample few-shot verification."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter

import torch
from torch import Tensor, nn

from ..eval.metrics import velocity_metrics
from ..losses.physics_losses import confidence_weight, subset_misfit
from ..models.conditioning import encode_condition, propose_update
from ..models.normalization import SignPreservingNormalizer
from ..physics.acoustic2d import Acoustic2DSolver, check_cfl
from ..physics.geometry import AcquisitionGeometry, select_sparse_shots
from ..physics.residuals import compute_multimodal_residuals
from ..physics.wavelet import highpass_filter
from .baseline_fwi import _synchronize


@dataclass
class GuidedFWIConfig:
    """Configuration for uncertainty-gated closed-loop FWI."""

    n_iters: int = 10
    lr: float = 5.0
    v_bounds: tuple[float, float] = (1500.0, 5000.0)
    kappa: float = 2.0
    sigma_threshold: float = 0.5
    n_check_shots: int = 2
    max_backtracks: int = 3
    max_update_mps: float = 150.0
    sparse_blend_weight: float = 0.5
    enable_fallback: bool = True
    f_cut: float = 0.0
    sparse_ratio: float = 0.25
    seed: int = 0
    optimizer: str = "sgd"

    def __post_init__(self) -> None:
        if self.n_iters < 0 or self.n_check_shots < 1 or self.max_backtracks < 0:
            raise ValueError("invalid iteration, check-shot, or backtrack count")
        if self.lr <= 0.0 or self.max_update_mps <= 0.0:
            raise ValueError("lr and max_update_mps must be positive")
        if not 0.0 < self.v_bounds[0] < self.v_bounds[1]:
            raise ValueError("v_bounds must be positive and increasing")
        if self.optimizer.lower() not in ("adam", "sgd"):
            raise ValueError("optimizer must be 'adam' or 'sgd'")


@dataclass
class _AdamState:
    """Reversible per-pixel Adam moments for safeguarded updates."""

    m: Tensor
    v_sq: Tensor
    step: int = 0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def propose(self, grad: Tensor, conf: Tensor | float, lr: float, max_update_mps: float):
        t = self.step + 1
        m_new = self.beta1 * self.m + (1.0 - self.beta1) * grad
        v_new = self.beta2 * self.v_sq + (1.0 - self.beta2) * (grad * grad)
        m_hat = m_new / (1.0 - self.beta1**t)
        v_hat = v_new / (1.0 - self.beta2**t)
        step = lr * conf * (m_hat / (v_hat.sqrt() + self.eps))
        peak = step.abs().amax()
        if peak > max_update_mps:
            step = step * (max_update_mps / peak)
        return step, (m_new, v_new, t)

    def commit(self, next_state: tuple[Tensor, Tensor, int]) -> None:
        self.m, self.v_sq, self.step = next_state


@dataclass
class StepResult:
    """Outcome of one verified velocity update."""

    velocity: Tensor
    loss_before: float
    loss_after: float
    decision: str
    check_shots: list[int]
    next_lr: float


def _select_check_shots(
    n_shots: int,
    sparse_shots: Sequence[int],
    n_check_shots: int,
    seed: int,
) -> list[int]:
    """Select verification shots from the complement of ``sparse_shots`` when possible."""
    sparse_set = set(sparse_shots)
    complement = [s for s in range(n_shots) if s not in sparse_set]
    pool = complement if complement else list(sparse_shots)
    count = min(n_check_shots, len(pool))
    picked = select_sparse_shots(len(pool), count, mode="random", seed=seed)
    return [pool[i] for i in picked]


def _predict(
    model: nn.Module,
    normalizer: SignPreservingNormalizer,
    v: Tensor,
    initial: Tensor,
    sparse: dict,
    observed: Tensor,
    solver: Acoustic2DSolver,
    shots: Sequence[int],
    config: GuidedFWIConfig,
    iter_k: int,
) -> tuple[Tensor, Tensor]:
    """Run ``model`` on the current sparse-shot state and denormalize to physical units."""
    syn = sparse["d_syn"]
    obs = observed[list(shots)]
    if config.f_cut > 0.0:
        syn = highpass_filter(syn, solver.dt, config.f_cut)
        obs = highpass_filter(obs, solver.dt, config.f_cut)
    residuals = compute_multimodal_residuals(syn, obs)

    shot_fraction = len(shots) / observed.shape[0]
    q_k = encode_condition(config.f_cut, iter_k, shot_fraction, device=v.device, dtype=v.dtype)
    inputs = {
        "v_k": v[None, None],
        "v_init": initial[None, None],
        "r_wave": residuals["r_wave"][None],
        "r_env": residuals["r_env"][None],
        "r_phase": residuals["r_phase"][None],
        "illumination": sparse["illumination"][None, None],
        "g_sparse": sparse["grad"][None, None],
        "q_k": q_k[None],
    }
    param_dtype = next(model.parameters(), v).dtype
    inputs = {key: value.to(dtype=param_dtype) for key, value in inputs.items()}
    mask = torch.ones((1, len(shots)), dtype=torch.bool, device=v.device)
    out = model(**inputs, shot_mask=mask)

    scale = normalizer.compute_scale(sparse["grad"])
    grad_phys = normalizer.denormalize(out["g_pred"][0, 0].to(v), scale)
    return grad_phys, out["sigma"][0, 0].to(v)


def _verify_step(
    v: Tensor,
    pred: Tensor,
    sigma: Tensor,
    sparse_grad: Tensor,
    check_shots: list[int],
    cached_before: float | None,
    d_obs: Tensor,
    wavelet: Tensor,
    geom: AcquisitionGeometry,
    solver: Acoustic2DSolver,
    config: GuidedFWIConfig,
    mean_sigma: float,
    current_lr: float,
    adam: _AdamState | None = None,
    surrogate_weight: float = 1.0,
) -> StepResult:
    """Verify a candidate update on ``check_shots`` with backtracking and fallback."""
    if cached_before is not None:
        before = cached_before
    else:
        before = subset_misfit(v, d_obs, solver, wavelet, geom, check_shots, config.f_cut).item()

    def try_direction(grad: Tensor, lr: float, conf: Tensor | float, backtracks: int, use_adam: bool = True):
        next_adam = None
        if adam is not None and use_adam:
            step, next_adam = adam.propose(grad, conf, lr, config.max_update_mps)
        elif adam is not None:
            # Keep Adam's diagonal preconditioner on fallback steps so depth sensitivity stays balanced.
            t = adam.step + 1
            v_new = adam.beta2 * adam.v_sq + (1.0 - adam.beta2) * (grad * grad)
            v_hat = v_new / (1.0 - adam.beta2**t)
            step = lr * conf * (grad / (v_hat.sqrt() + adam.eps))
            peak = step.abs().amax()
            if peak > config.max_update_mps:
                step = step * (config.max_update_mps / peak)
            next_adam = ((1.0 - adam.beta1) * grad, v_new, t)
        else:
            step = propose_update(grad, conf, lr=lr, max_update_mps=config.max_update_mps)
        if not torch.isfinite(step).all().item():
            return None
        for attempt in range(backtracks + 1):
            trial = (v - step * (0.5**attempt)).clamp(*config.v_bounds)
            try:
                check_cfl(trial.max().item(), geom.dz, geom.dx, solver.dt)
            except ValueError:
                continue
            after = subset_misfit(
                trial, d_obs, solver, wavelet, geom, check_shots, config.f_cut
            ).item()
            if math.isfinite(after) and after < before:
                if adam is not None:
                    if next_adam is not None:
                        adam.commit(next_adam)
                    else:
                        adam.m.zero_()
                return trial, after, attempt
        return None

    lr = current_lr * (0.5 if mean_sigma > config.sigma_threshold else 1.0)
    conf = confidence_weight(sigma, config.kappa)
    if adam is not None:
        # Mean-normalize confidence so uncertainty redistributes trust without shrinking Adam step size.
        conf = (conf / conf.mean().clamp_min(1e-6)).clamp(0.2, 1.0)
        conf = (1.0 - surrogate_weight) + surrogate_weight * conf
    result = try_direction(pred, lr, conf, config.max_backtracks, use_adam=True)
    decision = "predicted"
    if result is not None and result[2] > 0:
        decision = "backtracked"
    if result is None:
        w = config.sparse_blend_weight
        result = try_direction((1.0 - w) * pred + w * sparse_grad, lr, 1.0, 1, use_adam=False)
        decision = "sparse_blend"
    if result is None and config.enable_fallback:
        full_grad = solver.adjoint_gradient(v, d_obs, wavelet, geom, f_cut=config.f_cut)["grad"]
        lr = current_lr
        result = try_direction(full_grad, lr, 1.0, config.max_backtracks, use_adam=False)
        decision = "full_fallback"

    if result is None:
        if adam is not None:
            adam.m.zero_()
        floor = (0.35 if adam is not None else 0.05) * config.lr
        next_lr = max(floor, current_lr * 0.5)
        return StepResult(v, before, before, "rejected", check_shots, next_lr)

    trial, after, attempt = result
    if attempt > 0:
        if adam is not None:
            next_lr = max(0.35 * config.lr, current_lr * 0.85)
        else:
            next_lr = max(0.05 * config.lr, lr * (0.5**attempt))
    else:
        next_lr = min(current_lr * 1.25, config.lr)
    return StepResult(trial, before, after, decision, check_shots, next_lr)


@torch.no_grad()
def run_physics_guided_fwi(
    v_init: Tensor,
    v_true: Tensor,
    d_obs: Tensor,
    model: nn.Module,
    solver: Acoustic2DSolver,
    wavelet: Tensor,
    geom: AcquisitionGeometry,
    config: GuidedFWIConfig,
    normalizer: SignPreservingNormalizer | None = None,
    device: torch.device | str | None = None,
    *,
    on_step: Callable[[Tensor], None] | None = None,
) -> dict:
    """Run closed-loop FWI guided by ``model`` with out-of-sample few-shot verification."""
    v = v_init.detach().to(device=device).clone()
    initial = v.clone()
    truth = v_true.to(v)
    wavelet = wavelet.to(v)
    observed = d_obs.detach().to(v)
    if observed.shape[0] != geom.n_shots:
        raise ValueError("guided FWI requires observations for all acquisition shots")

    sparse_shots = select_sparse_shots(geom.n_shots, float(config.sparse_ratio))
    check_shots = _select_check_shots(geom.n_shots, sparse_shots, config.n_check_shots, config.seed)
    if normalizer is None:
        normalizer = getattr(model, "normalizer", None) or SignPreservingNormalizer()

    records: list[dict] = []
    current_lr = config.lr
    cached_check_loss: float | None = None
    init_check_loss: float | None = None
    adam = (
        _AdamState(torch.zeros_like(v), torch.zeros_like(v))
        if config.optimizer.lower() == "adam"
        else None
    )
    was_training = model.training
    model.to(v.device).eval()

    try:
        for k in range(config.n_iters):
            _synchronize(v.device)
            start = perf_counter()
            sparse = solver.adjoint_gradient(v, observed, wavelet, geom, sparse_shots, f_cut=config.f_cut)
            _synchronize(v.device)
            sparse_end = perf_counter()

            pred, sigma = _predict(
                model, normalizer, v, initial, sparse, observed, solver, sparse_shots, config, k
            )
            mean_sigma = sigma.mean().item()
            _synchronize(v.device)
            net_end = perf_counter()

            # Reuse check loss from the previous accepted step (or sparse loss if check == sparse).
            if cached_check_loss is None and check_shots == sparse_shots:
                cached_check_loss = sparse["loss"]
            if adam is not None and init_check_loss is not None and cached_check_loss is not None:
                w_k = min(1.0, max(0.0, cached_check_loss / max(init_check_loss, 1e-12)))
            else:
                w_k = 1.0
            pred_eff = sparse["grad"] + w_k * (pred - sparse["grad"]) if w_k < 1.0 else pred
            mean_sigma_eff = mean_sigma * w_k
            step = _verify_step(
                v,
                pred_eff,
                sigma,
                sparse["grad"],
                check_shots,
                cached_check_loss,
                observed,
                wavelet,
                geom,
                solver,
                config,
                mean_sigma_eff,
                current_lr,
                adam=adam,
                surrogate_weight=w_k,
            )
            if init_check_loss is None:
                init_check_loss = step.loss_before
            v = step.velocity
            current_lr = step.next_lr
            cached_check_loss = step.loss_after
            _synchronize(v.device)
            verified = perf_counter()

            metrics = velocity_metrics(v, truth)
            _synchronize(v.device)
            records.append(
                {
                    "loss_before": step.loss_before,
                    "loss": step.loss_after,
                    "check_shots": step.check_shots,
                    "velocity_metrics": metrics,
                    "decision": step.decision,
                    "mean_sigma": mean_sigma,
                    "sparse_time": sparse_end - start,
                    "net_time": net_end - sparse_end,
                    "grad_time": net_end - start,
                    "verify_time": verified - net_end,
                    "iter_time": perf_counter() - start,
                }
            )
            if on_step is not None:
                on_step(v)
    finally:
        model.train(was_training)

    decisions = [row["decision"] for row in records]
    count = max(len(records), 1)
    return {
        "v_final": v,
        "iterations": records,
        "loss_history": [row["loss"] for row in records],
        "metric_history": [row["velocity_metrics"] for row in records],
        "decisions": decisions,
        "descent_success_rate": sum(d != "rejected" for d in decisions) / count,
        "predicted_accept_rate": decisions.count("predicted") / count,
        "grad_time_mean": sum(row["grad_time"] for row in records) / count,
        "verify_time_mean": sum(row["verify_time"] for row in records) / count,
        "iter_time_mean": sum(row["iter_time"] for row in records) / count,
        "final_metrics": records[-1]["velocity_metrics"] if records else velocity_metrics(v, truth),
    }
