"""Local image similarity and norm-based velocity/gradient evaluation."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def _metric_pair(pred: Tensor | np.ndarray, ref: Tensor | np.ndarray) -> tuple[Tensor, Tensor]:
    device = pred.device if isinstance(pred, Tensor) else (
        ref.device if isinstance(ref, Tensor) else torch.device("cpu")
    )
    # Local variances of nearly constant velocities benefit from double precision.
    dtype = torch.float32 if device.type == "mps" else torch.float64
    pred = torch.as_tensor(pred, dtype=dtype, device=device).detach()
    ref = torch.as_tensor(ref, dtype=dtype, device=device).detach()
    if pred.shape != ref.shape or pred.numel() == 0:
        raise ValueError("prediction and reference must have the same nonempty shape")
    return pred, ref


def _ssim(pred: Tensor, ref: Tensor) -> Tensor:
    """Mean Gaussian-window SSIM, shrinking the 11x11 window on small models."""
    def kernel(length: int) -> Tensor:
        radius = min(5, (length - 1) // 2)
        offsets = torch.arange(-radius, radius + 1, dtype=ref.dtype, device=ref.device)
        weights = torch.exp(-0.5 * (offsets / 1.5).square())
        return weights / weights.sum()

    window = torch.outer(kernel(ref.shape[0]), kernel(ref.shape[1]))[None, None]
    # Center first to avoid cancellation in regions of nearly constant velocity.
    center = ref.mean()
    p, r = pred - center, ref - center
    fields = torch.stack((p, r, p.square(), r.square(), p * r))[:, None]
    mp, mr, p2, r2, pr = F.conv2d(fields, window)[:, 0].unbind()
    vp, vr = (p2 - mp.square()).clamp_min(0), (r2 - mr.square()).clamp_min(0)
    cov = pr - mp * mr
    mp, mr = mp + center, mr + center
    span = ref.max() - ref.min()
    span = torch.where(span > 0, span, ref.abs().max().clamp_min(1))
    c1, c2 = (0.01 * span).square(), (0.03 * span).square()
    luminance = (2 * mp * mr + c1) / (mp.square() + mr.square() + c1)
    contrast = (2 * cov + c2) / (vp + vr + c2)
    return (luminance * contrast).mean().clamp(-1, 1)


def velocity_metrics(v_pred: Tensor | np.ndarray, v_true: Tensor | np.ndarray) -> dict[str, float]:
    """Compare [nz, nx] velocities using m/s and a dimensionless relative error.

    Norms in denominators and logarithms use a 1e-12 floor, keeping statistics
    finite for a perfect reconstruction and for an all-zero reference.
    """
    pred, ref = _metric_pair(v_pred, v_true)
    if pred.ndim != 2:
        raise ValueError("velocity metrics expect [nz, nx] velocity images")
    diff = pred - ref
    signal = torch.linalg.vector_norm(ref).clamp_min(1e-12)
    error = torch.linalg.vector_norm(diff)
    return {
        "mae": diff.abs().mean().item(),
        "rmse": diff.square().mean().sqrt().item(),
        "re": (error / signal).item(),
        "ssim": _ssim(pred, ref).item(),
        "snr_db": (20 * torch.log10(signal / error.clamp_min(1e-12))).item(),
    }


def gradient_metrics(g_pred: Tensor | np.ndarray, g_ref: Tensor | np.ndarray) -> dict[str, float]:
    """Compare gradient directions and relative magnitude. Zero vectors have cosine zero."""
    pred, ref = _metric_pair(g_pred, g_ref)
    p, r = pred.flatten(), ref.flatten()
    ref_norm = torch.linalg.vector_norm(r)
    denominator = torch.linalg.vector_norm(p) * ref_norm
    cosine = torch.dot(p, r) / torch.where(denominator > 0, denominator, 1)
    relative = torch.linalg.vector_norm(p - r) / torch.where(ref_norm > 0, ref_norm, 1e-12)
    return {"cosine_similarity": cosine.clamp(-1, 1).item(), "rel_l2_error": relative.item()}


def _pearson(x: Tensor, y: Tensor) -> float:
    # Subtract an observed value first so constant arrays center to exactly zero.
    x, y = x - x[0], y - y[0]
    x, y = x - x.mean(), y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    return (torch.dot(x, y) / torch.where(denominator > 0, denominator, 1)).clamp(-1, 1).item()


def _average_ranks(x: Tensor) -> Tensor:
    ordered, indices = x.sort()
    _, counts = torch.unique_consecutive(ordered, return_counts=True)
    # All tied values get the midpoint of their ranks (one-based).
    midpoints = counts.cumsum(0).to(x.dtype) - (counts.to(x.dtype) - 1) / 2
    ranks = torch.empty_like(x)
    return ranks.scatter_(0, indices, midpoints.repeat_interleave(counts))


def uncertainty_calibration_metrics(
    g_pred_norm: Tensor, g_ref_norm: Tensor, sigma: Tensor, eps: float = 1e-4,
) -> dict[str, float]:
    """Pixelwise scale/error correlations and Laplace NLL (omitting log 2).

    A constant error or scale has undefined correlation, reported here as zero.
    """
    pred, ref = _metric_pair(g_pred_norm, g_ref_norm)
    if sigma.shape != pred.shape or eps <= 0:
        raise ValueError("sigma must match the gradient shape and eps must be positive")
    error = (pred - ref).abs().flatten()
    sigma = sigma.detach().to(pred).flatten()
    scale = sigma + eps
    return {
        "pearson_r": _pearson(sigma, error),
        "spearman_r": _pearson(_average_ranks(sigma), _average_ranks(error)),
        "nll": (error / scale + scale.log()).mean().item(),
        "mean_sigma": sigma.mean().item(),
        "mean_abs_err": error.mean().item(),
    }
