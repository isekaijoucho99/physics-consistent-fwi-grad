"""Reversible gradient compression and spatial wavenumber decomposition."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


class SignPreservingNormalizer:
    """Map a spatial magnitude quantile to one, retaining outliers for inversion."""

    def __init__(self, quantile: float = 0.99, alpha: float = 10.0, eps: float = 1e-8):
        if not 0 <= quantile <= 1:
            raise ValueError("quantile must be in [0, 1]")
        if not all(math.isfinite(x) and x > 0 for x in (alpha, eps)):
            raise ValueError("alpha and eps must be finite and positive")
        self.quantile = quantile
        self.alpha = alpha
        self.eps = eps
        self._log_base = math.log1p(alpha)

    def compute_scale(self, g: Tensor) -> Tensor:
        """Return one broadcastable scale per image, shaped [..., 1, 1]."""
        flat = g.abs().flatten(start_dim=-2)
        if flat.dtype not in (torch.float32, torch.float64):
            flat = flat.float()
        return torch.quantile(flat, self.quantile, dim=-1, keepdim=True).unsqueeze(-1).clamp_min(self.eps)

    def _spatial_scale(self, scale: Tensor, field: Tensor) -> Tensor:
        # Dataset samples store a scalar as [1], so a collated scale is [B, 1].
        shape = scale.shape + (1,) * (field.ndim - scale.ndim)
        return scale.reshape(shape).clamp_min(self.eps)

    def normalize(self, g: Tensor, scale: Tensor | None = None) -> tuple[Tensor, Tensor]:
        scale = self.compute_scale(g) if scale is None else self._spatial_scale(scale, g)
        magnitude = torch.log1p(self.alpha * (g.abs() / scale)) / self._log_base
        return g.sign() * magnitude, scale

    def denormalize(self, g_norm: Tensor, scale: Tensor) -> Tensor:
        scale = self._spatial_scale(scale, g_norm)
        magnitude = torch.expm1(g_norm.abs() * self._log_base) / self.alpha
        return g_norm.sign() * magnitude * scale


def split_wavenumber(field: Tensor, sigma_low: float = 3.0) -> tuple[Tensor, Tensor]:
    """Gaussian low-pass and its complementary high-pass on the last two axes."""
    if not math.isfinite(sigma_low) or sigma_low < 0:
        raise ValueError("sigma_low must be finite and nonnegative")
    if sigma_low == 0:
        return field, torch.zeros_like(field)
    radius = math.ceil(3 * sigma_low)
    offsets = torch.arange(-radius, radius + 1, device=field.device, dtype=field.dtype)
    weights = torch.exp(-0.5 * (offsets / sigma_low).square())
    weights = weights / weights.sum()
    images = field.reshape(-1, 1, *field.shape[-2:])
    low = F.conv2d(F.pad(images, (radius, radius, 0, 0), mode="replicate"), weights.view(1, 1, 1, -1))
    low = F.conv2d(F.pad(low, (0, 0, radius, radius), mode="replicate"), weights.view(1, 1, -1, 1))
    low = low.reshape_as(field)
    return low, field - low
