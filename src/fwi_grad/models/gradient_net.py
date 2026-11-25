"""Three-branch multi-modal network predicting FWI gradients and uncertainty."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .normalization import SignPreservingNormalizer, split_wavenumber


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups != 0:
            groups -= 1
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
        )
        if in_channels == out_channels and stride == 1:
            self.skip: nn.Module = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return F.silu(self.body(x) + self.skip(x))


class _SpatialEncoder(nn.Module):
    """Three-scale residual pyramid returning full, half, and quarter resolution maps."""

    def __init__(self, in_channels: int, width: int) -> None:
        super().__init__()
        self.full = _ResidualBlock(in_channels, width)
        self.middle = _ResidualBlock(width, 2 * width, stride=2)
        self.coarse = _ResidualBlock(2 * width, 4 * width, stride=2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        full = self.full(x)
        middle = self.middle(full)
        return full, middle, self.coarse(middle)


class _ShotEncoder(nn.Module):
    """Encode multi-modal shot gathers and aggregate across shots via masked mean-max pooling."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            _ResidualBlock(3, width, stride=2),
            _ResidualBlock(width, 2 * width, stride=2),
        )

    def forward(self, gathers: Tensor, shot_mask: Tensor, size: tuple[int, int]) -> Tensor:
        batch, shots = shot_mask.shape
        valid = shot_mask[:, :, None, None, None]
        gathers = gathers.masked_fill(~valid, 0.0)
        # Normalize each residual channel per shot over receivers and time.
        scale = gathers.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        gathers = gathers / scale
        # Transpose [n_rec, nt] -> [nt, n_rec] so time aligns with depth (z) and receivers with x.
        gathers = gathers.transpose(-2, -1).contiguous()
        encoded = self.encoder(gathers.flatten(0, 1))
        features = F.adaptive_avg_pool2d(encoded, size).reshape(batch, shots, -1, *size)

        count = valid.sum(dim=1)
        mean = features.masked_fill(~valid, 0.0).sum(dim=1) / count.clamp_min(1)
        maximum = features.masked_fill(~valid, -torch.inf).amax(dim=1)
        return torch.cat((mean, maximum.masked_fill(count == 0, 0.0)), dim=1)


def _modulate(x: Tensor, film: Tensor) -> Tensor:
    gamma, beta = film[:, :, None, None].chunk(2, dim=1)
    return (1.0 + gamma) * x + beta


def _upsample(x: Tensor, target: Tensor) -> Tensor:
    return F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)


class MultiModalGradientNet(nn.Module):
    """Predict normalized full-shot FWI gradients and pixelwise aleatoric uncertainty.

    Fuses three domains at the coarsest scale:
    1. Model domain: current velocity ``v_k`` and cumulative deviation ``v_k - v_init``.
    2. Data domain: per-shot waveform, envelope, and phase residuals.
    3. Auxiliary physics domain: acquisition illumination, sparse adjoint gradient, and grid coordinates.
    """

    def __init__(
        self,
        base_channels: int = 32,
        cond_dim: int = 32,
        eps_sigma: float = 1e-3,
        alpha: float = 10.0,
        disable_model_branch: bool = False,
        disable_data_branch: bool = False,
    ) -> None:
        super().__init__()
        if not math.isfinite(eps_sigma) or eps_sigma <= 0.0:
            raise ValueError("eps_sigma must be positive")
        self.eps_sigma = eps_sigma
        self.disable_model_branch = disable_model_branch
        self.disable_data_branch = disable_data_branch
        self.normalizer = SignPreservingNormalizer(alpha=alpha)

        width = base_channels
        self.model_encoder = _SpatialEncoder(2, width)
        self.data_encoder = _ShotEncoder(width)
        self.aux_encoder = _SpatialEncoder(4, width)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(12 * width, width, 1),
            nn.SiLU(),
            nn.Conv2d(width, 12 * width, 1),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(12 * width, 4 * width, 1),
            _ResidualBlock(4 * width, 4 * width),
        )
        self.film_sizes = (8 * width, 4 * width, 2 * width)
        self.film = nn.Sequential(
            nn.Linear(3, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, sum(self.film_sizes)),
        )
        self.mid_decoder = _ResidualBlock(8 * width, 2 * width)
        self.full_decoder = _ResidualBlock(4 * width, width)
        self.low_head = nn.Conv2d(2 * width, 1, 1)
        self.high_head = nn.Conv2d(width, 1, 1, bias=False)
        self.gate = nn.Sequential(nn.Linear(3, 1), nn.Sigmoid())
        self.sigma_head = nn.Conv2d(width, 1, 1)
        nn.init.normal_(self.low_head.weight, std=1e-3)
        nn.init.zeros_(self.low_head.bias)
        nn.init.normal_(self.high_head.weight, std=1e-3)

    def forward(
        self,
        v_k: Tensor,
        v_init: Tensor,
        r_wave: Tensor,
        r_env: Tensor,
        r_phase: Tensor,
        shot_mask: Tensor,
        illumination: Tensor,
        g_sparse: Tensor,
        q_k: Tensor,
    ) -> dict[str, Tensor]:
        """Run a forward pass and return ``g_pred``, ``g_low``, ``g_high``, and ``sigma``."""
        v_mean = v_init.detach().mean(dim=(-2, -1), keepdim=True).clamp_min(100.0)
        velocity = torch.cat((v_k / v_mean, 5.0 * (v_k - v_init) / v_mean), dim=1)
        m_full, m_mid, m_coarse = self.model_encoder(velocity)
        if self.disable_model_branch:
            m_full = torch.zeros_like(m_full)
            m_mid = torch.zeros_like(m_mid)
            m_coarse = torch.zeros_like(m_coarse)

        batch, _, nz, nx = v_k.shape
        coords = torch.zeros((batch, 2, nz, nx), device=v_k.device, dtype=v_k.dtype)

        illum_scale = torch.quantile(illumination.flatten(2), 0.99, dim=-1)[:, :, None, None]
        illum_norm = torch.log1p(illumination / (illum_scale + self.normalizer.eps))
        sparse_norm, _ = self.normalizer.normalize(g_sparse)
        a_full, a_mid, a_coarse = self.aux_encoder(torch.cat((illum_norm, sparse_norm, coords), dim=1))

        if self.disable_data_branch:
            data = torch.zeros_like(m_coarse)
        else:
            gathers = torch.stack((r_wave, r_env, r_phase), dim=2)
            data = self.data_encoder(gathers, shot_mask, m_coarse.shape[-2:])

        q_cond = torch.zeros_like(q_k)
        q_cond[:, 0] = q_k[:, 0]
        coarse_film, mid_film, full_film = self.film(q_cond).split(self.film_sizes, dim=1)
        combined = torch.cat((m_coarse, data, a_coarse), dim=1)
        bottleneck = _modulate(self.fusion(combined * self.attention(combined)), coarse_film)

        mid = self.mid_decoder(torch.cat((_upsample(bottleneck, m_mid), m_mid, a_mid), dim=1))
        mid = _modulate(mid, mid_film)
        full = self.full_decoder(torch.cat((_upsample(mid, m_full), m_full, a_full), dim=1))
        full = _modulate(full, full_film)

        low = _upsample(self.low_head(mid), full)
        _, high = split_wavenumber(self.high_head(full))
        gradient = sparse_norm + low + self.gate(q_cond)[:, :, None, None] * high
        sigma = F.softplus(self.sigma_head(full)) + self.eps_sigma
        return {
            "g_pred": gradient,
            "g_low": low,
            "g_high": high,
            "sigma": sigma,
        }
