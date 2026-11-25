"""Bounded multiscale convolutional parameterization of a 2-D velocity field."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _conv_layers(inputs, hidden, activation):
    head = nn.Conv2d(hidden, 1, 3, padding=1)
    nn.init.normal_(head.weight, std=1e-4)
    nn.init.zeros_(head.bias)
    return nn.Conv2d(inputs, hidden, 3, padding=1), activation(), head


class _LowBranch(nn.Sequential):
    def __init__(self, hidden, coarse_size):
        super().__init__(*_conv_layers(hidden + 2, hidden, nn.Tanh))
        self.latent = nn.Parameter(torch.randn(1, hidden, *coarse_size))

    def forward(self, coarse_coords):
        return super().forward(torch.cat((coarse_coords, self.latent), dim=1))


class MultiScaleVelocityReparam(nn.Module):
    """Add coarse background, full-resolution detail and grid corrections in logit space."""

    def __init__(
        self, v_init: Tensor, v_bounds: tuple[float, float] = (1500., 5000.),
        low_hidden: int = 32, high_hidden: int = 64, n_fourier: int = 32,
        fourier_scale: float = 2., max_grid_logit: float = .25, seed: int = 0,
    ):
        super().__init__()
        lower, upper = v_bounds
        if not 0 < lower < upper:
            raise ValueError("velocity bounds must be positive and increasing")
        nz, nx = v_init.shape
        v = v_init.detach()
        self.register_buffer("base_logit", ((v - lower) / (upper - lower)).clamp(1e-4, 1 - 1e-4).logit())
        self.register_buffer("_bounds", v.new_tensor(v_bounds))
        margin = (upper - lower) * 1e-4
        self.register_buffer("_initial_velocity", v.clamp(lower + margin, upper - margin))
        axes = [torch.linspace(-1, 1, n, dtype=v.dtype, device=v.device) for n in (nz, nx)]
        self.register_buffer("coords", torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1))
        coarse_size = (max(4, (nz + 3) // 4), max(4, (nx + 3) // 4))
        coords = self.coords.permute(2, 0, 1).unsqueeze(0)
        coarse_coords = F.interpolate(coords, size=coarse_size, mode="bilinear", align_corners=True)
        self.register_buffer("_coarse_coords", coarse_coords)
        # Initialize on CPU with a private RNG stream for reproducible CPU/CUDA comparisons.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            self.register_buffer("B", fourier_scale * torch.randn(2, n_fourier))
            self.low_branch = _LowBranch(low_hidden, coarse_size)
            self.high_branch = nn.Sequential(*_conv_layers(2 + 2 * n_fourier, high_hidden, nn.SiLU))
        self.max_grid_logit = max_grid_logit
        self.grid_residual = nn.Parameter(torch.zeros_like(v))
        self.to(v)
        phase = (2 * math.pi) * (self.coords @ self.B)
        features = torch.cat((self.coords, phase.sin(), phase.cos()), dim=-1)
        self.register_buffer("_high_features", features.permute(2, 0, 1).unsqueeze(0).contiguous())
        # Fixed offsets keep the initial corrections zero while preserving each branch's VJP.
        with torch.no_grad():
            self.register_buffer("initial_low", self._low_logits())
            self.register_buffer("initial_high", self._high_logits())

    def _velocity(self, logits):
        lower, upper = self._bounds.unbind()
        # Center the same sigmoid map at v_init to avoid a logit round-trip error at zero update.
        velocity = self._initial_velocity + (upper - lower) * (logits.sigmoid() - self.base_logit.sigmoid())
        # Even a saturated floating-point sigmoid must give an interior velocity.
        return velocity.clamp(torch.nextafter(lower, upper), torch.nextafter(upper, lower))

    def _low_logits(self):
        coarse = self.low_branch(self._coarse_coords)
        return F.interpolate(coarse, size=self.base_logit.shape, mode="bilinear", align_corners=True)[0, 0]

    def _high_logits(self):
        return self.high_branch(self._high_features)[0, 0]

    def _deltas(self, detail_weight, grid_weight):
        low = self._low_logits() - self.initial_low
        high = grid = torch.zeros_like(low)
        if detail_weight:
            high = detail_weight * (self._high_logits() - self.initial_high)
        if grid_weight:
            grid = grid_weight * self.max_grid_logit * self.grid_residual.tanh()
        return low, high, grid

    def forward(self, detail_weight: float = 1., grid_weight: float = 0.) -> Tensor:
        low, high, grid = self._deltas(detail_weight, grid_weight)
        return self._velocity(self.base_logit + low + high + grid)

    def components(self, detail_weight: float = 1., grid_weight: float = 0.) -> dict[str, Tensor]:
        """Return velocities in m/s and weighted additive corrections in logit space."""
        low, high, grid = self._deltas(detail_weight, grid_weight)
        background = self.base_logit + low
        return {"v_total": self._velocity(background + high + grid),
                "v_background": self._velocity(background),
                "delta_low": low, "delta_high": high, "delta_grid": grid}

    @torch.enable_grad()
    def project_gradient(self, g_v: Tensor, detail_weight: float = 1., grid_weight: float = 0.) -> list[Tensor]:
        """Replace parameter gradients by the VJP, in trainable parameter order."""
        params = [p for p in self.parameters() if p.requires_grad]
        velocity = self(detail_weight, grid_weight)
        grads = torch.autograd.grad(velocity, params, grad_outputs=g_v.detach(), allow_unused=True)
        for p, grad in zip(params, grads):
            p.grad = grad if grad is not None else torch.zeros_like(p)
        return [p.grad for p in params]
