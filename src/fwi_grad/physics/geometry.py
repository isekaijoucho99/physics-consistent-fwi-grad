"""Integer-grid surface acquisition geometry for a common receiver spread."""

from dataclasses import dataclass
import math
from numbers import Integral, Real

import torch
from torch import Tensor


def _indices(values: Tensor, limit: int, name: str) -> Tensor:
    coords = torch.as_tensor(values)
    if coords.ndim != 1 or coords.numel() == 0:
        raise ValueError(f"{name} must be a nonempty 1D array")
    if coords.is_floating_point() or coords.is_complex() or coords.dtype == torch.bool:
        raise ValueError(f"{name} must contain integer grid indices")
    coords = coords.to(torch.long)
    if torch.any((coords < 0) | (coords >= limit)):
        raise ValueError(f"{name} contains a grid index outside [0, {limit})")
    return coords


@dataclass
class AcquisitionGeometry:
    """Source and receiver indices in the physical, unpadded [z, x] grid."""

    nz: int
    nx: int
    dz: float
    dx: float
    src_z: Tensor
    src_x: Tensor
    rec_z: Tensor
    rec_x: Tensor

    def __post_init__(self) -> None:
        if min(self.nz, self.nx) < 1:
            raise ValueError("grid dimensions must be positive")
        if not all(math.isfinite(d) and d > 0 for d in (self.dz, self.dx)):
            raise ValueError("grid spacings must be finite and positive")
        self.src_z = _indices(self.src_z, self.nz, "src_z")
        self.src_x = _indices(self.src_x, self.nx, "src_x")
        self.rec_z = _indices(self.rec_z, self.nz, "rec_z")
        self.rec_x = _indices(self.rec_x, self.nx, "rec_x")
        if self.src_x.shape != self.src_z.shape or self.rec_x.shape != self.rec_z.shape:
            raise ValueError("each x array must match its corresponding z array")

    @property
    def n_shots(self) -> int:
        return self.src_x.numel()

    @property
    def n_rec(self) -> int:
        return self.rec_x.numel()


def _uniform_indices(size: int, count: int) -> Tensor:
    if not 1 <= count <= size:
        raise ValueError(f"count must be in [1, {size}] for unique grid indices")
    if count == 1:
        return torch.tensor([size // 2], dtype=torch.long)
    return torch.linspace(0, size - 1, count, dtype=torch.float64).round().long()


def create_surface_geometry(
    nz: int,
    nx: int,
    dz: float,
    dx: float,
    n_shots: int,
    n_rec: int,
    src_depth_idx: int = 2,
    rec_depth_idx: int = 2,
    margin: int = 4,
) -> AcquisitionGeometry:
    """Distribute sensors uniformly between the left and right lateral margins."""
    if margin < 0 or 2 * margin >= nx:
        raise ValueError("margin must leave at least one interior column")
    src_x = margin + _uniform_indices(nx - 2 * margin, n_shots)
    rec_x = margin + _uniform_indices(nx - 2 * margin, n_rec)
    return AcquisitionGeometry(
        nz, nx, dz, dx,
        torch.full((n_shots,), src_depth_idx), src_x,
        torch.full((n_rec,), rec_depth_idx), rec_x,
    )


def select_sparse_shots(
    n_shots: int, ratio_or_count: float | int, mode: str = "uniform", seed: int | None = None
) -> list[int]:
    """Select sorted unique shots by integer count or fraction rounded to nearest."""
    if n_shots < 1:
        raise ValueError("n_shots must be positive")
    if isinstance(ratio_or_count, Integral) and not isinstance(ratio_or_count, bool):
        count = int(ratio_or_count)
    elif isinstance(ratio_or_count, Real) and not isinstance(ratio_or_count, bool):
        if not 0 < ratio_or_count <= 1:
            raise ValueError("a sparse shot fraction must be in (0, 1]")
        count = max(1, math.floor(n_shots * ratio_or_count + 0.5))
    else:
        raise ValueError("ratio_or_count must be a fraction or integer count")
    if not 1 <= count <= n_shots:
        raise ValueError("shot count must be between one and n_shots")
    if mode == "uniform":
        return _uniform_indices(n_shots, count).tolist()
    if mode == "random":
        generator = None if seed is None else torch.Generator().manual_seed(seed)
        return torch.randperm(n_shots, generator=generator)[:count].sort().values.tolist()
    raise ValueError("mode must be 'uniform' or 'random'")


def geometry_encoding(geom: AcquisitionGeometry, device: torch.device | str | None = None) -> Tensor:
    """Count sources (channel 0) and receivers (channel 1) at each grid point."""
    device = geom.src_z.device if device is None else device
    encoding = torch.zeros((2, geom.nz * geom.nx), dtype=torch.float32, device=device)
    for channel, (z, x) in enumerate(((geom.src_z, geom.src_x), (geom.rec_z, geom.rec_x))):
        flat = z.to(device) * geom.nx + x.to(device)
        encoding[channel].scatter_add_(0, flat, torch.ones_like(flat, dtype=encoding.dtype))
    return encoding.reshape(2, geom.nz, geom.nx)
