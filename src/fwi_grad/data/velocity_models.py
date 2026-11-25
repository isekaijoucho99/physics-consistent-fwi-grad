"""Velocity models in metres per second and acquisition noise at a target SNR."""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def load_openfwi_velocity(
    npy_path: str | Path, index: int = 0, target_shape: tuple[int, int] | None = None
) -> np.ndarray:
    """Load a [z, x] section or one image from a channel-first OpenFWI stack.

    OpenFWI distributes velocities in m/s. Also accept geophysical models
    stored in km/s when their maximum velocity is below 20.
    """
    models = np.load(npy_path, mmap_mode="r", allow_pickle=False)
    if models.ndim == 2:
        models = models[None]
    if models.ndim == 4 and models.shape[1] == 1:
        data = models[index, 0]
    elif models.ndim == 3:
        data = models[index]
    else:
        raise ValueError("expected a velocity array shaped [H, W], [N, H, W] or [N, 1, H, W]")
    velocity = np.array(data, dtype=np.float32, copy=True)
    if velocity.size == 0 or not np.isfinite(velocity).all() or velocity.min() <= 0:
        raise ValueError("velocity must contain finite positive values")
    if velocity.max() < 20:
        velocity *= 1000
    if target_shape is not None:
        if len(target_shape) != 2 or min(target_shape) < 1:
            raise ValueError("target_shape must specify positive (nz, nx)")
        if tuple(velocity.shape) != tuple(target_shape):
            image = torch.from_numpy(velocity)[None, None]
            velocity = F.interpolate(
                image, size=target_shape, mode="bilinear", align_corners=False, antialias=True
            )[0, 0].numpy()
    return velocity


def generate_synthetic_complex_model(nz: int, nx: int, seed: int = 0) -> np.ndarray:
    """Generate folded sedimentary layers cut by a fault around a salt body."""
    if min(nz, nx) < 1:
        raise ValueError("nz and nx must be positive")
    rng = np.random.default_rng(seed)
    z, x = np.ogrid[0:1:complex(nz), 0:1:complex(nx)]
    frequency = rng.uniform(0.8, 1.5)
    phase = rng.uniform(-np.pi, np.pi)
    fold = (0.035 + 0.035 * z) * np.sin(2 * np.pi * frequency * x + phase)
    dip = rng.uniform(-0.08, 0.08) * (x - 0.5)
    fault = x > rng.uniform(0.40, 0.68) + rng.uniform(-0.18, 0.18) * z
    depth = z + fold + dip - rng.uniform(0.07, 0.14) * fault

    interfaces = np.linspace(0.09, 0.89, 7) + rng.uniform(-0.025, 0.025, 7)
    contrasts = rng.uniform(100, 240, len(interfaces))
    velocity = 1530 + 1050 * z + np.zeros_like(x)
    for interface, contrast in zip(interfaces, contrasts):
        velocity += contrast * (depth > interface)

    cx, cz = rng.uniform(0.28, 0.74), rng.uniform(0.48, 0.76)
    rx, rz = rng.uniform(0.12, 0.23), rng.uniform(0.10, 0.20)
    salt_topography = 0.022 * np.sin(4 * np.pi * x + phase)
    inside = ((x - cx) / rx) ** 2 + ((z - cz + salt_topography) / rz) ** 2 <= 1
    velocity[inside] = rng.uniform(4150, 4500)
    return np.clip(velocity, 1500, 4500).astype(np.float32)


def build_initial_model(
    v_true: np.ndarray | Tensor,
    method: str = "gaussian",
    sigma: float = 8.0,
    v_min: float | None = None,
    v_max: float | None = None,
) -> np.ndarray:
    """Smooth the model, or broadcast a linear velocity profile across columns.

    A linear profile spans the given bounds or the extrema of v_true. For a
    smoothed model the optional bounds clip the result. Sigma is in grid cells.
    """
    image = torch.as_tensor(v_true).detach().to(device="cpu", dtype=torch.float32)
    if image.ndim != 2 or image.numel() == 0 or not torch.isfinite(image).all():
        raise ValueError("v_true must be a nonempty finite [nz, nx] image")
    low = image.min().item() if v_min is None else v_min
    high = image.max().item() if v_max is None else v_max
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError("velocity bounds must be finite and ordered")
    if method == "linear":
        profile = torch.linspace(low, high, image.shape[0], dtype=image.dtype)
        return profile[:, None].expand_as(image).contiguous().numpy()
    if method != "gaussian":
        raise ValueError("method must be 'gaussian' or 'linear'")
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    if sigma > 0:
        radius = math.ceil(3 * sigma)
        offsets = torch.arange(-radius, radius + 1, dtype=image.dtype)
        weights = torch.exp(-0.5 * (offsets / sigma).square())
        weights = weights / weights.sum()
        image = F.pad(image[None, None], (radius, radius, 0, 0), mode="replicate")
        image = F.conv2d(image, weights.view(1, 1, 1, -1))
        image = F.pad(image, (0, 0, radius, radius), mode="replicate")
        image = F.conv2d(image, weights.view(1, 1, -1, 1))[0, 0]
    return image.clamp(low, high).numpy()


def add_gaussian_noise(d_obs: Tensor, snr_db: float, seed: int | None = None) -> Tensor:
    """Set the realised whole-gather signal/noise power ratio in decibels.

    A supplied seed draws noise using a private CPU generator so its samples
    reproduce across accelerators and leave the global random stream intact.
    """
    if not d_obs.is_floating_point() or d_obs.numel() == 0:
        raise ValueError("d_obs must be a nonempty real floating-point tensor")
    if snr_db == math.inf:
        return d_obs.clone()
    if not math.isfinite(snr_db):
        raise ValueError("snr_db must be finite or positive infinity")
    dtype = torch.float64 if d_obs.dtype == torch.float64 else torch.float32
    device = d_obs.device if seed is None else torch.device("cpu")
    generator = None if seed is None else torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(d_obs.shape, dtype=dtype, device=device, generator=generator)
    noise = noise.to(d_obs.device)
    signal_norm = torch.linalg.vector_norm(d_obs.to(dtype))
    noise_norm = torch.linalg.vector_norm(noise).clamp_min(torch.finfo(dtype).tiny)
    scale = (signal_norm / noise_norm) * (10.0 ** (-snr_db / 20.0))
    return (d_obs + noise * scale).to(d_obs.dtype)
