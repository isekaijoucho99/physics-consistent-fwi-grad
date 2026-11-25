"""Waveform, Hilbert envelope, and instantaneous phase residuals."""

from __future__ import annotations

import torch
from torch import Tensor


def compute_hilbert_analytic(x: Tensor) -> Tensor:
    """Compute the discrete analytic signal ``x + i H(x)`` along the time axis."""
    work = x if x.dtype == torch.float64 else x.float()
    n = work.shape[-1]
    multiplier = work.new_zeros(n)
    multiplier[0] = 1.0
    multiplier[1 : (n + 1) // 2] = 2.0
    if n % 2 == 0:
        multiplier[n // 2] = 1.0
    return torch.fft.ifft(torch.fft.fft(work, dim=-1) * multiplier, dim=-1)


def compute_multimodal_residuals(
    d_syn: Tensor,
    d_obs: Tensor,
    eps: float = 1e-6,
) -> dict[str, Tensor]:
    """Compute waveform, envelope, and amplitude-weighted instantaneous phase residuals."""
    syn = compute_hilbert_analytic(d_syn)
    obs = compute_hilbert_analytic(d_obs)
    env_syn, env_obs = syn.abs(), obs.abs()

    amplitude = env_syn * env_obs
    rel_eps = eps * amplitude.amax().clamp_min(1e-12)
    weight = amplitude / (amplitude + rel_eps)

    real = syn.real * obs.real + syn.imag * obs.imag
    imag = syn.imag * obs.real - syn.real * obs.imag
    return {
        "r_wave": d_syn - d_obs,
        "r_env": env_syn - env_obs,
        "r_phase": weight * torch.atan2(imag, real),
    }
