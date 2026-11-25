"""Source wavelets and zero-phase frequency filters along the time axis."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def ricker_wavelet(
    nt: int,
    dt: float,
    f0: float,
    t0: float | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Sample a Ricker wavelet peaking at ``t0`` (defaults to one period ``1 / f0``)."""
    if nt < 1 or dt <= 0.0 or f0 <= 0.0:
        raise ValueError("nt, dt, and f0 must be positive")
    t0 = 1.0 / f0 if t0 is None else t0
    t = torch.arange(nt, device=device, dtype=dtype) * dt
    arg = (math.pi * f0 * (t - t0)).square()
    return (1.0 - 2.0 * arg) * torch.exp(-arg)


def bandpass_filter(
    signal: Tensor,
    dt: float,
    f_low: float | None,
    f_high: float | None,
    order: int = 4,
) -> Tensor:
    """Apply a zero-phase Butterworth amplitude response along the last axis (Hz)."""
    if dt <= 0.0 or order <= 0:
        raise ValueError("dt and order must be positive")
    if f_low is not None and f_high is not None and f_low >= f_high:
        raise ValueError("f_low must be smaller than f_high")
    if not f_low and f_high is None:
        return signal

    work = signal if signal.dtype == torch.float64 else signal.float()
    nt = work.shape[-1]
    freq = torch.fft.rfftfreq(nt, d=dt, dtype=work.dtype, device=work.device)
    log_freq = freq.log()
    gain = torch.ones_like(freq)
    if f_low:
        gain = gain * torch.sigmoid(2.0 * order * (log_freq - math.log(f_low))).sqrt()
    if f_high is not None:
        gain = gain * torch.sigmoid(2.0 * order * (math.log(f_high) - log_freq)).sqrt()

    spectrum = torch.fft.rfft(work, dim=-1)
    return torch.fft.irfft(spectrum * gain, n=nt, dim=-1).to(signal.dtype)


def highpass_filter(signal: Tensor, dt: float, f_cut: float, order: int = 4) -> Tensor:
    """Attenuate energy below ``f_cut`` Hz with a zero-phase Butterworth filter."""
    return bandpass_filter(signal, dt, f_cut, None, order)
