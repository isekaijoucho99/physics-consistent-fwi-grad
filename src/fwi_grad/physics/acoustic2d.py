"""Differentiable constant-density acoustic propagation on a Cartesian grid."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .geometry import AcquisitionGeometry
from .residuals import compute_hilbert_analytic
from .wavelet import highpass_filter


def check_cfl(v_max: float, dz: float, dx: float, dt: float) -> None:
    """Check the leapfrog bound for the fourth-order axial Laplacian."""
    if not all(math.isfinite(a) and a > 0 for a in (v_max, dz, dx, dt)):
        raise ValueError("CFL check requires positive finite velocity, spacings and dt")
    # Each axial second derivative has maximum spectral magnitude 16/(3*h**2).
    dt_limit = math.sqrt(3) / (2 * v_max * math.hypot(1 / dz, 1 / dx))
    if dt >= dt_limit:
        raise ValueError(f"Unstable CFL: dt={dt:g} must be below {dt_limit:.8g} s")


def _laplacian(u: Tensor, inv_dz2: float, inv_dx2: float, free_surface: bool) -> Tensor:
    if free_surface:
        # Odd pressure extension around physical row z=0 (pressure release).
        ghosts = -u[:, 1:3].flip((1,))
        padded = F.pad(torch.cat((ghosts, u), dim=1), (2, 2, 0, 2))
    else:
        padded = F.pad(u, (2, 2, 2, 2))
    near_z = padded[:, 1:-3, 2:-2] + padded[:, 3:-1, 2:-2]
    far_z = padded[:, :-4, 2:-2] + padded[:, 4:, 2:-2]
    near_x = padded[:, 2:-2, 1:-3] + padded[:, 2:-2, 3:-1]
    far_x = padded[:, 2:-2, :-4] + padded[:, 2:-2, 4:]
    if inv_dz2 == inv_dx2:
        return ((near_z + near_x) * (4 / 3) - (far_z + far_x) / 12 - 5 * u) * inv_dx2
    return (
        near_z * (4 / 3 * inv_dz2) + near_x * (4 / 3 * inv_dx2)
        - far_z * (inv_dz2 / 12) - far_x * (inv_dx2 / 12)
        - u * (2.5 * (inv_dz2 + inv_dx2))
    )


def _objective_sum(syn: Tensor, obs: Tensor, dt: float, f_cut: float, misfit: str) -> Tensor:
    if not math.isfinite(f_cut) or f_cut < 0:
        raise ValueError("f_cut must be finite and nonnegative")
    if f_cut > 0:
        syn = highpass_filter(syn, dt, f_cut)
        obs = highpass_filter(obs, dt, f_cut)
    if misfit == "envelope":
        syn = compute_hilbert_analytic(syn).abs()
        obs = compute_hilbert_analytic(obs).abs()
    elif misfit != "l2":
        raise ValueError("misfit must be 'l2' or 'envelope'")
    return 0.5 * (syn - obs).square().sum()


class Acoustic2DSolver:
    """Fourth-order spatial, second-order temporal acoustic FDTD.

    The sponge extends beyond the physical model and copies its boundary
    velocities. A wavelet sample is the forcing s in one physical source cell.
    Receivers sample the updated pressure after each forcing sample.
    """

    check_cfl = staticmethod(check_cfl)

    def __init__(
        self, dt: float, npml: int = 16, shot_batch_size: int = 4, free_surface: bool = False
    ) -> None:
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if not isinstance(npml, int) or npml < 0:
            raise ValueError("npml must be a nonnegative integer")
        if not isinstance(shot_batch_size, int) or shot_batch_size < 1:
            raise ValueError("shot_batch_size must be a positive integer")
        self.dt = float(dt)
        self.npml = npml
        self.shot_batch_size = shot_batch_size
        self.free_surface = free_surface

    @property
    def _top(self) -> int:
        return 0 if self.free_surface else self.npml

    def _inputs(
        self, v: Tensor, wavelet: Tensor, geom: AcquisitionGeometry,
        shot_indices: Sequence[int] | Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if v.shape != (geom.nz, geom.nx) or not v.is_floating_point():
            raise ValueError("v must be a floating-point [nz, nx] velocity model")
        if self.free_surface and geom.nz + self.npml < 3:
            raise ValueError("free surface requires at least three padded rows")
        if wavelet.ndim != 1 or wavelet.numel() == 0:
            raise ValueError("wavelet must have shape [nt] with nt > 0")
        v_min, v_max = torch.aminmax(v.detach())
        if not v_min.item() > 0:
            raise ValueError("velocities must be positive and finite")
        check_cfl(v_max.item(), geom.dz, geom.dx, self.dt)

        n_shots = len(geom.src_x)
        if shot_indices is None:
            shots = torch.arange(n_shots, device=v.device)
        else:
            shots = torch.as_tensor(shot_indices, device=v.device)
        if shots.ndim != 1 or shots.numel() == 0:
            raise ValueError("shot_indices must select at least one shot in a 1-D sequence")
        if shots.dtype == torch.bool or shots.is_floating_point() or shots.is_complex():
            raise ValueError("shot_indices must contain integers")
        shots = shots.long()
        if (shots < 0).any().item() or (shots >= n_shots).any().item():
            raise ValueError("shot index outside acquisition geometry")

        width = geom.nx + 2 * self.npml
        def offset(z: Tensor, x: Tensor) -> Tensor:
            z = torch.as_tensor(z, dtype=torch.long, device=v.device)
            x = torch.as_tensor(x, dtype=torch.long, device=v.device)
            return (z + self._top) * width + (x + self.npml)

        sources = offset(geom.src_z, geom.src_x).index_select(0, shots)
        receivers = offset(geom.rec_z, geom.rec_x)
        return wavelet.to(v), shots, sources, receivers

    def _factor(self, v: Tensor) -> Tensor:
        bounds = (self.npml, self.npml, self._top, self.npml)
        extended = F.pad(v[None, None], bounds, mode="replicate")[0, 0]
        return (extended * self.dt).square()

    def _sponge(self, v: Tensor) -> Tensor:
        nz, nx = v.shape
        n = self.npml
        z = torch.arange(nz + self._top + n, device=v.device, dtype=v.dtype) - self._top
        x = torch.arange(nx + 2 * n, device=v.device, dtype=v.dtype) - n
        z_dist = torch.maximum(-z, z - (nz - 1)).clamp_min(0)
        x_dist = torch.maximum(-x, x - (nx - 1)).clamp_min(0)
        # Exponential pressure taper. Physical cells have unit transmission.
        depth2 = z_dist.square()[:, None] + x_dist.square()[None, :]
        damping = torch.exp((-0.18 / max(n, 1)**2) * depth2)
        if self.free_surface:
            damping[0] = 0
        return damping

    def _simulate_batch(
        self, v: Tensor, wavelet: Tensor, geom: AcquisitionGeometry,
        sources: Tensor, receivers: Tensor, sponge: Tensor,
        illumination: Tensor | None, checkpoint_step: int | None,
    ) -> Tensor:
        factor = self._factor(v)
        shape = (sources.numel(), *factor.shape)
        current = factor.new_zeros(shape)
        previous = torch.zeros_like(current)
        source_map = torch.zeros_like(current)
        source_map.flatten(1).scatter_(1, sources[:, None], 1.0)
        inv_dz2, inv_dx2 = 1 / geom.dz**2, 1 / geom.dx**2

        def advance(
            older: Tensor, pressure: Tensor, vdt2: Tensor, amplitudes: Tensor
        ) -> tuple[Tensor, Tensor, Tensor]:
            # Reentrant checkpoint runs forward under no_grad and replay under grad.
            # Restrict this side effect to the original pass so replay is exact.
            accumulate = illumination is not None and (
                not checkpoint_step or not torch.is_grad_enabled()
            )
            samples = []
            for amplitude in amplitudes.unbind():
                lap = _laplacian(pressure, inv_dz2, inv_dx2, self.free_surface)
                forcing = source_map * amplitude
                updated = sponge * (2 * pressure - sponge * older + vdt2 * (lap + forcing))
                samples.append(updated.flatten(1).index_select(1, receivers))
                if accumulate:
                    with torch.no_grad():
                        interior = updated[:, self._top:self._top+geom.nz, self.npml:self.npml+geom.nx]
                        illumination.add_(interior.square().sum(dim=0))
                older, pressure = pressure, updated
            return older, pressure, torch.stack(samples, dim=-1)

        gathers = []
        for chunk in wavelet.split(checkpoint_step or wavelet.numel()):
            if checkpoint_step:
                previous, current, gather = checkpoint(
                    advance, previous, current, factor, chunk,
                    use_reentrant=True, preserve_rng_state=False,
                )
            else:
                previous, current, gather = advance(previous, current, factor, chunk)
            gathers.append(gather)
        return torch.cat(gathers, dim=-1)

    def forward(
        self, v: Tensor, wavelet: Tensor, geom: AcquisitionGeometry,
        shot_indices: Sequence[int] | Tensor | None = None, return_illumination: bool = True,
    ) -> dict[str, Tensor | None]:
        """Return differentiable raw recordings and optional detached illumination."""
        wavelet, _, sources, receivers = self._inputs(v, wavelet, geom, shot_indices)
        sponge = self._sponge(v)
        illumination = torch.zeros_like(v) if return_illumination else None
        traces = [
            self._simulate_batch(v, wavelet, geom, batch, receivers, sponge, illumination, None)
            for batch in sources.split(self.shot_batch_size)
        ]
        return {"seismograms": torch.cat(traces), "illumination": illumination}

    @staticmethod
    def _observations(
        obs: Tensor, shots: Tensor, geom: AcquisitionGeometry, wavelet: Tensor
    ) -> Tensor:
        if obs.ndim != 3 or obs.shape[1:] != (len(geom.rec_x), wavelet.numel()):
            raise ValueError("d_obs must have shape [shots, n_rec, nt]")
        obs = obs.detach().to(wavelet)
        if obs.shape[0] == len(geom.src_x):
            return obs.index_select(0, shots)
        if obs.shape[0] != shots.numel():
            raise ValueError("d_obs must contain the acquisition or the selected shots")
        return obs

    @torch.enable_grad()
    def adjoint_gradient(
        self, v: Tensor, d_obs: Tensor, wavelet: Tensor, geom: AcquisitionGeometry,
        shot_indices: Sequence[int] | Tensor | None = None, misfit: str = "l2",
        f_cut: float = 0.0, checkpoint_step: int | None = 25,
    ) -> dict[str, Tensor | float]:
        """Differentiate half the mean squared residual over all selected samples.

        Each batch releases its graph after its backward pass. Set checkpoint_step
        to None or zero to keep a full time history during each batch's backward.
        """
        if checkpoint_step is not None and (not isinstance(checkpoint_step, int) or checkpoint_step < 0):
            raise ValueError("checkpoint_step must be a nonnegative integer or None")
        if v.is_cuda:
            torch.cuda.synchronize(v.device)
        start = time.perf_counter()
        model = v.detach().requires_grad_(True)
        wavelet, shots, sources, receivers = self._inputs(model, wavelet.detach(), geom, shot_indices)
        obs = self._observations(d_obs, shots, geom, wavelet)
        sponge = self._sponge(model)
        illumination = torch.zeros_like(model)
        total_loss = model.new_zeros(())
        traces = []
        for first in range(0, sources.numel(), self.shot_batch_size):
            batch = slice(first, first + self.shot_batch_size)
            syn = self._simulate_batch(
                model, wavelet, geom, sources[batch], receivers, sponge, illumination, checkpoint_step
            )
            loss = _objective_sum(syn, obs[batch], self.dt, f_cut, misfit) / obs.numel()
            loss.backward()
            total_loss.add_(loss.detach())
            traces.append(syn.detach())
        d_syn = torch.cat(traces)
        value = total_loss.item()
        return {
            "loss": value, "grad": model.grad.detach(), "d_syn": d_syn,
            "illumination": illumination, "elapsed_sec": time.perf_counter() - start,
        }

    @torch.no_grad()
    def directional_derivative_fd(
        self, v: Tensor, p: Tensor, d_obs: Tensor, wavelet: Tensor, geom: AcquisitionGeometry,
        eps: float = 1e-2, shot_indices: Sequence[int] | Tensor | None = None,
        misfit: str = "l2", f_cut: float = 0.0,
    ) -> float:
        """Evaluate a central difference in float64 to resolve small m/s perturbations.

        MPS has no float64 arithmetic, so this diagnostic runs on CPU for MPS inputs.
        The two objectives use the same discretization and normalization as the adjoint.
        """
        if not math.isfinite(eps) or eps <= 0 or p.shape != v.shape:
            raise ValueError("eps must be finite and positive and p must match the model shape")
        device = "cpu" if v.device.type == "mps" else v.device
        model = v.to(device=device, dtype=torch.float64)
        direction = p.to(model)
        wavelet, shots, _, _ = self._inputs(model, wavelet, geom, shot_indices)
        obs = self._observations(d_obs, shots, geom, wavelet)

        def objective(m: Tensor) -> Tensor:
            syn = self.forward(m, wavelet, geom, shots, return_illumination=False)["seismograms"]
            return _objective_sum(syn, obs, self.dt, f_cut, misfit) / obs.numel()

        return ((objective(model + eps * direction) - objective(model - eps * direction)) / (2 * eps)).item()
