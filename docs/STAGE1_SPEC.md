# Stage 1 Engineering Specification: 2D Acoustic FWI Physics Engine, Baseline & Multi-Modal Dataset Pipeline

## 1. Objective
Build the reusable, GPU-accelerated, pure-PyTorch 2D acoustic Full Waveform Inversion (FWI) physics foundation for the project **"面向低频缺失的物理约束与重参数化全波形反演梯度预测"**.

Why **Pure-PyTorch Batched 2D FDTD** (instead of compiled C++/CUDA third-party extensions):
- Zero compilation issues on Windows (`RTX 5060 Ti`, `torch 2.9 + cu128`) and macOS.
- Native `torch.autograd` support for exact discrete adjoint gradients, directional derivatives ($\mathcal{L}_{\text{dir}}$), one-step forward checks ($\mathcal{L}_{\text{data}}$), and VJP Jacobian projection ($\hat{g}_\theta = (J_\theta)^T \hat{g}_v$) in later stages.
- Direct access to internal forward wavefields to compute exact **acquisition illumination** $I_k = \sum_{s,t} u_s(x,t)^2$.

## 2. Target Directory Layout (under `<repo_root>`)
Do NOT modify or delete the existing PDF proposal file in the root directory.

```text
Project02/
├── docs/
│   └── STAGE1_SPEC.md
├── configs/
│   ├── stage1_smoke.yaml
│   └── stage1_default.yaml
├── src/
│   └── fwi_grad/
│       ├── __init__.py
│       ├── physics/
│       │   ├── __init__.py
│       │   ├── wavelet.py
│       │   ├── geometry.py
│       │   ├── acoustic2d.py
│       │   └── residuals.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── velocity_models.py
│       │   └── dataset_builder.py
│       ├── inversion/
│       │   ├── __init__.py
│       │   └── baseline_fwi.py
│       └── eval/
│           ├── __init__.py
│           └── metrics.py
├── scripts/
│   ├── run_stage1_baseline.py
│   └── sync_to_winpc.sh
├── tests/
│   └── test_stage1_physics.py
└── pyproject.toml
```

## 3. Detailed Module Requirements

### 3.1 `physics/wavelet.py`
- `ricker_wavelet(nt: int, dt: float, f0: float, t0: float | None = None, device=None, dtype=torch.float32) -> Tensor`
- `highpass_filter(signal: Tensor, dt: float, f_cut: float, order: int = 4) -> Tensor`:
  - Smooth zero-phase frequency-domain high-pass filter along the last dimension (`nt`) to simulate **low-frequency missing** conditions (e.g., `f_cut = 3.0 Hz`, `5.0 Hz`, `7.0 Hz`).
  - Must be pure PyTorch (`torch.fft.rfft` / `irfft`) so it works on GPU and CPU without SciPy CPU transfers.
- `bandpass_filter(signal: Tensor, dt: float, f_low: float | None, f_high: float | None, order: int = 4) -> Tensor` for multiscale FWI continuation.

### 3.2 `physics/geometry.py`
- Dataclass `AcquisitionGeometry`:
  - `nz`, `nx`, `dz`, `dx`
  - `src_z`, `src_x` (`[n_shots]` integer grid indices)
  - `rec_z`, `rec_x` (`[n_rec]` integer grid indices)
- Factory `create_surface_geometry(nz, nx, dz, dx, n_shots, n_rec, src_depth_idx=2, rec_depth_idx=2, margin=4)`
- Helper `select_sparse_shots(n_shots, ratio_or_count, mode="uniform"|"random", seed=None) -> list[int]`
- Helper `geometry_encoding(geom, device=None) -> Tensor`: returns a `[2, nz, nx]` spatial map encoding source and receiver density/locations.

### 3.3 `physics/acoustic2d.py`
- `Acoustic2DSolver`:
  - 2D constant-density acoustic wave equation:
    $$\frac{1}{v^2}\frac{\partial^2 u}{\partial t^2} = \nabla^2 u + s(x, z, t)$$
  - 4th-order spatial finite difference stencil (`[-1/12, 4/3, -5/2, 4/3, -1/12]` in both `z` and `x`) via vectorized `F.conv2d` or direct tensor slicing (prefer direct slicing for speed & exactness on GPU).
  - Sponge / Cerjan exponential or quadratic damping layer of width `npml` (e.g., 16–20 grid points) on all 4 boundaries (with optional free-surface top).
  - Strict CFL stability validator: `check_cfl(v_max, dz, dx, dt)` raising clear error if unstable.
  - Batched across shots: processes `shot_batch_size` shots in parallel `[B, nz_pad, nx_pad]`.
  - `forward(v, wavelet, geom, shot_indices=None, return_illumination=True) -> dict`:
    - Returns `{"seismograms": [n_sel, n_rec, nt], "illumination": [nz, nx]}`.
    - Illumination $I(z,x) = \sum_{s,t} u_s(z,x,t)^2$ accumulated in `torch.no_grad()` on the interior domain.
  - `adjoint_gradient(v, d_obs, wavelet, geom, shot_indices=None, misfit="l2", f_cut=0.0, checkpoint_step=25) -> dict`:
    - Uses `torch.utils.checkpoint.checkpoint` over time-step chunks (`checkpoint_step` steps per chunk) so GPU memory stays well within 16 GB even for large grids (`128x256`, `nt=1000`) while guaranteeing **exact discrete gradients**.
    - Supports `misfit="l2"` (and optional `"envelope"`).
    - If `f_cut > 0`, applies `highpass_filter` consistently to both `d_syn` and `d_obs` before computing misfit.
    - Returns `{"loss": float, "grad": Tensor[nz, nx], "d_syn": Tensor[n_sel, n_rec, nt], "illumination": Tensor[nz, nx], "elapsed_sec": float}`.
  - `directional_derivative_fd(v, p, d_obs, wavelet, geom, eps=1e-2, shot_indices=None, misfit="l2", f_cut=0.0) -> float`:
    - Computes central finite difference $[J(v + \varepsilon p) - J(v - \varepsilon p)] / (2\varepsilon)$ with `torch.no_grad()`.

### 3.4 `physics/residuals.py`
- `compute_hilbert_analytic(x: Tensor) -> Tensor`: Pure-PyTorch FFT Hilbert transform along last dim (`nt`).
- `compute_multimodal_residuals(d_syn: Tensor, d_obs: Tensor, eps: float = 1e-6) -> dict`:
  - `r_wave`: `d_syn - d_obs`
  - `r_env`: `abs(analytic(d_syn)) - abs(analytic(d_obs))`
  - `r_phase`: wrapped instantaneous phase difference `atan2(Im(z_syn * conj(z_obs)), Re(z_syn * conj(z_obs)) + eps)` masked smoothly by envelope weight so zero-amplitude regions do not produce phase noise.

### 3.5 `data/velocity_models.py`
- `load_openfwi_velocity(npy_path, index=0, target_shape=None) -> np.ndarray`: loads OpenFWI `model*.npy` (`[N, 1, H, W]` or `[N, H, W]`), converts to float32 in m/s, optionally resizes cleanly.
- `generate_synthetic_complex_model(nz, nx, seed=0) -> np.ndarray`: generates realistic complex geological velocity models (increasing background velocity + sinusoidal folded layers + fault offset + high-velocity salt/lens anomaly) in `[1500, 4500] m/s`, so tests and experiments work immediately even without external files.
- `build_initial_model(v_true, method="gaussian"|"linear", sigma=8.0, v_min=None, v_max=None) -> np.ndarray`.
- `add_gaussian_noise(d_obs, snr_db, seed=None) -> Tensor`.

### 3.6 `data/dataset_builder.py`
- `build_trajectory_samples(config, output_dir, device="cuda")`:
  - Generates FWI iteration trajectories across multiple velocity models and scenarios (`normal`, `low_freq_cut`, `noisy`, `sparse`, `biased_init`).
  - Saves each sample as `.pt` (or `.npz`) with standardized keys:
    `v_k`, `v_init`, `v_true`, `r_wave`, `r_env`, `r_phase`, `illumination`, `g_sparse`, `g_ref`, `meta` (`iter_k`, `f_cut`, `snr_db`, `sparse_ratio`, `loss_k`, `t_full_grad`, `t_sparse_grad`).
  - Writes a `manifest.json` index with train/val/test split by **velocity model ID** (strictly zero data leakage across splits!).

### 3.7 `inversion/baseline_fwi.py` & `eval/metrics.py`
- `eval/metrics.py`:
  - `velocity_metrics(v_pred, v_true)` → `mae`, `rmse`, `re` (relative L2 error), `ssim`, `snr_db`.
  - `gradient_metrics(g_pred, g_ref)` → `cosine_similarity`, `rel_l2_error`.
- `inversion/baseline_fwi.py`:
  - `run_traditional_fwi(v_init, v_true, d_obs, solver, wavelet, geom, n_iters, lr, shot_indices=None, f_cut=0.0, optimizer="adam", v_bounds=(1500.0, 5000.0))` → returns final velocity, loss history, velocity metric history, and per-iteration timing breakdown (`grad_time_mean`, `iter_time_mean`).

### 3.8 `scripts/run_stage1_baseline.py` & `scripts/sync_to_winpc.sh`
- `run_stage1_baseline.py --config configs/stage1_smoke.yaml --device auto`:
  - Runs finite-difference gradient check, full-shot FWI baseline, sparse-shot FWI baseline, and builds a small multi-modal dataset, writing `summary.json` to `outputs/`.
- `sync_to_winpc.sh`:
  - Syncs `Project02` code (`src/`, `configs/`, `scripts/`, `tests/`, `docs/`, `pyproject.toml`) from Mac to `winpc:F:\project\Project02` cleanly using `tar` over `ssh winpc` (so it works with Windows built-in `tar.exe` + OpenSSH without requiring `rsync`).

### 3.9 `tests/test_stage1_physics.py`
Comprehensive pytest suite covering:
1. CFL validation (passes valid params, raises on invalid dt).
2. Forward wavefield & illumination finiteness and symmetry.
3. **Exact gradient verification**: compares `adjoint_gradient` inner product $\langle g, p \rangle$ against `directional_derivative_fd` for random perturbation $p$ (assert relative error `< 5e-3` in float32 / `< 1e-4` in float64, cosine similarity `> 0.999` on small grid).
4. High-pass filter energy attenuation below `f_cut`.
5. Multi-modal residuals (`r_wave`, `r_env`, `r_phase`) zero-check when `d_syn == d_obs` and non-zero finite values under time shift.
6. Baseline FWI smoke test: verifies loss decreases and velocity MAE improves on a small `48x64` synthetic model.
7. Dataset builder smoke test: verifies `.pt` files and `manifest.json` are written with all required keys and strict model-level train/val/test split.

## 4. Acceptance Criteria
- Clean, typed, modular PyTorch code with no external compiled extensions required.
- All tests in `tests/test_stage1_physics.py` pass on `winpc` (`cu128`, RTX 5060 Ti).
- `run_stage1_baseline.py --config configs/stage1_smoke.yaml` completes end-to-end and outputs valid metrics & timing baseline.
