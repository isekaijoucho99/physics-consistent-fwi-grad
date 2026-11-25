# Stage 4 Engineering Specification: Multi-Scale Neural Reparameterization & Stable Closed-Loop FWI

## 1. Objective
Implement **Research Content 3 (研究内容三)** from the proposal for **Paper 2** and full-system integration:
- Bounded multi-scale neural network velocity reparameterization:
  $$v_\theta(x) = v_{\min} + (v_{\max} - v_{\min})\,\sigma\big[\text{logit}(\tilde{v}_0(x)) + G_{\theta_{\text{low}}}(z, x) + \alpha_{\text{detail}} G_{\theta_{\text{high}}}(\gamma_{\text{FF}}(z, x)) + \alpha_{\text{grid}} \Delta_{\text{grid}}(z, x)\big]$$
- Exact parameter-space gradient projection via Vector-Jacobian Product (VJP):
  $$\hat{g}_\theta = (J_\theta)^T \big(C(\hat{\sigma}_k) \odot \hat{g}_v\big)$$
- Uncertainty-controlled progressive update (low-wavenumber background first → Fourier high-wavenumber detail unfreezing → optional bounded late-stage grid refinement) with conservative rollback & quality control.
- End-to-end low-frequency missing (`f_cut = 5.0 Hz`) comparison script benchmarking:
  1. Traditional Full-Shot Grid FWI
  2. Direct-Grid Stage 3 Guided FWI (no reparameterization)
  3. Full-Adjoint Neural Reparameterization FWI (INR/NNFWI baseline)
  4. **Proposed Full Framework**: Stage 3 Predicted Gradient + Uncertainty + Multi-Scale Neural Reparameterization FWI.

## 2. Target Files to Add (under `<repo_root>`)
Do NOT break any existing Stage 1/2/3 APIs or tests (`test_stage1_physics.py`, `test_stage2_model.py`, `test_stage3_physics.py`, `test_plotting.py` must continue to pass 100%) and do NOT touch the root PDF. Keep code concise and direct without excessive defensive boilerplate.

```text
Project02/
├── docs/
│   └── STAGE4_SPEC.md
├── configs/
│   ├── stage4_smoke.yaml
│   └── stage4_openfwi.yaml
├── src/
│   └── fwi_grad/
│       ├── models/
│       │   └── reparam_net.py
│       └── inversion/
│           └── reparam_fwi.py
├── scripts/
│   └── run_stage4_reparam.py
└── tests/
    └── test_stage4_reparam.py
```

## 3. Detailed Module Requirements

### 3.1 `models/reparam_net.py`
- `MultiScaleVelocityReparam(v_init: Tensor, v_bounds: tuple[float, float] = (1500.0, 5000.0), low_hidden: int = 32, high_hidden: int = 64, n_fourier: int = 32, fourier_scale: float = 4.0, max_grid_logit: float = 0.25, seed: int = 0)`:
  - Stores normalized coordinate grid `(z, x)` in `[-1, 1]` and precomputed `base_logit = logit(clamp((v_init - v_min) / (v_max - v_min), 1e-4, 1 - 1e-4))` as buffers.
  - **Low-wavenumber background branch (`low_branch`)**: 3-layer MLP with `tanh` activations on raw `(z, x)` coordinates; final linear layer zero-initialized so initial output is identically `0`.
  - **High-wavenumber detail branch (`high_branch`)**: fixed Random Fourier Feature matrix `B ~ N(0, fourier_scale^2)` (`[2, n_fourier]`) generating `[sin(2π x B), cos(2π x B)]`, followed by a 3-layer MLP with `SiLU` activations; final linear layer zero-initialized so initial output is identically `0`.
  - **Late-stage bounded grid correction (`grid_residual`)**: `nn.Parameter(torch.zeros(nz, nx))`, bounded via `max_grid_logit * tanh(grid_residual)` so late-stage refinement stays smooth and bounded.
  - `forward(detail_weight: float = 1.0, grid_weight: float = 0.0) -> Tensor`: returns bounded velocity field `v_theta` (`[nz, nx]`), guaranteed in `(v_min, v_max)` and equal to `v_init` at initialization.
  - `components(detail_weight: float = 1.0, grid_weight: float = 0.0) -> dict[str, Tensor]`: returns `{"v_total", "v_background", "delta_low", "delta_high", "delta_grid"}` for paper visualization of multi-scale decomposition.
  - `project_gradient(g_v: Tensor, detail_weight: float = 1.0, grid_weight: float = 0.0) -> list[Tensor]`:
    - Computes exact VJP $\hat{g}_\theta = (J_\theta)^T \hat{g}_v$ via `torch.autograd.grad(v_theta, active_params, grad_outputs=g_v, allow_unused=True)` and assigns `.grad` cleanly onto model parameters (zeroing inactive branches when their weight is 0).

### 3.2 `inversion/reparam_fwi.py`
- `ReparamFWIConfig` dataclass:
  - `n_iters: int = 30`
  - `lr_low: float = 0.015`
  - `lr_high: float = 0.008`
  - `lr_grid: float = 0.01`
  - `v_bounds: tuple[float, float] = (1500.0, 5000.0)`
  - `kappa: float = 2.0`
  - `sigma_threshold: float = 0.5`
  - `n_check_shots: int = 2`
  - `max_backtracks: int = 3`
  - `sparse_ratio: float = 0.25`
  - `f_cut: float = 5.0`
  - `detail_unlock_frac: float = 0.3` (unlock high-wavenumber branch after 30% of iterations, ramping linearly over 20% of iterations)
  - `grid_unlock_frac: float = 0.75` (unlock optional bounded grid correction in final 25% of iterations)
  - `illumination_precond: bool = True` (divide model-space gradient by `(illumination / quantile(illumination, 0.9) + 0.1)` so deep strata update evenly)
  - `enable_fallback: bool = True`
  - `low_hidden: int = 32`, `high_hidden: int = 64`, `n_fourier: int = 32`, `fourier_scale: float = 4.0`, `seed: int = 0`
- `run_reparam_guided_fwi(v_init, v_true, d_obs, solver, wavelet, geom, config, grad_model=None, normalizer=None, device=None) -> dict`:
  - Works seamlessly in two modes:
    - **`grad_model is not None`**: Stage 3 predicted gradient + uncertainty $\hat{\sigma}_k$ (from sparse shots) → confidence weighting $C(\hat{\sigma}_k)$ + optional illumination preconditioning → VJP projection $\hat{g}_\theta = (J_\theta)^T \tilde{g}_v$ → Adam parameter step → few-shot verification (reusing `sparse["loss"]` as `before`!).
    - **`grad_model is None`**: Full-shot adjoint gradient baseline with neural reparameterization (INR/NNFWI baseline) for fair ablation!
  - **Conservative quality control on verification failure**:
    - Save parameter snapshot before stepping; if trial `v_theta` does not decrease check-shot misfit, restore parameters, halve effective LR (up to `max_backtracks`), and if still unstable, pause detail ramp (`detail_weight` held at previous value / 0) or fallback to sparse/full adjoint gradient.
  - Records `v_final`, `v_background`, `loss_history`, `full_loss_history` (evaluated cleanly for fair apples-to-apples curves), `metric_history`, `decisions`, `detail_weights`, `grad_time_mean`, `verify_time_mean`, `iter_time_mean`, `final_metrics`.

### 3.3 `scripts/run_stage4_reparam.py` & Configs
- `scripts/run_stage4_reparam.py --config configs/stage4_smoke.yaml --device auto`:
  - Loads Stage 3 checkpoint (`stage3_checkpoint`; optional scratch fallback only if `allow_scratch: true`).
  - Runs all 4 comparison methods on the test velocity model under **low-frequency missing (`f_cut = 5.0 Hz`)**:
    1. `full_grid_fwi` (traditional full-shot Adam FWI)
    2. `stage3_grid_fwi` (Stage 3 guided FWI on direct grid)
    3. `full_reparam_fwi` (full-shot adjoint + multi-scale neural reparameterization)
    4. `stage3_reparam_fwi` (proposed full framework: Stage 3 predicted gradient + uncertainty + multi-scale neural reparameterization)
  - Saves `stage4_metrics.json` and uses `plotting.py` (`plot_velocity_and_profile`, `plot_closed_loop_convergence`) to output 300 DPI PNG + PDF comparison figures to `output_dir`.
- `configs/stage4_smoke.yaml`: fast 6-iteration smoke test on `48x64` synthetic model.
- `configs/stage4_openfwi.yaml`: real OpenFWI `CurveVel_A` unseen test model (`64x64`, `nt=550` so waves fully penetrate `0.64 km`, `f_cut=5.0 Hz`, `n_iters=35`, using `outputs/stage3_openfwi/best_stage3.pt`).

### 3.4 `tests/test_stage4_reparam.py`
Pytest suite verifying:
1. Initial identity `||v_theta(init) - v_init||_inf < 1e-2 m/s` and strict bounds `(v_min, v_max)`.
2. Exact VJP check: finite-difference directional derivative of $\langle v_\theta, g_v \rangle$ along random parameter perturbation $\delta\theta$ matches $\langle \hat{g}_\theta, \delta\theta \rangle$ to relative error `< 1e-3`.
3. Progressive unfreezing: `high_branch` receives zero gradient when `detail_weight=0` and non-zero gradient when `detail_weight>0`.
4. `run_reparam_guided_fwi` (both `grad_model=None` and with `MultiModalGradientNet`) reduces full-shot misfit under `f_cut=5.0 Hz` and produces finite velocity metrics.
5. `run_stage4_reparam.py --config configs/stage4_smoke.yaml` completes end-to-end and writes `stage4_metrics.json`.
