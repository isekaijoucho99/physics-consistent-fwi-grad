# Stage 3 Engineering Specification: Wave-Equation Physics-Consistent Training & Inference Gating

## 1. Objective
Implement **Research Content 2 (研究内容二)** from the proposal:
- **Physics-Consistent Losses** ($\mathcal{L}_{\text{dir}}$, $\mathcal{L}_{\text{data}}$, $\mathcal{L}_{\text{str}}$) combined with Stage 2 losses ($\mathcal{L}_{\text{pred}}$, $\mathcal{L}_{\text{unc}}$) in a **progressive physics-weighting** two-stage training schedule.
- **Inference-Time Quality Control & Adaptive Fallback** (`PhysicsGuidedFWI`): uncertainty screening + lightweight few-shot directional-derivative / one-step forward verification + step-size backtracking / sparse-gradient blending / full-adjoint fallback.
- **Ablation & Evaluation Pipeline** (`scripts/train_stage3.py`) directly quantifying **Stage 2 (supervised only)** vs **Stage 3 (physics-consistent)** on cosine similarity, directional derivative error, one-step descent success rate, and closed-loop FWI efficiency.

## 2. Target Files to Add (under `<repo_root>`)
Do NOT break any existing Stage 1 or Stage 2 APIs or tests (`test_stage1_physics.py` and `test_stage2_model.py` must continue to pass 100%).

```text
Project02/
├── docs/
│   └── STAGE3_SPEC.md
├── configs/
│   ├── stage3_smoke.yaml
│   └── stage3_openfwi.yaml
├── src/
│   └── fwi_grad/
│       ├── losses/
│       │   └── physics_losses.py
│       └── inversion/
│           └── guided_fwi.py
├── scripts/
│   └── train_stage3.py
└── tests/
    └── test_stage3_physics.py
```

## 3. Detailed Module Requirements

### 3.1 `losses/physics_losses.py`
- `confidence_weight(sigma: Tensor, kappa: float = 2.0, min_weight: float = 0.1) -> Tensor`:
  - Computes $C(\hat{\sigma}_k) = \text{clamp}(1 / (1 + \kappa \hat{\sigma}_k), \min=\text{min\_weight}, \max=1.0)$.
- `structural_continuity_loss(g_pred_norm: Tensor, v_k: Tensor, beta: float = 5.0, eps: float = 1e-6) -> Tensor`:
  - Computes spatial finite differences $\nabla_z, \nabla_x$ of `v_k` and `g_pred_norm`.
  - Boundary-aware weights:
    $$W_z = \exp\left(-\beta \frac{|\nabla_z v_k|}{\text{mean}(|\nabla_z v_k|) + \varepsilon}\right), \quad W_x = \exp\left(-\beta \frac{|\nabla_x v_k|}{\text{mean}(|\nabla_x v_k|) + \varepsilon}\right)$$
    (computed detached from `v_k`).
  - Returns $\mathcal{L}_{\text{str}} = \text{mean}(W_z |\nabla_z \hat{g}|) + \text{mean}(W_x |\nabla_x \hat{g}|)$.
  - Smooth strata ($|\nabla v_k| \approx 0 \Rightarrow W \approx 1$) penalize gradient roughness; velocity interfaces ($|\nabla v_k| \gg 0 \Rightarrow W \approx 0$) preserve sharp structural edges.
- `directional_derivative_loss(g_pred_phys: Tensor, v_k: Tensor, d_obs: Tensor, solver: Acoustic2DSolver, wavelet: Tensor, geom: AcquisitionGeometry, shot_indices: Sequence[int], f_cut: float = 0.0, eps_fd: float = 0.05, eps_denom: float = 1e-8, p: Tensor | None = None) -> Tensor`:
  - Samples a random unit vector $p$ (`||p||_2 = 1`, same shape as `v_k`) if `p is None`.
  - Computes reference finite-difference directional derivative $D_p J(v_k) = \text{solver.directional\_derivative\_fd}(v_k, p, d_{\text{obs}}, \text{wavelet}, \text{geom}, \text{eps}=\text{eps\_fd}, \text{shot\_indices}=\text{shot\_indices}, f_{\text{cut}}=f_{\text{cut}})$ (detached scalar).
  - Computes predicted directional derivative $\langle g_{\text{pred\_phys}}, p \rangle$ (differentiable w.r.t. `g_pred_phys`!).
  - Returns $\mathcal{L}_{\text{dir}} = |\langle g_{\text{pred\_phys}}, p \rangle - D_p J(v_k)| / (|D_p J(v_k)| + \text{eps\_denom})$, clamped at a safe upper cap (e.g. `10.0`) so outlier steps cannot explode training.
- `one_step_data_consistency_loss(g_pred_phys: Tensor, sigma: Tensor, v_k: Tensor, d_obs: Tensor, solver: Acoustic2DSolver, wavelet: Tensor, geom: AcquisitionGeometry, shot_indices: Sequence[int], eta: float, rho: float = 0.0, kappa: float = 2.0, f_cut: float = 0.0, v_bounds: tuple[float, float] = (1500.0, 5000.0), max_update_mps: float = 100.0) -> dict[str, Tensor]`:
  - Computes current subset misfit $J_0 = J_{\text{sub}}(v_k)$ under `torch.no_grad()`.
  - Constructs differentiable trial step:
    `step = confidence_weight(sigma, kappa) * g_pred_phys`
    scaled/clamped so max velocity perturbation stays bounded (`<= max_update_mps`) for CFL safety:
    `v_trial = (v_k - eta * step).clamp(v_bounds[0], v_bounds[1])`.
  - Runs differentiable `solver.forward(v_trial, wavelet, geom, shot_indices=shot_indices, return_illumination=False)` and computes normalized misfit $J_1 = J_{\text{sub}}(v_{\text{trial}})$ (using the exact same objective normalization as `Acoustic2DSolver.adjoint_gradient`, respecting `f_cut`).
  - Relative hinge loss (scale-invariant across models):
    $$\mathcal{L}_{\text{data}} = \text{ReLU}\left(\frac{J_1 - J_0}{J_0 + 10^{-8}} + \rho\right)$$
  - Returns `{"loss_data": loss_data, "loss_before": J_0, "loss_after": J_1, "descended": (J_1 < J_0)}`.
- `ProgressivePhysicsWeightSchedule(lambda_dir, lambda_data, lambda_str, warmup_epochs)`:
  - Returns `(w_dir, w_data, w_str)` linearly ramped from `0` to target over `warmup_epochs` (for epoch 1..warmup).

### 3.2 `inversion/guided_fwi.py`
- `GuidedFWIConfig` dataclass:
  - `n_iters: int = 10`
  - `lr: float = 5.0`
  - `v_bounds: tuple[float, float] = (1500.0, 5000.0)`
  - `kappa: float = 2.0`
  - `sigma_threshold: float = 0.5` (mean uncertainty threshold triggering cautious mode)
  - `n_check_shots: int = 2` (few random shots for lightweight verification)
  - `max_backtracks: int = 3`
  - `max_update_mps: float = 150.0`
  - `sparse_blend_weight: float = 0.5`
  - `enable_fallback: bool = True`
  - `f_cut: float = 0.0`
  - `sparse_ratio: float = 0.25`
- `run_physics_guided_fwi(v_init, v_true, d_obs, model, solver, wavelet, geom, config, normalizer=None, device=None) -> dict`:
  - At each iteration $k$:
    1. Computes `g_sparse`, `d_syn_sparse`, `illumination_sparse` on `sparse_shots` (timed as `sparse_time`).
    2. Computes multi-modal residuals `(r_wave, r_env, r_phase)` and runs `model(...)` under `torch.no_grad()` to get `(g_pred_norm, sigma)` (timed as `net_time`).
    3. Denormalizes `g_pred_norm` using `normalizer.compute_scale(g_sparse)` to obtain physical-scale `g_pred_phys`, and computes `C = confidence_weight(sigma, kappa)`.
    4. Selects `n_check_shots` verification shots and evaluates `J_before` on `v_k`.
    5. **Verification & Fallback Ladder** (timed as `verify_time`):
       - **Try 1 (Predicted + Uncertainty Step)**: normalize update step scale consistently (Adam-like or RMS-normalized step `lr * C * g_pred_phys / (rms + eps)` capped by `max_update_mps`), test `J_after < J_before` on check shots; if `mean(sigma) > sigma_threshold`, start with `0.5 * lr`.
       - **Try 2 (Backtracking)**: halve step size up to `max_backtracks` times until `J_after < J_before`.
       - **Try 3 (Sparse Blending)**: blend `(1 - w) * g_pred_phys + w * g_sparse` and test `J_after < J_before`.
       - **Try 4 (Full Adjoint Fallback)**: if `enable_fallback` and still no descent, compute full-shot adjoint gradient and take a verified step.
    6. Records per-iteration `loss`, `velocity_metrics`, `decision` (`"predicted"`, `"backtracked"`, `"sparse_blend"`, `"full_fallback"`, `"rejected"`), `mean_sigma`, `grad_time` (`sparse_time + net_time`), `verify_time`, `iter_time`.
  - Returns comprehensive result dict including `v_final`, `loss_history`, `metric_history`, `decisions`, `descent_success_rate`, `predicted_accept_rate`, `grad_time_mean`, `verify_time_mean`, `iter_time_mean`, `final_metrics`.

### 3.3 `scripts/train_stage3.py` & Configs
- `scripts/train_stage3.py --config configs/stage3_smoke.yaml --device auto`:
  - Loads pretrained Stage 2 checkpoint (`pretrained_checkpoint`; if missing and `auto_train_stage2` is true, raises clear error or loads freshly initialized model when `allow_scratch: true`).
  - Fine-tunes `MultiModalGradientNet` with progressive physics weighting:
    - Every batch computes `Stage2GradientLoss` + `structural_continuity_loss`.
    - On `physics_subset_size` samples per batch (e.g., 1–2 samples, using `n_physics_shots` random shots to keep training fast!), computes `directional_derivative_loss` and `one_step_data_consistency_loss`.
  - Evaluates before fine-tuning (Stage 2 baseline) vs after fine-tuning (Stage 3 physics-consistent) on `val` split:
    - Cosine similarity, rel L2 error, uncertainty correlation
    - **Directional derivative relative error**
    - **One-step descent success rate**
  - Runs closed-loop `run_physics_guided_fwi` on a test model and compares against `run_traditional_fwi` (full-shot & sparse-shot), saving `best_stage3.pt` and `stage3_metrics.json`.
- `configs/stage3_smoke.yaml`: fast 2-epoch smoke test on Stage 1/2 smoke dataset.
- `configs/stage3_openfwi.yaml`: production-ready config using Stage 2 OpenFWI dataset & `outputs/stage2_openfwi/best.pt`.

### 3.4 `tests/test_stage3_physics.py`
Pytest suite verifying:
1. `structural_continuity_loss`: smooth-region gradient roughness is penalized strictly more than interface-aligned gradient jump.
2. `directional_derivative_loss`: near-zero (`< 1e-3`) for true adjoint gradient, high (`~ 2.0`) for negated gradient, and produces finite gradients w.r.t. `g_pred_phys`.
3. `one_step_data_consistency_loss`: end-to-end differentiable through `Acoustic2DSolver.forward` back to `MultiModalGradientNet` parameters; `descended == True` for true gradient direction and `loss_data > 0` for negated gradient.
4. `run_physics_guided_fwi`: decreases loss on synthetic model; when given a dummy model that outputs a **negated** gradient, triggers fallback (`full_fallback` / `sparse_blend`) instead of diverging!
5. `train_stage3.py` smoke run completes end-to-end and outputs valid `stage3_metrics.json` with Stage 2 vs Stage 3 comparison fields.

## 4. Acceptance Criteria
- All tests across `test_stage1_physics.py`, `test_stage2_model.py`, and `test_stage3_physics.py` pass on both Mac and `winpc` (`cu128`, RTX 5060 Ti).
- `train_stage3.py --config configs/stage3_smoke.yaml --device cuda` completes cleanly on `winpc`.
