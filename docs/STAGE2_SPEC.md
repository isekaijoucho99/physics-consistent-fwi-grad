# Stage 2 Engineering Specification: Multi-Modal Gradient & Uncertainty Prediction Network

## 1. Objective
Implement **Research Content 1 (研究内容一)** from the proposal:
$$(\hat{g}_k, \hat{\sigma}_k) = G_\psi(v_k, r_{\text{wave}}, r_{\text{env}}, r_{\text{phase}}, I_k, g_{\text{sparse}}, q_k)$$
along with sign-preserving quantile normalization, multi-scale low/high wavenumber decoding, Stage-1 supervised pretraining loss ($\mathcal{L}_{\text{pred}} + \lambda_\sigma \mathcal{L}_{\text{unc}}$), PyTorch DataLoader over Stage 1 manifests, uncertainty calibration metrics, and an OpenFWI-ready training pipeline.

## 2. Target Files to Add (under `<repo_root>`)
Do NOT break any existing Stage 1 APIs or tests (`tests/test_stage1_physics.py` must continue to pass 100%).

```text
Project02/
├── docs/
│   └── STAGE2_SPEC.md
├── configs/
│   ├── stage2_smoke.yaml
│   └── stage2_openfwi.yaml
├── src/
│   └── fwi_grad/
│       ├── data/
│       │   └── torch_dataset.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── normalization.py
│       │   └── gradient_net.py
│       ├── losses/
│       │   ├── __init__.py
│       │   └── gradient_losses.py
│       └── eval/
│           └── metrics.py          # extend with uncertainty_calibration_metrics
├── scripts/
│   └── train_stage2.py
└── tests/
    └── test_stage2_model.py
```

## 3. Detailed Module Requirements

### 3.1 `models/normalization.py`
- `SignPreservingNormalizer(quantile: float = 0.99, alpha: float = 10.0, eps: float = 1e-8)`:
  - `compute_scale(g: Tensor) -> Tensor`: per-sample quantile of `|g|` over spatial dims `(..., nz, nx)`, clamped by `eps`.
  - `normalize(g: Tensor, scale: Tensor | None = None) -> tuple[Tensor, Tensor]`:
    $$\mathcal{N}(g) = \text{sign}(g) \cdot \frac{\log(1 + \alpha |g| / s)}{\log(1 + \alpha)}$$
    Returns `(g_norm, scale)`.
  - `denormalize(g_norm: Tensor, scale: Tensor) -> Tensor`: exact inverse of `normalize`.
- `split_wavenumber(field: Tensor, sigma_low: float = 3.0) -> tuple[Tensor, Tensor]`:
  - Separable 2D Gaussian low-pass filter in PyTorch (`F.conv2d` with replicate padding) returning `(low_wn, high_wn)` such that `low_wn + high_wn == field` identically.

### 3.2 `models/gradient_net.py`
- `MultiModalGradientNet(base_channels: int = 32, cond_dim: int = 32, eps_sigma: float = 1e-3, alpha: float = 10.0)`:
  - Inputs in `forward(...)`:
    - `v_k`: `[B, 1, nz, nx]` (m/s, internally normalized to ~`[-1, 1]` via `(v - 3000) / 1500`)
    - `v_init`: `[B, 1, nz, nx]` (m/s; computes initial deviation `(v_k - v_init) / 500`)
    - `r_wave`, `r_env`, `r_phase`: `[B, S, n_rec, nt]` (per-shot normalized internally to avoid scale explosion)
    - `shot_mask`: `[B, S]` boolean mask (`True` for valid shots, `False` for padded shots)
    - `illumination`: `[B, 1, nz, nx]` (internally log-compressed via `log1p(I / (quantile(I, 0.99) + eps))`)
    - `g_sparse`: `[B, 1, nz, nx]` (internally sign-preserving quantile normalized)
    - `q_k`: `[B, 3]` conditioning vector `[f_cut / 10.0, iter_norm, sparse_ratio]`
  - Architecture:
    1. **Model-Domain Encoder**: 3-scale Conv+GroupNorm+SiLU ResBlocks on `[v_k_norm, dv_init]` → features at `(H, W)`, `(H/2, W/2)`, `(H/4, W/4)`.
    2. **Shared Shot-Gather Data-Domain Encoder**: shared 2D CNN applied to `[B*S, 3, n_rec, nt]`, downsampling time/receiver dims, followed by `AdaptiveAvgPool2d((H//4, W//4))` (or dynamic target size matching the model bottleneck), then **masked mean + masked max pooling** across the shot dimension `S` using `shot_mask` → `[B, C_data, H_b, W_b]`.
    3. **Auxiliary Physics-Domain Encoder**: 3-scale Conv+GroupNorm+SiLU ResBlocks on `[illumination_norm, g_sparse_norm, coord_z, coord_x]` → features at `(H, W)`, `(H/2, W/2)`, `(H/4, W/4)`.
    4. **FiLM Conditioning**: MLP mapping `q_k` → `(gamma, beta)` modulating the bottleneck fusion and decoder blocks.
    5. **Cross-Modal Bottleneck Fusion**: concatenates Model, Data, and Auxiliary bottleneck features, applies channel-attention gating + residual fusion block.
    6. **Multi-Scale Dual-Head Decoder**:
       - Upsamples cleanly via bilinear interpolation + Conv (handles arbitrary `(nz, nx)`, including odd/non-power-of-2 sizes like `70x70` OpenFWI and `48x64` / `128x256`!).
       - **Low-wavenumber head**: predicts coarse smooth gradient $\hat{g}_{\text{low}}$ from the `1/2` / coarse decoder stage upsampled smoothly.
       - **High-wavenumber head**: predicts fine structural detail $\hat{g}_{\text{high}}$ from full-resolution skip-fused features.
       - Combines via $\hat{g}_k = \hat{g}_{\text{low}} + \text{gate}(q_k) \cdot \hat{g}_{\text{high}}$.
       - **Uncertainty head**: predicts $\hat{\sigma}_k = \text{Softplus}(\text{raw}) + \text{eps\_sigma} > 0$ (`[B, 1, nz, nx]`).
  - Returns dict: `{"g_pred": Tensor[B,1,nz,nx], "g_low": Tensor[B,1,nz,nx], "g_high": Tensor[B,1,nz,nx], "sigma": Tensor[B,1,nz,nx]}`.

### 3.3 `losses/gradient_losses.py`
- `Stage2GradientLoss(mu: float = 0.5, lambda_unc: float = 0.1, eps: float = 1e-4, detach_err_for_unc: bool = False)`:
  - Given `g_pred` (already in normalized gradient space), `sigma`, and `g_ref_norm` (target in normalized gradient space):
    - $\mathcal{L}_{\text{pred}} = \text{mean}_b\left(1 - \cos(\hat{g}_b, g_b^{\text{ref}})\right) + \mu \cdot \text{mean}(|\hat{g} - g^{\text{ref}}|)$
    - Let $e = |\hat{g} - g^{\text{ref}}|$ (optionally detached if `detach_err_for_unc=True`):
      $$\mathcal{L}_{\text{unc}} = \text{mean}\left(\frac{e}{\hat{\sigma} + \varepsilon} + \log(\hat{\sigma} + \varepsilon)\right)$$
    - Total loss: $\mathcal{L} = \mathcal{L}_{\text{pred}} + \lambda_\sigma \mathcal{L}_{\text{unc}}$.
  - Returns dict: `{"loss": Tensor, "loss_pred": Tensor, "loss_cos": Tensor, "loss_l1": Tensor, "loss_unc": Tensor}`.

### 3.4 `data/torch_dataset.py`
- `FWIGradientDataset(manifest_path: str | Path, split: str = "train", scenarios: Sequence[str] | None = None, max_iters: int = 20, normalizer: SignPreservingNormalizer | None = None)`:
  - Loads sample list from `manifest.json` filtered by `split` (and optional `scenarios`).
  - `__getitem__` loads the `.pt` sample on CPU and returns tensors:
    `v_k [1,nz,nx]`, `v_init [1,nz,nx]`, `v_true [1,nz,nx]`, `r_wave [S,n_rec,nt]`, `r_env [S,n_rec,nt]`, `r_phase [S,n_rec,nt]`, `illumination [1,nz,nx]`, `g_sparse [1,nz,nx]`, `g_ref [1,nz,nx]`, `g_ref_norm [1,nz,nx]`, `g_ref_scale [1]`, `q_k [3]`, `scenario` (str).
- `collate_fwi_batch(batch: list[dict]) -> dict`:
  - Pads `r_wave`, `r_env`, `r_phase` along shot dimension `S` to `max_S` in batch and builds `shot_mask [B, max_S]` (`torch.bool`).

### 3.5 `eval/metrics.py` Extension
- Add `uncertainty_calibration_metrics(g_pred_norm: Tensor, g_ref_norm: Tensor, sigma: Tensor, eps: float = 1e-4) -> dict[str, float]`:
  - Computes `pearson_r`, `spearman_r`, `nll`, `mean_sigma`, `mean_abs_err` over flattened spatial pixels (pure PyTorch, handling constant-tensor edge cases gracefully).

### 3.6 `scripts/train_stage2.py` & Configs
- `scripts/train_stage2.py --config configs/stage2_smoke.yaml --device auto`:
  - If `dataset.manifest` does not exist and `dataset.auto_build` is true, automatically invokes Stage 1 `build_trajectory_samples` first!
  - Trains `MultiModalGradientNet` with AdamW + CosineAnnealingLR + gradient clipping (`max_norm=1.0`).
  - Evaluates after each epoch on `val` split (overall + per-scenario breakdown: cosine similarity, rel L2 error, uncertainty Pearson/Spearman correlation, inference time per sample).
  - Saves `best.pt`, `last.pt`, and `stage2_metrics.json` to `output_dir`.
- `configs/stage2_smoke.yaml`: fast 3-epoch smoke test using Stage 1 smoke dataset (`48x64`).
- `configs/stage2_openfwi.yaml`: production-ready config using `F:\OpenFWI\Vel Family\CurveVel_A\model\model1.npy` (resized to `64x64`, `nt=350`, `n_shots=8`, `n_models=20`, `n_iters=4`, 5 scenarios → 400 trajectory samples, 40 epochs on RTX 5060 Ti).

### 3.7 `tests/test_stage2_model.py`
Pytest suite verifying:
1. `SignPreservingNormalizer`: odd symmetry, boundedness, exact `denormalize` round-trip, zero-tensor safety.
2. `split_wavenumber`: `low + high == original` within `1e-5`, and `low` has strictly lower high-frequency energy than `original`.
3. `MultiModalGradientNet`: forward/backward pass with variable shot counts (`shot_mask`) and non-power-of-2 grid (`70x70` OpenFWI native size + `48x64`), positive finite `sigma`, and non-zero gradients on all 3 encoders and FiLM layer.
4. `Stage2GradientLoss`: zero prediction loss for identical normalized tensors; uncertainty loss decreases when `sigma` matches true error magnitude.
5. `FWIGradientDataset` + `collate_fwi_batch` + 2-epoch end-to-end training smoke test verifying decreasing training loss and valid `stage2_metrics.json` output.

## 4. Acceptance Criteria
- All Stage 1 tests (`tests/test_stage1_physics.py`) AND Stage 2 tests (`tests/test_stage2_model.py`) pass on both Mac and `winpc` (`cu128`, RTX 5060 Ti).
- `train_stage2.py --config configs/stage2_smoke.yaml --device cuda` runs end-to-end on `winpc` cleanly.
