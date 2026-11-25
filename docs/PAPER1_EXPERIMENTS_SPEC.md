# Paper 1 Formal Experiments & Layer-by-Layer Ablation Specification

## 1. Objective
Build a clean, resumable, publication-grade experimental suite for **Paper 1** (*Physics-Consistent Multi-Modal Gradient and Uncertainty Prediction for Full Waveform Inversion*) covering:
1. **Multi-family OpenFWI stratified trajectory dataset** (`CurveVel_A`, `CurveVel_B`, `FlatVel_A`, `FlatVel_B`) with strict model-disjoint train/val/test splits + unseen Out-of-Distribution (OOD) complex fault-and-salt models.
2. **6-rung layer-by-layer ablation ladder (`A0` → `A5`)** isolating each architectural and physics-loss contribution on identical splits and seeds.
3. **Multi-model closed-loop FWI benchmark** (mean ± std across all unseen test models + OOD fault-salt models under both normal and 5 Hz low-frequency-missing conditions), reporting full accuracy, stability, and speedup metrics + LaTeX `booktabs` tables + 300 DPI figures.

## 2. Minimal Extensions to Existing Modules (100% Backwards-Compatible)
Keep code clean and direct without excessive defensive boilerplate. Do NOT touch the root PDF or break any existing tests (all 61 existing tests must continue passing).

### 2.1 `src/fwi_grad/data/dataset_builder.py`
- Extend `build_trajectory_samples` to optionally accept `model.sources`: a list of dicts `[{"name": "CurveVel_A", "npy_path": "...", "n_models": 10, "start_index": 0}, ...]`.
- When `model.sources` is present:
  - Assigns each source's models stratified into `train/val/test` by `split_fractions` (zero model leakage across splits, every family represented in train/val/test!).
  - Records `"family"` in both `manifest.json` entries and each `.pt` sample's `meta` dict.
- When `model.sources` is absent, existing single-source behavior is 100% unchanged.

### 2.2 `src/fwi_grad/models/gradient_net.py`
- Add two optional boolean flags to `MultiModalGradientNet.__init__` (both defaulting to `False` so existing checkpoints and callers are 100% unaffected):
  - `disable_model_branch: bool = False`: zeros `m_full, m_mid, m_coarse` features before fusion/skip connections.
  - `disable_data_branch: bool = False`: zeros `data` shot-encoder bottleneck features before fusion.

### 2.3 `scripts/run_paper1_ablation.py` & Configs
- `scripts/run_paper1_ablation.py --config <yaml> --device auto`:
  - **Step 1 (Dataset)**: builds multi-family dataset if `manifest.json` does not exist yet (resumable!).
  - **Step 2 (6-Rung Layer-by-Layer Ablation)**: runs and evaluates each rung on the exact same `train/val/test` split (skipping any rung whose `<output_dir>/<rung>/eval.json` already exists unless `--force` is passed):
    - `A0_sparse_only`: `disable_model_branch=True`, `disable_data_branch=True`, `lambda_unc=0.0`
    - `A1_no_data`: `disable_model_branch=False`, `disable_data_branch=True`, `lambda_unc=0.0`
    - `A2_multimodal_no_unc`: full 3-branch, `lambda_unc=0.0`
    - `A3_stage2_full`: full 3-branch, `lambda_unc=0.1` (Stage 2 supervised + uncertainty)
    - `A4_stage3_dir_str`: initialized from `A3_stage2_full/best.pt`, fine-tuned with `L_pred + L_unc + L_str + L_dir` (`lambda_data=0.0`)
    - `A5_proposed_full`: initialized from `A3_stage2_full/best.pt`, fine-tuned with `L_pred + L_unc + L_str + L_dir + L_data`
  - Evaluates **all 6 rungs on the held-out `test` split** (overall + per-scenario + per-family):
    - `cosine_similarity`, `rel_l2_error`, `directional_derivative_relative_error` (unbiased full-shot), `one_step_descent_success_rate`, `spearman_r` (uncertainty calibration), `inference_ms`.
  - **Step 3 (Multi-Model Closed-Loop Benchmark)**:
    - Runs closed-loop FWI across all held-out `test` models + `n_ood_models` unseen synthetic complex fault-salt models under both `f_cut = 0.0 Hz` (normal) and `f_cut = 5.0 Hz` (low-frequency missing), comparing:
      - `full_adjoint`, `sparse_adjoint`, `A2_guided`, `A3_guided`, `A5_proposed_guided`.
    - Aggregates **mean ± std** of `final_full_loss`, `mae`, `rmse`, `re`, `ssim`, `snr_db`, `predicted_accept_rate`, `descent_success_rate`, `grad_time_mean`, `iter_time_mean`, and `% time reduction vs full_adjoint`.
  - **Step 4 (Publication Outputs)**:
    - Writes `paper1_ablation_summary.json`
    - Writes `paper1_ablation_tables.tex` (Table 1: 6-rung layer-by-layer ablation on held-out test set; Table 2: multi-model closed-loop normal & 5 Hz low-freq + OOD comparison with mean ± std and time reduction %)
    - Generates 300 DPI PNG + PDF ablation bar chart (`ablation_ladder.png/.pdf`) + closed-loop summary figure (`closed_loop_summary.png/.pdf`) + sample gradient/uncertainty visual (`best_gradient_uncertainty.png/.pdf`).

### 2.4 Configs
- `configs/paper1_ablation_smoke.yaml`: fast 2-epoch / 2-model smoke test verifying the entire 6-rung ablation + closed-loop + LaTeX/figure export pipeline in < 30 seconds.
- `configs/paper1_ablation_formal.yaml`: production formal Paper 1 experiment on `winpc` RTX 5060 Ti:
  - 4 OpenFWI families (`CurveVel_A`, `CurveVel_B`, `FlatVel_A`, `FlatVel_B` from `F:\OpenFWI\Vel Family\<family>\model\model1.npy`), `n_models: 10` per family (40 total → 28 train / 6 val / 6 test, stratified!), `grid: 64x64`, `nt: 500`, `n_shots: 8`, `n_rec: 56`, `sparse_ratio: 0.25`, `n_iters: 4`, 5 scenarios → **800 samples**.
  - Stage 2 rungs (`A0`–`A3`): `epochs: 35`, `batch_size: 8`, `lr: 5e-4`, `num_workers: 0`.
  - Stage 3 rungs (`A4`–`A5`): `epochs: 12`, `batch_size: 8`, `lr: 1e-4`, `lambda_dir: 0.05`, `lambda_data: 0.3`, `lambda_str: 0.01`, `warmup_epochs: 4`, `physics_subset_size: 2`, `n_physics_shots: 2`.
  - Closed-loop benchmark: all 6 held-out OpenFWI test models + 2 OOD fault-salt models, `n_iters: 25`, `lr: 5.0`, `sparse_ratio: 0.25`, `n_check_shots: 2`, tested at both `f_cut = 0.0 Hz` and `f_cut = 5.0 Hz`.

### 2.5 `tests/test_paper1_ablation.py`
- Verifies stratified multi-family dataset split (all families in every split, zero model ID overlap).
- Verifies `disable_model_branch` and `disable_data_branch` cleanly zero the respective branch without breaking gradients on active branches.
- Verifies `run_paper1_ablation.py --config configs/paper1_ablation_smoke.yaml` completes end-to-end, is resumable on a second run, and writes valid JSON, `.tex`, and 300 DPI PNG/PDF figures.
