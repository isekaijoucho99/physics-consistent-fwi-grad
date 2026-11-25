# Physics-Consistent Multi-Modal Gradient and Uncertainty Prediction for Efficient FWI

PyTorch implementation of a physics-consistent multi-modal gradient and uncertainty prediction framework for 2D acoustic full waveform inversion (FWI).

## Overview
Adjoint-state FWI evaluates multi-shot forward and backward wave simulations at every iteration. This repository pairs a pure-PyTorch 2D acoustic FDTD solver with a compact three-branch residual network (`MultiModalGradientNet`, 1.30M parameters) that predicts full-shot-equivalent gradients and pixelwise aleatoric uncertainty from sparse-shot adjoint cues:

- **Dimensionless, Resolution- and Shot-Agnostic Design:** Normalizes velocity states by the initial-model mean, aggregates arbitrary shot counts via permutation-invariant masked mean-max pooling, omits fixed grid coordinates, and adds dual-scale residual corrections on top of log-compressed sparse gradients.
- **Physics-Consistent Two-Stage Training:** Combines supervised directional/amplitude and Laplace NLL calibration (`Stage 2`) with edge-weighted structural continuity ($\mathcal{L}_{\text{str}}$), multi-probe finite-difference directional-derivative matching ($\mathcal{L}_{\text{dir}}$), and differentiable one-step PDE descent ($\mathcal{L}_{\text{data}}$) (`Stage 3`).
- **Safeguarded Misfit-Proportional Closed-Loop FWI:** Anneals surrogate residual corrections in proportion to out-of-sample check-shot misfit reduction ($w_k = \min(1, J_{\mathcal{S}_{\text{chk}}}(v_k)/J_{\mathcal{S}_{\text{chk}}}(v_0))$), weights reversible Adam updates by calibrated confidence, and enforces wave-equation descent via a four-tier verification ladder (immediate acceptance, backtracking line search, sparse blending, and full-adjoint fallback).

## Repository Layout
```text
├── configs/                    # YAML configs for smoke tests, ablations, and cross-geology benchmarks
├── docs/                       # Notation system and stage specifications
├── scripts/
│   ├── run_stage1_baseline.py  # Gradient check, baseline FWI, and trajectory dataset builder
│   ├── train_stage2.py         # Supervised + uncertainty pretraining
│   ├── train_stage3.py         # Physics-consistent fine-tuning and guided FWI
│   ├── run_paper1_ablation.py  # Resumable 6-rung ablation (A0-A5) and multi-model closed-loop benchmark
│   ├── run_paper1_benchmarks.py# Zero-shot Marmousi2 and SEG/EAGE Overthrust cross-resolution benchmarks
│   ├── plot_paper1_figures.py  # 300 DPI PNG/PDF figure and LaTeX booktabs table generator
│   ├── export_fig1_assets.py   # Marmousi2 intermediate tensor exporter for architecture overview plot
│   ├── plot_framework_diagram.py # Publication-style architecture overview plotter
│   └── run_stage4_reparam.py   # Multi-scale neural velocity reparameterization (follow-up study)
├── src/fwi_grad/               # Core library (physics, data, models, losses, inversion, eval)
├── tests/                      # Unit and end-to-end smoke tests (70 tests)
├── pyproject.toml
└── requirements.txt
```

## Installation
Requires Python 3.10+ and PyTorch 2.3+ (CPU or CUDA; no compiled C++/CUDA extensions required):
```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### 1. Run Unit & Smoke Tests
```bash
pytest tests -q
```

### 2. Fast End-to-End Smoke Run (CPU or GPU)
```bash
python scripts/run_stage1_baseline.py --config configs/stage1_smoke.yaml --device auto
python scripts/train_stage2.py --config configs/stage2_smoke.yaml --device auto
python scripts/train_stage3.py --config configs/stage3_smoke.yaml --device auto
python scripts/run_paper1_ablation.py --config configs/paper1_ablation_smoke.yaml --device auto
```

### 3. Reproducing Formal Experiments
Download the public [OpenFWI](https://openfwi-lanl.github.io/) velocity families (`CurveVel_A`, `CurveVel_B`, `FlatVel_A`, `FlatVel_B`) and standard 2D [Marmousi2](https://uh.edu/nsm/geoscience/research/marmousi2/) / [SEG/EAGE Overthrust](https://wiki.seg.org/wiki/SEG/EAGE_Salt_and_Overthrust_Models) velocity sections as 2D/3D NumPy `.npy` arrays (`m/s`), and place them under `data/` as follows:
```text
data/
├── openfwi/
│   └── Vel Family/
│       ├── CurveVel_A/model/model1.npy
│       ├── CurveVel_B/model/model1.npy
│       ├── FlatVel_A/model/model1.npy
│       └── FlatVel_B/model/model1.npy
└── benchmarks/
    ├── marmousi2.npy
    └── overthrust.npy
```
Then run from the repository root (note config relative paths resolve relative to `configs/`):
```bash
# 4-family 800-sample active-descent dataset, A0-A5 ablation, and OpenFWI/OOD closed-loop benchmarks
python scripts/run_paper1_ablation.py --config configs/paper1_ablation_formal.yaml --device cuda

# Zero-shot cross-resolution Marmousi2 & SEG/EAGE Overthrust benchmarks
python scripts/run_paper1_benchmarks.py --config configs/paper1_benchmarks.yaml --device cuda
```
Both scripts cache completed rungs/cases and resume automatically; pass `--force` to recompute from scratch.

> **Experimental Scope Note:** All synthetic observations are generated with the same 2D constant-density acoustic FDTD solver used in inversion (noise-free on the cross-geology benchmarks, and 15 dB Gaussian noise in the noisy OpenFWI scenario). Time reductions are reported as positive percentages $100(1 - t / t_{\text{full}})$ based on total wall-clock iteration time including verification.

## Citation
```bibtex
@article{fwi_grad_surrogate_2026,
  title   = {Physics-Consistent Multi-Modal Gradient and Uncertainty Prediction for Efficient Full Waveform Inversion},
  author  = {Author, First and Co-authors},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year    = {2026},
  note    = {Submitted}
}
```

## License
Released under the [MIT License](LICENSE).
