# Paper 1 Unified Notation System (Minimal IEEE TGRS Standard)

> **Design Principle**: Every symbol has exactly **one** meaning across the entire manuscript, figures, and Algorithm 1. Avoid nested subscripts/superscripts (e.g., $\hat{g}_{\text{sp},k}^{\text{phys}}$), remove minor numerical stabilizers ($\epsilon, \epsilon_u, \epsilon_v, \beta, \rho$) from main display equations (report them once in *Implementation Details*), and keep Algorithm 1 concise (12–14 lines).

## 1. Core Symbol Table

| Symbol | Meaning | Scope |
| :--- | :--- | :--- |
| $v_k, v_0$ | Velocity model at iteration $k$, initial model | Sec. III–IV |
| $u_s$ | Acoustic forward wavefield of shot $s$ | Sec. III-A |
| $d_s^{\text{obs}}, d_s^{\text{syn}}$ | Observed and synthetic shot gathers | Sec. III-A |
| $\mathcal{S}, \mathcal{S}_{\text{sp}}, \mathcal{S}_{\text{chk}}$ | Full, sparse ($25\%$), and out-of-sample verification shot sets | Sec. III–IV |
| $J_{\mathcal{S}}(v)$ | Waveform $L_2$ misfit evaluated on shot set $\mathcal{S}$ | Sec. III–IV |
| $r_k = (r_{\text{w}}, r_{\text{e}}, r_{\text{p}})$ | Multi-modal waveform, envelope, and phase residuals | Sec. III-A/B |
| $I_k$ | Source-side wavefield illumination map | Sec. III-A/B |
| $g_k^{\text{sp}}, g_k^{\text{ref}}$ | Sparse-shot and full-shot reference adjoint gradients | Sec. III–IV |
| $q_k$ | Acquisition/iteration conditioning vector | Sec. III-B |
| $G_\psi$ | Multi-modal gradient and uncertainty network | Sec. III-B/C/D |
| $\hat{g}_k, \hat{\sigma}_k$ | Predicted gradient and pixelwise uncertainty map | Sec. III–IV |
| $\mathcal{N}(\cdot), \mathcal{N}^{-1}(\cdot)$ | Signed-log normalization and inverse operator | Sec. III-B |
| $\mathcal{L}_{\text{sup}}, \mathcal{L}_{\text{unc}}$ | Supervised gradient loss and Laplace NLL loss | Sec. III-C |
| $\mathcal{L}_{\text{str}}, \mathcal{L}_{\text{dir}}, \mathcal{L}_{\text{data}}$ | Structural TV, directional-derivative, and 1-step descent losses | Sec. III-C |
| $C_k$ | Pixelwise confidence weight derived from $\hat{\sigma}_k$ | Sec. III-C/D |
| $\eta_k, \delta v_k$ | Adaptive step size and damped-Adam velocity update | Sec. III-D |

## 2. Equation & Pseudocode Simplification Rules

1. **Drop redundant arguments**: Write $J_{\mathcal{S}}(v)$ instead of $J_{\mathcal{S}}(v; f_{\text{cut}})$ (state once that high-pass filtering is applied to gathers when $f_{\text{cut}} > 0$).
2. **Shorten subscripts**: Use $r_{\text{w}}, r_{\text{e}}, r_{\text{p}}$ instead of $r_{\text{wave}}, r_{\text{env}}, r_{\text{phase}}$; use $g_k^{\text{sp}}$ instead of $g_{\text{sp},k}$.
3. **Move hyperparameters out of display equations**: Keep equations focused on the mathematical idea; list exact constants ($\alpha=10, \beta=5, \kappa=2, \varepsilon_{\text{fd}}=0.05$, etc.) concisely in one paragraph in Section IV-A.
4. **Slim Algorithm 1 to ~13 lines**: Show only the clean logical flow (sparse adjoint $\to$ network prediction $\to$ confidence-weighted damped-Adam proposal $\to$ out-of-sample 4-tier verification $\to$ state commit/update), omitting low-level clamping and step-size algebra.
