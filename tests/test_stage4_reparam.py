"""Multi-scale VJP, transactional inversion and four-way benchmark checks."""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from fwi_grad.inversion.reparam_fwi import ReparamFWIConfig, run_reparam_guided_fwi
from fwi_grad.losses.physics_losses import subset_misfit
from fwi_grad.models import MultiModalGradientNet, SignPreservingNormalizer
from fwi_grad.models.reparam_net import MultiScaleVelocityReparam
from fwi_grad.physics.acoustic2d import Acoustic2DSolver
from fwi_grad.physics.geometry import create_surface_geometry
from fwi_grad.physics.wavelet import ricker_wavelet


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_net(dtype=torch.float32):
    initial = torch.linspace(1550, 4950, 99, dtype=dtype).reshape(9, 11)
    net = MultiScaleVelocityReparam(initial, low_hidden=8, high_hidden=10, n_fourier=6)
    return initial, net


def test_initial_identity_and_strict_bounds():
    initial, net = make_net()
    fields = net.components(1., 1.)
    assert set(fields) == {"v_total", "v_background", "delta_low", "delta_high", "delta_grid"}
    for key in ("v_total", "v_background"):
        torch.testing.assert_close(fields[key], initial, rtol=0, atol=0)
    for key in ("delta_low", "delta_high", "delta_grid"):
        assert torch.count_nonzero(fields[key]) == 0
    with torch.no_grad():
        net.grid_residual.fill_(100)
        assert net.components(0., 1.)["delta_grid"].abs().max() <= .25
        for bias in (-100., 100.):
            net.low_branch[-1].bias.fill_(bias)
            velocity = net()
            assert (velocity > 1500.).all() and (velocity < 5000.).all()


def test_shallow_latent_perturbation_has_local_support():
    initial = torch.full((64, 64), 2500.)
    net = MultiScaleVelocityReparam(initial, low_hidden=8, high_hidden=8, n_fourier=4)
    latent = net.low_branch.latent
    assert latent.shape[-2:] == (16, 16)
    with torch.no_grad():
        latent[0, :, 0, 8] += 1.
        change = net(0., 0.) - initial
    assert change[:16].any()
    assert not change[16:].any()

    shallow = torch.zeros_like(initial)
    shallow[0] = 1.
    net.project_gradient(shallow, 0., 0.)
    assert latent.grad[:, :, :3].any()
    assert not latent.grad[:, :, 3:].any()


def test_exact_vjp_directional_derivative():
    _, net = make_net(torch.float64)
    rng = torch.Generator().manual_seed(41)
    params = list(net.parameters())
    with torch.no_grad():
        for p in params:  # Exercise hidden layers and a nonzero grid as well as the heads.
            p.add_(.1 * torch.randn(p.shape, dtype=p.dtype, generator=rng))
    g_v = torch.randn(net.grid_residual.shape, dtype=torch.float64, generator=rng)
    directions = [torch.randn(p.shape, dtype=p.dtype, generator=rng) for p in params]
    grads = net.project_gradient(g_v, .6, .4)
    assert len(grads) == len(params)
    projected = sum((p.grad * d).sum() for p, d in zip(params, directions))
    eps = 1e-5
    with torch.no_grad():
        for p, d in zip(params, directions):
            p.add_(d, alpha=eps)
        plus = (g_v * net(.6, .4)).sum()
        for p, d in zip(params, directions):
            p.add_(d, alpha=-2 * eps)
        minus = (g_v * net(.6, .4)).sum()
    assert projected.item() == pytest.approx(((plus - minus) / (2 * eps)).item(), rel=1e-3)


def test_branch_gradients_clear_on_freeze():
    initial, net = make_net()
    for weight in (0., .5, 0.):
        net.project_gradient(torch.ones_like(initial), weight, weight)
        for branch in (list(net.high_branch.parameters()), [net.grid_residual]):
            assert any(p.grad is not None and p.grad.any() for p in branch) == bool(weight)
        assert net.low_branch[-1].bias.grad.abs().item() > 0
        for idx in (0, 2):
            assert net.low_branch[idx].weight.grad.any()
            assert bool(net.high_branch[idx].weight.grad.any()) == bool(weight)


@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def experiment(request):
    geom = create_surface_geometry(13, 17, 10., 10., 3, 9, margin=2)
    solver = Acoustic2DSolver(.001, npml=3, shot_batch_size=2)
    wave = ricker_wavelet(80, solver.dt, 30., device=request.param)
    initial = wave.new_full((13, 17), 2100.)
    truth = initial + 100 * torch.exp(-torch.arange(13, device=wave.device)[:, None] / 7)
    obs = solver.forward(truth, wave, geom, return_illumination=False)["seismograms"]
    return initial, truth, obs, solver, wave, geom


@pytest.mark.parametrize("guided", [False, True])
def test_filtered_full_shot_descent(experiment, guided):
    initial, truth, obs, solver, wave, geom = experiment
    torch.manual_seed(7)
    model = MultiModalGradientNet(base_channels=4, cond_dim=8) if guided else None
    cfg = ReparamFWIConfig(n_iters=5, low_hidden=8, high_hidden=8, n_fourier=4,
                           sparse_ratio=2/3, n_check_shots=1, sigma_threshold=10.)
    result = run_reparam_guided_fwi(*experiment, cfg, grad_model=model)
    history = result["full_loss_history"]
    assert len(history) == cfg.n_iters + 1 and history[-1] < history[0]
    for field, expected in ((initial, history[0]), (result["v_final"], history[-1])):
        actual = subset_misfit(field, obs, solver, wave, geom, range(geom.n_shots), 5.)
        assert actual.item() == pytest.approx(expected, rel=1e-5)
    assert len(result["metric_history"]) == len(result["decisions"]) == cfg.n_iters
    assert result["detail_weights"][:2] == [0., 0.]
    assert result["v_background"].shape == truth.shape
    assert all(math.isfinite(v) for v in result["final_metrics"].values())
    assert result["final_metrics"]["mae"] == pytest.approx((result["v_final"] - truth).abs().mean().item())
    if guided:
        assert model.training and all(p.grad is None for p in model.parameters())


def test_rejected_steps_keep_last_verified_model(experiment):
    initial, _, _, solver, wave, geom = experiment
    obs = solver.forward(initial, wave, geom, return_illumination=False)["seismograms"]
    cfg = ReparamFWIConfig(n_iters=4, detail_unlock_frac=0., grid_unlock_frac=0.,
                           enable_fallback=False, max_backtracks=1)
    result = run_reparam_guided_fwi(initial, initial, obs, solver, wave, geom, cfg)
    assert result["decisions"] == ["rejected"] * cfg.n_iters
    assert result["detail_weights"] == [0.] * cfg.n_iters
    torch.testing.assert_close(result["v_final"], initial, atol=0., rtol=0.)


def test_large_adam_proposal_obeys_velocity_cap(experiment):
    initial = experiment[0]
    cfg = ReparamFWIConfig(n_iters=1, lr_low=.05, max_backtracks=0, enable_fallback=False)
    result = run_reparam_guided_fwi(*experiment, cfg)
    step = (result["v_final"] - initial).abs().max().item()
    assert 0. < step <= 20.


def test_full_adjoint_matches_projection_point_during_ramp(experiment, monkeypatch):
    _, _, obs, solver, wave, geom = experiment
    project = MultiScaleVelocityReparam.project_gradient

    def checked_project(net, g_v, detail, grid):
        v = net(detail, grid)
        reference = solver.adjoint_gradient(v, obs, wave, geom, f_cut=5.)["grad"]
        reference = reference / torch.quantile(reference.abs(), .99).clamp_min(1e-12)
        torch.testing.assert_close(g_v, reference, atol=1e-12, rtol=1e-5)
        return project(net, g_v, detail, grid)

    monkeypatch.setattr(MultiScaleVelocityReparam, "project_gradient", checked_project)
    cfg = ReparamFWIConfig(n_iters=8, illumination_precond=False, lr_low=.002,
                           detail_unlock_frac=.1, n_check_shots=3, sparse_ratio=1.)
    result = run_reparam_guided_fwi(*experiment, cfg)
    assert any(0. < w < 1. for w in result["detail_weights"])
    assert 1. in result["detail_weights"]


def test_rollback_restores_adam_moments(experiment):
    class PrescribedDirection(torch.nn.Module):
        normalizer = SignPreservingNormalizer()

        def __init__(self, fail_step: int | None = None):
            super().__init__()
            self.fail_step = fail_step

        def forward(self, g_sparse, q_k, **inputs):
            scale = self.normalizer.compute_scale(g_sparse)
            step_idx = round(q_k[0, 1].item() * 25.0)
            direction = -100 * g_sparse if step_idx == self.fail_step else g_sparse
            pred, _ = self.normalizer.normalize(direction, scale)
            sigma = 0.8 if step_idx == 1 else 0.0
            return {"g_pred": pred, "sigma": torch.full_like(pred, sigma)}

    def run(count, fallback):
        # The clean run uses a half step on the second update, matching post-rejection damping.
        cfg = ReparamFWIConfig(n_iters=count, lr_low=.002, enable_fallback=fallback,
                               kappa=0., sigma_threshold=.75 if count == 2 else 1.,
                               grid_unlock_frac=1., detail_unlock_frac=1., illumination_precond=False)
        return run_reparam_guided_fwi(*experiment, cfg, grad_model=PrescribedDirection(1 if count == 3 else None))

    clean = run(2, False)
    rejected = run(3, False)
    assert rejected["decisions"] == ["predicted", "rejected", "predicted"]
    assert rejected["full_loss_history"][1] == rejected["full_loss_history"][2]
    torch.testing.assert_close(rejected["v_final"], clean["v_final"], rtol=0., atol=0.)
    recovered = run(3, True)
    assert recovered["decisions"][1] == "sparse_fallback"
    assert recovered["loss_history"][1] < recovered["loss_history"][0]


def test_smoke_cli(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "scripts/run_stage4_reparam.py"),
                    "--config", str(ROOT / "configs/stage4_smoke.yaml"), "--device", "auto",
                    "--output-dir", str(tmp_path)], cwd=tmp_path, check=True)
    report = json.loads((tmp_path / "stage4_metrics.json").read_text(encoding="utf-8"))
    has_checkpoint = (ROOT / "outputs/stage3_smoke/best_stage3.pt").is_file()
    assert report["initialization"] == ("stage3_checkpoint" if has_checkpoint else "scratch")
    assert report["f_cut"] == 5.
    for key in ("full_grid_fwi", "stage3_grid_fwi", "full_reparam_fwi", "stage3_reparam_fwi"):
        method = report["closed_loop"][key]
        history = method["full_loss_history"]
        assert len(history) == 7 and len(method["metric_history"]) == 6
        assert history[0] == pytest.approx(report["closed_loop"]["initial_full_loss"], rel=1e-5)
        assert history[-1] == method["final_full_loss"]
        assert all(math.isfinite(v) for v in method["final_metrics"].values())
    for name in ("velocity_and_profile", "closed_loop_convergence"):
        assert (tmp_path / (name + ".png")).read_bytes().startswith(b"\x89PNG")
        assert (tmp_path / (name + ".pdf")).read_bytes().startswith(b"%PDF")


def test_missing_checkpoint_requires_explicit_scratch(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from run_stage4_reparam import _load_predictor

    config = {"stage3_checkpoint": "missing.pt", "grad_model": {"base_channels": 4}}
    with pytest.raises(FileNotFoundError, match="missing.pt"):
        _load_predictor(config, tmp_path, "cpu")
    model, source = _load_predictor(dict(config, allow_scratch=True), tmp_path, "cpu")
    assert source == "scratch" and isinstance(model, MultiModalGradientNet)
