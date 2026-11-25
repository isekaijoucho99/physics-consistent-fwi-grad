"""Acceptance tests for differentiable physics training and verified FWI updates."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import yaml

from fwi_grad.inversion.guided_fwi import GuidedFWIConfig, run_physics_guided_fwi
from fwi_grad.losses.physics_losses import (
    ProgressivePhysicsWeightSchedule, confidence_weight, directional_derivative_loss,
    one_step_data_consistency_loss, structural_continuity_loss,
)
from fwi_grad.models import MultiModalGradientNet, SignPreservingNormalizer
from fwi_grad.physics.acoustic2d import Acoustic2DSolver
from fwi_grad.physics.geometry import create_surface_geometry
from fwi_grad.physics.residuals import compute_multimodal_residuals
from fwi_grad.physics.wavelet import ricker_wavelet


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request):
    return torch.device(request.param)


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def experiment(device):
    geom = create_surface_geometry(13, 17, 10., 10., 3, 9, margin=2)
    solver = Acoustic2DSolver(0.001, npml=3, shot_batch_size=2)
    wave = ricker_wavelet(56, solver.dt, 34., device=device, dtype=torch.float64)
    v = wave.new_full((geom.nz, geom.nx), 2100.)
    truth = v + 100 * torch.exp(-torch.arange(geom.nz, device=device)[:, None] / 7)
    obs = solver.forward(truth, wave, geom, return_illumination=False)["seismograms"]
    return v, truth, obs, solver, wave, geom


@pytest.mark.parametrize("axis", [-1, -2])
def test_structure_and_detached_interface_weights(axis, device):
    aligned = torch.zeros(9, 9, device=device)
    aligned[4:] = 1
    in_strata = aligned.roll(2, dims=0)
    in_strata[:2] = 0  # a single jump of the same magnitude
    if axis == -1:
        aligned, in_strata = aligned.T, in_strata.T
    v = (2000 + 500 * aligned).requires_grad_()
    aligned.requires_grad_()
    edge = structural_continuity_loss(aligned, v)
    smooth = structural_continuity_loss(in_strata, v)
    assert smooth > 1e4 * edge
    torch.testing.assert_close(smooth, smooth.new_tensor(1 / 8))
    edge.backward()
    assert v.grad is None and aligned.grad is not None
    assert structural_continuity_loss(torch.ones_like(v), v) == 0


def test_confidence_and_progressive_weights(device):
    sigma = torch.tensor([0., 0.5, 1., 100.], device=device)
    torch.testing.assert_close(confidence_weight(sigma), sigma.new_tensor([1., 0.5, 1 / 3, 0.1]))
    schedule = ProgressivePhysicsWeightSchedule(2., 0.4, 0.1, 4)
    for epoch, fraction in [(0, 0.), (1, .25), (2, .5), (4, 1.), (5, 1.)]:
        assert schedule(epoch) == pytest.approx((2 * fraction, .4 * fraction, .1 * fraction))
    assert ProgressivePhysicsWeightSchedule(1., 1., 1., 0)(1) == (1., 1., 1.)


def test_directional_derivative_against_subset_adjoint(experiment):
    v, _, obs, solver, wave, geom = experiment
    shots, cutoff = [2, 0], 9.
    grad = solver.adjoint_gradient(v, obs, wave, geom, shots, f_cut=cutoff)["grad"]
    p = grad / grad.norm()
    args = (v, obs, solver, wave, geom, shots)
    correct = directional_derivative_loss(grad, *args, p=p, f_cut=cutoff)
    wrong = (-grad).requires_grad_()
    error = directional_derivative_loss(wrong, *args, p=p, f_cut=cutoff)
    assert correct.item() < 1e-3 and error.item() == pytest.approx(2., abs=1e-3)
    error.backward()
    assert torch.isfinite(wrong.grad).all() and wrong.grad.norm() > 0
    assert directional_derivative_loss(1000 * grad, *args, p=p, f_cut=cutoff) == 10.
    torch.manual_seed(24)
    assert directional_derivative_loss(grad, *args, f_cut=cutoff) < 1e-3


def test_trial_step_matches_filtered_adjoint_and_cap(experiment):
    v, _, obs, solver, wave, geom = experiment
    shots, cutoff = [2, 0], 8.
    ref = solver.adjoint_gradient(v, obs, wave, geom, shots, f_cut=cutoff)
    grad = ref["grad"]
    sigma = torch.zeros_like(v)
    eta, cap = 300. / grad.abs().max().item(), 6.
    for sign in (1., -1.):
        # Supply already-selected observations to exercise the Stage 1 convention.
        result = one_step_data_consistency_loss(sign * grad, sigma, v, obs[shots], solver, wave,
                                                geom, shots, eta, f_cut=cutoff, max_update_mps=cap)
        expected_v = v - sign * cap * grad / grad.abs().max()
        expected = solver.adjoint_gradient(expected_v, obs, wave, geom, shots, f_cut=cutoff)
        assert result["loss_before"].item() == pytest.approx(ref["loss"], rel=1e-10)
        assert result["loss_after"].item() == pytest.approx(expected["loss"], rel=1e-10)
        assert result["descended"].item() == (sign > 0)
        assert (result["loss_data"].item() > 0) == (sign < 0)


@pytest.mark.parametrize("training", [True, False])
def test_physics_sample_matches_full_adjoint(experiment, training, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import train_stage3

    v, truth, obs, solver, wave, geom = experiment
    ref = solver.adjoint_gradient(v, obs, wave, geom)["grad"]
    net = _SparsePredictor(1)
    norm, scale = net.normalizer.normalize(ref)
    survey = SimpleNamespace(solver=solver, wavelet=wave, geom=geom, observations=lambda *_: obs)
    batch = dict(model_id=["model_0"], scenario=["normal"], v_true=truth[None, None],
                 q_k=v.new_zeros(1, 3), g_ref_scale=scale.reshape(1, 1).cpu())
    inputs = dict(v_k=v[None, None], g_sparse=ref[None, None])
    output = dict(g_pred=norm[None, None].requires_grad_(), sigma=torch.zeros_like(v)[None, None])
    rng = torch.Generator().manual_seed(42)
    error, step = train_stage3._physics_sample(
        net, batch, inputs, output, 0, survey, dict(n_physics_shots=1, eta=1.), rng, training,
    )
    assert error.item() < 1e-3
    shots = torch.randperm(geom.n_shots, generator=rng.manual_seed(42))[:1].tolist()
    before = solver.adjoint_gradient(v, obs, wave, geom, shots)["loss"]
    assert step["loss_before"].item() == pytest.approx(before, rel=1e-9)


def test_trial_loss_backpropagates_into_both_network_heads(experiment):
    v, _, obs, solver, wave, geom = experiment
    torch.manual_seed(17)
    net = MultiModalGradientNet(base_channels=4, cond_dim=8).to(v)
    shots = [1]
    sparse = solver.adjoint_gradient(v, obs, wave, geom, shots)
    res = compute_multimodal_residuals(sparse["d_syn"], obs[shots])
    output = net(v[None, None], v[None, None], **{k: t[None] for k, t in res.items()},
                 shot_mask=torch.ones(1, 1, dtype=torch.bool, device=v.device),
                 illumination=sparse["illumination"][None, None], g_sparse=sparse["grad"][None, None],
                 q_k=v.new_tensor([[0., 0., 1 / 3]]))
    scale = net.normalizer.compute_scale(sparse["grad"])
    pred = net.normalizer.denormalize(output["g_pred"][0, 0], scale)
    eta = 2. / pred.detach().abs().max().item()
    result = one_step_data_consistency_loss(pred, output["sigma"][0, 0], v, obs, solver,
                                            wave, geom, shots, eta, rho=1.)
    assert not result["loss_before"].requires_grad and result["loss_after"].requires_grad
    result["loss_data"].backward()
    for param in (net.model_encoder.full.body[0].weight, net.low_head.weight, net.sigma_head.weight):
        assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.norm() > 0


class _SparsePredictor(torch.nn.Module):
    def __init__(self, sign, sigma=0.1):
        super().__init__()
        self.sign, self.sigma = sign, sigma
        self.normalizer = SignPreservingNormalizer()

    def forward(self, g_sparse, **inputs):
        pred, _ = self.normalizer.normalize(self.sign * g_sparse)
        return {"g_pred": pred, "sigma": torch.full_like(pred, self.sigma)}


@pytest.mark.parametrize("sign,blend,fallback,decision", [
    (1, .25, True, "predicted"), (-1, 1., True, "sparse_blend"),
    (-1, .25, True, "full_fallback"), (-1, .25, False, "rejected"),
])
def test_guided_decisions_and_actual_descent(experiment, sign, blend, fallback, decision):
    v, truth, obs, solver, wave, geom = experiment
    cfg = GuidedFWIConfig(n_iters=2, lr=2., sparse_ratio=2/3, n_check_shots=3, f_cut=7.,
                          sparse_blend_weight=blend, enable_fallback=fallback)
    result = run_physics_guided_fwi(v, truth, obs, _SparsePredictor(sign), solver, wave, geom, cfg)
    assert result["decisions"] == [decision] * cfg.n_iters
    assert result["descent_success_rate"] == float(decision != "rejected")
    assert result["predicted_accept_rate"] == float(decision == "predicted")
    for row in result["iterations"]:
        assert (row["loss"] < row["loss_before"]) == (decision != "rejected")
        assert row["grad_time"] == pytest.approx(row["sparse_time"] + row["net_time"])
        assert row["iter_time"] >= row["grad_time"] + row["verify_time"] > row["grad_time"] > 0
    final = result["v_final"]
    assert row["check_shots"] == [1]
    actual = solver.adjoint_gradient(final, obs, wave, geom, row["check_shots"], f_cut=cfg.f_cut)["loss"]
    assert result["loss_history"][-1] == pytest.approx(actual, rel=1e-9)
    if decision == "rejected":
        torch.testing.assert_close(final, v, atol=0, rtol=0)
    else:
        assert actual < result["iterations"][0]["loss_before"]
    assert result["final_metrics"]["mae"] == pytest.approx((truth - final).abs().mean().item())


def test_uncertainty_halves_initial_step(experiment):
    v, truth, obs, solver, wave, geom = experiment
    cfg = GuidedFWIConfig(n_iters=1, n_check_shots=3, lr=1., kappa=0.)
    def update(sigma):
        result = run_physics_guided_fwi(v, truth, obs, _SparsePredictor(1, sigma), solver, wave, geom, cfg)
        assert result["decisions"] == ["predicted"]
        return result["v_final"] - v
    torch.testing.assert_close(update(0.8), 0.5 * update(0.1))


@pytest.mark.parametrize("count", [1, 2])
def test_guided_first_trial_uses_one_forward(experiment, count):
    v, truth, obs, solver, wave, geom = experiment
    cfg = GuidedFWIConfig(n_iters=2, lr=1., kappa=0., sparse_ratio=2/3, n_check_shots=count, f_cut=7.)
    with patch.object(solver, "forward", wraps=solver.forward) as forward:
        result = run_physics_guided_fwi(v, truth, obs, _SparsePredictor(1), solver, wave, geom, cfg)
    row = result["iterations"][0]
    assert result["decisions"] == ["predicted", "predicted"]
    # Iter 0 evaluates out-of-sample before + trial (2 forwards); iter 1 reuses cached before (1 forward).
    assert forward.call_count == 3
    shots = row["check_shots"]
    assert shots == [1]
    before = solver.adjoint_gradient(v, obs, wave, geom, shots, f_cut=cfg.f_cut)["loss"]
    assert row["loss_before"] == pytest.approx(before, rel=1e-9)


def test_verified_backtracking_near_solution(experiment):
    v, _, _, solver, wave, geom = experiment
    truth = v + 2.
    obs = solver.forward(truth, wave, geom, return_illumination=False)["seismograms"]
    cfg = GuidedFWIConfig(n_iters=3, lr=50., kappa=0., sparse_ratio=1., n_check_shots=geom.n_shots,
                          max_backtracks=8)
    with patch.object(solver, "forward", wraps=solver.forward) as forward:
        result = run_physics_guided_fwi(v, truth, obs, _SparsePredictor(1), solver, wave, geom, cfg)
    assert result["decisions"][:2] == ["backtracked", "predicted"]
    assert result["loss_history"][0] < result["iterations"][0]["loss_before"]
    # Four halvings accept 3.125 m/s. The next iteration keeps it, then grows by 25%.
    trials = [call.args[0] for call in forward.call_args_list]
    for previous, trial, lr in zip((v, trials[4], trials[5]), trials[4:7], (3.125, 3.125, 3.90625)):
        assert torch.quantile((trial - previous).abs(), .99).item() == pytest.approx(lr)


def test_train_stage3_cli(tmp_path, device):
    config = yaml.safe_load((ROOT / "configs/stage3_smoke.yaml").read_text(encoding="utf-8"))
    source = yaml.safe_load((ROOT / "configs/stage1_smoke.yaml").read_text(encoding="utf-8"))
    source["grid"].update(nz=14, nx=19)
    source["physics"].update(nt=48, npml=3, f0=40.)
    source["acquisition"].update(n_shots=3, n_rec=9, margin=2)
    source["dataset"].update(n_iters=1)
    config["dataset"].update(stage1_config=source, manifest="data/manifest.json", max_iters=1)
    config.update(output_dir="results", pretrained_checkpoint="pretrained/best.pt", allow_scratch=False)
    config["guided_fwi"].update(n_iters=1)
    # Exercise the real Stage 2 checkpoint format without relying on workspace outputs.
    stage2 = {key: config[key] for key in ("dataset", "model", "loss", "training")}
    stage2.update(output_dir="pretrained", training=dict(stage2["training"], epochs=1))
    for name, settings in (("stage2", stage2), ("stage3", config)):
        path = tmp_path / (name + ".yaml")
        path.write_text(yaml.safe_dump(settings), encoding="utf-8")
        subprocess.run([sys.executable, str(ROOT / f"scripts/train_{name}.py"), "--config", str(path),
                        "--device", str(device)], cwd=tmp_path, check=True)
    report = json.loads((tmp_path / "results/stage3_metrics.json").read_text(encoding="utf-8"))
    assert report["initialization"] == "stage2_checkpoint"
    assert len(report["history"]) == 2
    for key in ("stage2_baseline", "stage3_physics"):
        metrics = report[key]
        assert metrics["samples"] == len(source["dataset"]["scenarios"])
        assert all(math.isfinite(value) for value in metrics.values())
        assert abs(metrics["cosine_similarity"]) <= 1 and metrics["rel_l2_error"] >= 0
        assert abs(metrics["pearson_r"]) <= 1 and abs(metrics["spearman_r"]) <= 1
        assert metrics["directional_derivative_relative_error"] >= 0
        assert 0 <= metrics["one_step_descent_success_rate"] <= 1
    best = min(report["history"], key=lambda row: row["val"]["loss"])
    checkpoint = torch.load(tmp_path / "results/best_stage3.pt", weights_only=True, map_location="cpu")
    assert checkpoint["epoch"] == best["epoch"] == report["best_epoch"]
    assert report["stage3_physics"] == best["val"]
    MultiModalGradientNet(**checkpoint["config"]["model"]).load_state_dict(checkpoint["model_state_dict"])
    loop = report["closed_loop"]
    manifest = json.loads((tmp_path / "data/manifest.json").read_text(encoding="utf-8"))
    assert loop["model_id"] in manifest["splits"]["test"]
    for key in ("full_adjoint", "sparse_adjoint", "stage2_guided", "stage3_guided"):
        assert loop[key]["final_full_loss"] >= 0
        assert loop[key]["iter_time_mean"] >= loop[key]["grad_time_mean"] > 0
        assert all(math.isfinite(val) for val in loop[key]["final_metrics"].values())
    assert (loop["stage2_guided"]["iterations"][0]["check_shots"]
            == loop["stage3_guided"]["iterations"][0]["check_shots"])
