"""Stage 1 numerical and file-format acceptance tests."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from fwi_grad.data.dataset_builder import build_trajectory_samples, make_experiment
from fwi_grad.data.velocity_models import (
    add_gaussian_noise,
    build_initial_model,
    generate_synthetic_complex_model,
    load_openfwi_velocity,
)
from fwi_grad.eval.metrics import gradient_metrics, velocity_metrics
from fwi_grad.inversion.baseline_fwi import run_traditional_fwi
from fwi_grad.physics.acoustic2d import Acoustic2DSolver
from fwi_grad.physics.geometry import (
    AcquisitionGeometry,
    create_surface_geometry,
    geometry_encoding,
    select_sparse_shots,
)
from fwi_grad.physics.residuals import compute_hilbert_analytic, compute_multimodal_residuals
from fwi_grad.physics.wavelet import bandpass_filter, highpass_filter, ricker_wavelet


@pytest.fixture(scope="module", params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request):
    return torch.device(request.param)


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_cfl():
    # Exact 4th-order bound for a rectangular grid.
    limit = math.sqrt(3) / (2 * 4000 * math.hypot(1 / 8, 1 / 12))
    Acoustic2DSolver.check_cfl(4000, 8, 12, 0.99 * limit)
    with pytest.raises(ValueError, match="(?i)CFL"):
        Acoustic2DSolver.check_cfl(4000, 8, 12, 1.01 * limit)


def test_geometry():
    geom = create_surface_geometry(16, 31, 8, 10, 5, 1, margin=3)
    assert geom.src_x.tolist() == [3, 9, 15, 21, 27]
    assert geom.rec_x.tolist() == [15]
    encoded = geometry_encoding(geom)
    assert encoded.shape == (2, 16, 31)
    assert (encoded[0, geom.src_z, geom.src_x] > 0).all()
    assert (encoded[1, geom.rec_z, geom.rec_x] > 0).all()
    assert encoded[0].count_nonzero() == 5
    assert encoded[1].count_nonzero() == 1
    assert len(select_sparse_shots(8, 1)) == 1
    assert select_sparse_shots(8, 1.0) == list(range(8))
    assert select_sparse_shots(7, 3) == [0, 3, 6]
    random_shots = select_sparse_shots(8, 0.5, mode="random", seed=10)
    assert len(set(random_shots)) == 4
    assert random_shots == select_sparse_shots(8, 0.5, mode="random", seed=10)


def test_wavefield_and_illumination(device):
    # Sampling every physical grid point makes sum(u**2) independently observable.
    nz, nx = 13, 17
    zz, xx = torch.meshgrid(torch.arange(nz), torch.arange(nx), indexing="ij")
    geom = AcquisitionGeometry(nz, nx, 10, 10, torch.tensor([6]), torch.tensor([8]),
                               zz.flatten(), xx.flatten())
    velocity = torch.full((nz, nx), 2200., device=device)
    solver = Acoustic2DSolver(dt=0.001, npml=5, shot_batch_size=1)
    wavelet = ricker_wavelet(73, 0.001, 30, t0=0.024, device=device)
    predicted = solver.forward(velocity, wavelet, geom)
    seismograms = predicted["seismograms"]
    illumination = predicted["illumination"]
    assert seismograms.shape == (1, nz * nx, len(wavelet))
    field = seismograms.reshape(nz, nx, -1)
    assert torch.isfinite(field).all() and field.abs().max() > 0
    assert torch.isfinite(illumination).all() and (illumination >= 0).all()
    torch.testing.assert_close(illumination, field.square().sum(-1), rtol=2e-5, atol=1e-6)
    for axis in [0, 1]:
        assert (field - field.flip([axis])).abs().max() < 2e-5 * field.abs().max()
        assert (illumination - illumination.flip([axis])).abs().max() < 2e-5 * illumination.max()


@pytest.mark.parametrize("dtype,tolerance,epsilon", [
    (torch.float32, 5e-3, 1.0),
    (torch.float64, 1e-4, 0.05),
])
def test_discrete_gradient(device, dtype, tolerance, epsilon):
    geom = create_surface_geometry(15, 21, 9., 11., 4, 10, margin=2)
    solver = Acoustic2DSolver(dt=0.001, npml=4, shot_batch_size=2)
    wave = ricker_wavelet(57, 0.001, 30, t0=0.022, device=device, dtype=dtype)
    velocity = torch.full((15, 21), 2100., device=device, dtype=dtype)
    truth = velocity + 100
    truth[3:8, 6:15] += 150
    shots = [3, 1, 0]  # Reordered subset and a short final batch.
    obs = solver.forward(truth, wave, geom)["seismograms"]
    result = solver.adjoint_gradient(velocity, obs, wave, geom, shots, f_cut=12., checkpoint_step=8)
    model = velocity.clone().requires_grad_(True)
    forward = solver.forward(model, wave, geom, shots)
    residual = highpass_filter(forward["seismograms"], 0.001, 12.) - highpass_filter(obs[shots], 0.001, 12.)
    objective = 0.5 * residual.square().mean()
    (expected,) = torch.autograd.grad(objective, model)
    assert result["loss"] == pytest.approx(objective.item(), rel=2e-6)
    for name, value in [("grad", expected), ("d_syn", forward["seismograms"]),
                        ("illumination", forward["illumination"])]:
        torch.testing.assert_close(result[name], value, rtol=2e-5, atol=1e-8)
    assert not result["illumination"].requires_grad
    assert result["elapsed_sec"] > 0

    rng = torch.Generator().manual_seed(819)
    projections, differences = [], []
    for _ in range(3):
        # Unit RMS directions, eps expressed in m/s. Keep float32 perturbations
        # representable around 2100 m/s when differencing the two losses.
        direction = torch.randn(velocity.shape, generator=rng, dtype=dtype).to(device)
        projections.append((result["grad"] * direction).sum().item())
        differences.append(solver.directional_derivative_fd(
            velocity, direction, obs[shots], wave, geom, eps=epsilon, shot_indices=shots, f_cut=12.
        ))
        if dtype == torch.float32:
            # Also difference two native float32 wave propagations. The solver's
            # public FD diagnostic promotes precision for its much smaller default eps.
            def native_objective(offset):
                trial = velocity + offset * direction
                pred = solver.forward(trial, wave, geom, shots, False)["seismograms"]
                delta = highpass_filter(pred, 0.001, 12.) - highpass_filter(obs[shots], 0.001, 12.)
                return 0.5 * delta.square().mean().item()

            native_fd = (native_objective(epsilon) - native_objective(-epsilon)) / (2 * epsilon)
            assert native_fd == pytest.approx(projections[-1], rel=tolerance, abs=0)
    ad, fd = np.array(projections), np.array(differences)
    assert np.all(np.abs(ad - fd) / np.maximum(np.abs(ad), np.abs(fd)) < tolerance), (ad, fd)
    assert np.dot(ad, fd) / (np.linalg.norm(ad) * np.linalg.norm(fd)) > 0.999


@pytest.mark.parametrize("nt", [255, 256])
def test_filters_and_residuals(device, nt):
    # Choose whole periods so attenuation is measured independently of FFT leakage.
    t = torch.arange(nt, dtype=torch.float64, device=device) / nt
    low, mid, high = [torch.cos(2 * math.pi * f * t) for f in [2, 18, 56]]
    signal = torch.stack([low, mid, high])
    hp = highpass_filter(signal, 1 / nt, 8.0)
    bp = bandpass_filter(signal, 1 / nt, 8.0, 28.0)
    assert hp[0].norm() < 0.02 * low.norm()
    assert (hp[1] - mid).norm() < 0.02 * mid.norm()
    assert (bp[1] - mid).norm() < 0.04 * mid.norm()
    assert bp[2].norm() < 0.10 * high.norm()
    torch.testing.assert_close(compute_hilbert_analytic(low), torch.polar(torch.ones_like(t), 4 * math.pi * t))
    traces = (mid * (1 + 0.3 * low)).reshape(1, 1, -1)
    for key, value in compute_multimodal_residuals(traces, traces).items():
        assert value.abs().max() < 1e-12, key
    shifted = compute_multimodal_residuals(traces.roll(2, -1), traces)
    assert set(shifted) == {"r_wave", "r_env", "r_phase"}
    assert all(torch.isfinite(v).all() and v.abs().max() > 0.01 for v in shifted.values())
    assert compute_multimodal_residuals(traces, torch.zeros_like(traces))["r_phase"].abs().max() == 0


def test_velocity_models_and_metrics(tmp_path):
    truth = generate_synthetic_complex_model(32, 48, seed=11)
    assert truth.shape == (32, 48) and truth.dtype == np.float32
    assert 1500 <= truth.min() < truth.max() <= 4500
    np.testing.assert_array_equal(truth, generate_synthetic_complex_model(32, 48, seed=11))
    assert not np.array_equal(truth, generate_synthetic_complex_model(32, 48, seed=12))
    two = np.stack([truth[::-1], truth])
    for i, models in enumerate([two, two[:, None]]):
        filename = tmp_path / f"model{i}.npy"
        np.save(filename, models)
        np.testing.assert_allclose(load_openfwi_velocity(filename, index=1), truth)
        assert load_openfwi_velocity(filename, 0, (15, 23)).shape == (15, 23)
    smooth = build_initial_model(truth, sigma=3)
    assert smooth.shape == truth.shape and smooth.dtype == np.float32
    assert np.abs(np.diff(smooth, axis=0)).mean() < np.abs(np.diff(truth, axis=0)).mean()
    linear = build_initial_model(truth, method="linear", v_min=1600., v_max=3200.)
    np.testing.assert_allclose(linear[:, 0], np.linspace(1600, 3200, 32), rtol=1e-6)
    obs = torch.tensor(truth)
    noisy = add_gaussian_noise(obs, 16, seed=9)
    torch.testing.assert_close(noisy, add_gaussian_noise(obs, 16, seed=9))
    assert 20 * torch.log10(obs.norm() / (noisy - obs).norm()).item() == pytest.approx(16, abs=0.6)
    accuracy = velocity_metrics(obs, obs)
    assert accuracy["mae"] == accuracy["rmse"] == accuracy["re"] == 0
    assert accuracy["ssim"] == pytest.approx(1.0, abs=1e-5)
    assert velocity_metrics(noisy, obs)["ssim"] < accuracy["ssim"]
    grad = gradient_metrics(-obs, obs)
    assert grad["cosine_similarity"] == pytest.approx(-1., abs=1e-5)
    assert grad["rel_l2_error"] == pytest.approx(2., abs=1e-5)


def test_baseline_on_synthetic_model(device):
    truth = torch.from_numpy(generate_synthetic_complex_model(48, 64, seed=13)).to(device)
    # Smooth near-surface bias, directly illuminated by the survey.
    depth = torch.arange(48, device=device)[:, None]
    initial = truth + 100 * torch.exp(-depth / 10)
    geom = create_surface_geometry(48, 64, 10., 10., 4, 24)
    wave = ricker_wavelet(160, 0.001, 20., device=device)
    solver = Acoustic2DSolver(0.001, npml=8, shot_batch_size=2)
    observed = solver.forward(truth, wave, geom)["seismograms"]
    initial_error = (initial - truth).abs().mean().item()
    result = run_traditional_fwi(initial, truth, observed, solver, wave, geom, n_iters=3, lr=3.)
    final = result["v_final"]
    loss = 0.5 * (solver.forward(final, wave, geom)["seismograms"] - observed).square().mean()
    assert loss.item() < result["loss_history"][-1] < result["loss_history"][0]
    assert len(result["metric_history"]) == len(result["loss_history"]) == 3
    assert (final - truth).abs().mean().item() < initial_error
    assert result["metric_history"][-1]["mae"] == pytest.approx((final - truth).abs().mean().item(), rel=1e-5)
    assert result["iter_time_mean"] >= result["grad_time_mean"] > 0


def test_trajectory_files(tmp_path, device):
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/stage1_smoke.yaml").read_text(encoding="utf-8"))
    config["grid"].update(nz=14, nx=19)
    config["physics"].update(nt=49, f0=42., npml=3)
    config["acquisition"].update(n_shots=3, n_rec=8, margin=2)
    config["dataset"].update(n_models=3, n_iters=2, sparse_ratio=0.34)
    build_trajectory_samples(config, tmp_path, device=str(device))
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    splits = manifest["splits"]
    assert set(splits) == {"train", "val", "test"}
    assert sorted(map(len, splits.values())) == [1, 1, 1]
    assert len({model_id for ids in splits.values() for model_id in ids}) == 3
    samples = manifest["samples"]
    assert len(samples) == len({s["path"] for s in samples}) == 30
    expected = {"v_k", "v_init", "v_true", "r_wave", "r_env", "r_phase", "illumination", "g_sparse", "g_ref", "meta"}
    seen, truths, split_for_truth = set(), {}, {}
    for row in samples:
        model_id, split, scenario, k = [row[key] for key in ("model_id", "split", "scenario", "iter_k")]
        assert model_id in splits[split]
        seen.add((model_id, scenario, k))
        sample = torch.load(tmp_path / row["path"], map_location="cpu", weights_only=True)
        assert expected <= sample.keys()
        meta = sample["meta"]
        assert {"iter_k", "f_cut", "snr_db", "sparse_ratio", "loss_k", "t_full_grad", "t_sparse_grad"} <= meta.keys()
        assert meta["iter_k"] == k and meta["t_full_grad"] > 0 and meta["t_sparse_grad"] > 0
        for key in expected - {"meta"}:
            assert torch.isfinite(sample[key]).all(), key
            if key.startswith("r_"):
                assert sample[key].shape == (1, 8, 49)
            else:
                assert sample[key].shape == (14, 19)
        truth = sample["v_true"].numpy().tobytes()
        assert split_for_truth.setdefault(truth, split) == split
        assert truths.setdefault(model_id, truth) == truth
        if k == 0:
            torch.testing.assert_close(sample["v_k"], sample["v_init"])
        else:
            assert not torch.equal(sample["v_k"], sample["v_init"])
        assert meta["loss_k"] >= 0.0
    assert len(split_for_truth) == 3
    assert seen == {(mid, scenario, k) for mid in truths for scenario in config["dataset"]["scenarios"] for k in (0, 1)}

    # Reconstruct full and sparse labels independently from a stored trajectory state.
    row = next(row for row in samples if row["scenario"] == "low_freq_cut" and row["iter_k"] == 1)
    sample = torch.load(tmp_path / row["path"], map_location=device, weights_only=True)
    _, _, wavelet, geom, solver = make_experiment(config, 0, device)
    obs = solver.forward(sample["v_true"], wavelet, geom)["seismograms"]
    cutoff = config["dataset"]["f_cut"]
    sparse_shots = select_sparse_shots(3, 0.34)
    full = solver.adjoint_gradient(sample["v_k"], obs, wavelet, geom, f_cut=cutoff)
    sparse = solver.adjoint_gradient(sample["v_k"], obs, wavelet, geom, sparse_shots, f_cut=cutoff)
    torch.testing.assert_close(sample["g_ref"], full["grad"])
    torch.testing.assert_close(sample["g_sparse"], sparse["grad"])
    torch.testing.assert_close(sample["illumination"], sparse["illumination"])
    residual = highpass_filter(sparse["d_syn"], solver.dt, cutoff) - highpass_filter(obs[sparse_shots], solver.dt, cutoff)
    torch.testing.assert_close(sample["r_wave"], residual)


def test_fourth_order_stencil_and_source_scaling(device):
    nz, nx = 11, 13
    z, x = 5, 6
    zz, xx = torch.meshgrid(torch.arange(nz), torch.arange(nx), indexing="ij")
    geom = AcquisitionGeometry(nz, nx, 8., 13., torch.tensor([z]), torch.tensor([x]),
                               zz.flatten(), xx.flatten())
    dt, speed = 0.001, 2000.
    solver = Acoustic2DSolver(dt, npml=4)
    v = torch.full((nz, nx), speed, dtype=torch.float64, device=device)
    impulse = v.new_tensor([1., 0.])
    field = solver.forward(v, impulse, geom)["seismograms"].reshape(nz, nx, 2)
    scale = (speed * dt) ** 2
    expected = torch.zeros_like(v)
    expected[z, x] = scale
    torch.testing.assert_close(field[..., 0], expected)
    stencil = v.new_tensor([-1 / 12, 4 / 3, -2.5, 4 / 3, -1 / 12])
    expected *= 2
    expected[z - 2:z + 3, x] += scale ** 2 / geom.dz ** 2 * stencil
    expected[z, x - 2:x + 3] += scale ** 2 / geom.dx ** 2 * stencil
    torch.testing.assert_close(field[..., 1], expected)


def test_free_surface_pressure(device):
    v = torch.full((17, 21), 2200., device=device)
    wave = ricker_wavelet(53, 0.001, 40., device=device)
    geom = create_surface_geometry(17, 21, 10., 10., 1, 7, src_depth_idx=3, rec_depth_idx=0)
    wall = Acoustic2DSolver(0.001, npml=4, free_surface=True)
    sponge = Acoustic2DSolver(0.001, npml=4)
    assert wall.forward(v, wave, geom)["seismograms"].abs().max() == 0
    assert sponge.forward(v, wave, geom)["seismograms"].abs().max() > 0


def test_gradient_image_against_coordinate_differences(device):
    # Finite-difference every velocity cell to check the entire gradient image.
    geom = create_surface_geometry(5, 7, 8., 10., 2, 5, margin=1)
    solver = Acoustic2DSolver(0.001, npml=2, shot_batch_size=2)
    wave = ricker_wavelet(31, 0.001, 45., t0=0.012, dtype=torch.float64, device=device)
    v = torch.full((5, 7), 1900., dtype=wave.dtype, device=device)
    obs = solver.forward(v + 120, wave, geom)["seismograms"]
    grad = solver.adjoint_gradient(v, obs, wave, geom, checkpoint_step=7)["grad"]
    finite_difference = torch.empty_like(v)
    for z in range(geom.nz):
        for x in range(geom.nx):
            unit = torch.zeros_like(v)
            unit[z, x] = 1.
            finite_difference[z, x] = solver.directional_derivative_fd(v, unit, obs, wave, geom)
    score = gradient_metrics(grad, finite_difference)
    assert score["cosine_similarity"] > 0.999
    assert score["rel_l2_error"] < 1e-4
