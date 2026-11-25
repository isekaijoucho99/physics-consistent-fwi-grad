"""Stage 2 numerical, masking and training-pipeline acceptance tests."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from fwi_grad.data.dataset_builder import SCENARIOS
from fwi_grad.data.torch_dataset import FWIGradientDataset, collate_fwi_batch
from fwi_grad.data.velocity_models import generate_synthetic_complex_model
from fwi_grad.eval.metrics import uncertainty_calibration_metrics
from fwi_grad.losses import Stage2GradientLoss
from fwi_grad.models import MultiModalGradientNet, SignPreservingNormalizer, split_wavenumber


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


def test_sign_preserving_quantiles(device):
    # Each image has a different scale. The top quartile includes outliers.
    g = torch.tensor([[-2, -1, 0, 4], [-3, 1, 9, 2]], dtype=torch.float64, device=device)
    g = torch.stack([g, 7 * g])[:, None]
    norm = SignPreservingNormalizer(0.75, alpha=4)
    compressed, scale = norm.normalize(g)
    expected = g.new_tensor([3.25, 7 * 3.25]).view(2, 1, 1, 1)
    torch.testing.assert_close(scale, expected)
    torch.testing.assert_close(compressed, g.sign() * torch.log1p(4 * g.abs() / expected) / math.log(5))
    assert (compressed[g.abs() <= scale].abs() <= 1).all()
    assert (compressed[g.abs() > scale].abs() > 1).all()
    torch.testing.assert_close(norm.normalize(-g)[0], -compressed)
    torch.testing.assert_close(norm.denormalize(compressed, scale), g, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(norm.denormalize(compressed, scale.view(2, 1)), g)
    torch.testing.assert_close(norm.normalize(g, scale.flatten())[0], compressed)
    zeros = torch.zeros_like(g)
    z, s = norm.normalize(zeros)
    assert (s == norm.eps).all() and torch.equal(z, zeros)
    assert torch.equal(norm.denormalize(z, s), zeros)


def test_wavenumber_split(device):
    torch.manual_seed(3)
    field = torch.randn(2, 3, 35, 46, device=device, dtype=torch.float64)
    field[0, 2] = 7.0
    low, high = split_wavenumber(field, sigma_low=2)
    torch.testing.assert_close(low + high, field, rtol=0, atol=1e-5)
    # Replicated boundaries preserve a constant image all the way to its corners.
    torch.testing.assert_close(low[0, 2], field[0, 2], rtol=0, atol=1e-5)
    power = torch.fft.rfft2(field).abs().square()
    low_power = torch.fft.rfft2(low).abs().square()
    kz = torch.fft.fftfreq(35, device=device)[:, None]
    kx = torch.fft.rfftfreq(46, device=device)[None, :]
    high_band = kz.square() + kx.square() > 0.25 ** 2
    assert low_power[..., high_band].sum() < 0.05 * power[..., high_band].sum()


def _network_inputs(device, nz, nx):
    shape = (2, 1, nz, nx)
    initial = 2500 + 400 * torch.randn(shape, device=device)
    inputs = {"v_init": initial, "v_k": initial + 25 * torch.randn_like(initial),
              "illumination": 1e4 * torch.rand(shape, device=device),
              "g_sparse": 1e-5 * torch.randn(shape, device=device),
              "q_k": torch.tensor([[0., 0., 0.5], [0.7, 0.5, 0.5]], device=device),
              "shot_mask": torch.tensor([[True, True, True], [True, False, False]], device=device)}
    for key in ("r_wave", "r_env", "r_phase"):
        inputs[key] = torch.randn(2, 3, 11, 37, device=device)
    return inputs


@pytest.mark.parametrize("shape", [(70, 70), (48, 64), (64, 64), (128, 256), (37, 55)])
def test_network_shapes_and_gradients(device, shape):
    torch.manual_seed(19)
    inputs = _network_inputs(device, *shape)
    model = MultiModalGradientNet(base_channels=4, cond_dim=8).to(device)
    pred = model(**inputs)
    assert set(pred) == {"g_pred", "g_low", "g_high", "sigma"}
    for value in pred.values():
        assert value.shape == (2, 1, *shape) and torch.isfinite(value).all()
    assert (pred["sigma"] >= model.eps_sigma).all()
    q_cond = torch.zeros_like(inputs["q_k"])
    q_cond[:, 0] = inputs["q_k"][:, 0]
    gate = model.gate(q_cond)[:, :, None, None]
    sparse_norm, _ = model.normalizer.normalize(inputs["g_sparse"])
    torch.testing.assert_close(pred["g_pred"], sparse_norm + pred["g_low"] + gate * pred["g_high"])
    loss = Stage2GradientLoss()(pred["g_pred"], pred["sigma"], torch.randn_like(pred["g_pred"]))
    loss["loss"].backward()
    for block in (model.model_encoder, model.data_encoder, model.aux_encoder, model.film,
                  model.attention, model.low_head, model.high_head, model.gate, model.sigma_head):
        grads = [p.grad for p in block.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(torch.count_nonzero(g) > 0 for g in grads)


def test_shot_mask_ignores_padding_and_shot_order(device):
    torch.manual_seed(21)
    inputs = _network_inputs(device, 19, 27)
    # Compare different convolution batch sizes independently of CUDA TF32 rounding.
    inputs = {k: v.double() if v.is_floating_point() else v for k, v in inputs.items()}
    model = MultiModalGradientNet(base_channels=4).to(device=device, dtype=torch.float64).eval()
    with torch.no_grad():
        original = model(**inputs)
        order = [2, 0, 1]
        changed = dict(inputs, shot_mask=inputs["shot_mask"][:, order])
        for key in ("r_wave", "r_env", "r_phase"):
            changed[key] = inputs[key].masked_fill(~inputs["shot_mask"][:, :, None, None], torch.nan)[:, order]
        reordered = model(**changed)
        # Compare against physically removing the two missing shots.
        single = {key: value[1:] for key, value in inputs.items()}
        for key in ("r_wave", "r_env", "r_phase", "shot_mask"):
            single[key] = single[key][:, :1]
        unpadded = model(**single)
        for key in original:
            torch.testing.assert_close(reordered[key], original[key], rtol=1e-9, atol=1e-10)
            torch.testing.assert_close(unpadded[key], original[key][1:], rtol=1e-9, atol=1e-10)


def test_stage2_gradient_loss(device):
    pred = torch.tensor([[[[3., 4.]]], [[[0., 0.]]]], device=device, requires_grad=True)
    ref = torch.tensor([[[[4., 3.]]], [[[1., 2.]]]], device=device)
    error = (pred.detach() - ref).abs()
    sigma = error.clone().requires_grad_(True)
    objective = Stage2GradientLoss(mu=0.7, lambda_unc=0.2)
    exact = objective(pred, sigma, pred.detach())
    assert exact["loss_pred"].item() == 0
    losses = objective(pred, sigma, ref)
    expected_cos = (1 - 24 / 25 + 1) / 2
    assert losses["loss_cos"].item() == pytest.approx(expected_cos)
    assert losses["loss_l1"].item() == pytest.approx(error.mean().item())
    expected_pred = expected_cos + 0.7 * error.mean().item()
    expected_unc = (error / (sigma + objective.eps) + (sigma + objective.eps).log()).mean().item()
    assert losses["loss_pred"].item() == pytest.approx(expected_pred)
    assert losses["loss_unc"].item() == pytest.approx(expected_unc)
    assert losses["loss"].item() == pytest.approx(expected_pred + 0.2 * expected_unc)
    for bad_scale in (0.15, 8):
        assert losses["loss_unc"] < objective(pred, bad_scale * sigma, ref)["loss_unc"]
    for detach in (False, True):
        unc = Stage2GradientLoss(detach_err_for_unc=detach)(pred, sigma, ref)["loss_unc"]
        grad_pred, grad_sigma = torch.autograd.grad(unc, (pred, sigma), allow_unused=True)
        assert (grad_pred is None) == detach
        assert grad_sigma is not None and torch.count_nonzero(grad_sigma)


def test_calibration_uses_pixelwise_errors_and_average_ranks(device):
    pred = torch.tensor([[1., -1., 3., -7.]], dtype=torch.float64, device=device)
    sigma = pred.new_tensor([[1., 4., 4., 2.]])
    def correlation(x, y):
        return np.corrcoef(x, y)[0, 1]

    metrics = uncertainty_calibration_metrics(pred, torch.zeros_like(pred), sigma)
    assert metrics["pearson_r"] == pytest.approx(correlation([1, 1, 3, 7], [1, 4, 4, 2]))
    assert metrics["spearman_r"] == pytest.approx(correlation([1.5, 1.5, 3, 4], [1, 3.5, 3.5, 2]))
    assert metrics["mean_sigma"] == pytest.approx(2.75)
    assert metrics["mean_abs_err"] == pytest.approx(3)
    scale = sigma + 1e-4
    assert metrics["nll"] == pytest.approx((pred.abs() / scale + scale.log()).mean().item())
    for err, std in [(pred, sigma * 0 + 0.3), (pred * 0, sigma), (pred[:, :1], sigma[:, :1])]:
        stats = uncertainty_calibration_metrics(err, torch.zeros_like(err), std)
        assert stats["spearman_r"] == stats["pearson_r"] == 0
        assert all(math.isfinite(value) for value in stats.values())


def test_openfwi_configuration():
    config = yaml.safe_load((ROOT / "configs/stage2_openfwi.yaml").read_text(encoding="utf-8"))
    source = config["dataset"]["stage1_config"]
    data = source["dataset"]
    assert set(data["scenarios"]) == set(SCENARIOS)
    assert data["n_models"] == 20 and data["n_iters"] == config["dataset"]["max_iters"] == 4
    assert data["n_models"] * data["n_iters"] * len(data["scenarios"]) == 400
    assert source["grid"]["nz"] == source["grid"]["nx"] == 64
    assert source["physics"]["nt"] == 350 and source["acquisition"]["n_shots"] == 8
    model_path = Path(source["model"]["npy_path"])
    assert model_path == Path("../data/openfwi/Vel Family/CurveVel_A/model/model1.npy")
    assert config["training"]["epochs"] == 40


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory, device):
    """Exercise the public CLI, including auto-build from a miniature OpenFWI stack."""
    tmp = tmp_path_factory.mktemp("stage2")
    path = tmp / "train.yaml"
    config = yaml.safe_load((ROOT / "configs/stage2_smoke.yaml").read_text(encoding="utf-8"))
    stage1 = yaml.safe_load((ROOT / "configs/stage1_smoke.yaml").read_text(encoding="utf-8"))
    models = np.stack([generate_synthetic_complex_model(70, 70, seed=i) for i in range(3)])
    np.save(tmp / "model stack.npy", models[:, None])
    stage1["model"].update(npy_path="model stack.npy", sigma=2)
    stage1["grid"].update(nz=23, nx=31)
    stage1["physics"].update(nt=64, f0=36., npml=4)
    stage1["acquisition"].update(n_shots=3, n_rec=12, margin=2)
    config["dataset"].update(manifest="data/manifest.json", stage1_config=stage1)
    config["training"].update(epochs=2)
    config["output_dir"] = "checkpoints"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "scripts/train_stage2.py"), "--config", str(path),
                    "--device", str(device)], cwd=tmp, check=True)
    return tmp


def test_stage1_manifest_and_collation(trained_run):
    path = trained_run / "data/manifest.json"
    dataset = FWIGradientDataset(path, split="train", max_iters=2)
    assert len(dataset) == 10
    sparse = FWIGradientDataset(path, scenarios=["sparse"], max_iters=2)
    normal = FWIGradientDataset(path, scenarios=["low_freq_cut"], max_iters=2)
    assert len(sparse) == len(normal) == 2
    assert not len(FWIGradientDataset(path, scenarios=[]))
    a, b = normal[1], sparse[0]
    assert a["scenario"] == "low_freq_cut" and b["scenario"] == "sparse"
    images = {"v_k", "v_init", "v_true", "g_sparse", "illumination", "g_ref", "g_ref_norm"}
    gathers = {"r_wave", "r_env", "r_phase"}
    assert a.keys() == images | gathers | {"q_k", "g_ref_scale", "scenario", "f_cut"}
    assert all(a[key].shape == (1, 23, 31) and a[key].device.type == "cpu" for key in images)
    assert a["g_ref_scale"].shape == (1,)
    torch.testing.assert_close(a["q_k"], torch.tensor([0.7, 1.0 / 25.0, 2.0 / 3.0]))
    torch.testing.assert_close(b["q_k"], torch.tensor([0.0, 0.0, 2.0 / 3.0]))
    b_short = dict(b, **{key: b[key][:1] for key in gathers})
    batch = collate_fwi_batch([a, b_short])
    assert batch["scenario"] == ["low_freq_cut", "sparse"]
    assert batch["shot_mask"].dtype == torch.bool
    assert batch["shot_mask"].tolist() == [[True, True], [True, False]]
    for key in gathers:
        assert batch[key].shape == (2, 2, 12, 64)
        torch.testing.assert_close(batch[key][0], a[key])
        torch.testing.assert_close(batch[key][1, :1], b_short[key])
        assert torch.count_nonzero(batch[key][1, 1]) == 0
    torch.testing.assert_close(SignPreservingNormalizer().denormalize(batch["g_ref_norm"], batch["g_ref_scale"]),
                               batch["g_ref"], rtol=2e-5, atol=1e-12)


def test_two_epoch_training_and_checkpoints(trained_run):
    out = trained_run / "checkpoints"
    report = json.loads((out / "stage2_metrics.json").read_text(encoding="utf-8"))
    first, last = report["history"]
    assert last["train_loss"] < first["train_loss"]
    assert 0 < last["lr"] < first["lr"]
    for epoch in (first, last):
        val = epoch["val"]
        assert val["overall"]["samples"] == 10 and set(val["per_scenario"]) == set(SCENARIOS)
        assert all(group["samples"] == 2 for group in val["per_scenario"].values())
        for metrics in [val["overall"], *val["per_scenario"].values()]:
            assert abs(metrics["cosine_similarity"]) <= 1 and metrics["rel_l2_error"] >= 0
            assert abs(metrics["pearson_r"]) <= 1 and abs(metrics["spearman_r"]) <= 1
            assert metrics["mean_sigma"] > 0 and metrics["inference_seconds_per_sample"] > 0
            assert all(math.isfinite(value) for value in metrics.values())
    best_epoch = min(report["history"], key=lambda e: e["val"]["overall"]["loss"])
    assert report["best_epoch"] == best_epoch["epoch"]
    for filename, epoch in [("best.pt", best_epoch), ("last.pt", last)]:
        checkpoint = torch.load(out / filename, weights_only=True, map_location="cpu")
        assert checkpoint["epoch"] == epoch["epoch"]
        assert checkpoint["val_loss"] == epoch["val"]["overall"]["loss"]
        model = MultiModalGradientNet(**checkpoint["config"]["model"])
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
