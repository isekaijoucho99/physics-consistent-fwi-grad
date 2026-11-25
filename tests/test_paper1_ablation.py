"""Paper 1 split, branch ablation, and resumable CLI acceptance tests."""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from fwi_grad.data.dataset_builder import build_trajectory_samples
from fwi_grad.models import MultiModalGradientNet

ROOT = Path(__file__).resolve().parents[1]
RUNGS = ["A0_sparse_only", "A1_no_data", "A2_multimodal_no_unc", "A3_stage2_full",
         "A4_stage3_dir_str", "A5_proposed_full"]


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_stratified_openfwi_trajectories(tmp_path):
    families = ["CurveVel_A", "CurveVel_B", "FlatVel_A", "FlatVel_B"]
    sources = []
    for k, family in enumerate(families):
        path = tmp_path / f"{family}.npy"
        layers = np.arange(8, dtype="float32")[None, :, None] * 50
        stack = 1800 + 100 * k + np.arange(11)[:, None, None] + layers + np.zeros((1, 1, 8))
        np.save(path, stack.astype("float32"))
        sources.append(dict(name=family, npy_path=str(path), n_models=10, start_index=1))
    config = dict(grid=dict(nz=8, nx=8), physics=dict(nt=8, npml=1, f0=100.),
                  acquisition=dict(n_shots=2, n_rec=4, margin=1), model=dict(sources=sources, seed=42),
                  dataset=dict(n_iters=1, scenarios=["normal"], split_fractions=[.7, .15, .15]))
    index = build_trajectory_samples(config, tmp_path, "cpu")
    splits = index["splits"]
    assert [len(splits[s]) for s in ("train", "val", "test")] == [28, 6, 6]
    assert len({m for ids in splits.values() for m in ids}) == 40
    assert len(index["samples"]) == 40
    for split in splits:
        rows = [r for r in index["samples"] if r["split"] == split]
        assert {r["family"] for r in rows} == set(families)
        assert {r["model_id"] for r in rows} == set(splits[split])
    origins = set()
    for row in index["samples"]:
        sample = torch.load(tmp_path / row["path"], weights_only=True)
        family = row["family"]
        assert sample["meta"]["family"] == family
        origins.add((family, sample["v_true"][0, 0].item() - 1800 - 100 * families.index(family)))
    assert origins == {(family, i) for family in families for i in range(1, 11)}


def test_multifamily_requires_nonempty_splits(tmp_path):
    source = dict(name="Flat", npy_path="unused.npy", n_models=3)
    cfg = dict(model=dict(sources=[source]), dataset=dict(split_fractions=[.8, .2, 0.]))
    with pytest.raises(ValueError, match="positive split_fractions"):
        build_trajectory_samples(cfg, tmp_path, "cpu")


@pytest.mark.parametrize("no_model,no_data", [(True, True), (True, False), (False, True), (False, False)])
def test_branch_flags_preserve_active_gradients(no_model, no_data):
    torch.manual_seed(31)
    net = MultiModalGradientNet(4, 8, disable_model_branch=no_model, disable_data_branch=no_data)
    net.load_state_dict(MultiModalGradientNet(4, 8).state_dict(), strict=True)
    field = torch.randn(1, 1, 8, 8)
    data = torch.randn(1, 2, 4, 12)
    args = dict(v_k=2700 + field, v_init=2700 - field, r_wave=data, r_env=data.abs(), r_phase=data,
                shot_mask=torch.ones(1, 2, dtype=torch.bool), illumination=field.abs(),
                g_sparse=field, q_k=torch.tensor([[0., 0., .5]]))
    pred = net(**args)["g_pred"]
    reference = MultiModalGradientNet(4, 8)
    reference.load_state_dict(net.state_dict())
    with torch.no_grad():
        for encoder, disabled in [(reference.model_encoder, no_model), (reference.data_encoder, no_data)]:
            if disabled:
                for p in encoder.parameters():
                    p.zero_()
        torch.testing.assert_close(pred, reference(**args)["g_pred"], rtol=0, atol=0)
    pred.square().mean().backward()
    for encoder, disabled in [(net.model_encoder, no_model), (net.data_encoder, no_data),
                              (net.aux_encoder, False)]:
        has_gradient = any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())
        assert has_gradient == (not disabled)
    if no_model:
        args.update(v_k=4000 - field, v_init=1800 + field)
    if no_data:
        args.update(r_wave=data.flip(-1), r_env=data.flip(-2), r_phase=-data)
    torch.testing.assert_close(net(**args)["g_pred"], pred, rtol=0, atol=0)


def test_held_out_physics_uses_deployable_inputs(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from scripts import run_paper1_ablation as paper

    cfg = yaml.safe_load((ROOT / "configs/paper1_ablation_smoke.yaml").read_text())
    experiment = cfg["experiment"]
    truth, initial, _, geom, _ = paper.make_experiment(experiment, 0, "cpu")
    survey = paper.stage3._Survey(experiment, torch.device("cpu"))
    obs = survey.observations("model_0000", "normal", truth)
    grad = survey.solver.adjoint_gradient(initial, obs, survey.wavelet, geom, shot_indices=[0])["grad"]
    model = MultiModalGradientNet(**cfg["network"]).eval()
    g = grad[None, None]
    inputs = {"v_k": initial[None, None], "g_sparse": g}
    batch = {"g_ref_scale": torch.tensor([[1.0]]), "v_true": truth[None, None],
             "model_id": ["model_0000"], "scenario": ["normal"], "q_k": torch.zeros(1, 3)}
    output = {"g_pred": model.normalizer.normalize(g)[0], "sigma": torch.zeros_like(g)}

    def score():
        return paper._physics_sample(model, batch, inputs, output, 0, survey, cfg["physics_loss"],
                                     torch.Generator().manual_seed(7), held_out=True, use_unc=False)

    error, step = score()
    batch["g_ref_scale"] *= 100
    output["sigma"] += 1.e3
    other_error, other_step = score()
    torch.testing.assert_close(error, other_error)
    torch.testing.assert_close(step["loss_after"], other_step["loss_after"])
    full = list(range(geom.n_shots))
    expected = paper.subset_misfit(initial, obs, survey.solver, survey.wavelet, geom, full).item()
    assert step["loss_before"].item() == pytest.approx(expected)
    assert torch.isfinite(error) and error.item() >= 0.0


def test_paper1_smoke_and_resume(tmp_path):
    relative = Path("configs/paper1_ablation_smoke.yaml")
    path = tmp_path / relative
    path.parent.mkdir()
    path.write_bytes((ROOT / relative).read_bytes())
    out = path.parent / yaml.safe_load(path.read_text())["output_dir"]
    command = [sys.executable, str(ROOT / "scripts/run_paper1_ablation.py"), "--config", str(relative)]

    def run():
        return subprocess.run(command, cwd=tmp_path, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, check=True, timeout=90).stdout

    run()
    report = json.loads((out / "paper1_ablation_summary.json").read_text())
    manifest = out / "dataset/manifest.json"
    index = json.loads(manifest.read_text())
    test = [r for r in index["samples"] if r["split"] == "test"]
    assert len(index["splits"]["test"]) == 2
    metrics = {"cosine_similarity", "rel_l2_error", "directional_derivative_relative_error",
               "one_step_descent_success_rate", "spearman_r", "inference_ms"}
    times = {manifest: manifest.stat().st_mtime_ns}
    for rung in RUNGS:
        path = out / rung / "eval.json"
        evaluation = json.loads(path.read_text())
        assert report["ablation"][rung] == evaluation
        assert evaluation["split"] == "test"
        assert len(evaluation["history"]) == 2
        assert evaluation["lambda_unc"] == (0. if rung in RUNGS[:3] else .1)
        assert evaluation["model"]["disable_model_branch"] == (rung == RUNGS[0])
        assert evaluation["model"]["disable_data_branch"] == (rung in RUNGS[:2])
        if rung in RUNGS[4:]:
            assert evaluation["initialization"] == "A3_stage2_full/best.pt"
            assert evaluation["physics_loss"]["lambda_data"] == (0. if rung == RUNGS[4] else .3)
        assert evaluation["overall"]["samples"] == len(test)
        assert evaluation["overall"]["inference_ms"] > 0
        for key, field in [("per_family", "family"), ("per_scenario", "scenario")]:
            assert evaluation[key].keys() == {r[field] for r in test}
            for label, group in evaluation[key].items():
                assert group["samples"] == sum(r[field] == label for r in test)
                assert all(math.isfinite(group[m]) for m in metrics)
        for file in (path, path.with_name("best.pt")):
            times[file] = file.stat().st_mtime_ns

    loop = report["closed_loop"]
    methods = {"full_adjoint", "sparse_adjoint", "A2_guided", "A3_guided", "A5_proposed_guided"}
    for cohort, n in [("test", 2), ("ood", 1)]:
        cases = [c for c in loop["cases"] if c["cohort"] == cohort]
        ids = {c["model_id"] for c in cases}
        assert len(ids) == n
        if cohort == "test":
            assert ids == set(index["splits"]["test"])
        for cutoff in (0., 5.):
            group = [c for c in cases if c["f_cut"] == cutoff]
            assert len(group) == n
            for c in group:
                assert set(c["methods"]) == methods
            for name in methods:
                values = [c["methods"][name] for c in group]
                reduced = [100 * (1 - v["iter_time_mean"] / c["methods"]["full_adjoint"]["iter_time_mean"])
                           for c, v in zip(group, values)]
                stats = loop["aggregate"][cohort][str(cutoff)][name]
                for key in ("final_full_loss", "mae", "rmse", "re", "ssim", "snr_db", "descent_success_rate",
                            "grad_time_mean", "iter_time_mean", "predicted_accept_rate", "time_reduction_pct"):
                    if key == "predicted_accept_rate" and name.endswith("adjoint"):
                        assert stats[key] is None
                        continue
                    numbers = reduced if key == "time_reduction_pct" else [v[key] for v in values]
                    assert stats[key]["mean"] == pytest.approx(np.mean(numbers))
                    assert stats[key]["std"] == pytest.approx(np.std(numbers))

    table = (out / "paper1_ablation_tables.tex").read_text()
    assert table.count(r"\toprule") >= 2 and r"\bottomrule" in table and r"\pm" in table
    for name in ("ablation_ladder", "closed_loop_summary", "best_gradient_uncertainty"):
        with Image.open(out / f"{name}.png") as im:
            im.load()
            assert im.info["dpi"] == pytest.approx((300, 300), abs=.1)
        pdf = (out / f"{name}.pdf").read_bytes()
        assert pdf.startswith(b"%PDF-") and pdf.rstrip().endswith(b"%%EOF")
    progress = run()
    assert all(file.stat().st_mtime_ns == stamp for file, stamp in times.items())
    assert all(any(rung in line and "skip" in line.lower() for line in progress.splitlines())
               for rung in RUNGS)
