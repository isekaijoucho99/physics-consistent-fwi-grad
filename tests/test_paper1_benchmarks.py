"""Two-step zero-shot CLI run on wide synthetic sections and ablation checkpoints."""

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def test_cross_structure_benchmarks_and_resume(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs/paper1_benchmarks.yaml").read_text())
    cfg["output_dir"] = "results"
    smoke = ROOT / "outputs/paper1_ablation_smoke"
    for key in ("a3_checkpoint", "a5_checkpoint"):
        cfg[key] = str(smoke / Path(cfg[key]).parent.name / "best.pt")
    cfg["physics"].update(nt=24, f0=80., npml=2)
    cfg["acquisition"]["n_rec"] = 16
    cfg["closed_loop"]["n_iters"] = 2
    z, x = np.ogrid[:64, :160]
    for i, bench in enumerate(cfg["benchmarks"]):
        bench["npy_path"] = bench["name"] + ".npy"
        interface = 4 + 3 * np.sin(x / 15) + i * (x > 80) * 4
        truth = 2100 + 12 * z + 500 * (z > interface) + i * 250
        np.save(tmp_path / bench["npy_path"], truth.astype("float32"))
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / cfg["output_dir"]

    def run():
        subprocess.run([sys.executable, str(ROOT / "scripts/run_paper1_benchmarks.py"),
                        "--config", str(path), "--device", "auto"], cwd=ROOT, check=True, timeout=240)
        return json.loads((out / "paper1_benchmarks_summary.json").read_text())

    cases = run()["cases"]
    assert {(c["benchmark"], c["f_cut"]) for c in cases} == {
        (name, cut) for name in ("marmousi2", "overthrust") for cut in (0., 5.)}
    methods = {"full_adjoint", "sparse_adjoint", "A3_guided", "A5_proposed_guided"}
    markers = []
    for case in cases:
        assert case["sparse_shots"] == [0, 12, 23]
        assert set(case["open_loop"]) == {"A3", "A5"}
        assert case["open_loop"]["A3"] != case["open_loop"]["A5"]
        for row in case["open_loop"].values():
            assert -1 <= row["cosine_similarity"] <= 1 and -1 <= row["spearman_r"] <= 1
            for key in ("rel_l2_error", "directional_derivative_relative_error"):
                assert 0 <= row[key] < math.inf
        loop = case["closed_loop"]
        full = loop["full_adjoint"]
        assert methods <= loop.keys()
        for name in methods:
            row = loop[name]
            history = row["full_loss_history"]
            assert len(history) == 3 and len(row["metric_history"]) == 2
            assert history[0] == pytest.approx(loop["initial_full_loss"], rel=1e-5, abs=0.)
            assert history[-1] == row["final_full_loss"]
            assert all(0 < loss < math.inf for loss in history)
            assert all(math.isfinite(row[key]) for key in ("mae", "rmse", "ssim", "snr_db"))
            assert row["mae"] == row["metric_history"][-1]["mae"]
            assert row["iter_time_total"] == pytest.approx(2 * row["iter_time_mean"])
            assert 0 <= row["descent_success_rate"] <= 1
            if name.endswith("adjoint"):
                assert row["predicted_accept_rate"] is None
            else:
                assert 0 <= row["predicted_accept_rate"] <= 1
            for time_key, reduction in (("grad_time_mean", "grad_time_reduction_pct"),
                                        ("iter_time_mean", "time_reduction_pct")):
                assert row[time_key] > 0
                assert row[reduction] == pytest.approx(100 * (1 - row[time_key] / full[time_key]))

        directory = out / case["benchmark"] / f"f_{case['f_cut']:g}Hz"
        markers.append(directory / "case.json")
        for stem in ("A3_gradient_uncertainty", "A5_gradient_uncertainty",
                     "closed_loop_convergence", "velocity_and_profile"):
            with Image.open(directory / f"{stem}.png") as im:
                im.load()
                assert im.info["dpi"] == pytest.approx((300, 300), abs=.1)
            pdf = (directory / f"{stem}.pdf").read_bytes()
            assert pdf.startswith(b"%PDF-") and pdf.rstrip().endswith(b"%%EOF")

    for name in ("marmousi2", "overthrust"):
        losses = {c["f_cut"]: c["closed_loop"]["initial_full_loss"] for c in cases if c["benchmark"] == name}
        assert losses[5.] < losses[0.]
    table = out / "paper1_benchmarks_table.tex"
    tex = table.read_text()
    assert all(token in tex for token in (r"\usepackage{booktabs}", r"\toprule", r"\bottomrule",
                                          "Marmousi2", "Overthrust", "A3", "A5"))
    if shutil.which("pdflatex"):
        subprocess.run(["pdflatex", "-interaction=batchmode", "-halt-on-error", str(table)],
                       cwd=tmp_path, check=True, timeout=30)

    # Interrupt just before the last case commits, then rebuild the aggregate from all four.
    stamps = {p: p.stat().st_mtime_ns for p in markers[:-1]}
    markers[-1].unlink()
    (out / "paper1_benchmarks_summary.json").unlink()
    resumed = run()["cases"]
    assert resumed[:-1] == cases[:-1] and markers[-1].is_file()
    for rung, scores in cases[-1]["open_loop"].items():
        assert resumed[-1]["open_loop"][rung] == pytest.approx(scores, rel=1e-4, abs=1e-6)
    assert all(p.stat().st_mtime_ns == stamp for p, stamp in stamps.items())
