#!/usr/bin/env python3
"""Export real Marmousi2 A5 tensors for the visual Figure 1 framework diagram."""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from fwi_grad.data.dataset_builder import make_experiment
from fwi_grad.inversion.guided_fwi import GuidedFWIConfig, _predict, run_physics_guided_fwi
from fwi_grad.physics.geometry import select_sparse_shots
from fwi_grad.physics.residuals import compute_multimodal_residuals
from fwi_grad.physics.wavelet import highpass_filter
from scripts.run_paper1_ablation import _load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "paper1_benchmarks.yaml"))
    parser.add_argument("--output", default=str(ROOT / "outputs" / "fig1_marmousi2_assets.npz"))
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bench = config["benchmarks"][0]  # marmousi2
    survey = {k: config[k] for k in ("grid", "acquisition", "physics")}
    model_cfg = dict(bench, npy_path=cfg_path.parent / bench["npy_path"])
    truth, initial, wavelet, geom, solver = make_experiment(dict(survey, model=model_cfg), 0, device)
    initial = initial.clamp(*bench["v_bounds"])

    with torch.no_grad():
        obs = solver.forward(truth, wavelet, geom, return_illumination=False)["seismograms"]

    cfg = GuidedFWIConfig(**config["closed_loop"], v_bounds=bench["v_bounds"], f_cut=5.0)
    shots = select_sparse_shots(geom.n_shots, float(cfg.sparse_ratio))
    full = solver.adjoint_gradient(initial, obs, wavelet, geom, f_cut=5.0)
    sparse = solver.adjoint_gradient(initial, obs, wavelet, geom, shots, f_cut=5.0)

    net = _load_model(cfg_path.parent / config["a5_checkpoint"], device)
    with torch.no_grad():
        pred, sigma = _predict(net, net.normalizer, initial, initial, sparse, obs, solver, shots, cfg, 0)
        sp_norm, scale = net.normalizer.normalize(sparse["grad"])
        ref_norm, _ = net.normalizer.normalize(full["grad"], scale)
        pred_norm, _ = net.normalizer.normalize(pred, scale)
        syn_hp = highpass_filter(sparse["d_syn"], solver.dt, 5.0)
        obs_hp = highpass_filter(obs[list(shots)], solver.dt, 5.0)
        res = compute_multimodal_residuals(syn_hp, obs_hp)

    loop = run_physics_guided_fwi(
        initial, truth, obs, net, solver, wavelet, geom, replace(cfg, seed=config.get("seed", 42))
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        v_true=truth.cpu().numpy(),
        v_0=initial.cpu().numpy(),
        v_updated=loop["v_final"].detach().cpu().numpy(),
        wavelet=wavelet.cpu().numpy(),
        obs_sp=obs_hp.cpu().numpy(),
        syn_sp=syn_hp.cpu().numpy(),
        r_wave=res["r_wave"].cpu().numpy(),
        r_env=res["r_env"].cpu().numpy(),
        r_phase=res["r_phase"].cpu().numpy(),
        illum=sparse["illumination"].cpu().numpy(),
        g_sp=sp_norm.cpu().numpy(),
        g_ref=ref_norm.cpu().numpy(),
        g_pred=pred_norm.cpu().numpy(),
        sigma=sigma.cpu().numpy(),
    )
    print(f"Saved {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
