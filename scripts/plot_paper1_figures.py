#!/usr/bin/env python3
"""Generate Paper 1 figures and booktabs tables from a Stage 3 evaluation report.

Example:
    python scripts/plot_paper1_figures.py --metrics outputs/stage3_smoke/stage3_metrics.json \
        --output-dir outputs/paper1_figures

The checkpoint beside the report supplies the architecture and survey settings.
Validation scenarios use paired finite-difference probes. Maps use test data.
Use --metrics-only for an archived report without companion files.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

if __package__:
    from . import train_stage3 as stage3
else:
    import train_stage3 as stage3

from fwi_grad.data.torch_dataset import collate_fwi_batch
from fwi_grad.eval.metrics import gradient_metrics, velocity_metrics
from fwi_grad.eval.plotting import (
    export_latex_tables, plot_closed_loop_convergence, plot_gradient_and_uncertainty,
    plot_scenario_comparison, plot_velocity_and_profile,
)
from fwi_grad.inversion.baseline_fwi import run_traditional_fwi
from fwi_grad.inversion.guided_fwi import GuidedFWIConfig, run_physics_guided_fwi
from fwi_grad.losses.physics_losses import subset_misfit
from fwi_grad.models import MultiModalGradientNet
from fwi_grad.physics.geometry import select_sparse_shots


def _load_model(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    model = MultiModalGradientNet(**config.get('model', {}))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.eval().to(device), config


@torch.no_grad()
def _by_scenario(models, dataset, config, survey, seed):
    """Preserve the validation loader order and the original sequence of physical probes."""
    loader = DataLoader(dataset, batch_size=config['training']['batch_size'], collate_fn=collate_fwi_batch)
    results = {}
    for name, model in models.items():
        rng = torch.Generator().manual_seed(seed)
        grouped = defaultdict(list)
        for batch in loader:
            inputs = stage3.stage2._inputs(batch, survey.wavelet.device)
            output = model(**inputs)
            for i, scenario in enumerate(batch['scenario']):
                row = gradient_metrics(output['g_pred'][i], batch['g_ref_norm'][i])
                error, _ = stage3._physics_sample(model, batch, inputs, output, i, survey,
                                                  config['physics_loss'], rng, False)
                row['directional_derivative_relative_error'] = error.item()
                grouped[scenario].append(row)
        results[name] = {scene: {key: np.mean([r[key] for r in rows]) for key in rows[0]}
                         for scene, rows in grouped.items()}
    return results


class _FullShotHistory:
    """Observe pre-update velocities during replay while using the native FWI optimizers."""

    def __init__(self, solver):
        self.solver = solver
        self.values = []

    def __getattr__(self, name):
        return getattr(self.solver, name)

    def adjoint_gradient(self, v, d_obs, wavelet, geom, shot_indices=None, **kwargs):
        result = self.solver.adjoint_gradient(v, d_obs, wavelet, geom, shot_indices=shot_indices, **kwargs)
        # Both optimizers pass an explicit shot list once per update. A guided
        # full-adjoint fallback omits it and revisits the same pre-update velocity.
        if shot_indices is not None:
            if len(shot_indices) == geom.n_shots:
                self.values.append(result['loss'])
            else:
                with torch.no_grad():
                    loss = subset_misfit(v, d_obs, self.solver, wavelet, geom, range(geom.n_shots),
                                         kwargs.get('f_cut', 0.))
                self.values.append(loss.item())
        return result


def _replay(loop, sample, models, survey, config, seed):
    truth, initial = (sample[k][0].to(survey.wavelet) for k in ('v_true', 'v_init'))
    loop['initial_metrics'] = velocity_metrics(initial, truth)
    obs = survey.observations(sample['model_id'], sample['scenario'], truth)
    cfg = GuidedFWIConfig(**config.get('guided_fwi', {}))
    cfg.f_cut = cfg.f_cut or float(sample['q_k'][0] * 10)
    full = list(range(survey.geom.n_shots))
    shots = {'full_adjoint': full, 'sparse_adjoint': select_sparse_shots(len(full), float(cfg.sparse_ratio))}
    velocities = {}
    for key in (*shots, *models):
        if key not in loop:
            continue
        recorder = _FullShotHistory(survey.solver)
        common = (recorder, survey.wavelet, survey.geom)
        if key in shots:
            result = run_traditional_fwi(initial, truth, obs, *common, cfg.n_iters, cfg.lr,
                                         shot_indices=shots[key], f_cut=cfg.f_cut, v_bounds=cfg.v_bounds)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(seed)
                result = run_physics_guided_fwi(initial, truth, obs, models[key], *common, cfg)
        final = result['v_final']
        with torch.no_grad():
            loss = subset_misfit(final, obs, survey.solver, survey.wavelet, survey.geom, full, cfg.f_cut).item()
        history = [*recorder.values, loss]
        record = loop[key]
        # Catch a different survey, checkpoint, or trajectory before associating
        # these maps with the archived timings and accuracy measurements.
        if (len(history) != len(record['metric_history']) + 1
                or not np.allclose([history[0], loss], [loop['initial_full_loss'], record['final_full_loss']],
                                   rtol=.002, atol=1e-10)):
            raise ValueError(f'{key} replay differs from the report. Check the checkpoint, survey and test sample.')
        record['full_loss_history'] = history
        velocities[key] = final.cpu().numpy()
    return velocities


def _single_test_sample(path, dataset_options):
    # Reuse Stage 1 tensor shaping, gradient normalization and q_k construction
    # even for a standalone .pt copied away from its original manifest.
    meta = torch.load(path, map_location='cpu', weights_only=True)['meta']
    if meta['split'] != 'test':
        raise ValueError('--sample must select a Stage 1 test sample')
    entry = dict(meta, path=str(path.resolve()))
    with TemporaryDirectory() as tmp:
        manifest = Path(tmp) / 'manifest.json'
        manifest.write_text(json.dumps({'samples': [entry], 'splits': {'test': [meta['model_id']]}}))
        return stage3._TrajectoryDataset(manifest, split='test', **dataset_options)[0]


def _config_dir(checkpoint, config, override):
    if override:
        return override
    # These are the layouts used by the repository and by train_stage3's CLI tests.
    for directory in (ROOT / 'configs', checkpoint.parent.parent, checkpoint.parent.parent / 'configs'):
        if (directory / config['output_dir']).resolve() == checkpoint.parent.resolve():
            return directory
    return checkpoint.parent


def _field_figures(args, report, stages, checkpoint, out):
    device = torch.device(args.device)
    if device.type == 'cpu':
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    model, config = _load_model(checkpoint, device)
    directory = _config_dir(checkpoint, config, args.config_dir)
    data = config['dataset']
    source = data['stage1_config']
    if isinstance(source, str):
        path = directory / source
        source = yaml.safe_load(path.read_text(encoding='utf-8')) if path.is_file() else None
    survey = stage3._Survey(source, device) if source is not None else None
    spacing = dict(dx=survey.geom.dx, dz=survey.geom.dz) if survey else {}
    options = dict(max_iters=data.get('max_iters', 20), normalizer=model.normalizer)
    manifest = args.manifest or directory / data['manifest']
    loop = report['closed_loop']
    test_case = None
    val = None
    if manifest.is_file():
        shared = dict(scenarios=data.get('scenarios'), **options)
        val = stage3._TrajectoryDataset(manifest, split='val', **shared)
        test = stage3._TrajectoryDataset(manifest, split='test', **shared)
        index = next((i for i, row in enumerate(test.entries) if row['model_id'] == loop.get('model_id')
                      and row['scenario'] == loop.get('scenario')), None)
        if index is not None:
            test_case = test[index]
    sample = _single_test_sample(args.sample, options) if args.sample else test_case
    if sample is not None:
        with torch.no_grad():
            batch = collate_fwi_batch([sample])
            prediction = model(**stage3.stage2._inputs(batch, device))
            # The learned sigma and absolute residual both live in the normalized
            # reference target space. Use that same transform for the sparse map.
            sparse, _ = model.normalizer.normalize(sample['g_sparse'], scale=sample['g_ref_scale'])
            plot_gradient_and_uncertainty(sample['v_k'][0], sample['g_ref_norm'][0], sparse[0],
                                          prediction['g_pred'][0, 0].cpu(), prediction['sigma'][0, 0].cpu(),
                                          out / 'gradient_and_uncertainty', normalized=True, **spacing)
        print(f"Gradient and uncertainty: {sample['model_id']}, {sample['scenario']}.")
        if test_case is None and (sample['model_id'], sample['scenario']) == (loop.get('model_id'), loop.get('scenario')):
            test_case = sample
    else:
        print('Gradient figure skipped. Supply a test trajectory with --sample or --manifest.')

    if survey is None:
        print(f"Scenario evaluation and velocity replay skipped. Use --config-dir to locate {data['stage1_config']}.")
        return
    seed = config['training'].get('seed', 42)
    models = {}
    baseline_path = args.stage2_checkpoint
    if baseline_path is None and report.get('pretrained_checkpoint'):
        baseline_path = Path(report['pretrained_checkpoint'])
        if not baseline_path.is_absolute():
            baseline_path = directory / baseline_path
        if not baseline_path.is_file():
            baseline_path = directory / config['pretrained_checkpoint']
    if baseline_path and baseline_path.is_file():
        models['stage2_guided'], _ = _load_model(baseline_path, device)
    elif report.get('initialization') == 'scratch':
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            models['stage2_guided'] = MultiModalGradientNet(**config.get('model', {})).eval().to(device)
        print('Stage 2 baseline uses the original seeded scratch initialization.')
    else:
        print('Stage 2 replay unavailable. Supply the matching --stage2-checkpoint.')
    models['stage3_guided'] = model
    if val:
        grouped = _by_scenario(models, val, config, survey, seed)
        for stage, name in enumerate(('stage2_guided', 'stage3_guided')):
            if name in grouped:
                stages[stage]['per_scenario'] = grouped[name]
    else:
        print('Validation scenarios unavailable. Supply the original --manifest.')
    if test_case is None:
        print('Velocity replay skipped. Supply the test case named in closed_loop via --sample or --manifest.')
        return
    velocities = _replay(loop, test_case, models, survey, config, seed)
    nx = test_case['v_true'].shape[-1]
    indices = args.trace_x_indices or sorted({nx // 4, nx // 2, 3 * nx // 4})
    plot_velocity_and_profile(test_case['v_true'][0], test_case['v_init'][0], velocities,
                              indices, out / 'velocity_and_profile', **spacing)
    print(f"Velocity profiles: {test_case['model_id']}, {test_case['scenario']}, x = {indices}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--metrics', '--stage3-metrics', type=Path, default=ROOT / 'outputs/stage3_smoke/stage3_metrics.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/paper1_figures')
    parser.add_argument('--checkpoint', type=Path, help='Stage 3 checkpoint (default: best_stage3.pt beside metrics)')
    parser.add_argument('--stage2-checkpoint', type=Path, help='Override the baseline path in the report')
    parser.add_argument('--sample', type=Path, help='Stage 1 test .pt for the gradient figure (default: closed-loop case)')
    parser.add_argument('--manifest', type=Path, help='Stage 1 manifest (default: checkpoint dataset configuration)')
    parser.add_argument('--config-dir', type=Path, help='Directory against which checkpoint config paths were resolved')
    parser.add_argument('--trace-x-indices', type=int, nargs='+', help='Zero-based columns for the velocity profiles')
    parser.add_argument('--metrics-only', action='store_true', help='Use only the recorded metrics and full-shot endpoints')
    parser.add_argument('--device', default='cpu', help='PyTorch device for optional evaluation and replay')
    args = parser.parse_args()
    for path in (args.checkpoint, args.stage2_checkpoint, args.manifest, args.sample):
        if path and not path.is_file():
            parser.error(f'File does not exist: {path}')
    report = json.loads(args.metrics.read_text(encoding='utf-8'))
    stages = [dict(report[name]) for name in ('stage2_baseline', 'stage3_physics')]
    stages[0]['initialization'] = report.get('initialization')
    checkpoint = args.checkpoint or args.metrics.parent / 'best_stage3.pt'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Finish inference, replay checks and rendering before replacing a previous run.
    with TemporaryDirectory(dir=args.output_dir) as tmp:
        out = Path(tmp)
        if not args.metrics_only and checkpoint.is_file():
            _field_figures(args, report, stages, checkpoint, out)
        else:
            print('Using archived metrics. Gradient and velocity figures require a checkpoint and test sample.')
        plot_scenario_comparison(*stages, out / 'scenario_comparison')
        plot_closed_loop_convergence(report['closed_loop'], out / 'closed_loop_convergence')
        export_latex_tables(*stages, report['closed_loop'], out / 'paper1_tables.tex')
        generated = sorted(out.iterdir())
        for file in generated:
            file.replace(args.output_dir / file.name)
    print(f'Paper 1 output directory: {args.output_dir.resolve()}')
    print('Generated: ' + ', '.join(file.name for file in generated))


if __name__ == '__main__':
    main()
