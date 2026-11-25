"""Paper outputs must remain usable with either archived metrics or full smoke artifacts."""

import json
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import yaml
from matplotlib.figure import Figure
from PIL import Image

from fwi_grad.eval import plotting


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ('normal', 'low_freq_cut', 'noisy', 'sparse', 'biased_init')


def check_pair(prefix):
    with Image.open(prefix.with_suffix('.png')) as png:
        assert png.size[0] > 1000
        assert png.info['dpi'] == pytest.approx((300, 300), abs=0.1)
        png.verify()
    pdf = prefix.with_suffix('.pdf').read_bytes()
    assert pdf.startswith(b'%PDF') and b'%%EOF' in pdf[-100:] and len(pdf) > 1000


@pytest.fixture
def exported(monkeypatch):
    figures = []
    savefig = Figure.savefig

    def capture(figure, filename, **kwargs):
        savefig(figure, filename, **kwargs)
        figures.append(figure)

    monkeypatch.setattr(Figure, 'savefig', capture)
    return figures


def test_fields_and_depth_profiles(tmp_path, exported):
    grad = np.linspace(-1, 2, 63).reshape(7, 9)
    velocity = 2000 + 15 * grad
    prefix = tmp_path / 'nested' / 'gradient'
    plotting.plot_gradient_and_uncertainty(velocity, grad, -grad, grad / 2, abs(grad), prefix, dx=20, dz=5)
    check_pair(prefix)
    maps = [ax for ax in exported[-1].axes if ax.images]
    assert maps[0].get_xlabel() == 'Distance (km)' and maps[0].get_ylabel() == 'Depth (km)'
    assert maps[0].images[0].get_extent() == pytest.approx([-.01, .17, .0325, -.0025])
    sigma, error = (ax.images[0] for ax in maps[-2:])
    assert sigma.get_clim()[1] == pytest.approx(np.percentile(abs(grad), 99))
    assert error.get_clim()[1] == pytest.approx(np.percentile(abs(grad / 2), 99))
    assert sigma.colorbar.ax is not error.colorbar.ax
    assert matplotlib.get_backend().lower() == 'agg'
    prefix = tmp_path / 'profiles'
    plotting.plot_velocity_and_profile(velocity, velocity + 50, {'Full adjoint': velocity + 1}, [0, 8], prefix)
    check_pair(prefix)
    assert exported[-1].axes[0].lines[-1].get_xdata()[0] == pytest.approx(.08)
    profile = next(ax for ax in exported[-1].axes if ax.get_title().startswith('Profile'))
    np.testing.assert_allclose(profile.lines[0].get_ydata(), np.arange(7) * .01)
    with pytest.raises(ValueError, match='trace'):
        plotting.plot_velocity_and_profile(velocity, velocity, {}, [9], prefix)
    with pytest.raises(ValueError, match='shape'):
        plotting.plot_gradient_and_uncertainty(velocity, grad, grad.T, grad, abs(grad), prefix)


def test_scenarios_and_closed_loop_tables(tmp_path, exported):
    def scenario(value):
        return {name: dict(cosine_similarity=value, rel_l2_error=0.2,
                           directional_derivative_relative_error=i + 1) for i, name in enumerate(SCENARIOS)}

    s2, s3 = {'per_scenario': scenario(0.8)}, scenario(0.9)
    del s3['noisy']['directional_derivative_relative_error']
    method = dict(loss_history=[2., 1.], metric_history=[{'mae': 9., 'ssim': .8}, {'mae': 8., 'ssim': .9}],
                  final_metrics={'mae': 8., 'ssim': .9}, final_full_loss=.5, grad_time_mean=.4, iter_time_mean=.5)
    loop = dict(initial_full_loss=2., full_adjoint=method,
                sparse_adjoint=dict(method, grad_time_mean=.1, iter_time_mean=.25))
    plotting.plot_scenario_comparison(s2, s3, tmp_path / 'scenario')
    plotting.plot_closed_loop_convergence(loop, tmp_path / 'convergence')
    times = next(ax for ax in exported[-1].axes if ax.get_title().startswith('(c)'))
    assert {'-75%', '-50%'} <= {label.get_text() for label in times.texts}
    for name in ('scenario', 'convergence'):
        check_pair(tmp_path / name)
    dest = tmp_path / 'tables' / 'results.tex'
    plotting.export_latex_tables(s2, s3, loop, dest)
    tex = dest.read_text()
    assert tex.count(r'\toprule') == tex.count(r'\bottomrule') == 2
    assert 'Biased init' in tex and 'Noisy' in tex and '--' in tex
    assert '100.0' in tex and '0.250' in tex  # 0.5 s / 0.25 s is a 100% speed gain.


def run_cli(metrics, out, *args):
    return subprocess.run([sys.executable, str(ROOT / 'scripts/plot_paper1_figures.py'), '--metrics',
                           str(metrics), '--output-dir', str(out), *args], cwd=out.parent, check=True,
                          capture_output=True, text=True).stdout


def test_archived_json_without_checkpoints(tmp_path):
    metrics = tmp_path / 'stage3_metrics.json'
    stats = dict(cosine_similarity=.95, rel_l2_error=.25, directional_derivative_relative_error=5.)
    method = dict(loss_history=[2.], final_full_loss=1., metric_history=[dict(mae=80., ssim=.8)],
                  final_metrics=dict(mae=80., ssim=.8), grad_time_mean=.1, iter_time_mean=.2)
    metrics.write_text(json.dumps(dict(stage2_baseline=stats, stage3_physics=stats,
                                      closed_loop=dict(initial_full_loss=2., full_adjoint=method))))
    out = tmp_path / 'out'
    message = run_cli(metrics, out)
    assert 'checkpoint' in message.lower()
    check_pair(out / 'scenario_comparison')
    check_pair(out / 'closed_loop_convergence')
    tex = (out / 'paper1_tables.tex').read_text()
    assert 'Overall' in tex and 'Biased init' not in tex
    assert not (out / 'velocity_and_profile.png').exists()


def test_zero_initial_objective(tmp_path):
    # A perfect starting model has no meaningful normalized misfit.
    method = dict(loss_history=[0.], final_full_loss=0., metric_history=[dict(mae=0., ssim=1.)],
                  grad_time_mean=.1, iter_time_mean=.2)
    plotting.plot_closed_loop_convergence(dict(initial_full_loss=0., full_adjoint=method), tmp_path / 'zero')
    check_pair(tmp_path / 'zero')


def test_large_directional_error_preserves_scenario_ticks(tmp_path, exported):
    # Inspect the labeled axes at export, independently of the PDF text extractor.
    metrics = {name: dict(cosine_similarity=.9, rel_l2_error=.3,
                          directional_derivative_relative_error=60. if name == 'noisy' else .5)
               for name in SCENARIOS}
    plotting.plot_scenario_comparison(metrics, metrics, tmp_path / 'scenarios')
    axes = [ax for ax in exported[0].axes if ax.get_title()]
    assert len(axes) == 3
    for ax in axes:
        assert [tick.get_text() for tick in ax.get_xticklabels()] == [
            'Normal', 'Low-freq. cut', 'Noisy', 'Sparse', 'Biased init.']


@pytest.mark.skipif(not all(path.is_file() for path in (
    ROOT / 'outputs/stage2_smoke/best.pt', ROOT / 'outputs/dataset/manifest.json',
)), reason='optional Stage 2 smoke checkpoint and trajectory data')
def test_full_smoke_cli(tmp_path):
    out = tmp_path / 'figs'
    manifest = ROOT / 'outputs/dataset/manifest.json'
    # Pair the figure replay with metrics from the same validation objective and update rule.
    config = yaml.safe_load((ROOT / 'configs/stage3_smoke.yaml').read_text())
    config.update(output_dir='stage3', pretrained_checkpoint=str(ROOT / 'outputs/stage2_smoke/best.pt'))
    config['dataset'].update(manifest=str(manifest), stage1_config=str(ROOT / 'configs/stage1_smoke.yaml'))
    config['training']['epochs'] = 1
    path = tmp_path / 'smoke.yaml'
    path.write_text(yaml.safe_dump(config))
    subprocess.run([sys.executable, str(ROOT / 'scripts/train_stage3.py'), '--config', str(path),
                    '--device', 'cpu'], check=True)
    metrics = tmp_path / 'stage3/stage3_metrics.json'
    entries = json.loads(manifest.read_text())['samples']
    sample = next(row for row in entries if row['split'] == 'test' and row['scenario'] == 'noisy')
    run_cli(metrics, out, '--sample', str(manifest.parent / sample['path']), '--trace-x-indices', '1', '8')
    for name in ('gradient_and_uncertainty', 'velocity_and_profile', 'scenario_comparison', 'closed_loop_convergence'):
        check_pair(out / name)
    tex = (out / 'paper1_tables.tex').read_text()
    assert 'Biased init' in tex and 'Stage 3' in tex and '--' not in '\n'.join(
        row for row in tex.splitlines() if ' & ' in row)
    # The validation probes must match train_stage3, including their RNG order.
    # Check scenario means weighted by the actual number of validation samples.
    report = json.loads(metrics.read_text())
    for stage, key in (('Stage 2', 'stage2_baseline'), ('Stage 3', 'stage3_physics')):
        rows = [line.split(' & ') for line in tex.splitlines() if f' & {stage} & ' in line]
        errors = [float(row[-1].split()[0]) for row in rows if row[0] != 'Overall']
        counts = [sum(e['scenario'] == scene and e['split'] == 'val' for e in entries) for scene in SCENARIOS]
        assert len(set(errors)) > 1
        assert np.average(errors, weights=counts) == pytest.approx(
            report[key]['directional_derivative_relative_error'], abs=.001)
