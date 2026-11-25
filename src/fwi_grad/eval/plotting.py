"""Headless Paper 1 figures and booktabs tables for Stage 2/3 FWI results."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator


_SCENARIOS = {
    'normal': 'Normal', 'low_freq_cut': 'Low-freq. cut', 'noisy': 'Noisy',
    'sparse': 'Sparse', 'biased_init': 'Biased init.',
}
_METRICS = {
    'cosine_similarity': 'Cosine similarity', 'rel_l2_error': r'Relative $L_2$ error',
    'directional_derivative_relative_error': 'Directional derivative\nrelative error',
}
_METHODS = {
    'full_adjoint': 'Full adjoint', 'sparse_adjoint': 'Sparse adjoint',
    'stage2_guided': 'Stage 2 guided', 'stage3_guided': 'Stage 3 guided',
}
_COLORS = ('#254e70', '#ce6a2b', '#318475', '#995981', '#696969', '#947123')
_MARKERS = ('o', 's', '^', 'D', 'v', 'P')
_STYLE = {
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 8,
    'mathtext.fontset': 'dejavuserif', 'text.usetex': False, 'pdf.fonttype': 42,
    'axes.titlesize': 9, 'axes.labelsize': 8, 'axes.linewidth': .6,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'lines.linewidth': 1.2,
}


@contextmanager
def _figure(prefix, size):
    with plt.rc_context(_STYLE):
        fig = plt.figure(figsize=size, layout='constrained')
        try:
            yield fig
            prefix = Path(prefix)
            prefix.parent.mkdir(parents=True, exist_ok=True)
            for suffix in ('.png', '.pdf'):
                fig.savefig(prefix.parent / (prefix.name + suffix), dpi=300, facecolor='white')
        finally:
            plt.close(fig)


def _fields(*images):
    fields = [np.asarray(image, dtype=float) for image in images]
    for field in fields:
        if field.ndim != 2 or not field.size or field.shape != fields[0].shape:
            raise ValueError('all fields must have the same nonempty [depth, x] shape')
        if not np.isfinite(field).all():
            raise ValueError('fields contain non-finite values')
    return fields


def _image(ax, field, title, cmap, norm, dx, dz):
    nz, nx = field.shape
    # Pixel centers are the grid nodes, including the surface at z = 0.
    extent = (-.5 * dx / 1000, (nx - .5) * dx / 1000, (nz - .5) * dz / 1000, -.5 * dz / 1000)
    image = ax.imshow(field, norm=norm, cmap=cmap, origin='upper', interpolation='nearest',
                      aspect='equal', extent=extent)
    ax.set(title=title, xlabel='Distance (km)', ylabel='Depth (km)')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    return image


def plot_gradient_and_uncertainty(v_k, g_ref, g_sparse, g_pred, sigma, save_prefix, *,
                                  normalized=False, dx=10., dz=10.):
    """Plot aligned [nz, nx] fields with grid spacings in metres and gradients and sigma in the same space."""
    v, ref, sparse, pred, sigma = _fields(v_k, g_ref, g_sparse, g_pred, sigma)
    if sigma.min() < 0:
        raise ValueError('sigma must be nonnegative')
    error = np.abs(pred - ref)
    limit = max(np.abs(g).max() for g in (ref, sparse, pred)) or 1.
    signed = Normalize(-limit, limit)
    with _figure(save_prefix, (7.16, 5.45)) as fig:
        gs = fig.add_gridspec(4, 3, height_ratios=[1, .065, 1, .065])
        panels = [
            (v, '(a) Current velocity', 'viridis', Normalize(v.min(), v.max())),
            (ref, '(b) Full-shot reference', 'RdBu_r', signed),
            (sparse, '(c) Sparse-shot gradient', 'RdBu_r', signed),
            (pred, '(d) Predicted gradient', 'RdBu_r', signed),
            (sigma, r'(e) Predicted uncertainty $\sigma$', 'magma', Normalize(0., np.percentile(sigma, 99) or 1.)),
            (error, r'(f) $|g_{\mathrm{pred}}-g_{\mathrm{ref}}|$', 'magma', Normalize(0., np.percentile(error, 99) or 1.)),
        ]
        images = [_image(fig.add_subplot(gs[2 * (i // 3), i % 3]), *panel, dx, dz)
                  for i, panel in enumerate(panels)]
        unit = ' (norm.)' if normalized else ''
        for idx, slot, label in ((0, gs[1, 0], 'Velocity (m/s)'), (1, gs[1, 1:], 'Gradient (b–d)' + unit),
                                 (4, gs[3, 1], 'Uncertainty' + unit), (5, gs[3, 2], 'Absolute error' + unit)):
            fig.colorbar(images[idx], cax=fig.add_subplot(slot), orientation='horizontal', label=label)


def _scenario_rows(metrics, stage):
    """Accept a full native report, best validation block, or scenario-keyed map."""
    key = 'stage2_baseline' if stage == 2 else 'stage3_physics'
    if key in metrics:
        metrics = metrics[key]
    elif 'history' in metrics and 'best_epoch' in metrics:
        metrics = next(row['val'] for row in metrics['history'] if row['epoch'] == metrics['best_epoch'])
    groups = metrics.get('per_scenario', metrics)
    rows = {name: groups[name] for name in _SCENARIOS if name in groups}
    overall = metrics.get('overall', metrics)
    if any(key in overall for key in _METRICS):
        rows['overall'] = overall
    return rows


def _number(row, key):
    value = row.get(key)
    return float(value) if value is not None and np.isfinite(float(value)) else np.nan


def plot_scenario_comparison(stage2_metrics, stage3_metrics, save_prefix):
    """Compare stage means in scenario order. Archived overall-only reports get one group."""
    stages = [_scenario_rows(stage2_metrics, 2), _scenario_rows(stage3_metrics, 3)]
    names = list(_SCENARIOS) if any(any(k in rows for k in _SCENARIOS) for rows in stages) else ['overall']
    ticks = [_SCENARIOS.get(name, 'Overall') for name in names]
    stage_labels = ['Stage 2 (scratch)' if stage2_metrics.get('initialization') == 'scratch' else 'Stage 2', 'Stage 3']
    with _figure(save_prefix, (7.16, 2.95)) as fig:
        grid = fig.add_gridspec(2, 3, height_ratios=[.08, 1])
        legend_ax = fig.add_subplot(grid[0, :])
        legend_ax.axis('off')
        axes = [fig.add_subplot(grid[1, col]) for col in range(3)]
        for idx, (ax, (key, title)) in enumerate(zip(axes, _METRICS.items())):
            x = np.arange(len(names))
            heights = np.array([[_number(rows.get(n, {}), key) for n in names] for rows in stages])
            finite = heights[np.isfinite(heights)]
            # Retain zero as the bar baseline while keeping small errors legible
            # alongside large finite-difference errors in the noisy scenario.
            if idx == 2 and finite.size and finite.max() > 20 and finite.min() >= 0:
                ax.set_yscale('symlog', linthresh=1)
                ax.set_ylabel('Linear to 1, then log')
                if finite.max() < 100:
                    ax.set_yticks([tick for tick in (0, 1, 3, 10, 30) if tick <= finite.max()])
                    ax.yaxis.set_major_formatter('{x:g}')
            for j, values in enumerate(heights):
                positions = x + (j - .5) * .36
                ax.bar(positions, values, .35, color=_COLORS[j], hatch=('', '///')[j],
                       edgecolor='white', linewidth=.4, label=stage_labels[j])
                for pos in positions[~np.isfinite(values)]:
                    ax.annotate('NA', (pos, 0), rotation=90, ha='center', va='bottom', fontsize=6)
            ax.set_title(f'({chr(97 + idx)}) {title}')
            ax.set_xticks(x, ticks, rotation=38 if len(names) > 1 else 0, ha='right' if len(names) > 1 else 'center')
            ax.set_axisbelow(True)
            ax.grid(axis='y', alpha=.25)
            ax.axhline(0, linewidth=.5, color='0.4')
        handles, labels = axes[0].get_legend_handles_labels()
        legend_ax.legend(handles, labels, loc='center', ncol=2, fontsize=7)


def _methods(loop, labels=_METHODS):
    records = [(key, loop[key]) for key in labels if key in loop]
    if not records:
        raise ValueError('closed_loop requires at least one method record')
    return records


def _gain(full_time, method_time):
    return 100 * (full_time / method_time - 1) if full_time > 0 and method_time > 0 else np.nan


def plot_closed_loop_convergence(closed_loop_dict, save_prefix, *, method_labels=_METHODS):
    """Use full-shot evaluations at velocity iterates, and post-update MAE/SSIM.

    A full_loss_history, when supplied, starts at v_0. Native sparse/guided
    reports only measure the common objective at the endpoints. Their internal
    sparse/check-shot loss_history is on a different (possibly varying) survey.
    The first key in method_labels names the full-shot timing baseline.
    """
    loop = closed_loop_dict
    records = _methods(loop, method_labels)
    baseline_key = next(iter(method_labels))
    initial = _number(loop, 'initial_full_loss')
    if not np.isfinite(initial) or initial < 0:
        raise ValueError('initial_full_loss must be finite and nonnegative')
    with _figure(save_prefix, (7.16, 3.75)) as fig:
        grid = fig.add_gridspec(3, 3, height_ratios=[.13, 1, 1], width_ratios=[1, 1, 1.17])
        legend_ax = fig.add_subplot(grid[0, :])
        legend_ax.axis('off')
        objective = fig.add_subplot(grid[1:, 0])
        mae = fig.add_subplot(grid[1, 1])
        ssim = fig.add_subplot(grid[2, 1], sharex=mae)
        times = fig.add_subplot(grid[1:, 2])
        handles = []
        endpoints_only = False
        for key, record in records:
            idx = list(method_labels).index(key)
            style = dict(color=_COLORS[idx], marker=_MARKERS[idx], markersize=3)
            history = record['metric_history']
            steps = np.arange(1, len(history) + 1)
            if 'initial_metrics' in loop:
                steps = np.arange(len(history) + 1)
                history = [loop['initial_metrics'], *history]
            handle, = mae.plot(steps, [_number(row, 'mae') for row in history], label=method_labels[key], **style)
            handles.append(handle)
            ssim.plot(steps, [_number(row, 'ssim') for row in history], **style)
            if 'full_loss_history' in record:
                values = record['full_loss_history']
                iterations = np.arange(len(values))
            elif key == baseline_key:
                values = [*record['loss_history'], record['final_full_loss']]
                iterations = np.arange(len(values))
            else:
                values = [initial, record['final_full_loss']]
                iterations = [0, len(record['metric_history'])]
                style['linestyle'] = ':'
                endpoints_only = True
            objective.plot(iterations, np.asarray(values) / initial if initial else values, **style)
        legend_ax.legend(handles=handles, loc='center', ncol=len(handles), fontsize=7)
        objective.set(title='(a) Full-shot objective', xlabel='Iteration',
                      ylabel=r'$J(v_k)\,/\,J(v_0)$' if initial else r'$J(v_k)$ (initial misfit is zero)')
        if endpoints_only:
            objective.set_xlabel('Iteration\nDotted: endpoints only')
        mae.set(title='(b) Velocity accuracy', ylabel='MAE (m/s)')
        mae.tick_params(labelbottom=False)
        ssim.set(xlabel='Iteration', ylabel='SSIM')
        for ax in (objective, mae, ssim):
            ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
            ax.grid(alpha=.22)

        y = np.arange(len(records))
        full = loop.get(baseline_key, {})
        for i, (key, label) in enumerate((('grad_time_mean', 'Gradient'), ('iter_time_mean', 'Iteration'))):
            seconds = [_number(row, key) for _, row in records]
            bars = times.barh(y + (i - .5) * .36, seconds, .34, color=('#adc4d6', '#405a71')[i], label=label)
            baseline = _number(full, key)
            labels = [f'{100 * (value / baseline - 1):+.0f}%' if baseline > 0 and np.isfinite(value) else ''
                      for value in seconds]
            times.bar_label(bars, labels=labels, padding=2, fontsize=6)
        times.set_yticks(y, [method_labels[k].replace(' guided', '') for k, _ in records], fontsize=6.5)
        times.invert_yaxis()
        times.margins(x=.45)
        times.xaxis.set_major_locator(MaxNLocator(3))
        times.set(title='(c) Mean step time', xlabel='Seconds\nTime change vs. full (%)')
        times.legend(loc='upper center', bbox_to_anchor=(.5, -.26), ncol=2, fontsize=6)


def plot_velocity_and_profile(v_true, v_init, v_methods: dict[str, np.ndarray], trace_x_indices, save_prefix,
                              *, dx=10., dz=10.):
    """Compare velocities in m/s at zero-based trace indices with grid spacings in metres."""
    fields = _fields(v_true, v_init, *v_methods.values())
    labels = ['True', 'Initial', *[_METHODS.get(name, name) for name in v_methods]]
    nz, nx = fields[0].shape
    indices = list(trace_x_indices)
    if not indices or any(not isinstance(i, (int, np.integer)) or not 0 <= i < nx for i in indices):
        raise ValueError(f'trace_x_indices must contain integer trace indices in [0, {nx - 1}]')
    cols = min(3, len(fields))
    image_rows = (len(fields) + cols - 1) // cols
    trace_cols = min(3, len(indices))
    trace_rows = (len(indices) + trace_cols - 1) // trace_cols
    norm = Normalize(min(a.min() for a in fields), max(a.max() for a in fields))
    with _figure(save_prefix, (7.16, 2.05 * image_rows + 2.15 * trace_rows + .9)) as fig:
        grid = fig.add_gridspec(4, 1, height_ratios=[image_rows, .055, trace_rows, .10])
        maps = grid[0].subgridspec(image_rows, cols)
        for i, (label, field) in enumerate(zip(labels, fields)):
            ax = fig.add_subplot(maps[i // cols, i % cols])
            image = _image(ax, field, label, 'viridis', norm, dx, dz)
            for x in indices:
                ax.axvline(x * dx / 1000, color='white', ls='--', linewidth=.7)
        fig.colorbar(image, cax=fig.add_subplot(grid[1]), orientation='horizontal', label='Velocity (m/s)')
        traces = grid[2].subgridspec(trace_rows, trace_cols)
        depth = np.arange(nz) * dz / 1000
        for i, x in enumerate(indices):
            ax = fig.add_subplot(traces[i // trace_cols, i % trace_cols])
            for j, (label, field) in enumerate(zip(labels, fields)):
                ax.plot(field[:, x], depth, label=label, color=_COLORS[j % len(_COLORS)],
                        ls=('-', '--', '-.', ':')[j % 4])
            ax.set(xlabel='Velocity (m/s)', ylabel='Depth (km)', title=f'Profile at x = {x * dx / 1000:g} km')
            ax.invert_yaxis()
            ax.grid(alpha=.22)
        handles, names = ax.get_legend_handles_labels()
        legend_ax = fig.add_subplot(grid[3])
        legend_ax.axis('off')
        legend_ax.legend(handles, names, loc='center', ncol=min(6, len(names)), fontsize=7)


def _table(caption, label, columns, header, rows):
    return '\n'.join([r'\begin{table*}[t]', r'\centering', rf'\caption{{{caption}}}',
                      rf'\label{{tab:paper1-{label}}}', rf'\begin{{tabular}}{{{columns}}}',
                      r'\toprule', ' & '.join(header) + r' \\', r'\midrule',
                      *[' & '.join(row) + r' \\' for row in rows],
                      r'\bottomrule', r'\end{tabular}', r'\end{table*}'])


def _fmt(value, digits=3):
    return f'{value:.{digits}f}' if np.isfinite(value) else '--'


def export_latex_tables(stage2_metrics, stage3_metrics, closed_loop_dict, save_path):
    """Write paste-ready gradient and closed-loop tables. Requires the booktabs package."""
    stages = [_scenario_rows(stage2_metrics, 2), _scenario_rows(stage3_metrics, 3)]
    rows = []
    baseline_label = 'Stage 2 (scratch)' if stage2_metrics.get('initialization') == 'scratch' else 'Stage 2'
    for name, label in {**_SCENARIOS, 'overall': 'Overall'}.items():
        if any(name in stage for stage in stages):
            for model, stage in zip((baseline_label, 'Stage 3'), stages):
                rows.append([label, model, *[_fmt(_number(stage.get(name, {}), k)) for k in _METRICS]])
    caption = 'Gradient prediction accuracy by validation scenario.'
    if any('--' in row for row in rows):
        caption += ' Dashes denote unavailable values.'
    gradient = _table(caption,
                      'gradients', 'llrrr', ['Scenario', 'Model', 'Cosine similarity',
                                            r'Relative $L_2$ error', 'Directional derivative rel. error'], rows)
    rows = []
    loop = closed_loop_dict
    full = _number(loop.get('full_adjoint', {}), 'iter_time_mean')
    for key, record in _methods(loop):
        final = record['final_metrics']
        sec = _number(record, 'iter_time_mean')
        rows.append([_METHODS[key], _fmt(_number(record, 'final_full_loss'), 6), _fmt(_number(final, 'mae')),
                     _fmt(_number(final, 'ssim'), 4), _fmt(_number(record, 'grad_time_mean'), 4),
                     _fmt(sec, 4), _fmt(_gain(full, sec), 1)])
    closed = _table(r'Closed-loop inversion on the test model. Times are means per iteration. '
                    r'Speed gain is $100(t_{\mathrm{full}}/t-1)$ using total iteration time.',
                    'closed-loop', 'lrrrrrr', ['Method', r'Final $J_{\mathrm{full}}$', 'MAE (m/s)', 'SSIM',
                                              'Gradient (s)', 'Iteration (s)', r'Gain (\%)'], rows)
    dest = Path(save_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text('% Preamble: \\usepackage{booktabs}\n' + gradient + '\n\n' + closed + '\n', encoding='utf-8')
