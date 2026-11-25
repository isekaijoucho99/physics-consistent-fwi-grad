#!/usr/bin/env python3
"""Draw a clean, minimal IEEE TGRS Figure 1 in Times New Roman."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.colors import LogNorm, Normalize, to_rgb
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib.path import Path as PlotPath

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper1_tgrs" / "figures"

INK, MUTED, RULE = "#1A2530", "#4A5A6A", "#CFD8DF"
SLATE, AMBER, TEAL = "#3B5E7A", "#9A6310", "#0E6B65"
SLATE_BG, AMBER_BG, TEAL_BG = "#F8FAFC", "#FFFDF5", "#F2FBFA"
GRAD = Normalize(-0.85, 0.85)
VELOCITY = Normalize(1.45, 4.70)
METHODS = {
    "full_adjoint": ("Full adjoint", "#2C3E50", "-", 1.4),
    "sparse_adjoint": ("Sparse adjoint", "#C05610", "--", 1.4),
    "A5_proposed_guided": ("Proposed", "#0E7A74", "-", 2.0),
}


def load_case():
    with np.load(ROOT / "outputs" / "fig1_marmousi2_assets.npz") as archive:
        arrays = dict(archive)
    summary = ROOT / "outputs" / "paper1_benchmarks" / "paper1_benchmarks_summary.json"
    report = json.loads(summary.read_text(encoding="utf-8"))
    case, = (row for row in report["cases"]
             if row["benchmark"] == "marmousi2" and row["f_cut"] == 5.0)
    return arrays, case, report["requested_config"]


def label(ax, x, y, text, size=9.0, *, color=INK, weight="normal", ha="center", **kwargs):
    return ax.text(x, y, text, ha=ha, va="center", fontsize=size, color=color,
                   fontweight=weight, linespacing=1.15, zorder=6, **kwargs)


def box(ax, x, y, w, h, fill, edge="none", *, lw=0.75, radius=0.007):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=1,
    ))


def line(ax, points, color=RULE, lw=0.75, **kwargs):
    return ax.plot(*zip(*points), color=color, lw=lw, solid_capstyle="round", **kwargs)[0]


def arrow(ax, points, color=SLATE, lw=1.05, *, head=6.5, curve=False, **kwargs):
    codes = [PlotPath.MOVETO] + [PlotPath.CURVE4 if curve else PlotPath.LINETO] * (len(points) - 1)
    ax.add_patch(FancyArrowPatch(
        path=PlotPath(points, codes), arrowstyle="-|>", mutation_scale=head,
        facecolor=color, edgecolor=color, linewidth=lw, joinstyle="round", **kwargs,
    ))


def tile(fig, rect, data, cmap, norm, *, extent=None, zorder=2):
    ax = fig.add_axes(rect, zorder=zorder)
    ax.imshow(data, cmap=cmap, norm=norm, extent=extent, origin="upper",
              interpolation="nearest", aspect="auto")
    ax.set(xticks=[], yticks=[])
    for spine in ax.spines.values():
        spine.set(color="#8A9BA5", linewidth=0.45)
    return ax


def top_card(ax, x, w, title, bg, edge):
    box(ax, x, 0.405, w, 0.535, bg, edge, lw=0.95, radius=0.010)
    label(ax, x + w / 2, 0.912, title, 10.2, weight="bold")
    line(ax, [(x + 0.014, 0.888), (x + w - 0.014, 0.888)], edge, 0.55)


def draw_physics(fig, ax, a, dt):
    top_card(ax, 0.015, 0.240, "Sparse-Shot Physics", SLATE_BG, "#94A3B8")
    label(ax, 0.081, 0.858, "Multi-Modal Residuals", 8.8, weight="bold")
    label(ax, 0.200, 0.858, "Physics Cues", 8.8, weight="bold")

    for layer, key in enumerate(("r_phase", "r_env", "r_wave")):
        shot = a[key][0].T
        vmax = np.percentile(np.abs(shot), 97)
        sheet = tile(fig, [0.027 + 0.013 * layer, 0.742 - 0.016 * layer, 0.078, 0.090],
                     shot, "seismic", Normalize(-vmax, vmax), zorder=2 + layer)
        if key == "r_wave":
            sheet.plot(np.linspace(0.06, 0.94, 7), np.ones(7), "v", color=INK,
                       markeredgecolor="white", markeredgewidth=0.3, markersize=3.5,
                       transform=sheet.transAxes, clip_on=False)
    label(ax, 0.082, 0.690, "Wave · Env. · Phase", 8.2, color=MUTED)

    illum = np.log1p(a["illum"] / (np.percentile(a["illum"], 99) + 1e-8))
    label(ax, 0.155, 0.805, r"$I_k$", 9.5)
    tile(fig, [0.171, 0.774, 0.072, 0.062], illum, "cividis",
         Normalize(*np.percentile(illum, [1, 99])))
    label(ax, 0.155, 0.720, r"$g_{\mathrm{sp},k}$", 9.0)
    tile(fig, [0.171, 0.689, 0.072, 0.062], a["g_sp"], "RdBu_r", GRAD)

    gathers = [a[key][0].T for key in ("obs_sp", "syn_sp")]
    limit = np.percentile(np.abs(gathers), 97)
    for x, name, gather in zip((0.027, 0.088), ("Observed", "Synthetic"), gathers):
        label(ax, x + 0.024, 0.650, name, 8.2)
        tile(fig, [x, 0.554, 0.048, 0.078], gather, "Greys", Normalize(-limit, limit))
    arrow(ax, [(0.082, 0.634), (0.082, 0.674)], head=5.5)

    label(ax, 0.082, 0.526, r"Velocity $v_k$", 9.0, weight="bold")
    tile(fig, [0.027, 0.425, 0.110, 0.080], a["v_0"] / 1000, "turbo", VELOCITY)

    box(ax, 0.149, 0.468, 0.095, 0.168, "#FFFFFF", "#9AAEBF")
    label(ax, 0.1965, 0.612, "Acoustic FDTD", 8.8, weight="bold")
    wave = fig.add_axes([0.156, 0.495, 0.081, 0.095], zorder=2)
    t_ms = np.arange(140) * dt * 1000
    wave.axhline(0, color="#ADC1CF", lw=0.6)
    wave.plot(t_ms, a["wavelet"][:140], color="#20587C", lw=1.4)
    wave.set(xlim=(t_ms[0], t_ms[-1]), ylim=(-0.58, 1.08))
    wave.axis("off")
    arrow(ax, [(0.138, 0.465), (0.144, 0.465), (0.144, 0.492), (0.149, 0.492)], head=5.5)
    arrow(ax, [(0.149, 0.593), (0.137, 0.593)], head=5.5)


def cuboid(ax, x, cy, width, height, depth, color):
    slope = np.tan(np.pi / 6) * ax.figure.get_figwidth() / ax.figure.get_figheight()
    dy = depth * slope
    drop = width * slope
    y = cy - height / 2 + drop / 2
    left = (x, y)
    front = (x + width, y - drop)
    right = (front[0] + depth, front[1] + dy)

    def above(point):
        return (point[0], point[1] + height)

    rgb = np.array(to_rgb(color))
    faces = (
        ([left, front, above(front), above(left)], rgb),
        ([above(left), (x + depth, y + height + dy), above(right), above(front)],
         rgb + (1 - rgb) * 0.46),
        ([front, right, above(right), above(front)], rgb * 0.70),
    )
    for vertices, fill in faces:
        ax.add_patch(Polygon(vertices, facecolor=fill, edgecolor=rgb * 0.50,
                             lw=0.62, joinstyle="round", zorder=4))
    for fraction in (0.33, 0.67):
        bottom = (front[0] + depth * fraction, front[1] + dy * fraction)
        line(ax, [bottom, above(bottom)], rgb * 0.52, 0.35, zorder=5)


def draw_network(fig, ax, a, unc_norm):
    top_card(ax, 0.285, 0.430, r"Multi-Modal Network $G_{\psi}$", AMBER_BG, "#E6A832")

    for points in (
        [(0.355, 0.808), (0.414, 0.848), (0.495, 0.849), (0.530, 0.705)],
        [(0.354, 0.498), (0.410, 0.415), (0.574, 0.414), (0.571, 0.592)],
    ):
        arrow(ax, points, "#718F9A", 0.8, head=5, curve=True, linestyle=(0, (3.2, 2.0)))

    for cy, name, color in (
        (0.772, "Model Enc.", "#6599BF"),
        (0.646, "Shot Enc.", "#8296C7"),
        (0.520, "Aux. Enc.", "#60A69D"),
    ):
        label(ax, 0.298, cy + 0.068, name, 8.6, weight="bold", ha="left")
        for x, w, h, d in ((0.303, 0.012, 0.072, 0.012),
                           (0.333, 0.016, 0.055, 0.010),
                           (0.365, 0.020, 0.039, 0.008)):
            cuboid(ax, x, cy, w, h, d, color)
        line(ax, [(0.326, cy), (0.365, cy)], "#8298A4", 0.75, zorder=2)
        target_y = 0.646 + (cy - 0.646) * 0.16
        arrow(ax, [(0.395, cy), (0.429, cy), (0.426, target_y), (0.432, target_y)],
              "#657B86", 0.95, curve=True)

    cuboid(ax, 0.434, 0.646, 0.031, 0.080, 0.018, "#E5A549")
    label(ax, 0.456, 0.762, "SE + FiLM", 9.2, color="#815511", weight="bold")
    box(ax, 0.422, 0.522, 0.082, 0.044, "#FFF4D2", "#D5AE5B", radius=0.015)
    label(ax, 0.463, 0.544, r"Condition $q_k$", 8.5)
    arrow(ax, [(0.455, 0.568), (0.455, 0.598)], "#AD852F", 0.85, linestyle=":")

    label(ax, 0.550, 0.848, "Dual-Scale U-Net", 9.0, weight="bold")
    label(ax, 0.436, 0.858, "Skip connections", 8.0, color=MUTED)
    arrow(ax, [(0.485, 0.646), (0.514, 0.646)], AMBER)
    cuboid(ax, 0.517, 0.646, 0.022, 0.070, 0.010, "#D88E67")
    arrow(ax, [(0.550, 0.646), (0.559, 0.646)], AMBER, head=5)
    cuboid(ax, 0.560, 0.646, 0.017, 0.096, 0.012, "#CA7859")

    for cy, name, data, cmap, norm in (
        (0.748, r"Predicted Gradient $\hat{g}_k$", a["g_pred"], "RdBu_r", GRAD),
        (0.544, r"Uncertainty $\hat{\sigma}_k$", a["sigma"], "magma", unc_norm),
    ):
        label(ax, 0.654, cy + 0.072, name, 8.8, weight="bold")
        tile(fig, [0.605, cy - 0.045, 0.098, 0.090], data, cmap, norm)
        arrow(ax, [(0.594, 0.646), (0.594, cy), (0.603, cy)], "#687F84", 0.90)
    line(ax, [(0.589, 0.646), (0.594, 0.646)], "#687F84", 0.90)


def draw_verification(fig, ax, a):
    top_card(ax, 0.745, 0.240, "Safeguarded Update", TEAL_BG, "#0D9488")

    box(ax, 0.759, 0.780, 0.212, 0.068, "white", "#9DCAC2")
    label(ax, 0.865, 0.824, "Uncertainty-Gated Step", 9.0, weight="bold")
    label(ax, 0.865, 0.798, "Confidence-weighted update", 8.2, color=MUTED)

    arrow(ax, [(0.865, 0.779), (0.865, 0.748)], TEAL, head=6)

    box(ax, 0.759, 0.668, 0.212, 0.076, "white", "#9DCAC2")
    label(ax, 0.865, 0.718, "Out-of-Sample Check", 9.0, weight="bold")
    label(ax, 0.865, 0.688, r"Accept $\to$ Backtrack $\to$ Blend $\to$ Fallback",
          8.1, color=TEAL)

    arrow(ax, [(0.865, 0.667), (0.865, 0.630)], TEAL, head=6)

    label(ax, 0.865, 0.604, r"Updated Velocity $v_{k+1}$", 9.2, weight="bold")
    blend = 0.35 * a["v_true"] + 0.65 * a["v_0"]
    tile(fig, [0.768, 0.428, 0.194, 0.148], blend / 1000, "turbo", VELOCITY)


def draw_flow(ax):
    line(ax, [(0.273, 0.520), (0.273, 0.772)], SLATE, 1.0, zorder=3)
    line(ax, [(0.255, 0.646), (0.273, 0.646)], SLATE, 1.0, zorder=3)
    ax.plot(0.273, 0.646, "o", ms=2.5, color=SLATE, zorder=4)
    for cy in (0.772, 0.646, 0.520):
        arrow(ax, [(0.273, cy), (0.301, cy)], SLATE, 1.0, head=6.5, zorder=3)

    line(ax, [(0.705, 0.748), (0.728, 0.748)], TEAL, 1.15, zorder=3)
    arrow(ax, [(0.705, 0.544), (0.728, 0.544), (0.728, 0.814), (0.756, 0.814)],
          TEAL, 1.15, head=7.5, zorder=3)
    ax.plot(0.728, 0.748, "o", ms=2.5, color=TEAL, zorder=4)

    for points in (
        [(0.215, 0.405), (0.198, 0.365)], [(0.192, 0.350), (0.171, 0.308)],
        [(0.704, 0.405), (0.722, 0.382), (0.722, 0.365)],
        [(0.722, 0.350), (0.642, 0.308)], [(0.956, 0.405), (0.932, 0.308)],
    ):
        line(ax, points, "#8EA4A9", 0.75, linestyle=(0, (2.7, 2.4)), zorder=0)

    arrow(ax, [(0.865, 0.426), (0.865, 0.358), (0.082, 0.358), (0.082, 0.423)],
          TEAL, 1.65, head=8, zorder=3)
    label(ax, 0.474, 0.376, r"Next Iteration ($k \to k+1$)",
          9.0, color=TEAL, weight="bold",
          bbox=dict(facecolor="white", edgecolor="none", pad=2.2))


def detail_card(ax, x, title):
    width = 0.310
    box(ax, x, 0.020, width, 0.288, "white", "#B8C6CD", lw=0.8)
    label(ax, x + 0.012, 0.282, title, 9.6, weight="bold", ha="left")
    line(ax, [(x + 0.012, 0.260), (x + width - 0.012, 0.260)], RULE, 0.6)


def draw_map_details(fig, ax, a, grid, unc_norm):
    detail_card(ax, 0.015, "(b) Physics-Consistent Losses")
    detail_card(ax, 0.345, "(c) Uncertainty Calibration")

    dz, dx = grid["dz"] / 1000, grid["dx"] / 1000
    nz, nx = a["g_sp"].shape
    extent = (-dx / 2, (nx - 0.5) * dx, (nz - 0.5) * dz, -dz / 2)
    groups = (
        (0.015, (r"Sparse $g_{\rm sp}$", r"Predicted $\hat{g}_k$", r"Reference $g_{\rm ref}$"),
         (a["g_sp"], a["g_pred"], a["g_ref"]), "RdBu_r", GRAD),
        (0.345, (r"Uncertainty $\hat{\sigma}_k$", r"Error $|\hat{g}_k - g_{\rm ref}|$"),
         (a["sigma"], np.abs(a["g_pred"] - a["g_ref"])), "magma", unc_norm),
    )
    for left, names, fields, cmap, norm in groups:
        gap = 0.009
        width = (0.288 - (len(names) - 1) * gap) / len(names)
        for i, (name, field) in enumerate(zip(names, fields)):
            x = left + 0.011 + i * (width + gap)
            label(ax, x + width / 2, 0.232, name, 8.8, weight="bold")
            panel = tile(fig, [x, 0.048, width, 0.160], field, cmap, norm, extent=extent)
            panel.set(xlim=(0.80, 1.70), ylim=(0.45, 0))
            if field is a["g_sp"]:
                ring = Ellipse((1.25, 0.145), 0.142, 0.128, fill=False,
                               edgecolor="#FFE361", lw=1.1, ls=(0, (2.0, 1.3)))
                ring.set_path_effects([patheffects.Stroke(linewidth=2.1, foreground="#292521"),
                                       patheffects.Normal()])
                panel.add_patch(ring)


def draw_descent(fig, ax, case, n_iters):
    loop = case["closed_loop"]
    baseline = loop["initial_full_loss"]
    histories = {name: np.asarray(loop[name]["full_loss_history"]) / baseline for name in METHODS}
    detail_card(ax, 0.675, "(d) Closed-Loop Convergence")

    k = np.arange(n_iters + 1)
    chart = fig.add_axes([0.722, 0.078, 0.242, 0.158], zorder=2)
    chart.spines[["top", "right"]].set_visible(False)
    for side in ("bottom", "left"):
        chart.spines[side].set(color="#7A8E99", linewidth=0.6)
    chart.tick_params(length=2.5, width=0.55, pad=2.5, labelsize=8.2, colors=MUTED)
    chart.grid(axis="y", color="#E2E8EC", lw=0.5)
    chart.set_axisbelow(True)

    for name, (legend, color, style, lw) in METHODS.items():
        chart.plot(k, histories[name], label=legend, color=color, ls=style, lw=lw)

    chart.set(xlim=(0, n_iters), ylim=(0, 1.03),
              xticks=[0, 5, 10, 15, 20, 25], yticks=[0, 0.5, 1.0])
    chart.set_xlabel(r"Iteration $k$", fontsize=8.5, color=INK, labelpad=2.0)
    chart.set_ylabel(r"Normalized $J_{\rm full}$", fontsize=8.5, color=INK, labelpad=2.5)
    chart.legend(loc="upper right", frameon=False, fontsize=8.2,
                 handlelength=1.8, handletextpad=0.5, labelspacing=0.28)


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.labelcolor": INK,
        "text.color": INK,
    })
    arrays, case, config = load_case()
    fig = plt.figure(figsize=(11.6, 6.2))
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set(xlim=(0, 1), ylim=(0, 1))
    canvas.set_axis_off()

    label(canvas, 0.015, 0.968, "(a) Closed-Loop Surrogate-Gradient FWI",
          10.8, weight="bold", ha="left")

    unc_norm = LogNorm(0.002, 0.7, clip=True)
    draw_physics(fig, canvas, arrays, config["physics"]["dt"])
    draw_network(fig, canvas, arrays, unc_norm)
    draw_verification(fig, canvas, arrays)
    draw_flow(canvas)
    draw_map_details(fig, canvas, arrays, config["grid"], unc_norm)
    draw_descent(fig, canvas, case, config["closed_loop"]["n_iters"])

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = OUTPUT / f"fig_framework_overview.{suffix}"
        fig.savefig(path, dpi=300)
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
