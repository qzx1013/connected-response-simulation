"""Build the review-oriented six-figure suite from the frozen package data.

This script only reads data/assets inside its containing figure package and
writes ``figures/final_six_v2``.  It never reads an older package copy.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from pathlib import Path

# Keep the rendering dependencies local to the 20260721 package workflow.
# The default desktop Python does not ship Matplotlib, while this directory is
# explicitly disposable and does not affect either figure source package.
import numpy as np
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.text import Text


from residual_tn.paths import RESULTS

DATA = RESULTS
OUT = RESULTS / "figures"

COL = {
    "ink": "#20252b",
    "muted": "#626a73",
    "grid": "#d7dce2",
    "blue": "#2f6fae",
    "purple": "#7b5bb6",
    "green": "#4c9b75",
    "orange": "#e18e3e",
    "red": "#bd3b2c",
    "inside": "#4c9b75",
    "outside": "#d76b4a",
}

SOFT_DELTA_CMAP = LinearSegmentedColormap.from_list(
    "soft_delta",
    ("#fffdf9", "#fee8c8", "#fdbb84", "#ed8a55", "#c95f48"),
)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.titlesize": 8.3,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finish(ax, *, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", color=COL["grid"], linewidth=0.55, alpha=0.75)
        ax.set_axisbelow(True)


def label(ax, value: str):
    ax.text(
        -0.10,
        1.05,
        value,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def save(fig, stem: str):
    OUT.mkdir(parents=True, exist_ok=True)
    if stem in {"Fig2", "Fig3", "Fig4", "Fig5", "Fig6"}:
        # Keep the established panel geometry while making every visible text
        # element--including inset labels and explicit annotations--slightly
        # larger.  Scaling here also covers text with a locally supplied
        # fontsize that would not respond to the global rcParams.
        for text_artist in fig.findobj(match=Text):
            text_artist.set_fontsize(text_artist.get_fontsize() * 1.15)
    fig.savefig(OUT / f"{stem}.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def centered(values):
    values = np.sort(np.asarray(values, dtype=float))[::-1]
    positions = [0]
    radius = 1
    while len(positions) < len(values):
        positions.append(-radius)
        if len(positions) < len(values):
            positions.append(radius)
        radius += 1
    out = np.empty_like(values)
    out[np.asarray(positions)] = values
    return out


def plot_fig2():
    payload = read_json(DATA / "fig2_liouville" / "fig2_error_statistics.json")
    runs = payload["runs"]
    singletons = {float(x["noise_time"]): x for x in payload["singleton_trajectories"]}

    fig = plt.figure(figsize=(11.6, 7.15), facecolor="white")
    gs = fig.add_gridspec(
        2,
        6,
        left=0.055,
        right=0.988,
        bottom=0.075,
        top=0.93,
        wspace=0.62,
        hspace=0.52,
    )
    a = gs[0, :2].subgridspec(2, 1, height_ratios=[2.45, 1.0], hspace=0.13)
    ax_af = fig.add_subplot(a[0, 0])
    ax_ae = fig.add_subplot(a[1, 0], sharex=ax_af)
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[0, 4:])
    ax_d = fig.add_subplot(gs[1, :3])
    ax_e = fig.add_subplot(gs[1, 3:])

    # Panel a contains a second, compact error strip below the fidelity plot.
    # Panels b/c must occupy the full height of panel a, including its compact
    # error strip.  Matching only the upper fidelity axis left a large blank
    # area below b/c in the earlier v2 render.
    main_a_pos = ax_af.get_position()
    error_a_pos = ax_ae.get_position()
    for ax in (ax_b, ax_c):
        pos = ax.get_position()
        ax.set_position(
            [
                pos.x0,
                error_a_pos.y0,
                pos.width,
                main_a_pos.y1 - error_a_pos.y0,
            ]
        )

    label(ax_af, "a")
    label(ax_b, "b")
    label(ax_c, "c")
    label(ax_d, "d")
    label(ax_e, "e")

    colors = [COL["blue"], COL["green"], COL["purple"]]
    noise = [1000.0, 3000.0, 5000.0]
    all_f, all_err = [], []
    for color, t in zip(colors, noise):
        row = singletons[t]
        exact = np.asarray(row["F_liouville_trajectory"], dtype=float)
        f2 = np.asarray(row["F2_full_state_trajectory"], dtype=float)
        layers = np.arange(len(exact))
        err = np.maximum(np.abs(f2 - exact), 1e-7)
        ax_af.plot(layers, f2, color=color, lw=1.55, zorder=2)
        ax_af.plot(layers, exact, color=COL["red"], lw=1.35, ls=(0, (4, 2)), zorder=4)
        ax_ae.fill_between(layers, 1e-7, err, color=color, alpha=0.13, linewidth=0)
        ax_ae.plot(layers, err, color=color, lw=1.15)
        all_f.extend([exact, f2])
        all_err.append(err)
    ax_af.set_xlim(0, 20)
    ax_af.set_ylim(max(0.0, min(x.min() for x in all_f) - 0.04), 1.02)
    ax_af.set_ylabel("fidelity")
    ax_af.set_title("one circuit: fidelity evolution", pad=7)
    ax_af.tick_params(axis="x", labelbottom=False)
    ax_af.legend(
        [
            Line2D([0], [0], color=COL["red"], ls=(0, (4, 2)), lw=1.35),
            Line2D([0], [0], color=COL["blue"], lw=1.55),
            Line2D([0], [0], color=COL["green"], lw=1.55),
            Line2D([0], [0], color=COL["purple"], lw=1.55),
        ],
        [
            "exact qutrit Liouville",
            r"full-state $F_2$ (1 us)",
            r"full-state $F_2$ (3 us)",
            r"full-state $F_2$ (5 us)",
        ],
        frameon=False,
        loc="lower left",
        fontsize=5.2,
        handlelength=1.6,
    )
    ax_ae.set_yscale("log")
    ax_ae.set_ylim(1e-7, max(x.max() for x in all_err) * 1.35)
    ax_ae.set_xlabel("layer")
    ax_ae.set_ylabel(r"$|F_2-F_{\rm exact}|$")
    ax_ae.set_xticks([0, 5, 10, 15, 20])
    finish(ax_af, grid=False)
    finish(ax_ae, grid=False)

    groups = {
        "1 us": [
            r
            for r in runs
            if int(r["layers"]) == 20 and math.isclose(float(r["noise_time"]), 1000.0)
        ],
        "3 us": [
            r
            for r in runs
            if int(r["layers"]) == 20 and math.isclose(float(r["noise_time"]), 3000.0)
        ],
        "5 us": [
            r
            for r in runs
            if int(r["layers"]) == 20 and math.isclose(float(r["noise_time"]), 5000.0)
        ],
        "5 us, 40 layers": [
            r
            for r in runs
            if int(r["layers"]) == 40 and math.isclose(float(r["noise_time"]), 5000.0)
        ],
    }

    def ranked(rows):
        vals = [
            abs(float(r["error_F2_minus_Liouville"])) / float(r["F_liouville"]) * 100
            for r in rows
        ]
        return centered(vals)

    def draw_stats(ax, names, title):
        cols = (
            [COL["blue"], COL["green"], COL["purple"]]
            if len(names) == 3
            else [COL["purple"], COL["orange"]]
        )
        max_y = 0.0
        for name, color in zip(names, cols):
            values = ranked(groups[name])
            x = np.arange(len(values))
            ax.bar(
                x,
                values,
                width=0.80,
                color=color,
                alpha=0.50,
                edgecolor=color,
                linewidth=0.5,
                label=f"{name} (max {values.max():.2f}%)",
            )
            max_y = max(max_y, values.max())
        ax.set_xlim(-1.0, 20.0)
        ax.set_ylim(0, max_y * 1.22)
        ax.set_xticks([])
        ax.set_xlabel("20 circuit samples, ranked by final-layer error", labelpad=7)
        # The full ratio is defined in the caption; the compact label keeps
        # the shortened b/c axes clear of the figure title.
        ax.set_ylabel("relative error (%)")
        ax.set_title(title, pad=8)
        ax.legend(frameon=False, loc="upper right", fontsize=5.4, handlelength=1.2)
        finish(ax)

    draw_stats(ax_b, ["1 us", "3 us", "5 us"], "20 circuits, 20 layers")
    draw_stats(ax_c, ["5 us", "5 us, 40 layers"], "20 circuits, same 5 us noise")

    pauli = read_json(DATA / "pauli_observables" / "pauli_8q_exact_vs_fullstate.json")
    exact = np.asarray(pauli["exact_qutrit"]["pauli_conditional"], dtype=float)
    f1 = np.asarray(
        pauli["full_state"]["first_order"]["pauli_conditional"], dtype=float
    )
    f2 = np.asarray(
        pauli["full_state"]["second_order"]["pauli_conditional"], dtype=float
    )
    markers = {0: "o", 1: "^", 2: "s"}
    basis_names = ["X", "Y", "Z"]
    for basis in range(3):
        # Put the exact value on the horizontal axis so the diagonal is a
        # meaningful one-to-one reference, rather than a categorical index.
        x = exact[:, basis]
        ax_d.plot(
            x,
            f1[:, basis],
            ls="None",
            marker=markers[basis],
            ms=4.0,
            markerfacecolor="white",
            markeredgecolor=COL["orange"],
            markeredgewidth=1.0,
        )
        ax_d.plot(
            x,
            f2[:, basis],
            ls="None",
            marker=markers[basis],
            ms=4.0,
            markerfacecolor=COL["blue"],
            markeredgecolor="white",
            markeredgewidth=0.4,
        )
    lo = min(exact.min(), f1.min(), f2.min()) - 0.04
    hi = max(exact.max(), f1.max(), f2.max()) + 0.04
    ax_d.plot([lo, hi], [lo, hi], color=COL["muted"], lw=0.8, zorder=0)
    ax_d.set_xlim(lo, hi)
    ax_d.set_ylim(lo, hi)
    ax_d.set_ylabel("full-state conditional Pauli expectation")
    ax_d.set_xlabel("exact qutrit-Liouville reference")
    ax_d.set_title("local Pauli parity: one reference per observable", pad=8)
    ax_d.legend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="white",
                markeredgecolor=COL["orange"],
                markersize=5,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=COL["blue"],
                markeredgecolor="white",
                markersize=5,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="none",
                markeredgecolor=COL["muted"],
                markersize=5,
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color="none",
                markerfacecolor="none",
                markeredgecolor=COL["muted"],
                markersize=5,
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor="none",
                markeredgecolor=COL["muted"],
                markersize=5,
            ),
            Line2D([0], [0], color=COL["muted"], lw=0.8),
        ],
        ["first order", "connected C2", "X", "Y", "Z", "$y=x$"],
        frameon=False,
        loc="upper left",
        fontsize=5.6,
    )
    finish(ax_d, grid=False)

    err1 = np.abs(f1 - exact) * 1e3
    err2 = np.abs(f2 - exact) * 1e3
    positions = np.arange(24)
    # Group the 24 bars as X(8), Y(8), Z(8), matching the point ordering above.
    ax_e.bar(
        positions - 0.20,
        err1.T.reshape(-1),
        width=0.38,
        color=COL["orange"],
        alpha=0.58,
        label="first order",
    )
    ax_e.bar(
        positions + 0.20,
        err2.T.reshape(-1),
        width=0.38,
        color=COL["blue"],
        alpha=0.78,
        label="connected C2",
    )
    ax_e.set_xticks([3.5, 11.5, 19.5], ["X", "Y", "Z"])
    ax_e.set_xticks(
        np.arange(24), [str(i + 1) for _ in range(3) for i in range(8)], minor=True
    )
    ax_e.tick_params(axis="x", which="minor", length=2, labelsize=5)
    ax_e.set_ylabel(r"absolute error ($\times10^{-3}$)")
    ax_e.set_xlabel("site within basis")
    ax_e.set_title("absolute errors by basis", pad=8)
    ax_e.legend(frameon=False, loc="upper right", fontsize=5.7)
    finish(ax_e)
    save(fig, "Fig2")


def plot_fig3():
    ensemble = read_json(
        DATA / "connected_ensemble_20260716" / "analysis_summary.json"
    )["circuits"]
    light = read_csv(DATA / "connected_ensemble_20260716" / "lightcone_20circuits.csv")
    layer_rows = read_csv(
        DATA / "connected_ensemble_20260716" / "sample00_layer_pair_c2.csv"
    )
    fig, axs = plt.subplots(2, 3, figsize=(11.6, 7.0), facecolor="white")
    for ax, value in zip(axs.flat, "abcdef"):
        label(ax, value)

    # Keep the three top-row plotting boxes identical even after the figure
    # allocates space for the two lower heat-map colorbars.
    top_pos = axs[0, 0].get_position()
    for ax in (axs[0, 1], axs[0, 2]):
        pos = ax.get_position()
        ax.set_position([pos.x0, top_pos.y0, pos.width, top_pos.height])

    complete = [r for r in ensemble if r.get("c3_complete", False)]
    c1 = centered([r["C1_magnitude"] for r in complete])
    c2 = centered([r["C2_magnitude"] for r in complete])
    c3 = centered([r["C3_distinct_magnitude"] for r in complete])
    x = np.arange(len(complete))
    ax = axs[0, 0]
    floor = max(min(c3[c3 > 0]) / 2.5, 1e-8)
    for vals, width, color, alpha, text in [
        (c1, 0.88, COL["purple"], 0.33, r"$|C_1|$"),
        (c2, 0.69, COL["blue"], 0.55, r"$|C_2|$"),
        (c3, 0.50, COL["orange"], 0.82, r"$|C_3^{\rm distinct}|$"),
    ]:
        ax.bar(
            x,
            np.maximum(vals - floor, 0),
            bottom=floor,
            width=width,
            color=color,
            alpha=alpha,
            label=text,
        )
    ax.set_yscale("log")
    ax.set_ylim(floor, c1.max() * 1.8)
    ax.set_xticks([])
    ax.set_xlabel("20 random-circuit samples, ranked by contribution")
    ax.set_ylabel("magnitude of log-fidelity contribution")
    ax.set_title("Connected-order hierarchy", pad=7)
    ax.legend(
        frameon=True,
        facecolor="white",
        framealpha=0.92,
        edgecolor="none",
        fontsize=5.6,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=3,
        borderaxespad=0.0,
        handlelength=1.3,
        columnspacing=0.8,
    )
    finish(ax)

    ax = axs[0, 1]
    r21 = centered([100 * r["C2_over_C1_magnitude"] for r in complete])
    r32 = centered([100 * r["C3_over_C2_magnitude"] for r in complete])
    ax.bar(x, r21, width=0.82, color=COL["blue"], alpha=0.48, label=r"$|C_2/C_1|$")
    ax.bar(
        x,
        r32,
        width=0.58,
        color=COL["orange"],
        alpha=0.72,
        label=r"$|C_3^{\rm distinct}/C_2|$",
    )
    ax.set_yscale("log")
    ax.set_xticks([])
    ax.set_xlabel("20 random-circuit samples, ranked by ratio")
    ax.set_ylabel("successive correction (%)")
    ax.set_title("Successive-order suppression", pad=7)
    ax.legend(
        frameon=True,
        facecolor="white",
        framealpha=0.92,
        edgecolor="none",
        fontsize=5.6,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=2,
        borderaxespad=0.0,
        handlelength=1.3,
        columnspacing=0.8,
    )
    finish(ax)

    arr = sorted(
        light, key=lambda row: float(row["outside_abs_fraction"]), reverse=True
    )
    ax = axs[0, 2]
    inside = np.array([100 * float(r["inside_abs_fraction"]) for r in arr])
    outside = np.array([100 * float(r["outside_abs_fraction"]) for r in arr])
    ax.bar(
        x,
        inside,
        width=0.86,
        color=COL["inside"],
        alpha=0.72,
        label="inside forward cone",
    )
    ax.bar(
        x,
        outside,
        bottom=inside,
        width=0.86,
        color=COL["outside"],
        alpha=0.76,
        label="outside forward cone",
    )
    ax.set_ylim(0, 103)
    ax.set_xticks([])
    ax.set_xlabel("20 random-circuit samples, ranked by outside mass")
    ax.set_ylabel(r"fraction of $\sum|\kappa^{(2)}|$ (%)")
    ax.set_title("Strict light-cone partition", pad=7)
    ax.legend(
        frameon=True,
        facecolor="white",
        framealpha=0.92,
        edgecolor="none",
        fontsize=5.6,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=2,
        borderaxespad=0.0,
        handlelength=1.3,
        columnspacing=0.8,
    )
    finish(ax, grid=False)

    ax = axs[1, 0]
    radii = np.arange(4)
    tails = np.array(
        [[100 * float(r[f"tail_abs_fraction_R{k}"]) for k in radii] for r in light]
    )
    for row in tails:
        ax.plot(
            radii,
            np.maximum(row, 1e-6),
            color=COL["muted"],
            alpha=0.18,
            lw=0.65,
            marker="o",
            ms=1.8,
        )
    ax.plot(
        radii,
        np.maximum(np.median(tails, axis=0), 1e-6),
        color=COL["outside"],
        lw=1.8,
        marker="o",
        ms=3.5,
        label="median",
    )
    ax.set_yscale("log")
    ax.set_xlabel("support-gap buffer $R$")
    ax.set_ylabel(r"outside-cone $\sum|\kappa^{(2)}|$ (%)")
    ax.set_title("Buffer-tail decay across 20 circuits", pad=7)
    ax.legend(frameon=False, fontsize=5.8)
    finish(ax)

    matrix_signed = np.full((20, 20), np.nan)
    matrix_abs = np.full((20, 20), np.nan)
    for row in layer_rows:
        a, b = int(row["source_layer"]), int(row["target_layer"])
        matrix_signed[a, b] = abs(float(row["signed_kappa2"])) or np.nan
        matrix_abs[a, b] = abs(float(row["abs_kappa2"])) or np.nan
    for ax, matrix, title in [
        (axs[1, 1], matrix_signed, r"signed aggregation $|K_{ab}|$"),
        (axs[1, 2], matrix_abs, r"absolute event mass $A_{ab}$"),
    ]:
        positive = matrix[np.isfinite(matrix) & (matrix > 0)]
        image = ax.imshow(
            matrix,
            origin="lower",
            cmap="Blues",
            norm=LogNorm(
                vmin=max(float(np.percentile(positive, 2)), 1e-9),
                vmax=float(positive.max()),
            ),
        )
        ax.set_xticks([0, 5, 10, 15, 19])
        ax.set_yticks([0, 5, 10, 15, 19])
        ax.set_xlabel("target layer $b$")
        ax.set_ylabel("source layer $a$")
        ax.set_title(title, pad=7)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        finish(ax, grid=False)
    fig.subplots_adjust(
        left=0.055, right=0.985, bottom=0.075, top=0.95, wspace=0.40, hspace=0.48
    )
    save(fig, "Fig3")


def plot_fig4():
    p = read_json(
        DATA
        / "pauli_observables"
        / "mc_4x4_validation"
        / "qutrit_mc_4x4_vs_fullstate_f2.json"
    )
    z = np.load(
        DATA / "pauli_observables" / "mc_4x4_validation" / "qutrit_mc_4x4_samples.npz"
    )
    layer_samples = z["layer_fidelity_samples"]
    mc_mean = np.asarray(p["qutrit_monte_carlo"]["trajectory_mean"], dtype=float)
    mc_ci = np.asarray(
        p["qutrit_monte_carlo"]["trajectory_ci95_half_width"], dtype=float
    )
    f2_traj = np.asarray(p["full_state_f2"]["trajectory"], dtype=float)
    checkpoints = [
        int(x["trajectories"]) for x in p["qutrit_monte_carlo"]["checkpoints"]
    ]
    f2 = np.asarray(p["local_pauli"]["full_state_c2"]["conditional"], dtype=float)
    f1 = np.asarray(
        read_json(DATA / "pauli_observables" / "pauli_16q_fullstate.json")[
            "full_state"
        ]["first_order"]["pauli_conditional"],
        dtype=float,
    )
    mc_pauli = np.asarray(
        p["local_pauli"]["qutrit_mc"]["pauli_conditional_mean"], dtype=float
    )
    mc_se = np.asarray(
        p["local_pauli"]["qutrit_mc"]["pauli_conditional_ci95_half_width"], dtype=float
    )
    fig, axs = plt.subplots(2, 3, figsize=(11.6, 7.0), facecolor="white")
    for ax, value in zip(axs.flat, "abcdef"):
        label(ax, value)

    ax = axs[0, 0]
    layers = np.arange(len(f2_traj))
    ax.plot(layers, f2_traj, color=COL["purple"], lw=1.6, label=r"full-state $F_2$")
    ax.plot(
        layers,
        mc_mean,
        color=COL["blue"],
        lw=1.25,
        marker="o",
        ms=2.4,
        label="qutrit MC",
    )
    ax.fill_between(
        layers,
        mc_mean - mc_ci,
        mc_mean + mc_ci,
        color=COL["blue"],
        alpha=0.16,
        linewidth=0,
    )
    ax.set_xlabel("layer")
    ax.set_ylabel("fidelity")
    ax.set_title("Fidelity trajectory", pad=7)
    ax.legend(frameon=False, fontsize=5.8)
    finish(ax)

    ax = axs[0, 1]
    cp_mean = []
    cp_half = []
    for m in checkpoints:
        vals = layer_samples[:m, -1]
        cp_mean.append(float(vals.mean()))
        cp_half.append(float(1.96 * vals.std(ddof=1) / math.sqrt(m)))
    cp_mean, cp_half = np.asarray(cp_mean), np.asarray(cp_half)
    ax.axhline(f2_traj[-1], color=COL["purple"], lw=1.45, label=r"full-state $F_2$")
    ax.errorbar(
        checkpoints,
        cp_mean,
        yerr=cp_half,
        color=COL["blue"],
        marker="o",
        lw=1.25,
        capsize=2.5,
        label="MC mean +/- 95% CI",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(checkpoints, [str(x) for x in checkpoints])
    ax.set_xlabel("qutrit MC trajectories")
    ax.set_ylabel("final-layer fidelity")
    ax.set_title("Trajectory-number convergence", pad=7)
    ax.legend(frameon=False, fontsize=5.5)
    finish(ax)

    ax = axs[0, 2]
    markers = {0: "o", 1: "^", 2: "s"}
    basis_names = ["X", "Y", "Z"]
    for basis in range(3):
        x = mc_pauli[:, basis]
        ax.errorbar(
            x,
            f1[:, basis],
            xerr=mc_se[:, basis],
            fmt=markers[basis],
            ms=3.1,
            color=COL["orange"],
            markerfacecolor="white",
            markeredgewidth=0.85,
            ecolor=COL["muted"],
            elinewidth=0.45,
            capsize=1.2,
            alpha=0.82,
        )
        ax.errorbar(
            x,
            f2[:, basis],
            xerr=mc_se[:, basis],
            fmt=markers[basis],
            ms=3.1,
            color=COL["blue"],
            markerfacecolor=COL["blue"],
            markeredgecolor="white",
            markeredgewidth=0.35,
            ecolor=COL["muted"],
            elinewidth=0.45,
            capsize=1.2,
            alpha=0.82,
        )
    lo = min(mc_pauli.min(), f1.min(), f2.min()) - 0.035
    hi = max(mc_pauli.max(), f1.max(), f2.max()) + 0.035
    ax.plot([lo, hi], [lo, hi], color=COL["muted"], lw=0.75)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("MC conditional estimate (horizontal 95% CI)")
    ax.set_ylabel("full-state estimate")
    ax.set_title("Pauli parity across X/Y/Z", pad=7)
    ax.legend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="white",
                markeredgecolor=COL["orange"],
                markersize=4.5,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=COL["blue"],
                markeredgecolor="white",
                markersize=4.5,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="none",
                markeredgecolor=COL["muted"],
                markersize=4.5,
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color="none",
                markerfacecolor="none",
                markeredgecolor=COL["muted"],
                markersize=4.5,
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor="none",
                markeredgecolor=COL["muted"],
                markersize=4.5,
            ),
            Line2D([0], [0], color=COL["muted"], lw=0.75),
        ],
        ["first order", "connected C2", "X", "Y", "Z", "$y=x$"],
        frameon=False,
        fontsize=5.3,
        loc="upper left",
    )
    finish(ax, grid=False)

    ax = axs[1, 0]
    errors1 = np.abs(f1 - mc_pauli)
    errors2 = np.abs(f2 - mc_pauli)
    means1 = errors1.mean(axis=0)
    means2 = errors2.mean(axis=0)
    xx = np.arange(3)
    ax.bar(
        xx - 0.18,
        means1,
        width=0.34,
        color=COL["orange"],
        alpha=0.60,
        label="first order",
    )
    ax.bar(
        xx + 0.18,
        means2,
        width=0.34,
        color=COL["blue"],
        alpha=0.80,
        label="connected C2",
    )
    cover = np.abs(f2 - mc_pauli) <= mc_se
    counts = cover.sum(axis=0)
    ax.set_xticks(xx, [f"{b}\n{n}/16" for b, n in zip(basis_names, counts)])
    ax.set_xlabel("basis; C2 CI coverage shown below labels")
    ax.set_ylabel("MAE versus MC")
    ax.set_title("Basis-resolved error and CI coverage", pad=7)
    ax.legend(
        frameon=True,
        facecolor="white",
        framealpha=0.92,
        edgecolor="none",
        fontsize=5.6,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=2,
        borderaxespad=0.0,
        handlelength=1.3,
        columnspacing=0.9,
    )
    ax.text(
        0.02,
        0.97,
        "C2 pointwise 95% CI coverage",
        transform=ax.transAxes,
        va="top",
        fontsize=5.7,
        color=COL["muted"],
    )
    finish(ax)

    ax = axs[1, 1]
    # Read measured costs from the figure payload, including any audited retiming.
    runtime_mc = float(p["runtime_seconds"]["qutrit_monte_carlo"])
    runtime_f2 = float(p["runtime_seconds"]["full_state_f2"])
    runtime = runtime_mc * np.asarray(checkpoints) / checkpoints[-1]
    errors = np.abs(cp_mean - f2_traj[-1])
    ax.plot(
        runtime,
        np.maximum(errors, 1e-5),
        color=COL["blue"],
        marker="o",
        lw=1.1,
        label="MC prefix difference",
    )
    # Display the actual interval width instead of clipping an error bar
    # around an absolute difference to make it fit a logarithmic axis.
    ax.plot(
        runtime, cp_half, color=COL["orange"], ls="--", marker="s",
        markersize=3, lw=1.0, label="MC 95% CI half-width",
    )
    ax.scatter(
        [runtime_f2],
        [1e-5],
        color=COL["purple"],
        marker="D",
        s=22,
        label="deterministic full-state $F_2$",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("online time (s; MC prefixes scaled from measured run)")
    ax.set_ylabel(r"$|\hat F_{\rm MC}-F_2|$; MC 95% CI half-width")
    ax.set_title("Accuracy-cost trajectory", pad=7)
    # Keep the legend away from the Monte Carlo trajectory.  In the former
    # upper-right position, its diamond and circle sat next to the measured
    # markers and could be mistaken for two additional data points.
    ax.legend(
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        edgecolor=COL["grid"],
        fontsize=5.3,
        loc="lower right",
    )
    finish(ax)

    ax = axs[1, 2]
    vals = [runtime_f2, runtime_mc]
    bars = ax.bar([0, 1], vals, color=[COL["purple"], COL["orange"]], width=0.58)
    ax.set_yscale("log")
    f2_label = "optimized full-state $F_2$" if p.get("timing_provenance", {}).get("full_state_schedule") == "shared_prefix" else "full-state $F_2$"
    ax.set_xticks([0, 1], [f2_label, "qutrit MC\n4096 trajectories"])
    ax.set_ylabel("measured online wall time (s)")
    ax.set_title("Same-device runtime comparison", pad=7)
    for bar, value in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.12,
            f"{value:.2f} s",
            ha="center",
            va="bottom",
            fontsize=6,
        )
    ax.text(
        0.40,
        0.55,
        (rf"${runtime_mc / runtime_f2 / 1000:.2f}\times10^3$" if runtime_mc / runtime_f2 >= 1000 else f"{runtime_mc / runtime_f2:.0f}x"),
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=COL["ink"],
    )
    ax.text(
        0.40,
        0.45,
        "online-time ratio",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.2,
        color=COL["muted"],
    )
    finish(ax, grid=False)

    fig.subplots_adjust(
        left=0.055, right=0.985, bottom=0.075, top=0.96, wspace=0.40, hspace=0.50
    )
    save(fig, "Fig4")


# Values below are digitized from source_assets/natcomm_exact_peps_layer_fidelity.png.
# The frozen package contains the plotted curves but not their original arrays.
# They are sufficient for the requested re-statistics, but are not new simulations.
_TN_NOISE = (10.0, 5.0, 1.0)
_TN_EXACT_F2 = {
    10.0: [
        0.99505,
        0.97583,
        0.94859,
        0.92703,
        0.90141,
        0.87981,
        0.85270,
        0.83419,
        0.81288,
        0.79766,
        0.77939,
        0.76560,
        0.74743,
        0.73035,
        0.71081,
        0.69238,
        0.67137,
        0.65585,
        0.64079,
        0.62722,
    ],
    5.0: [
        0.98298,
        0.95570,
        0.91216,
        0.87215,
        0.82698,
        0.78837,
        0.74496,
        0.71521,
        0.68047,
        0.65611,
        0.62732,
        0.60436,
        0.57844,
        0.55079,
        0.52364,
        0.49746,
        0.47055,
        0.44881,
        0.42778,
        0.41108,
    ],
    1.0: [
        0.93218,
        0.81420,
        0.66540,
        0.53446,
        0.41729,
        0.32950,
        0.25716,
        0.21015,
        0.16769,
        0.14160,
        0.11435,
        0.09597,
        0.07767,
        0.06082,
        0.04736,
        0.03795,
        0.02897,
        0.02283,
        0.01832,
        0.01540,
    ],
}
_TN_CAVITY_F2 = {
    10.0: [
        0.99619,
        0.97636,
        0.94864,
        0.92846,
        0.90293,
        0.88133,
        0.85570,
        0.83718,
        0.81445,
        0.80065,
        0.78091,
        0.76716,
        0.74894,
        0.73047,
        0.71090,
        0.69247,
        0.67280,
        0.65589,
        0.64079,
        0.62590,
    ],
    5.0: [
        0.98450,
        0.95707,
        0.91222,
        0.87215,
        0.82844,
        0.78990,
        0.74785,
        0.71690,
        0.68199,
        0.65758,
        0.62872,
        0.60573,
        0.57834,
        0.55069,
        0.52217,
        0.49613,
        0.47032,
        0.44733,
        0.42627,
        0.40956,
    ],
    1.0: [
        0.93259,
        0.81427,
        0.66446,
        0.53300,
        0.41584,
        0.32789,
        0.25570,
        0.20723,
        0.16453,
        0.13709,
        0.11115,
        0.09142,
        0.07315,
        0.05774,
        0.04431,
        0.03497,
        0.02743,
        0.02134,
        0.01683,
        0.01364,
    ],
}
_TN_EXACT_LAYER_C2 = {
    10.0: [
        0.0,
        0.0001,
        -0.0032,
        0.0,
        0.00184,
        -0.00016,
        -0.00045,
        -0.00019,
        0.00185,
        0.00000,
        0.00307,
        0.00015,
        0.00581,
        -0.00008,
        0.00322,
        -0.00004,
        0.00264,
        0.00004,
        0.00609,
        0.00003,
    ],
    5.0: [
        0.0,
        -0.00011,
        -0.00319,
        0.00004,
        0.00207,
        0.00013,
        -0.00070,
        0.00013,
        0.00219,
        0.00047,
        0.00359,
        0.00087,
        0.00650,
        0.00054,
        0.00374,
        0.00043,
        0.00302,
        0.00073,
        0.00675,
        0.00084,
    ],
    1.0: [
        0.0,
        0.00057,
        -0.00194,
        0.00188,
        0.00495,
        0.00345,
        0.00260,
        0.00582,
        0.00827,
        0.00766,
        0.01366,
        0.01128,
        0.01720,
        0.00867,
        0.01283,
        0.00827,
        0.01086,
        0.00928,
        0.01553,
        0.00898,
    ],
}
_TN_CAVITY_LAYER_C2 = {
    10.0: [
        0.0,
        0.0001,
        -0.0035,
        0.0,
        0.0033,
        0.0,
        0.00128,
        0.00014,
        0.00248,
        -0.00004,
        0.00275,
        0.00005,
        0.00551,
        0.0,
        0.00423,
        0.0,
        0.00260,
        -0.00005,
        0.00676,
        -0.00006,
    ],
    5.0: [
        0.0,
        -0.00014,
        -0.00373,
        -0.00007,
        0.00324,
        0.00012,
        0.00106,
        -0.00002,
        0.00218,
        0.00028,
        0.00305,
        0.00055,
        0.00527,
        0.00000,
        0.00410,
        0.00002,
        0.00230,
        0.00031,
        0.00639,
        0.00036,
    ],
    1.0: [
        0.0,
        0.00001,
        -0.00332,
        0.00046,
        0.00378,
        0.00062,
        0.00025,
        -0.00057,
        0.00127,
        0.00106,
        0.00503,
        0.00247,
        0.00745,
        0.00082,
        0.00594,
        0.00060,
        0.00268,
        0.00135,
        0.00756,
        0.00139,
    ],
}
