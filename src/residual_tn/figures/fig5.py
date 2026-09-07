"""Render all six panels only from completed, matching-model numerical data."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(root, name):
    return json.loads((root / name / "result.json").read_text())


def metrics(estimate, reference):
    values = np.abs(np.asarray(estimate) - np.asarray(reference)).ravel()
    assert np.isfinite(values).all()
    return dict(
        mae=float(values.mean()),
        p95=float(np.quantile(values, 0.95)),
        max=float(values.max()),
    )


def render(root):
    root = Path(root)
    out = root / "figures"
    out.mkdir(exist_ok=True)
    exact = read(root, "exact_T5")
    peps = read(root, "pauli")
    keys = (
        "manifest_sha256",
        "gate_table_sha256",
        "kraus_sha256",
        "noise_T1_T2_us",
        "layers",
        "width",
        "length",
        "kernel_scheme",
    )
    assert all(exact["identity"][k] == peps["identity"][k] for k in keys)
    assert peps["raw"]["site_indices"] == list(range(20)) and peps["raw"][
        "source_layers"
    ] == list(range(20))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), layout="constrained")
    colors = ["#2468A2", "#D88B20", "#42966D"]
    report = {}
    plots = [
        ("a", "Background Pauli", exact["ideal_pauli"], peps["ideal_pauli"]),
        (
            "b",
            "Source-layer conditional corrections",
            [r["delta_pauli"] for r in exact["per_source_layer"]],
            [r["delta_pauli"] for r in peps["per_source_layer"]],
        ),
        (
            "c",
            "All one-location insertions",
            exact["additive_first_order_pauli"],
            peps["additive_first_order_pauli"],
        ),
    ]
    for ax, (letter, title, reference, estimate) in zip(axes[0], plots):
        reference = np.asarray(reference)
        estimate = np.asarray(estimate)
        assert reference.shape == estimate.shape
        low = min(reference.min(), estimate.min())
        high = max(reference.max(), estimate.max())
        margin = 0.07 * max(high - low, 1e-6)
        limits = (low - margin, high + margin)
        ax.plot(limits, limits, "--", color="#8C9299", lw=0.8)
        for k, color in enumerate(colors):
            ax.scatter(
                reference[..., k].ravel(),
                estimate[..., k].ravel(),
                s=7 if letter == "b" else 20,
                color=color,
                alpha=0.65,
                label="XYZ"[k],
                linewidths=0.2,
                edgecolors="white",
            )
        ax.set(
            xlim=limits,
            ylim=limits,
            xlabel="Exact full state",
            ylabel="PEPS/BP",
            title=title,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.legend(frameon=False, ncol=3, loc="upper left")
        report[letter] = metrics(estimate, reference)
        m = report[letter]
        ax.text(
            0.03,
            0.04,
            f"MAE / P95 / max\n{m['mae']:.2e} / {m['p95']:.2e} / {m['max']:.2e}",
            transform=ax.transAxes,
            fontsize=7,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
        )
    reference = read(root, "exact_source0")
    pair_keys = sorted((r["source_gate"], r["target_gate"]) for r in reference["pairs"])

    def values(result):
        rows = {
            (r["source_gate"], r["target_gate"]): r["delta"] for r in result["pairs"]
        }
        assert sorted(rows) == pair_keys and len(rows) == len(result["pairs"])
        return np.array([rows[k] for k in pair_keys])

    reference_values = values(reference)
    names = [
        "baseline",
        "compression1e-3",
        "compression1e-5",
        "bp250",
        "iter_baseline",
        "bp1000",
        "bptol1e-8",
        "tol_baseline",
        "bptol1e-12",
    ]
    controls = {name: read(root, "control_" + name) for name in names}
    baseline = values(controls["baseline"])
    control_report = {}
    shared = {
        group: controls["tol_baseline" if group == "e" else "iter_baseline"][
            "identity"
        ]["shared_background"]
        for group in "ef"
    }
    for name, value in controls.items():
        identity = value["identity"]
        assert identity["requested_source_layers"] == [0]
        assert identity["numerics"]["local_readout_method"] == "gloop"
        assert (
            identity["target_sample_sha256"]
            == reference["identity"]["target_sample_sha256"]
        )
        assert all(identity[k] == reference["identity"][k] for k in keys)
        if name not in ("baseline", "compression1e-3", "compression1e-5"):
            assert (
                value["one_location_fidelity"]
                == controls["baseline"]["one_location_fidelity"]
            ), "e/f normalizers must be fixed"
            group = (
                "e"
                if name.startswith("bptol") or name == "tol_baseline"
                else ("f" if name.startswith("bp") or name == "iter_baseline" else "d")
            )
            assert identity["shared_background"]["group"] == group
            assert (
                identity["shared_background"]["cache_sha256"]
                == shared[group]["cache_sha256"]
            )
        estimate = values(value)
        source = json.loads((root / ("control_" + name) / "source_00.json").read_text())
        residuals = [x for row in source["bp_records"] for x in row["residuals"]]
        iterations = [x for row in source["bp_records"] for x in row["iterations"]]
        control_report[name] = dict(
            vs_matched_full_state=metrics(estimate, reference_values),
            vs_baseline=metrics(
                estimate,
                values(
                    controls[
                        "tol_baseline"
                        if name.startswith("bptol") or name == "tol_baseline"
                        else (
                            "iter_baseline"
                            if name.startswith("bp") or name == "iter_baseline"
                            else "baseline"
                        )
                    ]
                ),
            ),
            signed_sample_sum_error=float((estimate - reference_values).sum()),
            absolute_sample_error_sum=float(np.abs(estimate - reference_values).sum()),
            mixed_bp_max_residual=max(residuals),
            mixed_bp_max_iterations=max(iterations),
            mixed_bp_fraction_within_tol=float(
                np.mean(np.array(residuals) <= identity["numerics"]["bp_tol"])
            ),
            pair_count=len(pair_keys),
            identity=identity,
        )
    series = [
        ("mae", "Mean", "#2468A2", "o"),
        ("p95", "95th percentile", "#D88B20", "s"),
        ("max", "Maximum", "#BB4B4B", "^"),
    ]
    groups = [
        (
            ["compression1e-3", "baseline", "compression1e-5"],
            [r"$10^{-3}$", r"$10^{-4}$", r"$10^{-5}$"],
            "Dynamic BP compression tolerance",
            "Fidelity pairs: compression",
        ),
        (
            ["bptol1e-8", "tol_baseline", "bptol1e-12"],
            [r"$10^{-8}$", r"$10^{-10}$", r"$10^{-12}$"],
            "Readout BP convergence tolerance",
            "Fidelity pairs: BP tolerance",
        ),
        (
            ["bp250", "iter_baseline", "bp1000"],
            ["250", "500", "1000"],
            "Readout BP iteration cap",
            "Fidelity pairs: iteration cap",
        ),
    ]
    for ax, (order, labels, xlabel, title) in zip(axes[1], groups):
        x = np.arange(len(order))
        for key, label, color, marker in series:
            y = [control_report[name]["vs_matched_full_state"][key] for name in order]
            ax.plot(x, y, marker + "-", color=color, lw=1, ms=4, label=label)
        ax.set_xticks(x, labels)
        ax.set(
            xlabel=xlabel,
            ylabel=r"Absolute error in sampled $\delta_{ab}$",
            title=title,
        )
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        ax.grid(axis="y", alpha=0.2)
    axes[1, 0].legend(frameon=False, fontsize=7)
    for letter, ax in zip("abcdef", axes.flat):
        ax.text(
            -0.12, 1.05, letter, transform=ax.transAxes, fontweight="bold", fontsize=13
        )
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "5 × 4 qubits · 20 layers · 4 sparse diagonal residual couplings", fontsize=12
    )
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(out / f"Fig5.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)
    report.update(
        source0_controls=control_report,
        shared_background=shared,
        reference_identity=reference["identity"],
        identity=peps["identity"],
        pauli_c_definition="conditional ratio after additive one-location probability assembly",
        control_definition="Source-0 sampled fidelity pair delta_ab=M_ab/(M_a*M_b)-1; fixed matched Kraus cutoff 1e-8",
        sample_pair_keys=pair_keys,
        target_sample=reference["identity"]["target_sample"],
        scope="a-c: all 20 sources; d-f: sampled targets from source 0; no complete C2 or fidelity curve is inferred",
    )
    (out / "Fig5_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    caption = """Figure 5. Full-state benchmark of Pauli readout and sampled fidelity-pair numerical controls.
All branches use the same 5×4, 20-layer circuit and four endpoint-disjoint diagonal residual couplings.
(a–c) All-source Pauli comparisons: background components,
source-layer conditional corrections, and additive one-location probability assembly.
(d–f) Mean, 95th-percentile and maximum absolute errors of the dimensionless fidelity connected pair
quantity delta_ab=M_ab/(M_a M_b)-1, relative to independent full-state calculations of exactly the same pairs.
Every source-0 gate and retained Kraus mode is included. Three fixed target gates per layer are sampled
with seed 20260906, stratifying residual-endpoint and other supports where available; only b>a is retained.
The target sample does not depend on measured signals or errors. It is shared by every control.
In panel d, each compression tolerance applies to both background and source-branch construction. Panels e/f instead fix their
background caches, clean BP messages, Kraus amplitudes and normalizers M_a. Their baseline history was built with
Dmax=64, compression tolerance 1e-4, BP tolerance 1e-10 and BP cap 500.
(d) Full-pipeline dynamic BP-weighted compression tolerances 1e-3, 1e-4 and 1e-5.
(e) Readout BP tolerances 1e-8, 1e-10 and 1e-12 at iteration cap 500.
(f) Readout BP iteration caps 250, 500 and 1000 at tolerance 1e-10.
The gloop size remains 8 for both one-insertion normalization and two-insertion reads. Unless varied: Dmax=64, dynamic tolerance=1e-4, BP tolerance=1e-10,
BP cap=500, gloop size=8. Full-state and PEPS use identical Kraus cutoff 1e-8 and noise T1=T2=5 us.
In (e,f), source evolution and norm-BP
compression settings stay fixed; only mixed-BP readout stopping criteria vary. One common source
evolution and one BP sweep trajectory produce all e/f snapshots. Each branch is saved at its first
converged check or its iteration cap, whichever comes first. Identical snapshots reuse their gloop reads.
Panels d–f test the sampled fidelity-pair readout; no terminal Pauli components are recomputed.
Panels e/f do not test background-construction convergence. No panel provides a complete source-0 C2 sum, full all-source fidelity, or a large-system error bound.
The three curves are distinct error-distribution statistics, not three physical noise strengths.
The fidelity-pair target is the normalized residual-dressed coherent background, not the intended control-only state.
"""
    (out / "Fig5_caption.txt").write_text(caption)
    print("FIGURE_COMPLETE=" + str(out / "Fig5.pdf"), flush=True)
