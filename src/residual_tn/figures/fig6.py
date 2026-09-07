"""Compact Fig6, drawn only from a single completed sparse-residual run."""

from pathlib import Path
import json


def load_completed(directory):
    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text())
    identity, raw = result["identity"], result["raw"]
    if (identity["width"], identity["length"], identity["layers"]) != (6, 6, 20):
        raise ValueError("Fig6 requires the 6×6, 20-layer circuit")
    if len(identity["residual_edges"]) != 7:
        raise ValueError("Fig6 requires the approved seven-edge residual model")
    from residual_tn.geometry import validate_geometry

    validate_geometry(
        dict(
            width=6,
            length=6,
            n_qubits=36,
            n_residual_edges=7,
            edges=identity["residual_edges"],
        )
    )
    if result["definition"]["validation_subset"] or raw["source_layers"] != list(
        range(20)
    ):
        raise ValueError("Cannot render final Fig6 from incomplete source subsets")
    if raw["site_indices"] != list(range(36)):
        raise ValueError("All 108 terminal Pauli components are required")
    if len(raw["source_gate_ids"]) != sum(identity["gate_counts_by_layer"]) or len(
        set(raw["source_gate_ids"])
    ) != len(raw["source_gate_ids"]):
        raise ValueError("Missing or repeated source gate contributions")
    committed_ids = []
    for source in range(20):
        row = json.loads((directory / f"source_{source:02}.json").read_text())
        if row["identity"] != identity or row["source_layer"] != source:
            raise ValueError(
                "Source result belongs to another model or numerical configuration"
            )
        committed_ids.extend(row["source_gate_ids"])
    if committed_ids != raw["source_gate_ids"]:
        raise ValueError("Assembly and committed source gate lists differ")
    events = [
        json.loads(line)
        for line in (directory / "batch_resources.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return result, events


def resource_summary(events):
    # Evolution's aggregate already contains its norm-BP calls; don't count both.
    seconds = dict(background=0.0, evolution=0.0, readout=0.0)
    for row in events:
        event = row["event"]
        if event == "background_saved":
            seconds["background"] += row["seconds"]
        elif event == "evolution_layer":
            seconds["evolution"] += (
                row["gates_seconds"]
                + row["bp_seconds"]
                + row.get("projection_and_checkpoint_seconds", 0.0)
            )
        elif event == "norm_bp" and row.get("layer") == "terminal":
            seconds["evolution"] += row["seconds"]
        elif event in ("pauli_read", "background_pauli_read"):
            seconds["readout"] += row["seconds"]
    memory = [row for row in events if "peak_allocated_gib" in row]
    memory += [row["gates_peak"] for row in events if "gates_peak" in row]
    return dict(
        completed_phase_seconds=seconds,
        peak_allocated_gib=max(
            (row["peak_allocated_gib"] for row in memory), default=0.0
        ),
        peak_reserved_gib=max(
            (row["peak_reserved_gib"] for row in memory), default=0.0
        ),
        timing_scope="Recorded completed phase wall times, excluding model compilation, unrecorded interruption work and orchestration; not end-to-end or summed GPU kernel time",
        memory_scope="PyTorch process allocated/reserved peaks across recorded phases; driver/other-process memory is not included",
    )


def render(directory):
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result, events = load_completed(directory)
    q0 = np.asarray(result["ideal_pauli"], dtype=float).reshape(6, 6, 3)
    q1 = np.asarray(result["additive_first_order_pauli"], dtype=float).reshape(6, 6, 3)
    if not np.isfinite(q0).all() or not np.isfinite(q1).all():
        raise ValueError("Nonfinite Pauli result")
    delta = q1 - q0
    resources = resource_summary(events)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(6.8, 5.4), layout="constrained")
    grid = fig.add_gridspec(
        3, 4, width_ratios=(1, 1, 1, 0.07), height_ratios=(1, 1, 0.8)
    )
    for row, (values, label) in enumerate(
        (
            (q1, "First-order conditional Pauli"),
            (delta, "Change from coherent background"),
        )
    ):
        limit = max(1e-12, float(np.max(np.abs(values))))
        for basis, name in enumerate("XYZ"):
            ax = fig.add_subplot(grid[row, basis])
            im = ax.imshow(values[:, :, basis], cmap="RdBu_r", vmin=-limit, vmax=limit)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(name if row == 0 else "")
            if basis == 0:
                ax.set_ylabel(label)
                if row == 0:
                    ax.text(
                        -0.22,
                        1.14,
                        "a",
                        transform=ax.transAxes,
                        fontweight="bold",
                        fontsize=12,
                    )
        fig.colorbar(im, cax=fig.add_subplot(grid[row, 3]))
    lower = grid[2, :].subgridspec(1, 2, width_ratios=(1.5, 1))
    timing = fig.add_subplot(lower[0, 0])
    memory = fig.add_subplot(lower[0, 1])
    timing.text(
        -0.13, 1.08, "b", transform=timing.transAxes, fontweight="bold", fontsize=12
    )
    labels = ["Background", "Branch evolution + BP", "Pauli readout"]
    hours = [
        resources["completed_phase_seconds"][key] / 3600
        for key in ("background", "evolution", "readout")
    ]
    timing.barh(labels, hours, color=["#42966D", "#D88B20", "#2468A2"])
    timing.invert_yaxis()
    timing.set_xlabel("Recorded completed phases (h)")
    memory.bar(
        ["Allocated", "Reserved"],
        [resources["peak_allocated_gib"], resources["peak_reserved_gib"]],
        color=["#2468A2", "#91B6D5"],
    )
    memory.set_ylabel("PyTorch peak (GiB)")
    out = Path(directory) / "figures"
    out.mkdir(exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(out / f"Fig6.{suffix}", dpi=300)
    plt.close(fig)
    identity = result["identity"]
    metrics = dict(
        identity=identity,
        resources=resources,
        source_count=len(result["raw"]["source_layers"]),
        gate_count=len(result["raw"]["source_gate_ids"]),
        retained_mode_count=sum(
            row["retained_mode_count"] for row in result["source_layer_schedule"]
        ),
        pauli_components=108,
        max_absolute_first_order_change=float(np.abs(delta).max()),
        observable="conditional ratio after additive one-location probability assembly",
        validation="large-system demonstration; no full-state reference or accuracy bound inferred",
    )
    (out / "Fig6_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    controls = identity["numerics"]
    (out / "Fig6_caption.txt").write_text(
        f"""Figure 6. Sparse-residual 36-qubit demonstration. The 6×6 circuit has 20 layers and seven fixed, endpoint-disjoint diagonal residual pairs. The coherent background includes the same pulse-dressed Magnus-1 residual kernels in every source branch. All source gates and retained Kraus modes contribute to all 108 final local Pauli components. Panel a shows the conditional ratios after additive one-location probability assembly and their change relative to the residual-dressed background. Dynamic BP compression uses tolerance {identity["dynamic_bp_threshold"]:g}, bond cap {controls["chi_max"]}, BP tolerance {controls["bp_tol"]:g}, iteration cap {controls["bp_max_iter"]} and gloop size {controls["gloop_size"]}. Ragged branch tensors retain independent dimensions; norm BP uses temporary padded batches and terminal reads use adaptive bounded batches on one CUDA stream. Panel b reports completed phase wall times and observed PyTorch allocated/reserved peaks from this run only. The phase times exclude model construction, orchestration and unrecorded interrupted work; they are not an end-to-end GPU time. Driver and other-process memory are excluded from the memory bars. This figure demonstrates a large-system calculation and does not establish an exact full-state error bound.\n"""
    )
    return metrics
