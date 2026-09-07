"""Rendering fixtures are synthetic and are never production figure data."""

import json
import numpy as np
import pytest
from residual_tn.config import read
from residual_tn.geometry import load
from residual_tn.paths import CONFIGS
from residual_tn.figures.fig6 import render, load_completed


def test_complete_run_render_and_subset_rejection(tmp_path):
    spec = read("fig6")
    geometry = load(CONFIGS / "residual_manifest.json", 6, 6)
    identity = dict(
        width=6,
        length=6,
        layers=20,
        residual_edges=geometry["edges"],
        gate_counts_by_layer=[1] * 20,
        dynamic_bp_threshold=1e-4,
        numerics=dict(chi_max=64, bp_tol=1e-10, bp_max_iter=500, gloop_size=8),
    )
    q0 = np.linspace(-0.3, 0.3, 108).reshape(36, 3)
    result = dict(
        identity=identity,
        raw=dict(
            source_layers=list(range(20)),
            site_indices=list(range(36)),
            source_gate_ids=list(range(20)),
        ),
        definition=dict(validation_subset=False),
        ideal_pauli=q0.tolist(),
        additive_first_order_pauli=(q0 * 0.98).tolist(),
        source_layer_schedule=[dict(retained_mode_count=1) for _ in range(20)],
    )
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result))
    for source in range(20):
        (tmp_path / f"source_{source:02}.json").write_text(
            json.dumps(
                dict(identity=identity, source_layer=source, source_gate_ids=[source])
            )
        )
    (tmp_path / "batch_resources.jsonl").write_text(
        json.dumps(
            dict(
                event="pauli_read",
                seconds=3600,
                peak_allocated_gib=3,
                peak_reserved_gib=4,
            )
        )
        + "\n"
    )
    metrics = render(tmp_path)
    assert (
        metrics["pauli_components"] == 108
        and metrics["resources"]["peak_allocated_gib"] == 3
    )
    assert all(
        (tmp_path / "figures" / ("Fig6." + suffix)).stat().st_size > 100
        for suffix in ("png", "pdf", "svg")
    )
    result["definition"]["validation_subset"] = True
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="incomplete"):
        load_completed(tmp_path)
