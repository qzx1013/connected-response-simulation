from pathlib import Path
import json
import subprocess
import sys
import pytest

from residual_tn.geometry import load, validate_geometry
from residual_tn.identity import RunLock, claim_run
from residual_tn.paths import CONFIGS


def test_frozen_geometries():
    expected = {
        (4, 2): [(1, 4), (2, 5), (3, 6)],
        (4, 4): [(1, 4), (2, 7), (9, 12), (10, 13)],
        (5, 4): [(1, 5), (3, 9), (10, 16), (11, 17)],
        (6, 6): [(1, 8), (3, 10), (9, 16), (19, 26), (22, 29), (24, 31), (28, 35)],
    }
    for shape, pairs in expected.items():
        geometry = load(CONFIGS / "residual_manifest.json", *shape)
        audit = validate_geometry(geometry)
        assert audit["pairs"] == [list(pair) for pair in pairs]
        if shape == (6, 6):
            assert (
                audit["shared_base_route_vertices"]
                == audit["shared_base_route_bonds"]
                == 0
            )


def test_plan_is_portable_and_never_imports_torch(tmp_path):
    code = """import sys
from residual_tn.config import describe
assert 'torch' not in sys.modules
value = describe('fig6')
assert value['geometry']['edges'] == 7
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path, check=True)
    value = subprocess.check_output(
        [sys.executable, "-I", "-m", "residual_tn", "plan", "fig6"],
        cwd=tmp_path,
        text=True,
    )
    assert json.loads(value)["model"]["width"] == 6


def test_checkpoint_identity_and_single_writer(tmp_path):
    identity = {"geometry": "one", "compression": 1e-4}
    claim_run(tmp_path, identity)
    claim_run(tmp_path, dict(identity))
    with pytest.raises(ValueError):
        claim_run(tmp_path, dict(identity, compression=1e-5))
    with RunLock(tmp_path):
        with pytest.raises(RuntimeError):
            with RunLock(tmp_path):
                pass
    with RunLock(tmp_path):
        pass


def test_fig6_rejects_full_state_before_allocating():
    from types import SimpleNamespace
    from residual_tn.experiments.pauli_exact import calculate_all

    with pytest.raises(ValueError, match="20 qubits"):
        calculate_all(SimpleNamespace(n_qubits=36), None, None, None)


def test_fig6_local_model_construction(monkeypatch):
    import torch
    from residual_tn.peps import model

    original = torch.zeros

    def bounded(*shape, **kwargs):
        import math

        dimensions = (
            shape[0]
            if len(shape) == 1 and isinstance(shape[0], (tuple, list))
            else shape
        )
        assert math.prod(dimensions) < 2**25, (
            "Local model attempted a large state allocation"
        )
        return original(*shape, **kwargs)

    monkeypatch.setattr(torch, "zeros", bounded)
    ctx, geometry, kernels, identity = model.model(
        torch.device("cpu"), width=6, length=6, depth=2, pieces=4
    )
    assert ctx.n_qubits == 36 and len(geometry["edges"]) == 7 and len(kernels) == 2
    assert len(kernels[0]) == 7 and all(
        len(k.support) <= 4 for row in kernels for k in row
    )
    assert json.loads(json.dumps(identity)) == identity


def test_resource_accounting_does_not_double_count_bp():
    from residual_tn.figures.fig6 import resource_summary

    result = resource_summary(
        [
            dict(
                event="evolution_layer",
                gates_seconds=2,
                bp_seconds=3,
                projection_and_checkpoint_seconds=1,
            ),
            dict(event="norm_bp", layer=2, seconds=3),
            dict(event="norm_bp", layer="terminal", seconds=4),
            dict(event="pauli_read", seconds=5),
            dict(event="background_saved", seconds=6),
        ]
    )
    assert result["completed_phase_seconds"] == dict(
        evolution=10.0, background=6.0, readout=5.0
    )
