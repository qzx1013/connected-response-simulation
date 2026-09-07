"""Four-qubit end-to-end checks, never the production GPU study."""

import dataclasses
import json
import torch
from residual_tn.peps import (
    model as p,
    source_bank as bank,
    fidelity_batch as f,
    background_cache as shared,
)
from residual_tn.experiments import (
    pauli_exact,
    fidelity_pairs as fs,
    fidelity_controls as panel,
)


def test_pauli_source_bank_exact_and_resume(tmp_path, monkeypatch):
    ctx, geometry, kernels, identity = p.model(torch.device("cpu"), small=True)
    # Keep real residual kernels/control channels and the complete toy depth.
    cfg = p.config(device="cpu", gloop=4, bond=16, bp_tol=1e-12)
    identity.update(numerics=dataclasses.asdict(cfg), dynamic_bp_threshold=1e-10)
    assert json.loads(json.dumps(identity)) == identity
    exact = pauli_exact.calculate_all(
        ctx, kernels, identity, tmp_path / "exact", source_layers=[0]
    )
    out = tmp_path / "peps"
    with p.DynamicEvolution(
        ctx,
        geometry,
        kernels,
        cfg,
        out / "compression.jsonl",
        threshold=1e-10,
        residual_svd_tol=1e-12,
    ) as evo:
        result = bank.calculate(ctx, identity, evo, cfg, out, source_layers=[0])
    x = torch.tensor(exact["additive_first_order_pauli"], dtype=torch.float64)
    y = torch.tensor(result["additive_first_order_pauli"], dtype=torch.float64)
    assert float((x - y).abs().max()) < 1e-7
    assert not list(out.glob("evolved_source_*.pt"))
    assert (out / "background_layers.pt").exists()

    def unexpected(*args, **kwargs):
        raise AssertionError("Committed source was recomputed")

    monkeypatch.setattr(bank, "make_bank", unexpected)
    with p.DynamicEvolution(
        ctx,
        geometry,
        kernels,
        cfg,
        out / "compression.jsonl",
        threshold=1e-10,
        residual_svd_tol=1e-12,
    ) as evo:
        resumed = bank.calculate(ctx, identity, evo, cfg, out, source_layers=[0])
    assert resumed["raw"] == result["raw"]


def test_fidelity_target_restart_and_shared_normalizers(tmp_path):
    ctx, geometry, kernels, identity = p.model(torch.device("cpu"), small=True)
    cfg = p.config(device="cpu", gloop=4, bond=16)
    identity = f.schedule_identity(
        fs.sample_identity(identity, fs.target_sample(ctx, geometry))
    )
    identity.update(numerics=dataclasses.asdict(cfg), dynamic_bp_threshold=1e-4)
    baseline = tmp_path / "baseline"
    with p.DynamicEvolution(
        ctx, geometry, kernels, cfg, baseline / "compression.jsonl"
    ) as evo:
        original = f.calculate(ctx, identity, evo, cfg, baseline)
    shared.export(baseline / "background_layers.pt", baseline, "e")
    layers, plan, descriptor = shared.load(ctx, identity, cfg, baseline)
    descriptors = {group: dict(descriptor, group=group) for group in "ef"}
    out = tmp_path / "panel_ef"
    with p.DynamicEvolution(
        ctx, geometry, kernels, cfg, out / "compression.jsonl"
    ) as evo:
        try:
            panel.calculate(
                ctx,
                identity,
                evo,
                cfg,
                layers,
                plan,
                descriptors,
                out,
                stop_after_target=1,
            )
        except InterruptedError:
            pass
        else:
            raise AssertionError("Restart injection did not fire")
        results = panel.calculate(
            ctx, identity, evo, cfg, layers, plan, descriptors, out
        )
    reference = {
        (r["source_gate"], r["target_gate"]): r["delta"] for r in original["pairs"]
    }
    observed = results["tol_baseline"]
    assert observed["one_location_fidelity"] == original["one_location_fidelity"]
    assert len(observed["pairs"]) == len(reference)
    assert (
        max(
            abs(r["delta"] - reference[r["source_gate"], r["target_gate"]])
            for r in observed["pairs"]
        )
        < 2e-10
    )
