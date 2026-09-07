"""Source-restricted fidelity pairs; no terminal Pauli observable is evaluated."""

import argparse, dataclasses, math, time, hashlib, json, random
from pathlib import Path
import torch
from residual_tn.peps import model as p


def target_sample(ctx, geometry, per_layer=3):
    """Fixed, outcome-independent spatial sample, stratified by residual support."""
    endpoints = {q for edge in geometry["edges"] for q in (edge["q1"], edge["q2"])}
    rng = random.Random(20260906)
    chosen = []
    for layer, gates in enumerate(ctx.layers):
        groups = [
            [g for g in gates if bool(set(g["qubits"]) & endpoints) == touch]
            for touch in (True, False)
        ]
        selected = []
        for group in groups:
            if group:
                selected.append(rng.choice(group))
        remainder = [
            g
            for g in gates
            if all(int(g["gate_idx"]) != int(h["gate_idx"]) for h in selected)
        ]
        rng.shuffle(remainder)
        selected = (selected + remainder)[: min(per_layer, len(gates))]
        for gate in sorted(selected, key=lambda g: int(g["gate_idx"])):
            chosen.append(
                dict(
                    layer=layer,
                    gate=int(gate["gate_idx"]),
                    qubits=list(gate["qubits"]),
                    touches_residual_endpoint=bool(set(gate["qubits"]) & endpoints),
                )
            )
    return chosen


def sample_identity(identity, sample):
    return dict(
        identity,
        target_sample=sample,
        target_sample_sha256=hashlib.sha256(
            json.dumps(sample, sort_keys=True).encode()
        ).hexdigest(),
        pair_scope="Sampled second-insertion target gates, with all source-0 gates and retained modes; ordered distinct gate IDs",
        observable="fidelity_second_insertion_pairs",
        requested_source_layers=[0],
    )


def assemble(ctx, one, pairs, sources=(0,), sample=None):
    selected = {int(g["gate_idx"]) for s in sources for g in ctx.layers[s]}
    ids = (
        [int(g["gate_idx"]) for layer in ctx.layers for g in layer]
        if sample is None
        else [r["gate"] for r in sample]
    )
    expected = {(a, b) for a in selected for b in ids if b > a}
    actual = {(r["source_gate"], r["target_gate"]) for r in pairs}
    assert actual == expected and len(actual) == len(pairs), (
        len(expected),
        len(actual),
        len(pairs),
    )
    by_layer = [
        sum(r["delta"] for r in pairs if r["target_layer"] == t)
        for t in range(len(ctx.layers))
    ]
    cumulative = []
    total = 0.0
    for value in by_layer:
        total += value
        cumulative.append(total)
    return dict(
        source_layers=list(sources),
        pair_count=len(pairs),
        pairs=pairs,
        one_location_fidelity=one,
        pair_delta_by_target_layer=by_layer,
        cumulative_pair_delta=cumulative,
        pair_delta_sum=total,
        scope="Sampled fidelity second-insertion pairs from selected sources; sums are unweighted sample sums, not a complete C2 or fidelity",
        target_sample=sample,
    )


def exact(ctx, kernels, identity, out, source=0):
    start = time.perf_counter()
    history = []
    modes = {}
    one = {}
    state = torch.zeros(2**ctx.n_qubits, dtype=ctx.dtype, device=ctx.device)
    state[0] = 1
    for t, gates in enumerate(ctx.layers):
        state = p.exact_ops.apply_background_layer(
            state, kernels[t], gates, n_qubits=ctx.n_qubits
        )
        history.append(state.clone())
        for gate in gates:
            gid = int(gate["gate_idx"])
            dim = 2 ** len(gate["qubits"])
            modes[gid] = p.core.compute_dressed_kraus(
                gate["kraus_ops"], gate["ideal_unitary"]
            ).reshape(-1, dim, dim)
            transition = p.exact_ops._local_transition_overlap(
                state, state.unsqueeze(0), gate["qubits"], n_qubits=ctx.n_qubits
            )
            amplitude = torch.einsum("voi,boi->bv", modes[gid], transition)
            one[gid] = float(amplitude.abs().square().sum())
    pairs = []
    for gate in ctx.layers[source]:
        gid = int(gate["gate_idx"])
        branch = p.exact_ops.apply_mode_bank(
            history[source], modes[gid], gate["qubits"], n_qubits=ctx.n_qubits
        )
        for target in range(source, len(ctx.layers)):
            if target > source:
                branch = p.exact_ops.apply_background_layer(
                    branch, kernels[target], ctx.layers[target], n_qubits=ctx.n_qubits
                )
            for other in ctx.layers[target]:
                bid = int(other["gate_idx"])
                if bid <= gid or (
                    identity.get("target_sample") is not None
                    and bid not in {r["gate"] for r in identity["target_sample"]}
                ):
                    continue
                transition = p.exact_ops._local_transition_overlap(
                    history[target], branch, other["qubits"], n_qubits=ctx.n_qubits
                )
                amplitude = torch.einsum("voi,boi->bv", modes[bid], transition)
                fab = float(amplitude.abs().square().sum())
                pairs.append(
                    dict(
                        source_gate=gid,
                        target_gate=bid,
                        target_layer=target,
                        pair_fidelity=fab,
                        delta=fab / (one[gid] * one[bid]) - 1.0,
                    )
                )
    result = assemble(ctx, one, pairs, (source,), identity.get("target_sample"))
    result.update(identity=identity, seconds=time.perf_counter() - start)
    p.save(Path(out) / "result.json", result)
    return result
