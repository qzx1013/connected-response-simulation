"""Independent direct full-state Pauli reference, restricted to small systems."""

from pathlib import Path
import json
import time
import torch
from residual_tn.peps import model as p


def calculate_all(ctx, kernels, identity, out, source_layers=None):
    if ctx.n_qubits > 20:
        raise ValueError(
            "Explicit full-state reference is limited to 20 qubits; Fig6 uses PEPS only"
        )
    out = Path(out)
    sources = (
        list(range(len(ctx.layers))) if source_layers is None else list(source_layers)
    )
    state = torch.zeros(2**ctx.n_qubits, dtype=ctx.dtype, device=ctx.device)
    state[0] = 1
    history = []
    for layer, gates in enumerate(ctx.layers):
        state = p.exact_ops.apply_background_layer(
            state, kernels[layer], gates, n_qubits=ctx.n_qubits
        )
        history.append(state.clone())
    q0 = p.cpu(p.positive(state, ctx.n_qubits))
    records = []
    for source in sources:
        path = out / f"source_{source:02}.json"
        if path.exists():
            row = json.loads(path.read_text())
            if row["identity"] != identity:
                raise ValueError("Exact reference checkpoint identity changed")
        else:
            tick = time.perf_counter()
            values = []
            ids = []
            for gate in ctx.layers[source]:
                dimension = 2 ** len(gate["qubits"])
                modes = p.core.compute_dressed_kraus(
                    gate["kraus_ops"], gate["ideal_unitary"]
                ).reshape(-1, dimension, dimension)
                branch = p.exact_ops.apply_mode_bank(
                    history[source], modes, gate["qubits"], n_qubits=ctx.n_qubits
                )
                for target in range(source + 1, len(ctx.layers)):
                    branch = p.exact_ops.apply_background_layer(
                        branch,
                        kernels[target],
                        ctx.layers[target],
                        n_qubits=ctx.n_qubits,
                    )
                values.append(p.cpu(p.positive(branch, ctx.n_qubits)))
                ids.append(int(gate["gate_idx"]))
            row = dict(
                identity=identity,
                source_layer=source,
                source_gate_ids=ids,
                qg=torch.stack(values).tolist(),
                seconds=time.perf_counter() - tick,
            )
            p.save(path, row)
        records.append(row)
    qg = torch.tensor(
        [value for row in records for value in row["qg"]], dtype=torch.float64
    )
    delta = (qg - q0).sum(0)
    per_source = []
    for row in records:
        d = (torch.tensor(row["qg"], dtype=torch.float64) - q0).sum(0)
        per_source.append(
            dict(
                source_layer=row["source_layer"],
                delta_q=d.tolist(),
                delta_pauli=p.correction(q0, d).tolist(),
            )
        )
    result = dict(
        identity=identity,
        raw=dict(
            q0=q0.tolist(),
            qg=qg.tolist(),
            q1=(q0 + delta).tolist(),
            source_gate_ids=[i for row in records for i in row["source_gate_ids"]],
            source_layers=sources,
            site_indices=list(range(ctx.n_qubits)),
        ),
        per_source_layer=per_source,
        ideal_pauli=p.conditional(q0).tolist(),
        additive_first_order_pauli=p.conditional(q0 + delta).tolist(),
        strict_first_order_pauli=(p.conditional(q0) + p.correction(q0, delta)).tolist(),
    )
    p.save(out / "result.json", result)
    return result
