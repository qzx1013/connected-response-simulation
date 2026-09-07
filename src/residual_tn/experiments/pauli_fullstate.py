#!/usr/bin/env python
"""Local Pauli observables from exact qutrit Liouville and full-state C2.

The production definitions are deliberately leakage-aware and positive-effect
based.  For every qubit i and basis alpha in {X,Y,Z}, the code evaluates

    E(i,alpha,+/-) = (I +/- sigma_alpha) / 2

inside the global computational subspace.  The reported Pauli expectation is
conditional on global computational-subspace survival,

    <sigma_alpha>_cond = (p_plus - p_minus) / (p_plus + p_minus).

For eight qutrits, the reference is direct local-axis Liouville evolution with
the raw qutrit channels.  For eight, sixteen, and twenty-four projected qubits,
the estimator uses exact full-state Kraus branches and all one- and two-location
channel responses.  Positive terminal probabilities use the logarithmic
connected formula unless their ideal/one-location reference falls below
``reference_floor``; those entries automatically use the additive connected
expansion.

The 16-qubit production case reuses the 1000-slice, T1=T2=5 us projected Kraus
operators in ``0522_reference_data.pt``.  The 24-qubit 6x4 case compiles the
same 0522 qutrit channels and uses target-Kraus chunking to keep exact
full-state branch memory below the A100 limit.  No qutrit Liouville state is
attempted for 16 or 24 qubits.
"""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import replace
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch


from residual_tn.paths import CONFIGS, INPUTS, RESULTS

PACKAGE_ROOT = RESULTS
from residual_tn.experiments import fig2 as qcore
from residual_tn.backend.context import FidelityContext  # noqa: E402
from residual_tn.backend.kraus import compute_dressed_kraus  # noqa: E402
from residual_tn.backend.residual_magnus import ResidualMagnusLibrary  # noqa: E402
from residual_tn.backend import residual_background_exact as residual_bg  # noqa: E402


BASES = ("X", "Y", "Z")
SIGNS = ("+", "-")
FORMAT_VERSION = 1
DEFAULT_RESIDUAL_MANIFEST = CONFIGS / "residual_manifest.json"


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_memory_gib(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024**3))


def apply_1q_batch(
    states: torch.Tensor,
    gate: torch.Tensor,
    qubit: int,
    n_qubits: int,
) -> torch.Tensor:
    """Apply one common 1Q operator to a state-vector batch."""
    if states.ndim == 1:
        states = states.unsqueeze(0)
    batch = int(states.shape[0])
    x = states.reshape(batch, *([2] * n_qubits))
    x = torch.movedim(x, qubit + 1, 1)
    shape = x.shape
    x = x.reshape(batch, 2, -1)
    y = torch.einsum("oi,bik->bok", gate.reshape(2, 2), x)
    y = y.reshape(shape)
    y = torch.movedim(y, 1, qubit + 1)
    return y.reshape(batch, 2**n_qubits)


def apply_2q_batch(
    states: torch.Tensor,
    gate: torch.Tensor,
    q1: int,
    q2: int,
    n_qubits: int,
) -> torch.Tensor:
    """Apply one common 2Q operator to a state-vector batch."""
    if states.ndim == 1:
        states = states.unsqueeze(0)
    batch = int(states.shape[0])
    x = states.reshape(batch, *([2] * n_qubits))
    x = torch.movedim(x, (q1 + 1, q2 + 1), (1, 2))
    shape = x.shape
    x = x.reshape(batch, 4, -1)
    y = torch.einsum("oi,bik->bok", gate.reshape(4, 4), x)
    y = y.reshape(shape)
    y = torch.movedim(y, (1, 2), (q1 + 1, q2 + 1))
    return y.reshape(batch, 2**n_qubits)


def apply_1q_per_state(
    states: torch.Tensor,
    gates: torch.Tensor,
    qubit: int,
    n_qubits: int,
) -> torch.Tensor:
    """Apply one 1Q operator per state in a state-vector batch."""
    batch = int(states.shape[0])
    x = states.reshape(batch, *([2] * n_qubits))
    x = torch.movedim(x, qubit + 1, 1)
    shape = x.shape
    x = x.reshape(batch, 2, -1)
    y = torch.einsum("boi,bik->bok", gates.reshape(batch, 2, 2), x)
    y = y.reshape(shape)
    y = torch.movedim(y, 1, qubit + 1)
    return y.reshape(batch, 2**n_qubits)


def apply_2q_per_state(
    states: torch.Tensor,
    gates: torch.Tensor,
    q1: int,
    q2: int,
    n_qubits: int,
) -> torch.Tensor:
    """Apply one 2Q operator per state in a state-vector batch."""
    batch = int(states.shape[0])
    x = states.reshape(batch, *([2] * n_qubits))
    x = torch.movedim(x, (q1 + 1, q2 + 1), (1, 2))
    shape = x.shape
    x = x.reshape(batch, 4, -1)
    y = torch.einsum("boi,bik->bok", gates.reshape(batch, 4, 4), x)
    y = y.reshape(shape)
    y = torch.movedim(y, (1, 2), (q1 + 1, q2 + 1))
    return y.reshape(batch, 2**n_qubits)


def apply_mode_bank(
    states: torch.Tensor,
    modes: torch.Tensor,
    qubits: Sequence[int],
    n_qubits: int,
) -> torch.Tensor:
    """Apply every local operator mode to every input branch state."""
    if states.ndim == 1:
        states = states.unsqueeze(0)
    batch = int(states.shape[0])
    n_modes = int(modes.shape[0])
    expanded_states = (
        states[:, None, :]
        .expand(batch, n_modes, states.shape[-1])
        .reshape(batch * n_modes, states.shape[-1])
    )
    local_dim = 2 ** len(qubits)
    expanded_modes = (
        modes.reshape(n_modes, local_dim, local_dim)[None, :, :, :]
        .expand(batch, n_modes, local_dim, local_dim)
        .reshape(batch * n_modes, local_dim, local_dim)
    )
    if len(qubits) == 1:
        return apply_1q_per_state(
            expanded_states, expanded_modes, int(qubits[0]), n_qubits
        )
    if len(qubits) == 2:
        return apply_2q_per_state(
            expanded_states,
            expanded_modes,
            int(qubits[0]),
            int(qubits[1]),
            n_qubits,
        )
    raise ValueError(f"unsupported gate arity: {len(qubits)}")


def propagate_layers(
    states: torch.Tensor,
    ctx: FidelityContext,
    start_layer: int,
    stop_layer: int | None = None,
    residual_layers: Sequence[Sequence[Any]] | None = None,
) -> torch.Tensor:
    """Propagate a branch batch through common-background layers [start, stop)."""
    stop = len(ctx.layers) if stop_layer is None else int(stop_layer)
    for layer_idx in range(int(start_layer), stop):
        if residual_layers is not None:
            states = residual_bg.apply_background_layer(
                states,
                residual_layers[layer_idx],
                ctx.layers[layer_idx],
                n_qubits=ctx.n_qubits,
            )
            continue
        for gate in ctx.layers[layer_idx]:
            qubits = tuple(int(q) for q in gate["qubits"])
            if len(qubits) == 1:
                states = apply_1q_batch(
                    states, gate["ideal_unitary"], qubits[0], ctx.n_qubits
                )
            else:
                states = apply_2q_batch(
                    states,
                    gate["ideal_unitary"],
                    qubits[0],
                    qubits[1],
                    ctx.n_qubits,
                )
    return states


def ideal_history_and_dressed_ops(
    ctx: FidelityContext,
    residual_layers: Sequence[Sequence[Any]] | None = None,
) -> tuple[list[torch.Tensor], dict[int, torch.Tensor]]:
    """Return common-background states and every local replacement bank."""
    psi = torch.zeros(2**ctx.n_qubits, dtype=ctx.dtype, device=ctx.device)
    psi[0] = 1.0
    history: list[torch.Tensor] = []
    dressed: dict[int, torch.Tensor] = {}
    for layer_index, layer in enumerate(ctx.layers):
        if residual_layers is not None:
            psi = residual_bg.apply_background_layer(
                psi,
                residual_layers[layer_index],
                layer,
                n_qubits=ctx.n_qubits,
            )
        else:
            for gate in layer:
                qubits = tuple(int(q) for q in gate["qubits"])
                if len(qubits) == 1:
                    psi = qcore.apply_1q_state(
                        psi, gate["ideal_unitary"], qubits[0], ctx.n_qubits, d=2
                    )
                else:
                    psi = qcore.apply_2q_state(
                        psi,
                        gate["ideal_unitary"],
                        qubits[0],
                        qubits[1],
                        ctx.n_qubits,
                        d=2,
                    )
        history.append(psi.clone())
        for gate in layer:
            dressed[int(gate["gate_idx"])] = compute_dressed_kraus(
                gate["kraus_ops"], gate["ideal_unitary"]
            ).reshape(
                -1,
                2 ** len(tuple(gate["qubits"])),
                2 ** len(tuple(gate["qubits"])),
            )
    return history, dressed


def local_rhos_from_state_batch(
    states: torch.Tensor,
    n_qubits: int,
) -> torch.Tensor:
    """Sum one-qubit reduced density matrices over a branch-state batch."""
    if states.ndim == 1:
        states = states.unsqueeze(0)
    batch = int(states.shape[0])
    tensor = states.reshape(batch, *([2] * n_qubits))
    local = []
    for qubit in range(n_qubits):
        x = torch.movedim(tensor, qubit + 1, 1).reshape(batch, 2, -1)
        local.append(torch.einsum("bik,bjk->ij", x, x.conj()))
    return torch.stack(local, dim=0)


def local_rhos_after_mode_bank(
    states: torch.Tensor,
    modes: torch.Tensor,
    qubits: Sequence[int],
    ctx: FidelityContext,
    start_layer: int,
    mode_chunk_size: int | None,
    residual_layers: Sequence[Sequence[Any]] | None = None,
) -> torch.Tensor:
    """Apply a Kraus bank in exact chunks and sum its final local density matrices.

    Chunking changes only the order of the Kraus-mode sum.  It prevents the
    two-location 24-qubit calculation from materializing every source-target
    Kraus combination at once.
    """
    n_modes = int(modes.shape[0])
    chunk = n_modes if mode_chunk_size is None else int(mode_chunk_size)
    if chunk <= 0:
        raise ValueError("mode_chunk_size must be positive or None")
    local_total = torch.zeros((ctx.n_qubits, 2, 2), dtype=ctx.dtype, device=ctx.device)
    for first in range(0, n_modes, chunk):
        branches = apply_mode_bank(
            states,
            modes[first : first + chunk],
            qubits,
            ctx.n_qubits,
        )
        branches = propagate_layers(
            branches,
            ctx,
            start_layer,
            residual_layers=residual_layers,
        )
        local_total += local_rhos_from_state_batch(branches, ctx.n_qubits)
        del branches
    return local_total


def pauli_matrices(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.stack(
        [
            torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device),
            torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device),
            torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device),
        ],
        dim=0,
    )


def positive_terminal_probabilities(local_rhos: torch.Tensor) -> torch.Tensor:
    """Return p(+/-) with shape (n_qubits, 3 bases, 2 signs)."""
    paulis = pauli_matrices(local_rhos.device, local_rhos.dtype)
    trace = torch.diagonal(local_rhos, dim1=-2, dim2=-1).sum(-1).real
    expectation = torch.einsum("aij,qji->qa", paulis, local_rhos).real
    return torch.stack(
        ((trace[:, None] + expectation) / 2, (trace[:, None] - expectation) / 2),
        dim=-1,
    ).to(torch.float64)


def summarize_probabilities(probabilities: torch.Tensor) -> dict[str, Any]:
    p = probabilities.detach().to(torch.float64).cpu()
    survival = p[..., 0] + p[..., 1]
    signed = p[..., 0] - p[..., 1]
    conditional = signed / survival.clamp_min(1e-300)
    return {
        "p_plus": p[..., 0].tolist(),
        "p_minus": p[..., 1].tolist(),
        "survival_by_basis": survival.tolist(),
        "pauli_unconditional": signed.tolist(),
        "pauli_conditional": conditional.tolist(),
        "survival_basis_spread_max": float(
            (survival.max(dim=1).values - survival.min(dim=1).values).max()
        ),
        "probability_min": float(p.min()),
        "probability_max": float(p.max()),
    }


def _trace_local_axis_tensor(
    tensor: torch.Tensor,
    diagonal_indices: torch.Tensor,
) -> torch.Tensor:
    out = tensor
    for axis in reversed(range(out.ndim)):
        out = out.index_select(axis, diagonal_indices).sum(dim=axis)
    return out


def qutrit_global_computational_local_rhos(
    rho_local: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    """Project all qutrits to {0,1}, then return unnormalized local qubit rhos."""
    n_qubits = rho_local.ndim
    device = rho_local.device
    comp = torch.tensor([0, 1, 3, 4], dtype=torch.long, device=device)
    diag_qutrit = torch.tensor([0, 4, 8], dtype=torch.long, device=device)
    raw_trace = float(
        _trace_local_axis_tensor(rho_local, diag_qutrit).real.detach().cpu()
    )
    projected = rho_local
    for axis in range(n_qubits):
        projected = projected.index_select(axis, comp)
    diag_qubit = torch.tensor([0, 3], dtype=torch.long, device=device)
    survival = float(
        _trace_local_axis_tensor(projected, diag_qubit).real.detach().cpu()
    )
    local = []
    for qubit in range(n_qubits):
        reduced = projected
        for axis in reversed(range(n_qubits)):
            if axis == qubit:
                continue
            reduced = reduced.index_select(axis, diag_qubit).sum(dim=axis)
        local.append(reduced.reshape(2, 2))
    return torch.stack(local, dim=0), survival, raw_trace


def run_qutrit_liouville_final(
    pattern: Sequence[Any],
    channels: qcore.ChannelSet,
    width: int,
    length: int,
    residual_layers: Sequence[Sequence[Any]] | None = None,
) -> torch.Tensor:
    n_qubits = width * length
    rho = torch.zeros(
        (9,) * n_qubits,
        dtype=channels.ideal_single.dtype,
        device=channels.ideal_single.device,
    )
    rho[(0,) * n_qubits] = 1.0
    for layer_idx, layer in enumerate(pattern):
        if residual_layers is not None:
            rho = residual_bg.apply_residual_layer_to_liouville(
                rho, residual_layers[layer_idx], local_dim=3
            )
        if layer_idx % 2 == 0:
            for phys in range(n_qubits):
                mode = int(layer[phys])
                rho = qcore.apply_1q_liouville(
                    rho, channels.single_superoperators[phys, mode], phys
                )
        else:
            for q1, q2 in layer:
                rho = qcore.apply_2q_liouville(
                    rho, channels.two_superoperator, int(q1), int(q2)
                )
    return rho


def channel_cache_payload(
    channels: qcore.ChannelSet,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "metadata": metadata,
        "ideal_single": channels.ideal_single.detach().cpu(),
        "ideal_two": channels.ideal_two.detach().cpu(),
        "single_superoperators": channels.single_superoperators.detach().cpu(),
        "two_superoperator": channels.two_superoperator.detach().cpu(),
        "single_kraus": {
            tuple(int(x) for x in key): value.detach().cpu()
            for key, value in channels.single_kraus.items()
        },
        "two_kraus": channels.two_kraus.detach().cpu(),
    }


def channels_from_cache(
    payload: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> qcore.ChannelSet:
    return qcore.ChannelSet(
        ideal_single=payload["ideal_single"].to(device=device, dtype=dtype),
        ideal_two=payload["ideal_two"].to(device=device, dtype=dtype),
        single_superoperators=payload["single_superoperators"].to(
            device=device, dtype=dtype
        ),
        two_superoperator=payload["two_superoperator"].to(device=device, dtype=dtype),
        single_kraus={
            tuple(int(x) for x in key): value.to(device=device, dtype=dtype)
            for key, value in payload["single_kraus"].items()
        },
        two_kraus=payload["two_kraus"].to(device=device, dtype=dtype),
    )


def load_or_build_channels(
    width: int,
    length: int,
    piece_num: int,
    noise_time_ns: float,
    cache_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[qcore.ChannelSet, dict[str, Any]]:
    expected = {
        "width": int(width),
        "length": int(length),
        "piece_num": int(piece_num),
        "noise_time_ns": float(noise_time_ns),
    }
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        metadata = dict(payload.get("metadata", {}))
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(
                    f"channel cache mismatch for {key}: {metadata.get(key)!r} != {value!r}"
                )
        metadata["loaded_from_cache"] = True
        metadata["cache_path"] = str(cache_path)
        return channels_from_cache(payload, device, dtype), metadata
    channels = qcore.build_channels(
        width, length, piece_num, noise_time_ns, device, dtype
    )
    metadata = {
        **expected,
        "compiler": "0522 qutrit local Liouville, Crank-Nicolson slices",
        "loaded_from_cache": False,
        "cache_path": str(cache_path),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(channel_cache_payload(channels, metadata), cache_path)
    return channels, metadata


def axis_for_physical(reference: dict[str, Any], physical: int) -> int:
    mapping = reference["phys_to_1d"]
    if int(physical) in mapping:
        return int(mapping[int(physical)])
    return int(mapping[str(int(physical))])


def remap_residual_layers_to_reference_axes(
    layers: Sequence[Sequence[Any]],
    reference: dict[str, Any],
) -> list[list[Any]]:
    """Map physical residual supports onto the full-state tensor axes."""
    return [
        [
            replace(
                kernel,
                support=tuple(
                    axis_for_physical(reference, physical)
                    for physical in kernel.support
                ),
            )
            for kernel in layer
        ]
        for layer in layers
    ]


def pair_index(reference: dict[str, Any], q1: int, q2: int) -> int:
    pair = tuple(sorted((int(q1), int(q2))))
    mapping = reference["pair_to_idx"]
    if pair in mapping:
        return int(mapping[pair])
    for key, value in mapping.items():
        parsed = ast.literal_eval(key) if isinstance(key, str) else key
        if tuple(sorted(int(x) for x in parsed)) == pair:
            return int(value)
    raise KeyError(pair)


def build_reference_context(
    reference: dict[str, Any],
    device: torch.device,
) -> FidelityContext:
    n_qubits = int(reference["N_QUBITS"])
    width = int(round(math.sqrt(n_qubits)))
    length = n_qubits // width
    ctx = FidelityContext(n_qubits, width, length, device, torch.complex128)
    ideal_single = reference["ideal_single_2d"].to(
        device=device, dtype=torch.complex128
    )
    ideal_two = reference["ideal_two_gate_qubit_4x4"].to(
        device=device, dtype=torch.complex128
    )
    single = {
        (int(key[0]), int(key[1])): value.to(device=device, dtype=torch.complex128)
        for key, value in reference["kraus_single_dict"].items()
    }
    two = {
        int(key): value.to(device=device, dtype=torch.complex128)
        for key, value in reference["kraus_2q_dict"].items()
    }
    gate_idx = 0
    for layer_idx, pattern_layer in enumerate(reference["pattern"]):
        gates: list[dict[str, Any]] = []
        if layer_idx % 2 == 0:
            for physical in range(n_qubits):
                mode = int(pattern_layer[physical])
                gates.append(
                    {
                        "gate_idx": gate_idx,
                        "qubits": (axis_for_physical(reference, physical),),
                        "ideal_unitary": ideal_single[mode],
                        "kraus_ops": single[(physical, mode)],
                    }
                )
                gate_idx += 1
        else:
            for pair in pattern_layer:
                q1, q2 = int(pair[0]), int(pair[1])
                gates.append(
                    {
                        "gate_idx": gate_idx,
                        "qubits": (
                            axis_for_physical(reference, q1),
                            axis_for_physical(reference, q2),
                        ),
                        "ideal_unitary": ideal_two,
                        "kraus_ops": two[pair_index(reference, q1, q2)],
                    }
                )
                gate_idx += 1
        ctx.add_layer(gates)
    return ctx


def gate_records(ctx: FidelityContext) -> list[dict[str, Any]]:
    records = []
    for layer_idx, layer in enumerate(ctx.layers):
        for gate in layer:
            records.append(
                {
                    "gate_idx": int(gate["gate_idx"]),
                    "layer": int(layer_idx),
                    "qubits": tuple(int(q) for q in gate["qubits"]),
                    "mode_count": int(gate["kraus_ops"].shape[0]),
                    "gate": gate,
                }
            )
    records.sort(key=lambda item: item["gate_idx"])
    return records


def context_signature(
    ctx: FidelityContext,
    max_layer_distance: int | None,
    residual_layers: Sequence[Sequence[Any]] | None = None,
) -> str:
    payload = {
        "n_qubits": ctx.n_qubits,
        "n_layers": len(ctx.layers),
        "max_layer_distance": max_layer_distance,
        "gates": [
            (
                int(gate["gate_idx"]),
                int(layer_idx),
                tuple(int(q) for q in gate["qubits"]),
                int(gate["kraus_ops"].shape[0]),
            )
            for layer_idx, layer in enumerate(ctx.layers)
            for gate in layer
        ],
        "residual": None
        if residual_layers is None
        else [
            (
                int(kernel.layer),
                tuple(int(x) for x in kernel.edge),
                tuple(int(x) for x in kernel.support),
                float(kernel.epsilon_rad_ns),
                str(kernel.scheme),
            )
            for layer in residual_layers
            for kernel in layer
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def count_expected_pairs(
    records: Sequence[dict[str, Any]],
    max_layer_distance: int | None,
) -> int:
    count = 0
    for i, source in enumerate(records):
        for target in records[i + 1 :]:
            distance = int(target["layer"]) - int(source["layer"])
            if max_layer_distance is not None and distance > max_layer_distance:
                continue
            count += 1
    return count


def full_state_connected_pauli(
    ctx: FidelityContext,
    *,
    reference_floor: float,
    max_layer_distance: int | None,
    checkpoint_path: Path | None,
    resume: bool,
    progress_every: int,
    mode_chunk_size: int | None,
    residual_layers: Sequence[Sequence[Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate all 3N Pauli outcome pairs through connected order two."""
    device = ctx.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    history, dressed = ideal_history_and_dressed_ops(ctx, residual_layers)
    records = gate_records(ctx)
    gate_to_pos = {int(item["gate_idx"]): i for i, item in enumerate(records)}
    q0 = positive_terminal_probabilities(
        local_rhos_from_state_batch(history[-1], ctx.n_qubits)
    )
    signature = context_signature(ctx, max_layer_distance, residual_layers)

    qg: torch.Tensor
    pair_log_delta: torch.Tensor
    pair_additive_c2: torch.Tensor
    next_source = 0
    pair_count = 0
    one_point_complete = False

    if resume and checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("signature") != signature:
            raise ValueError("checkpoint context signature mismatch")
        qg = checkpoint["qg"].to(device=device, dtype=torch.float64)
        pair_log_delta = checkpoint["pair_log_delta"].to(
            device=device, dtype=torch.float64
        )
        pair_additive_c2 = checkpoint["pair_additive_c2"].to(
            device=device, dtype=torch.float64
        )
        next_source = int(checkpoint["next_source"])
        pair_count = int(checkpoint["pair_count"])
        one_point_complete = bool(checkpoint.get("one_point_complete", False))
        print(
            f"[resume] source={next_source}/{len(records)} pairs={pair_count}",
            flush=True,
        )
    else:
        qg = torch.empty(
            (len(records),) + tuple(q0.shape),
            dtype=torch.float64,
            device=device,
        )
        pair_log_delta = torch.zeros_like(q0)
        pair_additive_c2 = torch.zeros_like(q0)

    if not one_point_complete:
        print(
            f"[full-state] computing {len(records)} one-location responses", flush=True
        )
        for position, item in enumerate(records):
            layer = int(item["layer"])
            gate_idx = int(item["gate_idx"])
            qg[position] = positive_terminal_probabilities(
                local_rhos_after_mode_bank(
                    history[layer],
                    dressed[gate_idx],
                    item["qubits"],
                    ctx,
                    layer + 1,
                    mode_chunk_size,
                    residual_layers,
                )
            )
            if (position + 1) % max(1, progress_every) == 0 or position + 1 == len(
                records
            ):
                print(
                    f"[one] {position + 1}/{len(records)} "
                    f"peak={peak_memory_gib(device):.2f} GiB",
                    flush=True,
                )
        one_point_complete = True
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "format_version": FORMAT_VERSION,
                    "signature": signature,
                    "q0": q0.detach().cpu(),
                    "qg": qg.detach().cpu(),
                    "pair_log_delta": pair_log_delta.detach().cpu(),
                    "pair_additive_c2": pair_additive_c2.detach().cpu(),
                    "next_source": next_source,
                    "pair_count": pair_count,
                    "one_point_complete": True,
                },
                checkpoint_path,
            )

    log_eligible = (q0 > reference_floor) & torch.all(qg > reference_floor, dim=0)
    expected_pairs = count_expected_pairs(records, max_layer_distance)
    print(
        f"[full-state] computing {expected_pairs} two-location responses "
        f"from source {next_source}",
        flush=True,
    )

    for source_pos in range(next_source, len(records)):
        source = records[source_pos]
        source_layer = int(source["layer"])
        source_idx = int(source["gate_idx"])
        active = apply_mode_bank(
            history[source_layer],
            dressed[source_idx],
            source["qubits"],
            ctx.n_qubits,
        )
        current_layer = source_layer
        for target in records[source_pos + 1 :]:
            target_layer = int(target["layer"])
            layer_distance = target_layer - source_layer
            if max_layer_distance is not None and layer_distance > max_layer_distance:
                break
            if target_layer > current_layer:
                active = propagate_layers(
                    active,
                    ctx,
                    current_layer + 1,
                    target_layer + 1,
                    residual_layers,
                )
                current_layer = target_layer
            target_idx = int(target["gate_idx"])
            qgh = positive_terminal_probabilities(
                local_rhos_after_mode_bank(
                    active,
                    dressed[target_idx],
                    target["qubits"],
                    ctx,
                    target_layer + 1,
                    mode_chunk_size,
                    residual_layers,
                )
            )
            target_pos = gate_to_pos[target_idx]
            pair_additive_c2 += qgh - qg[source_pos] - qg[target_pos] + q0
            denominator = qg[source_pos] * qg[target_pos]
            delta = torch.zeros_like(q0)
            delta[log_eligible] = (
                qgh[log_eligible] * q0[log_eligible] / denominator[log_eligible] - 1.0
            )
            pair_log_delta += delta
            pair_count += 1
            del qgh, delta

        next_source = source_pos + 1
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "format_version": FORMAT_VERSION,
                    "signature": signature,
                    "q0": q0.detach().cpu(),
                    "qg": qg.detach().cpu(),
                    "pair_log_delta": pair_log_delta.detach().cpu(),
                    "pair_additive_c2": pair_additive_c2.detach().cpu(),
                    "next_source": next_source,
                    "pair_count": pair_count,
                    "one_point_complete": True,
                },
                checkpoint_path,
            )
        if next_source % max(1, progress_every) == 0 or next_source == len(records):
            cuda_sync(device)
            print(
                f"[pair] source {next_source}/{len(records)} "
                f"pairs={pair_count}/{expected_pairs} "
                f"elapsed={time.time() - started:.1f}s "
                f"peak={peak_memory_gib(device):.2f} GiB",
                flush=True,
            )

    if pair_count != expected_pairs:
        raise AssertionError(f"pair count {pair_count} != expected {expected_pairs}")

    safe_q0 = q0.clamp_min(reference_floor)
    safe_qg = qg.clamp_min(reference_floor)
    log_q1 = torch.log(safe_q0) + torch.log(safe_qg / safe_q0.unsqueeze(0)).sum(dim=0)
    q1_log = torch.exp(log_q1)
    q2_log = torch.exp(log_q1 + pair_log_delta)
    q1_add = q0 + (qg - q0.unsqueeze(0)).sum(dim=0)
    q2_add = q1_add + pair_additive_c2
    q1 = torch.where(log_eligible, q1_log, q1_add)
    q2 = torch.where(log_eligible, q2_log, q2_add)
    cuda_sync(device)
    elapsed = time.time() - started

    return {
        "definition": {
            "terminal_effects": "positive p_plus/p_minus effects",
            "reported_expectation": "conditional on global computational-subspace survival",
            "log_reference_floor": float(reference_floor),
            "fallback": "additive connected expansion",
            "pair_scope": "all unordered gate pairs"
            if max_layer_distance is None
            else f"layer distance <= {max_layer_distance}",
        },
        "ideal": summarize_probabilities(q0),
        "first_order": summarize_probabilities(q1),
        "second_order": summarize_probabilities(q2),
        "raw": {
            "q0": q0.detach().cpu().tolist(),
            "q1": q1.detach().cpu().tolist(),
            "q2": q2.detach().cpu().tolist(),
            "pair_log_delta_sum": pair_log_delta.detach().cpu().tolist(),
            "pair_additive_c2_sum": pair_additive_c2.detach().cpu().tolist(),
            "log_eligible": log_eligible.detach().cpu().tolist(),
        },
        "diagnostics": {
            "n_qubits": int(ctx.n_qubits),
            "n_layers": int(len(ctx.layers)),
            "gate_count": int(len(records)),
            "pair_count": int(pair_count),
            "expected_pair_count": int(expected_pairs),
            "log_effect_count": int(log_eligible.sum().item()),
            "additive_effect_count": int((~log_eligible).sum().item()),
            "runtime_seconds": float(elapsed),
            "peak_cuda_memory_gib": peak_memory_gib(device),
            "target_mode_chunk_size": mode_chunk_size,
            "device": str(device),
            "dtype": str(ctx.dtype),
            "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
            "context_signature": signature,
            "common_residual_background": residual_layers is not None,
        },
    }


def exact_payload_from_qutrit(rho_local: torch.Tensor) -> dict[str, Any]:
    local_rhos, survival, trace = qutrit_global_computational_local_rhos(rho_local)
    probabilities = positive_terminal_probabilities(local_rhos)
    payload = summarize_probabilities(probabilities)
    payload["global_computational_survival"] = survival
    payload["qutrit_trace"] = trace
    return payload


def error_metrics(estimate: dict[str, Any], exact: dict[str, Any]) -> dict[str, Any]:
    est = np.asarray(estimate["pauli_conditional"], dtype=float)
    ref = np.asarray(exact["pauli_conditional"], dtype=float)
    error = np.abs(est - ref)
    bloch = np.linalg.norm(est - ref, axis=1)
    per_basis = {}
    for basis_idx, basis in enumerate(BASES):
        values = error[:, basis_idx]
        per_basis[basis] = {
            "mae": float(np.mean(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }
    p_est = np.stack(
        [
            np.asarray(estimate["p_plus"], dtype=float),
            np.asarray(estimate["p_minus"], dtype=float),
            1.0 - np.asarray(estimate["survival_by_basis"], dtype=float),
        ],
        axis=-1,
    )
    p_ref = np.stack(
        [
            np.asarray(exact["p_plus"], dtype=float),
            np.asarray(exact["p_minus"], dtype=float),
            1.0 - np.asarray(exact["survival_by_basis"], dtype=float),
        ],
        axis=-1,
    )
    tv = 0.5 * np.sum(np.abs(p_est - p_ref), axis=-1)
    return {
        "conditional_pauli_mae": float(np.mean(error)),
        "conditional_pauli_p95": float(np.percentile(error, 95)),
        "conditional_pauli_max": float(np.max(error)),
        "per_basis": per_basis,
        "bloch_vector_error_mean": float(np.mean(bloch)),
        "bloch_vector_error_p95": float(np.percentile(bloch, 95)),
        "bloch_vector_error_max": float(np.max(bloch)),
        "three_outcome_tv_mean": float(np.mean(tv)),
        "three_outcome_tv_p95": float(np.percentile(tv, 95)),
        "three_outcome_tv_max": float(np.max(tv)),
    }


def save_case(output_dir: Path, stem: str, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    methods: list[tuple[str, dict[str, Any]]] = [
        ("ideal", payload["full_state"]["ideal"]),
        ("first_order", payload["full_state"]["first_order"]),
        ("second_order", payload["full_state"]["second_order"]),
    ]
    if payload.get("exact_qutrit") is not None:
        methods.append(("exact_qutrit", payload["exact_qutrit"]))
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "method",
            "qubit",
            "basis",
            "pauli_conditional",
            "pauli_unconditional",
            "p_plus",
            "p_minus",
            "survival",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, record in methods:
            for qubit in range(int(payload["metadata"]["n_qubits"])):
                for basis_idx, basis in enumerate(BASES):
                    writer.writerow(
                        {
                            "method": method,
                            "qubit": qubit,
                            "basis": basis,
                            "pauli_conditional": record["pauli_conditional"][qubit][
                                basis_idx
                            ],
                            "pauli_unconditional": record["pauli_unconditional"][qubit][
                                basis_idx
                            ],
                            "p_plus": record["p_plus"][qubit][basis_idx],
                            "p_minus": record["p_minus"][qubit][basis_idx],
                            "survival": record["survival_by_basis"][qubit][basis_idx],
                        }
                    )
    print(f"[save] {json_path}", flush=True)
    print(f"[save] {csv_path}", flush=True)


def run_8q(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    width, length = 4, 2
    n_qubits = width * length
    patterns = qcore.generate_circuit_patterns(
        1,
        args.layers8,
        n_qubits,
        qcore.row_major_neighbors(width, length),
        args.seed8,
    )
    pattern = patterns[0]
    manifest, geometry = residual_bg.load_residual_geometry(
        args.residual_manifest, width, length
    )
    residual_steps = (
        args.piece_num
        if int(args.residual_integration_steps) == 0
        else int(args.residual_integration_steps)
    )
    residual_started = time.time()
    projected_residual = residual_bg.compile_pattern_residual(
        ResidualMagnusLibrary(
            n_qubits,
            integration_steps=residual_steps,
            dtype=torch.complex128,
            device=device,
            projected_qubits=True,
        ),
        pattern,
        geometry["edges"],
        scheme=args.kernel_scheme,
    )
    qutrit_residual = residual_bg.compile_pattern_residual(
        ResidualMagnusLibrary(
            n_qubits,
            integration_steps=residual_steps,
            dtype=torch.complex128,
            device=device,
            projected_qubits=False,
        ),
        pattern,
        geometry["edges"],
        scheme=args.kernel_scheme,
    )
    residual_compile_seconds = time.time() - residual_started
    print(
        f"[8q] compiled shared {args.kernel_scheme} residual background: "
        f"layers={args.layers8} edges/layer={len(geometry['edges'])} "
        f"seconds={residual_compile_seconds:.1f}",
        flush=True,
    )
    cache_path = Path(args.channel_cache)
    channels, channel_meta = load_or_build_channels(
        width,
        length,
        args.piece_num,
        args.noise_time,
        cache_path,
        device,
        torch.complex128,
    )
    print("[8q] exact qutrit Liouville", flush=True)
    exact_started = time.time()
    rho_qutrit = run_qutrit_liouville_final(
        pattern,
        channels,
        width,
        length,
        qutrit_residual,
    )
    exact = exact_payload_from_qutrit(rho_qutrit)
    exact_runtime = time.time() - exact_started
    del rho_qutrit
    if device.type == "cuda":
        torch.cuda.empty_cache()

    ctx = qcore.build_projected_context(
        pattern, width, length, channels, device, torch.complex128
    )
    checkpoint = Path(args.output_dir) / "pauli_8q_fullstate_checkpoint.pt"
    full_state = full_state_connected_pauli(
        ctx,
        reference_floor=args.reference_floor,
        max_layer_distance=args.max_layer_distance,
        checkpoint_path=checkpoint,
        resume=args.resume,
        progress_every=args.progress_every,
        mode_chunk_size=args.mode_chunk_size,
        residual_layers=projected_residual,
    )
    payload = {
        "metadata": {
            "case": "8-qutrit exact Liouville vs 8-qubit projected full-state C2",
            "n_qubits": n_qubits,
            "width": width,
            "length": length,
            "n_layers": int(args.layers8),
            "circuit_seed": int(args.seed8),
            "noise_time_ns": float(args.noise_time),
            "piece_num": int(args.piece_num),
            "channel_metadata": channel_meta,
            "exact_runtime_seconds": float(exact_runtime),
            "residual_manifest": str(
                Path(args.residual_manifest).expanduser().resolve()
            ),
            "residual_manifest_base_seed": int(manifest["base_seed"]),
            "residual_kernel_scheme": args.kernel_scheme,
            "residual_integration_steps": residual_steps,
            "residual_edges_per_layer": len(geometry["edges"]),
            "residual_compile_seconds": residual_compile_seconds,
            "background_layer_order": [
                "interaction_picture_residual",
                "intended_or_noisy_local_maps",
            ],
        },
        "exact_qutrit": exact,
        "full_state": full_state,
        "metrics": {
            "first_order_vs_exact": error_metrics(full_state["first_order"], exact),
            "second_order_vs_exact": error_metrics(full_state["second_order"], exact),
        },
    }
    save_case(Path(args.output_dir), "pauli_8q_exact_vs_fullstate", payload)
    return payload


def run_16q(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    reference_path = Path(args.reference16)
    reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    if int(reference["N_QUBITS"]) != 16:
        raise ValueError("the 16-qubit production reference must contain N_QUBITS=16")
    manifest, geometry = residual_bg.load_residual_geometry(
        args.residual_manifest, 4, 4
    )
    residual_steps = (
        args.piece_num
        if int(args.residual_integration_steps) == 0
        else int(args.residual_integration_steps)
    )
    residual_started = time.time()
    physical_residual = residual_bg.compile_pattern_residual(
        ResidualMagnusLibrary(
            16,
            integration_steps=residual_steps,
            dtype=torch.complex128,
            device=device,
            projected_qubits=True,
        ),
        reference["pattern"],
        geometry["edges"],
        scheme=args.kernel_scheme,
    )
    projected_residual = remap_residual_layers_to_reference_axes(
        physical_residual, reference
    )
    residual_compile_seconds = time.time() - residual_started
    print(
        f"[16q] compiled shared {args.kernel_scheme} residual background: "
        f"layers={len(projected_residual)} edges/layer={len(geometry['edges'])} "
        f"seconds={residual_compile_seconds:.1f}",
        flush=True,
    )
    ctx = build_reference_context(reference, device)
    checkpoint = Path(args.output_dir) / "pauli_16q_fullstate_checkpoint.pt"
    full_state = full_state_connected_pauli(
        ctx,
        reference_floor=args.reference_floor,
        max_layer_distance=args.max_layer_distance,
        checkpoint_path=checkpoint,
        resume=args.resume,
        progress_every=args.progress_every,
        mode_chunk_size=args.mode_chunk_size,
        residual_layers=projected_residual,
    )
    payload = {
        "metadata": {
            "case": "16-qubit projected full-state connected C2 only",
            "n_qubits": 16,
            "width": int(ctx.width),
            "length": int(ctx.length),
            "n_layers": int(len(ctx.layers)),
            "noise_time_ns": 5000.0,
            "piece_num": 1000,
            "source_reference": str(reference_path),
            "physical_to_axis": {
                str(int(physical)): int(axis)
                for physical, axis in reference["phys_to_1d"].items()
            },
            "residual_manifest": str(
                Path(args.residual_manifest).expanduser().resolve()
            ),
            "residual_manifest_base_seed": int(manifest["base_seed"]),
            "residual_kernel_scheme": args.kernel_scheme,
            "residual_integration_steps": residual_steps,
            "residual_edges_per_layer": len(geometry["edges"]),
            "residual_compile_seconds": residual_compile_seconds,
            "background_layer_order": [
                "interaction_picture_residual",
                "intended_or_noisy_local_maps",
            ],
            "exact_liouville": False,
            "exact_liouville_reason": "9^16 qutrit Liouville state is intentionally not materialized",
        },
        "exact_qutrit": None,
        "full_state": full_state,
        "metrics": None,
    }
    save_case(Path(args.output_dir), "pauli_16q_fullstate", payload)
    return payload


def run_24q(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    width, length = 6, 4
    n_qubits = width * length
    pattern = qcore.generate_circuit_patterns(
        1,
        args.layers24,
        n_qubits,
        qcore.row_major_neighbors(width, length),
        args.seed24,
    )[0]
    channels, channel_meta = load_or_build_channels(
        width,
        length,
        args.piece_num,
        args.noise_time,
        Path(args.channel_cache24),
        device,
        torch.complex128,
    )
    ctx = qcore.build_projected_context(
        pattern, width, length, channels, device, torch.complex128
    )
    checkpoint = Path(args.output_dir) / "pauli_24q_6x4_fullstate_checkpoint.pt"
    full_state = full_state_connected_pauli(
        ctx,
        reference_floor=args.reference_floor,
        max_layer_distance=args.max_layer_distance,
        checkpoint_path=checkpoint,
        resume=args.resume,
        progress_every=args.progress_every,
        mode_chunk_size=args.mode_chunk_size,
    )
    payload = {
        "metadata": {
            "case": "24-qubit 6x4 projected full-state connected C2 only",
            "n_qubits": n_qubits,
            "width": width,
            "length": length,
            "n_layers": int(args.layers24),
            "circuit_seed": int(args.seed24),
            "noise_time_ns": float(args.noise_time),
            "piece_num": int(args.piece_num),
            "channel_metadata": channel_meta,
            "exact_liouville": False,
            "exact_liouville_reason": "9^24 qutrit Liouville state is intentionally not materialized",
        },
        "exact_qutrit": None,
        "full_state": full_state,
        "metrics": None,
    }
    save_case(Path(args.output_dir), "pauli_24q_6x4_fullstate", payload)
    return payload


def smoke_test(args: argparse.Namespace) -> None:
    device = torch.device("cpu")
    width, length, layers = 2, 1, 2
    pattern = qcore.generate_circuit_patterns(
        1,
        layers,
        width * length,
        qcore.row_major_neighbors(width, length),
        17,
    )[0]
    cache = Path(args.output_dir) / "smoke_channels.pt"
    channels, _ = load_or_build_channels(
        width, length, 8, 5000.0, cache, device, torch.complex128
    )
    rho = run_qutrit_liouville_final(pattern, channels, width, length)
    exact = exact_payload_from_qutrit(rho)
    if abs(float(exact["qutrit_trace"]) - 1.0) > 2e-7:
        raise AssertionError(f"qutrit trace drift: {exact['qutrit_trace']}")
    ctx = qcore.build_projected_context(
        pattern, width, length, channels, device, torch.complex128
    )
    result = full_state_connected_pauli(
        ctx,
        reference_floor=1e-10,
        max_layer_distance=None,
        checkpoint_path=None,
        resume=False,
        progress_every=1,
        mode_chunk_size=2,
    )
    if result["diagnostics"]["pair_count"] != 3:
        raise AssertionError(result["diagnostics"])
    q0_survival = np.asarray(result["ideal"]["survival_by_basis"])
    if not np.allclose(q0_survival, 1.0, atol=1e-10):
        raise AssertionError("ideal terminal probabilities are not normalized")
    print(
        json.dumps(
            {
                "smoke": "PASS",
                "qutrit_trace": exact["qutrit_trace"],
                "qutrit_global_comp_survival": exact["global_computational_survival"],
                "pair_count": result["diagnostics"]["pair_count"],
                "second_order_probability_min": result["second_order"][
                    "probability_min"
                ],
            },
            indent=2,
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    data_dir = PACKAGE_ROOT / "data" / "pauli_observables"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run8", action="store_true")
    parser.add_argument("--run16", action="store_true")
    parser.add_argument("--run24", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--allow-cpu-production",
        action="store_true",
        help="allow the production calculation on a CPU-only local machine",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--layers8", type=int, default=20)
    parser.add_argument("--seed8", type=int, default=20260718)
    parser.add_argument("--layers24", type=int, default=20)
    parser.add_argument("--seed24", type=int, default=20260718)
    parser.add_argument("--noise-time", type=float, default=5000.0)
    parser.add_argument("--piece-num", type=int, default=1000)
    parser.add_argument("--residual-manifest", default=str(DEFAULT_RESIDUAL_MANIFEST))
    parser.add_argument(
        "--kernel-scheme",
        choices=("dyson-1", "magnus-1"),
        default="magnus-1",
    )
    parser.add_argument(
        "--residual-integration-steps",
        type=int,
        default=0,
        help="0 reuses --piece-num for the offline residual integration",
    )
    parser.add_argument("--reference-floor", type=float, default=1e-10)
    parser.add_argument("--max-layer-distance", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument(
        "--mode-chunk-size",
        type=int,
        default=None,
        help="target Kraus modes per exact full-state chunk; use 4 for 6x4 on A100",
    )
    parser.add_argument("--output-dir", default=str(data_dir))
    parser.add_argument(
        "--channel-cache",
        default=str(data_dir / "qutrit_channels_4x2_T5000_piece1000.pt"),
    )
    parser.add_argument(
        "--reference16",
        default=str(INPUTS / "0522_reference_data.pt"),
    )
    parser.add_argument(
        "--channel-cache24",
        default=str(data_dir / "qutrit_channels_6x4_T5000_piece1000.pt"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke:
        smoke_test(args)
        return
    run8 = bool(args.run8 or args.run_all)
    run16 = bool(args.run16 or args.run_all)
    run24 = bool(args.run24 or args.run_all)
    if not run8 and not run16 and not run24:
        build_parser().print_help()
        return
    device = torch.device(args.device)
    if (
        device.type == "cpu"
        and (run8 or run16 or run24)
        and not args.allow_cpu_production
    ):
        raise RuntimeError(
            "Production full-state C2 runs require --device cuda unless "
            "--allow-cpu-production is set explicitly."
        )
    if run8:
        run_8q(args, device)
    if run16:
        run_16q(args, device)
    if run24:
        run_24q(args, device)


if __name__ == "__main__":
    main()
