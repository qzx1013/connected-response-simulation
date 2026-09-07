#!/usr/bin/env python
"""Exact 4x4, 20-layer connected-order ensemble audit.

The qutrit channels are not recompiled.  This script reuses the projected
qubit Kraus operators exported from the 1000-slice 0522 calculation, generates
independent random circuits, evaluates every one- and two-location response
with the exact full-state backend, and evaluates the complete distinct-layer
third-order sum by streaming full-state branch banks.

The reported third order contains triples with a < b < c only.  Same-layer
and mixed-layer triples are deliberately excluded and are labelled as such in
every output file.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np
import torch


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PEPS_CORR2BP_PAIR_FORMULA", "centered_C")
os.environ.setdefault("CENTERED_FIDELITY_EXACT_BRANCH_CHUNK_MODES", "128")

from residual_tn.paths import CONFIGS, INPUTS, RESULTS

PACKAGE_ROOT = RESULTS
from residual_tn.backend.context import FidelityContext  # noqa: E402
from residual_tn.backend.exact_small import exact_compute_fidelity  # noqa: E402
from residual_tn.experiments.cumulant_exact import (  # noqa: E402
    apply_event_modes_to_branches,
    apply_event_modes_to_one_state,
    build_event_layers,
    compute_c2_audit,
    pair_response,
    propagate_ideal,
    readout_event_sum_over_modes,
)


GateId = Union[int, Tuple[int, int]]
PairKey = Tuple[int, GateId, int, GateId]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
    temp.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pattern_sha256(pattern: Sequence[Any]) -> str:
    text = json.dumps(json_safe(pattern), separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_gate_id(value: Any) -> GateId:
    if isinstance(value, np.generic):
        return int(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return int(value.item())
        return tuple(int(x) for x in value.detach().cpu().flatten().tolist())
    if isinstance(value, (list, tuple)):
        return tuple(int(x) for x in value)
    return int(value)


def row_major_neighbors(width: int, length: int) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    for row in range(length):
        for col in range(width):
            q = row * width + col
            if col + 1 < width:
                edges.append((q, q + 1))
            if row + 1 < length:
                edges.append((q, q + width))
    return edges


def generate_random_circuit_patterns(
    n_circuits: int,
    n_layers: int,
    n_qubits: int,
    neighbors: Sequence[Tuple[int, int]],
    seed: int,
) -> List[List[Any]]:
    """Match the random-circuit generator used by physics_0522.py."""
    np_rng = np.random.RandomState(seed)
    py_rng = random.Random(seed)
    single_pool = np_rng.randint(0, 4, size=(n_circuits, n_layers, n_qubits))
    patterns: List[List[Any]] = []
    for circuit in range(n_circuits):
        layers: List[Any] = []
        for layer in range(n_layers):
            if layer % 2 == 0:
                layers.append(single_pool[circuit, layer, :].copy())
                continue
            shuffled = list(neighbors)
            py_rng.shuffle(shuffled)
            used: set[int] = set()
            selected: List[List[int]] = []
            for q1, q2 in shuffled:
                if q1 not in used and q2 not in used:
                    selected.append([int(q1), int(q2)])
                    used.add(int(q1))
                    used.add(int(q2))
            layers.append(selected)
        patterns.append(layers)
    return patterns


def axis_for_physical(reference: Dict[str, Any], physical: int) -> int:
    mapping = reference["phys_to_1d"]
    if isinstance(mapping, dict):
        if physical in mapping:
            return int(mapping[physical])
        return int(mapping[str(physical)])
    return int(mapping[physical])


def pair_index(reference: Dict[str, Any], q1: int, q2: int) -> int:
    key = tuple(sorted((int(q1), int(q2))))
    mapping = reference["pair_to_idx"]
    if key in mapping:
        return int(mapping[key])
    for raw, value in mapping.items():
        parsed = ast.literal_eval(raw) if isinstance(raw, str) else raw
        if tuple(sorted(int(x) for x in parsed)) == key:
            return int(value)
    raise KeyError(key)


def build_context(
    reference: Dict[str, Any],
    pattern: Sequence[Any],
    device: torch.device,
) -> Tuple[FidelityContext, Dict[int, Tuple[int, GateId]]]:
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
    gate_meta: Dict[int, Tuple[int, GateId]] = {}
    gate_idx = 0
    for layer_idx, layer_pattern in enumerate(pattern):
        gates: List[Dict[str, Any]] = []
        if layer_idx % 2 == 0:
            for physical in range(n_qubits):
                mode = int(layer_pattern[physical])
                gates.append(
                    {
                        "gate_idx": gate_idx,
                        "qubits": (axis_for_physical(reference, physical),),
                        "ideal_unitary": ideal_single[mode],
                        "kraus_ops": single[(physical, mode)],
                    }
                )
                gate_meta[gate_idx] = (layer_idx, int(physical))
                gate_idx += 1
        else:
            for raw_pair in layer_pattern:
                q1, q2 = int(raw_pair[0]), int(raw_pair[1])
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
                gate_meta[gate_idx] = (layer_idx, (q1, q2))
                gate_idx += 1
        ctx.add_layer(gates)
    return ctx, gate_meta


def apply_1q_state(
    state: torch.Tensor,
    gate: torch.Tensor,
    q: int,
    n_qubits: int,
) -> torch.Tensor:
    tensor = state.reshape(*([2] * n_qubits))
    out = torch.tensordot(gate, tensor, dims=([1], [q]))
    return torch.movedim(out, 0, q).reshape(-1)


def apply_2q_state(
    state: torch.Tensor,
    gate: torch.Tensor,
    q1: int,
    q2: int,
    n_qubits: int,
) -> torch.Tensor:
    tensor = state.reshape(*([2] * n_qubits))
    matrix = gate.reshape(2, 2, 2, 2)
    out = torch.tensordot(matrix, tensor, dims=([2, 3], [q1, q2]))
    permutation: List[int] = []
    other = 0
    for position in range(n_qubits):
        if position == q1:
            permutation.append(0)
        elif position == q2:
            permutation.append(1)
        else:
            permutation.append(2 + other)
            other += 1
    return out.permute(*permutation).reshape(-1)


def ideal_state_history(ctx: FidelityContext) -> List[torch.Tensor]:
    n_qubits = int(ctx.n_qubits)
    state = torch.zeros(2**n_qubits, dtype=ctx.dtype, device=ctx.device)
    state[0] = 1.0
    history = [state.unsqueeze(0)]
    for layer in ctx.layers:
        for gate in layer:
            qubits = tuple(int(x) for x in gate["qubits"])
            if len(qubits) == 1:
                state = apply_1q_state(
                    state, gate["ideal_unitary"], qubits[0], n_qubits
                )
            else:
                state = apply_2q_state(
                    state, gate["ideal_unitary"], qubits[0], qubits[1], n_qubits
                )
        history.append(state.unsqueeze(0))
    return history


def exact_pair_payload(
    ctx: FidelityContext,
    gate_meta: Dict[int, Tuple[int, GateId]],
) -> Tuple[Any, Dict[Tuple[int, GateId], float], Dict[PairKey, float]]:
    result = exact_compute_fidelity(
        ctx,
        max_layer_distance=None,
        pair_selection="all",
        store_pair_C=False,
        pair_C_to_cpu=True,
    )
    local: Dict[Tuple[int, GateId], float] = {}
    for gate_idx, value in result.local_fidelities.items():
        layer, gate_id = gate_meta[int(gate_idx)]
        local[(int(layer), canonical_gate_id(gate_id))] = float(value)

    pair_response_values: Dict[PairKey, float] = {}
    for (left_idx, right_idx), delta in result.pair_deltas.items():
        left_layer, left_gate = gate_meta[int(left_idx)]
        right_layer, right_gate = gate_meta[int(right_idx)]
        left_gate = canonical_gate_id(left_gate)
        right_gate = canonical_gate_id(right_gate)
        ma = local[(left_layer, left_gate)]
        mb = local[(right_layer, right_gate)]
        ratio = 1.0 + float(delta)
        if ratio <= 0:
            raise RuntimeError(
                "non-positive exact pair ratio for gates %s and %s"
                % (left_idx, right_idx)
            )
        pair_response_values[(left_layer, left_gate, right_layer, right_gate)] = (
            ratio * ma * mb
        )
    return result, local, pair_response_values


def layer_pair_rows(
    c2_rows: Sequence[Dict[str, Any]], n_layers: int
) -> List[Dict[str, Any]]:
    bins: Dict[Tuple[int, int], Dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for row in c2_rows:
        key = (int(row["src_layer"]), int(row["tgt_layer"]))
        bins[key]["count"] += 1
        bins[key]["signed_kappa2"] += float(row["kappa_log"])
        bins[key]["abs_kappa2"] += abs(float(row["kappa_log"]))
    output: List[Dict[str, Any]] = []
    for source in range(n_layers):
        for target in range(source, n_layers):
            values = bins.get((source, target), {})
            output.append(
                {
                    "source_layer": source,
                    "target_layer": target,
                    "pair_count": int(values.get("count", 0)),
                    "signed_kappa2": float(values.get("signed_kappa2", 0.0)),
                    "abs_kappa2": float(values.get("abs_kappa2", 0.0)),
                }
            )
    return output


def lightcone_metrics(
    c2_rows: Sequence[Dict[str, Any]], summary: Dict[str, Any]
) -> Dict[str, float]:
    total_signed = float(summary.get("sum_log_all", 0.0))
    total_abs = float(summary.get("sum_abs_log_all", 0.0))
    inside_signed = float(summary.get("sum_log_inside", 0.0))
    inside_abs = float(summary.get("sum_abs_log_inside", 0.0))
    outside_signed = float(summary.get("sum_log_outside", 0.0))
    outside_abs = float(summary.get("sum_abs_log_outside", 0.0))
    output = {
        "c2_signed": total_signed,
        "c2_abs_mass": total_abs,
        "inside_signed": inside_signed,
        "outside_signed": outside_signed,
        "inside_abs_fraction": inside_abs / max(total_abs, 1e-300),
        "outside_abs_fraction": outside_abs / max(total_abs, 1e-300),
        "outside_signed_over_total": outside_signed / total_signed
        if total_signed != 0
        else float("nan"),
    }
    max_gap = max((int(row["cone_gap"]) for row in c2_rows), default=0)
    for radius in range(max(6, max_gap) + 1):
        tail = [row for row in c2_rows if int(row["cone_gap"]) > radius]
        tail_abs = sum(abs(float(row["kappa_log"])) for row in tail)
        tail_signed = sum(float(row["kappa_log"]) for row in tail)
        output[f"tail_abs_fraction_R{radius}"] = tail_abs / max(total_abs, 1e-300)
        output[f"tail_signed_over_total_R{radius}"] = (
            tail_signed / total_signed if total_signed != 0 else float("nan")
        )
    return output


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_csv_gz(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compute_distinct_layer_c3(
    reference: Dict[str, Any],
    local_f: Dict[Tuple[int, GateId], float],
    m_pair: Dict[PairKey, float],
    device: torch.device,
    state_chunk: int,
    checkpoint_path: Path,
    circuit: int,
    pattern_hash: str,
    resume: bool,
    max_source_layers: int,
    max_middle_per_source: int,
    max_target_layers: int,
    connected_threshold: float,
    record_layer_triples: bool,
) -> Tuple[Dict[str, float], List[Dict[str, Any]], bool]:
    n_layers = int(reference["N_LAYERS"])
    n_qubits = int(reference["N_QUBITS"])
    source_layers = list(range(n_layers - 2))
    if max_source_layers > 0:
        source_layers = source_layers[:max_source_layers]

    global_summary: Dict[str, float] = defaultdict(float)
    layer_summaries: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    completed_sources: List[int] = []
    if resume and checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("pattern_sha256") == pattern_hash:
            global_summary.update(
                {
                    str(k): float(v)
                    for k, v in checkpoint.get("global_summary", {}).items()
                }
            )
            completed_sources = [
                int(x) for x in checkpoint.get("completed_source_layers", [])
            ]
            for row in checkpoint.get("layer_triples", []):
                key = (int(row["layer_a"]), int(row["layer_b"]), int(row["layer_c"]))
                layer_summaries[key] = row
            print(
                "[resume-c3] circuit=%02d completed_sources=%s"
                % (circuit, completed_sources),
                flush=True,
            )

    event_layers, _ = build_event_layers(reference, device)
    ideal_states = [
        state.to(device=device, dtype=torch.complex128)
        for state in reference["ideal_state_history"]
    ]
    started = time.time()

    for source_position, a in enumerate(source_layers):
        if a in completed_sources:
            continue
        events_a = event_layers[a]
        source_parts: List[torch.Tensor] = []
        source_event_indices: List[int] = []
        psi_after_a = ideal_states[a + 1]
        for event_index, event_a in enumerate(events_a):
            branches = apply_event_modes_to_one_state(psi_after_a, event_a, n_qubits)
            source_parts.append(branches)
            source_event_indices.extend([event_index] * int(branches.shape[0]))
        source_states = torch.cat(source_parts, dim=0)
        source_index = torch.tensor(
            source_event_indices, dtype=torch.long, device=device
        )

        middle_done = 0
        for b in range(a + 1, n_layers - 1):
            if max_middle_per_source > 0 and middle_done >= max_middle_per_source:
                break
            source_states = propagate_ideal(source_states, b, b, reference, state_chunk)
            middle_done += 1
            events_b = event_layers[b]
            for event_b in events_b:
                states_ab = apply_event_modes_to_branches(
                    source_states, event_b, n_qubits
                )
                rank_b = int(event_b.kraus.shape[0])
                source_index_ab = source_index.repeat_interleave(rank_b)
                current_states = states_ab
                target_seen = 0
                for c in range(b + 1, n_layers):
                    if max_target_layers > 0 and target_seen >= max_target_layers:
                        break
                    current_states = propagate_ideal(
                        current_states, c, c, reference, state_chunk
                    )
                    target_seen += 1
                    psi_after_c = ideal_states[c + 1]
                    events_c = event_layers[c]
                    layer_key = (a, b, c)
                    if record_layer_triples and layer_key not in layer_summaries:
                        layer_summaries[layer_key] = {
                            "layer_a": a,
                            "layer_b": b,
                            "layer_c": c,
                            "count": 0,
                            "sum_log_c3": 0.0,
                            "sum_abs_log_c3": 0.0,
                            "max_abs_log_c3": 0.0,
                        }
                    for event_c in events_c:
                        per_branch = readout_event_sum_over_modes(
                            psi_after_c, current_states, event_c, n_qubits
                        )
                        by_source = torch.zeros(
                            len(events_a), dtype=torch.float64, device=device
                        )
                        by_source.index_add_(
                            0, source_index_ab, per_branch.real.to(torch.float64)
                        )
                        values = by_source.detach().cpu().numpy()
                        for event_index, event_a in enumerate(events_a):
                            mabc = float(values[event_index])
                            ma = float(local_f[(a, event_a.gate_id)])
                            mb = float(local_f[(b, event_b.gate_id)])
                            mc = float(local_f[(c, event_c.gate_id)])
                            mab = float(pair_response(m_pair, event_a, event_b))
                            mac = float(pair_response(m_pair, event_a, event_c))
                            mbc = float(pair_response(m_pair, event_b, event_c))
                            if min(mabc, ma, mb, mc, mab, mac, mbc) <= 0:
                                continue
                            ratio3 = (mabc * ma * mb * mc) / (mab * mac * mbc)
                            if ratio3 <= 0:
                                continue
                            k3 = math.log(ratio3)
                            k2_ab = math.log(mab / (ma * mb))
                            k2_ac = math.log(mac / (ma * mc))
                            k2_bc = math.log(mbc / (mb * mc))
                            abs_k3 = abs(k3)
                            global_summary["count"] += 1
                            global_summary["sum_log_c3"] += k3
                            global_summary["sum_abs_log_c3"] += abs_k3
                            global_summary["max_abs_log_c3"] = max(
                                global_summary["max_abs_log_c3"], abs_k3
                            )
                            edges = (
                                int(abs(k2_ab) >= connected_threshold)
                                + int(abs(k2_ac) >= connected_threshold)
                                + int(abs(k2_bc) >= connected_threshold)
                            )
                            global_summary[f"edge_count_{edges}_abs"] += abs_k3
                            global_summary[f"edge_count_{edges}_count"] += 1
                            if record_layer_triples:
                                layer_summary = layer_summaries[layer_key]
                                layer_summary["count"] += 1
                                layer_summary["sum_log_c3"] += k3
                                layer_summary["sum_abs_log_c3"] += abs_k3
                                layer_summary["max_abs_log_c3"] = max(
                                    layer_summary["max_abs_log_c3"], abs_k3
                                )
                del current_states, states_ab

        completed_sources.append(a)
        checkpoint_payload = {
            "circuit": circuit,
            "pattern_sha256": pattern_hash,
            "completed_source_layers": completed_sources,
            "requested_source_layers": source_layers,
            "global_summary": dict(global_summary),
            "layer_triples": [layer_summaries[key] for key in sorted(layer_summaries)],
            "elapsed_this_attempt_sec": time.time() - started,
            "distinct_layer_only": True,
            "same_layer_triples_included": False,
            "mixed_layer_triples_included": False,
        }
        atomic_json(checkpoint_path, checkpoint_payload)
        print(
            "[c3-source] circuit=%02d source=%02d (%d/%d) sum=%+.6e abs=%.6e elapsed=%.1fs"
            % (
                circuit,
                a,
                source_position + 1,
                len(source_layers),
                global_summary["sum_log_c3"],
                global_summary["sum_abs_log_c3"],
                time.time() - started,
            ),
            flush=True,
        )

    complete = (
        max_source_layers == 0
        and max_middle_per_source == 0
        and max_target_layers == 0
        and sorted(completed_sources) == list(range(n_layers - 2))
    )
    return (
        dict(global_summary),
        [layer_summaries[key] for key in sorted(layer_summaries)],
        complete,
    )


def result_row(
    circuit: int,
    pattern_hash: str,
    exact_result: Any,
    c2_summary: Dict[str, Any],
    lightcone: Dict[str, float],
    c3_summary: Dict[str, float] | None,
    c3_complete: bool,
    runtime: Dict[str, float],
) -> Dict[str, Any]:
    c1 = float(exact_result.log_fidelity_first_order)
    c2 = float(c2_summary["sum_log_all"])
    row: Dict[str, Any] = {
        "circuit": circuit,
        "pattern_sha256": pattern_hash,
        "C1_signed_log": c1,
        "C1_magnitude": abs(c1),
        "C2_signed_log": c2,
        "C2_magnitude": abs(c2),
        "C2_abs_mass": float(c2_summary["sum_abs_log_all"]),
        "C2_over_C1_magnitude": abs(c2) / max(abs(c1), 1e-300),
        "pair_count": int(c2_summary["count"]),
        "fidelity_F1": float(exact_result.fidelity_first_order),
        "fidelity_F2_log_cumulant": float(math.exp(c1 + c2)),
        "c3_complete": bool(c3_complete),
        **lightcone,
        **runtime,
    }
    if c3_summary is not None:
        c3 = float(c3_summary.get("sum_log_c3", 0.0))
        c3_abs = float(c3_summary.get("sum_abs_log_c3", 0.0))
        row.update(
            {
                "C3_distinct_signed_log": c3,
                "C3_distinct_magnitude": abs(c3),
                "C3_distinct_abs_mass": c3_abs,
                "C3_over_C2_magnitude": abs(c3) / max(abs(c2), 1e-300),
                "C3_abs_mass_over_C2_abs_mass": c3_abs
                / max(float(c2_summary["sum_abs_log_all"]), 1e-300),
                "fidelity_F3_distinct_log_cumulant": float(math.exp(c1 + c2 + c3)),
                "c3_event_triple_count": int(c3_summary.get("count", 0)),
                "c3_max_abs_single_term": float(c3_summary.get("max_abs_log_c3", 0.0)),
            }
        )
    return row


def write_aggregate_outputs(
    outdir: Path,
    metadata: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    started: float,
) -> None:
    ordered = sorted(rows, key=lambda row: int(row["circuit"]))
    payload = {
        "metadata": {
            **metadata,
            "completed_circuits": [int(row["circuit"]) for row in ordered],
            "elapsed_sec": time.time() - started,
        },
        "circuits": ordered,
    }
    atomic_json(outdir / "analysis_summary.json", payload)
    if ordered:
        all_fields: List[str] = []
        for row in ordered:
            for key in row:
                if key not in all_fields:
                    all_fields.append(key)
        normalized = [{key: row.get(key, "") for key in all_fields} for row in ordered]
        write_csv(outdir / "connected_order_20circuits.csv", normalized)
        lightcone_fields = [
            key
            for key in all_fields
            if key
            in {
                "circuit",
                "pattern_sha256",
                "C2_signed_log",
                "C2_abs_mass",
                "inside_signed",
                "outside_signed",
                "inside_abs_fraction",
                "outside_abs_fraction",
                "outside_signed_over_total",
            }
            or key.startswith("tail_")
        ]
        lightcone_rows = [
            {key: row.get(key, "") for key in lightcone_fields} for row in ordered
        ]
        write_csv(outdir / "lightcone_20circuits.csv", lightcone_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, default=Path("0522_reference_data.pt")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-circuits", type=int, default=20)
    parser.add_argument("--n-layers", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument(
        "--pattern-source",
        choices=("generated", "reference"),
        default="generated",
        help="Use fresh random circuits or the single pattern stored in the reference file.",
    )
    parser.add_argument("--c2-only", action="store_true")
    parser.add_argument("--state-chunk", type=int, default=2048)
    parser.add_argument("--connected-threshold", type=float, default=1e-6)
    parser.add_argument("--max-c3-source-layers", type=int, default=0)
    parser.add_argument("--max-c3-middle-per-source", type=int, default=0)
    parser.add_argument("--max-c3-target-layers", type=int, default=0)
    parser.add_argument("--start-circuit", type=int, default=0)
    parser.add_argument("--stop-circuit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    started = time.time()
    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable; refusing a silent CPU fallback"
        )
    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    print("[setup] device=%s" % device, flush=True)
    if device.type == "cuda":
        print("[setup] gpu=%s" % torch.cuda.get_device_name(device), flush=True)

    reference_path = args.reference.resolve()
    if not reference_path.exists():
        raise FileNotFoundError(args.reference)
    reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    if int(reference["N_QUBITS"]) != 16:
        raise ValueError("this ensemble audit requires the 4x4 (16-qubit) reference")
    if int(args.n_layers) != 20:
        raise ValueError("the requested manuscript audit is fixed to 20 layers")

    if args.pattern_source == "reference":
        patterns = [list(reference["pattern"][: args.n_layers])]
    else:
        patterns = generate_random_circuit_patterns(
            args.n_circuits,
            args.n_layers,
            int(reference["N_QUBITS"]),
            row_major_neighbors(4, 4),
            args.seed,
        )
    atomic_json(
        outdir / "random_circuit_patterns.json",
        {
            "seed": args.seed,
            "pattern_source": args.pattern_source,
            "n_circuits": len(patterns),
            "n_layers": args.n_layers,
            "patterns": patterns,
        },
    )

    metadata = {
        "experiment": "4x4 20-layer exact connected-order ensemble audit",
        "width": 4,
        "length": 4,
        "n_qubits": 16,
        "n_layers": 20,
        "n_circuits_requested": len(patterns),
        "pattern_source": args.pattern_source,
        "random_seed": args.seed,
        "reference_file": reference_path.name,
        "reference_sha256": file_sha256(reference_path),
        "channel_source": "0522 projected-qubit Kraus operators compiled from 1000 qutrit Liouville slices",
        "noise_time_ns": 5000,
        "piece_num": 1000,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "pair_order": "kappa2 = log(M_ab/(M_a M_b)) from all exact full-state gate pairs",
        "third_order": "complete a<b<c distinct-layer log cumulant; same-layer and mixed-layer triples excluded",
        "lightcone": "forward physical support through intervening two-qubit layers; R tail means cone_gap > R",
        "c2_only": bool(args.c2_only),
        "state_chunk": int(args.state_chunk),
        "connected_threshold": float(args.connected_threshold),
        "c3_caps": {
            "max_source_layers": int(args.max_c3_source_layers),
            "max_middle_per_source": int(args.max_c3_middle_per_source),
            "max_target_layers": int(args.max_c3_target_layers),
        },
    }

    existing_rows: List[Dict[str, Any]] = []
    summary_path = outdir / "analysis_summary.json"
    if args.resume and summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        existing_rows = list(previous.get("circuits", []))
    rows_by_circuit = {int(row["circuit"]): row for row in existing_rows}

    stop = args.stop_circuit if args.stop_circuit > 0 else len(patterns)
    selected = range(max(0, args.start_circuit), min(stop, len(patterns)))
    for circuit in selected:
        pattern = patterns[circuit]
        p_hash = pattern_sha256(pattern)
        previous_row = rows_by_circuit.get(circuit)
        if previous_row is not None and (
            args.c2_only or bool(previous_row.get("c3_complete", False))
        ):
            print("[skip] circuit=%02d already complete" % circuit, flush=True)
            continue

        circuit_started = time.time()
        ctx, gate_meta = build_context(reference, pattern, device)
        history = ideal_state_history(ctx)
        runtime_reference = dict(reference)
        runtime_reference["pattern"] = pattern
        runtime_reference["ideal_state_history"] = history
        runtime_reference["N_LAYERS"] = args.n_layers

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        pair_started = time.time()
        exact_result, local_f, m_pair = exact_pair_payload(ctx, gate_meta)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        pair_seconds = time.time() - pair_started
        c2_rows, c2_summary = compute_c2_audit(runtime_reference, local_f, m_pair, 4)
        lightcone = lightcone_metrics(c2_rows, c2_summary)

        if circuit == 0:
            write_csv_gz(outdir / "sample00_c2_event_pairs.csv.gz", c2_rows)
            write_csv(
                outdir / "sample00_layer_pair_c2.csv",
                layer_pair_rows(c2_rows, args.n_layers),
            )

        c3_summary: Dict[str, float] | None = None
        c3_rows: List[Dict[str, Any]] = []
        c3_complete = False
        c3_seconds = 0.0
        if not args.c2_only:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            c3_started = time.time()
            checkpoint_path = outdir / ("circuit_%02d_c3_checkpoint.json" % circuit)
            c3_summary, c3_rows, c3_complete = compute_distinct_layer_c3(
                runtime_reference,
                local_f,
                m_pair,
                device,
                args.state_chunk,
                checkpoint_path,
                circuit,
                p_hash,
                args.resume,
                args.max_c3_source_layers,
                args.max_c3_middle_per_source,
                args.max_c3_target_layers,
                args.connected_threshold,
                record_layer_triples=(circuit == 0),
            )
            c3_seconds = time.time() - c3_started
            if circuit == 0 and c3_rows:
                write_csv(outdir / "sample00_distinct_c3_by_layer_triple.csv", c3_rows)

        runtime = {
            "runtime_exact_C1_C2_sec": pair_seconds,
            "runtime_distinct_C3_sec": c3_seconds,
            "runtime_total_circuit_sec": time.time() - circuit_started,
        }
        row = result_row(
            circuit,
            p_hash,
            exact_result,
            c2_summary,
            lightcone,
            c3_summary,
            c3_complete,
            runtime,
        )
        rows_by_circuit[circuit] = row
        atomic_json(
            outdir / ("circuit_%02d_summary.json" % circuit),
            {
                "metadata": metadata,
                "result": row,
                "c2_summary": c2_summary,
                "c3_summary": c3_summary,
            },
        )
        write_aggregate_outputs(
            outdir, metadata, list(rows_by_circuit.values()), started
        )
        print(
            "[circuit-done] %02d C1=%+.6e C2=%+.6e C3=%s pair=%.1fs c3=%.1fs"
            % (
                circuit,
                row["C1_signed_log"],
                row["C2_signed_log"],
                ("%+.6e" % row["C3_distinct_signed_log"])
                if "C3_distinct_signed_log" in row
                else "skipped",
                pair_seconds,
                c3_seconds,
            ),
            flush=True,
        )
        del exact_result, ctx, history, runtime_reference, local_f, m_pair, c2_rows
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_aggregate_outputs(outdir, metadata, list(rows_by_circuit.values()), started)
    print(
        "[done] completed=%d output=%s elapsed=%.1fs"
        % (len(rows_by_circuit), outdir, time.time() - started),
        flush=True,
    )


if __name__ == "__main__":
    main()
