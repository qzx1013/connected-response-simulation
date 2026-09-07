#!/usr/bin/env python
"""4x4 20-layer cumulant audit built from exported 0522 data.

This script intentionally does not import or modify 0522_unified.py. It reuses
0522_reference_data.pt for the circuit, ideal state history, Kraus operators and
ideal gates, and M_pair_from_0522.pt for the exact two-location branch responses.

Outputs:
  - C2 all-pair versus support-retained summaries.
  - Exact C3 log-cumulants for one selected layer triple.
  - CSV/JSON data and one manuscript-ready diagnostic figure.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import ast
import csv
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GateId = Union[int, Tuple[int, int]]
PairKey = Tuple[int, GateId, int, GateId]


def canonical_gate_id(gate_id) -> GateId:
    if isinstance(gate_id, np.generic):
        return int(gate_id)
    if isinstance(gate_id, torch.Tensor):
        if gate_id.ndim == 0:
            return int(gate_id.item())
        return tuple(int(x) for x in gate_id.detach().cpu().flatten().tolist())
    if isinstance(gate_id, (list, tuple)):
        return tuple(int(x) for x in gate_id)
    return int(gate_id)


def gate_label(gate_id: GateId) -> str:
    if isinstance(gate_id, tuple):
        return "q%d-q%d" % gate_id
    return "q%d" % gate_id


def event_label(layer: int, gate_id: GateId) -> str:
    return "L%d:%s" % (layer, gate_label(gate_id))


def support_from_gate(gate_id: GateId) -> Tuple[int, ...]:
    if isinstance(gate_id, tuple):
        return tuple(int(x) for x in gate_id)
    return (int(gate_id),)


def parse_local_fidelities(raw: Dict) -> Dict[Tuple[int, GateId], float]:
    out: Dict[Tuple[int, GateId], float] = {}
    for key, value in raw.items():
        parsed = ast.literal_eval(key) if isinstance(key, str) else key
        out[(int(parsed[0]), canonical_gate_id(parsed[1]))] = float(value)
    return out


def parse_pair_responses(raw: Dict) -> Dict[PairKey, float]:
    out: Dict[PairKey, float] = {}
    for key, value in raw.items():
        i, gi, j, gj = key
        out[(int(i), canonical_gate_id(gi), int(j), canonical_gate_id(gj))] = float(
            value
        )
    return out


def manhattan(q1: int, q2: int, width: int) -> int:
    r1, c1 = divmod(int(q1), width)
    r2, c2 = divmod(int(q2), width)
    return abs(r1 - r2) + abs(c1 - c2)


def min_support_distance(a: Iterable[int], b: Iterable[int], width: int) -> int:
    return min(manhattan(x, y, width) for x in a for y in b)


def normalize_pair_tuple(pair) -> Tuple[int, int]:
    return tuple(int(x) for x in pair)


def forward_support(
    pattern: Sequence, source_support: Iterable[int], src_layer: int, tgt_layer: int
) -> Tuple[int, ...]:
    """Propagate a qubit support through two-qubit ideal gates up to tgt_layer."""
    support = set(int(x) for x in source_support)
    for layer in range(src_layer + 1, tgt_layer + 1):
        if layer % 2 == 1:
            for pair in pattern[layer]:
                q0, q1 = normalize_pair_tuple(pair)
                if q0 in support or q1 in support:
                    support.add(q0)
                    support.add(q1)
    return tuple(sorted(support))


def apply_1q_sv(
    state: torch.Tensor, gate: torch.Tensor, target: int, n_qubits: int
) -> torch.Tensor:
    bsz = state.shape[0]
    state = state.view(bsz, *([2] * n_qubits))
    state = torch.movedim(state, target + 1, 1)
    orig_shape = state.shape
    state = state.reshape(bsz, 2, -1)
    if gate.ndim == 3:
        state = torch.einsum("boi,bik->bok", gate, state)
    else:
        state = torch.einsum("oi,bik->bok", gate, state)
    state = state.reshape(*orig_shape)
    state = torch.movedim(state, 1, target + 1)
    return state.reshape(bsz, -1)


def apply_2q_sv(
    state: torch.Tensor, gate: torch.Tensor, t1: int, t2: int, n_qubits: int
) -> torch.Tensor:
    bsz = state.shape[0]
    state = state.view(bsz, *([2] * n_qubits))
    state = torch.movedim(state, (t1 + 1, t2 + 1), (1, 2))
    orig_shape = state.shape
    state = state.reshape(bsz, 4, -1)
    is_batched = gate.ndim == 3 or gate.ndim == 5
    gate_mat = gate.reshape(-1, 4, 4) if is_batched else gate.reshape(4, 4)
    if is_batched:
        state = torch.einsum("boi,bik->bok", gate_mat, state)
    else:
        state = torch.einsum("oi,bik->bok", gate_mat, state)
    state = state.reshape(*orig_shape)
    state = torch.movedim(state, (1, 2), (t1 + 1, t2 + 1))
    return state.reshape(bsz, -1)


def get_reshaped_states_1q(
    psi: torch.Tensor, step_sv: torch.Tensor, target: int, n_qubits: int
):
    bsz = step_sv.shape[0]
    psi_view = psi.squeeze(0).view(*([2] * n_qubits))
    step_view = step_sv.view(bsz, *([2] * n_qubits))
    psi_perm = torch.movedim(psi_view, target, 0)
    step_perm = torch.movedim(step_view, target + 1, 1)
    return psi_perm.reshape(2, -1), step_perm.reshape(bsz, 2, -1)


def get_reshaped_states_2q(
    psi: torch.Tensor, step_sv: torch.Tensor, t1: int, t2: int, n_qubits: int
):
    bsz = step_sv.shape[0]
    psi_view = psi.squeeze(0).view(*([2] * n_qubits))
    step_view = step_sv.view(bsz, *([2] * n_qubits))
    psi_perm = torch.movedim(psi_view, (t1, t2), (0, 1))
    step_perm = torch.movedim(step_view, (t1 + 1, t2 + 1), (1, 2))
    return psi_perm.reshape(4, -1), step_perm.reshape(bsz, 4, -1)


@dataclass(frozen=True)
class Event:
    layer: int
    gate_id: GateId
    support: Tuple[int, ...]
    axes: Tuple[int, ...]
    kraus: torch.Tensor
    ideal: torch.Tensor

    @property
    def label(self) -> str:
        return event_label(self.layer, self.gate_id)


def build_event_layers(
    ref: Dict, device: torch.device
) -> Tuple[List[List[Event]], Dict]:
    n_qubits = int(ref["N_QUBITS"])
    n_layers = int(ref["N_LAYERS"])
    pattern = ref["pattern"]
    phys_to_1d = ref["phys_to_1d"]
    pair_to_idx = {
        tuple(int(x) for x in k): int(v) for k, v in ref["pair_to_idx"].items()
    }
    single = {
        (int(k[0]), int(k[1])): v.to(device=device, dtype=torch.complex128)
        for k, v in ref["kraus_single_dict"].items()
    }
    twoq = {
        int(k): v.to(device=device, dtype=torch.complex128)
        for k, v in ref["kraus_2q_dict"].items()
    }
    ideal_single = ref["ideal_single_2d"].to(device=device, dtype=torch.complex128)
    ideal_two = ref["ideal_two_gate_qubit_4x4"].to(
        device=device, dtype=torch.complex128
    )

    def axis(q: int) -> int:
        return (
            int(phys_to_1d[q])
            if not isinstance(phys_to_1d, dict)
            else int(phys_to_1d[int(q)])
        )

    layers: List[List[Event]] = []
    for layer in range(n_layers):
        events: List[Event] = []
        pat = pattern[layer]
        if layer % 2 == 0:
            for q in range(n_qubits):
                mode = int(pat[q])
                events.append(
                    Event(
                        layer=layer,
                        gate_id=int(q),
                        support=(int(q),),
                        axes=(axis(q),),
                        kraus=single[(int(q), mode)],
                        ideal=ideal_single[mode],
                    )
                )
        else:
            for pair in pat:
                q0, q1 = normalize_pair_tuple(pair)
                idx = pair_to_idx[tuple(sorted((q0, q1)))]
                events.append(
                    Event(
                        layer=layer,
                        gate_id=(q0, q1),
                        support=(q0, q1),
                        axes=(axis(q0), axis(q1)),
                        kraus=twoq[idx],
                        ideal=ideal_two,
                    )
                )
        layers.append(events)
    meta = {"pattern": pattern, "n_qubits": n_qubits, "n_layers": n_layers}
    return layers, meta


def apply_event_modes_to_one_state(
    psi_after: torch.Tensor, ev: Event, n_qubits: int
) -> torch.Tensor:
    dressed = torch.matmul(ev.kraus, ev.ideal.conj().T)
    states = psi_after.repeat(dressed.shape[0], 1)
    if len(ev.axes) == 1:
        return apply_1q_sv(states, dressed, ev.axes[0], n_qubits)
    return apply_2q_sv(states, dressed, ev.axes[0], ev.axes[1], n_qubits)


def apply_event_modes_to_branches(
    states: torch.Tensor, ev: Event, n_qubits: int
) -> torch.Tensor:
    dressed = torch.matmul(ev.kraus, ev.ideal.conj().T)
    bsz = states.shape[0]
    r = dressed.shape[0]
    states_rep = states.repeat_interleave(r, dim=0)
    gate_batch = dressed.repeat(bsz, 1, 1)
    if len(ev.axes) == 1:
        return apply_1q_sv(states_rep, gate_batch, ev.axes[0], n_qubits)
    return apply_2q_sv(states_rep, gate_batch, ev.axes[0], ev.axes[1], n_qubits)


def readout_event_sum_over_modes(
    psi_after: torch.Tensor, states: torch.Tensor, ev: Event, n_qubits: int
) -> torch.Tensor:
    dressed = torch.matmul(ev.kraus, ev.ideal.conj().T)
    if len(ev.axes) == 1:
        psi_r, states_r = get_reshaped_states_1q(
            psi_after, states, ev.axes[0], n_qubits
        )
    else:
        psi_r, states_r = get_reshaped_states_2q(
            psi_after, states, ev.axes[0], ev.axes[1], n_qubits
        )
    c_tensor = torch.einsum("ok,bik->boi", psi_r.conj(), states_r)
    ov = torch.einsum("noi,boi->bn", dressed, c_tensor)
    return torch.sum(torch.abs(ov) ** 2, dim=1)


def apply_ideal_layer(
    states: torch.Tensor,
    layer: int,
    pattern: Sequence,
    ideal_single: torch.Tensor,
    ideal_two: torch.Tensor,
    phys_to_1d,
    n_qubits: int,
) -> torch.Tensor:
    def axis(q: int) -> int:
        return (
            int(phys_to_1d[q])
            if not isinstance(phys_to_1d, dict)
            else int(phys_to_1d[int(q)])
        )

    pat = pattern[layer]
    if layer % 2 == 0:
        for q in range(n_qubits):
            mode = int(pat[q])
            states = apply_1q_sv(states, ideal_single[mode], axis(q), n_qubits)
    else:
        for pair in pat:
            q0, q1 = normalize_pair_tuple(pair)
            states = apply_2q_sv(states, ideal_two, axis(q0), axis(q1), n_qubits)
    return states


def propagate_ideal(
    states: torch.Tensor, start_layer: int, end_layer: int, ref: Dict, state_chunk: int
) -> torch.Tensor:
    if start_layer > end_layer:
        return states
    if states.shape[0] > state_chunk:
        parts = [
            propagate_ideal(
                states[i : i + state_chunk], start_layer, end_layer, ref, state_chunk
            )
            for i in range(0, states.shape[0], state_chunk)
        ]
        return torch.cat(parts, dim=0)

    ideal_single = ref["ideal_single_2d"].to(
        device=states.device, dtype=torch.complex128
    )
    ideal_two = ref["ideal_two_gate_qubit_4x4"].to(
        device=states.device, dtype=torch.complex128
    )
    for layer in range(start_layer, end_layer + 1):
        states = apply_ideal_layer(
            states,
            layer,
            ref["pattern"],
            ideal_single,
            ideal_two,
            ref["phys_to_1d"],
            int(ref["N_QUBITS"]),
        )
    return states


def compute_c2_audit(
    ref: Dict,
    local_f: Dict[Tuple[int, GateId], float],
    m_pair: Dict[PairKey, float],
    width: int,
) -> Tuple[List[Dict], Dict]:
    pattern = ref["pattern"]
    rows: List[Dict] = []
    summary = defaultdict(float)
    by_target = {layer: defaultdict(float) for layer in range(int(ref["N_LAYERS"]))}
    by_gap = defaultdict(lambda: defaultdict(float))

    for (i, gi, j, gj), mab in sorted(
        m_pair.items(),
        key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2], str(kv[0][3])),
    ):
        ma = local_f.get((i, gi))
        mb = local_f.get((j, gj))
        if ma is None or mb is None or ma <= 0 or mb <= 0 or mab <= 0:
            continue
        ratio = mab / (ma * mb)
        if ratio <= 0:
            continue
        kappa = math.log(ratio)
        delta = ratio - 1.0
        src_support = support_from_gate(gi)
        tgt_support = support_from_gate(gj)
        if i == j:
            cone = tuple(sorted(src_support))
        else:
            cone = forward_support(pattern, src_support, i, j)
        inside = bool(set(cone).intersection(tgt_support))
        gap = 0 if inside else min_support_distance(cone, tgt_support, width)
        row = {
            "src_layer": i,
            "src_gate": gate_label(gi),
            "tgt_layer": j,
            "tgt_gate": gate_label(gj),
            "src_support": " ".join(map(str, src_support)),
            "tgt_support": " ".join(map(str, tgt_support)),
            "forward_support": " ".join(map(str, cone)),
            "inside_forward_cone": int(inside),
            "cone_gap": int(gap),
            "delta_linear": delta,
            "kappa_log": kappa,
            "abs_kappa_log": abs(kappa),
        }
        rows.append(row)
        bucket = "inside" if inside else "outside"
        summary["count"] += 1
        summary["sum_log_all"] += kappa
        summary["sum_abs_log_all"] += abs(kappa)
        summary[f"count_{bucket}"] += 1
        summary[f"sum_log_{bucket}"] += kappa
        summary[f"sum_abs_log_{bucket}"] += abs(kappa)
        if i < j:
            summary["cross_count"] += 1
            summary[f"cross_count_{bucket}"] += 1
            summary["cross_sum_log_all"] += kappa
            summary["cross_sum_abs_log_all"] += abs(kappa)
            summary[f"cross_sum_log_{bucket}"] += kappa
            summary[f"cross_sum_abs_log_{bucket}"] += abs(kappa)
        else:
            summary["same_layer_count"] += 1
            summary[f"same_layer_count_{bucket}"] += 1
            summary["same_layer_sum_log_all"] += kappa
            summary["same_layer_sum_abs_log_all"] += abs(kappa)
            summary[f"same_layer_sum_log_{bucket}"] += kappa
            summary[f"same_layer_sum_abs_log_{bucket}"] += abs(kappa)
        by_target[j]["all"] += kappa
        by_target[j][bucket] += kappa
        by_target[j]["abs_all"] += abs(kappa)
        by_target[j][f"abs_{bucket}"] += abs(kappa)
        by_gap[gap]["count"] += 1
        by_gap[gap]["sum_abs"] += abs(kappa)
        by_gap[gap]["max_abs"] = max(by_gap[gap]["max_abs"], abs(kappa))

    summary = dict(summary)
    summary["outside_abs_fraction"] = summary.get("sum_abs_log_outside", 0.0) / max(
        summary.get("sum_abs_log_all", 0.0), 1e-300
    )
    summary["outside_signed_fraction"] = abs(summary.get("sum_log_outside", 0.0)) / max(
        abs(summary.get("sum_log_all", 0.0)), 1e-300
    )
    summary["cross_outside_abs_fraction"] = summary.get(
        "cross_sum_abs_log_outside", 0.0
    ) / max(summary.get("cross_sum_abs_log_all", 0.0), 1e-300)
    summary["cross_outside_signed_fraction"] = abs(
        summary.get("cross_sum_log_outside", 0.0)
    ) / max(abs(summary.get("cross_sum_log_all", 0.0)), 1e-300)
    summary["by_target"] = {str(k): dict(v) for k, v in by_target.items()}
    summary["by_gap"] = {str(k): dict(v) for k, v in sorted(by_gap.items())}
    return rows, summary


def pair_response(m_pair: Dict[PairKey, float], left: Event, right: Event) -> float:
    key = (left.layer, left.gate_id, right.layer, right.gate_id)
    if key in m_pair:
        return m_pair[key]
    rev = (right.layer, right.gate_id, left.layer, left.gate_id)
    if rev in m_pair:
        return m_pair[rev]
    raise KeyError("missing pair response %s -> %s" % (left.label, right.label))


def compute_c3_for_layer_triple(
    ref: Dict,
    event_layers: List[List[Event]],
    local_f: Dict[Tuple[int, GateId], float],
    m_pair: Dict[PairKey, float],
    layers: Tuple[int, int, int],
    device: torch.device,
    state_chunk: int,
) -> Tuple[List[Dict], Dict]:
    la, lb, lc = layers
    if not (la < lb < lc):
        raise ValueError("selected layers must satisfy a < b < c")
    n_qubits = int(ref["N_QUBITS"])
    ideal_states = [
        x.to(device=device, dtype=torch.complex128) for x in ref["ideal_state_history"]
    ]

    events_a = event_layers[la]
    events_b = event_layers[lb]
    events_c = event_layers[lc]

    source_states = []
    source_event_indices: List[int] = []
    psi_after_a = ideal_states[la + 1]
    for ai, ev_a in enumerate(events_a):
        states_a = apply_event_modes_to_one_state(psi_after_a, ev_a, n_qubits)
        source_states.append(states_a)
        source_event_indices.extend([ai] * states_a.shape[0])
    source_states_t = torch.cat(source_states, dim=0)
    source_index_t = torch.tensor(source_event_indices, dtype=torch.long, device=device)
    source_states_t = propagate_ideal(source_states_t, la + 1, lb, ref, state_chunk)

    rows: List[Dict] = []
    summary = defaultdict(float)
    psi_after_c = ideal_states[lc + 1]
    t0 = time.time()

    for bj, ev_b in enumerate(events_b):
        states_ab = apply_event_modes_to_branches(source_states_t, ev_b, n_qubits)
        rb = int(ev_b.kraus.shape[0])
        source_index_ab = source_index_t.repeat_interleave(rb)
        states_ab = propagate_ideal(states_ab, lb + 1, lc, ref, state_chunk)

        for cj, ev_c in enumerate(events_c):
            per_branch = readout_event_sum_over_modes(
                psi_after_c, states_ab, ev_c, n_qubits
            )
            by_source = torch.zeros(len(events_a), dtype=torch.float64, device=device)
            by_source.index_add_(0, source_index_ab, per_branch.real.to(torch.float64))
            by_source_cpu = by_source.detach().cpu().numpy()

            for ai, ev_a in enumerate(events_a):
                ma = local_f[(la, ev_a.gate_id)]
                mb = local_f[(lb, ev_b.gate_id)]
                mc = local_f[(lc, ev_c.gate_id)]
                mab = pair_response(m_pair, ev_a, ev_b)
                mac = pair_response(m_pair, ev_a, ev_c)
                mbc = pair_response(m_pair, ev_b, ev_c)
                mabc = float(by_source_cpu[ai])
                if (
                    mabc <= 0
                    or ma <= 0
                    or mb <= 0
                    or mc <= 0
                    or mab <= 0
                    or mac <= 0
                    or mbc <= 0
                ):
                    continue
                ratio3 = (mabc * ma * mb * mc) / (mab * mac * mbc)
                if ratio3 <= 0:
                    continue
                k3 = math.log(ratio3)
                k2_ab = math.log(mab / (ma * mb))
                k2_ac = math.log(mac / (ma * mc))
                k2_bc = math.log(mbc / (mb * mc))
                row = {
                    "layer_a": la,
                    "gate_a": gate_label(ev_a.gate_id),
                    "layer_b": lb,
                    "gate_b": gate_label(ev_b.gate_id),
                    "layer_c": lc,
                    "gate_c": gate_label(ev_c.gate_id),
                    "Mabc": mabc,
                    "ratio3": ratio3,
                    "kappa3_log": k3,
                    "abs_kappa3_log": abs(k3),
                    "kappa2_ab_log": k2_ab,
                    "kappa2_ac_log": k2_ac,
                    "kappa2_bc_log": k2_bc,
                }
                rows.append(row)
                summary["count"] += 1
                summary["sum_log_c3"] += k3
                summary["sum_abs_log_c3"] += abs(k3)
                summary["max_abs_log_c3"] = max(summary["max_abs_log_c3"], abs(k3))
        elapsed = time.time() - t0
        print(
            "[c3] finished middle event %s (%d/%d), rows=%d, elapsed=%.1fs"
            % (ev_b.label, bj + 1, len(events_b), len(rows), elapsed),
            flush=True,
        )

    matched_pair_keys = set()
    pair_sums = {}
    for left_layer, right_layer in [(la, lb), (la, lc), (lb, lc)]:
        s = 0.0
        s_abs = 0.0
        for ev_l in event_layers[left_layer]:
            for ev_r in event_layers[right_layer]:
                ma = local_f[(ev_l.layer, ev_l.gate_id)]
                mb = local_f[(ev_r.layer, ev_r.gate_id)]
                mab = pair_response(m_pair, ev_l, ev_r)
                k2 = math.log(mab / (ma * mb))
                s += k2
                s_abs += abs(k2)
                matched_pair_keys.add(
                    (ev_l.layer, ev_l.gate_id, ev_r.layer, ev_r.gate_id)
                )
        pair_sums["L%d_L%d_sum_log_c2" % (left_layer, right_layer)] = s
        pair_sums["L%d_L%d_sum_abs_log_c2" % (left_layer, right_layer)] = s_abs
    summary.update(pair_sums)
    summary["selected_layers"] = list(layers)
    summary["c3_abs_over_matched_c2_abs"] = summary.get("sum_abs_log_c3", 0.0) / max(
        sum(v for k, v in pair_sums.items() if k.endswith("sum_abs_log_c2")), 1e-300
    )
    summary["c3_signed_over_matched_c2_signed"] = abs(
        summary.get("sum_log_c3", 0.0)
    ) / max(
        abs(sum(v for k, v in pair_sums.items() if k.endswith("sum_log_c2"))), 1e-300
    )
    return rows, dict(summary)


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(
    out_path_png: Path,
    out_path_pdf: Path,
    c2_rows: List[Dict],
    c2_summary: Dict,
    c3_rows: List[Dict],
    c3_summary: Dict,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2))

    by_target = c2_summary["by_target"]
    layers = sorted(int(k) for k in by_target.keys())
    all_vals = np.array(
        [by_target[str(k)].get("all", 0.0) for k in layers], dtype=float
    )
    inside_vals = np.array(
        [by_target[str(k)].get("inside", 0.0) for k in layers], dtype=float
    )
    outside_vals = np.array(
        [by_target[str(k)].get("outside", 0.0) for k in layers], dtype=float
    )
    axes[0, 0].plot(layers, np.cumsum(all_vals), label="all pairs", lw=2.2)
    axes[0, 0].plot(layers, np.cumsum(inside_vals), label="support-retained", lw=2.0)
    axes[0, 0].plot(
        layers, np.cumsum(outside_vals), label="outside contribution", lw=1.8
    )
    axes[0, 0].set_title("Second-order log-cumulant audit")
    axes[0, 0].set_xlabel("target layer")
    axes[0, 0].set_ylabel("cumulative sum kappa2")
    axes[0, 0].legend(frameon=False, fontsize=9)
    axes[0, 0].axhline(0.0, color="0.75", lw=0.8)

    gap_data = c2_summary["by_gap"]
    gaps = sorted(int(k) for k in gap_data.keys())
    gap_abs = np.array([gap_data[str(k)].get("sum_abs", 0.0) for k in gaps])
    gap_mean = np.array(
        [
            gap_data[str(k)].get("sum_abs", 0.0)
            / max(gap_data[str(k)].get("count", 1.0), 1.0)
            for k in gaps
        ]
    )
    axes[0, 1].bar(gaps, gap_abs, color="#4C78A8", alpha=0.82, label="sum |kappa2|")
    axr = axes[0, 1].twinx()
    axr.plot(gaps, gap_mean, color="#F58518", marker="o", label="mean |kappa2|")
    axes[0, 1].set_yscale("log")
    axr.set_yscale("log")
    axes[0, 1].set_title("Outside-cone decay by support gap")
    axes[0, 1].set_xlabel("support gap after propagation")
    axes[0, 1].set_ylabel("sum |kappa2|")
    axr.set_ylabel("mean |kappa2|")

    pair_abs_labels = []
    pair_abs_values = []
    for key, value in c3_summary.items():
        if key.endswith("sum_abs_log_c2"):
            parts = key.split("_")
            pair_abs_labels.append(parts[0] + "-" + parts[1])
            pair_abs_values.append(float(value))
    pair_abs_labels.append(
        "C3 " + "-".join("L%d" % x for x in c3_summary["selected_layers"])
    )
    pair_abs_values.append(float(c3_summary.get("sum_abs_log_c3", 0.0)))
    axes[1, 0].bar(
        pair_abs_labels,
        pair_abs_values,
        color=["#54A24B"] * (len(pair_abs_labels) - 1) + ["#E45756"],
    )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Selected-layer third order is a small tail")
    axes[1, 0].set_ylabel("sum |log-cumulant|")
    axes[1, 0].tick_params(axis="x", rotation=25)

    selected_layers = set(c3_summary["selected_layers"])
    c2_selected_abs = []
    for row in c2_rows:
        if (
            row["src_layer"] in selected_layers
            and row["tgt_layer"] in selected_layers
            and row["src_layer"] < row["tgt_layer"]
        ):
            c2_selected_abs.append(max(float(row["abs_kappa_log"]), 1e-300))
    c3_abs = [max(float(row["abs_kappa3_log"]), 1e-300) for row in c3_rows]
    bins = np.linspace(-18, 0, 55)
    if c2_selected_abs:
        axes[1, 1].hist(
            np.log10(c2_selected_abs),
            bins=bins,
            alpha=0.62,
            label="selected C2",
            color="#72B7B2",
        )
    if c3_abs:
        axes[1, 1].hist(
            np.log10(c3_abs),
            bins=bins,
            alpha=0.62,
            label="selected C3",
            color="#E45756",
        )
    axes[1, 1].set_title("Magnitude distribution")
    axes[1, 1].set_xlabel("log10 |kappa|")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].legend(frameon=False, fontsize=9)

    fig.suptitle(
        "4x4, 20-layer exact branch cumulant audit from 0522 exports", fontsize=13
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_path_png, dpi=220)
    fig.savefig(out_path_pdf)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="4x4 exact branch C2/C3 audit from 0522 exports"
    )
    parser.add_argument("--ref", default="0522_reference_data.pt")
    parser.add_argument("--pairs", default="M_pair_from_0522.pt")
    parser.add_argument(
        "--layers", default="0,10,18", help="comma-separated layer triple a,b,c"
    )
    parser.add_argument("--outdir", default="natcomm_4x4_cumulant_audit")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--state-chunk", type=int, default=384)
    args = parser.parse_args()

    t_start = time.time()
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    torch.set_grad_enabled(False)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[setup] device=%s" % device, flush=True)
    if device.type == "cuda":
        print("[setup] gpu=%s" % torch.cuda.get_device_name(0), flush=True)

    ref = torch.load(args.ref, map_location="cpu", weights_only=False)
    pair_cache = torch.load(args.pairs, map_location="cpu", weights_only=False)
    local_f = parse_local_fidelities(pair_cache["local_fidelities"])
    m_pair = parse_pair_responses(pair_cache["M_pair"])
    width = int(round(math.sqrt(int(ref["N_QUBITS"]))))

    c2_rows, c2_summary = compute_c2_audit(ref, local_f, m_pair, width)
    write_csv(outdir / "c2_all_vs_support_cone.csv", c2_rows)
    print(
        "[c2] rows=%d outside_abs_fraction=%.6e cross_outside_abs_fraction=%.6e"
        % (
            len(c2_rows),
            c2_summary.get("outside_abs_fraction", float("nan")),
            c2_summary.get("cross_outside_abs_fraction", float("nan")),
        ),
        flush=True,
    )

    event_layers, _ = build_event_layers(ref, device)
    layers = tuple(int(x) for x in args.layers.split(","))
    c3_rows, c3_summary = compute_c3_for_layer_triple(
        ref, event_layers, local_f, m_pair, layers, device, args.state_chunk
    )
    write_csv(outdir / "c3_selected_layer_triple.csv", c3_rows)

    payload = {
        "metadata": {
            "script": Path(__file__).name,
            "ref": args.ref,
            "pairs": args.pairs,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "N_QUBITS": int(ref["N_QUBITS"]),
            "N_LAYERS": int(ref["N_LAYERS"]),
            "selected_layers": list(layers),
            "elapsed_sec": time.time() - t_start,
        },
        "c2_summary": c2_summary,
        "c3_summary": c3_summary,
        "top_outside_c2_by_abs": sorted(
            [r for r in c2_rows if not r["inside_forward_cone"]],
            key=lambda r: r["abs_kappa_log"],
            reverse=True,
        )[:20],
        "top_c3_by_abs": sorted(
            c3_rows, key=lambda r: r["abs_kappa3_log"], reverse=True
        )[:20],
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    make_figure(
        outdir / "natcomm_4x4_cumulant_audit.png",
        outdir / "natcomm_4x4_cumulant_audit.pdf",
        c2_rows,
        c2_summary,
        c3_rows,
        c3_summary,
    )

    print(
        "[c3] rows=%d sum_abs=%.6e ratio_abs_to_matched_c2=%.6e"
        % (
            len(c3_rows),
            c3_summary.get("sum_abs_log_c3", float("nan")),
            c3_summary.get("c3_abs_over_matched_c2_abs", float("nan")),
        ),
        flush=True,
    )
    print("[done] wrote %s elapsed=%.1fs" % (outdir, time.time() - t_start), flush=True)


if __name__ == "__main__":
    main()
