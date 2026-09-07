"""
exact_small.py: Exact full-state connected-fidelity reference backend.
Implements the mathematical definition using direct state-vector evolution.
This is the semantic reference for all approximate backends.

Formulas:
  E[g,u] = K[g,u] U[g]^dag
  a[g,u] = <psi_{ell(g)+1} | E[g,u] | psi_{ell(g)+1}>
  F_g    = sum_u |a[g,u]|^2

  A_vu(a,b)[v,u] =
    <psi_{ell_b+1}| E[b,v] U_{ell_a+1->ell_b+1} E[a,u]
     |psi_{ell_a+1}>

  C_vu(a,b)[v,u] = A_vu(a,b)[v,u] - a[b,v] * a[a,u]

  Delta_raw = sum_{u,v} |A_vu[v,u]|^2 / (F_a F_b) - 1
  Delta_cent =
    (2 Re sum_{u,v} conj(a[b,v] * a[a,u]) * C_vu[v,u]
     + sum_{u,v} |C_vu[v,u]|^2) / (F_a F_b)

  log_F1 = sum_g log(F_g)
  log_F_connected = log_F1 + sum_{pairs} Delta_ab
"""

import os
from typing import Optional, Set, Tuple
import torch
from residual_tn.backend.context import FidelityContext, FidelityResult
from residual_tn.backend.kraus import (
    compute_dressed_kraus,
    compute_one_point_amplitude,
    compute_centered_operator,
)
from residual_tn.backend.fidelity_formula import delta_from_centered_C, delta_from_raw_A
from residual_tn.backend.pair_accumulator import pair_mode_diagnostic_from_centered_C


def _apply_1q_gate(state: torch.Tensor, gate: torch.Tensor, q: int, n_qubits: int):
    """Apply single-qubit gate to full state vector.
    state: (2**n_qubits,), gate: (2,2), q: target qubit index.
    """
    dim = 2**n_qubits
    state = state.reshape(*([2] * n_qubits))
    state = torch.tensordot(gate, state, dims=([1], [q]))
    state = torch.moveaxis(state, 0, q)
    return state.reshape(dim)


def _apply_2q_gate(
    state: torch.Tensor, gate: torch.Tensor, q1: int, q2: int, n_qubits: int
):
    """Apply two-qubit gate to full state vector.
    gate: (4,4) or (2,2,2,2)
    """
    d = 2
    dim = 2**n_qubits
    if gate.shape == (d * d, d * d):
        gate = gate.reshape(d, d, d, d)
    state = state.reshape(*([d] * n_qubits))
    state = torch.tensordot(gate, state, dims=([2, 3], [q1, q2]))
    ndim_new = state.ndim
    # Build permutation: move the two output dims (0,1) to q1, q2 positions
    perm = []
    out_idx = 0
    for pos in range(n_qubits):
        if pos == q1:
            perm.append(0)
        elif pos == q2:
            perm.append(1)
        else:
            perm.append(2 + out_idx)
            out_idx += 1
    state = state.permute(*perm)
    return state.reshape(dim)


def _apply_1q_gate_batched(
    states: torch.Tensor,
    gate: torch.Tensor,
    q: int,
    n_qubits: int,
) -> torch.Tensor:
    """Apply one single-qubit gate to a batch of full state vectors."""
    if states.ndim == 1:
        return _apply_1q_gate(states, gate, q, n_qubits)

    batch = states.shape[0]
    dim = 2**n_qubits
    gate_mat = gate.reshape(2, 2)
    x = states.reshape(batch, *([2] * n_qubits))
    x = torch.movedim(x, q + 1, 1)
    orig_shape = x.shape
    x = x.reshape(batch, 2, -1)
    y = torch.einsum("oi, bik -> bok", gate_mat, x)
    y = y.reshape(*orig_shape)
    y = torch.movedim(y, 1, q + 1)
    return y.reshape(batch, dim)


def _apply_2q_gate_batched(
    states: torch.Tensor,
    gate: torch.Tensor,
    q1: int,
    q2: int,
    n_qubits: int,
) -> torch.Tensor:
    """Apply one two-qubit gate to a batch of full state vectors."""
    if states.ndim == 1:
        return _apply_2q_gate(states, gate, q1, q2, n_qubits)

    batch = states.shape[0]
    dim = 2**n_qubits
    gate_mat = gate.reshape(4, 4)
    x = states.reshape(batch, *([2] * n_qubits))
    x = torch.movedim(x, (q1 + 1, q2 + 1), (1, 2))
    orig_shape = x.shape
    x = x.reshape(batch, 4, -1)
    y = torch.einsum("oi, bik -> bok", gate_mat, x)
    y = y.reshape(*orig_shape)
    y = torch.movedim(y, (1, 2), (q1 + 1, q2 + 1))
    return y.reshape(batch, dim)


def _apply_1q_gate_per_state(
    states: torch.Tensor,
    gates: torch.Tensor,
    q: int,
    n_qubits: int,
) -> torch.Tensor:
    """Apply one local 1q operator per state in a full-state batch."""
    batch = states.shape[0]
    dim = 2**n_qubits
    gate_mat = gates.reshape(batch, 2, 2)
    x = states.reshape(batch, *([2] * n_qubits))
    x = torch.movedim(x, q + 1, 1)
    orig_shape = x.shape
    x = x.reshape(batch, 2, -1)
    y = torch.einsum("boi, bik -> bok", gate_mat, x)
    y = y.reshape(*orig_shape)
    y = torch.movedim(y, 1, q + 1)
    return y.reshape(batch, dim)


def _apply_2q_gate_per_state(
    states: torch.Tensor,
    gates: torch.Tensor,
    q1: int,
    q2: int,
    n_qubits: int,
) -> torch.Tensor:
    """Apply one local 2q operator per state in a full-state batch."""
    batch = states.shape[0]
    dim = 2**n_qubits
    gate_mat = gates.reshape(batch, 4, 4)
    x = states.reshape(batch, *([2] * n_qubits))
    x = torch.movedim(x, (q1 + 1, q2 + 1), (1, 2))
    orig_shape = x.shape
    x = x.reshape(batch, 4, -1)
    y = torch.einsum("boi, bik -> bok", gate_mat, x)
    y = y.reshape(*orig_shape)
    y = torch.movedim(y, (1, 2), (q1 + 1, q2 + 1))
    return y.reshape(batch, dim)


def _get_local_rho(state: torch.Tensor, qubits, n_qubits: int) -> torch.Tensor:
    """Get reduced density matrix for given qubits from pure state.
    Preserves the order of qubits as given.
    """
    d = 2
    n_local = len(qubits)
    state_t = state.reshape(*([d] * n_qubits))
    # Move target qubits to first positions, preserving order
    axes = list(range(n_qubits))
    for i, q in enumerate(qubits):
        axes.remove(q)
        axes.insert(i, q)
    state_t = state_t.permute(*axes)
    dim_local = d**n_local
    dim_env = d ** (n_qubits - n_local)
    psi = state_t.reshape(dim_local, dim_env)
    rho = psi @ psi.conj().T
    return rho


def exact_compute_one_point_amplitudes(
    ctx: FidelityContext,
    layer_index: int,
) -> dict[int, torch.Tensor]:
    """Return exact one-point dressed-Kraus amplitudes for one layer.

    This is a small diagnostic helper.  It evolves the ideal state exactly to
    the requested post-layer state and returns
    ``<psi_layer|E_u|psi_layer>`` without applying any mode filter.
    """
    if not 0 <= int(layer_index) < len(ctx.layers):
        raise ValueError(f"invalid layer_index {layer_index}")
    state = torch.zeros(2**ctx.n_qubits, dtype=ctx.dtype, device=ctx.device)
    state[0] = 1.0
    for current_layer, gates in enumerate(ctx.layers):
        for gate in gates:
            qubits = gate["qubits"]
            if len(qubits) == 1:
                state = _apply_1q_gate(
                    state, gate["ideal_unitary"], qubits[0], ctx.n_qubits
                )
            else:
                state = _apply_2q_gate(
                    state, gate["ideal_unitary"], qubits[0], qubits[1], ctx.n_qubits
                )
        if current_layer != int(layer_index):
            continue
        amplitudes = {}
        for gate in gates:
            gate_id = int(gate["gate_idx"])
            dressed = compute_dressed_kraus(gate["kraus_ops"], gate["ideal_unitary"])
            rho = _get_local_rho(state, gate["qubits"], ctx.n_qubits)
            amplitudes[gate_id] = compute_one_point_amplitude(dressed, rho)
        return amplitudes
    raise RuntimeError(f"failed to reach layer_index {layer_index}")


def _local_transition_overlap(
    psi: torch.Tensor,
    branches: torch.Tensor,
    qubits,
    n_qubits: int,
) -> torch.Tensor:
    """Return C[b,o,i] = sum_env conj(psi[o,env]) * branch[b,i,env]."""
    d = 2
    if branches.ndim == 1:
        branches = branches.reshape(1, -1)
    batch = int(branches.shape[0])
    qubits = tuple(int(q) for q in qubits)
    n_local = len(qubits)
    dim_local = d**n_local
    psi_view = psi.reshape(*([d] * n_qubits))
    branch_view = branches.reshape(batch, *([d] * n_qubits))
    psi_perm = torch.movedim(psi_view, qubits, tuple(range(n_local)))
    branch_perm = torch.movedim(
        branch_view,
        tuple(q + 1 for q in qubits),
        tuple(range(1, n_local + 1)),
    )
    psi_local = psi_perm.reshape(dim_local, -1)
    branch_local = branch_perm.reshape(batch, dim_local, -1)
    return torch.einsum("ok,bik->boi", psi_local.conj(), branch_local)


def _positive_int_env(name: str, default: int) -> int:
    """Read a positive integer environment setting."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return int(default)
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {raw!r}.")
    return value


def _group_mode_count(group) -> int:
    """Number of full-state branch modes stored in one active source group."""
    return int(group["branches"].shape[0])


def _sum_group_modes(groups) -> int:
    """Total branch-mode count across active source groups."""
    return sum(_group_mode_count(group) for group in groups)


def _iter_group_chunks(groups, max_modes: int):
    """Yield source-group chunks with at most max_modes branch states."""
    chunk = []
    count = 0
    for group in groups:
        modes = _group_mode_count(group)
        if chunk and count + modes > max_modes:
            yield chunk
            chunk = []
            count = 0
        chunk.append(group)
        count += modes
        if count >= max_modes:
            yield chunk
            chunk = []
            count = 0
    if chunk:
        yield chunk


def _concat_group_branches(groups: list) -> torch.Tensor:
    """Concatenate a source-group chunk into one branch-mode batch."""
    if len(groups) == 1:
        return groups[0]["branches"]
    return torch.cat([group["branches"] for group in groups], dim=0)


def _assign_group_branches(groups: list, branches: torch.Tensor) -> None:
    """Split a branch-mode batch back onto its source-group records."""
    offset = 0
    for group in groups:
        modes = _group_mode_count(group)
        group["branches"] = branches[offset : offset + modes]
        offset += modes


def _propagate_group_chunks(
    groups: list,
    layer_gates: list,
    n_qubits: int,
    max_modes: int,
    diagnostics: dict,
) -> None:
    """Propagate active full-state branch groups through one ideal layer."""
    for chunk in _iter_group_chunks(groups, max_modes):
        branches = _concat_group_branches(chunk)
        diagnostics["branch_propagation_chunks"] += 1
        diagnostics["source_group_layer_propagations"] += len(chunk)
        diagnostics["max_exact_branch_chunk_modes"] = max(
            diagnostics["max_exact_branch_chunk_modes"],
            int(branches.shape[0]),
        )
        for gate in layer_gates:
            qubits = gate["qubits"]
            ideal_u = gate["ideal_unitary"]
            if len(qubits) == 1:
                branches = _apply_1q_gate_batched(
                    branches, ideal_u, qubits[0], n_qubits
                )
            else:
                branches = _apply_2q_gate_batched(
                    branches, ideal_u, qubits[0], qubits[1], n_qubits
                )
        _assign_group_branches(chunk, branches)


def _build_result(
    ctx: FidelityContext,
    log_F1: float,
    local_F: dict,
    pair_deltas: dict,
    pair_C: dict,
    max_layer_distance: int,
    selected_pairs,
    store_pair_C: bool,
    pair_C_to_cpu: bool,
    diagnostics: dict,
) -> FidelityResult:
    """Create a FidelityResult and layer self-energy summaries."""
    log_F_connected = log_F1 + sum(pair_deltas.values())

    self_energy_by_target_layer = {}
    self_energy_by_source_layer = {}
    for (a_idx, b_idx), delta_val in pair_deltas.items():
        if hasattr(ctx, "gate_map") and b_idx in ctx.gate_map:
            b_layer = ctx.gate_map[b_idx][0]
            self_energy_by_target_layer[b_layer] = (
                self_energy_by_target_layer.get(b_layer, 0.0) + delta_val
            )
        if hasattr(ctx, "gate_map") and a_idx in ctx.gate_map:
            a_layer = ctx.gate_map[a_idx][0]
            self_energy_by_source_layer[a_layer] = (
                self_energy_by_source_layer.get(a_layer, 0.0) + delta_val
            )

    return FidelityResult(
        log_fidelity_first_order=log_F1,
        fidelity_first_order=torch.exp(torch.tensor(log_F1)).item(),
        log_fidelity_connected=log_F_connected,
        fidelity_connected=torch.exp(torch.tensor(log_F_connected)).item(),
        local_fidelities=local_F,
        pair_deltas=pair_deltas,
        pair_C=pair_C,
        self_energy_by_layer=self_energy_by_target_layer,
        self_energy_by_source_layer=self_energy_by_source_layer,
        diagnostics={
            "backend": "exact",
            "max_layer_distance": max_layer_distance,
            "selected_pairs": None if selected_pairs is None else len(selected_pairs),
            "store_pair_C": store_pair_C,
            "pair_C_to_cpu": pair_C_to_cpu,
            **diagnostics,
        },
    )


def _propagate_flat_branch_batch(
    states: torch.Tensor,
    layer_gates: list,
    n_qubits: int,
) -> torch.Tensor:
    """Propagate a batch of full-state branches through one ideal layer."""
    for gate in layer_gates:
        qubits = gate["qubits"]
        ideal_u = gate["ideal_unitary"]
        if len(qubits) == 1:
            states = _apply_1q_gate_batched(states, ideal_u, qubits[0], n_qubits)
        else:
            states = _apply_2q_gate_batched(
                states, ideal_u, qubits[0], qubits[1], n_qubits
            )
    return states


def _exact_compute_fidelity_flat_centered(
    ctx: FidelityContext,
    max_layer_distance: int,
    selected_pairs,
    state_history: list,
    local_a: dict,
    local_F: dict,
    centered_ops: dict,
    log_F1: float,
    source_mode_indices: dict[int, torch.Tensor] | None = None,
    target_mode_indices: dict[int, torch.Tensor] | None = None,
) -> FidelityResult:
    """Flat source-branch pipeline with centered Kraus contractions."""
    n_qubits = ctx.n_qubits
    device = ctx.device
    source_mode_indices = source_mode_indices or {}
    target_mode_indices = target_mode_indices or {}

    def _source_ids(gate_id: int, count: int) -> torch.Tensor:
        value = source_mode_indices.get(gate_id)
        if value is None:
            return torch.arange(count, dtype=torch.long, device=device)
        return value

    def _target_ids(gate_id: int, count: int) -> torch.Tensor:
        value = target_mode_indices.get(gate_id)
        if value is None:
            return torch.arange(count, dtype=torch.long, device=device)
        return value

    branch_chunk_modes = _positive_int_env(
        "CENTERED_FIDELITY_EXACT_BRANCH_CHUNK_MODES",
        128,
    )
    progress = os.environ.get("CENTERED_FIDELITY_PROGRESS", "0") == "1"
    skip_same_layer = (
        os.environ.get("CENTERED_FIDELITY_EXACT_SKIP_SAME_LAYER", "0") == "1"
    )
    selected_source_gates = (
        None if selected_pairs is None else {int(a_idx) for a_idx, _ in selected_pairs}
    )

    active_branches = []
    pair_cross = {}
    pair_quad = {}
    diagnostics = {
        "flat_centered_pipeline": True,
        "layer_window_pipeline": True,
        "exact_branch_mode_batching": True,
        "exact_skip_same_layer": skip_same_layer,
        "exact_branch_chunk_modes": branch_chunk_modes,
        "source_groups_spawned": 0,
        "source_modes_spawned": 0,
        "source_groups_pruned_before_readout": 0,
        "source_modes_pruned_before_readout": 0,
        "source_groups_pruned_before_propagation": 0,
        "source_modes_pruned_before_propagation": 0,
        "max_active_source_groups": 0,
        "max_active_source_modes": 0,
        "max_exact_branch_chunk_modes": 0,
        "target_pairs_evaluated": 0,
        "source_group_layer_propagations": 0,
        "branch_readout_chunks": 0,
        "branch_propagation_chunks": 0,
        "target_mode_batch_applications": 0,
        "local_transition_readouts": 0,
        "pair_mode_diagnostics": {},
    }
    record_pair_modes = os.environ.get("CENTERED_PAIR_MODE_DIAGNOSTICS", "0") == "1"

    for layer_idx, layer_gates in enumerate(ctx.layers):
        layer_pairs_before = diagnostics["target_pairs_evaluated"]
        layer_readout_chunks_before = diagnostics["branch_readout_chunks"]
        layer_prop_chunks_before = diagnostics["branch_propagation_chunks"]
        layer_prop_groups_before = diagnostics["source_group_layer_propagations"]
        if progress:
            print(
                "[exact-layer] "
                f"start {layer_idx + 1}/{len(ctx.layers)} "
                f"active_groups={len(active_branches)} "
                f"active_modes={len(active_branches)} "
                f"completed_pairs={len(pair_cross)}",
                flush=True,
            )

        if max_layer_distance is not None and active_branches:
            before = len(active_branches)
            active_branches = [
                branch
                for branch in active_branches
                if layer_idx - int(branch["source_layer"]) <= max_layer_distance
            ]
            pruned = before - len(active_branches)
            diagnostics["source_groups_pruned_before_readout"] += pruned
            diagnostics["source_modes_pruned_before_readout"] += pruned

        # Inject this layer's centered source modes at |psi_{layer+1}>.
        for gate_a in layer_gates:
            a_idx = gate_a["gate_idx"]
            if selected_source_gates is not None and a_idx not in selected_source_gates:
                continue
            qubits_a = gate_a["qubits"]
            source_ids = _source_ids(a_idx, int(centered_ops[a_idx].shape[0]))
            E_a_center = centered_ops[a_idx].index_select(0, source_ids)
            n_u = int(E_a_center.shape[0])
            source_base = state_history[layer_idx].expand(n_u, -1).clone()
            if len(qubits_a) == 1:
                branches = _apply_1q_gate_per_state(
                    source_base, E_a_center, qubits_a[0], n_qubits
                )
            else:
                branches = _apply_2q_gate_per_state(
                    source_base, E_a_center, qubits_a[0], qubits_a[1], n_qubits
                )
            for local_mode, source_mode in enumerate(source_ids.tolist()):
                active_branches.append(
                    {
                        "source_layer": layer_idx,
                        "source_gate_idx": a_idx,
                        "source_mode": source_mode,
                        "state": branches[local_mode].clone(),
                    }
                )
            diagnostics["source_groups_spawned"] += 1
            diagnostics["source_modes_spawned"] += n_u

        diagnostics["max_active_source_groups"] = max(
            diagnostics["max_active_source_groups"],
            len(active_branches),
        )
        diagnostics["max_active_source_modes"] = max(
            diagnostics["max_active_source_modes"],
            len(active_branches),
        )

        psi_target = state_history[layer_idx]
        for gate_b in layer_gates:
            b_idx = gate_b["gate_idx"]
            eligible = []
            for branch in active_branches:
                a_idx = branch["source_gate_idx"]
                source_layer = branch["source_layer"]
                if a_idx == b_idx:
                    continue
                if max_layer_distance is not None:
                    if layer_idx - source_layer > max_layer_distance:
                        continue
                if layer_idx == source_layer:
                    if skip_same_layer:
                        continue
                    if a_idx > b_idx:
                        continue
                if selected_pairs is not None and (a_idx, b_idx) not in selected_pairs:
                    continue
                eligible.append(branch)

            if not eligible:
                continue

            qubits_b = gate_b["qubits"]
            target_ids = _target_ids(b_idx, int(centered_ops[b_idx].shape[0]))
            E_b_center = centered_ops[b_idx].index_select(0, target_ids)
            n_v = int(E_b_center.shape[0])
            a_b_conj = local_a[b_idx].index_select(0, target_ids).conj()

            for start in range(0, len(eligible), branch_chunk_modes):
                chunk_meta = eligible[start : start + branch_chunk_modes]
                states = torch.stack([branch["state"] for branch in chunk_meta])
                diagnostics["branch_readout_chunks"] += 1
                diagnostics["max_exact_branch_chunk_modes"] = max(
                    diagnostics["max_exact_branch_chunk_modes"],
                    int(states.shape[0]),
                )
                transition = _local_transition_overlap(
                    psi_target, states, qubits_b, n_qubits
                )
                E_target = E_b_center.reshape(
                    n_v, transition.shape[-2], transition.shape[-1]
                )
                values_by_branch = torch.einsum("voi,boi->bv", E_target, transition)
                diagnostics["local_transition_readouts"] += 1
                diagnostics["target_mode_batch_applications"] += n_v

                rows_by_source = {}
                for local_idx, branch in enumerate(chunk_meta):
                    rows_by_source.setdefault(branch["source_gate_idx"], []).append(
                        (local_idx, branch["source_mode"])
                    )

                for a_idx, rows_modes in rows_by_source.items():
                    rows = torch.tensor(
                        [row for row, _ in rows_modes],
                        dtype=torch.long,
                        device=device,
                    )
                    modes = torch.tensor(
                        [mode for _, mode in rows_modes],
                        dtype=torch.long,
                        device=device,
                    )
                    values = values_by_branch.index_select(0, rows)
                    a_source_conj = local_a[a_idx].index_select(0, modes).conj()
                    cross = (
                        2.0
                        * torch.sum(
                            values * a_source_conj.unsqueeze(1) * a_b_conj.unsqueeze(0)
                        ).real
                    )
                    quad = torch.sum(torch.abs(values) ** 2).real
                    pair_key = (a_idx, b_idx)
                    pair_cross[pair_key] = pair_cross.get(pair_key, 0.0) + cross
                    pair_quad[pair_key] = pair_quad.get(pair_key, 0.0) + quad
                    diagnostics["target_pairs_evaluated"] += 1

        if max_layer_distance is not None and active_branches:
            before = len(active_branches)
            active_branches = [
                branch
                for branch in active_branches
                if (layer_idx + 1) - int(branch["source_layer"]) <= max_layer_distance
            ]
            pruned = before - len(active_branches)
            diagnostics["source_groups_pruned_before_propagation"] += pruned
            diagnostics["source_modes_pruned_before_propagation"] += pruned

        if layer_idx + 1 < len(ctx.layers) and active_branches:
            for start in range(0, len(active_branches), branch_chunk_modes):
                chunk_meta = active_branches[start : start + branch_chunk_modes]
                states = torch.stack([branch["state"] for branch in chunk_meta])
                diagnostics["branch_propagation_chunks"] += 1
                diagnostics["source_group_layer_propagations"] += len(chunk_meta)
                diagnostics["max_exact_branch_chunk_modes"] = max(
                    diagnostics["max_exact_branch_chunk_modes"],
                    int(states.shape[0]),
                )
                states = _propagate_flat_branch_batch(
                    states, ctx.layers[layer_idx + 1], n_qubits
                )
                for local_idx, branch in enumerate(chunk_meta):
                    branch["state"] = states[local_idx].clone()

        if progress:
            print(
                "[exact-layer] "
                f"done {layer_idx + 1}/{len(ctx.layers)} "
                f"active_groups={len(active_branches)} "
                f"active_modes={len(active_branches)} "
                f"new_pairs={diagnostics['target_pairs_evaluated'] - layer_pairs_before} "
                f"completed_pairs={len(pair_cross)} "
                f"readout_chunks={diagnostics['branch_readout_chunks'] - layer_readout_chunks_before} "
                f"prop_chunks={diagnostics['branch_propagation_chunks'] - layer_prop_chunks_before} "
                f"prop_groups={diagnostics['source_group_layer_propagations'] - layer_prop_groups_before}",
                flush=True,
            )

    pair_deltas = {}
    for pair_key, cross in pair_cross.items():
        a_idx, b_idx = pair_key
        delta = (cross + pair_quad.get(pair_key, 0.0)) / (
            local_F[a_idx] * local_F[b_idx]
        )
        pair_deltas[pair_key] = float(delta.detach().cpu().item())

    return _build_result(
        ctx=ctx,
        log_F1=log_F1,
        local_F=local_F,
        pair_deltas=pair_deltas,
        pair_C={},
        max_layer_distance=max_layer_distance,
        selected_pairs=selected_pairs,
        store_pair_C=False,
        pair_C_to_cpu=False,
        diagnostics=diagnostics,
    )


def exact_compute_fidelity(
    ctx: FidelityContext,
    max_layer_distance: int = None,
    pair_selection: str = "all",
    selected_pairs: Optional[Set[Tuple[int, int]]] = None,
    store_pair_C: bool = False,
    pair_C_to_cpu: bool = False,
    source_mode_masks: Optional[dict[int, list[int]]] = None,
    target_mode_masks: Optional[dict[int, list[int]]] = None,
) -> FidelityResult:
    """
    Exact full-state connected fidelity computation.

    Args:
        ctx: FidelityContext with circuit definition and Kraus operators.
        max_layer_distance: None = all future layers; 0 = same-layer only.
        pair_selection: "all" includes all valid pairs.
        selected_pairs: optional explicit set of (source_gate_idx,
            target_gate_idx) pairs to evaluate after the standard pair rules.
        store_pair_C: whether to retain finalized selected amplitude matrices
            in the historical pair_C result field.
        pair_C_to_cpu: move retained amplitude matrices to CPU.
        source_mode_masks: optional fixed source gate -> mode-ID mask.
        target_mode_masks: optional fixed target gate -> mode-ID mask.

    Returns:
        FidelityResult
    """
    n_qubits = ctx.n_qubits
    dim = 2**n_qubits
    device = ctx.device
    dtype = ctx.dtype

    # ---- Step 1: Evolve ideal state, cache psi_{g+1} and one-point amplitudes ----
    psi_ideal = torch.zeros(dim, dtype=dtype, device=device)
    psi_ideal[0] = 1.0

    state_history = []  # state_history[g] = |psi_{g+1}>
    local_a = {}  # gate_idx -> a_u tensor (n_u,)
    local_F = {}  # gate_idx -> scalar F_g
    dressed_ops = {}  # gate_idx -> E_u tensor (n_u,d,d)

    log_F1 = 0.0

    for layer_idx, layer_gates in enumerate(ctx.layers):
        for gate_info in layer_gates:
            ideal_u = gate_info["ideal_unitary"]
            qubits = gate_info["qubits"]
            if len(qubits) == 1:
                psi_ideal = _apply_1q_gate(psi_ideal, ideal_u, qubits[0], n_qubits)
            else:
                psi_ideal = _apply_2q_gate(
                    psi_ideal, ideal_u, qubits[0], qubits[1], n_qubits
                )

        state_history.append(psi_ideal.clone())

        # Compute one-point amplitudes for all gates in this layer
        for gate_info in layer_gates:
            g_idx = gate_info["gate_idx"]
            K = gate_info["kraus_ops"]
            U = gate_info["ideal_unitary"]
            qubits = gate_info["qubits"]

            E_u = compute_dressed_kraus(K, U)
            dressed_ops[g_idx] = E_u
            rho = _get_local_rho(psi_ideal, qubits, n_qubits)
            a_u = compute_one_point_amplitude(E_u, rho)
            local_a[g_idx] = a_u

            F_g = torch.sum(torch.abs(a_u) ** 2).item()
            local_F[g_idx] = F_g
            log_F1 += torch.log(torch.tensor(F_g, dtype=torch.float64)).item()

    def _normalize_mode_masks(
        masks: Optional[dict[int, list[int]]],
        label: str,
    ) -> dict[int, torch.Tensor]:
        normalized = {}
        if masks is None:
            return normalized
        for raw_gate_id, raw_ids in masks.items():
            gate_id = int(raw_gate_id)
            if gate_id not in local_a:
                raise ValueError(f"{label} contains unknown gate {gate_id}")
            ids = torch.as_tensor(raw_ids, dtype=torch.long, device=device)
            if ids.ndim != 1 or ids.numel() == 0:
                raise ValueError(
                    f"{label}[{gate_id}] must contain at least one mode ID"
                )
            count = int(local_a[gate_id].numel())
            if bool((ids < 0).any()) or bool((ids >= count).any()):
                raise ValueError(f"{label}[{gate_id}] contains an out-of-range mode ID")
            if int(torch.unique(ids).numel()) != int(ids.numel()):
                raise ValueError(f"{label}[{gate_id}] contains duplicate mode IDs")
            normalized[gate_id] = ids
        return normalized

    source_mode_indices = _normalize_mode_masks(source_mode_masks, "source_mode_masks")
    target_mode_indices = _normalize_mode_masks(target_mode_masks, "target_mode_masks")

    # A mode-matched comparison uses the fixed BP masks in the one-point
    # normalization factors while leaving centered operator construction based
    # on the exact, unmasked amplitudes.
    masked_local_F = dict(local_F)
    masked_log_F1 = 0.0
    for g_idx, amplitudes in local_a.items():
        ids = source_mode_indices.get(g_idx)
        if ids is None:
            ids = target_mode_indices.get(g_idx)
        if ids is None:
            ids = torch.arange(int(amplitudes.numel()), dtype=torch.long, device=device)
        value = torch.sum(
            amplitudes.index_select(0, ids).abs().square(),
            dtype=torch.float64,
        )
        masked_local_F[g_idx] = float(value.item())
        masked_log_F1 += torch.log(value).item()

    pair_formula = os.environ.get("PEPS_CORR2BP_PAIR_FORMULA", "centered_C")
    if pair_formula not in {"centered_C", "raw_A"}:
        raise ValueError("PEPS_CORR2BP_PAIR_FORMULA must be 'centered_C' or 'raw_A'.")

    # ---- Step 2: Build centered operators ----
    centered_ops = {}  # gate_idx -> E_center tensor (n_u, *dims, *dims)
    for g_idx in local_a:
        centered_ops[g_idx] = compute_centered_operator(
            dressed_ops[g_idx], local_a[g_idx]
        )

    if pair_formula == "centered_C" and not store_pair_C:
        return _exact_compute_fidelity_flat_centered(
            ctx=ctx,
            max_layer_distance=max_layer_distance,
            selected_pairs=selected_pairs,
            state_history=state_history,
            local_a=local_a,
            local_F=masked_local_F,
            centered_ops=centered_ops,
            log_F1=masked_log_F1,
            source_mode_indices=source_mode_indices,
            target_mode_indices=target_mode_indices,
        )

    # The raw-A path below shares the same masked normalization factors.
    local_F = masked_local_F
    log_F1 = masked_log_F1

    # ---- Step 3: Layer-window pair loop - compute selected matrix and Delta ----
    pair_deltas = {}
    pair_C = {}
    selected_source_gates = (
        None if selected_pairs is None else {int(a_idx) for a_idx, _ in selected_pairs}
    )
    branch_chunk_modes = _positive_int_env(
        "CENTERED_FIDELITY_EXACT_BRANCH_CHUNK_MODES",
        128,
    )
    progress = os.environ.get("CENTERED_FIDELITY_PROGRESS", "0") == "1"
    active_groups = []
    exact_diag = {
        "layer_window_pipeline": True,
        "exact_branch_chunk_modes": branch_chunk_modes,
        "source_groups_spawned": 0,
        "source_modes_spawned": 0,
        "source_groups_pruned_before_readout": 0,
        "source_modes_pruned_before_readout": 0,
        "source_groups_pruned_before_propagation": 0,
        "source_modes_pruned_before_propagation": 0,
        "max_active_source_groups": 0,
        "max_active_source_modes": 0,
        "max_exact_branch_chunk_modes": 0,
        "target_pairs_evaluated": 0,
        "source_group_layer_propagations": 0,
        "branch_readout_chunks": 0,
        "branch_propagation_chunks": 0,
        "target_mode_batch_applications": 0,
        "local_transition_readouts": 0,
        "pair_mode_diagnostics": {},
    }
    record_pair_modes = os.environ.get("CENTERED_PAIR_MODE_DIAGNOSTICS", "0") == "1"

    for layer_idx, layer_gates in enumerate(ctx.layers):
        layer_pairs_before = exact_diag["target_pairs_evaluated"]
        layer_readout_chunks_before = exact_diag["branch_readout_chunks"]
        layer_prop_chunks_before = exact_diag["branch_propagation_chunks"]
        layer_prop_groups_before = exact_diag["source_group_layer_propagations"]
        if progress:
            print(
                "[exact-layer] "
                f"start {layer_idx + 1}/{len(ctx.layers)} "
                f"active_groups={len(active_groups)} "
                f"active_modes={_sum_group_modes(active_groups)} "
                f"completed_pairs={len(pair_deltas)}",
                flush=True,
            )

        if max_layer_distance is not None and active_groups:
            groups_before = len(active_groups)
            modes_before = _sum_group_modes(active_groups)
            active_groups = [
                group
                for group in active_groups
                if layer_idx - int(group["source_layer"]) <= max_layer_distance
            ]
            exact_diag["source_groups_pruned_before_readout"] += groups_before - len(
                active_groups
            )
            exact_diag["source_modes_pruned_before_readout"] += (
                modes_before - _sum_group_modes(active_groups)
            )

        # Spawn current-layer source branches after this layer's ideal state.
        for gate_a in layer_gates:
            a_idx = gate_a["gate_idx"]
            if selected_source_gates is not None and a_idx not in selected_source_gates:
                continue

            qubits_a = gate_a["qubits"]
            E_a_source = (
                dressed_ops[a_idx] if pair_formula == "raw_A" else centered_ops[a_idx]
            )
            source_ids = source_mode_indices.get(a_idx)
            if source_ids is None:
                source_ids = torch.arange(
                    int(E_a_source.shape[0]), dtype=torch.long, device=device
                )
            E_a_source = E_a_source.index_select(0, source_ids)
            n_u = E_a_source.shape[0]
            source_base = state_history[layer_idx].expand(n_u, -1).clone()
            if len(qubits_a) == 1:
                branches = _apply_1q_gate_per_state(
                    source_base, E_a_source, qubits_a[0], n_qubits
                )
            else:
                branches = _apply_2q_gate_per_state(
                    source_base, E_a_source, qubits_a[0], qubits_a[1], n_qubits
                )
            active_groups.append(
                {
                    "source_layer": layer_idx,
                    "source_gate_idx": a_idx,
                    "branches": branches,
                    "mode_ids": source_ids,
                }
            )
            exact_diag["source_groups_spawned"] += 1
            exact_diag["source_modes_spawned"] += int(n_u)

        exact_diag["max_active_source_groups"] = max(
            exact_diag["max_active_source_groups"],
            len(active_groups),
        )
        exact_diag["max_active_source_modes"] = max(
            exact_diag["max_active_source_modes"],
            _sum_group_modes(active_groups),
        )

        # Read current-layer targets against all active source groups.
        psi_target = state_history[layer_idx]
        psi_target_conj = psi_target.conj()
        for gate_b in layer_gates:
            b_idx = gate_b["gate_idx"]
            eligible_groups = []
            for group in active_groups:
                a_idx = group["source_gate_idx"]
                source_layer = group["source_layer"]
                if a_idx == b_idx:
                    continue
                if max_layer_distance is not None:
                    if layer_idx - source_layer > max_layer_distance:
                        continue
                if layer_idx == source_layer and a_idx > b_idx:
                    continue
                if selected_pairs is not None and (a_idx, b_idx) not in selected_pairs:
                    continue
                eligible_groups.append(group)

            if not eligible_groups:
                continue

            qubits_b = gate_b["qubits"]
            E_b_readout = (
                dressed_ops[b_idx] if pair_formula == "raw_A" else centered_ops[b_idx]
            )
            target_ids = target_mode_indices.get(b_idx)
            if target_ids is None:
                target_ids = torch.arange(
                    int(E_b_readout.shape[0]), dtype=torch.long, device=device
                )
            E_b_readout = E_b_readout.index_select(0, target_ids)
            n_v = E_b_readout.shape[0]

            for group_chunk in _iter_group_chunks(eligible_groups, branch_chunk_modes):
                branches = _concat_group_branches(group_chunk)
                chunk_modes = int(branches.shape[0])
                exact_diag["branch_readout_chunks"] += 1
                exact_diag["max_exact_branch_chunk_modes"] = max(
                    exact_diag["max_exact_branch_chunk_modes"],
                    chunk_modes,
                )
                transition = _local_transition_overlap(
                    psi_target, branches, qubits_b, n_qubits
                )
                E_target = E_b_readout.reshape(
                    n_v, transition.shape[-2], transition.shape[-1]
                )
                values_by_branch = torch.einsum("voi,boi->bv", E_target, transition)
                exact_diag["local_transition_readouts"] += 1
                exact_diag["target_mode_batch_applications"] += n_v

                C_by_group = []
                offset = 0
                for group in group_chunk:
                    modes = _group_mode_count(group)
                    C_by_group.append(
                        values_by_branch[offset : offset + modes].T.contiguous()
                    )
                    offset += modes

                for group, C_vu in zip(group_chunk, C_by_group):
                    a_idx = group["source_gate_idx"]
                    b_idx = gate_b["gate_idx"]
                    F_a_tensor = torch.tensor(
                        local_F[a_idx], dtype=dtype, device=device
                    )
                    F_b_tensor = torch.tensor(
                        local_F[b_idx], dtype=dtype, device=device
                    )
                    if pair_formula == "raw_A":
                        delta = delta_from_raw_A(
                            C_vu,
                            F_a_tensor,
                            F_b_tensor,
                        )
                    else:
                        source_mode_ids = group["mode_ids"]
                        delta = delta_from_centered_C(
                            C_vu,
                            local_a[a_idx].index_select(0, source_mode_ids),
                            local_a[b_idx].index_select(0, target_ids),
                            F_a_tensor,
                            F_b_tensor,
                        )
                    pair_deltas[(a_idx, b_idx)] = delta.item()
                    exact_diag["target_pairs_evaluated"] += 1
                    if record_pair_modes:
                        exact_diag["pair_mode_diagnostics"][(a_idx, b_idx)] = (
                            pair_mode_diagnostic_from_centered_C(
                                C_vu, local_a[a_idx], local_a[b_idx]
                            )
                        )
                    if store_pair_C:
                        pair_C[(a_idx, b_idx)] = (
                            C_vu.detach().cpu() if pair_C_to_cpu else C_vu
                        )

        # Drop groups that cannot contribute to the next layer before paying
        # to propagate their full state-vector branches.
        if max_layer_distance is not None and active_groups:
            groups_before = len(active_groups)
            modes_before = _sum_group_modes(active_groups)
            active_groups = [
                group
                for group in active_groups
                if (layer_idx + 1) - int(group["source_layer"]) <= max_layer_distance
            ]
            exact_diag["source_groups_pruned_before_propagation"] += (
                groups_before - len(active_groups)
            )
            exact_diag["source_modes_pruned_before_propagation"] += (
                modes_before - _sum_group_modes(active_groups)
            )

        # Propagate active source branches through the next ideal layer so they
        # are at |B_{a,u}^{(layer+2)}> on the next iteration.
        if layer_idx + 1 < len(ctx.layers):
            _propagate_group_chunks(
                active_groups,
                ctx.layers[layer_idx + 1],
                n_qubits,
                branch_chunk_modes,
                exact_diag,
            )

        if progress:
            print(
                "[exact-layer] "
                f"done {layer_idx + 1}/{len(ctx.layers)} "
                f"active_groups={len(active_groups)} "
                f"active_modes={_sum_group_modes(active_groups)} "
                f"new_pairs={exact_diag['target_pairs_evaluated'] - layer_pairs_before} "
                f"completed_pairs={len(pair_deltas)} "
                f"readout_chunks={exact_diag['branch_readout_chunks'] - layer_readout_chunks_before} "
                f"prop_chunks={exact_diag['branch_propagation_chunks'] - layer_prop_chunks_before} "
                f"prop_groups={exact_diag['source_group_layer_propagations'] - layer_prop_groups_before}",
                flush=True,
            )

    log_F_connected = log_F1 + sum(pair_deltas.values())

    # Build self_energy_by_layer
    self_energy_by_target_layer = {}
    self_energy_by_source_layer = {}
    for (a_idx, b_idx), delta_val in pair_deltas.items():
        if hasattr(ctx, "gate_map") and b_idx in ctx.gate_map:
            b_layer = ctx.gate_map[b_idx][0]
            self_energy_by_target_layer[b_layer] = (
                self_energy_by_target_layer.get(b_layer, 0.0) + delta_val
            )
        if hasattr(ctx, "gate_map") and a_idx in ctx.gate_map:
            a_layer = ctx.gate_map[a_idx][0]
            self_energy_by_source_layer[a_layer] = (
                self_energy_by_source_layer.get(a_layer, 0.0) + delta_val
            )

    return FidelityResult(
        log_fidelity_first_order=log_F1,
        fidelity_first_order=torch.exp(torch.tensor(log_F1)).item(),
        log_fidelity_connected=log_F_connected,
        fidelity_connected=torch.exp(torch.tensor(log_F_connected)).item(),
        local_fidelities=local_F,
        pair_deltas=pair_deltas,
        pair_C=pair_C,
        self_energy_by_layer=self_energy_by_target_layer,
        self_energy_by_source_layer=self_energy_by_source_layer,
        diagnostics={
            "backend": "exact",
            "max_layer_distance": max_layer_distance,
            "selected_pairs": None if selected_pairs is None else len(selected_pairs),
            "store_pair_C": store_pair_C,
            "pair_C_to_cpu": pair_C_to_cpu,
            "pair_formula": pair_formula,
            "exact_branch_mode_batching": True,
            **exact_diag,
        },
    )
