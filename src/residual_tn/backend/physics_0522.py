from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

NOISE_T_US = 5.0  # Driver sets T1=T2 explicitly in microseconds.

from residual_tn.backend.channel import superoperator_to_kraus
from residual_tn.backend.context import FidelityContext


def row_major_neighbors(width: int, length: int) -> List[Tuple[int, int]]:
    out = []
    for q in range(width * length):
        row, col = divmod(q, width)
        if col < width - 1:
            out.append(tuple(sorted((q, q + 1))))
        if row < length - 1:
            out.append(tuple(sorted((q, q + width))))
    return out


def _batch_kron(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    sA, sB = A.shape, B.shape
    N, M = sA[-2], sA[-1]
    P, Q = sB[-2], sB[-1]
    return (A.unsqueeze(-1).unsqueeze(-3) * B.unsqueeze(-2).unsqueeze(-4)).reshape(
        *sA[:-2], N * P, M * Q
    )


def _ladder(dim: int, device: torch.device, dtype: torch.dtype):
    a = torch.zeros((dim, dim), dtype=dtype, device=device)
    for i in range(1, dim):
        a[i - 1, i] = math.sqrt(i)
    adag = a.conj().T
    num = adag @ a
    eye = torch.eye(dim, dtype=dtype, device=device)
    return a, adag, num, eye


def _ideal_single(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    sq2 = 2**0.5 / 2.0
    return torch.stack(
        [
            torch.tensor(
                [[sq2, -1j * sq2], [-1j * sq2, sq2]], dtype=dtype, device=device
            ),
            torch.tensor([[sq2, -sq2], [sq2, sq2]], dtype=dtype, device=device),
            torch.tensor(
                [[sq2, -0.5 - 0.5j], [0.5 - 0.5j, sq2]], dtype=dtype, device=device
            ),
            torch.eye(2, dtype=dtype, device=device),
        ]
    )


def _ideal_two(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    gate = torch.tensor(
        [
            [1.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j],
            [0.0000 + 0.0000j, 0.7660 + 0.0000j, 0.0000 + 0.6429j, 0.0000 + 0.0000j],
            [0.0000 + 0.0000j, 0.0000 + 0.6429j, 0.7660 + 0.0000j, 0.0000 + 0.0000j],
            [0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.9970 - 0.0767j],
        ],
        dtype=dtype,
        device=device,
    )
    u, _, vh = torch.linalg.svd(gate)
    return u @ vh


def _drag_coeffs(n_qubits: int, device: torch.device) -> torch.Tensor:
    vals = [
        -0.0224,
        -0.0253,
        -0.0935,
        -0.0756,
        -0.0228,
        -0.0243,
        -0.0988,
        -0.0820,
        -0.0458,
        -0.0382,
        -0.1060,
        -0.1068,
        -0.0224,
        -0.0253,
        -0.0935,
        -0.0756,
        -0.0224,
        -0.0253,
        -0.0935,
        -0.0756,
        -0.0228,
        -0.0243,
        -0.0988,
        -0.0820,
        -0.0458,
        -0.0382,
        -0.1060,
        -0.1068,
        -0.0224,
        -0.0253,
        -0.0935,
        -0.0756,
        -0.0228,
        -0.0243,
        -0.0988,
        -0.0820,
        -0.0224,
        -0.0253,
        -0.0935,
        -0.0756,
        -0.0228,
        -0.0243,
        -0.0988,
        -0.0820,
        -0.0458,
        -0.0382,
        -0.1060,
        -0.1068,
        -0.0224,
        -0.0253,
        -0.0935,
        -0.0756,
        -0.0224,
        -0.0253,
        -0.0935,
        -0.0756,
        -0.0228,
        -0.0243,
        -0.0988,
        -0.0820,
        -0.0458,
        -0.0382,
        -0.1060,
        -0.1068,
        -0.0224,
        -0.0253,
        -0.0935,
        -0.0756,
        -0.0228,
        -0.0243,
        -0.0988,
        -0.0820,
    ]
    reps = (n_qubits + len(vals) - 1) // len(vals)
    return torch.tensor((vals * reps)[:n_qubits], dtype=torch.float64, device=device)


def generate_random_circuit_patterns(
    batch_size: int,
    n_layers: int,
    n_qubits: int,
    all_neighbors: Sequence[Tuple[int, int]],
    seed: int | None = 0,
):
    np_rng = (
        np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    )
    py_rng = random.Random(seed) if seed is not None else random.Random()
    single_gates_pool = np_rng.randint(0, 4, size=(batch_size, n_layers, n_qubits))
    all_pattern = []
    for b in range(batch_size):
        circuit_layers = []
        for d in range(n_layers):
            if d % 2 == 0:
                circuit_layers.append(single_gates_pool[b, d, :])
            else:
                shuffled_edges = list(all_neighbors)
                py_rng.shuffle(shuffled_edges)
                selected_pairs = []
                used_qubits = set()
                for u, v in shuffled_edges:
                    if u not in used_qubits and v not in used_qubits:
                        selected_pairs.append([u, v])
                        used_qubits.add(u)
                        used_qubits.add(v)
                circuit_layers.append(selected_pairs)
        all_pattern.append(circuit_layers)
    return all_pattern


def calculate_mapping_cost(circuit_pattern, mapping_1d_to_phys):
    phys_to_1d = {phys_idx: pos for pos, phys_idx in enumerate(mapping_1d_to_phys)}
    total_cost = 0
    max_distance = 0
    for layer in circuit_pattern:
        if (
            isinstance(layer, (list, tuple))
            and len(layer) > 0
            and isinstance(layer[0], (list, tuple, np.ndarray))
        ):
            for pair in layer:
                dist = abs(phys_to_1d[int(pair[0])] - phys_to_1d[int(pair[1])])
                total_cost += dist
                max_distance = max(max_distance, dist)
    return total_cost, max_distance


def find_optimal_mapping_sa(
    circuit_pattern, n_qubits: int, steps: int = 200000, seed: int | None = 0
):
    rng = random.Random(seed) if seed is not None else random.Random()
    current = list(range(n_qubits))
    current_cost, _ = calculate_mapping_cost(circuit_pattern, current)
    best = current[:]
    best_cost = current_cost
    temp = 100.0
    cooling = 0.995
    for _ in range(max(0, int(steps))):
        new = current[:]
        i, j = rng.sample(range(n_qubits), 2)
        new[i], new[j] = new[j], new[i]
        new_cost, _ = calculate_mapping_cost(circuit_pattern, new)
        if new_cost < current_cost or rng.random() < math.exp(
            (current_cost - new_cost) / max(temp, 1e-300)
        ):
            current = new
            current_cost = new_cost
            if current_cost < best_cost:
                best = current[:]
                best_cost = current_cost
        temp *= cooling
    return best, best_cost


def _single_superoperators(
    width: int, length: int, piece_num: int, device: torch.device, dtype: torch.dtype
):
    n_qubits = width * length
    dim = 3
    dim2 = 9
    dtype_r = torch.float64
    a, adag, num, eye = _ladder(dim, device, dtype)
    t1 = (NOISE_T_US * 1e3) * torch.ones(n_qubits, dtype=dtype_r, device=device)
    t2 = (NOISE_T_US * 1e3) * torch.ones(n_qubits, dtype=dtype_r, device=device)
    anh = -2 * math.pi * 242 * 1e-3 * torch.ones(n_qubits, dtype=dtype_r, device=device)
    drive_duration = torch.max(8 * math.pi / torch.abs(anh))
    basefreq = 4.9
    irr1 = (np.sqrt(5) - 1) / 2
    irr2 = (np.sqrt(7) - 1) / 2
    ov = 0.006
    q_freqs = torch.zeros(n_qubits, dtype=dtype_r, device=device)
    for i in range(n_qubits):
        x = i % width
        y = int(i / width)
        q_freqs[i] = basefreq + abs(0.002) / ov * (
            np.cos(x * 2 * np.pi * irr1) + np.cos(y * 2 * np.pi * irr2)
        )
    drive_freqs = q_freqs.clone()
    drive_omegas = 0.2051 * torch.ones(n_qubits, dtype=dtype_r, device=device) / 2
    drags = _drag_coeffs(n_qubits, device)
    phase_map = torch.tensor(
        [0, math.pi / 2, math.pi / 4], dtype=dtype_r, device=device
    )
    P = 4
    T = int(piece_num)
    eye_flat = torch.eye(dim2, dtype=dtype, device=device)
    D_ops = torch.zeros((n_qubits, 1, dim2, dim2), dtype=dtype, device=device)
    term1 = torch.kron(a.conj(), a)
    nadaga = adag @ a
    term2 = -0.5 * (torch.kron(eye, nadaga) + torch.kron(nadaga.T.contiguous(), eye))
    D_ops += (1.0 / t1).reshape(n_qubits, 1, 1, 1) * (term1 + term2).reshape(
        1, 1, dim2, dim2
    )
    term1 = torch.kron(num.conj(), num)
    n2 = num @ num
    term2 = -0.5 * (torch.kron(eye, n2) + torch.kron(n2.T.contiguous(), eye))
    D_ops += (1.0 / t2).reshape(n_qubits, 1, 1, 1) * (term1 + term2).reshape(
        1, 1, dim2, dim2
    )
    D_ops = D_ops.expand(n_qubits, P, dim2, dim2)
    if T <= 1:
        raise ValueError(
            "piece_num must be greater than 1 for qutrit channel generation."
        )
    drive_duration_value = float(drive_duration.detach().cpu().item())
    t_raw = torch.linspace(0, drive_duration_value, T, dtype=dtype_r, device=device)
    t_mid = (t_raw[:-1] + t_raw[1:]) / 2
    t_grid = torch.cat([t_mid, t_raw[-1].unsqueeze(0)], dim=0)
    dt = drive_duration / (T - 1)
    t_rel = t_grid.reshape(T, 1, 1)
    sigma = drive_duration / 4.0
    center = drive_duration / 2.0
    gaussian = torch.exp(-0.5 * ((t_rel - center) / sigma) ** 2)
    env_deriv = -(t_rel - center) / (sigma**2) * gaussian
    shift = math.exp(-2.0)
    env = gaussian - shift
    beta = drive_omegas.reshape(1, n_qubits, 1) * env + 1j * (
        drags.reshape(1, n_qubits, 1) * env_deriv / anh.reshape(1, n_qubits, 1)
    )
    beta = beta.expand(T, n_qubits, P).clone()
    phis = phase_map.reshape(1, 1, -1).to(device)
    total_phase = (q_freqs - drive_freqs).reshape(1, n_qubits, 1) * t_rel + phis
    beta[:, :, : P - 1] *= torch.exp(1j * total_phase)
    beta[:, :, -1] = 0
    nn_minus_n = num @ (num - eye)
    H_static = (
        0.5 * anh.reshape(1, n_qubits, 1, 1, 1) * nn_minus_n.reshape(1, 1, 1, dim, dim)
    )
    H_static = H_static.expand(T, n_qubits, P, dim, dim)
    H_all = (
        H_static
        + beta.reshape(T, n_qubits, P, 1, 1) * adag.reshape(1, 1, 1, dim, dim)
        + beta.conj().reshape(T, n_qubits, P, 1, 1) * a.reshape(1, 1, 1, dim, dim)
    )
    H_kron_I = (
        H_all.unsqueeze(-1).unsqueeze(-3) * eye.reshape(1, 1, 1, 1, 3, 1, 3)
    ).reshape(T, n_qubits, P, dim2, dim2)
    I_kron_HT = (
        eye.reshape(1, 1, 1, 3, 1, 3, 1)
        * H_all.transpose(-1, -2).unsqueeze(-2).unsqueeze(-4)
    ).reshape(T, n_qubits, P, dim2, dim2)
    L_total = -1j * (H_kron_I - I_kron_HT) + D_ops.unsqueeze(0)
    eye_super = eye_flat.reshape(1, 1, 1, dim2, dim2).expand(T, n_qubits, P, dim2, dim2)
    S_slices = torch.linalg.solve(
        (eye_super - 0.5 * dt * L_total).reshape(T * n_qubits * P, dim2, dim2),
        (eye_super + 0.5 * dt * L_total).reshape(T * n_qubits * P, dim2, dim2),
    ).view(T, n_qubits, P, dim2, dim2)
    S_cum = S_slices[0]
    for t in range(1, T):
        S_cum = S_slices[t] @ S_cum
    return S_cum


def _two_superoperator(piece_num: int, device: torch.device, dtype: torch.dtype):
    dim = 3
    dim2 = 9
    dim4 = 81
    dtype_r = torch.float64
    op_a, op_adag, _, eye = _ladder(dim, device, dtype)
    op_adag = op_adag.contiguous()
    a_L = torch.kron(op_a, eye).reshape(1, 1, dim2, dim2)
    adag_L = torch.kron(op_adag, eye).reshape(1, 1, dim2, dim2)
    a_R = torch.kron(eye, op_a).reshape(1, 1, dim2, dim2)
    adag_R = torch.kron(eye, op_adag).reshape(1, 1, dim2, dim2)
    n_L = adag_L @ a_L
    n_R = adag_R @ a_R
    eye_M = torch.eye(dim2, dtype=dtype, device=device).view(1, 1, dim2, dim2)
    eye_super = torch.eye(dim4, dtype=dtype, device=device).view(1, 1, dim4, dim4)

    def dissipator(op, op_dag, gamma):
        jump = op_dag @ op
        return gamma * (
            _batch_kron(op, op.conj())
            - 0.5
            * (_batch_kron(jump, eye_M) + _batch_kron(eye_M, jump.transpose(-1, -2)))
        )

    gamma = torch.tensor(1.0 / (NOISE_T_US * 1e3), dtype=dtype_r, device=device).view(
        1, 1, 1, 1
    )
    D_ops = (
        dissipator(a_L, adag_L, gamma)
        + dissipator(a_R, adag_R, gamma)
        + dissipator(n_L, n_L, gamma)
        + dissipator(n_R, n_R, gamma)
    )
    anh = torch.tensor(-2 * math.pi * 242 * 1e-3, dtype=dtype_r, device=device)
    drive_duration = 8 * math.pi / torch.abs(anh)
    T = int(piece_num)
    if T <= 1:
        raise ValueError(
            "piece_num must be greater than 1 for qutrit channel generation."
        )
    dt = drive_duration / (T - 1)
    H_static = 0.5 * anh * (n_L @ (n_L - eye_M)) + 0.5 * anh * (n_R @ (n_R - eye_M))
    H_dynamic = torch.tensor(-0.0422, dtype=dtype_r, device=device).view(1, 1, 1, 1) * (
        (adag_L @ a_R) + (a_L @ adag_R)
    )
    H_total = H_static + H_dynamic
    L_total = (
        -1j
        * (_batch_kron(H_total, eye_M) - _batch_kron(eye_M, H_total.transpose(-1, -2)))
        + D_ops
    )
    S_step = torch.linalg.solve(
        (eye_super - 0.5 * dt * L_total).reshape(dim4, dim4),
        (eye_super + 0.5 * dt * L_total).reshape(dim4, dim4),
    )
    S_cum = torch.eye(dim4, dtype=dtype, device=device)
    for _ in range(T):
        S_cum = S_step @ S_cum
    return S_cum


def _p_map(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    P = torch.zeros((9, 2, 2), dtype=dtype, device=device)
    P[0, 0, 0] = 1.0
    P[1, 0, 1] = 1.0
    P[3, 1, 0] = 1.0
    P[4, 1, 1] = 1.0
    return P


def fix_superoperator_indexing(S: torch.Tensor) -> torch.Tensor:
    return (
        S.view(3, 3, 3, 3, 3, 3, 3, 3)
        .permute(0, 2, 1, 3, 4, 6, 5, 7)
        .reshape(9, 9, 9, 9)
    )


def _load_pattern_from_file(
    pattern_file: str,
    width: int,
    length: int,
    n_layers: int,
    n_qubits: int,
    all_neighbors: Sequence[Tuple[int, int]],
    pattern_index: int,
):
    path = Path(pattern_file).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Pattern file not found: {path}")
    patterns = torch.load(path, map_location="cpu", weights_only=False)
    if not patterns:
        raise ValueError(f"Pattern file is empty: {path}")
    pattern = patterns[int(pattern_index) % len(patterns)]
    if len(pattern) < n_layers:
        raise ValueError(
            f"Pattern file has only {len(pattern)} layers, requested {n_layers}."
        )
    pattern = list(pattern[:n_layers])
    allowed_edges = {tuple(sorted(pair)) for pair in all_neighbors}
    for layer_idx, layer in enumerate(pattern):
        if layer_idx % 2 == 0:
            if len(layer) != n_qubits:
                raise ValueError(
                    f"Pattern layer {layer_idx} has {len(layer)} 1Q modes, expected {n_qubits}."
                )
            continue
        used = set()
        for pair in layer:
            u, v = int(pair[0]), int(pair[1])
            if tuple(sorted((u, v))) not in allowed_edges:
                raise ValueError(
                    f"Pattern layer {layer_idx} has non-neighbor pair {(u, v)} for {width}x{length}."
                )
            if u in used or v in used:
                raise ValueError(
                    f"Pattern layer {layer_idx} has overlapping pair {(u, v)}."
                )
            used.add(u)
            used.add(v)
    return pattern, {"pattern_file": str(path), "pattern_file_samples": len(patterns)}


def build_generated_0522_context(
    width: int,
    length: int,
    n_layers: int,
    device: torch.device,
    dtype: torch.dtype = torch.complex128,
    *,
    piece_num: int = 1000,
    pattern_seed: int | None = 0,
    pattern_index: int = 6,
    pattern_batch_size: int = 50,
    pattern_file: str | None = None,
    mapping_steps: int = 0,
    mapping_seed: int | None = 0,
    mapping_mode: str = "identity",
    kraus_tol: float = 1e-8,
):
    """Build a generated 0522-style context.

    Inputs/outputs use computational dimension 2. Qubit labels in generated
    contexts default to one convention: physical lattice label equals
    state-vector axis equals PEPS site. Set ``mapping_mode="annealed"`` only
    for legacy/debug comparisons that intentionally use a nontrivial
    ``phys_to_1d`` permutation.
    """
    n_qubits = width * length
    all_neighbors = row_major_neighbors(width, length)
    pair_to_idx = {tuple(sorted(pair)): i for i, pair in enumerate(all_neighbors)}
    pattern_meta: Dict[str, Any] = {}
    if pattern_file:
        pattern, pattern_meta = _load_pattern_from_file(
            pattern_file,
            width,
            length,
            n_layers,
            n_qubits,
            all_neighbors,
            pattern_index,
        )
        pattern_source = "file"
    else:
        patterns = generate_random_circuit_patterns(
            pattern_batch_size, n_layers, n_qubits, all_neighbors, pattern_seed
        )
        pattern = patterns[int(pattern_index) % len(patterns)]
        pattern_source = "generated"
    if mapping_mode == "identity":
        mapping = list(range(n_qubits))
        mapping_cost, _ = calculate_mapping_cost(pattern, mapping)
        mapping_steps_used = 0
    elif mapping_mode == "annealed":
        mapping, mapping_cost = find_optimal_mapping_sa(
            pattern, n_qubits, steps=mapping_steps, seed=mapping_seed
        )
        mapping_steps_used = int(mapping_steps)
    else:
        raise ValueError(
            f"Unsupported mapping_mode={mapping_mode!r}; use 'identity' or 'annealed'."
        )
    phys_to_1d = {phys: pos for pos, phys in enumerate(mapping)}
    ideal_single = _ideal_single(dtype, device)
    ideal_two = _ideal_two(dtype, device)
    S_single = _single_superoperators(width, length, piece_num, device, dtype)
    P = _p_map(device, dtype)
    kraus_single: Dict[Tuple[int, int], torch.Tensor] = {}
    for phys_u in range(n_qubits):
        for mode in range(4):
            S_proj = torch.einsum(
                "xab,xy,ycd->abcd", P, S_single[phys_u, mode], P
            ).reshape(4, 4)
            kraus_single[(phys_u, mode)] = superoperator_to_kraus(
                S_proj, d=2, tol=kraus_tol
            )
    S_2q_fixed = fix_superoperator_indexing(
        _two_superoperator(piece_num, device, dtype)
    )
    S_2q_proj = torch.einsum("uab,vcd,uvij,ief,jgh->abcdefgh", P, P, S_2q_fixed, P, P)
    S_2q_reordered = (
        S_2q_proj.permute(0, 2, 1, 3, 4, 6, 5, 7).contiguous().reshape(16, 16)
    )
    kraus_2q_ops = superoperator_to_kraus(S_2q_reordered, d=4, tol=kraus_tol)
    ctx = FidelityContext(n_qubits, width, length, device, dtype)
    gate_table: Dict[int, Dict[str, Any]] = {}
    gate_idx = 0
    for layer_idx in range(n_layers):
        pat = pattern[layer_idx]
        gates = []
        if layer_idx % 2 == 0:
            for phys_u in range(n_qubits):
                mode = int(pat[phys_u])
                q = int(phys_u)
                q_1d = int(phys_to_1d[phys_u])
                gates.append(
                    {
                        "gate_idx": gate_idx,
                        "qubits": (q,),
                        "ideal_unitary": ideal_single[mode],
                        "kraus_ops": kraus_single[(phys_u, mode)],
                        "mode": mode,
                        "control_kind": "single_qutrit_drag",
                    }
                )
                gate_table[gate_idx] = {
                    "layer": layer_idx,
                    "qubits": (q,),
                    "arity": 1,
                    "gate_id_0522": int(phys_u),
                    "physical_qubits": [int(phys_u)],
                    "package_qubits": [q],
                    "mapped_qubits_1d": [q_1d],
                    "mode": mode,
                }
                gate_idx += 1
        else:
            for gate_id_in_layer, pair in enumerate(pat):
                u, v = int(pair[0]), int(pair[1])
                q_u, q_v = u, v
                mapped_u, mapped_v = int(phys_to_1d[u]), int(phys_to_1d[v])
                idx = pair_to_idx[tuple(sorted((u, v)))]
                gates.append(
                    {
                        "gate_idx": gate_idx,
                        "qubits": (q_u, q_v),
                        "ideal_unitary": ideal_two,
                        "kraus_ops": kraus_2q_ops,
                        "control_kind": "two_qutrit_exchange",
                    }
                )
                gate_table[gate_idx] = {
                    "layer": layer_idx,
                    "qubits": (q_u, q_v),
                    "arity": 2,
                    "gate_id_0522": [u, v],
                    "gate_id_0522_layer_enum": int(gate_id_in_layer),
                    "physical_qubits": [u, v],
                    "package_qubits": [q_u, q_v],
                    "mapped_qubits_1d": [mapped_u, mapped_v],
                    "neighbor_pair_index": idx,
                }
                gate_idx += 1
        ctx.add_layer(gates)
    ctx.preparation_diagnostics = {
        "context_source": "0522-generated",
        "piece_num": int(piece_num),
        "pattern_source": pattern_source,
        "pattern_seed": pattern_seed,
        "pattern_index": int(pattern_index),
        "pattern_file": pattern_meta.get("pattern_file"),
        "mapping_mode": mapping_mode,
        "mapping_steps": int(mapping_steps_used),
        "mapping_seed": mapping_seed,
        "mapping_cost": int(mapping_cost),
        "kraus_single_entries": len(kraus_single),
        "kraus_2q_shared": True,
    }
    meta = {
        "context_source": "0522-generated",
        "width": width,
        "length": length,
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "n_gates": gate_idx,
        "gate_table": gate_table,
        "phys_to_1d": phys_to_1d,
        "mapping_1d_to_phys": [int(x) for x in mapping],
        "mapping_mode": mapping_mode,
        "mapping_steps": int(mapping_steps_used),
        "mapping_cost": int(mapping_cost),
        "pattern_source": pattern_source,
        **pattern_meta,
        "all_neighbors": [list(map(int, p)) for p in all_neighbors],
        "pair_to_idx": {str(k): int(v) for k, v in pair_to_idx.items()},
        "ideal_circuit": "0522_unified generated qutrit-projected",
    }
    return ctx, meta
