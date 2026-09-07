"""Standalone data/figure pipeline for the new Fig. 2 validation panels.

This file is intentionally outside the manuscript source tree.  It does not
modify ``main.tex`` or any existing figure asset.

The comparison is:

* ``F_liouville``: global qutrit density-matrix evolution using the raw local
  qutrit superoperators before computational-subspace projection.
* ``F2_full_state``: the existing exact full-state branch backend applied to
  the projected qubit Kraus channels, accumulated through second order.
* ``F_liouville`` is then evaluated from the raw qutrit channels already held
  in the same ``ChannelSet``; it is not a second channel compilation.

The global qutrit superoperator is never materialized.  The state is stored as
one local-Liouville tensor with shape ``(9,)*N`` and each 1Q/2Q local map is
applied on the selected local Liouville axes.

Default figure statistics use ``T1=T2=1000`` and ``5000`` for the 20-layer
distributions, then ``5000`` again for the 40-layer distribution.  Panel a
uses one common circuit at ``1000``, ``3000``, and ``5000`` ns.  These are
command-line parameters and are not hard-coded into the manuscript.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch


from residual_tn.paths import CONFIGS, INPUTS, RESULTS

PROJECT_ROOT = RESULTS

# Keep the existing exact backend in its centered-C convention.
os.environ.setdefault("PEPS_CORR2BP_PAIR_FORMULA", "centered_C")
os.environ.setdefault("CENTERED_FIDELITY_EXACT_BRANCH_CHUNK_MODES", "128")

from residual_tn.backend.channel import superoperator_to_kraus  # noqa: E402
from residual_tn.backend.context import FidelityContext  # noqa: E402
from residual_tn.backend.residual_magnus import ResidualMagnusLibrary  # noqa: E402
from residual_tn.backend import residual_background_exact as residual_bg  # noqa: E402


DEFAULT_RESIDUAL_MANIFEST = CONFIGS / "residual_manifest.json"


@dataclass
class ChannelSet:
    ideal_single: torch.Tensor
    ideal_two: torch.Tensor
    single_superoperators: torch.Tensor  # (N, 4, 9, 9), raw qutrit channels
    two_superoperator: torch.Tensor  # (9, 9, 9, 9), local-Liouville axes
    single_kraus: dict[tuple[int, int], torch.Tensor]
    two_kraus: torch.Tensor


def _batch_kron(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    sa, sb = a.shape, b.shape
    n, m = sa[-2], sa[-1]
    p, q = sb[-2], sb[-1]
    return (a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)).reshape(
        *sa[:-2], n * p, m * q
    )


def _ladder(dim: int, device: torch.device, dtype: torch.dtype):
    a = torch.zeros((dim, dim), dtype=dtype, device=device)
    for i in range(1, dim):
        a[i - 1, i] = math.sqrt(i)
    adag = a.conj().T
    num = adag @ a
    eye = torch.eye(dim, dtype=dtype, device=device)
    return a, adag, num, eye


def ideal_single_gates(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
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


def ideal_two_gate(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
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
    ]
    reps = (n_qubits + len(vals) - 1) // len(vals)
    return torch.tensor((vals * reps)[:n_qubits], dtype=torch.float64, device=device)


def row_major_neighbors(width: int, length: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for r in range(length):
        for c in range(width):
            q = r * width + c
            if c + 1 < width:
                out.append((q, q + 1))
            if r + 1 < length:
                out.append((q, q + width))
    return out


def generate_circuit_patterns(
    batch_size: int,
    n_layers: int,
    n_qubits: int,
    neighbors: Sequence[tuple[int, int]],
    seed: int,
) -> list[list[Any]]:
    np_rng = np.random.RandomState(seed)
    py_rng = random.Random(seed)
    single_pool = np_rng.randint(0, 4, size=(batch_size, n_layers, n_qubits))
    all_patterns: list[list[Any]] = []
    for b in range(batch_size):
        layers: list[Any] = []
        for layer in range(n_layers):
            if layer % 2 == 0:
                layers.append(single_pool[b, layer, :].copy())
            else:
                edges = list(neighbors)
                py_rng.shuffle(edges)
                selected: list[list[int]] = []
                used: set[int] = set()
                for u, v in edges:
                    if u not in used and v not in used:
                        selected.append([u, v])
                        used.add(u)
                        used.add(v)
                layers.append(selected)
        all_patterns.append(layers)
    return all_patterns


def _single_superoperators(
    width: int,
    length: int,
    piece_num: int,
    t1: float,
    t2: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Parameterized copy of the existing 0522 qutrit 1Q channel compiler."""
    n_qubits = width * length
    dim, dim2 = 3, 9
    dtype_r = torch.float64
    a, adag, num, eye = _ladder(dim, device, dtype)
    t1s = t1 * torch.ones(n_qubits, dtype=dtype_r, device=device)
    t2s = t2 * torch.ones(n_qubits, dtype=dtype_r, device=device)
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
    n_pulses = 4
    steps = int(piece_num)
    eye_flat = torch.eye(dim2, dtype=dtype, device=device)
    d_ops = torch.zeros((n_qubits, 1, dim2, dim2), dtype=dtype, device=device)

    term1 = torch.kron(a.conj(), a)
    nadaga = adag @ a
    term2 = -0.5 * (torch.kron(eye, nadaga) + torch.kron(nadaga.T.contiguous(), eye))
    d_ops += (1.0 / t1s).reshape(n_qubits, 1, 1, 1) * (term1 + term2).reshape(
        1, 1, dim2, dim2
    )

    term1 = torch.kron(num.conj(), num)
    n2 = num @ num
    term2 = -0.5 * (torch.kron(eye, n2) + torch.kron(n2.T.contiguous(), eye))
    d_ops += (1.0 / t2s).reshape(n_qubits, 1, 1, 1) * (term1 + term2).reshape(
        1, 1, dim2, dim2
    )
    d_ops = d_ops.expand(n_qubits, n_pulses, dim2, dim2)

    if steps <= 1:
        raise ValueError("piece_num must be greater than 1")
    duration = float(drive_duration.detach().cpu().item())
    t_raw = torch.linspace(0, duration, steps, dtype=dtype_r, device=device)
    t_mid = (t_raw[:-1] + t_raw[1:]) / 2
    t_grid = torch.cat([t_mid, t_raw[-1].unsqueeze(0)], dim=0)
    dt = drive_duration / (steps - 1)
    t_rel = t_grid.reshape(steps, 1, 1)
    sigma = drive_duration / 4.0
    center = drive_duration / 2.0
    gaussian = torch.exp(-0.5 * ((t_rel - center) / sigma) ** 2)
    env_deriv = -(t_rel - center) / (sigma**2) * gaussian
    env = gaussian - math.exp(-2.0)
    beta = drive_omegas.reshape(1, n_qubits, 1) * env + 1j * (
        drags.reshape(1, n_qubits, 1) * env_deriv / anh.reshape(1, n_qubits, 1)
    )
    beta = beta.expand(steps, n_qubits, n_pulses).clone()
    phis = phase_map.reshape(1, 1, -1).to(device)
    total_phase = (q_freqs - drive_freqs).reshape(1, n_qubits, 1) * t_rel + phis
    beta[:, :, : n_pulses - 1] *= torch.exp(1j * total_phase)
    beta[:, :, -1] = 0

    nn_minus_n = num @ (num - eye)
    h_static = (
        0.5 * anh.reshape(1, n_qubits, 1, 1, 1) * nn_minus_n.reshape(1, 1, 1, dim, dim)
    )
    h_static = h_static.expand(steps, n_qubits, n_pulses, dim, dim)
    h_all = h_static + beta.reshape(steps, n_qubits, n_pulses, 1, 1) * adag.reshape(
        1, 1, 1, dim, dim
    )
    h_all = h_all + beta.conj().reshape(steps, n_qubits, n_pulses, 1, 1) * a.reshape(
        1, 1, 1, dim, dim
    )

    h_kron_i = (
        h_all.unsqueeze(-1).unsqueeze(-3) * eye.reshape(1, 1, 1, 1, 3, 1, 3)
    ).reshape(steps, n_qubits, n_pulses, dim2, dim2)
    i_kron_ht = (
        eye.reshape(1, 1, 1, 3, 1, 3, 1)
        * h_all.transpose(-1, -2).unsqueeze(-2).unsqueeze(-4)
    ).reshape(steps, n_qubits, n_pulses, dim2, dim2)
    l_total = -1j * (h_kron_i - i_kron_ht) + d_ops.unsqueeze(0)
    eye_super = eye_flat.reshape(1, 1, 1, dim2, dim2).expand_as(l_total)
    a_mat = (eye_super - 0.5 * dt * l_total).reshape(
        steps * n_qubits * n_pulses, dim2, dim2
    )
    b_mat = (eye_super + 0.5 * dt * l_total).reshape(
        steps * n_qubits * n_pulses, dim2, dim2
    )
    s_slices = torch.linalg.solve(a_mat, b_mat).view(
        steps, n_qubits, n_pulses, dim2, dim2
    )

    s_cum = s_slices[0]
    for step in range(1, steps):
        s_cum = s_slices[step] @ s_cum
    return s_cum


def _two_superoperator(
    piece_num: int,
    t1: float,
    t2: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Parameterized copy of the existing 0522 qutrit 2Q compiler."""
    dim, dim2, dim4 = 3, 9, 81
    dtype_r = torch.float64
    op_a, op_adag, _, eye = _ladder(dim, device, dtype)
    op_adag = op_adag.contiguous()
    a_l = torch.kron(op_a, eye).reshape(1, 1, dim2, dim2)
    adag_l = torch.kron(op_adag, eye).reshape(1, 1, dim2, dim2)
    a_r = torch.kron(eye, op_a).reshape(1, 1, dim2, dim2)
    adag_r = torch.kron(eye, op_adag).reshape(1, 1, dim2, dim2)
    n_l = adag_l @ a_l
    n_r = adag_r @ a_r
    eye_m = torch.eye(dim2, dtype=dtype, device=device).view(1, 1, dim2, dim2)
    eye_super = torch.eye(dim4, dtype=dtype, device=device).view(1, 1, dim4, dim4)

    def dissipator(
        op: torch.Tensor, op_dag: torch.Tensor, gamma: torch.Tensor
    ) -> torch.Tensor:
        jump = op_dag @ op
        return gamma * (
            _batch_kron(op, op.conj())
            - 0.5
            * (_batch_kron(jump, eye_m) + _batch_kron(eye_m, jump.transpose(-1, -2)))
        )

    gamma1 = torch.tensor(1.0 / t1, dtype=dtype_r, device=device).view(1, 1, 1, 1)
    gamma2 = torch.tensor(1.0 / t2, dtype=dtype_r, device=device).view(1, 1, 1, 1)
    d_ops = (
        dissipator(a_l, adag_l, gamma1)
        + dissipator(a_r, adag_r, gamma1)
        + dissipator(n_l, n_l, gamma2)
        + dissipator(n_r, n_r, gamma2)
    )

    anh = torch.tensor(-2 * math.pi * 242 * 1e-3, dtype=dtype_r, device=device)
    drive_duration = 8 * math.pi / torch.abs(anh)
    steps = int(piece_num)
    dt = drive_duration / (steps - 1)
    h_static = 0.5 * anh * (n_l @ (n_l - eye_m)) + 0.5 * anh * (n_r @ (n_r - eye_m))
    h_dynamic = torch.tensor(-0.0422, dtype=dtype_r, device=device).view(1, 1, 1, 1) * (
        (adag_l @ a_r) + (a_l @ adag_r)
    )
    h_total = h_static + h_dynamic
    l_total = (
        -1j
        * (_batch_kron(h_total, eye_m) - _batch_kron(eye_m, h_total.transpose(-1, -2)))
        + d_ops
    )
    s_step = torch.linalg.solve(
        (eye_super - 0.5 * dt * l_total).reshape(dim4, dim4),
        (eye_super + 0.5 * dt * l_total).reshape(dim4, dim4),
    )
    s_cum = torch.eye(dim4, dtype=dtype, device=device)
    for _ in range(steps):
        s_cum = s_step @ s_cum
    return s_cum


def fix_superoperator_indexing(s: torch.Tensor) -> torch.Tensor:
    return (
        s.view(3, 3, 3, 3, 3, 3, 3, 3)
        .permute(0, 2, 1, 3, 4, 6, 5, 7)
        .reshape(9, 9, 9, 9)
    )


def qubit_projection(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    p = torch.zeros((9, 2, 2), dtype=dtype, device=device)
    p[0, 0, 0] = 1.0
    p[1, 0, 1] = 1.0
    p[3, 1, 0] = 1.0
    p[4, 1, 1] = 1.0
    return p


def build_channels(
    width: int,
    length: int,
    piece_num: int,
    noise_time: float,
    device: torch.device,
    dtype: torch.dtype,
) -> ChannelSet:
    n_qubits = width * length
    ideal_1 = ideal_single_gates(dtype, device)
    ideal_2 = ideal_two_gate(dtype, device)
    raw_1 = _single_superoperators(
        width, length, piece_num, noise_time, noise_time, device, dtype
    )
    raw_2 = fix_superoperator_indexing(
        _two_superoperator(piece_num, noise_time, noise_time, device, dtype)
    )
    p = qubit_projection(device, dtype)

    single_kraus: dict[tuple[int, int], torch.Tensor] = {}
    for phys in range(n_qubits):
        for mode in range(4):
            s_proj = torch.einsum("xab,xy,ycd->abcd", p, raw_1[phys, mode], p).reshape(
                4, 4
            )
            single_kraus[(phys, mode)] = superoperator_to_kraus(s_proj, d=2, tol=1e-10)

    s2_proj = torch.einsum("uab,vcd,uvij,ief,jgh->abcdefgh", p, p, raw_2, p, p)
    s2_qubit = s2_proj.permute(0, 2, 1, 3, 4, 6, 5, 7).contiguous().reshape(16, 16)
    two_kraus = superoperator_to_kraus(s2_qubit, d=4, tol=1e-10)
    return ChannelSet(ideal_1, ideal_2, raw_1, raw_2, single_kraus, two_kraus)


def build_projected_context(
    pattern: Sequence[Any],
    width: int,
    length: int,
    channels: ChannelSet,
    device: torch.device,
    dtype: torch.dtype,
) -> FidelityContext:
    n_qubits = width * length
    ctx = FidelityContext(n_qubits, width, length, device, dtype)
    gate_idx = 0
    for layer_idx, layer in enumerate(pattern):
        gates: list[dict[str, Any]] = []
        if layer_idx % 2 == 0:
            for phys in range(n_qubits):
                mode = int(layer[phys])
                gates.append(
                    {
                        "gate_idx": gate_idx,
                        "qubits": (phys,),
                        "mode": mode,
                        "ideal_unitary": channels.ideal_single[mode],
                        "kraus_ops": channels.single_kraus[(phys, mode)],
                    }
                )
                gate_idx += 1
        else:
            for q1, q2 in layer:
                gates.append(
                    {
                        "gate_idx": gate_idx,
                        "qubits": (int(q1), int(q2)),
                        "ideal_unitary": channels.ideal_two,
                        "kraus_ops": channels.two_kraus,
                    }
                )
                gate_idx += 1
        ctx.add_layer(gates)
    return ctx


def embed_1q_qutrit(u2: torch.Tensor) -> torch.Tensor:
    out = torch.eye(3, dtype=u2.dtype, device=u2.device)
    out[:2, :2] = u2
    return out


def embed_2q_qutrit(u4: torch.Tensor) -> torch.Tensor:
    out = torch.eye(9, dtype=u4.dtype, device=u4.device)
    comp = torch.tensor([0, 1, 3, 4], dtype=torch.long, device=u4.device)
    out[comp[:, None], comp[None, :]] = u4
    return out


def apply_1q_state(
    psi: torch.Tensor, u: torch.Tensor, q: int, n_qubits: int, d: int = 3
) -> torch.Tensor:
    tensor = psi.reshape([d] * n_qubits)
    out = torch.tensordot(u, tensor, dims=([1], [q]))
    perm = list(range(1, n_qubits))
    perm.insert(q, 0)
    return out.permute(*perm).reshape(-1)


def apply_2q_state(
    psi: torch.Tensor,
    u: torch.Tensor,
    q1: int,
    q2: int,
    n_qubits: int,
    d: int = 3,
) -> torch.Tensor:
    tensor = psi.reshape([d] * n_qubits)
    others = [i for i in range(n_qubits) if i not in (q1, q2)]
    perm = others + [q1, q2]
    x = tensor.permute(*perm).reshape(-1, d * d)
    y = x @ u.reshape(d * d, d * d).T
    y = y.reshape([d] * (n_qubits - 2) + [d, d])
    inverse = [0] * n_qubits
    for i, p in enumerate(perm):
        inverse[p] = i
    return y.permute(*inverse).reshape(-1)


def apply_1q_liouville(
    rho_local: torch.Tensor,
    s1: torch.Tensor,
    q: int,
) -> torch.Tensor:
    x = rho_local.movedim(q, 0)
    shape = x.shape
    y = s1 @ x.reshape(9, -1)
    return y.reshape(shape).movedim(0, q)


def apply_2q_liouville(
    rho_local: torch.Tensor,
    s2: torch.Tensor,
    q1: int,
    q2: int,
) -> torch.Tensor:
    x = rho_local.movedim((q1, q2), (0, 1))
    shape = x.shape
    x = x.reshape(9, 9, -1)
    y = torch.einsum("abij,ijr->abr", s2, x).reshape(shape)
    return y.movedim((0, 1), (q1, q2))


def local_liouville_to_matrix(
    rho_local: torch.Tensor, n_qubits: int, d: int = 3
) -> torch.Tensor:
    x = rho_local.reshape([d, d] * n_qubits)
    ket_axes = list(range(0, 2 * n_qubits, 2))
    bra_axes = list(range(1, 2 * n_qubits, 2))
    return x.permute(*(ket_axes + bra_axes)).reshape(d**n_qubits, d**n_qubits)


def liouville_fidelity(
    rho_local: torch.Tensor,
    psi_ideal: torch.Tensor,
    n_qubits: int,
) -> tuple[float, float]:
    rho_matrix = local_liouville_to_matrix(rho_local, n_qubits)
    fidelity = torch.einsum("i,ij,j->", psi_ideal.conj(), rho_matrix, psi_ideal).real
    trace = torch.trace(rho_matrix).real
    return float(fidelity.detach().cpu()), float(trace.detach().cpu())


def run_qutrit_liouville(
    pattern: Sequence[Any],
    channels: ChannelSet,
    width: int,
    length: int,
    compiled_residual: Sequence[Sequence[Any]],
) -> dict[str, list[float]]:
    n_qubits = width * length
    d2 = 9
    rho = torch.zeros(
        (d2,) * n_qubits,
        dtype=channels.ideal_single.dtype,
        device=channels.ideal_single.device,
    )
    rho[(0,) * n_qubits] = 1.0
    psi = torch.zeros(
        3**n_qubits,
        dtype=channels.ideal_single.dtype,
        device=channels.ideal_single.device,
    )
    psi[0] = 1.0
    fidelities = [1.0]
    traces = [1.0]

    if len(compiled_residual) < len(pattern):
        raise ValueError("residual library is shorter than the circuit")
    for layer_idx, layer in enumerate(pattern):
        rho = residual_bg.apply_residual_layer_to_liouville(
            rho, compiled_residual[layer_idx], local_dim=3
        )
        if layer_idx % 2 == 0:
            for phys in range(n_qubits):
                mode = int(layer[phys])
                rho = apply_1q_liouville(
                    rho, channels.single_superoperators[phys, mode], phys
                )
                psi = apply_1q_state(
                    psi, embed_1q_qutrit(channels.ideal_single[mode]), phys, n_qubits
                )
        else:
            for q1, q2 in layer:
                q1, q2 = int(q1), int(q2)
                rho = apply_2q_liouville(rho, channels.two_superoperator, q1, q2)
                psi = apply_2q_state(
                    psi, embed_2q_qutrit(channels.ideal_two), q1, q2, n_qubits
                )
        f, tr = liouville_fidelity(rho, psi, n_qubits)
        fidelities.append(f)
        traces.append(tr)
    return {"fidelity": fidelities, "trace": traces}


def full_state_f2_trajectory(
    ctx: FidelityContext,
    compiled_residual: Sequence[Sequence[Any]],
    *,
    all_prefixes: bool,
) -> dict[str, Any]:
    """Evaluate the F0-normalized response on the shared residual background."""
    if all_prefixes:
        trajectory = residual_bg.connected_fidelity_trajectory(ctx, compiled_residual)
        return {
            **trajectory,
            "final_result_f2": float(trajectory["fidelity_second_order"][-1]),
            "pair_count": int(trajectory["final_pair_count"]),
        }
    final = residual_bg.connected_fidelity_at_depth(ctx, compiled_residual)
    return {
        "zero_insertion_fidelity": [1.0, float(final["zero_insertion_fidelity"])],
        "fidelity_first_order": [1.0, float(final["fidelity_first_order"])],
        "fidelity_second_order": [1.0, float(final["fidelity_second_order"])],
        "final_result_f2": float(final["fidelity_second_order"]),
        "pair_count": int(final["pair_count"]),
    }


def run_one_case(
    pattern: Sequence[Any],
    channels: ChannelSet,
    width: int,
    length: int,
    n_layers: int,
    projected_residual: Sequence[Sequence[Any]],
    qutrit_residual: Sequence[Sequence[Any]],
    *,
    full_trajectory: bool,
) -> dict[str, Any]:
    pattern = list(pattern[:n_layers])
    ctx = build_projected_context(
        pattern,
        width,
        length,
        channels,
        channels.ideal_single.device,
        channels.ideal_single.dtype,
    )
    # Use the same 0522-derived ChannelSet for both quantities.  The requested
    # order is the full-state F2 calculation followed by the exact Liouville
    # readout from the raw qutrit superoperators already stored in `channels`.
    full_state = full_state_f2_trajectory(
        ctx,
        projected_residual[:n_layers],
        all_prefixes=full_trajectory,
    )
    liouville = run_qutrit_liouville(
        pattern,
        channels,
        width,
        length,
        qutrit_residual[:n_layers],
    )
    return {
        "F_liouville": liouville["fidelity"],
        "trace_liouville": liouville["trace"],
        "F1_full_state": full_state["fidelity_first_order"],
        "F2_full_state": full_state["fidelity_second_order"],
        "F0_full_state": full_state["zero_insertion_fidelity"],
        "final_F_liouville": liouville["fidelity"][-1],
        "final_F2_full_state": full_state["fidelity_second_order"][-1],
        "final_error": full_state["fidelity_second_order"][-1]
        - liouville["fidelity"][-1],
        "pair_count": full_state["pair_count"],
    }


def _singleton_record(
    noise_time: float,
    n_layers: int,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Serialize one complete trajectory used by the original-style panel a."""
    liouville = [float(x) for x in result["F_liouville"]]
    f2 = [float(x) for x in result["F2_full_state"]]
    f0 = [float(x) for x in result["F0_full_state"]]
    return {
        "circuit": 0,
        "layers": int(n_layers),
        "noise_time": float(noise_time),
        "F_liouville_trajectory": liouville,
        "F2_full_state_trajectory": f2,
        "F0_full_state_trajectory": f0,
        "abs_error_trajectory": [abs(a - b) for a, b in zip(f2, liouville)],
        "trace_liouville_trajectory": [float(x) for x in result["trace_liouville"]],
        "pair_count": int(result["pair_count"]),
    }


def save_records(
    output_dir: Path,
    metadata: dict[str, Any],
    runs: list[dict[str, Any]],
    singleton_trajectories: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "runs": runs,
        "singleton_trajectories": singleton_trajectories,
    }
    (output_dir / "fig2_error_statistics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "fig2_error_statistics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        fields = [
            "circuit",
            "layers",
            "noise_time",
            "F_liouville",
            "F2_full_state",
            "error_F2_minus_Liouville",
            "abs_error",
            "trace_liouville",
            "pair_count",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            writer.writerow({k: run.get(k) for k in fields})


def run_ensemble(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dtype = torch.complex128
    width, length = 4, 2
    n_qubits = width * length
    singleton_layers = int(args.singleton_layers)
    if singleton_layers <= 0:
        raise ValueError("--singleton-layers must be positive")
    singleton_noise = [float(x) for x in args.singleton_noise]
    n_total_layers = max(40, singleton_layers)
    patterns = generate_circuit_patterns(
        args.n_circuits,
        n_total_layers,
        n_qubits,
        row_major_neighbors(width, length),
        args.seed,
    )
    manifest, geometry = residual_bg.load_residual_geometry(
        args.residual_manifest, width, length
    )
    residual_steps = (
        args.piece_num
        if int(args.residual_integration_steps) == 0
        else int(args.residual_integration_steps)
    )
    print(
        f"[residual] compiling {args.kernel_scheme} common backgrounds once: "
        f"circuits={len(patterns)} layers={n_total_layers} "
        f"edges/layer={len(geometry['edges'])} steps={residual_steps}",
        flush=True,
    )
    projected_library = ResidualMagnusLibrary(
        n_qubits,
        integration_steps=residual_steps,
        dtype=dtype,
        device=device,
        projected_qubits=True,
    )
    qutrit_library = ResidualMagnusLibrary(
        n_qubits,
        integration_steps=residual_steps,
        dtype=dtype,
        device=device,
        projected_qubits=False,
    )
    projected_residual = []
    qutrit_residual = []
    residual_started = time.time()
    for circuit, pattern in enumerate(patterns):
        projected_residual.append(
            residual_bg.compile_pattern_residual(
                projected_library,
                pattern,
                geometry["edges"],
                scheme=args.kernel_scheme,
            )
        )
        qutrit_residual.append(
            residual_bg.compile_pattern_residual(
                qutrit_library,
                pattern,
                geometry["edges"],
                scheme=args.kernel_scheme,
            )
        )
        print(
            f"[residual] circuit={circuit + 1}/{len(patterns)} compiled",
            flush=True,
        )
    residual_compile_seconds = time.time() - residual_started
    required: dict[float, set[int]] = {}
    for t in args.noise20:
        required.setdefault(float(t), set()).add(20)
    for t in args.noise40:
        required.setdefault(float(t), set()).add(40)
    # Panel a uses one common circuit at 1, 3, and 5 us.  Reuse an ensemble
    # trajectory whenever it is already available; otherwise run only that
    # one circuit through the same F2-then-Liouville calculation.
    for t in singleton_noise:
        required.setdefault(t, set())

    runs: list[dict[str, Any]] = []
    singleton_trajectories: list[dict[str, Any]] = []
    for noise_time, lengths in sorted(required.items()):
        print(
            f"[channels] T1=T2={noise_time:g}; piece_num={args.piece_num}", flush=True
        )
        channels = build_channels(
            width, length, args.piece_num, noise_time, device, dtype
        )
        is_singleton_noise = any(
            math.isclose(noise_time, target) for target in singleton_noise
        )
        singleton_result: dict[str, Any] | None = None
        for circuit, pattern in enumerate(patterns):
            for n_layers in sorted(lengths):
                start = time.time()
                need_trajectory = (
                    is_singleton_noise and circuit == 0 and n_layers == singleton_layers
                )
                result = run_one_case(
                    pattern,
                    channels,
                    width,
                    length,
                    n_layers,
                    projected_residual[circuit],
                    qutrit_residual[circuit],
                    full_trajectory=need_trajectory,
                )
                row = {
                    "circuit": circuit,
                    "layers": n_layers,
                    "noise_time": noise_time,
                    "F_liouville": result["final_F_liouville"],
                    "F2_full_state": result["final_F2_full_state"],
                    "error_F2_minus_Liouville": result["final_error"],
                    "abs_error": abs(result["final_error"]),
                    "trace_liouville": result["trace_liouville"][-1],
                    "pair_count": result["pair_count"],
                    "F_liouville_trajectory": result["F_liouville"],
                    "F2_full_state_trajectory": result["F2_full_state"],
                    "F1_full_state_trajectory": result["F1_full_state"],
                    "F0_full_state_trajectory": result["F0_full_state"],
                }
                runs.append(row)
                if is_singleton_noise and circuit == 0 and n_layers == singleton_layers:
                    singleton_result = result
                print(
                    f"[run] circuit={circuit:02d} layers={n_layers:02d} "
                    f"T={noise_time:g} error={row['error_F2_minus_Liouville']:+.6e} "
                    f"trace={row['trace_liouville']:.9f} elapsed={time.time() - start:.1f}s",
                    flush=True,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        if is_singleton_noise:
            if singleton_result is None:
                start = time.time()
                singleton_result = run_one_case(
                    patterns[0],
                    channels,
                    width,
                    length,
                    singleton_layers,
                    projected_residual[0],
                    qutrit_residual[0],
                    full_trajectory=True,
                )
                print(
                    f"[singleton] layers={singleton_layers:02d} T={noise_time:g} "
                    f"error={singleton_result['final_error']:+.6e} "
                    f"trace={singleton_result['trace_liouville'][-1]:.9f} "
                    f"elapsed={time.time() - start:.1f}s",
                    flush=True,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            singleton_trajectories.append(
                _singleton_record(noise_time, singleton_layers, singleton_result)
            )

    metadata = {
        "width": width,
        "length": length,
        "n_qubits": n_qubits,
        "n_circuits": args.n_circuits,
        "seed": args.seed,
        "piece_num": args.piece_num,
        "noise20": [float(x) for x in args.noise20],
        "noise40": [float(x) for x in args.noise40],
        "singleton_layers": singleton_layers,
        "singleton_noise": singleton_noise,
        "device": str(device),
        "dtype": str(dtype),
        "liouville": "raw qutrit local superoperators plus the common qutrit residual background",
        "full_state": "projected-qubit exact F0-normalized one-/two-location response on the common residual background",
        "residual_manifest": str(Path(args.residual_manifest).expanduser().resolve()),
        "residual_manifest_base_seed": int(manifest["base_seed"]),
        "residual_kernel_scheme": args.kernel_scheme,
        "residual_integration_steps": residual_steps,
        "residual_edges_per_layer": len(geometry["edges"]),
        "residual_compile_seconds": residual_compile_seconds,
        "background_layer_order": [
            "interaction_picture_residual",
            "intended_or_noisy_local_maps",
        ],
    }
    save_records(Path(args.output_dir), metadata, runs, singleton_trajectories)


def load_runs(
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (
        payload["metadata"],
        payload["runs"],
        payload.get("singleton_trajectories", []),
    )


def _absolute_errors(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    """Return the final absolute F2-vs-Liouville errors for one ensemble."""
    return np.asarray(
        [
            float(row.get("abs_error", abs(float(row["error_F2_minus_Liouville"]))))
            for row in rows
        ],
        dtype=float,
    )


def _relative_errors(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    """Return ``|F2 - F_exact| / F_exact`` for one ensemble."""
    values: list[float] = []
    for row in rows:
        exact = float(row["F_liouville"])
        if exact <= 0.0:
            raise ValueError(
                f"relative error requires positive exact fidelity, got {exact}"
            )
        absolute = float(
            row.get("abs_error", abs(float(row["error_F2_minus_Liouville"])))
        )
        values.append(absolute / exact)
    return np.asarray(values, dtype=float)


def _centered_descending_positions(count: int) -> np.ndarray:
    """Put the largest of ``count`` values at the centre, then alternate sides."""
    if count <= 0:
        return np.empty(0, dtype=float)
    positions = [0]
    radius = 1
    while len(positions) < count:
        positions.append(-radius)
        if len(positions) < count:
            positions.append(radius)
        radius += 1
    return np.asarray(positions, dtype=float)


def _plot_ranked_error_panel(
    ax,
    rows: list[dict[str, Any]],
    title: str,
    *,
    y_limit_percent: float,
) -> None:
    """Plot all 20 final relative errors as a centre-out ranked bar profile."""
    errors = _relative_errors(rows)
    if errors.size == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    # The bars are individual circuits, not histogram bins.  Sorting their
    # heights and placing the largest bar in the middle makes the spread clear
    # without treating circuit index as a physical horizontal variable.
    heights = np.sort(errors)[::-1] * 100.0
    positions = _centered_descending_positions(len(heights))
    ax.bar(positions, heights, width=0.78, color="#3f78a8", linewidth=0.0)
    ax.set_xlim(-len(heights) / 2 - 0.9, len(heights) / 2 + 0.9)
    ax.set_ylim(0.0, y_limit_percent)
    ax.set_xticks([])
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.set_ylabel(r"relative error $|F_2-F_{\mathrm{exact}}|/F_{\mathrm{exact}}$ (%)")
    ax.set_title(title, fontsize=10, pad=13)
    ax.text(
        0.0,
        heights[0] + 0.035 * y_limit_percent,
        f"max = {heights[0]:.2f}%",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#244f70",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)


def _noise_us(noise_time: float) -> str:
    """Compact display label for a noise time stored internally in ns."""
    return f"{noise_time / 1000.0:g} μs"


def _noise_math_label(noise_time: float) -> str:
    return rf"$T_1=T_2={noise_time / 1000.0:g}\,\mu\mathrm{{s}}$"


def _plot_original_style_singleton(
    ax_fidelity,
    ax_error,
    trajectories: Sequence[dict[str, Any]],
) -> None:
    """Recreate panel a as fidelity and layer-resolved error for three noises."""
    from matplotlib.lines import Line2D

    if len(trajectories) != 3:
        raise ValueError("Panel a requires three singleton trajectories")

    colors = ["#2868a5", "#2f8c5b", "#7b5bb6"]
    all_fidelities: list[np.ndarray] = []
    all_errors: list[np.ndarray] = []
    n_layers: int | None = None

    for color, row in zip(colors, trajectories):
        liouville = np.asarray(row["F_liouville_trajectory"], dtype=float)
        f2 = np.asarray(row["F2_full_state_trajectory"], dtype=float)
        if len(liouville) != len(f2):
            raise ValueError(
                "F2 and Liouville singleton trajectories have different lengths"
            )
        current_layers = len(liouville) - 1
        if n_layers is None:
            n_layers = current_layers
        elif n_layers != current_layers:
            raise ValueError(
                "The three singleton trajectories must have the same layer count"
            )

        layers = np.arange(len(liouville))
        # Match the original figure: the estimator is the coloured solid curve,
        # while the exact reference is a red dashed overlay.  Drawing the exact
        # curve last keeps it visible even when the two values nearly coincide.
        ax_fidelity.plot(layers, f2, color=color, lw=1.85, zorder=2)
        ax_fidelity.plot(
            layers,
            liouville,
            color="#e31a1c",
            lw=1.55,
            ls=(0, (5.0, 3.0)),
            zorder=4,
        )
        errors = np.abs(f2 - liouville) * 1e3
        ax_error.fill_between(
            layers,
            0.0,
            errors,
            color=color,
            alpha=0.13,
            linewidth=0.0,
            zorder=1,
        )
        ax_error.plot(
            layers,
            errors,
            color=color,
            lw=1.35,
            zorder=2,
        )
        all_fidelities.extend([liouville, f2])
        all_errors.append(errors)

    assert n_layers is not None
    low = max(0.0, min(float(values.min()) for values in all_fidelities) - 0.05)
    max_error = max(float(values.max()) for values in all_errors)
    tick_step = 5 if n_layers <= 25 else 10

    ax_fidelity.set_xlim(0, n_layers)
    ax_fidelity.set_ylim(low, 1.02)
    ax_fidelity.set_ylabel("fidelity", fontsize=8)
    ax_fidelity.set_title("One circuit: fidelity evolution", fontsize=9.5, pad=9)
    ax_fidelity.tick_params(axis="x", labelbottom=False)
    ax_fidelity.tick_params(axis="y", labelsize=7)
    ax_fidelity.spines["top"].set_visible(False)
    ax_fidelity.spines["right"].set_visible(False)

    ax_fidelity.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="#e31a1c",
                lw=1.55,
                ls=(0, (5.0, 3.0)),
                label="Exact Liouville",
            ),
            *[
                Line2D(
                    [0],
                    [0],
                    color=color,
                    lw=1.85,
                    label=rf"Full-state $F_2$ ({_noise_us(float(row['noise_time']))})",
                )
                for color, row in zip(colors, trajectories)
            ],
        ],
        loc="lower left",
        frameon=False,
        fontsize=5.8,
        handlelength=2.0,
        borderpad=0.1,
        labelspacing=0.28,
    )

    ax_error.set_xlim(0, n_layers)
    ax_error.set_ylim(0.0, max(1e-6, 1.18 * max_error))
    ax_error.set_xticks(np.arange(0, n_layers + 1, tick_step))
    ax_error.set_xlabel("layer", fontsize=8)
    ax_error.set_ylabel(r"$|F_2-F_{\mathrm{exact}}|$ ($\times10^{-3}$)", fontsize=7)
    ax_error.tick_params(labelsize=7)
    ax_error.spines["top"].set_visible(False)
    ax_error.spines["right"].set_visible(False)


def plot_figure(data_path: Path, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metadata, runs, singleton_trajectories = load_runs(data_path)
    noise20 = [float(x) for x in metadata.get("noise20", [])]
    noise40 = [float(x) for x in metadata.get("noise40", [])]
    if len(noise20) < 2 or not noise40:
        raise ValueError(
            "The figure requires two 20-layer noise values and one 40-layer noise value"
        )
    if not math.isclose(noise20[1], noise40[0]):
        raise ValueError(
            "Panels c and d must use the same noise time: set noise40 equal to the second noise20 value"
        )

    rows20_a = [
        row
        for row in runs
        if int(row["layers"]) == 20
        and math.isclose(float(row["noise_time"]), noise20[0])
    ]
    rows20_b = [
        row
        for row in runs
        if int(row["layers"]) == 20
        and math.isclose(float(row["noise_time"]), noise20[1])
    ]
    rows40 = [
        row
        for row in runs
        if int(row["layers"]) == 40
        and math.isclose(float(row["noise_time"]), noise40[0])
    ]
    ensembles = [rows20_a, rows20_b, rows40]
    if any(not rows for rows in ensembles):
        raise ValueError("Missing one of the requested ensemble data sets")

    singleton_noise = [float(x) for x in metadata.get("singleton_noise", [])]
    if len(singleton_noise) != 3:
        raise ValueError("The figure requires three singleton noise values for panel a")
    singleton_by_noise = {
        float(row["noise_time"]): row for row in singleton_trajectories
    }
    try:
        panel_a_trajectories = [singleton_by_noise[noise] for noise in singleton_noise]
    except KeyError as exc:
        raise ValueError(
            f"Missing singleton trajectory for T1=T2={exc.args[0]:g}"
        ) from exc

    # All statistical panels share a percentage scale so both the numerator
    # discrepancy and the fidelity level of each noise/depth setting are kept.
    max_error_percent = (
        max(float(_relative_errors(rows).max()) for rows in ensembles) * 100.0
    )
    y_limit_percent = max(1e-6, 1.18 * max_error_percent)

    # The outer four panels have equal footprints.  Panel a follows the
    # original two-row design: fidelity above, layer-resolved error below.
    fig = plt.figure(figsize=(10.6, 7.55), facecolor="white")
    outer = fig.add_gridspec(
        2,
        2,
        left=0.10,
        right=0.985,
        bottom=0.085,
        top=0.93,
        wspace=0.34,
        hspace=0.46,
    )
    panel_a = outer[0, 0].subgridspec(2, 1, height_ratios=[2.5, 1.0], hspace=0.12)
    ax_a_fidelity = fig.add_subplot(panel_a[0, 0])
    ax_a_error = fig.add_subplot(panel_a[1, 0], sharex=ax_a_fidelity)
    ax_b = fig.add_subplot(outer[0, 1])
    ax_c = fig.add_subplot(outer[1, 0])
    ax_d = fig.add_subplot(outer[1, 1])

    _plot_original_style_singleton(ax_a_fidelity, ax_a_error, panel_a_trajectories)
    _plot_ranked_error_panel(
        ax_b,
        rows20_a,
        f"20 circuits, 20 layers\n{_noise_math_label(noise20[0])}",
        y_limit_percent=y_limit_percent,
    )
    _plot_ranked_error_panel(
        ax_c,
        rows20_b,
        f"20 circuits, 20 layers\n{_noise_math_label(noise20[1])}",
        y_limit_percent=y_limit_percent,
    )
    _plot_ranked_error_panel(
        ax_d,
        rows40,
        f"20 circuits, 40 layers\n{_noise_math_label(noise40[0])}",
        y_limit_percent=y_limit_percent,
    )
    for label, ax, y in (
        ("a", ax_a_fidelity, 1.18),
        ("b", ax_b, 1.19),
        ("c", ax_c, 1.19),
        ("d", ax_d, 1.19),
    ):
        ax.text(
            -0.02,
            y,
            label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom",
            clip_on=False,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / "Fig2_exact_liouville_vs_fullstate_v5.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "Fig2_exact_liouville_vs_fullstate_v5.pdf", bbox_inches="tight"
    )
    plt.close(fig)


def smoke_test() -> None:
    """Small CPU-only test for channel projection, local Liouville action and F2 wiring."""
    device = torch.device("cpu")
    dtype = torch.complex128
    width, length, n_layers = 3, 1, 1
    patterns = generate_circuit_patterns(
        1, n_layers, width * length, row_major_neighbors(width, length), 0
    )
    channels = build_channels(width, length, 8, 5000.0, device, dtype)
    edge = [{"q1": 0, "q2": 2, "epsilon_rad_ns": 0.0015}]
    projected_library = ResidualMagnusLibrary(
        width * length,
        integration_steps=8,
        dtype=dtype,
        device=device,
        projected_qubits=True,
    )
    qutrit_library = ResidualMagnusLibrary(
        width * length,
        integration_steps=8,
        dtype=dtype,
        device=device,
        projected_qubits=False,
    )
    projected = residual_bg.compile_pattern_residual(
        projected_library, patterns[0], edge, scheme="dyson-1"
    )
    qutrit = residual_bg.compile_pattern_residual(
        qutrit_library, patterns[0], edge, scheme="dyson-1"
    )
    result = run_one_case(
        patterns[0],
        channels,
        width,
        length,
        n_layers,
        projected,
        qutrit,
        full_trajectory=True,
    )
    trace = result["trace_liouville"][-1]
    if not math.isfinite(trace) or trace <= 0.0:
        raise AssertionError(f"invalid Liouville trace in smoke test: {trace}")
    if not math.isfinite(result["final_F_liouville"]):
        raise AssertionError(f"invalid smoke fidelity: {result['final_F_liouville']}")
    print(
        json.dumps(
            {
                "smoke": "PASS",
                "F_liouville": result["final_F_liouville"],
                "F2_full_state": result["final_F2_full_state"],
                "trace": trace,
                "pair_count": result["pair_count"],
            },
            indent=2,
        )
    )


def device_info() -> None:
    """Print the selected Torch device facts without running a simulation."""
    available = torch.cuda.is_available()
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "cuda_available": available,
                "cuda_count": torch.cuda.device_count() if available else 0,
                "cuda_device": torch.cuda.get_device_name(0) if available else None,
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true", help="run the 20-circuit ensemble")
    p.add_argument(
        "--plot", action="store_true", help="plot from an existing JSON data file"
    )
    p.add_argument(
        "--smoke", action="store_true", help="run a small CPU-only smoke test"
    )
    p.add_argument(
        "--device-info", action="store_true", help="print Torch/CUDA device information"
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-circuits", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260711)
    p.add_argument("--piece-num", type=int, default=1000)
    p.add_argument("--residual-manifest", default=str(DEFAULT_RESIDUAL_MANIFEST))
    p.add_argument(
        "--kernel-scheme",
        choices=("dyson-1", "magnus-1"),
        default="magnus-1",
    )
    p.add_argument(
        "--residual-integration-steps",
        type=int,
        default=0,
        help="0 reuses --piece-num for the offline residual integration",
    )
    p.add_argument(
        "--noise20",
        nargs="+",
        type=float,
        default=[1000.0, 5000.0],
        metavar="T20",
        help="one or more noise times for 20-layer ensembles",
    )
    p.add_argument(
        "--noise40",
        nargs="*",
        type=float,
        default=[5000.0],
        metavar="T40",
        help="zero or more noise times for 40-layer ensembles",
    )
    p.add_argument(
        "--singleton-noise",
        nargs=3,
        type=float,
        default=[1000.0, 3000.0, 5000.0],
        metavar=("TSINGLEA", "TSINGLEB", "TSINGLEC"),
        help="three noise times for panel a (default: 1, 3, 5 us)",
    )
    p.add_argument("--singleton-layers", type=int, default=20)
    p.add_argument(
        "--output-dir", default=str(PROJECT_ROOT / "fig2_error_statistics" / "data_v4")
    )
    p.add_argument(
        "--data",
        default=str(
            PROJECT_ROOT
            / "fig2_error_statistics"
            / "data_v4"
            / "fig2_error_statistics.json"
        ),
    )
    p.add_argument(
        "--figure-dir", default=str(PROJECT_ROOT / "fig2_error_statistics" / "figures")
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.device_info:
        device_info()
        return
    if args.smoke:
        smoke_test()
        return
    if args.run:
        if args.device == "cpu" and args.n_circuits >= 20:
            raise RuntimeError(
                "Refusing the full ensemble on CPU. Use the A100 with --device cuda; "
                "--smoke is available for CPU validation."
            )
        run_ensemble(args)
    if args.plot:
        plot_figure(Path(args.data), Path(args.figure_dir))
    if not args.run and not args.plot:
        build_parser().print_help()


if __name__ == "__main__":
    main()
