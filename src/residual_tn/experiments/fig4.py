"""4x4 qutrit Monte Carlo benchmark against the full-state F2 estimator.

The Monte Carlo reference evolves pure qutrit states (local dimension d=3)
with Kraus operators obtained from the same 1000-slice 0522 local
superoperators.  The deterministic comparison is the existing full-state
second-order connected estimator evaluated with the projected qubit Kraus
channels for exactly the same circuit.

Channel compilation is an offline step and is reported separately.  The two
online wall times are measured on the same device after the channels and
circuit have been loaded.
"""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import replace
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

DEFAULT_REFERENCE = INPUTS / "0522_reference_data.pt"
DEFAULT_PAIR_DATA = None
DEFAULT_OUTPUT = RESULTS / "fig4/mc"
DEFAULT_FIGURES = RESULTS / "fig4/figures"
DEFAULT_CHANNEL_CACHE = INPUTS / "qutrit_kraus.pt"
DEFAULT_PAULI_REFERENCE = RESULTS / "fig4/pauli/pauli_16q_fullstate.json"
PROJECT_ROOT = RESULTS
os.environ.setdefault("PEPS_CORR2BP_PAIR_FORMULA", "centered_C")
os.environ.setdefault("CENTERED_FIDELITY_EXACT_BRANCH_CHUNK_MODES", "128")

from residual_tn.backend.channel import kraus_to_superoperator, superoperator_to_kraus  # noqa: E402
from residual_tn.backend.context import FidelityContext  # noqa: E402
from residual_tn.backend.exact_small import exact_compute_fidelity  # noqa: E402
from residual_tn.experiments import fig2 as fig2_core  # noqa: E402
from residual_tn.backend.residual_magnus import (  # noqa: E402
    ResidualMagnusLibrary,
    apply_operator_to_state,
)
from residual_tn.backend import residual_background_exact as residual_bg  # noqa: E402


DEFAULT_RESIDUAL_MANIFEST = CONFIGS / "residual_manifest.json"


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def choose_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def canonical_gate_id(value: Any) -> int | tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(x) for x in value)
    return int(value)


def remap_residual_layers(
    layers: Sequence[Sequence[Any]],
    reference: dict[str, Any],
) -> list[list[Any]]:
    """Map physical residual supports onto the reference state axes."""
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


def parse_local_fidelities(raw: dict[Any, Any]) -> dict[tuple[int, Any], float]:
    out: dict[tuple[int, Any], float] = {}
    for key, value in raw.items():
        parsed = ast.literal_eval(key) if isinstance(key, str) else key
        out[(int(parsed[0]), canonical_gate_id(parsed[1]))] = float(value)
    return out


def dense_two_qutrit_superoperator(local_axes: torch.Tensor) -> torch.Tensor:
    """Invert fix_superoperator_indexing for a two-qutrit local-axis map."""
    if tuple(local_axes.shape) != (9, 9, 9, 9):
        raise ValueError(f"expected (9,9,9,9), received {tuple(local_axes.shape)}")
    return (
        local_axes.reshape(3, 3, 3, 3, 3, 3, 3, 3)
        .permute(0, 2, 1, 3, 4, 6, 5, 7)
        .contiguous()
        .reshape(81, 81)
    )


def tp_residual(kraus: torch.Tensor) -> float:
    dim = int(kraus.shape[-1])
    closure = torch.einsum("uoi,uoj->ij", kraus.conj(), kraus)
    eye = torch.eye(dim, dtype=kraus.dtype, device=kraus.device)
    return float(torch.linalg.norm(closure - eye).detach().cpu())


def channel_roundtrip_residual(
    superoperator: torch.Tensor, kraus: torch.Tensor
) -> float:
    rebuilt = kraus_to_superoperator(kraus)
    denom = torch.linalg.norm(superoperator).clamp_min(1e-300)
    return float((torch.linalg.norm(rebuilt - superoperator) / denom).detach().cpu())


def trace_preserving_kraus_repair(kraus: torch.Tensor) -> torch.Tensor:
    """Apply the common right-normalization K_u <- K_u (sum K^dag K)^-1/2.

    The 1000-slice Crank-Nicolson map is trace preserving but is only
    approximately completely positive.  Removing its tiny negative Choi mode
    produces a CP Kraus set with a small closure residual.  This normalization
    restores a linear CPTP channel, which is required by standard trajectory
    sampling, while retaining the inherited Kraus span.
    """
    closure = torch.einsum("uoi,uoj->ij", kraus.conj(), kraus)
    closure = 0.5 * (closure + closure.conj().T)
    eigenvalues, eigenvectors = torch.linalg.eigh(closure)
    if float(eigenvalues.min().detach().cpu()) <= 0:
        raise RuntimeError("Kraus closure is not positive definite")
    inverse_sqrt = (
        eigenvectors
        @ torch.diag(torch.rsqrt(eigenvalues).to(kraus.dtype))
        @ eigenvectors.conj().T
    )
    return torch.einsum("uoi,ij->uoj", kraus, inverse_sqrt)


def build_qutrit_kraus_cache(
    *,
    width: int,
    length: int,
    piece_num: int,
    noise_time: float,
    kraus_tol: float,
    device: torch.device,
    cache_path: Path,
) -> tuple[dict[tuple[int, int], torch.Tensor], torch.Tensor, dict[str, Any]]:
    started = time.perf_counter()
    channels = fig2_core.build_channels(
        width,
        length,
        piece_num,
        noise_time,
        device,
        torch.complex128,
    )
    synchronize(device)
    compiled_seconds = time.perf_counter() - started

    single: dict[tuple[int, int], torch.Tensor] = {}
    single_cp_projection = 0.0
    single_cptp_adjustment = 0.0
    single_tp_before = 0.0
    single_tp_after = 0.0
    for phys in range(width * length):
        for mode in range(4):
            superop = channels.single_superoperators[phys, mode]
            current_cp = superoperator_to_kraus(superop, d=3, tol=kraus_tol)
            current = trace_preserving_kraus_repair(current_cp)
            single[(phys, mode)] = current.detach().cpu()
            single_cp_projection = max(
                single_cp_projection,
                channel_roundtrip_residual(superop, current_cp),
            )
            single_cptp_adjustment = max(
                single_cptp_adjustment,
                channel_roundtrip_residual(superop, current),
            )
            single_tp_before = max(single_tp_before, tp_residual(current_cp))
            single_tp_after = max(single_tp_after, tp_residual(current))

    two_superop = dense_two_qutrit_superoperator(channels.two_superoperator)
    two_cp = superoperator_to_kraus(two_superop, d=9, tol=kraus_tol)
    two = trace_preserving_kraus_repair(two_cp)
    two_cp_projection = channel_roundtrip_residual(two_superop, two_cp)
    two_cptp_adjustment = channel_roundtrip_residual(two_superop, two)
    two_tp_before = tp_residual(two_cp)
    two_tp_after = tp_residual(two)

    diagnostics = {
        "source": "0522 qutrit superoperators compiled with Crank-Nicolson slices",
        "width": int(width),
        "length": int(length),
        "piece_num": int(piece_num),
        "noise_time_ns": float(noise_time),
        "kraus_tol": float(kraus_tol),
        "compile_and_decompose_seconds": float(compiled_seconds),
        "single_kraus_ranks": sorted(
            {int(value.shape[0]) for value in single.values()}
        ),
        "two_kraus_rank": int(two.shape[0]),
        "cptp_repair": "positive-Choi Kraus modes followed by right closure normalization",
        "single_cp_projection_relative_residual_max": float(single_cp_projection),
        "two_cp_projection_relative_residual": float(two_cp_projection),
        "single_cptp_relative_adjustment_max": float(single_cptp_adjustment),
        "two_cptp_relative_adjustment": float(two_cptp_adjustment),
        "single_tp_residual_before_repair_max": float(single_tp_before),
        "two_tp_residual_before_repair": float(two_tp_before),
        "single_tp_residual_after_repair_max": float(single_tp_after),
        "two_tp_residual_after_repair": float(two_tp_after),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": diagnostics,
            "single_kraus": single,
            "two_kraus": two.detach().cpu(),
        },
        cache_path,
    )
    del channels, two_superop
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (
        {key: value.to(device=device) for key, value in single.items()},
        two.to(device=device),
        diagnostics,
    )


def load_or_build_qutrit_kraus(
    *,
    width: int,
    length: int,
    piece_num: int,
    noise_time: float,
    kraus_tol: float,
    device: torch.device,
    cache_path: Path,
    rebuild: bool,
) -> tuple[dict[tuple[int, int], torch.Tensor], torch.Tensor, dict[str, Any]]:
    if cache_path.exists() and not rebuild:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        metadata = dict(payload["metadata"])
        expected = {
            "width": int(width),
            "length": int(length),
            "piece_num": int(piece_num),
            "noise_time_ns": float(noise_time),
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(
                    f"qutrit channel cache mismatch for {key}: "
                    f"{metadata.get(key)!r} != {value!r}"
                )
        single = {
            (int(key[0]), int(key[1])): value.to(device=device, dtype=torch.complex128)
            for key, value in payload["single_kraus"].items()
        }
        two = payload["two_kraus"].to(device=device, dtype=torch.complex128)
        metadata["loaded_from_cache"] = True
        metadata["cache_path"] = str(cache_path)
        return single, two, metadata
    single, two, metadata = build_qutrit_kraus_cache(
        width=width,
        length=length,
        piece_num=piece_num,
        noise_time=noise_time,
        kraus_tol=kraus_tol,
        device=device,
        cache_path=cache_path,
    )
    metadata["loaded_from_cache"] = False
    metadata["cache_path"] = str(cache_path)
    return single, two, metadata


def axis_for_physical(reference: dict[str, Any], physical: int) -> int:
    mapping = reference["phys_to_1d"]
    if isinstance(mapping, dict):
        return int(mapping[int(physical)])
    return int(mapping[int(physical)])


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


def build_projected_reference_context(
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


def f2_trajectory_from_result(
    ctx: FidelityContext,
    result: Any,
) -> list[float]:
    n_layers = len(ctx.layers)
    log_local = [0.0] * n_layers
    for gate_idx, value in result.local_fidelities.items():
        layer = int(ctx.gate_map[int(gate_idx)][0])
        log_local[layer] += math.log(max(float(value), 1e-300))
    trajectory = [1.0]
    total = 0.0
    for layer in range(n_layers):
        total += log_local[layer]
        total += float(result.self_energy_by_layer.get(layer, 0.0))
        trajectory.append(float(math.exp(total)))
    return trajectory


def reference_ideal_history_residual(
    reference: dict[str, Any],
    ctx: FidelityContext,
) -> float:
    n_qubits = int(reference["N_QUBITS"])
    psi = torch.zeros(2**n_qubits, dtype=torch.complex128, device=ctx.device)
    psi[0] = 1.0
    maximum = 0.0
    for layer_idx, gates in enumerate(ctx.layers):
        for gate in gates:
            qubits = tuple(int(q) for q in gate["qubits"])
            if len(qubits) == 1:
                psi = fig2_core.apply_1q_state(
                    psi, gate["ideal_unitary"], qubits[0], n_qubits, d=2
                )
            else:
                psi = fig2_core.apply_2q_state(
                    psi, gate["ideal_unitary"], qubits[0], qubits[1], n_qubits, d=2
                )
        expected = (
            reference["ideal_state_history"][layer_idx + 1]
            .to(device=ctx.device, dtype=torch.complex128)
            .reshape(-1)
        )
        maximum = max(maximum, float(torch.linalg.norm(psi - expected).detach().cpu()))
    return maximum


def cached_f2_trajectory(pair_payload: dict[str, Any], n_layers: int) -> dict[str, Any]:
    local = parse_local_fidelities(pair_payload["local_fidelities"])
    log_local = [0.0] * n_layers
    for (layer, _), value in local.items():
        log_local[int(layer)] += math.log(max(float(value), 1e-300))

    pair_delta = [0.0] * n_layers
    valid_pairs = 0
    for key, response in pair_payload["M_pair"].items():
        source_layer, source_gate, target_layer, target_gate = key
        source_id = canonical_gate_id(source_gate)
        target_id = canonical_gate_id(target_gate)
        first = local.get((int(source_layer), source_id))
        second = local.get((int(target_layer), target_id))
        if first is None or second is None or first <= 0 or second <= 0:
            continue
        delta = float(response) / (first * second) - 1.0
        pair_delta[int(target_layer)] += delta
        valid_pairs += 1

    trajectory = [1.0]
    total = 0.0
    for layer in range(n_layers):
        total += log_local[layer] + pair_delta[layer]
        trajectory.append(float(math.exp(total)))
    return {
        "trajectory": trajectory,
        "pair_count": int(valid_pairs),
        "log_f1": float(sum(log_local)),
        "sum_delta": float(sum(pair_delta)),
    }


def run_full_state_f2(
    reference: dict[str, Any],
    device: torch.device,
    projected_residual: Sequence[Sequence[Any]],
) -> tuple[dict[str, Any], float]:
    ctx = build_projected_reference_context(reference, device)
    ideal_residual = reference_ideal_history_residual(reference, ctx)
    if ideal_residual > 1e-9:
        raise RuntimeError(
            f"reconstructed circuit does not match cached ideal history: residual={ideal_residual}"
        )
    synchronize(device)
    started = time.perf_counter()
    result = residual_bg.connected_fidelity_trajectory(ctx, projected_residual)
    synchronize(device)
    elapsed = time.perf_counter() - started
    payload = {
        "trajectory": result["fidelity_second_order"],
        "zero_insertion_trajectory": result["zero_insertion_fidelity"],
        "first_order_trajectory": result["fidelity_first_order"],
        "final_fidelity": float(result["fidelity_second_order"][-1]),
        "fidelity_first_order": float(result["fidelity_first_order"][-1]),
        "log_fidelity_first_order": float(
            math.log(
                result["fidelity_first_order"][-1]
                / result["zero_insertion_fidelity"][-1]
            )
        ),
        "sum_delta": float(
            math.log(
                result["fidelity_second_order"][-1] / result["fidelity_first_order"][-1]
            )
        ),
        "pair_count": int(result["final_pair_count"]),
        "ideal_history_max_l2_residual": float(ideal_residual),
        "common_background": "B_l = U_l K_res,I,l",
    }
    del ctx, result
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload, float(elapsed)


def precompute_effects(kraus: torch.Tensor) -> torch.Tensor:
    return torch.einsum("uoi,uoj->uij", kraus.conj(), kraus)


def build_qutrit_events(
    reference: dict[str, Any],
    single_kraus: dict[tuple[int, int], torch.Tensor],
    two_kraus: torch.Tensor,
    *,
    proposal_mode: str,
    proposal_mix: float,
) -> list[list[dict[str, Any]]]:
    n_qubits = int(reference["N_QUBITS"])
    single_effects = {
        key: precompute_effects(value) for key, value in single_kraus.items()
    }
    two_effects = precompute_effects(two_kraus)

    def trace_proposal(kraus: torch.Tensor, effects: torch.Tensor) -> torch.Tensor:
        local_size = int(kraus.shape[-1])
        probabilities = torch.diagonal(effects, dim1=-2, dim2=-1).real.sum(dim=-1)
        probabilities = probabilities.clamp_min(0.0)
        return probabilities / probabilities.sum()

    def reduced_density(psi: torch.Tensor, targets: Sequence[int]) -> torch.Tensor:
        targets = tuple(int(x) for x in targets)
        tensor = psi.reshape(*([2] * n_qubits))
        moved = tensor.movedim(targets, tuple(range(len(targets))))
        flat = moved.reshape(2 ** len(targets), -1)
        return flat @ flat.conj().T

    def ideal_proposal(
        effects: torch.Tensor,
        ideal_qubit_state: torch.Tensor,
        targets: Sequence[int],
    ) -> torch.Tensor:
        rho_qubit = reduced_density(ideal_qubit_state, targets)
        if len(targets) == 1:
            comp = torch.tensor([0, 1], dtype=torch.long, device=effects.device)
            local_size = 3
        else:
            comp = torch.tensor([0, 1, 3, 4], dtype=torch.long, device=effects.device)
            local_size = 9
        rho_qutrit = torch.zeros(
            (local_size, local_size), dtype=effects.dtype, device=effects.device
        )
        rho_qutrit[comp[:, None], comp[None, :]] = rho_qubit
        probabilities = torch.einsum("uij,ji->u", effects, rho_qutrit).real.clamp_min(
            0.0
        )
        return probabilities / probabilities.sum()

    def proposal(
        kraus: torch.Tensor,
        effects: torch.Tensor,
        ideal_qubit_state: torch.Tensor,
        targets: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trace_probabilities = trace_proposal(kraus, effects)
        if proposal_mode == "trace":
            probabilities = trace_probabilities
        elif proposal_mode == "ideal":
            ideal_probabilities = ideal_proposal(effects, ideal_qubit_state, targets)
            probabilities = (
                1.0 - proposal_mix
            ) * ideal_probabilities + proposal_mix * trace_probabilities
            probabilities = probabilities / probabilities.sum()
        else:
            raise ValueError(f"unsupported fixed proposal mode: {proposal_mode}")
        scaled = kraus / torch.sqrt(probabilities.clamp_min(1e-300))[:, None, None]
        return probabilities, scaled

    layers: list[list[dict[str, Any]]] = []
    for layer_idx, pattern_layer in enumerate(reference["pattern"]):
        current: list[dict[str, Any]] = []
        ideal_before = (
            reference["ideal_state_history"][layer_idx]
            .to(device=two_kraus.device, dtype=two_kraus.dtype)
            .reshape(-1)
        )
        if layer_idx % 2 == 0:
            for physical in range(n_qubits):
                mode = int(pattern_layer[physical])
                key = (physical, mode)
                targets = (axis_for_physical(reference, physical),)
                current_proposal, current_scaled = proposal(
                    single_kraus[key], single_effects[key], ideal_before, targets
                )
                current.append(
                    {
                        "targets": targets,
                        "kraus": single_kraus[key],
                        "effects": single_effects[key],
                        "proposal": current_proposal,
                        "scaled_kraus": current_scaled,
                    }
                )
        else:
            for pair in pattern_layer:
                q1, q2 = int(pair[0]), int(pair[1])
                targets = (
                    axis_for_physical(reference, q1),
                    axis_for_physical(reference, q2),
                )
                current_proposal, current_scaled = proposal(
                    two_kraus, two_effects, ideal_before, targets
                )
                current.append(
                    {
                        "targets": targets,
                        "kraus": two_kraus,
                        "effects": two_effects,
                        "proposal": current_proposal,
                        "scaled_kraus": current_scaled,
                    }
                )
        layers.append(current)
    return layers


def sample_local_kraus(
    states: torch.Tensor,
    *,
    targets: Sequence[int],
    kraus: torch.Tensor,
    effects: torch.Tensor,
    n_qubits: int,
    local_dim: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, float]:
    batch = int(states.shape[0])
    targets = tuple(int(x) for x in targets)
    local_axes = tuple(target + 1 for target in targets)
    front_axes = tuple(range(1, 1 + len(targets)))
    tensor = states.reshape(batch, *([local_dim] * n_qubits))
    moved = tensor.movedim(local_axes, front_axes)
    moved_shape = moved.shape
    local_size = local_dim ** len(targets)
    flat = moved.reshape(batch, local_size, -1)

    rho = torch.matmul(flat, flat.conj().transpose(1, 2))
    probabilities = torch.einsum("uij,bji->bu", effects, rho).real
    probabilities = probabilities.clamp_min(0.0)
    totals = probabilities.sum(dim=1)
    if bool(torch.any(totals <= 0)):
        raise RuntimeError("encountered a non-positive total Kraus probability")
    max_probability_residual = float(torch.max(torch.abs(totals - 1.0)).detach().cpu())
    normalized = probabilities / totals.unsqueeze(1)
    cumulative = torch.cumsum(normalized, dim=1)
    cumulative[:, -1] = 1.0
    random_values = torch.rand(
        (batch, 1),
        dtype=states.real.dtype,
        device=states.device,
        generator=generator,
    )
    choices = torch.searchsorted(cumulative.contiguous(), random_values).squeeze(1)
    choices = choices.clamp_max(kraus.shape[0] - 1)
    selected = kraus.index_select(0, choices)
    selected_probabilities = probabilities.gather(1, choices[:, None]).squeeze(1)
    output = torch.einsum("boi,bir->bor", selected, flat)
    output = (
        output / torch.sqrt(selected_probabilities.clamp_min(1e-300))[:, None, None]
    )
    restored = output.reshape(moved_shape).movedim(front_axes, local_axes)
    return restored.reshape(batch, -1), max_probability_residual


def sample_fixed_proposal_kraus(
    states: torch.Tensor,
    *,
    targets: Sequence[int],
    proposal: torch.Tensor,
    scaled_kraus: torch.Tensor,
    n_qubits: int,
    local_dim: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Unbiased state-independent Kraus Monte Carlo with importance weights."""
    batch = int(states.shape[0])
    targets = tuple(int(x) for x in targets)
    local_axes = tuple(target + 1 for target in targets)
    front_axes = tuple(range(1, 1 + len(targets)))
    tensor = states.reshape(batch, *([local_dim] * n_qubits))
    moved = tensor.movedim(local_axes, front_axes)
    moved_shape = moved.shape
    local_size = local_dim ** len(targets)
    flat = moved.reshape(batch, local_size, -1)
    cumulative = torch.cumsum(proposal, dim=0)
    cumulative[-1] = 1.0
    random_values = torch.rand(
        batch,
        dtype=states.real.dtype,
        device=states.device,
        generator=generator,
    )
    choices = torch.searchsorted(cumulative.contiguous(), random_values)
    choices = choices.clamp_max(scaled_kraus.shape[0] - 1)
    selected = scaled_kraus.index_select(0, choices)
    output = torch.einsum("boi,bir->bor", selected, flat)
    restored = output.reshape(moved_shape).movedim(front_axes, local_axes)
    return restored.reshape(batch, -1)


def computational_qutrit_indices(n_qubits: int, device: torch.device) -> torch.Tensor:
    binary = torch.arange(2**n_qubits, dtype=torch.long, device=device)
    output = torch.zeros_like(binary)
    for axis in range(n_qubits):
        bit = (binary >> (n_qubits - 1 - axis)) & 1
        output += bit * (3 ** (n_qubits - 1 - axis))
    return output


def qutrit_fidelity_values(
    states: torch.Tensor,
    ideal_qubit: torch.Tensor,
    comp_indices: torch.Tensor,
) -> torch.Tensor:
    computational = states.index_select(1, comp_indices)
    amplitudes = torch.einsum("j,bj->b", ideal_qubit.conj(), computational)
    return torch.abs(amplitudes) ** 2


def qutrit_local_pauli_probabilities(
    states: torch.Tensor,
    n_qubits: int,
    comp_indices: torch.Tensor,
) -> torch.Tensor:
    """Return terminal p(+/-) for all local X/Y/Z effects.

    The qutrit trajectories are first projected onto the global computational
    subspace, matching ``compute_local_pauli_fullstate.py``.  The returned
    probabilities are intentionally *not* normalized per trajectory.  For
    fixed-proposal sampling, the squared trajectory norm is the importance
    weight, so the Monte Carlo estimator must average p_plus and p_minus first
    and take their ratio only afterwards.

    Shape: ``(batch, n_qubits, 3, 2)`` with the last axis ``(+,-)``.
    """
    if states.ndim != 2:
        raise ValueError(
            f"expected a state batch, received shape {tuple(states.shape)}"
        )
    batch = int(states.shape[0])
    computational = states.index_select(1, comp_indices)
    tensor = computational.reshape(batch, *([2] * int(n_qubits)))
    probabilities = []
    for qubit in range(int(n_qubits)):
        local = torch.movedim(tensor, qubit + 1, 1).reshape(batch, 2, -1)
        rho00 = torch.sum(torch.abs(local[:, 0, :]) ** 2, dim=-1).real
        rho11 = torch.sum(torch.abs(local[:, 1, :]) ** 2, dim=-1).real
        rho01 = torch.einsum("bi,bi->b", local[:, 0, :], local[:, 1, :].conj())
        expectation = torch.stack(
            (
                2.0 * rho01.real,
                -2.0 * rho01.imag,
                rho00 - rho11,
            ),
            dim=1,
        )
        trace = (rho00 + rho11)[:, None]
        probabilities.append(
            torch.stack(
                ((trace + expectation) / 2.0, (trace - expectation) / 2.0),
                dim=-1,
            )
        )
    return torch.stack(probabilities, dim=1).to(torch.float64)


def run_mc_batch(
    *,
    batch_size: int,
    n_qubits: int,
    event_layers: Sequence[Sequence[dict[str, Any]]],
    ideal_history: torch.Tensor,
    comp_indices: torch.Tensor,
    generator: torch.Generator,
    sampling: str,
    record_local_pauli: bool,
    residual_layers: Sequence[Sequence[Any]],
) -> tuple[np.ndarray, dict[str, float], np.ndarray | None]:
    device = ideal_history.device
    dim = 3**n_qubits
    states = torch.zeros((batch_size, dim), dtype=ideal_history.dtype, device=device)
    states[:, 0] = 1.0
    values = torch.empty(
        (batch_size, len(event_layers) + 1), dtype=torch.float64, device=device
    )
    values[:, 0] = 1.0
    max_probability_residual = 0.0
    for layer_idx, events in enumerate(event_layers):
        for kernel in residual_layers[layer_idx]:
            states = apply_operator_to_state(
                states,
                kernel.operator,
                kernel.support,
                n_qubits=n_qubits,
                local_dim=3,
            )
        for event in events:
            if sampling == "state-dependent":
                states, residual = sample_local_kraus(
                    states,
                    targets=event["targets"],
                    kraus=event["kraus"],
                    effects=event["effects"],
                    n_qubits=n_qubits,
                    local_dim=3,
                    generator=generator,
                )
                max_probability_residual = max(max_probability_residual, residual)
            elif sampling == "fixed-proposal":
                states = sample_fixed_proposal_kraus(
                    states,
                    targets=event["targets"],
                    proposal=event["proposal"],
                    scaled_kraus=event["scaled_kraus"],
                    n_qubits=n_qubits,
                    local_dim=3,
                    generator=generator,
                )
            else:
                raise ValueError(f"unsupported Monte Carlo sampling scheme: {sampling}")
        values[:, layer_idx + 1] = qutrit_fidelity_values(
            states,
            ideal_history[layer_idx + 1],
            comp_indices,
        )
    norms = torch.sum(torch.abs(states) ** 2, dim=1).real
    max_norm_deviation = float(torch.max(torch.abs(norms - 1.0)).detach().cpu())
    local_pauli = None
    if record_local_pauli:
        local_pauli = (
            qutrit_local_pauli_probabilities(
                states,
                n_qubits,
                comp_indices,
            )
            .detach()
            .cpu()
            .numpy()
        )
    return (
        values.detach().cpu().numpy(),
        {
            "max_gate_probability_sum_residual": float(max_probability_residual),
            # Fixed-proposal trajectories carry importance weights in their norms,
            # so deviation from unity is expected and is not a normalization error.
            "max_final_sample_norm_deviation_from_unity": float(max_norm_deviation),
        },
        local_pauli,
    )


def summarize_samples(samples: np.ndarray) -> dict[str, list[float]]:
    count = int(samples.shape[0])
    mean = samples.mean(axis=0)
    if count > 1:
        standard_error = samples.std(axis=0, ddof=1) / math.sqrt(count)
    else:
        standard_error = np.full(samples.shape[1], np.nan, dtype=float)
    ci95 = 1.959963984540054 * standard_error
    return {
        "mean": mean.astype(float).tolist(),
        "standard_error": standard_error.astype(float).tolist(),
        "ci95_half_width": ci95.astype(float).tolist(),
    }


def summarize_local_pauli_samples(samples: np.ndarray) -> dict[str, Any]:
    """Summarize p(+/-) and the conditional Pauli ratio across trajectories."""
    if samples.ndim != 4 or samples.shape[-2:] != (3, 2):
        raise ValueError(
            "expected local Pauli samples with shape "
            f"(trajectories, sites, 3, 2), received {samples.shape}"
        )
    count = int(samples.shape[0])
    if count < 2:
        raise ValueError(
            "at least two trajectories are required for a confidence interval"
        )

    plus = np.asarray(samples[..., 0], dtype=np.float64)
    minus = np.asarray(samples[..., 1], dtype=np.float64)
    mean_plus = plus.mean(axis=0)
    mean_minus = minus.mean(axis=0)
    se_plus = plus.std(axis=0, ddof=1) / math.sqrt(count)
    se_minus = minus.std(axis=0, ddof=1) / math.sqrt(count)
    ci_plus = 1.959963984540054 * se_plus
    ci_minus = 1.959963984540054 * se_minus

    mean_survival = mean_plus + mean_minus
    mean_signed = mean_plus - mean_minus
    conditional = mean_signed / np.maximum(mean_survival, 1e-300)

    # Delta-method standard error for the ratio of the two sample means.  The
    # influence variable below includes the p_plus/p_minus covariance, which is
    # important for fixed-proposal trajectories.
    survival = plus + minus
    signed = plus - minus
    influence = (signed - conditional[None, ...] * survival) / np.maximum(
        mean_survival[None, ...], 1e-300
    )
    conditional_se = influence.std(axis=0, ddof=1) / math.sqrt(count)
    conditional_ci = 1.959963984540054 * conditional_se

    return {
        "trajectories": count,
        "ci_method": "normal delta-method CI for ratio of mean p_plus and p_minus",
        "p_plus_mean": mean_plus.tolist(),
        "p_plus_standard_error": se_plus.tolist(),
        "p_plus_ci95_half_width": ci_plus.tolist(),
        "p_minus_mean": mean_minus.tolist(),
        "p_minus_standard_error": se_minus.tolist(),
        "p_minus_ci95_half_width": ci_minus.tolist(),
        "survival_mean": mean_survival.tolist(),
        "pauli_unconditional_mean": mean_signed.tolist(),
        "pauli_conditional_mean": conditional.tolist(),
        "pauli_conditional_standard_error": conditional_se.tolist(),
        "pauli_conditional_ci95_half_width": conditional_ci.tolist(),
        "probability_min": float(np.min(samples)),
        "probability_max": float(np.max(samples)),
    }


def load_full_state_pauli_reference(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    second_order = payload["full_state"]["second_order"]
    conditional = np.asarray(second_order["pauli_conditional"], dtype=np.float64)
    p_plus = np.asarray(second_order["p_plus"], dtype=np.float64)
    p_minus = np.asarray(second_order["p_minus"], dtype=np.float64)
    if conditional.shape != (16, 3):
        raise ValueError(
            f"expected 16x3 full-state Pauli reference, received {conditional.shape}"
        )
    return {
        "source": str(path),
        "case": payload.get("metadata", {}).get("case"),
        "conditional": conditional.tolist(),
        "p_plus": p_plus.tolist(),
        "p_minus": p_minus.tolist(),
    }


def compare_local_pauli_to_full_state(
    mc_summary: dict[str, Any],
    full_state: dict[str, Any],
) -> dict[str, Any]:
    mc = np.asarray(mc_summary["pauli_conditional_mean"], dtype=np.float64)
    ref = np.asarray(full_state["conditional"], dtype=np.float64)
    ci = np.asarray(mc_summary["pauli_conditional_ci95_half_width"], dtype=np.float64)
    absolute = np.abs(mc - ref)
    coverage = absolute <= ci
    per_basis = {
        basis: {
            "mae": float(np.mean(absolute[:, basis_idx])),
            "max_absolute_error": float(np.max(absolute[:, basis_idx])),
            "ci_coverage_count": int(np.sum(coverage[:, basis_idx])),
            "ci_coverage_fraction": float(np.mean(coverage[:, basis_idx])),
        }
        for basis_idx, basis in enumerate(("X", "Y", "Z"))
    }
    return {
        "absolute_error": absolute.tolist(),
        "relative_error_to_full_state": (
            absolute / np.maximum(np.abs(ref), 1e-12)
        ).tolist(),
        "ci_coverage": coverage.tolist(),
        "mae": float(np.mean(absolute)),
        "p95_absolute_error": float(np.percentile(absolute, 95)),
        "max_absolute_error": float(np.max(absolute)),
        "ci_coverage_count": int(np.sum(coverage)),
        "ci_coverage_fraction": float(np.mean(coverage)),
        "per_basis": per_basis,
    }


def parse_checkpoints(text: str | None, total: int) -> list[int]:
    if text:
        values = sorted(
            {int(value.strip()) for value in text.split(",") if value.strip()}
        )
    else:
        values = []
        current = 16
        while current < total:
            values.append(current)
            current *= 2
    values = [value for value in values if 0 < value <= total]
    if total not in values:
        values.append(total)
    return sorted(set(values))


def run_qutrit_monte_carlo(
    *,
    reference: dict[str, Any],
    single_kraus: dict[tuple[int, int], torch.Tensor],
    two_kraus: torch.Tensor,
    trajectories: int,
    batch_size: int,
    checkpoint_values: Sequence[int],
    seed: int,
    device: torch.device,
    progress: bool,
    sampling: str,
    proposal_mode: str,
    proposal_mix: float,
    mc_dtype: torch.dtype,
    record_local_pauli: bool,
    qutrit_residual: Sequence[Sequence[Any]],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray | None, float]:
    n_qubits = int(reference["N_QUBITS"])
    single_kraus = {
        key: value.to(dtype=mc_dtype) for key, value in single_kraus.items()
    }
    two_kraus = two_kraus.to(dtype=mc_dtype)
    event_layers = build_qutrit_events(
        reference,
        single_kraus,
        two_kraus,
        proposal_mode=proposal_mode,
        proposal_mix=proposal_mix,
    )
    ideal_history = torch.stack(
        [
            state.to(dtype=mc_dtype).reshape(-1)
            for state in reference["ideal_state_history"]
        ]
    ).to(device=device)
    residual_layers = [
        [
            replace(kernel, operator=kernel.operator.to(dtype=mc_dtype))
            for kernel in layer
        ]
        for layer in qutrit_residual
    ]
    comp_indices = computational_qutrit_indices(n_qubits, device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    collected: list[np.ndarray] = []
    local_pauli_collected: list[np.ndarray] = []
    checkpoints: list[dict[str, Any]] = []
    completed = 0
    max_probability_residual = 0.0
    max_norm_residual = 0.0
    synchronize(device)
    started = time.perf_counter()
    for checkpoint in checkpoint_values:
        while completed < checkpoint:
            current_batch = min(int(batch_size), checkpoint - completed)
            batch_values, diagnostics, batch_local_pauli = run_mc_batch(
                batch_size=current_batch,
                n_qubits=n_qubits,
                event_layers=event_layers,
                ideal_history=ideal_history,
                comp_indices=comp_indices,
                generator=generator,
                sampling=sampling,
                record_local_pauli=record_local_pauli,
                residual_layers=residual_layers,
            )
            collected.append(batch_values)
            if batch_local_pauli is not None:
                local_pauli_collected.append(batch_local_pauli)
            completed += current_batch
            max_probability_residual = max(
                max_probability_residual,
                diagnostics["max_gate_probability_sum_residual"],
            )
            max_norm_residual = max(
                max_norm_residual,
                diagnostics["max_final_sample_norm_deviation_from_unity"],
            )
            if progress:
                synchronize(device)
                print(
                    f"[qutrit-mc] trajectories={completed}/{trajectories} "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
        synchronize(device)
        current_samples = np.concatenate(collected, axis=0)
        summary = summarize_samples(current_samples)
        checkpoints.append(
            {
                "trajectories": int(checkpoint),
                "final_mean": float(summary["mean"][-1]),
                "final_standard_error": float(summary["standard_error"][-1]),
                "final_ci95_half_width": float(summary["ci95_half_width"][-1]),
                "cumulative_wall_seconds": float(time.perf_counter() - started),
            }
        )
    synchronize(device)
    elapsed = time.perf_counter() - started
    samples = np.concatenate(collected, axis=0)
    local_pauli_samples = (
        np.concatenate(local_pauli_collected, axis=0) if local_pauli_collected else None
    )
    final_summary = summarize_samples(samples)
    payload = {
        "trajectories": int(trajectories),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "sampling": sampling,
        "fixed_proposal_mode": proposal_mode,
        "fixed_proposal_trace_mix": float(proposal_mix),
        "state_dtype": str(mc_dtype),
        "trajectory_mean": final_summary["mean"],
        "trajectory_standard_error": final_summary["standard_error"],
        "trajectory_ci95_half_width": final_summary["ci95_half_width"],
        "checkpoints": checkpoints,
        "max_gate_probability_sum_residual": float(max_probability_residual),
        "max_final_sample_norm_deviation_from_unity": float(max_norm_residual),
        "final_sample_norm_interpretation": (
            "importance-weight magnitude; deviation from unity is expected"
            if sampling == "fixed-proposal"
            else "normalized-state residual; should be close to zero"
        ),
    }
    return payload, samples, local_pauli_samples, float(elapsed)


def write_results(
    output_dir: Path,
    payload: dict[str, Any],
    samples: np.ndarray,
    local_pauli_samples: np.ndarray | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "qutrit_mc_4x4_vs_fullstate_f2.json"
    csv_path = output_dir / "qutrit_mc_4x4_vs_fullstate_f2.csv"
    npz_path = output_dir / "qutrit_mc_4x4_samples.npz"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    arrays: dict[str, np.ndarray] = {"layer_fidelity_samples": samples}
    if local_pauli_samples is not None:
        arrays["local_pauli_probability_samples"] = local_pauli_samples
    np.savez_compressed(npz_path, **arrays)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "layer",
                "f2_full_state",
                "qutrit_mc_mean",
                "qutrit_mc_standard_error",
                "qutrit_mc_ci95_half_width",
            ],
        )
        writer.writeheader()
        f2 = payload["full_state_f2"]["trajectory"]
        mc = payload["qutrit_monte_carlo"]
        for layer in range(len(f2)):
            writer.writerow(
                {
                    "layer": layer,
                    "f2_full_state": f2[layer],
                    "qutrit_mc_mean": mc["trajectory_mean"][layer],
                    "qutrit_mc_standard_error": mc["trajectory_standard_error"][layer],
                    "qutrit_mc_ci95_half_width": mc["trajectory_ci95_half_width"][
                        layer
                    ],
                }
            )
    print(json_path, flush=True)
    print(csv_path, flush=True)
    print(npz_path, flush=True)

    if "local_pauli" in payload:
        local_csv_path = output_dir / "qutrit_mc_local_pauli_vs_fullstate_c2.csv"
        local = payload["local_pauli"]
        mc = local["qutrit_mc"]["pauli_conditional_mean"]
        ci = local["qutrit_mc"]["pauli_conditional_ci95_half_width"]
        full = local["full_state_c2"]["conditional"]
        error = local["comparison"]["absolute_error"]
        with local_csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "site_axis",
                    "basis",
                    "full_state_c2",
                    "qutrit_mc_mean",
                    "qutrit_mc_ci95_half_width",
                    "absolute_error",
                    "mc_ci_contains_full_state",
                ],
            )
            writer.writeheader()
            for site in range(len(full)):
                for basis_idx, basis in enumerate(("X", "Y", "Z")):
                    writer.writerow(
                        {
                            "site_axis": site,
                            "basis": basis,
                            "full_state_c2": full[site][basis_idx],
                            "qutrit_mc_mean": mc[site][basis_idx],
                            "qutrit_mc_ci95_half_width": ci[site][basis_idx],
                            "absolute_error": error[site][basis_idx],
                            "mc_ci_contains_full_state": local["comparison"][
                                "ci_coverage"
                            ][site][basis_idx],
                        }
                    )
        print(local_csv_path, flush=True)


def run_production(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    width = int(round(math.sqrt(int(reference["N_QUBITS"]))))
    length = int(reference["N_QUBITS"]) // width
    n_layers = int(reference["N_LAYERS"])
    if (width, length, n_layers) != (4, 4, 20):
        raise ValueError(
            f"production reference must be 4x4 with 20 layers, got {width}x{length} L={n_layers}"
        )
    print(
        f"[device] {device}"
        + (f" / {torch.cuda.get_device_name(device)}" if device.type == "cuda" else ""),
        flush=True,
    )
    print(
        f"[config] qutrit d=3, 4x4, layers=20, T1=T2={args.noise_time:g} ns, "
        f"piece_num={args.piece_num}, trajectories={args.trajectories}",
        flush=True,
    )

    single_qutrit, two_qutrit, channel_diagnostics = load_or_build_qutrit_kraus(
        width=width,
        length=length,
        piece_num=args.piece_num,
        noise_time=args.noise_time,
        kraus_tol=args.kraus_tol,
        device=device,
        cache_path=args.channel_cache,
        rebuild=args.rebuild_channels,
    )
    manifest, geometry = residual_bg.load_residual_geometry(
        args.residual_manifest, width, length
    )
    residual_steps = (
        args.piece_num
        if int(args.residual_integration_steps) == 0
        else int(args.residual_integration_steps)
    )
    residual_started = time.perf_counter()
    projected_physical = residual_bg.compile_pattern_residual(
        ResidualMagnusLibrary(
            int(reference["N_QUBITS"]),
            integration_steps=residual_steps,
            dtype=torch.complex128,
            device=device,
            projected_qubits=True,
        ),
        reference["pattern"],
        geometry["edges"],
        scheme=args.kernel_scheme,
    )
    qutrit_physical = residual_bg.compile_pattern_residual(
        ResidualMagnusLibrary(
            int(reference["N_QUBITS"]),
            integration_steps=residual_steps,
            dtype=torch.complex128,
            device=device,
            projected_qubits=False,
        ),
        reference["pattern"],
        geometry["edges"],
        scheme=args.kernel_scheme,
    )
    projected_residual = remap_residual_layers(projected_physical, reference)
    qutrit_residual = remap_residual_layers(qutrit_physical, reference)
    residual_compile_seconds = time.perf_counter() - residual_started
    print(
        f"[residual] compiled projected and qutrit {args.kernel_scheme} "
        f"backgrounds: layers={n_layers} edges/layer={len(geometry['edges'])} "
        f"steps={residual_steps} seconds={residual_compile_seconds:.1f}",
        flush=True,
    )
    cached = {
        "source": "legacy no-residual pair cache intentionally not used",
        "compatible_with_common_background": False,
    }

    if args.skip_direct_f2:
        raise ValueError(
            "--skip-direct-f2 is incompatible with the residual-dressed run; "
            "the legacy pair cache has no coherent residual"
        )
    else:
        print(
            "[full-state F2] starting deterministic second-order calculation",
            flush=True,
        )
        direct, f2_seconds = run_full_state_f2(reference, device, projected_residual)
        direct["source"] = "direct exact full-state second-order calculation"
        print(
            f"[full-state F2] F={direct['final_fidelity']:.12f} "
            f"wall={f2_seconds:.1f}s pairs={direct['pair_count']}",
            flush=True,
        )

    checkpoints = parse_checkpoints(args.checkpoints, args.trajectories)
    mc_dtype = torch.complex64 if args.mc_dtype == "complex64" else torch.complex128
    mc, samples, local_pauli_samples, mc_seconds = run_qutrit_monte_carlo(
        reference=reference,
        single_kraus=single_qutrit,
        two_kraus=two_qutrit,
        trajectories=args.trajectories,
        batch_size=args.batch_size,
        checkpoint_values=checkpoints,
        seed=args.seed,
        device=device,
        progress=args.progress,
        sampling=args.mc_sampling,
        proposal_mode=args.fixed_proposal,
        proposal_mix=args.fixed_proposal_mix,
        mc_dtype=mc_dtype,
        record_local_pauli=args.record_local_pauli,
        qutrit_residual=qutrit_residual,
    )
    final_mean = float(mc["trajectory_mean"][-1])
    final_ci = float(mc["trajectory_ci95_half_width"][-1])
    print(
        f"[qutrit MC] F={final_mean:.12f} +/- {final_ci:.3e} (95% CI) "
        f"wall={mc_seconds:.1f}s",
        flush=True,
    )

    direct_final = float(direct["trajectory"][-1])
    local_pauli_payload = None
    if args.record_local_pauli:
        if local_pauli_samples is None:
            raise RuntimeError(
                "local Pauli recording was requested but no samples were returned"
            )
        if not args.local_pauli_reference.exists():
            raise FileNotFoundError(
                "local Pauli full-state reference was not found: "
                f"{args.local_pauli_reference}"
            )
        mc_local = summarize_local_pauli_samples(local_pauli_samples)
        full_local = load_full_state_pauli_reference(args.local_pauli_reference)
        local_comparison = compare_local_pauli_to_full_state(mc_local, full_local)
        local_pauli_payload = {
            "definition": {
                "terminal_effects": "global computational-subspace projected p_plus/p_minus",
                "reported_expectation": "(mean p_plus - mean p_minus) / (mean p_plus + mean p_minus)",
                "basis_order": ["X", "Y", "Z"],
                "trajectory_weighting": (
                    "fixed-proposal importance weights are retained; no per-trajectory normalization"
                ),
            },
            "full_state_c2": full_local,
            "qutrit_mc": mc_local,
            "comparison": local_comparison,
        }
        print(
            f"[local Pauli] MAE={local_comparison['mae']:.3e} "
            f"p95={local_comparison['p95_absolute_error']:.3e} "
            f"95% CI coverage={local_comparison['ci_coverage_count']}/48 "
            f"({local_comparison['ci_coverage_fraction']:.1%})",
            flush=True,
        )
    payload = {
        "metadata": {
            "benchmark": "4x4 qutrit Monte Carlo versus projected full-state F2",
            "local_dimension_monte_carlo": 3,
            "width": width,
            "length": length,
            "n_qubits": int(reference["N_QUBITS"]),
            "n_layers": n_layers,
            "noise_time_ns": float(args.noise_time),
            "piece_num": int(args.piece_num),
            "mc_sampling": args.mc_sampling,
            "fixed_proposal": args.fixed_proposal,
            "fixed_proposal_trace_mix": float(args.fixed_proposal_mix),
            "record_local_pauli": bool(args.record_local_pauli),
            "mc_state_dtype": args.mc_dtype,
            "same_circuit": True,
            "reference_file": str(args.reference),
            "pair_data_file": str(args.pair_data),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
            "torch_version": torch.__version__,
            "timing_scope": "online evolution only; channel compilation/decomposition excluded",
            "residual_manifest": str(Path(args.residual_manifest).resolve()),
            "residual_manifest_base_seed": int(manifest["base_seed"]),
            "residual_kernel_scheme": args.kernel_scheme,
            "residual_integration_steps": int(residual_steps),
            "residual_edges_per_layer": len(geometry["edges"]),
            "residual_compile_seconds_excluded_from_online_timing": float(
                residual_compile_seconds
            ),
            "common_background": "B_l = U_l K_res,I,l",
        },
        "channel_preparation": channel_diagnostics,
        "full_state_f2": direct,
        "cached_full_state_f2": cached,
        "qutrit_monte_carlo": mc,
        "runtime_seconds": {
            "full_state_f2": f2_seconds,
            "qutrit_monte_carlo": mc_seconds,
        },
        "comparison": {
            "absolute_error_f2_minus_mc": abs(direct_final - final_mean),
            "relative_error_f2_vs_mc": abs(direct_final - final_mean)
            / max(abs(final_mean), 1e-300),
            "mc_final_ci95_half_width": final_ci,
            "f2_inside_mc_95ci": bool(abs(direct_final - final_mean) <= final_ci),
            "direct_vs_cached_f2_absolute_difference": None,
        },
    }
    if local_pauli_payload is not None:
        payload["local_pauli"] = local_pauli_payload
    write_results(args.output_dir, payload, samples, local_pauli_samples)


def plot_results(data_path: Path, figure_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = json.loads(data_path.read_text(encoding="utf-8"))
    f2 = np.asarray(payload["full_state_f2"]["trajectory"], dtype=float)
    mc = payload["qutrit_monte_carlo"]
    mean = np.asarray(mc["trajectory_mean"], dtype=float)
    ci = np.asarray(mc["trajectory_ci95_half_width"], dtype=float)
    layers = np.arange(len(f2))
    checkpoints = mc["checkpoints"]
    checkpoint_n = np.asarray([row["trajectories"] for row in checkpoints], dtype=int)
    checkpoint_mean = np.asarray(
        [row["final_mean"] for row in checkpoints], dtype=float
    )
    checkpoint_ci = np.asarray(
        [row["final_ci95_half_width"] for row in checkpoints], dtype=float
    )
    runtime = payload["runtime_seconds"]

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.55), facecolor="white")
    colors = {"mc": "#2868a5", "f2": "#7b5bb6", "time": "#d9822b"}

    ax = axes[0]
    ax.fill_between(
        layers, mean - ci, mean + ci, color=colors["mc"], alpha=0.18, linewidth=0
    )
    ax.plot(
        layers,
        mean,
        color=colors["mc"],
        lw=1.9,
        marker="o",
        ms=3.0,
        label=rf"Qutrit fixed-proposal MC ($d=3$, $n={mc['trajectories']}$)",
    )
    ax.plot(
        layers,
        f2,
        color=colors["f2"],
        lw=2.0,
        label=r"Full-state $F_2$ (projected qubit)",
    )
    ax.set_xlabel("layer")
    ax.set_ylabel("fidelity")
    ax.set_title(r"$4\times4$ circuit: fidelity evolution", fontsize=10, pad=12)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.set_xlim(0, len(f2) - 1)

    ax = axes[1]
    ax.errorbar(
        checkpoint_n,
        checkpoint_mean,
        yerr=checkpoint_ci,
        color=colors["mc"],
        marker="o",
        ms=4.2,
        lw=1.5,
        capsize=3,
        label="Fixed-proposal qutrit MC (95% CI)",
    )
    ax.axhline(f2[-1], color=colors["f2"], lw=1.8, label=r"Full-state $F_2$")
    ax.set_xscale("log", base=2)
    ax.set_xticks(checkpoint_n)
    ax.set_xticklabels([f"{int(value):,}" for value in checkpoint_n])
    ax.set_xlabel("qutrit Monte Carlo trajectories")
    ax.set_ylabel("final-layer fidelity")
    ax.set_title("final-layer Monte Carlo convergence", fontsize=10, pad=12)
    ax.legend(frameon=False, fontsize=7.5, loc="best")

    ax = axes[2]
    labels = [r"Full-state $F_2$", f"Qutrit MC\n$n={mc['trajectories']}$"]
    values = [float(runtime["full_state_f2"]), float(runtime["qutrit_monte_carlo"])]
    bars = ax.bar(
        labels, values, color=[colors["f2"], colors["time"]], alpha=0.78, width=0.58
    )
    ax.set_yscale("log")
    ax.set_ylim(8.0, max(values) * 3.0)
    ax.set_ylabel("wall time (s, log scale)")
    ax.set_title(
        "same-device online runtime\n(channel compilation excluded)",
        fontsize=10,
        pad=7,
    )
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.08,
            f"{value:.1f} s" + (f"\n({value / 60.0:.1f} min)" if value >= 60 else ""),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    runtime_ratio = values[1] / values[0]
    ax.text(
        0.50,
        0.60,
        rf"$\mathbf{{{runtime_ratio:.0f}\times}}$",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=16,
        color="#333333",
    )
    ax.text(
        0.50,
        0.535,
        r"MC / $F_2$ online time",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.8,
        color="#555555",
    )
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
    for label, ax in zip(("a", "b", "c"), axes):
        position = ax.get_position()
        fig.text(
            position.x0 - 0.02,
            0.965,
            label,
            fontsize=12,
            fontweight="bold",
            va="bottom",
        )
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.18, top=0.87, wspace=0.34)

    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / "Fig2_qutrit_mc_4x4_vs_fullstate_f2.png"
    pdf = figure_dir / "Fig2_qutrit_mc_4x4_vs_fullstate_f2.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(png)
    print(pdf)


def self_test(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    channels = fig2_core.build_channels(
        2,
        1,
        1000,
        5000.0,
        device,
        torch.complex128,
    )
    dense = dense_two_qutrit_superoperator(channels.two_superoperator)
    kraus_cp = superoperator_to_kraus(dense, d=9, tol=1e-10)
    cp_projection = channel_roundtrip_residual(dense, kraus_cp)
    tp_before = tp_residual(kraus_cp)
    kraus = trace_preserving_kraus_repair(kraus_cp)
    cptp_adjustment = channel_roundtrip_residual(dense, kraus)
    tp_after = tp_residual(kraus)
    if cp_projection > 2e-3:
        raise AssertionError(
            f"two-qutrit CP projection residual is too large: {cp_projection}"
        )
    if cptp_adjustment > 3e-3:
        raise AssertionError(
            f"two-qutrit CPTP adjustment is too large: {cptp_adjustment}"
        )
    if tp_after > 5e-10:
        raise AssertionError(
            f"two-qutrit post-repair TP residual is too large: {tp_after}"
        )

    generator = torch.Generator(device=device)
    generator.manual_seed(1234)
    psi = torch.randn(9, dtype=torch.complex128, device=device)
    psi = psi / torch.linalg.norm(psi)
    target = torch.randn(9, dtype=torch.complex128, device=device)
    target = target / torch.linalg.norm(target)
    exact = torch.sum(
        torch.abs(torch.einsum("j,uji,i->u", target.conj(), kraus, psi)) ** 2
    ).real
    count = 20000
    states = psi.expand(count, -1).clone()
    sampled, probability_residual = sample_local_kraus(
        states,
        targets=(0, 1),
        kraus=kraus,
        effects=precompute_effects(kraus),
        n_qubits=2,
        local_dim=3,
        generator=generator,
    )
    values = torch.abs(torch.einsum("j,bj->b", target.conj(), sampled)) ** 2
    mean = values.mean()
    se = values.std(unbiased=True) / math.sqrt(count)
    error = torch.abs(mean - exact)
    if float(error) > 5.0 * float(se) + 2e-4:
        raise AssertionError(
            f"Monte Carlo estimator failed: mean={float(mean)} exact={float(exact)} se={float(se)}"
        )

    effects = precompute_effects(kraus)
    proposal = torch.diagonal(effects, dim1=-2, dim2=-1).real.sum(dim=-1)
    proposal = proposal.clamp_min(0.0) / proposal.sum()
    scaled = kraus / torch.sqrt(proposal.clamp_min(1e-300))[:, None, None]
    fixed_states = sample_fixed_proposal_kraus(
        psi.expand(count, -1).clone(),
        targets=(0, 1),
        proposal=proposal,
        scaled_kraus=scaled,
        n_qubits=2,
        local_dim=3,
        generator=generator,
    )
    fixed_values = torch.abs(torch.einsum("j,bj->b", target.conj(), fixed_states)) ** 2
    fixed_mean = fixed_values.mean()
    fixed_se = fixed_values.std(unbiased=True) / math.sqrt(count)
    if float(torch.abs(fixed_mean - exact)) > 5.0 * float(fixed_se) + 2e-4:
        raise AssertionError(
            "fixed-proposal Monte Carlo estimator failed: "
            f"mean={float(fixed_mean)} exact={float(exact)} se={float(fixed_se)}"
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "device": str(device),
                "two_qutrit_kraus_rank": int(kraus.shape[0]),
                "cp_projection_relative_residual": cp_projection,
                "cptp_relative_adjustment": cptp_adjustment,
                "tp_residual_before_repair": tp_before,
                "tp_residual_after_repair": tp_after,
                "probability_sum_residual": probability_residual,
                "mc_mean": float(mean),
                "exact": float(exact),
                "standard_error": float(se),
                "fixed_proposal_mc_mean": float(fixed_mean),
                "fixed_proposal_standard_error": float(fixed_se),
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--pair-data", type=Path, default=DEFAULT_PAIR_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--channel-cache", type=Path, default=DEFAULT_CHANNEL_CACHE)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_OUTPUT / "qutrit_mc_4x4_vs_fullstate_f2.json",
    )
    parser.add_argument("--piece-num", type=int, default=1000)
    parser.add_argument(
        "--residual-manifest", type=Path, default=DEFAULT_RESIDUAL_MANIFEST
    )
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
    parser.add_argument("--noise-time", type=float, default=5000.0)
    parser.add_argument("--kraus-tol", type=float, default=1e-10)
    parser.add_argument("--trajectories", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--mc-dtype",
        choices=("complex64", "complex128"),
        default="complex128",
        help="qutrit trajectory state precision; channel preparation and full-state F2 remain complex128",
    )
    parser.add_argument(
        "--mc-sampling",
        choices=("state-dependent", "fixed-proposal"),
        default="state-dependent",
        help="standard normalized trajectories or unbiased fixed-proposal Kraus sampling",
    )
    parser.add_argument(
        "--fixed-proposal",
        choices=("ideal", "trace"),
        default="ideal",
        help="proposal used by fixed-proposal Kraus Monte Carlo",
    )
    parser.add_argument(
        "--fixed-proposal-mix",
        type=float,
        default=1e-3,
        help="trace-proposal mixture that preserves support for ideal-guided sampling",
    )
    parser.add_argument("--checkpoints", default=None)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--rebuild-channels", action="store_true")
    parser.add_argument("--skip-direct-f2", action="store_true")
    parser.add_argument(
        "--record-local-pauli",
        action="store_true",
        help=(
            "at the end of each qutrit trajectory, record all 16 local X/Y/Z "
            "positive-effect probabilities for the full-state comparison"
        ),
    )
    parser.add_argument(
        "--local-pauli-reference",
        type=Path,
        default=DEFAULT_PAULI_REFERENCE,
        help="16-site full-state C2 JSON used for the terminal local-Pauli comparison",
    )
    parser.add_argument("--progress", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.self_test:
        self_test(args)
    if args.run:
        run_production(args)
    if args.plot:
        plot_results(args.data_path, args.figure_dir)
    if not (args.self_test or args.run or args.plot):
        parser.print_help()


if __name__ == "__main__":
    main()
