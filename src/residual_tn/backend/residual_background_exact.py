"""Exact small-system helpers for the shared coherent-residual background.

The response locations remain the local environmental maps.  Residual Magnus
kernels are compiled once per circuit and are then applied to the zero-source
trajectory and to every one-/two-source branch.  The layer convention is

    B_l = U_l K_res,I,l,

so the residual kernels act first on the incoming state and the disjoint
intended gates act second.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from residual_tn.backend.kraus import compute_dressed_kraus
from residual_tn.backend.residual_magnus import (
    CompiledResidualOperator,
    ResidualMagnusLibrary,
    apply_operator_to_state,
)


def load_residual_geometry(
    manifest_path: str | Path,
    width: int,
    length: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one exact geometry record from the reproducible manifest."""
    path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for geometry in manifest.get("geometries", []):
        if int(geometry.get("width", -1)) == int(width) and int(
            geometry.get("length", -1)
        ) == int(length):
            return manifest, geometry
    raise ValueError(f"no {width}x{length} residual geometry in {path}")


def control_layers_from_pattern(
    pattern: Sequence[Any], n_qubits: int
) -> list[list[dict[str, Any]]]:
    """Translate the random-circuit pattern into residual control blocks."""
    layers: list[list[dict[str, Any]]] = []
    for layer_index, layer in enumerate(pattern):
        if layer_index % 2 == 0:
            layers.append(
                [
                    {"qubits": (site,), "mode": int(layer[site])}
                    for site in range(n_qubits)
                ]
            )
        else:
            layers.append([{"qubits": (int(pair[0]), int(pair[1]))} for pair in layer])
    return layers


def compile_pattern_residual(
    library: ResidualMagnusLibrary,
    pattern: Sequence[Any],
    residual_edges: Iterable[dict[str, Any]],
    *,
    scheme: str,
) -> list[list[CompiledResidualOperator]]:
    """Compile one circuit while reusing the library's pulse histories."""
    controls = control_layers_from_pattern(pattern, library.n_qubits)
    return library.compile_circuit(controls, residual_edges, scheme=scheme)


def apply_background_layer(
    states: torch.Tensor,
    kernels: Sequence[CompiledResidualOperator],
    gates: Sequence[dict[str, Any]],
    *,
    n_qubits: int,
    adjoint: bool = False,
) -> torch.Tensor:
    """Apply one common qubit background layer, or its adjoint."""
    result = states
    if not adjoint:
        for kernel in kernels:
            result = apply_operator_to_state(
                result,
                kernel.operator,
                kernel.support,
                n_qubits=n_qubits,
                local_dim=2,
            )
        for gate in gates:
            result = apply_operator_to_state(
                result,
                gate["ideal_unitary"],
                gate["qubits"],
                n_qubits=n_qubits,
                local_dim=2,
            )
        return result

    for gate in reversed(gates):
        result = apply_operator_to_state(
            result,
            gate["ideal_unitary"].conj().transpose(-1, -2),
            gate["qubits"],
            n_qubits=n_qubits,
            local_dim=2,
        )
    for kernel in reversed(kernels):
        result = apply_operator_to_state(
            result,
            kernel.operator.conj().transpose(-1, -2),
            kernel.support,
            n_qubits=n_qubits,
            local_dim=2,
        )
    return result


def propagate_background_layers(
    states: torch.Tensor,
    layers: Sequence[Sequence[dict[str, Any]]],
    compiled_layers: Sequence[Sequence[CompiledResidualOperator]],
    *,
    n_qubits: int,
    start_layer: int,
    stop_layer: int | None = None,
) -> torch.Tensor:
    stop = len(layers) if stop_layer is None else int(stop_layer)
    result = states
    for layer in range(int(start_layer), stop):
        result = apply_background_layer(
            result,
            compiled_layers[layer],
            layers[layer],
            n_qubits=n_qubits,
        )
    return result


def _apply_matrix_to_axes(
    tensor: torch.Tensor,
    matrix: torch.Tensor,
    axes: Sequence[int],
    *,
    local_dim: int,
) -> torch.Tensor:
    axes_tuple = tuple(int(axis) for axis in axes)
    if len(set(axes_tuple)) != len(axes_tuple):
        raise ValueError("operator axes contain duplicates")
    rest = tuple(axis for axis in range(tensor.ndim) if axis not in axes_tuple)
    order = axes_tuple + rest
    inverse = [order.index(axis) for axis in range(tensor.ndim)]
    local_size = local_dim ** len(axes_tuple)
    permuted = tensor.permute(order)
    tail_shape = permuted.shape[len(axes_tuple) :]
    transformed = matrix.reshape(local_size, local_size) @ permuted.reshape(
        local_size, -1
    )
    return transformed.reshape(*([local_dim] * len(axes_tuple)), *tail_shape).permute(
        inverse
    )


def apply_residual_layer_to_liouville(
    rho_local: torch.Tensor,
    kernels: Sequence[CompiledResidualOperator],
    *,
    local_dim: int = 3,
) -> torch.Tensor:
    """Apply K rho K^dagger without forming a (d^2)^k superoperator."""
    n_sites = rho_local.ndim
    tensor = rho_local.reshape(*([local_dim, local_dim] * n_sites))
    for kernel in kernels:
        ket_axes = tuple(2 * site for site in kernel.support)
        bra_axes = tuple(2 * site + 1 for site in kernel.support)
        tensor = _apply_matrix_to_axes(
            tensor,
            kernel.operator,
            ket_axes,
            local_dim=local_dim,
        )
        tensor = _apply_matrix_to_axes(
            tensor,
            kernel.operator.conj(),
            bra_axes,
            local_dim=local_dim,
        )
    return tensor.reshape(*([local_dim * local_dim] * n_sites))


def _apply_operator_per_state(
    states: torch.Tensor,
    operators: torch.Tensor,
    support: Sequence[int],
    *,
    n_qubits: int,
) -> torch.Tensor:
    """Apply one local qubit operator to each state in a batch."""
    if states.ndim != 2:
        raise ValueError("states must have shape (batch, 2**n_qubits)")
    sites = tuple(int(site) for site in support)
    batch = int(states.shape[0])
    local_size = 2 ** len(sites)
    if tuple(operators.shape) != (batch, local_size, local_size):
        raise ValueError("operator bank does not match the state batch")
    rest = tuple(site for site in range(n_qubits) if site not in sites)
    order = sites + rest
    inverse = [order.index(site) for site in range(n_qubits)]
    x = states.reshape(batch, *([2] * n_qubits)).permute(
        0, *(site + 1 for site in order)
    )
    y = torch.einsum("boi,bik->bok", operators, x.reshape(batch, local_size, -1))
    y = y.reshape(batch, *([2] * n_qubits)).permute(0, *(site + 1 for site in inverse))
    return y.reshape(batch, 2**n_qubits)


def apply_mode_bank(
    states: torch.Tensor,
    modes: torch.Tensor,
    support: Sequence[int],
    *,
    n_qubits: int,
) -> torch.Tensor:
    """Apply every local mode to every input state."""
    if states.ndim == 1:
        states = states.unsqueeze(0)
    batch = int(states.shape[0])
    mode_count = int(modes.shape[0])
    local_size = 2 ** len(tuple(support))
    expanded_states = (
        states[:, None, :]
        .expand(batch, mode_count, states.shape[-1])
        .reshape(batch * mode_count, states.shape[-1])
    )
    expanded_modes = (
        modes.reshape(mode_count, local_size, local_size)[None, ...]
        .expand(batch, mode_count, local_size, local_size)
        .reshape(batch * mode_count, local_size, local_size)
    )
    return _apply_operator_per_state(
        expanded_states,
        expanded_modes,
        support,
        n_qubits=n_qubits,
    )


def _local_transition_overlap(
    effect_state: torch.Tensor,
    branches: torch.Tensor,
    support: Sequence[int],
    *,
    n_qubits: int,
) -> torch.Tensor:
    """Return T[b,o,i]=sum_env conj(effect[o,env])*branch[b,i,env]."""
    if branches.ndim == 1:
        branches = branches.unsqueeze(0)
    sites = tuple(int(site) for site in support)
    local_size = 2 ** len(sites)
    batch = int(branches.shape[0])
    effect = effect_state.reshape(*([2] * n_qubits))
    branch = branches.reshape(batch, *([2] * n_qubits))
    effect = torch.movedim(effect, sites, tuple(range(len(sites))))
    branch = torch.movedim(
        branch,
        tuple(site + 1 for site in sites),
        tuple(range(1, len(sites) + 1)),
    )
    return torch.einsum(
        "ok,bik->boi",
        effect.reshape(local_size, -1).conj(),
        branch.reshape(batch, local_size, -1),
    )


def _background_and_target_histories(
    layers: Sequence[Sequence[dict[str, Any]]],
    compiled_layers: Sequence[Sequence[CompiledResidualOperator]],
    *,
    n_qubits: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    background = torch.zeros(2**n_qubits, device=device, dtype=dtype)
    target = torch.zeros_like(background)
    background[0] = 1.0
    target[0] = 1.0
    background_history: list[torch.Tensor] = []
    target_history: list[torch.Tensor] = []
    for layer, gates in enumerate(layers):
        background = apply_background_layer(
            background,
            compiled_layers[layer],
            gates,
            n_qubits=n_qubits,
        )
        for gate in gates:
            target = apply_operator_to_state(
                target,
                gate["ideal_unitary"],
                gate["qubits"],
                n_qubits=n_qubits,
                local_dim=2,
            )
        background_history.append(background.clone())
        target_history.append(target.clone())
    return background_history, target_history


def connected_fidelity_at_depth(
    ctx: Any,
    compiled_layers: Sequence[Sequence[CompiledResidualOperator]],
    *,
    depth: int | None = None,
    store_responses: bool = False,
) -> dict[str, Any]:
    """Exact projected-qubit F0 and all one-/two-location responses."""
    total_depth = len(ctx.layers) if depth is None else int(depth)
    if not 1 <= total_depth <= len(ctx.layers):
        raise ValueError("depth lies outside the circuit")
    layers = list(ctx.layers[:total_depth])
    kernels = list(compiled_layers[:total_depth])
    n_qubits = int(ctx.n_qubits)
    background_history, target_history = _background_and_target_histories(
        layers,
        kernels,
        n_qubits=n_qubits,
        device=ctx.device,
        dtype=ctx.dtype,
    )

    effects: list[torch.Tensor | None] = [None] * total_depth
    effect = target_history[-1]
    for layer in reversed(range(total_depth)):
        effects[layer] = effect
        if layer > 0:
            effect = apply_background_layer(
                effect,
                kernels[layer],
                layers[layer],
                n_qubits=n_qubits,
                adjoint=True,
            )

    f0_tensor = torch.vdot(target_history[-1], background_history[-1]).abs().square()
    f0 = float(f0_tensor.real.detach().cpu())
    if not math.isfinite(f0) or f0 <= 0.0:
        raise FloatingPointError(f"invalid zero-insertion fidelity {f0}")

    dressed: dict[int, torch.Tensor] = {}
    for layer in layers:
        for gate in layer:
            gate_id = int(gate["gate_idx"])
            local_size = 2 ** len(tuple(gate["qubits"]))
            dressed[gate_id] = compute_dressed_kraus(
                gate["kraus_ops"], gate["ideal_unitary"]
            ).reshape(-1, local_size, local_size)

    active: list[dict[str, Any]] = []
    one_fidelity: dict[int, float] = {}
    pair_fidelity: dict[tuple[int, int], float] = {}
    pair_delta_sum = 0.0
    pair_count = 0

    for layer_index, gates in enumerate(layers):
        if active:
            counts = [int(group["branches"].shape[0]) for group in active]
            joined = torch.cat([group["branches"] for group in active], dim=0)
            joined = apply_background_layer(
                joined,
                kernels[layer_index],
                gates,
                n_qubits=n_qubits,
            )
            offset = 0
            for group, count in zip(active, counts):
                group["branches"] = joined[offset : offset + count]
                offset += count

        current: list[dict[str, Any]] = []
        effect_state = effects[layer_index]
        assert effect_state is not None
        for gate in gates:
            gate_id = int(gate["gate_idx"])
            support = tuple(int(site) for site in gate["qubits"])
            modes = dressed[gate_id]

            branches = apply_mode_bank(
                background_history[layer_index],
                modes,
                support,
                n_qubits=n_qubits,
            )
            amplitudes_one = torch.mv(branches, effect_state.conj())
            f_b = float(
                amplitudes_one.abs().square().sum(dtype=torch.float64).detach().cpu()
            )
            if not math.isfinite(f_b) or f_b <= 0.0:
                raise FloatingPointError(
                    f"invalid one-location fidelity at gate {gate_id}: {f_b}"
                )
            one_fidelity[gate_id] = f_b

            eligible = active + current
            if eligible:
                counts = [int(group["branches"].shape[0]) for group in eligible]
                joined = torch.cat([group["branches"] for group in eligible], dim=0)
                transition = _local_transition_overlap(
                    effect_state,
                    joined,
                    support,
                    n_qubits=n_qubits,
                )
                amplitudes = torch.einsum("voi,boi->bv", modes, transition)
                offset = 0
                for group, count in zip(eligible, counts):
                    f_ab = float(
                        amplitudes[offset : offset + count]
                        .abs()
                        .square()
                        .sum(dtype=torch.float64)
                        .detach()
                        .cpu()
                    )
                    f_a = float(group["one_fidelity"])
                    pair_delta_sum += f_ab * f0 / (f_a * f_b) - 1.0
                    pair_count += 1
                    if store_responses:
                        pair_fidelity[(int(group["gate_idx"]), gate_id)] = f_ab
                    offset += count
            current.append(
                {
                    "gate_idx": gate_id,
                    "branches": branches,
                    "one_fidelity": f_b,
                }
            )
        active.extend(current)

    c1 = sum(math.log(value / f0) for value in one_fidelity.values())
    first = f0 * math.exp(c1)
    second = f0 * math.exp(c1 + pair_delta_sum)
    result = {
        "zero_insertion_fidelity": f0,
        "fidelity_first_order": first,
        "fidelity_second_order": second,
        "log_one_location_sum": c1,
        "pair_delta_sum": pair_delta_sum,
        "pair_count": pair_count,
        "gate_count": len(one_fidelity),
    }
    if store_responses:
        result["one_location_fidelity"] = one_fidelity
        result["two_location_fidelity"] = pair_fidelity
    return result


def _all_prefix_transition_overlap(
    effect_states: torch.Tensor,
    branches: torch.Tensor,
    support: Sequence[int],
    *,
    n_qubits: int,
) -> torch.Tensor:
    """T[t,b,o,i] for terminal-specific bras and shared source kets."""
    sites = tuple(int(q) for q in support)
    local_size = 2 ** len(sites)
    targets, batch = effect_states.shape[0], branches.shape[0]
    effect = effect_states.reshape(targets, *([2] * n_qubits))
    branch = branches.reshape(batch, *([2] * n_qubits))
    axes = tuple(q + 1 for q in sites)
    front = tuple(range(1, len(sites) + 1))
    effect = torch.movedim(effect, axes, front).reshape(targets, local_size, -1)
    branch = torch.movedim(branch, axes, front).reshape(batch, local_size, -1)
    return torch.einsum("tok,bik->tboi", effect.conj(), branch)


def connected_fidelity_trajectory(
    ctx: Any,
    compiled_layers: Sequence[Sequence[CompiledResidualOperator]],
) -> dict[str, Any]:
    """Read all terminal prefixes during one source-bank forward sweep.

    Only source kets are shared. Every prefix keeps its own intended-control
    target, backward-propagated effect, F0, single-source responses and pairs.
    All response pairs are retained, including pairs within a layer. The
    at-depth reader remains available for individual depths and raw responses.

    The source bank is propagated once per layer, independent of how many
    terminal prefixes are requested. Pair reductions stay on the device until
    the completed curves are returned. State memory scales with the number of
    source modes plus the quadratic-size history of terminal-specific effects.
    """
    layers = ctx.layers
    depth = len(layers)
    if not depth:
        raise ValueError("the circuit must contain at least one layer")
    if len(compiled_layers) != depth:
        raise ValueError("one residual-kernel sequence is required per layer")
    n_qubits, device = int(ctx.n_qubits), ctx.device
    background, targets = _background_and_target_histories(
        layers,
        compiled_layers,
        n_qubits=n_qubits,
        device=device,
        dtype=ctx.dtype,
    )
    f0 = torch.stack(
        [torch.vdot(t, b).abs().square() for t, b in zip(targets, background)]
    ).to(torch.float64)
    if not bool((torch.isfinite(f0) & (f0 > 0)).all()):
        raise FloatingPointError("invalid zero-insertion fidelity")

    # effects[l][t-l] is the target at terminal layer t propagated back to l.
    effects = [None] * depth
    effects[-1] = targets[-1].unsqueeze(0)
    for layer in range(depth - 2, -1, -1):
        pulled = apply_background_layer(
            effects[layer + 1],
            compiled_layers[layer + 1],
            layers[layer + 1],
            n_qubits=n_qubits,
            adjoint=True,
        )
        effects[layer] = torch.cat((targets[layer].unsqueeze(0), pulled), dim=0)

    descriptors, counts = [], []
    for gates in layers:
        row = []
        for gate in gates:
            support = tuple(int(q) for q in gate["qubits"])
            size = 2 ** len(support)
            modes = compute_dressed_kraus(
                gate["kraus_ops"], gate["ideal_unitary"]
            ).reshape(-1, size, size)
            row.append((support, modes))
            counts.append(int(modes.shape[0]))
        descriptors.append(row)
    gate_count = len(counts)
    bank = torch.empty((sum(counts), 2**n_qubits), dtype=ctx.dtype, device=device)
    # Group ragged source-mode ranks by gate without a scalar CPU read per pair.
    mode_gate = torch.repeat_interleave(
        torch.arange(gate_count, device=device),
        torch.tensor(counts, device=device, dtype=torch.long),
    )
    # Only prefixes at or after a source birth access that source column.
    one_fidelity = torch.empty((depth, gate_count), device=device, dtype=torch.float64)
    c1 = torch.zeros(depth, device=device, dtype=torch.float64)
    pair_delta = torch.zeros_like(c1)
    all_valid = torch.ones((), device=device, dtype=torch.bool)
    used = gate_index = 0
    for layer, gates in enumerate(layers):
        if used:
            propagated = apply_background_layer(
                bank[:used],
                compiled_layers[layer],
                gates,
                n_qubits=n_qubits,
            )
            bank[:used].copy_(propagated)
            del propagated
        for support, modes in descriptors[layer]:
            branches = apply_mode_bank(
                background[layer], modes, support, n_qubits=n_qubits
            )
            f_b = (
                (effects[layer].conj() @ branches.T)
                .abs()
                .square()
                .sum(dim=1, dtype=torch.float64)
            )
            one_fidelity[layer:, gate_index] = f_b
            all_valid &= (torch.isfinite(f_b) & (f_b > 0)).all()
            c1[layer:] += torch.log(f_b / f0[layer:])
            if used:
                transition = _all_prefix_transition_overlap(
                    effects[layer],
                    bank[:used],
                    support,
                    n_qubits=n_qubits,
                )
                amplitudes = torch.einsum("voi,tboi->tbv", modes, transition)
                per_source_mode = (
                    amplitudes.abs().square().sum(dim=2, dtype=torch.float64)
                )
                f_ab = torch.zeros(
                    (depth - layer, gate_index), device=device, dtype=torch.float64
                )
                f_ab.index_add_(1, mode_gate[:used], per_source_mode)
                pair_delta[layer:] += (
                    f_ab
                    * f0[layer:, None]
                    / (one_fidelity[layer:, :gate_index] * f_b[:, None])
                    - 1
                ).sum(dim=1)
                del transition, amplitudes, per_source_mode, f_ab
            count = int(branches.shape[0])
            bank[used : used + count].copy_(branches)
            used += count
            gate_index += 1
            del branches
    if not bool(all_valid):
        raise FloatingPointError(
            "invalid one-location fidelity for at least one prefix"
        )
    first = f0 * torch.exp(c1)
    second = f0 * torch.exp(c1 + pair_delta)
    packed = torch.stack((f0, first, second)).detach().cpu().tolist()
    return {
        "zero_insertion_fidelity": [1.0, *packed[0]],
        "fidelity_first_order": [1.0, *packed[1]],
        "fidelity_second_order": [1.0, *packed[2]],
        "final_pair_count": gate_count * (gate_count - 1) // 2,
    }
