#!/usr/bin/env python
"""Residual-background exact 4x4 connected-order ensemble audit.

This is the coherent-residual counterpart of
``compute_4x4_connected_ensemble.py``.  It deliberately reuses that script's
spatial C2 audit, streaming distinct-layer C3 implementation, checkpointing,
and output format.  Only the common propagation and response normalization
are replaced:

    B_l = U_l K_res,I,l,
    M_a = F_a/F_0,
    M_ab = F_ab/F_0,
    M_abc = F_abc/F_0.

Residual Magnus kernels are compiled once for each random circuit and reused
by the zero-insertion trajectory and every one-, two-, and three-location
branch.  Physical residual supports are remapped to the full-state tensor axes
stored in the 0522 reference before application.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Sequence

import torch


from residual_tn.paths import CONFIGS, INPUTS, RESULTS

PACKAGE_ROOT = RESULTS
from residual_tn.backend import context as context  # noqa: F401,E402
from residual_tn.backend import exact_small as exact_small  # noqa: F401,E402
from residual_tn.experiments import cumulant_exact as natcomm_4x4_cumulant_audit  # noqa: F401,E402
from residual_tn.experiments import connected_ensemble as base  # noqa: E402
from residual_tn.backend import residual_background_exact as residual_bg  # noqa: E402
from residual_tn.backend.residual_magnus import (  # noqa: E402
    CompiledResidualOperator,
    ResidualMagnusLibrary,
)


_ORIGINAL_BUILD_CONTEXT = base.build_context
_ORIGINAL_RESULT_ROW = base.result_row
_ORIGINAL_WRITE_AGGREGATE_OUTPUTS = base.write_aggregate_outputs
_ORIGINAL_READOUT = base.readout_event_sum_over_modes

_CONFIG: argparse.Namespace | None = None
_LIBRARY: ResidualMagnusLibrary | None = None
_GEOMETRY: dict[str, Any] | None = None
_MANIFEST: dict[str, Any] | None = None
_CURRENT_CTX: Any | None = None
_CURRENT_KERNELS: list[list[CompiledResidualOperator]] | None = None
_CURRENT_BACKGROUND_HISTORY: list[torch.Tensor] | None = None
_CURRENT_EFFECTS: list[torch.Tensor] | None = None
_CURRENT_F0: float | None = None
_CURRENT_COMPILE_SECONDS: float = 0.0


def _axis(reference: dict[str, Any], physical: int) -> int:
    mapping = reference["phys_to_1d"]
    if isinstance(mapping, dict):
        if physical in mapping:
            return int(mapping[physical])
        return int(mapping[str(physical)])
    return int(mapping[physical])


def _remap_kernels(
    compiled: Sequence[Sequence[CompiledResidualOperator]],
    reference: dict[str, Any],
) -> list[list[CompiledResidualOperator]]:
    """Preserve operator tensor order while replacing physical sites by axes."""
    return [
        [
            replace(
                kernel,
                support=tuple(_axis(reference, site) for site in kernel.support),
            )
            for kernel in layer
        ]
        for layer in compiled
    ]


def build_context(
    reference: dict[str, Any], pattern: Sequence[Any], device: torch.device
):
    global _LIBRARY, _CURRENT_CTX, _CURRENT_KERNELS, _CURRENT_COMPILE_SECONDS
    if _CONFIG is None or _GEOMETRY is None:
        raise RuntimeError("residual configuration was not initialized")
    ctx, gate_meta = _ORIGINAL_BUILD_CONTEXT(reference, pattern, device)
    if _LIBRARY is None:
        _LIBRARY = ResidualMagnusLibrary(
            int(reference["N_QUBITS"]),
            integration_steps=int(_CONFIG.residual_integration_steps),
            dtype=torch.complex128,
            device=device,
            projected_qubits=True,
        )
    started = time.time()
    physical = residual_bg.compile_pattern_residual(
        _LIBRARY,
        pattern,
        _GEOMETRY["edges"],
        scheme=_CONFIG.kernel_scheme,
    )
    _CURRENT_KERNELS = _remap_kernels(physical, reference)
    _CURRENT_CTX = ctx
    _CURRENT_COMPILE_SECONDS = time.time() - started
    print(
        "[residual] compiled common background: layers=%d edges/layer=%d "
        "scheme=%s seconds=%.2f"
        % (
            len(_CURRENT_KERNELS),
            len(_GEOMETRY["edges"]),
            _CONFIG.kernel_scheme,
            _CURRENT_COMPILE_SECONDS,
        ),
        flush=True,
    )
    return ctx, gate_meta


def _histories() -> tuple[list[torch.Tensor], list[torch.Tensor], float]:
    global _CURRENT_BACKGROUND_HISTORY, _CURRENT_EFFECTS, _CURRENT_F0
    if _CURRENT_CTX is None or _CURRENT_KERNELS is None:
        raise RuntimeError("no active residual-background circuit")
    background, target = residual_bg._background_and_target_histories(
        _CURRENT_CTX.layers,
        _CURRENT_KERNELS,
        n_qubits=int(_CURRENT_CTX.n_qubits),
        device=_CURRENT_CTX.device,
        dtype=_CURRENT_CTX.dtype,
    )
    effects: list[torch.Tensor | None] = [None] * len(_CURRENT_CTX.layers)
    effect = target[-1]
    for layer in reversed(range(len(_CURRENT_CTX.layers))):
        effects[layer] = effect
        if layer > 0:
            effect = residual_bg.apply_background_layer(
                effect,
                _CURRENT_KERNELS[layer],
                _CURRENT_CTX.layers[layer],
                n_qubits=int(_CURRENT_CTX.n_qubits),
                adjoint=True,
            )
    f0 = float(torch.vdot(target[-1], background[-1]).abs().square().real.cpu())
    if not math.isfinite(f0) or f0 <= 0.0:
        raise FloatingPointError(f"invalid residual zero-insertion fidelity {f0}")
    _CURRENT_BACKGROUND_HISTORY = background
    _CURRENT_EFFECTS = [value for value in effects if value is not None]
    _CURRENT_F0 = f0
    return background, _CURRENT_EFFECTS, f0


def ideal_state_history(ctx: Any) -> list[torch.Tensor]:
    """Compatibility hook: return the residual background prefix states."""
    background, _, _ = _histories()
    initial = torch.zeros(2 ** int(ctx.n_qubits), dtype=ctx.dtype, device=ctx.device)
    initial[0] = 1.0
    return [initial.unsqueeze(0)] + [state.unsqueeze(0) for state in background]


def exact_pair_payload(ctx: Any, gate_meta: dict[int, tuple[int, Any]]):
    """Return normalized M_a and M_ab while retaining the physical F0 factor."""
    if _CURRENT_KERNELS is None:
        raise RuntimeError("residual kernels are unavailable")
    payload = residual_bg.connected_fidelity_at_depth(
        ctx,
        _CURRENT_KERNELS,
        store_responses=True,
    )
    f0 = float(payload["zero_insertion_fidelity"])
    local: dict[tuple[int, Any], float] = {}
    for gate_idx, value in payload["one_location_fidelity"].items():
        layer, gate_id = gate_meta[int(gate_idx)]
        local[(int(layer), base.canonical_gate_id(gate_id))] = float(value) / f0
    pairs: dict[base.PairKey, float] = {}
    for (left_idx, right_idx), value in payload["two_location_fidelity"].items():
        left_layer, left_gate = gate_meta[int(left_idx)]
        right_layer, right_gate = gate_meta[int(right_idx)]
        pairs[
            (
                int(left_layer),
                base.canonical_gate_id(left_gate),
                int(right_layer),
                base.canonical_gate_id(right_gate),
            )
        ] = float(value) / f0
    exact_result = SimpleNamespace(
        log_fidelity_first_order=float(payload["log_one_location_sum"]),
        fidelity_first_order=float(payload["fidelity_first_order"]),
        zero_insertion_fidelity=f0,
        pair_count=int(payload["pair_count"]),
    )
    return exact_result, local, pairs


def propagate_background(
    states: torch.Tensor,
    start_layer: int,
    end_layer: int,
    reference: dict[str, Any],
    state_chunk: int,
) -> torch.Tensor:
    """Drop-in replacement for the legacy ideal-only C3 propagation."""
    if start_layer > end_layer:
        return states
    if _CURRENT_CTX is None or _CURRENT_KERNELS is None:
        raise RuntimeError("no active residual-background circuit")
    if int(states.shape[0]) > int(state_chunk):
        result = torch.empty_like(states)
        for offset in range(0, int(states.shape[0]), int(state_chunk)):
            stop = min(offset + int(state_chunk), int(states.shape[0]))
            result[offset:stop].copy_(
                propagate_background(
                    states[offset:stop], start_layer, end_layer, reference, state_chunk
                )
            )
        return result

    result = states
    for layer in range(int(start_layer), int(end_layer) + 1):
        result = residual_bg.apply_background_layer(
            result,
            _CURRENT_KERNELS[layer],
            _CURRENT_CTX.layers[layer],
            n_qubits=int(_CURRENT_CTX.n_qubits),
        )
    return result


def readout_normalized(
    ignored_state: torch.Tensor,
    states: torch.Tensor,
    event: Any,
    n_qubits: int,
) -> torch.Tensor:
    """Read F_abc with the residual-dressed backward effect and divide by F0."""
    if _CURRENT_EFFECTS is None or _CURRENT_F0 is None:
        raise RuntimeError("residual effects are unavailable")
    return (
        _ORIGINAL_READOUT(_CURRENT_EFFECTS[int(event.layer)], states, event, n_qubits)
        / _CURRENT_F0
    )


def result_row(*args: Any, **kwargs: Any) -> dict[str, Any]:
    row = _ORIGINAL_RESULT_ROW(*args, **kwargs)
    exact_result = args[2] if len(args) >= 3 else kwargs["exact_result"]
    f0 = float(exact_result.zero_insertion_fidelity)
    row["zero_insertion_fidelity_F0"] = f0
    row["fidelity_F1"] = f0 * math.exp(float(row["C1_signed_log"]))
    row["fidelity_F2_log_cumulant"] = f0 * math.exp(
        float(row["C1_signed_log"]) + float(row["C2_signed_log"])
    )
    if "C3_distinct_signed_log" in row:
        row["fidelity_F3_distinct_log_cumulant"] = f0 * math.exp(
            float(row["C1_signed_log"])
            + float(row["C2_signed_log"])
            + float(row["C3_distinct_signed_log"])
        )
    row["runtime_residual_compile_sec"] = float(_CURRENT_COMPILE_SECONDS)
    return row


def write_aggregate_outputs(
    outdir: Path,
    metadata: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    started: float,
) -> None:
    if _CONFIG is None or _GEOMETRY is None or _MANIFEST is None:
        raise RuntimeError("residual metadata is unavailable")
    enriched = dict(metadata)
    enriched.update(
        {
            "common_background": "B_l = U_l K_res,I,l",
            "response_normalization": {
                "M_a": "F_a/F_0",
                "M_ab": "F_ab/F_0",
                "M_abc": "F_abc/F_0",
                "kappa2": "log(F_ab F_0/(F_a F_b))",
            },
            "residual_kernel_scheme": _CONFIG.kernel_scheme,
            "residual_integration_steps": int(_CONFIG.residual_integration_steps),
            "residual_manifest": str(Path(_CONFIG.residual_manifest).resolve()),
            "residual_manifest_base_seed": int(_MANIFEST["base_seed"]),
            "residual_edges_per_layer": len(_GEOMETRY["edges"]),
            "residual_compiled_once_per_circuit_and_reused": True,
        }
    )
    _ORIGINAL_WRITE_AGGREGATE_OUTPUTS(outdir, enriched, rows, started)


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser_original()
    parser.add_argument(
        "--residual-manifest",
        type=Path,
        default=CONFIGS / "residual_manifest.json",
    )
    parser.add_argument(
        "--kernel-scheme",
        choices=("dyson-1", "magnus-1"),
        default="magnus-1",
    )
    parser.add_argument("--residual-integration-steps", type=int, default=1000)
    return parser


def main() -> None:
    global _CONFIG, _GEOMETRY, _MANIFEST
    # Preserve the original parser before installing the compatibility hooks.
    _CONFIG = build_parser().parse_args()
    _MANIFEST, _GEOMETRY = residual_bg.load_residual_geometry(
        _CONFIG.residual_manifest, 4, 4
    )
    print(
        "[residual-setup] scheme=%s steps=%d edges/layer=%d"
        % (
            _CONFIG.kernel_scheme,
            _CONFIG.residual_integration_steps,
            len(_GEOMETRY["edges"]),
        ),
        flush=True,
    )
    base.build_parser = build_parser
    base.build_context = build_context
    base.ideal_state_history = ideal_state_history
    base.exact_pair_payload = exact_pair_payload
    base.propagate_ideal = propagate_background
    base.readout_event_sum_over_modes = readout_normalized
    base.result_row = result_row
    base.write_aggregate_outputs = write_aggregate_outputs
    base.main()


# The wrapper needs an unmodified parser factory after monkeypatching.
base.build_parser_original = base.build_parser


if __name__ == "__main__":
    main()
