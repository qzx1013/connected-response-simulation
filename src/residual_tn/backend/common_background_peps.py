#!/usr/bin/env python
"""Build reusable PEPS checkpoints for the coherent common background.

For each circuit layer, the pulse-resolved library integrates
``U_0(t)^dagger H_res U_0(t)`` once for every residual edge/control signature.
It therefore produces a two-site operator in a one-qubit layer and a two-,
three-, or four-site operator in a disjoint two-qubit layer.  The incoming
PEPS is acted on first by this interaction-picture residual operator and then
by the intended layer, realizing ``U_0 K_res,I``.  Nonlocal support is routed
only inside the PEPS backend; full-state backends can act on its tensor axes
directly.  The resulting layer checkpoints are reusable seeds for all
environmental-response branches, and the pulse integration is never repeated
per branch.

This script deliberately constructs only the coherent background.  It does
not insert a noisy Kraus map and it does not build a response branch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Sequence

import torch


from residual_tn.paths import CONFIGS

from residual_tn.backend.physics_0522 import build_generated_0522_context  # noqa: E402
from residual_tn.backend import residual_magnus as residual  # noqa: E402
from residual_tn.backend import single_site_bp_streamed_optimized as bp  # noqa: E402


DEFAULT_MANIFEST = CONFIGS / "residual_manifest.json"
DEFAULT_LAYER_DURATION_NS = residual.LAYER_DURATION_NS
KERNEL_SCHEMES = residual.KERNEL_SCHEMES


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_residual_geometry(
    manifest_path: str | Path,
    width: int,
    length: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate one geometry from the residual manifest."""
    path = Path(manifest_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    matches = [
        geometry
        for geometry in manifest.get("geometries", [])
        if int(geometry.get("width", -1)) == int(width)
        and int(geometry.get("length", -1)) == int(length)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {width}x{length} geometry in {path}, "
            f"found {len(matches)}"
        )
    geometry = matches[0]
    if int(geometry.get("n_qubits", -1)) != int(width) * int(length):
        raise ValueError("manifest geometry has an inconsistent qubit count")
    edges = geometry.get("edges")
    if not isinstance(edges, list) or not edges:
        raise ValueError("manifest geometry contains no residual edges")
    if int(geometry.get("n_residual_edges", -1)) != len(edges):
        raise ValueError("manifest residual-edge count does not match its list")
    return manifest, geometry


def swap_operator(*, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return the two-qubit SWAP operator in the |00>,|01>,|10>,|11> basis."""
    return torch.tensor(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=dtype,
        device=device,
    )


def _operator_to_left_canonical_mpo(
    operator: torch.Tensor,
    site_count: int,
) -> list[torch.Tensor]:
    """Historical gauge, retained only for controlled compression comparisons."""
    expected = 2**site_count
    if tuple(operator.shape) != (expected, expected):
        raise ValueError("multi-site operator shape does not match its support")
    tensor = operator.reshape(*([2] * site_count), *([2] * site_count))
    interleaved = []
    for site in range(site_count):
        interleaved.extend((site, site_count + site))
    remainder = tensor.permute(interleaved).reshape(*([4] * site_count))
    cores: list[torch.Tensor] = []
    left_rank = 1
    for site in range(site_count - 1):
        matrix = remainder.reshape(left_rank * 4, -1)
        left, singular, right = torch.linalg.svd(matrix, full_matrices=False)
        threshold = (
            torch.finfo(singular.dtype).eps
            * max(matrix.shape)
            * singular[0].abs().clamp_min(1.0)
        )
        rank = max(1, int(torch.sum(singular > threshold).item()))
        left = left[:, :rank]
        singular = singular[:rank]
        right = right[:rank, :]
        cores.append(left.reshape(left_rank, 2, 2, rank))
        remainder = (singular.unsqueeze(1) * right).reshape(
            rank, *([4] * (site_count - site - 1))
        )
        left_rank = rank
    cores.append(remainder.reshape(left_rank, 2, 2, 1))
    return cores


def _operator_to_mpo(
    operator: torch.Tensor,
    site_count: int,
    *,
    gauge: str = "right-canonical",
) -> list[torch.Tensor]:
    """Factor an operator with weights visible to the left-to-right zip-up.

    Right-canonical MPO cores keep the singular weights in the unprocessed
    left prefix.  A left-canonical MPO instead pushes them toward its last
    core, so intermediate PEPS truncations can discard important components
    before seeing their actual operator weights.  This is still a local
    compression, not a full PEPS-environment error bound.
    """
    if gauge == "left-canonical":
        return _operator_to_left_canonical_mpo(operator, site_count)
    if gauge != "right-canonical":
        raise ValueError(f"unsupported residual MPO gauge {gauge!r}")
    expected = 2**site_count
    if site_count < 1 or tuple(operator.shape) != (expected, expected):
        raise ValueError("multi-site operator shape does not match its support")
    tensor = operator.reshape(*([2] * site_count), *([2] * site_count))
    interleaved = [
        axis for site in range(site_count) for axis in (site, site + site_count)
    ]
    remainder = tensor.permute(interleaved).reshape(*([4] * site_count))
    reverse_cores: list[torch.Tensor] = []
    right_rank = 1
    for site in range(site_count - 1, 0, -1):
        matrix = remainder.reshape(-1, 4 * right_rank)
        left, singular, right = torch.linalg.svd(matrix, full_matrices=False)
        threshold = (
            torch.finfo(singular.dtype).eps
            * max(matrix.shape)
            * singular[0].abs().clamp_min(1.0)
        )
        rank = max(1, int(torch.sum(singular > threshold).item()))
        reverse_cores.append(right[:rank].reshape(rank, 2, 2, right_rank))
        remainder = (left[:, :rank] * singular[:rank]).reshape(
            *([4] * site),
            rank,
        )
        right_rank = rank
    return [remainder.reshape(1, 2, 2, right_rank), *reversed(reverse_cores)]


def _path_axis(
    site: int,
    other: int,
    neighbors: dict[int, dict[str, int]],
) -> int:
    direction = next(
        (side for side, neighbor in neighbors[site].items() if neighbor == other),
        None,
    )
    if direction is None:
        raise ValueError(f"path step {(site, other)} is not nearest-neighbour")
    return int(bp.DIR_TO_AXIS[direction])


def _absorb_mpo_core(
    tensor: torch.Tensor,
    core: torch.Tensor,
    *,
    previous_axis: int | None,
    next_axis: int | None,
) -> torch.Tensor:
    """Absorb one MPO core and merge its ranks into PEPS path bonds."""
    value = torch.einsum("budlri,aoic->budlroac", tensor, core)
    # value axes: batch, U, D, L, R, physical_out, mpo_left, mpo_right
    permutation = [0]
    shape = [int(tensor.shape[0])]
    used_mpo_axes: set[int] = set()
    for peps_axis in range(4):
        value_axis = peps_axis + 1
        dimension = int(tensor.shape[value_axis])
        permutation.append(value_axis)
        if previous_axis == peps_axis:
            permutation.append(6)
            dimension *= int(core.shape[0])
            used_mpo_axes.add(6)
        if next_axis == peps_axis:
            permutation.append(7)
            dimension *= int(core.shape[-1])
            used_mpo_axes.add(7)
        shape.append(dimension)
    for mpo_axis in (6, 7):
        if mpo_axis not in used_mpo_axes:
            if int(value.shape[mpo_axis]) != 1:
                raise ValueError("nontrivial MPO rank has no matching path bond")
            permutation.append(mpo_axis)
    permutation.append(5)
    shape.append(2)
    return value.permute(permutation).reshape(shape).contiguous()


def _zip_absorb_mpo_core(
    tensors: dict[int, torch.Tensor],
    previous_site: int,
    site: int,
    core: torch.Tensor,
    *,
    next_axis: int | None,
    config: Any,
    neighbors: dict[int, dict[str, int]],
    svd_relative_error: float = 0.0,
    compression_callback: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Absorb one MPO core while streaming its enlarged right-rank axis.

    Materializing an interior four-site MPO core first multiplies both path
    bonds of a D=64 PEPS tensor by the operator-Schmidt ranks (typically
    4x16).  In complex128 that temporary alone can occupy 64 GiB.  This is the
    same QR--SVD zip-up used by the ordinary two-site update, written through
    the right reduced Gram matrix so that only one outgoing MPO-rank slice is
    live at a time.  No MPO singular values are discarded here; the sole
    truncation remains the requested PEPS ``chi_max``.
    """
    left_tensor = tensors[int(previous_site)]
    right_tensor = tensors[int(site)]
    if left_tensor.ndim != 6 or right_tensor.ndim != 6:
        raise ValueError("streamed MPO zip expects batched rank-6 PEPS tensors")
    if int(left_tensor.shape[0]) != int(right_tensor.shape[0]):
        raise ValueError("streamed MPO zip batch dimensions do not match")

    left_axis = _path_axis(int(previous_site), int(site), neighbors)
    right_axis = _path_axis(int(site), int(previous_site), neighbors)
    mpo_left, physical_out, physical_in, mpo_right = map(int, core.shape)
    if physical_out != 2 or physical_in != 2:
        raise ValueError("residual MPO core must have qubit physical legs")
    base_common = int(right_tensor.shape[right_axis + 1])
    common = base_common * mpo_left
    if int(left_tensor.shape[left_axis + 1]) != common:
        raise ValueError(
            "incoming expanded MPO bond is inconsistent: "
            f"left={int(left_tensor.shape[left_axis + 1])}, "
            f"right={base_common}x{mpo_left}"
        )

    batch = int(left_tensor.shape[0])
    left_other_axes = [axis for axis in range(4) if axis != left_axis]
    left_other_shape = [int(left_tensor.shape[axis + 1]) for axis in left_other_axes]
    right_other_axes = [axis for axis in range(4) if axis != right_axis]
    right_other_shape = [int(right_tensor.shape[axis + 1]) for axis in right_other_axes]

    target_chunk_bytes = int(
        os.environ.get("NCS_RESIDUAL_MPO_STREAM_CHUNK_BYTES", 512 * 1024**2)
    )
    if target_chunk_bytes < 1:
        raise ValueError("NCS_RESIDUAL_MPO_STREAM_CHUNK_BYTES must be positive")

    # A conventional reduced QR materializes Q with shape
    # [all external left legs, expanded path bond].  At D=96 this Q alone can
    # require about 30 GiB.  Instead form H=A^H A in external-leg chunks and
    # use its eigendecomposition to obtain the polar/QR-equivalent factors
    #
    #   A = (A V Lambda^{-1/2}) (Lambda^{1/2} V^H) = Q R.
    #
    # The zero-eigenvalue subspace is removed with a numerical pseudoinverse.
    # This keeps only O(common^2) factorization storage and later emits A times
    # the retained rank directly, never the much wider Q.
    left_ordered = left_tensor.permute(
        0,
        left_axis + 1,
        *[axis + 1 for axis in left_other_axes],
        5,
    )
    left_stream_position = max(
        range(len(left_other_shape)),
        key=lambda position: left_other_shape[position],
    )
    left_stream_peps_axis = left_other_axes[left_stream_position]
    left_stream_dimension = left_other_shape[left_stream_position]
    left_other_per_stream = math.prod(
        dimension
        for position, dimension in enumerate(left_other_shape)
        if position != left_stream_position
    )
    left_bytes_per_stream_index = (
        batch
        * common
        * left_other_per_stream
        * physical_out
        * int(left_tensor.element_size())
    )
    left_stream_step = max(
        1,
        min(
            left_stream_dimension,
            target_chunk_bytes // max(1, left_bytes_per_stream_index),
        ),
    )

    def left_matrix_chunks(ordered):
        for start in range(0, left_stream_dimension, left_stream_step):
            stop = min(start + left_stream_step, left_stream_dimension)
            ordered_chunk = ordered.narrow(
                2 + left_stream_position, start, stop - start
            )
            # The permutation gives A_chunk^T; conjugation makes this the
            # adjoint needed by both A^H A and the later A @ projector pass.
            yield start, stop, ordered_chunk.reshape(batch, common, -1).conj()

    left_gram = torch.zeros(
        (batch, common, common),
        dtype=left_tensor.dtype,
        device=left_tensor.device,
    )
    for _, _, chunk in left_matrix_chunks(left_ordered):
        left_gram.add_(chunk @ chunk.mH)
        del chunk
    left_gram = (left_gram + left_gram.mH) * 0.5
    left_values, left_vectors = torch.linalg.eigh(left_gram)
    left_values = left_values.flip(-1).clamp_min(0.0)
    left_vectors = left_vectors.flip(-1)
    left_scale = left_values[:, :1].clamp_min(1.0)
    left_threshold = torch.finfo(left_values.dtype).eps * max(1, common) * left_scale
    left_nonzero = left_values > left_threshold
    left_root = torch.sqrt(left_values)
    left_inverse_root = torch.where(
        left_nonzero,
        torch.reciprocal(left_root.clamp_min(torch.finfo(left_root.dtype).tiny)),
        0.0,
    )
    # R = Lambda^(1/2) V^H and W = V Lambda^(-1/2), so Q = A W.
    r_left = left_root.unsqueeze(-1) * left_vectors.mH
    left_whitener = left_vectors * left_inverse_root.unsqueeze(1)
    del (
        left_gram,
        left_values,
        left_scale,
        left_threshold,
        left_nonzero,
        left_root,
        left_inverse_root,
    )

    # Directly emit [batch, base_common, mpo_left, other..., physical]
    # order.  The largest external PEPS leg is streamed as well as the MPO
    # right rank: a single outgoing-rank slice can still be several GiB once
    # residual routing has populated many D=64 bonds.  Gram accumulation and
    # the later projection are separable over these external columns, so this
    # second level of streaming is exact.
    right_ordered = right_tensor.permute(
        0,
        right_axis + 1,
        *[axis + 1 for axis in right_other_axes],
        5,
    )
    stream_position = max(
        range(len(right_other_shape)),
        key=lambda position: right_other_shape[position],
    )
    stream_peps_axis = right_other_axes[stream_position]
    stream_dimension = right_other_shape[stream_position]
    other_per_stream = math.prod(
        dimension
        for position, dimension in enumerate(right_other_shape)
        if position != stream_position
    )
    bytes_per_stream_index = (
        batch
        * common
        * other_per_stream
        * physical_out
        * int(right_tensor.element_size())
    )
    stream_step = max(
        1,
        min(
            stream_dimension,
            target_chunk_bytes // max(1, bytes_per_stream_index),
        ),
    )

    def right_matrix_chunks(mpo_index: int):
        for start in range(0, stream_dimension, stream_step):
            stop = min(start + stream_step, stream_dimension)
            ordered_chunk = right_ordered.narrow(
                2 + stream_position, start, stop - start
            )
            value = torch.einsum(
                "bdxyzi,aoi->bdaxyzo",
                ordered_chunk,
                core[..., int(mpo_index)],
            )
            yield start, stop, value.reshape(batch, common, -1)

    gram = torch.zeros(
        (batch, common, common),
        dtype=right_tensor.dtype,
        device=right_tensor.device,
    )
    for mpo_index in range(mpo_right):
        for _, _, chunk in right_matrix_chunks(mpo_index):
            gram.add_(chunk @ chunk.mH)
            del chunk

    reduced = r_left @ gram @ r_left.mH
    reduced = (reduced + reduced.mH) * 0.5
    eigenvalues, eigenvectors, eig_metadata = bp._compression_hermitian_eigh(
        reduced,
        max_rank=int(config.chi_max),
        solver=str(config.svd_solver),
        oversample=int(config.svd_oversample),
        power_iterations=int(config.svd_power_iterations),
        random_seed=int(config.svd_random_seed),
    )
    available = int(eig_metadata["available_rank"])
    singular = torch.sqrt(eigenvalues)

    def record_zip_compression(record: dict[str, Any]) -> None:
        if compression_callback is not None:
            compression_callback(
                {
                    **record,
                    "bond": [int(previous_site), int(site)],
                    "mpo_left_rank": mpo_left,
                    "mpo_right_rank": mpo_right,
                    "left_factorization": "streamed_gram_whitening",
                }
            )

    rank = bp._adaptive_svd_rank(
        singular,
        chi_max=int(config.chi_max),
        relative_error=float(svd_relative_error),
        total_squared_norm=eig_metadata["total_squared_norm"],
        available_rank=available,
        spectrum_complete=bool(eig_metadata["spectrum_complete"]),
        solver=str(eig_metadata["solver"]),
        diagnostics_callback=(
            record_zip_compression if compression_callback is not None else None
        ),
    )
    eigenvalues = eigenvalues[:, :rank]
    eigenvectors = eigenvectors[:, :, :rank]
    singular = singular[:, :rank]
    singular_threshold = (
        torch.finfo(singular.dtype).eps
        * max(int(reduced.shape[-2]), int(reduced.shape[-1]))
        * singular[:, :1].clamp_min(1.0)
    )
    nonzero = singular > singular_threshold
    root = torch.where(nonzero, torch.sqrt(singular), 0.0)
    inverse_root = torch.where(
        nonzero,
        torch.reciprocal(root.clamp_min(torch.finfo(root.dtype).tiny)),
        0.0,
    )

    left_small = eigenvectors * root.unsqueeze(1)
    left_projector = left_whitener @ left_small
    left_tag_order = ["batch", *left_other_axes, "physical", left_axis]
    left_tag_position = {tag: position for position, tag in enumerate(left_tag_order)}
    left_restore = [left_tag_position["batch"]]
    left_restore.extend(left_tag_position[axis] for axis in range(4))
    left_restore.append(left_tag_position["physical"])
    left_final_shape = [int(dimension) for dimension in left_tensor.shape]
    left_final_shape[left_axis + 1] = rank
    new_left = torch.empty(
        left_final_shape, dtype=left_tensor.dtype, device=left_tensor.device
    )
    for start, stop, chunk in left_matrix_chunks(left_ordered):
        chunk_other_shape = list(left_other_shape)
        chunk_other_shape[left_stream_position] = stop - start
        projected = chunk.mH @ left_projector
        standard = projected.reshape(batch, *chunk_other_shape, 2, rank).permute(
            *left_restore
        )
        new_left.narrow(left_stream_peps_axis + 1, start, stop - start).copy_(standard)
        del chunk, projected, standard

    # sqrt(S) V^H = S^(-1/2) U^H R_left B.  Forming this projector now lets
    # us release Q_left and the enlarged incoming PEPS tensor before the
    # second streamed pass over B.
    right_projector = (eigenvectors.mH @ r_left) * inverse_root.unsqueeze(-1)
    tensors[int(previous_site)] = new_left
    del (
        left_tensor,
        left_ordered,
        left_vectors,
        left_whitener,
        left_projector,
        r_left,
        gram,
        reduced,
        eigenvalues,
        eigenvectors,
        singular,
        root,
        inverse_root,
        nonzero,
        singular_threshold,
        left_small,
        new_left,
    )

    final_shape = [int(dimension) for dimension in right_tensor.shape]
    final_shape[right_axis + 1] = rank
    if next_axis is None:
        if mpo_right != 1:
            raise ValueError("terminal MPO core has a nontrivial outgoing rank")
    else:
        final_shape[next_axis + 1] *= mpo_right
    final = torch.empty(
        final_shape, dtype=right_tensor.dtype, device=right_tensor.device
    )
    if next_axis is not None:
        split_shape: list[int] = [batch]
        for axis in range(4):
            dimension = (
                rank if axis == right_axis else int(right_tensor.shape[axis + 1])
            )
            split_shape.append(dimension)
            if axis == next_axis:
                split_shape.append(mpo_right)
        split_shape.append(2)
        final_split = final.reshape(split_shape)
    else:
        final_split = final

    right_tag_order = ["batch", right_axis, *right_other_axes, "physical"]
    right_tag_position = {tag: position for position, tag in enumerate(right_tag_order)}
    right_restore = [right_tag_position["batch"]]
    right_restore.extend(right_tag_position[axis] for axis in range(4))
    right_restore.append(right_tag_position["physical"])
    for mpo_index in range(mpo_right):
        if next_axis is None:
            destination = final_split
        else:
            destination = final_split.select(next_axis + 2, mpo_index)
        for start, stop, chunk in right_matrix_chunks(mpo_index):
            chunk_other_shape = list(right_other_shape)
            chunk_other_shape[stream_position] = stop - start
            projected = right_projector @ chunk
            standard = projected.reshape(batch, rank, *chunk_other_shape, 2).permute(
                *right_restore
            )
            destination.narrow(stream_peps_axis + 1, start, stop - start).copy_(
                standard
            )
            del chunk, projected, standard
    tensors[int(site)] = final.contiguous()


def _apply_adjacent_mpo_operator(
    tensors: dict[int, torch.Tensor],
    path_sites: Sequence[int],
    operator: torch.Tensor,
    *,
    config: Any,
    neighbors: dict[int, dict[str, int]],
    svd_relative_error: float = 0.0,
    mpo_gauge: str = "right-canonical",
    compression_callback: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Apply a shared 3/4-site operator along an adjacent PEPS path."""
    sites = tuple(map(int, path_sites))
    if len(sites) < 3:
        raise ValueError("the MPO path is reserved for operators of arity >= 3")
    cores = _operator_to_mpo(operator, len(sites), gauge=mpo_gauge)
    for index, (site, core) in enumerate(zip(sites, cores)):
        previous = None if index == 0 else sites[index - 1]
        following = None if index + 1 == len(sites) else sites[index + 1]
        previous_axis = (
            None if previous is None else _path_axis(site, previous, neighbors)
        )
        next_axis = (
            None if following is None else _path_axis(site, following, neighbors)
        )
        if previous is None:
            tensors[site] = _absorb_mpo_core(
                tensors[site],
                core,
                previous_axis=previous_axis,
                next_axis=next_axis,
            )
        else:
            _zip_absorb_mpo_core(
                tensors,
                previous,
                site,
                core,
                next_axis=next_axis,
                config=config,
                neighbors=neighbors,
                svd_relative_error=svd_relative_error,
                compression_callback=compression_callback,
            )


def _embed_operator_on_path(
    operator: torch.Tensor,
    support: Sequence[int],
    path_sites: Sequence[int],
) -> torch.Tensor:
    """Embed a support operator into an adjacent path with spectator identities.

    This is the exact PEPO/MPO alternative to physically swapping support
    tokens together and then swapping them back.  Residual paths in the
    supplied geometries contain at most a handful of qubits, so constructing
    this small dense path operator is inexpensive; its MPO factorization keeps
    the long-range action compressed during PEPS absorption.
    """
    support_sites = tuple(map(int, support))
    path = tuple(map(int, path_sites))
    if len(set(path)) != len(path):
        raise ValueError("residual MPO path contains duplicate sites")
    if not set(support_sites).issubset(path):
        raise ValueError("residual operator support is not contained in path")
    expected = 2 ** len(support_sites)
    if tuple(operator.shape) != (expected, expected):
        raise ValueError("residual operator shape does not match its support")
    spectators = tuple(site for site in path if site not in set(support_sites))
    expanded = operator
    identity = torch.eye(2, dtype=operator.dtype, device=operator.device)
    source_order = list(support_sites)
    for site in spectators:
        expanded = torch.kron(expanded, identity)
        source_order.append(site)
    return residual.permute_operator_sites(
        expanded,
        source_order,
        path,
        local_dim=2,
    )


def _compiled_cluster_path(
    kernel: residual.CompiledResidualOperator,
    edge_record: dict[str, Any],
    neighbors: dict[int, dict[str, int]],
) -> list[int]:
    """Return a simple nearest-neighbour path containing the kernel support."""
    m, n = kernel.edge
    route = [int(site) for site in edge_record["swap_routing"]["path"]]
    if route[0] == n and route[-1] == m:
        route.reverse()
    if route[0] != m or route[-1] != n:
        raise ValueError(f"manifest path does not connect residual edge {(m, n)}")
    left_block, right_block = kernel.control_blocks
    left_partners = [site for site in left_block if site != m]
    right_partners = [site for site in right_block if site != n]
    if left_partners and left_partners[0] not in route:
        route.insert(0, int(left_partners[0]))
    if right_partners and right_partners[0] not in route:
        route.append(int(right_partners[0]))
    if len(route) != len(set(route)):
        raise ValueError(f"residual support route is not simple: {route}")
    for first, second in zip(route, route[1:]):
        _path_axis(first, second, neighbors)
    if not set(kernel.support).issubset(route):
        raise ValueError("residual route does not contain the dressed support")
    return route


def _compact_support_swaps(
    cluster_path: Sequence[int], support: Sequence[int]
) -> tuple[list[tuple[int, int]], tuple[int, ...]]:
    """Pack support tokens into the left end of a path with adjacent SWAPs."""
    positions = list(map(int, cluster_path))
    tokens = positions.copy()
    support_set = set(map(int, support))
    swaps: list[tuple[int, int]] = []
    for target in range(len(support_set)):
        source = next(
            index
            for index in range(target, len(tokens))
            if tokens[index] in support_set
        )
        while source > target:
            swaps.append((positions[source - 1], positions[source]))
            tokens[source - 1], tokens[source] = tokens[source], tokens[source - 1]
            source -= 1
    routed_order = tuple(tokens[: len(support_set)])
    if set(routed_order) != support_set:
        raise RuntimeError("support compaction failed")
    return swaps, routed_order


def apply_compiled_residual_operator(
    tensors: dict[int, torch.Tensor],
    kernel: residual.CompiledResidualOperator,
    edge_record: dict[str, Any],
    *,
    config: Any,
    neighbors: dict[int, dict[str, int]],
    svd_relative_error: float = 0.0,
    routing_mode: str = "swap",
    mpo_gauge: str = "right-canonical",
    compression_callback: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Apply one pulse-resolved 2/3/4-site residual Magnus operator."""
    cluster_path = _compiled_cluster_path(kernel, edge_record, neighbors)
    if routing_mode not in {"swap", "direct-mpo"}:
        raise ValueError("residual routing mode must be swap or direct-mpo")

    def tagged(stage: str) -> Callable[[dict[str, Any]], None] | None:
        if compression_callback is None:
            return None

        def emit(record: dict[str, Any]) -> None:
            compression_callback(
                {
                    **record,
                    "stage": stage,
                    "edge": [int(kernel.edge[0]), int(kernel.edge[1])],
                    "support_size": len(kernel.support),
                    "cluster_path": [int(site) for site in cluster_path],
                    "routing_mode": str(routing_mode),
                    "mpo_gauge": str(mpo_gauge),
                }
            )

        return emit

    if routing_mode == "direct-mpo":
        path_operator = _embed_operator_on_path(
            kernel.operator,
            kernel.support,
            cluster_path,
        )
        if len(cluster_path) == 2:
            bp._apply_operator_microbatched(
                tensors,
                tuple(cluster_path),
                path_operator,
                config=config,
                neighbors=neighbors,
                svd_relative_error=svd_relative_error,
                svd_diagnostics_callback=tagged("residual_direct_two_site"),
            )
        else:
            _apply_adjacent_mpo_operator(
                tensors,
                cluster_path,
                path_operator,
                config=config,
                neighbors=neighbors,
                svd_relative_error=svd_relative_error,
                mpo_gauge=mpo_gauge,
                compression_callback=tagged("residual_direct_mpo_zip"),
            )
        return

    swaps, routed_order = _compact_support_swaps(cluster_path, kernel.support)
    swap = swap_operator(dtype=kernel.operator.dtype, device=kernel.operator.device)

    for pair in swaps:
        bp._apply_operator_microbatched(
            tensors,
            pair,
            swap,
            config=config,
            neighbors=neighbors,
            svd_relative_error=svd_relative_error,
            svd_diagnostics_callback=tagged("route_swap_forward"),
        )
    routed_operator = residual.permute_operator_sites(
        kernel.operator,
        kernel.support,
        routed_order,
        local_dim=2,
    )
    routed_sites = tuple(cluster_path[: len(kernel.support)])
    if len(routed_sites) == 2:
        bp._apply_operator_microbatched(
            tensors,
            routed_sites,
            routed_operator,
            config=config,
            neighbors=neighbors,
            svd_relative_error=svd_relative_error,
            svd_diagnostics_callback=tagged("residual_two_site"),
        )
    else:
        _apply_adjacent_mpo_operator(
            tensors,
            routed_sites,
            routed_operator,
            config=config,
            neighbors=neighbors,
            svd_relative_error=svd_relative_error,
            mpo_gauge=mpo_gauge,
            compression_callback=tagged("residual_mpo_zip"),
        )
    for pair in reversed(swaps):
        bp._apply_operator_microbatched(
            tensors,
            pair,
            swap,
            config=config,
            neighbors=neighbors,
            svd_relative_error=svd_relative_error,
            svd_diagnostics_callback=tagged("route_swap_reverse"),
        )


def apply_compiled_residual_layer(
    tensors: dict[int, torch.Tensor],
    kernels: Sequence[residual.CompiledResidualOperator],
    geometry: dict[str, Any],
    *,
    config: Any,
    neighbors: dict[int, dict[str, int]],
    svd_relative_error: float = 0.0,
    routing_mode: str = "swap",
    mpo_gauge: str = "right-canonical",
    compression_callback: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    records = {(int(edge["q1"]), int(edge["q2"])): edge for edge in geometry["edges"]}
    for kernel in kernels:
        apply_compiled_residual_operator(
            tensors,
            kernel,
            records[kernel.edge],
            config=config,
            neighbors=neighbors,
            svd_relative_error=svd_relative_error,
            routing_mode=routing_mode,
            mpo_gauge=mpo_gauge,
            compression_callback=compression_callback,
        )


def _unbatched_cpu_peps(tensors: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    return {
        int(site): value[0].detach().to(device="cpu").contiguous()
        for site, value in sorted(tensors.items())
    }


def load_common_background_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device | None = None,
    batched: bool = False,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Load a reusable common-background PEPS checkpoint.

    The returned metadata is the full checkpoint payload except for the tensor
    dictionary.  Set ``batched=True`` when passing the result back to the
    streamed PEPS engine, whose leading dimension is the branch batch.
    """
    path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != "coherent_common_background_peps":
        raise ValueError(f"{path} is not a common-background PEPS checkpoint")
    target = torch.device("cpu" if device is None else device)
    tensors = {
        int(site): value.to(device=target).contiguous()
        for site, value in payload["tensors"].items()
    }
    shape = payload.get("shape", [])
    if len(shape) != 3 or len(tensors) != int(shape[0]) * int(shape[1]):
        raise ValueError(f"{path} has an inconsistent PEPS geometry")
    if batched:
        tensors = {site: value.unsqueeze(0) for site, value in tensors.items()}
    metadata = {key: value for key, value in payload.items() if key != "tensors"}
    return tensors, metadata


def _max_virtual_bond(tensors: dict[int, torch.Tensor]) -> int:
    return max(int(size) for value in tensors.values() for size in value.shape[1:5])


def build_common_background_peps(
    ctx: Any,
    geometry: dict[str, Any],
    *,
    peps_bond: int,
    compiled_layers: Sequence[Sequence[residual.CompiledResidualOperator]],
    checkpoint: Callable[[int, dict[int, torch.Tensor]], None] | None = None,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Evolve pulse-resolved residual and intended layers, with checkpoints.

    ``checkpoint(layer, tensors)`` receives the batched PEPS after the complete
    common-background layer.  ``layer=-1`` denotes the initial product state.
    """
    if peps_bond <= 0:
        raise ValueError("peps_bond must be positive")
    config = bp.SingleSiteConfig(
        chi_max=int(peps_bond),
        two_qubit_apply_batch_size=1024,
        two_qubit_layer_gate_batch_size=32,
        batch_disjoint_two_qubit_gates=True,
    )
    config.validate()
    if len(compiled_layers) != len(ctx.layers):
        raise ValueError("compiled residual library does not match circuit depth")
    neighbors = bp._build_neighbors(int(ctx.width), int(ctx.length))
    tensors = {site: value.unsqueeze(0) for site, value in bp._zero_peps(ctx).items()}
    if checkpoint is not None:
        checkpoint(-1, tensors)

    _sync(ctx.device)
    started = time.perf_counter()
    layer_records = []
    for layer, gates in enumerate(ctx.layers):
        tick = time.perf_counter()
        residual_tick = time.perf_counter()
        # The interaction-picture layer background is U_{0,l} K_{res,I,l};
        # hence K_{res,I,l} acts on the incoming state before U_{0,l}.
        apply_compiled_residual_layer(
            tensors,
            compiled_layers[layer],
            geometry,
            config=config,
            neighbors=neighbors,
        )
        _sync(ctx.device)
        residual_seconds = time.perf_counter() - residual_tick

        intended_tick = time.perf_counter()
        bp._apply_layer(tensors, gates, config=config, neighbors=neighbors)
        _sync(ctx.device)
        intended_seconds = time.perf_counter() - intended_tick
        if checkpoint is not None:
            checkpoint(layer, tensors)
        record = {
            "layer": layer,
            "intended_seconds": intended_seconds,
            "residual_seconds": residual_seconds,
            "total_seconds": time.perf_counter() - tick,
            "max_virtual_bond": _max_virtual_bond(tensors),
        }
        layer_records.append(record)
        print(
            f"[common-background] layer={layer + 1}/{len(ctx.layers)} "
            f"intended={intended_seconds:.3f}s "
            f"residual={residual_seconds:.3f}s "
            f"bond={record['max_virtual_bond']}",
            flush=True,
        )
    return _unbatched_cpu_peps(tensors), {
        "seconds": time.perf_counter() - started,
        "layers": layer_records,
        "max_virtual_bond": _max_virtual_bond(tensors),
        "n_residual_edges_per_layer": len(geometry["edges"]),
        "kernel_scheme": compiled_layers[0][0].scheme,
        "layer_duration_ns": residual.LAYER_DURATION_NS,
        "edge_composition": "ordered_product_in_manifest_order",
        "background_layers": ["interaction_picture_residual", "intended"],
        "residual_support_sizes": sorted(
            {len(kernel.support) for layer in compiled_layers for kernel in layer}
        ),
    }


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--width", type=int, required=True)
    value.add_argument("--length", type=int, required=True)
    value.add_argument("--layers", type=int, required=True)
    value.add_argument("--peps-bond", type=int, default=64)
    value.add_argument(
        "--layer-duration-ns", type=float, default=DEFAULT_LAYER_DURATION_NS
    )
    value.add_argument("--residual-manifest", default=str(DEFAULT_MANIFEST))
    value.add_argument("--kernel-scheme", choices=KERNEL_SCHEMES, default="magnus-1")
    value.add_argument("--piece-num", type=int, default=1000)
    value.add_argument(
        "--residual-integration-steps",
        type=int,
        default=0,
        help="Midpoint slices for the offline residual library; 0 uses piece-num.",
    )
    value.add_argument("--pattern-seed", type=int, default=0)
    value.add_argument("--pattern-index", type=int, default=6)
    value.add_argument("--pattern-file")
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--save-layers", choices=("all", "final"), default="all")
    value.add_argument("--checkpoint-dir", required=True)
    value.add_argument("--overwrite", action="store_true")
    return value
