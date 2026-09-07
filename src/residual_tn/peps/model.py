"""Shared circuit model and dynamic PEPS evolution for Figures 5 and 6."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import gc
import hashlib
import inspect
import json
import math
from pathlib import Path
import time

pass  # Package imports need no path bootstrap.
import torch
import torch.backends.opt_einsum
from residual_tn.peps import compression as bp
from residual_tn.backend import common_background_peps as common
from residual_tn.backend import physics_0522 as physics
from residual_tn.backend import residual_background_exact as exact_ops
from residual_tn.backend import residual_magnus as residual

from residual_tn.paths import CONFIGS
from residual_tn.geometry import load as load_geometry
from residual_tn.identity import source_identity

core = bp.core


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bp.write_json(path, data)


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu(v) for v in value)
    return value


def save_tensor(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(cpu(value), temporary)
    temporary.replace(path)


def synchronize():
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()


def setup_device(device, allocator_gib=76.0, reserve_gib=4.0, threads=4):
    torch.set_num_threads(threads)
    torch.set_grad_enabled(False)
    if device.type == "cuda":
        index = 0 if device.index is None else device.index
        torch.cuda.set_device(index)
        total = torch.cuda.get_device_properties(index).total_memory
        # The physical device may report slightly less than its marketed size.
        limit = min(allocator_gib * 1024**3, total - reserve_gib * 1024**3)
        if limit <= 0:
            raise ValueError("Insufficient device memory after reserve")
        torch.cuda.set_per_process_memory_fraction(limit / total, index)


def config(
    *, bond=64, bp_tol=1e-10, bp_max_iter=500, gloop=8, cavity=False, device="cuda"
):
    changes = dict(
        chi_max=bond,
        bp_tol=bp_tol,
        bp_max_iter=bp_max_iter,
        svd_solver="randomized",
        svd_oversample=24,
        svd_power_iterations=2,
        svd_random_seed=0,
        two_qubit_apply_batch_size=1,
        two_qubit_layer_gate_batch_size=8,
        gloop_size=gloop,
        gloop_memory_target_gib=24.0,
        gloop_peak_budget_gib=72.0,
        gloop_peak_reserve_gib=4.0,
        gloop_peak_safety_factor=1.25,
        gloop_memory_max_slices=262144,
        gloop_oom_max_retries=3,
        einsum_memory_gib=24.0,
        retain_pair_ledger=True,
        readout_patch_batch_size=1,
        readout_branch_stream_count=1,
        pauli_readout_schedule="tile_major",
    )
    if cavity:
        # All two-insertion correlator reads now use generalized loops.
        changes.update(
            local_readout_method="gloop", readout_boundary_mode="single_site"
        )
    if str(device) == "cpu":
        changes.update(
            gloop_memory_target_gib=None, gloop_peak_budget_gib=None, svd_solver="full"
        )
    return bp.configuration(**changes)


def model(
    device,
    noise_us=5.0,
    kraus_tol=1e-8,
    small=False,
    *,
    width=5,
    length=4,
    depth=20,
    pieces=1000,
    pattern_seed=0,
    pattern_index=6,
    manifest=None,
):
    """Build local channels and residual kernels, never a full state vector."""
    physics.NOISE_T_US = float(noise_us)
    if small:
        width, length, depth, pieces = 2, 2, 4, 20
    manifest = (
        CONFIGS / "residual_manifest.json" if manifest is None else Path(manifest)
    )
    if small:
        _, geometry = common.load_residual_geometry(manifest, 5, 4)
        geometry = dict(
            geometry,
            width=2,
            length=2,
            n_qubits=4,
            n_residual_edges=1,
            edges=[
                dict(
                    geometry["edges"][0],
                    q1=0,
                    q2=3,
                    swap_routing=dict(
                        path=[0, 1, 3],
                        forward_swaps=[[0, 1]],
                        reverse_swaps=[[0, 1]],
                        kernel_support_after_routing=[1, 3],
                    ),
                )
            ],
        )
    else:
        geometry = load_geometry(manifest, width, length)
    ctx, meta = physics.build_generated_0522_context(
        width,
        length,
        depth,
        device,
        torch.complex128,
        piece_num=pieces,
        pattern_seed=pattern_seed,
        pattern_index=pattern_index,
        mapping_mode="identity",
        kraus_tol=kraus_tol,
    )
    kernels = residual.ResidualMagnusLibrary(
        ctx.n_qubits,
        integration_steps=pieces,
        dtype=ctx.dtype,
        device=device,
        projected_qubits=True,
    ).compile_circuit(ctx.layers, geometry["edges"], scheme="magnus-1")
    fingerprint = hashlib.sha256()
    for gates in ctx.layers:
        for gate in gates:
            fingerprint.update(
                gate["kraus_ops"].detach().cpu().contiguous().numpy().tobytes()
            )
    identity = dict(
        width=width,
        length=length,
        layers=depth,
        piece_num=pieces,
        pattern_seed=pattern_seed,
        pattern_index=pattern_index,
        noise_T1_T2_us=noise_us,
        kraus_eigenvalue_cutoff=kraus_tol,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        geometry_sha256=hashlib.sha256(
            json.dumps(geometry, sort_keys=True).encode()
        ).hexdigest(),
        residual_edges=geometry["edges"],
        common_residual_background=True,
        kernel_scheme="magnus-1",
        gate_counts_by_layer=[len(row) for row in ctx.layers],
        retained_modes_by_layer=[
            sum(int(g["kraus_ops"].shape[0]) for g in row) for row in ctx.layers
        ],
        compiled_support_counts={
            str(size): count
            for size, count in collections.Counter(
                len(k.support) for row in kernels for k in row
            ).items()
        },
        gate_table_sha256=hashlib.sha256(
            json.dumps(meta["gate_table"], sort_keys=True).encode()
        ).hexdigest(),
        kraus_sha256=fingerprint.hexdigest(),
        code_manifest_sha256=source_identity(),
        fidelity_target="normalized residual-dressed coherent background",
        pair_scope="all unordered distinct gate locations, including same-layer disjoint gates",
    )
    return ctx, geometry, kernels, identity


class DynamicEvolution:
    """One cold norm BP and fixed-environment compression after each layer.

    No post-compression BP is done here. The caller solves it only when it
    actually reads an observable or stores source amplitudes. Intermediate
    Pauli branch propagation therefore has no unused repeated solve.
    """

    def __init__(
        self,
        ctx,
        geometry,
        kernels,
        cfg,
        audit_path,
        threshold=1e-4,
        residual_svd_tol=1e-3,
    ):
        self.ctx, self.geometry, self.kernels, self.cfg = ctx, geometry, kernels, cfg
        self.threshold = threshold
        self.residual_svd_tol = residual_svd_tol
        self.original_apply = core._apply_layer
        self.path = Path(audit_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.counter = 0
        self.layer_ids = {id(gates): i for i, gates in enumerate(ctx.layers)}

    def residual_hook(self, tensors, layer, *, config, neighbors):
        common.apply_compiled_residual_layer(
            tensors,
            self.kernels[layer],
            self.geometry,
            config=config,
            neighbors=neighbors,
            svd_relative_error=self.residual_svd_tol,
            routing_mode="direct-mpo",
            mpo_gauge="right-canonical",
        )
        return {
            (site, side)
            for site, directions in neighbors.items()
            for side in directions
        }

    def apply(self, tensors, gates, *, config, neighbors):
        started = time.perf_counter()
        invalidated = self.original_apply(
            tensors, gates, config=config, neighbors=neighbors
        )
        if self.threshold is None:
            return invalidated
        batch = int(next(iter(tensors.values())).shape[0])
        if batch != 1:
            raise RuntimeError(
                "Adaptive ranks require branch_batch_size=1; prebuild all clean layers"
            )
        peps = {site: value[0] for site, value in tensors.items()}
        messages, _, info = bp.solve(peps, self.ctx, config)
        frozen = {k: v.detach().cpu() for k, v in messages.items()}
        del messages
        audit = []
        before = sum(v.numel() * v.element_size() for v in peps.values())
        for a, b, side, reverse, axis, baxis in bp.edges(
            self.ctx.width, self.ctx.length
        ):
            p, q, record = bp.projectors(
                frozen[a, side], frozen[b, reverse], self.threshold
            )
            if record["rank_after"] < record["rank_before"]:
                peps[a] = bp.absorb(peps[a], p.to(self.ctx.device), axis)
                peps[b] = bp.absorb(peps[b], q.to(self.ctx.device), baxis)
            audit.append(dict(bond=[a, b], **record))
        tensors.update({site: value.unsqueeze(0) for site, value in peps.items()})
        synchronize()
        record = dict(
            call=self.counter,
            layer=self.layer_ids.get(id(gates)),
            storage_before_bytes=before,
            storage_after_bytes=sum(
                v.numel() * v.element_size() for v in peps.values()
            ),
            bp=info,
            compression=audit,
            seconds=time.perf_counter() - started,
        )
        with self.path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        self.counter += 1
        return invalidated | {
            (site, side)
            for site, directions in neighbors.items()
            for side in directions
        }

    def __enter__(self):
        core._apply_layer = self.apply
        return self

    def __exit__(self, *args):
        core._apply_layer = self.original_apply


def conditional(q):
    return (q[..., 0] - q[..., 1]) / q.sum(-1)


def correction(q0, delta):
    return (delta[..., 0] - delta[..., 1] - conditional(q0) * delta.sum(-1)) / q0.sum(
        -1
    )


def positive(state, n):
    # Independent Pauli evaluation; state may contain a Kraus-mode batch.
    state = state.reshape(-1, 2**n)
    paulis = torch.tensor(
        [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]],
        dtype=state.dtype,
        device=state.device,
    )
    output = []
    for site in range(n):
        mat = (
            state.reshape(-1, *([2] * n))
            .movedim(site + 1, 1)
            .reshape(state.shape[0], 2, -1)
        )
        rho = torch.einsum("boe,bpe->op", mat, mat.conj())
        trace = torch.trace(rho).real
        signal = torch.einsum("ab,kba->k", rho, paulis).real
        output.append(torch.stack(((trace + signal) / 2, (trace - signal) / 2), -1))
    return torch.stack(output)
