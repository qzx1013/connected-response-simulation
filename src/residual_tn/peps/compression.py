"""BP-weighted projectors from the validated sparse-residual experiment."""

import dataclasses, json, time
from pathlib import Path
from types import SimpleNamespace

pass  # Package imports need no path bootstrap.
import torch
from residual_tn.backend import single_site_bp_streamed_optimized as core


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def edges(width, length):
    for a in range(width * length):
        if a % width < width - 1:
            yield a, a + 1, "R", "L", 3, 2
        if a // width < length - 1:
            yield a, a + width, "D", "U", 1, 0


def projectors(left, right, tol, force_rank=None):
    roots = []
    metrics = []
    for m in (left, right):
        herm = float(torch.linalg.norm(m - m.mH) / torch.linalg.norm(m))
        e, v = torch.linalg.eigh((m + m.mH) * 0.5)
        assert herm < 1e-12 and e[-1] > 0 and e[0] >= -1e-12 * e[-1]
        roots.append((v * e.clamp_min(0).sqrt()) @ v.mH)
        metrics.append(
            {
                "eigen_min_max_ratio": float(e[0] / e[-1]),
                "negative_roundoff_clipped": int((e < 0).sum()),
            }
        )
    x, y = roots
    c = x.T @ y
    u, s, vh = torch.linalg.svd(c)
    tails = torch.cat(
        (s.square().flip(0).cumsum(0).flip(0)[1:], s.new_zeros(1))
    ).sqrt() / torch.linalg.norm(s)
    r = int(torch.nonzero(tails <= tol)[0, 0]) + 1 if force_rank is None else force_rank
    d = len(s)
    if r == d:
        p = q = torch.eye(d, dtype=c.dtype, device=c.device)
        mode = "unchanged_full_rank"
    else:
        if s[r - 1] <= 1e-12 * s[0]:
            raise RuntimeError(
                "Unresolved retained subspace; candidate aborted without silently dropping ranks"
            )
        p = (y @ vh[:r].mH) / s[:r].sqrt()
        q = (x @ u[:, :r].conj()) / s[:r].sqrt()
        mode = "retained_subspace"
    measured = float(torch.linalg.norm(x.T @ p @ q.T @ y - c) / torch.linalg.norm(c))
    assert abs(measured - float(tails[r - 1])) < 1e-10
    return (
        p,
        q,
        {
            "rank_before": d,
            "rank_after": r,
            "tol": tol,
            "predicted_local_tail": float(tails[r - 1]),
            "measured_weighted_error": measured,
            "metrics": metrics,
            "mode": mode,
            "insertion_operator_norm": float(torch.linalg.matrix_norm(p @ q.T, ord=2)),
        },
    )


def absorb(tensor, matrix, axis):
    # new[..., r, ...] = sum_d old[..., d, ...] * matrix[d,r].
    return (
        torch.tensordot(tensor, matrix, dims=([axis], [0]))
        .movedim(-1, axis)
        .contiguous()
    )


def ctx_for(width, length, device):
    # These calls require geometry/dtype only; no circuit is regenerated.
    return SimpleNamespace(
        width=width,
        length=length,
        n_qubits=width * length,
        dtype=torch.complex128,
        device=torch.device(device),
        layers=[[]],
    )


def configuration(**overrides):
    cfg = core.SingleSiteConfig(
        chi_max=64,
        branch_batch_size=1,
        bp_branch_chunk_size=1,
        readout_branch_chunk_size=1,
        readout_site_chunk_size=1,
        bethe_site_batch_size=1,
        bp_max_iter=500,
        bp_tol=1e-10,
        bp_damping=0.5,
        bp_residual_check_interval=4,
        branch_weight_mode="source_anchored",
        local_readout_method="gloop",
        gloop_size=4,
        gloop_optimize="greedy",
        gloop_expression_cache_size=32768,
        gloop_memory_target_gib=8.0,
        gloop_memory_search_repeats=64,
        gloop_memory_max_slices=4096,
        gloop_peak_budget_gib=60.0,
        gloop_peak_reserve_gib=8.0,
        gloop_peak_safety_factor=2.0,
        gloop_oom_max_retries=2,
        readout_boundary_mode="single_site",
        einsum_memory_gib=16.0,
        retain_pair_ledger=False,
    )
    cfg = dataclasses.replace(cfg, **overrides)
    cfg.validate()
    return cfg


def solve(peps, ctx, cfg):
    start = time.perf_counter()
    plan = core._build_plan(ctx, cfg)
    batched = {k: v.unsqueeze(0) for k, v in peps.items()}
    initial, label = core._initialize_messages(
        batched, peps, plan, warm=None, clean=None
    )
    solved, info = core._solve_bp(
        batched,
        peps,
        initial,
        plan,
        max_iter=cfg.bp_max_iter,
        tol=cfg.bp_tol,
        damping=cfg.bp_damping,
        residual_check_interval=cfg.bp_residual_check_interval,
        step_residual_gate_factor=cfg.bp_step_residual_gate_factor,
        reuse_opposite_cavities=cfg.reuse_opposite_bp_cavities,
        active_compaction_ratio=cfg.bp_active_compaction_ratio,
    )
    z = core._bethe_factors(batched, peps, solved, plan)["Z"][0]
    if ctx.device.type == "cuda":
        torch.cuda.synchronize()
    return (
        {k: v[0] for k, v in solved.items()},
        z,
        {**info, "init": label, "seconds": time.perf_counter() - start},
    )


def readout(peps, messages, z, ctx, cfg, size, sites):
    cfg = dataclasses.replace(cfg, gloop_size=size)
    plan = core._build_plan(ctx, cfg)
    if ctx.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    rho = core._pauli_read_local_rhos(
        {k: v.unsqueeze(0) for k, v in peps.items()},
        peps,
        {k: v.unsqueeze(0) for k, v in messages.items()},
        plan,
        z.real.reshape(1),
        ctx.n_qubits,
        cfg,
        site_indices=sites,
    )[0]
    q = core._pauli_positive_probabilities(rho)
    if ctx.device.type == "cuda":
        torch.cuda.synchronize()
    return {
        "gloop": size,
        "site_indices": list(sites),
        "ideal": core._summarize_pauli_probabilities(q),
        "raw_q0": q.detach().cpu().tolist(),
        "seconds": time.perf_counter() - start,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3
        if ctx.device.type == "cuda"
        else 0,
        "gloop_audit": dict(plan["gloop_audit"]),
    }
