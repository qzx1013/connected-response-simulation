import math
import torch
import pytest
from residual_tn.peps import (
    model as p,
    norm_batch as norm,
    source_bank as b,
    continuation,
)


def branches():
    torch.manual_seed(20260905)
    rows = []
    for d in (2, 3):
        shapes = [(1, d, 1, d, 2), (1, d, d, 1, 2), (d, 1, 1, d, 2), (d, 1, d, 1, 2)]
        rows.append(
            {
                i: torch.randn((1, *s), dtype=torch.complex128)
                / math.sqrt(math.prod(s))
                for i, s in enumerate(shapes)
            }
        )
    return rows


def dense(branch):
    a, c, d, e = [branch[i][0] for i in range(4)]
    return torch.einsum("axbyi,czyej,xfghk,zmhwl->ijkl", a, c, d, e).reshape(-1)


def test_ragged_bp_and_read_against_serial_and_exact(tmp_path, monkeypatch):
    ctx = p.bp.ctx_for(2, 2, "cpu")
    cfg = p.config(gloop=4, bp_tol=1e-12, device="cpu")
    plan = p.core._build_plan(ctx, cfg)
    bank = branches()
    reference = []
    for branch in bank:
        m, z, _ = p.bp.solve({k: v[0] for k, v in branch.items()}, ctx, cfg)
        reference.append((m, z))
    messages, _ = norm.solve(bank, plan, cfg)
    assert (
        max(
            float((messages[i][k] - reference[i][0][k]).abs().max())
            for i in range(2)
            for k in messages[i]
        )
        < 1e-10
    )
    rows = [
        dict(branch=branch, messages=m, z=z.real.reshape(1))
        for branch, (m, z) in zip(bank, reference)
    ]
    serial = torch.cat([b.read_group(ctx, cfg, [row], (0, 1, 2, 3))[0] for row in rows])
    joined, _ = b.read_group(ctx, cfg, rows, (0, 1, 2, 3))
    assert float((serial - joined).abs().max()) < 1e-10
    exact = torch.stack(
        [p.conditional(p.positive(dense(branch), 4)) for branch in bank]
    )
    measured = p.conditional(p.core._pauli_positive_probabilities(joined))
    assert float((measured - exact).abs().max()) < 1e-9
    ordinary = b.read_group
    for exception in (
        torch.OutOfMemoryError("injected"),
        RuntimeError("no exact contraction plan fits the requested peak budget"),
    ):

        def injected(ctx, cfg, selected, sites):
            if len(selected) > 1:
                raise type(exception)(str(exception))
            return ordinary(ctx, cfg, selected, sites)

        monkeypatch.setattr(b, "read_group", injected)
        retried, records = b.read_rows(ctx, cfg, rows, (0, 1, 2, 3), tmp_path, 1)
        assert float((retried - serial).abs().max()) < 1e-10
        assert len(records) == 2


def test_continuation_matches_independently_stopped_bp():
    ctx = p.bp.ctx_for(2, 2, "cpu")
    cfg = p.config(gloop=4, bp_tol=1e-12, device="cpu")
    plan = p.core._build_plan(ctx, cfg)
    rows = [dict(branch=branch) for branch in branches()]
    clean = dict(peps={s: v[0] for s, v in rows[0]["branch"].items()})
    ket, bra, initial = continuation.mixed_inputs(ctx, plan, rows, clean)
    criteria = {
        "cap2": (1e-30, 2),
        "cap5": (1e-30, 5),
        "cap9": (1e-30, 9),
        "loose": (1e-2, 80),
        "strict": (1e-6, 80),
    }
    snapshots, infos = continuation.solve(
        ket, bra, {k: v.clone() for k, v in initial.items()}, plan, cfg, criteria
    )
    for name, (tol, cap) in criteria.items():
        independent, info = p.core._solve_bp(
            ket,
            bra,
            {k: v.clone() for k, v in initial.items()},
            plan,
            max_iter=cap,
            tol=tol,
            damping=cfg.bp_damping,
            residual_check_interval=cfg.bp_residual_check_interval,
            step_residual_gate_factor=cfg.bp_step_residual_gate_factor,
            reuse_opposite_cavities=cfg.reuse_opposite_bp_cavities,
            active_compaction_ratio=cfg.bp_active_compaction_ratio,
            allow_nonconverged=True,
        )
        assert (
            max(
                float((snapshots[name][k] - value).abs().max())
                for k, value in independent.items()
            )
            < 2e-11
        )
        assert info["iterations_per_branch"] == infos[name]["iterations_per_branch"]
