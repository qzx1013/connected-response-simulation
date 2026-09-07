"""Save independently stopped BP solutions along one shared sweep trajectory.

Each branch/criterion gets its first converged snapshot, or its cap snapshot.
Further iterations never overwrite an earlier snapshot. Contractions and
phase/normalization conventions are the frozen core's Gauss-Seidel update.
"""

import math
import time
import torch
from residual_tn.peps import model as p
from residual_tn.peps import fidelity_batch as f
from residual_tn.peps import source_bank as b


def solve(ket, bra, initial, plan, cfg, criteria):
    core = p.core
    messages = dict(initial)
    count = next(iter(ket.values())).shape[0]
    device = next(iter(ket.values())).device
    pending = {
        name: torch.ones(count, dtype=torch.bool, device=device) for name in criteria
    }
    outputs = {
        name: {
            k: torch.empty(v.shape, dtype=v.dtype, device="cpu")
            for k, v in messages.items()
        }
        for name in criteria
    }
    infos = {
        name: dict(
            iterations_per_branch=[0] * count,
            residual_per_branch=[None] * count,
            raw_map_residual_per_branch=[None] * count,
            step_residual_per_branch=[None] * count,
            converged_per_branch=[False] * count,
        )
        for name in criteria
    }
    site_plan = core._bp_site_plan(plan)
    bra_conj = {site: value.conj() for site, value in bra.items()}
    maximum = max(cap for tol, cap in criteria.values())
    interval = cfg.bp_residual_check_interval
    factor = cfg.bp_step_residual_gate_factor
    checks = 0
    for iteration in range(1, maximum + 1):
        active = torch.stack(list(pending.values())).any(dim=0)
        step = torch.zeros(count, dtype=torch.float64, device=device)
        bad = torch.zeros(count, dtype=torch.bool, device=device)
        for site, entries in site_plan:
            raw_by_key = core._contract_bp_site_messages(
                ket,
                bra_conj,
                messages,
                site,
                entries,
                plan,
                reuse_opposite_cavities=cfg.reuse_opposite_bp_cavities,
            )
            for _, direction, _, _ in entries:
                key = site, direction
                old = messages[key]
                raw, bad_raw, _ = core._normalize_message(raw_by_key[key])
                raw = core._phase_align(raw, old)
                updated, bad_updated, _ = core._normalize_message(
                    (1 - cfg.bp_damping) * raw + cfg.bp_damping * old
                )
                delta = (
                    (updated - old)
                    .abs()
                    .reshape(count, -1)
                    .max(dim=1)
                    .values.to(torch.float64)
                )
                step = torch.maximum(
                    step, torch.where(active, delta, torch.zeros_like(step))
                )
                messages[key] = torch.where(active.reshape(count, 1, 1), updated, old)
                bad |= (bad_raw | bad_updated) & active
        if bool(bad.any()):
            raise RuntimeError("Non-finite continued mixed BP update")
        eligible = {}
        for name, (tol, cap) in criteria.items():
            if iteration > cap or not (
                iteration == 1 or iteration % interval == 0 or iteration == cap
            ):
                continue
            mask = pending[name].clone()
            if iteration != cap and factor is not None:
                mask &= step <= factor * tol
            eligible[name] = mask
        if not eligible:
            continue
        union = torch.stack(list(eligible.values())).any(dim=0)
        positions = torch.nonzero(union, as_tuple=False).flatten()
        if not positions.numel():
            continue
        checks += 1
        if positions.numel() == count:
            subket, submessages = ket, messages
        else:
            subket = {k: v.index_select(0, positions) for k, v in ket.items()}
            submessages = {k: v.index_select(0, positions) for k, v in messages.items()}
        residual, raw_residual, invalid = core._strict_bp_map_residual(
            subket,
            bra_conj,
            submessages,
            site_plan,
            plan,
            reuse_opposite_cavities=cfg.reuse_opposite_bp_cavities,
        )
        if bool(invalid.any()):
            raise RuntimeError("Invalid continued mixed BP strict residual")
        projective = torch.full(
            (count,), float("inf"), dtype=torch.float64, device=device
        )
        rawmap = torch.full_like(projective, float("inf"))
        projective.index_copy_(0, positions, residual)
        rawmap.index_copy_(0, positions, raw_residual)
        for name, mask in eligible.items():
            tol, cap = criteria[name]
            accepted = mask & ((projective < tol) | (iteration == cap))
            ids = torch.nonzero(accepted, as_tuple=False).flatten()
            if not ids.numel():
                continue
            cpuids = ids.cpu()
            # Pack once per snapshot rather than copy each edge separately.
            saved = f.pack_cpu_messages(
                {k: v.index_select(0, ids) for k, v in messages.items()}
            )
            for k, v in saved.items():
                outputs[name][k].index_copy_(0, cpuids, v)
            r = projective.index_select(0, ids).cpu().tolist()
            raw = rawmap.index_select(0, ids).cpu().tolist()
            steps = step.index_select(0, ids).cpu().tolist()
            for j, idx in enumerate(cpuids.tolist()):
                info = infos[name]
                info["iterations_per_branch"][idx] = iteration
                info["residual_per_branch"][idx] = r[j]
                info["raw_map_residual_per_branch"][idx] = raw[j]
                info["step_residual_per_branch"][idx] = steps[j]
                info["converged_per_branch"][idx] = r[j] < tol
            pending[name] &= ~accepted
        if not bool(torch.stack(list(pending.values())).any()):
            break
    assert not any(bool(mask.any()) for mask in pending.values())
    for name, info in infos.items():
        info.update(
            iterations=max(info["iterations_per_branch"]),
            shared_sweeps=iteration,
            shared_strict_checks=checks,
            projective_raw_residual_per_branch=info["residual_per_branch"],
            failed_positions=[
                i for i, v in enumerate(info["converged_per_branch"]) if not v
            ],
        )
    return outputs, infos


def mixed_inputs(ctx, plan, rows, clean):
    ket = f.join_branches(rows, ctx.device)
    bra = clean["peps"]
    ref = next(iter(ket.values()))
    initial = {}
    for site, side in plan["directed_edges"]:
        axis = p.core.DIR_TO_AXIS[side]
        dk, db = ket[site].shape[axis + 1], bra[site].shape[axis]
        value = torch.zeros(len(rows), dk, db, dtype=ref.dtype, device=ctx.device)
        for i, row in enumerate(rows):
            actual = row["branch"][site].shape[axis + 1]
            value[i, :actual, :] = torch.eye(
                actual, db, dtype=ref.dtype, device=ctx.device
            ) / math.sqrt(min(actual, db))
        initial[site, side] = value
    return ket, bra, initial


def mixed_bank(ctx, cfg, plan, rows, clean, out, target, criteria):
    records = {name: [] for name in criteria}

    def attempt(group):
        ket, bra, initial = mixed_inputs(ctx, plan, group, clean)
        return solve(ket, bra, initial, plan, cfg, criteria)

    def chunk(group):
        if (
            len(group) > 1
            and b.padded_storage([r["branch"] for r in group]) * 12
            > b.available_budget()
        ):
            mid = len(group) // 2
            chunk(group[:mid])
            chunk(group[mid:])
            return
        b.reset_peak()
        tick = time.perf_counter()
        failed = False
        try:
            snapshots, infos = attempt(group)
        except RuntimeError as error:
            if not f.memory_error(error):
                raise
            error.__traceback__ = None
            error.__context__ = None
            error.__cause__ = None
            failed = True
        if failed:
            b.recover()
            b.emit(out, "continued_bp_oom_split", target=target, batch=len(group))
            if len(group) == 1:
                raise RuntimeError(
                    "One continued mixed-BP branch exceeds memory budget"
                )
            mid = len(group) // 2
            chunk(group[:mid])
            chunk(group[mid:])
            return
        for i, row in enumerate(group):
            row["snapshots"] = {
                name: {
                    (site, side): value[
                        i, : row["branch"][site].shape[p.core.DIR_TO_AXIS[side] + 1], :
                    ].contiguous()
                    for (site, side), value in messages.items()
                }
                for name, messages in snapshots.items()
            }
        for name, info in infos.items():
            records[name].append(
                dict(
                    target_layer=target,
                    B=len(group),
                    branch_ids=[r["id"] for r in group],
                    iterations=info["iterations_per_branch"],
                    residuals=info["residual_per_branch"],
                    converged=info["converged_per_branch"],
                    init="single_cold_trajectory_snapshots",
                )
            )
        b.emit(
            out,
            "continued_mixed_bp",
            target=target,
            batch=len(group),
            seconds=time.perf_counter() - tick,
            shared_sweeps=max(info["shared_sweeps"] for info in infos.values()),
            snapshot_max_iterations={
                name: info["iterations"] for name, info in infos.items()
            },
            **b.memory(),
        )

    ordered = sorted(rows, key=lambda r: b.padded_storage([r["branch"]]))
    for start in range(0, len(ordered), f.MIXED_CAP):
        chunk(ordered[start : start + f.MIXED_CAP])
    return records
