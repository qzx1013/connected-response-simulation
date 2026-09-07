"""Target-major generalized-loop fidelity banks: batched compression/mixed BP, bounded reads.

The source-anchored transition estimator and source-mode weights are unchanged.
All insertion readouts use generalized loops, not a 2x2 cluster closure. Ragged
branches live on CPU between phases. GPU padding is temporary, and each
branch keeps its own convergence test and bond projectors.
"""

from __future__ import annotations
import dataclasses
import math
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from residual_tn.peps import model as p
from residual_tn.peps import source_bank as b

EVOLUTION_CAP = 128
MIXED_CAP = 128
READ_CAP = 32


def schedule_identity(identity):
    return dict(
        identity,
        fidelity_schedule="target_major_ragged_v1",
        evolution_batch_cap=EVOLUTION_CAP,
        mixed_bp_batch_cap=MIXED_CAP,
        fidelity_read_batch_cap=READ_CAP,
        clean_cache_device="cpu",
        observable="fidelity_second_insertion_pairs",
        requested_source_layers=[0],
    )


def join_branches(rows, device):
    branches = [row["branch"] for row in rows]
    result = {}
    for site in branches[0]:
        target = [max(v[site].shape[a] for v in branches) for a in range(1, 6)]
        result[site] = torch.cat(
            [
                F.pad(
                    v[site].to(device),
                    tuple(
                        x
                        for a in reversed(range(5))
                        for x in (0, target[a] - v[site].shape[a + 1])
                    ),
                )
                for v in branches
            ]
        )
    return result


def pack_cpu_messages(values):
    # One device-to-host transfer, avoiding synchronization on each edge.
    keys = list(values)
    packed = torch.cat([values[k].reshape(-1) for k in keys]).cpu()
    result = {}
    offset = 0
    for key in keys:
        value = values[key]
        n = value.numel()
        result[key] = packed[offset : offset + n].reshape(value.shape)
        offset += n
    return result


def mixed_group(ctx, cfg, plan, rows, clean):
    ket = join_branches(rows, ctx.device)
    bra = clean["peps"]
    ref = next(iter(ket.values()))
    initial = {}
    for site, side in plan["directed_edges"]:
        axis = p.core.DIR_TO_AXIS[side]
        dk, db = ket[site].shape[axis + 1], bra[site].shape[axis]
        value = torch.zeros(len(rows), dk, db, dtype=ref.dtype, device=ctx.device)
        for i, row in enumerate(rows):
            actual = row["branch"][site].shape[axis + 1]
            # Padding must not add identity entries outside this branch's leg.
            value[i, :actual, :] = torch.eye(
                actual, db, dtype=ref.dtype, device=ctx.device
            ) / math.sqrt(min(actual, db))
        initial[site, side] = value
    messages, info = p.core._solve_bp(
        ket,
        bra,
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
    cpu = pack_cpu_messages(messages)
    cropped = []
    for i, row in enumerate(rows):
        cropped.append(
            {
                (site, side): value[
                    i, : row["branch"][site].shape[p.core.DIR_TO_AXIS[side] + 1], :
                ].contiguous()
                for (site, side), value in cpu.items()
            }
        )
    return cropped, info


def memory_error(error):
    return isinstance(error, torch.OutOfMemoryError) or any(
        s in str(error)
        for s in (
            "exceeds einsum memory limit",
            "no exact contraction plan fits",
            "resident tensors plus reserve already exceed",
            "bounded gloop needs ",
            "bounded gloop exhausted exact slicing retries",
        )
    )


def mixed_bank(ctx, cfg, plan, rows, clean, out, source, target):
    records = []

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
            messages, info = mixed_group(ctx, cfg, plan, group, clean)
        except RuntimeError as error:
            if not memory_error(error):
                raise
            error.__traceback__ = None
            error.__context__ = None
            error.__cause__ = None
            failed = True
        if failed:
            b.recover()
            b.emit(
                out, "mixed_oom_split", source=source, target=target, batch=len(group)
            )
            if len(group) == 1:
                raise RuntimeError("One mixed-BP branch exceeds the memory budget")
            mid = len(group) // 2
            chunk(group[:mid])
            chunk(group[mid:])
            return
        for i, row in enumerate(group):
            row["messages"] = messages[i]
        record = dict(
            target_layer=target,
            B=len(group),
            branch_ids=[r["id"] for r in group],
            init="cold_ragged_padded",
            iterations=info["iterations_per_branch"],
            residuals=info["residual_per_branch"],
        )
        records.append(record)
        b.emit(
            out,
            "mixed_bp",
            source=source,
            target=target,
            batch=len(group),
            seconds=time.perf_counter() - tick,
            bp=info,
            **b.memory(),
        )

    # Nearby shapes reduce wasted padding; identities preserve accumulation order.
    ordered = sorted(rows, key=lambda r: b.padded_storage([r["branch"]]))
    for start in range(0, len(ordered), MIXED_CAP):
        chunk(ordered[start : start + MIXED_CAP])
    return records


def read_group(ctx, cfg, plan, rows, clean, target):
    ket = join_branches(rows, ctx.device)
    messages = {}
    for site, side in rows[0]["messages"]:
        dk = ket[site].shape[p.core.DIR_TO_AXIS[side] + 1]
        messages[site, side] = torch.stack(
            [
                F.pad(
                    row["messages"][site, side].to(ctx.device),
                    (0, 0, 0, dk - row["messages"][site, side].shape[0]),
                )
                for row in rows
            ]
        )
    amplitude = torch.stack([r["amplitude"] for r in rows]).to(ctx.device)
    # Keep the established estimator. No per-branch warning ledger is retained
    # here; finite values, BP convergence and unresolved flags remain mandatory.
    num, den, bad = p.core._read_target_chunk(
        ket,
        clean,
        messages,
        plan["targets"][target],
        plan,
        source_amplitudes=amplitude,
        target_layer=target,
        gate_ids_override=plan.get("sampled_gate_ids", {}).get(target),
    )
    if bool(bad.any()) or not bool(
        torch.isfinite(num).all() & torch.isfinite(den).all()
    ):
        raise RuntimeError("Invalid or unresolved fidelity read")
    return p.cpu(num), p.cpu(den)


def read_bank(ctx, cfg, plan, rows, clean, out, source, target, identity):
    Path(out).mkdir(parents=True, exist_ok=True)
    path = Path(out) / f"read_source_{source:02}_target_{target:02}.pt"
    key = [(r["id"], r["gate"], r["mode"]) for r in rows]
    if path.exists():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert checkpoint["identity"] == identity and checkpoint["branch_keys"] == key
        results = checkpoint["results"]
    else:
        results = {}
    cap = 1
    stats = []

    def chunk(group):
        b.reset_peak()
        tick = time.perf_counter()
        failed = False
        try:
            num, den = read_group(ctx, cfg, plan, group, clean, target)
        except RuntimeError as error:
            if not memory_error(error):
                raise
            error.__traceback__ = None
            error.__context__ = None
            error.__cause__ = None
            failed = True
        if failed:
            b.recover()
            b.emit(
                out,
                "fidelity_read_oom_split",
                source=source,
                target=target,
                batch=len(group),
            )
            if len(group) == 1:
                raise RuntimeError("One fidelity read exceeds the memory budget")
            mid = len(group) // 2
            chunk(group[:mid])
            chunk(group[mid:])
            return
        for i, row in enumerate(group):
            results[row["id"]] = (num[i], den[i])
        record = dict(
            source=source,
            target=target,
            batch=len(group),
            seconds=time.perf_counter() - tick,
            **b.memory(),
        )
        stats.append(record)
        b.emit(out, "fidelity_read", **record)

    remaining = [r for r in rows if r["id"] not in results]
    while remaining:
        group = remaining[:cap]
        before = len(stats)
        chunk(group)
        p.save_tensor(path, dict(identity=identity, branch_keys=key, results=results))
        measured = stats[before:]
        # Grow from measured peak, not from the PEPS storage size. A layer with
        # new ranks starts with B=1; one CUDA stream bounds simultaneous readers.
        safe = min(
            max(1, int(56 / max(r["peak_allocated_gib"], 0.001) * r["batch"] / 1.3))
            for r in measured
        )
        cap = min(READ_CAP, max(1, min(cap * 2, safe)))
        remaining = remaining[len(group) :]
    return (
        torch.stack([results[r["id"]][0] for r in rows]),
        torch.stack([results[r["id"]][1] for r in rows]),
        stats,
    )


def make_bank(ctx, clean, source, cfg, plan):
    loaded = p.core._move_clean_layer_tensors(clean, ctx.device)
    records, filters = p.core._source_records(ctx, loaded, source, cfg)
    rows = []
    for tile in p.core._iter_source_tiles(
        records, loaded["peps"], config=cfg, neighbors=plan["neighbors"]
    ):
        local = int(tile["source_gate_local"][0])
        rows.append(
            dict(
                id=len(rows),
                gate=int(ctx.layers[source][local]["gate_idx"]),
                local=local,
                mode=int(tile["source_mode_local"][0]),
                amplitude=p.cpu(tile["source_amplitudes"][0]),
                branch=p.cpu(tile["tensors"]),
            )
        )
    return rows, filters


def source(ctx, identity, evolution, cfg, layers, plan, out, s, *, limit_branches=None):
    tick = time.perf_counter()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    progress = out / f"progress_source_{s:02}.pt"
    if progress.exists():
        state = torch.load(progress, map_location="cpu", weights_only=False)
        assert state["identity"] == identity
        rows, filters, pairs, bp_records, start = (
            state[k] for k in ("rows", "filters", "pairs", "bp_records", "next_target")
        )
    else:
        rows, filters = make_bank(ctx, layers[s], s, cfg, plan)
        if limit_branches is not None:
            rows = rows[:limit_branches]
        pairs = []
        bp_records = []
        start = s
    for target in range(start, len(ctx.layers)):
        if target > s:
            rows = b.evolve_bank(
                ctx,
                identity,
                evolution,
                cfg,
                plan,
                rows,
                s,
                0,
                out,
                through_layer=target,
            )
        clean = p.core._move_clean_layer_tensors(layers[target], ctx.device)
        bp_records.extend(mixed_bank(ctx, cfg, plan, rows, clean, out, s, target))
        num, den, stats = read_bank(
            ctx, cfg, plan, rows, clean, out, s, target, identity
        )
        del clean
        # Accumulate in the original source-tile order, independent of BP batches.
        totals = torch.zeros(
            len(ctx.layers[s]), len(ctx.layers[target]), dtype=torch.float64
        )
        for row, value in zip(rows, num):
            totals[row["local"]] += value
        fs = torch.tensor(
            [layers[s]["gate_data"][int(g["gate_idx"])]["F"] for g in ctx.layers[s]],
            dtype=torch.float64,
        )
        ft = torch.tensor(
            [
                layers[target]["gate_data"][int(g["gate_idx"])]["F"]
                for g in ctx.layers[target]
            ],
            dtype=torch.float64,
        )
        denominator = fs[:, None] * ft[None, :]
        assert bool(torch.isfinite(denominator).all() & (denominator > 1e-30).all())
        matrix = totals / denominator - 1.0
        assert bool(torch.isfinite(matrix).all())
        for a, source_gate in enumerate(ctx.layers[s]):
            for c, target_gate in enumerate(ctx.layers[target]):
                ga, gb = int(source_gate["gate_idx"]), int(target_gate["gate_idx"])
                if gb > ga and (
                    not plan.get("sampled_gate_ids")
                    or gb in plan["sampled_gate_ids"][target]
                ):
                    pairs.append(
                        dict(
                            source_gate=ga,
                            target_gate=gb,
                            target_layer=target,
                            delta=float(matrix[a, c]),
                        )
                    )
        for row in rows:
            row.pop("messages", None)
        p.save_tensor(
            progress,
            dict(
                identity=identity,
                rows=rows,
                filters=filters,
                pairs=pairs,
                bp_records=bp_records,
                next_target=target + 1,
            ),
        )
        b.emit(
            out, "fidelity_target_complete", source=s, target=target, branches=len(rows)
        )
    result = dict(
        identity=identity,
        source_layer=s,
        pairs=pairs,
        bp_records=bp_records,
        seconds=time.perf_counter() - tick,
        branch_count=len(rows),
        mode_filter=filters,
    )
    p.save(out / f"source_{s:02}.json", result)
    return result


def calculate(
    ctx, identity, evolution, cfg, out, source_layers=(0,), shared_layers=None
):
    assert cfg.local_readout_method == "gloop"
    assert (
        cfg.branch_weight_mode == "source_anchored"
        and cfg.quarantine_granularity == "branch"
    )
    assert (
        not cfg.auto_quarantine_ill_conditioned
        and cfg.source_amplitude_filter_tol == 0.0
    )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tick = time.perf_counter()
    b.EVOLUTION_CAP = EVOLUTION_CAP
    if shared_layers is None:
        layers, plan = b.background(ctx, identity, evolution, cfg, out)
    else:
        layers, plan = shared_layers
        b.emit(
            out,
            "background_reused",
            layers=len(layers),
            cache_sha256=identity["shared_background"]["cache_sha256"],
            background_group=identity["shared_background"]["group"],
        )
    if identity.get("target_sample") is not None:
        plan["sampled_gate_ids"] = {
            t: tuple(r["gate"] for r in identity["target_sample"] if r["layer"] == t)
            for t in range(len(ctx.layers))
        }
    one = {
        int(g["gate_idx"]): float(layers[i]["gate_data"][int(g["gate_idx"])]["F"])
        for i, gates in enumerate(ctx.layers)
        for g in gates
    }
    p.save(out / "background.json", dict(identity=identity, one_location_fidelity=one))
    pairs = []
    for s in source_layers:
        path = out / f"source_{s:02}.json"
        if path.exists():
            import json

            row = json.loads(path.read_text())
            assert row["identity"] == identity
        else:
            row = source(ctx, identity, evolution, cfg, layers, plan, out, s)
        pairs.extend(row["pairs"])
        b.emit(out, "fidelity_source_complete", source=s, seconds=row["seconds"])
        # Completed JSON is the permanent checkpoint; large branch banks are
        # only needed for the active source. Keep the background cache.
        for pattern in (
            f"progress_source_{s:02}.pt",
            f"evolved_source_{s:02}_wave_*.pt",
            f"read_source_{s:02}_target_*.pt",
        ):
            for file in out.glob(pattern):
                file.unlink()
    from residual_tn.experiments.fidelity_pairs import assemble

    result = assemble(ctx, one, pairs, source_layers, identity.get("target_sample"))
    result.update(identity=identity, seconds=time.perf_counter() - tick)
    p.save(out / "result.json", result)
    return result
