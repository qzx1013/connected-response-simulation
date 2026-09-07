"""Source-layer banks with independent adaptive ranks and memory-bounded reads.

The dominant compression norm BP is batched (up to 128); gate/MPO updates and
projectors keep their branch-local order. CPU offload separates live storage
from GPU workspace. Terminal reads run on one stream, initially B=1 and then
adapt to measured peaks. No exact result enters this computation.
"""

from __future__ import annotations
import dataclasses
import builtins
import gc
import json
import math
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from residual_tn.peps import model as p
from residual_tn.peps import norm_batch as norm

GIB = 1024**3
EVOLUTION_CAP = 128
SOURCE_WAVE_CAP = 128  # Stable checkpoint membership; execution batch may change.
READ_CAP = 16
WORKSPACE_CAP_GIB = 64.0
HEADROOM_GIB = 6.0
READ_GROWTH_TARGET_GIB = 56.0


def configure_logging():
    # Keep retries/errors and audit counters; avoid per-expression flushes.
    from residual_tn.backend import bounded_gloop as bounded_gloop

    def brief_print(*args, **kwargs):
        if args and str(args[0]).startswith("[bounded-gloop-plan]"):
            return
        builtins.print(*args, **kwargs)

    bounded_gloop.print = brief_print


def gpu(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: gpu(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(gpu(v, device) for v in value)
    return value


def memory():
    if not torch.cuda.is_initialized():
        return dict(allocated_gib=0.0, peak_allocated_gib=0.0, peak_reserved_gib=0.0)
    return dict(
        allocated_gib=torch.cuda.memory_allocated() / GIB,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / GIB,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / GIB,
    )


def reset_peak():
    p.synchronize()
    if torch.cuda.is_initialized():
        torch.cuda.reset_peak_memory_stats()


def recover():
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def emit(out, kind, **row):
    row = dict(event=kind, **row)
    with (Path(out) / "batch_resources.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    brief = {
        k: v
        for k, v in row.items()
        if k not in ("bp", "compression", "ranks", "gloop_audit")
    }
    print("[source-bank] " + json.dumps(brief), flush=True)


def padded_storage(branches):
    ref = next(iter(branches[0].values()))
    return (
        len(branches)
        * ref.element_size()
        * sum(
            math.prod(max(b[s].shape[a] for b in branches) for a in range(1, 6))
            for s in branches[0]
        )
    )


def available_budget():
    if not torch.cuda.is_initialized():
        return float("inf")
    free, total = torch.cuda.mem_get_info()
    # Include reusable allocator blocks, preserving 6 GiB of physical headroom.
    reusable = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    return max(0.0, min(WORKSPACE_CAP_GIB * GIB, free + reusable - HEADROOM_GIB * GIB))


def solve_bank(branches, plan, cfg, out, tag, warm=None):
    """Retry only norm work; inputs remain on CPU and cannot be half-updated."""

    def attempt(selected):
        device = torch.device(
            plan.get("_batch80_device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        live = [gpu(v, device) for v in selected]
        messages, info = norm.solve(live, plan, cfg, warm=warm)
        # Pack all small messages into a single device-to-host transfer.
        keys = [list(v) for v in messages]
        sizes = [[v[k].numel() for k in ks] for v, ks in zip(messages, keys)]
        packed = torch.cat(
            [v[k].reshape(-1) for v, ks in zip(messages, keys) for k in ks]
        ).cpu()
        offset = 0
        result = []
        for v, ks, ns in zip(messages, keys, sizes):
            row = {}
            for k, n in zip(ks, ns):
                row[k] = packed[offset : offset + n].reshape(v[k].shape)
                offset += n
            result.append(row)
        return result, info

    def chunk(selected):
        estimate = padded_storage(selected)
        if len(selected) > 1 and estimate * 12 > available_budget():
            middle = len(selected) // 2
            emit(
                out,
                "bp_admission_split",
                **tag,
                requested_batch=len(selected),
                padded_peps_gib=estimate / GIB,
            )
            return chunk(selected[:middle]) + chunk(selected[middle:])
        reset_peak()
        tick = time.perf_counter()
        failed = False
        try:
            values, info = attempt(selected)
        except torch.OutOfMemoryError as error:
            error.__traceback__ = None
            error.__context__ = None
            error.__cause__ = None
            failed = True
        if failed:
            recover()
            emit(out, "bp_oom_split", **tag, requested_batch=len(selected), **memory())
            if len(selected) == 1:
                raise RuntimeError("One BP branch exceeds the GPU memory limit")
            middle = len(selected) // 2
            return chunk(selected[:middle]) + chunk(selected[middle:])
        emit(
            out,
            "norm_bp",
            **tag,
            actual_batch=len(selected),
            seconds=time.perf_counter() - tick,
            bp=info,
            **memory(),
        )
        return values

    result = []
    for start in range(0, len(branches), EVOLUTION_CAP):
        result.extend(chunk(branches[start : start + EVOLUTION_CAP]))
    return result


def background(ctx, identity, evolution, cfg, out):
    configure_logging()
    reset_peak()
    tick = time.perf_counter()
    path = Path(out) / "background_layers.pt"
    if path.exists():
        data = torch.load(path, map_location="cpu", weights_only=False)
        assert data["identity"] == identity
        return data["layers"], p.core._build_plan(ctx, cfg)
    cache = p.core.build_clean_cache(
        ctx, cfg, background_layer_hook=evolution.residual_hook, offload_device="cpu"
    )
    p.core._extend_clean_cache(ctx, cache, len(ctx.layers) - 1)
    p.save_tensor(path, dict(identity=identity, layers=cache["layers"]))
    emit(
        out,
        "background_saved",
        layers=len(cache["layers"]),
        seconds=time.perf_counter() - tick,
        **memory(),
    )
    return cache["layers"], cache["plan"]


def make_bank(ctx, clean, source, cfg, plan):
    loaded = p.core._move_clean_layer_tensors(clean, torch.device(ctx.device))
    records, filters = p.core._source_records(ctx, loaded, source, cfg)
    rows = []
    for tile in p.core._iter_source_tiles(
        records, loaded["peps"], config=cfg, neighbors=plan["neighbors"]
    ):
        assert tile["branch_ids"].numel() == 1
        local = int(tile["source_gate_local"][0])
        mode = int(tile["source_mode_local"][0])
        rows.append(
            dict(
                gate=int(ctx.layers[source][local]["gate_idx"]),
                mode=mode,
                branch=p.cpu(tile["tensors"]),
            )
        )
    return rows, filters


def evolve_bank(
    ctx, identity, evolution, cfg, plan, rows, source, wave, out, through_layer=None
):
    path = Path(out) / f"evolved_source_{source:02}_wave_{wave:02}.pt"
    start = source + 1
    if path.exists():
        saved = torch.load(path, weights_only=False)
        assert saved["identity"] == identity
        assert [(v["gate"], v["mode"]) for v in rows] == saved["branch_keys"]
        rows = saved["rows"]
        start = saved["through_layer"] + 1
    branch_keys = [(v["gate"], v["mode"]) for v in rows]
    end = len(ctx.layers) - 1 if through_layer is None else through_layer
    for layer in range(start, end + 1):
        reset_peak()
        tick = time.perf_counter()
        # Keep the ragged source bank on CPU while individual gate kernels run.
        for row in rows:
            branch = gpu(row["branch"], ctx.device)
            evolution.residual_hook(
                branch, layer, config=cfg, neighbors=plan["neighbors"]
            )
            evolution.original_apply(
                branch, ctx.layers[layer], config=cfg, neighbors=plan["neighbors"]
            )
            row["branch"] = p.cpu(branch)
            del branch
        gate_seconds = time.perf_counter() - tick
        gate_peak = memory()
        tick = time.perf_counter()
        if evolution.threshold is None:
            p.save_tensor(
                path,
                dict(
                    identity=identity,
                    source=source,
                    wave=wave,
                    through_layer=layer,
                    branch_keys=branch_keys,
                    rows=rows,
                ),
            )
            emit(
                out,
                "evolution_layer",
                source=source,
                wave=wave,
                layer=layer,
                branches=len(rows),
                gates_seconds=gate_seconds,
                gates_peak=gate_peak,
                bp_seconds=0.0,
                compression_disabled=True,
                **memory(),
            )
            continue
        messages = solve_bank(
            [v["branch"] for v in rows],
            plan,
            cfg,
            out,
            dict(source=source, wave=wave, layer=layer),
        )
        bp_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        ranks = []
        tails = []
        for row, frozen in zip(rows, messages):
            branch = gpu(row["branch"], ctx.device)
            rank = []
            for a, b, side, reverse, axis, baxis in p.bp.edges(ctx.width, ctx.length):
                left, right, record = p.bp.projectors(
                    frozen[a, side], frozen[b, reverse], evolution.threshold
                )
                if record["rank_after"] < record["rank_before"]:
                    branch[a] = p.bp.absorb(
                        branch[a][0], left.to(ctx.device), axis
                    ).unsqueeze(0)
                    branch[b] = p.bp.absorb(
                        branch[b][0], right.to(ctx.device), baxis
                    ).unsqueeze(0)
                rank.append(record["rank_after"])
                tails.append(record["measured_weighted_error"])
            row["branch"] = p.cpu(branch)
            del branch
            ranks.append(rank)
        del messages
        p.save_tensor(
            path,
            dict(
                identity=identity,
                source=source,
                wave=wave,
                through_layer=layer,
                branch_keys=branch_keys,
                rows=rows,
            ),
        )
        emit(
            out,
            "evolution_layer",
            source=source,
            wave=wave,
            layer=layer,
            branches=len(rows),
            gates_seconds=gate_seconds,
            gates_peak=gate_peak,
            bp_seconds=bp_seconds,
            projection_and_checkpoint_seconds=time.perf_counter() - tick,
            ranks=ranks,
            max_weighted_tail=max(tails, default=0.0),
            **memory(),
        )
    if not path.exists():
        p.save_tensor(
            path,
            dict(
                identity=identity,
                source=source,
                wave=wave,
                through_layer=end,
                branch_keys=branch_keys,
                rows=rows,
            ),
        )
    return rows


def terminal_bank(ctx, cfg, plan, rows, clean_messages, out, source, wave):
    messages = solve_bank(
        [v["branch"] for v in rows],
        plan,
        cfg,
        out,
        dict(source=source, wave=wave, layer="terminal"),
        warm=clean_messages,
    )
    for row, values in zip(rows, messages):
        branch = gpu(row["branch"], ctx.device)
        msg = {k: v.to(ctx.device).unsqueeze(0) for k, v in values.items()}
        # Preserve the old reader's exact contraction convention and norm.
        z = p.core._bethe_factors(
            branch, {k: v[0] for k, v in branch.items()}, msg, plan
        )["Z"].real.to(torch.float64)
        row.update(messages=values, z=p.cpu(z))
        del branch, msg, z
    return rows


def join_read(rows, device):
    branches = [v["branch"] for v in rows]
    joined = {}
    for site in branches[0]:
        target = [max(b[site].shape[a] for b in branches) for a in range(1, 6)]
        joined[site] = torch.cat(
            [
                F.pad(
                    b[site].to(device),
                    tuple(
                        x
                        for a in reversed(range(5))
                        for x in (0, target[a] - b[site].shape[a + 1])
                    ),
                )
                for b in branches
            ]
        )
    messages = {}
    for key in rows[0]["messages"]:
        site, side = key
        d = joined[site].shape[p.core.DIR_TO_AXIS[side] + 1]
        messages[key] = torch.stack(
            [
                F.pad(
                    row["messages"][key].to(device),
                    (
                        0,
                        d - row["messages"][key].shape[1],
                        0,
                        d - row["messages"][key].shape[0],
                    ),
                )
                for row in rows
            ]
        )
    z = torch.cat([row["z"] for row in rows]).to(device)
    return joined, messages, z


def read_group(ctx, cfg, rows, sites):
    ket, messages, z = join_read(rows, ctx.device)
    read_cfg = dataclasses.replace(
        cfg,
        readout_branch_chunk_size=len(rows),
        readout_branch_stream_count=1,
        readout_site_chunk_size=1,
    )
    plan = p.core._build_plan(ctx, read_cfg)
    rho = p.core._pauli_read_local_rhos(
        ket, ket, messages, plan, z, ctx.n_qubits, read_cfg, site_indices=sites
    )
    result = p.cpu(rho)
    p.synchronize()
    return result, dict(plan.get("gloop_audit", {}))


def read_rows(
    ctx,
    cfg,
    rows,
    sites,
    out,
    source,
    prior_peak=0.0,
    prior_batch=1,
    checkpoint_path=None,
    identity=None,
):
    """One CUDA stream; tentative growth, padded-shape admission, atomic OOM split."""
    cap = 1 if source == 0 else min(READ_CAP, max(2, prior_batch * 2))
    # A 1.5x safety factor limits growth from the measured prior source peak.
    if source > 0 and prior_peak > 0:
        cap = min(
            cap, max(1, int(READ_GROWTH_TARGET_GIB / (1.5 * prior_peak) * prior_batch))
        )
    records = []
    results = []
    branch_keys = [(row.get("gate"), row.get("mode")) for row in rows]
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert (
            saved["identity"] == identity
            and saved["branch_keys"] == branch_keys
            and saved["sites"] == list(sites)
        )
        results = list(saved["results"])
        records = saved["records"]

    def commit():
        if checkpoint_path is not None:
            p.save_tensor(
                checkpoint_path,
                dict(
                    identity=identity,
                    branch_keys=branch_keys,
                    sites=list(sites),
                    results=results,
                    records=records,
                ),
            )

    def chunk(group, requested):
        estimate = padded_storage([v["branch"] for v in group])
        if len(group) > 1 and estimate * 8 > available_budget():
            middle = len(group) // 2
            return chunk(group[:middle], requested) + chunk(group[middle:], requested)
        reset_peak()
        tick = time.perf_counter()
        failed = False
        try:
            rho, audit = read_group(ctx, cfg, group, sites)
        except (torch.OutOfMemoryError, RuntimeError) as error:
            message = str(error)
            known_memory_failure = isinstance(error, torch.OutOfMemoryError) or any(
                token in message
                for token in (
                    "resident tensors plus reserve already exceed memory budget",
                    "bounded gloop needs ",
                    "no exact contraction plan fits the requested peak budget",
                    "bounded gloop exhausted exact slicing retries",
                )
            )
            if not known_memory_failure:
                raise
            error.__traceback__ = None
            error.__context__ = None
            error.__cause__ = None
            failed = True
        if failed:
            recover()
            emit(
                out,
                "read_oom_split",
                source=source,
                requested_batch=len(group),
                **memory(),
            )
            if len(group) == 1:
                raise RuntimeError(
                    "One Pauli read exceeds the GPU budget, even with sliced gloop"
                )
            middle = len(group) // 2
            return chunk(group[:middle], requested) + chunk(group[middle:], requested)
        row = dict(
            source=source,
            requested_batch=requested,
            actual_batch=len(group),
            seconds=time.perf_counter() - tick,
            gloop_audit=audit,
            effective_branch_chunk=min(
                len(group), audit.get("oom_retry_min_branch_chunk") or len(group)
            ),
            **memory(),
        )
        records.append(row)
        emit(out, "pauli_read", **row)
        return list(rho.unbind(0))

    start = len(results)
    while start < len(rows):
        selected = rows[start : start + cap]
        first_record = len(records)
        results.extend(chunk(selected, cap))
        commit()
        start += len(selected)
        measured = records[first_record:]
        safe = min(
            max(
                1,
                int(
                    READ_GROWTH_TARGET_GIB
                    / (1.5 * max(row["peak_allocated_gib"], 0.001))
                    * row["actual_batch"]
                ),
            )
            for row in measured
        )
        cap = min(READ_CAP, max(1, min(cap * 2, safe)))
    return torch.stack(results), records


def assemble(ctx, identity, q0, rows):
    ids = [gid for row in rows for gid in row["source_gate_ids"]]
    qg = torch.tensor([q for row in rows for q in row["qg"]], dtype=torch.float64)
    delta = (qg - q0).sum(0)
    layers = []
    for row in rows:
        d = (torch.tensor(row["qg"], dtype=torch.float64) - q0).sum(0)
        layers.append(
            dict(
                source_layer=row["source_layer"],
                delta_q=d.tolist(),
                delta_pauli=p.correction(q0, d).tolist(),
            )
        )
    return dict(
        identity=identity,
        raw=dict(
            site_indices=list(range(ctx.n_qubits)),
            source_layers=[row["source_layer"] for row in rows],
            source_gate_ids=ids,
            q0=q0.tolist(),
            qg=qg.tolist(),
            q1=(q0 + delta).tolist(),
        ),
        definition=dict(
            connected_order=1,
            source_operator="complete relative channel K U_dagger",
            assembly="sum unnormalized Kraus effects per gate, subtract q0 once per gate",
            terminal_readout="gloop",
            gloop_size=cfg_gloop(identity),
            validation_subset=len(rows) != len(ctx.layers),
        ),
        source_layer_schedule=[
            {k: v for k, v in row.items() if k not in ("qg", "identity")}
            for row in rows
        ],
        per_source_layer=layers,
        ideal_pauli=p.conditional(q0).tolist(),
        additive_first_order_pauli=p.conditional(q0 + delta).tolist(),
        strict_first_order_pauli=(p.conditional(q0) + p.correction(q0, delta)).tolist(),
    )


def cfg_gloop(identity):
    return identity.get("numerics", {}).get("gloop_size", 8)


def calculate(ctx, identity, evolution, cfg, out, source_layers=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    assert cfg.branch_batch_size == 1 and cfg.source_amplitude_filter_tol == 0.0
    selected_sources = (
        list(range(len(ctx.layers))) if source_layers is None else list(source_layers)
    )
    assert selected_sources and selected_sources == sorted(set(selected_sources))
    assert all(0 <= s < len(ctx.layers) for s in selected_sources)
    layers, plan = background(ctx, identity, evolution, cfg, out)
    qpath = out / "background.json"
    if qpath.exists():
        saved = json.loads(qpath.read_text())
        assert saved["identity"] == identity
        q0 = torch.tensor(saved["raw_q0"], dtype=torch.float64)
    else:
        reset_peak()
        tick = time.perf_counter()
        final = p.core._move_clean_layer_tensors(
            layers[len(ctx.layers) - 1], ctx.device
        )
        rho = p.core._pauli_read_local_rhos(
            {k: v.unsqueeze(0) for k, v in final["peps"].items()},
            final["peps"],
            {k: v.unsqueeze(0) for k, v in final["messages"].items()},
            plan,
            final["Z_clean"].real.reshape(1),
            ctx.n_qubits,
            cfg,
        )[0]
        q0 = p.cpu(p.core._pauli_positive_probabilities(rho))
        del final, rho
        emit(
            out, "background_pauli_read", seconds=time.perf_counter() - tick, **memory()
        )
        p.save(
            qpath,
            dict(
                identity=identity,
                raw_q0=q0.tolist(),
                ideal_pauli=p.conditional(q0).tolist(),
            ),
        )
    completed = []
    prior_peak = 0.0
    prior_batch = 1
    for source in selected_sources:
        spath = out / f"source_{source:02}.json"
        if spath.exists():
            row = json.loads(spath.read_text())
            assert row["identity"] == identity
        else:
            tick = time.perf_counter()
            bank, filters = make_bank(ctx, layers[source], source, cfg, plan)
            gate_ids = [int(g["gate_idx"]) for g in ctx.layers[source]]
            totals = {
                gid: torch.zeros(ctx.n_qubits, 2, 2, dtype=ctx.dtype)
                for gid in gate_ids
            }
            reads = []
            for start in range(0, len(bank), SOURCE_WAVE_CAP):
                wave = start // SOURCE_WAVE_CAP
                rows = evolve_bank(
                    ctx,
                    identity,
                    evolution,
                    cfg,
                    plan,
                    bank[start : start + SOURCE_WAVE_CAP],
                    source,
                    wave,
                    out,
                )
                rows = terminal_bank(
                    ctx,
                    cfg,
                    plan,
                    rows,
                    layers[len(ctx.layers) - 1]["messages"],
                    out,
                    source,
                    wave,
                )
                # Checkpoint per wave read; resume never double-counts partial output.
                rpath = out / f"read_source_{source:02}_wave_{wave:02}.pt"
                if rpath.exists():
                    r = torch.load(rpath, weights_only=False)
                    assert r["identity"] == identity
                    rhos, stats = r["rhos"], r["stats"]
                else:
                    rhos, stats = read_rows(
                        ctx,
                        cfg,
                        rows,
                        tuple(range(ctx.n_qubits)),
                        out,
                        source,
                        prior_peak,
                        prior_batch,
                        checkpoint_path=out
                        / f"read_progress_source_{source:02}_wave_{wave:02}.pt",
                        identity=identity,
                    )
                    p.save_tensor(
                        rpath, dict(identity=identity, rhos=rhos, stats=stats)
                    )
                for branch, rho in zip(rows, rhos):
                    totals[branch["gate"]] += rho
                reads.extend(stats)
                del rows, rhos
            qg = p.core._pauli_positive_probabilities(
                torch.stack([totals[gid] for gid in gate_ids])
            )
            assert torch.isfinite(qg).all()
            row = dict(
                identity=identity,
                source_layer=source,
                source_gate_ids=gate_ids,
                qg=qg.tolist(),
                retained_mode_count=len(bank),
                mode_filter=filters,
                seconds=time.perf_counter() - tick,
                read_peak_allocated_gib=max(v["peak_allocated_gib"] for v in reads),
                read_max_actual_batch=max(v["actual_batch"] for v in reads),
                read_safe_batch=min(
                    v.get("effective_branch_chunk", v["actual_batch"])
                    for v in reads
                    if v["actual_batch"] == v["requested_batch"]
                )
                if any(v["actual_batch"] == v["requested_batch"] for v in reads)
                else 1,
                completed_before_next_source=True,
            )
            p.save(spath, row)
            del bank, totals, qg
        completed.append(row)
        prior_peak = row["read_peak_allocated_gib"]
        prior_batch = row["read_safe_batch"]
        result = assemble(ctx, identity, q0, completed)
        p.save(out / f"through_source_{source:02}.json", result)
        # Committed source JSON owns the contribution; discard only transient banks.
        for pattern in (
            f"evolved_source_{source:02}_wave_*.pt",
            f"read_source_{source:02}_wave_*.pt",
            f"read_progress_source_{source:02}_wave_*.pt",
        ):
            for transient in out.glob(pattern):
                transient.unlink()
        emit(
            out,
            "source_complete",
            source=source,
            read_peak_gib=prior_peak,
            read_safe_batch=prior_batch,
        )
    assert result["raw"]["source_gate_ids"] == [
        int(g["gate_idx"]) for s in selected_sources for g in ctx.layers[s]
    ]
    p.save(out / "result.json", result)
    return result
