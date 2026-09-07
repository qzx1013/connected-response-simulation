"""One source evolution for panels e/f and one BP trajectory per target.

Panels keep separately identified background caches. Sharing evolution
requires identical cache checksums and identical evolution settings.
"""

import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path
import torch
from residual_tn.peps import model as p
from residual_tn.peps import source_bank as b
from residual_tn.peps import fidelity_batch as f
from residual_tn.experiments import fidelity_pairs as fs
from residual_tn.peps import background_cache as shared
from residual_tn.peps import continuation as continuation

POINTS = {
    "bptol1e-8": ("e", 1e-8, 500),
    "tol_baseline": ("e", 1e-10, 500),
    "bptol1e-12": ("e", 1e-12, 500),
    "bp250": ("f", 1e-10, 250),
    "iter_baseline": ("f", 1e-10, 500),
    "bp1000": ("f", 1e-10, 1000),
}


def message_fingerprint(rows, label):
    digest = hashlib.sha256()
    for row in rows:
        for key in sorted(row["snapshots"][label]):
            value = row["snapshots"][label][key].contiguous()
            digest.update(str((key, tuple(value.shape), value.dtype)).encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def accumulate(ctx, layers, rows, num, target, selected):
    totals = torch.zeros(
        len(ctx.layers[0]), len(ctx.layers[target]), dtype=torch.float64
    )
    for row, value in zip(rows, num):
        totals[row["local"]] += value
    fs0 = torch.tensor(
        [layers[0]["gate_data"][int(g["gate_idx"])]["F"] for g in ctx.layers[0]],
        dtype=torch.float64,
    )
    ft = torch.tensor(
        [
            layers[target]["gate_data"][int(g["gate_idx"])]["F"]
            for g in ctx.layers[target]
        ],
        dtype=torch.float64,
    )
    denominator = fs0[:, None] * ft[None, :]
    assert bool(torch.isfinite(denominator).all() & (denominator > 1e-30).all())
    delta = totals / denominator - 1
    assert bool(torch.isfinite(delta).all())
    return [
        dict(
            source_gate=int(a["gate_idx"]),
            target_gate=int(g["gate_idx"]),
            target_layer=target,
            delta=float(delta[i, j]),
        )
        for i, a in enumerate(ctx.layers[0])
        for j, g in enumerate(ctx.layers[target])
        if int(g["gate_idx"]) > int(a["gate_idx"]) and int(g["gate_idx"]) in selected
    ]


def calculate(
    ctx,
    identity,
    evo,
    cfg,
    layers,
    plan,
    descriptors,
    out,
    points=None,
    stop_after_target=None,
):
    points = points or POINTS
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    criteria = {}
    point_labels = {}
    for name, (group, tol, cap) in points.items():
        label = f"{tol:.0e}_cap{cap}"
        criteria[label] = (tol, cap)
        point_labels[name] = label
    identities = {
        name: dict(
            identity,
            numerics=dataclasses.asdict(
                dataclasses.replace(cfg, bp_tol=tol, bp_max_iter=cap)
            ),
            evolution_numerics=dataclasses.asdict(cfg),
            shared_background=descriptors[group],
            bp_scan_scope="readout mixed BP only; source evolution and compression settings fixed",
            fidelity_read_batch_cap=64,
            bp_continuation="per-branch first-convergence or cap snapshot of a single trajectory",
        )
        for name, (group, tol, cap) in points.items()
    }
    bank_identity = dict(
        identity,
        panel_backgrounds=descriptors,
        criteria=criteria,
        points=points,
        read_pool_batch_cap=64,
        bp_continuation="shared_e_f_source_evolution_and_mixed_bp_v1",
    )
    progress = out / "progress.pt"
    if progress.exists():
        data = torch.load(progress, map_location="cpu", weights_only=False)
        assert data["identity"] == bank_identity
        rows, filters, pairs, records, next_target = (
            data[k] for k in ("rows", "filters", "pairs", "records", "next_target")
        )
    else:
        rows, filters = f.make_bank(ctx, layers[0], 0, cfg, plan)
        pairs = {name: [] for name in points}
        records = {name: [] for name in points}
        next_target = 0
    plan["sampled_gate_ids"] = {
        t: tuple(r["gate"] for r in identity["target_sample"] if r["layer"] == t)
        for t in range(len(ctx.layers))
    }
    for target in range(next_target, len(ctx.layers)):
        if target > 0:
            rows = b.evolve_bank(
                ctx,
                bank_identity,
                evo,
                cfg,
                plan,
                rows,
                0,
                0,
                out,
                through_layer=target,
            )
        clean = p.core._move_clean_layer_tensors(layers[target], ctx.device)
        snapshot_path = out / f"snapshots_target_{target:02}.pt"
        if snapshot_path.exists():
            saved = torch.load(snapshot_path, map_location="cpu", weights_only=False)
            assert saved["identity"] == bank_identity
            assert [(r["gate"], r["mode"]) for r in saved["rows"]] == [
                (r["gate"], r["mode"]) for r in rows
            ]
            rows, infos = saved["rows"], saved["infos"]
        else:
            infos = continuation.mixed_bank(
                ctx, cfg, plan, rows, clean, out, target, criteria
            )
            p.save_tensor(
                snapshot_path, dict(identity=bank_identity, rows=rows, infos=infos)
            )
        # Pool branch x snapshot reads at this target. Equal message banks
        # share one contraction; all remaining tasks use bounded GPU batches.
        tasks = []
        label_indices = {}
        fingerprints = {}
        for label in criteria:
            key = message_fingerprint(rows, label)
            if key in fingerprints:
                label_indices[label] = fingerprints[key]
                b.emit(
                    out,
                    "identical_snapshot_read_reused",
                    target=target,
                    label=label,
                    message_sha256=key,
                )
                continue
            indices = []
            for row in rows:
                indices.append(len(tasks))
                tasks.append(dict(row, id=len(tasks), messages=row["snapshots"][label]))
            label_indices[label] = indices
            fingerprints[key] = indices
        previous_cap = f.READ_CAP
        f.READ_CAP = 64
        try:
            pooled_num, pooled_den, stats = f.read_bank(
                ctx, cfg, plan, tasks, clean, out, 0, target, bank_identity
            )
        finally:
            f.READ_CAP = previous_cap
        for name, (group, tol, cap) in points.items():
            label = point_labels[name]
            num = pooled_num[label_indices[label]]
            pairs[name].extend(
                accumulate(
                    ctx, layers, rows, num, target, plan["sampled_gate_ids"][target]
                )
            )
            records[name].extend(infos[label])
        unique_reads = len(fingerprints)
        del tasks, pooled_num, pooled_den
        for row in rows:
            row.pop("messages", None)
            row.pop("snapshots", None)
        p.save_tensor(
            progress,
            dict(
                identity=bank_identity,
                rows=rows,
                filters=filters,
                pairs=pairs,
                records=records,
                next_target=target + 1,
            ),
        )
        snapshot_path.unlink()
        # The atomic panel checkpoint owns all committed target results.
        temporary_read = out / f"read_source_00_target_{target:02}.pt"
        if temporary_read.exists():
            temporary_read.unlink()
        del clean
        b.emit(
            out,
            "bp_panel_target_complete",
            target=target,
            branches=len(rows),
            unique_reads=unique_reads,
            points=len(points),
        )
        if stop_after_target == target:
            raise InterruptedError("Injected panel target checkpoint restart")
    one = {
        int(g["gate_idx"]): float(layers[t]["gate_data"][int(g["gate_idx"])]["F"])
        for t, gates in enumerate(ctx.layers)
        for g in gates
    }
    results = {}
    for name in points:
        point = out.parent / ("control_" + name)
        source = dict(
            identity=identities[name],
            source_layer=0,
            pairs=pairs[name],
            bp_records=records[name],
            branch_count=len(rows),
            mode_filter=filters,
            seconds=time.perf_counter() - start,
            timing_scope="shared e/f job, not independent per-point runtime",
        )
        p.save(point / "source_00.json", source)
        result = fs.assemble(ctx, one, pairs[name], (0,), identity["target_sample"])
        result.update(
            identity=identities[name], seconds_shared_panel=time.perf_counter() - start
        )
        p.save(point / "result.json", result)
        results[name] = result
    p.save(
        out / "result.json",
        dict(
            identity=bank_identity,
            completed_points=list(points),
            pair_count=next(iter(results.values()))["pair_count"],
            seconds=time.perf_counter() - start,
        ),
    )
    for file in (progress, out / "evolved_source_00_wave_00.pt"):
        if file.exists():
            file.unlink()
    return results
