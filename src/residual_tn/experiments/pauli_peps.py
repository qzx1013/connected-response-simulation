"""One generalized Fig5/Fig6 PEPS computation; no full-state allocation."""

from pathlib import Path
import dataclasses
import json
import time

from residual_tn.identity import RunLock, atomic_json, claim_run


def run(
    spec,
    output,
    device="cuda",
    source_layers=None,
    evolution_batch=None,
    read_batch=None,
):
    import torch
    from residual_tn.peps import model as p, source_bank as bank

    output = Path(output)
    model = spec["model"]
    numerics = spec["numerics"]
    execution = dict(spec["resources"])
    if evolution_batch is not None:
        execution["evolution_batch_cap"] = evolution_batch
    if read_batch is not None:
        execution["readout_batch_cap"] = read_batch
    if min(execution["evolution_batch_cap"], execution["readout_batch_cap"]) < 1:
        raise ValueError("Batch caps must be positive")
    with RunLock(output):
        p.setup_device(
            torch.device(device),
            execution["allocator_gib"],
            execution["reserve_gib"],
            execution["cpu_threads"],
        )
        ctx, geometry, kernels, identity = p.model(torch.device(device), **model)
        cfg = p.config(
            device=device,
            bond=numerics["bond_cap"],
            bp_tol=numerics["bp_tol"],
            bp_max_iter=numerics["bp_max_iter"],
            gloop=numerics["gloop_size"],
        )
        cfg = dataclasses.replace(
            cfg,
            gloop_memory_target_gib=execution["gloop_intermediate_gib"]
            if device != "cpu"
            else None,
            gloop_peak_budget_gib=execution["gloop_peak_budget_gib"]
            if device != "cpu"
            else None,
            gloop_peak_reserve_gib=execution["reserve_gib"],
        )
        cfg.validate()
        identity.update(
            numerics=dataclasses.asdict(cfg),
            dynamic_bp_threshold=numerics["compression_tol"],
            residual_mpo_svd_tol=numerics["residual_mpo_svd_tol"],
            redundant_post_compression_bp=False,
            pauli_schedule="source_bank_padded_norm_bp",
            readout_policy="single stream, adaptive branch batches and exact slicing",
        )
        # Batch caps / requested sources do not change the scientific identity.
        # A rerun may lower concurrency and consume compatible committed sources.
        claim_run(output, identity)
        atomic_json(
            output / "execution.json",
            dict(resources=execution, requested_sources=source_layers, device=device),
        )
        bank.EVOLUTION_CAP = execution["evolution_batch_cap"]
        bank.READ_CAP = execution["readout_batch_cap"]
        bank.WORKSPACE_CAP_GIB = execution["workspace_cap_gib"]
        bank.HEADROOM_GIB = execution["admission_headroom_gib"]
        bank.READ_GROWTH_TARGET_GIB = execution["read_growth_target_gib"]
        tick = time.perf_counter()
        atomic_json(output / "status.json", {"state": "running"})
        try:
            with p.DynamicEvolution(
                ctx,
                geometry,
                kernels,
                cfg,
                output / "compression.jsonl",
                threshold=numerics["compression_tol"],
                residual_svd_tol=numerics["residual_mpo_svd_tol"],
            ) as evolution:
                result = bank.calculate(
                    ctx, identity, evolution, cfg, output, source_layers=source_layers
                )
            state = (
                "subset_complete"
                if result["definition"]["validation_subset"]
                else "complete"
            )
            atomic_json(
                output / "status.json",
                dict(state=state, seconds_this_invocation=time.perf_counter() - tick),
            )
            return result
        except BaseException as error:
            atomic_json(
                output / "status.json",
                dict(
                    state="failed",
                    error=repr(error),
                    seconds_this_invocation=time.perf_counter() - tick,
                ),
            )
            raise
