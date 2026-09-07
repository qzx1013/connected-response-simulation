"""Current Fig5: all-source Pauli benchmark and source-0 fidelity pairs."""

import dataclasses


def run(spec, job, out, root, device):
    from residual_tn.peps import (
        model as p,
        fidelity_batch as f,
        source_bank as b,
        background_cache as shared,
    )
    from residual_tn.experiments import (
        fidelity_pairs as fs,
        fidelity_controls as controls,
        pauli_exact,
    )

    b.configure_logging()
    ctx, geometry, kernels, identity = p.model(device, **spec["model"])
    if ctx.n_qubits > 20:
        raise ValueError("Fig5 matched full-state study is limited to 20 qubits")
    if job["driver"] == "pauli_exact":
        return pauli_exact.calculate_all(ctx, kernels, identity, out)
    identity = fs.sample_identity(identity, fs.target_sample(ctx, geometry))
    if job["driver"] == "fidelity_exact":
        return fs.exact(ctx, kernels, identity, out, source=0)
    num = spec["numerics"]
    cfg = p.config(
        device=device,
        cavity=True,
        bond=num["bond_cap"],
        bp_tol=num["bp_tol"],
        bp_max_iter=num["bp_max_iter"],
        gloop=num["gloop_size"],
    )
    threshold = job.get("compression_tol", num["compression_tol"])
    identity = f.schedule_identity(
        dict(
            identity,
            numerics=dataclasses.asdict(cfg),
            dynamic_bp_threshold=threshold,
            redundant_post_compression_bp=False,
        )
    )
    with p.DynamicEvolution(
        ctx,
        geometry,
        kernels,
        cfg,
        out / "compression.jsonl",
        threshold=threshold,
        residual_svd_tol=num["residual_mpo_svd_tol"],
    ) as evolution:
        if job["driver"] == "fidelity_compression":
            return f.calculate(ctx, identity, evolution, cfg, out, source_layers=(0,))
        if job["driver"] != "fidelity_controls":
            raise ValueError(job["driver"])
        # One physical file, two panel descriptors. No duplicate 20-layer cache.
        baseline = root / "control_baseline"
        shared.export(baseline / "background_layers.pt", baseline, "e")
        layers, plan, descriptor = shared.load(ctx, identity, cfg, baseline)
        descriptors = {group: dict(descriptor, group=group) for group in "ef"}
        return controls.calculate(
            ctx, identity, evolution, cfg, layers, plan, descriptors, out
        )
