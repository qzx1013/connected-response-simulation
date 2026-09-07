"""Bound CUDA allocation and add restartable work units to frozen scripts."""

import argparse, hashlib, importlib.util, json, os, sys, time
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def install_case_cache(module, directory):
    original = module.run_one_case
    position = 0

    def call(*args, **kwargs):
        nonlocal position
        path = directory / f"case_{position:04d}.json"
        position += 1
        if path.exists():
            return json.loads(path.read_text())
        value = original(*args, **kwargs)
        atomic_json(path, value)
        return value

    module.run_one_case = call


def install_mc_cache(module, directory):
    """Save samples and RNG after each logical batch; never normalize weights."""
    import numpy as np
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    original_batch = module.run_mc_batch
    original_mc = module.run_qutrit_monte_carlo
    offset = 0
    replayed_seconds = 0.0
    ledger = []

    def attempt(kwargs):
        before = kwargs["generator"].get_state()
        try:
            return original_batch(**kwargs)
        except torch.cuda.OutOfMemoryError as error:
            error.__traceback__ = None
            if kwargs["batch_size"] <= 1:
                raise
        # Failed stochastic work is discarded before restoring the RNG.
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        kwargs["generator"].set_state(before)
        left = kwargs["batch_size"] // 2
        a = attempt(dict(kwargs, batch_size=left))
        b = attempt(dict(kwargs, batch_size=kwargs["batch_size"] - left))
        return (
            np.concatenate((a[0], b[0])),
            {k: max(a[1][k], b[1][k]) for k in a[1]},
            None if a[2] is None else np.concatenate((a[2], b[2])),
        )

    def batch(**kwargs):
        nonlocal offset, replayed_seconds
        path = directory / f"batch_{offset:06d}.pt"
        before = kwargs["generator"].get_state()
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=False)
            assert saved["batch_size"] == kwargs["batch_size"] and torch.equal(
                saved["rng_before"], before
            )
            kwargs["generator"].set_state(saved["rng_after"])
            result = saved["result"]
            replayed_seconds += saved["compute_seconds"]
            ledger.append(
                dict(
                    end=offset + kwargs["batch_size"], replayed_seconds=replayed_seconds
                )
            )
        else:
            module.synchronize(kwargs["ideal_history"].device)
            tick = time.perf_counter()
            result = attempt(kwargs)
            module.synchronize(kwargs["ideal_history"].device)
            saved = dict(
                batch_size=kwargs["batch_size"],
                rng_before=before,
                rng_after=kwargs["generator"].get_state(),
                result=result,
                compute_seconds=time.perf_counter() - tick,
            )
            temp = path.with_suffix(".tmp")
            torch.save(saved, temp)
            temp.replace(path)
        offset += kwargs["batch_size"]
        return result

    def mc(**kwargs):
        result = original_mc(**kwargs)
        payload, samples, local, elapsed = result
        for row in payload["checkpoints"]:
            previous = max(
                [
                    x["replayed_seconds"]
                    for x in ledger
                    if x["end"] <= row["trajectories"]
                ]
                + [0.0]
            )
            row["cumulative_wall_seconds"] += previous
        payload["checkpoint_replayed_compute_seconds"] = replayed_seconds
        payload["resume_timing_definition"] = (
            "current online wall time plus previously saved successful batch compute time; replay loading is included"
        )
        return payload, samples, local, elapsed + replayed_seconds

    module.run_mc_batch = batch
    module.run_qutrit_monte_carlo = mc


def main():
    import importlib
    from residual_tn.config import read
    from residual_tn.identity import RunLock, claim_run, source_identity
    from residual_tn.paths import CONFIGS
    from residual_tn.studies import input_identity

    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    spec = read(args.figure)
    job = next(j for j in spec["jobs"] if j["name"] == args.job)
    out = args.output / job["directory"]
    identity = dict(
        code=source_identity(),
        specification=spec,
        job=job,
        inputs=input_identity(spec, args.inputs),
        device=args.device,
    )
    import torch

    torch.set_num_threads(job["threads"])
    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA unavailable; use --device cpu only for small tests"
            )
        index = 0 if device.index is None else device.index
        torch.cuda.set_device(index)
        total = torch.cuda.get_device_properties(index).total_memory
        limit = min(job["allocator_gib"] * 2**30, total - 4 * 2**30)
        if limit <= 0:
            raise ValueError("Device too small for memory reserve")
        torch.cuda.set_per_process_memory_fraction(limit / total, index)
        torch.cuda.reset_peak_memory_stats(index)
    # The Pauli driver owns this stage's lock and scientific checkpoint identity.
    if job["driver"] == "pauli_peps":
        from residual_tn.experiments.pauli_peps import run

        run(spec, out, args.device)
        return
    with RunLock(out):
        claim_run(out, identity)
        tick = time.perf_counter()
        passed = False
        try:
            if job["driver"] == "module_cli":
                module = importlib.import_module(job["module"])
                if job["module"].endswith(".fig2"):
                    install_case_cache(module, out / "cases")
                if job["module"].endswith(".fig4"):
                    install_mc_cache(module, out / "mc_batches")
                fields = dict(
                    output=str(args.output),
                    stage=str(out),
                    inputs=str(args.inputs),
                    manifest=str(CONFIGS / "residual_manifest.json"),
                    device=args.device,
                )
                sys.argv = [
                    job["module"],
                    *[token.format(**fields) for token in job["args"]],
                ]
                module.main()
            else:
                from residual_tn.experiments.fig5_stages import run

                run(spec, job, out, args.output, device)
            passed = True
        finally:
            atomic_json(
                out / "worker_resources.json",
                dict(
                    passed=passed,
                    seconds=time.perf_counter() - tick,
                    max_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30
                    if device.type == "cuda"
                    else 0,
                    max_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30
                    if device.type == "cuda"
                    else 0,
                    allocator_limit_gib=job["allocator_gib"],
                    threads=job["threads"],
                ),
            )


if __name__ == "__main__":
    main()
