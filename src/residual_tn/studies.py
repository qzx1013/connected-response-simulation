"""Restartable manual studies, with bounded concurrency only for Fig2."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os
import subprocess
import sys
import time

from residual_tn.config import read
from residual_tn.identity import (
    RunLock,
    atomic_json,
    claim_run,
    sha256,
    source_identity,
)


def input_identity(spec, inputs):
    inputs = Path(inputs)
    names = spec.get("required_inputs", [])
    if not names:
        return {}
    manifest = json.loads((inputs / "sha256.json").read_text())
    hashes = {}
    for name in names:
        hashes[name] = sha256(inputs / name)
        if hashes[name] != manifest[name]["sha256"]:
            raise ValueError("Input checksum mismatch: " + name)
    return hashes


def run_study(figure, output, inputs, device):
    spec = read(figure)
    output, inputs = Path(output).resolve(), Path(inputs).resolve()
    identity = dict(
        code=source_identity(),
        specification=spec,
        inputs=input_identity(spec, inputs),
        device=device,
    )
    with RunLock(output):
        claim_run(output, identity)
        status = {"state": "running", "stages": {}}
        atomic_json(output / "status.json", status)

        def stage(job):
            target = output / job["directory"] / job["expected"]
            record = output / ("stage_" + job["name"] + ".json")
            if record.exists():
                previous = json.loads(record.read_text())
                if (
                    previous["state"] == "complete"
                    and target.exists()
                    and sha256(target) == previous["output_sha256"]
                ):
                    return previous
            command = [
                sys.executable,
                "-u",
                "-m",
                "residual_tn.worker",
                "--figure",
                figure,
                "--job",
                job["name"],
                "--output",
                str(output),
                "--inputs",
                str(inputs),
                "--device",
                device,
            ]
            tick = time.perf_counter()
            print("[stage-start]", figure, job["name"], flush=True)
            with (output / (job["name"] + ".log")).open("a") as stream:
                completed = subprocess.run(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env=dict(
                        os.environ,
                        OMP_NUM_THREADS=str(job["threads"]),
                        MKL_NUM_THREADS=str(job["threads"]),
                    ),
                )
            result = dict(
                state="complete"
                if completed.returncode == 0 and target.exists()
                else "failed",
                returncode=completed.returncode,
                seconds=time.perf_counter() - tick,
            )
            if result["state"] == "complete":
                result["output_sha256"] = sha256(target)
            atomic_json(record, result)
            if result["state"] != "complete":
                raise RuntimeError(
                    "Stage failed: "
                    + job["name"]
                    + "; retained checkpoints and log: "
                    + str(output)
                )
            print("[stage-complete]", figure, job["name"], flush=True)
            return result

        try:
            if spec["parallel_stages"] > 1:
                with ThreadPoolExecutor(max_workers=spec["parallel_stages"]) as pool:
                    futures = [(job, pool.submit(stage, job)) for job in spec["jobs"]]
                    for job, future in futures:
                        status["stages"][job["name"]] = future.result()
            else:
                for job in spec["jobs"]:
                    status["stages"][job["name"]] = stage(job)
                    atomic_json(output / "status.json", status)
            from residual_tn.cli import render

            render(figure, output)
            status["state"] = "complete"
        except BaseException as error:
            status.update(state="failed", error=repr(error))
            raise
        finally:
            atomic_json(output / "status.json", status)
