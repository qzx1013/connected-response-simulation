"""One immutable baseline PEPS history and normalization within each panel.

Scanned parameters affect source evolution and mixed-BP readout only. The
cached background tensors, clean messages, Kraus amplitudes and M_a are fixed.
"""

import dataclasses
import hashlib
import json
import shutil
from pathlib import Path
import torch
from residual_tn.peps import model as p

PHYSICAL_KEYS = (
    "width",
    "length",
    "layers",
    "piece_num",
    "pattern_seed",
    "pattern_index",
    "noise_T1_T2_us",
    "kraus_eigenvalue_cutoff",
    "manifest_sha256",
    "common_residual_background",
    "kernel_scheme",
    "gate_table_sha256",
    "kraus_sha256",
    "fidelity_target",
)
SCANNED_BP_KEYS = {"bp_tol", "bp_max_iter"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def export(source, destination, group):
    assert group in ("d", "e", "f")
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    data = torch.load(source, map_location="cpu", weights_only=False)
    identity = data["identity"]
    assert sorted(data["layers"]) == list(range(identity["layers"]))
    for layer in data["layers"].values():
        assert dataclasses.asdict(layer["config"]) == identity["numerics"]
    target = destination / "background_layers.pt"
    if source.resolve() != target.resolve():
        assert not target.exists()
        shutil.copy2(source, target)
    metadata = dict(
        schema=1,
        group=group,
        cache_sha256=sha256(target),
        physical_identity={k: identity[k] for k in PHYSICAL_KEYS},
        background_numerics=identity["numerics"],
        background_compression_tol=identity["dynamic_bp_threshold"],
        original_identity=identity,
        original_cache_path=str(source),
        scope="Fixed baseline background, clean BP messages, source amplitudes and one-insertion normalizations shared across controls",
    )
    p.save(destination / "metadata.json", metadata)
    return metadata


def load(ctx, identity, cfg, directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["schema"] == 1
    assert metadata["physical_identity"] == {k: identity[k] for k in PHYSICAL_KEYS}, (
        "Different physical background or Kraus model"
    )
    current = dataclasses.asdict(cfg)
    expected = metadata["background_numerics"]
    assert set(current) == set(expected)
    assert all(
        current[k] == expected[k] for k in current if k not in SCANNED_BP_KEYS
    ), "Only source compression and BP convergence controls may differ"
    path = directory / "background_layers.pt"
    assert sha256(path) == metadata["cache_sha256"], (
        "Shared background checksum mismatch"
    )
    data = torch.load(path, map_location="cpu", weights_only=False)
    assert data["identity"] == metadata["original_identity"]
    assert sorted(data["layers"]) == list(range(len(ctx.layers)))
    layers = {}
    for t, layer in data["layers"].items():
        assert dataclasses.asdict(layer["config"]) == expected
        assert set(layer["gate_data"]) == {int(g["gate_idx"]) for g in ctx.layers[t]}
        # The reader uses this field for algorithm options. Tensor payloads
        # and normalizers remain byte-for-byte those of the fixed baseline.
        layers[t] = dict(layer, config=cfg)
    descriptor = {
        k: metadata[k]
        for k in (
            "group",
            "cache_sha256",
            "physical_identity",
            "background_numerics",
            "background_compression_tol",
            "scope",
        )
    }
    return layers, p.core._build_plan(ctx, cfg), descriptor
