"""Validated, tensor-only background persistence for terminal gloop scans."""

from __future__ import annotations

import json
from pathlib import Path

import torch


def _map_tensors(value, function):
    if isinstance(value, torch.Tensor):
        return function(value)
    if isinstance(value, dict):
        return {key: _map_tensors(item, function) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_tensors(item, function) for item in value]
    if isinstance(value, tuple):
        return tuple(_map_tensors(item, function) for item in value)
    return value


def save_background(path: Path, clean, metadata):
    if not metadata:
        raise ValueError("background checkpoint requires model/code identity")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite background: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        key: clean[key]
        for key in ("peps", "messages", "Z_clean", "bp_info", "bond_audit")
    }
    state = _map_tensors(
        state, lambda tensor: tensor.detach().to("cpu", copy=True).contiguous()
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"schema": 1, "metadata": metadata, "state": state}, temporary)
    temporary.replace(path)


def load_background(path: Path, metadata, device):
    # Validate on CPU before allocating any GPU memory.  No arbitrary pickle
    # execution; only our tensor/primitive checkpoint schema is accepted.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema") != 1
        or not metadata
        or payload.get("metadata") != metadata
    ):
        raise ValueError("background checkpoint model/code identity mismatch")
    return _map_tensors(payload["state"], lambda tensor: tensor.to(device))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
