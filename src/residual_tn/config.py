"""Figure specifications can be inspected without importing Torch or CUDA."""

from copy import deepcopy
import json
from pathlib import Path
from residual_tn.paths import CONFIGS
from residual_tn.geometry import load, validate_geometry


def read(name):
    path = CONFIGS / (name + ".json")
    value = json.loads(path.read_text())
    if value["figure"] != name:
        raise ValueError("Figure configuration/name mismatch")
    return value


def describe(name):
    value = deepcopy(read(name))
    if "model" in value:
        model = value["model"]
        value["geometry"] = validate_geometry(
            load(CONFIGS / "residual_manifest.json", model["width"], model["length"])
        )
    value["execution"] = "manual only; inspecting this plan starts no computation"
    return value
