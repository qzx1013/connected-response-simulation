"""Portable defaults, relative to the checkout or the working directory."""

from pathlib import Path

CHECKOUT = Path(__file__).resolve().parents[2]
CONFIGS = Path(__file__).resolve().parent / "configs"
INPUTS = Path.cwd() / "inputs"
RESULTS = Path.cwd() / "results"
