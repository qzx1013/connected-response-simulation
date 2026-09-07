"""Audit the physical residual pairs without importing tensor libraries."""

from collections import Counter
import json
import math
from pathlib import Path


def validate_geometry(geometry):
    width, length = geometry["width"], geometry["length"]
    n = width * length
    edges = geometry["edges"]
    if n != geometry["n_qubits"] or len(edges) != geometry["n_residual_edges"]:
        raise ValueError("Geometry counts are inconsistent")
    endpoints = Counter()
    bonds = Counter()
    vertices = Counter()
    for edge in edges:
        a, b = edge["q1"], edge["q2"]
        if not (0 <= a < b < n):
            raise ValueError("Residual endpoints must be ordered in range")
        if abs(a // width - b // width) != 1 or abs(a % width - b % width) != 1:
            raise ValueError("Only diagonal next-nearest neighbors are allowed")
        endpoints.update((a, b))
        mhz = edge["epsilon_over_2pi_mhz"]
        if not 0.1 <= mhz <= 0.5 or not math.isclose(
            edge["epsilon_rad_ns"], 2 * math.pi * mhz * 1e-3, rel_tol=1e-14
        ):
            raise ValueError("Residual strength/unit mismatch")
        route = edge["swap_routing"]["path"]
        if len(route) != 3 or route[0] != a or route[-1] != b:
            raise ValueError("Base route must be a shortest path between endpoints")
        for u, v in zip(route, route[1:]):
            if (
                not 0 <= u < n
                or not 0 <= v < n
                or abs(u // width - v // width) + abs(u % width - v % width) != 1
            ):
                raise ValueError("Route contains a non-nearest-neighbor step")
            bonds.update([tuple(sorted((u, v)))])
        vertices.update(route)
    if max(endpoints.values(), default=0) > 1:
        raise ValueError("Residual endpoints overlap")
    return dict(
        width=width,
        length=length,
        n_qubits=n,
        edges=len(edges),
        pairs=[[e["q1"], e["q2"]] for e in edges],
        endpoint_disjoint=True,
        all_diagonal=True,
        shared_base_route_bonds=sum(v * (v - 1) // 2 for v in bonds.values()),
        shared_base_route_vertices=sum(v * (v - 1) // 2 for v in vertices.values()),
    )


def load(path, width, length):
    data = json.loads(Path(path).read_text())
    matches = [
        g for g in data["geometries"] if (g["width"], g["length"]) == (width, length)
    ]
    if len(matches) != 1 or data["kernel_scheme"] != "magnus-1":
        raise ValueError("Missing/ambiguous geometry or unsupported residual scheme")
    validate_geometry(matches[0])
    return matches[0]
