"""Pulse-resolved interaction-picture residual-operator library.

For a residual exchange edge ``(m, n)`` this module compiles

    Omega_1 = -i epsilon_mn integral dt [a_m^dagger(t) a_n(t)
                                        + a_n^dagger(t) a_m(t)]

in the interaction picture of the intended 0522 control acting during one
circuit layer.  The time integral is performed once per control signature.
The cached unit-coupling generator is then scaled by ``epsilon_mn`` and used
to form either ``I + Omega_1`` or, preferentially, ``exp(Omega_1)``.

One-qubit control layers retain two-site support.  During a disjoint two-qubit
gate layer, the support is the union of the two control blocks containing the
residual-edge endpoints and can therefore contain two, three, or four sites.
The physical partner labels only determine how a cached block operator is
embedded; they do not require another pulse integration.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


ANHARMONICITY_RAD_NS = -2.0 * math.pi * 242.0e-3
DRIVE_OMEGA_RAD_NS = 0.2051 / 2.0
TWO_QUTRIT_EXCHANGE_RAD_NS = -0.0422
LAYER_DURATION_NS = 8.0 * math.pi / abs(ANHARMONICITY_RAD_NS)
SINGLE_PHASES = (0.0, math.pi / 2.0, math.pi / 4.0)
KERNEL_SCHEMES = ("dyson-1", "magnus-1")


_DRAG_VALUES = (
    -0.0224,
    -0.0253,
    -0.0935,
    -0.0756,
    -0.0228,
    -0.0243,
    -0.0988,
    -0.0820,
    -0.0458,
    -0.0382,
    -0.1060,
    -0.1068,
    -0.0224,
    -0.0253,
    -0.0935,
    -0.0756,
    -0.0224,
    -0.0253,
    -0.0935,
    -0.0756,
    -0.0228,
    -0.0243,
    -0.0988,
    -0.0820,
    -0.0458,
    -0.0382,
    -0.1060,
    -0.1068,
    -0.0224,
    -0.0253,
    -0.0935,
    -0.0756,
    -0.0228,
    -0.0243,
    -0.0988,
    -0.0820,
    -0.0224,
    -0.0253,
    -0.0935,
    -0.0756,
    -0.0228,
    -0.0243,
    -0.0988,
    -0.0820,
    -0.0458,
    -0.0382,
    -0.1060,
    -0.1068,
    -0.0224,
    -0.0253,
    -0.0935,
    -0.0756,
    -0.0224,
    -0.0253,
    -0.0935,
    -0.0756,
    -0.0228,
    -0.0243,
    -0.0988,
    -0.0820,
    -0.0458,
    -0.0382,
    -0.1060,
    -0.1068,
    -0.0224,
    -0.0253,
    -0.0935,
    -0.0756,
    -0.0228,
    -0.0243,
    -0.0988,
    -0.0820,
)


def _ladder(
    dim: int, *, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    annihilation = torch.zeros((dim, dim), dtype=dtype, device=device)
    for level in range(1, dim):
        annihilation[level - 1, level] = math.sqrt(level)
    creation = annihilation.conj().T.contiguous()
    number = creation @ annihilation
    identity = torch.eye(dim, dtype=dtype, device=device)
    return annihilation, creation, number, identity


def _batch_kron(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Kronecker product with arbitrary shared leading dimensions."""
    lrows, lcols = left.shape[-2:]
    rrows, rcols = right.shape[-2:]
    return (
        left.unsqueeze(-1).unsqueeze(-3) * right.unsqueeze(-2).unsqueeze(-4)
    ).reshape(*left.shape[:-2], lrows * rrows, lcols * rcols)


def permute_operator_sites(
    operator: torch.Tensor,
    source_sites: Sequence[int],
    target_sites: Sequence[int],
    *,
    local_dim: int,
) -> torch.Tensor:
    """Reorder tensor-product sites of a possibly batched operator."""
    source = tuple(map(int, source_sites))
    target = tuple(map(int, target_sites))
    if sorted(source) != sorted(target) or len(set(source)) != len(source):
        raise ValueError("source_sites and target_sites must be permutations")
    count = len(source)
    expected = local_dim**count
    if tuple(operator.shape[-2:]) != (expected, expected):
        raise ValueError("operator dimension does not match its site count")
    lead = operator.shape[:-2]
    offset = len(lead)
    tensor = operator.reshape(*lead, *([local_dim] * count), *([local_dim] * count))
    row_axes = [source.index(site) for site in target]
    permutation = (
        list(range(offset))
        + [offset + axis for axis in row_axes]
        + [offset + count + axis for axis in row_axes]
    )
    return tensor.permute(permutation).reshape(*lead, expected, expected)


def _computational_indices(
    site_count: int, *, qutrit_dim: int = 3, device: torch.device
) -> torch.Tensor:
    indices = []
    for bits in itertools.product((0, 1), repeat=site_count):
        value = 0
        for bit in bits:
            value = value * qutrit_dim + bit
        indices.append(value)
    return torch.tensor(indices, dtype=torch.long, device=device)


def project_qutrit_operator_to_qubits(
    operator: torch.Tensor, site_count: int
) -> torch.Tensor:
    """Restrict a qutrit operator to the tensor-product |0>,|1> subspace."""
    indices = _computational_indices(site_count, device=operator.device)
    return operator.index_select(-2, indices).index_select(-1, indices)


def _site_drag(site: int) -> float:
    return float(_DRAG_VALUES[int(site) % len(_DRAG_VALUES)])


@dataclass(frozen=True)
class ControlHistory:
    sites: tuple[int, ...]
    midpoint_unitaries: torch.Tensor
    dt_ns: float
    kind: str
    mode: int | None = None


@dataclass
class CompiledResidualOperator:
    layer: int
    edge: tuple[int, int]
    support: tuple[int, ...]
    control_blocks: tuple[tuple[int, ...], tuple[int, ...]]
    epsilon_rad_ns: float
    unit_omega: torch.Tensor
    omega: torch.Tensor
    operator: torch.Tensor
    scheme: str

    def metadata(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "edge": list(self.edge),
            "support": list(self.support),
            "support_size": len(self.support),
            "control_blocks": [list(block) for block in self.control_blocks],
            "epsilon_rad_ns": self.epsilon_rad_ns,
            "scheme": self.scheme,
        }


class ResidualMagnusLibrary:
    """Cache 0522 control histories and unit-coupling Magnus generators."""

    def __init__(
        self,
        n_qubits: int,
        *,
        integration_steps: int = 1000,
        dtype: torch.dtype = torch.complex128,
        device: str | torch.device = "cpu",
        projected_qubits: bool = True,
    ) -> None:
        if n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        if integration_steps <= 0:
            raise ValueError("integration_steps must be positive")
        if not torch.is_complex(torch.empty((), dtype=dtype)):
            raise TypeError("the residual Magnus library needs a complex dtype")
        self.n_qubits = int(n_qubits)
        self.integration_steps = int(integration_steps)
        self.dtype = dtype
        self.device = torch.device(device)
        self.projected_qubits = bool(projected_qubits)
        self.dt_ns = LAYER_DURATION_NS / self.integration_steps
        self._single_histories: dict[tuple[int, int], torch.Tensor] = {}
        self._two_history: torch.Tensor | None = None
        self._unit_omega_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    def _midpoint_grid(self) -> torch.Tensor:
        real_dtype = torch.float64 if self.dtype == torch.complex128 else torch.float32
        return (
            torch.arange(
                self.integration_steps,
                dtype=real_dtype,
                device=self.device,
            )
            + 0.5
        ) * self.dt_ns

    def _single_history(self, site: int, mode: int) -> torch.Tensor:
        key = (int(site), int(mode))
        cached = self._single_histories.get(key)
        if cached is not None:
            return cached
        if mode not in (0, 1, 2, 3):
            raise ValueError(f"unsupported one-qutrit mode {mode}")

        annihilation, creation, number, identity = _ladder(
            3, dtype=self.dtype, device=self.device
        )
        static = 0.5 * ANHARMONICITY_RAD_NS * (number @ (number - identity))
        times = self._midpoint_grid()
        center = LAYER_DURATION_NS / 2.0
        sigma = LAYER_DURATION_NS / 4.0
        gaussian = torch.exp(-0.5 * ((times - center) / sigma) ** 2)
        envelope = gaussian - math.exp(-2.0)
        derivative = -(times - center) / (sigma**2) * gaussian
        if mode == 3:
            beta = torch.zeros_like(times, dtype=self.dtype)
        else:
            beta = (
                DRIVE_OMEGA_RAD_NS * envelope
                + 1j * _site_drag(site) * derivative / ANHARMONICITY_RAD_NS
            ).to(self.dtype) * torch.exp(
                torch.tensor(
                    1j * SINGLE_PHASES[mode],
                    dtype=self.dtype,
                    device=self.device,
                )
            )
        hamiltonians = (
            static.unsqueeze(0)
            + beta[:, None, None] * creation.unsqueeze(0)
            + beta.conj()[:, None, None] * annihilation.unsqueeze(0)
        )
        midpoint = self._propagate_midpoints(hamiltonians)
        self._single_histories[key] = midpoint
        return midpoint

    def _two_qutrit_history(self) -> torch.Tensor:
        if self._two_history is not None:
            return self._two_history
        annihilation, creation, number, identity = _ladder(
            3, dtype=self.dtype, device=self.device
        )
        a_left = torch.kron(annihilation, identity)
        a_right = torch.kron(identity, annihilation)
        adag_left = torch.kron(creation, identity)
        adag_right = torch.kron(identity, creation)
        n_left = adag_left @ a_left
        n_right = adag_right @ a_right
        eye_pair = torch.eye(9, dtype=self.dtype, device=self.device)
        hamiltonian = 0.5 * ANHARMONICITY_RAD_NS * (
            n_left @ (n_left - eye_pair) + n_right @ (n_right - eye_pair)
        ) + TWO_QUTRIT_EXCHANGE_RAD_NS * (adag_left @ a_right + a_left @ adag_right)
        times = self._midpoint_grid()
        self._two_history = torch.matrix_exp(
            (-1j * times[:, None, None]) * hamiltonian.unsqueeze(0)
        )
        return self._two_history

    def _propagate_midpoints(self, hamiltonians: torch.Tensor) -> torch.Tensor:
        identity = torch.eye(
            hamiltonians.shape[-1], dtype=self.dtype, device=self.device
        )
        current = identity
        midpoints = []
        for hamiltonian in hamiltonians:
            half = torch.matrix_exp(-0.5j * self.dt_ns * hamiltonian)
            midpoints.append(half @ current)
            current = (half @ half) @ current
        return torch.stack(midpoints, dim=0)

    def _layer_blocks(
        self, gates: Sequence[dict[str, Any]]
    ) -> dict[int, tuple[tuple[int, ...], str, int | None]]:
        blocks: dict[int, tuple[tuple[int, ...], str, int | None]] = {}
        for gate in gates:
            sites = tuple(map(int, gate["qubits"]))
            if len(sites) == 1:
                mode = int(gate.get("mode", 3))
                descriptor = (sites, "single", mode)
            elif len(sites) == 2:
                descriptor = (sites, "two", None)
            else:
                raise ValueError("0522 layers may contain only one- or two-site gates")
            for site in sites:
                if site in blocks:
                    raise ValueError(f"overlapping intended gates at site {site}")
                blocks[site] = descriptor
        for site in range(self.n_qubits):
            blocks.setdefault(site, ((site,), "single", 3))
        return blocks

    def _history_for_block(
        self, descriptor: tuple[tuple[int, ...], str, int | None]
    ) -> ControlHistory:
        sites, kind, mode = descriptor
        if kind == "single":
            assert mode is not None
            return ControlHistory(
                sites,
                self._single_history(sites[0], mode),
                self.dt_ns,
                kind,
                mode,
            )
        if kind == "two":
            return ControlHistory(
                sites, self._two_qutrit_history(), self.dt_ns, kind, None
            )
        raise ValueError(f"unsupported control kind {kind!r}")

    def _dressed_annihilation(
        self, endpoint: int, history: ControlHistory
    ) -> torch.Tensor:
        annihilation, _, _, identity = _ladder(3, dtype=self.dtype, device=self.device)
        if len(history.sites) == 1:
            bare = annihilation
        elif history.sites.index(endpoint) == 0:
            bare = torch.kron(annihilation, identity)
        else:
            bare = torch.kron(identity, annihilation)
        unitary = history.midpoint_unitaries
        return unitary.conj().transpose(-1, -2) @ bare @ unitary

    def _template_key(
        self,
        edge: tuple[int, int],
        left: tuple[tuple[int, ...], str, int | None],
        right: tuple[tuple[int, ...], str, int | None],
    ) -> tuple[Any, ...]:
        def signature(
            endpoint: int,
            descriptor: tuple[tuple[int, ...], str, int | None],
        ) -> tuple[Any, ...]:
            sites, kind, mode = descriptor
            return (
                kind,
                int(mode) if mode is not None else None,
                sites.index(endpoint),
                int(endpoint) if kind == "single" else None,
            )

        return (
            signature(edge[0], left),
            signature(edge[1], right),
            self.integration_steps,
            self.projected_qubits,
        )

    def unit_omega_for_edge(
        self,
        gates: Sequence[dict[str, Any]],
        edge: tuple[int, int],
    ) -> tuple[tuple[int, ...], tuple[tuple[int, ...], tuple[int, ...]], torch.Tensor]:
        """Return support, control blocks, and Omega_1 for epsilon=1 rad/ns."""
        m, n = tuple(map(int, edge))
        blocks = self._layer_blocks(gates)
        left_descriptor = blocks[m]
        right_descriptor = blocks[n]
        left_sites = left_descriptor[0]
        right_sites = right_descriptor[0]
        if set(left_sites) & set(right_sites):
            raise ValueError(
                "a residual edge inside one intended two-site block is not "
                "part of the configured nonlocal residual ensemble"
            )
        support = tuple(sorted((*left_sites, *right_sites)))
        key = self._template_key((m, n), left_descriptor, right_descriptor)
        cached = self._unit_omega_cache.get(key)
        if cached is not None:
            source_sites = tuple((*left_sites, *right_sites))
            omega = permute_operator_sites(
                cached,
                source_sites,
                support,
                local_dim=2 if self.projected_qubits else 3,
            )
            return support, (left_sites, right_sites), omega

        left_history = self._history_for_block(left_descriptor)
        right_history = self._history_for_block(right_descriptor)
        dressed_m = self._dressed_annihilation(m, left_history)
        dressed_n = self._dressed_annihilation(n, right_history)
        exchange = _batch_kron(
            dressed_m.conj().transpose(-1, -2), dressed_n
        ) + _batch_kron(dressed_m, dressed_n.conj().transpose(-1, -2))
        canonical_sites = tuple((*left_sites, *right_sites))
        integral = self.dt_ns * torch.sum(exchange, dim=0)
        unit_omega_qutrit = -1j * integral
        unit_omega_qutrit = 0.5 * (unit_omega_qutrit - unit_omega_qutrit.conj().T)
        if self.projected_qubits:
            canonical = project_qutrit_operator_to_qubits(
                unit_omega_qutrit, len(canonical_sites)
            )
            local_dim = 2
        else:
            canonical = unit_omega_qutrit
            local_dim = 3
        self._unit_omega_cache[key] = canonical.detach().clone()
        omega = permute_operator_sites(
            canonical,
            canonical_sites,
            support,
            local_dim=local_dim,
        )
        return support, (left_sites, right_sites), omega

    def compile_layer(
        self,
        layer_index: int,
        gates: Sequence[dict[str, Any]],
        residual_edges: Iterable[dict[str, Any]],
        *,
        scheme: str = "magnus-1",
    ) -> list[CompiledResidualOperator]:
        if scheme not in KERNEL_SCHEMES:
            raise ValueError(f"unsupported residual kernel scheme {scheme!r}")
        compiled = []
        for record in residual_edges:
            edge = (int(record["q1"]), int(record["q2"]))
            support, blocks, unit_omega = self.unit_omega_for_edge(gates, edge)
            epsilon = float(record["epsilon_rad_ns"])
            omega = epsilon * unit_omega
            identity = torch.eye(omega.shape[-1], dtype=self.dtype, device=self.device)
            operator = (
                identity + omega if scheme == "dyson-1" else torch.matrix_exp(omega)
            )
            compiled.append(
                CompiledResidualOperator(
                    layer=int(layer_index),
                    edge=edge,
                    support=support,
                    control_blocks=blocks,
                    epsilon_rad_ns=epsilon,
                    unit_omega=unit_omega,
                    omega=omega,
                    operator=operator,
                    scheme=scheme,
                )
            )
        return compiled

    def compile_circuit(
        self,
        layers: Sequence[Sequence[dict[str, Any]]],
        residual_edges: Iterable[dict[str, Any]],
        *,
        scheme: str = "magnus-1",
    ) -> list[list[CompiledResidualOperator]]:
        edges = list(residual_edges)
        return [
            self.compile_layer(index, gates, edges, scheme=scheme)
            for index, gates in enumerate(layers)
        ]

    def save_operator_library(
        self,
        path: str | Path,
        circuit_kernels: Sequence[Sequence[CompiledResidualOperator]],
    ) -> None:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "kind": "pulse_resolved_interaction_picture_residual_library",
            "n_qubits": self.n_qubits,
            "integration_steps": self.integration_steps,
            "layer_duration_ns": LAYER_DURATION_NS,
            "projected_qubits": self.projected_qubits,
            "layers": [
                [
                    {
                        **kernel.metadata(),
                        "unit_omega": kernel.unit_omega.detach().cpu(),
                        "omega": kernel.omega.detach().cpu(),
                        "operator": kernel.operator.detach().cpu(),
                    }
                    for kernel in layer
                ]
                for layer in circuit_kernels
            ],
        }
        temporary = target.with_name(target.name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(target)


def apply_operator_to_state(
    states: torch.Tensor,
    operator: torch.Tensor,
    support: Sequence[int],
    *,
    n_qubits: int,
    local_dim: int = 2,
) -> torch.Tensor:
    """Apply a 2/3/4-site operator directly to full-state tensor axes."""
    sites = tuple(map(int, support))
    if len(set(sites)) != len(sites):
        raise ValueError("operator support contains duplicate sites")
    if any(site < 0 or site >= n_qubits for site in sites):
        raise ValueError("operator support lies outside the state")
    expected = local_dim ** len(sites)
    if tuple(operator.shape[-2:]) != (expected, expected):
        raise ValueError("operator shape does not match support")
    if states.shape[-1] != local_dim**n_qubits:
        raise ValueError("state dimension does not match n_qubits")
    rest = tuple(site for site in range(n_qubits) if site not in sites)
    order = sites + rest
    inverse = [order.index(site) for site in range(n_qubits)]
    leading = states.shape[:-1]
    tensor = states.reshape(*leading, *([local_dim] * n_qubits))
    offset = len(leading)
    tensor = tensor.permute(*range(offset), *(offset + site for site in order)).reshape(
        *leading, expected, -1
    )
    transformed = torch.einsum("ab,...bc->...ac", operator, tensor)
    transformed = transformed.reshape(*leading, *([local_dim] * n_qubits)).permute(
        *range(offset), *(offset + site for site in inverse)
    )
    return transformed.reshape(*leading, local_dim**n_qubits)


def load_operator_library(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> tuple[list[list[CompiledResidualOperator]], dict[str, Any]]:
    """Load a precomputed library without repeating pulse integration."""
    source = Path(path).expanduser().resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("kind") != "pulse_resolved_interaction_picture_residual_library":
        raise ValueError(f"{source} is not a residual Magnus operator library")
    target_device = torch.device(device)
    layers: list[list[CompiledResidualOperator]] = []
    for layer_records in payload["layers"]:
        layer: list[CompiledResidualOperator] = []
        for record in layer_records:
            tensor_dtype = dtype or record["operator"].dtype

            def move(name: str) -> torch.Tensor:
                return (
                    record[name]
                    .to(device=target_device, dtype=tensor_dtype)
                    .contiguous()
                )

            blocks = tuple(tuple(map(int, block)) for block in record["control_blocks"])
            layer.append(
                CompiledResidualOperator(
                    layer=int(record["layer"]),
                    edge=tuple(map(int, record["edge"])),
                    support=tuple(map(int, record["support"])),
                    control_blocks=(blocks[0], blocks[1]),
                    epsilon_rad_ns=float(record["epsilon_rad_ns"]),
                    unit_omega=move("unit_omega"),
                    omega=move("omega"),
                    operator=move("operator"),
                    scheme=str(record["scheme"]),
                )
            )
        layers.append(layer)
    metadata = {key: value for key, value in payload.items() if key != "layers"}
    metadata["path"] = str(source)
    return layers, metadata


def apply_residual_layer_to_state(
    states: torch.Tensor,
    kernels: Sequence[CompiledResidualOperator],
    *,
    n_qubits: int,
) -> torch.Tensor:
    """Apply a cached residual layer to full states with no SWAP routing."""
    result = states
    for kernel in kernels:
        result = apply_operator_to_state(
            result,
            kernel.operator,
            kernel.support,
            n_qubits=n_qubits,
            local_dim=2,
        )
    return result


def apply_background_layer_to_state(
    states: torch.Tensor,
    kernels: Sequence[CompiledResidualOperator],
    intended_gates: Sequence[dict[str, Any]],
    *,
    n_qubits: int,
) -> torch.Tensor:
    """Apply U_0 K_res,I to full-state branches using a cached library."""
    result = apply_residual_layer_to_state(states, kernels, n_qubits=n_qubits)
    for gate in intended_gates:
        result = apply_operator_to_state(
            result,
            gate["ideal_unitary"],
            gate["qubits"],
            n_qubits=n_qubits,
            local_dim=2,
        )
    return result
