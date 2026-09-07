"""
context.py: Defines input data structures and output result type.
Explicitly defines qubit mapping, layer index, gate index, tensor shapes.
No physical calculations.
"""

from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import torch


class FidelityContext:
    def __init__(
        self,
        n_qubits: int,
        width: int,
        length: int,
        device: torch.device,
        dtype: torch.dtype = torch.complex128,
    ):
        self.n_qubits = n_qubits
        self.width = width
        self.length = length
        self.device = device
        self.dtype = dtype

        # Gates are organized by layer. Each layer is a list of dicts:
        # {
        #    'gate_idx': int,
        #    'qubits': Tuple[int, ...],
        #    'ideal_unitary': torch.Tensor,
        #    'kraus_ops': torch.Tensor
        # }
        self.layers: List[List[Dict[str, Any]]] = []

        # Mapping from gate_idx to (layer_idx, index_in_layer)
        self.gate_map: Dict[int, Tuple[int, int]] = {}

    def add_layer(self, gates: List[Dict[str, Any]]):
        layer_idx = len(self.layers)
        self.layers.append(gates)
        for i, g in enumerate(gates):
            self.gate_map[g["gate_idx"]] = (layer_idx, i)


@dataclass
class FidelityResult:
    """Output of the centered lightcone fidelity computation.

    `pair_C` is a historical field name.  It stores `A_vu` when
    diagnostics["pair_formula"] == "raw_A" and `C_vu` when
    diagnostics["pair_formula"] == "centered_C".
    """

    log_fidelity_first_order: float
    fidelity_first_order: float
    log_fidelity_connected: float
    fidelity_connected: float
    local_fidelities: Dict[int, float]
    pair_deltas: Dict[tuple, float]
    pair_C: Dict[tuple, torch.Tensor]
    self_energy_by_layer: Dict[
        int, float
    ]  # per-layer sum of Deltas for target in that layer
    self_energy_by_source_layer: Dict[
        int, float
    ]  # per-layer sum of Deltas for source in that layer
    diagnostics: Dict[str, Any]

    def summary(self) -> str:
        lines = []
        lines.append(f"F1          = {self.fidelity_first_order:.12e}")
        lines.append(f"log_F1      = {self.log_fidelity_first_order:.12f}")
        lines.append(f"F_connected = {self.fidelity_connected:.12e}")
        lines.append(f"log_Fc      = {self.log_fidelity_connected:.12f}")
        lines.append(f"# pairs     = {len(self.pair_deltas)}")
        for (a, b), delta in sorted(self.pair_deltas.items(), key=lambda x: x[0]):
            lines.append(f"  Delta[{a},{b}] = {delta:+.6e}")
        if self.self_energy_by_layer:
            lines.append(f"Self-energy by target layer:")
            for lay, se in sorted(self.self_energy_by_layer.items()):
                lines.append(f"  layer {lay}: sum Delta = {se:+.6e}")
        return "\n".join(lines)
