"""Small exact tests for the residual-dressed reusable PEPS."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


from residual_tn.backend import (
    common_background_peps as common,
    residual_magnus as residual,
)


DTYPE = torch.complex128


class LinearContext:
    width = 4
    length = 1

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.dtype = DTYPE


def dense_linear_state(tensors: dict[int, torch.Tensor]) -> torch.Tensor:
    q0 = tensors[0][0, 0, 0, 0, :, :]
    q1 = tensors[1][0, 0, 0, :, :, :]
    q2 = tensors[2][0, 0, 0, :, :, :]
    q3 = tensors[3][0, 0, 0, :, 0, :]
    return torch.einsum("ap,abq,bcr,cs->pqrs", q0, q1, q2, q3).reshape(-1)


def product_preparation(device: torch.device) -> list[torch.Tensor]:
    vectors = [
        (1.0, 0.2 + 0.1j),
        (0.7 - 0.1j, 0.4j),
        (0.6 + 0.2j, 0.5 - 0.1j),
        (0.3j, 0.8 + 0.1j),
    ]
    gates = []
    for values in vectors:
        column = torch.tensor(values, dtype=DTYPE, device=device)
        column = column / torch.linalg.vector_norm(column)
        a, b = column
        gates.append(torch.stack((column, torch.stack((-b.conj(), a.conj()))), dim=1))
    return gates


class CommonBackgroundPepsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cpu")

    def _prepared(self, chi_max: int):
        ctx = LinearContext(self.device)
        config = common.bp.SingleSiteConfig(
            chi_max=chi_max,
            two_qubit_apply_batch_size=32,
            two_qubit_layer_gate_batch_size=4,
            batch_disjoint_two_qubit_gates=False,
        )
        config.validate()
        neighbors = common.bp._build_neighbors(ctx.width, ctx.length)
        tensors = {
            site: value.unsqueeze(0)
            for site, value in common.bp._zero_peps(ctx).items()
        }
        for site, gate in enumerate(product_preparation(self.device)):
            common.bp._apply_operator_microbatched(
                tensors, (site,), gate, config=config, neighbors=neighbors
            )
        return tensors, config, neighbors

    def test_four_body_mpo_matches_direct_full_state(self) -> None:
        tensors, config, neighbors = self._prepared(128)
        before = dense_linear_state(tensors)
        identity = torch.eye(4, dtype=DTYPE, device=self.device)
        intended_layer = [
            {"qubits": (0, 1), "ideal_unitary": identity},
            {"qubits": (2, 3), "ideal_unitary": identity},
        ]
        library = residual.ResidualMagnusLibrary(
            4,
            integration_steps=128,
            dtype=DTYPE,
            device=self.device,
            projected_qubits=True,
        )
        record = {
            "q1": 0,
            "q2": 3,
            "epsilon_rad_ns": 0.0023,
            "swap_routing": {"path": [0, 1, 2, 3]},
        }
        kernel = library.compile_layer(1, intended_layer, [record], scheme="magnus-1")[
            0
        ]
        self.assertEqual(kernel.support, (0, 1, 2, 3))
        common.apply_compiled_residual_operator(
            tensors,
            kernel,
            record,
            config=config,
            neighbors=neighbors,
        )
        actual = dense_linear_state(tensors)
        expected = residual.apply_operator_to_state(
            before, kernel.operator, kernel.support, n_qubits=4
        )
        self.assertTrue(torch.allclose(actual, expected, atol=2e-10))

    def test_nonlocal_two_body_route_matches_direct_full_state(self) -> None:
        tensors, config, neighbors = self._prepared(64)
        before = dense_linear_state(tensors)
        identity = torch.eye(2, dtype=DTYPE, device=self.device)
        intended_layer = [
            {"qubits": (site,), "mode": 3, "ideal_unitary": identity}
            for site in range(4)
        ]
        library = residual.ResidualMagnusLibrary(
            4,
            integration_steps=64,
            dtype=DTYPE,
            device=self.device,
            projected_qubits=True,
        )
        record = {
            "q1": 0,
            "q2": 3,
            "epsilon_rad_ns": 0.0023,
            "swap_routing": {"path": [0, 1, 2, 3]},
        }
        kernel = library.compile_layer(0, intended_layer, [record], scheme="magnus-1")[
            0
        ]
        self.assertEqual(kernel.support, (0, 3))
        common.apply_compiled_residual_operator(
            tensors,
            kernel,
            record,
            config=config,
            neighbors=neighbors,
        )
        actual = dense_linear_state(tensors)
        expected = residual.apply_operator_to_state(
            before, kernel.operator, kernel.support, n_qubits=4
        )
        self.assertTrue(torch.allclose(actual, expected, atol=2e-10))


if __name__ == "__main__":
    unittest.main()
