"""Numerical tests for the pulse-resolved residual Magnus library."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import torch


from residual_tn.backend import residual_magnus as rm


def single_layer(modes: list[int], device: torch.device) -> list[dict]:
    identity = torch.eye(2, dtype=torch.complex128, device=device)
    return [
        {
            "qubits": (site,),
            "mode": mode,
            "control_kind": "single_qutrit_drag",
            "ideal_unitary": identity,
        }
        for site, mode in enumerate(modes)
    ]


def two_qubit_layer(device: torch.device) -> list[dict]:
    identity = torch.eye(4, dtype=torch.complex128, device=device)
    return [
        {
            "qubits": (0, 1),
            "control_kind": "two_qutrit_exchange",
            "ideal_unitary": identity,
        },
        {
            "qubits": (2, 3),
            "control_kind": "two_qutrit_exchange",
            "ideal_unitary": identity,
        },
    ]


class ResidualMagnusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cpu")
        cls.library = rm.ResidualMagnusLibrary(
            4,
            integration_steps=128,
            dtype=torch.complex128,
            device=cls.device,
            projected_qubits=True,
        )
        cls.edge = [{"q1": 0, "q2": 3, "epsilon_rad_ns": 0.0023}]

    def test_single_qubit_layer_has_two_site_support(self) -> None:
        kernels = self.library.compile_layer(
            0,
            single_layer([0, 1, 2, 3], self.device),
            self.edge,
            scheme="magnus-1",
        )
        kernel = kernels[0]
        self.assertEqual(kernel.support, (0, 3))
        self.assertEqual(tuple(kernel.operator.shape), (4, 4))

    def test_two_qubit_layer_has_four_site_support(self) -> None:
        kernel = self.library.compile_layer(
            1, two_qubit_layer(self.device), self.edge, scheme="magnus-1"
        )[0]
        self.assertEqual(kernel.support, (0, 1, 2, 3))
        self.assertEqual(tuple(kernel.operator.shape), (16, 16))
        antihermitian_error = torch.max(torch.abs(kernel.omega + kernel.omega.conj().T))
        identity = torch.eye(16, dtype=kernel.operator.dtype, device=self.device)
        unitarity_error = torch.max(
            torch.abs(kernel.operator.conj().T @ kernel.operator - identity)
        )
        self.assertLess(float(antihermitian_error.item()), 1e-12)
        self.assertLess(float(unitarity_error.item()), 1e-10)

    def test_one_active_block_has_three_site_support(self) -> None:
        identity = torch.eye(4, dtype=torch.complex128, device=self.device)
        kernel = self.library.compile_layer(
            1,
            [{"qubits": (0, 1), "ideal_unitary": identity}],
            self.edge,
            scheme="magnus-1",
        )[0]
        self.assertEqual(kernel.support, (0, 1, 3))
        self.assertEqual(tuple(kernel.operator.shape), (8, 8))

    def test_idle_layer_reduces_to_integrated_exchange(self) -> None:
        kernel = self.library.compile_layer(
            0,
            single_layer([3, 3, 3, 3], self.device),
            self.edge,
            scheme="magnus-1",
        )[0]
        exchange = torch.zeros((4, 4), dtype=torch.complex128, device=self.device)
        exchange[1, 2] = 1.0
        exchange[2, 1] = 1.0
        expected = -1j * rm.LAYER_DURATION_NS * exchange
        self.assertTrue(torch.allclose(kernel.unit_omega, expected, atol=2e-10))

    def test_full_state_application_preserves_norm_for_magnus(self) -> None:
        kernel = self.library.compile_layer(
            1, two_qubit_layer(self.device), self.edge, scheme="magnus-1"
        )[0]
        generator = torch.Generator(device=self.device)
        generator.manual_seed(7)
        state = torch.randn(
            16,
            dtype=torch.complex128,
            device=self.device,
            generator=generator,
        )
        state = state / torch.linalg.vector_norm(state)
        result = rm.apply_operator_to_state(
            state, kernel.operator, kernel.support, n_qubits=4
        )
        self.assertTrue(
            torch.allclose(
                torch.linalg.vector_norm(result),
                torch.ones((), dtype=torch.float64, device=self.device),
                atol=1e-10,
            )
        )

    def test_saved_library_loads_without_reintegration(self) -> None:
        kernels = [
            self.library.compile_layer(
                1,
                two_qubit_layer(self.device),
                self.edge,
                scheme="magnus-1",
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "residual_library.pt"
            self.library.save_operator_library(path, kernels)
            loaded, metadata = rm.load_operator_library(
                path, device=self.device, dtype=torch.complex128
            )
        self.assertEqual(
            metadata["kind"],
            "pulse_resolved_interaction_picture_residual_library",
        )
        self.assertEqual(loaded[0][0].support, (0, 1, 2, 3))
        self.assertTrue(
            torch.allclose(
                loaded[0][0].operator,
                kernels[0][0].operator,
                atol=0.0,
                rtol=0.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
