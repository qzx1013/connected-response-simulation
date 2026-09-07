"""Independent dense branch references for the all-prefix fidelity schedule."""

from itertools import combinations, product
from types import SimpleNamespace
import math
import torch
import pytest
from residual_tn.backend import residual_background_exact as bg
from residual_tn.backend.residual_magnus import apply_operator_to_state


def fixture_context(seed=12, residual=True, dyson=False):
    rng = torch.Generator().manual_seed(seed)
    dtype = torch.complex128

    def unitary(size):
        h = torch.randn(size, size, generator=rng, dtype=dtype)
        return torch.matrix_exp(0.18j * (h + h.conj().T))

    layers, kernels = [], []
    gate_id = 13
    for layer in range(4):
        supports = [(i,) for i in range(4)] if layer % 2 == 0 else [(2, 0), (1, 3)]
        gates = []
        for sites in supports:
            size = 2 ** len(sites)
            u = unitary(size)
            noise = torch.diag(torch.tensor([1, -1] * (size // 2), dtype=dtype))
            # Trace-changing Kraus map with unequal mode ranks across gates.
            weights = [0.982, 0.007] if gate_id % 2 else [0.977, 0.008, 0.003]
            modes = [math.sqrt(weights[0]) * u, math.sqrt(weights[1]) * noise @ u]
            if len(weights) == 3:
                modes.append(math.sqrt(weights[2]) * unitary(size) @ u)
            gates.append(
                dict(
                    gate_idx=gate_id,
                    qubits=sites,
                    ideal_unitary=u,
                    kraus_ops=torch.stack(modes),
                )
            )
            gate_id += 3
        layers.append(gates)
        sites = (3, 0) if layer % 2 == 0 else (3, 1, 0)
        size = 2 ** len(sites)
        h = torch.randn(size, size, generator=rng, dtype=dtype)
        omega = 0.01j * (h + h.conj().T)
        operator = (
            torch.eye(size, dtype=dtype) + omega if dyson else torch.matrix_exp(omega)
        )
        kernels.append(
            [SimpleNamespace(support=sites, operator=operator)] if residual else []
        )
    return SimpleNamespace(
        n_qubits=4, device=torch.device("cpu"), dtype=dtype, layers=layers
    ), kernels


def dense_enumeration(ctx, kernels):
    """Propagate each inserted mode combination to each actual target."""
    result = {
        k: [1.0]
        for k in [
            "zero_insertion_fidelity",
            "fidelity_first_order",
            "fidelity_second_order",
        ]
    }
    for depth in range(1, len(ctx.layers) + 1):
        target = torch.zeros(16, dtype=ctx.dtype)
        target[0] = 1
        gates = [g for layer in ctx.layers[:depth] for g in layer]
        for g in gates:
            target = apply_operator_to_state(
                target, g["ideal_unitary"], g["qubits"], n_qubits=4
            )

        def response(selected):
            value = 0.0
            for ids in product(*(range(len(g["kraus_ops"])) for g in selected)):
                source = {
                    g["gate_idx"]: (g["kraus_ops"][mode] @ g["ideal_unitary"].conj().T)
                    for g, mode in zip(selected, ids)
                }
                state = torch.zeros(16, dtype=ctx.dtype)
                state[0] = 1
                for index, layer in enumerate(ctx.layers[:depth]):
                    for kernel in kernels[index]:
                        state = apply_operator_to_state(
                            state, kernel.operator, kernel.support, n_qubits=4
                        )
                    for g in layer:
                        state = apply_operator_to_state(
                            state, g["ideal_unitary"], g["qubits"], n_qubits=4
                        )
                    for g in layer:
                        if g["gate_idx"] in source:
                            state = apply_operator_to_state(
                                state, source[g["gate_idx"]], g["qubits"], n_qubits=4
                            )
                value += float(torch.vdot(target, state).abs().square())
            return value

        f0 = response([])
        singles = {g["gate_idx"]: response([g]) for g in gates}
        c1 = sum(math.log(x / f0) for x in singles.values())
        delta = sum(
            response([a, b]) * f0 / (singles[a["gate_idx"]] * singles[b["gate_idx"]])
            - 1
            for a, b in combinations(gates, 2)
        )
        result["zero_insertion_fidelity"].append(f0)
        result["fidelity_first_order"].append(f0 * math.exp(c1))
        result["fidelity_second_order"].append(f0 * math.exp(c1 + delta))
    result["final_pair_count"] = len(gates) * (len(gates) - 1) // 2
    return result


@pytest.mark.parametrize(
    "residual,dyson", [(False, False), (True, False), (True, True)]
)
def test_shared_prefix_matches_independent_dense_enumeration(residual, dyson):
    ctx, kernels = fixture_context(residual=residual, dyson=dyson)
    expected = dense_enumeration(ctx, kernels)
    observed = bg.connected_fidelity_trajectory(ctx, kernels)
    for key in expected:
        torch.testing.assert_close(
            torch.as_tensor(observed[key], dtype=torch.float64),
            torch.as_tensor(expected[key], dtype=torch.float64),
            rtol=1e-11,
            atol=1e-12,
        )


def test_shared_prefix_preserves_inputs_and_at_depth_reader():
    ctx, kernels = fixture_context()
    before = [[g["kraus_ops"].clone() for g in layer] for layer in ctx.layers]
    a = bg.connected_fidelity_trajectory(ctx, kernels)
    b = bg.connected_fidelity_trajectory(ctx, kernels)
    assert a == b
    for layer, saved in zip(ctx.layers, before):
        for g, k in zip(layer, saved):
            assert torch.equal(g["kraus_ops"], k)
    for depth in range(1, 5):
        ref = bg.connected_fidelity_at_depth(
            ctx, kernels, depth=depth, store_responses=True
        )
        for key in [
            "zero_insertion_fidelity",
            "fidelity_first_order",
            "fidelity_second_order",
        ]:
            assert abs(a[key][depth] - ref[key]) < 1e-12
        n = sum(len(x) for x in ctx.layers[:depth])
        assert len(ref["two_location_fidelity"]) == n * (n - 1) // 2


def test_single_prefix_and_nonpositive_fidelity():
    ctx, kernels = fixture_context()
    ctx.layers = ctx.layers[:1]
    kernels = kernels[:1]
    assert bg.connected_fidelity_trajectory(ctx, kernels)["final_pair_count"] == 6
    ctx.layers[0][0]["kraus_ops"].zero_()
    with pytest.raises(FloatingPointError, match="one-location"):
        bg.connected_fidelity_trajectory(ctx, kernels)
