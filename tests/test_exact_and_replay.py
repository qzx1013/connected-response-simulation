from pathlib import Path
from types import SimpleNamespace
import math
import numpy as np
import torch
from residual_tn.worker import install_mc_cache


def test_mc_checkpoint_replays_samples_and_rng(tmp_path):
    def sample(**kw):
        value = torch.rand(kw["batch_size"], 3, generator=kw["generator"]).numpy()
        return value, {"norm": float(value.max())}, value.copy()

    fake = SimpleNamespace(
        synchronize=lambda _: None,
        run_mc_batch=sample,
        run_qutrit_monte_carlo=lambda **kw: None,
    )
    install_mc_cache(fake, tmp_path)
    rng = torch.Generator().manual_seed(17)
    expected = [
        fake.run_mc_batch(batch_size=3, generator=rng, ideal_history=torch.ones(1))
        for _ in range(2)
    ]
    final = rng.get_state().clone()

    def unexpected(**kw):
        raise AssertionError("Replayed batch ran again")

    replay = SimpleNamespace(
        synchronize=lambda _: None,
        run_mc_batch=unexpected,
        run_qutrit_monte_carlo=lambda **kw: None,
    )
    install_mc_cache(replay, tmp_path)
    other = torch.Generator().manual_seed(17)
    for reference in expected:
        observed = replay.run_mc_batch(
            batch_size=3, generator=other, ideal_history=torch.ones(1)
        )
        assert np.array_equal(observed[0], reference[0]) and np.array_equal(
            observed[2], reference[2]
        )
        assert observed[1] == reference[1]
    assert torch.equal(final, other.get_state())


def test_fig3_chunk_preallocation_preserves_input(monkeypatch):
    from residual_tn.experiments import fig3

    dtype = torch.complex128
    context = SimpleNamespace(
        n_qubits=4,
        layers=[[dict(qubits=(0, 1), ideal_unitary=torch.eye(4, dtype=dtype))]],
    )
    torch.manual_seed(91)
    kernel = SimpleNamespace(
        support=(0, 3), operator=torch.linalg.qr(torch.randn(4, 4, dtype=dtype))[0]
    )
    monkeypatch.setattr(fig3, "_CURRENT_CTX", context)
    monkeypatch.setattr(fig3, "_CURRENT_KERNELS", [[kernel]])
    states = torch.randn(11, 16, dtype=dtype)
    before = states.clone()
    observed = fig3.propagate_background(states, 0, 0, {}, 3)
    reference = torch.cat(
        [
            fig3.propagate_background(states[i : i + 3], 0, 0, {}, 100)
            for i in range(0, 11, 3)
        ]
    )
    assert (
        torch.equal(before, states)
        and float((observed - reference).abs().max()) < 1e-13
    )


def test_bundled_reference_has_no_private_class_dependency():
    inputs = Path(__file__).resolve().parents[1] / "inputs"
    reference = torch.load(
        inputs / "0522_reference_data.pt", map_location="cpu", weights_only=False
    )
    assert reference["N_QUBITS"] == 16 and "pattern" in reference
    assert isinstance(
        torch.load(inputs / "qutrit_kraus.pt", map_location="cpu", weights_only=False),
        dict,
    )


def test_exact_flat_branch_modes_match_matrix_reference(monkeypatch):
    from residual_tn.backend.context import FidelityContext
    from residual_tn.backend.exact_small import exact_compute_fidelity

    ctx = FidelityContext(2, 2, 1, torch.device("cpu"))
    eye = torch.eye(2, dtype=torch.complex128)
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    h = torch.tensor(
        [[math.cos(0.3), -math.sin(0.3)], [math.sin(0.3), math.cos(0.3)]],
        dtype=torch.complex128,
    )
    for layer in range(3):
        gates = []
        for site in range(2):
            u = h if layer == 0 else eye
            noise = torch.stack([0.99**0.5 * u, 0.01**0.5 * x @ u])
            gates.append(
                dict(
                    gate_idx=layer * 2 + site,
                    qubits=(site,),
                    ideal_unitary=u,
                    kraus_ops=noise,
                )
            )
        ctx.add_layer(gates)
    monkeypatch.setenv("PEPS_CORR2BP_PAIR_FORMULA", "centered_C")
    for masks in ({}, dict(source_mode_masks={0: [1]}, target_mode_masks={4: [1]})):
        flat = exact_compute_fidelity(ctx, **masks)
        matrix = exact_compute_fidelity(ctx, store_pair_C=True, **masks)
        assert flat.pair_deltas.keys() == matrix.pair_deltas.keys()
        assert (
            max(abs(v - matrix.pair_deltas[k]) for k, v in flat.pair_deltas.items())
            < 1e-12
        )


def test_qutrit_cpu_channel_self_test():
    from residual_tn.experiments.fig4 import self_test

    self_test(SimpleNamespace(device="cpu"))
