# What each figure tests

| Figure | Computation | Scope |
|---|---|---|
| 2 | 8-qubit Liouville/full-state and local Pauli comparisons | 3 sparse residual pairs; original circuit/noise ensembles retained |
| 3 | 16-qubit connected ensemble | 4 sparse residual pairs; complete distinct-layer C3, with the existing same-layer exclusions |
| 4 | 16-qubit full-state response versus qutrit trajectories | 4 sparse residual pairs; 4096 fixed-proposal trajectories, physical batch 8 and exact RNG/sample replay |
| 5 a–c | Full-state versus PEPS Pauli | All 20 source layers and 60 terminal local Pauli components |
| 5 d–f | Source-0 fidelity-pair δ_ab | All source-0 gates/modes, three outcome-independent sampled target gates per layer, ordered b > a |
| 6 (printed manuscript) | Historical zero-residual 36-qubit PEPS Pauli and resources | All 20 source layers and 108 terminal components; 20.38 aggregate GPU-hours; no exact 36-qubit reference |
| `fig6` command | Seven-edge residual 36-qubit extension | Current Figure 5 algorithm; no complete extension result reported in the manuscript |

## Coherent background and insertions

Residual coupling is part of the coherent background. Each layer applies its compiled interaction-picture residual operator followed by the intended control layer. Kernels are compiled once per circuit and reused across response branches. Kraus channels are projected from the same local qutrit model and dressed as E = K U†.

These E operators represent the **complete relative channel**, not ΔE = E − I. Pauli assembly sums the unnormalized Kraus effects for each gate and subtracts the common background contribution once per gate. It does not subtract once per mode. The reported additive conditional observable is the ratio after this additive probability assembly. The strict linearized conditional ratio is also saved under a distinct result key.

Figure 5 d–f use δ_ab = M_ab / (M_a M_b) − 1 relative to the **normalized residual-dressed coherent background**. They do not measure residual-induced loss relative to the intended control-only final state. The Figure 2–4 fidelity experiments retain their intended-control target convention and F0 normalization. These target conventions must remain explicit when combining the figures in the manuscript.

## Figure 5 controls

- d: compression tolerances 1e-3, 1e-4, 1e-5 affect the full PEPS pipeline. Each point constructs and caches its own background and response branches.
- e: readout mixed-BP tolerances 1e-8, 1e-10, 1e-12; cap 500. Background, source propagation, compression and one-insertion normalizers are fixed.
- f: readout mixed-BP iteration caps 250, 500, 1000; tolerance 1e-10. The same background/evolution policy as e applies.

The e/f jobs share the same baseline tensor history in one physical cache file with separate panel descriptors. They share a single source evolution and a continued BP trajectory at each target. Every criterion saves each branch at its first converged check or its cap. Identical message snapshots share a gloop contraction. All pair reads and their one-insertion normalizations use gloop size 8. Kraus cutoff is fixed and matched between full-state and PEPS; neither Kraus nor gloop size is scanned in d–f.

Flat e/f curves demonstrate saturation of those readout stopping criteria in this sample, not exactness of the whole PEPS approximation. The sample contains near-zero correlations, so MAE alone is insufficient; the renderer reports P95, maximum, signed sample-sum error and absolute sample-error sum as well. Unweighted sample sums are not a complete C2 or fidelity estimate. No large-qubit accuracy bound is inferred.

The Figure 3 cone diagnostic is the original **intended-control support cone**. Its construction does not include propagation through residual kernels; do not label it an exact causal cone of the full Hamiltonian.

## Figure 6 reporting

The printed manuscript retains the historical zero-residual Figure 6. Its data, source partitions and measured 20.38-hour runtime remain the archived benchmark. Nonzero residual coupling is demonstrated in Figures 2–5; the manuscript does not claim a complete residual 36-qubit run or unchanged 36-qubit accuracy/runtime.

The `fig6` residual-extension command reuses the Figure 5 default PEPS settings: complex128, bond cap 64, dynamic weighted compression tolerance 1e-4, BP tolerance 1e-10, cap 500, damping 0.5, gloop size 8, Kraus cutoff 1e-8, and T1 = T2 = 5 μs. Residual MPO zip-up tolerance is separately fixed at 1e-3. These tolerances control different approximations and must not be conflated.

The renderer only reads one complete run under the seven-edge model. It does not reuse a no-residual q0, a fixed survival factor, old source partitions, old mode-count assumptions or another job's timing. Resource plots show recorded completed phase wall time, not integrated GPU-kernel time or total elapsed calendar time. Peak memory is PyTorch allocated/reserved memory; it excludes the CUDA driver and other processes.

A complete nonzero-residual 36-qubit result and its production runtime/peak memory are not reported here. The default residual-extension renderer is therefore not the provenance of the printed historical Figure 6.
