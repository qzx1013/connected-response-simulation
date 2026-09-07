# Release validation

The final package was built as a standard wheel, installed **without editable mode**, and imported from an isolated environment's `site-packages`. Commands were also invoked from an otherwise empty directory with Python's isolated mode (`-I`). The original source workspace was not on the runtime import path.

Environment: Python 3.12.14, Torch 2.6.0+cpu. Exact public package versions are in `requirements-tested.txt`. `pip check` reported no broken requirements. Static checks for undefined/local-before-assignment names passed. The import audit covers deferred imports as well as imports executed by the tests.

**27 tests passed in 22.19 seconds.** These are CPU checks, not production A100 benchmarks.

| Check | Outcome |
|---|---|
| Packaged Fig2–6 entry points and modules | Import successfully without private path bootstrapping |
| Residual geometries | 8q/3, 16q/4, 20q/4, 36q/7; diagonal and endpoint-disjoint |
| Existing 8q, 16q and 20q edge records | Exactly preserved, including strengths |
| New 36-qubit local model | Two layers at four integration steps construct successfully without large-state allocation |
| Pulse-resolved residual operators | Small-system invariants and direct-state comparisons pass |
| Four-body residual MPO application | Matches direct full-state application in the regression fixtures |
| Ragged norm BP versus separate solves | Maximum message discrepancy below 1e-10 |
| Padded gloop reads versus separate reads | Maximum discrepancy below 1e-10 |
| Full-cover gloop versus exact Pauli | Maximum discrepancy below 1e-9 |
| Allocator/planner memory failures | Recursive splits preserve branch order and results |
| BP continuation versus independently stopped solves | Message discrepancies below 2e-11; stopping iterations match |
| Four-qubit, four-layer source-0 Pauli end-to-end comparison | Maximum absolute error 3.1008529077780622e-12 with strict toy compression settings |
| Completed Pauli source replay | No recomputation; identical assembled raw results |
| Fidelity target interruption/resume and shared normalizers | Maximum δ_ab difference from the independent baseline 2.220446049250313e-16 |
| MC checkpoint replay | Identical weighted samples and restored RNG state |
| Fig3 chunked propagation | Preserves inputs and agrees with concatenated independent chunks |
| Exact flat/matrix reference paths, including mode masks | Agree within 1e-12 |
| Included input tensor files | Load without private Python classes; original SHA-256 checks pass |
| Fig6 plotting | Synthetic fixture renders PNG/PDF/SVG; incomplete sources are rejected |

The four-qubit tests are deliberately small and include strict tolerances for checking implementation equivalence. Their tiny errors must not be quoted as accuracy estimates for the 20- or 36-qubit production circuits. The synthetic plotting fixture is not distributed as scientific result data.

The shared backend was consolidated with explicit package imports. Redundant serial/cavity launch drivers and the duplicate full-pipeline tree were removed. The batched-bra contraction previously assembled through source rewriting now uses an explicit argument. A dormant recursive full-state sweep helper was removed and the retained flat path's source-mode indexing was repaired and tested. These edits are confined to the new repository; the frozen local Fig2–4 release passed its original source-hash checks unchanged.

The new Figure 6 production evolution/readout has **not** been run. Its runtime, peak GPU memory and large-system convergence remain unmeasured. Existing server calculations were neither modified nor queried during this preparation. The machine-readable record is `provenance/validation.json`.

## Figure 4 all-prefix full-state optimization (2026-09-07)

The exact projected-qubit fidelity trajectory now shares each source bank's
forward propagation across terminal prefixes. Each prefix retains its own
intended-control target, adjoint effect, F0 and one-/two-location responses.
Pair-mode reductions remain on the device. No Kraus mode or response pair is
dropped. `connected_fidelity_at_depth` still exposes the individual-depth
reference reader and raw responses.

On the same A100-SXM4-80GB, two interleaved comparisons using the original
16-qubit, 20-layer input and four-edge P1000 Magnus-1 residual gave baseline
times 49.68114/52.95833 s and optimized times
4.20436/4.85022 s (ratio of means 11.336). All 21
prefix points and all 26,335 final-depth response pairs are included; kernel
compilation is excluded for both cases. The maximum F2 discrepancy from the
retained production curve is 1.7943e-12.
Peak allocated CUDA memory fell from 7.5292 to 6.5838 GiB; allocator-reserved
memory was approximately 31.6 GiB in both schedules. Timings reflect this
server session and are not a universal performance guarantee.

The final packaged source passed 15 related CPU tests (pytest 8.4.2,
Torch 2.6.0+cu124); the unrelated bundled-input test was
excluded from the isolated source staging directory. New independent dense
enumeration checks cover zero residual, unitary residual and a nonunitary
Dyson kernel, unsorted supports, unequal mode ranks, all source pairs, input
preservation and invalid fidelities. The formal Fig4 entry point also ran on
the production 16-qubit input and agreed within
1.7936e-12.

The detailed machine-readable record is
`provenance/fig4_fullstate_optimization_20260907.json`. This benchmark changes
only the F2 evaluation schedule. Figure 4 now uses the mean optimized time
(4.52729 s) and the validated optimized fidelity curve. The plotting payload
records the timing provenance, retains the MC samples and reads both measured
costs directly; the MC/F2 online-time ratio is 1359.35. This focused verification
does not repeat the earlier full-wheel audit.
