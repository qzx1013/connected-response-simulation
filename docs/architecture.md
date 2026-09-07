# Architecture and restart rules

`cli` reads packaged JSON specifications. `plan` and `validate` use only the standard library. `studies` starts isolated per-stage Python workers; numerical modules are imported by package name, without adding private folders to `sys.path` or searching for legacy code. An installed wheel contains every project-specific Python dependency and the physical configuration files.

`backend` contains local channel construction, dressed Kraus operations, the pulse-resolved residual Magnus library, direct full-state operations and the shared BP/gloop engine. The low-level engine is intentionally kept together in this release to limit changes to the validated contraction conventions. The batched-bra opposite-cavity contraction is now an explicit parameter of the shared helper, replacing runtime source-string rewriting.

`peps.model` builds local gates/kernels and a content identity, without allocating 2^N amplitudes. `DynamicEvolution` applies residual and intended operations and performs one cold norm-BP solve for each dynamic compression step. No unused post-compression solve is added to propagated source branches. Clean-cache normalizations and terminal readout still require their own appropriate solves.

`peps.source_bank` caches all coherent layers on CPU. Each branch keeps its own actual virtual dimensions. Gate/MPO kernels retain their branch-local order; norm BP temporarily zero-pads compatible banks, solves independently per branch, then crops messages before constructing projectors. The fixed source-wave size is 128; changing the execution BP batch cap does not change the membership of an existing checkpoint wave.

Pauli terminal reads run on one stream and adapt branch batches using observed peaks and admission estimates. A known allocator or contraction-planner memory failure discards only the failed read and retries smaller groups in the same order. At one branch, a memory failure stops rather than lowering numerical tolerances silently. Cache release is reserved for recovery instead of every successful tile.

## Persistent state

- `background_layers.pt`: immutable per-run coherent histories, clean messages and source normalizers.
- `evolved_source_SS_wave_WW.pt`: atomic active-wave checkpoint after each propagated layer.
- `read_progress_source_SS_wave_WW.pt`: atomic completed terminal reads for an active wave.
- `source_SS.json`: permanent committed contribution, mode accounting and readout resource summary.
- `through_source_SS.json`, `result.json`: additive assembly from committed source rows, never an increment applied twice.
- Fidelity controls use target checkpoints and continue from the last committed target, with shared BP snapshots as needed.
- Figure 2 cases and Figure 4 logical Monte Carlo batches have separate atomic checkpoints. MC replay restores the exact RNG state and weighted samples.

Transient source-wave files are deleted only after the source JSON and assembly have committed. A crash leaves the active source resumable; completed sources are never recomputed. No blanket cleanup of unrelated output directories is performed.

Physical/model identity includes the geometry manifest hash, actual edge list, gate-table hash, Kraus hash, kernel scheme and numerical settings. Code identity covers the packaged Python/config files. Incompatible checkpoints are rejected. Relocating an unchanged checkout and output tree does not change these content identities. Moving caches from the historical script releases into the reorganized package is **not** automatic: the pickled class paths and code identities differ. Keep the original frozen releases for resuming jobs that were already launched there.

Output directories have an advisory Linux process lock. Two processes cannot write the same run concurrently; a crash releases the lock automatically. Default contraction-tree persistence lives under the user's standard cache directory, not inside installed source code. `TANGENT_BP_BOUNDED_2X2_TREE_CACHE` can select a path or disable this optional performance cache; it is not a private dependency.

## Memory defaults

The 80 GB profile caps PyTorch allocation at the smaller of 76 GiB and reported device capacity minus 4 GiB. Gloop planning uses a 72 GiB peak budget, 24 GiB intermediate target, explicit safety factor and exact slicing. Admission estimates leave additional headroom and reads begin at batch one. These controls reduce risk but do not constitute a measured 36-qubit peak guarantee. A fresh full run remains necessary on the target GPU.

The runner neither controls other users' GPU processes nor waits for unrelated jobs. Start it manually after the existing computations finish.
