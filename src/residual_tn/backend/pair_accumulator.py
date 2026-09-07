"""
pair_accumulator.py: Pair-level amplitude-matrix accumulation and Delta computation.

This module is responsible ONLY for:
  - Receiving one target-mode amplitude vector (shape (n_v,)) for a given
    (source gate, target gate, source mode u).
  - Assembling the selected pair amplitude matrix (n_v, n_u).
  - Computing Delta_ab via the configured pair formula when all source modes for a pair arrive.
  - Recording self-energy diagnostics (per-layer Delta sums).

It does NOT:
  - Define whether that matrix is bare A_vu or centered C_vu; that is set by
    the branch/target readout pipeline plus pair_formula.
  - Contract tensor networks.
  - Import PEPS, BP, branch metadata, or patch contraction modules.

Allowed imports:
  - torch
  - context.FidelityContext  (for gate_map lookups only)
  - fidelity_formula.delta_from_raw_A
  - fidelity_formula.delta_from_centered_C (explicit diagnostic only)
"""

import os
import torch
from typing import Any, Dict, Optional, Tuple
from residual_tn.backend.context import FidelityContext
from residual_tn.backend.fidelity_formula import delta_from_centered_C, delta_from_raw_A


def _complex_vector_as_pairs(x: torch.Tensor):
    """Return a JSON-safe list of [real, imag] entries for a complex vector."""
    flat = x.detach().cpu().reshape(-1)
    return [[float(v.real.item()), float(v.imag.item())] for v in flat]


def _complex_matrix_as_pairs(x: torch.Tensor):
    """Return a JSON-safe nested list of [real, imag] entries for a matrix."""
    mat = x.detach().cpu()
    return [[[float(v.real.item()), float(v.imag.item())] for v in row] for row in mat]


def _real_vector(x: torch.Tensor):
    """Return a JSON-safe list of real tensor entries."""
    return [float(v.item()) for v in x.detach().cpu().reshape(-1)]


def _real_matrix(x: torch.Tensor):
    """Return a JSON-safe nested list of real matrix entries."""
    mat = x.detach().cpu()
    return [[float(v.item()) for v in row] for row in mat]


def pair_mode_diagnostic_from_centered_C(
    C_vu: torch.Tensor,
    source_a: torch.Tensor,
    target_a: torch.Tensor,
    source_backend_a: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Break the centered numerator and denominator into Kraus-mode contributions.

    Inputs:
      C_vu: (n_v, n_u) complex centered two-point matrix.
      source_a: (n_u,) complex local/source one-point amplitudes.
      target_a: (n_v,) complex target one-point amplitudes.
      source_backend_a: optional (n_u,) complex source one-point amplitudes
        measured after source propagation by the same approximate backend.
    Output:
      JSON-safe dict with per-mode denominator, cross, quadratic, and numerator
      contributions.  This is diagnostic-only and does not change formulas.
    Physical convention:
      Numerator entries use
      N_vu = 2 * Re(conj(target_a[v] * source_a[u]) * C_vu[v,u])
             + abs(C_vu[v,u])**2.
    """
    C = C_vu.detach()
    source = source_a.to(device=C.device, dtype=C.dtype).detach()
    target = target_a.to(device=C.device, dtype=C.dtype).detach()
    source_den = torch.abs(source) ** 2
    target_den = torch.abs(target) ** 2
    cross = 2.0 * (target.conj().unsqueeze(1) * source.conj().unsqueeze(0) * C).real
    quad = torch.abs(C) ** 2
    numerator = cross + quad
    out: Dict[str, Any] = {
        "source_one_point_local": _complex_vector_as_pairs(source),
        "target_one_point_local": _complex_vector_as_pairs(target),
        "source_denominator_by_mode_local": _real_vector(source_den.real),
        "target_denominator_by_mode_local": _real_vector(target_den.real),
        "F_a_local": float(torch.sum(source_den).real.detach().cpu().item()),
        "F_b_local": float(torch.sum(target_den).real.detach().cpu().item()),
        "denominator_local": float(
            (torch.sum(source_den).real * torch.sum(target_den).real)
            .detach()
            .cpu()
            .item()
        ),
        "C_vu": _complex_matrix_as_pairs(C),
        "cross_by_uv_local": _real_matrix(cross),
        "quad_by_uv": _real_matrix(quad.real),
        "numerator_by_uv_local": _real_matrix(numerator),
        "numerator_by_source_mode_local": _real_vector(torch.sum(numerator, dim=0)),
        "numerator_by_target_mode_local": _real_vector(torch.sum(numerator, dim=1)),
        "numerator_local": float(torch.sum(numerator).detach().cpu().item()),
    }
    if source_backend_a is not None:
        source_backend = source_backend_a.to(device=C.device, dtype=C.dtype).detach()
        source_residual = source_backend - source
        source_backend_den = torch.abs(source_backend) ** 2
        source_residual_den = torch.abs(source_residual) ** 2
        cross_backend = (
            2.0
            * (target.conj().unsqueeze(1) * source_backend.conj().unsqueeze(0) * C).real
        )
        numerator_backend = cross_backend + quad
        out.update(
            {
                "source_one_point_backend": _complex_vector_as_pairs(source_backend),
                "source_centered_residual_backend": _complex_vector_as_pairs(
                    source_residual
                ),
                "source_centered_residual_abs_by_mode": _real_vector(
                    torch.abs(source_residual)
                ),
                "source_centered_residual_norm2": float(
                    torch.sum(source_residual_den).real.detach().cpu().item()
                ),
                "source_denominator_by_mode_backend": _real_vector(
                    source_backend_den.real
                ),
                "F_a_backend": float(
                    torch.sum(source_backend_den).real.detach().cpu().item()
                ),
                "denominator_backend_source": float(
                    (torch.sum(source_backend_den).real * torch.sum(target_den).real)
                    .detach()
                    .cpu()
                    .item()
                ),
                "cross_by_uv_backend_source": _real_matrix(cross_backend),
                "numerator_by_uv_backend_source": _real_matrix(numerator_backend),
                "numerator_by_source_mode_backend_source": _real_vector(
                    torch.sum(numerator_backend, dim=0)
                ),
                "numerator_by_target_mode_backend_source": _real_vector(
                    torch.sum(numerator_backend, dim=1)
                ),
                "numerator_backend_source": float(
                    torch.sum(numerator_backend).detach().cpu().item()
                ),
            }
        )
    return out


class PairAccumulator:
    """
    Accumulate amplitude vectors for selected (source gate, target gate) pairs.

    Physics:
      This object does not define the physical amplitude representation and does
      not contract tensor networks. It only assembles already-read mode
      vectors into a (n_v, n_u) matrix and
      computes Delta_ab when all source modes u have arrived.

    Convention:
      pair_formula="raw_A" (default):
        A_vu[b_mode, a_mode] has shape (n_v, n_u), built from bare dressed
        Kraus operators, and Delta_ab follows fidelity_formula.delta_from_raw_A.
      pair_formula="centered_C":
        C_vu[b_mode, a_mode] has shape (n_v, n_u), and Delta_ab follows
        fidelity_formula.delta_from_centered_C.  This is now an explicit
        diagnostic convention, not the default physical standard.

    Shapes:
      amplitude_v: (n_v,) complex tensor, one per target Kraus mode.
      A_vu or C_vu: (n_v, n_u) complex tensor, assembled from source-mode rows.

    The formula mode is a physical convention and must match
    docs/physics_contract.md.
    """

    def __init__(
        self,
        ctx: FidelityContext,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
        store_pair_C: bool = True,
        pair_C_to_cpu: bool = False,
        pair_formula: str = "raw_A",
        correction_fn: Optional[callable] = None,
    ):
        """
        Args:
            ctx: FidelityContext providing gate_map for layer lookups.
            dtype: torch dtype for Delta computation (default torch.float64).
            device: torch device for Delta computation tensors.
            store_pair_C: whether finalized amplitude matrices are retained in
                the historical `pair_C` field.  Delta and self-energy accumulation do not require
                retaining them after a pair is finalized.
            pair_C_to_cpu: if storing `pair_C`, detach and move finalized
                matrices to CPU immediately.  This changes result storage only,
                not Delta computation.
            pair_formula: "raw_A" for the bare dressed-Kraus formula
                Delta_ab = M_ab/(F_a F_b)-1, or "centered_C" for the explicit
                centered diagnostic convention.
            correction_fn: optional callable(a_idx, b_idx, amplitude) -> corrected.
                Called after assembling the full amplitude matrix from all
                source modes but before Delta computation. The tensor has
                shape (n_v, n_u).  The
                callable must accept and return complex torch tensors.  When
                None (default), no correction is applied.  This is the
                intended hook for loop-calculus corrections.
        """
        if pair_formula not in {"centered_C", "raw_A"}:
            raise ValueError("pair_formula must be 'centered_C' or 'raw_A'.")
        self.ctx = ctx
        self._dtype = dtype
        self._device = device if device is not None else ctx.device
        self.store_pair_C = store_pair_C
        self.pair_C_to_cpu = pair_C_to_cpu
        self.pair_formula = pair_formula

        # Amplitude-vector storage: key (a_idx, b_idx) -> {u_idx: vector}
        self._C_accum: Dict[tuple, Dict[int, torch.Tensor]] = {}
        self._source_centered_identity_accum: Dict[tuple, Dict[int, torch.Tensor]] = {}

        # Finalized results
        self.pair_C: Dict[tuple, torch.Tensor] = {}
        self.pair_deltas: Dict[tuple, float] = {}
        self.self_consistent_pair_diagnostics: Dict[tuple, Dict[str, Any]] = {}
        self.pair_mode_diagnostics: Dict[tuple, Dict[str, Any]] = {}

        # Prevent re-finalization
        self._finalized_pairs: set = set()

        # Optional correction callback
        self._correction_fn = correction_fn

        # Self-energy per layer
        self.self_energy_by_target_layer: Dict[int, float] = {}
        self.self_energy_by_source_layer: Dict[int, float] = {}

        # Require external metadata for Delta computation
        self.local_a: Dict[int, torch.Tensor] = {}  # gate_idx -> a_u (n_u,)
        self.local_F: Dict[int, float] = {}  # gate_idx -> F_g

    # ------------------------------------------------------------------
    # External metadata registration
    # ------------------------------------------------------------------
    def register_gate_metadata(self, gate_idx: int, a_u: torch.Tensor, F_g: float):
        """
        Register one-point amplitudes and local fidelity for a gate.

        Args:
            gate_idx: gate index.
            a_u: (n_u,) complex one-point amplitudes a_{gate,u}.
            F_g: scalar F_g = sum_u |a_u|^2.
        """
        self.local_a[gate_idx] = a_u
        self.local_F[gate_idx] = F_g

    def set_correction_fn(self, fn: Optional[callable]) -> None:
        """Register or clear the amplitude correction callback.

        The callback must have signature ``(a_idx, b_idx, amplitude) -> corrected``
        where ``amplitude`` is a ``(n_v, n_u)`` complex tensor containing
        `A_vu` for raw_A mode or `C_vu` for centered_C mode.  When ``fn`` is None,
        no correction is applied.  This changes the formula used for all
        *future* pair finalizations; already-finalized pairs are unaffected.
        """
        self._correction_fn = fn

    # ------------------------------------------------------------------
    # Pair accumulation
    # ------------------------------------------------------------------
    def add_source_mode_C(
        self,
        a_idx: int,
        b_idx: int,
        source_mode_u: int,
        C_v: torch.Tensor,
        source_centered_identity: Optional[torch.Tensor] = None,
    ):
        """
        Add one amplitude vector for source gate a, target gate b, source mode u.

        If all n_u source modes for this pair have been accumulated,
        Delta_ab is computed automatically and stored in pair_deltas.

        Args:
            a_idx: source gate index.
            b_idx: target gate index.
            source_mode_u: source Kraus mode index (0 <= u < n_u).
            C_v: historical parameter name; (n_v,) complex vector. It is A_vu[:,u] in raw_A mode or C_vu[:,u] in centered_C mode.
        """
        key = (a_idx, b_idx)
        if key not in self._C_accum:
            self._C_accum[key] = {}
        self._C_accum[key][source_mode_u] = C_v
        if source_centered_identity is not None:
            if key not in self._source_centered_identity_accum:
                self._source_centered_identity_accum[key] = {}
            self._source_centered_identity_accum[key][source_mode_u] = (
                source_centered_identity
            )
        self._try_finalize(a_idx, b_idx)

    # ------------------------------------------------------------------
    # Read-only access
    # ------------------------------------------------------------------
    def has_pair(self, a_idx: int, b_idx: int) -> bool:
        """Check if a pair has been finalized (Delta computed)."""
        return (a_idx, b_idx) in self._finalized_pairs

    def get_delta(self, a_idx: int, b_idx: int) -> Optional[float]:
        """Get Delta_ab for a finalized pair, or None if not yet computed."""
        return self.pair_deltas.get((a_idx, b_idx))

    def get_C_vu(self, a_idx: int, b_idx: int) -> Optional[torch.Tensor]:
        """Get the finalized pair amplitude matrix, or None.

        Historical method name: returns `A_vu` in raw_A mode and `C_vu`
        in centered_C mode.
        """
        return self.pair_C.get((a_idx, b_idx))

    def total_delta_sum(self) -> float:
        """Sum of all finalized Deltas."""
        return sum(self.pair_deltas.values())

    def incomplete_pair_diagnostics(self, max_entries: Optional[int] = None) -> Dict:
        """
        Report pairs with partial amplitude data that were not finalized.

        Shapes:
          Each accumulated vector has shape (n_v,) complex. A pair is complete
          only after every expected source mode u has arrived, so the amplitude
          matrix can be assembled with shape (n_v, n_u).

        Returns:
          dict with:
            count: number of incomplete pair entries.
            pairs: list of diagnostic records.  Each record contains pair,
              expected_source_modes, received_source_modes, missing_source_modes,
              and missing metadata flags.

        This is diagnostic-only.  It does not change Delta_ab, pair_C, or
        total_delta_sum.
        """
        entries = []
        for key, by_mode in sorted(self._C_accum.items()):
            if key in self._finalized_pairs:
                continue

            a_idx, b_idx = key
            source_metadata_missing = a_idx not in self.local_a
            target_metadata_missing = b_idx not in self.local_a
            source_fidelity_missing = a_idx not in self.local_F
            target_fidelity_missing = b_idx not in self.local_F
            received_modes = sorted(int(u) for u in by_mode.keys())

            if source_metadata_missing:
                expected_modes = None
                missing_modes = None
            else:
                expected_modes = int(self.local_a[a_idx].shape[0])
                received_set = set(received_modes)
                missing_modes = [
                    u for u in range(expected_modes) if u not in received_set
                ]

            entries.append(
                {
                    "pair": key,
                    "expected_source_modes": expected_modes,
                    "received_source_modes": received_modes,
                    "received_count": len(received_modes),
                    "missing_source_modes": missing_modes,
                    "source_metadata_missing": source_metadata_missing,
                    "target_metadata_missing": target_metadata_missing,
                    "source_fidelity_missing": source_fidelity_missing,
                    "target_fidelity_missing": target_fidelity_missing,
                }
            )

        count = len(entries)
        if max_entries is not None:
            entries = entries[:max_entries]
        return {"count": count, "pairs": entries}

    # ------------------------------------------------------------------
    # Internal: finalize pair and record self-energy
    # ------------------------------------------------------------------
    def _try_finalize(self, a_idx: int, b_idx: int):
        """If all u modes for this pair arrived, compute Delta."""
        key = (a_idx, b_idx)
        if key in self._finalized_pairs:
            return
        if key not in self._C_accum:
            return
        if a_idx not in self.local_a:
            return  # metadata not registered yet
        if b_idx not in self.local_a:
            return  # metadata not registered yet
        if a_idx not in self.local_F or b_idx not in self.local_F:
            return  # fidelity metadata not registered yet

        n_u = self.local_a[a_idx].shape[0]
        n_v = self.local_a[b_idx].shape[0]
        if len(self._C_accum[key]) < n_u:
            return

        # Assemble selected amplitude matrix: stack source modes in u order.
        C_vu_list = [self._C_accum[key][u] for u in range(n_u)]
        C_vu = torch.stack(C_vu_list, dim=1)  # (n_v, n_u)

        # Optional correction callback (e.g., loop-calculus correction)
        if self._correction_fn is not None:
            C_vu = self._correction_fn(a_idx, b_idx, C_vu)

        F_a_tensor = torch.tensor(
            self.local_F[a_idx], dtype=self._dtype, device=self._device
        )
        F_b_tensor = torch.tensor(
            self.local_F[b_idx], dtype=self._dtype, device=self._device
        )
        F_a_for_delta = F_a_tensor.to(device=C_vu.device)
        F_b_for_delta = F_b_tensor.to(device=C_vu.device)
        local_source_a = self.local_a[a_idx].to(device=C_vu.device, dtype=C_vu.dtype)
        local_target_a = self.local_a[b_idx].to(device=C_vu.device, dtype=C_vu.dtype)
        if self.pair_formula == "raw_A":
            delta = delta_from_raw_A(C_vu, F_a_for_delta, F_b_for_delta)
        else:
            delta = delta_from_centered_C(
                C_vu,
                local_source_a,
                local_target_a,
                F_a_for_delta,
                F_b_for_delta,
            )
        record_pair_modes = os.environ.get("CENTERED_PAIR_MODE_DIAGNOSTICS", "0") == "1"
        source_backend_a = None
        residual_by_mode = self._source_centered_identity_accum.get(key)
        if (
            self.pair_formula == "centered_C"
            and residual_by_mode is not None
            and all(u in residual_by_mode for u in range(n_u))
        ):
            source_centered_identity = torch.stack(
                [residual_by_mode[u].reshape(()) for u in range(n_u)]
            ).to(device=C_vu.device, dtype=C_vu.dtype)
            source_backend_a = local_source_a + source_centered_identity
            source_backend_F = torch.sum(torch.abs(source_backend_a) ** 2).real
            target_F = torch.tensor(
                self.local_F[b_idx], dtype=self._dtype, device=C_vu.device
            )
            term1 = torch.sum(
                local_target_a.conj().unsqueeze(1)
                * source_backend_a.conj().unsqueeze(0)
                * C_vu
            )
            term2 = torch.sum(torch.abs(C_vu) ** 2)
            numerator_backend_source = (2 * term1.real + term2).real
            denominator_backend_source = source_backend_F * target_F
            delta_backend_source = numerator_backend_source / denominator_backend_source
            numerator_local_source = delta * torch.tensor(
                self.local_F[a_idx] * self.local_F[b_idx],
                dtype=self._dtype,
                device=C_vu.device,
            )
            self.self_consistent_pair_diagnostics[key] = {
                "source_centered_identity": _complex_vector_as_pairs(
                    source_centered_identity
                ),
                "source_one_point_local": _complex_vector_as_pairs(local_source_a),
                "source_one_point_backend": _complex_vector_as_pairs(source_backend_a),
                "source_F_local": float(self.local_F[a_idx]),
                "source_F_backend": float(source_backend_F.detach().cpu().item()),
                "target_F_local": float(self.local_F[b_idx]),
                "numerator_local_source": float(
                    numerator_local_source.detach().cpu().item()
                ),
                "numerator_backend_source": float(
                    numerator_backend_source.detach().cpu().item()
                ),
                "delta_backend_source": float(
                    delta_backend_source.detach().cpu().item()
                ),
                "max_abs_source_centered_identity": float(
                    torch.max(torch.abs(source_centered_identity)).detach().cpu().item()
                ),
            }
        if record_pair_modes and self.pair_formula == "centered_C":
            self.pair_mode_diagnostics[key] = pair_mode_diagnostic_from_centered_C(
                C_vu,
                local_source_a,
                local_target_a,
                source_backend_a=source_backend_a,
            )
        elif record_pair_modes and self.pair_formula == "raw_A":
            raw_M = torch.sum(torch.abs(C_vu) ** 2).real
            denominator = F_a_tensor.to(device=C_vu.device) * F_b_tensor.to(
                device=C_vu.device
            )
            self.pair_mode_diagnostics[key] = {
                "formula": "raw_A",
                "A_vu": _complex_matrix_as_pairs(C_vu),
                "M_ab_raw": float(raw_M.detach().cpu().item()),
                "denominator_local": float(denominator.detach().cpu().item()),
                "delta_raw": float(delta.detach().cpu().item()),
            }
        if self.store_pair_C:
            self.pair_C[key] = C_vu.detach().cpu() if self.pair_C_to_cpu else C_vu
        self.pair_deltas[key] = delta.item()
        self._finalized_pairs.add(key)
        del self._C_accum[key]
        if key in self._source_centered_identity_accum:
            del self._source_centered_identity_accum[key]

        # Record self-energy per layer
        self._record_delta_by_layer(a_idx, b_idx, delta.item())

    def _record_delta_by_layer(self, a_idx: int, b_idx: int, delta_val: float):
        """Add delta_val to the appropriate per-layer self-energy sums."""
        if hasattr(self.ctx, "gate_map"):
            gm = self.ctx.gate_map
            if b_idx in gm:
                b_layer = gm[b_idx][0]
                self.self_energy_by_target_layer[b_layer] = (
                    self.self_energy_by_target_layer.get(b_layer, 0.0) + delta_val
                )
            if a_idx in gm:
                a_layer = gm[a_idx][0]
                self.self_energy_by_source_layer[a_layer] = (
                    self.self_energy_by_source_layer.get(a_layer, 0.0) + delta_val
                )
