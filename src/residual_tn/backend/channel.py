"""
channel.py: Dense quantum-channel preparation utilities.

This module is an input-preparation layer for the fidelity backend.  It turns
Hamiltonian/Lindblad or superoperator descriptions into Kraus operators that
can be placed in `FidelityContext`.

It does NOT compute centered correlations, Delta_ab, PEPS contractions, or
fidelity.  No physical convention in docs/physics_contract.md is changed.
"""

from typing import Iterable, List, Optional, Sequence
import torch


def kraus_to_superoperator(kraus_ops: torch.Tensor) -> torch.Tensor:
    """
    Build the row-major Liouville superoperator for a Kraus channel.

    Formula:
      rho_out[o,p] = sum_{i,j} S[o,p,i,j] * rho_in[i,j]
      S[o,p,i,j] = sum_u K_u[o,i] * conj(K_u[p,j])

    Args:
        kraus_ops: (n_kraus, d, d) complex tensor.

    Returns:
        superoperator: (d*d, d*d) complex tensor in row-major matrix
            vectorization, compatible with `superoperator_to_kraus`.

    Exact dense algebra.  No fidelity convention changes.
    """
    if kraus_ops.ndim != 3:
        raise ValueError("kraus_ops must have shape (n_kraus, d, d).")
    if kraus_ops.shape[-1] != kraus_ops.shape[-2]:
        raise ValueError("Kraus operators must be square.")
    return torch.einsum("uoi,upj->opij", kraus_ops, kraus_ops.conj()).reshape(
        kraus_ops.shape[-1] ** 2, kraus_ops.shape[-1] ** 2
    )


def superoperator_to_kraus(
    superoperator: torch.Tensor,
    d: Optional[int] = None,
    tol: float = 1e-10,
) -> torch.Tensor:
    """
    Decompose a dense channel superoperator into Kraus operators.

    Convention:
      The input superoperator uses row-major matrix vectorization:
        vec(rho_out)[o,p] = S[o,p,i,j] * vec(rho_in)[i,j].
      The Choi matrix is formed as:
        Choi[o,i,p,j] = S[o,p,i,j].

    Args:
        superoperator: (d*d, d*d) complex tensor.
        d: Hilbert-space dimension.  If omitted, inferred from shape.
        tol: discard Choi eigenmodes with eigenvalue <= tol.

    Returns:
        kraus_ops: (n_kraus, d, d) complex tensor.

    Approximation:
        Eigenmodes below `tol` are discarded.  This is a numerical rank
        cutoff only; it does not change the centered-fidelity definition.
    """
    if superoperator.ndim != 2 or superoperator.shape[0] != superoperator.shape[1]:
        raise ValueError("superoperator must be a square matrix.")
    if d is None:
        root = int(round(superoperator.shape[0] ** 0.5))
        if root * root != superoperator.shape[0]:
            raise ValueError("Cannot infer Hilbert dimension from superoperator shape.")
        d = root
    if superoperator.shape != (d * d, d * d):
        raise ValueError(
            f"Expected superoperator shape {(d * d, d * d)}, got {tuple(superoperator.shape)}."
        )

    S4 = superoperator.reshape(d, d, d, d)
    choi = S4.permute(0, 2, 1, 3).reshape(d * d, d * d)
    choi = 0.5 * (choi + choi.conj().T)
    eigvals, eigvecs = torch.linalg.eigh(choi)

    kraus: List[torch.Tensor] = []
    for idx in range(eigvals.numel()):
        val = eigvals[idx].real
        if val > tol:
            kraus.append(
                torch.sqrt(val).to(superoperator.dtype) * eigvecs[:, idx].reshape(d, d)
            )

    if not kraus:
        return torch.zeros(
            (0, d, d), dtype=superoperator.dtype, device=superoperator.device
        )
    return torch.stack(kraus, dim=0)


def lindblad_liouvillian(
    hamiltonian: torch.Tensor,
    collapse_ops: Optional[Sequence[torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Build a dense Lindblad generator in row-major Liouville convention.

    Formula on a density matrix rho:
      L(rho) = -i * (H rho - rho H)
             + sum_c [ C rho C^dag
                       - 0.5 * (C^dag C rho + rho C^dag C) ]

    Args:
        hamiltonian: (d,d) Hermitian complex tensor.
        collapse_ops: sequence of (d,d) complex collapse operators.

    Returns:
        liouvillian: (d*d, d*d) complex tensor.  Column `i*d+j` is
            `L(|i><j|)` flattened in row-major order.

    Exact dense algebra.  No fidelity convention changes.
    """
    if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("hamiltonian must have shape (d, d).")
    d = hamiltonian.shape[0]
    device = hamiltonian.device
    dtype = hamiltonian.dtype
    cops = list(collapse_ops or [])
    for c in cops:
        if c.shape != (d, d):
            raise ValueError(
                f"collapse operator has shape {tuple(c.shape)}, expected {(d, d)}."
            )

    columns = []
    for i in range(d):
        for j in range(d):
            rho = torch.zeros((d, d), dtype=dtype, device=device)
            rho[i, j] = 1.0
            out = -1j * (hamiltonian @ rho - rho @ hamiltonian)
            for c in cops:
                cdc = c.conj().T @ c
                out = out + c @ rho @ c.conj().T - 0.5 * (cdc @ rho + rho @ cdc)
            columns.append(out.reshape(d * d))
    return torch.stack(columns, dim=1)


def superoperator_from_lindblad(
    hamiltonian: torch.Tensor,
    duration: float,
    collapse_ops: Optional[Sequence[torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Evolve a time-independent Lindblad generator to a superoperator.

    Args:
        hamiltonian: (d,d) complex Hamiltonian.
        duration: scalar evolution time in the same units as H and collapse
            rates.
        collapse_ops: sequence of (d,d) collapse operators.

    Returns:
        superoperator: (d*d, d*d) complex tensor equal to exp(duration * L).

    Approximation:
        Uses dense `torch.matrix_exp`; no time discretization is introduced.
    """
    L = lindblad_liouvillian(hamiltonian, collapse_ops)
    return torch.matrix_exp(
        torch.as_tensor(duration, dtype=L.real.dtype, device=L.device) * L
    )


def kraus_from_lindblad(
    hamiltonian: torch.Tensor,
    duration: float,
    collapse_ops: Optional[Sequence[torch.Tensor]] = None,
    tol: float = 1e-10,
) -> torch.Tensor:
    """
    Compute Kraus operators from a time-independent Lindblad model.

    Args:
        hamiltonian: (d,d) complex Hamiltonian.
        duration: evolution time.
        collapse_ops: sequence of (d,d) collapse operators.
        tol: Choi eigenvalue cutoff for Kraus extraction.

    Returns:
        kraus_ops: (n_kraus, d, d) complex tensor.

    This is a preparation utility; centered fidelity formulas are not applied.
    """
    S = superoperator_from_lindblad(hamiltonian, duration, collapse_ops)
    return superoperator_to_kraus(S, d=hamiltonian.shape[0], tol=tol)


def project_kraus_to_subspace(
    kraus_ops: torch.Tensor,
    subspace: Sequence[int],
) -> torch.Tensor:
    """
    Project dense Kraus operators onto a computational subspace.

    Args:
        kraus_ops: (n_kraus, D, D) complex tensor.
        subspace: sequence of basis indices to keep, e.g. [0, 1].

    Returns:
        projected_kraus: (n_kraus, d, d), where d=len(subspace).

    Approximation:
        Leakage outside the chosen subspace is discarded.  The projected
        channel may be trace-decreasing; this is a physical modeling choice
        made before the fidelity backend.
    """
    if kraus_ops.ndim != 3:
        raise ValueError("kraus_ops must have shape (n_kraus, D, D).")
    idx = torch.as_tensor(subspace, dtype=torch.long, device=kraus_ops.device)
    return kraus_ops.index_select(1, idx).index_select(2, idx)
