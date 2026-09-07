"""
fidelity_formula.py: Pure fidelity formulas, no PEPS/BP/state-vector dependency.

This module contains only the algebraic formulas for local fidelity and
connected fidelity correction.  Every function operates on torch tensors
with documented shapes.

Formulas implemented:
  F_g       = sum_u |a_u|^2
  A_vu[v,u] = bare / uncentered two-Kraus amplitude
  C_vu[v,u] = A_vu[v,u] - a_b[v] * a_a[u]
  Delta_raw = sum_{u,v} |A_vu[v,u]|^2 / (F_a F_b) - 1
  Delta_cent =
    (2 Re sum_{u,v} conj(a_b[v] * a_a[u]) * C_vu[v,u]
     + sum_{u,v} |C_vu[v,u]|^2) / (F_a F_b)
  log_F1    = sum_g log(F_g)
  log_Fc    = log_F1 + sum Delta_ab
"""

import torch


def local_fidelity_from_amplitudes(a_u: torch.Tensor) -> torch.Tensor:
    """
    Compute gate-level local fidelity F_g = sum_u |a_u|^2.

    Args:
        a_u: (n_u,) complex one-point amplitudes.

    Returns:
        F_g: scalar tensor (real, float64).
    """
    return torch.sum(torch.abs(a_u) ** 2).real


def accumulate_log_F1(local_fidelities, dtype=torch.float64) -> torch.Tensor:
    """
    Compute log_F1 = sum_g log(F_g).

    Args:
        local_fidelities: dict gate_idx -> float F_g, or iterable of scalar tensors.
        dtype: torch dtype for accumulation.

    Returns:
        log_F1: scalar tensor.
    """
    if isinstance(local_fidelities, dict):
        vals = list(local_fidelities.values())
    else:
        vals = list(local_fidelities)
    log_sum = torch.tensor(0.0, dtype=dtype)
    for v in vals:
        F = torch.as_tensor(v, dtype=dtype)
        log_sum += torch.log(F)
    return log_sum


def delta_from_centered_C(
    C_vu: torch.Tensor,
    a_a: torch.Tensor,
    a_b: torch.Tensor,
    F_a: torch.Tensor,
    F_b: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Delta_ab from centered C_vu and one-point amplitudes.

    Delta_ab =
        (2 * Re sum_{u,v} conj(a_b,v * a_a,u) * C_vu
         + sum_{u,v} |C_vu|^2)
        / (F_a * F_b)

    This is the exact centered pair correction (no -1 term).

    Args:
        C_vu: (n_v, n_u) complex amplitude matrix.
        a_a:  (n_u,) source one-point amplitudes a_{a,u}.
        a_b:  (n_v,) target one-point amplitudes a_{b,v}.
        F_a:  scalar F_a = sum_u |a_a[u]|^2.
        F_b:  scalar F_b = sum_v |a_b[v]|^2.

    Returns:
        delta: scalar Delta_ab (real, float).
    """
    # term1 = 2 * Re sum_{u,v} (a_b[v] * a_a[u])^* * C_vu
    # = 2 * Re sum_{u,v} conj(a_b[v]) * conj(a_a[u]) * C_vu
    term1 = torch.sum(a_b.conj().unsqueeze(1) * a_a.conj().unsqueeze(0) * C_vu)
    term1 = 2 * term1.real
    # term2 = sum_{u,v} |C_vu|^2
    term2 = torch.sum(torch.abs(C_vu) ** 2)
    delta = (term1 + term2) / (F_a * F_b)
    return delta.real


def delta_from_raw_A(
    A_vu: torch.Tensor,
    F_a: torch.Tensor,
    F_b: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Delta_ab from uncentered two-Kraus amplitudes.

    Original formula:
      M_ab = sum_{u,v} |A_vu|^2
      Delta_ab = M_ab / (F_a * F_b) - 1

    Inputs:
      A_vu: (n_v, n_u) complex raw amplitude matrix.
      F_a: scalar source one-location fidelity sum_u |a_u|^2.
      F_b: scalar target one-location fidelity sum_v |b_v|^2.
    Output:
      delta: scalar real Delta_ab.
    Physical convention:
      This is algebraically equivalent to delta_from_centered_C only when
      A_vu = a_b[v] * a_a[u] + C_vu[v,u] is evaluated in one consistent global
      normalization.  It does not use one-point amplitudes explicitly.
    """
    M_ab = torch.sum(torch.abs(A_vu) ** 2).real
    return (M_ab / (F_a * F_b) - 1.0).real
