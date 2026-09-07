"""
kraus.py: Dressed Kraus and Centered operators.
Implements E = K U_dag, a_u = <E_u>, E_u_center = E_u - a_u * I
"""

import torch


def compute_dressed_kraus(
    kraus_ops: torch.Tensor, ideal_unitary: torch.Tensor
) -> torch.Tensor:
    """
    Computes E_u = K_u U_dag.

    Args:
        kraus_ops: shape (n_kraus, d_out, d_in) or for 2Q (n_kraus, d1_out, d2_out, d1_in, d2_in)
                   Generally (n_kraus, *dims, *dims)
        ideal_unitary: shape (*dims, *dims)

    Returns:
        dressed_kraus: shape (n_kraus, *dims, *dims)
    """
    n_kraus = kraus_ops.shape[0]
    n_qubits = (ideal_unitary.ndim) // 2

    # Flatten physical dimensions for matrix multiplication
    dim = 1
    for i in range(n_qubits):
        dim *= ideal_unitary.shape[i]

    K_mat = kraus_ops.reshape(n_kraus, dim, dim)
    U_mat = ideal_unitary.reshape(dim, dim)
    U_dag = U_mat.conj().transpose(-1, -2)

    E_mat = torch.matmul(K_mat, U_dag.unsqueeze(0))
    return E_mat.reshape(kraus_ops.shape)


def compute_one_point_amplitude(
    E_u: torch.Tensor, local_rho: torch.Tensor
) -> torch.Tensor:
    """
    Computes a_u = <E_u> = Tr(E_u * rho).
    Since rho is formed from the ideal state |psi><psi|, this is exactly <psi | E_u | psi>.

    Args:
        E_u: shape (n_kraus, *dims, *dims)
        local_rho: shape (*dims, *dims)

    Returns:
        a_u: shape (n_kraus,)
    """
    n_kraus = E_u.shape[0]
    n_qubits = (local_rho.ndim) // 2
    dim = 1
    for i in range(n_qubits):
        dim *= local_rho.shape[i]

    E_mat = E_u.reshape(n_kraus, dim, dim)
    rho_mat = local_rho.reshape(dim, dim)

    # a_u = Tr(E_mat @ rho_mat.T) but rho is density matrix, Tr(AB) = sum(A * B.T). Wait, Tr(E*rho) = sum_ij E_ij rho_ji.
    # Actually, rho is formed as rho_ij = <i|rho|j>. E is E_ij = <i|E|j>.
    # Tr(E * rho) = sum_ij E_ij rho_ji = einsum('kab, ba -> k', E_mat, rho_mat)
    # However, standard convention: if rho = |psi><psi|, rho_ab = psi_a psi^*_b.
    # E |psi> = sum_a (sum_b E_ab psi_b) |a>. <psi|E|psi> = sum_ab psi^*_a E_ab psi_b = sum_ab E_ab rho_ba.
    a_u = torch.einsum("kab, ba -> k", E_mat, rho_mat)
    return a_u


def compute_centered_operator(E_u: torch.Tensor, a_u: torch.Tensor) -> torch.Tensor:
    """
    Computes E_u_center = E_u - a_u * I.
    NOTE: This is an algebraic subtraction proportional to the identity operator,
    NOT a Hilbert-Schmidt trace subtraction.

    Args:
        E_u: shape (n_kraus, *dims, *dims)
        a_u: shape (n_kraus,)

    Returns:
        E_u_center: shape (n_kraus, *dims, *dims)
    """
    n_kraus = E_u.shape[0]
    n_qubits = (E_u.ndim - 1) // 2
    dim = 1
    for i in range(n_qubits):
        dim *= E_u.shape[i + 1]

    E_mat = E_u.reshape(n_kraus, dim, dim)
    I_mat = torch.eye(dim, dtype=E_u.dtype, device=E_u.device).unsqueeze(0)

    E_center_mat = E_mat - a_u.view(n_kraus, 1, 1) * I_mat
    return E_center_mat.reshape(E_u.shape)
