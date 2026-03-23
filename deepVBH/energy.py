from __future__ import annotations

import torch


def symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def generalized_lowest_eigenvalue(
    hamiltonian: torch.Tensor,
    overlap: torch.Tensor,
    *,
    jitter: float = 1.0e-8,
    max_attempts: int = 6,
) -> torch.Tensor:
    if hamiltonian.ndim != 2 or overlap.ndim != 2:
        raise ValueError("hamiltonian and overlap must be rank-2 tensors")
    if hamiltonian.shape != overlap.shape:
        raise ValueError("hamiltonian and overlap must have matching shapes")

    hamiltonian = symmetrize(hamiltonian)
    overlap = symmetrize(overlap)
    identity = torch.eye(overlap.shape[0], dtype=overlap.dtype, device=overlap.device)

    scale = jitter
    last_error: RuntimeError | None = None
    for _ in range(max_attempts):
        try:
            chol = torch.linalg.cholesky(overlap + scale * identity)
            break
        except RuntimeError as error:
            last_error = error
            scale *= 10.0
    else:
        assert last_error is not None
        raise last_error

    left_solved = torch.linalg.solve_triangular(
        chol,
        hamiltonian,
        upper=False,
    )
    transformed = torch.linalg.solve_triangular(
        chol,
        left_solved.transpose(-1, -2),
        upper=False,
    ).transpose(-1, -2)
    transformed = symmetrize(transformed)
    eigenvalues = torch.linalg.eigvalsh(transformed)
    return eigenvalues[0]


def total_energy_from_parts(
    exact_reference_energy: torch.Tensor,
    reference_energy_residual: torch.Tensor,
    structure_hamiltonian: torch.Tensor,
    overlap_matrix: torch.Tensor,
    nuclear_repulsion_energy: torch.Tensor,
) -> torch.Tensor:
    return (
        exact_reference_energy
        + reference_energy_residual
        + generalized_lowest_eigenvalue(structure_hamiltonian, overlap_matrix)
        + nuclear_repulsion_energy
    )
