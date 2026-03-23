from __future__ import annotations

import jax.numpy as jnp


def symmetrize(matrix: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (matrix + jnp.swapaxes(matrix, -1, -2))


def generalized_lowest_eigenvalue(
    hamiltonian: jnp.ndarray,
    overlap: jnp.ndarray,
    *,
    jitter: float = 1.0e-8,
) -> jnp.ndarray:
    hamiltonian = symmetrize(hamiltonian)
    overlap = symmetrize(overlap)
    identity = jnp.eye(overlap.shape[0], dtype=overlap.dtype)
    chol = jnp.linalg.cholesky(overlap + jitter * identity)
    left_solved = jnp.linalg.solve(chol, hamiltonian)
    transformed = jnp.swapaxes(
        jnp.linalg.solve(chol, jnp.swapaxes(left_solved, -1, -2)),
        -1,
        -2,
    )
    transformed = symmetrize(transformed)
    eigenvalues = jnp.linalg.eigvalsh(transformed)
    return eigenvalues[0]


def total_energy_from_parts(
    exact_reference_energy: jnp.ndarray,
    reference_energy_residual: jnp.ndarray,
    structure_hamiltonian: jnp.ndarray,
    overlap_matrix: jnp.ndarray,
    nuclear_repulsion_energy: jnp.ndarray,
) -> jnp.ndarray:
    return (
        exact_reference_energy
        + reference_energy_residual
        + generalized_lowest_eigenvalue(structure_hamiltonian, overlap_matrix)
        + nuclear_repulsion_energy
    )
