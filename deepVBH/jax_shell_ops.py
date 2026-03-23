from __future__ import annotations

import itertools
import math
from functools import lru_cache
from typing import Any

import e3nn_jax as e3nn
import jax.numpy as jnp
import numpy as np


def cartesian_shell_dimension(angular_momentum: int) -> int:
    if angular_momentum < 0:
        raise ValueError("angular_momentum must be non-negative")
    return (angular_momentum + 1) * (angular_momentum + 2) // 2


def canonical_cartesian_exponents(angular_momentum: int) -> list[tuple[int, int, int]]:
    if angular_momentum < 0:
        raise ValueError("angular_momentum must be non-negative")
    exponents: list[tuple[int, int, int]] = []
    for lx in range(angular_momentum, -1, -1):
        for ly in range(angular_momentum - lx, -1, -1):
            lz = angular_momentum - lx - ly
            exponents.append((lx, ly, lz))
    return exponents


def _flatten_tensor_index(indices: tuple[int, ...]) -> int:
    flat_index = 0
    for axis in indices:
        flat_index = flat_index * 3 + axis
    return flat_index


@lru_cache(maxsize=None)
def _symmetric_cartesian_basis_rows(
    angular_momentum: int,
) -> tuple[tuple[float, ...], ...]:
    if angular_momentum == 0:
        return ((1.0,),)

    basis_rows: list[tuple[float, ...]] = []
    tensor_dimension = 3**angular_momentum
    for exponent in canonical_cartesian_exponents(angular_momentum):
        multiset = (
            (0,) * exponent[0]
            + (1,) * exponent[1]
            + (2,) * exponent[2]
        )
        permutations = sorted(set(itertools.permutations(multiset)))
        row = [0.0] * tensor_dimension
        scale = 1.0 / math.sqrt(len(permutations))
        for order in permutations:
            row[_flatten_tensor_index(order)] = scale
        basis_rows.append(tuple(row))
    return tuple(basis_rows)


def symmetric_cartesian_basis(
    angular_momentum: int,
    *,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    return jnp.asarray(
        _symmetric_cartesian_basis_rows(angular_momentum),
        dtype=dtype,
    )


@lru_cache(maxsize=None)
def irreps_for_cartesian_shell(angular_momentum: int) -> e3nn.Irreps:
    if angular_momentum == 0:
        return e3nn.Irreps("1x0e")
    if angular_momentum == 1:
        return e3nn.Irreps("1x1o")
    return e3nn.reduced_symmetric_tensor_product_basis(
        e3nn.Irreps("1o"),
        angular_momentum,
    ).irreps.simplify()


@lru_cache(maxsize=None)
def _cartesian_to_irreps_matrix_rows(
    angular_momentum: int,
) -> tuple[tuple[float, ...], ...]:
    if angular_momentum == 0:
        return ((1.0,),)
    if angular_momentum == 1:
        return (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )

    tensor_product = e3nn.reduced_symmetric_tensor_product_basis(
        e3nn.Irreps("1o"),
        angular_momentum,
    )
    change_of_basis = np.moveaxis(
        np.asarray(tensor_product.array, dtype=np.float64),
        -1,
        0,
    ).reshape(tensor_product.irreps.dim, -1)
    symmetric_basis = np.asarray(
        _symmetric_cartesian_basis_rows(angular_momentum),
        dtype=np.float64,
    )
    projection = change_of_basis @ symmetric_basis.T
    return tuple(
        tuple(float(value) for value in row)
        for row in projection.tolist()
    )


def cartesian_to_irreps_matrix(
    angular_momentum: int,
    *,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    return jnp.asarray(
        _cartesian_to_irreps_matrix_rows(angular_momentum),
        dtype=dtype,
    )


def cartesian_shell_rotation_matrix(
    angular_momentum: int,
    rotation_matrix: jnp.ndarray,
) -> jnp.ndarray:
    if angular_momentum == 0:
        return jnp.ones((1, 1), dtype=rotation_matrix.dtype)

    symmetric_basis = symmetric_cartesian_basis(
        angular_momentum,
        dtype=rotation_matrix.dtype,
    )
    full_rotation = rotation_matrix
    for _ in range(1, angular_momentum):
        full_rotation = jnp.kron(full_rotation, rotation_matrix)
    return symmetric_basis @ full_rotation @ symmetric_basis.T


def sparse_to_dense_ao_coefficients(
    sample: dict[str, Any],
    sparse_coefficients: jnp.ndarray,
) -> jnp.ndarray:
    n_orbitals, _ = sparse_coefficients.shape
    n_basis_functions = sample["ao_to_atom"].shape[0]
    orbital_basis_mask = sample["orbital_basis_mask"]
    ao_indices = sample["orbital_basis_index_table"] - 1
    safe_ao_indices = jnp.clip(ao_indices, a_min=0)

    orbital_indices = jnp.arange(n_orbitals, dtype=jnp.int32)[:, None]
    flat_indices = orbital_indices * n_basis_functions + safe_ao_indices
    flat_values = jnp.where(
        orbital_basis_mask,
        sparse_coefficients,
        jnp.zeros_like(sparse_coefficients),
    ).reshape(-1)

    dense_flat = jnp.zeros((n_orbitals * n_basis_functions,), dtype=sparse_coefficients.dtype)
    dense_flat = dense_flat.at[flat_indices.reshape(-1)].add(flat_values)
    return dense_flat.reshape(n_orbitals, n_basis_functions)


def dense_to_sparse_ao_coefficients(
    sample: dict[str, Any],
    dense_coefficients: jnp.ndarray,
) -> jnp.ndarray:
    orbital_basis_mask = sample["orbital_basis_mask"]
    ao_indices = sample["orbital_basis_index_table"] - 1
    gathered = jnp.take_along_axis(
        dense_coefficients,
        jnp.clip(ao_indices, a_min=0),
        axis=1,
    )
    return jnp.where(
        orbital_basis_mask,
        gathered,
        jnp.zeros_like(gathered),
    )


def validate_cartesian_shell_layout(sample: dict[str, Any]) -> None:
    n_shells = int(np.asarray(sample["shell_to_atom"]).shape[0])
    shell_angular_momenta = np.asarray(sample["shell_angular_momenta"])
    shell_ao_starts = np.asarray(sample["shell_ao_starts"])
    shell_ao_counts = np.asarray(sample["shell_ao_counts"])
    ao_cartesian_exponents = np.asarray(sample["ao_cartesian_exponents"])

    for shell_index in range(n_shells):
        angular_momentum = int(shell_angular_momenta[shell_index])
        shell_start = int(shell_ao_starts[shell_index])
        shell_count = int(shell_ao_counts[shell_index])
        expected_dimension = cartesian_shell_dimension(angular_momentum)
        if shell_count != expected_dimension:
            raise ValueError(
                "strict shell-equivariant encoding currently requires full Cartesian shells; "
                f"shell {shell_index} has l={angular_momentum} with count={shell_count}, "
                f"expected {expected_dimension}"
            )
        shell_exponents = [
            tuple(int(component) for component in exponent)
            for exponent in ao_cartesian_exponents[
                shell_start : shell_start + shell_count
            ].tolist()
        ]
        if shell_exponents != canonical_cartesian_exponents(angular_momentum):
            raise ValueError(
                "unexpected AO ordering for shell-equivariant encoding; "
                f"shell {shell_index} exponents are {shell_exponents}"
            )


def rotate_dense_ao_coefficients(
    sample: dict[str, Any],
    dense_coefficients: jnp.ndarray,
    rotation_matrix: jnp.ndarray,
) -> jnp.ndarray:
    validate_cartesian_shell_layout(sample)
    rotated = jnp.zeros_like(dense_coefficients)
    n_shells = int(np.asarray(sample["shell_to_atom"]).shape[0])
    shell_angular_momenta = np.asarray(sample["shell_angular_momenta"])
    shell_ao_starts = np.asarray(sample["shell_ao_starts"])
    shell_ao_counts = np.asarray(sample["shell_ao_counts"])

    for shell_index in range(n_shells):
        angular_momentum = int(shell_angular_momenta[shell_index])
        shell_start = int(shell_ao_starts[shell_index])
        shell_count = int(shell_ao_counts[shell_index])
        shell_rotation = cartesian_shell_rotation_matrix(
            angular_momentum,
            rotation_matrix,
        )
        shell_block = dense_coefficients[:, shell_start : shell_start + shell_count]
        rotated = rotated.at[:, shell_start : shell_start + shell_count].set(
            shell_block @ shell_rotation.T
        )
    return rotated
