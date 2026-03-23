from __future__ import annotations

import itertools
import math
from functools import lru_cache
from typing import Any

import torch
from e3nn import o3


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


def _symmetry_formula(angular_momentum: int) -> str:
    if angular_momentum == 0:
        raise ValueError("rank-0 tensors do not need a symmetry formula")
    letters = "ijklmnopqrstuvwxyzabcdefgh"
    if angular_momentum > len(letters):
        raise ValueError("angular_momentum is too large for the generated formula")
    indices = letters[:angular_momentum]
    permutations = sorted(
        {"".join(order) for order in itertools.permutations(indices)}
    )
    return "=".join(permutations)


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
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> torch.Tensor:
    return torch.tensor(
        _symmetric_cartesian_basis_rows(angular_momentum),
        dtype=dtype,
        device=device,
    )


@lru_cache(maxsize=None)
def irreps_for_cartesian_shell(angular_momentum: int) -> o3.Irreps:
    if angular_momentum == 0:
        return o3.Irreps("1x0e")
    if angular_momentum == 1:
        return o3.Irreps("1x1o")
    tensor_product = o3.ReducedTensorProducts(
        _symmetry_formula(angular_momentum),
        i="1o",
    )
    return tensor_product.irreps_out.simplify()


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

    tensor_product = o3.ReducedTensorProducts(
        _symmetry_formula(angular_momentum),
        i="1o",
    )
    change_of_basis = tensor_product.change_of_basis.reshape(
        tensor_product.irreps_out.dim,
        -1,
    ).to(dtype=torch.float64)
    symmetric_basis = symmetric_cartesian_basis(
        angular_momentum,
        dtype=torch.float64,
    )
    projection = change_of_basis @ symmetric_basis.transpose(0, 1)
    return tuple(
        tuple(float(value) for value in row)
        for row in projection.tolist()
    )


def cartesian_to_irreps_matrix(
    angular_momentum: int,
    *,
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> torch.Tensor:
    return torch.tensor(
        _cartesian_to_irreps_matrix_rows(angular_momentum),
        dtype=dtype,
        device=device,
    )


def cartesian_shell_rotation_matrix(
    angular_momentum: int,
    rotation_matrix: torch.Tensor,
) -> torch.Tensor:
    if angular_momentum == 0:
        return rotation_matrix.new_ones((1, 1))

    symmetric_basis = symmetric_cartesian_basis(
        angular_momentum,
        dtype=rotation_matrix.dtype,
        device=rotation_matrix.device,
    )
    full_rotation = rotation_matrix
    for _ in range(1, angular_momentum):
        full_rotation = torch.kron(full_rotation, rotation_matrix)
    return symmetric_basis @ full_rotation @ symmetric_basis.transpose(0, 1)


def sparse_to_dense_ao_coefficients(
    sample: dict[str, Any],
    sparse_coefficients: torch.Tensor,
) -> torch.Tensor:
    n_orbitals, _ = sparse_coefficients.shape
    n_basis_functions = sample["ao_to_atom"].shape[0]
    orbital_basis_mask = sample["orbital_basis_mask"]
    ao_indices = sample["orbital_basis_index_table"] - 1
    safe_ao_indices = ao_indices.clamp_min(0)

    orbital_indices = torch.arange(
        n_orbitals,
        device=sparse_coefficients.device,
        dtype=torch.int64,
    ).unsqueeze(1).expand_as(safe_ao_indices)
    flat_indices = orbital_indices * n_basis_functions + safe_ao_indices
    flat_values = torch.where(
        orbital_basis_mask,
        sparse_coefficients,
        torch.zeros_like(sparse_coefficients),
    ).reshape(-1)

    dense_flat = sparse_coefficients.new_zeros(n_orbitals * n_basis_functions)
    dense_flat = dense_flat.scatter_add(0, flat_indices.reshape(-1), flat_values)
    return dense_flat.reshape(n_orbitals, n_basis_functions)


def dense_to_sparse_ao_coefficients(
    sample: dict[str, Any],
    dense_coefficients: torch.Tensor,
) -> torch.Tensor:
    orbital_basis_mask = sample["orbital_basis_mask"]
    ao_indices = sample["orbital_basis_index_table"] - 1
    gathered = dense_coefficients.gather(1, ao_indices.clamp_min(0))
    return torch.where(
        orbital_basis_mask,
        gathered,
        torch.zeros_like(gathered),
    )


def validate_cartesian_shell_layout(sample: dict[str, Any]) -> None:
    n_shells = int(sample["shell_to_atom"].shape[0])
    for shell_index in range(n_shells):
        angular_momentum = int(sample["shell_angular_momenta"][shell_index].item())
        shell_start = int(sample["shell_ao_starts"][shell_index].item())
        shell_count = int(sample["shell_ao_counts"][shell_index].item())
        expected_dimension = cartesian_shell_dimension(angular_momentum)
        if shell_count != expected_dimension:
            raise ValueError(
                "strict shell-equivariant encoding currently requires full Cartesian shells; "
                f"shell {shell_index} has l={angular_momentum} with count={shell_count}, "
                f"expected {expected_dimension}"
            )
        shell_exponents = [
            tuple(int(component) for component in exponent)
            for exponent in sample["ao_cartesian_exponents"][
                shell_start : shell_start + shell_count
            ].detach().cpu().tolist()
        ]
        if shell_exponents != canonical_cartesian_exponents(angular_momentum):
            raise ValueError(
                "unexpected AO ordering for shell-equivariant encoding; "
                f"shell {shell_index} exponents are {shell_exponents}"
            )


def rotate_dense_ao_coefficients(
    sample: dict[str, Any],
    dense_coefficients: torch.Tensor,
    rotation_matrix: torch.Tensor,
) -> torch.Tensor:
    validate_cartesian_shell_layout(sample)
    rotated = torch.zeros_like(dense_coefficients)
    n_shells = int(sample["shell_to_atom"].shape[0])
    for shell_index in range(n_shells):
        angular_momentum = int(sample["shell_angular_momenta"][shell_index].item())
        shell_start = int(sample["shell_ao_starts"][shell_index].item())
        shell_count = int(sample["shell_ao_counts"][shell_index].item())
        shell_rotation = cartesian_shell_rotation_matrix(
            angular_momentum,
            rotation_matrix,
        )
        shell_block = dense_coefficients[:, shell_start : shell_start + shell_count]
        rotated[:, shell_start : shell_start + shell_count] = (
            shell_block @ shell_rotation.transpose(0, 1)
        )
    return rotated
