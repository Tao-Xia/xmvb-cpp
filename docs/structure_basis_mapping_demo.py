#!/usr/bin/env python3

"""
Minimal demo of how structure-basis matrix elements are assembled from
unique alpha/beta spin-string channels.

This script computes the same structure overlap / Hamiltonian matrices in two
ways:

1. Brute force over full determinant pairs:
     M_IJ = sum_{d,e} t_{I,d} M_de t_{J,e}

2. Matrix form on the unique alpha/beta spaces:
     Build one coefficient matrix C_I(a, b) per structure, then evaluate
     Frobenius inner products such as
       S_IJ = <C_I, S_alpha C_J S_beta^T>_F

The two paths should agree to numerical precision.
"""

from __future__ import annotations

import numpy as np


def build_demo_data():
    """Return a small hand-written example."""
    # Full determinants D_d = (alpha_id(d), beta_id(d)).
    det_to_alpha = [0, 0, 1, 1]
    det_to_beta = [0, 1, 0, 1]

    # determinant_to_structure_terms[d] = [(structure_id, coefficient), ...]
    structure_terms = [
        [(0, +1.0)],
        [(0, +1.0), (1, +1.0)],
        [(0, -1.0), (1, +1.0)],
        [(1, +1.0)],
    ]

    # Unique-spin kernels.
    s_alpha = np.array([
        [1.00, 0.20],
        [0.20, 1.10],
    ])
    s_beta = np.array([
        [1.00, 0.30],
        [0.30, 0.90],
    ])
    h1_alpha = np.array([
        [2.00, 0.40],
        [0.40, 1.80],
    ])
    h1_beta = np.array([
        [1.50, 0.10],
        [0.10, 1.20],
    ])
    h_alpha = np.array([
        [2.50, 0.50],
        [0.50, 2.20],
    ])
    h_beta = np.array([
        [1.80, 0.20],
        [0.20, 1.40],
    ])

    # Opposite-spin channels grouped by packed active-pair index P.
    u_alpha = [
        np.array([
            [0.30, 0.10],
            [0.05, 0.25],
        ]),
        np.array([
            [0.20, 0.00],
            [-0.10, 0.15],
        ]),
    ]
    gu_beta = [
        np.array([
            [0.40, 0.20],
            [0.10, 0.35],
        ]),
        np.array([
            [0.10, 0.30],
            [0.20, 0.05],
        ]),
    ]

    return {
        "det_to_alpha": det_to_alpha,
        "det_to_beta": det_to_beta,
        "structure_terms": structure_terms,
        "n_structures": 2,
        "n_unique_alpha": 2,
        "n_unique_beta": 2,
        "s_alpha": s_alpha,
        "s_beta": s_beta,
        "h1_alpha": h1_alpha,
        "h1_beta": h1_beta,
        "h_alpha": h_alpha,
        "h_beta": h_beta,
        "u_alpha": u_alpha,
        "gu_beta": gu_beta,
    }


def structure_coefficient(det_index: int, structure_index: int, structure_terms):
    """Return t_{I,d} for one (structure, determinant) pair."""
    coeff = 0.0
    for target_structure, value in structure_terms[det_index]:
        if target_structure == structure_index:
            coeff += value
    return coeff


def build_structure_coefficient_matrices(
    det_to_alpha,
    det_to_beta,
    structure_terms,
    n_structures,
    n_unique_alpha,
    n_unique_beta,
):
    """
    Build one matrix C_I(a, b) per structure:

      C_I(a, b) = sum_{d : alpha(d)=a, beta(d)=b} t_{I,d}
    """
    coefficient_matrices = [
        np.zeros((n_unique_alpha, n_unique_beta)) for _ in range(n_structures)
    ]

    for det_index in range(len(det_to_alpha)):
        alpha_id = det_to_alpha[det_index]
        beta_id = det_to_beta[det_index]
        for structure_index, coeff in structure_terms[det_index]:
            coefficient_matrices[structure_index][alpha_id, beta_id] += coeff

    return coefficient_matrices


def determinant_overlap(det_left, det_right, det_to_alpha, det_to_beta, s_alpha, s_beta):
    """S_de = S_alpha(aL, aR) * S_beta(bL, bR)."""
    alpha_left = det_to_alpha[det_left]
    alpha_right = det_to_alpha[det_right]
    beta_left = det_to_beta[det_left]
    beta_right = det_to_beta[det_right]
    return s_alpha[alpha_left, alpha_right] * s_beta[beta_left, beta_right]


def determinant_one_electron(
    det_left,
    det_right,
    det_to_alpha,
    det_to_beta,
    s_alpha,
    s_beta,
    h1_alpha,
    h1_beta,
):
    """One-electron full determinant matrix element."""
    alpha_left = det_to_alpha[det_left]
    alpha_right = det_to_alpha[det_right]
    beta_left = det_to_beta[det_left]
    beta_right = det_to_beta[det_right]
    return (
        h1_alpha[alpha_left, alpha_right] * s_beta[beta_left, beta_right]
        + s_alpha[alpha_left, alpha_right] * h1_beta[beta_left, beta_right]
    )


def determinant_total(
    det_left,
    det_right,
    det_to_alpha,
    det_to_beta,
    s_alpha,
    s_beta,
    h_alpha,
    h_beta,
    u_alpha,
    gu_beta,
):
    """Total full determinant matrix element for the toy example."""
    alpha_left = det_to_alpha[det_left]
    alpha_right = det_to_alpha[det_right]
    beta_left = det_to_beta[det_left]
    beta_right = det_to_beta[det_right]

    same_spin_part = (
        h_alpha[alpha_left, alpha_right] * s_beta[beta_left, beta_right]
        + s_alpha[alpha_left, alpha_right] * h_beta[beta_left, beta_right]
    )

    opposite_spin_part = 0.0
    for packed_pair_index in range(len(u_alpha)):
        opposite_spin_part += (
            u_alpha[packed_pair_index][alpha_left, alpha_right]
            * gu_beta[packed_pair_index][beta_left, beta_right]
        )

    return same_spin_part + opposite_spin_part


def brute_force_structure_matrices(data):
    """Compute structure matrices by summing over all full determinant pairs."""
    n_det = len(data["det_to_alpha"])
    n_structures = data["n_structures"]
    structure_terms = data["structure_terms"]

    overlap_matrix = np.zeros((n_structures, n_structures))
    one_electron_matrix = np.zeros((n_structures, n_structures))
    total_matrix = np.zeros((n_structures, n_structures))

    for left_structure in range(n_structures):
        for right_structure in range(n_structures):
            for det_left in range(n_det):
                coeff_left = structure_coefficient(
                    det_left, left_structure, structure_terms
                )
                if abs(coeff_left) < 1.0e-15:
                    continue

                for det_right in range(n_det):
                    coeff_right = structure_coefficient(
                        det_right, right_structure, structure_terms
                    )
                    if abs(coeff_right) < 1.0e-15:
                        continue

                    weight = coeff_left * coeff_right
                    overlap_matrix[left_structure, right_structure] += (
                        weight
                        * determinant_overlap(
                            det_left,
                            det_right,
                            data["det_to_alpha"],
                            data["det_to_beta"],
                            data["s_alpha"],
                            data["s_beta"],
                        )
                    )
                    one_electron_matrix[left_structure, right_structure] += (
                        weight
                        * determinant_one_electron(
                            det_left,
                            det_right,
                            data["det_to_alpha"],
                            data["det_to_beta"],
                            data["s_alpha"],
                            data["s_beta"],
                            data["h1_alpha"],
                            data["h1_beta"],
                        )
                    )
                    total_matrix[left_structure, right_structure] += (
                        weight
                        * determinant_total(
                            det_left,
                            det_right,
                            data["det_to_alpha"],
                            data["det_to_beta"],
                            data["s_alpha"],
                            data["s_beta"],
                            data["h_alpha"],
                            data["h_beta"],
                            data["u_alpha"],
                            data["gu_beta"],
                        )
                    )

    return overlap_matrix, one_electron_matrix, total_matrix


def frobenius_inner_product(left_matrix, right_matrix):
    """<A, B>_F = sum_ij A_ij B_ij."""
    return np.sum(left_matrix * right_matrix)


def matrix_form_structure_matrices(data, coefficient_matrices):
    """Compute the same matrices directly on the unique alpha/beta spaces."""
    n_structures = data["n_structures"]

    overlap_matrix = np.zeros((n_structures, n_structures))
    one_electron_matrix = np.zeros((n_structures, n_structures))
    total_matrix = np.zeros((n_structures, n_structures))

    for left_structure in range(n_structures):
        left_coefficients = coefficient_matrices[left_structure]

        for right_structure in range(n_structures):
            right_coefficients = coefficient_matrices[right_structure]

            overlap_image = (
                data["s_alpha"] @ right_coefficients @ data["s_beta"].T
            )

            one_electron_image = (
                data["h1_alpha"] @ right_coefficients @ data["s_beta"].T
                + data["s_alpha"] @ right_coefficients @ data["h1_beta"].T
            )

            total_image = (
                data["h_alpha"] @ right_coefficients @ data["s_beta"].T
                + data["s_alpha"] @ right_coefficients @ data["h_beta"].T
            )

            opposite_spin_image = np.zeros_like(right_coefficients)
            for packed_pair_index in range(len(data["u_alpha"])):
                opposite_spin_image += (
                    data["u_alpha"][packed_pair_index]
                    @ right_coefficients
                    @ data["gu_beta"][packed_pair_index].T
                )

            total_image += opposite_spin_image

            overlap_matrix[left_structure, right_structure] = frobenius_inner_product(
                left_coefficients, overlap_image
            )
            one_electron_matrix[left_structure, right_structure] = (
                frobenius_inner_product(left_coefficients, one_electron_image)
            )
            total_matrix[left_structure, right_structure] = (
                frobenius_inner_product(left_coefficients, total_image)
            )

    return overlap_matrix, one_electron_matrix, total_matrix


def print_matrix(label: str, value: np.ndarray):
    print(f"{label} =")
    print(value)
    print()


def main():
    np.set_printoptions(precision=6, suppress=True)

    data = build_demo_data()
    coefficient_matrices = build_structure_coefficient_matrices(
        data["det_to_alpha"],
        data["det_to_beta"],
        data["structure_terms"],
        data["n_structures"],
        data["n_unique_alpha"],
        data["n_unique_beta"],
    )

    print("=== Structure coefficient matrices C_I(alpha, beta) ===")
    for structure_index, coefficients in enumerate(coefficient_matrices):
        print_matrix(f"C[{structure_index}]", coefficients)

    brute_overlap, brute_one_electron, brute_total = brute_force_structure_matrices(
        data
    )
    mf_overlap, mf_one_electron, mf_total = matrix_form_structure_matrices(
        data, coefficient_matrices
    )

    print("=== Brute force over full determinant pairs ===")
    print_matrix("overlap", brute_overlap)
    print_matrix("one_electron", brute_one_electron)
    print_matrix("total", brute_total)

    print("=== Matrix form on unique alpha/beta spaces ===")
    print_matrix("overlap", mf_overlap)
    print_matrix("one_electron", mf_one_electron)
    print_matrix("total", mf_total)

    print("=== Max absolute differences ===")
    print("overlap:", np.max(np.abs(brute_overlap - mf_overlap)))
    print("one_electron:", np.max(np.abs(brute_one_electron - mf_one_electron)))
    print("total:", np.max(np.abs(brute_total - mf_total)))


if __name__ == "__main__":
    main()
