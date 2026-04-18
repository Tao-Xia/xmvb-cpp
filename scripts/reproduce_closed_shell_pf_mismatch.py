#!/usr/bin/env python3
"""
Reproduce the mismatch between:

1. the exact raw-VB overlap from determinant expansion, and
2. the current prototype closed-shell Pfaffian encoding used by the
   symmetric `alpha_beta_block` + normalization + trace-projector path.

This script is intentionally not a "correct Pfaffian raw-VB overlap" example.
It exists only to reproduce the current mismatch in a minimal, explicit form.
"""

import numpy as np


structure_left = "1-2 3-4"
structure_right = "1-2 3-4"


all_tokens = structure_left.split() + structure_right.split()
largest_label = 0
for token in all_tokens:
    left_text, right_text = token.split("-")
    left_label = int(left_text)
    right_label = int(right_text)
    if left_label > largest_label:
        largest_label = left_label
    if right_label > largest_label:
        largest_label = right_label

n_orbitals = largest_label
n_pairs = len(structure_left.split())


left_pairs = []
for token in structure_left.split():
    left_text, right_text = token.split("-")
    i = int(left_text) - 1
    j = int(right_text) - 1
    left_pairs.append((i, j))

right_pairs = []
for token in structure_right.split():
    left_text, right_text = token.split("-")
    i = int(left_text) - 1
    j = int(right_text) - 1
    right_pairs.append((i, j))


left_adjacency = np.zeros((n_orbitals, n_orbitals), dtype=float)
for i, j in left_pairs:
    left_adjacency[i, j] = 1.0
    left_adjacency[j, i] = 1.0

right_adjacency = np.zeros((n_orbitals, n_orbitals), dtype=float)
for i, j in right_pairs:
    right_adjacency[i, j] = 1.0
    right_adjacency[j, i] = 1.0


spatial_overlap = np.eye(n_orbitals, dtype=float)


left_partial_terms = []
left_partial_terms.append(
    {
        "alpha_occ": [],
        "beta_occ": [],
        "coefficient": 1.0,
    }
)

for i, j in left_pairs:
    next_terms = []
    for term in left_partial_terms:
        direct_term = {
            "alpha_occ": term["alpha_occ"][:],
            "beta_occ": term["beta_occ"][:],
            "coefficient": term["coefficient"],
        }
        direct_term["alpha_occ"].append(i)
        direct_term["beta_occ"].append(j)
        next_terms.append(direct_term)

        if i != j:
            swapped_term = {
                "alpha_occ": term["alpha_occ"][:],
                "beta_occ": term["beta_occ"][:],
                "coefficient": -term["coefficient"],
            }
            swapped_term["alpha_occ"].append(j)
            swapped_term["beta_occ"].append(i)
            next_terms.append(swapped_term)
    left_partial_terms = next_terms

left_accumulated_terms = {}
for term in left_partial_terms:
    alpha_occ = term["alpha_occ"][:]
    beta_occ = term["beta_occ"][:]
    coefficient = term["coefficient"]

    for left_index in range(len(alpha_occ) - 1):
        for right_index in range(left_index + 1, len(alpha_occ)):
            if alpha_occ[left_index] > alpha_occ[right_index]:
                alpha_occ[left_index], alpha_occ[right_index] = (
                    alpha_occ[right_index],
                    alpha_occ[left_index],
                )
                coefficient = -coefficient

    for left_index in range(len(beta_occ) - 1):
        for right_index in range(left_index + 1, len(beta_occ)):
            if beta_occ[left_index] > beta_occ[right_index]:
                beta_occ[left_index], beta_occ[right_index] = (
                    beta_occ[right_index],
                    beta_occ[left_index],
                )
                coefficient = -coefficient

    key = (tuple(alpha_occ), tuple(beta_occ))
    if key not in left_accumulated_terms:
        left_accumulated_terms[key] = 0.0
    left_accumulated_terms[key] += coefficient

left_terms = []
for key, coefficient in left_accumulated_terms.items():
    if abs(coefficient) <= 1.0e-12:
        continue
    left_terms.append(
        {
            "alpha_occ": list(key[0]),
            "beta_occ": list(key[1]),
            "coefficient": coefficient,
        }
    )


right_partial_terms = []
right_partial_terms.append(
    {
        "alpha_occ": [],
        "beta_occ": [],
        "coefficient": 1.0,
    }
)

for i, j in right_pairs:
    next_terms = []
    for term in right_partial_terms:
        direct_term = {
            "alpha_occ": term["alpha_occ"][:],
            "beta_occ": term["beta_occ"][:],
            "coefficient": term["coefficient"],
        }
        direct_term["alpha_occ"].append(i)
        direct_term["beta_occ"].append(j)
        next_terms.append(direct_term)

        if i != j:
            swapped_term = {
                "alpha_occ": term["alpha_occ"][:],
                "beta_occ": term["beta_occ"][:],
                "coefficient": -term["coefficient"],
            }
            swapped_term["alpha_occ"].append(j)
            swapped_term["beta_occ"].append(i)
            next_terms.append(swapped_term)
    right_partial_terms = next_terms

right_accumulated_terms = {}
for term in right_partial_terms:
    alpha_occ = term["alpha_occ"][:]
    beta_occ = term["beta_occ"][:]
    coefficient = term["coefficient"]

    for left_index in range(len(alpha_occ) - 1):
        for right_index in range(left_index + 1, len(alpha_occ)):
            if alpha_occ[left_index] > alpha_occ[right_index]:
                alpha_occ[left_index], alpha_occ[right_index] = (
                    alpha_occ[right_index],
                    alpha_occ[left_index],
                )
                coefficient = -coefficient

    for left_index in range(len(beta_occ) - 1):
        for right_index in range(left_index + 1, len(beta_occ)):
            if beta_occ[left_index] > beta_occ[right_index]:
                beta_occ[left_index], beta_occ[right_index] = (
                    beta_occ[right_index],
                    beta_occ[left_index],
                )
                coefficient = -coefficient

    key = (tuple(alpha_occ), tuple(beta_occ))
    if key not in right_accumulated_terms:
        right_accumulated_terms[key] = 0.0
    right_accumulated_terms[key] += coefficient

right_terms = []
for key, coefficient in right_accumulated_terms.items():
    if abs(coefficient) <= 1.0e-12:
        continue
    right_terms.append(
        {
            "alpha_occ": list(key[0]),
            "beta_occ": list(key[1]),
            "coefficient": coefficient,
        }
    )


exact_overlap = 0.0
for left_term in left_terms:
    for right_term in right_terms:
        alpha_submatrix = np.zeros((n_pairs, n_pairs), dtype=float)
        for row in range(n_pairs):
            for column in range(n_pairs):
                left_orbital = left_term["alpha_occ"][row]
                right_orbital = right_term["alpha_occ"][column]
                alpha_submatrix[row, column] = spatial_overlap[left_orbital, right_orbital]

        beta_submatrix = np.zeros((n_pairs, n_pairs), dtype=float)
        for row in range(n_pairs):
            for column in range(n_pairs):
                left_orbital = left_term["beta_occ"][row]
                right_orbital = right_term["beta_occ"][column]
                beta_submatrix[row, column] = spatial_overlap[left_orbital, right_orbital]

        exact_overlap += (
            left_term["coefficient"]
            * right_term["coefficient"]
            * np.linalg.det(alpha_submatrix)
            * np.linalg.det(beta_submatrix)
        )


prototype_left_block = left_adjacency.copy()
prototype_right_block = right_adjacency.copy()

prototype_left_block /= np.sqrt(max(1.0e-30, np.sum(prototype_left_block * prototype_left_block)))
prototype_right_block /= np.sqrt(max(1.0e-30, np.sum(prototype_right_block * prototype_right_block)))


left_ba = -prototype_left_block.transpose()
right_ab = prototype_right_block

left_pair = left_ba.transpose()
left_pair_times_overlap = left_pair @ spatial_overlap
right_pair_transpose = right_ab.transpose()

pair_core = left_pair_times_overlap @ right_pair_transpose
spatial_kernel = pair_core @ spatial_overlap


trace_1 = 2.0 * np.trace(spatial_kernel)
trace_2 = 2.0 * np.trace(spatial_kernel @ spatial_kernel)


coefficient_0 = 1.0
coefficient_1 = 0.5 * trace_1 * coefficient_0
coefficient_2 = (
    0.5 * trace_1 * coefficient_1
    - 0.5 * trace_2 * coefficient_0
) / 2.0

prototype_overlap = coefficient_2


print(f"left VB structure : {structure_left}")
print(f"right VB structure: {structure_right}")
print()
print("left adjacency matrix:")
print(left_adjacency)
print()
print("right adjacency matrix:")
print(right_adjacency)
print()
print("spatial-orbital overlap matrix S:")
print(spatial_overlap)
print()
print("prototype normalized alpha_beta_block (left):")
print(prototype_left_block)
print()
print("prototype normalized alpha_beta_block (right):")
print(prototype_right_block)
print()
print("prototype pair core = L S R^T:")
print(pair_core)
print()
print("prototype spatial kernel = L S R^T S:")
print(spatial_kernel)
print()
print(f"exact determinant overlap = {exact_overlap}")
print(f"prototype trace-projector overlap = {prototype_overlap}")
print(f"absolute difference = {abs(exact_overlap - prototype_overlap)}")
