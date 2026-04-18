#!/usr/bin/env python3
"""
Exact closed-shell VB-structure overlap example.

This script starts from two VB structures written in the chemistry-style form

    1-2 3-4

meaning that orbitals 1 and 2 form one singlet bond, and orbitals 3 and 4
form another singlet bond.

Unlike the previous toy script, this file computes the overlap by the exact
singlet determinant expansion that matches the determinant-basis reference.
So if the spatial-orbital overlap matrix is the identity and the two
structures are the same, the self-overlap of

    (1-2)(3-4)

is 4, not 6 and not 0.375.
"""

import math

import numpy as np


# ---------------------------------------------------------------------------
# 1. Input two VB structures.
#    You can change these two strings directly.
# ---------------------------------------------------------------------------
structure_left = "1-2 3-4"
structure_right = "1-2 3-4"


# ---------------------------------------------------------------------------
# 2. Determine the active orbital dimension M from the largest orbital label.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 3. Parse the left VB structure into an explicit pair list.
#    The orbital labels are converted from 1-based to 0-based indexing.
# ---------------------------------------------------------------------------
left_pairs = []
for token in structure_left.split():
    left_text, right_text = token.split("-")
    i = int(left_text) - 1
    j = int(right_text) - 1
    left_pairs.append((i, j))


# ---------------------------------------------------------------------------
# 4. Parse the right VB structure into an explicit pair list.
# ---------------------------------------------------------------------------
right_pairs = []
for token in structure_right.split():
    left_text, right_text = token.split("-")
    i = int(left_text) - 1
    j = int(right_text) - 1
    right_pairs.append((i, j))


# ---------------------------------------------------------------------------
# 5. Build adjacency matrices only as a direct visualization of the VB graph.
#    These adjacency matrices are not the final overlap formula.
# ---------------------------------------------------------------------------
left_adjacency = np.zeros((n_orbitals, n_orbitals), dtype=float)
for i, j in left_pairs:
    left_adjacency[i, j] = 1.0
    left_adjacency[j, i] = 1.0

right_adjacency = np.zeros((n_orbitals, n_orbitals), dtype=float)
for i, j in right_pairs:
    right_adjacency[i, j] = 1.0
    right_adjacency[j, i] = 1.0


# ---------------------------------------------------------------------------
# 6. Define the spatial-orbital overlap matrix S.
#    For the simplest check, use the identity matrix.
#    Then the self-overlap of (1-2)(3-4) should be 4.
# ---------------------------------------------------------------------------
spatial_overlap = np.eye(n_orbitals, dtype=float)


# ---------------------------------------------------------------------------
# 7. Expand the left VB structure into determinant terms.
#
#    One covalent singlet pair (i-j) is expanded as
#
#        alpha_i beta_j - alpha_j beta_i
#
#    so each non-diagonal pair creates two determinant terms with coefficients
#    +1 and -1.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 8. Expand the right VB structure into determinant terms by the same rule.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 9. Print the structures and their adjacency matrices.
# ---------------------------------------------------------------------------
print("left VB structure :", structure_left)
print("right VB structure:", structure_right)
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


# ---------------------------------------------------------------------------
# 10. Print the determinant expansions explicitly.
# ---------------------------------------------------------------------------
print("left determinant terms:")
for term in left_terms:
    print(term)
print()

print("right determinant terms:")
for term in right_terms:
    print(term)
print()


# ---------------------------------------------------------------------------
# 11. Compute the exact overlap S12 from the determinant expansion.
# ---------------------------------------------------------------------------
overlap_s12 = 0.0
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

        alpha_overlap = np.linalg.det(alpha_submatrix)
        beta_overlap = np.linalg.det(beta_submatrix)

        contribution = (
            left_term["coefficient"]
            * right_term["coefficient"]
            * alpha_overlap
            * beta_overlap
        )
        overlap_s12 += contribution


# ---------------------------------------------------------------------------
# 12. Compute the self-overlap S11 of the left structure.
# ---------------------------------------------------------------------------
overlap_s11 = 0.0
for left_term_a in left_terms:
    for left_term_b in left_terms:
        alpha_submatrix = np.zeros((n_pairs, n_pairs), dtype=float)
        for row in range(n_pairs):
            for column in range(n_pairs):
                left_orbital = left_term_a["alpha_occ"][row]
                right_orbital = left_term_b["alpha_occ"][column]
                alpha_submatrix[row, column] = spatial_overlap[left_orbital, right_orbital]

        beta_submatrix = np.zeros((n_pairs, n_pairs), dtype=float)
        for row in range(n_pairs):
            for column in range(n_pairs):
                left_orbital = left_term_a["beta_occ"][row]
                right_orbital = left_term_b["beta_occ"][column]
                beta_submatrix[row, column] = spatial_overlap[left_orbital, right_orbital]

        alpha_overlap = np.linalg.det(alpha_submatrix)
        beta_overlap = np.linalg.det(beta_submatrix)

        contribution = (
            left_term_a["coefficient"]
            * left_term_b["coefficient"]
            * alpha_overlap
            * beta_overlap
        )
        overlap_s11 += contribution


# ---------------------------------------------------------------------------
# 13. Compute the self-overlap S22 of the right structure.
# ---------------------------------------------------------------------------
overlap_s22 = 0.0
for right_term_a in right_terms:
    for right_term_b in right_terms:
        alpha_submatrix = np.zeros((n_pairs, n_pairs), dtype=float)
        for row in range(n_pairs):
            for column in range(n_pairs):
                left_orbital = right_term_a["alpha_occ"][row]
                right_orbital = right_term_b["alpha_occ"][column]
                alpha_submatrix[row, column] = spatial_overlap[left_orbital, right_orbital]

        beta_submatrix = np.zeros((n_pairs, n_pairs), dtype=float)
        for row in range(n_pairs):
            for column in range(n_pairs):
                left_orbital = right_term_a["beta_occ"][row]
                right_orbital = right_term_b["beta_occ"][column]
                beta_submatrix[row, column] = spatial_overlap[left_orbital, right_orbital]

        alpha_overlap = np.linalg.det(alpha_submatrix)
        beta_overlap = np.linalg.det(beta_submatrix)

        contribution = (
            right_term_a["coefficient"]
            * right_term_b["coefficient"]
            * alpha_overlap
            * beta_overlap
        )
        overlap_s22 += contribution


# ---------------------------------------------------------------------------
# 14. Print the final overlap values.
# ---------------------------------------------------------------------------
normalized_overlap = overlap_s12 / math.sqrt(overlap_s11 * overlap_s22)

print("exact S11 =", overlap_s11)
print("exact S22 =", overlap_s22)
print("exact S12 =", overlap_s12)
print("normalized overlap =", normalized_overlap)
