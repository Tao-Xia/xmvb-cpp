#!/usr/bin/env python3
"""
Compare two coefficient maps for one closed-shell structure:

1. exact raw-VB determinant expansion;
2. determinant coefficients from the oriented spatial AGP matrix F.

If these maps differ, then the oriented single-AGP encoding is not the same
many-electron state as the raw-VB structure.
"""

import itertools
import numpy as np


def parse_pairs(text):
    return [(int(a) - 1, int(b) - 1) for a, b in (token.split("-") for token in text.split())]


def canonical_with_sign(indices):
    indices = list(indices)
    sign = 1.0
    for i in range(len(indices) - 1):
        for j in range(i + 1, len(indices)):
            if indices[i] > indices[j]:
                indices[i], indices[j] = indices[j], indices[i]
                sign = -sign
    return tuple(indices), sign


def raw_vb_terms(pairs):
    terms = {((), ()): 1.0}
    for i, j in pairs:
        next_terms = {}
        for (alpha_occ, beta_occ), coefficient in terms.items():
            key = (alpha_occ + (i,), beta_occ + (j,))
            next_terms[key] = next_terms.get(key, 0.0) + coefficient
            if i != j:
                key = (alpha_occ + (j,), beta_occ + (i,))
                next_terms[key] = next_terms.get(key, 0.0) - coefficient
        terms = next_terms

    out = {}
    for (alpha_occ, beta_occ), coefficient in terms.items():
        alpha_occ, alpha_sign = canonical_with_sign(alpha_occ)
        beta_occ, beta_sign = canonical_with_sign(beta_occ)
        key = (alpha_occ, beta_occ)
        out[key] = out.get(key, 0.0) + coefficient * alpha_sign * beta_sign
    return {key: value for key, value in sorted(out.items()) if abs(value) > 1.0e-12}


def oriented_pair_matrix(pairs):
    n_orbitals = max(max(i, j) for i, j in pairs) + 1
    F = np.zeros((n_orbitals, n_orbitals), dtype=float)
    for i, j in pairs:
        if i == j:
            F[i, i] += 1.0
        elif i < j:
            F[i, j] += 1.0
            F[j, i] -= 1.0
        else:
            F[j, i] += 1.0
            F[i, j] -= 1.0
    return F


def agp_terms(F, n_pairs):
    n_orbitals = F.shape[0]
    out = {}
    for alpha_occ in itertools.combinations(range(n_orbitals), n_pairs):
        for beta_occ in itertools.combinations(range(n_orbitals), n_pairs):
            coefficient = float(np.linalg.det(F[np.ix_(alpha_occ, beta_occ)]))
            if abs(coefficient) > 1.0e-12:
                out[(alpha_occ, beta_occ)] = coefficient
    return {key: value for key, value in sorted(out.items())}


test_structures = [
    "1-1 2-2",
    "1-2 3-4",
    "1-2 1-2",
]

for structure_text in test_structures:
    pairs = parse_pairs(structure_text)
    F = oriented_pair_matrix(pairs)
    raw_terms = raw_vb_terms(pairs)
    agp_minor_terms = agp_terms(F, len(pairs))
    print(f"\nstructure = {structure_text}")
    print("F =")
    print(F)
    print("raw-VB determinant coefficients:")
    for key, value in raw_terms.items():
        print(f"  {key}: {value:+.6f}")
    print("oriented-AGP determinant coefficients:")
    for key, value in agp_minor_terms.items():
        print(f"  {key}: {value:+.6f}")
    print(f"match = {raw_terms == agp_minor_terms}")
