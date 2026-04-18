#!/usr/bin/env python3
"""
Verify the exact bond-labeled determinant generating-function formula for the
closed-shell legacy raw-VB overlap.

This script compares:

1. the explicit determinant expansion used as the exact oracle, and
2. the constant-term formula built from selector matrices U and V;
3. the equivalent exact hypercube-average coefficient extraction.

The polynomial determinant implementation below is intentionally verification
only. It uses permutation expansion and is not meant as a production algorithm.
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


def expand_legacy_terms(pairs):
    terms = {((), ()): 1.0}
    for i, j in pairs:
        next_terms = {}
        for (alpha_occ, beta_occ), coefficient in terms.items():
            key = (alpha_occ + (i,), beta_occ + (j,))
            next_terms[key] = next_terms.get(key, 0.0) + coefficient
            if i != j:
                key = (alpha_occ + (j,), beta_occ + (i,))
                next_terms[key] = next_terms.get(key, 0.0) + coefficient
        terms = next_terms
    out = {}
    for (alpha_occ, beta_occ), coefficient in terms.items():
        alpha_occ, alpha_sign = canonical_with_sign(alpha_occ)
        beta_occ, beta_sign = canonical_with_sign(beta_occ)
        key = (alpha_occ, beta_occ)
        out[key] = out.get(key, 0.0) + coefficient * alpha_sign * beta_sign
    return {key: value for key, value in out.items() if abs(value) > 1.0e-12}


def det_overlap_legacy(left_pairs, right_pairs, S):
    if len(left_pairs) != len(right_pairs):
        return 0.0
    left_terms = expand_legacy_terms(left_pairs)
    right_terms = expand_legacy_terms(right_pairs)
    total = 0.0
    for (left_alpha, left_beta), left_coeff in left_terms.items():
        for (right_alpha, right_beta), right_coeff in right_terms.items():
            alpha_block = S[np.ix_(left_alpha, right_alpha)]
            beta_block = S[np.ix_(left_beta, right_beta)]
            total += left_coeff * right_coeff * np.linalg.det(alpha_block) * np.linalg.det(beta_block)
    return total


def zero_exp(n_vars):
    return (0,) * n_vars


def monomial(n_vars, exponent, coefficient):
    if abs(coefficient) <= 1.0e-12:
        return {}
    return {exponent: float(coefficient)}


def add_poly(a, b):
    out = dict(a)
    for exponent, coefficient in b.items():
        out[exponent] = out.get(exponent, 0.0) + coefficient
        if abs(out[exponent]) <= 1.0e-12:
            del out[exponent]
    return out


def mul_poly(a, b):
    if not a or not b:
        return {}
    out = {}
    for exp_a, coeff_a in a.items():
        for exp_b, coeff_b in b.items():
            exponent = tuple(x + y for x, y in zip(exp_a, exp_b))
            out[exponent] = out.get(exponent, 0.0) + coeff_a * coeff_b
    return {key: value for key, value in out.items() if abs(value) > 1.0e-12}


def permutation_sign(perm):
    sign = 1.0
    for i in range(len(perm) - 1):
        for j in range(i + 1, len(perm)):
            if perm[i] > perm[j]:
                sign = -sign
    return sign


def determinant_poly(entries):
    n_rows = len(entries)
    n_vars = len(next(iter(entries[0][0].keys()), ()))
    total = {}
    # Verification-only Leibniz expansion of an n x n polynomial matrix.
    for perm in itertools.permutations(range(n_rows)):
        term = monomial(n_vars, zero_exp(n_vars), permutation_sign(perm))
        for row, column in enumerate(perm):
            term = mul_poly(term, entries[row][column])
            if not term:
                break
        total = add_poly(total, term)
    return total


def selector_choices(pair, variable_index, n_vars, is_alpha):
    i, j = pair
    if i == j:
        return [(i, zero_exp(n_vars))]
    if variable_index is None:
        raise ValueError("covalent pair requires a variable index")
    exponent = [0] * n_vars
    if is_alpha:
        exponent[variable_index] = 1
        return [(i, zero_exp(n_vars)), (j, tuple(exponent))]
    exponent[variable_index] = -1
    return [(j, zero_exp(n_vars)), (i, tuple(exponent))]


def entry_poly(left_pair, right_pair, left_var, right_var, n_vars, S, is_alpha):
    left_choices = selector_choices(left_pair, left_var, n_vars, is_alpha)
    right_choices = selector_choices(right_pair, right_var, n_vars, is_alpha)
    poly = {}
    for left_orbital, left_exp in left_choices:
        for right_orbital, right_exp in right_choices:
            exponent = tuple(x + y for x, y in zip(left_exp, right_exp))
            poly = add_poly(poly, monomial(n_vars, exponent, S[left_orbital, right_orbital]))
    return poly


def overlap_by_generating_function(left_pairs, right_pairs, S):
    if len(left_pairs) != len(right_pairs):
        return 0.0
    left_cov_count = sum(i != j for i, j in left_pairs)
    right_cov_count = sum(i != j for i, j in right_pairs)
    n_vars = left_cov_count + right_cov_count
    left_var_indices = []
    right_var_indices = []
    offset = 0
    for i, j in left_pairs:
        left_var_indices.append(None if i == j else offset)
        if i != j:
            offset += 1
    for i, j in right_pairs:
        right_var_indices.append(None if i == j else offset)
        if i != j:
            offset += 1
    alpha_entries = []
    beta_entries = []
    for row, left_pair in enumerate(left_pairs):
        alpha_row = []
        beta_row = []
        for column, right_pair in enumerate(right_pairs):
            # Each entry is a Laurent polynomial in the bond labels of the left
            # and right covalent pairs. Ionic pairs contribute fixed selectors.
            alpha_row.append(entry_poly(left_pair, right_pair, left_var_indices[row], right_var_indices[column], n_vars, S, True))
            beta_row.append(entry_poly(left_pair, right_pair, left_var_indices[row], right_var_indices[column], n_vars, S, False))
        alpha_entries.append(alpha_row)
        beta_entries.append(beta_row)
    alpha_poly = determinant_poly(alpha_entries)
    beta_poly = determinant_poly(beta_entries)
    total_poly = mul_poly(alpha_poly, beta_poly)
    return total_poly.get(zero_exp(n_vars), 0.0)


def build_signed_selector(pairs, signs, is_alpha):
    rows = []
    sign_index = 0
    n_orbitals = max(max(i, j) for i, j in pairs) + 1 if pairs else 0
    for i, j in pairs:
        row = np.zeros(n_orbitals, dtype=float)
        if i == j:
            row[i] = 1.0
        elif is_alpha:
            row[i] = 1.0
            row[j] = signs[sign_index]
            sign_index += 1
        else:
            row[j] = 1.0
            row[i] = signs[sign_index]
            sign_index += 1
        rows.append(row)
    return np.vstack(rows) if rows else np.zeros((0, n_orbitals), dtype=float)


def overlap_by_hypercube_average(left_pairs, right_pairs, S):
    if len(left_pairs) != len(right_pairs):
        return 0.0
    left_cov_count = sum(i != j for i, j in left_pairs)
    right_cov_count = sum(i != j for i, j in right_pairs)
    total = 0.0
    for left_signs in itertools.product((-1.0, 1.0), repeat=left_cov_count):
        left_alpha = build_signed_selector(left_pairs, left_signs, True)
        left_beta = build_signed_selector(left_pairs, left_signs, False)
        for right_signs in itertools.product((-1.0, 1.0), repeat=right_cov_count):
            right_alpha = build_signed_selector(right_pairs, right_signs, True)
            right_beta = build_signed_selector(right_pairs, right_signs, False)
            total += (
                np.linalg.det(left_alpha @ S @ right_alpha.T)
                * np.linalg.det(left_beta @ S @ right_beta.T)
            )
    return total / (2.0 ** (left_cov_count + right_cov_count))


def main():
    rng = np.random.default_rng(7)
    base = rng.normal(size=(5, 5))
    S_full = np.eye(5) + 0.15 * (base + base.T)
    cases = [
        ("1-2 3-4", "1-2 3-4"),
        ("1-2 3-4", "1-3 2-4"),
        ("1-1 2-3", "1-2 3-3"),
        ("1-1 2-2", "1-1 2-2"),
    ]
    tolerance = 1.0e-10
    max_abs_diff = 0.0
    for left_text, right_text in cases:
        left_pairs = parse_pairs(left_text)
        right_pairs = parse_pairs(right_text)
        n_orbitals = max(max(i, j) for i, j in left_pairs + right_pairs) + 1
        S = S_full[:n_orbitals, :n_orbitals]
        overlap_det = det_overlap_legacy(left_pairs, right_pairs, S)
        overlap_gen = overlap_by_generating_function(left_pairs, right_pairs, S)
        overlap_avg = overlap_by_hypercube_average(left_pairs, right_pairs, S)
        diff_gen = abs(overlap_det - overlap_gen)
        diff_avg = abs(overlap_det - overlap_avg)
        max_abs_diff = max(max_abs_diff, diff_gen, diff_avg)
        print(
            f"{left_text:>11s} | {right_text:<11s} "
            f"det = {overlap_det:+.12f}  poly = {overlap_gen:+.12f}  "
            f"avg = {overlap_avg:+.12f}  "
            f"diff_poly = {diff_gen:.3e}  diff_avg = {diff_avg:.3e}"
        )
    print(f"max_abs_diff = {max_abs_diff:.3e}")
    if max_abs_diff > tolerance:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
