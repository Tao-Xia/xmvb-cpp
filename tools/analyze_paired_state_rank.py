#!/usr/bin/env python3
"""Diagnose the compressibility of one nonorthogonal paired VB structure.

This is a deliberately small, exact kill test for the proposed large-active-
space route.  It constructs one covalent structure containing adjacent singlet
pairs, applies the exterior power of a Cholesky overlap factor, and measures
the Schmidt spectrum across every spatial-orbital cut.  No Hamiltonian, tensor-
network library, or production VBSCF code is involved.

The explicit fixed-spin carrier has size ``binomial(K, K/2)**2``.  Consequently
this diagnostic is intended for K <= 14; its purpose is to reveal rank growth,
not to perform the eventual K = 100 calculation.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import scipy.linalg


@dataclass(frozen=True)
class PairTerm:
    """One alpha/beta determinant product in a singlet-pair expansion."""

    alpha_mask: int
    beta_mask: int
    coefficient: float


@dataclass(frozen=True)
class CutSpectrum:
    """Schmidt information for one spatial-orbital bipartition."""

    cut: int
    exact_rank: int
    effective_ranks: tuple[int, ...]
    entropy: float
    largest_singular_value: float
    smallest_retained_singular_value: float


def bit_count(value: int) -> int:
    """Return the number of occupied bits in ``value``."""

    return int(value.bit_count())


def inversion_parity(indices: Sequence[int]) -> int:
    """Return +1 or -1 for the permutation that sorts ``indices``."""

    inversions = 0
    for right in range(1, len(indices)):
        for left in range(right):
            inversions += indices[left] > indices[right]
    return -1 if inversions % 2 else 1


def adjacent_singlet_terms(n_orbitals: int) -> list[PairTerm]:
    """Expand adjacent singlet pairs only for the small exact reference.

    Spin-orbitals are canonically ordered with every alpha orbital preceding
    every beta orbital.  Fermionic reorder signs are therefore included here,
    rather than inferred from the apparent product of local pair signs.
    """

    if n_orbitals <= 0 or n_orbitals % 2:
        raise ValueError("the paired diagnostic requires a positive even K")
    n_pairs = n_orbitals // 2
    normalization = 2.0 ** (-0.5 * n_pairs)
    terms: list[PairTerm] = []
    for choices in itertools.product((0, 1), repeat=n_pairs):
        alpha_mask = 0
        beta_mask = 0
        operator_order: list[int] = []
        local_sign = 1
        for pair, choice in enumerate(choices):
            first = 2 * pair
            second = first + 1
            if choice == 0:
                # b(first,alpha)^+ b(second,beta)^+
                alpha_mask |= 1 << first
                beta_mask |= 1 << second
                operator_order.extend((first, n_orbitals + second))
            else:
                # -b(first,beta)^+ b(second,alpha)^+
                alpha_mask |= 1 << second
                beta_mask |= 1 << first
                operator_order.extend((n_orbitals + first, second))
                local_sign *= -1
        terms.append(
            PairTerm(
                alpha_mask,
                beta_mask,
                normalization * local_sign * inversion_parity(operator_order),
            )
        )
    return terms


def fixed_weight_masks(n_orbitals: int, n_electrons: int) -> list[int]:
    """Enumerate fixed-particle bit strings in lexical occupation order."""

    masks: list[int] = []
    for occupied in itertools.combinations(range(n_orbitals), n_electrons):
        mask = 0
        for orbital in occupied:
            mask |= 1 << orbital
        masks.append(mask)
    return masks


def replacement_sign(source: int, removed: int, inserted: int) -> float:
    """Fermionic sign of replacing one occupied spatial orbital."""

    low = min(removed, inserted)
    high = max(removed, inserted)
    if high - low <= 1:
        return 1.0
    between = source & (((1 << high) - 1) ^ ((1 << (low + 1)) - 1))
    return 1.0 if bit_count(between) % 2 == 0 else -1.0


def build_exterior_shears(
    masks: Sequence[int], upper_transform: np.ndarray
) -> tuple[np.ndarray, list[tuple[float, list[tuple[int, int, float]]]]]:
    """Build the same triangular exterior transform used by xmvb-cpp."""

    n_orbitals = upper_transform.shape[0]
    if upper_transform.shape != (n_orbitals, n_orbitals):
        raise ValueError("orbital transform must be square")
    if np.max(np.abs(np.tril(upper_transform, -1)), initial=0.0) > 1.0e-12:
        raise ValueError("orbital transform must be upper triangular")
    if np.any(np.diag(upper_transform) <= 0.0):
        raise ValueError("orbital transform must have a positive diagonal")

    mask_index = {mask: index for index, mask in enumerate(masks)}
    scales = np.ones(len(masks))
    diagonal = np.diag(upper_transform)
    for index, mask in enumerate(masks):
        for orbital in range(n_orbitals):
            if mask & (1 << orbital):
                scales[index] *= diagonal[orbital]

    unit_upper = np.array(upper_transform, dtype=float, copy=True)
    unit_upper /= diagonal[np.newaxis, :]
    operations: list[tuple[int, int, float]] = []
    for removed in range(1, n_orbitals):
        for inserted in range(removed - 1, -1, -1):
            coefficient = float(unit_upper[inserted, removed])
            operations.append((inserted, removed, coefficient))
            unit_upper[:, removed] -= coefficient * unit_upper[:, inserted]

    shears: list[tuple[float, list[tuple[int, int, float]]]] = []
    for inserted, removed, coefficient in operations:
        pairs: list[tuple[int, int, float]] = []
        removed_bit = 1 << removed
        inserted_bit = 1 << inserted
        for source_index, source_mask in enumerate(masks):
            if not source_mask & removed_bit or source_mask & inserted_bit:
                continue
            target_mask = (source_mask ^ removed_bit) | inserted_bit
            pairs.append(
                (
                    source_index,
                    mask_index[target_mask],
                    replacement_sign(source_mask, removed, inserted),
                )
            )
        shears.append((coefficient, pairs))
    return scales, shears


def apply_exterior_left(
    coefficients: np.ndarray,
    scales: np.ndarray,
    shears: Sequence[tuple[float, Sequence[tuple[int, int, float]]]],
) -> None:
    """Apply an exterior transform to coefficient-matrix rows in place."""

    coefficients *= scales[:, np.newaxis]
    for coefficient, pairs in shears:
        if coefficient == 0.0:
            continue
        for source, target, sign in pairs:
            coefficients[target, :] += coefficient * sign * coefficients[source, :]


def apply_exterior_right(
    coefficients: np.ndarray,
    scales: np.ndarray,
    shears: Sequence[tuple[float, Sequence[tuple[int, int, float]]]],
) -> None:
    """Apply an exterior transform to coefficient-matrix columns in place."""

    coefficients *= scales[np.newaxis, :]
    for coefficient, pairs in shears:
        if coefficient == 0.0:
            continue
        for source, target, sign in pairs:
            coefficients[:, target] += coefficient * sign * coefficients[:, source]


def transformed_pair_coefficients(
    overlap: np.ndarray,
) -> tuple[np.ndarray, list[int], list[PairTerm]]:
    """Return the exact orthonormal-carrier coefficients of one pair state."""

    n_orbitals = overlap.shape[0]
    n_alpha = n_orbitals // 2
    masks = fixed_weight_masks(n_orbitals, n_alpha)
    mask_index = {mask: index for index, mask in enumerate(masks)}
    terms = adjacent_singlet_terms(n_orbitals)
    coefficients = np.zeros((len(masks), len(masks)))
    for term in terms:
        coefficients[
            mask_index[term.alpha_mask], mask_index[term.beta_mask]
        ] += term.coefficient

    # numpy returns S = L L^T.  R = L^T gives S = R^T R and C = Q R.
    upper_transform = np.linalg.cholesky(overlap).T
    scales, shears = build_exterior_shears(masks, upper_transform)
    apply_exterior_left(coefficients, scales, shears)
    apply_exterior_right(coefficients, scales, shears)
    return coefficients, masks, terms


def direct_nonorthogonal_norm(
    overlap: np.ndarray,
    terms: Sequence[PairTerm],
) -> float:
    """Evaluate the pair-state norm by the explicit small determinant sum."""

    n_orbitals = overlap.shape[0]
    occupied_cache: dict[int, list[int]] = {}

    def occupied(mask: int) -> list[int]:
        if mask not in occupied_cache:
            occupied_cache[mask] = [
                orbital
                for orbital in range(n_orbitals)
                if mask & (1 << orbital)
            ]
        return occupied_cache[mask]

    result = 0.0
    for left in terms:
        left_alpha = occupied(left.alpha_mask)
        left_beta = occupied(left.beta_mask)
        for right in terms:
            right_alpha = occupied(right.alpha_mask)
            right_beta = occupied(right.beta_mask)
            alpha_overlap = np.linalg.det(
                overlap[np.ix_(left_alpha, right_alpha)]
            )
            beta_overlap = np.linalg.det(
                overlap[np.ix_(left_beta, right_beta)]
            )
            result += (
                left.coefficient
                * right.coefficient
                * alpha_overlap
                * beta_overlap
            )
    return float(result)


def alpha_beta_to_site_sign(alpha_mask: int, beta_mask: int) -> float:
    """Convert all-alpha-then-beta ordering to spatial-site Fock ordering."""

    crossings = 0
    remaining_alpha = alpha_mask
    while remaining_alpha:
        alpha = (remaining_alpha & -remaining_alpha).bit_length() - 1
        crossings += bit_count(beta_mask & ((1 << alpha) - 1))
        remaining_alpha &= remaining_alpha - 1
    return -1.0 if crossings % 2 else 1.0


def split_mask_ids(
    masks: Sequence[int], cut: int, n_orbitals: int
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Group fixed-weight strings by the particle number left of a cut."""

    left_mask = (1 << cut) - 1
    grouped: dict[int, list[tuple[int, int, int]]] = {}
    for index, mask in enumerate(masks):
        left = mask & left_mask
        right = mask >> cut
        grouped.setdefault(bit_count(left), []).append((index, left, right))

    result: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for particle_count, values in grouped.items():
        left_values = sorted({left for _, left, _ in values})
        right_values = sorted({right for _, _, right in values})
        left_index = {mask: index for index, mask in enumerate(left_values)}
        right_index = {mask: index for index, mask in enumerate(right_values)}
        determinant_indices = np.asarray([value[0] for value in values], dtype=int)
        left_ids = np.asarray([left_index[value[1]] for value in values], dtype=int)
        right_ids = np.asarray([right_index[value[2]] for value in values], dtype=int)
        result[particle_count] = determinant_indices, left_ids, right_ids
    return result


def schmidt_singular_values(
    site_coefficients: np.ndarray,
    masks: Sequence[int],
    cut: int,
    n_orbitals: int,
) -> np.ndarray:
    """Compute singular values in particle-number-resolved cut blocks."""

    alpha_groups = split_mask_ids(masks, cut, n_orbitals)
    beta_groups = split_mask_ids(masks, cut, n_orbitals)
    singular_values: list[np.ndarray] = []

    for alpha_count, (alpha_indices, alpha_left, alpha_right) in alpha_groups.items():
        del alpha_count
        n_alpha_left = int(alpha_left.max(initial=-1)) + 1
        n_alpha_right = int(alpha_right.max(initial=-1)) + 1
        for beta_indices, beta_left, beta_right in beta_groups.values():
            n_beta_left = int(beta_left.max(initial=-1)) + 1
            n_beta_right = int(beta_right.max(initial=-1)) + 1
            block = np.zeros(
                (n_alpha_left * n_beta_left, n_alpha_right * n_beta_right)
            )
            rows = (
                alpha_left[:, np.newaxis] * n_beta_left
                + beta_left[np.newaxis, :]
            )
            columns = (
                alpha_right[:, np.newaxis] * n_beta_right
                + beta_right[np.newaxis, :]
            )
            block[rows.ravel(), columns.ravel()] = site_coefficients[
                np.ix_(alpha_indices, beta_indices)
            ].ravel()
            if block.size:
                singular_values.append(scipy.linalg.svdvals(block))
    if not singular_values:
        return np.empty(0)
    return np.sort(np.concatenate(singular_values))[::-1]


def summarize_cut(
    singular_values: np.ndarray,
    cut: int,
    discarded_weight_tolerances: Sequence[float],
) -> CutSpectrum:
    """Convert a singular spectrum into exact and tolerance-dependent ranks."""

    squared = singular_values * singular_values
    norm = float(np.sum(squared))
    if norm <= 0.0:
        raise ValueError("zero norm encountered in Schmidt analysis")
    probabilities = squared / norm
    numerical_threshold = np.finfo(float).eps * max(1, singular_values.size) * singular_values[0]
    exact_rank = int(np.count_nonzero(singular_values > numerical_threshold))
    cumulative = np.cumsum(probabilities)
    effective_ranks = tuple(
        int(np.searchsorted(cumulative, 1.0 - tolerance, side="left") + 1)
        for tolerance in discarded_weight_tolerances
    )
    positive = probabilities[probabilities > 0.0]
    entropy = float(-np.sum(positive * np.log(positive)))
    return CutSpectrum(
        cut=cut,
        exact_rank=exact_rank,
        effective_ranks=effective_ranks,
        entropy=entropy,
        largest_singular_value=float(singular_values[0]),
        smallest_retained_singular_value=float(
            singular_values[effective_ranks[-1] - 1]
        ),
    )


def toeplitz_overlap(n_orbitals: int, decay: float) -> np.ndarray:
    """Return the positive-definite local model S_ij = decay**|i-j|."""

    indices = np.arange(n_orbitals)
    return decay ** np.abs(indices[:, np.newaxis] - indices[np.newaxis, :])


def dense_overlap_from_spectrum(local_overlap: np.ndarray, seed: int) -> np.ndarray:
    """Make a dense overlap with the same eigenvalues as a local model."""

    eigenvalues = np.linalg.eigvalsh(local_overlap)
    generator = np.random.default_rng(seed)
    orthogonal, _ = np.linalg.qr(generator.standard_normal(local_overlap.shape))
    result = (orthogonal * eigenvalues[np.newaxis, :]) @ orthogonal.T
    return 0.5 * (result + result.T)


def hydrogen_chain_overlap(n_orbitals: int, spacing_bohr: float) -> np.ndarray:
    """Build the STO-3G AO overlap of an equally spaced hydrogen chain."""

    try:
        from pyscf import gto
    except ImportError as error:
        raise RuntimeError("the hchain model requires PySCF") from error
    molecule = gto.M(
        atom=[("H", (spacing_bohr * atom, 0.0, 0.0)) for atom in range(n_orbitals)],
        basis="sto-3g",
        unit="Bohr",
        spin=0,
        verbose=0,
    )
    overlap = molecule.intor_symmetric("int1e_ovlp")
    if overlap.shape != (n_orbitals, n_orbitals):
        raise RuntimeError("H/STO-3G did not produce one AO per active orbital")
    return overlap


def overlap_model(
    name: str,
    n_orbitals: int,
    decay: float,
    spacing_bohr: float,
    seed: int,
) -> np.ndarray:
    """Construct one overlap model requested on the command line."""

    if name == "identity":
        return np.eye(n_orbitals)
    local = toeplitz_overlap(n_orbitals, decay)
    if name == "local":
        return local
    if name == "dense":
        return dense_overlap_from_spectrum(local, seed + n_orbitals)
    if name == "hchain":
        return hydrogen_chain_overlap(n_orbitals, spacing_bohr)
    raise ValueError(f"unknown overlap model: {name}")


def analyze_case(
    model: str,
    n_orbitals: int,
    tolerances: Sequence[float],
    decay: float,
    spacing_bohr: float,
    seed: int,
) -> tuple[float, float, list[CutSpectrum]]:
    """Run one exact rank diagnostic and its independent norm check."""

    overlap = overlap_model(model, n_orbitals, decay, spacing_bohr, seed)
    eigenvalues = np.linalg.eigvalsh(overlap)
    if eigenvalues[0] <= 0.0:
        raise ValueError(f"{model} overlap is not positive definite")
    coefficients, masks, terms = transformed_pair_coefficients(overlap)
    transformed_norm = float(np.vdot(coefficients, coefficients).real)
    reference_norm = direct_nonorthogonal_norm(overlap, terms)
    relative_norm_error = abs(transformed_norm - reference_norm) / max(
        1.0, abs(reference_norm)
    )
    site_sign = np.empty_like(coefficients)
    for alpha_index, alpha_mask in enumerate(masks):
        for beta_index, beta_mask in enumerate(masks):
            site_sign[alpha_index, beta_index] = alpha_beta_to_site_sign(
                alpha_mask, beta_mask
            )
    site_coefficients = coefficients * site_sign
    spectra = [
        summarize_cut(
            schmidt_singular_values(site_coefficients, masks, cut, n_orbitals),
            cut,
            tolerances,
        )
        for cut in range(1, n_orbitals)
    ]
    return float(eigenvalues[-1] / eigenvalues[0]), relative_norm_error, spectra


def write_rows(
    output: Path,
    rows: Iterable[dict[str, object]],
    tolerances: Sequence[float],
) -> None:
    """Write machine-readable cut diagnostics."""

    fields = [
        "model",
        "n_orbitals",
        "cut",
        "condition_number",
        "norm_relative_error",
        "exact_rank",
        *[f"rank_dw_{tolerance:.0e}" for tolerance in tolerances],
        "entropy",
        "largest_singular_value",
        "smallest_retained_singular_value",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_self_test() -> None:
    """Verify pair signs, exterior action, norm, and orthogonal cut ranks."""

    tolerances = (1.0e-12,)
    for n_orbitals in (4, 6, 8):
        condition, norm_error, spectra = analyze_case(
            "identity", n_orbitals, tolerances, 0.35, 2.0, 7
        )
        if abs(condition - 1.0) > 1.0e-14 or norm_error > 1.0e-13:
            raise AssertionError("identity-model normalization failed")
        for spectrum in spectra:
            expected_rank = 2 if spectrum.cut % 2 else 1
            if spectrum.exact_rank != expected_rank:
                raise AssertionError(
                    f"K={n_orbitals} cut={spectrum.cut}: "
                    f"rank {spectrum.exact_rank}, expected {expected_rank}"
                )
    _, norm_error, _ = analyze_case(
        "local", 8, tolerances, 0.35, 2.0, 7
    )
    if norm_error > 2.0e-12:
        raise AssertionError("nonorthogonal exterior-transform norm check failed")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[6, 8, 10, 12],
        help="even active-orbital counts for the exact diagnostic",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("identity", "local", "dense", "hchain"),
        default=["identity", "local", "dense", "hchain"],
        help="overlap models to compare",
    )
    parser.add_argument(
        "--tolerances",
        nargs="+",
        type=float,
        default=[1.0e-8, 1.0e-10, 1.0e-12],
        help="discarded-weight tolerances used for effective ranks",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=0.35,
        help="nearest-neighbor scale in the local Toeplitz overlap",
    )
    parser.add_argument(
        "--spacing-bohr",
        type=float,
        default=2.0,
        help="H-H separation for the physical H/STO-3G overlap model",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/diagnostics/paired_state_rank.csv"),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run internal exact checks before the requested cases",
    )
    return parser.parse_args()


def main() -> None:
    """Run requested rank diagnostics and print compact worst-cut summaries."""

    arguments = parse_arguments()
    if arguments.self_test:
        run_self_test()
        print("paired-state rank self-test: passed")

    for size in arguments.sizes:
        if size <= 0 or size % 2:
            raise ValueError(f"invalid paired active-space size: {size}")
        carrier_dimension = math.comb(size, size // 2)
        carrier_bytes = carrier_dimension * carrier_dimension * 8
        if carrier_bytes > 2_000_000_000:
            raise ValueError(
                f"K={size} needs {carrier_bytes / 2**30:.2f} GiB for the "
                "exact diagnostic; use K <= 14"
            )
    tolerances = tuple(sorted(arguments.tolerances, reverse=True))
    if any(tolerance <= 0.0 or tolerance >= 1.0 for tolerance in tolerances):
        raise ValueError("discarded-weight tolerances must lie between zero and one")

    rows: list[dict[str, object]] = []
    for model in arguments.models:
        for n_orbitals in arguments.sizes:
            condition, norm_error, spectra = analyze_case(
                model,
                n_orbitals,
                tolerances,
                arguments.decay,
                arguments.spacing_bohr,
                arguments.seed,
            )
            worst = max(
                spectra,
                key=lambda spectrum: spectrum.effective_ranks[-1],
            )
            ranks = ", ".join(
                f"D({tolerance:.0e})={rank}"
                for tolerance, rank in zip(tolerances, worst.effective_ranks)
            )
            print(
                f"{model:8s} K={n_orbitals:2d} cond(S)={condition:.3e} "
                f"norm_err={norm_error:.3e} worst_cut={worst.cut:2d} "
                f"D_exact={worst.exact_rank:4d} {ranks}",
                flush=True,
            )
            for spectrum in spectra:
                row: dict[str, object] = {
                    "model": model,
                    "n_orbitals": n_orbitals,
                    "cut": spectrum.cut,
                    "condition_number": condition,
                    "norm_relative_error": norm_error,
                    "exact_rank": spectrum.exact_rank,
                    "entropy": spectrum.entropy,
                    "largest_singular_value": spectrum.largest_singular_value,
                    "smallest_retained_singular_value": (
                        spectrum.smallest_retained_singular_value
                    ),
                }
                for tolerance, rank in zip(tolerances, spectrum.effective_ranks):
                    row[f"rank_dw_{tolerance:.0e}"] = rank
                rows.append(row)
    write_rows(arguments.output, rows, tolerances)
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
