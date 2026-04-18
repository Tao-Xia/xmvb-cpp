#!/usr/bin/env python3
"""
Show how same-spin deduplication works for closed-shell active spaces.

For a singlet closed-shell active space with N electrons in N orbitals:

  - alpha strings contain N/2 occupied orbitals
  - beta strings contain N/2 occupied orbitals
  - full determinants are the Cartesian product alpha x beta

Same-spin deduplication does not cache full determinant pairs. Instead, it
deduplicates the repeated alpha and beta strings that appear across many full
determinants and then reuses the unique alpha-pair / beta-pair kernels.
"""

from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CaseSummary:
    n_electrons: int
    n_orbitals: int
    n_alpha: int
    n_beta: int
    unique_alpha_strings: int
    unique_beta_strings: int
    unique_alpha_pairs: int
    unique_beta_pairs: int
    full_determinants: int
    total_unique_spin_strings: int
    same_spin_direct_kernel_count: int
    same_spin_cached_kernel_count: int
    reuse_factor: float


def enumerate_spin_strings(n_orbitals: int, n_spin_electrons: int) -> list[tuple[int, ...]]:
    return list(itertools.combinations(range(n_orbitals), n_spin_electrons))


def summarize_case(n_electrons: int, n_orbitals: int) -> CaseSummary:
    if n_electrons != n_orbitals:
        raise ValueError("this demo expects N electrons in N orbitals")
    if n_electrons % 2 != 0:
        raise ValueError("this demo expects a closed-shell even-electron singlet")

    n_alpha = n_electrons // 2
    n_beta = n_electrons // 2

    alpha_strings = enumerate_spin_strings(n_orbitals, n_alpha)
    beta_strings = enumerate_spin_strings(n_orbitals, n_beta)

    full_determinants = len(alpha_strings) * len(beta_strings)

    # Without dedup, each unordered full-determinant pair needs one alpha
    # same-spin kernel and one beta same-spin kernel.
    same_spin_direct_kernel_count = full_determinants * (full_determinants + 1)

    # With dedup, we only need all ordered unique alpha pairs and beta pairs.
    same_spin_cached_kernel_count = (
        len(alpha_strings) * len(alpha_strings) +
        len(beta_strings) * len(beta_strings)
    )

    return CaseSummary(
        n_electrons=n_electrons,
        n_orbitals=n_orbitals,
        n_alpha=n_alpha,
        n_beta=n_beta,
        unique_alpha_strings=len(alpha_strings),
        unique_beta_strings=len(beta_strings),
        unique_alpha_pairs=len(alpha_strings) * len(alpha_strings),
        unique_beta_pairs=len(beta_strings) * len(beta_strings),
        full_determinants=full_determinants,
        total_unique_spin_strings=len(alpha_strings) + len(beta_strings),
        same_spin_direct_kernel_count=same_spin_direct_kernel_count,
        same_spin_cached_kernel_count=same_spin_cached_kernel_count,
        reuse_factor=same_spin_direct_kernel_count / same_spin_cached_kernel_count,
    )


def format_spin_string(spin_string: tuple[int, ...]) -> str:
    return "(" + ",".join(str(index + 1) for index in spin_string) + ")"


def print_example(n_electrons: int, n_orbitals: int, max_examples: int) -> None:
    n_spin = n_electrons // 2
    alpha_strings = enumerate_spin_strings(n_orbitals, n_spin)
    beta_strings = enumerate_spin_strings(n_orbitals, n_spin)

    repeated_alpha = alpha_strings[0]
    print(f"Example for {n_electrons}e{n_orbitals}o")
    print(
        "  One alpha spin-string "
        f"{format_spin_string(repeated_alpha)} appears in "
        f"{len(beta_strings)} different full determinants:"
    )
    for beta_string in beta_strings[:max_examples]:
        print(
            "    D = (alpha="
            f"{format_spin_string(repeated_alpha)}, beta={format_spin_string(beta_string)})"
        )
    if len(beta_strings) > max_examples:
        print(f"    ... {len(beta_strings) - max_examples} more determinants share the same alpha")

    repeated_beta = beta_strings[0]
    print(
        "  Symmetrically, one beta spin-string "
        f"{format_spin_string(repeated_beta)} appears in "
        f"{len(alpha_strings)} different full determinants."
    )
    print()


def print_table(summaries: list[CaseSummary]) -> None:
    header = (
        f"{'case':>8}  {'alpha_str':>10}  {'beta_str':>9}  "
        f"{'full_det':>10}  {'alpha_pairs':>12}  {'beta_pairs':>11}  "
        f"{'direct_same_spin':>18}  {'cached_same_spin':>18}  {'reuse':>10}"
    )
    print(header)
    print("-" * len(header))
    for summary in summaries:
        case_label = f"{summary.n_electrons}e{summary.n_orbitals}o"
        print(
            f"{case_label:>8}  "
            f"{summary.unique_alpha_strings:10d}  "
            f"{summary.unique_beta_strings:9d}  "
            f"{summary.full_determinants:10d}  "
            f"{summary.unique_alpha_pairs:12d}  "
            f"{summary.unique_beta_pairs:11d}  "
            f"{summary.same_spin_direct_kernel_count:18d}  "
            f"{summary.same_spin_cached_kernel_count:18d}  "
            f"{summary.reuse_factor:10.1f}"
        )


def print_explanation(summaries: list[CaseSummary]) -> None:
    print()
    print("Interpretation")
    for summary in summaries:
        case_label = f"{summary.n_electrons}e{summary.n_orbitals}o"
        print(
            f"  {case_label}: "
            f"full determinants = alpha_str x beta_str = "
            f"{summary.unique_alpha_strings} x {summary.unique_beta_strings} = "
            f"{summary.full_determinants}"
        )
        print(
            f"           same-spin cached kernels = alpha_pairs + beta_pairs = "
            f"{summary.unique_alpha_pairs} + {summary.unique_beta_pairs} = "
            f"{summary.same_spin_cached_kernel_count}"
        )
        print(
            f"           unique spin strings are "
            f"{summary.unique_alpha_strings} alpha + {summary.unique_beta_strings} beta = "
            f"{summary.total_unique_spin_strings}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Count unique alpha/beta spin strings and same-spin kernel reuse "
            "for closed-shell Ne=No active spaces."
        )
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        type=int,
        default=[6, 8, 10, 12],
        help="Active-space sizes N in Ne=No. Default: 6 8 10 12",
    )
    parser.add_argument(
        "--show-example",
        action="store_true",
        help="Print a small explicit determinant example showing how repetition arises.",
    )
    parser.add_argument(
        "--example-case",
        type=int,
        default=6,
        help="Which Ne=No case to use for the explicit example. Default: 6",
    )
    parser.add_argument(
        "--max-example-dets",
        type=int,
        default=5,
        help="How many repeated determinants to print in the explicit example. Default: 5",
    )
    args = parser.parse_args()

    summaries = [summarize_case(n_electrons=n, n_orbitals=n) for n in args.cases]

    if args.show_example:
        print_example(
            n_electrons=args.example_case,
            n_orbitals=args.example_case,
            max_examples=args.max_example_dets,
        )

    print_table(summaries)
    print_explanation(summaries)


if __name__ == "__main__":
    main()
