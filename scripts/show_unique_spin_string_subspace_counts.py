#!/usr/bin/env python3
"""
Generate closed-shell active-space subspace statistics for PPT tables.

This extends the earlier unique-spin-string counting script to the cumulative
raw-structure subspaces used by XMVB:

  - cov   : covalent structures only
  - 0-k   : cumulative subspace containing ion classes 0..k
  - full  : all ion classes

For a closed-shell Ne=No=2p active space, the exact-k ion-class counts follow
the legacy formulas in `src/runtime_c/local_runtime/genstr.c`:

  structures_exact(k) = C(2p, k) C(2p-k, k) Catalan(p-k)
  unique_det_exact(k) = C(2p, k) C(2p-k, k) C(2p-2k, p-k)

The expanded determinant-term count before deduplication is

  expanded_terms_exact(k) = structures_exact(k) * 2^(p-k),

because each of the p-k covalent pairs contributes two spin orientations while
each ionic pair is orientation-fixed.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SubspaceRow:
    label: str
    max_ionic_pairs: int
    structure_count: int
    expanded_determinant_terms: int
    unique_determinants: int
    unique_alpha_strings: int
    unique_beta_strings: int
    determinant_redundancy: float
    same_spin_reuse: float


def catalan(number_of_pairs: int) -> int:
    return math.comb(2 * number_of_pairs, number_of_pairs) // (number_of_pairs + 1)


def exact_ion_class_structure_count(n_active_orbitals: int, ionic_pairs: int) -> int:
    n_pairs = n_active_orbitals // 2
    return (
        math.comb(n_active_orbitals, ionic_pairs)
        * math.comb(n_active_orbitals - ionic_pairs, ionic_pairs)
        * catalan(n_pairs - ionic_pairs)
    )


def exact_ion_class_expanded_terms(n_active_orbitals: int, ionic_pairs: int) -> int:
    n_pairs = n_active_orbitals // 2
    return exact_ion_class_structure_count(n_active_orbitals, ionic_pairs) * (2 ** (n_pairs - ionic_pairs))


def exact_ion_class_unique_determinants(n_active_orbitals: int, ionic_pairs: int) -> int:
    n_pairs = n_active_orbitals // 2
    return (
        math.comb(n_active_orbitals, ionic_pairs)
        * math.comb(n_active_orbitals - ionic_pairs, ionic_pairs)
        * math.comb(n_active_orbitals - 2 * ionic_pairs, n_pairs - ionic_pairs)
    )


def subspace_label(max_ionic_pairs: int, n_pairs: int) -> str:
    if max_ionic_pairs == 0:
        return "cov"
    if max_ionic_pairs == n_pairs:
        return "full"
    return f"0-{max_ionic_pairs}"


def summarize_subspaces(n_active_orbitals: int) -> list[SubspaceRow]:
    if n_active_orbitals % 2 != 0:
        raise ValueError("closed-shell statistics require an even active-space size")

    n_pairs = n_active_orbitals // 2
    unique_spin_strings = math.comb(n_active_orbitals, n_pairs)
    rows: list[SubspaceRow] = []

    cumulative_structures = 0
    cumulative_expanded_terms = 0
    cumulative_unique_determinants = 0
    for ionic_pairs in range(n_pairs + 1):
        cumulative_structures += exact_ion_class_structure_count(n_active_orbitals, ionic_pairs)
        cumulative_expanded_terms += exact_ion_class_expanded_terms(n_active_orbitals, ionic_pairs)
        cumulative_unique_determinants += exact_ion_class_unique_determinants(
            n_active_orbitals,
            ionic_pairs,
        )
        rows.append(
            SubspaceRow(
                label=subspace_label(ionic_pairs, n_pairs),
                max_ionic_pairs=ionic_pairs,
                structure_count=cumulative_structures,
                expanded_determinant_terms=cumulative_expanded_terms,
                unique_determinants=cumulative_unique_determinants,
                unique_alpha_strings=unique_spin_strings,
                unique_beta_strings=unique_spin_strings,
                determinant_redundancy=(
                    cumulative_expanded_terms / cumulative_unique_determinants
                    if cumulative_unique_determinants > 0
                    else 0.0
                ),
                same_spin_reuse=(
                    cumulative_unique_determinants / unique_spin_strings
                    if unique_spin_strings > 0
                    else 0.0
                ),
            )
        )
    return rows


def format_integer(value: int) -> str:
    return f"{value:,}"


def format_ratio(value: float) -> str:
    return f"{value:.4f}"


def render_case_markdown(n_active_orbitals: int) -> str:
    rows = summarize_subspaces(n_active_orbitals)
    lines = [
        f"## {n_active_orbitals}-{n_active_orbitals}",
        "",
        "| 子空间 | 价键结构数 | 展开总行列式数 | unique 总行列式数 | unique alpha string | unique beta string | 行列式冗余度 | same-spin reuse |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.label} | "
            f"{format_integer(row.structure_count)} | "
            f"{format_integer(row.expanded_determinant_terms)} | "
            f"{format_integer(row.unique_determinants)} | "
            f"{format_integer(row.unique_alpha_strings)} | "
            f"{format_integer(row.unique_beta_strings)} | "
            f"{format_ratio(row.determinant_redundancy)} | "
            f"{format_ratio(row.same_spin_reuse)} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_markdown(cases: list[int]) -> str:
    sections = [
        "# Closed-Shell Unique-String 子空间统计",
        "",
        "## 口径",
        "",
        "- 只统计 closed-shell `Ne = No` 活性空间。",
        "- `cov` 表示只保留 0 个 ionic pair 的共价结构。",
        "- `0-k` 表示累计保留 `0, 1, ..., k` 个 ionic pairs 的结构子空间。",
        "- `full` 表示累计到最大 ionic pair 数，也就是完整结构空间。",
        "- `展开总行列式数` 是结构展开后的 determinant term 总数，尚未做 full determinant 去重。",
        "- `unique 总行列式数` 是按 `(alpha, beta)` 成对去重后的 determinant 数。",
        "- `行列式冗余度 = 展开总行列式数 / unique 总行列式数`。",
        "- `same-spin reuse = unique 总行列式数 / unique alpha string`，closed-shell 下对 beta 完全相同。",
        "",
        "## 公式",
        "",
        "设活性空间为 `2p-2p`，`k` 为 ionic pair 数，则 legacy `genstr.c` 对应的 exact-`k` ion-class 计数为：",
        "",
        "- `structures_exact(k) = C(2p, k) C(2p-k, k) Catalan(p-k)`",
        "- `expanded_terms_exact(k) = structures_exact(k) * 2^(p-k)`",
        "- `unique_det_exact(k) = C(2p, k) C(2p-k, k) C(2p-2k, p-k)`",
        "- `unique_alpha = unique_beta = C(2p, p)`",
        "",
        "累计子空间 `0-k` 的数据就是对 exact ion classes `0..k` 求和。",
        "",
        "## 结果",
        "",
    ]
    for case in cases:
        sections.append(render_case_markdown(case))

    sections.extend(
        [
            "## 交叉核对",
            "",
            "- `6-6` 的 exact ion-class `(structures, unique determinants)` 为 `(5, 20), (60, 180), (90, 180), (20, 20)`，与 `test/10698_VBSCF.xmo` 一致。",
            "- `8-8` 的 exact ion-class `(structures, unique determinants)` 为 `(14, 70), (280, 1120), (840, 2520), (560, 1120), (70, 70)`，与 `test/240_VBSCF.xmo` 一致。",
            "- 这说明这里的累计子空间表与 legacy XMVB 的 ion-class 计数口径对齐。",
            "",
        ]
    )
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate cumulative subspace statistics for closed-shell Ne=No "
            "active spaces."
        )
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        type=int,
        default=[4, 6, 8, 10, 12],
        help="Active-space sizes Ne=No. Default: 6 8 10 12",
    )
    args = parser.parse_args()

    for case in args.cases:
        if case <= 0 or case % 2 != 0:
            raise ValueError("all cases must be positive even integers")

    print(render_markdown(args.cases), end="")


if __name__ == "__main__":
    main()
