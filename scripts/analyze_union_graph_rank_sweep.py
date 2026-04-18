#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import random
import re
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path


PAIR_LINE_RE = re.compile(r"^component\[(?P<index>\d+)\]\s+(?P<body>.*)$")
RANK_LINE_RE = re.compile(r"^rank_sweep\s+(?P<body>.*)$")


@dataclass
class RankSweepEntry:
    rank_cap: int
    offblock_fro_error: float
    offblock_rel_error: float
    approximate_overlap: float
    abs_error: float
    rel_error: float


@dataclass
class PairResult:
    left_structure: int
    right_structure: int
    component_count: int
    offblock_overlap_fraction: float
    exact_overlap: float
    block_diagonal_abs_error: float
    max_available_rank: int
    component_types: tuple[str, ...]
    rank_sweeps: dict[int, RankSweepEntry]


def parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch statistics for union-graph low-rank cross-block correction "
            "diagnostics using check_union_graph_overlap_blocks."
        )
    )
    parser.add_argument("input_file", type=Path, help="Input .xmi file")
    parser.add_argument(
        "--analyzer-binary",
        type=Path,
        default=Path("build/src/check_union_graph_overlap_blocks"),
        help="Path to check_union_graph_overlap_blocks. Default: build/src/check_union_graph_overlap_blocks",
    )
    parser.add_argument(
        "--count-binary",
        type=Path,
        default=Path("build/src/check_geminal_structure_overlap"),
        help="Path used to probe available structure count. Default: build/src/check_geminal_structure_overlap",
    )
    parser.add_argument(
        "--structure-count",
        type=int,
        default=0,
        help="Optional override for the number of structures. Default: 0 = probe from count binary",
    )
    parser.add_argument(
        "--include-diagonal",
        action="store_true",
        help="Include diagonal pairs (i,i). Default: off",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Analyze at most this many structure pairs. Default: 0 = all selected pairs",
    )
    parser.add_argument(
        "--pair-order",
        choices=["lexicographic", "random"],
        default="lexicographic",
        help="Pair traversal order. Default: lexicographic",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when --pair-order random. Default: 0",
    )
    parser.add_argument(
        "--max-rank-cap",
        type=int,
        default=-1,
        help="Maximum rank cap forwarded to the analyzer. Default: -1 = analyzer chooses",
    )
    parser.add_argument(
        "--singular-value-threshold",
        type=float,
        default=1.0e-8,
        help="Numerical rank threshold forwarded to the analyzer. Default: 1e-8",
    )
    parser.add_argument(
        "--rank-caps",
        type=parse_int_list,
        default=[0, 1, 2],
        help="Comma-separated rank caps to summarize. Default: 0,1,2",
    )
    parser.add_argument(
        "--normalized-error-floor",
        type=float,
        default=1.0e-8,
        help=(
            "Stabilizer used in abs_error / max(abs(exact_overlap), floor). "
            "Default: 1e-8"
        ),
    )
    parser.add_argument(
        "--normalized-error-thresholds",
        type=parse_float_list,
        default=[0.1, 0.5, 1.0],
        help=(
            "Comma-separated normalized-error thresholds for success-rate reporting. "
            "Default: 0.1,0.5,1.0"
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV path for per-pair results",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout


def parse_value_block(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if " = " not in line:
            continue
        key, value = line.split(" = ", 1)
        values[key.strip()] = value.strip()
    return values


def probe_structure_count(args: argparse.Namespace) -> int:
    if args.structure_count > 0:
        return args.structure_count
    output = run_command(
        [
            str(args.count_binary),
            str(args.input_file),
            "--max-structures",
            "1",
            "--report-count",
            "1",
            "--candidate",
            "exact_terms",
            "--pair-phase-mode",
            "legacy",
        ]
    )
    values = parse_value_block(output)
    if "available_structures" not in values:
        raise RuntimeError("failed to parse available_structures from count binary output")
    return int(values["available_structures"])


def build_pair_list(
    structure_count: int,
    include_diagonal: bool,
    pair_order: str,
    max_pairs: int,
    seed: int,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for left_structure in range(structure_count):
        for right_structure in range(left_structure + 1):
            if not include_diagonal and left_structure == right_structure:
                continue
            pairs.append((left_structure, right_structure))
    if pair_order == "random":
        rng = random.Random(seed)
        rng.shuffle(pairs)
    if max_pairs > 0:
        return pairs[:max_pairs]
    return pairs


def parse_component_type(line: str) -> str | None:
    match = PAIR_LINE_RE.match(line)
    if match is None:
        return None
    fields = {}
    for token in match.group("body").split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    return fields.get("type")


def parse_rank_line(line: str) -> RankSweepEntry | None:
    match = RANK_LINE_RE.match(line)
    if match is None:
        return None
    fields = {}
    for token in match.group("body").split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    return RankSweepEntry(
        rank_cap=int(fields["rank_cap"]),
        offblock_fro_error=float(fields["offblock_fro_error"]),
        offblock_rel_error=float(fields["offblock_rel_error"]),
        approximate_overlap=float(fields["approximate_overlap"]),
        abs_error=float(fields["abs_error"]),
        rel_error=float(fields["rel_error"]),
    )


def analyze_pair(args: argparse.Namespace, left_structure: int, right_structure: int) -> PairResult:
    command = [
        str(args.analyzer_binary),
        str(args.input_file),
        "--left-structure",
        str(left_structure),
        "--right-structure",
        str(right_structure),
        "--singular-value-threshold",
        str(args.singular_value_threshold),
    ]
    if args.max_rank_cap >= 0:
        command.extend(["--max-rank-cap", str(args.max_rank_cap)])
    output = run_command(command)
    values = parse_value_block(output)

    component_types: list[str] = []
    rank_sweeps: dict[int, RankSweepEntry] = {}
    for line in output.splitlines():
        component_type = parse_component_type(line)
        if component_type is not None:
            component_types.append(component_type)
            continue
        rank_entry = parse_rank_line(line)
        if rank_entry is not None:
            rank_sweeps[rank_entry.rank_cap] = rank_entry

    return PairResult(
        left_structure=int(values["left_structure"]),
        right_structure=int(values["right_structure"]),
        component_count=int(values["component_count"]),
        offblock_overlap_fraction=float(values["offblock_overlap_fraction"]),
        exact_overlap=float(values["exact_overlap"]),
        block_diagonal_abs_error=float(values["block_diagonal_abs_error"]),
        max_available_rank=int(values["max_available_rank"]),
        component_types=tuple(component_types),
        rank_sweeps=rank_sweeps,
    )


def median_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(statistics.median(values))


def mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(statistics.fmean(values))


def summarize_rank_cap(
    pair_results: list[PairResult],
    rank_cap: int,
    normalized_error_floor: float,
    normalized_error_thresholds: list[float],
) -> list[str]:
    available_results = [
        result.rank_sweeps[rank_cap]
        for result in pair_results
        if rank_cap in result.rank_sweeps
    ]
    if not available_results:
        return [f"rank_cap[{rank_cap}]_available_pairs = 0"]

    abs_errors = [entry.abs_error for entry in available_results]
    normalized_errors = []
    for pair_result in pair_results:
        if rank_cap not in pair_result.rank_sweeps:
            continue
        entry = pair_result.rank_sweeps[rank_cap]
        scale = max(abs(pair_result.exact_overlap), normalized_error_floor)
        normalized_errors.append(entry.abs_error / scale)

    lines = [
        f"rank_cap[{rank_cap}]_available_pairs = {len(available_results)}",
        f"rank_cap[{rank_cap}]_abs_error_mean = {mean_or_nan(abs_errors):.6e}",
        f"rank_cap[{rank_cap}]_abs_error_median = {median_or_nan(abs_errors):.6e}",
        f"rank_cap[{rank_cap}]_normalized_error_mean = {mean_or_nan(normalized_errors):.6e}",
        f"rank_cap[{rank_cap}]_normalized_error_median = {median_or_nan(normalized_errors):.6e}",
    ]
    for threshold in normalized_error_thresholds:
        success_count = sum(error <= threshold for error in normalized_errors)
        success_fraction = success_count / len(normalized_errors)
        lines.append(
            f"rank_cap[{rank_cap}]_normalized_error_le_{threshold:g} = {success_fraction:.6f}"
        )
    return lines


def write_csv(
    path: Path,
    pair_results: list[PairResult],
    rank_caps: list[int],
    normalized_error_floor: float,
) -> None:
    fieldnames = [
        "left_structure",
        "right_structure",
        "component_count",
        "component_types",
        "offblock_overlap_fraction",
        "exact_overlap",
        "block_diagonal_abs_error",
        "max_available_rank",
    ]
    for rank_cap in rank_caps:
        fieldnames.extend(
            [
                f"rank_{rank_cap}_available",
                f"rank_{rank_cap}_approximate_overlap",
                f"rank_{rank_cap}_abs_error",
                f"rank_{rank_cap}_normalized_error",
            ]
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in pair_results:
            row = {
                "left_structure": result.left_structure,
                "right_structure": result.right_structure,
                "component_count": result.component_count,
                "component_types": " ".join(result.component_types),
                "offblock_overlap_fraction": result.offblock_overlap_fraction,
                "exact_overlap": result.exact_overlap,
                "block_diagonal_abs_error": result.block_diagonal_abs_error,
                "max_available_rank": result.max_available_rank,
            }
            for rank_cap in rank_caps:
                entry = result.rank_sweeps.get(rank_cap)
                row[f"rank_{rank_cap}_available"] = int(entry is not None)
                if entry is None:
                    row[f"rank_{rank_cap}_approximate_overlap"] = ""
                    row[f"rank_{rank_cap}_abs_error"] = ""
                    row[f"rank_{rank_cap}_normalized_error"] = ""
                else:
                    row[f"rank_{rank_cap}_approximate_overlap"] = entry.approximate_overlap
                    row[f"rank_{rank_cap}_abs_error"] = entry.abs_error
                    row[f"rank_{rank_cap}_normalized_error"] = (
                        entry.abs_error / max(abs(result.exact_overlap), normalized_error_floor)
                    )
            writer.writerow(row)


def histogram(values: list[int | str]) -> dict[int | str, int]:
    counts: dict[int | str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def main() -> int:
    args = parse_args()
    structure_count = probe_structure_count(args)
    pair_list = build_pair_list(
        structure_count=structure_count,
        include_diagonal=args.include_diagonal,
        pair_order=args.pair_order,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )
    if not pair_list:
        raise RuntimeError("no structure pairs selected")

    pair_results = [analyze_pair(args, left, right) for left, right in pair_list]

    component_count_hist = histogram([result.component_count for result in pair_results])
    max_rank_hist = histogram([result.max_available_rank for result in pair_results])

    print(f"input_file = {args.input_file}")
    print(f"structure_count = {structure_count}")
    print(f"selected_pairs = {len(pair_results)}")
    print(f"pair_order = {args.pair_order}")
    print(f"include_diagonal = {int(args.include_diagonal)}")
    print(f"normalized_error_floor = {args.normalized_error_floor}")
    print(f"rank_caps = {args.rank_caps}")
    print(f"component_count_histogram = {component_count_hist}")
    print(f"max_available_rank_histogram = {max_rank_hist}")
    print(
        "offblock_overlap_fraction_mean = "
        f"{mean_or_nan([result.offblock_overlap_fraction for result in pair_results]):.6e}"
    )
    print(
        "offblock_overlap_fraction_median = "
        f"{median_or_nan([result.offblock_overlap_fraction for result in pair_results]):.6e}"
    )
    print(
        "block_diagonal_abs_error_mean = "
        f"{mean_or_nan([result.block_diagonal_abs_error for result in pair_results]):.6e}"
    )
    print(
        "block_diagonal_abs_error_median = "
        f"{median_or_nan([result.block_diagonal_abs_error for result in pair_results]):.6e}"
    )

    for rank_cap in args.rank_caps:
        for line in summarize_rank_cap(
            pair_results=pair_results,
            rank_cap=rank_cap,
            normalized_error_floor=args.normalized_error_floor,
            normalized_error_thresholds=args.normalized_error_thresholds,
        ):
            print(line)

    if args.csv is not None:
        write_csv(
            path=args.csv,
            pair_results=pair_results,
            rank_caps=args.rank_caps,
            normalized_error_floor=args.normalized_error_floor,
        )
        print(f"csv = {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
