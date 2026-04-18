#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
import re
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path


VALUE_RE = re.compile(r"^\s*([^=]+?)\s*=\s*(.*)\s*$")
RANK_LINE_RE = re.compile(r"^rank_sweep\s+(?P<body>.*)$")


@dataclass
class PairFeatures:
    input_file: str
    left_structure: int
    right_structure: int
    component_count: int
    n_general_components: int
    max_available_rank: int
    offblock_overlap_fraction: float
    max_cross_block_second_singular: float
    max_cross_block_third_singular: float
    exact_overlap: float
    required_rank: int


@dataclass
class RuleMetrics:
    offblock_threshold: float
    second_singular_threshold: float
    third_singular_threshold: float
    underprediction_count: int
    exact_match_count: int
    mean_predicted_rank: float
    mean_overprediction: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search a simple conservative rank-screening rule for union-graph "
            "cross-block correction diagnostics."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Input .xmi files or directories containing .xmi files",
    )
    parser.add_argument(
        "--analyzer-binary",
        type=Path,
        default=Path("build/src/check_union_graph_overlap_blocks"),
        help="Path to check_union_graph_overlap_blocks",
    )
    parser.add_argument(
        "--count-binary",
        type=Path,
        default=Path("build/src/check_geminal_structure_overlap"),
        help="Path to structure-count probe binary",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Analyze at most this many files after sorting. Default: 0 = all",
    )
    parser.add_argument(
        "--pairs-per-file",
        type=int,
        default=8,
        help="Analyze at most this many pairs per file. Default: 8",
    )
    parser.add_argument(
        "--pair-order",
        choices=["lexicographic", "random"],
        default="random",
        help="Pair traversal order inside each file. Default: random",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed used when --pair-order random. Default: 0",
    )
    parser.add_argument(
        "--max-rank-cap",
        type=int,
        default=3,
        help="Maximum rank cap forwarded to the analyzer. Default: 3",
    )
    parser.add_argument(
        "--normalized-error-floor",
        type=float,
        default=1.0e-14,
        help="Normalization floor used when computing required rank. Default: 1e-14",
    )
    parser.add_argument(
        "--target-normalized-error",
        type=float,
        default=1.0e-2,
        help="Tolerance used to define required rank. Default: 1e-2",
    )
    parser.add_argument(
        "--quantiles",
        type=int,
        default=9,
        help="Number of quantile cut points per scalar feature. Default: 9",
    )
    parser.add_argument(
        "--top-rules",
        type=int,
        default=10,
        help="Print at most this many candidate rules. Default: 10",
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
        match = VALUE_RE.match(line)
        if match is None:
            continue
        values[match.group(1).strip()] = match.group(2).strip()
    return values


def parse_key_value_tokens(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in text.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def resolve_input_files(inputs: list[Path], max_files: int) -> list[Path]:
    resolved: list[Path] = []
    for input_path in inputs:
        if input_path.is_dir():
            resolved.extend(sorted(input_path.glob("*.xmi")))
            continue
        if input_path.suffix == ".xmi" and input_path.is_file():
            resolved.append(input_path)
    unique_files = sorted({path.resolve() for path in resolved})
    if max_files > 0:
        return unique_files[:max_files]
    return unique_files


def probe_structure_count(count_binary: Path, input_file: Path) -> int:
    output = run_command(
        [
            str(count_binary),
            str(input_file),
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
    return int(values["available_structures"])


def build_pair_list(
    structure_count: int,
    pair_order: str,
    pairs_per_file: int,
    seed: int,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for left_structure in range(structure_count):
        for right_structure in range(left_structure):
            pairs.append((left_structure, right_structure))
    if pair_order == "random":
        rng = random.Random(seed)
        rng.shuffle(pairs)
    if pairs_per_file > 0:
        return pairs[:pairs_per_file]
    return pairs


def compute_required_rank(
    output: str,
    exact_overlap: float,
    normalized_error_floor: float,
    target_normalized_error: float,
) -> int:
    scale = max(abs(exact_overlap), normalized_error_floor)
    rank_errors: dict[int, float] = {}
    for line in output.splitlines():
        match = RANK_LINE_RE.match(line)
        if match is None:
            continue
        fields = parse_key_value_tokens(match.group("body"))
        rank_cap = int(fields["rank_cap"])
        abs_error = float(fields["abs_error"])
        rank_errors[rank_cap] = abs_error / scale
    for rank_cap in sorted(rank_errors):
        if rank_errors[rank_cap] <= target_normalized_error:
            return rank_cap
    return max(rank_errors)


def analyze_pair(
    analyzer_binary: Path,
    input_file: Path,
    left_structure: int,
    right_structure: int,
    max_rank_cap: int,
    normalized_error_floor: float,
    target_normalized_error: float,
) -> PairFeatures:
    output = run_command(
        [
            str(analyzer_binary),
            str(input_file),
            "--left-structure",
            str(left_structure),
            "--right-structure",
            str(right_structure),
            "--max-rank-cap",
            str(max_rank_cap),
        ]
    )
    values = parse_value_block(output)
    exact_overlap = float(values["exact_overlap"])
    required_rank = compute_required_rank(
        output,
        exact_overlap=exact_overlap,
        normalized_error_floor=normalized_error_floor,
        target_normalized_error=target_normalized_error,
    )
    return PairFeatures(
        input_file=str(input_file),
        left_structure=int(values["left_structure"]),
        right_structure=int(values["right_structure"]),
        component_count=int(values["component_count"]),
        n_general_components=int(values["n_general_components"]),
        max_available_rank=int(values["max_available_rank"]),
        offblock_overlap_fraction=float(values["offblock_overlap_fraction"]),
        max_cross_block_second_singular=float(values["max_cross_block_second_singular"]),
        max_cross_block_third_singular=float(values["max_cross_block_third_singular"]),
        exact_overlap=exact_overlap,
        required_rank=required_rank,
    )


def quantile_candidates(values: list[float], n_quantiles: int) -> list[float]:
    if not values:
        return [0.0]
    sorted_values = sorted(values)
    candidates = {0.0, sorted_values[0], sorted_values[-1]}
    if len(sorted_values) == 1:
        return sorted(candidates)
    for quantile_index in range(n_quantiles):
        fraction = quantile_index / max(1, n_quantiles - 1)
        position = int(round(fraction * (len(sorted_values) - 1)))
        candidates.add(sorted_values[position])
    return sorted(candidates)


def predict_rank(
    pair: PairFeatures,
    offblock_threshold: float,
    second_singular_threshold: float,
    third_singular_threshold: float,
    max_rank_cap: int,
) -> int:
    if pair.component_count <= 1:
      return 0
    if pair.offblock_overlap_fraction <= offblock_threshold and pair.n_general_components == 0:
      return 0
    if pair.max_cross_block_second_singular <= second_singular_threshold:
      return min(1, max_rank_cap)
    if pair.max_cross_block_third_singular <= third_singular_threshold:
      return min(2, max_rank_cap)
    return min(max_rank_cap, max(2, pair.max_available_rank))


def evaluate_rule(
    pairs: list[PairFeatures],
    offblock_threshold: float,
    second_singular_threshold: float,
    third_singular_threshold: float,
    max_rank_cap: int,
) -> RuleMetrics:
    predicted_ranks = [
        predict_rank(
            pair,
            offblock_threshold=offblock_threshold,
            second_singular_threshold=second_singular_threshold,
            third_singular_threshold=third_singular_threshold,
            max_rank_cap=max_rank_cap,
        )
        for pair in pairs
    ]
    underprediction_count = sum(
        predicted_rank < pair.required_rank
        for predicted_rank, pair in zip(predicted_ranks, pairs)
    )
    exact_match_count = sum(
        predicted_rank == pair.required_rank
        for predicted_rank, pair in zip(predicted_ranks, pairs)
    )
    overpredictions = [
        max(predicted_rank - pair.required_rank, 0)
        for predicted_rank, pair in zip(predicted_ranks, pairs)
    ]
    return RuleMetrics(
        offblock_threshold=offblock_threshold,
        second_singular_threshold=second_singular_threshold,
        third_singular_threshold=third_singular_threshold,
        underprediction_count=underprediction_count,
        exact_match_count=exact_match_count,
        mean_predicted_rank=statistics.fmean(predicted_ranks),
        mean_overprediction=statistics.fmean(overpredictions),
    )


def main() -> int:
    args = parse_args()
    input_files = resolve_input_files(args.inputs, args.max_files)
    if not input_files:
        raise RuntimeError("no input .xmi files found")

    pairs: list[PairFeatures] = []
    for file_index, input_file in enumerate(input_files):
        structure_count = probe_structure_count(args.count_binary, input_file)
        if structure_count < 2:
            continue
        pair_list = build_pair_list(
            structure_count=structure_count,
            pair_order=args.pair_order,
            pairs_per_file=args.pairs_per_file,
            seed=args.seed + file_index,
        )
        for left_structure, right_structure in pair_list:
            pairs.append(
                analyze_pair(
                    analyzer_binary=args.analyzer_binary,
                    input_file=input_file,
                    left_structure=left_structure,
                    right_structure=right_structure,
                    max_rank_cap=args.max_rank_cap,
                    normalized_error_floor=args.normalized_error_floor,
                    target_normalized_error=args.target_normalized_error,
                )
            )

    if not pairs:
        raise RuntimeError("no pairs analyzed")

    offblock_thresholds = quantile_candidates(
        [pair.offblock_overlap_fraction for pair in pairs],
        args.quantiles,
    )
    second_singular_thresholds = quantile_candidates(
        [pair.max_cross_block_second_singular for pair in pairs],
        args.quantiles,
    )
    third_singular_thresholds = quantile_candidates(
        [pair.max_cross_block_third_singular for pair in pairs],
        args.quantiles,
    )

    rules: list[RuleMetrics] = []
    for offblock_threshold in offblock_thresholds:
        for second_singular_threshold in second_singular_thresholds:
            for third_singular_threshold in third_singular_thresholds:
                rules.append(
                    evaluate_rule(
                        pairs,
                        offblock_threshold=offblock_threshold,
                        second_singular_threshold=second_singular_threshold,
                        third_singular_threshold=third_singular_threshold,
                        max_rank_cap=args.max_rank_cap,
                    )
                )

    rules.sort(
        key=lambda item: (
            item.underprediction_count,
            item.mean_predicted_rank,
            item.mean_overprediction,
            -item.exact_match_count,
        )
    )

    print(f"input_files = {len(input_files)}")
    print(f"analyzed_pairs = {len(pairs)}")
    print(f"max_rank_cap = {args.max_rank_cap}")
    print(f"target_normalized_error = {args.target_normalized_error}")
    print(
        "required_rank_histogram = "
        f"{ {rank: sum(pair.required_rank == rank for pair in pairs) for rank in sorted({pair.required_rank for pair in pairs})} }"
    )
    print()
    print("best_rules:")
    for rule in rules[:args.top_rules]:
        print(
            "offblock_threshold = "
            f"{rule.offblock_threshold:.6e} | "
            "second_singular_threshold = "
            f"{rule.second_singular_threshold:.6e} | "
            "third_singular_threshold = "
            f"{rule.third_singular_threshold:.6e} | "
            f"underprediction_count = {rule.underprediction_count} | "
            f"exact_match_count = {rule.exact_match_count} | "
            f"mean_predicted_rank = {rule.mean_predicted_rank:.6f} | "
            f"mean_overprediction = {rule.mean_overprediction:.6f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
