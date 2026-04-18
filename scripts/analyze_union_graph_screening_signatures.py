#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


VALUE_RE = re.compile(r"^\s*([^=]+?)\s*=\s*(.*)\s*$")
COMPONENT_LINE_RE = re.compile(r"^component\[(?P<index>\d+)\]\s+(?P<body>.*)$")
RANK_LINE_RE = re.compile(r"^rank_sweep\s+(?P<body>.*)$")


@dataclass
class ComponentInfo:
    component_index: int
    component_type: str
    orbital_count: int


@dataclass
class PairSummary:
    input_file: str
    left_structure: int
    right_structure: int
    structure_count: int
    component_count: int
    component_signature: str
    max_available_rank: int
    exact_overlap: float
    offblock_overlap_fraction: float
    required_rank_for_tolerance: int | None


def parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate union-graph component signatures across many XMI inputs and "
            "measure how strongly they predict the required low-rank correction."
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
        default=10,
        help="Analyze at most this many pairs per file. Default: 10",
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
        "--top-signatures",
        type=int,
        default=12,
        help="Print at most this many signatures in the summary. Default: 12",
    )
    parser.add_argument(
        "--min-signature-count",
        type=int,
        default=2,
        help="Only print signatures seen at least this many times. Default: 2",
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


def parse_component_line(line: str) -> ComponentInfo | None:
    match = COMPONENT_LINE_RE.match(line)
    if match is None:
        return None
    fields = parse_key_value_tokens(match.group("body"))
    orbitals_match = re.search(r"orbitals=\[(?P<body>[^\]]*)\]", line)
    orbital_count = 0
    if orbitals_match is not None:
        orbitals_body = orbitals_match.group("body").strip()
        if orbitals_body:
            orbital_count = len(orbitals_body.split())
    return ComponentInfo(
        component_index=int(match.group("index")),
        component_type=fields.get("type", "unknown"),
        orbital_count=orbital_count,
    )


def parse_rank_errors(
    text: str,
    normalized_error_floor: float,
    exact_overlap: float,
) -> dict[int, float]:
    errors: dict[int, float] = {}
    scale = max(abs(exact_overlap), normalized_error_floor)
    for line in text.splitlines():
        match = RANK_LINE_RE.match(line)
        if match is None:
            continue
        fields = parse_key_value_tokens(match.group("body"))
        rank_cap = int(fields["rank_cap"])
        abs_error = float(fields["abs_error"])
        errors[rank_cap] = abs_error / scale
    return errors


def component_signature(components: list[ComponentInfo]) -> str:
    labels = sorted(
        f"{component.component_type}:{component.orbital_count}"
        for component in components
    )
    return "+".join(labels)


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


def analyze_pair(
    analyzer_binary: Path,
    input_file: Path,
    structure_count: int,
    left_structure: int,
    right_structure: int,
    max_rank_cap: int,
    normalized_error_floor: float,
    target_normalized_error: float,
) -> PairSummary:
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
    components = [
        component
        for line in output.splitlines()
        for component in [parse_component_line(line)]
        if component is not None
    ]
    exact_overlap = float(values["exact_overlap"])
    rank_errors = parse_rank_errors(
        output,
        normalized_error_floor=normalized_error_floor,
        exact_overlap=exact_overlap,
    )
    required_rank: int | None = None
    for rank_cap in sorted(rank_errors):
        if rank_errors[rank_cap] <= target_normalized_error:
            required_rank = rank_cap
            break

    return PairSummary(
        input_file=str(input_file),
        left_structure=int(values["left_structure"]),
        right_structure=int(values["right_structure"]),
        structure_count=structure_count,
        component_count=int(values["component_count"]),
        component_signature=component_signature(components),
        max_available_rank=int(values["max_available_rank"]),
        exact_overlap=exact_overlap,
        offblock_overlap_fraction=float(values["offblock_overlap_fraction"]),
        required_rank_for_tolerance=required_rank,
    )


def mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(statistics.fmean(values))


def median_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(statistics.median(values))


def main() -> int:
    args = parse_args()
    input_files = resolve_input_files(args.inputs, args.max_files)
    if not input_files:
        raise RuntimeError("no input .xmi files found")

    pair_summaries: list[PairSummary] = []
    structure_counts: list[int] = []
    skipped_files = 0

    for file_index, input_file in enumerate(input_files):
        structure_count = probe_structure_count(args.count_binary, input_file)
        structure_counts.append(structure_count)
        if structure_count < 2:
            skipped_files += 1
            continue
        pair_list = build_pair_list(
            structure_count=structure_count,
            pair_order=args.pair_order,
            pairs_per_file=args.pairs_per_file,
            seed=args.seed + file_index,
        )
        for left_structure, right_structure in pair_list:
            pair_summaries.append(
                analyze_pair(
                    analyzer_binary=args.analyzer_binary,
                    input_file=input_file,
                    structure_count=structure_count,
                    left_structure=left_structure,
                    right_structure=right_structure,
                    max_rank_cap=args.max_rank_cap,
                    normalized_error_floor=args.normalized_error_floor,
                    target_normalized_error=args.target_normalized_error,
                )
            )

    if not pair_summaries:
        raise RuntimeError("no pair summaries collected")

    signature_counter = Counter(
        summary.component_signature for summary in pair_summaries
    )
    rank_counter = Counter(summary.max_available_rank for summary in pair_summaries)
    required_counter = Counter(
        summary.required_rank_for_tolerance for summary in pair_summaries
    )

    per_signature_rank_counter: dict[str, Counter[int]] = defaultdict(Counter)
    per_signature_required_counter: dict[str, Counter[int | None]] = defaultdict(Counter)
    for summary in pair_summaries:
        per_signature_rank_counter[summary.component_signature][summary.max_available_rank] += 1
        per_signature_required_counter[summary.component_signature][
            summary.required_rank_for_tolerance
        ] += 1

    print(f"input_files = {len(input_files)}")
    print(f"skipped_files = {skipped_files}")
    print(f"analyzed_pairs = {len(pair_summaries)}")
    print(f"pairs_per_file = {args.pairs_per_file}")
    print(f"pair_order = {args.pair_order}")
    print(f"structure_count_mean = {mean_or_nan(structure_counts):.6f}")
    print(f"structure_count_median = {median_or_nan(structure_counts):.6f}")
    print(f"max_available_rank_histogram = {dict(sorted(rank_counter.items()))}")
    print(f"required_rank_histogram = {dict(sorted(required_counter.items(), key=lambda item: (-1 if item[0] is None else item[0])))}")
    print(
        "offblock_overlap_fraction_mean = "
        f"{mean_or_nan([summary.offblock_overlap_fraction for summary in pair_summaries]):.6e}"
    )
    print(
        "offblock_overlap_fraction_median = "
        f"{median_or_nan([summary.offblock_overlap_fraction for summary in pair_summaries]):.6e}"
    )
    print()
    print("signature_summaries:")

    printed = 0
    for signature, count in signature_counter.most_common():
        if count < args.min_signature_count:
            continue
        rank_hist = dict(sorted(per_signature_rank_counter[signature].items()))
        required_hist = dict(
            sorted(
                per_signature_required_counter[signature].items(),
                key=lambda item: (-1 if item[0] is None else item[0]),
            )
        )
        print(
            f"signature = {signature} | pair_count = {count} | "
            f"max_rank_hist = {rank_hist} | required_rank_hist = {required_hist}"
        )
        printed += 1
        if printed >= args.top_signatures:
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
