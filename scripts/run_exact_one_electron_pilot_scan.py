#!/usr/bin/env python3

import argparse
import csv
import os
import pathlib
import random
import re
import statistics
import subprocess
import sys
from typing import Dict, List, Sequence, Tuple


DEFAULT_BINARY = "build/src/analyze_star_separator_one_electron_dataset"
DEFAULT_DATASET_ROOT = "data/training_xmi/6e6o_full"
EXAMPLE_PATTERN = re.compile(r"^example\[(\d+)\]\s+(.*)$")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a pilot exact one-electron separator-state scan over a sampled "
            "subset of the 6e6o_full dataset and emit pair-level CSV output."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        help="Path to the 6e6o_full dataset directory.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional manifest path. Defaults to <dataset-root>/manifest.txt.",
    )
    parser.add_argument(
        "--binary",
        default=DEFAULT_BINARY,
        help="Path to analyze_star_separator_one_electron_dataset.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=3,
        help="Maximum number of molecule files to sample from the manifest.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=2,
        help="Maximum number of random structure pairs to analyze per molecule.",
    )
    parser.add_argument(
        "--top-examples",
        type=int,
        default=None,
        help=(
            "Number of example rows requested from the analyzer per file. "
            "Defaults to --max-pairs."
        ),
    )
    parser.add_argument(
        "--target-pair-rows",
        type=int,
        default=0,
        help=(
            "Stop once at least this many pair rows have been collected across "
            "all processed files. Zero means no early stop."
        ),
    )
    parser.add_argument(
        "--pair-order",
        default="random",
        choices=["random", "lexicographic"],
        help="Pair order passed to the analyzer.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260402,
        help="Seed used both for file sampling and analyzer pair-order sampling.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=240.0,
        help="Per-file timeout passed to subprocess.run.",
    )
    parser.add_argument(
        "--output",
        default="/tmp/exact_one_electron_pilot_pairs.csv",
        help="CSV path for pair-level output.",
    )
    parser.add_argument(
        "--summary-output",
        default="/tmp/exact_one_electron_pilot_summary.csv",
        help="CSV path for file-level summary output.",
    )
    return parser.parse_args()


def read_manifest(dataset_root: pathlib.Path, manifest_path: pathlib.Path) -> List[pathlib.Path]:
    paths: List[pathlib.Path] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            entry = raw_line.strip()
            if not entry:
                continue
            path = dataset_root / entry
            if path.is_file():
                paths.append(path)
    return paths


def select_files(paths: Sequence[pathlib.Path], max_files: int, seed: int) -> List[pathlib.Path]:
    if max_files <= 0 or max_files >= len(paths):
        return list(paths)
    rng = random.Random(seed)
    return sorted(rng.sample(list(paths), max_files))


def parse_global_metrics(stdout: str) -> Dict[str, str]:
    metrics: Dict[str, str] = {}
    for line in stdout.splitlines():
        if " = " not in line:
            continue
        key, value = line.split(" = ", 1)
        metrics[key.strip()] = value.strip()
    return metrics


def parse_example_line(line: str) -> Dict[str, str]:
    match = EXAMPLE_PATTERN.match(line.strip())
    if match is None:
        return {}
    fields: Dict[str, str] = {"example_index": match.group(1)}
    for token in match.group(2).split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def parse_examples(stdout: str) -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []
    for line in stdout.splitlines():
        if not line.startswith("example["):
            continue
        parsed = parse_example_line(line)
        if parsed:
            examples.append(parsed)
    return examples


def run_analyzer(
    binary: str,
    input_path: pathlib.Path,
    max_pairs: int,
    top_examples: int,
    pair_order: str,
    seed: int,
    timeout_seconds: float,
) -> Tuple[Dict[str, str], List[Dict[str, str]], str, int]:
    command = [
        binary,
        str(input_path),
        "--pair-order",
        pair_order,
        "--max-pairs",
        str(max_pairs),
        "--top-examples",
        str(top_examples),
        "--seed",
        str(seed),
    ]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as error:
        return {}, [], f"timeout after {timeout_seconds}s: {error}", 124
    metrics = parse_global_metrics(completed.stdout)
    examples = parse_examples(completed.stdout)
    return metrics, examples, completed.stderr.strip(), completed.returncode


def safe_float(value: str) -> float:
    if value == "":
        return float("nan")
    return float(value)


def build_pair_rows(
    input_path: pathlib.Path,
    metrics: Dict[str, str],
    examples: Sequence[Dict[str, str]],
    stderr: str,
    returncode: int,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for example in examples:
        row = {
            "input_file": str(input_path),
            "molecule_id": input_path.stem,
            "status": "ok" if returncode == 0 else f"exit_{returncode}",
            "selected_pairs": metrics.get("selected_pairs", ""),
            "covered_star_pair_count": metrics.get("covered_star_pair_count", ""),
            "unique_reference_determinant_pair_count": metrics.get(
                "unique_reference_determinant_pair_count", ""
            ),
            "unique_exact_separator_state_count": metrics.get(
                "unique_exact_separator_state_count", ""
            ),
            "unique_exact_leaf_message_bundle_count": metrics.get(
                "unique_exact_leaf_message_bundle_count", ""
            ),
            "unique_exact_merge_state_count": metrics.get(
                "unique_exact_merge_state_count", ""
            ),
            "elapsed_wall_time_seconds": metrics.get("elapsed_wall_time_seconds", ""),
            "stderr": stderr.replace("\n", "\\n"),
        }
        row.update(example)
        rows.append(row)
    return rows


def build_summary_row(
    input_path: pathlib.Path,
    metrics: Dict[str, str],
    pair_rows: Sequence[Dict[str, str]],
    returncode: int,
) -> Dict[str, str]:
    alpha_ranks = [
        safe_float(row.get("explicit_one_leaf_alpha_matrix_target_span_rank", ""))
        for row in pair_rows
        if row.get("explicit_one_leaf_alpha_matrix_target_span_rank", "") != ""
    ]
    beta_ranks = [
        safe_float(row.get("explicit_one_leaf_beta_matrix_target_span_rank", ""))
        for row in pair_rows
        if row.get("explicit_one_leaf_beta_matrix_target_span_rank", "") != ""
    ]
    one_electron_errors = [
        safe_float(row.get("one_electron_abs_error", ""))
        for row in pair_rows
        if row.get("one_electron_abs_error", "") != ""
    ]
    widths = [
        safe_float(row.get("width_upper_bound", ""))
        for row in pair_rows
        if row.get("width_upper_bound", "") != ""
    ]
    return {
        "input_file": str(input_path),
        "molecule_id": input_path.stem,
        "status": "ok" if returncode == 0 else f"exit_{returncode}",
        "selected_pairs": metrics.get("selected_pairs", ""),
        "covered_star_pair_count": metrics.get("covered_star_pair_count", ""),
        "pair_rows": str(len(pair_rows)),
        "max_width_upper_bound": (
            f"{max(widths):.16g}" if widths else ""
        ),
        "median_alpha_target_span_rank": (
            f"{statistics.median(alpha_ranks):.16g}" if alpha_ranks else ""
        ),
        "median_beta_target_span_rank": (
            f"{statistics.median(beta_ranks):.16g}" if beta_ranks else ""
        ),
        "max_alpha_target_span_rank": (
            f"{max(alpha_ranks):.16g}" if alpha_ranks else ""
        ),
        "max_beta_target_span_rank": (
            f"{max(beta_ranks):.16g}" if beta_ranks else ""
        ),
        "max_one_electron_abs_error": (
            f"{max(one_electron_errors):.16g}" if one_electron_errors else ""
        ),
        "unique_reference_determinant_pair_count": metrics.get(
            "unique_reference_determinant_pair_count", ""
        ),
        "unique_exact_leaf_message_bundle_count": metrics.get(
            "unique_exact_leaf_message_bundle_count", ""
        ),
        "unique_exact_merge_state_count": metrics.get(
            "unique_exact_merge_state_count", ""
        ),
        "elapsed_wall_time_seconds": metrics.get("elapsed_wall_time_seconds", ""),
    }


def write_csv(path: pathlib.Path, rows: Sequence[Dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_console_summary(summary_rows: Sequence[Dict[str, str]]) -> None:
    if not summary_rows:
        print("no summary rows collected")
        return
    print("pilot_scan_summary")
    for row in summary_rows:
        print(
            " ".join(
                [
                    f"molecule_id={row.get('molecule_id', '')}",
                    f"status={row.get('status', '')}",
                    f"pair_rows={row.get('pair_rows', '')}",
                    f"selected_pairs={row.get('selected_pairs', '')}",
                    f"median_alpha_rank={row.get('median_alpha_target_span_rank', '')}",
                    f"median_beta_rank={row.get('median_beta_target_span_rank', '')}",
                    f"max_error={row.get('max_one_electron_abs_error', '')}",
                    f"elapsed_s={row.get('elapsed_wall_time_seconds', '')}",
                ]
            )
        )


def print_rank_summary(pair_rows: Sequence[Dict[str, str]]) -> None:
    def collect_ints(key: str) -> List[int]:
        values: List[int] = []
        for row in pair_rows:
            value = row.get(key, "")
            if value == "":
                continue
            values.append(int(float(value)))
        return values

    alpha_ranks = collect_ints("explicit_one_leaf_alpha_matrix_target_span_rank")
    beta_ranks = collect_ints("explicit_one_leaf_beta_matrix_target_span_rank")
    widths = collect_ints("width_upper_bound")
    print("pair_rank_summary")
    print(f"pair_row_count={len(pair_rows)}")
    if alpha_ranks:
        print(
            f"alpha_rank_values={alpha_ranks} alpha_rank_min={min(alpha_ranks)} "
            f"alpha_rank_median={statistics.median(alpha_ranks):.16g} "
            f"alpha_rank_max={max(alpha_ranks)}"
        )
    if beta_ranks:
        print(
            f"beta_rank_values={beta_ranks} beta_rank_min={min(beta_ranks)} "
            f"beta_rank_median={statistics.median(beta_ranks):.16g} "
            f"beta_rank_max={max(beta_ranks)}"
        )
    if widths:
        print(
            f"width_values={widths} width_min={min(widths)} "
            f"width_median={statistics.median(widths):.16g} "
            f"width_max={max(widths)}"
        )


def main() -> int:
    args = parse_arguments()
    dataset_root = pathlib.Path(args.dataset_root).resolve()
    manifest_path = (
        pathlib.Path(args.manifest).resolve()
        if args.manifest is not None
        else dataset_root / "manifest.txt"
    )
    output_path = pathlib.Path(args.output).resolve()
    summary_output_path = pathlib.Path(args.summary_output).resolve()

    paths = read_manifest(dataset_root, manifest_path)
    selected_files = select_files(paths, args.max_files, args.seed)
    if not selected_files:
        raise RuntimeError("no input files selected for pilot scan")

    pair_rows: List[Dict[str, str]] = []
    summary_rows: List[Dict[str, str]] = []
    top_examples = args.top_examples if args.top_examples is not None else args.max_pairs
    for file_index, input_path in enumerate(selected_files):
        file_seed = args.seed + 1009 * file_index
        print(
            f"[pilot] running {input_path.name} max_pairs={args.max_pairs} seed={file_seed}",
            file=sys.stderr,
            flush=True,
        )
        metrics, examples, stderr, returncode = run_analyzer(
            args.binary,
            input_path,
            args.max_pairs,
            top_examples,
            args.pair_order,
            file_seed,
            args.timeout_seconds,
        )
        file_pair_rows = build_pair_rows(input_path, metrics, examples, stderr, returncode)
        summary_rows.append(build_summary_row(input_path, metrics, file_pair_rows, returncode))
        pair_rows.extend(file_pair_rows)
        if args.target_pair_rows > 0 and len(pair_rows) >= args.target_pair_rows:
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_path, pair_rows)
    write_csv(summary_output_path, summary_rows)
    print_console_summary(summary_rows)
    print_rank_summary(pair_rows)
    print(f"pair_csv={output_path}")
    print(f"summary_csv={summary_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
