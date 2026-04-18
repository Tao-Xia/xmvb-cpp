#!/usr/bin/env python3

import argparse
import csv
import os
import subprocess
import sys
from typing import Dict, Iterable, List


DEFAULT_METRICS = [
    "selected_pairs",
    "covered_star_pair_count",
    "one_electron_max_abs_error",
    "total_reference_determinant_pair_count",
    "total_local_term_pair_visits",
    "unique_reference_determinant_pair_count",
    "unique_exact_leaf_message_state_count",
    "unique_exact_merge_state_count",
    "layered_state_total_over_unique_reference_pair_ratio",
    "total_collapsed_subdeterminant_evaluations",
    "total_collapsed_dp_transition_count",
    "total_one_electron_subdeterminant_evaluations",
    "total_explicit_one_leaf_one_electron_subdeterminant_evaluations",
    "elapsed_wall_time_seconds",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep edge thresholds for analyze_star_separator_one_electron_dataset "
            "and emit a TSV summary."
        )
    )
    parser.add_argument("input_path", help="Path to the .xmi input file.")
    parser.add_argument(
        "--binary",
        default="build/src/analyze_star_separator_one_electron_dataset",
        help="Analyzer executable to run.",
    )
    parser.add_argument(
        "--threshold",
        dest="thresholds",
        action="append",
        type=float,
        required=True,
        help="Edge max-abs threshold to test. Repeat this flag for multiple values.",
    )
    parser.add_argument(
        "--active-overlap-source",
        default="input",
        choices=["input", "optimized_vbscf"],
        help="Active overlap source passed to the analyzer.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=50,
        help="Maximum number of structure pairs to analyze.",
    )
    parser.add_argument(
        "--left-structure",
        type=int,
        default=None,
        help="Restrict analysis to one left structure index.",
    )
    parser.add_argument(
        "--right-structure",
        type=int,
        default=None,
        help="Restrict analysis to one right structure index.",
    )
    parser.add_argument(
        "--top-examples",
        type=int,
        default=1,
        help="Number of examples requested from the analyzer.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=180.0,
        help="Per-threshold timeout for the analyzer.",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="Output TSV path. Use '-' for stdout.",
    )
    return parser.parse_args()


def parse_key_value_lines(stdout: str) -> Dict[str, str]:
    metrics: Dict[str, str] = {}
    for line in stdout.splitlines():
        if " = " not in line:
            continue
        key, value = line.split(" = ", 1)
        metrics[key.strip()] = value.strip()
    return metrics


def to_float(metrics: Dict[str, str], key: str) -> float:
    value = metrics.get(key)
    if value is None or value == "":
        return float("nan")
    return float(value)


def to_int(metrics: Dict[str, str], key: str) -> int:
    value = metrics.get(key)
    if value is None or value == "":
        return -1
    return int(float(value))


def run_threshold(args: argparse.Namespace, threshold: float) -> Dict[str, str]:
    command = [
        args.binary,
        args.input_path,
        "--top-examples",
        str(args.top_examples),
        "--active-overlap-source",
        args.active_overlap_source,
        "--edge-max-abs-threshold",
        f"{threshold:.16g}",
    ]
    if args.left_structure is not None or args.right_structure is not None:
        if args.left_structure is None or args.right_structure is None:
            raise ValueError(
                "--left-structure and --right-structure must be provided together"
            )
        command.extend(["--left-structure", str(args.left_structure)])
        command.extend(["--right-structure", str(args.right_structure)])
    else:
        command.extend(["--max-pairs", str(args.max_pairs)])
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")

    print(
        f"[threshold={threshold:.16g}] running {' '.join(command)}",
        file=sys.stderr,
        flush=True,
    )
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=args.timeout_seconds,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {
            "threshold": f"{threshold:.16g}",
            "status": "timeout",
        }

    metrics = parse_key_value_lines(completed.stdout)
    row: Dict[str, str] = {
        "threshold": f"{threshold:.16g}",
        "status": "ok" if completed.returncode == 0 else f"exit_{completed.returncode}",
    }
    for key in DEFAULT_METRICS:
        row[key] = metrics.get(key, "")

    unique_reference_pair_count = to_int(metrics, "unique_reference_determinant_pair_count")
    unique_leaf_message_state_count = to_int(metrics, "unique_exact_leaf_message_state_count")
    unique_merge_state_count = to_int(metrics, "unique_exact_merge_state_count")
    layered_state_total = unique_leaf_message_state_count + unique_merge_state_count
    row["unique_layered_state_total"] = str(layered_state_total)
    if unique_reference_pair_count > 0:
        saved_count = unique_reference_pair_count - layered_state_total
        saved_fraction = 1.0 - (layered_state_total / unique_reference_pair_count)
        row["unique_layered_state_saved_count"] = str(saved_count)
        row["unique_layered_state_saved_fraction"] = f"{saved_fraction:.16g}"
    else:
        row["unique_layered_state_saved_count"] = ""
        row["unique_layered_state_saved_fraction"] = ""

    selected_pairs = to_int(metrics, "selected_pairs")
    covered_star_pair_count = to_int(metrics, "covered_star_pair_count")
    if selected_pairs > 0:
        row["covered_star_pair_fraction"] = f"{covered_star_pair_count / selected_pairs:.16g}"
    else:
        row["covered_star_pair_fraction"] = ""

    exact_subdet = to_float(metrics, "total_one_electron_subdeterminant_evaluations")
    explicit_subdet = to_float(
        metrics,
        "total_explicit_one_leaf_one_electron_subdeterminant_evaluations",
    )
    if explicit_subdet > 0.0:
        row["one_electron_subdet_saved_fraction"] = f"{1.0 - (exact_subdet / explicit_subdet):.16g}"
    else:
        row["one_electron_subdet_saved_fraction"] = ""

    row["stderr"] = completed.stderr.strip().replace("\n", "\\n")
    if completed.returncode != 0 and not metrics:
        return row

    return row


def ordered_thresholds(values: Iterable[float]) -> List[float]:
    return sorted(set(values))


def write_rows(rows: List[Dict[str, str]], output_path: str) -> None:
    fieldnames = [
        "threshold",
        "status",
        "selected_pairs",
        "covered_star_pair_count",
        "covered_star_pair_fraction",
        "one_electron_max_abs_error",
        "total_reference_determinant_pair_count",
        "total_local_term_pair_visits",
        "unique_reference_determinant_pair_count",
        "unique_exact_leaf_message_state_count",
        "unique_exact_merge_state_count",
        "unique_layered_state_total",
        "unique_layered_state_saved_count",
        "unique_layered_state_saved_fraction",
        "layered_state_total_over_unique_reference_pair_ratio",
        "total_collapsed_subdeterminant_evaluations",
        "total_collapsed_dp_transition_count",
        "total_one_electron_subdeterminant_evaluations",
        "total_explicit_one_leaf_one_electron_subdeterminant_evaluations",
        "one_electron_subdet_saved_fraction",
        "elapsed_wall_time_seconds",
        "stderr",
    ]
    if output_path == "-":
        output_stream = sys.stdout
        close_stream = False
    else:
        output_stream = open(output_path, "w", newline="", encoding="utf-8")
        close_stream = True

    try:
        writer = csv.DictWriter(output_stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    finally:
        if close_stream:
            output_stream.close()


def main() -> int:
    args = parse_arguments()
    rows = []
    for threshold in ordered_thresholds(args.thresholds):
        rows.append(run_threshold(args, threshold))
    write_rows(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
