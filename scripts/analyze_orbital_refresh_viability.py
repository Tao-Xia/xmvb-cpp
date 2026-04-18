#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class StepData:
    step_name: str
    accepted_iteration_index: int
    energy: float
    parameter_vector: np.ndarray
    gradient_vector: np.ndarray


def parse_int_list(text: str) -> list[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise argparse.ArgumentTypeError("list must contain at least one integer")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline diagnostics for stale-gradient and low-rank orbital-update "
            "ideas using an xmvb-cpp accepted-iteration trace."
        )
    )
    parser.add_argument("sample_dir", type=Path, help="Trace sample directory")
    parser.add_argument(
        "--lags",
        type=parse_int_list,
        default=[1, 2, 3],
        help="Comma-separated lag list for coherence/model-fidelity checks. Default: 1,2,3",
    )
    parser.add_argument(
        "--subspace-ranks",
        type=parse_int_list,
        default=[4, 8, 12, 16],
        help="Comma-separated ranks for recent-update subspace projection diagnostics. Default: 4,8,12,16",
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=8,
        help="Number of previous accepted updates used to build the recent-update subspace. Default: 8",
    )
    parser.add_argument(
        "--step-stride",
        type=int,
        default=1,
        help="Only analyze every N-th accepted step. Default: 1",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Analyze at most this many accepted steps after stride filtering. Default: 0 = all",
    )
    parser.add_argument(
        "--energy-delta-floor",
        type=float,
        default=1.0e-12,
        help="Stabilizer for relative energy-change errors. Default: 1e-12",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path for per-lag diagnostics",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_step_dirs(steps_dir: Path, step_stride: int, max_steps: int) -> Iterable[Path]:
    step_dirs = sorted(path for path in steps_dir.iterdir() if path.is_dir() and path.name.startswith("step_"))
    filtered = step_dirs[::step_stride]
    if max_steps > 0:
        filtered = filtered[:max_steps]
    return filtered


def load_step_data(sample_dir: Path, step_stride: int, max_steps: int) -> list[StepData]:
    static_dir = sample_dir / "static"
    steps_dir = sample_dir / "steps"
    differentiable_parameter_indices = np.fromfile(
        static_dir / "differentiable_parameter_indices_i32.bin",
        dtype=np.int32,
    )
    if differentiable_parameter_indices.size == 0:
        raise RuntimeError("differentiable_parameter_indices_i32.bin is empty")

    step_data: list[StepData] = []
    for step_dir in iter_step_dirs(steps_dir, step_stride, max_steps):
        metadata = load_json(step_dir / "metadata.json")
        orbital_value_table = np.fromfile(
            step_dir / "orbital_value_table_f64.bin",
            dtype=np.float64,
        )
        gradient_vector_full = np.fromfile(
            step_dir / "sparse_orbital_energy_gradient_f64.bin",
            dtype=np.float64,
        )
        if orbital_value_table.size != gradient_vector_full.size:
            raise RuntimeError(
                f"orbital value and gradient sizes differ in {step_dir}: "
                f"{orbital_value_table.size} vs {gradient_vector_full.size}"
            )
        parameter_vector = orbital_value_table[differentiable_parameter_indices]
        gradient_vector = gradient_vector_full[differentiable_parameter_indices]
        step_data.append(
            StepData(
                step_name=step_dir.name,
                accepted_iteration_index=int(metadata["accepted_iteration_index"]),
                energy=float(metadata["total_energy"]),
                parameter_vector=parameter_vector,
                gradient_vector=gradient_vector,
            )
        )
    if len(step_data) < 3:
        raise RuntimeError("need at least three accepted steps for refresh diagnostics")
    return step_data


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": float(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def format_summary(prefix: str, values: list[float]) -> list[str]:
    stats = summarize(values)
    return [
        f"{prefix}_count = {int(stats['count'])}",
        f"{prefix}_mean = {stats['mean']:.6f}",
        f"{prefix}_median = {stats['median']:.6f}",
        f"{prefix}_min = {stats['min']:.6f}",
        f"{prefix}_max = {stats['max']:.6f}",
    ]


def rank_for_threshold(singular_values: np.ndarray, threshold: float) -> int:
    if singular_values.size == 0:
        return 0
    energy = np.square(singular_values)
    total = float(np.sum(energy))
    if total <= 0.0:
        return 0
    cumulative = np.cumsum(energy) / total
    return int(np.searchsorted(cumulative, threshold, side="left") + 1)


def build_update_matrix(step_data: list[StepData]) -> np.ndarray:
    updates = [
        step_data[index + 1].parameter_vector - step_data[index].parameter_vector
        for index in range(len(step_data) - 1)
    ]
    return np.column_stack(updates)


def analyze_lagged_coherence_and_model_fidelity(
    step_data: list[StepData],
    lags: list[int],
    energy_delta_floor: float,
) -> tuple[dict[int, list[float]], dict[int, list[float]], dict[int, list[float]]]:
    coherence_by_lag: dict[int, list[float]] = {lag: [] for lag in lags}
    model_relative_error_by_lag: dict[int, list[float]] = {lag: [] for lag in lags}
    model_sign_agreement_by_lag: dict[int, list[float]] = {lag: [] for lag in lags}

    for lag in lags:
        for index in range(len(step_data) - lag):
            current = step_data[index]
            future = step_data[index + lag]
            coherence_by_lag[lag].append(
                cosine_similarity(current.gradient_vector, future.gradient_vector)
            )
            parameter_delta = future.parameter_vector - current.parameter_vector
            energy_delta = future.energy - current.energy
            predicted_energy_delta = float(np.dot(current.gradient_vector, parameter_delta))
            relative_error = abs(energy_delta - predicted_energy_delta) / (
                abs(energy_delta) + energy_delta_floor
            )
            model_relative_error_by_lag[lag].append(relative_error)
            sign_agreement = 1.0 if predicted_energy_delta * energy_delta > 0.0 else 0.0
            model_sign_agreement_by_lag[lag].append(sign_agreement)

    return coherence_by_lag, model_relative_error_by_lag, model_sign_agreement_by_lag


def analyze_recent_update_subspace(
    step_data: list[StepData],
    subspace_ranks: list[int],
    history_window: int,
) -> dict[int, list[float]]:
    capture_by_rank: dict[int, list[float]] = {rank: [] for rank in subspace_ranks}
    updates = [
        step_data[index + 1].parameter_vector - step_data[index].parameter_vector
        for index in range(len(step_data) - 1)
    ]

    for gradient_index in range(1, len(step_data)):
        history_end = gradient_index - 1
        history_begin = max(0, history_end - history_window + 1)
        history_updates = updates[history_begin : history_end + 1]
        if not history_updates:
            continue
        update_matrix = np.column_stack(history_updates)
        left_vectors, singular_values, _ = np.linalg.svd(update_matrix, full_matrices=False)
        if singular_values.size == 0:
            continue
        gradient_vector = step_data[gradient_index].gradient_vector
        gradient_norm_sq = float(np.dot(gradient_vector, gradient_vector))
        if gradient_norm_sq <= 0.0:
            continue
        for rank in subspace_ranks:
            effective_rank = min(rank, left_vectors.shape[1])
            basis = left_vectors[:, :effective_rank]
            projected_gradient = basis.T @ gradient_vector
            capture = float(np.dot(projected_gradient, projected_gradient) / gradient_norm_sq)
            capture_by_rank[rank].append(capture)
    return capture_by_rank


def write_csv(
    path: Path,
    step_data: list[StepData],
    lags: list[int],
    coherence_by_lag: dict[int, list[float]],
    model_relative_error_by_lag: dict[int, list[float]],
    model_sign_agreement_by_lag: dict[int, list[float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "start_step_name",
                "start_iter",
                "lag",
                "end_step_name",
                "end_iter",
                "gradient_cosine",
                "linear_model_relative_error",
                "linear_model_sign_agreement",
            ]
        )
        for lag in lags:
            for index in range(len(step_data) - lag):
                writer.writerow(
                    [
                        step_data[index].step_name,
                        step_data[index].accepted_iteration_index,
                        lag,
                        step_data[index + lag].step_name,
                        step_data[index + lag].accepted_iteration_index,
                        f"{coherence_by_lag[lag][index]:.10f}",
                        f"{model_relative_error_by_lag[lag][index]:.10f}",
                        f"{model_sign_agreement_by_lag[lag][index]:.10f}",
                    ]
                )


def main() -> int:
    args = parse_args()
    sample_dir = args.sample_dir.resolve()
    step_data = load_step_data(sample_dir, args.step_stride, args.max_steps)
    update_matrix = build_update_matrix(step_data)
    _, singular_values, _ = np.linalg.svd(update_matrix, full_matrices=False)

    coherence_by_lag, model_relative_error_by_lag, model_sign_agreement_by_lag = (
        analyze_lagged_coherence_and_model_fidelity(
            step_data,
            args.lags,
            args.energy_delta_floor,
        )
    )
    capture_by_rank = analyze_recent_update_subspace(
        step_data,
        args.subspace_ranks,
        args.history_window,
    )

    if args.csv is not None:
        write_csv(
            args.csv.resolve(),
            step_data,
            args.lags,
            coherence_by_lag,
            model_relative_error_by_lag,
            model_sign_agreement_by_lag,
        )

    print(f"sample_dir = {sample_dir}")
    print(f"accepted_step_count = {len(step_data)}")
    print(f"parameter_dimension = {step_data[0].parameter_vector.size}")
    print(f"history_window = {args.history_window}")
    print(f"rank_for_update_variance_0.80 = {rank_for_threshold(singular_values, 0.80)}")
    print(f"rank_for_update_variance_0.90 = {rank_for_threshold(singular_values, 0.90)}")
    print(f"rank_for_update_variance_0.95 = {rank_for_threshold(singular_values, 0.95)}")

    for lag in args.lags:
        for line in format_summary(f"gradient_cosine_lag_{lag}", coherence_by_lag[lag]):
            print(line)
        for line in format_summary(
            f"linear_model_relative_error_lag_{lag}",
            model_relative_error_by_lag[lag],
        ):
            print(line)
        sign_agreement_rate = float(np.mean(model_sign_agreement_by_lag[lag]))
        print(f"linear_model_sign_agreement_rate_lag_{lag} = {sign_agreement_rate:.6f}")

    for rank in args.subspace_ranks:
        values = capture_by_rank[rank]
        if not values:
            continue
        for line in format_summary(f"recent_update_gradient_capture_rank_{rank}", values):
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
