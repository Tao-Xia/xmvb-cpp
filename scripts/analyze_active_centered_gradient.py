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
class StaticMetadata:
    n_atoms: int
    n_basis_functions: int
    n_orbitals: int
    n_active_orbitals: int
    active_orbital_start_index: int


@dataclass
class StepSummary:
    step_name: str
    accepted_iteration_index: int
    total_energy: float
    active_atom_count: int
    frontier_orbital_count: int
    active_share: float
    frontier_share: float
    other_share: float
    active_plus_frontier_share: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze accepted-iteration orbital-gradient concentration in an xmvb-cpp "
            "trace dataset and report active/frontier/other shares."
        )
    )
    parser.add_argument("sample_dir", type=Path, help="Trace sample directory")
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
        help="Analyze at most this many steps after stride filtering. Default: 0 = all",
    )
    parser.add_argument(
        "--active-atom-weight-threshold",
        type=float,
        default=0.05,
        help=(
            "Atoms carrying at least this fraction of total active-orbital squared weight "
            "are considered active-centered. Default: 0.05"
        ),
    )
    parser.add_argument(
        "--min-active-atoms",
        type=int,
        default=1,
        help="Ensure at least this many active-centered atoms are kept. Default: 1",
    )
    parser.add_argument(
        "--frontier-orbital-overlap-threshold",
        type=float,
        default=0.20,
        help=(
            "A non-active orbital is labeled frontier when this fraction of its squared "
            "coefficient weight lies on active-centered atoms. Default: 0.20"
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path for per-step CSV output",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_static_metadata(static_dir: Path) -> StaticMetadata:
    metadata = load_json(static_dir / "metadata.json")
    return StaticMetadata(
        n_atoms=int(metadata["n_atoms"]),
        n_basis_functions=int(metadata["n_basis_functions"]),
        n_orbitals=int(metadata["n_orbitals"]),
        n_active_orbitals=int(metadata["n_active_orbitals"]),
        active_orbital_start_index=int(metadata["active_orbital_start_index"]),
    )


def load_i32(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.int32)


def load_f64(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float64)


def reshape_orbital_major(values: np.ndarray, n_orbitals: int, n_basis_functions: int) -> np.ndarray:
    expected_size = n_orbitals * n_basis_functions
    if values.size != expected_size:
        raise ValueError(f"{expected_size=} does not match {values.size=} for orbital-major matrix")
    return values.reshape((n_orbitals, n_basis_functions))


def build_orbital_atom_weights(
    orbital_value_table: np.ndarray,
    orbital_basis_index_table: np.ndarray,
    orbital_basis_counts: np.ndarray,
    ao_to_atom: np.ndarray,
) -> np.ndarray:
    n_orbitals, n_basis_functions = orbital_value_table.shape
    orbital_atom_weights = np.zeros((n_orbitals, int(np.max(ao_to_atom)) + 1), dtype=np.float64)
    orbital_basis_index_table = orbital_basis_index_table.reshape((n_orbitals, n_basis_functions))
    for orbital_index in range(n_orbitals):
        coefficient_count = int(orbital_basis_counts[orbital_index])
        for coefficient_index in range(coefficient_count):
            basis_function_index = int(orbital_basis_index_table[orbital_index, coefficient_index])
            if basis_function_index <= 0:
                continue
            atom_index = int(ao_to_atom[basis_function_index - 1])
            coefficient = orbital_value_table[orbital_index, coefficient_index]
            orbital_atom_weights[orbital_index, atom_index] += coefficient * coefficient
    return orbital_atom_weights


def select_active_atoms(
    active_atom_weights: np.ndarray,
    threshold: float,
    min_active_atoms: int,
) -> np.ndarray:
    total_weight = float(np.sum(active_atom_weights))
    if total_weight <= 0.0:
        return np.arange(min(max(min_active_atoms, 0), active_atom_weights.size), dtype=np.int32)

    normalized = active_atom_weights / total_weight
    selected = np.flatnonzero(normalized >= threshold).astype(np.int32)
    if selected.size >= min_active_atoms:
        return selected

    ranked = np.argsort(normalized)[::-1]
    keep_count = min(max(min_active_atoms, 1), normalized.size)
    return np.sort(ranked[:keep_count].astype(np.int32))


def classify_orbitals(
    orbital_atom_weights: np.ndarray,
    active_orbital_start_index: int,
    n_active_orbitals: int,
    active_atoms: np.ndarray,
    frontier_orbital_overlap_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_orbitals = orbital_atom_weights.shape[0]
    active_mask = np.zeros(n_orbitals, dtype=bool)
    active_stop_index = active_orbital_start_index + n_active_orbitals
    active_mask[active_orbital_start_index:active_stop_index] = True

    frontier_mask = np.zeros(n_orbitals, dtype=bool)
    active_atom_mask = np.zeros(orbital_atom_weights.shape[1], dtype=bool)
    active_atom_mask[active_atoms] = True

    for orbital_index in range(n_orbitals):
        if active_mask[orbital_index]:
            continue
        total_weight = float(np.sum(orbital_atom_weights[orbital_index]))
        if total_weight <= 0.0:
            continue
        active_atom_weight = float(np.sum(orbital_atom_weights[orbital_index, active_atom_mask]))
        if active_atom_weight / total_weight >= frontier_orbital_overlap_threshold:
            frontier_mask[orbital_index] = True

    other_mask = ~(active_mask | frontier_mask)
    return active_mask, frontier_mask, other_mask


def compute_share(mask: np.ndarray, per_orbital_gradient_norm_sq: np.ndarray, total_gradient_norm_sq: float) -> float:
    if total_gradient_norm_sq <= 0.0:
        return 0.0
    return float(np.sum(per_orbital_gradient_norm_sq[mask]) / total_gradient_norm_sq)


def iter_step_dirs(steps_dir: Path, step_stride: int, max_steps: int) -> Iterable[Path]:
    step_dirs = sorted(path for path in steps_dir.iterdir() if path.is_dir() and path.name.startswith("step_"))
    filtered = step_dirs[::step_stride]
    if max_steps > 0:
        filtered = filtered[:max_steps]
    return filtered


def summarize_sample(
    sample_dir: Path,
    step_stride: int,
    max_steps: int,
    active_atom_weight_threshold: float,
    min_active_atoms: int,
    frontier_orbital_overlap_threshold: float,
) -> list[StepSummary]:
    static_dir = sample_dir / "static"
    steps_dir = sample_dir / "steps"
    static_metadata = load_static_metadata(static_dir)

    orbital_basis_index_table = load_i32(static_dir / "orbital_basis_index_table_i32.bin")
    orbital_basis_counts = load_i32(static_dir / "orbital_basis_counts_i32.bin")
    ao_to_atom = load_i32(static_dir / "ao_to_atom_i32.bin")

    summaries: list[StepSummary] = []
    for step_dir in iter_step_dirs(steps_dir, step_stride, max_steps):
        step_metadata = load_json(step_dir / "metadata.json")
        orbital_value_table = reshape_orbital_major(
            load_f64(step_dir / "orbital_value_table_f64.bin"),
            static_metadata.n_orbitals,
            static_metadata.n_basis_functions,
        )
        orbital_gradient = reshape_orbital_major(
            load_f64(step_dir / "sparse_orbital_energy_gradient_f64.bin"),
            static_metadata.n_orbitals,
            static_metadata.n_basis_functions,
        )

        orbital_atom_weights = build_orbital_atom_weights(
            orbital_value_table,
            orbital_basis_index_table,
            orbital_basis_counts,
            ao_to_atom,
        )
        active_slice = slice(
            static_metadata.active_orbital_start_index,
            static_metadata.active_orbital_start_index + static_metadata.n_active_orbitals,
        )
        active_atoms = select_active_atoms(
            np.sum(orbital_atom_weights[active_slice], axis=0),
            active_atom_weight_threshold,
            min_active_atoms,
        )
        active_mask, frontier_mask, other_mask = classify_orbitals(
            orbital_atom_weights,
            static_metadata.active_orbital_start_index,
            static_metadata.n_active_orbitals,
            active_atoms,
            frontier_orbital_overlap_threshold,
        )

        per_orbital_gradient_norm_sq = np.sum(np.square(orbital_gradient), axis=1)
        total_gradient_norm_sq = float(np.sum(per_orbital_gradient_norm_sq))
        active_share = compute_share(active_mask, per_orbital_gradient_norm_sq, total_gradient_norm_sq)
        frontier_share = compute_share(frontier_mask, per_orbital_gradient_norm_sq, total_gradient_norm_sq)
        other_share = compute_share(other_mask, per_orbital_gradient_norm_sq, total_gradient_norm_sq)

        summaries.append(
            StepSummary(
                step_name=step_dir.name,
                accepted_iteration_index=int(step_metadata["accepted_iteration_index"]),
                total_energy=float(step_metadata["total_energy"]),
                active_atom_count=int(active_atoms.size),
                frontier_orbital_count=int(np.count_nonzero(frontier_mask)),
                active_share=active_share,
                frontier_share=frontier_share,
                other_share=other_share,
                active_plus_frontier_share=active_share + frontier_share,
            )
        )
    return summaries


def write_csv(path: Path, summaries: list[StepSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "step_name",
                "accepted_iteration_index",
                "total_energy",
                "active_atom_count",
                "frontier_orbital_count",
                "active_share",
                "frontier_share",
                "other_share",
                "active_plus_frontier_share",
            ]
        )
        for summary in summaries:
            writer.writerow(
                [
                    summary.step_name,
                    summary.accepted_iteration_index,
                    f"{summary.total_energy:.15f}",
                    summary.active_atom_count,
                    summary.frontier_orbital_count,
                    f"{summary.active_share:.10f}",
                    f"{summary.frontier_share:.10f}",
                    f"{summary.other_share:.10f}",
                    f"{summary.active_plus_frontier_share:.10f}",
                ]
            )


def print_report(sample_dir: Path, summaries: list[StepSummary]) -> None:
    if not summaries:
        raise RuntimeError(f"no accepted steps found under {sample_dir}")

    print(f"sample_dir = {sample_dir}")
    print("step_name  iter   energy              active_atoms  frontier_orbs  active_share  frontier_share  other_share  active_plus_frontier")
    for summary in summaries:
        print(
            f"{summary.step_name:>9}  "
            f"{summary.accepted_iteration_index:>4d}  "
            f"{summary.total_energy:>18.12f}  "
            f"{summary.active_atom_count:>12d}  "
            f"{summary.frontier_orbital_count:>14d}  "
            f"{summary.active_share:>12.6f}  "
            f"{summary.frontier_share:>14.6f}  "
            f"{summary.other_share:>11.6f}  "
            f"{summary.active_plus_frontier_share:>21.6f}"
        )

    active_values = np.array([summary.active_share for summary in summaries], dtype=np.float64)
    frontier_values = np.array([summary.frontier_share for summary in summaries], dtype=np.float64)
    combined_values = np.array(
        [summary.active_plus_frontier_share for summary in summaries],
        dtype=np.float64,
    )
    print("summary_mean_active_share = {:.6f}".format(float(np.mean(active_values))))
    print("summary_mean_frontier_share = {:.6f}".format(float(np.mean(frontier_values))))
    print("summary_mean_active_plus_frontier_share = {:.6f}".format(float(np.mean(combined_values))))
    print("summary_median_active_plus_frontier_share = {:.6f}".format(float(np.median(combined_values))))
    print("summary_min_active_plus_frontier_share = {:.6f}".format(float(np.min(combined_values))))
    print("summary_max_active_plus_frontier_share = {:.6f}".format(float(np.max(combined_values))))


def main() -> int:
    args = parse_args()
    sample_dir = args.sample_dir.resolve()
    summaries = summarize_sample(
        sample_dir=sample_dir,
        step_stride=args.step_stride,
        max_steps=args.max_steps,
        active_atom_weight_threshold=args.active_atom_weight_threshold,
        min_active_atoms=args.min_active_atoms,
        frontier_orbital_overlap_threshold=args.frontier_orbital_overlap_threshold,
    )
    if args.csv is not None:
        write_csv(args.csv.resolve(), summaries)
    print_report(sample_dir, summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
