#!/usr/bin/env python3
"""Run reproducible VBSCF optimizer benchmarks and preserve raw evidence."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import socket
import subprocess
import sys
from typing import Iterable


BACKENDS = {
    "tnhvp": "nonredundant_truncated_newton",
    "lbfgs": "nonredundant_lbfgspp",
}

SUMMARY_COLUMNS = [
    "system",
    "backend",
    "repeat",
    "exit_code",
    "converged",
    "termination_reason",
    "iterations",
    "initial_energy",
    "final_energy",
    "projected_gradient_inf",
    "projected_gradient_l2",
    "hvp_directions",
    "hvp_seconds",
    "kkt_solved",
    "kkt_total",
    "scf_seconds",
    "end_to_end_seconds",
    "external_seconds",
    "max_rss_kb",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(arguments: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=cwd, text=True, capture_output=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def report_value(text: str, label: str) -> str:
    match = re.search(rf"^{re.escape(label)}\s*:\s*(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def parse_float(value: str) -> str:
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", value)
    return match.group(0) if match else ""


def parse_run(text: str, exit_code: int) -> dict[str, str | int]:
    convergence = re.search(
        r"VBSCF (converged|did not converge) in\s+(\d+) iterations", text
    )
    kkt = re.search(
        r"^Inner solves meeting KKT target\s*:\s*(\d+)\s*/\s*(\d+)",
        text,
        re.MULTILINE,
    )
    return {
        "exit_code": exit_code,
        "converged": int(bool(convergence and convergence.group(1) == "converged")),
        "termination_reason": report_value(text, "Termination reason"),
        "iterations": convergence.group(2) if convergence else "",
        "initial_energy": parse_float(report_value(text, "Initial total energy")),
        "final_energy": parse_float(report_value(text, "Final total energy")),
        "projected_gradient_inf": parse_float(
            report_value(text, "Final projected |g|_inf")
        ),
        "projected_gradient_l2": parse_float(
            report_value(text, "Final projected |g|_2")
        ),
        "hvp_directions": parse_float(
            report_value(text, "Matrix-free HVP directions")
        ),
        "hvp_seconds": parse_float(
            report_value(text, "Matrix-free HVP wall time")
        ),
        "kkt_solved": kkt.group(1) if kkt else "",
        "kkt_total": kkt.group(2) if kkt else "",
        "scf_seconds": parse_float(report_value(text, "SCF iteration wall time")),
        "end_to_end_seconds": parse_float(
            report_value(text, "End-to-end wall time")
        ),
    }


def parse_time_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(errors="replace").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def write_summary(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_COLUMNS, dialect="excel-tab")
        writer.writeheader()
        writer.writerows(rows)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run versioned TNHVP/L-BFGS comparisons from VBSCF input decks."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--executable", type=Path, default=Path("build/src/xmvb-cpp.exe")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backends", nargs="+", choices=BACKENDS, default=["tnhvp", "lbfgs"]
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--omp-threads", type=int, default=1)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--gradient-tolerance", type=float)
    parser.add_argument("--energy-tolerance", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    repository = Path(__file__).resolve().parents[1]
    executable = args.executable.resolve()
    output = args.output.resolve()
    inputs = [path.resolve() for path in args.inputs]
    if not executable.is_file():
        raise SystemExit(f"executable does not exist: {executable}")
    if args.repeats <= 0 or args.omp_threads <= 0 or args.blas_threads <= 0:
        raise SystemExit("repeats and thread counts must be positive")
    missing = [path for path in inputs if not path.is_file()]
    if missing:
        raise SystemExit(f"input does not exist: {missing[0]}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": str(args.omp_threads),
            "OPENBLAS_NUM_THREADS": str(args.blas_threads),
            "GOTO_NUM_THREADS": str(args.blas_threads),
            "MKL_NUM_THREADS": str(args.blas_threads),
            "MKL_DYNAMIC": "FALSE",
            "OMP_PROC_BIND": "close",
            "OMP_PLACES": "cores",
        }
    )
    manifest = {
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": git_value(["rev-parse", "HEAD"], repository),
        "git_status": git_value(["status", "--short"], repository),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": str(executable),
        "executable_sha256": sha256(executable),
        "inputs": [
            {"path": str(path), "sha256": sha256(path)} for path in inputs
        ],
        "backends": args.backends,
        "repeats": args.repeats,
        "omp_threads": args.omp_threads,
        "blas_threads": args.blas_threads,
        "max_iterations": args.max_iterations,
        "gradient_tolerance": args.gradient_tolerance,
        "energy_tolerance": args.energy_tolerance,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    rows: list[dict[str, object]] = []
    any_failed = False
    for input_path in inputs:
        for short_backend in args.backends:
            for repeat in range(1, args.repeats + 1):
                run_name = f"repeat_{repeat:03d}"
                run_dir = output / input_path.stem / short_backend
                run_dir.mkdir(parents=True, exist_ok=True)
                stdout_path = run_dir / f"{run_name}.out"
                stderr_path = run_dir / f"{run_name}.err"
                time_path = run_dir / f"{run_name}.time"
                command_path = run_dir / f"{run_name}.command"
                tnhvp_path = run_dir / f"{run_name}.tnhvp.tsv"
                command = [
                    "/usr/bin/time",
                    "-f",
                    "external_seconds=%e\nmax_rss_kb=%M",
                    "-o",
                    str(time_path),
                    str(executable),
                    str(input_path),
                    "--optimizer-backend",
                    BACKENDS[short_backend],
                ]
                if short_backend == "tnhvp":
                    command += [
                        "--nonredundant-truncated-newton-hvp-mode",
                        "exact_ctx",
                        "--tnhvp-trace",
                        str(tnhvp_path),
                    ]
                for option, value in (
                    ("--max-iterations", args.max_iterations),
                    ("--gradient-tolerance", args.gradient_tolerance),
                    ("--energy-tolerance", args.energy_tolerance),
                ):
                    if value is not None:
                        command += [option, str(value)]

                print(shlex.join(command), flush=True)
                command_path.write_text(shlex.join(command) + "\n")
                with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
                    completed = subprocess.run(
                        command,
                        cwd=repository,
                        env=environment,
                        stdout=stdout,
                        stderr=stderr,
                        check=False,
                    )
                text = stdout_path.read_text(errors="replace")
                row: dict[str, object] = {
                    "system": input_path.stem,
                    "backend": short_backend,
                    "repeat": repeat,
                    **parse_run(text, completed.returncode),
                }
                timing = parse_time_file(time_path)
                row["external_seconds"] = timing.get("external_seconds", "")
                row["max_rss_kb"] = timing.get("max_rss_kb", "")
                rows.append(row)
                write_summary(output / "summary.tsv", rows)
                any_failed = any_failed or completed.returncode != 0

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
