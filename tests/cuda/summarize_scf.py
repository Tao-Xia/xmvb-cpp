#!/usr/bin/env python3
"""Summarize completed GPU SCF runs without rerunning a CPU baseline."""

import argparse
import csv
from pathlib import Path
import re


def summarize(root: Path, output: Path) -> None:
    rows = []
    for case in ("mixed20", "cov", "ion02", "full"):
        directory = root / case
        if not directory.is_dir():
            continue
        data = dict(line.split("=", 1) for line in
                    (directory / "gpu.out").read_text().splitlines() if "=" in line)
        if data.get("converged") != "1":
            raise ValueError(f"{case}: SCF has not converged")
        if data.get("preconditioner") != "analytic_hessian_diagonal":
            raise ValueError(f"{case}: wrong preconditioner")
        samples = {}
        for fields in csv.reader((directory / "gpu_samples.csv").read_text().splitlines()):
            if len(fields) == 3:
                samples.setdefault(fields[1].strip(), []).append(float(fields[2]))
        # Do not attribute another process's memory to this calculation.
        pid_file = directory / "gpu.pid"
        if pid_file.exists():
            process_samples = samples.get(pid_file.read_text().strip(), [])
        elif len(samples) > 1:
            raise ValueError(f"{case}: ambiguous GPU process samples")
        else:
            process_samples = next(iter(samples.values()), [])
        peak = max(process_samples) if process_samples else ""
        rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)",
                        (directory / "gpu.time").read_text())
        rows.append({
            "case": case,
            "structures": data["structures"],
            "preconditioner": data["preconditioner"],
            "energy_hartree": data["total_energy_hartree"],
            "iterations": data["scf_iterations"],
            "hvp": data["orbital_hvp_actions"],
            "ci_actions": data["ci_block_actions"],
            "ci_prepared_points": data.get("ci_prepared_points", ""),
            "selected_metric_builds": data.get("selected_metric_builds", ""),
            "scf_seconds": data["scf_seconds"],
            "process_seconds": data["process_seconds"],
            "input_seconds": data.get("input_seconds", ""),
            "input_integral_provider_seconds": data.get("input_integral_provider_seconds", ""),
            "input_pair_graph_seconds": data.get("input_pair_graph_seconds", ""),
            "input_orbital_guess_seconds": data.get("input_orbital_guess_seconds", ""),
            "input_structure_expansion_seconds": data.get("input_structure_expansion_seconds", ""),
            "context_seconds": data.get("context_seconds", ""),
            "solve_call_seconds": data.get("solve_call_seconds", ""),
            "solver_teardown_seconds": data.get("solver_teardown_seconds", ""),
            "orbital_output_seconds": data.get("orbital_output_seconds", ""),
            "context_peak_gib": int(data["gpu_peak_device_bytes"]) / 1024**3,
            "sampled_process_gpu_peak_mib": peak,
            "host_peak_rss_mib": int(rss[1]) / 1024 if rss else "",
        })
    if not rows:
        raise ValueError("no SCF case directories found")
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summarize(args.run, args.output)
