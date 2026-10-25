#!/usr/bin/env python3
"""Compare GPU SCF summaries against saved CPU results, never execute a baseline."""

import argparse
import csv
import hashlib
from pathlib import Path


def compare(cpu_reference: Path, gpu_runs: list[Path], output: Path) -> None:
    with cpu_reference.open() as stream:
        cpu = {row["case"]: row for row in csv.DictReader(stream)}
    rows = []
    for run in gpu_runs:
        with (run / "summary.csv").open() as stream:
            for gpu in csv.DictReader(stream):
                key = "c12-" + gpu["case"]
                if key not in cpu:
                    continue
                baseline = cpu[key]
                input_hash = hashlib.sha256((run / gpu["case"] / "input.xmi").read_bytes()).hexdigest()
                if input_hash != baseline["input_sha256"]:
                    raise ValueError(f"{gpu['case']}: CPU/GPU inputs differ")
                rows.append({
                    "case": gpu["case"],
                    "cpu_reference": str(cpu_reference),
                    "gpu_run": str(run),
                    "input_sha256": input_hash,
                    "cpu_energy_hartree": baseline["old_energy_Eh"],
                    "gpu_energy_hartree": gpu["energy_hartree"],
                    "energy_difference_hartree": float(gpu["energy_hartree"]) -
                        float(baseline["old_energy_Eh"]),
                    "cpu_steps": baseline["old_steps"], "gpu_steps": gpu["iterations"],
                    "cpu_hvp": baseline["old_hvp"], "gpu_hvp": gpu["hvp"],
                    "cpu_scf_seconds": baseline["old_scf_s"],
                    "gpu_scf_seconds": gpu["scf_seconds"],
                    "cpu_process_seconds": baseline["old_total_s"],
                    "gpu_process_seconds": gpu["process_seconds"],
                    "gpu_over_cpu_process_time": float(gpu["process_seconds"]) /
                        float(baseline["old_total_s"]),
                    "cpu_node": baseline["old_node"],
                    "gpu_context_peak_gib": gpu["context_peak_gib"],
                    "gpu_sampled_process_peak_mib": gpu["sampled_process_gpu_peak_mib"],
                })
    if not rows:
        raise ValueError("no matching completed cases")
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-reference", required=True, type=Path)
    parser.add_argument("--gpu-run", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    compare(args.cpu_reference, args.gpu_run, args.output)
