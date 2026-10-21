#!/usr/bin/env python3
"""Compare production GPU CI with saved prototype GPU records, not SCF runs."""

import argparse
import csv
from pathlib import Path
import re
import sys

from summarize import fields, process_gpu_peak_gib


def summarize(reference, current):
    for case in ("mixed20", "cov", "ion02", "full"):
        old = fields(reference / case / "gpu.out")
        new = fields(current / case / "gpu.out")
        if old["structures"] != new["structures"]:
            raise ValueError(f"{case}: selected spaces differ")
        error = abs(float(old["total_energy_hartree"]) -
                    float(new["total_energy_hartree"]))
        if error > 1e-7 or float(new["ci_stationarity_residual_hartree"]) > 1e-7:
            raise ValueError(f"{case}: requested CI accuracy is not met")
        memory = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)",
                           (current / case / "gpu.time").read_text())
        yield dict(
            case=case, scope="frozen_orbital_ci", structures=new["structures"],
            gpu_energy=new["total_energy_hartree"], energy_error_hartree=error,
            prototype_actions=old["ci_block_actions"], actions=new["ci_block_actions"],
            prototype_ci_seconds=old["ci_solve_seconds"], ci_seconds=new["ci_solve_seconds"],
            model_setup_seconds=new["ci_model_init_seconds"],
            prototype_process_seconds=old["benchmark_elapsed_seconds"],
            process_seconds=new["benchmark_elapsed_seconds"],
            preconditioner=new["ci_preconditioner"], identity_overlap=new["ci_identity_overlap"],
            stationarity_hartree=new["ci_stationarity_residual_hartree"],
            device_payload_mib=float(new["gpu_peak_device_bytes"]) / 2**20,
            sampled_process_vram_gib=process_gpu_peak_gib(
                current / case / "gpu_samples.csv", new["process_id"]),
            host_rss_mib=int(memory[1]) / 1024 if memory else "missing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("current", type=Path)
    args = parser.parse_args()
    rows = list(summarize(args.reference, args.current))
    writer = csv.DictWriter(sys.stdout, fieldnames=rows[0], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
