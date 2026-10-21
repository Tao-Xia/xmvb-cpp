#!/usr/bin/env python3
"""Compare GPU-only C12 solves against saved CPU/GPU CI records."""

import argparse
import csv
from pathlib import Path
import sys

from summarize import fields, process_gpu_peak_gib


def summarize(reference, current):
    for case in ("mixed20", "cov", "ion02", "full"):
        cpu = fields(reference / case / "cpu.out")
        old = fields(reference / case / "gpu.out")
        gpu = fields(current / case / "gpu.out")
        if cpu["structures"] != gpu["structures"]:
            raise ValueError(f"{case}: different selected spaces")
        error = abs(float(cpu["total_energy_hartree"])-float(gpu["total_energy_hartree"]))
        if error > 5e-7 or float(gpu["ci_stationarity_residual_hartree"]) > 1.01e-7:
            raise ValueError(f"{case}: GPU CI accuracy contract not met")
        yield dict(case=case, structures=gpu["structures"],
                   cpu_reference_energy=cpu["total_energy_hartree"],
                   gpu_energy=gpu["total_energy_hartree"], energy_error=error,
                   cpu_reference_ci_seconds=cpu["ci_solve_seconds"],
                   old_gpu_ci_seconds=old["ci_solve_seconds"],
                   gpu_ci_seconds=gpu["ci_solve_seconds"],
                   old_gpu_actions=old["ci_block_actions"], gpu_actions=gpu["ci_block_actions"],
                   old_gpu_total_seconds=old["benchmark_elapsed_seconds"],
                   gpu_total_seconds=gpu["benchmark_elapsed_seconds"],
                   gpu_graph_build_seconds=gpu["gpu_graph_build_seconds"],
                   gpu_peak_payload_bytes=gpu["gpu_peak_device_bytes"],
                   gpu_sampled_process_peak_gib=process_gpu_peak_gib(
                       current / case / "gpu_samples.csv",gpu["process_id"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("current", type=Path)
    args = parser.parse_args()
    rows = list(summarize(args.reference,args.current))
    writer = csv.DictWriter(sys.stdout,fieldnames=rows[0],lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
