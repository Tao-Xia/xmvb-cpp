#!/usr/bin/env python3
"""Check paired, same-input CPU/GPU C12 CASCI records."""

import argparse
import csv
from pathlib import Path
import re
import sys


def fields(path):
    result = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def rss_gib(path):
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)",
                      path.read_text())
    return int(match[1]) / 1048576 if match else None


def process_gpu_peak_gib(path, pid):
    """NVML process memory includes CUDA context and library allocations."""
    if not path.exists():
        return None
    samples = []
    for line in path.read_text().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3 and parts[1] == pid:
            try:
                samples.append(float(parts[2]))
            except ValueError:
                pass
    return max(samples) / 1024 if samples else None


def summarize(root):
    for case in ("mixed3", "mixed20", "cov", "ion02", "full"):
        folder = root / case
        cpu = fields(folder / "cpu.out")
        gpu = fields(folder / "gpu.out")
        check = fields(folder / "check.out")
        if cpu.get("mode") != "--cpu" or gpu.get("mode") != "--gpu":
            raise ValueError(f"{case}: missing CPU/GPU solve")
        if check.get("mode") != "--check":
            raise ValueError(f"{case}: missing H/S block check")
        for key in ("structures",):
            if cpu[key] != gpu[key] or cpu[key] != check[key]:
                raise ValueError(f"{case}: different {key} in paired runs")
        offsets = [float(run["energy_offset_hartree"]) for run in (cpu,gpu,check)]
        if max(offsets) - min(offsets) > 1e-10:
            raise ValueError(f"{case}: inconsistent inactive/nuclear energy offset")
        cpu_energy = float(cpu["total_energy_hartree"])
        gpu_energy = float(gpu["total_energy_hartree"])
        gap = abs(cpu_energy - gpu_energy)
        if gap > 5e-7:
            raise ValueError(f"{case}: CPU/GPU total energy gap {gap:.3g} Eh")
        if float(gpu["ci_stationarity_residual_hartree"]) > 1.01e-7:
            raise ValueError(f"{case}: GPU CI stationarity budget not met")
        yield dict(case=case, structures=cpu["structures"],
                   cpu_total_energy_hartree=cpu_energy,
                   gpu_total_energy_hartree=gpu_energy,
                   absolute_energy_gap_hartree=gap,
                   h_block_relative_error=check["h_block_relative_error"],
                   s_block_relative_error=check["s_block_relative_error"],
                   cpu_end_to_end_seconds=cpu["benchmark_elapsed_seconds"],
                   gpu_end_to_end_seconds=gpu["benchmark_elapsed_seconds"],
                   cpu_ci_solve_seconds=cpu["ci_solve_seconds"],
                   gpu_ci_solve_seconds=gpu["ci_solve_seconds"],
                   cpu_actions=cpu["ci_block_actions"],
                   gpu_actions=gpu["ci_block_actions"],
                   cpu_iterations=cpu["ci_iterations"],
                   gpu_iterations=gpu["ci_iterations"],
                   gpu_peak_device_gib=int(gpu["gpu_peak_device_bytes"]) / 1073741824,
                   gpu_sampled_process_peak_gib=process_gpu_peak_gib(
                       folder / "gpu_samples.csv", gpu.get("process_id")),
                   cpu_peak_rss_gib=rss_gib(folder / "cpu.time"),
                   gpu_peak_rss_gib=rss_gib(folder / "gpu.time"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    rows = list(summarize(args.run_root))
    writer = csv.DictWriter(sys.stdout, fieldnames=rows[0], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
