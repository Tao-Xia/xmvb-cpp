#!/usr/bin/env python3
"""Summarize GPU physical-derivative diagnostics; never label them SCF runs."""

import argparse
import csv
from pathlib import Path
import re


def summarize(root: Path, output: Path) -> None:
    rows = []
    for case in ("mixed20", "cov", "ion02", "full"):
        directory = root / case
        data = dict(line.split("=", 1) for line in
                    (directory / "gpu.out").read_text().splitlines() if "=" in line)
        if data.get("point_scope") != "physical_derivatives_not_scf":
            raise ValueError(f"{case}: missing successful derivative diagnostic")
        pid = data["process_id"]
        samples = []
        for fields in csv.reader((directory / "gpu_samples.csv").read_text().splitlines()):
            if len(fields) == 3 and fields[1].strip() == pid:
                samples.append(float(fields[2]))
        rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)",
                        (directory / "gpu.time").read_text())
        row = {
            "case": case,
            "scope": "physical_derivatives_not_scf",
            "structures": data["structures"],
            "energy_hartree": data["point_total_energy_hartree"],
            "energy_gradient_seconds": data["point_energy_gradient_seconds"],
            "joint_hvp_seconds": data["point_joint_hvp_seconds"],
            "gradient_difference_error": data["point_gradient_difference_error"],
            "hvp_difference_error": data["point_hvp_difference_error"],
            "context_peak_gib": int(data["gpu_peak_device_bytes"]) / 1024**3,
            "sampled_process_gpu_peak_mib": max(samples) if samples else "",
            "host_peak_rss_mib": int(rss[1]) / 1024 if rss else "",
            "hvp_context_h2d_bytes": data["point_hvp_context_h2d_bytes"],
            "hvp_context_d2h_bytes": data["point_hvp_context_d2h_bytes"],
        }
        rows.append(row)
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
