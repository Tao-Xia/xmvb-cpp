#!/usr/bin/env python3
"""Reuse a saved CPU phase baseline; compare native, resident GPU H/M actions."""

import argparse
import csv
from pathlib import Path
import sys

from summarize import fields


def summarize(baseline, current):
    for case in ("mixed20", "cov", "ion02", "full"):
        cpu = fields(baseline / case / "cpu_phases.out")
        previous = fields(baseline / case / "action.out")
        gpu = fields(current / case / "action.out")
        check = fields(current / case / "check.out")
        if cpu["structures"] != gpu["structures"]:
            raise ValueError(f"{case}: different selected spaces")
        for width in (1, 3):
            prefix = f"action_b{width}_"
            cpu_seconds = float(cpu[prefix + "cpu_seconds"])
            old_seconds = float(previous[prefix + "gpu_wall_seconds"])
            seconds = float(gpu[prefix + "gpu_wall_seconds"])
            if any(int(gpu[prefix + key]) for key in ("h2d_bytes", "d2h_bytes")):
                raise ValueError(f"{case}: transfer in resident timing window")
            row = dict(case=case, structures=cpu["structures"], block_width=width,
                       cpu_baseline_seconds=cpu_seconds,
                       gpu_before_seconds=old_seconds, gpu_after_seconds=seconds,
                       gpu_improvement=old_seconds / seconds,
                       cpu_to_gpu_ratio=cpu_seconds / seconds,
                       gpu_event_seconds=gpu[prefix + "gpu_event_seconds"],
                       h_relative_error=check["h_block_relative_error"],
                       m_relative_error=check["s_block_relative_error"],
                       gpu_peak_payload_bytes=gpu["gpu_peak_device_bytes"])
            for key, value in cpu.items():
                if key.startswith(prefix + "cpu_phase_"):
                    row[key.removeprefix(prefix)] = value
            yield row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    args = parser.parse_args()
    rows = list(summarize(args.baseline, args.current))
    columns = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(sys.stdout, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
