#!/usr/bin/env python3
"""Plot optimizer iteration/time curves directly from the run CSV.

Usage: python3 plot_optimizer_comparison.py paired_runs.csv output_prefix
Writes output_prefix.png and output_prefix.svg.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def read_runs(source):
    runs = defaultdict(dict)
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            runs[row["case"]][row["method"]] = row
    selected = []
    for case, methods in runs.items():
        if all(methods.get(m, {}).get("termination", "").endswith(
                "projected_gradient_tolerance") for m in ("neo", "block_lbfgs")):
            selected.append((case, methods))
    return sorted(selected, key=lambda item: float(item[1]["block_lbfgs"]["total_wall_s"]))


def plot(runs, destination):
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.0), dpi=170, sharex=True)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.13, hspace=0.14)
    fig.suptitle("VBSCF optimizer comparison: all completed paired runs", x=0.10, y=0.975,
                 ha="left", fontsize=16, weight="bold")
    fig.text(0.09, 0.925,
             "Matched Hanhai25 runs · 32 threads · Davidson · Libcint · revision c2776c4",
             fontsize=10, color="#526477")

    x = list(range(len(runs)))
    metrics = (("accepted_steps", "Accepted orbital steps"),
               ("total_wall_s", "End-to-end wall time (s)"))
    methods = (("block_lbfgs", "block-LBFGS", "#2469a8"),
               ("neo", "NEO", "#d97932"))
    for ax, (field, ylabel) in zip(axes, metrics):
        for method, title, color in methods:
            values = [float(pair[method][field]) for _, pair in runs]
            ax.plot(x, values, "-", color=color, linewidth=0.65, alpha=0.38)
            ax.scatter(x, values, s=12, color=color, alpha=0.82,
                       linewidths=0, label=title)
        ax.set_yscale("log")
        ax.minorticks_off()
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(axis="y", which="major", color="#e3e8ee", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="upper left", ncol=2, frameon=False)
    axes[1].set_xlabel("Case rank (sorted by block-LBFGS end-to-end wall time)", fontsize=10)
    fig.text(0.09, 0.04,
             f"All {len(runs)} pairs converged. Lines only guide the eye; each dot is one input. "
             "Log scales keep small and large systems visible.",
             fontsize=8.5, color="#66798b")
    for suffix in ("png", "svg"):
        fig.savefig(destination.with_suffix("." + suffix), facecolor="white")
    plt.close(fig)
    svg = destination.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    runs = read_runs(Path(sys.argv[1]))
    if not runs:
        raise SystemExit("No converged pairs found for selected cases")
    plot(runs, Path(sys.argv[2]))


if __name__ == "__main__":
    main()
