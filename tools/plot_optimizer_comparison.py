#!/usr/bin/env python3
"""Create direct NEO/block-LBFGS parity plots from paired benchmark CSV data.

Usage: python3 plot_optimizer_comparison.py paired_runs.csv output_prefix
Writes output_prefix.png and output_prefix.svg.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def read_pairs(source):
    by_case = defaultdict(dict)
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            by_case[row["case"]][row["method"]] = row

    pairs = []
    for case, methods in by_case.items():
        if not {"neo", "block_lbfgs"}.issubset(methods):
            continue
        neo, block = methods["neo"], methods["block_lbfgs"]
        if not (neo["termination"].endswith("projected_gradient_tolerance")
                and block["termination"].endswith("projected_gradient_tolerance")):
            continue
        pairs.append({
            "case": case,
            "orbital_type": neo["orbital_type"].upper(),
            "block_steps": float(block["accepted_steps"]),
            "neo_steps": float(neo["accepted_steps"]),
            "block_wall_s": float(block["total_wall_s"]),
            "neo_wall_s": float(neo["total_wall_s"]),
        })
    return pairs


def parity_axis(ax, pairs, x_field, y_field, xlabel, ylabel):
    colors = {"HAO": "#2878b5", "OEO": "#d9752b"}
    for orbital_type in ("HAO", "OEO"):
        group = [pair for pair in pairs if pair["orbital_type"] == orbital_type]
        if group:
            ax.scatter([pair[x_field] for pair in group], [pair[y_field] for pair in group],
                       s=26, color=colors.get(orbital_type, "#6e7d8e"), alpha=0.72,
                       linewidths=0, label=f"{orbital_type} ({len(group)})")
    low = min(min(pair[x_field], pair[y_field]) for pair in pairs)
    high = max(max(pair[x_field], pair[y_field]) for pair in pairs)
    ax.plot([low, high], [low, high], color="#596a7c", linewidth=1.1, zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(low * 0.8, high * 1.25)
    ax.set_ylim(low * 0.8, high * 1.25)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(which="major", color="#e0e6ed", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def plot(pairs, destination):
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8), dpi=180)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.17, top=0.78, wspace=0.28)
    fig.suptitle("VBSCF optimizer comparison: completed paired runs", x=0.09, y=0.97,
                 ha="left", fontsize=15, weight="bold")
    fig.text(0.09, 0.905,
             f"{len(pairs)} matched Hanhai25 runs · 32 threads · Davidson · Libcint · revision c2776c4",
             fontsize=9.5, color="#526477")
    parity_axis(axes[0], pairs, "block_steps", "neo_steps",
                "block-LBFGS accepted orbital steps", "NEO accepted orbital steps")
    parity_axis(axes[1], pairs, "block_wall_s", "neo_wall_s",
                "block-LBFGS end-to-end wall time (s)", "NEO end-to-end wall time (s)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    axes[0].text(0.96, 0.06, "NEO fewer steps", transform=axes[0].transAxes,
                 ha="right", va="bottom", fontsize=8.5, color="#526477")
    axes[1].text(0.96, 0.06, "NEO faster", transform=axes[1].transAxes,
                 ha="right", va="bottom", fontsize=8.5, color="#526477")
    fig.text(0.09, 0.05, "Diagonal: equal performance. Points below the diagonal favor NEO.",
             fontsize=8.5, color="#66798b")
    for suffix in ("png", "svg"):
        fig.savefig(destination.with_suffix("." + suffix), facecolor="white")
    plt.close(fig)
    svg = destination.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    pairs = read_pairs(Path(sys.argv[1]))
    if not pairs:
        raise SystemExit("No completed, converged pairs found")
    plot(pairs, Path(sys.argv[2]))


if __name__ == "__main__":
    main()
