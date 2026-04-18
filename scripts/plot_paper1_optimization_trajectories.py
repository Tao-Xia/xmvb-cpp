#!/usr/bin/env python3

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ITERATION_RE = re.compile(
    r"^\s*(\d+)\s+"
    r"([-+]?\d+\.\d+(?:[Ee][-+]?\d+)?)\s+"
    r"([-+]?\d+\.\d+(?:[Ee][-+]?\d+)?)\s+"
    r"([-+]?\d+\.\d+(?:[Ee][-+]?\d+)?)\s+"
    r"([-+]?\d+\.\d+(?:[Ee][-+]?\d+)?)\s+"
    r"([-+]?\d+\.\d+(?:[Ee][-+]?\d+)?)\s*$"
)

KEY_VALUE_RE = re.compile(r"^([A-Za-z0-9_]+)\s*=\s*(.+?)\s*$")


def parse_log(path: Path) -> dict:
    iterations = []
    energies = []
    metadata = {}

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            match = ITERATION_RE.match(line)
            if match:
                iterations.append(int(match.group(1)))
                energies.append(float(match.group(2)))
                continue

            match = KEY_VALUE_RE.match(line)
            if match:
                key = match.group(1)
                value = match.group(2)
                metadata[key] = value

    if "initial_total_energy" not in metadata:
        raise ValueError(f"missing initial_total_energy in {path}")
    if "final_total_energy" not in metadata:
        raise ValueError(f"missing final_total_energy in {path}")

    initial_energy = float(metadata["initial_total_energy"])
    final_energy = float(metadata["final_total_energy"])

    series_iterations = [0] + iterations
    series_energies = [initial_energy] + energies

    return {
        "path": path,
        "iterations": series_iterations,
        "energies": series_energies,
        "initial_energy": initial_energy,
        "final_energy": final_energy,
        "selected_raw_structure_count": int(metadata.get("selected_raw_structure_count", "0")),
        "expanded_determinant_count": int(metadata.get("expanded_determinant_count", "0")),
        "converged": metadata.get("converged", "unknown"),
        "termination_reason": metadata.get("termination_reason", ""),
    }


def plot_system(ax, title: str, determinant: dict, pfaffian: dict) -> None:
    reference_energy = min(determinant["final_energy"], pfaffian["final_energy"])
    floor = 1.0e-12

    det_errors = [max(energy - reference_energy, floor) for energy in determinant["energies"]]
    pf_errors = [max(energy - reference_energy, floor) for energy in pfaffian["energies"]]

    ax.semilogy(
        determinant["iterations"],
        det_errors,
        marker="o",
        markersize=3.5,
        linewidth=1.8,
        color="#1f77b4",
        label="Determinant-expanded reference",
    )
    ax.semilogy(
        pfaffian["iterations"],
        pf_errors,
        marker="s",
        markersize=3.5,
        linewidth=1.8,
        color="#d62728",
        label="Projected-Pfaffian structure-space",
    )

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Optimization Iteration")
    ax.set_ylabel(r"$E_i - E_{\mathrm{ref}}$ (Hartree)")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_ylim(5.0e-13, None)

    structures = pfaffian["selected_raw_structure_count"]
    determinants = pfaffian["expanded_determinant_count"]
    if structures > 0 and determinants > 0:
        ax.text(
            0.03,
            0.05,
            f"{structures} VB structures / {determinants} determinants",
            transform=ax.transAxes,
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.75"},
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Paper I closed-shell optimization trajectories."
    )
    parser.add_argument("--f2-det", default="F2_full_det_vbscf.log")
    parser.add_argument("--f2-pf", default="F2_full_pf_vbscf.log")
    parser.add_argument("--c6h6-det", default="C6H6_full.log")
    parser.add_argument("--c6h6-pf", default="C6H6_full_pf_vbscf.log")
    parser.add_argument(
        "--output-prefix",
        default="figures/paper1_energy_vs_iteration",
        help="Output prefix without extension.",
    )
    args = parser.parse_args()

    datasets = {
        "F$_2$": (
            parse_log(Path(args.f2_det)),
            parse_log(Path(args.f2_pf)),
        ),
        "C$_6$H$_6$": (
            parse_log(Path(args.c6h6_det)),
            parse_log(Path(args.c6h6_pf)),
        ),
    }

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True)

    for ax, (title, (determinant, pfaffian)) in zip(axes, datasets.items()):
        plot_system(ax, title, determinant, pfaffian)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_prefix.with_suffix(".pdf")
    png_path = output_prefix.with_suffix(".png")
    figure.savefig(pdf_path, dpi=300, bbox_inches="tight")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")

    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
