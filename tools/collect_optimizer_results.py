#!/usr/bin/env python3
"""Collect provenance-preserving VBSCF optimizer results from Slurm run folders.

Usage: python3 collect_optimizer_results.py ROOT REVISION LAYOUT > runs.csv
LAYOUT is ``method-case`` or ``case-method``. This script only reads run files.
"""

import csv
import re
import sys
from pathlib import Path


FIELDS = (
    "case method revision run_dir exit_code termination eigensolver integrals "
    "orbital_type basis n_active_orbitals n_active_electrons n_structures n_states "
    "n_threads energy_initial_hartree energy_final_hartree gradient_final_inf "
    "accepted_steps scf_wall_s total_wall_s peak_rss_kb orbital_hvp_actions "
    "structure_response_actions coupled_block_actions exact_hvp_block_actions "
    "structure_response_block_actions rejected_trials"
).split()


def collect(run, case, method, revision):
    result = dict.fromkeys(FIELDS, "")
    result.update(case=case, method=method, revision=revision, run_dir=str(run))
    status = run / "status.txt"
    if status.exists():
        for line in status.read_text(errors="replace").splitlines():
            if line.startswith("exit_code="):
                result["exit_code"] = line.partition("=")[2]

    xmi = run / "input.xmi"
    if xmi.exists():
        in_control = False
        for line in xmi.read_text(errors="replace").splitlines():
            stripped = line.strip().lower()
            if stripped.startswith("$ctrl"):
                in_control = True
                continue
            if in_control and stripped.startswith("$end"):
                break
            if not in_control or stripped.startswith("#"):
                continue
            for key, value in re.findall(r"\b(orbtyp|basis|nao|nae)\s*=\s*([^\s]+)",
                                         stripped):
                field = {"orbtyp": "orbital_type", "basis": "basis",
                         "nao": "n_active_orbitals", "nae": "n_active_electrons"}[key]
                result[field] = value

    output = run / "run.out"
    if output.exists():
        values = {
            "Termination reason": "termination",
            "Structure eigensolver": "eigensolver",
            "CPU threads": "n_threads",
            "Initial total energy": "energy_initial_hartree",
            "Final total energy": "energy_final_hartree",
            "Final projected |g|_inf": "gradient_final_inf",
            "SCF iteration wall time": "scf_wall_s",
            "End-to-end wall time": "total_wall_s",
            "Number of structures": "n_structures",
            "Number of equally averaged states": "n_states",
        }
        with output.open(errors="replace") as handle:
            for line in handle:
                if ":" in line:
                    key, _, value = line.partition(":")
                    field = values.get(key.strip())
                    if field is not None:
                        result[field] = value.strip().removesuffix(" s")
                match = re.search(r"VBSCF converged in\s+(\d+) iterations", line)
                if match:
                    result["accepted_steps"] = match.group(1)
                if "Integral evaluation:" in line:
                    result["integrals"] = line.partition(":")[2].strip().rstrip(".")

    resource = run / "resource.time"
    if resource.exists():
        for line in resource.read_text(errors="replace").splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            if key.strip() == "Maximum resident set size (kbytes)":
                result["peak_rss_kb"] = value.strip()
            elif key.strip() == "Exit status" and not result["exit_code"]:
                result["exit_code"] = value.strip()

    trace = run / "optimizer.tsv"
    if trace.exists():
        sums = {key: 0 for key in ("orbital_hvp_actions", "structure_response_actions",
                                     "coupled_block_actions", "exact_hvp_block_actions",
                                     "structure_response_block_actions", "rejected_trials")}
        with trace.open() as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            available = set(reader.fieldnames or ()) & sums.keys()
            for row in reader:
                for key in available:
                    sums[key] += int(row.get(key, "0") or "0")
        for key in available:
            result[key] = sums[key]
    return result


def main():
    if len(sys.argv) != 4 or sys.argv[3] not in ("method-case", "case-method"):
        raise SystemExit(__doc__)
    root, revision, layout = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    for outer in sorted(root.iterdir()):
        if not outer.is_dir():
            continue
        for inner in sorted(outer.iterdir()):
            if not inner.is_dir() or not (inner / "run.out").exists():
                continue
            case, method = ((inner.name, outer.name) if layout == "method-case"
                            else (outer.name, inner.name))
            writer.writerow(collect(inner, case, method, revision))


if __name__ == "__main__":
    main()
