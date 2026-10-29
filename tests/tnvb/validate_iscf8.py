#!/usr/bin/env python3
"""Run ISCF=8 integration checks without changing optimizer tolerances.

This is an explicit numerical validation driver, not a production command or
an ordinary fast CTest. Generated decks/results live only in --output.
"""
import argparse
import json
import math
from pathlib import Path
import re
import subprocess


def run(args):
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[2]
    inputs = root / "inputs"
    inputs.mkdir(exist_ok=True)
    cases = [("c6_hao", "C6_random5_hao.xmi", args.benzene_steps, True),
             ("c6_oeo", "C6_random5_oeo.xmi", args.benzene_steps, True),
             ("c14_hao", "C14_4k_hao.xmi", 1, False)]
    checks = []
    for name, fixture, steps, convergence_required in cases:
        prefix = root / f"{name}_{args.mode}_iscf8"
        deck = inputs / f"{name}_{args.mode}_iscf8.xmi"
        original = (repository / "testdata" / "tnvb" / fixture).read_text()
        assert len(re.findall(r"(?mi)^iscf=8$", original)) == 1
        text = original
        deck.write_text(re.sub(r"(?mi)^itmax=\d+$", f"itmax={steps}", text))
        report_path = prefix.with_suffix(".json")
        if report_path.exists():
            raise RuntimeError(f"refusing to overwrite validation evidence: {report_path}")
        with prefix.with_suffix(".log").open("w") as log:
            completed = subprocess.run([args.optimizer, str(deck), str(report_path),
                                        args.mode, str(steps)], stdout=log,
                                       stderr=subprocess.STDOUT)
        if not report_path.exists():
            raise RuntimeError(f"optimizer did not produce {report_path}, status={completed.returncode}")
        report = json.loads(report_path.read_text())
        reference, tensor = report["reference"], report["tnvb"]
        valid = (report["optimizer"] == "plbfgs" and
                 report["convergence_profile"] == "original" and
                 report["energy_tolerance"] == 1e-7 and
                 report["gradient_tolerance"] == 1e-3 and
                 report["gradient_metric"] == "projected_inf" and
                 report["final_energy_error"] <= 1e-6 and
                 report["tnvb_endpoint_crosscheck"]["passed"] and
                 report["reference_endpoint_crosscheck"]["passed"] and
                 report["tnvb_export_energy_error"] <= 1e-9 and
                 report["reference_export_energy_error"] <= 1e-9 and
                 tensor["fit_energy_error"] <= 1e-7 and
                 tensor["rdm_identity_error"] <= 1e-8)
        for result in (reference, tensor):
            history = result["energy_history"]
            valid = valid and len(history) >= 2 and result["iterations"] >= 1
            valid = valid and all(math.isfinite(e) for e in history)
            valid = valid and all(b <= a + 1e-10 for a, b in zip(history, history[1:]))
            if convergence_required:
                valid = valid and result["converged"] and result["projected_gradient_inf"] < 1e-3
                valid = valid and abs(history[-1] - history[-2]) < 1e-7
            else:
                # A one-step C14 cap is deliberately NOT convergence evidence.
                valid = valid and result["iterations"] == 1 and len(history) == 2
                valid = valid and result["termination"] in (
                    "max_iterations", "plbfgs_projected_gradient_tolerance")
        if convergence_required:
            valid = valid and completed.returncode == 0 and report["passed"]
        else:
            valid = valid and completed.returncode in (0, 1)
            valid = valid and report["final_coefficient_error"] <= 1e-4
            valid = valid and report["final_weight_error"] <= 1e-4
        check = dict(case=name, integral_mode=args.mode, passed=bool(valid),
                     convergence_required=convergence_required, report=str(report_path),
                     exit_code=completed.returncode,
                     final_energy_error=report["final_energy_error"])
        checks.append(check)
        (root / f"summary_{args.mode}.json").write_text(json.dumps(checks, indent=2) + "\n")
        print(json.dumps(check), flush=True)
        if not valid:
            return 1

    # Exercise the actual user-facing executable with normal .xmo output and
    # its standard orbital/density analyses, in addition to diagnostic JSON.
    deck = inputs / f"c6_hao_{args.mode}_iscf8.xmi"
    xmo = root / f"c6_hao_{args.mode}_iscf8_cli.xmo"
    if xmo.exists():
        raise RuntimeError(f"refusing to overwrite validation evidence: {xmo}")
    with xmo.open("w") as log:
        completed = subprocess.run([args.executable, str(deck),
                                    "--standard-two-electron-mode", args.mode],
                                   stdout=log, stderr=subprocess.STDOUT)
    output = xmo.read_text()
    valid = completed.returncode == 0 and "VBSCF converged" in output
    for label in ("block-LBFGS (nonredundant orbital-block inverse)",
                  "Final projected |g|_inf", "OVERLAP OF VB STRUCTURES",
                  "HAMILTONIAN OF VB STRUCTURES", "WEIGHTS OF STRUCTURES",
                  "ORBITALS IN PRIMITIVE BASIS FUNCTIONS", "COMPUTED NATURAL ORBITALS",
                  "XMVB ATOMIC POPULATION ANALYSIS"):
        valid = valid and label in output
    checks.append(dict(case="production_cli", integral_mode=args.mode,
                       passed=bool(valid), xmo=str(xmo), exit_code=completed.returncode))
    (root / f"summary_{args.mode}.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks[-1]), flush=True)
    return 0 if valid else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", required=True)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("exact", "ri"), required=True)
    parser.add_argument("--benzene-steps", type=int, default=500)
    raise SystemExit(run(parser.parse_args()))
