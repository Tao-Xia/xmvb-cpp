#!/usr/bin/env python3
"""Read-only numerical evidence audit; not a full implementation sign-off.

Original optimization gates and supplementary diagnostics remain separate.
No result file, acceptance limit, production option or job is modified.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

OPTIMIZED_ENERGY_ERROR_LIMIT = 1e-6


class Audit:
    def __init__(self, document):
        if not isinstance(document, dict):
            raise ValueError("report must be a JSON object")
        self.document = document
        self.failures = []
        self.metrics = {}
        self.notes = []
        self.check(type(document.get("passed")) is bool, "missing boolean passed flag")
        self._finite_tree(document)

    def check(self, condition, message):
        if not condition:
            self.failures.append(message)

    def _finite_tree(self, value):
        if isinstance(value, float):
            self.check(math.isfinite(value), "nonfinite numeric value in report")
        elif isinstance(value, dict):
            for item in value.values():
                self._finite_tree(item)
        elif isinstance(value, list):
            for item in value:
                self._finite_tree(item)

    def number(self, source, key):
        value = source.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("missing/nonfinite number: " + key)
        return value

    def error(self, source, key, limit=None):
        value = self.number(source, key)
        self.metrics[key] = value
        self.check(value >= 0, key + " is negative")
        if limit is not None:
            self.check(value <= limit, "{}={} exceeds {}".format(key, value, limit))
        return value

    def source_flag(self, c14_historical=False, optimized_energy_update=False):
        if self.document.get("passed") is not True:
            matrix_only_historical = (
                c14_historical and not self.failures and
                "acceptance_profile" not in self.document and
                (self.document.get("H_error", 0) > 1e-8 or
                 self.document.get("S_error", 0) > 1e-10))
            if matrix_only_historical:
                self.notes.append("Historical C14 flag retained; measured errors pass the user-confirmed profile; H/S are diagnostics.")
            elif (optimized_energy_update and not self.failures and
                  "final_energy_error_limit" not in self.document and
                  1e-8 < self.document.get("final_energy_error", 0) <= OPTIMIZED_ENERGY_ERROR_LIMIT):
                self.notes.append("Historical optimization flag retained; measured errors pass the user-updated 1e-6 Eh terminal-energy target with unchanged optimizer convergence settings.")
            else:
                self.check(False, "source report is not passed")

    def result(self):
        return dict(status="failed" if self.failures else "passed",
                    source_passed=self.document.get("passed"),
                    failures=self.failures, metrics=self.metrics, notes=self.notes)


def comparison(audit, k, gradient, integral):
    doc = audit.document
    audit.check(doc.get("integrals") == integral, "integral model mismatch")
    audit.check(doc.get("mode") == ("gradient" if gradient else "energy"), "evaluation mode mismatch")
    limits = {"energy_error": 1e-9 if k == 6 else 1e-6,
              "coefficient_error": 1e-8 if k == 6 else 1e-4,
              "all_weight_error": 1e-8 if k == 6 else 1e-4,
              "H_error": 1e-8 if k == 6 else None,
              "S_error": 1e-10 if k == 6 else None}
    for key, limit in limits.items():
        audit.error(doc, key, limit)
    audit.check(doc.get("tnvb_determinants") == 0, "TNVB has expanded determinants")
    if gradient:
        audit.error(doc, "gradient_error", 1e-7)
        audit.error(doc, "rdm_identity_error", 1e-8)
        audit.error(doc, "fit_energy_error", 1e-7)
    else:
        audit.check(doc.get("gradient_error") is None, "ENERGY report contains gradient error")
    actual = abs(audit.number(doc, "tnvb_energy") - audit.number(doc, "determinant_energy"))
    audit.check(abs(actual - doc["energy_error"]) <= 1e-12, "energy-error field disagrees with energies")
    # Only this explicitly approved C14 scope may differ from an old flag.
    audit.source_flag(c14_historical=k == 14)


def directions(audit, integral):
    doc = audit.document
    audit.check(doc.get("integrals") == integral, "integral model mismatch")
    expected = {"S", "h", "packed_g", "inactive_orbitals", "active_orbitals"}
    items = doc.get("directions", [])
    audit.check(len(items) == 5 and {v.get("component") for v in items} == expected,
                "missing/duplicate finite-difference component")
    for item in items:
        steps = item.get("steps", [])
        audit.check([v.get("epsilon") for v in steps] == [1e-4, 3e-5, 1e-5],
                    "finite-difference step schedule mismatch")
        analytic = audit.number(item, "analytic")
        errors = [audit.error(v, "error") for v in steps]
        for step, error in zip(steps, errors):
            measured = abs(audit.number(step, "numerical") - analytic)
            audit.check(abs(measured - error) <= 1e-12, "FD error disagrees with derivatives")
        stable = len(errors) == 3 and any(errors[i] <= 1e-6 and errors[i+1] <= 1e-6 for i in (0, 1))
        audit.check(stable, "no adjacent passing FD steps: " + str(item.get("component")))
        audit.metrics[str(item.get("component")) + "_worst_error"] = max(errors, default=0)
    audit.source_flag()


def legacy(audit):
    limits = dict(H=1e-8, S=1e-10, coefficients=1e-8,
                  P=1e-8, Pi=1e-8, D3G=1e-7, energy=1e-9)
    for key, limit in limits.items():
        audit.error(audit.document.get("errors", {}), key, limit)
    audit.source_flag()


def endpoint(audit, doc, k, label):
    audit.check(isinstance(doc, dict), "missing endpoint: " + label)
    for key, limit in {"energy_error": 1e-9 if k == 6 else 1e-6,
                       "coefficient_error": 1e-8 if k == 6 else 1e-4,
                       "weight_error": 1e-8 if k == 6 else 1e-4,
                       "gradient_error": 1e-7, "fit_energy_error": 1e-7,
                       "H_error": 1e-8 if k == 6 else None,
                       "S_error": 1e-10 if k == 6 else None}.items():
        value = audit.error(doc, key, limit)
        audit.metrics[label + "." + key] = value
    audit.check(doc.get("passed") is True, "endpoint not passed: " + label)


def optimization(audit, k, integral, supplementary=False):
    doc = audit.document
    audit.check(doc.get("integrals") == integral, "integral model mismatch")
    e_tol = audit.number(doc, "energy_tolerance")
    g_tol = audit.number(doc, "gradient_l2_tolerance")
    if supplementary:
        audit.check(k == 6 and doc.get("convergence_profile") == "stricter_diagnostic",
                    "supplementary result lacks its diagnostic profile")
        audit.check(0 < e_tol <= 1e-7 and 0 < g_tol <= 1e-3, "invalid supplementary tolerances")
    else:
        audit.check(doc.get("convergence_profile", "original") == "original",
                    "supplementary result cannot replace original acceptance")
        audit.check(e_tol == 1e-7 and g_tol == 1e-3, "original optimizer tolerances changed")
    cap = audit.number(doc, "max_iterations")
    audit.check(type(cap) is int and cap > 0 and (k != 14 or cap <= 200), "invalid iteration cap")
    for label in ("reference", "tnvb"):
        state = doc.get(label, {})
        audit.check(state.get("converged") is True, label + " did not converge")
        audit.check(audit.number(state, "gradient_l2") < g_tol, label + " gradient not converged")
        iterations = audit.number(state, "iterations")
        audit.check(type(iterations) is int and 0 <= iterations <= cap, label + " iteration count invalid")
        history = state.get("energy_history", [])
        audit.check(len(history) == iterations + 1, label + " accepted energy history incomplete")
        if iterations > 0:
            audit.check(len(history) >= 2 and abs(history[-1] - history[-2]) < e_tol,
                        label + " accepted energy change not converged")
        energy = audit.number(state, "energy")
        audit.check(bool(history) and abs(history[-1] - energy) <= 1e-9,
                    label + " final energy differs from accepted history")
        audit.error(state, "wall_seconds")
        endpoint(audit, doc.get(label + "_endpoint_crosscheck", {}), k, label)
    measured = abs(doc["tnvb"]["energy"] - doc["reference"]["energy"])
    error = audit.error(doc, "final_energy_error", OPTIMIZED_ENERGY_ERROR_LIMIT)
    audit.metrics["final_energy_error_limit"] = OPTIMIZED_ENERGY_ERROR_LIMIT
    if "final_energy_error_limit" in doc:
        audit.check(audit.number(doc, "final_energy_error_limit") == OPTIMIZED_ENERGY_ERROR_LIMIT,
                    "reported optimized-energy acceptance limit differs from user requirement")
    audit.check(abs(measured - error) <= 1e-12, "final energy error disagrees with energies")
    for key in ("final_coefficient_error", "final_weight_error"):
        audit.error(doc, key, 1e-4 if k == 14 else None)
    reference_c = doc["reference"].get("coefficients", [])
    tensor_c = doc["tnvb"].get("coefficients", [])
    audit.check(len(reference_c) == len(tensor_c) > 0, "missing/mismatched final coefficient arrays")
    phase = -1 if sum(t * r for t, r in zip(tensor_c, reference_c)) < 0 else 1
    dc = max((abs(phase * t - r) for t, r in zip(tensor_c, reference_c)), default=math.inf)
    audit.check(abs(dc - doc["final_coefficient_error"]) <= 1e-12,
                "coefficient-error field disagrees with aligned coefficients")
    reference_w = doc["reference"].get("weights_column_major", [])
    tensor_w = doc["tnvb"].get("weights_column_major", [])
    audit.check(len(reference_w) == len(tensor_w) == 4 * len(reference_c),
                "missing/mismatched final weight arrays")
    dw = max((abs(t - r) for t, r in zip(tensor_w, reference_w)), default=math.inf)
    audit.check(abs(dw - doc["final_weight_error"]) <= 1e-12,
                "weight-error field disagrees with weights")
    audit.error(doc["tnvb"], "fit_energy_error", 1e-7)
    audit.error(doc["tnvb"], "rdm_identity_error", 1e-8)
    for key in ("tnvb_export_energy_error", "reference_export_energy_error"):
        if k == 14 or key in doc:
            audit.error(doc, key, 1e-9)
        else:
            audit.notes.append("Historical benzene report lacks " + key + "; exported ENERGY evidence is listed separately.")
    audit.source_flag(optimized_energy_update=k == 6 and not supplementary)


def audit_file(path, validator, *args):
    entry = dict(path=str(path))
    try:
        content = path.read_bytes()
        entry["sha256"] = hashlib.sha256(content).hexdigest()
        audit = Audit(json.loads(content))
        try:
            validator(audit, *args)
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
            audit.failures.append("malformed/incomplete evidence: " + str(error))
        entry.update(audit.result())
    except FileNotFoundError:
        entry.update(status="pending_evidence", failures=["report not present; this does not establish job state"])
    except (OSError, ValueError, TypeError) as error:
        entry.update(status="failed", failures=["unreadable/malformed evidence: " + str(error)])
    return entry


def summarize(build, c14):
    required, supplementary = {}, {}
    for orbitals in ("hao", "oeo"):
        for integral in ("exact", "ri"):
            name = "c6_" + orbitals + "_" + integral
            required[name + "_same_orbital"] = audit_file(build / (name + "_phase_refresh.json"), comparison, 6, True, integral)
            required[name + "_directions"] = audit_file(build / (name + "_directions.json"), directions, integral)
            required[name + "_legacy"] = audit_file(build / (name + "_legacy.json"), legacy)
            suffix = "_optimization.json" if orbitals == "hao" else "_optimization500.json"
            required[name + "_optimization_original"] = audit_file(build / (name + suffix), optimization, 6, integral)
            if orbitals == "oeo":
                supplementary[name + "_200_step_outcome"] = audit_file(build / (name + "_optimization.json"), optimization, 6, integral)
                supplementary[name + "_stricter"] = audit_file(build / (name + "_stricter_optimization.json"), optimization, 6, integral, True)
            else:
                required[name + "_energy"] = audit_file(build / (name + "_energy.json"), comparison, 6, False, integral)
                exported = audit_file(build / (name + "_export_check.json"), comparison, 6, False, integral)
                if exported["status"] == "passed":
                    try:
                        original = json.loads((build / (name + suffix)).read_text())
                        exported_doc = json.loads(Path(exported["path"]).read_text())
                        de = abs(exported_doc["tnvb_energy"] - original["tnvb"]["energy"])
                        exported["metrics"]["exported_vs_optimized_energy_error"] = de
                        if not math.isfinite(de) or de > 1e-9:
                            raise ValueError("exported energy differs from optimized energy")
                    except (OSError, ValueError, TypeError, KeyError) as error:
                        exported["status"] = "failed"
                        exported["failures"].append("export crosscheck incomplete: " + str(error))
                required[name + "_exported_energy"] = exported
    for integral in ("exact", "ri"):
        name = "c14_" + integral
        required[name + "_same_orbital"] = audit_file(build / ("c14_hao_" + integral + "_gradient.json"), comparison, 14, True, integral)
        required[name + "_optimization"] = audit_file(c14 / (name + "_opt200.json"), optimization, 14, integral)
    for name, item in supplementary.items():
        item["role"] = "historical_only" if name.endswith("_stricter") else "bounded_original_run_diagnostic"
    return dict(scope="numerical evidence only; build/install/isolation and full completion audit remain separate",
                optimization_energy_error_limit=OPTIMIZED_ENERGY_ERROR_LIMIT,
                convergence_policy="original parameters required; stricter historical results cannot supply formal acceptance",
                c14_matrix_policy="user-confirmed: H/S finite diagnostics; energy/gradient/coefficients/weights are gates",
                all_required_numerical_checks_passed=all(v["status"] == "passed" for v in required.values()),
                required=required, supplementary=supplementary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path, help="ON build containing benzene and C14 same-orbital JSON evidence")
    parser.add_argument("c14", type=Path, help="frozen C14 optimization evidence directory")
    parser.add_argument("--output", type=Path, help="write a new summary JSON (input evidence is never overwritten)")
    args = parser.parse_args()
    report = summarize(args.build.resolve(), args.c14.resolve())
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        inputs = {Path(item["path"]).resolve() for group in ("required", "supplementary") for item in report[group].values()}
        if args.output.resolve() in inputs:
            parser.error("output must not overwrite input evidence")
        args.output.write_text(serialized)
    print(serialized, end="")
    return 0 if report["all_required_numerical_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
