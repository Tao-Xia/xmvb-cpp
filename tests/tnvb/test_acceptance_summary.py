#!/usr/bin/env python3
"""No Block2 or quantum calculation: regress the evidence classifier."""
import copy
import tempfile
import unittest
from pathlib import Path

from summarize_acceptance import Audit, audit_file, comparison, directions, optimization


def comparison_document(k=6):
    return dict(passed=True, mode="gradient", integrals="exact",
                tnvb_energy=-10.0, determinant_energy=-10.0,
                energy_error=0.0, H_error=0.0, S_error=0.0,
                coefficient_error=0.0, all_weight_error=0.0,
                gradient_error=0.0, rdm_identity_error=0.0,
                fit_energy_error=0.0, tnvb_determinants=0)


def optimization_document():
    cross = dict(passed=True, energy_error=0.0, H_error=0.0, S_error=0.0,
                 coefficient_error=0.0, weight_error=0.0, gradient_error=0.0,
                 fit_energy_error=0.0)
    state = dict(converged=True, iterations=1, gradient_l2=1e-4,
                 energy=-10.0, energy_history=[-9.99999999, -10.0],
                 wall_seconds=1.0, fit_energy_error=0.0, rdm_identity_error=0.0,
                 coefficients=[1.0, 0.0], weights_column_major=[1.0, 0.0] * 4)
    return dict(passed=True, integrals="exact", max_iterations=200,
                convergence_profile="original", energy_tolerance=1e-7,
                gradient_l2_tolerance=1e-3, reference=copy.deepcopy(state),
                tnvb=copy.deepcopy(state), reference_endpoint_crosscheck=cross,
                tnvb_endpoint_crosscheck=copy.deepcopy(cross), final_energy_error=0.0,
                final_coefficient_error=0.0, final_weight_error=0.0,
                tnvb_export_energy_error=0.0, reference_export_energy_error=0.0)


class AcceptanceSummaryTest(unittest.TestCase):
    def test_c14_user_profile_does_not_hide_historical_matrix_error(self):
        doc = comparison_document(14)
        doc.update(passed=False, S_error=1.52e-10)
        audit = Audit(doc)
        comparison(audit, 14, True, "exact")
        self.assertEqual(audit.result()["status"], "passed")
        self.assertFalse(audit.result()["source_passed"])
        self.assertEqual(audit.metrics["S_error"], 1.52e-10)
        self.assertTrue(audit.notes)
        benzene = Audit(doc)
        comparison(benzene, 6, True, "exact")
        self.assertEqual(benzene.result()["status"], "failed")

    def test_false_source_flag_cannot_be_overridden_for_benzene(self):
        doc = comparison_document()
        doc["passed"] = False
        audit = Audit(doc)
        comparison(audit, 6, True, "exact")
        self.assertTrue(audit.failures)

    def test_unknown_c14_failure_is_not_overridden(self):
        doc = comparison_document(14)
        doc["passed"] = False
        audit = Audit(doc)
        comparison(audit, 14, True, "exact")
        self.assertTrue(audit.failures)

    def test_nonfinite_extra_numeric_field_fails(self):
        doc = comparison_document()
        doc["coefficients"] = [float("nan")]
        audit = Audit(doc)
        comparison(audit, 6, True, "exact")
        self.assertTrue(audit.failures)

    def test_missing_numeric_field_is_not_a_pass(self):
        doc = comparison_document()
        del doc["S_error"]
        with self.assertRaises(ValueError):
            comparison(Audit(doc), 6, True, "exact")

    def test_missing_evidence_is_pending_not_running_or_passed(self):
        with tempfile.TemporaryDirectory() as folder:
            result = audit_file(Path(folder) / "missing.json", comparison, 6, True, "exact")
        self.assertEqual(result["status"], "pending_evidence")

    def test_converged_flag_does_not_override_terminal_energy_mismatch(self):
        doc = optimization_document()
        doc["tnvb"]["energy"] += 2e-6
        doc["tnvb"]["energy_history"] = [doc["tnvb"]["energy"] + 1e-8, doc["tnvb"]["energy"]]
        doc["final_energy_error"] = 2e-6
        audit = Audit(doc)
        optimization(audit, 6, "exact")
        self.assertTrue(any("final_energy_error" in v for v in audit.failures))

    def test_approved_energy_target_preserves_old_flag_and_convergence_settings(self):
        doc = optimization_document()
        doc["passed"] = False
        doc["tnvb"]["energy"] += 7.7e-7
        doc["tnvb"]["energy_history"] = [doc["tnvb"]["energy"] + 1e-8, doc["tnvb"]["energy"]]
        doc["final_energy_error"] = 7.7e-7
        audit = Audit(doc)
        optimization(audit, 6, "exact")
        self.assertFalse(audit.failures)
        self.assertFalse(audit.result()["source_passed"])
        self.assertTrue(audit.notes)
        self.assertEqual(doc["energy_tolerance"], 1e-7)
        self.assertEqual(doc["gradient_l2_tolerance"], 1e-3)
        # An unexplained failure under the new limit cannot be reclassified.
        doc["final_energy_error_limit"] = 1e-6
        audit = Audit(doc)
        optimization(audit, 6, "exact")
        self.assertTrue(audit.failures)

    def test_same_orbital_energy_target_remains_strict(self):
        doc = comparison_document()
        doc["tnvb_energy"] += 2e-7
        doc["energy_error"] = abs(doc["tnvb_energy"] - doc["determinant_energy"])
        audit = Audit(doc)
        comparison(audit, 6, True, "exact")
        self.assertTrue(any("energy_error" in v for v in audit.failures))

    def test_supplementary_settings_cannot_replace_original(self):
        doc = optimization_document()
        doc.update(convergence_profile="stricter_diagnostic", energy_tolerance=1e-10,
                   gradient_l2_tolerance=1e-5)
        for label in ("reference", "tnvb"):
            doc[label].update(gradient_l2=1e-6, energy_history=[-9.99999999999, -10.0])
        original = Audit(doc)
        optimization(original, 6, "exact")
        self.assertTrue(original.failures)
        diagnostic = Audit(doc)
        optimization(diagnostic, 6, "exact", supplementary=True)
        self.assertFalse(diagnostic.failures)

    def test_c14_cap_and_export_checks_are_required(self):
        doc = optimization_document()
        doc["max_iterations"] = 201
        del doc["tnvb_export_energy_error"]
        audit = Audit(doc)
        with self.assertRaises(ValueError):
            optimization(audit, 14, "exact")
        self.assertTrue(any("iteration cap" in v for v in audit.failures))

    def test_coefficient_field_is_recomputed_with_phase_alignment(self):
        doc = optimization_document()
        doc["tnvb"]["coefficients"] = [-1.0, 0.0]
        audit = Audit(doc)
        optimization(audit, 6, "exact")
        self.assertFalse(audit.failures)
        doc["tnvb"]["coefficients"][1] = 0.01
        audit = Audit(doc)
        optimization(audit, 6, "exact")
        self.assertTrue(any("aligned coefficients" in v for v in audit.failures))

    def test_rdm_limit_cannot_be_bypassed_by_passed_flag(self):
        doc = optimization_document()
        doc["tnvb"]["rdm_identity_error"] = 2e-8
        audit = Audit(doc)
        optimization(audit, 14, "exact")
        self.assertTrue(any("rdm_identity_error" in v for v in audit.failures))

    def test_fd_single_lucky_step_is_not_stable(self):
        doc = dict(passed=True, integrals="exact", directions=[])
        for component in ("S", "h", "packed_g", "inactive_orbitals", "active_orbitals"):
            steps = [dict(epsilon=e, numerical=v, error=v)
                     for e, v in zip((1e-4, 3e-5, 1e-5), (1e-5, 1e-8, 1e-5))]
            doc["directions"].append(dict(component=component, analytic=0.0, steps=steps))
        audit = Audit(doc)
        directions(audit, "exact")
        self.assertTrue(audit.failures)


if __name__ == "__main__":
    unittest.main()
