#!/usr/bin/env python3
"""Exercise production RDM gates without loading Block2 or computing RDMs."""
import ast
import math
from pathlib import Path
import unittest


SOURCE_ROOT = Path(__file__).resolve().parents[2]


class RdmPolicyTest(unittest.TestCase):
    def setUp(self):
        core = SOURCE_ROOT / "src/tnvb/worker/pair_core.py"
        tree = ast.parse(core.read_text())
        selected = [node for node in tree.body if (
            isinstance(node, ast.FunctionDef) and node.name == "check_total_state_rdm"
        ) or (
            isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "RDM_CHECK_WARNING"
                for target in node.targets
            )
        )]
        self.messages = []
        namespace = {"math": math, "log_summary": self.messages.append}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(core), "exec"), namespace)
        self.check = namespace["check_total_state_rdm"]

    def test_warning_boundary_is_quiet(self):
        self.check(0.0, 0.0, 1e-8, 0.0, 1e-7)
        self.assertEqual(self.messages, [])

    def test_each_residual_warns_in_band(self):
        for index in range(4):
            with self.subTest(index=index):
                self.messages.clear()
                errors = [0.0] * 4
                errors[index] = 1.781e-8
                self.check(*errors, 1e-7)
                self.assertEqual(len(self.messages), 1)
                self.assertIn("[TNVB-WARNING]", self.messages[0])
                self.assertIn("fatal_limit=1.000e-07", self.messages[0])

    def test_fatal_boundary_is_warning_only(self):
        self.check(0.0, 0.0, 1e-7, 0.0, 1e-7)
        self.assertEqual(len(self.messages), 1)

    def test_each_residual_above_fatal_limit_stops(self):
        for index in range(4):
            with self.subTest(index=index):
                errors = [0.0] * 4
                errors[index] = 1.001e-7
                with self.assertRaisesRegex(RuntimeError, "RDM check failed"):
                    self.check(*errors, 1e-7)

    def test_invalid_residuals_stop(self):
        for invalid in (float("nan"), float("inf"), -1e-9):
            for index in range(4):
                with self.subTest(invalid=invalid, index=index):
                    errors = [0.0] * 4
                    errors[index] = invalid
                    with self.assertRaisesRegex(RuntimeError, "invalid residual"):
                        self.check(*errors, 1e-7)

    def test_invalid_fatal_limit_stops(self):
        for limit in (float("nan"), float("inf"), 0.0, -1e-7):
            with self.subTest(limit=limit):
                with self.assertRaisesRegex(RuntimeError, "invalid fatal limit"):
                    self.check(0.0, 0.0, 0.0, 0.0, limit)

    def test_production_entrypoints_and_cpp_use_same_fatal_limit(self):
        wrapper = (SOURCE_ROOT / "src/tnvb/worker/evaluate.py").read_text()
        process = (SOURCE_ROOT / "src/tnvb/communication/process.cpp").read_text()
        evaluator = (SOURCE_ROOT / "src/tnvb/electronic/evaluator.cpp").read_text()
        self.assertIn('os.environ["XMVB_BLOCK2_RDM_CHECK_TOL"] = "1e-7"', wrapper)
        self.assertIn('environment["XMVB_BLOCK2_RDM_CHECK_TOL"] = "1e-7"', process)
        self.assertIn("response.rdm_identity_error > 1e-7", evaluator)


if __name__ == "__main__":
    unittest.main()
