"""Full 3-PDM reduction tests independent of Block2 and chemical fixtures."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import tracemalloc
import unittest
from unittest.mock import patch

import numpy as np

PATH = Path(__file__).resolve().parents[2] / "src/tnvb/worker/rdm_full.py"
SPEC = importlib.util.spec_from_file_location("rdm_full", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FullRdmTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3039)

    def verify(self, density, integrals, norm):
        before = density.copy()
        g_before = integrals.copy()
        contracted, particle, times = MODULE.reduce_full_3pdm(density, integrals, norm)
        expected_c = np.einsum("faeb,abcdef->dc", integrals, before / norm,
                               optimize=True)
        expected_p = np.einsum("ijkkba->ijba", before / norm)
        np.testing.assert_allclose(contracted, expected_c, rtol=2e-13, atol=2e-12)
        np.testing.assert_allclose(particle, expected_p, rtol=2e-13, atol=2e-12)
        np.testing.assert_array_equal(density, before)
        np.testing.assert_array_equal(integrals, g_before)
        self.assertGreaterEqual(times["integral_seconds"], 0)

    def test_layout_and_nonunit_norm(self):
        for k in (2, 4, 6):
            for norm in (0.19, 1.0, 8.1):
                density = self.rng.normal(size=(k,) * 6)
                integrals = self.rng.normal(size=(k,) * 4)
                for value in (density, np.asfortranarray(density),
                              density.transpose(5, 3, 2, 1, 0, 4)):
                    with self.subTest(k=k, norm=norm, strides=value.strides):
                        self.verify(value, integrals, norm)

    def test_read_only_data(self):
        density = self.rng.normal(size=(4,) * 6)
        integrals = self.rng.normal(size=(4,) * 4)
        density.flags.writeable = False
        integrals.flags.writeable = False
        self.verify(density, integrals, 1.03)

    def test_workspace_has_no_six_index_copy(self):
        density = np.ones((10,) * 6)
        integrals = np.ones((10,) * 4)
        tracemalloc.start()
        try:
            MODULE.reduce_full_3pdm(density, integrals, 0.99)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak, density.nbytes // 4)

    def test_invalid_norm(self):
        for norm in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                MODULE.reduce_full_3pdm(np.ones((2,) * 6), np.ones((2,) * 4), norm)

    def test_optional_profiler_restores_trace_and_keeps_result(self):
        class Driver:
            def get_npdm(self, state, **kwargs):
                return state.copy()

            def get_3pdm(self, state, **kwargs):
                return self.get_npdm(state, **kwargs)

        state = np.ones((2,) * 6)
        messages = []
        previous = sys.gettrace()
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict(os.environ, {"XMVB_BLOCK2_RDM_PROFILE": folder}):
                value = MODULE.get_profiled_3pdm(Driver(), state, 9, messages.append)
            np.testing.assert_array_equal(value, state)
            self.assertIs(sys.gettrace(), previous)
            reports = list(Path(folder).glob("dm3-*.json"))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text())
            self.assertGreaterEqual(report["seconds"], sum(report["phases"].values()) - 1e-12)
            self.assertTrue(messages[0].startswith("[TNVB-3RDM-PROFILE]"))

    def test_profiler_restores_trace_after_native_failure(self):
        class Driver:
            def get_npdm(self, state, **kwargs):
                raise RuntimeError("injected native failure")

            def get_3pdm(self, state, **kwargs):
                return self.get_npdm(state, **kwargs)

        previous = sys.gettrace()
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict(os.environ, {"XMVB_BLOCK2_RDM_PROFILE": folder}):
                with self.assertRaisesRegex(RuntimeError, "injected native failure"):
                    MODULE.get_profiled_3pdm(Driver(), None, 9, lambda _: None)
            self.assertIs(sys.gettrace(), previous)
            self.assertEqual(len(list(Path(folder).glob("dm3-*.json"))), 1)


if __name__ == "__main__":
    unittest.main()
