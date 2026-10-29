#!/usr/bin/env python3
"""Pure shared-bank safety/routing tests; no Block2 import or computation."""
from pathlib import Path
import ast
import importlib.util
import os
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("hs_shared", ROOT / "src/tnvb/worker/hs_shared.py")
SHARED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SHARED)


class SharedPolicyTest(unittest.TestCase):
    def test_energy_only_parallel_routing(self):
        self.assertTrue(SHARED.use_shared_hs(None, 2))
        self.assertTrue(SHARED.use_shared_hs(None, 16))
        self.assertFalse(SHARED.use_shared_hs(None, 1))
        for root in (0, 1):
            self.assertFalse(SHARED.use_shared_hs(root, 16))

    def test_file_selection_matches_original(self):
        for tags in (["S1", "S12"], ["S1.P2", "S2"]):
            names = ["F.MPS.S1.0", "F.MPS.S12.0", "F.MPS.S11.0",
                     "S1-mps_info.bin", "S12-mps_info.bin", "S1", "S12.",
                     "F.MPS.S1.P2.0", "S1.P2-mps_info.bin", "F.MPS.S2.0"]
            expected = [name for name in names if any(
                name == f"{tag}-mps_info.bin" or f".{tag}." in name for tag in tags)]
            self.assertEqual(SHARED.selected_names(names, tags), expected)

    def test_bank_integrity_detects_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "tensor"
            file.write_bytes(b"original")
            before = SHARED.fingerprints(directory)
            self.assertEqual(before, SHARED.fingerprints(directory))
            file.write_bytes(b"modified")
            self.assertNotEqual(before, SHARED.fingerprints(directory))

    def test_invalid_manifests_fail_before_block2(self):
        good = dict(norb=14, nelec=14, bond=200, hs_workers=4, hs_threads=2,
                    symmetry="SU2", state_tags=["S1", "S2"])
        SHARED.validate_manifest(good)
        for change in (dict(hs_threads=0), dict(hs_workers=0), dict(symmetry="SZ"),
                       dict(state_tags=[]), dict(state_tags=["S1", "S1"]),
                       dict(state_tags=["../S1"])):
            with self.assertRaises(ValueError):
                SHARED.validate_manifest(dict(good, **change))

    def test_cutoff_default_and_cache_are_consistent(self):
        source = (ROOT / "src/tnvb/worker/pair_core.py").read_text()
        tree = ast.parse(source)
        builder = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                       and node.name == 'build_su2_states')
        cutoff = next(node.value for node in builder.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'cutoff'
                              for target in node.targets))
        manifest = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'warm_cache_manifest')
        result = next(node.value for node in manifest.body if isinstance(node, ast.Return))
        settings = next(value for key, value in zip(result.keys, result.values)
                        if isinstance(key, ast.Constant) and key.value == 'pair_fit')
        cutoff_code = compile(ast.Expression(cutoff), 'pair-cutoff', 'eval')
        settings_code = compile(ast.Expression(settings), 'cache-cutoff', 'eval')
        for override, expected in ((None, '1e-14'), ('0', '0'), ('1e-12', '1e-12')):
            with self.subTest(override=override), patch.dict(os.environ, {}, clear=True):
                if override is not None:
                    os.environ['XMVB_BLOCK2_SU2_CUTOFF'] = override
                self.assertEqual(eval(cutoff_code, {'os': os}), float(expected))
                self.assertEqual(eval(settings_code, {'os': os})['XMVB_BLOCK2_SU2_CUTOFF'],
                                 expected)


if __name__ == "__main__":
    unittest.main()
