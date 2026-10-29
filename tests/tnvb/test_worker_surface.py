"""Check the production worker call graph without importing native Block2."""
import ast
import builtins
from pathlib import Path
import symtable
import unittest


WORKER = Path(__file__).resolve().parents[2] / 'src/tnvb/worker'


class WorkerSurfaceTests(unittest.TestCase):
    def test_core_functions_are_referenced(self):
        tree = ast.parse((WORKER / 'pair_core.py').read_text())
        functions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        references = {n.id for n in ast.walk(tree)
                      if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        # Includes ProcessPool initializer/map callbacks as well as direct calls.
        for path in WORKER.glob('*.py'):
            references.update(n.attr for n in ast.walk(ast.parse(path.read_text()))
                              if isinstance(n, ast.Attribute) and
                              isinstance(n.value, ast.Name) and n.value.id == 'core')
        self.assertEqual(functions - references, set())

    def test_no_dangling_global_references(self):
        for path in WORKER.glob('*.py'):
            table = symtable.symtable(path.read_text(), str(path), 'exec')
            defined = set(table.get_identifiers()) | set(dir(builtins)) | {'__file__'}
            def check(scope):
                for symbol in scope.get_symbols():
                    if symbol.is_global() and symbol.is_referenced():
                        self.assertIn(symbol.get_name(), defined,
                                      f'{path.name}: {scope.get_name()}')
                for child in scope.get_children():
                    check(child)
            check(table)

    def test_core_exports_used_by_adapters_exist(self):
        exports = set(symtable.symtable((WORKER / 'pair_core.py').read_text(),
                                       'pair_core.py', 'exec').get_identifiers())
        for path in WORKER.glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if (isinstance(node, ast.Attribute) and
                        isinstance(node.value, ast.Name) and node.value.id == 'core'):
                    self.assertIn(node.attr, exports, f'{path.name}: core.{node.attr}')

    def test_only_direct_full_su2_execution_surface(self):
        core = (WORKER / 'pair_core.py').read_text()
        all_sources = '\n'.join(p.read_text() for p in WORKER.glob('*.py'))
        for removed in ('def contracted_d3g_', 'def build_sz_states',
                        'def structure_determinants', 'def fit_weighted_total_state',
                        'def build_weighted_total_state', 'SymmetryTypes.SZ',
                        'XMVB_BLOCK2_D3G_', 'XMVB_BLOCK2_VALIDATE_DIRECT_D3G',
                        'XMVB_BLOCK2_SUM_ENERGY_TOL'):
            self.assertNotIn(removed, all_sources)
        functions = {n.name: n for n in ast.parse(core).body
                     if isinstance(n, ast.FunctionDef)}
        for name in ('build_hs', 'compute_total_state_rdms'):
            arguments = {a.arg for a in functions[name].args.args}
            self.assertFalse(arguments & {'sum_mode', 'd3g_mode', 'd3g_backend',
                                         'd3g_mode_code', 'd3g_backend_code',
                                         'orthogonal_output', 'rinv'})
        self.assertIn('build_direct_total_state', functions)
        self.assertIn('rdm_full.reduce_full_3pdm', core)


if __name__ == '__main__':
    unittest.main()
