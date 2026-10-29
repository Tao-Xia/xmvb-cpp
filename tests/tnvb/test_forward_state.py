"""Cheap one-shot handoff/signature/file-transfer tests, without Block2."""
import ast
import importlib.util
import os
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
WORKER=ROOT/'src/tnvb/worker'
sys.path.insert(0,str(WORKER))
import forward_state as cache
import hs_shared


class HandoffTests(unittest.TestCase):
    def request(self,path,mode=0,solver=0,bond=32,payload=b'integrals'):
        path.write_bytes(b'CPTNREQ3'+struct.pack('<10i',3,1,2,2,2,bond,64,mode,solver,0)+
                         struct.pack('<2d',1e-7,1e-3)+payload)

    def test_only_mode_can_differ(self):
        with tempfile.TemporaryDirectory() as folder:
            a,b=Path(folder)/'a',Path(folder)/'b'
            self.request(a);self.request(b,mode=1)
            self.assertEqual(cache.signature(a),cache.signature(b))
            for change in (dict(solver=1),dict(bond=33),dict(payload=b'changed')):
                self.request(b,**change)
                self.assertNotEqual(cache.signature(a),cache.signature(b))

    def test_numeric_settings_but_not_failure_injection(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'request';self.request(path)
            before=cache.signature(path)
            with patch.dict(os.environ,{'XMVB_BLOCK2_SU2_TOL':'1e-9'}):
                self.assertNotEqual(before,cache.signature(path))
            with patch.dict(os.environ,{'XMVB_BLOCK2_FORBID_3PDM':'1'}):
                self.assertEqual(before,cache.signature(path))

    def test_cutoff_default_and_zero_override_have_distinct_signatures(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            path=Path(folder)/'request';self.request(path)
            default=cache.signature(path)
            with patch.dict(os.environ,{'XMVB_BLOCK2_SU2_CUTOFF':'1e-14'}):
                self.assertEqual(default,cache.signature(path))
            with patch.dict(os.environ,{'XMVB_BLOCK2_SU2_CUTOFF':'0'}):
                self.assertNotEqual(default,cache.signature(path))

    def fixture(self,folder):
        root=Path(folder)/'forward';root.mkdir();(root/'mps').mkdir()
        (root/'mps/S1-mps_info.bin').write_bytes(b'info')
        (root/'context.npz').write_bytes(b'context')
        request,response=Path(folder)/'request',Path(folder)/'response'
        self.request(request);response.write_bytes(b'response')
        cache.publish(root,request,response,dict(state_tags=['S1'],files=1,seconds=0))
        return root,request,response

    def test_claim_once(self):
        with tempfile.TemporaryDirectory() as folder:
            root,request,response=self.fixture(folder)
            self.request(request,mode=1)
            self.assertEqual(cache.claim(root,request,response)['state_tags'],['S1'])
            with self.assertRaises(FileNotFoundError):cache.claim(root,request,response)

    def test_changed_file_response_and_request_rejected(self):
        for target in ('state','response','request'):
            with self.subTest(target=target),tempfile.TemporaryDirectory() as folder:
                root,request,response=self.fixture(folder)
                if target=='state':(root/'mps/S1-mps_info.bin').write_bytes(b'corrupt-longer')
                elif target=='response':response.write_bytes(b'corrupt')
                else:self.request(request,bond=9)
                with self.assertRaises(RuntimeError):cache.claim(root,request,response)
                self.assertTrue((root/'ready.json').is_file())

    def test_move_only_selected_final_files(self):
        tree=ast.parse((WORKER/'pair_core.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and
                  n.name=='move_parallel_state_files')
        scope=dict(os=os,hs_shared=hs_shared)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'move-test','exec'),scope)
        with tempfile.TemporaryDirectory() as folder:
            source,dest=Path(folder)/'source',Path(folder)/'dest';source.mkdir();dest.mkdir()
            for name in ('S1-mps_info.bin','F.MPS.S1.0','S1.P1-mps_info.bin','unrelated'):
                (source/name).write_bytes(b'payload')
            self.assertEqual(scope['move_parallel_state_files'](source,dest,['S1']),2)
            self.assertEqual(set(p.name for p in dest.iterdir()),{'S1-mps_info.bin','F.MPS.S1.0'})
            self.assertEqual(set(p.name for p in source.iterdir()),{'S1.P1-mps_info.bin','unrelated'})


if __name__=='__main__':unittest.main()
