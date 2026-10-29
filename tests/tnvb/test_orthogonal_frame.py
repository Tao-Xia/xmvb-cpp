"""Orthogonal integral/response protocol checks without native Block2."""
import ast
from pathlib import Path
import struct
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import scipy.linalg

ROOT=Path(__file__).resolve().parents[2]/'src/tnvb/worker'


def functions(path,names,scope):
    nodes=[n for n in ast.parse(path.read_text()).body
           if isinstance(n,ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)


SCOPE=dict(np=np,scipy=__import__('scipy'))
functions(ROOT/'pair_core.py',{'pair_index','unpack_eri',
          'accept_orthogonal_integrals','orthogonal_reduced_data',
          'read_exact'},SCOPE)
READER=dict(np=np,struct=struct,REQUEST_MAGIC=b'CPTNREQ3',VERSION=3,
            core=types.SimpleNamespace(read_exact=SCOPE['read_exact']))
functions(ROOT/'evaluate.py',{'read_array','read_request'},READER)


# Independent raw-frame reference, kept outside the production worker.
def transform_reduced_data_to_xmvb(rinv, dm1, dm2, contracted_orth):
    """Transform the 1/2-RDM and contracted 3-RDM to XMVB conventions."""
    density1 = rinv @ dm1 @ rinv.T
    density2 = np.einsum(
        "ip,jq,kr,ls,pqrs->ijkl",
        rinv,
        rinv,
        rinv,
        rinv,
        dm2,
        optimize=True,
    )
    # Block2: dm2[x1,x2,y2,y1].  XMVB: Pit[y2,y1,x2,x1].
    pit = np.asfortranarray(density2.transpose(2, 3, 1, 0))
    contracted_nonorth = rinv @ contracted_orth @ rinv.T
    # Block2 dm1[x1,y1]; XMVB P[y1,x1].  It is symmetric for the real
    # state used here, but retain the explicit convention.
    pmat = np.asfortranarray(density1.T)
    return pmat, pit, np.asfortranarray(contracted_nonorth)


class OrthogonalFrameTests(unittest.TestCase):
    def setUp(self):
        self.r=np.array([[1.2,0.35],[0.,0.8]])
        self.s=self.r.T@self.r
        self.h=np.array([[-0.9,0.17],[0.17,-0.6]])
        self.g=np.arange(6,dtype=float)/7

    def test_accepts_without_another_integral_transformation(self):
        with patch.dict(SCOPE,orthogonalize_integrals=lambda *args:
                        self.fail('pretransformed integrals were orthogonalized twice')):
            r,h,g=SCOPE['accept_orthogonal_integrals'](self.s,self.h,self.g,self.r)
        self.assertIs(h,self.h)
        self.assertIs(r,self.r)
        np.testing.assert_array_equal(g,SCOPE['unpack_eri'](self.g,2))
        q=np.linalg.inv(self.r)
        self.assertGreater(np.max(np.abs(q.T@h@q-h)),0.1)

    def test_invalid_frames_fail_before_native_work(self):
        for r in (np.eye(2),self.r.T,-self.r,self.r*np.nan,np.ones((3,3))):
            with self.subTest(r=r),self.assertRaises(ValueError):
                SCOPE['accept_orthogonal_integrals'](self.s,self.h,self.g,r)

    def test_response_uses_orthogonal_density_indices(self):
        dm1=np.array([[1.3,.12],[.12,.7]])
        dm2=np.arange(16.).reshape((2,)*4)
        contracted=np.array([[.3,.04],[.04,.2]])
        p,pi,c=SCOPE['orthogonal_reduced_data'](dm1,dm2,contracted)
        np.testing.assert_array_equal(p,dm1.T)
        np.testing.assert_array_equal(pi,dm2.transpose(2,3,1,0))
        np.testing.assert_array_equal(c,contracted)
        q=np.linalg.inv(self.r)
        raw=transform_reduced_data_to_xmvb(q,dm1,dm2,contracted)
        expected=np.einsum('ia,jb,kc,ld,abcd->ijkl',q,q,q,q,pi)
        np.testing.assert_allclose(raw[1],expected,atol=2e-14)

    def test_protocol_carries_frame_and_rejects_old_protocol(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'request'
            header=struct.pack('<10i',3,1,2,2,2,16,32,1,0,0)
            payload=struct.pack('<2d',1e-7,1e-3)+struct.pack('<2i',1,2)
            for a in (self.s,self.r,self.h,self.g):
                payload+=a.astype('<f8').tobytes(order='F')
            path.write_bytes(b'CPTNREQ3'+header+payload)
            request,cap,mode,options,frame=READER['read_request'](path)
            self.assertEqual((cap,mode,options),(32,1,None))
            np.testing.assert_array_equal(frame,self.r)
            np.testing.assert_array_equal(request[-2],self.h)
            np.testing.assert_array_equal(request[-1],self.g)
            path.write_bytes(b'CPTNREQ2'+header+payload)
            with self.assertRaises(ValueError):READER['read_request'](path)


if __name__=='__main__':unittest.main()
