"""Synthetic callback-only Davidson regression (does not import Block2)."""
import sys
from pathlib import Path
import unittest
import numpy as np
import scipy.linalg

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src/tnvb/worker'))
from davidson import solve


class DavidsonTests(unittest.TestCase):
    def problem(self, n, space=32, seed=4):
        rng = np.random.default_rng(seed)
        a = rng.normal(size=(n,n))*.015
        s = np.eye(n)+a@a.T
        b = rng.normal(size=(n,n))*.03
        h = np.diag(np.arange(n)*.2-3)+(b+b.T)*.5
        expected, vectors = scipy.linalg.eigh(h,s)
        energy, c, info = solve(lambda x:(h@x,s@x),np.diag(h),np.diag(s),
                                max_space=space)
        self.assertLess(abs(energy-expected[0]),1e-11)
        self.assertLess(np.max(np.abs(c*np.sign(c@s@vectors[:,0])-vectors[:,0])),1e-8)
        self.assertLess(abs(c@s@c-1),1e-12)
        self.assertLess(np.linalg.norm(h@c-energy*s@c),1e-10)
        return info

    def test_one(self): self.problem(1)
    def test_small(self): self.problem(7)
    def test_restart(self): self.assertGreater(self.problem(50,8)['restarts'],0)
    def test_near_degenerate(self):
        h=np.diag([-2.,-2.+1e-8,0.,1.])
        e,c,_=solve(lambda x:(h@x,x),np.diag(h),np.ones(4),initial=np.eye(4)[:,3])
        self.assertLess(abs(e+2),1e-11)
    def test_structure_rescaling(self):
        h=np.array([[-2.,.13],[.13,-1.]])
        d=np.diag([1e-4,1e4]); h=d@h@d; s=d@d
        e,_,_=solve(lambda x:(h@x,s@x),np.diag(h),np.diag(s))
        self.assertLess(abs(e-scipy.linalg.eigh(h,s)[0][0]),1e-12)
    def test_invalid(self):
        with self.assertRaises(ValueError): solve(lambda x:(x,x),[1.],[-1.])
        with self.assertRaises(RuntimeError):
            solve(lambda x:(np.full(3,np.nan),x),np.ones(3),np.ones(3))
    def test_nonconvergence(self):
        rng=np.random.default_rng(3);h=rng.normal(size=(20,20));h+=h.T
        with self.assertRaisesRegex(RuntimeError,'failed to converge'):
            solve(lambda x:(h@x,x),np.diag(h),np.ones(20),max_iterations=1)


if __name__=='__main__': unittest.main()
