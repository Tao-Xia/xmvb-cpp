"""Fixed linear real-SU(2) VB-family map; no determinant expansion.

Append an SU(2)-scalar structure-label site, then use lossless block QR.
The physical-site tensors and final coefficient map are independent of v.
Native left-coupled singlets are embedded in a common local physical basis.
"""
import time
import numpy as np
from pyblock2.algebra.core import MPS, Tensor, SubTensor


def bytes_in(state):
    return sum(block.reduced.nbytes for tensor in state.tensors for block in tensor.blocks)


def bond_profile(state):
    return [sum(values.values()) for values in state.get_bond_dims()]


def addition_bytes(a, b):
    bonds = [x+y for x,y in zip(a.get_bond_dims(), b.get_bond_dims())]
    total = 0
    for site, (x,y) in enumerate(zip(a.tensors,b.tensors)):
        shapes = {}
        for block in x.blocks+y.blocks:
            shape = list(block.reduced.shape)
            if site: shape[0] = bonds[site-1][block.q_labels[0]]
            if site != a.n_sites-1: shape[-1] = bonds[site][block.q_labels[-1]]
            shapes[block.q_labels] = shape
        total += sum(8*int(np.prod(shape)) for shape in shapes.values())
    return total


class FixedFamily:
    def __init__(self, states, zero, target, max_bytes=2*1024**3):
        if not states or any(s.n_sites != states[0].n_sites for s in states):
            raise ValueError('inconsistent or empty state family')
        started = time.perf_counter()
        self.nstr, self.nsites = len(states), states[0].n_sites
        self.zero, self.target = zero, target
        self.max_bytes = max_bytes
        total = None
        self.history = []
        for index, source in enumerate(states):
            branch = source.deep_copy()
            branch.tensors[-1] = Tensor(blocks=[SubTensor(
                q_labels=(*block.q_labels,target), reduced=block.reduced[...,None].copy())
                for block in branch.tensors[-1].blocks])
            label = np.zeros((1,self.nstr))
            label[0,index] = 1.0
            branch.tensors.append(Tensor(blocks=[SubTensor((target,zero),label)]))
            if total is None:
                total = branch
            else:
                estimate = addition_bytes(total,branch)
                # Budget includes source, result and a conservative QR copy.
                if 2*estimate+bytes_in(total)+bytes_in(branch) > max_bytes:
                    raise MemoryError('TNVB common-family construction exceeds XMVB_BLOCK2_COMMON_GB; use DENSE or increase the resource budget')
                total = total + branch
            # Opposite QR first trims shape-redundant direct-sum dimensions
            # from the finite structure-label boundary. Neither QR uses a
            # singular-value cutoff or rank cap; this does not discard states.
            total.canonicalize(0)
            total.canonicalize(self.nsites)
            self.history.append(dict(structures=index+1,bytes=bytes_in(total),
                                     bonds=bond_profile(total)))
        self.family = total
        if len(total.tensors[-1].blocks) != 1:
            raise RuntimeError('unexpected non-scalar family label sector')
        final = total.tensors[-1].blocks[0]
        if final.q_labels != (target,zero) or final.reduced.shape[1] != self.nstr:
            raise RuntimeError('inconsistent family coefficient map')
        self.z = final.reduced.copy()
        self.build_seconds = time.perf_counter()-started
        self.isometry_error = self.check_isometry()
        if self.isometry_error > 1e-11:
            raise RuntimeError(f'family left-isometry error {self.isometry_error}')
        self.digest = self.fingerprint()

    def fingerprint(self):
        import hashlib
        h = hashlib.sha256()
        for tensor in self.family.tensors:
            for block in tensor.blocks:
                h.update(np.asarray([q.data for q in block.q_labels],dtype='<u8').tobytes())
                h.update(np.asarray(block.reduced.shape,dtype='<u8').tobytes())
                h.update(block.reduced.tobytes())
        return h.hexdigest()

    def check_isometry(self):
        largest = 0.0
        for tensor in self.family.tensors[:-1]:
            grams = {}
            for block in tensor.blocks:
                a = block.reduced.reshape((-1,block.reduced.shape[-1]))
                q = block.q_labels[-1]
                gram = a.T@a
                grams[q] = grams.get(q,0.0)+gram
            for gram in grams.values():
                largest = max(largest,float(np.max(np.abs(gram-np.eye(gram.shape[0])))))
        return largest

    def vector(self, coefficients):
        v = np.asarray(coefficients,dtype=float)
        if v.shape != (self.nstr,) or not np.all(np.isfinite(v)):
            raise ValueError('invalid family coefficients')
        x = self.z@v
        # Keep the stored family immutable: canonicalization later mutates copies.
        tensors = [tensor.deep_copy() for tensor in self.family.tensors[:self.nsites-1]]
        tensors.append(Tensor(blocks=[SubTensor(block.q_labels[:-1],
            np.einsum('apr,r->ap',block.reduced,x,optimize=True))
            for block in self.family.tensors[self.nsites-1].blocks]))
        return MPS(tensors=tensors)

    def overlap_action(self, coefficients):
        # First K tensors are isometries; do not materialize the N x N Gram.
        return self.z.T@(self.z@np.asarray(coefficients,dtype=float))

    def project(self, state):
        """Apply B^T once to a left-coupled singlet MPS, retaining all labels."""
        if state.n_sites != self.nsites:
            raise ValueError('projection site count differs')
        environment = {self.zero:np.ones((1,1))}
        for site in range(self.nsites):
            bra_blocks = self.family.tensors[site].blocks
            ket_blocks = {}
            for block in state.tensors[site].blocks:
                labels, data = block.q_labels, block.reduced
                if site == 0:
                    labels = (self.zero,*labels); data = data[None,...]
                if site == self.nsites-1:
                    labels = (*labels,self.target); data = data[...,None]
                if labels in ket_blocks:
                    raise RuntimeError('duplicate physical ket block')
                ket_blocks[labels] = data
            output = {}
            for block in bra_blocks:
                labels, a = block.q_labels, block.reduced
                if site == 0:
                    labels = (self.zero,*labels); a = a[None,...]
                left,physical,right = labels
                if left not in environment or labels not in ket_blocks:
                    continue
                term = np.einsum('ab,api,bpj->ij',environment[left],a,
                                 ket_blocks[labels],optimize=True)
                output[right] = output.get(right,0.0)+term
            environment = output
        final = environment.get(self.target)
        if final is None or final.shape != (self.z.shape[0],1):
            raise RuntimeError('projection did not reach singlet target')
        return self.z.T@final[:,0]
