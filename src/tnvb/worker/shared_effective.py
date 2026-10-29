"""Fixed common-basis action using one reusable native SU(2) environment.

No finite-bond Hamiltonian-image fit, no dense structure H/S. All structure
dependence is a fixed linear map into one native center tensor. Never invoke
another native sweep/frame reset while this operator is alive.
"""
import hashlib
import time
import numpy as np
import pair_core as core


def layout(matrix):
    info = matrix.info
    return (int(info.delta_quantum.data), bool(info.is_wavefunction),
            tuple(int(q.data) for q in info.quanta),
            tuple(map(int,info.n_states_bra)),tuple(map(int,info.n_states_ket)))


def prefix_digest(state):
    h = hashlib.sha256()
    for site in range(state.n_sites-1):
        state.load_tensor(site)
        mat = state.tensors[site]
        h.update(repr((site,state.canonical_form[site],layout(mat),float(mat.factor))).encode())
        h.update(np.asarray(mat.data).tobytes())
        state.unload_tensor(site)
    return h.hexdigest()


class SharedEffectiveAction:
    def __init__(self,driver,mpo,family,basis):
        self.driver,self.mpo,self.family = driver,mpo,family
        started = time.perf_counter()
        columns = []
        signature = None
        prefix = None
        # Repeat exactly the shape-only left QR used by sum_from_python,
        # but update the entire terminal structure map in one pass.
        canonical = family.family.deep_copy()
        for site in range(family.nsites-1):
            factors = canonical.tensors[site].left_canonicalize()
            canonical.tensors[site+1].left_multiply(factors)
        terminal = canonical.tensors[family.nsites-1]
        z = canonical.tensors[-1].blocks[0].reduced
        map_check_error = 0.0
        for index in range(family.nstr):
            # Independent full-chain conversions at both endpoints validate
            # the native gauge/phase and prefix without N full-chain copies.
            slow = None
            if index in (0,family.nstr-1):
                unit = np.zeros(family.nstr);unit[index]=1.0
                py = family.vector(unit)
                state = core.sum_from_python(driver,py,basis,f'COMMON-CENTER-{index}')
                state,_ = driver.adjust_mps(state,dot=1)
                if state.center != state.n_sites-1 or state.canonical_form[-1]!='S':
                    raise RuntimeError('common native center was not preserved')
                current = prefix_digest(state)
                if prefix is not None and current != prefix:
                    raise RuntimeError('native coefficient conversion changed the fixed prefix')
                prefix = current
                state.load_tensor(state.center)
                center = state.tensors[state.center]
                if center.factor != 1.0:
                    raise RuntimeError('non-unit native center factor')
                if signature is not None and layout(center) != signature:
                    raise RuntimeError('native coefficient conversion changed center layout')
                signature = layout(center)
                slow = np.asarray(center.data).copy()
                if index == 0:self.state = state
                state.unload_tensor(state.center)
                if slow.nbytes*family.nstr > family.max_bytes:
                    raise MemoryError('TNVB common center map exceeds XMVB_BLOCK2_COMMON_GB')
            # Only the terminal tensor depends on the structure coefficient.
            # Native backward fusion and CG recoupling preserve the K->S
            # convention; no Python imitation of non-Abelian recoupling.
            bw = driver.bw
            info = self.state.info
            site = family.nsites-1
            physical = info.basis[site]
            sparse = bw.bs.SparseTensor()
            sparse.data = bw.bs.VectorVectorPSSTensor([
                bw.bs.VectorPSSTensor() for _ in range(physical.n)])
            for block in terminal.blocks:
                left,q,right = block.q_labels
                data = np.einsum('apr,r->ap',block.reduced,z[:,index],optimize=True)[...,None]
                m = physical.find_state(q)
                if m<0:raise RuntimeError('terminal physical sector missing')
                value = bw.bx.Tensor(bw.b.VectorMKLInt(data.shape))
                np.array(value,copy=False)[:] = data
                sparse.data[m].append(((left,right),value))
            left_fused = bw.bs.UnfusedMPS.backward_left_fused(site,info,sparse,True)
            right_fused = info.swap_wfn_to_fused_right(site,left_fused,mpo.tf.opf.cg)
            if layout(right_fused) != signature:
                raise RuntimeError('terminal-only native conversion layout differs')
            column = np.asarray(right_fused.data).copy()
            if slow is not None:
                map_check_error = max(map_check_error,float(np.max(np.abs(column-slow))))
                if map_check_error>1e-12:
                    raise RuntimeError('terminal-only conversion differs from full native chain')
            columns.append(column)
        self.p = np.ascontiguousarray(np.column_stack(columns))
        self.prefix_digest = prefix
        self.terminal_map_check_error = map_check_error
        self.center_map_seconds = time.perf_counter()-started
        # P is a center-by-structure map, not H/S. Full Gram checks belong
        # exclusively to the external validation harness.
        bw = driver.bw
        tick = time.perf_counter()
        self.environment = bw.bs.MovingEnvironment(mpo,self.state,self.state,'COMMON-EFF')
        self.environment.delayed_contraction = bw.b.OpNamesSet.normal_ops()
        self.environment.cached_contraction = False
        self.environment.fused_contraction_rotation = True
        self.environment.save_environments = False
        self.environment.init_environments(False)
        self.environment_seconds = time.perf_counter()-tick
        tick = time.perf_counter()
        self.state.load_tensor(self.state.center)
        center = self.state.tensors[self.state.center]
        self.effective = self.environment.eff_ham(bw.b.FuseTypes.FuseR,False,False,center,center)
        # A raw EffectiveHamiltonian.__call__ is not a standalone Tasked
        # driver: that mode requires native precompute()/post_precompute().
        # Match native EffectiveHamiltonian.multiply's immediate-execution
        # mode using this effective object's private sequence only.
        self.effective.tf.opf.seq.mode = bw.b.SeqTypes.Nothing
        self.effective_seconds = time.perf_counter()-tick
        self.const_e = float(mpo.const_e)
        self.setup_seconds = time.perf_counter()-started
        self.closed = False

    def __call__(self,vector):
        if self.closed:
            raise RuntimeError('common effective action has been closed')
        v = np.asarray(vector,dtype=float)
        if v.shape != (self.family.nstr,) or not np.all(np.isfinite(v)):
            raise ValueError('invalid common-basis coefficients')
        tick = time.perf_counter()
        b = np.ascontiguousarray((self.p@v).reshape((-1,1)))
        c = np.zeros_like(b)
        self.effective(self.driver.bw.b.Matrix(b),self.driver.bw.b.Matrix(c))
        sv = self.p.T@b[:,0]
        hv = self.p.T@c[:,0]+self.const_e*sv
        return hv,sv,dict(seconds=time.perf_counter()-tick,center_size=b.size)

    def close(self):
        if not self.closed:
            self.effective.deallocate()
            self.closed = True
