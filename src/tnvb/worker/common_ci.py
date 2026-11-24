"""Isolated common-SU(2)-basis Davidson execution and result export.

Only coefficient space is varied: H v = P.T H_eff P v, S v = P.T P v.
The native effective Hamiltonian lives for the entire solve, then is released
before any RDM driver is created. No free DMRG optimization is performed.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
import time
import numpy as np


def _solve(source, scratch, tags, norb, nelec, bond, threads, h1, g2, options):
    import pair_core as core
    from fixed_family import FixedFamily, bytes_in, bond_profile
    from shared_effective import SharedEffectiveAction
    from davidson import solve
    core.configure_phase_environment(threads)
    if Path(source).resolve()!=Path(scratch).resolve():
        core.copy_parallel_state_files(source, scratch, tags)
    # A common family is larger than one capped structure. The individual
    # state stack heuristic is not a valid environment budget here. Use the
    # same 8-GiB floor as FULL RDM, unless the user explicitly sets the stack.
    stack_bytes=core.stack_memory_bytes(norb,bond)
    if 'XMVB_BLOCK2_STACK_GB' not in os.environ:
        stack_bytes=max(stack_bytes,8*1024**3)
    driver = core.DMRGDriver(scratch=scratch, symm_type=core.SymmetryTypes.SU2,
        n_threads=threads, stack_mem=stack_bytes,
        clean_scratch=True)
    driver.initialize_system(n_sites=norb,n_elec=nelec,spin=0,orb_sym=[0]*norb)
    mpo = core.get_hamiltonian_mpo(driver,h1,g2,log=False)
    basis = [Counter({driver.bw.SX(q.n,q.twos,q.pg):int(info.n_states[i])
                      for i,q in enumerate(info.quanta)}) for info in driver.ghamil.basis]
    states = []
    for index,tag in enumerate(tags):
        state,local = core.sum_left_coupled_python(driver,driver.load_mps(tag),f'CI-IN-{index}')
        if any(n>0 and full.get(q)!=n for given,full in zip(local,basis) for q,n in given.items()):
            raise RuntimeError('Davidson family has an incompatible physical basis')
        states.append(state)
    budget = float(os.environ.get('XMVB_BLOCK2_COMMON_GB','2'))
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError('XMVB_BLOCK2_COMMON_GB must be positive')
    family = FixedFamily(states,driver.bw.SX(0,0,0),driver.bw.SX(nelec,0,0),
                         max_bytes=int(budget*1024**3))
    del states
    core.log_summary(f'[TNVB-COMMON] prepare structures={len(tags)} '
        f'common_bond={max(bond_profile(family.family))} '
        f'family_mib={bytes_in(family.family)/1024**2:.3f} '
        f'native_stack_gib={stack_bytes/1024**3:.3f}')
    operator = SharedEffectiveAction(driver,mpo,family,basis)
    try:
        p = operator.p
        n = len(tags)
        sdiag = np.einsum('ij,ij->j',p,p)
        if np.any(sdiag <= 0) or not np.all(np.isfinite(sdiag)):
            raise RuntimeError('Davidson family contains a zero/nonfinite structure')
        singular = np.linalg.svd(p/np.sqrt(sdiag)[None,:],compute_uv=False)
        eigen = singular**2
        if len(eigen)<n or eigen[-1] <= max(1e-12,eigen[0]*1e-12):
            raise RuntimeError('Davidson structure metric is rank deficient; no structures were removed')
        hdiag = np.empty(n)
        for i in range(n):
            unit=np.zeros(n);unit[i]=1
            hdiag[i]=operator(unit)[0][i]
        started=time.perf_counter()
        energy,c,info=solve(operator,hdiag,sdiag,
            options['energy_tolerance'],options['gradient_tolerance'],
            initial=options.get('initial'))
        solve_seconds=time.perf_counter()-started
        # Materialize for the existing XMO matrices/weight/analysis contract
        # AFTER convergence; these columns were not used by the CI solver.
        started=time.perf_counter()
        h=np.empty((n,n));s=np.empty((n,n))
        for i in range(n):
            unit=np.zeros(n);unit[i]=1
            h[:,i],s[:,i],_=operator(unit)
        hasym=float(np.max(np.abs(h-h.T)));sasym=float(np.max(np.abs(s-s.T)))
        if hasym>1e-10*max(1.,float(np.max(np.abs(h)))) or sasym>1e-10:
            raise RuntimeError('Davidson common-basis export lost H/S symmetry')
        h=(h+h.T)*.5;s=(s+s.T)*.5
        core.validate_structure_metric(h,s)
        if abs(c@s@c-1)>1e-9 or np.linalg.norm((h@c-energy*s@c)/np.sqrt(sdiag))>1e-9:
            raise RuntimeError('Davidson exported eigenpair residual failed')
        if family.fingerprint()!=family.digest:
            raise RuntimeError('Davidson mutated its fixed structure family')
        export_seconds=time.perf_counter()-started
        core.log_summary(f'[TNVB-DAVIDSON] iterations={info["iterations"]} '
            f'actions={info["block_actions"]} subspace={info["peak_subspace_dimension"]} '
            f'restarts={info["restarts"]} residual={info["maximum_stationarity_residual"]:.3e} '
            f'relative={info["maximum_relative_residual"]:.3e} solve={solve_seconds:.3f}s')
        core.log_summary(f'[TNVB-COMMON] structures={n} threads={threads} '
            f'family={family.build_seconds:.3f}s bytes={bytes_in(family.family)} '
            f'common_bond={max(bond_profile(family.family))} center={p.shape[0]} '
            f'setup={operator.setup_seconds:.3f}s export={export_seconds:.3f}s '
            f'H_asym={hasym:.3e} S_asym={sasym:.3e}')
        return h,s,energy,c,info
    finally:
        operator.close()


def contract(source, scratch_parent, tags, norb, nelec, bond, threads, h1, g2, options,
             private_source=False):
    """Use spawn: never reset the parent construction driver's global frame."""
    scratch=(source if private_source else
             tempfile.mkdtemp(prefix='tnvb-common-ci-',dir=scratch_parent))
    try:
        with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
            return pool.submit(_solve,source,scratch,tags,norb,nelec,bond,threads,h1,g2,options).result()
    finally:
        if not private_source:
            shutil.rmtree(scratch,ignore_errors=True)
