#!/usr/bin/env python3
"""Owned paired-state numerical core for the independent C++ TNVB worker.

The production entrypoint and only wire protocol live in evaluate.py. The
stable SU(2) pair/total-state numerical routines are reused here; this module
does not provide a legacy XMVB executable or protocol compatibility path.
"""

from __future__ import annotations

import atexit
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing
import os
import shutil
import sys
import tempfile
import time

# NumPy/SciPy are loaded in every spawned task process.  Their BLAS runtime
# must be constrained before those modules are imported; changing these
# variables in a ProcessPool initializer is already too late.  Tensor-network
# parallelism is provided by Block2's explicit ``n_threads`` setting instead.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["GOTO_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ.setdefault("OMP_DYNAMIC", "FALSE")
os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")

import numpy as np
import scipy.linalg

from pyblock2.algebra.core import MPS as PyMPS, SubTensor, Tensor
from pyblock2.algebra.io import init_block2_types
from pyblock2.driver.core import (
    DMRGDriver,
    MPOAlgorithmTypes,
    NPDMAlgorithmTypes,
    SymmetryTypes,
)

import hs_shared
import rdm_full


# Diagnostic only: never reject an evaluation or promote a bond stage on this
# basis. This retains the former default accuracy reference, without an input
# keyword or an environment-variable override.
FIT_ENERGY_WARNING = 1.0e-7
# Production RDM diagnostics have a warning band before the fatal limit.
# These are numerical consistency checks, not orbital convergence criteria.
RDM_CHECK_WARNING = 1.0e-8
# Internal repair/cache criteria, not orbital convergence limits. Finite
# failures of these criteria still produce RDMs and a compression warning.
SUM_INFIDELITY_TOL = 1.0e-10
# Numerical allowance for repeated SVD/center moves as well as floating-point
# sums. It is below the repair threshold and is charged against that threshold,
# rather than interpreting a slightly negative infidelity as perfect accuracy.
SUM_FIDELITY_NUMERICAL_FLOOR = 1.0e-11


# Initialized separately in every spawned H/S process.  Block2 owns global
# threading and memory state, so sharing one DMRGDriver between Python threads
# is unsafe; process-private drivers also give every contraction private
# scratch filenames.
_PARALLEL_HS_CONTEXT = None
_PARALLEL_STATE_CONTEXT = None

# TNVB production output is intentionally fixed and independent of XMVB's
# traditional IPRINT setting.  Existing development prints remain in the
# implementation as local diagnostics but are silent on successful requests;
# fatal diagnostics and corrupt warm-cache details are always visible.
_builtin_print = print


def print(*values, **kwargs):
    stream = kwargs.get("file", sys.stdout)
    message = kwargs.get("sep", " ").join(str(value) for value in values)
    if stream is not sys.stdout or "load_failed=" in message:
        _builtin_print(*values, **kwargs)


def log_summary(message: str):
    _builtin_print(message, flush=True)


def check_total_state_rdm(
    trace_error, contraction2_error, contraction3_error, rdm_energy_error,
    rdm_tolerance,
):
    """Warn above 1e-8; reject nonfinite data or residuals above the fatal limit."""
    errors = (
        trace_error, contraction2_error, contraction3_error, rdm_energy_error,
    )
    detail = (
        f"trace1={trace_error:.3e}, "
        f"contract2={contraction2_error:.3e}, "
        f"contract3={contraction3_error:.3e}, "
        f"energy={rdm_energy_error:.3e}"
    )
    if not math.isfinite(rdm_tolerance) or rdm_tolerance <= 0.0:
        raise RuntimeError("TNVB total-state RDM check has an invalid fatal limit")
    if any(not math.isfinite(error) or error < 0.0 for error in errors):
        raise RuntimeError("TNVB total-state RDM check failed: invalid residual; " + detail)
    if max(errors) > rdm_tolerance:
        raise RuntimeError(
            "TNVB total-state RDM check failed: " + detail
            + f", limit={rdm_tolerance:.3e}"
        )
    if max(errors) > RDM_CHECK_WARNING:
        log_summary(
            "[TNVB-WARNING] total-state RDM check: " + detail
            + f", warning_limit={RDM_CHECK_WARNING:.3e}, "
            + f"fatal_limit={rdm_tolerance:.3e}; continuing RDM/gradient evaluation. "
            + "Numerical consistency is outside the warning reference; "
            + "verify gradient and bond convergence before interpreting optimized results."
        )


def warn_total_state_fit(fit_error: float, tnbond: int, sum_bond: int):
    """Report a finite fitting discrepancy without changing the calculation."""
    if not np.isfinite(fit_error) or fit_error < 0.0:
        raise RuntimeError("TNVB total MPS has an invalid fit energy deviation")
    if fit_error > FIT_ENERGY_WARNING:
        log_summary(
            f"[TNVB-WARNING] fit_dE={fit_error:.6e} Eh exceeds "
            f"warning_reference={FIT_ENERGY_WARNING:.1e} Eh "
            f"D={tnbond}/{sum_bond}; continuing RDM/gradient evaluation. "
            "The fitted total-state MPS and H/S energies are inconsistent; "
            "gradients and optimized results may be inaccurate. "
            "Check convergence with larger TNSUMBOND (and TNBOND)."
        )


def read_exact(stream, byte_count: int) -> bytes:
    data = stream.read(byte_count)
    if len(data) != byte_count:
        raise RuntimeError("truncated XMVB/Block2 request")
    return data


def pair_index(i: int, j: int) -> int:
    hi, lo = max(i, j), min(i, j)
    return hi * (hi + 1) // 2 + lo


def unpack_eri(packed: np.ndarray, n: int) -> np.ndarray:
    result = np.empty((n, n, n, n), dtype=np.float64)
    for p in range(n):
        for q in range(n):
            pq = pair_index(p, q)
            for r in range(n):
                for s in range(n):
                    rs = pair_index(r, s)
                    result[p, q, r, s] = packed[pair_index(pq, rs)]
    return result


def accept_orthogonal_integrals(sso, h_orth, packed_g_orth, rmat):
    """Accept C++'s framed h/G without another orbital transformation."""
    n = sso.shape[0]
    if (sso.shape != (n, n) or h_orth.shape != (n, n) or
            rmat.shape != (n, n) or
            packed_g_orth.shape != (n*(n+1)//2*(n*(n+1)//2+1)//2,) or
            not all(np.all(np.isfinite(x)) for x in (sso,h_orth,packed_g_orth,rmat))):
        raise ValueError("invalid orthogonal TNVB integral dimensions or values")
    scale = max(1.0, float(np.max(np.abs(rmat))))
    if (np.any(np.diag(rmat)<=0) or np.max(np.abs(np.tril(rmat,-1)))>1e-12*scale or
            np.max(np.abs(rmat.T@rmat-sso))>1e-10*max(1.0,float(np.max(np.abs(sso))))):
        raise ValueError("TNVB orthogonal integral frame does not match its VB overlap")
    if np.max(np.abs(h_orth-h_orth.T))>1e-10*max(1.0,float(np.max(np.abs(h_orth)))):
        raise ValueError("asymmetric orthogonal TNVB one-electron integrals")
    return rmat, h_orth, unpack_eri(packed_g_orth,n)


def orthogonal_reduced_data(dm1, dm2, contracted_orth):
    """Protocol v3 index layout, with every free index in the orthogonal basis."""
    return (np.asfortranarray(dm1.T),
            np.asfortranarray(dm2.transpose(2,3,1,0)),
            np.asfortranarray(contracted_orth))


def paired_structure_layout(structure, nao: int, nae: int, nel: int):
    """Validate occupations and return singlet pairs for one VB structure."""
    if nae % 2:
        raise RuntimeError("Block2 paired MPS currently requires even NAE")
    ndb = (nel - nae) // 2
    if 2 * ndb + nae != nel:
        raise RuntimeError("NEL-NAE must be even")
    active = np.asarray(structure[2 * ndb : 2 * ndb + nae], dtype=int) - ndb - 1
    site_kind = [("empty", -1) for _ in range(nao)]
    pairs = []
    structure_phase = 1.0
    for pair in range(nae // 2):
        original_left = int(active[2 * pair])
        original_right = int(active[2 * pair + 1])
        if not (0 <= original_left < nao and 0 <= original_right < nao):
            raise RuntimeError("active pair orbital is outside 1..NAO")
        if original_left == original_right:
            if site_kind[original_left][0] != "empty":
                raise RuntimeError("active orbital belongs to multiple pairs")
            site_kind[original_left] = ("double", -1)
            continue
        if original_left > original_right:
            structure_phase = -structure_phase
        left, right = sorted((original_left, original_right))
        if site_kind[left][0] != "empty" or site_kind[right][0] != "empty":
            raise RuntimeError("active orbital belongs to multiple pairs")
        pair_id = len(pairs)
        pairs.append((left, right))
        site_kind[left] = ("pair_left", pair_id)
        site_kind[right] = ("pair_right", pair_id)
    return site_kind, pairs, structure_phase


def stack_memory_bytes(nao: int, tnbond: int) -> int:
    if "XMVB_BLOCK2_STACK_GB" in os.environ:
        return int(float(os.environ["XMVB_BLOCK2_STACK_GB"]) * 1024**3)
    estimate = 32 * nao * max(2, tnbond) ** 2
    return max(512 * 1024**2, estimate)


def rdm_stack_memory_bytes(nao: int, sum_bond: int) -> int:
    """Size Block2's single-process stack for high-rank RDM environments."""
    if "XMVB_BLOCK2_RDM_STACK_GB" in os.environ:
        value = float(os.environ["XMVB_BLOCK2_RDM_STACK_GB"])
        if value <= 0.0:
            raise RuntimeError("XMVB_BLOCK2_RDM_STACK_GB must be positive")
        return int(value * 1024**3)
    # The 1-/2-/3-RDM environments retain many more operator blocks than an
    # H/S contraction.  This phase is serial at the process level, so a larger
    # private stack does not multiply across the H/S worker pool.
    return max(8 * 1024**3, 16 * stack_memory_bytes(nao, sum_bond))


def npdm_algorithm_type():
    """Return the selected Block2 NPDM implementation and its log label."""
    mode = os.environ.get("XMVB_BLOCK2_NPDM_ALGORITHM", "default")
    mode = mode.strip().lower().replace("_", "-")
    algorithms = {
        "default": NPDMAlgorithmTypes.Default,
        "symbol-free": NPDMAlgorithmTypes.SymbolFree,
        "symbol-free-low-mem": (
            NPDMAlgorithmTypes.SymbolFree
            | NPDMAlgorithmTypes.Compressed
            | NPDMAlgorithmTypes.LowMem
        ),
        "normal": NPDMAlgorithmTypes.Normal,
        "fast": NPDMAlgorithmTypes.Fast,
    }
    if mode not in algorithms:
        raise RuntimeError(
            "XMVB_BLOCK2_NPDM_ALGORITHM must be default, symbol-free, "
            "symbol-free-low-mem, normal, or fast"
        )
    return algorithms[mode], mode


def get_hamiltonian_mpo(driver, h_orth, g_orth, log=True):
    """Build the QC Hamiltonian with the selected Block2 MPO algorithm."""
    mpo_mode = os.environ.get("XMVB_BLOCK2_H_MPO", "no-trans-conventional")
    mpo_mode = mpo_mode.lower().replace("_", "-")
    algorithms = {
        "fast-bipartite": MPOAlgorithmTypes.FastBipartite,
        "no-trans-conventional": MPOAlgorithmTypes.NoTransConventional,
        "no-trans-conventional-nc": MPOAlgorithmTypes.NoTransConventionalNC,
        "no-trans-conventional-cn": MPOAlgorithmTypes.NoTransConventionalCN,
    }
    if mpo_mode not in algorithms:
        raise RuntimeError(
            "XMVB_BLOCK2_H_MPO must be fast-bipartite, no-trans-conventional, "
            "no-trans-conventional-nc, or no-trans-conventional-cn"
        )
    started = time.perf_counter()
    h_mpo = driver.get_qc_mpo(
        h1e=h_orth,
        g2e=g_orth,
        ecore=0.0,
        cutoff=1.0e-14,
        integral_cutoff=1.0e-15,
        symmetrize=False,
        algo_type=algorithms[mpo_mode],
        iprint=0,
    )
    summary = "unavailable"
    try:
        nnz, size, bond = h_mpo.get_summary()
        summary = f"bond={bond} nnz={nnz} size={size}"
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    if log:
        print(
            f"[TNVB-BLOCK2] Hamiltonian_MPO mode={mpo_mode} {summary} "
            f"build_seconds={time.perf_counter() - started:.3f}",
            flush=True,
        )
    return h_mpo


def structure_pair_sequence(structure, nao: int, nae: int, nel: int):
    """Return every active singlet pair and the XMVB structure phase."""
    # Reuse the stricter site-layout validation so that shared orbitals or an
    # unclosed pair fail before any expensive SU(2) MPO multiplication.
    paired_structure_layout(structure, nao, nae, nel)
    ndb = (nel - nae) // 2
    active = np.asarray(structure[2 * ndb : 2 * ndb + nae], dtype=int) - ndb - 1
    pairs = []
    phase = 1.0
    for ipair in range(nae // 2):
        original_left = int(active[2 * ipair])
        original_right = int(active[2 * ipair + 1])
        if original_left > original_right:
            phase = -phase
        pairs.append(
            (min(original_left, original_right), max(original_left, original_right))
        )
    return pairs, phase


def build_su2_pair_mpo(driver, rmat, left: int, right: int, scale: float):
    """Build one nonorthogonal XMVB singlet-pair creator in SU(2) form.

    Block2's reduced `(C+C)0` convention produces a normalized inter-orbital
    singlet and sqrt(2)|double> for equal indices.  XMVB instead uses unit
    coefficients for both members of an inter-orbital singlet and unit
    amplitude for a doubly occupied orbital.  The two cases therefore require
    sqrt(2) and 1/sqrt(2), respectively.
    """
    coefficient = np.outer(rmat[:, left], rmat[:, right])
    normalization = math.sqrt(2.0) if left != right else 1.0 / math.sqrt(2.0)
    builder = driver.expr_builder()
    builder.add_sum_term(
        "(C+C)0",
        coefficient,
        cutoff=1.0e-16,
        factor=scale * normalization,
    )
    return driver.get_mpo(
        builder.finalize(), cutoff=1.0e-16, add_ident=False, iprint=0
    )


def normalize_block2_mps(state):
    """Normalize a canonical center and persist its numerical tensor.

    Block2 save_data() writes metadata only.  Contractions and child drivers
    can reload tensors from disk, so saving the center is part of normalization.
    Callers must already have a canonical MPS (as returned by multiply/addition).
    """
    center = state.center
    if not 0 <= center < state.n_sites:
        raise RuntimeError("TNVB MPS has an invalid canonical center")
    # After a forward two-site sweep at the right boundary, Block2 keeps
    # center=n_sites-2 although the wavefunction tensor is the final C/S site
    # (e.g. LLLLLC).  Normalizing the L isometry there would not normalize MPS.
    if state.dot == 2 and state.canonical_form[center] == "L":
        center += 1
    if center >= state.n_sites or state.canonical_form[center] not in "CKS":
        raise RuntimeError("TNVB MPS normalization requires a canonical wavefunction tensor")
    if state.tensors[center] is None:
        raise RuntimeError("TNVB MPS has no tensor at its canonical center")
    # deep_copy() returns allocated tensor handles with deallocated storage;
    # testing only for None misses that case.
    if state.tensors[center].total_memory == 0:
        state.load_tensor(center)
    tensor = state.tensors[center]
    center_norm = float(tensor.norm())
    if not math.isfinite(center_norm) or center_norm <= 0.0:
        raise RuntimeError("TNVB MPS has a zero or nonfinite center norm")
    tensor.iscale(1.0 / center_norm)
    state.save_tensor(center)
    state.save_data()


def validate_structure_metric(hmat, smat):
    """Check a norm-scaled Gram matrix before either ENERGY or RDM solving.

    No chemical structures are silently dropped.  A retained-subspace
    condition number would hide rank loss, so reject rank-deficient inputs.
    Returned diagnostics refer to the normalized, dimensionless Gram matrix.
    """
    if (hmat.ndim != 2 or hmat.shape != smat.shape or
            hmat.shape[0] != hmat.shape[1] or hmat.shape[0] == 0):
        raise RuntimeError("TNVB H/S matrices have inconsistent dimensions")
    if not np.all(np.isfinite(hmat)) or not np.all(np.isfinite(smat)):
        raise RuntimeError("TNVB H/S matrices contain nonfinite elements")
    diagonal = np.diag(smat)
    if np.any(diagonal <= 0.0):
        raise RuntimeError("TNVB structure MPS has a nonpositive norm")
    norms = np.sqrt(diagonal)
    scaled_s = (smat / norms[:, None]) / norms[None, :]
    eigenvalues = np.linalg.eigvalsh(0.5 * (scaled_s + scaled_s.T))
    tolerance = max(1.0e-12, abs(float(eigenvalues[-1])) * 1.0e-12)
    rank = int(np.count_nonzero(eigenvalues > tolerance))
    if rank != smat.shape[0]:
        raise RuntimeError(
            "TNVB structure overlap is rank deficient or not positive definite: "
            f"S_rank={rank}/{smat.shape[0]} "
            f"S_min={eigenvalues[0]:.6e} rank_tol={tolerance:.6e}; "
            "use a linearly independent structure set (no structures were dropped)"
        )
    return float(eigenvalues[0]), float(eigenvalues[-1] / eigenvalues[0]), rank


def solve_generalized_state(hmat, smat, root: int):
    """Solve the same real symmetric generalized problem as XMVB eigencalc."""
    if not 0 <= root < hmat.shape[0]:
        raise RuntimeError(
            f"requested TNVB root {root + 1} is outside 1..{hmat.shape[0]}"
        )
    validate_structure_metric(hmat, smat)
    norms = np.sqrt(np.diag(smat))
    scaled_h = (hmat / norms[:, None]) / norms[None, :]
    scaled_s = (smat / norms[:, None]) / norms[None, :]
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        scaled_h, scaled_s, check_finite=True, driver="gvd"
    )
    vector = eigenvectors[:, root]
    residual = scaled_h @ vector - eigenvalues[root] * (scaled_s @ vector)
    scale = max(1.0, np.linalg.norm(scaled_h, ord=np.inf) +
                abs(eigenvalues[root]) * np.linalg.norm(scaled_s, ord=np.inf))
    if (not np.all(np.isfinite(eigenvalues)) or
            not np.all(np.isfinite(vector)) or
            np.linalg.norm(residual, ord=np.inf) >
            1.0e-9 * scale * max(1.0, np.linalg.norm(vector, ord=np.inf))):
        raise RuntimeError("TNVB generalized eigenproblem has an invalid residual")
    coefficient = np.asarray(vector / norms, dtype=np.float64)
    metric_norm = float(coefficient @ smat @ coefficient)
    if not np.isfinite(metric_norm) or metric_norm <= 0.0:
        raise RuntimeError("invalid metric norm from TNVB generalized eigenproblem")
    coefficient /= math.sqrt(metric_norm)
    return float(eigenvalues[root]), coefficient


def sum_quality_passes(quality):
    return bool(all(math.isfinite(quality[key]) and quality[key] >= 0.0
                    for key in ("infidelity", "roundoff", "energy_error")) and
                quality["infidelity"] + quality["roundoff"] <= SUM_INFIDELITY_TOL and
                quality["energy_error"] <= FIT_ENERGY_WARNING)


def total_state_quality(driver, total, states, coefficient, smat,
                        identity_mpo, h_mpo, eigenvalue):
    """Compare a compressed MPS with the actual H/S target, using no RDMs."""
    c = np.asarray(coefficient, dtype=np.float64)
    if (c.shape != (len(states),) or smat.shape != (len(states), len(states)) or
            not np.all(np.isfinite(c)) or not np.all(np.isfinite(smat))):
        raise RuntimeError("TNVB total-state quality has inconsistent/nonfinite inputs")
    q = float(c @ smat @ c)
    n = float(driver.expectation(total, identity_mpo, total))
    if not math.isfinite(q) or not math.isfinite(n) or min(q, n) <= 0.0:
        raise RuntimeError("TNVB total-state quality has an invalid target/MPS norm")
    terms = [float(ci) * float(driver.expectation(total, identity_mpo, state))
             for ci, state in zip(c, states)]
    z = math.fsum(terms)
    fidelity = (z / math.sqrt(n) / math.sqrt(q)) ** 2
    q_scale = float(np.abs(c) @ np.abs(smat) @ np.abs(c)) / q
    z_scale = math.fsum(abs(x) for x in terms) / math.sqrt(n * q)
    # Expectations and center moves accumulate rounding along the MPS chain;
    # a scalar dot-product error bound alone falsely rejects long chains.
    roundoff = max(SUM_FIDELITY_NUMERICAL_FLOOR,
                   float(64.0 * max(1, getattr(driver, "n_sites", 1)) * np.finfo(float).eps *
                         max(1.0, q_scale, z_scale) * (1.0 + fidelity)))
    if not math.isfinite(fidelity) or not math.isfinite(roundoff):
        raise RuntimeError("TNVB total-state fidelity is nonfinite")
    if fidelity > 1.0 + roundoff:
        raise RuntimeError(
            "TNVB total-state fidelity violates the Cauchy-Schwarz bound: "
            f"F={fidelity:.17e} bound={roundoff:.3e} "
            f"norm={n:.17e} target_norm={q:.17e} overlap={z:.17e}"
        )
    infidelity = max(0.0, 1.0 - fidelity)
    energy = float(driver.expectation(total, h_mpo, total)) / n
    error = abs(energy - eigenvalue)
    if not math.isfinite(energy) or not math.isfinite(error):
        raise RuntimeError("TNVB total MPS has an invalid energy")
    return dict(norm=n, target_norm=q, target_overlap=z,
                infidelity=infidelity, roundoff=roundoff,
                energy=energy, energy_error=error)


def sum_left_coupled_python(driver, state, tag):
    copied = driver.copy_mps(state, tag)
    copied, _ = driver.adjust_mps(copied, dot=1)
    driver.align_mps_center(copied, copied.n_sites - 1)
    # A right-boundary S tensor is still right-fused. Convert it to K using
    # Block2's SU(2) recoupling before decoding a purely left-coupled tensor.
    if copied.canonical_form[-1]=='S':
        copied.flip_fused_form(copied.n_sites-1,None,driver.prule)
        copied.save_data()
    copied.info.load_mutable()
    basis = [Counter({driver.bw.SX(b.quanta[i].n,b.quanta[i].twos,b.quanta[i].pg):
                      int(b.n_states[i]) for i in range(b.n)})
             for b in copied.info.basis]
    # The generic Python decoder misidentifies this build's right-boundary
    # fused tensor. Use Block2's native unfuser, retaining the K coupling tree.
    unfused=driver.bw.bs.UnfusedMPS()
    unfused.initialize(copied)
    tensors=[]
    clone=lambda q: driver.bw.SX(q.n,q.twos,q.pg)
    for site,tensor in enumerate(unfused.tensors):
        blocks=[]
        for m,sector in enumerate(tensor.data):
            physical=clone(copied.info.basis[site].quanta[m])
            for (left,right),reduced in sector:
                labels=(clone(left),physical,clone(right))
                data=np.array(reduced,copy=True)
                if site==0:
                    assert data.shape[0]==1 and labels[0]==driver.bw.SX(0,0,0)
                    data=data[0]; labels=labels[1:]
                if site==copied.n_sites-1:
                    assert data.shape[-1]==1 and labels[-1]==copied.info.target
                    data=data[...,0]; labels=labels[:-1]
                blocks.append(SubTensor(q_labels=labels,reduced=data))
        tensors.append(Tensor(blocks=blocks))
    result=PyMPS(tensors=tensors)
    assert all(t is not None for t in result.tensors)
    return result, basis


def sum_from_python(driver, state, basis, tag):
    # Construct an all-left-coupled UnfusedMPS ending in K. The generic upstream
    # to_block2 chooses S at the right boundary; SU(2) K/S recoupling must instead
    # be done by Block2 AFTER the tensor is assembled in the intended tree.
    state.canonicalize(state.n_sites-1)
    b,bs,brs,bx=init_block2_types(driver.bw.SX,np.float64)
    vacuum=driver.bw.SX(0,0,0)
    target=driver.bw.SX(driver.n_elec,0,0)
    basis_info=[]
    def fill(info,values):
        info.allocate(len(values))
        for i,(q,n) in enumerate(values.items()):
            info.quanta[i]=q; info.n_states[i]=n
        info.sort_states()
        return info
    for values in basis:
        basis_info.append(fill(brs.StateInfo(),values))
    info=brs.MPSInfo(state.n_sites,vacuum,target,brs.VectorStateInfo(basis_info))
    info.tag=tag
    info.set_bond_dimension_full_fci(vacuum,vacuum)
    info.left_dims[0]=brs.StateInfo(vacuum)
    for i,values in enumerate(state.get_left_dims()):
        fill(info.left_dims[i+1],values)
    info.left_dims[state.n_sites]=brs.StateInfo(target)
    info.right_dims[0]=brs.StateInfo(target)
    for i,values in enumerate(state.get_right_dims()):
        fill(info.right_dims[i+1],{(target-q)[0]:n for q,n in values.items()})
    info.right_dims[state.n_sites]=brs.StateInfo(vacuum)
    info.bond_dim=info.get_max_bond_dimension()
    info.save_mutable()
    info.save_data(driver.scratch+'/'+tag+'-mps_info.bin')
    tensors=[]
    for site,physical_basis in enumerate(basis_info):
        tensor=bs.SparseTensor()
        tensor.data=bs.VectorVectorPSSTensor([bs.VectorPSSTensor() for _ in range(physical_basis.n)])
        for block in state.tensors[site].blocks:
            if site==0:
                physical,right=block.q_labels;left=vacuum
                data=block.reduced[None,...]
            elif site==state.n_sites-1:
                left,physical=block.q_labels;right=target
                data=block.reduced[...,None]
            else:
                left,physical,right=block.q_labels;data=block.reduced
            m=physical_basis.find_state(physical)
            assert m!=-1
            tensor.data[m].append(((left,right),bx.Tensor(b.VectorMKLInt(data.shape))))
            np.array(tensor.data[m][-1][1],copy=False)[:]=data
        tensors.append(tensor)
    unfused=bs.UnfusedMPS()
    unfused.info=info;unfused.n_sites=state.n_sites
    unfused.canonical_form='L'*(state.n_sites-1)+'K'
    unfused.center=state.n_sites-1;unfused.dot=1
    unfused.tensors=bs.VectorSpTensor(tensors)
    result=unfused.finalize()
    # This Block2 release's adjust_mps(dot=2) leaves a terminal S while
    # align_mps_center expects C in two-site form (otherwise it loops in T).
    # Use the same boundary representation as the saved production MPS:
    # right-fused terminal tensor, center=N-2, dot=2, canonical ...LC.
    result.flip_fused_form(state.n_sites-1,None,driver.prule)
    assert result.canonical_form[-1]=='S'
    result.canonical_form=result.canonical_form[:-1]+'C'
    result.center=state.n_sites-2
    result.dot=2
    result.save_data()
    return result


def sum_python(states, coefficients, order):
    order = list(order)
    total = states[order[0]] * float(coefficients[order[0]])
    for index in order[1:]:
        total = total + states[index] * float(coefficients[index])
    return total


def max_python_bond(state):
    return max(sum(d.values()) for d in state.get_bond_dims())


def sum_native_compress(driver, exact, limit, tag):
    started = time.perf_counter()
    state = driver.copy_mps(exact, tag)
    driver.compress_mps(state, max_bond_dim=limit)
    # Return to the same canonical orientation for independent measurements.
    driver.compress_mps(state, max_bond_dim=limit)
    # align_mps_center similarly leaves terminal S after a two-site sweep.
    assert state.dot==2 and state.center==state.n_sites-2
    if state.canonical_form[-1]=='S':
        state.canonical_form=state.canonical_form[:-1]+'C'
        state.save_data()
    assert state.canonical_form[-1]=='C'
    return state, time.perf_counter() - started

def build_direct_total_state(driver, states, coefficient, smat, nelec,
                             bond_dimension, root, identity_mpo, h_mpo,
                             eigenvalue, state_scratch,
                             warm_write_directory=None):
    """Deterministic exact block sum followed by bounded SU(2) compression."""
    started = time.perf_counter()
    coefficients = np.asarray(coefficient, dtype=float)
    if (not states or coefficients.shape != (len(states),) or
            not np.all(np.isfinite(coefficients)) or bond_dimension < 1):
        raise RuntimeError("Invalid deterministic total-state inputs")
    target = driver.bw.SX(nelec, 0, 0)
    if any(s.info.target != target or s.n_sites != driver.n_sites for s in states):
        raise RuntimeError("Direct SU(2) sum requires the current singlet, pg=0 sector")
    order = [i for i in range(len(states)) if coefficients[i] != 0.0]
    if not order:
        raise RuntimeError("Cannot construct a zero total-state coefficient vector")
    python_states = []
    common_basis = None
    for index in order:
        py, basis = sum_left_coupled_python(driver, states[index], f"TNVB-DIRECT-IN-{index}")
        if common_basis is not None and basis != common_basis:
            raise RuntimeError("Incompatible physical bases in direct SU(2) sum")
        common_basis = basis
        python_states.append(py)
    exact_py = sum_python(python_states, coefficients[order], range(len(order)))
    raw_bond = max_python_bond(exact_py)
    exact = sum_from_python(driver, exact_py.deep_copy(), common_basis, "TNVB-DIRECT-EXACT")
    build_seconds = time.perf_counter() - started
    total, compress_seconds = sum_native_compress(
        driver, exact, int(bond_dimension), "TNVB-DIRECT-TOTAL")
    actual_bond = max(reduced_bond_profile(total))
    hard_cap = False
    if actual_bond > bond_dimension:
        # Native center sweeps retain at least one vector per sector. If that
        # violates the requested cap, truncate the exact all-left-coupled block
        # tensors globally instead. This deletes whole sectors when necessary;
        # recoupling back to Block2 remains native. Quality is measured below,
        # never inferred from the Python reduced-tensor truncation estimate.
        cap_started = time.perf_counter()
        capped = exact_py.deep_copy()
        capped.compress(k=int(bond_dimension), cutoff=0.0, left=True)
        total = sum_from_python(driver, capped, common_basis, "TNVB-DIRECT-CAPPED")
        actual_bond = max(reduced_bond_profile(total))
        compress_seconds += time.perf_counter() - cap_started
        hard_cap = True
    if actual_bond > bond_dimension:
        raise RuntimeError(f"Direct sum compression exceeded cap: {actual_bond}>{bond_dimension}")
    quality = total_state_quality(
        driver, total, states, coefficients, smat, identity_mpo, h_mpo, eigenvalue)
    valid = sum_quality_passes(quality)
    # Store only a verified final state. Outer cache commit still occurs only
    # after successful RDM evaluation. No prefix manifest can seed another sum.
    if warm_write_directory:
        context = dict(version=2, constructor="SU2_DIRECT_SVD", nstr=len(states),
                       nsites=driver.n_sites, nelec=nelec, bond=int(bond_dimension), root=root)
        metadata = dict(context=context, valid=valid, quality=quality,
                        order=[i + 1 for i in order], prefixes={},
                        actual_bond=int(actual_bond))
        if valid:
            cached = driver.copy_mps(total, "TNVB-TOTAL-FINAL")
            persist_warm_start_mps(cached, state_scratch, warm_write_directory)
        path = os.path.join(warm_write_directory, "sum_manifest.json")
        with open(path + ".tmp", "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(path + ".tmp", path)
    log_summary(
        f"[TNVB-SUM] method=SU2_DIRECT_SVD structures={len(order)} "
        f"raw_bond={raw_bond} actual_bond={actual_bond} cap={bond_dimension} "
        f"hard_cap={int(hard_cap)} eta={quality['infidelity']:.3e} "
        f"fit_dE={quality['energy_error']:.3e} cache={'verified' if valid else 'invalid'} "
        f"build={build_seconds:.3f}s compress={compress_seconds:.3f}s")
    if quality["infidelity"] + quality["roundoff"] > SUM_INFIDELITY_TOL:
        log_summary(
            f"[TNVB-WARNING] sum_eta={quality['infidelity']:.6e} "
            f"roundoff={quality['roundoff']:.3e} reference={SUM_INFIDELITY_TOL:.1e}; "
            "direct total-state compression did not reach target fidelity; "
            "continuing RDM/gradient evaluation without a reusable sum cache. "
            "Gradients and optimized results may be inaccurate.")
    return total, quality, dict(method="SU2_DIRECT_SVD", actual_bond=actual_bond,
                               build_seconds=build_seconds, compress_seconds=compress_seconds)


def reduced_bond_profile(state):
    """Return actual retained SU(2) multiplet counts on every left bond."""
    state.info.load_mutable()
    return [int(sum(dimension.n_states)) for dimension in state.info.left_dims]


def pair_occupation_guess(rmat, pairs):
    """Exact empty-site mask and a soft (not physical-RDM) occupation hint.

    Only exactly zero rows of all consumed orbital columns are frozen empty.
    No magnitude threshold is used, including for tiny Cholesky tails.
    """
    columns = [i for pair in pairs for i in pair]
    vectors = np.asarray(rmat, dtype=float)[:, columns]
    if not np.all(np.isfinite(vectors)):
        raise RuntimeError("Nonfinite orbital transformation in pair initial guess")
    norms = np.sum(vectors * vectors, axis=0)
    if np.any(norms <= 0):
        raise RuntimeError("Zero orbital column in pair initial guess")
    support = np.any(vectors != 0.0, axis=1)
    count = int(np.count_nonzero(support))
    electrons = len(columns)
    if count == 0 or electrons > 2 * count:
        raise RuntimeError("Inconsistent pair target/support")
    weights = np.sum(vectors[support] ** 2 / norms, axis=1)
    # Spread a small amount of hint probability over every allowed site.
    # Project onto [0,2] with the correct total, without freezing occupied sites.
    weights = 0.9 * weights + 0.1 * electrons / count
    if electrons == 2 * count:
        projected = np.full(count, 2.0)
    else:
        lo, hi = -float(np.max(weights)) - 2.0, 2.0
        for _ in range(80):
            mid = (lo + hi) * 0.5
            if np.clip(weights + mid, 1e-6, 2.0 - 1e-6).sum() < electrons:
                lo = mid
            else:
                hi = mid
        projected = np.clip(weights + (lo + hi) * 0.5, 1e-6, 2.0 - 1e-6)
    occs = np.zeros(len(support))
    occs[support] = projected
    return occs, ''.join('A' if x else 'V' for x in support)


def pair_leading_csf(nsites, pairs):
    """A singlet CSF with the consumed orbitals' leading occupations.

    With the upper-triangular orbital map, the leading occupation component
    is retained. Adjacent coupling of the sorted singly occupied sites has
    nonzero recoupling overlap with any singlet pairing of those sites.
    This is an initial guess, not the nonorthogonal target wavefunction.
    """
    counts = np.zeros(nsites, dtype=int)
    for left, right in pairs:
        counts[left] += 1
        counts[right] += 1
    if np.any(counts > 2):
        raise RuntimeError("Invalid leading CSF occupation")
    pattern = []
    single = 0
    for count in counts:
        if count == 1:
            pattern.append('+' if single % 2 == 0 else '-')
            single += 1
        else:
            pattern.append('2' if count == 2 else '0')
    if single % 2:
        raise RuntimeError("Leading CSF is not a singlet")
    return ''.join(pattern)


def pair_center_norm(state):
    """Read the canonical wavefunction norm, without normalizing a trial."""
    center = state.center
    if state.dot == 2 and state.canonical_form[center] == 'L':
        center += 1
    if center >= state.n_sites or state.canonical_form[center] not in 'CKS':
        raise RuntimeError("Pair fitting returned an invalid canonical center")
    if state.tensors[center] is None:
        raise RuntimeError("Pair fitting returned a missing center tensor")
    if state.tensors[center].total_memory == 0:
        state.load_tensor(center)
    return float(state.tensors[center].norm())


def continue_pair_fit(initial, multiply, center_norm, initial_sweeps,
                      tolerance, report):
    """Two bounded attempts; continue the same nonzero, unnormalized trial.

    A stable norm across batches is a stopping diagnostic, not a residual or
    fidelity certificate. Each attempt uses at most max(12, initial_sweeps+4)
    sweeps. Failed/nonconverged candidates are never normalized or cached.
    """
    budget = max(12, initial_sweeps + 4)
    total_sweeps = 0
    for attempt in range(2):
        output = initial(attempt)
        previous = None
        used = 0
        while used < budget:
            batch = initial_sweeps if used == 0 else min(2, budget - used)
            fitted = float(multiply(output, batch))
            used += batch
            total_sweeps += batch
            norm = center_norm(output)
            if not (math.isfinite(fitted) and fitted > 0 and
                    math.isfinite(norm) and norm > 0):
                reason = 'ZERO_OR_NONFINITE'
                break
            if previous is not None and abs(fitted - previous) <= tolerance * max(1.0, abs(fitted)):
                return output, fitted, dict(attempts=attempt + 1, sweeps=total_sweeps)
            previous = fitted
        else:
            reason = 'NORM_NOT_STABLE'
        report(attempt + 1, reason, used, fitted)
    raise RuntimeError("SU2 pair fitting failed after two bounded attempts; "
                       f"reason={reason} sweeps={total_sweeps}")


def build_su2_states(
    driver,
    identity_mpo,
    ntstr,
    nstr: int,
    nao: int,
    nae: int,
    nel: int,
    tnbond: int,
    rmat,
    structure_indices=None,
    state_scratch=None,
    warm_read_directory=None,
    warm_write_directory=None,
):
    """Build all physical VB structures by variational SU(2) pair products."""
    sweeps = int(os.environ.get("XMVB_BLOCK2_SU2_SWEEPS", "4"))
    tolerance = float(os.environ.get("XMVB_BLOCK2_SU2_TOL", "1e-10"))
    # Trim negligible pair-fit subspaces before summing structures and building
    # high-rank RDM environments. Keep zero cutoff as an explicit override.
    cutoff = float(os.environ.get("XMVB_BLOCK2_SU2_CUTOFF", "1e-14"))
    seed = int(os.environ.get("XMVB_BLOCK2_SU2_SEED", "20260813"))
    iprint = int(os.environ.get("XMVB_BLOCK2_SU2_IPRINT", "0"))
    pair_log = os.environ.get("XMVB_BLOCK2_SU2_PAIR_LOG", "0") == "1"
    if sweeps < 1:
        raise RuntimeError("XMVB_BLOCK2_SU2_SWEEPS must be positive")
    if tolerance <= 0.0:
        raise RuntimeError("XMVB_BLOCK2_SU2_TOL must be positive")
    if cutoff < 0.0:
        raise RuntimeError("XMVB_BLOCK2_SU2_CUTOFF must be nonnegative")

    bond_dimension = max(2, tnbond)
    vacuum = driver.get_mps_from_csf_coefficients(
        dets=["0" * nao],
        dvals=np.asarray([1.0]),
        tag="SU2-VACUUM",
        target=driver.bw.SX(0, 0, 0),
        dot=2,
        iprint=0,
    )
    pair_cache = {}
    states = []
    if structure_indices is None:
        structure_indices = range(nstr)
    if (warm_read_directory or warm_write_directory) and not state_scratch:
        raise RuntimeError("TNVB warm starts require the driver scratch path")
    for istr in structure_indices:
        structure_started = time.perf_counter()
        pairs, phase = structure_pair_sequence(ntstr[:, istr], nao, nae, nel)
        state = vacuum
        max_pair_seconds = 0.0
        last_fit_norm = 0.0
        warm_pairs = 0
        total_pair_sweeps = 0
        retried_pairs = 0
        for ipair, (left, right) in enumerate(pairs):
            pair_started = time.perf_counter()
            scale = phase if ipair == 0 else 1.0
            key = (left, right, scale)
            if key not in pair_cache:
                pair_cache[key] = build_su2_pair_mpo(
                    driver, rmat, left, right, scale
                )
            target = driver.bw.SX(2 * (ipair + 1), 0, 0)
            tag = f"SU2-S{istr + 1}-P{ipair + 1}"
            output = load_warm_start_mps(
                driver,
                state_scratch,
                warm_read_directory,
                tag,
                bond_dimension,
            )
            used_warm_start = output is not None
            if used_warm_start:
                warm_pairs += 1
            warm_output = output
            occupations, mask = pair_occupation_guess(rmat, pairs[:ipair + 1])

            def initial(attempt):
                if attempt == 0 and warm_output is not None:
                    return warm_output
                driver.bw.b.Random.rand_seed(seed + 1000 * istr + ipair + 104729 * attempt)
                if attempt == 0:
                    return driver.get_mps_from_csf_coefficients(
                        dets=[pair_leading_csf(nao, pairs[:ipair + 1])],
                        dvals=np.asarray([1.0]), tag=tag, target=target, dot=2,
                        full_fci=False, casci_nvirt=mask.count('V'),
                        casci_mask=mask if 'V' in mask else None, iprint=0,
                    )
                hints = occupations.copy()
                if attempt:
                    # A second supported distribution, not unrestricted noise.
                    allowed = np.array([x == 'A' for x in mask])
                    hints[allowed] = 0.5 * hints[allowed] + (ipair + 1) / np.count_nonzero(allowed)
                return driver.get_random_mps(
                    tag=tag if attempt == 0 else tag + '-RETRY',
                    bond_dim=bond_dimension, target=target, dot=2,
                    full_fci=False, occs=hints.tolist(),
                    casci_nvirt=mask.count('V'),
                    casci_mask=mask if 'V' in mask else None,
                )

            def multiply(trial, batch):
                return driver.multiply(
                    trial,
                    pair_cache[key],
                    state,
                    n_sweeps=batch,
                    tol=tolerance,
                    bra_bond_dims=[bond_dimension],
                    cutoff=cutoff,
                    iprint=iprint,
                )

            def report(attempt, reason, used, fitted):
                log_summary(f"[TNVB-PAIR-RETRY] structure={istr + 1} pair={ipair + 1} "
                            f"attempt={attempt}/2 reason={reason} sweeps={used} fit_norm={fitted:.6e}")

            output, last_fit_norm, details = continue_pair_fit(
                initial, multiply, pair_center_norm, sweeps, tolerance, report
            )
            total_pair_sweeps += details['sweeps']
            retried_pairs += details['attempts'] - 1
            if details['attempts'] > 1:
                output = driver.copy_mps(output, tag)
            normalize_block2_mps(output)
            persist_warm_start_mps(
                output, state_scratch, warm_write_directory
            )
            elapsed = time.perf_counter() - pair_started
            max_pair_seconds = max(max_pair_seconds, elapsed)
            if pair_log:
                _builtin_print(
                    f"[TNVB-BLOCK2-SU2] structure={istr + 1} "
                    f"pair={ipair + 1}/{len(pairs)} orbitals=({left + 1},{right + 1}) "
                    f"initial={'warm' if used_warm_start else 'occupation-csf'} "
                    f"empty_sites={mask.count('V')} sweeps={details['sweeps']} attempts={details['attempts']} "
                    f"fit_norm={last_fit_norm:.12e} seconds={elapsed:.3f}",
                    flush=True,
                )
            state = output

        profile = reduced_bond_profile(state)
        norm2 = float(driver.expectation(state, identity_mpo, state))
        states.append(state)
        print(
            f"[TNVB-BLOCK2] physical_MPS progress={istr + 1}/{nstr} "
            f"symmetry=SU2 builder=variational-pair pairs={len(pairs)} "
            f"warm_pairs={warm_pairs}/{len(pairs)} "
            f"sweeps={sweeps} reduced_bond={max(profile)} "
            f"physical_norm2={norm2:.12e} last_fit_norm={last_fit_norm:.12e} "
            f"max_pair_seconds={max_pair_seconds:.3f} "
            f"structure_seconds={time.perf_counter() - structure_started:.3f}",
            flush=True,
        )
    del pair_cache, vacuum
    return states


def allocated_cpu_count():
    """Return the CPU allocation without introducing an XMVB input keyword."""
    slurm_value = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
    if slurm_value:
        # Slurm normally exports a plain integer.  Accept a possible suffix so
        # an unusual site-local representation cannot disable parallelism.
        digits = ""
        for character in slurm_value:
            if not character.isdigit():
                break
            digits += character
        if digits and int(digits) > 0:
            return int(digits), "SLURM_CPUS_PER_TASK"
    try:
        affinity_count = len(os.sched_getaffinity(0))
        if affinity_count > 0:
            return affinity_count, "CPU-affinity"
    except (AttributeError, OSError):
        pass
    cpu_count = os.cpu_count() or 1
    return max(1, int(cpu_count)), "os.cpu_count"


def positive_environment_integer(name: str):
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        result = int(value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a positive integer") from error
    if result < 1:
        raise RuntimeError(f"{name} must be a positive integer")
    return result


def memory_allocation_bytes():
    """Return a Slurm memory limit when it is unambiguous for this one task."""
    for name, per_cpu in (
        ("SLURM_MEM_PER_NODE", False),
        ("SLURM_MEM_PER_CPU", True),
    ):
        raw = os.environ.get(name, "").strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        # Slurm uses zero for an unconstrained/default memory request.
        if value <= 0:
            continue
        if per_cpu:
            allocated, _ = allocated_cpu_count()
            value *= allocated
        return value * 1024**2, name
    return None, "unavailable"


def parallel_phase_layout(task_count: int, nao: int, tnbond: int, phase: str):
    """Choose process/thread layout from allocation and useful work.

    Tiny contractions do not amortize Python startup, MPS file copying, and
    Block2 driver initialization.  The work score prevents a small active
    space from spawning one process per available core, while D=192 C20 still
    exposes enough work to occupy a 16-core allocation.

    ``XMVB_BLOCK2_WORKERS`` is deliberately only an environment override for
    regression and benchmarking.  It is not, and must not become, an input
    keyword because worker placement is not part of the wavefunction model.
    """
    allocated, source = allocated_cpu_count()
    if task_count < 1:
        raise RuntimeError("TNVB parallel layout requires a positive task count")
    if phase not in ("state", "hs"):
        raise RuntimeError("internal TNVB parallel phase must be state or hs")

    thread_override = positive_environment_integer("XMVB_BLOCK2_THREADS")
    worker_override = positive_environment_integer("XMVB_BLOCK2_WORKERS")
    block2_threads = min(thread_override or 1, allocated)
    process_capacity = max(1, allocated // block2_threads)

    per_task_work = nao * max(2, tnbond) ** 2
    target_work = 10_000 if phase == "state" else 24_000
    useful_workers = max(1, math.ceil(task_count * per_task_work / target_work))
    process_capacity = min(process_capacity, useful_workers)
    source += "+work-model"

    memory_bytes, memory_source = memory_allocation_bytes()
    if memory_bytes is not None:
        # Each process owns a Block2 stack, Hamiltonian MPO and all reloaded
        # structure MPSs.  This is deliberately conservative: it trades a
        # little concurrency for avoiding a late Slurm OOM kill.
        per_process = 4 * stack_memory_bytes(nao, tnbond)
        per_process += 6 * nao * max(2, tnbond) ** 2 * 8
        memory_capacity = max(1, int(0.80 * memory_bytes) // per_process)
        process_capacity = min(process_capacity, memory_capacity)
        source += f"+{memory_source}"
    if worker_override is not None:
        process_capacity = min(process_capacity, worker_override)
        source += "+XMVB_BLOCK2_WORKERS"

    workers = min(max(1, task_count), process_capacity)
    # When only a few genuinely heavy tasks exist, give each process the idle
    # cores.  Do not do this for tiny tensors: Block2 team startup costs more
    # than the contraction itself there.
    if thread_override is None and per_task_work >= 250_000:
        block2_threads = max(1, allocated // workers)
    thread_source = "XMVB_BLOCK2_THREADS" if thread_override else "work-model"
    return workers, block2_threads, allocated, source, thread_source


def configure_phase_environment(block2_threads: int):
    """Set the environment inherited by newly spawned Block2 workers."""
    os.environ["OMP_NUM_THREADS"] = str(block2_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["GOTO_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"


def rdm_thread_count(nao: int, tnbond: int):
    """Choose Block2 threads for total-state fitting and 1/2/3-RDMs."""
    allocated, source = allocated_cpu_count()
    override = positive_environment_integer("XMVB_BLOCK2_RDM_THREADS")
    if override is not None:
        return min(override, allocated), source + "+XMVB_BLOCK2_RDM_THREADS"
    # A three-body density has a much steeper orbital/bond dependence than an
    # H/S element.  This model intentionally leaves 6e/6o D=32 serial.  Do not
    # use more than 16 shared-memory threads automatically: on the C20
    # Dsum=448 NPDM, 32 threads increased peak RSS from about 38 GiB to 95 GiB
    # and reduced effective CPU utilization.  An explicit benchmark override
    # remains possible through XMVB_BLOCK2_RDM_THREADS.
    work = nao**3 * max(2, tnbond) ** 2
    useful = max(1, math.ceil(work / 2_000_000))
    return min(allocated, useful, 16), source + "+work-model-cap16"


def hs_chunks(pairs, worker_count: int):
    """Create enough small chunks for dynamic H/S load balancing."""
    configured = os.environ.get("XMVB_BLOCK2_HS_CHUNK_SIZE", "").strip()
    if configured:
        try:
            chunk_size = int(configured)
        except ValueError as error:
            raise RuntimeError(
                "XMVB_BLOCK2_HS_CHUNK_SIZE must be a positive integer"
            ) from error
        if chunk_size < 1:
            raise RuntimeError(
                "XMVB_BLOCK2_HS_CHUNK_SIZE must be a positive integer"
            )
    else:
        # Four scheduling waves normally hide the variation between diagonal
        # and off-diagonal Hamiltonian contractions without excessive IPC.
        chunk_size = max(1, math.ceil(len(pairs) / (4 * worker_count)))
    return [pairs[start : start + chunk_size] for start in range(0, len(pairs), chunk_size)], chunk_size


def persist_parallel_states(states, scratch: str):
    """Make final MPS tensor and metadata files reloadable by child drivers."""
    tags = []
    for state in states:
        tag = str(state.info.tag)
        state.save_data()
        state.info.save_data(os.path.join(scratch, f"{tag}-mps_info.bin"))
        tags.append(tag)
    return tags


def copy_parallel_state_files(source: str, destination: str, tags):
    """Copy only final MPS files, excluding all pair-product intermediates."""
    names = hs_shared.selected_names(os.listdir(source), tags)
    copied = 0
    for name in names:
        source_path = os.path.join(source, name)
        if os.path.isfile(source_path):
            shutil.copyfile(source_path, os.path.join(destination, name))
            copied += 1
    for tag in tags:
        info_path = os.path.join(destination, f"{tag}-mps_info.bin")
        if not os.path.isfile(info_path):
            raise RuntimeError(f"parallel H/S cannot find persisted MPS tag {tag}")
    return copied


def move_parallel_state_files(source: str, destination: str, tags):
    """Transfer final private files after their last construction/HS reader.

    Both directories belong to one request on the same filesystem. Never use
    this for a shared bank, warm cache or a live native MPS owner.
    """
    names = hs_shared.selected_names(os.listdir(source), tags)
    for tag in tags:
        if not os.path.isfile(os.path.join(source, f"{tag}-mps_info.bin")):
            raise RuntimeError(f"retained state cannot find MPS tag {tag}")
    moved = 0
    for name in names:
        path = os.path.join(source, name)
        if os.path.isfile(path):
            os.replace(path, os.path.join(destination, name))
            moved += 1
    return moved


def warm_cache_manifest(nstr, nao, nae, nel, tnbond, sum_bond, ntstr, rmat=None):
    """Return the compatibility key for cross-call MPS initial guesses.

    The active-orbital transformation is intentionally absent: it changes at
    every macro iteration, while the previous MPS is precisely the desired
    initial guess for the new transformation.  Everything that changes the
    Hilbert space, VB structure ordering, or fixed bond limits is included.
    """
    structure_bytes = np.asarray(
        ntstr, dtype="<i4", order="F"
    ).tobytes(order="F")
    return {
        "schema": 1,
        "symmetry": "SU2",
        "pair_builder": 3,  # occupation-matched initialization and bounded continuation
        "pair_initializer": "occupation-csf-v2",
        "orbital_support_sha256": None if rmat is None else hashlib.sha256(
            np.asarray(rmat != 0.0, dtype=np.uint8).tobytes()).hexdigest(),
        "pair_fit": {name: os.environ.get(name, default) for name, default in (
            ('XMVB_BLOCK2_SU2_SWEEPS', '4'), ('XMVB_BLOCK2_SU2_TOL', '1e-10'),
            ('XMVB_BLOCK2_SU2_CUTOFF', '1e-14'), ('XMVB_BLOCK2_SU2_SEED', '20260813'))},
        "nstr": int(nstr),
        "nao": int(nao),
        "nae": int(nae),
        "nel": int(nel),
        "tnbond": int(tnbond),
        "sum_bond": int(sum_bond),
        "structures_sha256": hashlib.sha256(structure_bytes).hexdigest(),
    }


def prepare_warm_cache(manifest):
    """Create a transactional write generation and locate a compatible one."""
    root = os.environ.get("XMVB_BLOCK2_WARM_CACHE", "").strip()
    if not root:
        return None, None, None
    root = os.path.abspath(root)
    os.makedirs(root, exist_ok=True)
    current = os.path.join(root, "current")
    previous = os.path.join(root, "previous")
    # Recover the last complete generation if a process died between the two
    # atomic renames used by commit_warm_cache.
    if not os.path.isdir(current) and os.path.isdir(previous):
        os.replace(previous, current)

    read_directory = None
    manifest_path = os.path.join(current, "manifest.json")
    try:
        with open(manifest_path, "r", encoding="utf-8") as stream:
            cached_manifest = json.load(stream)
        if cached_manifest == manifest:
            read_directory = current
    except (FileNotFoundError, OSError, ValueError, TypeError):
        read_directory = None

    write_directory = tempfile.mkdtemp(prefix="candidate-", dir=root)
    atexit.register(shutil.rmtree, write_directory, ignore_errors=True)
    print(
        "[TNVB-BLOCK2-WARM] "
        f"cache={'hit' if read_directory else 'miss'} root={root}",
        flush=True,
    )
    return read_directory, write_directory, root


def commit_warm_cache(root, write_directory, manifest):
    """Atomically promote a complete MPS generation after a successful call."""
    if not root or not write_directory:
        return
    manifest_temp = os.path.join(write_directory, "manifest.json.tmp")
    manifest_path = os.path.join(write_directory, "manifest.json")
    with open(manifest_temp, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, sort_keys=True, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(manifest_temp, manifest_path)

    current = os.path.join(root, "current")
    previous = os.path.join(root, "previous")
    shutil.rmtree(previous, ignore_errors=True)
    if os.path.isdir(current):
        os.replace(current, previous)
    os.replace(write_directory, current)
    shutil.rmtree(previous, ignore_errors=True)
    print("[TNVB-BLOCK2-WARM] promoted complete generation", flush=True)


def load_warm_start_mps(
    driver, state_scratch, read_directory, tag, bond_dimension
):
    """Load one compatible previous MPS as a variational initial guess."""
    if not read_directory:
        return None
    if not os.path.isfile(os.path.join(read_directory, f"{tag}-mps_info.bin")):
        return None
    try:
        copy_parallel_state_files(read_directory, state_scratch, [tag])
        state = driver.load_mps(tag)
        state.info.load_mutable()
        state.info.bond_dim = int(bond_dimension)
        state.info.deallocate_mutable()
        return state
    except Exception as error:
        print(
            f"[TNVB-BLOCK2-WARM] tag={tag} load_failed={error}",
            flush=True,
        )
        return None


def persist_warm_start_mps(state, state_scratch, write_directory):
    """Save one converged fixed-bond MPS into the candidate generation."""
    if not write_directory:
        return 0
    tags = persist_parallel_states([state], state_scratch)
    return copy_parallel_state_files(state_scratch, write_directory, tags)


def initialize_parallel_state_worker(
    scratch_parent,
    state_store,
    nao,
    nae,
    nel,
    tnbond,
    ntstr,
    rmat,
    block2_threads,
    warm_read_directory,
    warm_write_directory,
):
    """Initialize one process that constructs a subset of independent MPSs."""
    global _PARALLEL_STATE_CONTEXT
    configure_phase_environment(block2_threads)
    scratch = tempfile.mkdtemp(
        prefix=f"tnvb-block2-state-{os.getpid()}-", dir=scratch_parent
    )
    atexit.register(shutil.rmtree, scratch, ignore_errors=True)
    driver = DMRGDriver(
        scratch=scratch,
        symm_type=SymmetryTypes.SU2,
        n_threads=block2_threads,
        stack_mem=stack_memory_bytes(nao, tnbond),
        clean_scratch=True,
    )
    driver.initialize_system(n_sites=nao, n_elec=nae, spin=0, orb_sym=[0] * nao)
    identity_mpo = driver.get_identity_mpo()
    _PARALLEL_STATE_CONTEXT = (
        driver,
        identity_mpo,
        scratch,
        state_store,
        nao,
        nae,
        nel,
        tnbond,
        ntstr,
        rmat,
        warm_read_directory,
        warm_write_directory,
    )


def build_parallel_state_batch(structure_indices):
    """Build, persist, and export one deterministic subset of SU(2) states."""
    if _PARALLEL_STATE_CONTEXT is None:
        raise RuntimeError("parallel MPS worker was not initialized")
    (
        driver,
        identity_mpo,
        scratch,
        state_store,
        nao,
        nae,
        nel,
        tnbond,
        ntstr,
        rmat,
        warm_read_directory,
        warm_write_directory,
    ) = _PARALLEL_STATE_CONTEXT
    started = time.perf_counter()
    states = build_su2_states(
        driver,
        identity_mpo,
        ntstr,
        ntstr.shape[1],
        nao,
        nae,
        nel,
        tnbond,
        rmat,
        structure_indices=structure_indices,
        state_scratch=scratch,
        warm_read_directory=warm_read_directory,
        warm_write_directory=warm_write_directory,
    )
    tags = persist_parallel_states(states, scratch)
    copied = copy_parallel_state_files(scratch, state_store, tags)
    print(
        f"[TNVB-BLOCK2-PARALLEL] state_worker={os.getpid()} "
        f"structures={','.join(str(index + 1) for index in structure_indices)} "
        f"copied_files={copied} seconds={time.perf_counter() - started:.3f}",
        flush=True,
    )
    return [(index, tag) for index, tag in zip(structure_indices, tags)]


def build_parallel_su2_states(
    scratch_parent,
    state_store,
    nao,
    nae,
    nel,
    tnbond,
    ntstr,
    rmat,
    worker_count,
    block2_threads,
    warm_read_directory=None,
    warm_write_directory=None,
):
    """Construct independent VB structures concurrently and return ordered tags."""
    structure_indices = list(range(ntstr.shape[1]))
    batches = [structure_indices[index::worker_count] for index in range(worker_count)]
    configure_phase_environment(block2_threads)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=initialize_parallel_state_worker,
        initargs=(
            scratch_parent,
            state_store,
            nao,
            nae,
            nel,
            tnbond,
            ntstr,
            rmat,
            block2_threads,
            warm_read_directory,
            warm_write_directory,
        ),
    ) as executor:
        batch_results = list(executor.map(build_parallel_state_batch, batches))
    tagged = [item for batch in batch_results for item in batch]
    tagged.sort(key=lambda item: item[0])
    return [tag for _, tag in tagged]


def initialize_parallel_hs_worker(
    source_scratch,
    scratch_parent,
    symmetry,
    nao,
    nae,
    tnbond,
    h_orth,
    g_orth,
    state_tags,
    block2_threads,
):
    """Initialize one process-private Block2 driver and reload all final MPSs."""
    global _PARALLEL_HS_CONTEXT
    if symmetry != "SU2":
        raise ValueError("parallel H/S requires the SU(2) state bank")
    initialized = time.perf_counter()
    # Prevent inherited parent settings from multiplying BLAS/OpenMP threads
    # inside every task process.  DMRGDriver additionally configures Block2's
    # OpenMP runtime explicitly through ``n_threads``.
    configure_phase_environment(block2_threads)
    scratch = tempfile.mkdtemp(
        prefix=f"tnvb-block2-hs-{os.getpid()}-", dir=scratch_parent
    )
    atexit.register(shutil.rmtree, scratch, ignore_errors=True)
    copied = copy_parallel_state_files(source_scratch, scratch, state_tags)
    driver = DMRGDriver(
        scratch=scratch,
        symm_type=SymmetryTypes.SU2,
        n_threads=block2_threads,
        stack_mem=stack_memory_bytes(nao, tnbond),
        clean_scratch=True,
    )
    driver.initialize_system(n_sites=nao, n_elec=nae, spin=0, orb_sym=[0] * nao)
    h_mpo = get_hamiltonian_mpo(driver, h_orth, g_orth, log=False)
    identity_mpo = driver.get_identity_mpo()
    states = [driver.load_mps(tag) for tag in state_tags]
    _PARALLEL_HS_CONTEXT = (driver, identity_mpo, h_mpo, states)
    print(
        f"[TNVB-BLOCK2-PARALLEL] worker={os.getpid()} "
        f"loaded_states={len(states)} copied_files={copied} "
        f"init_seconds={time.perf_counter() - initialized:.3f}",
        flush=True,
    )


def contract_parallel_hs_batch(batch):
    """Contract a deterministic batch of independent upper-triangle elements."""
    if _PARALLEL_HS_CONTEXT is None:
        raise RuntimeError("parallel H/S worker was not initialized")
    driver, identity_mpo, h_mpo, states = _PARALLEL_HS_CONTEXT
    result = []
    for order, i, j in batch:
        overlap_started = time.perf_counter()
        overlap = float(driver.expectation(states[i], identity_mpo, states[j]))
        overlap_seconds = time.perf_counter() - overlap_started
        hamiltonian_started = time.perf_counter()
        hamiltonian = float(driver.expectation(states[i], h_mpo, states[j]))
        hamiltonian_seconds = time.perf_counter() - hamiltonian_started
        result.append(
            (
                order,
                i,
                j,
                overlap,
                hamiltonian,
                overlap_seconds,
                hamiltonian_seconds,
                os.getpid(),
            )
        )
    return result


def contract_parallel_hs(
    states,
    source_scratch,
    scratch_parent,
    symmetry,
    nao,
    nae,
    tnbond,
    h_orth,
    g_orth,
    pairs,
    worker_count,
    block2_threads,
    state_tags=None,
):
    """Run H/S batches in spawned processes and return canonical-order results."""
    if states is not None:
        state_tags = persist_parallel_states(states, source_scratch)
    elif state_tags is None:
        raise RuntimeError("parallel H/S requires states or persisted state tags")
    batches, chunk_size = hs_chunks(pairs, worker_count)
    print(
        f"[TNVB-BLOCK2-PARALLEL] H/S scheduler chunks={len(batches)} "
        f"chunk_size={chunk_size}",
        flush=True,
    )
    configure_phase_environment(block2_threads)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=initialize_parallel_hs_worker,
        initargs=(
            source_scratch,
            scratch_parent,
            symmetry,
            nao,
            nae,
            tnbond,
            h_orth,
            g_orth,
            state_tags,
            block2_threads,
        ),
    ) as executor:
        batch_results = list(
            executor.map(contract_parallel_hs_batch, batches, chunksize=1)
        )
    results = [item for batch in batch_results for item in batch]
    results.sort(key=lambda item: item[0])
    return results


def compute_total_state_rdms(
    source_scratch,
    scratch_parent,
    state_tags,
    nao,
    nae,
    tnbond,
    sum_bond,
    root,
    hmat,
    smat,
    h_orth,
    g_orth,
    warm_write_directory=None,
    selected_eigenpair=None,
    consume_source=False,
):
    """Reload final structures, form one weighted state, and compute its RDMs."""
    started = time.perf_counter()
    rdm_scratch = (source_scratch if consume_source else
                   tempfile.mkdtemp(prefix="tnvb-block2-rdm-", dir=scratch_parent))
    try:
        copied = (0 if consume_source else
                  copy_parallel_state_files(source_scratch, rdm_scratch, state_tags))
        rdm_threads, allocation_source = rdm_thread_count(nao, sum_bond)
        rdm_stack_bytes = rdm_stack_memory_bytes(nao, sum_bond)
        configure_phase_environment(rdm_threads)
        driver = DMRGDriver(
            scratch=rdm_scratch,
            symm_type=SymmetryTypes.SU2,
            n_threads=rdm_threads,
            stack_mem=rdm_stack_bytes,
            clean_scratch=True,
        )
        driver.initialize_system(
            n_sites=nao, n_elec=nae, spin=0, orb_sym=[0] * nao
        )
        states = [driver.load_mps(tag) for tag in state_tags]
        identity_mpo = driver.get_identity_mpo()
        h_mpo = get_hamiltonian_mpo(driver, h_orth, g_orth, log=False)
        if selected_eigenpair is None:
            eigenvalue, coefficient = solve_generalized_state(hmat, smat, root)
        else:
            eigenvalue, coefficient = selected_eigenpair
            coefficient = np.asarray(coefficient, dtype=float)
            sc = smat @ coefficient
            residual = hmat @ coefficient - eigenvalue * sc
            if (root != 0 or not np.isfinite(eigenvalue) or
                    not np.all(np.isfinite(coefficient)) or
                    abs(coefficient @ sc - 1.0) > 1e-9 or
                    np.linalg.norm(residual / np.sqrt(np.diag(smat))) > 1e-9):
                raise RuntimeError("invalid selected eigenpair supplied to TNVB RDM")
        print(
            f"[TNVB-BLOCK2-RDM] total_state start root={root + 1} "
            f"structures={len(states)} structure_bond={tnbond} "
            f"sum_bond={sum_bond} threads={rdm_threads} "
            f"stack_gib={rdm_stack_bytes / 1024**3:.3f} "
            f"copied_files={copied} allocation_source={allocation_source}",
            flush=True,
        )
        sum_started = time.perf_counter()
        total, sum_quality, sum_details = build_direct_total_state(
            driver,
            states,
            coefficient,
            smat,
            nae,
            max(2, sum_bond),
            root,
            identity_mpo,
            h_mpo,
            eigenvalue,
            state_scratch=rdm_scratch,
            warm_write_directory=warm_write_directory,
        )
        norm = sum_quality["norm"]
        total_energy = sum_quality["energy"]
        energy_error = sum_quality["energy_error"]
        sum_seconds = time.perf_counter() - sum_started
        print(
            f"[TNVB-BLOCK2-RDM] total_state complete norm={norm:.17e} "
            f"eigenvalue={eigenvalue:.17e} expectation={total_energy:.17e} "
            f"energy_error={energy_error:.3e} "
            f"seconds={sum_seconds:.3f}",
            flush=True,
        )
        warn_total_state_fit(energy_error, tnbond, sum_bond)

        rdm_started = time.perf_counter()
        npdm_algorithm, npdm_algorithm_name = npdm_algorithm_type()
        print(
            f"[TNVB-BLOCK2-RDM] npdm_algorithm={npdm_algorithm_name}",
            flush=True,
        )
        print("[TNVB-BLOCK2-RDM-STAGE] dm1 start", flush=True)
        stage_started = time.perf_counter()
        dm1 = np.asarray(
            driver.get_1pdm(total, algo_type=npdm_algorithm, iprint=0),
            dtype=np.float64,
        ) / norm
        dm1_seconds = time.perf_counter() - stage_started
        print(
            f"[TNVB-BLOCK2-RDM-STAGE] dm1 complete seconds={dm1_seconds:.3f}",
            flush=True,
        )
        print("[TNVB-BLOCK2-RDM-STAGE] dm2 start", flush=True)
        stage_started = time.perf_counter()
        dm2 = np.asarray(
            driver.get_2pdm(total, algo_type=npdm_algorithm, iprint=0),
            dtype=np.float64,
        ) / norm
        dm2_seconds = time.perf_counter() - stage_started
        print(
            f"[TNVB-BLOCK2-RDM-STAGE] dm2 complete seconds={dm2_seconds:.3f}",
            flush=True,
        )
        if os.environ.get("XMVB_BLOCK2_FORBID_3PDM", "0") == "1":
            raise RuntimeError("rank-six density construction is forbidden for this run")
        log_summary(
            f"[TNVB-RDM] dm3 start active={nao} bond={sum_bond} "
            f"threads={rdm_threads}"
        )
        stage_started = time.perf_counter()
        dm3 = np.asarray(rdm_full.get_profiled_3pdm(
            driver, total, npdm_algorithm, log_summary), dtype=np.float64)
        dm3_seconds = time.perf_counter() - stage_started
        (contracted_orth, particle3,
         reduction_times) = rdm_full.reduce_full_3pdm(dm3, g_orth, norm)
        dm3_reduce_seconds = reduction_times["integral_seconds"]
        stage_started = time.perf_counter()
        contraction3_error = float(np.max(np.abs(particle3 - (nae - 2) * dm2)))
        dm3_check_seconds = (reduction_times["particle_seconds"] +
                             time.perf_counter() - stage_started)
        tensor_gib = dm3.nbytes / 1024**3
        # Release the rank-six output; only O(K**4) and O(K**2) reductions survive.
        del dm3, particle3
        log_summary(
            f"[TNVB-3RDM] algorithm={npdm_algorithm_name} "
            f"actual_bond={sum_details.get('actual_bond', '-')} cap={sum_bond} "
            f"native={dm3_seconds:.3f}s reduce={dm3_reduce_seconds:.3f}s "
            f"check={dm3_check_seconds:.3f}s tensor={tensor_gib:.3f}GiB "
            "full_copy=0"
        )
        pmat, pit, contracted_d3 = orthogonal_reduced_data(dm1, dm2, contracted_orth)
        reconstructed = float(np.einsum("ij,ij", h_orth, dm1))
        reconstructed += 0.5 * float(np.einsum("ijkl,iklj", g_orth, dm2))
        trace_error = abs(float(np.trace(dm1)) - nae)
        contraction2_error = float(
            np.max(np.abs(np.einsum("ijja->ia", dm2) - (nae - 1) * dm1))
        )
        rdm_energy_error = abs(reconstructed - total_energy)
        identity_error = max(
            trace_error,
            contraction2_error,
            contraction3_error,
        )
        rdm_total_seconds = time.perf_counter() - started
        print(
            f"[TNVB-BLOCK2-RDM] complete trace1_error={trace_error:.3e} "
            f"contract2_error={contraction2_error:.3e} "
            f"contract3_error={contraction3_error:.3e} "
            f"energy_error={rdm_energy_error:.3e} "
            f"P_elements={pmat.size} Pi_elements={pit.size} "
            f"D3G_elements={contracted_d3.size} "
            f"rdm_seconds={time.perf_counter() - rdm_started:.3f} "
            f"total_seconds={time.perf_counter() - started:.3f}",
            flush=True,
        )
        log_summary(
            f"[TNVB-RDM] sum={sum_seconds:.3f}s dm1={dm1_seconds:.3f}s "
            f"dm2={dm2_seconds:.3f}s dm3={dm3_seconds:.3f}s "
            f"total={rdm_total_seconds:.3f}s "
            f"fit_dE={energy_error:.3e} RDM_dE={rdm_energy_error:.3e} "
            f"identity_max={identity_error:.3e} norm_error={abs(1.0 - norm):.3e}"
        )
        rdm_tolerance = float(os.environ.get("XMVB_BLOCK2_RDM_CHECK_TOL", "1e-7"))
        check_total_state_rdm(
            trace_error,
            contraction2_error,
            contraction3_error,
            rdm_energy_error,
            rdm_tolerance,
        )
        diagnostics = dict(identity_error=identity_error,
                           rdm_energy_error=rdm_energy_error,
                           total_seconds=rdm_total_seconds,
                           sum_seconds=sum_seconds, dm1_seconds=dm1_seconds,
                           dm2_seconds=dm2_seconds, dm3_seconds=dm3_seconds,
                           dm3_reduce_seconds=dm3_reduce_seconds,
                           dm3_check_seconds=dm3_check_seconds)
        return (energy_error, eigenvalue, coefficient, pmat, pit,
                contracted_d3, diagnostics)
    finally:
        shutil.rmtree(rdm_scratch, ignore_errors=True)


def build_hs(
    request, root=None, sum_bond=None, ci_options=None, ci_result=None, retain_state=None,
    integral_frame=None,
):
    nstr, nao, nae, nel, tnbond, ntstr, sso, hho, packed_ggo = request
    started = time.perf_counter()
    if integral_frame is None:
        raise ValueError("C++ TNVB requires its orthogonal integral frame")
    rmat, h_orth, g_orth = accept_orthogonal_integrals(sso, hho, packed_ggo, integral_frame)

    symmetry = "SU2"
    if retain_state is not None and root is not None:
        raise ValueError("TNVB can retain forward states only for ENERGY")
    if root is not None and (sum_bond is None or sum_bond < tnbond):
        raise RuntimeError("TNVB RDM sum bond must be at least TNBOND")

    # Pair construction persists every intermediate MPS.  Putting those files
    # beside the XMVB input can make a shared home filesystem dominate the
    # runtime.  Prefer an explicitly selected parent, then Slurm's node-local
    # temporary directory, and finally Python's platform temporary directory
    # (normally /tmp).  Each worker still gets a private mkdtemp directory.
    scratch_parent = os.environ.get("XMVB_BLOCK2_SCRATCH_PARENT")
    if not scratch_parent:
        scratch_parent = os.environ.get("SLURM_TMPDIR")
    if not scratch_parent:
        scratch_parent = tempfile.gettempdir()
    if not os.path.isdir(scratch_parent):
        raise RuntimeError(
            f"Block2 scratch parent does not exist: {scratch_parent}; "
            "set XMVB_BLOCK2_SCRATCH_PARENT to an existing local directory"
        )
    warm_manifest_data = None
    warm_read_directory = None
    warm_write_directory = None
    warm_root = None
    if root is not None:
        warm_manifest_data = warm_cache_manifest(
            nstr, nao, nae, nel, tnbond, sum_bond, ntstr, rmat
        )
        (
            warm_read_directory,
            warm_write_directory,
            warm_root,
        ) = prepare_warm_cache(warm_manifest_data)
    scratch = tempfile.mkdtemp(prefix="tnvb-block2-scratch-", dir=scratch_parent)
    atexit.register(shutil.rmtree, scratch, ignore_errors=True)
    # Each phase consumes the scheduler allocation through independent
    # processes first, then assigns otherwise idle cores to Block2 threads.
    # Parallel placement is execution policy, not part of the XMVB input.
    allocated_cpus, allocation_source = allocated_cpu_count()
    hs_task_count = nstr * (nstr + 1) // 2
    (
        state_workers,
        state_threads,
        _,
        state_worker_source,
        state_thread_source,
    ) = parallel_phase_layout(nstr, nao, tnbond, "state")
    (
        hs_workers,
        hs_threads,
        _,
        hs_worker_source,
        hs_thread_source,
    ) = parallel_phase_layout(hs_task_count, nao, tnbond, "hs")
    profile_limit = int(os.environ.get("XMVB_BLOCK2_HS_PROFILE_LIMIT", "0"))
    if profile_limit > 0:
        hs_workers = 1
        hs_worker_source += "+profile-serial"
    parallel_states_setting = os.environ.get(
        "XMVB_BLOCK2_PARALLEL_STATES", "1"
    ).strip().lower()
    if parallel_states_setting not in ("0", "1", "false", "true", "no", "yes"):
        raise RuntimeError(
            "XMVB_BLOCK2_PARALLEL_STATES must be 0/1, false/true, or no/yes"
        )
    parallel_states = (
        state_workers > 1
        and nstr > 1
        and parallel_states_setting in ("1", "true", "yes")
    )
    if not parallel_states:
        state_workers = 1

    d3g_name = "ENERGY" if root is None else "FULL"
    backend_name = "NONE" if root is None else "FULL"
    rdm_threads_preview = 0
    rdm_stack_preview_gib = 0.0
    if root is not None:
        rdm_threads_preview, _ = rdm_thread_count(nao, sum_bond)
        rdm_stack_preview_gib = (
            rdm_stack_memory_bytes(nao, sum_bond) / 1024**3
        )
    log_summary(
        f"[TNVB] repr=MPS/SU2 active={nae}e/{nao}o nstr={nstr} "
        f"D={tnbond}/{sum_bond if root is not None else '-'} "
        f"check={'GRADIENT' if root is not None else 'ENERGY'} "
        f"sum={'DIRECT' if root is not None else '-'} "
        f"d3g={d3g_name} backend={backend_name} "
        "integrals=ORTHOGONAL"
    )
    log_summary(
        f"[TNVB-PARALLEL] state={state_workers}x{state_threads} "
        f"hs={hs_workers}x{hs_threads} rdm=1x{rdm_threads_preview} "
        f"stack={rdm_stack_preview_gib:.3f}GiB"
    )

    print(
        f"[TNVB-BLOCK2] configuration symmetry={symmetry} "
        f"TNBOND={tnbond} "
        f"scratch_parent={scratch_parent} allocated_cpus={allocated_cpus} "
        f"allocation_source={allocation_source} "
        f"state_workers={state_workers} state_threads={state_threads} "
        f"state_worker_source={state_worker_source} "
        f"state_thread_source={state_thread_source} "
        f"hs_workers={hs_workers} hs_threads={hs_threads} "
        f"hs_worker_source={hs_worker_source} "
        f"hs_thread_source={hs_thread_source}",
        flush=True,
    )

    driver = None
    identity_mpo = None
    states = None
    state_tags = None
    mps_started = time.perf_counter()
    if parallel_states:
        print(
            f"[TNVB-BLOCK2-PARALLEL] MPS start structures={nstr} "
            f"workers={state_workers} block2_threads={state_threads}",
            flush=True,
        )
        state_tags = build_parallel_su2_states(
            scratch_parent,
            scratch,
            nao,
            nae,
            nel,
            tnbond,
            ntstr,
            rmat,
            state_workers,
            state_threads,
            warm_read_directory=warm_read_directory,
            warm_write_directory=warm_write_directory,
        )
        print(
            f"[TNVB-BLOCK2-PARALLEL] MPS joined structures={len(state_tags)} "
            f"workers={state_workers}",
            flush=True,
        )
    else:
        driver = DMRGDriver(
            scratch=scratch,
            symm_type=SymmetryTypes.SU2,
            n_threads=state_threads,
            stack_mem=stack_memory_bytes(nao, tnbond),
            clean_scratch=True,
        )
        driver.initialize_system(n_sites=nao, n_elec=nae, spin=0, orb_sym=[0] * nao)
        identity_mpo = driver.get_identity_mpo()
        states = build_su2_states(
            driver,
            identity_mpo,
            ntstr,
            nstr,
            nao,
            nae,
            nel,
            tnbond,
            rmat,
            state_scratch=scratch,
            warm_read_directory=warm_read_directory,
            warm_write_directory=warm_write_directory,
        )
    mps_seconds = time.perf_counter() - mps_started

    if ci_options is not None:
        # This branch never constructs pairwise H/S environments or solves a
        # dense structure CI problem. The native common operator owns a child
        # frame and is closed before the RDM driver is initialized below.
        import common_ci
        if ci_result is None:
            raise ValueError("Davidson requires an eigenpair result sink")
        if state_tags is None:
            state_tags = persist_parallel_states(states, scratch)
        del states, identity_mpo, driver
        hs_started = time.perf_counter()
        hmat, smat, eigenvalue, coefficient, diagnostics = common_ci.contract(
            scratch, scratch_parent, state_tags, nao, nae, tnbond,
            min(16, allocated_cpus), h_orth, g_orth, ci_options,
            private_source=True)
        ci_result.update(energy=eigenvalue, coefficient=coefficient,
                         diagnostics=diagnostics)
        diagonal = np.diag(smat)
        log_summary(
            f"[TNVB-MPS] structures={nstr} Dmax={tnbond} build={mps_seconds:.3f}s "
            f"norm=[{np.min(diagonal):.6e},{np.max(diagonal):.6e}] "
            f"warm={'hit' if warm_read_directory is not None else 'miss'}")
        log_summary(f"[TNVB-HS] backend=COMMON_DAVIDSON "
                    f"wall={time.perf_counter()-hs_started:.3f}s matrices=exported")
        rdm_result = None
        if root is not None:
            rdm_result = compute_total_state_rdms(
                scratch, scratch_parent, state_tags, nao, nae, tnbond, sum_bond,
                root, hmat, smat, h_orth, g_orth,
                warm_write_directory=warm_write_directory,
                selected_eigenpair=(eigenvalue, coefficient))
        if warm_root is not None:
            commit_warm_cache(warm_root, warm_write_directory, warm_manifest_data)
        if retain_state is not None:
            from forward_state import export_context
            export_context(retain_state, scratch, state_tags, h_orth, g_orth)
        shutil.rmtree(scratch, ignore_errors=True)
        return (hmat, smat) if rdm_result is None else (hmat, smat) + rdm_result

    hs_started = time.perf_counter()
    hmat = np.zeros((nstr, nstr), dtype=np.float64, order="F")
    smat = np.zeros((nstr, nstr), dtype=np.float64, order="F")
    pairs = []
    for j in range(nstr):
        for i in range(j, nstr):
            pairs.append((len(pairs), i, j))

    if hs_workers > 1:
        print(
            f"[TNVB-BLOCK2-PARALLEL] H/S start tasks={len(pairs)} "
            f"workers={hs_workers} block2_threads={hs_threads}",
            flush=True,
        )
        # ENERGY uses the verified immutable shared bank. FULL gradients keep
        # their existing private H/S/RDM objects and warm-cache lifecycle.
        contract_hs = hs_shared.contract if hs_shared.use_shared_hs(root, hs_workers) else contract_parallel_hs
        results = contract_hs(
            states,
            scratch,
            scratch_parent,
            symmetry,
            nao,
            nae,
            tnbond,
            h_orth,
            g_orth,
            pairs,
            hs_workers,
            hs_threads,
            state_tags=state_tags,
        )
        print(
            f"[TNVB-BLOCK2-PARALLEL] H/S joined tasks={len(results)} "
            f"workers={hs_workers}",
            flush=True,
        )
    else:
        if driver is None:
            # Parallel construction produced on-disk states only.  A serial
            # H/S phase (including profiling) still needs its own live driver.
            configure_phase_environment(hs_threads)
            driver = DMRGDriver(
                scratch=scratch,
                symm_type=SymmetryTypes.SU2,
                n_threads=hs_threads,
                stack_mem=stack_memory_bytes(nao, tnbond),
                clean_scratch=True,
            )
            driver.initialize_system(
                n_sites=nao, n_elec=nae, spin=0, orb_sym=[0] * nao
            )
            identity_mpo = driver.get_identity_mpo()
            states = [driver.load_mps(tag) for tag in state_tags]
        h_mpo = get_hamiltonian_mpo(driver, h_orth, g_orth)
        results = []
        for order, i, j in pairs:
            overlap_started = time.perf_counter()
            overlap = float(driver.expectation(states[i], identity_mpo, states[j]))
            overlap_seconds = time.perf_counter() - overlap_started
            hamiltonian_started = time.perf_counter()
            hamiltonian = float(driver.expectation(states[i], h_mpo, states[j]))
            hamiltonian_seconds = time.perf_counter() - hamiltonian_started
            results.append(
                (
                    order,
                    i,
                    j,
                    overlap,
                    hamiltonian,
                    overlap_seconds,
                    hamiltonian_seconds,
                    os.getpid(),
                )
            )

    completed, total = 0, len(pairs)
    for (
        _,
        i,
        j,
        overlap,
        hamiltonian,
        overlap_seconds,
        hamiltonian_seconds,
        worker_pid,
    ) in results:
        smat[i, j] = smat[j, i] = overlap
        hmat[i, j] = hmat[j, i] = hamiltonian
        completed += 1
        print(
            f"[TNVB-BLOCK2] H/S progress={completed}/{total} "
            f"expectation={'parallel' if hs_workers > 1 else 'fused'} "
            f"worker={worker_pid} "
            f"S_seconds={overlap_seconds:.3f} "
            f"H_seconds={hamiltonian_seconds:.3f} "
            f"S={overlap:.17e} H={hamiltonian:.17e} "
            f"wall_seconds={time.perf_counter() - started:.3f}",
            flush=True,
        )
        if 0 < profile_limit <= completed < total:
            raise RuntimeError(
                "intentional H/S profiling stop after "
                f"{completed} of {total} matrix elements"
            )

    hs_seconds = time.perf_counter() - hs_started
    s_min, s_condition, effective_rank = validate_structure_metric(hmat, smat)
    slowest = max(results, key=lambda item: item[5] + item[6])
    warm_status = "hit" if warm_read_directory is not None else "miss"
    diagonal = np.diag(smat)
    invalid_norms = np.where((~np.isfinite(diagonal)) | (diagonal <= 0.0))[0]
    if invalid_norms.size:
        details = ",".join(
            f"{int(index) + 1}:{diagonal[index]:.6e}"
            for index in invalid_norms[:8]
        )
        raise RuntimeError(
            "TNVB structure MPS has invalid norms at structures " + details
        )
    log_summary(
        f"[TNVB-MPS] structures={nstr} Dmax={tnbond} build={mps_seconds:.3f}s "
        f"norm=[{float(np.min(diagonal)):.6e},{float(np.max(diagonal)):.6e}] "
        f"warm={warm_status}"
    )
    log_summary(
        f"[TNVB-HS] tasks={total} workers={hs_workers} wall={hs_seconds:.3f}s "
        f"slowest=({slowest[1] + 1},{slowest[2] + 1},"
        f"{slowest[5] + slowest[6]:.3f}s) H_asym="
        f"{float(np.max(np.abs(hmat - hmat.T))):.3e} S_asym="
        f"{float(np.max(np.abs(smat - smat.T))):.3e} S_min={s_min:.6e} "
        f"S_cond={s_condition:.6e} S_rank={effective_rank}/{nstr} metric=normalized"
    )

    rdm_result = None
    if root is not None:
        if state_tags is None:
            state_tags = persist_parallel_states(states, scratch)
        # Release the construction/contraction objects before Block2 is
        # reinitialized with all scheduler-allocated cores for the single RDM job.
        del states, identity_mpo, driver
        if hs_workers == 1:
            del h_mpo
        rdm_result = compute_total_state_rdms(
            scratch,
            scratch_parent,
            state_tags,
            nao,
            nae,
            tnbond,
            sum_bond,
            root,
            hmat,
            smat,
            h_orth,
            g_orth,
            warm_write_directory=warm_write_directory,
        )
    else:
        # Keep objects alive until all contractions finish, then remove scratch.
        if retain_state is not None and state_tags is None:
            state_tags = persist_parallel_states(states, scratch)
        del states, identity_mpo, driver
        if hs_workers == 1:
            del h_mpo
    if warm_root is not None:
        commit_warm_cache(
            warm_root, warm_write_directory, warm_manifest_data
        )
    if retain_state is not None:
        from forward_state import export_context
        export_context(retain_state, scratch, state_tags, h_orth, g_orth)
    shutil.rmtree(scratch, ignore_errors=True)
    if rdm_result is None:
        return hmat, smat
    return (hmat, smat) + rdm_result
