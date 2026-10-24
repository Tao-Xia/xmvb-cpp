#!/usr/bin/env python3
"""Owned paired-state numerical core for the independent C++ TNVB worker.

The production entrypoint and only wire protocol live in evaluate.py. The
stable SU(2) pair/total-state numerical routines are reused here; this module
does not provide a legacy XMVB executable or protocol compatibility path.
"""

from __future__ import annotations

import atexit
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import hashlib
import json
import math
import multiprocessing
import os
import shutil
import struct
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

from block2 import SZ
from pyblock2.algebra.core import MPS as PyMPS, SubTensor, Tensor
from pyblock2.algebra.io import MPOTools, MPSTools, init_block2_types
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
_PARALLEL_D3G_CONTEXT = None
_PARALLEL_STATE_D3G_CONTEXT = None
_PARALLEL_NATIVE_D3G_CONTEXT = None

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
    direct_trace_error, direct_overlap_error, rdm_tolerance,
):
    """Warn above 1e-8; reject nonfinite data or residuals above the fatal limit."""
    errors = (
        trace_error, contraction2_error, contraction3_error, rdm_energy_error,
        direct_trace_error, direct_overlap_error,
    )
    detail = (
        f"trace1={trace_error:.3e}, "
        f"contract2={contraction2_error:.3e}, "
        f"contract3={contraction3_error:.3e}, "
        f"energy={rdm_energy_error:.3e}, "
        f"direct_trace={direct_trace_error:.3e}, "
        f"direct_overlap={direct_overlap_error:.3e}"
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


def orthogonalize_integrals(sso, hho, packed_ggo):
    # S = R.T R and b_i^dagger = sum_p R[p,i] a_p^dagger.
    rmat = scipy.linalg.cholesky(sso, lower=False, check_finite=True)
    rinv = scipy.linalg.solve_triangular(
        rmat, np.eye(rmat.shape[0]), lower=False, check_finite=True
    )
    h_orth = rinv.T @ hho @ rinv
    g_nonorth = unpack_eri(packed_ggo, rmat.shape[0])
    g_orth = np.einsum(
        "ip,jq,kr,ls,ijkl->pqrs",
        rinv,
        rinv,
        rinv,
        rinv,
        g_nonorth,
        optimize=True,
    )
    return rmat, h_orth, g_orth


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


def structure_determinants(structure, nao: int, nae: int, nel: int):
    """Expand a carrier into Block2 SZ determinants for regression only.

    This exact exponential implementation is retained as an independent
    small-system reference for the direct block-sparse paired MPS importer.
    """
    if nae % 2:
        raise RuntimeError("Block2 TNVB prototype currently requires even NAE")
    ndb = (nel - nae) // 2
    if 2 * ndb + nae != nel:
        raise RuntimeError("NEL-NAE must be even")
    active = np.asarray(structure[2 * ndb : 2 * ndb + nae], dtype=int) - ndb - 1
    terms = [(list("0" * nao), 1.0)]
    for pair in range(nae // 2):
        original_left = int(active[2 * pair])
        original_right = int(active[2 * pair + 1])
        if not (0 <= original_left < nao and 0 <= original_right < nao):
            raise RuntimeError("active pair orbital is outside 1..NAO")
        if original_left == original_right:
            updated = []
            for occupancy, coefficient in terms:
                if occupancy[original_left] != "0":
                    raise RuntimeError("active orbital belongs to multiple pairs")
                occupancy = occupancy.copy()
                occupancy[original_left] = "2"
                updated.append((occupancy, coefficient))
            terms = updated
            continue

        phase = -1.0 if original_left > original_right else 1.0
        left, right = sorted((original_left, original_right))
        updated = []
        for occupancy, coefficient in terms:
            if occupancy[left] != "0" or occupancy[right] != "0":
                raise RuntimeError("active orbital belongs to multiple pairs")
            up_down = occupancy.copy()
            up_down[left], up_down[right] = "a", "b"
            updated.append((up_down, coefficient * phase))
            down_up = occupancy.copy()
            down_up[left], down_up[right] = "b", "a"
            updated.append((down_up, -coefficient * phase))
        terms = updated

    combined = {}
    for occupancy, coefficient in terms:
        key = "".join(occupancy)
        combined[key] = combined.get(key, 0.0) + coefficient
    combined = {key: value for key, value in combined.items() if value != 0.0}
    return list(combined), np.asarray(list(combined.values()), dtype=np.float64)


def paired_structure_layout(structure, nao: int, nae: int, nel: int):
    """Return the site-local paired-state automaton for one VB structure."""
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


def build_direct_paired_mps(driver, structure, nao: int, nae: int, nel: int, tag: str):
    """Build an exact paired carrier without enumerating its 2**P determinants.

    A virtual state records only the spin choices of pairs crossing the current
    orbital cut.  States with the same particle/spin quantum number share one
    Block2 SZ sector and use its reduced dimension for the remaining choices.
    """
    site_kind, pairs, structure_phase = paired_structure_layout(
        structure, nao, nae, nel
    )
    q_empty = SZ(0, 0, 0)
    q_alpha = SZ(1, 1, 0)
    q_beta = SZ(1, -1, 0)
    q_double = SZ(2, 0, 0)
    target = SZ(nae, 0, 0)

    # Each state key is a sorted tuple of (open_pair_id, left_endpoint_spin),
    # where spin 0/1 means alpha/beta at the already visited left endpoint.
    bond_states = [{(): q_empty}]
    site_transitions = []
    max_crossing = 0
    for site, (kind, pair_id) in enumerate(site_kind):
        next_states = {}
        transitions = []

        def add_transition(left_key, right_key, q_local, coefficient):
            q_right = bond_states[-1][left_key] + q_local
            if right_key in next_states and next_states[right_key] != q_right:
                raise RuntimeError("inconsistent quantum number in paired MPS automaton")
            next_states[right_key] = q_right
            transitions.append((left_key, right_key, q_local, coefficient))

        for left_key in bond_states[-1]:
            choices = dict(left_key)
            if kind == "empty":
                add_transition(left_key, left_key, q_empty, 1.0)
            elif kind == "double":
                add_transition(left_key, left_key, q_double, 1.0)
            elif kind == "pair_left":
                if pair_id in choices:
                    raise RuntimeError("paired MPS opened the same pair twice")
                alpha_choices = dict(choices)
                alpha_choices[pair_id] = 0
                beta_choices = dict(choices)
                beta_choices[pair_id] = 1
                add_transition(
                    left_key, tuple(sorted(alpha_choices.items())), q_alpha, 1.0
                )
                add_transition(
                    left_key, tuple(sorted(beta_choices.items())), q_beta, 1.0
                )
            elif kind == "pair_right":
                if pair_id not in choices:
                    raise RuntimeError("paired MPS closed a pair that was not open")
                left_spin = choices.pop(pair_id)
                right_key = tuple(sorted(choices.items()))
                if left_spin == 0:
                    add_transition(left_key, right_key, q_beta, 1.0)
                else:
                    add_transition(left_key, right_key, q_alpha, -1.0)
            else:
                raise RuntimeError(f"unknown paired MPS site kind {kind}")
        bond_states.append(next_states)
        site_transitions.append(transitions)
        max_crossing = max(max_crossing, len(next(iter(next_states), ())))

    if bond_states[-1] != {(): target}:
        raise RuntimeError("paired MPS did not close all pairs at the target quantum number")

    # Map automaton states to (quantum sector, reduced index) on every bond.
    sector_maps = []
    sector_dims = []
    for states in bond_states:
        sectors = {}
        state_map = {}
        for state_key, quantum in states.items():
            index = len(sectors.setdefault(quantum, []))
            sectors[quantum].append(state_key)
            state_map[state_key] = (quantum, index)
        sector_maps.append(state_map)
        sector_dims.append({quantum: len(keys) for quantum, keys in sectors.items()})

    tensors = []
    for site, transitions in enumerate(site_transitions):
        blocks = {}
        for left_key, right_key, q_local, coefficient in transitions:
            q_left, i_left = sector_maps[site][left_key]
            q_right, i_right = sector_maps[site + 1][right_key]
            coefficient *= structure_phase if site == 0 else 1.0
            if site == 0:
                labels = (q_local, q_right)
                shape = (1, sector_dims[site + 1][q_right])
                position = (0, i_right)
            elif site == nao - 1:
                labels = (q_left, q_local)
                shape = (sector_dims[site][q_left], 1)
                position = (i_left, 0)
            else:
                labels = (q_left, q_local, q_right)
                shape = (
                    sector_dims[site][q_left],
                    1,
                    sector_dims[site + 1][q_right],
                )
                position = (i_left, 0, i_right)
            if labels not in blocks:
                blocks[labels] = SubTensor(
                    q_labels=labels, reduced=np.zeros(shape, dtype=np.float64)
                )
            blocks[labels].reduced[position] += coefficient
        tensors.append(Tensor(blocks=list(blocks.values())))

    basis_template = Counter(
        {q_empty: 1, q_alpha: 1, q_beta: 1, q_double: 1}
    )
    python_mps = PyMPS(tensors=tensors)
    carrier_norm2 = float(python_mps | python_mps)
    carrier = MPSTools.to_block2(
        python_mps,
        [basis_template.copy() for _ in range(nao)],
        center=0,
        tag=tag,
        left_vacuum=q_empty,
    )
    carrier = driver.adjust_mps(carrier, dot=2)[0]
    raw_bond_dimension = max(sum(dims.values()) for dims in sector_dims)
    return carrier, len(pairs), max_crossing, raw_bond_dimension, carrier_norm2


def local_basis_template():
    """Return the four SZ sectors of one spatial-orbital site."""
    return Counter(
        {
            SZ(0, 0, 0): 1,
            SZ(1, 1, 0): 1,
            SZ(1, -1, 0): 1,
            SZ(2, 0, 0): 1,
        }
    )


def vacuum_python_mps(nao: int):
    """Construct the block-sparse Fock vacuum in pyblock2 format."""
    q_empty = SZ(0, 0, 0)
    tensors = []
    for site in range(nao):
        if site == 0 or site == nao - 1:
            labels = (q_empty, q_empty)
            shape = (1, 1)
        else:
            labels = (q_empty, q_empty, q_empty)
            shape = (1, 1, 1)
        tensors.append(
            Tensor(
                blocks=[
                    SubTensor(q_labels=labels, reduced=np.ones(shape, dtype=np.float64))
                ]
            )
        )
    return PyMPS(tensors=tensors)


def build_creation_mpos(driver, rmat: np.ndarray):
    """Build alpha/beta creators for every nonorthogonal orbital column.

    Each returned MPO represents sum_p R[p,i] a_p^dagger.  The MPOs carry
    particle-number/spin quantum-number flux and can therefore create the
    physical VB state directly from the vacuum, without approximating
    Gamma(R) by time evolution.
    """
    creators = {"alpha": [], "beta": []}
    for spin, operator in (("alpha", "c"), ("beta", "C")):
        for orbital in range(rmat.shape[1]):
            builder = driver.expr_builder()
            builder.add_sum_term(
                operator,
                np.asarray(rmat[:, orbital], dtype=np.float64),
                cutoff=1.0e-16,
            )
            block_mpo = driver.get_mpo(
                builder.finalize(), cutoff=1.0e-16, iprint=0, add_ident=False
            )
            # get_mpo returns a SimplifiedMPO.  The unsimplified primitive is
            # deliberately requested because MPOTools translates that exact
            # operator, including its fermionic Jordan-Wigner signs.
            creators[spin].append(MPOTools.from_block2(block_mpo.prim_mpo))
    return creators


def build_pair_creation_mpo(creators, left: int, right: int):
    """Return the singlet-pair creator used by XMVB/ITensor."""
    alpha, beta = creators["alpha"], creators["beta"]
    if left == right:
        return alpha[left] @ beta[right]
    return alpha[left] @ beta[right] - beta[left] @ alpha[right]


def build_direct_physical_mps(
    driver,
    structure,
    nao: int,
    nae: int,
    nel: int,
    tnbond: int,
    creators,
    pair_cache,
    tag: str,
):
    """Build a physical VB structure directly from transformed pair creators.

    This mirrors the ITensor reference construction: start from the Fock
    vacuum, apply one Cholesky-transformed pair creator at a time, and compress
    to TNBOND after each pair.  It replaces 20 RK4 truncations by nae/2 direct
    pair applications and has no time-step error.
    """
    if nae % 2:
        raise RuntimeError("Block2 direct pair construction currently requires even NAE")
    ndb = (nel - nae) // 2
    if 2 * ndb + nae != nel:
        raise RuntimeError("NEL-NAE must be even")
    active = np.asarray(structure[2 * ndb : 2 * ndb + nae], dtype=int) - ndb - 1
    physical = vacuum_python_mps(nao)
    structure_phase = 1.0
    max_discarded_norm = 0.0
    cutoff = float(os.environ.get("XMVB_BLOCK2_PAIR_CUTOFF", "1e-13"))
    sweep = os.environ.get("XMVB_BLOCK2_PAIR_SWEEP", "right-to-left").lower()
    if sweep not in ("right-to-left", "left-to-right"):
        raise RuntimeError(
            "XMVB_BLOCK2_PAIR_SWEEP must be right-to-left or left-to-right"
        )
    compress_left = sweep == "right-to-left"

    for pair in range(nae // 2):
        original_left = int(active[2 * pair])
        original_right = int(active[2 * pair + 1])
        if not (
            0 <= original_left < nao
            and 0 <= original_right < nao
        ):
            raise RuntimeError("active pair orbital is outside 1..NAO")
        if original_left > original_right:
            structure_phase = -structure_phase
        left, right = sorted((original_left, original_right))
        key = (left, right)
        if key not in pair_cache:
            pair_cache[key] = build_pair_creation_mpo(creators, left, right)
        physical = pair_cache[key] @ physical
        discarded_norm = physical.compress(
            k=max(2, tnbond), cutoff=cutoff, left=compress_left
        )
        max_discarded_norm = max(max_discarded_norm, float(discarded_norm))

    if structure_phase < 0.0:
        physical.tensors[0] = -physical.tensors[0]
    physical_norm2 = float(physical | physical)
    basis = [local_basis_template() for _ in range(nao)]
    block_mps = MPSTools.to_block2(
        physical,
        basis,
        center=0,
        tag=tag,
        left_vacuum=SZ(0, 0, 0),
    )
    return block_mps, physical_norm2, max_discarded_norm


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


def state_d3g_stack_memory_bytes(nao: int, hole_bond: int) -> int:
    """Size one STATE matrix-element worker without changing FULL defaults.

    A two-electron transition environment needs appreciably more operator
    stack than H/S at the same MPS bond.  In particular, the generic 512 MiB
    floor exhausts Block2's double stack for C20 with a D=600 hole state.
    Keep this allowance local to STATE so increasing it cannot inflate the
    single-process FULL NPDM stack or the ordinary H/S worker pool.
    """
    override = os.environ.get("XMVB_BLOCK2_STATE_D3G_STACK_GB", "").strip()
    if override:
        value = float(override)
        if value <= 0.0:
            raise RuntimeError(
                "XMVB_BLOCK2_STATE_D3G_STACK_GB must be positive"
            )
        return int(value * 1024**3)
    estimate = 128 * nao * max(2, hole_bond) ** 2
    return max(stack_memory_bytes(nao, hole_bond), estimate)


def native_d3g_stack_memory_bytes(nao: int, total_bond: int) -> int:
    """Size one selected-output complementary-MPO worker stack.

    C14/D=384 reaches about 290 MiB in Block2's double arena for an eight-
    output batch, so the generic 512 MiB H/S stack (shared between arenas and
    bookkeeping) is insufficient.  This allowance remains process-local and
    cannot inflate FULL 3-RDM or ordinary H/S calculations.
    """
    override = os.environ.get("XMVB_BLOCK2_NATIVE_STACK_GB", "").strip()
    if override:
        value = float(override)
        if value <= 0.0:
            raise RuntimeError("XMVB_BLOCK2_NATIVE_STACK_GB must be positive")
        return int(value * 1024**3)
    # For the validated eight-output topology the C14/D=384 and C20/D=448
    # boundaries require about 1 and 4 GiB, respectively.  The observed
    # growth follows the complementary-channel count (roughly K^3) times MPS
    # block storage (D^2); retain a one-GiB floor for smaller active spaces.
    measured_scale = (
        (max(1, nao) / 14.0) ** 3
        * (max(2, total_bond) / 384.0) ** 2
    )
    estimate = int(math.ceil(measured_scale * 1024**3))
    return max(1024**3, estimate, 2 * stack_memory_bytes(nao, total_bond))


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


def contracted_d3g_mode(mode_code: int):
    """Select the contracted 3-RDM implementation.

    ``full`` is the established K**6 reference.  ``contracted`` evaluates
    SU(2) scalar ``E[c,d] V`` MPOs and reconstructs the complementary triangle
    from the 2-RDM commutator, without allocating a rank-three NPDM.
    ``validate`` computes both and refuses to continue unless every
    contracted element agrees.
    """
    modes = {0: "full", 1: "contracted", 2: "validate"}
    if mode_code not in modes:
        raise RuntimeError(
            "contracted-D3G mode must be FULL, CONTRACTED, or VALIDATE"
        )
    return modes[mode_code]


def contracted_d3g_backend(mode_code: int):
    """Select a development implementation for contracted-D3G validation.

    ``explicit`` is the established lower-triangle scalar-MPO reference;
    ``state`` is the SU(2) hole-state oracle.  ``native`` is the production
    SU(2) complementary-operator implementation.
    """
    backends = {
        0: "explicit",
        4: "native",
        5: "state",
    }
    if mode_code not in backends:
        raise RuntimeError(
            "contracted-D3G backend must be EXPLICIT, NATIVE, or STATE"
        )
    return backends[mode_code]


def get_hamiltonian_mpo(driver, h_orth, g_orth, log=True, mpo_mode=None):
    """Build the QC Hamiltonian with a selectable Block2 MPO algorithm.

    ``mpo_mode`` is an internal override used when an operator product needs a
    particular symbolic representation.  Ordinary energy/Hamiltonian callers
    retain the user-selected ``XMVB_BLOCK2_H_MPO`` mode.
    """
    if mpo_mode is None:
        mpo_mode = os.environ.get(
            "XMVB_BLOCK2_H_MPO", "no-trans-conventional"
        )
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
                             warm_read_directory=None, warm_write_directory=None):
    """Deterministic exact block sum followed by bounded SU(2) compression.

    Physical-state caches are unaffected. Partial sums from the old variational
    constructor are neither read nor used as initial guesses.
    """
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


def build_weighted_total_state(
    driver,
    states,
    coefficients,
    nelec: int,
    bond_dimension: int,
    state_scratch=None,
    warm_read_directory=None,
    tag_prefix="TNVB-SUM-A",
    root=0,
    attempt=0,
    cache_metadata=None,
    return_chain=False,
):
    """Build one candidate; never publish it to the cross-call cache here.

    Partial sums keep their fitted amplitudes. Only the final sum is normalized.
    Distinct attempt tags keep the first candidate valid during a cold retry.
    """
    if len(states) != len(coefficients) or not states:
        raise RuntimeError("TNVB total-state construction received inconsistent inputs")
    addition_sweeps = int(os.environ.get("XMVB_BLOCK2_SUM_SWEEPS", "8"))
    addition_tolerance = float(os.environ.get("XMVB_BLOCK2_SUM_TOL", "1e-11"))
    addition_cutoff = float(os.environ.get("XMVB_BLOCK2_SUM_CUTOFF", "1e-14"))
    addition_iprint = int(os.environ.get("XMVB_BLOCK2_SUM_IPRINT", "0"))
    if addition_sweeps < 1 or addition_tolerance <= 0.0 or addition_cutoff < 0.0:
        raise RuntimeError("invalid XMVB_BLOCK2_SUM_SWEEPS/TOL/CUTOFF setting")

    order = np.argsort(-np.abs(coefficients), kind="stable")
    prefixes = sum_prefixes(order)
    base_seed = int(os.environ.get("XMVB_BLOCK2_SU2_SEED", "20260813"))
    warm_stages = 0
    chain = []

    def initial_sum(position):
        nonlocal warm_stages
        tag = f"{tag_prefix}-{position}"
        previous = None
        if (cache_metadata is not None and
                cache_metadata["prefixes"].get(str(position)) ==
                prefixes[str(position)]):
            previous = load_warm_start_mps(
                driver, state_scratch, warm_read_directory,
                f"TNVB-TOTAL-{position}", bond_dimension,
            )
        if previous is not None:
            warm_stages += 1
            return driver.copy_mps(previous, tag)
        seed = (base_seed + 104729 * (root + 1) +
                1009 * (attempt + 1) + 9176 * position) % 2147483647
        driver.bw.b.Random.rand_seed(seed)
        return driver.get_random_mps(
            tag=tag, bond_dim=bond_dimension,
            target=driver.bw.SX(nelec, 0, 0), dot=2, full_fci=False,
        )

    if len(states) == 1:
        total = driver.copy_mps(states[int(order[0])], f"{tag_prefix}-1")
        chain.append(total)
    else:
        first, second = (int(order[0]), int(order[1]))
        total = initial_sum(2)
        driver.addition(
            total,
            states[first],
            states[second],
            mpo_a=float(coefficients[first]),
            mpo_b=float(coefficients[second]),
            n_sweeps=addition_sweeps,
            tol=addition_tolerance,
            bra_bond_dims=[bond_dimension],
            cutoff=addition_cutoff,
            iprint=addition_iprint,
        )
        chain.append(total)
        for position, state_index in enumerate(order[2:], start=3):
            state_index = int(state_index)
            output = initial_sum(position)
            driver.addition(
                output,
                total,
                states[state_index],
                mpo_a=1.0,
                mpo_b=float(coefficients[state_index]),
                n_sweeps=addition_sweeps,
                tol=addition_tolerance,
                bra_bond_dims=[bond_dimension],
                cutoff=addition_cutoff,
                iprint=addition_iprint,
            )
            chain.append(output)
            total = output
    normalize_block2_mps(total)
    if return_chain:
        return total, chain, warm_stages
    return total


def sum_prefixes(order):
    """O(nstr) cache identities for ordered structure-index prefixes."""
    digest = hashlib.sha256()
    result = {}
    for position, index in enumerate(order, start=1):
        digest.update(struct.pack("<i", int(index) + 1))
        result[str(position)] = digest.hexdigest()
    return result


def sum_cache_context(nstr, nsites, nelec, bond_dimension, root):
    # Orbitals and coefficients may change: the cache is an initial guess,
    # never an already evaluated sum. Root, representation and solver changes
    # invalidate total guesses independently of the physical-state cache.
    return dict(
        version=3, constructor="VARIATIONAL_FIT", nstr=int(nstr), nsites=int(nsites),
        nelec=int(nelec), bond=int(bond_dimension), root=int(root),
        sweeps=int(os.environ.get("XMVB_BLOCK2_SUM_SWEEPS", "8")),
        tolerance=float(os.environ.get("XMVB_BLOCK2_SUM_TOL", "1e-11")),
        cutoff=float(os.environ.get("XMVB_BLOCK2_SUM_CUTOFF", "1e-14")),
        seed=int(os.environ.get("XMVB_BLOCK2_SU2_SEED", "20260813")),
    )


def read_sum_cache(read_directory, context):
    """Old/unverified total sums are ignored without invalidating VB states."""
    if not read_directory:
        return None
    try:
        with open(os.path.join(read_directory, "sum_manifest.json"),
                  encoding="utf-8") as stream:
            metadata = json.load(stream)
        if (metadata.get("context") != context or
                metadata.get("valid") is not True or
                not isinstance(metadata.get("prefixes"), dict) or
                not sum_quality_passes(metadata["quality"])):
            return None
        return metadata
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None


def select_sum_candidate(qualities):
    """Choose by target fidelity, not by lowering the Hamiltonian energy."""
    best = 0
    for index in range(1, len(qualities)):
        a, b = qualities[best], qualities[index]
        if sum_quality_passes(a) != sum_quality_passes(b):
            if sum_quality_passes(b):
                best = index
            continue
        slack = max(a["roundoff"], b["roundoff"], 64 * np.finfo(float).eps)
        if (b["infidelity"] < a["infidelity"] - slack or
                (abs(b["infidelity"] - a["infidelity"]) <= slack and
                 b["energy_error"] < a["energy_error"])):
            best = index
    return best


def publish_sum_cache(driver, chain, order, context, quality,
                      state_scratch, write_directory):
    """Stage only the selected, verified chain; outer commit follows RDM success."""
    if not write_directory:
        return
    valid = sum_quality_passes(quality)
    metadata = dict(context=context, valid=valid, quality=quality,
                    order=[int(x) + 1 for x in order],
                    prefixes=sum_prefixes(order) if valid else {})
    if valid:
        first_position = 1 if len(order) == 1 else 2
        for position, partial in enumerate(chain, start=first_position):
            # Embedded Block2 tags must change along with filenames; raw file
            # renaming would not be sufficient. Candidate tags remain intact.
            cached = driver.copy_mps(partial, f"TNVB-TOTAL-{position}")
            persist_warm_start_mps(cached, state_scratch, write_directory)
    path = os.path.join(write_directory, "sum_manifest.json")
    with open(path + ".tmp", "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(path + ".tmp", path)


def fit_variational_total_state(driver, states, coefficient, smat, nelec,
                             bond_dimension, root, identity_mpo, h_mpo,
                             eigenvalue, state_scratch,
                             warm_read_directory=None, warm_write_directory=None):
    """At most two complete fits, then exactly one selected state for RDMs."""
    context = sum_cache_context(len(states), driver.n_sites, nelec, bond_dimension, root)
    cache = read_sum_cache(warm_read_directory, context)
    candidates, qualities, timings, warms = [], [], [], []
    for attempt in range(2):
        started = time.perf_counter()
        total, chain, warm_stages = build_weighted_total_state(
            driver, states, coefficient, nelec, bond_dimension,
            state_scratch=state_scratch,
            warm_read_directory=warm_read_directory if attempt == 0 else None,
            tag_prefix=f"TNVB-SUM-{'AB'[attempt]}", root=root, attempt=attempt,
            cache_metadata=cache if attempt == 0 else None, return_chain=True,
        )
        quality = total_state_quality(
            driver, total, states, coefficient, smat, identity_mpo, h_mpo, eigenvalue,
        )
        candidates.append((total, chain))
        qualities.append(quality)
        timings.append(time.perf_counter() - started)
        warms.append(warm_stages)
        if sum_quality_passes(quality):
            break
        if attempt == 0:
            log_summary(
                f"[TNVB-SUM-RETRY] eta={quality['infidelity']:.3e} "
                f"fit_dE={quality['energy_error']:.3e} warm={warm_stages} "
                "action=cold_rebuild rdm=not_started"
            )
    best = select_sum_candidate(qualities)
    total, chain = candidates[best]
    quality = qualities[best]
    order = np.argsort(-np.abs(coefficient), kind="stable")
    publish_sum_cache(driver, chain, order, context, quality,
                      state_scratch, warm_write_directory)
    valid = sum_quality_passes(quality)
    log_summary(
        f"[TNVB-SUM] method=FIT actual_bond={max(reduced_bond_profile(total))} cap={bond_dimension} attempts={len(candidates)} selected={'AB'[best]} "
        f"warm={warms[best]}/{max(0, len(states)-1)} "
        f"eta={quality['infidelity']:.3e} fit_dE={quality['energy_error']:.3e} "
        f"cache={'verified' if valid else 'invalid'} "
        f"fit_times={','.join(f'{x:.3f}' for x in timings)}s"
    )
    if quality["infidelity"] + quality["roundoff"] > SUM_INFIDELITY_TOL:
        log_summary(
            f"[TNVB-WARNING] sum_eta={quality['infidelity']:.6e} "
            f"roundoff={quality['roundoff']:.3e} reference={SUM_INFIDELITY_TOL:.1e}; "
            "bounded total-state repair did not reach target fidelity; "
            "continuing RDM/gradient evaluation without a reusable sum cache. "
            "Gradients and optimized results may be inaccurate."
        )
    return total, quality, dict(attempts=len(candidates), selected=best,
                              qualities=qualities, seconds=timings, warm_stages=warms)


def fit_weighted_total_state(driver, states, coefficient, smat, nelec,
                             bond_dimension, root, identity_mpo, h_mpo,
                             eigenvalue, state_scratch,
                             warm_read_directory=None, warm_write_directory=None,
                             sum_mode=0):
    """Select only the sum constructor; all Hamiltonian/RDM code is shared."""
    if sum_mode not in (0, 1):
        raise RuntimeError("Invalid TNSUMMODE code; expected DIRECT=0 or FIT=1")
    constructor = build_direct_total_state if sum_mode == 0 else fit_variational_total_state
    return constructor(driver, states, coefficient, smat, nelec, bond_dimension,
                       root, identity_mpo, h_mpo, eigenvalue, state_scratch,
                       warm_read_directory=warm_read_directory,
                       warm_write_directory=warm_write_directory)



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


def assemble_contracted_d3g(g_orth, dm2, transition):
    """Assemble ``X[d,c]`` from ``T[c,d]=<E[c,d] V>`` and the 2-RDM.

    This is the array form of the exact operator identity derived in
    ``iscf5/tnvb_contracted_gradient_derivation.tex``.  Keeping the index
    transpose in one helper prevents the validation and production paths
    from drifting apart.
    """
    norb = g_orth.shape[0]
    if g_orth.shape != (norb,) * 4 or dm2.shape != (norb,) * 4:
        raise RuntimeError("contracted D3G requires rank-four square tensors")
    if transition.shape != (norb, norb):
        raise RuntimeError("contracted D3G transition matrix has wrong shape")
    lower = np.einsum("abde,cabe->dc", g_orth, dm2, optimize=True)
    contracted = 2.0 * transition.T - 2.0 * lower
    return contracted, lower


def two_body_transition_commutator(g_orth, dm2):
    """Return ``< [E[p,q], V] >`` using only the spin-free 2-RDM.

    For a real wavefunction and real integrals,

    ``T[p,q] - T[q,p] = <[E[p,q], V]>``, where
    ``T[p,q] = <E[p,q] V>``.  The commutator of a one-body excitation with
    the two-body Hamiltonian remains a two-body operator, so this matrix does
    not require a 3-RDM.  It lets the explicit scalar-MPO backend evaluate
    only one triangle of ``T``.
    """
    if g_orth.ndim != 4 or dm2.shape != g_orth.shape:
        raise RuntimeError("two-body transition commutator shape mismatch")
    commutator = 0.5 * (
        np.einsum("qjkl,pklj->pq", g_orth, dm2, optimize=True)
        + np.einsum("ijql,iplj->pq", g_orth, dm2, optimize=True)
        - np.einsum("ijkp,ikqj->pq", g_orth, dm2, optimize=True)
        - np.einsum("ipkl,iklq->pq", g_orth, dm2, optimize=True)
    )
    antisymmetry_error = float(
        np.max(np.abs(commutator + commutator.T))
    )
    tolerance = 1.0e-9 * max(1.0, float(np.max(np.abs(commutator))))
    if antisymmetry_error > tolerance:
        raise RuntimeError(
            "two-body transition commutator is not antisymmetric: "
            f"error={antisymmetry_error:.3e}, limit={tolerance:.3e}"
        )
    # Remove harmless roundoff from the two algebraically equivalent halves.
    return 0.5 * (commutator - commutator.T)


def _decode_npdm_linear_index(linear_index, n_sites, rank):
    """Decode GeneralNPDMMPO's base-``n_sites`` flattened orbital tuple."""
    indices = [0] * rank
    value = int(linear_index)
    for position in range(rank - 1, -1, -1):
        indices[position] = value % n_sites
        value //= n_sites
    if value != 0:
        raise RuntimeError("NPDM orbital index exceeds its declared rank")
    return indices


def _packed_site_index_value(site_index):
    """Recover the 48-bit GeneralNPDMMPO flattened index."""
    return (
        (int(site_index[0]) << 36)
        | (int(site_index[1]) << 24)
        | (int(site_index[2]) << 12)
        | int(site_index[3])
    )


def _site_index_from_linear(bw, linear_index, component=0):
    """Create Block2's four-word encoding of a flattened NPDM index."""
    if linear_index < 0 or linear_index >= (1 << 48):
        raise RuntimeError("contracted NPDM index does not fit in 48 bits")
    return bw.b.SiteIndex(
        (
            (int(linear_index) >> 36) & 0xFFF,
            (int(linear_index) >> 24) & 0xFFF,
            (int(linear_index) >> 12) & 0xFFF,
            int(linear_index) & 0xFFF,
        ),
        (int(component),),
    )


def _su2_comp_enum_schemes(bw):
    """Return the pinned rank-6 input and rank-2 output SU(2) schemes.

    This is the deliberately enumerable reference for ``SU2COMP``.  It keeps
    every fermionic permutation and every allowed SU(2) recoupling generated
    by Block2, then reduces the observable from K**6 to K**2 at the middle
    expressions.  The production NC/CN builder must reproduce this result
    without first constructing the rank-6 environments.
    """
    op_1pdm = "(C+D)0"
    op_3pdm = op_1pdm
    for _ in range(2):
        op_3pdm = f"((C+{op_3pdm})1+D)0"
    input_perm = bw.b.SpinPermScheme.initialize_su2(6, op_3pdm, True)
    input_scheme = bw.b.NPDMScheme(
        bw.b.VectorSpinPermScheme([input_perm])
    )
    output_perm = bw.b.SpinPermScheme.initialize_su2(2, op_1pdm, True)
    output_scheme = bw.b.NPDMScheme(
        bw.b.VectorSpinPermScheme([output_perm])
    )
    return input_perm, input_scheme, output_perm, output_scheme


def _spin_perm_scheme_statistics(scheme):
    """Count all concrete permutation/recoupling branches in a scheme."""
    permutation_keys = 0
    recoupling_terms = 0
    coupled_expressions = set()
    coefficient_magnitudes = set()
    for channel_map in scheme.data:
        permutation_keys += len(channel_map)
        for permutation in channel_map:
            terms = channel_map[permutation]
            recoupling_terms += len(terms)
            for coefficient, expression in terms:
                coupled_expressions.add(str(expression))
                coefficient_magnitudes.add(round(abs(float(coefficient)), 12))
    return {
        "index_patterns": len(scheme.index_patterns),
        "channel_maps": len(scheme.data),
        "permutation_keys": permutation_keys,
        "recoupling_terms": recoupling_terms,
        "coupled_expressions": len(coupled_expressions),
        "coefficient_magnitudes": tuple(sorted(coefficient_magnitudes)),
    }


def su2_comp_enum_channel_statistics(bw):
    """Expose complete pinned-channel counts for the 4e/4o prototype.

    ``input`` is the pure rank-3 PDM tensor used by the multi-output
    contraction.  ``transition_product`` is the independently evaluated
    ``E[c,d] V`` operator used to check the product coupling tree.
    """
    input_perm, _, output_perm, _ = _su2_comp_enum_schemes(bw)
    one_body = "(C+D)0"
    two_body = "((C+(C+D)0)1+D)0"
    transition_product = bw.b.SpinPermScheme.initialize_su2(
        6, f"({one_body}+{two_body})0", True
    )
    return {
        "input": _spin_perm_scheme_statistics(input_perm),
        "output": _spin_perm_scheme_statistics(output_perm),
        "transition_product": _spin_perm_scheme_statistics(
            transition_product
        ),
    }


def contract_su2_3pdm_middle_operators(
    bw, mpo, output_scheme, g_orth, cutoff=0.0
):
    """Precontract every SU(2) 3PDM middle expression with ``g``.

    ``GeneralNPDMMPO`` normally emits one middle expression for every
    ``(a,b,c,d,e,f)`` and later allocates a K**6 result.  This transformation
    changes only those output expressions: all terms with common ``(d,c)``
    are accumulated with coefficient ``g[f,a,e,b]`` into one ``XPDM(d,c)``.
    The left/right complementary operators and their rigorously generated
    fermionic/SU(2) phases are untouched.

    After the transformation ``Expect.get_npdm`` sees a one-particle output
    scheme and therefore allocates only K**2 values.  The caller must still
    apply the SU(2) rank-three normalization factor ``sqrt(2)**3`` because the
    operator environments were generated from the rank-three scheme.
    """
    if cutoff < 0.0:
        raise RuntimeError("contracted 3PDM integral cutoff must be nonnegative")
    norb = int(mpo.n_sites)
    if g_orth.shape != (norb,) * 4:
        raise RuntimeError("contracted 3PDM requires a square rank-four tensor")
    use_cpp = os.environ.get("XMVB_BLOCK2_NPDM_CPP_CONTRACT", "1") == "1"
    if use_cpp and hasattr(mpo, "contract_eri_middle"):
        coefficients = bw.b.VectorDouble(
            np.ascontiguousarray(g_orth, dtype=np.float64).reshape(-1).tolist()
        )
        original_total, contracted_total = mpo.contract_eri_middle(
            coefficients, cutoff
        )
        mpo.npdm_scheme = output_scheme
        return int(original_total), int(contracted_total)
    kept_total = 0
    original_total = 0
    for middle in range(norb - 1):
        mpo.load_middle_operators(middle)
        names = mpo.middle_operator_names[middle]
        expressions = mpo.middle_operator_exprs[middle]
        if len(names) != len(expressions):
            raise RuntimeError("3PDM middle names/expressions have different sizes")
        original_total += len(names)
        grouped = {}
        q_labels = {}
        for position in range(len(names)):
            name = names[position]
            expression = expressions[position]
            if name.site_index.s() != 0:
                raise RuntimeError("SU(2) 3PDM must have one spin component")
            flat = _packed_site_index_value(name.site_index)
            a, b, c, d, e, f = _decode_npdm_linear_index(flat, norb, 6)
            weight = float(g_orth[f, a, e, b])
            if abs(weight) <= cutoff:
                continue
            try:
                product = bw.bs.OpProduct(
                    expression.a,
                    expression.b,
                    float(expression.factor) * weight,
                    int(expression.conj),
                )
            except AttributeError as exc:
                raise RuntimeError(
                    "3PDM middle expression is not a simple operator product"
                ) from exc
            output_flat = d * norb + c
            grouped.setdefault(output_flat, []).append(product)
            q_labels.setdefault(output_flat, name.q_label)

        ordered = sorted(grouped)
        new_names = bw.bs.SymbolicColumnVector(len(ordered))
        new_expressions = bw.bs.SymbolicColumnVector(len(ordered))
        for position, output_flat in enumerate(ordered):
            products = grouped[output_flat]
            new_names[position] = bw.bs.OpElement(
                names[0].name,
                _site_index_from_linear(bw, output_flat),
                q_labels[output_flat],
            )
            if len(products) == 1:
                new_expressions[position] = products[0]
            else:
                product_vector = bw.bs.VectorOpProduct(products)
                new_expressions[position] = bw.bs.OpSum(product_vector)
        mpo.middle_operator_names[middle] = new_names
        mpo.middle_operator_exprs[middle] = new_expressions
        mpo.save_middle_operators(middle)
        mpo.unload_middle_operators(middle)
        kept_total += len(ordered)
    mpo.npdm_scheme = output_scheme
    return original_total, kept_total


def contracted_d3g_from_weighted_su2_npdm(
    driver,
    total,
    norm,
    g_orth,
    cutoff=0.0,
    iprint=0,
):
    """Compute ``g[f,a,e,b] Gamma3[a,b,c,d,e,f]`` without a 3RDM."""
    if SymmetryTypes.SU2 not in driver.symm_type:
        raise RuntimeError("weighted contracted 3PDM requires an SU(2) driver")
    if driver.mpi is not None:
        raise NotImplementedError(
            "weighted contracted 3PDM is currently shared-memory only"
        )
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("weighted contracted 3PDM requires a positive norm")
    bw = driver.bw
    _, input_scheme, _, output_scheme = _su2_comp_enum_schemes(bw)

    mstate = total.deep_copy("TNVB-D3G-WEIGHTED-KET@TMP")
    if mstate.dot == 2:
        mstate.dot = 1
        if mstate.center == mstate.n_sites - 2:
            mstate.center = mstate.n_sites - 1
            mstate.canonical_form = mstate.canonical_form[:-1] + "S"
        elif mstate.center == 0:
            mstate.canonical_form = "K" + mstate.canonical_form[1:]
        else:
            raise RuntimeError(
                "two-site total MPS must be centered at an edge before the "
                "weighted contracted-3PDM sweep"
            )
        mstate.save_data()
    mstate.info.load_mutable()
    mstate.info.bond_dim = max(
        mstate.info.bond_dim, mstate.info.get_max_bond_dimension()
    )
    mstate.info.deallocate_mutable()
    driver.align_mps_center(mstate, mstate)
    build_started = time.perf_counter()
    print(
        "[TNVB-BLOCK2-D3G-NPDM-STAGE] general_3pdm_mpo_build start",
        flush=True,
    )
    primary_mpo = bw.bs.GeneralNPDMMPO(driver.ghamil, input_scheme, False)
    primary_mpo.delta_quantum = (mstate.info.target - mstate.info.target)[0]
    primary_mpo.iprint = 2 if iprint >= 4 else min(iprint, 1)
    primary_mpo.build()
    build_seconds = time.perf_counter() - build_started
    print(
        "[TNVB-BLOCK2-D3G-NPDM-STAGE] general_3pdm_mpo_build complete "
        f"seconds={build_seconds:.3f}",
        flush=True,
    )
    contract_started = time.perf_counter()
    original_middle, contracted_middle = contract_su2_3pdm_middle_operators(
        bw, primary_mpo, output_scheme, g_orth, cutoff=cutoff
    )
    contract_seconds = time.perf_counter() - contract_started
    print(
        "[TNVB-BLOCK2-D3G-NPDM-STAGE] middle_precontraction complete "
        f"original={original_middle} contracted={contracted_middle} "
        f"seconds={contract_seconds:.3f}",
        flush=True,
    )
    # Development probe for the native term-graph design.  Once the middle
    # outputs have been contracted from K**6 to K**2, SimplifiedMPO can trace
    # those outputs backwards and remove partial strings that no longer feed
    # the requested contraction.  This still pays the temporary rank-six
    # build cost, so it is not the production kernel; it tells the native
    # constructor exactly which left/right operator families survive.
    if os.environ.get("XMVB_BLOCK2_WEIGHTED_NPDM_SIMPLIFY", "0") == "1":
        simplify_started = time.perf_counter()
        primary_mpo = bw.bs.SimplifiedMPO(
            primary_mpo,
            bw.bs.Rule(),
            False,
            False,
            bw.b.OpNamesSet.all_ops(),
            "TNVB-D3G-WEIGHTED-SIMPLIFIED",
        )
        print(
            "[TNVB-BLOCK2-D3G-NPDM-STAGE] postcontract_simplify complete "
            f"seconds={time.perf_counter() - simplify_started:.3f}",
            flush=True,
        )

    environment_started = time.perf_counter()
    environment = bw.bs.MovingEnvironment(
        primary_mpo, mstate, mstate, "TNVB-D3G-WEIGHTED"
    )
    environment.cached_contraction = False
    environment.fused_contraction_rotation = True
    environment.init_environments(iprint >= 2)
    environment_seconds = time.perf_counter() - environment_started
    print(
        "[TNVB-BLOCK2-D3G-NPDM-STAGE] environment complete "
        f"seconds={environment_seconds:.3f}",
        flush=True,
    )
    expectation = bw.bs.Expect(
        environment, mstate.info.bond_dim, mstate.info.bond_dim
    )
    expectation.cutoff = 1.0e-24
    expectation.zero_dot_algo = True
    weighted_algorithm = os.environ.get(
        "XMVB_BLOCK2_WEIGHTED_NPDM_ALGORITHM", "fast"
    ).strip().lower()
    weighted_algorithms = {
        "fast": bw.b.ExpectationAlgorithmTypes.Fast,
        "normal": bw.b.ExpectationAlgorithmTypes.Normal,
        "automatic": bw.b.ExpectationAlgorithmTypes.Automatic,
    }
    if weighted_algorithm not in weighted_algorithms:
        raise RuntimeError(
            "XMVB_BLOCK2_WEIGHTED_NPDM_ALGORITHM must be fast, normal, "
            "or automatic"
        )
    expectation.algo_type = weighted_algorithms[weighted_algorithm]
    expectation.iprint = iprint
    sweep_started = time.perf_counter()
    expectation.solve(True, mstate.center == 0)
    npdms = list(expectation.get_npdm())
    sweep_seconds = time.perf_counter() - sweep_started
    print(
        "[TNVB-BLOCK2-D3G-NPDM-STAGE] npdm_sweep complete "
        f"seconds={sweep_seconds:.3f}",
        flush=True,
    )
    if driver.clean_scratch:
        environment.remove_partition_files()
    if len(npdms) != 1:
        raise RuntimeError("weighted SU(2) 3PDM returned multiple components")
    result = np.asarray(npdms[0], dtype=np.float64)
    result *= math.sqrt(2.0) ** 3 / norm
    if result.shape != (driver.n_sites, driver.n_sites):
        raise RuntimeError("weighted contracted 3PDM has the wrong shape")
    return result, original_middle, contracted_middle


def contracted_d3g_from_su2_comp_enum(
    driver,
    total,
    norm,
    g_orth,
    cutoff=0.0,
    iprint=0,
):
    """Enumerable SU(2), multi-output reference for the future NC/CN MPO.

    This intentionally retains the K**6 rank-3 NPDM environment topology and
    is therefore restricted to small validation systems.  Its K**2 result,
    channel census, signs, and recoupling factors form the exact oracle for
    the coefficient-aware complementary-operator implementation.
    """
    statistics = su2_comp_enum_channel_statistics(driver.bw)
    input_statistics = statistics["input"]
    print(
        "[TNVB-BLOCK2-SU2COMP-ENUM] channels "
        f"patterns={input_statistics['index_patterns']} "
        f"permutations={input_statistics['permutation_keys']} "
        f"recouplings={input_statistics['recoupling_terms']} "
        f"expressions={input_statistics['coupled_expressions']}",
        flush=True,
    )
    result, original_middle, contracted_middle = (
        contracted_d3g_from_weighted_su2_npdm(
            driver,
            total,
            norm,
            g_orth,
            cutoff=cutoff,
            iprint=iprint,
        )
    )
    return result, original_middle, contracted_middle, statistics


def contracted_d3g_from_su2_comp_mpo(
    driver,
    total,
    norm,
    g_orth,
    dm2,
    cutoff=0.0,
    iprint=0,
    _native=False,
    source_scratch=None,
    scratch_parent=None,
):
    """Evaluate ``<E[c,d] V>`` with the SU(2)-coupled product MPO.

    The first factor is the rank-two, multi-output ``(C+D)0`` NPDM MPO.  The
    second is an integral-aware scalar two-electron MPO.  ``SU2FusedMPO``
    couples their virtual irreps explicitly, retaining every allowed channel
    instead of selecting one branch as generic ``StackedMPO`` does.  Hence the
    virtual bond grows polynomially while neither a K**6 operator list nor a
    K**6 result tensor is formed.
    """
    if SymmetryTypes.SU2 not in driver.symm_type:
        raise RuntimeError("SU2COMP requires an SU(2) driver")
    if driver.mpi is not None:
        raise NotImplementedError("SU2COMP is currently shared-memory only")
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("SU2COMP requires a positive state norm")
    if cutoff < 0.0:
        raise RuntimeError("SU2COMP integral cutoff must be nonnegative")
    norb = int(driver.n_sites)
    if g_orth.shape != (norb,) * 4 or dm2.shape != (norb,) * 4:
        raise RuntimeError("SU2COMP requires square rank-four ERI/2-RDM tensors")

    bw = driver.bw
    state = total.deep_copy("TNVB-D3G-SU2COMP-KET@TMP")
    if state.dot == 2:
        state.dot = 1
        if state.center == state.n_sites - 2:
            state.center = state.n_sites - 1
            state.canonical_form = state.canonical_form[:-1] + "S"
        elif state.center == 0:
            state.canonical_form = "K" + state.canonical_form[1:]
        else:
            raise RuntimeError(
                "two-site total MPS must be centered at an edge before the "
                "SU2COMP sweep"
            )
        state.save_data()
    state.info.load_mutable()
    state.info.bond_dim = max(
        state.info.bond_dim, state.info.get_max_bond_dimension()
    )
    state.info.deallocate_mutable()
    driver.align_mps_center(state, state)

    build_started = time.perf_counter()
    op_string = "(C+D)0"
    permutation = bw.b.SpinPermScheme.initialize_su2(2, op_string, True)
    scheme = bw.b.NPDMScheme(bw.b.VectorSpinPermScheme([permutation]))
    primary_mpo = bw.bs.GeneralNPDMMPO(driver.ghamil, scheme, False)
    primary_mpo.delta_quantum = (
        state.info.target - state.info.target
    )[0]
    primary_mpo.iprint = 2 if iprint >= 4 else min(iprint, 1)
    primary_mpo.build()

    native_layout = "nc-raw"
    if _native:
        native_layout = os.environ.get(
            "XMVB_BLOCK2_NATIVE_V_MPO", "nc-raw"
        ).strip().lower()
    if _native and native_layout in (
        "nc", "cn", "nc-raw", "cn-raw", "nc-mat", "cn-mat"
    ):
        native_family = native_layout.split("-", 1)[0]
        native_algorithms = {
            "nc": MPOAlgorithmTypes.NoTransConventionalNC,
            "cn": MPOAlgorithmTypes.NoTransConventionalCN,
        }
        vee_wrapper = driver.get_qc_mpo(
            h1e=np.zeros((norb, norb), dtype=np.float64),
            g2e=g_orth,
            ecore=0.0,
            cutoff=max(cutoff, 1.0e-16),
            integral_cutoff=cutoff,
            symmetrize=False,
            algo_type=native_algorithms[native_family],
            add_ident=False,
            iprint=2 if iprint >= 4 else min(iprint, 1),
        )
        if native_layout.endswith("-raw"):
            # Preserve the native QC normal/complementary operator chain.
            # Its virtual-name factors are SU(2) gauge data and must enter the
            # fused recurrence; SU2ComplementaryMPO retains them explicitly.
            vee_mpo = vee_wrapper.prim_mpo
        elif native_layout.endswith("-mat"):
            # Build the exact blocking formulas without RuleQC operator
            # substitutions.  This materializes the scalar H middle
            # expression while retaining the primitive NC/CN virtual basis.
            # The recurrence tables are not needed after SU2FusedMPO has
            # copied the resulting local tensors and middle formulas.
            vee_mpo = bw.bs.SimplifiedMPO(
                vee_wrapper.prim_mpo, bw.bs.Rule(), False, False
            )
            vee_mpo.left_operator_exprs = bw.bs.VectorSymbolic()
            vee_mpo.right_operator_exprs = bw.bs.VectorSymbolic()
        else:
            # Experimental materialization probe.  Clearing these recurrence
            # tables is not a valid production conversion; 4e/4o must reject
            # this path unless the simplified chain is explicitly rebuilt.
            vee_wrapper.left_operator_exprs = bw.bs.VectorSymbolic()
            vee_wrapper.right_operator_exprs = bw.bs.VectorSymbolic()
            vee_mpo = vee_wrapper
    else:
        general_layouts = {
            "bipartite": bw.b.MPOAlgorithmTypes.FastBipartite,
            "general-nc": bw.b.MPOAlgorithmTypes.NC,
            "general-cn": bw.b.MPOAlgorithmTypes.CN,
            "general-svd": bw.b.MPOAlgorithmTypes.SVD,
            "general-fast-svd": bw.b.MPOAlgorithmTypes.FastSVD,
        }
        if _native and native_layout not in general_layouts:
            raise RuntimeError(
                "XMVB_BLOCK2_NATIVE_V_MPO must be nc, cn, nc-raw, cn-raw, "
                "nc-mat, cn-mat, "
                "bipartite, general-nc, general-cn, general-svd, or "
                "general-fast-svd"
            )
        builder = driver.expr_builder()
        builder.add_sum_term(
            "((C+(C+D)0)1+D)0",
            g_orth,
            cutoff=cutoff,
            perm=[0, 2, 3, 1],
        )
        vee_mpo = bw.bs.GeneralMPO(
            driver.ghamil,
            builder.finalize(),
            general_layouts.get(
                native_layout, bw.b.MPOAlgorithmTypes.FastBipartite
            ),
            max(cutoff, 1.0e-16),
            -1,
            2 if iprint >= 4 else min(iprint, 1),
        )
        vee_mpo.build()
    if iprint >= 3:
        print(
            "[TNVB-BLOCK2-SU2COMP] factor_metadata "
            f"primary_middle_names={len(primary_mpo.middle_operator_names)} "
            f"primary_middle_exprs={len(primary_mpo.middle_operator_exprs)} "
            f"vee_middle_names={len(vee_mpo.middle_operator_names)} "
            f"vee_middle_exprs={len(vee_mpo.middle_operator_exprs)} "
            f"primary_left_exprs={len(primary_mpo.left_operator_exprs)} "
            f"vee_left_exprs={len(vee_mpo.left_operator_exprs)}",
            flush=True,
        )
        for cut in range(norb - 1):
            primary_mpo.load_left_operators(cut)
            primary_mpo.load_right_operators(cut + 1)
            vee_mpo.load_left_operators(cut)
            vee_mpo.load_right_operators(cut + 1)
            print(
                "[TNVB-BLOCK2-SU2COMP] factor_bond "
                f"cut={cut} primary_left="
                f"{len(primary_mpo.left_operator_names[cut])} "
                f"primary_right={len(primary_mpo.right_operator_names[cut + 1])} "
                f"vee_left={len(vee_mpo.left_operator_names[cut])} "
                f"vee_right={len(vee_mpo.right_operator_names[cut + 1])}",
                flush=True,
            )
            primary_mpo.unload_right_operators(cut + 1)
            primary_mpo.unload_left_operators(cut)
            vee_mpo.unload_right_operators(cut + 1)
            vee_mpo.unload_left_operators(cut)

    if _native and os.environ.get("XMVB_BLOCK2_DEBUG_MPO_CONJ", "0") == "1":
        def expression_conjugations(expression):
            values = []
            if hasattr(expression, "conj"):
                values.append(int(expression.conj))
            if hasattr(expression, "strings"):
                for term in expression.strings:
                    if hasattr(term, "conj"):
                        values.append(int(term.conj))
            return values

        for label, mpo in (("free", primary_mpo), ("vee", vee_mpo)):
            site_counts = {}
            middle_counts = {}
            for site in range(norb):
                mpo.load_tensor(site)
                for matrix in (mpo.tensors[site].lmat, mpo.tensors[site].rmat):
                    for expression in matrix.data:
                        for value in expression_conjugations(expression):
                            site_counts[value] = site_counts.get(value, 0) + 1
                mpo.unload_tensor(site)
            if len(mpo.middle_operator_exprs) == norb - 1:
                for cut in range(norb - 1):
                    mpo.load_middle_operators(cut)
                    for expression in mpo.middle_operator_exprs[cut].data:
                        for value in expression_conjugations(expression):
                            middle_counts[value] = middle_counts.get(value, 0) + 1
                    mpo.unload_middle_operators(cut)
            _builtin_print(
                "[TNVB-BLOCK2-SU2COMP-CONJ] "
                f"factor={label} site={site_counts} middle={middle_counts}",
                flush=True,
            )
    product_class = (
        bw.bs.SU2ComplementaryMPO if _native else bw.bs.SU2FusedMPO
    )
    product_label = "SU2-NATIVE" if _native else "SU2COMP"
    cut_kernel = _native and (
        os.environ.get("XMVB_BLOCK2_CUT_KERNEL", "0") == "1"
    )
    batch_size = 0
    output_batches = [None]
    output_cuts = [None]
    triangular_outputs = False
    if _native:
        batch_size = int(
            os.environ.get("XMVB_BLOCK2_NATIVE_OUTPUT_BATCH", "8")
        )
        if batch_size < 0:
            raise RuntimeError(
                "XMVB_BLOCK2_NATIVE_OUTPUT_BATCH must be nonnegative"
            )
        if batch_size or cut_kernel:
            # For real states/integrals, the upper triangle follows exactly
            # from the 2-RDM commutator.  Never allocate product channels for
            # those redundant outputs.
            triangular_outputs = True
            output_keys = []
            cut_output_keys = []
            seen_output_keys = set()
            for cut in range(norb - 1):
                primary_mpo.load_middle_operators(cut)
                current_keys = []
                for expression in primary_mpo.middle_operator_names[cut].data:
                    if not hasattr(expression, "site_index"):
                        continue
                    flat = _packed_site_index_value(expression.site_index)
                    c, d = _decode_npdm_linear_index(flat, norb, 2)
                    if c < d:
                        continue
                    key = int(hash(expression.site_index))
                    if key not in seen_output_keys:
                        seen_output_keys.add(key)
                        output_keys.append(key)
                        current_keys.append(key)
                primary_mpo.unload_middle_operators(cut)
                cut_output_keys.append(current_keys)
            expected_outputs = norb * (norb + 1) // 2
            if len(output_keys) != expected_outputs:
                raise RuntimeError(
                    "native output batching expected "
                    f"{expected_outputs} triangular PDM1 outputs, "
                    f"found {len(output_keys)}"
                )
            if cut_kernel:
                output_batches = [keys for keys in cut_output_keys if keys]
                output_cuts = [
                    cut for cut, keys in enumerate(cut_output_keys) if keys
                ]
            else:
                output_batches = [
                    output_keys[start : start + batch_size]
                    for start in range(0, len(output_keys), batch_size)
                ]
                output_cuts = [None] * len(output_batches)

    transition = np.zeros((norb, norb), dtype=np.float64)
    build_seconds = 0.0
    environment_seconds = 0.0
    sweep_seconds = 0.0
    output_batch_count = len(output_batches)
    if (
        _native
        and batch_size > 0
        and native_layout in ("nc-raw", "cn-raw")
        and source_scratch
        and scratch_parent
    ):
        worker_count, block2_threads = native_d3g_parallel_layout(
            output_batch_count, norb, state.info.bond_dim
        )
        if worker_count > 1:
            (
                transition,
                build_seconds,
                environment_seconds,
                sweep_seconds,
            ) = contract_parallel_native_output_batches(
                state,
                source_scratch,
                scratch_parent,
                norb,
                driver.n_elec,
                state.info.bond_dim,
                g_orth,
                cutoff,
                native_layout,
                output_batches,
                output_cuts,
                worker_count,
                block2_threads,
            )
            # The child results already contain the complete sum.  Leave the
            # serial loop empty rather than rebuilding the same output MPOs.
            output_batches = []
    for batch_index, (selected_outputs, output_cut) in enumerate(
        zip(output_batches, output_cuts), 1
    ):
        batch_started = time.perf_counter()
        product_arguments = [
            primary_mpo,
            vee_mpo,
            min(iprint, 1),
            f"TNVB-{product_label}-{batch_index}",
        ]
        if selected_outputs is not None:
            product_arguments.append(bw.b.VectorULInt(selected_outputs))
        fused_mpo = product_class(*product_arguments)
        batch_build_seconds = time.perf_counter() - batch_started
        build_seconds += batch_build_seconds
        summary = "unavailable"
        try:
            nnz, size, bond = fused_mpo.get_summary()
            summary = f"bond={bond} nnz={nnz} size={size}"
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
        print(
            f"[TNVB-BLOCK2-{product_label}] build layout={native_layout} "
            f"batch={batch_index}/{output_batch_count} {summary} "
            f"seconds={batch_build_seconds:.3f}",
            flush=True,
        )

        if output_cut is not None:
            driver.align_mps_center(state, output_cut)
        environment_started = time.perf_counter()
        environment = bw.bs.MovingEnvironment(
            fused_mpo,
            state,
            state,
            f"TNVB-D3G-{product_label}-{batch_index}",
        )
        environment.cached_contraction = False
        environment.fused_contraction_rotation = True
        environment.init_environments(iprint >= 2)
        environment_seconds += time.perf_counter() - environment_started
        expectation = bw.bs.Expect(
            environment, state.info.bond_dim, state.info.bond_dim
        )
        expectation.cutoff = 1.0e-24
        expectation.zero_dot_algo = True
        expectation.algo_type = (
            bw.b.ExpectationAlgorithmTypes.Fast
            if _native
            else bw.b.ExpectationAlgorithmTypes.Normal
        )
        expectation.iprint = iprint
        sweep_started = time.perf_counter()
        if output_cut is None:
            expectation.solve(True, state.center == 0)
            npdms = list(expectation.get_npdm())
            if len(npdms) != 1:
                raise RuntimeError(
                    f"SU2COMP returned {len(npdms)} one-particle components"
                )
            batch_transition = np.asarray(npdms[0], dtype=np.float64)
        else:
            iteration = expectation.update_zero_dot(
                output_cut,
                True,
                True,
                state.info.bond_dim,
                state.info.bond_dim,
            )
            batch_transition = np.zeros((norb, norb), dtype=np.float64)
            for expression, value in iteration.expectations:
                flat = _packed_site_index_value(expression.site_index)
                c, d = _decode_npdm_linear_index(flat, norb, 2)
                batch_transition[c, d] += float(value)
        sweep_seconds += time.perf_counter() - sweep_started
        if driver.clean_scratch:
            environment.remove_partition_files()
        if batch_transition.shape != (norb, norb):
            raise RuntimeError(
                "SU2COMP returned shape "
                f"{batch_transition.shape}, expected {(norb, norb)}"
            )
        transition += batch_transition
        del expectation, environment
        fused_mpo.deallocate()
        del fused_mpo

    transition *= math.sqrt(2.0) / norm
    if triangular_outputs:
        commutator = two_body_transition_commutator(g_orth, dm2)
        for c in range(norb):
            for d in range(c + 1, norb):
                transition[c, d] = transition[d, c] + commutator[c, d]
    contracted, lower = assemble_contracted_d3g(g_orth, dm2, transition)
    vee_expectation = 0.5 * float(
        np.einsum("ijkl,iklj", g_orth, dm2, optimize=True)
    )
    trace_error = abs(
        float(np.trace(transition)) - driver.n_elec * vee_expectation
    )
    print(
        f"[TNVB-BLOCK2-{product_label}] "
        f"outputs={'triangle' if triangular_outputs else 'full'} "
        f"batches={output_batch_count} build_seconds={build_seconds:.3f} "
        f"environment_seconds={environment_seconds:.3f} "
        f"sweep_seconds={sweep_seconds:.3f} trace_error={trace_error:.3e}",
        flush=True,
    )
    return contracted, transition, lower, vee_expectation, trace_error


def contracted_d3g_from_su2_complementary(
    driver,
    total,
    norm,
    g_orth,
    dm2,
    cutoff=0.0,
    iprint=0,
    source_scratch=None,
    scratch_parent=None,
):
    """Exact integral-aware SU(2) complementary-operator contraction.

    The dedicated Block2 class retains all allowed recoupling channels but
    removes virtual PDM1 x V pairs that are unreachable from either physical
    edge before allocating an MPS environment.  Its output is the same K x K
    transition matrix as the product-MPO oracle and it never constructs a
    rank-six NPDM or calls ``get_3pdm``.
    """
    return contracted_d3g_from_su2_comp_mpo(
        driver,
        total,
        norm,
        g_orth,
        dm2,
        cutoff=cutoff,
        iprint=iprint,
        _native=True,
        source_scratch=source_scratch,
        scratch_parent=scratch_parent,
    )


def _stacked_sweep_direction(driver, state):
    """Canonicalize a diagnostic stacked sweep at the requested edge."""
    mode = os.environ.get(
        "XMVB_BLOCK2_D3G_STACKED_DIRECTION", "auto"
    ).lower()
    if mode == "forward":
        driver.align_mps_center(state, 0)
        return True
    if mode == "backward":
        driver.align_mps_center(state, state.n_sites - 1)
        return False
    if mode == "auto":
        return state.center == 0
    raise RuntimeError(
        "XMVB_BLOCK2_D3G_STACKED_DIRECTION must be auto, forward, or backward"
    )


def get_stacked_transition_1pdm(
    driver,
    state,
    stacked_mpo,
    algo_type=NPDMAlgorithmTypes.Default,
    cutoff=1.0e-24,
    iprint=0,
):
    """Return all ``<state|E[p,q] stacked_mpo|state>`` in one NPDM sweep.

    Block2 already supports a product of two MPOs through
    ``MovingEnvironment.stacked_mpo``.  Its public ``expectation`` wrapper
    exposes that feature only for a scalar primary MPO.  Here the primary MPO
    is instead the free-index SU(2) one-particle NPDM MPO.  Consequently one
    shared left/right environment sweep produces the complete transition
    matrix without either constructing ``stacked_mpo|state>`` or rebuilding
    an environment for every pair of orbital indices.

    The returned matrix uses the same convention and SU(2) normalization as
    ``DMRGDriver.get_1pdm``; it is deliberately *not* divided by the state
    norm.
    """
    if driver.mpi is not None:
        raise NotImplementedError(
            "stacked free-index NPDM is currently implemented for shared-memory "
            "Block2 only"
        )
    if cutoff < 0.0:
        raise RuntimeError("stacked-NPDM cutoff must be nonnegative")
    site_type = int(os.environ.get("XMVB_BLOCK2_D3G_SITE_TYPE", "0"))
    if site_type not in (0, 2):
        raise RuntimeError("XMVB_BLOCK2_D3G_SITE_TYPE must be 0 or 2")

    bw = driver.bw
    op_str = "(C+D)0"
    perm = bw.b.SpinPermScheme.initialize_su2(2, op_str, True)
    perms = bw.b.VectorSpinPermScheme([perm])

    # Match DMRGDriver.get_npdm exactly: a zero-/one-site NPDM sweep needs a
    # one-site MPS.  Work on a private copy so the total-state canonical form
    # remains available to the caller and to the reference checks.
    mstate = state.deep_copy("TNVB-D3G-NPDM-KET@TMP")
    if mstate.dot == 2 and site_type != 2:
        mstate.dot = 1
        if mstate.center == mstate.n_sites - 2:
            mstate.center = mstate.n_sites - 1
            mstate.canonical_form = mstate.canonical_form[:-1] + "S"
        elif mstate.center == 0:
            mstate.canonical_form = "K" + mstate.canonical_form[1:]
        else:
            raise RuntimeError(
                "two-site total MPS must be centered at an edge before the "
                "stacked-NPDM sweep"
            )
        mstate.save_data()

    mstate.info.load_mutable()
    mstate.info.bond_dim = max(
        mstate.info.bond_dim, mstate.info.get_max_bond_dimension()
    )
    mstate.info.deallocate_mutable()
    driver.align_mps_center(mstate, mstate)
    sweep_forward = _stacked_sweep_direction(driver, mstate)

    scheme = bw.b.NPDMScheme(perms)
    opdq = (mstate.info.target - mstate.info.target)[0]
    opdq.twos = opdq.twos_low = bw.b.SpinPermRecoupling.get_target_twos(op_str)
    # Symbol-free NPDM replaces the primary MPO's middle expressions by a
    # counter.  Those expressions are precisely what a stacked secondary MPO
    # has to multiply, so the Block2 0.5.3 combination is invalid (and can
    # segfault in environment construction).  Retain explicit NPDM symbols;
    # this still shares a single environment for all free indices.
    if NPDMAlgorithmTypes.SymbolFree in algo_type:
        algo_type = NPDMAlgorithmTypes.Normal
    symbol_free = False
    primary_mpo = bw.bs.GeneralNPDMMPO(
        driver.ghamil, scheme, symbol_free
    )
    primary_mpo.delta_quantum = opdq
    primary_mpo.iprint = 2 if iprint >= 4 else min(iprint, 1)
    primary_mpo.build()
    primary_mpo = bw.bs.SimplifiedMPO(
        primary_mpo, bw.bs.Rule(), False, False
    )

    environment = bw.bs.MovingEnvironment(
        primary_mpo, mstate, mstate, "TNVB-D3G-NPDM"
    )
    environment.stacked_mpo = stacked_mpo
    # This is the setting used by DMRGDriver.expectation for stacked MPOs.
    # Delaying only one factor would no longer represent their product.
    environment.delayed_contraction = bw.b.OpNamesSet()
    environment.cached_contraction = False
    environment.fused_contraction_rotation = (
        os.environ.get("XMVB_BLOCK2_D3G_FUSED", "1") == "1"
    )
    environment.init_environments(iprint >= 2)

    expectation = bw.bs.Expect(
        environment, mstate.info.bond_dim, mstate.info.bond_dim
    )
    expectation.cutoff = cutoff
    expectation.zero_dot_algo = site_type == 0
    # Block2 0.5.3's Fast expectation kernel contracts only ``op->mat`` and
    # does not receive ``op->stacked_mat``.  Automatic can select that kernel
    # for this explicit NPDM MPO and would then silently return an ordinary
    # 1-RDM.  Normal is the only correct algorithm for the product of a
    # free-index primary MPO and the stacked two-electron MPO.
    expectation.algo_type = bw.b.ExpectationAlgorithmTypes.Normal
    expectation.iprint = iprint
    expectation.solve(True, sweep_forward)

    npdms = list(expectation.get_npdm())
    if driver.clean_scratch:
        environment.remove_partition_files()
    if len(npdms) != 1:
        raise RuntimeError(
            f"SU(2) stacked one-particle NPDM returned {len(npdms)} components"
        )
    # ``(C+D)0`` is a normalized rank-zero tensor.  The ordinary spin-summed
    # E[p,q] therefore carries the same sqrt(2) factor used in get_npdm().
    result = np.asarray(npdms[0], dtype=np.float64) * math.sqrt(2.0)
    expected_shape = (driver.n_sites, driver.n_sites)
    if result.shape != expected_shape:
        raise RuntimeError(
            "stacked one-particle NPDM returned shape "
            f"{result.shape}, expected {expected_shape}"
        )
    return result


def get_sz_stacked_transition_1pdm(
    driver,
    state,
    stacked_mpo,
    cutoff=1.0e-24,
    iprint=0,
):
    """Return ``<state|E[p,q] stacked_mpo|state>`` in Abelian SZ symmetry.

    The TNVB state is optimized and summed in SU(2).  Before this routine is
    entered, its singlet is expanded exactly into the single ``M_S = 0`` SZ
    component.  In SZ symmetry the spin trace

        E[p,q] = c[p,alpha]^dagger c[q,alpha]
               + c[p,beta ]^dagger c[q,beta ]

    is represented by the two independent NPDM strings ``cd`` and ``CD``.
    Their environments contain only Abelian quantum numbers, so ordinary
    fermionic multiplication with the scalar two-electron MPO is sufficient;
    no SU(2) tensor-product recoupling is delegated to ``stacked_mpo``.

    The result is deliberately unnormalized.  It has the same index order as
    ``DMRGDriver.get_1pdm`` and is divided by the MPS norm by the caller.
    """
    if SymmetryTypes.SZ not in driver.symm_type:
        raise RuntimeError("contracted D3G measurement requires an SZ driver")
    if driver.mpi is not None:
        raise NotImplementedError(
            "stacked free-index SZ NPDM is currently shared-memory only"
        )
    if cutoff < 0.0:
        raise RuntimeError("stacked-NPDM cutoff must be nonnegative")
    site_type = int(os.environ.get("XMVB_BLOCK2_D3G_SITE_TYPE", "0"))
    if site_type not in (0, 2):
        raise RuntimeError("XMVB_BLOCK2_D3G_SITE_TYPE must be 0 or 2")

    bw = driver.bw
    op_order = os.environ.get(
        "XMVB_BLOCK2_D3G_STACKED_SZ_ORDER", "normal"
    ).lower()
    if op_order == "normal":
        op_strings = ("cd", "CD")
    elif op_order == "reverse":
        op_strings = ("dc", "DC")
    else:
        raise RuntimeError(
            "XMVB_BLOCK2_D3G_STACKED_SZ_ORDER must be normal or reverse"
        )
    perms = bw.b.VectorSpinPermScheme(
        [
            bw.b.SpinPermScheme.initialize_sz(2, op_string, True)
            for op_string in op_strings
        ]
    )

    mstate = state.deep_copy("TNVB-D3G-SZ-NPDM-KET@TMP")
    if mstate.dot == 2 and site_type != 2:
        mstate.dot = 1
        if mstate.center == mstate.n_sites - 2:
            mstate.center = mstate.n_sites - 1
            mstate.canonical_form = mstate.canonical_form[:-1] + "S"
        elif mstate.center == 0:
            mstate.canonical_form = "K" + mstate.canonical_form[1:]
        else:
            raise RuntimeError(
                "two-site SZ total MPS must be centered at an edge before "
                "the stacked-NPDM sweep"
            )
        mstate.save_data()

    mstate.info.load_mutable()
    mstate.info.bond_dim = max(
        mstate.info.bond_dim, mstate.info.get_max_bond_dimension()
    )
    mstate.info.deallocate_mutable()
    driver.align_mps_center(mstate, mstate)
    sweep_forward = _stacked_sweep_direction(driver, mstate)

    scheme = bw.b.NPDMScheme(perms)
    opdq = (mstate.info.target - mstate.info.target)[0]
    primary_mpo = bw.bs.GeneralNPDMMPO(driver.ghamil, scheme, False)
    primary_mpo.delta_quantum = opdq
    primary_mpo.iprint = 2 if iprint >= 4 else min(iprint, 1)
    primary_mpo.build()
    primary_raw = os.environ.get(
        "XMVB_BLOCK2_D3G_STACKED_PRIMARY_RAW", "0"
    ) == "1"
    if not primary_raw:
        primary_mpo = bw.bs.SimplifiedMPO(
            primary_mpo, bw.bs.Rule(), False, False
        )

    stacked_raw = os.environ.get(
        "XMVB_BLOCK2_D3G_STACKED_SECONDARY_RAW", "0"
    ) == "1"
    if stacked_raw:
        if not hasattr(stacked_mpo, "prim_mpo"):
            raise RuntimeError(
                "requested raw stacked MPO but the secondary MPO has no wrapper"
            )
        stacked_mpo = stacked_mpo.prim_mpo
    if iprint >= 1:
        print(
            "[TNVB-BLOCK2-D3G-SZ-STACKED] "
            f"primary_raw={int(primary_raw)} secondary_raw={int(stacked_raw)} "
            f"direction={'forward' if sweep_forward else 'backward'} "
            f"order={op_order}",
            flush=True,
        )
    if os.environ.get("XMVB_BLOCK2_D3G_STACKED_SYMBOLS", "0") == "1":
        for site in range(driver.n_sites):
            primary_mpo.load_left_operators(site)
            stacked_mpo.load_left_operators(site)
            primary_left_names = primary_mpo.left_operator_names[site]
            secondary_left_names = stacked_mpo.left_operator_names[site]
            primary_left_exprs = (
                primary_mpo.left_operator_exprs[site]
                if len(primary_mpo.left_operator_exprs) != 0
                else None
            )
            secondary_left_exprs = (
                stacked_mpo.left_operator_exprs[site]
                if len(stacked_mpo.left_operator_exprs) != 0
                else None
            )
            print(
                "[TNVB-BLOCK2-D3G-SZ-STACKED-LEFT-SYMBOLS] "
                f"site={site} primary_names="
                f"{[str(primary_left_names[i]) for i in range(len(primary_left_names))]} "
                f"primary_exprs="
                f"{None if primary_left_exprs is None else [str(primary_left_exprs[i]) for i in range(len(primary_left_exprs))]} "
                f"secondary_names="
                f"{[str(secondary_left_names[i]) for i in range(len(secondary_left_names))]} "
                f"secondary_exprs="
                f"{None if secondary_left_exprs is None else [str(secondary_left_exprs[i]) for i in range(len(secondary_left_exprs))]}",
                flush=True,
            )
            primary_mpo.unload_left_operators(site)
            stacked_mpo.unload_left_operators(site)
        for middle in range(driver.n_sites - 1):
            primary_mpo.load_middle_operators(middle)
            stacked_mpo.load_middle_operators(middle)
            primary_names = primary_mpo.middle_operator_names[middle]
            primary_exprs = primary_mpo.middle_operator_exprs[middle]
            secondary_names = stacked_mpo.middle_operator_names[middle]
            secondary_exprs = stacked_mpo.middle_operator_exprs[middle]
            print(
                "[TNVB-BLOCK2-D3G-SZ-STACKED-SYMBOLS] "
                f"middle={middle} primary_names="
                f"{[str(primary_names[i]) for i in range(len(primary_names))]} "
                f"primary_exprs="
                f"{[str(primary_exprs[i]) for i in range(len(primary_exprs))]} "
                f"secondary_names="
                f"{[str(secondary_names[i]) for i in range(len(secondary_names))]} "
                f"secondary_exprs="
                f"{[str(secondary_exprs[i]) for i in range(len(secondary_exprs))]}",
                flush=True,
            )
            primary_mpo.unload_middle_operators(middle)
            stacked_mpo.unload_middle_operators(middle)

    environment = bw.bs.MovingEnvironment(
        primary_mpo, mstate, mstate, "TNVB-D3G-SZ-NPDM"
    )
    environment.stacked_mpo = stacked_mpo
    environment.delayed_contraction = bw.b.OpNamesSet()
    environment.cached_contraction = False
    environment.fused_contraction_rotation = (
        os.environ.get("XMVB_BLOCK2_D3G_FUSED", "1") == "1"
    )
    environment.init_environments(iprint >= 2)

    expectation = bw.bs.Expect(
        environment, mstate.info.bond_dim, mstate.info.bond_dim
    )
    expectation.cutoff = cutoff
    expectation.zero_dot_algo = site_type == 0
    # Fast/Automatic do not contract ``stacked_mat`` in Block2 0.5.3.
    expectation.algo_type = bw.b.ExpectationAlgorithmTypes.Normal
    expectation.iprint = iprint
    expectation.solve(True, sweep_forward)

    npdms = [np.asarray(x, dtype=np.float64) for x in expectation.get_npdm()]
    if driver.clean_scratch:
        environment.remove_partition_files()
    if len(npdms) != 2:
        raise RuntimeError(
            f"SZ stacked one-particle NPDM returned {len(npdms)} components"
        )
    result = npdms[0] + npdms[1]
    expected_shape = (driver.n_sites, driver.n_sites)
    if result.shape != expected_shape:
        raise RuntimeError(
            "SZ stacked one-particle NPDM returned shape "
            f"{result.shape}, expected {expected_shape}"
        )
    return result


def save_su2_singlet_as_sz_component(driver, state, scratch, tag):
    """Expand an SU(2) singlet exactly and persist its M_S=0 SZ component.

    Block2 performs the transformation through unfused tensors and explicit
    Clebsch--Gordan coefficients.  No variational fit or bond truncation is
    used.  The returned MPS belongs to the SZ bindings and must only be used
    after the SU(2) driver has been finalized and an SZ driver initialized.
    """
    if SymmetryTypes.SU2 not in driver.symm_type:
        raise RuntimeError("SU2-to-SZ bridge requires an SU2 source driver")
    if state.info.target.twos != 0:
        raise RuntimeError("SU2-to-SZ CONTRACTED bridge currently requires a singlet")
    if not tag or tag == state.info.tag:
        raise RuntimeError("SU2-to-SZ bridge requires a distinct nonempty MPS tag")

    started = time.perf_counter()
    sz_state = driver.mps_change_to_sz(state, tag, sz=0)
    sz_state.info.load_mutable()
    expanded_bond = max(
        int(sz_state.info.bond_dim),
        int(sz_state.info.get_max_bond_dimension()),
    )
    sz_state.info.bond_dim = expanded_bond
    sz_state.save_data()
    sz_state.info.save_data(os.path.join(scratch, f"{tag}-mps_info.bin"))
    sz_state.info.deallocate_mutable()
    return sz_state, expanded_bond, time.perf_counter() - started


def contracted_d3g_from_sz_stacked_npdm(
    driver,
    total,
    norm,
    g_orth,
    dm2,
    algo_type=NPDMAlgorithmTypes.Default,
    iprint=0,
):
    """Compute the exact integral-contracted 3-RDM in an SZ environment."""
    if SymmetryTypes.SZ not in driver.symm_type:
        raise RuntimeError(
            "contracted D3G production path requires an SZ measurement state"
        )
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("stacked-NPDM contraction requires a positive norm")
    vee_mpo = get_hamiltonian_mpo(
        driver,
        np.zeros((g_orth.shape[0], g_orth.shape[0]), dtype=np.float64),
        g_orth,
        log=False,
        # Conventional QC MPOs use a middle-site complementary-operator
        # transform.  Block2 0.5.3's generic stacked environment does not
        # transform the secondary MPO factor, which drops four-distinct-site
        # channels.  FastBipartite has no such schemer and is exact here.
        mpo_mode="fast-bipartite",
    )
    transition = get_sz_stacked_transition_1pdm(
        driver, total, vee_mpo, iprint=iprint
    ) / norm
    # See tnvb_contracted_gradient_derivation.tex.  With Block2's array
    # convention, X[d,c] = 2*T[c,d] - 2*L[d,c].
    contracted, lower = assemble_contracted_d3g(g_orth, dm2, transition)
    # Reuse the already normalized 2-RDM instead of launching a second scalar
    # MPO sweep solely for the trace invariant.
    vee_expectation = 0.5 * float(
        np.einsum("ijkl,iklj", g_orth, dm2, optimize=True)
    )
    trace_error = abs(float(np.trace(transition)) - driver.n_elec * vee_expectation)
    return contracted, transition, lower, vee_expectation, trace_error


def get_sz_explicit_product_transition_1pdm(
    driver,
    state,
    norm,
    g_orth,
    cutoff=0.0,
    iprint=0,
):
    """Compute ``<state|E[c,d] V|state>`` from explicit operator products.

    Block2 0.5.3's ``stacked_mpo`` contraction does not preserve every
    fermionic product channel when a free-index NPDM MPO is stacked with the
    two-electron Hamiltonian.  This correctness-first implementation instead
    gives the literal six-operator products to ``ExprBuilder``.  Its spin and
    permutation machinery then builds an ordinary scalar MPO for each
    ``(c,d)``.  No rank-three NPDM scheme or K**6 output tensor is created.

    This reference routine is intentionally serial.  The production wrapper
    distributes the independent ``(c,d)`` scalar contractions across
    process-private Block2 drivers.
    """
    if SymmetryTypes.SZ not in driver.symm_type:
        raise RuntimeError("explicit-product D3G measurement requires SZ")
    if driver.mpi is not None:
        raise NotImplementedError(
            "explicit-product D3G measurement is currently shared-memory only"
        )
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("explicit-product D3G requires a positive norm")
    if cutoff < 0.0:
        raise RuntimeError("explicit-product integral cutoff must be nonnegative")
    norb = driver.n_sites
    if g_orth.shape != (norb,) * 4:
        raise RuntimeError("explicit-product D3G requires square rank-four ERIs")

    # Match DMRGDriver.get_qc_mpo's SZ Hamiltonian convention.  For an input
    # integral g[p,q,r,s], the aa/ab/bb strings carry sites (p,r,s,q), while
    # the ba string carries (r,p,q,s); every spin block has the Hamiltonian's
    # factor 1/2.  Prefixing alpha ``cd`` or beta ``CD`` forms E[c,d] V in the
    # required operator order before ExprBuilder performs site ordering.
    standard_indices = []
    reversed_spin_indices = []
    values = []
    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    value = float(g_orth[p, q, r, s])
                    if abs(value) <= cutoff:
                        continue
                    standard_indices.append((p, r, s, q))
                    reversed_spin_indices.append((r, p, q, s))
                    values.append(0.5 * value)

    transition = np.empty((norb, norb), dtype=np.float64)
    expressions = (
        ("cdccdd", standard_indices),
        ("cdcCDd", standard_indices),
        ("cdCcdD", reversed_spin_indices),
        ("cdCCDD", standard_indices),
        ("CDccdd", standard_indices),
        ("CDcCDd", standard_indices),
        ("CDCcdD", reversed_spin_indices),
        ("CDCCDD", standard_indices),
    )
    for c in range(norb):
        for d in range(norb):
            builder = driver.expr_builder()
            for expression, integral_indices in expressions:
                indices = []
                for suffix in integral_indices:
                    indices.extend((c, d, *suffix))
                builder.add_term(expression, indices, values)
            product_mpo = driver.get_mpo(
                builder.finalize(),
                cutoff=max(cutoff, 1.0e-16),
                add_ident=False,
                iprint=max(0, iprint - 1),
            )
            transition[c, d] = (
                float(driver.expectation(state, product_mpo, state)) / norm
            )
            if iprint >= 2:
                print(
                    "[TNVB-BLOCK2-D3G-EXPLICIT] "
                    f"element={c * norb + d + 1}/{norb * norb} "
                    f"c={c + 1} d={d + 1} value={transition[c, d]:.17e}",
                    flush=True,
                )
    return transition


def _su2_explicit_product_integral_terms(g_orth, cutoff):
    """Prepare the common scalar-MPO expression and its ERI suffix terms."""
    norb = g_orth.shape[0]
    integral_indices = []
    values = []
    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    value = float(g_orth[p, q, r, s])
                    if abs(value) <= cutoff:
                        continue
                    integral_indices.append((p, r, s, q))
                    values.append(math.sqrt(2.0) * value)
    one_body = "(C+D)0"
    two_body = "((C+(C+D)0)1+D)0"
    return f"({one_body}+{two_body})0", integral_indices, values


def _contract_su2_explicit_product_element(
    driver,
    state,
    norm,
    expression,
    integral_indices,
    values,
    cutoff,
    c,
    d,
):
    """Build and contract one literal SU(2) ``E[c,d] V`` scalar MPO."""
    indices = []
    for suffix in integral_indices:
        indices.extend((c, d, *suffix))
    builder = driver.expr_builder()
    builder.add_term(expression, indices, values)
    build_started = time.perf_counter()
    product_mpo = driver.get_mpo(
        builder.finalize(),
        cutoff=max(cutoff, 1.0e-16),
        add_ident=False,
        iprint=0,
    )
    build_seconds = time.perf_counter() - build_started
    expectation_started = time.perf_counter()
    value = float(driver.expectation(state, product_mpo, state)) / norm
    expectation_seconds = time.perf_counter() - expectation_started
    return value, build_seconds, expectation_seconds


def get_su2_explicit_product_transition_1pdm(
    driver,
    state,
    norm,
    g_orth,
    cutoff=0.0,
    iprint=0,
    commutator=None,
):
    """SU(2)-adapted scalar-MPO evaluation of ``<E[c,d] V>``.

    Both ``E[c,d]`` and the two-electron Hamiltonian are rank-zero spin
    tensors.  Coupling their literal product to rank zero lets Block2's
    ``SpinPermScheme`` perform all site permutations and recouplings while
    preserving the operator order.  Relative to the SZ reference this uses
    one expression instead of eight spin blocks and avoids expanding the MPS
    out of its reduced SU(2) representation.
    """
    if SymmetryTypes.SU2 not in driver.symm_type:
        raise RuntimeError("SU2 explicit-product D3G measurement requires SU2")
    if driver.mpi is not None:
        raise NotImplementedError(
            "SU2 explicit-product D3G measurement is shared-memory only"
        )
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("SU2 explicit-product D3G requires a positive norm")
    if cutoff < 0.0:
        raise RuntimeError("SU2 explicit-product cutoff must be nonnegative")
    norb = driver.n_sites
    if g_orth.shape != (norb,) * 4:
        raise RuntimeError("SU2 explicit-product D3G requires square ERIs")
    if commutator is not None and commutator.shape != (norb, norb):
        raise RuntimeError("SU2 explicit-product commutator has wrong shape")

    expression, integral_indices, values = _su2_explicit_product_integral_terms(
        g_orth, cutoff
    )
    transition = np.empty((norb, norb), dtype=np.float64)
    pairs = (
        [(c, d) for c in range(norb) for d in range(c + 1)]
        if commutator is not None
        else [(c, d) for c in range(norb) for d in range(norb)]
    )
    started = time.perf_counter()
    print(
        "[TNVB-BLOCK2-D3G-SU2-EXPLICIT] start "
        f"elements={len(pairs)} integral_terms={len(values)} "
        f"cutoff={cutoff:.3e}",
        flush=True,
    )
    for element, (c, d) in enumerate(pairs, 1):
        (
            transition[c, d],
            build_seconds,
            expectation_seconds,
        ) = _contract_su2_explicit_product_element(
            driver,
            state,
            norm,
            expression,
            integral_indices,
            values,
            cutoff,
            c,
            d,
        )
        if iprint >= 1:
            print(
                "[TNVB-BLOCK2-D3G-SU2-EXPLICIT] "
                f"element={element}/{len(pairs)} "
                f"c={c + 1} d={d + 1} value={transition[c, d]:.17e} "
                f"build_seconds={build_seconds:.3f} "
                f"expectation_seconds={expectation_seconds:.3f} "
                f"wall_seconds={time.perf_counter() - started:.3f}",
                flush=True,
            )
    if commutator is not None:
        for c in range(norb):
            for d in range(c + 1, norb):
                transition[c, d] = transition[d, c] + commutator[c, d]
    return transition


def contracted_d3g_from_su2_explicit_product(
    driver,
    total,
    norm,
    g_orth,
    dm2,
    cutoff=0.0,
    iprint=0,
    source_scratch=None,
    scratch_parent=None,
):
    """Return the exact contracted 3-RDM through scalar ``E[c,d] V`` MPOs."""
    started = time.perf_counter()
    allocated, _ = allocated_cpu_count()
    configured_workers = positive_environment_integer(
        "XMVB_BLOCK2_D3G_WORKERS"
    )
    # Scalar MPO contractions scale poorly with OpenMP threads, whereas
    # different (c,d) elements are independent.  C14/D=192 measurements on
    # the same 16-core node reduced this stage from 501 s (4x4) through
    # 223 s (8x2) to 47.7 s (16x1).  Prefer one Block2 thread per process;
    # the memory model below remains the hard cap on process replication.
    default_workers = min(32, max(1, allocated))
    scalar_mpos = driver.n_sites * (driver.n_sites + 1) // 2
    worker_count = min(
        scalar_mpos,
        allocated,
        configured_workers or default_workers,
    )
    memory_bytes, _ = memory_allocation_bytes()
    if memory_bytes is not None:
        parent_reserve = 2 * rdm_stack_memory_bytes(
            driver.n_sites, total.info.bond_dim
        )
        per_worker = 4 * stack_memory_bytes(
            driver.n_sites, total.info.bond_dim
        )
        usable = max(0, int(0.80 * memory_bytes) - parent_reserve)
        worker_count = min(worker_count, max(1, usable // per_worker))
    commutator = two_body_transition_commutator(g_orth, dm2)
    if source_scratch and scratch_parent and worker_count > 1:
        transition = contract_parallel_su2_explicit_transition(
            total,
            source_scratch,
            scratch_parent,
            driver.n_sites,
            driver.n_elec,
            total.info.bond_dim,
            norm,
            g_orth,
            cutoff,
            worker_count,
            max(1, allocated // worker_count)
            if configured_workers is not None
            else 1,
            commutator,
        )
    else:
        worker_count = 1
        transition = get_su2_explicit_product_transition_1pdm(
            driver,
            total,
            norm,
            g_orth,
            cutoff=cutoff,
            iprint=iprint,
            commutator=commutator,
        )
    contracted, lower = assemble_contracted_d3g(g_orth, dm2, transition)
    vee_expectation = 0.5 * float(
        np.einsum("ijkl,iklj", g_orth, dm2, optimize=True)
    )
    trace_error = abs(
        float(np.trace(transition)) - driver.n_elec * vee_expectation
    )
    print(
        "[TNVB-BLOCK2-D3G-EXPLICIT] complete "
        f"scalar_mpos={scalar_mpos} workers={worker_count} "
        f"vee_expectation={vee_expectation:.12e} "
        f"transition_trace={np.trace(transition):.12e} "
        f"trace_error={trace_error:.3e} "
        f"seconds={time.perf_counter() - started:.3f}",
        flush=True,
    )
    return contracted, transition, lower, vee_expectation, trace_error


def contracted_d3g_from_su2_hole_states(
    driver,
    total,
    norm,
    sum_bond,
    g_orth,
    dm1,
    dm2,
    allow_overlap_failure: bool = False,
):
    """Evaluate the dense-ERI contracted 3-RDM through ``a_d|Psi>`` MPSs.

    For real orbitals and the permutationally symmetric active-space ERIs used
    by XMVB, the requested contraction is exactly

    ``D3G[d,c] = 2 * <a_c Psi | V_2 | a_d Psi> / <Psi|Psi>``.

    A one-site annihilation MPO has unit virtual bond dimension, so every
    hole state is representable at the original total-state bond dimension;
    no Hamiltonian-MPS product and no rank-six NPDM is formed.  The remaining
    work is the lower triangle of a Hamiltonian matrix between ``K`` SU(2)
    doublet MPSs.  The present reference implementation evaluates these
    matrix elements independently.  It removes rank-six NPDM storage, but
    its work is ``O(K**2 * C_V(K, D))`` for a two-electron-MPO matrix-element
    cost ``C_V``.  Reaching the target contracted scaling requires a future
    batched/shared-environment implementation.
    """
    if SymmetryTypes.SU2 not in driver.symm_type:
        raise RuntimeError("STATE contracted D3G requires an SU(2) driver")
    if driver.mpi is not None:
        raise NotImplementedError(
            "STATE contracted D3G is currently shared-memory only"
        )
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("STATE contracted D3G requires a positive norm")
    norb = driver.n_sites
    if g_orth.shape != (norb,) * 4:
        raise RuntimeError("STATE contracted D3G requires square active ERIs")
    if dm1.shape != (norb, norb) or dm2.shape != (norb,) * 4:
        raise RuntimeError("STATE contracted D3G received incompatible RDMs")

    hole_builder = os.environ.get(
        "XMVB_BLOCK2_STATE_HOLE_BUILDER", "FIT"
    ).strip().upper()
    if hole_builder == "EXACT":
        # The pyblock2 MPO@MPS prototype can construct and convert all local
        # hole tensors, but the resulting non-singlet embedding is not yet a
        # valid input to Block2's expectation environment.  Reject it at the
        # Python boundary instead of exposing a compiled-code segmentation
        # fault.  FIT is the only validated STATE hole-state builder.
        raise RuntimeError(
            "XMVB_BLOCK2_STATE_HOLE_BUILDER=EXACT is an unavailable "
            "interface prototype; use FIT"
        )
    sweeps = int(os.environ.get("XMVB_BLOCK2_STATE_SWEEPS", "2"))
    tolerance = float(os.environ.get("XMVB_BLOCK2_STATE_TOL", "1e-12"))
    cutoff = float(os.environ.get("XMVB_BLOCK2_STATE_CUTOFF", "1e-14"))
    linear_threshold = float(
        os.environ.get("XMVB_BLOCK2_STATE_LINEAR_THRD", "1e-14")
    )
    right_weight = float(
        os.environ.get("XMVB_BLOCK2_STATE_RIGHT_WEIGHT", "0")
    )
    iprint = int(os.environ.get("XMVB_BLOCK2_STATE_IPRINT", "0"))
    overlap_tolerance = float(
        os.environ.get("XMVB_BLOCK2_STATE_OVERLAP_TOL", "1e-9")
    )
    hole_bond = int(
        os.environ.get("XMVB_BLOCK2_STATE_BOND", str(sum_bond))
    )
    hole_dot = int(os.environ.get("XMVB_BLOCK2_STATE_DOT", "2"))
    if (
        hole_builder != "FIT"
        or sweeps < 1
        or tolerance <= 0.0
        or cutoff < 0.0
        or linear_threshold <= 0.0
        or not 0.0 <= right_weight <= 1.0
        or iprint < 0
        or overlap_tolerance <= 0.0
        or hole_bond < sum_bond
        or hole_dot not in (1, 2)
    ):
        raise RuntimeError("invalid STATE contracted-D3G numerical setting")

    started = time.perf_counter()
    hole_started = started
    # SU(2) non-singlet states use Block2's singlet embedding.  The physical
    # N-1 doublet therefore has the singlet target of ``total`` and a
    # one-electron, spin-doublet left vacuum.
    hole_left_vacuum = driver.bw.SX(1, 1, 0)
    holes = []
    fit_norms = []
    if hole_builder == "EXACT":
        # A one-site annihilation MPO has virtual bond one, so MPO@MPS does
        # not increase the MPS bond.  pyblock2's block-sparse algebra can
        # therefore form every hole exactly and return it to Block2 without
        # the variational multiply residual that limits the FIT path.
        exact_source = driver.copy_mps(
            total, tag="TNVB-D3G-EXACT-SOURCE"
        )
        driver.adjust_mps(exact_source, dot=1)
        driver.align_mps_center(exact_source, 0)
        py_total = MPSTools.from_block2(exact_source)
        # ``MPSTools.from_block2`` keeps Block2's singlet-embedding right
        # boundary as a unit third axis on the last tensor.  It carries no
        # numerical degree of freedom, while pyblock2's MPO@MPS and
        # ``to_block2`` interfaces require the conventional rank-two last
        # MPS tensor.  Remove that unit boundary before applying the local
        # annihilation MPO; the remaining left/physical labels determine the
        # physical N-1 target unambiguously.
        if py_total.tensors[-1] is not None:
            last_blocks = []
            for block in py_total.tensors[-1].blocks:
                if block.rank == 3:
                    if block.reduced.shape[-1] != 1:
                        raise RuntimeError(
                            "non-unit exact STATE singlet-embedding boundary"
                        )
                    last_blocks.append(
                        SubTensor(
                            q_labels=block.q_labels[:-1],
                            reduced=block.reduced[..., 0].copy(),
                        )
                    )
                elif block.rank == 2:
                    last_blocks.append(block.copy())
                else:
                    raise RuntimeError(
                        "unsupported exact STATE source right-boundary "
                        f"rank {block.rank}"
                    )
            py_total.tensors[-1] = Tensor(blocks=last_blocks)
        basis_template = []
        for site_info in total.info.basis:
            basis_template.append(
                Counter(
                    {
                        site_info.quanta[index]: int(
                            site_info.n_states[index]
                        )
                        for index in range(site_info.n)
                    }
                )
            )
        for orbital in range(norb):
            builder = driver.expr_builder()
            builder.add_term("D", [orbital], 2**0.5)
            annihilation_mpo = driver.get_mpo(
                builder.finalize(),
                iprint=0,
                left_vacuum=hole_left_vacuum,
                add_ident=False,
            )
            py_annihilation = MPOTools.from_block2(annihilation_mpo)
            try:
                # This is pyblock2's MPO@MPS contraction written explicitly
                # so that any unsupported tensor rank produces a diagnostic
                # containing the site number and both input layouts.
                py_tensors = [None] * norb
                for site in range(norb):
                    mpo_tensor = py_annihilation.tensors[site]
                    mps_tensor = py_total.tensors[site]
                    if mpo_tensor is None or mps_tensor is None:
                        raise RuntimeError(
                            "exact STATE requires a one-site canonical MPS"
                        )
                    if site == 0:
                        py_tensors[site] = Tensor.contract(
                            mpo_tensor, mps_tensor, [1], [0]
                        )
                    elif site == norb - 1:
                        if mps_tensor.rank != 2:
                            raise RuntimeError(
                                "unsupported exact STATE right-boundary "
                                f"rank {mps_tensor.rank}"
                            )
                        py_tensors[site] = Tensor.contract(
                            mpo_tensor,
                            mps_tensor,
                            [2],
                            [1],
                            1,
                            0,
                            out_trans=(0, 2, 1),
                        )
                    else:
                        if mps_tensor.rank != 3:
                            raise RuntimeError(
                                "unsupported exact STATE internal tensor "
                                f"rank {mps_tensor.rank} at site {site + 1}"
                            )
                        py_tensors[site] = Tensor.contract(
                            mpo_tensor,
                            mps_tensor,
                            [2],
                            [1],
                            1,
                            0,
                            out_trans=(0, 3, 1, 2, 4),
                        )
                py_hole = PyMPS(tensors=py_tensors)
                py_hole.merge_virtual_dims()
                # Validate the pyblock2 quantum-number layout before handing
                # it to the compiled Block2 converter.  In particular,
                # ``to_block2`` assumes one physical label per site and one
                # common target across all last-site blocks; violating either
                # assumption can otherwise fail below the Python boundary.
                # ``exact_source`` was exported at center zero, so all right
                # tensors already carry Block2's target-complement quantum
                # labels.  Numerical right-canonicalization preserves those
                # labels and is therefore safe to map back at center zero.
                py_hole.canonicalize(0)
                bad_physical = []
                inferred_targets = set()
                for site, tensor in enumerate(py_hole.tensors):
                    expected_rank = 2 if site in (0, norb - 1) else 3
                    for block in tensor.blocks:
                        if block.rank != expected_rank:
                            raise RuntimeError(
                                "invalid exact STATE post-merge rank: "
                                f"site={site + 1}, rank={block.rank}, "
                                f"expected={expected_rank}"
                            )
                        physical_index = 0 if site == 0 else 1
                        physical_label = block.q_labels[physical_index]
                        if physical_label not in basis_template[site]:
                            bad_physical.append(
                                (site + 1, str(physical_label))
                            )
                        if site == norb - 1:
                            inferred_targets.add(
                                str(
                                    (
                                        block.q_labels[0]
                                        + block.q_labels[1]
                                    )[0]
                                )
                            )
                if bad_physical:
                    raise RuntimeError(
                        "invalid exact STATE physical labels: "
                        f"{bad_physical[:8]}"
                    )
                if len(inferred_targets) != 1:
                    raise RuntimeError(
                        "inconsistent exact STATE targets: "
                        f"{sorted(inferred_targets)}"
                    )
            except Exception as error:
                total_ranks = [
                    None if tensor is None else tensor.rank
                    for tensor in py_total.tensors
                ]
                mpo_ranks = [
                    None if tensor is None else tensor.rank
                    for tensor in py_annihilation.tensors
                ]
                raise RuntimeError(
                    "exact STATE MPO@MPS failed: "
                    f"orbital={orbital + 1}, total_ranks={total_ranks}, "
                    f"mpo_ranks={mpo_ranks}, cause={error}"
                ) from error
            print(
                "[TNVB-STATE-EXACT] "
                f"hole={orbital + 1}/{norb} converting "
                f"target={next(iter(inferred_targets))}",
                file=sys.stderr,
                flush=True,
            )
            embedded_hole = MPSTools.to_block2(
                py_hole,
                # MPSTools.to_block2 replaces the Counter entries in this
                # list by compiled StateInfo objects, so every orbital needs
                # a fresh copy of the immutable basis template.
                [Counter(site_basis) for site_basis in basis_template],
                center=0,
                tag=f"TNVB-D3G-HOLE-{orbital}",
                left_vacuum=hole_left_vacuum,
            )
            hole = driver.mps_change_singlet_embedding(
                embedded_hole,
                tag=f"TNVB-D3G-HOLE-NSE-{orbital}",
                forward=False,
            )
            print(
                "[TNVB-STATE-EXACT] "
                f"hole={orbital + 1}/{norb} converted "
                f"bond={hole.info.bond_dim} target={hole.info.target}",
                file=sys.stderr,
                flush=True,
            )
            if hole.info.bond_dim > hole_bond:
                raise RuntimeError(
                    "exact STATE hole bond exceeds configured allowance: "
                    f"actual={hole.info.bond_dim}, limit={hole_bond}"
                )
            fit_norms.append(float(py_hole.norm()))
            holes.append(hole)
            del py_hole, py_annihilation, annihilation_mpo
        del py_total, exact_source
    else:
        for orbital in range(norb):
            annihilation_mpo = driver.get_site_mpo("D", orbital, iprint=0)
            hole = driver.get_random_mps(
                tag=f"TNVB-D3G-HOLE-{orbital}",
                bond_dim=hole_bond,
                target=total.info.target,
                left_vacuum=hole_left_vacuum,
                dot=hole_dot,
                full_fci=True,
            )
            fit_norms.append(
                float(
                    driver.multiply(
                        hole,
                        annihilation_mpo,
                        total,
                        n_sweeps=sweeps,
                        tol=tolerance,
                        bra_bond_dims=[hole_bond],
                        cutoff=cutoff,
                        thrds=[linear_threshold],
                        right_weight=right_weight,
                        iprint=iprint,
                    )
                )
            )
            holes.append(hole)
    hole_seconds = time.perf_counter() - hole_started

    contracted = np.empty((norb, norb), dtype=np.float64)
    pair_count = norb * (norb + 1) // 2
    state_workers, state_threads = state_d3g_parallel_layout(
        pair_count, norb, hole_bond
    )
    overlap_error = 0.0
    if state_workers > 1:
        results, matrix_seconds = contract_parallel_state_d3g(
            holes,
            driver.scratch,
            os.path.dirname(os.path.abspath(driver.scratch)),
            norb,
            driver.n_elec,
            hole_bond,
            g_orth,
            state_workers,
            state_threads,
        )
        overlap_seconds = sum(item[4] for item in results)
        for _, c, d, overlap, _, value, _ in results:
            overlap /= norm
            overlap_error = max(overlap_error, abs(overlap - dm1[c, d]))
            value = 2.0 * value / norm
            contracted[c, d] = value
            contracted[d, c] = value
    else:
        identity_mpo = driver.get_identity_mpo()
        overlap_started = time.perf_counter()
        for c in range(norb):
            for d in range(c + 1):
                overlap = float(
                    driver.expectation(holes[c], identity_mpo, holes[d])
                ) / norm
                overlap_error = max(
                    overlap_error, abs(overlap - dm1[c, d])
                )
        overlap_seconds = time.perf_counter() - overlap_started

        vee_mpo = get_hamiltonian_mpo(
            driver,
            np.zeros((norb, norb), dtype=np.float64),
            g_orth,
            log=False,
            mpo_mode="no-trans-conventional",
        )
        matrix_started = time.perf_counter()
        for c in range(norb):
            for d in range(c + 1):
                value = 2.0 * float(
                    driver.expectation(holes[c], vee_mpo, holes[d])
                ) / norm
                contracted[c, d] = value
                contracted[d, c] = value
        matrix_seconds = time.perf_counter() - matrix_started
    if overlap_error > overlap_tolerance and not allow_overlap_failure:
        raise RuntimeError(
            "STATE hole-state construction changed the 1-RDM: "
            f"error={overlap_error:.3e}, limit={overlap_tolerance:.3e}"
        )

    vee_expectation = 0.5 * float(
        np.einsum("ijkl,iklj", g_orth, dm2, optimize=True)
    )
    trace_reference = (driver.n_elec - 2) * float(
        np.einsum("faeb,abef", g_orth, dm2, optimize=True)
    )
    trace_error = abs(float(np.trace(contracted)) - trace_reference)
    log_summary(
        "[TNVB-D3G-STATE] complete "
        f"holes={norb} pairs={pair_count} bond={hole_bond} dot={hole_dot} "
        f"builder={hole_builder.lower()} "
        f"right_weight={right_weight:.3g} "
        f"workers={state_workers}x{state_threads} "
        f"fit_norm=[{min(fit_norms):.6e},{max(fit_norms):.6e}] "
        f"overlap_error={overlap_error:.3e} trace_error={trace_error:.3e} "
        f"hole_seconds={hole_seconds:.3f} "
        f"overlap_seconds={overlap_seconds:.3f} "
        f"matrix_seconds={matrix_seconds:.3f} "
        f"seconds={time.perf_counter() - started:.3f}"
    )
    return (
        contracted,
        None,
        None,
        vee_expectation,
        trace_error,
        overlap_error,
    )


def contracted_d3g_from_su2_sigma_mps(
    driver,
    total,
    norm: float,
    sum_bond: int,
    g_orth,
    dm2,
):
    """Fit ``V|Psi>`` once and obtain the complete SU(2) contracted 3PDM.

    The fitted sigma MPS replaces K(K+1)/2 independent scalar-product MPO
    expectations by one Hamiltonian application and one transition-1PDM
    sweep.  The exact 2PDM commutator is imposed after the fit, so compression
    can affect only the symmetric part of ``<Psi|E_cd V|Psi>``.
    """
    if SymmetryTypes.SU2 not in driver.symm_type:
        raise RuntimeError("sigma-MPS contracted D3G requires an SU(2) driver")
    if driver.mpi is not None:
        raise NotImplementedError(
            "sigma-MPS contracted D3G is currently shared-memory only"
        )
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("sigma-MPS contracted D3G requires a positive norm")
    started = time.perf_counter()
    sweeps = int(os.environ.get("XMVB_BLOCK2_SIGMA_SWEEPS", "4"))
    tolerance = float(os.environ.get("XMVB_BLOCK2_SIGMA_TOL", "1e-11"))
    cutoff = float(os.environ.get("XMVB_BLOCK2_SIGMA_CUTOFF", "1e-14"))
    iprint = int(os.environ.get("XMVB_BLOCK2_SIGMA_IPRINT", "0"))
    if sweeps < 1 or tolerance <= 0.0 or cutoff < 0.0 or iprint < 0:
        raise RuntimeError("invalid sigma-MPS sweep/tolerance/cutoff setting")
    sigma_bond = int(
        os.environ.get("XMVB_BLOCK2_SIGMA_BOND", str(2 * sum_bond))
    )
    if sigma_bond < 2:
        raise RuntimeError("XMVB_BLOCK2_SIGMA_BOND must be at least 2")
    sigma_mpo_mode = os.environ.get(
        "XMVB_BLOCK2_SIGMA_H_MPO", "no-trans-conventional"
    )
    vee_mpo = get_hamiltonian_mpo(
        driver,
        np.zeros((g_orth.shape[0], g_orth.shape[0]), dtype=np.float64),
        g_orth,
        log=False,
        mpo_mode=sigma_mpo_mode,
    )
    driver.bw.b.Random.rand_seed(20260813)
    sigma_state = driver.get_random_mps(
        tag="TNVB-D3G-SIGMA-PSI",
        bond_dim=sigma_bond,
        target=total.info.target,
        dot=2,
        full_fci=False,
    )
    apply_started = time.perf_counter()
    fit_norm = float(
        driver.multiply(
            sigma_state,
            vee_mpo,
            total,
            n_sweeps=sweeps,
            tol=tolerance,
            bra_bond_dims=[sigma_bond],
            cutoff=cutoff,
            iprint=iprint,
        )
    )
    apply_seconds = time.perf_counter() - apply_started
    identity_mpo = driver.get_identity_mpo()
    vee_expectation = float(driver.expectation(total, vee_mpo, total)) / norm
    fitted_overlap = (
        float(driver.expectation(total, identity_mpo, sigma_state)) / norm
    )
    fitted_norm = (
        float(driver.expectation(sigma_state, identity_mpo, sigma_state)) / norm
    )
    transition_started = time.perf_counter()
    raw_transition = np.asarray(
        driver.get_trans_1pdm(total, sigma_state, iprint=iprint),
        dtype=np.float64,
    ) / norm
    transition_seconds = time.perf_counter() - transition_started

    # The fitted sigma vector is needed only for the symmetric part.  The
    # antisymmetric part is the exact two-body commutator and costs O(K**4).
    commutator = two_body_transition_commutator(g_orth, dm2)
    transition = 0.5 * (
        raw_transition + raw_transition.T + commutator
    )
    commutator_error = float(
        np.max(
            np.abs(
                transition - transition.T - commutator
            )
        )
    )
    contracted, lower = assemble_contracted_d3g(g_orth, dm2, transition)
    trace_reference = driver.n_elec * vee_expectation
    trace_error = abs(float(np.trace(transition)) - trace_reference)
    overlap_error = abs(fitted_overlap - vee_expectation)
    print(
        "[TNVB-BLOCK2-D3G-SIGMA] complete "
        f"bond={sigma_bond} sweeps={sweeps} mpo={sigma_mpo_mode} "
        f"fit_norm={fit_norm:.12e} fitted_norm={fitted_norm:.12e} "
        f"vee_expectation={vee_expectation:.12e} "
        f"fitted_overlap={fitted_overlap:.12e} "
        f"overlap_error={overlap_error:.3e} "
        f"transition_trace={np.trace(transition):.12e} "
        f"trace_error={trace_error:.3e} "
        f"commutator_error={commutator_error:.3e} "
        f"apply_seconds={apply_seconds:.3f} "
        f"transition1_seconds={transition_seconds:.3f} "
        f"seconds={time.perf_counter() - started:.3f}",
        flush=True,
    )
    diagnostics = {
        "fit_norm": fit_norm,
        "fitted_norm": fitted_norm,
        "fitted_overlap": fitted_overlap,
        "overlap_error": overlap_error,
        "commutator_error": commutator_error,
        "apply_seconds": apply_seconds,
        "transition1_seconds": transition_seconds,
    }
    return (
        contracted,
        transition,
        lower,
        vee_expectation,
        trace_error,
        diagnostics,
    )


def validate_direct_contracted_d3g(
    driver,
    total,
    norm: float,
    nae: int,
    sum_bond: int,
    g_orth,
    dm2,
    dm3=None,
    contracted_orth=None,
):
    """Validate an integral-contracted 3-RDM without constructing an NPDM.

    Apply the two-electron Hamiltonian once, then compute the transition 1PDM
    between V|Psi> and |Psi>.  Normal ordering V E_cd gives the desired 3PDM
    contraction plus two terms that are available from the ordinary 2PDM.
    This validation-only path compares the result with the current full-3PDM
    reference.  It is gated by XMVB_BLOCK2_VALIDATE_DIRECT_D3G so normal
    calculations are unchanged.
    """
    started = time.perf_counter()
    (
        direct,
        transition1,
        lower,
        vee_expectation,
        _,
        sigma_diagnostics,
    ) = contracted_d3g_from_su2_sigma_mps(
        driver, total, norm, sum_bond, g_orth, dm2
    )
    debug_path = os.environ.get("XMVB_BLOCK2_DIRECT_D3G_DEBUG_NPZ", "").strip()
    if debug_path and dm3 is not None and contracted_orth is not None:
        np.savez(
            debug_path,
            g_orth=g_orth,
            dm2=dm2,
            dm3=dm3,
            transition1=transition1,
            contracted_orth=contracted_orth,
            lower=lower,
            direct=direct,
        )
    common = (
        f"bond={int(os.environ.get('XMVB_BLOCK2_SIGMA_BOND', 2 * sum_bond))} "
        f"fit_norm={sigma_diagnostics['fit_norm']:.12e} "
        f"vee_expectation={vee_expectation:.12e} "
        f"fitted_overlap={sigma_diagnostics['fitted_overlap']:.12e} "
        f"fitted_norm={sigma_diagnostics['fitted_norm']:.12e} "
        f"transition_trace={np.trace(transition1):.12e} "
        f"apply_seconds={sigma_diagnostics['apply_seconds']:.3f} "
        f"transition1_seconds={sigma_diagnostics['transition1_seconds']:.3f} "
        f"seconds={time.perf_counter() - started:.3f}"
    )
    if contracted_orth is None:
        print(f"[TNVB-BLOCK2-D3G-DIRECT] complete {common}", flush=True)
        return direct

    transition_from_rdms = (lower + 0.5 * contracted_orth).T
    transition_reference_error = float(
        np.max(np.abs(transition1 - transition_from_rdms))
    )
    transition_reference_transpose_error = float(
        np.max(np.abs(transition1 - transition_from_rdms.T))
    )
    direct_error = float(np.max(np.abs(direct - contracted_orth)))
    direct_transpose_error = float(np.max(np.abs(direct - contracted_orth.T)))
    reference_scale = float(
        np.vdot(direct, contracted_orth) / max(np.vdot(direct, direct), 1.0e-30)
    )
    print(
        "[TNVB-BLOCK2-D3G-VALIDATE] complete "
        f"direct_error={direct_error:.3e} "
        f"direct_transpose_error={direct_transpose_error:.3e} "
        f"reference_scale={reference_scale:.12e} "
        f"transition_reference_error={transition_reference_error:.3e} "
        f"transition_reference_transpose_error="
        f"{transition_reference_transpose_error:.3e} "
        f"{common}",
        flush=True,
    )
    validation_tolerance = float(
        os.environ.get("XMVB_BLOCK2_DIRECT_D3G_CHECK_TOL", "1e-8")
    )
    if max(direct_error, transition_reference_error) > validation_tolerance:
        raise RuntimeError(
            "direct contracted D3G validation failed: "
            f"error={max(direct_error, transition_reference_error):.3e}, "
            f"limit={validation_tolerance:.3e}"
        )
    return direct


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


def build_sz_states(driver, ntstr, nstr, nao, nae, nel, tnbond, rmat):
    """Retain the validated pre-SU(2) construction as an explicit regression path."""
    state_builder = os.environ.get("XMVB_BLOCK2_STATE_BUILDER", "pair").lower()
    if state_builder not in ("pair", "evolution"):
        raise RuntimeError("XMVB_BLOCK2_STATE_BUILDER must be pair or evolution")

    rotation_norm = np.linalg.norm(rmat - np.eye(nao))
    rotation_mpo = None
    rotation_algorithm = os.environ.get("XMVB_BLOCK2_ROT_ALGORITHM", "rk4").lower()
    if rotation_algorithm not in ("rk4", "tdvp"):
        raise RuntimeError("XMVB_BLOCK2_ROT_ALGORITHM must be rk4 or tdvp")
    default_steps = "20" if rotation_algorithm == "rk4" else "1"
    rotation_steps = int(os.environ.get("XMVB_BLOCK2_ROT_STEPS", default_steps))
    if state_builder == "evolution" and rotation_norm > 1.0e-13:
        kappa = scipy.linalg.logm(rmat)
        if np.max(np.abs(np.imag(kappa))) > 1.0e-11:
            raise RuntimeError("active-overlap Cholesky factor has a complex logarithm")
        kappa = np.asarray(np.real(kappa), dtype=np.float64)
        rotation_builder = driver.expr_builder()
        rotation_builder.add_sum_term("cd", -kappa, cutoff=1.0e-16)
        rotation_builder.add_sum_term("CD", -kappa, cutoff=1.0e-16)
        rotation_builder.add_term("", [], 1.0e-14)
        rotation_mpo = driver.get_mpo(
            rotation_builder.finalize(), cutoff=1.0e-16, iprint=0
        )

    creators = build_creation_mpos(driver, rmat) if state_builder == "pair" else None
    pair_cache = {}
    states = []
    carrier_mode = os.environ.get("XMVB_BLOCK2_CARRIER", "paired").lower()
    if carrier_mode not in ("paired", "determinants"):
        raise RuntimeError("XMVB_BLOCK2_CARRIER must be paired or determinants")
    max_expansion = int(os.environ.get("XMVB_BLOCK2_MAX_DETS", "4096"))
    for istr in range(nstr):
        if state_builder == "pair":
            _, pairs, _ = paired_structure_layout(ntstr[:, istr], nao, nae, nel)
            max_crossing = max(
                (
                    sum(left <= cut < right for left, right in pairs)
                    for cut in range(max(0, nao - 1))
                ),
                default=0,
            )
            carrier_bond = 1 << max_crossing
            carrier_norm2 = float(1 << len(pairs))
            physical, physical_norm2, max_discarded_norm = build_direct_physical_mps(
                driver,
                ntstr[:, istr],
                nao,
                nae,
                nel,
                tnbond,
                creators,
                pair_cache,
                tag=f"VB-PHYSICAL-{istr + 1}",
            )
            carrier_detail = (
                f"symmetry=SZ builder=pair pairs={len(pairs)} "
                f"crossing={max_crossing} raw_bond={carrier_bond} "
                f"carrier_norm2={carrier_norm2:.12e} "
                f"physical_norm2={physical_norm2:.12e} "
                f"max_discarded_norm={max_discarded_norm:.3e}"
            )
        elif carrier_mode == "paired":
            carrier, pair_count, max_crossing, carrier_bond, carrier_norm2 = (
                build_direct_paired_mps(
                    driver,
                    ntstr[:, istr],
                    nao,
                    nae,
                    nel,
                    tag=f"VB-CARRIER-{istr + 1}",
                )
            )
            carrier_detail = (
                f"symmetry=SZ carrier=paired pairs={pair_count} "
                f"crossing={max_crossing} raw_bond={carrier_bond} "
                f"norm2={carrier_norm2:.12e}"
            )
        else:
            dets, coefficients = structure_determinants(
                ntstr[:, istr], nao, nae, nel
            )
            if len(dets) > max_expansion:
                raise RuntimeError(
                    f"structure {istr + 1} requires {len(dets)} determinant terms; "
                    "select XMVB_BLOCK2_CARRIER=paired"
                )
            carrier = driver.get_mps_from_csf_coefficients(
                dets=dets,
                dvals=coefficients,
                tag=f"VB-CARRIER-{istr + 1}",
                dot=2,
                full_fci=True,
                iprint=0,
            )
            carrier_detail = f"symmetry=SZ carrier=determinants terms={len(dets)}"
        if state_builder == "evolution":
            if rotation_mpo is None:
                physical = carrier
            else:
                physical = driver.td_dmrg(
                    rotation_mpo,
                    carrier,
                    delta_t=1.0 / rotation_steps,
                    n_steps=rotation_steps,
                    bond_dims=[tnbond],
                    te_type=rotation_algorithm,
                    n_sub_sweeps=2,
                    normalize_mps=False,
                    hermitian=False,
                    cutoff=1.0e-20,
                    iprint=0,
                    final_mps_tag=f"VB-PHYSICAL-{istr + 1}",
                )
        states.append(physical)
        print(
            f"[TNVB-BLOCK2] physical_MPS progress={istr + 1}/{nstr} "
            f"{carrier_detail} physical_bond={physical.info.bond_dim} "
            f"dot={physical.dot} center={physical.center}",
            flush=True,
        )
    del rotation_mpo, creators, pair_cache
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
        symm_type=SymmetryTypes.SU2 if symmetry == "SU2" else SymmetryTypes.SZ,
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


def state_d3g_parallel_layout(task_count: int, nao: int, total_bond: int):
    """Choose a bounded process/thread layout for hole-state matrix elements."""
    allocated, _ = allocated_cpu_count()
    if task_count < 1:
        raise RuntimeError("STATE contracted D3G requires at least one task")
    # Process startup is larger than the whole contraction below 10 orbitals.
    if nao < 10 or allocated < 4:
        return 1, allocated

    worker_override = positive_environment_integer(
        "XMVB_BLOCK2_STATE_D3G_WORKERS"
    )
    thread_override = positive_environment_integer(
        "XMVB_BLOCK2_STATE_D3G_THREADS"
    )
    if thread_override is not None:
        block2_threads = min(thread_override, allocated)
        worker_capacity = max(1, allocated // block2_threads)
    else:
        worker_capacity = min(8, max(1, allocated // 2))
        block2_threads = max(1, allocated // worker_capacity)
    if worker_override is not None:
        worker_capacity = min(worker_capacity, worker_override)
        if thread_override is None:
            block2_threads = max(1, allocated // worker_capacity)

    memory_bytes, _ = memory_allocation_bytes()
    if memory_bytes is not None:
        # Each process reloads all K hole MPSs and owns one two-electron MPO.
        # Keep a conservative four-stack allowance for renormalized operators.
        per_process = 4 * state_d3g_stack_memory_bytes(nao, total_bond)
        per_process += 64 * nao * max(2, total_bond) ** 2
        memory_capacity = max(1, int(0.75 * memory_bytes) // per_process)
        worker_capacity = min(worker_capacity, memory_capacity)
        if thread_override is None:
            block2_threads = max(1, allocated // worker_capacity)

    return min(task_count, worker_capacity), block2_threads


def native_d3g_parallel_layout(batch_count: int, nao: int, total_bond: int):
    """Choose process concurrency for pruned native output batches.

    Every batch performs one independent expectation sweep.  Measurements on
    the C14 prototype show that assigning more OpenMP threads to a single
    small-output MPO is much less effective than running independent batches
    concurrently.  Keep this execution choice outside the XMVB input model.
    """
    allocated, _ = allocated_cpu_count()
    if batch_count < 1:
        raise RuntimeError("native D3G requires at least one output batch")
    # Spawn/driver/MPS-copy overhead dominates the 4e/4o correctness test.
    if nao < 10 or allocated < 4:
        return 1, allocated

    worker_override = positive_environment_integer(
        "XMVB_BLOCK2_NATIVE_WORKERS"
    )
    thread_override = positive_environment_integer(
        "XMVB_BLOCK2_NATIVE_THREADS"
    )
    block2_threads = min(thread_override or 1, allocated)
    worker_capacity = max(1, allocated // block2_threads)
    # More than sixteen private symbolic MPO builders tends to contend for
    # memory bandwidth and scratch metadata even on wider allocations.
    worker_capacity = min(worker_capacity, 16)
    if worker_override is not None:
        worker_capacity = min(worker_capacity, worker_override)

    memory_bytes, _ = memory_allocation_bytes()
    if memory_bytes is not None:
        # The parent still owns the total-state/RDM driver.  A child owns its
        # Block2 stack, two factor MPOs, one pruned fused MPO and one MPS copy.
        parent_reserve = 2 * rdm_stack_memory_bytes(nao, total_bond)
        per_process = native_d3g_stack_memory_bytes(nao, total_bond)
        per_process += 128 * nao * max(2, total_bond) ** 2
        usable = max(0, int(0.80 * memory_bytes) - parent_reserve)
        worker_capacity = min(
            worker_capacity, max(1, usable // max(1, per_process))
        )

    return min(batch_count, worker_capacity), block2_threads


def initialize_parallel_native_d3g_worker(
    source_scratch,
    scratch_parent,
    state_tag,
    nao,
    nae,
    total_bond,
    g_orth,
    cutoff,
    native_layout,
    block2_threads,
):
    """Build process-private factors for selected-output native sweeps."""
    global _PARALLEL_NATIVE_D3G_CONTEXT
    if native_layout not in ("nc-raw", "cn-raw"):
        raise RuntimeError(
            "parallel native D3G currently requires nc-raw or cn-raw"
        )
    configure_phase_environment(block2_threads)
    scratch = tempfile.mkdtemp(
        prefix=f"tnvb-block2-native-d3g-{os.getpid()}-", dir=scratch_parent
    )
    atexit.register(shutil.rmtree, scratch, ignore_errors=True)
    copy_parallel_state_files(source_scratch, scratch, [state_tag])
    driver = DMRGDriver(
        scratch=scratch,
        symm_type=SymmetryTypes.SU2,
        n_threads=block2_threads,
        stack_mem=native_d3g_stack_memory_bytes(nao, total_bond),
        clean_scratch=True,
    )
    driver.initialize_system(n_sites=nao, n_elec=nae, spin=0, orb_sym=[0] * nao)
    state = driver.load_mps(state_tag)

    bw = driver.bw
    permutation = bw.b.SpinPermScheme.initialize_su2(2, "(C+D)0", True)
    scheme = bw.b.NPDMScheme(bw.b.VectorSpinPermScheme([permutation]))
    primary_mpo = bw.bs.GeneralNPDMMPO(driver.ghamil, scheme, False)
    primary_mpo.delta_quantum = (state.info.target - state.info.target)[0]
    primary_mpo.iprint = 0
    primary_mpo.build()

    family = native_layout.split("-", 1)[0]
    algorithm = {
        "nc": MPOAlgorithmTypes.NoTransConventionalNC,
        "cn": MPOAlgorithmTypes.NoTransConventionalCN,
    }[family]
    vee_wrapper = driver.get_qc_mpo(
        h1e=np.zeros((nao, nao), dtype=np.float64),
        g2e=g_orth,
        ecore=0.0,
        cutoff=max(cutoff, 1.0e-16),
        integral_cutoff=cutoff,
        symmetrize=False,
        algo_type=algorithm,
        add_ident=False,
        iprint=0,
    )
    # Retain the wrapper as well as prim_mpo: the Python owner must outlive
    # the borrowed native QC MPO used by SU2ComplementaryMPO.
    _PARALLEL_NATIVE_D3G_CONTEXT = (
        driver,
        state,
        primary_mpo,
        vee_wrapper,
        vee_wrapper.prim_mpo,
    )


def contract_parallel_native_d3g_batch(task):
    """Evaluate one selected-output complementary-MPO batch."""
    if _PARALLEL_NATIVE_D3G_CONTEXT is None:
        raise RuntimeError("parallel native D3G worker was not initialized")
    batch_index, output_cut, selected_outputs = task
    driver, state, primary_mpo, _, vee_mpo = _PARALLEL_NATIVE_D3G_CONTEXT
    bw = driver.bw

    if output_cut is not None:
        driver.align_mps_center(state, output_cut)
    started = time.perf_counter()
    fused_mpo = bw.bs.SU2ComplementaryMPO(
        primary_mpo,
        vee_mpo,
        0,
        f"TNVB-SU2-NATIVE-P{os.getpid()}-{batch_index}",
        bw.b.VectorULInt(selected_outputs),
    )
    build_seconds = time.perf_counter() - started

    started = time.perf_counter()
    environment = bw.bs.MovingEnvironment(
        fused_mpo,
        state,
        state,
        f"TNVB-D3G-SU2-NATIVE-P{os.getpid()}-{batch_index}",
    )
    environment.cached_contraction = False
    environment.fused_contraction_rotation = True
    environment.init_environments(False)
    environment_seconds = time.perf_counter() - started

    expectation = bw.bs.Expect(
        environment, state.info.bond_dim, state.info.bond_dim
    )
    expectation.cutoff = 1.0e-24
    expectation.zero_dot_algo = True
    expectation.algo_type = bw.b.ExpectationAlgorithmTypes.Fast
    expectation.iprint = 0
    started = time.perf_counter()
    if output_cut is None:
        expectation.solve(True, state.center == 0)
        npdms = list(expectation.get_npdm())
        if len(npdms) != 1:
            raise RuntimeError(
                f"native output batch returned {len(npdms)} components"
            )
        transition = np.asarray(npdms[0], dtype=np.float64).copy()
    else:
        iteration = expectation.update_zero_dot(
            output_cut,
            True,
            True,
            state.info.bond_dim,
            state.info.bond_dim,
        )
        transition = np.zeros(
            (driver.n_sites, driver.n_sites), dtype=np.float64
        )
        for expression, value in iteration.expectations:
            flat = _packed_site_index_value(expression.site_index)
            c, d = _decode_npdm_linear_index(flat, driver.n_sites, 2)
            transition[c, d] += float(value)
    sweep_seconds = time.perf_counter() - started
    if driver.clean_scratch:
        environment.remove_partition_files()
    expected_shape = (driver.n_sites, driver.n_sites)
    if transition.shape != expected_shape:
        raise RuntimeError(
            f"native output batch returned {transition.shape}, "
            f"expected {expected_shape}"
        )
    del expectation, environment
    fused_mpo.deallocate()
    return (
        batch_index,
        transition,
        build_seconds,
        environment_seconds,
        sweep_seconds,
    )


def contract_parallel_native_output_batches(
    state,
    source_scratch,
    scratch_parent,
    nao,
    nae,
    total_bond,
    g_orth,
    cutoff,
    native_layout,
    output_batches,
    output_cuts,
    worker_count,
    block2_threads,
):
    """Sum independently pruned native outputs in spawned processes."""
    started = time.perf_counter()
    state_tag = persist_parallel_states([state], source_scratch)[0]
    tasks = [
        (batch_index, output_cut, selected_outputs)
        for batch_index, (output_cut, selected_outputs) in enumerate(
            zip(output_cuts, output_batches), 1
        )
    ]
    log_summary(
        "[TNVB-D3G-NATIVE-PARALLEL] start "
        f"batches={len(tasks)} workers={worker_count}x{block2_threads}"
    )
    configure_phase_environment(block2_threads)
    context = multiprocessing.get_context("spawn")
    results = []
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=initialize_parallel_native_d3g_worker,
        initargs=(
            source_scratch,
            scratch_parent,
            state_tag,
            nao,
            nae,
            total_bond,
            g_orth,
            cutoff,
            native_layout,
            block2_threads,
        ),
    ) as executor:
        futures = [
            executor.submit(contract_parallel_native_d3g_batch, task)
            for task in tasks
        ]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item[0])
    transition = np.zeros((nao, nao), dtype=np.float64)
    build_seconds = 0.0
    environment_seconds = 0.0
    sweep_seconds = 0.0
    for _, batch_transition, build, environment, sweep in results:
        transition += batch_transition
        build_seconds += build
        environment_seconds += environment
        sweep_seconds += sweep
    wall_seconds = time.perf_counter() - started
    log_summary(
        "[TNVB-D3G-NATIVE-PARALLEL] complete "
        f"batches={len(results)} workers={worker_count}x{block2_threads} "
        f"wall={wall_seconds:.3f}s build_cpu={build_seconds:.3f}s "
        f"env_cpu={environment_seconds:.3f}s sweep_cpu={sweep_seconds:.3f}s"
    )
    return transition, build_seconds, environment_seconds, sweep_seconds


def initialize_parallel_state_d3g_worker(
    source_scratch,
    scratch_parent,
    state_tags,
    nao,
    nae,
    total_bond,
    g_orth,
    block2_threads,
):
    """Reload hole states and the two-electron MPO in one private process."""
    global _PARALLEL_STATE_D3G_CONTEXT
    configure_phase_environment(block2_threads)
    scratch = tempfile.mkdtemp(
        prefix=f"tnvb-block2-state-d3g-{os.getpid()}-", dir=scratch_parent
    )
    atexit.register(shutil.rmtree, scratch, ignore_errors=True)
    copy_parallel_state_files(source_scratch, scratch, state_tags)
    driver = DMRGDriver(
        scratch=scratch,
        symm_type=SymmetryTypes.SU2,
        n_threads=block2_threads,
        stack_mem=state_d3g_stack_memory_bytes(nao, total_bond),
        clean_scratch=True,
    )
    driver.initialize_system(n_sites=nao, n_elec=nae, spin=0, orb_sym=[0] * nao)
    identity_mpo = driver.get_identity_mpo()
    vee_mpo = get_hamiltonian_mpo(
        driver,
        np.zeros((nao, nao), dtype=np.float64),
        g_orth,
        log=False,
        mpo_mode="no-trans-conventional",
    )
    holes = [driver.load_mps(tag) for tag in state_tags]
    _PARALLEL_STATE_D3G_CONTEXT = (driver, identity_mpo, vee_mpo, holes)


def contract_parallel_state_d3g_batch(batch):
    """Evaluate a deterministic batch of hole overlaps and V matrix elements."""
    if _PARALLEL_STATE_D3G_CONTEXT is None:
        raise RuntimeError("parallel STATE D3G worker was not initialized")
    driver, identity_mpo, vee_mpo, holes = _PARALLEL_STATE_D3G_CONTEXT
    results = []
    for order, c, d in batch:
        started = time.perf_counter()
        overlap = float(driver.expectation(holes[c], identity_mpo, holes[d]))
        overlap_seconds = time.perf_counter() - started
        started = time.perf_counter()
        value = float(driver.expectation(holes[c], vee_mpo, holes[d]))
        value_seconds = time.perf_counter() - started
        results.append(
            (
                order,
                c,
                d,
                overlap,
                overlap_seconds,
                value,
                value_seconds,
            )
        )
    return results


def contract_parallel_state_d3g(
    holes,
    source_scratch,
    scratch_parent,
    nao,
    nae,
    total_bond,
    g_orth,
    worker_count,
    block2_threads,
):
    """Evaluate the lower hole-state Hamiltonian triangle in processes."""
    started = time.perf_counter()
    state_tags = persist_parallel_states(holes, source_scratch)
    tasks = [
        (c * nao + d, c, d)
        for c in range(nao)
        for d in range(c + 1)
    ]
    tasks.sort(key=lambda item: (-max(item[1], item[2]), -min(item[1], item[2])))
    chunk_size = positive_environment_integer(
        "XMVB_BLOCK2_STATE_D3G_CHUNK_SIZE"
    ) or 1
    batches = [
        tasks[start : start + chunk_size]
        for start in range(0, len(tasks), chunk_size)
    ]
    log_summary(
        "[TNVB-D3G-STATE-PARALLEL] start "
        f"pairs={len(tasks)} workers={worker_count}x{block2_threads} "
        f"chunks={len(batches)}"
    )
    configure_phase_environment(block2_threads)
    context = multiprocessing.get_context("spawn")
    results = []
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=initialize_parallel_state_d3g_worker,
        initargs=(
            source_scratch,
            scratch_parent,
            state_tags,
            nao,
            nae,
            total_bond,
            g_orth,
            block2_threads,
        ),
    ) as executor:
        futures = [
            executor.submit(contract_parallel_state_d3g_batch, batch)
            for batch in batches
        ]
        for future in as_completed(futures):
            results.extend(future.result())
    results.sort(key=lambda item: item[0])
    seconds = time.perf_counter() - started
    log_summary(
        "[TNVB-D3G-STATE-PARALLEL] complete "
        f"pairs={len(results)} workers={worker_count}x{block2_threads} "
        f"seconds={seconds:.3f}"
    )
    return results, seconds


def initialize_parallel_d3g_worker(
    source_scratch,
    scratch_parent,
    state_tag,
    nao,
    nae,
    total_bond,
    norm,
    g_orth,
    cutoff,
    block2_threads,
):
    """Initialize one process-private scalar ``E[c,d] V`` contractor."""
    global _PARALLEL_D3G_CONTEXT
    initialized = time.perf_counter()
    configure_phase_environment(block2_threads)
    scratch = tempfile.mkdtemp(
        prefix=f"tnvb-block2-d3g-{os.getpid()}-", dir=scratch_parent
    )
    atexit.register(shutil.rmtree, scratch, ignore_errors=True)
    copied = copy_parallel_state_files(source_scratch, scratch, [state_tag])
    driver = DMRGDriver(
        scratch=scratch,
        symm_type=SymmetryTypes.SU2,
        n_threads=block2_threads,
        stack_mem=stack_memory_bytes(nao, total_bond),
        clean_scratch=True,
    )
    driver.initialize_system(n_sites=nao, n_elec=nae, spin=0, orb_sym=[0] * nao)
    state = driver.load_mps(state_tag)
    expression, integral_indices, values = _su2_explicit_product_integral_terms(
        g_orth, cutoff
    )
    _PARALLEL_D3G_CONTEXT = (
        driver,
        state,
        norm,
        expression,
        integral_indices,
        values,
        cutoff,
    )
    print(
        f"[TNVB-BLOCK2-D3G-PARALLEL] worker={os.getpid()} "
        f"copied_files={copied} block2_threads={block2_threads} "
        f"integral_terms={len(values)} "
        f"init_seconds={time.perf_counter() - initialized:.3f}",
        flush=True,
    )


def contract_parallel_d3g_batch(batch):
    """Contract a deterministic batch of independent ``E[c,d] V`` MPOs."""
    if _PARALLEL_D3G_CONTEXT is None:
        raise RuntimeError("parallel D3G worker was not initialized")
    (
        driver,
        state,
        norm,
        expression,
        integral_indices,
        values,
        cutoff,
    ) = _PARALLEL_D3G_CONTEXT
    results = []
    for order, c, d in batch:
        value, build_seconds, expectation_seconds = (
            _contract_su2_explicit_product_element(
                driver,
                state,
                norm,
                expression,
                integral_indices,
                values,
                cutoff,
                c,
                d,
            )
        )
        results.append(
            (
                order,
                c,
                d,
                value,
                build_seconds,
                expectation_seconds,
                os.getpid(),
            )
        )
    return results


def contract_parallel_su2_explicit_transition(
    total,
    source_scratch,
    scratch_parent,
    nao,
    nae,
    total_bond,
    norm,
    g_orth,
    cutoff,
    worker_count,
    block2_threads,
    commutator,
):
    """Compute the cheaper lower triangle of ``T`` and recover the upper."""
    started = time.perf_counter()
    if commutator.shape != (nao, nao):
        raise RuntimeError("parallel D3G commutator has the wrong shape")
    state_tag = persist_parallel_states([total], source_scratch)[0]
    tasks = [
        (c * nao + d, c, d)
        for c in range(nao)
        for d in range(c + 1)
    ]
    # C14 measurements show the rightmost of the two free orbital sites
    # dominates the scalar-MPO cost.  Launch large max(c,d) elements first so
    # no expensive outer-orbital row remains as a serial pool-shutdown tail.
    tasks.sort(key=lambda item: (-max(item[1], item[2]), -min(item[1], item[2])))
    # Individual MPO costs vary strongly with the locations of c and d.  One
    # element per future prevents an unlucky orbital row from leaving all but
    # one worker idle near the end of the sweep.  A larger explicit chunk is
    # available for platforms where scheduler overhead is measurable.
    configured_chunk = positive_environment_integer(
        "XMVB_BLOCK2_D3G_CHUNK_SIZE"
    )
    chunk_size = configured_chunk or 1
    batches = [
        tasks[start : start + chunk_size]
        for start in range(0, len(tasks), chunk_size)
    ]
    print(
        "[TNVB-BLOCK2-D3G-PARALLEL] start "
        f"elements={len(tasks)} workers={worker_count} "
        f"block2_threads={block2_threads} chunks={len(batches)} "
        f"chunk_size={chunk_size}",
        flush=True,
    )
    configure_phase_environment(block2_threads)
    context = multiprocessing.get_context("spawn")
    results = []
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=initialize_parallel_d3g_worker,
        initargs=(
            source_scratch,
            scratch_parent,
            state_tag,
            nao,
            nae,
            total_bond,
            norm,
            g_orth,
            cutoff,
            block2_threads,
        ),
    ) as executor:
        futures = [
            executor.submit(contract_parallel_d3g_batch, batch)
            for batch in batches
        ]
        completed = 0
        for future in as_completed(futures):
            batch_result = future.result()
            results.extend(batch_result)
            for (
                _,
                c,
                d,
                value,
                build_seconds,
                expectation_seconds,
                worker_pid,
            ) in batch_result:
                completed += 1
                print(
                    "[TNVB-BLOCK2-D3G-PARALLEL] "
                    f"element={completed}/{len(tasks)} c={c + 1} d={d + 1} "
                    f"worker={worker_pid} value={value:.17e} "
                    f"build_seconds={build_seconds:.3f} "
                    f"expectation_seconds={expectation_seconds:.3f} "
                    f"wall_seconds={time.perf_counter() - started:.3f}",
                    flush=True,
                )
    results.sort(key=lambda item: item[0])
    transition = np.empty((nao, nao), dtype=np.float64)
    for (
        _,
        c,
        d,
        value,
        _,
        _,
        _,
    ) in results:
        transition[c, d] = value
    for c in range(nao):
        for d in range(c + 1, nao):
            transition[c, d] = transition[d, c] + commutator[c, d]
    print(
        "[TNVB-BLOCK2-D3G-PARALLEL] joined "
        f"elements={len(results)} workers={worker_count} "
        f"seconds={time.perf_counter() - started:.3f}",
        flush=True,
    )
    return transition


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
    rinv,
    h_orth,
    g_orth,
    d3g_mode_code,
    d3g_backend_code,
    sum_mode,
    warm_read_directory=None,
    warm_write_directory=None,
    selected_eigenpair=None,
    consume_source=False,
    orthogonal_output=False,
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
        total, sum_quality, sum_details = fit_weighted_total_state(
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
            warm_read_directory=warm_read_directory,
            warm_write_directory=warm_write_directory,
            sum_mode=sum_mode,
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
        d3g_mode = contracted_d3g_mode(d3g_mode_code)
        validate_direct = (
            os.environ.get("XMVB_BLOCK2_VALIDATE_DIRECT_D3G", "0") == "1"
        )
        dm3 = None
        dm3_seconds = 0.0
        contraction3_error = 0.0
        reduced_dm3 = False
        dm3_reduce_seconds = 0.0
        dm3_check_seconds = 0.0
        reference_contracted_orth = None
        requires_dm3 = d3g_mode in ("full", "validate") or validate_direct
        if (
            requires_dm3
            and os.environ.get("XMVB_BLOCK2_FORBID_3PDM", "0") == "1"
        ):
            raise RuntimeError(
                "rank-six density construction is forbidden for this run"
            )
        if requires_dm3:
            log_summary(
                f"[TNVB-RDM] dm3 start active={nao} bond={sum_bond} "
                f"threads={rdm_threads}"
            )
            stage_started = time.perf_counter()
            dm3 = np.asarray(rdm_full.get_profiled_3pdm(
                driver, total, npdm_algorithm, log_summary), dtype=np.float64)
            dm3_seconds = time.perf_counter() - stage_started
            if d3g_mode == "full" and not validate_direct:
                (reference_contracted_orth, particle3,
                 reduction_times) = rdm_full.reduce_full_3pdm(dm3, g_orth, norm)
                dm3_reduce_seconds = reduction_times["integral_seconds"]
                stage_started = time.perf_counter()
                contraction3_error = float(np.max(np.abs(particle3 - (nae - 2) * dm2)))
                dm3_check_seconds = (reduction_times["particle_seconds"] +
                                     time.perf_counter() - stage_started)
                tensor_gib = dm3.nbytes / 1024**3
                # Release the native rank-six output before AO transformations.
                # Its lifetime ends here; only O(K**4) and O(K**2) data survive.
                del dm3, particle3
                dm3 = None
                reduced_dm3 = True
                log_summary(
                    f"[TNVB-3RDM] algorithm={npdm_algorithm_name} "
                    f"actual_bond={sum_details.get('actual_bond', '-')} cap={sum_bond} "
                    f"native={dm3_seconds:.3f}s reduce={dm3_reduce_seconds:.3f}s "
                    f"check={dm3_check_seconds:.3f}s tensor={tensor_gib:.3f}GiB "
                    "full_copy=0"
                )
            else:
                # Research validation routines consume a normalized full tensor.
                dm3 = dm3 / norm
                reference_contracted_orth = np.einsum(
                    "faeb,abcdef->dc", g_orth, dm3, optimize=True)

        # This optional diagnostic applies V to the SU(2) MPS.  It is not part
        # of the production contracted-gradient algorithm.
        if validate_direct:
            validate_direct_contracted_d3g(
                driver,
                total,
                norm,
                nae,
                sum_bond,
                g_orth,
                dm2,
                dm3,
                reference_contracted_orth,
            )

        direct_contracted_orth = None
        direct_seconds = 0.0
        direct_trace_error = 0.0
        direct_overlap_error = 0.0
        direct_validation_error = 0.0
        if d3g_mode in ("contracted", "validate"):
            stage_started = time.perf_counter()
            direct_backend = contracted_d3g_backend(d3g_backend_code)
            integral_cutoff = float(
                os.environ.get("XMVB_BLOCK2_D3G_INTEGRAL_CUTOFF", "0")
            )
            if integral_cutoff < 0.0:
                raise RuntimeError(
                    "XMVB_BLOCK2_D3G_INTEGRAL_CUTOFF must be nonnegative"
                )
            direct_transition = None
            direct_lower = None
            vee_expectation = 0.5 * float(
                np.einsum("ijkl,iklj", g_orth, dm2, optimize=True)
            )
            if direct_backend == "state":
                (
                    direct_contracted_orth,
                    direct_transition,
                    direct_lower,
                    vee_expectation,
                    direct_trace_error,
                    direct_overlap_error,
                ) = contracted_d3g_from_su2_hole_states(
                    driver,
                    total,
                    norm,
                    sum_bond,
                    g_orth,
                    dm1,
                    dm2,
                    allow_overlap_failure=(d3g_mode == "validate"),
                )
            elif direct_backend == "native":
                (
                    direct_contracted_orth,
                    direct_transition,
                    direct_lower,
                    vee_expectation,
                    direct_trace_error,
                ) = contracted_d3g_from_su2_complementary(
                    driver,
                    total,
                    norm,
                    g_orth,
                    dm2,
                    cutoff=integral_cutoff,
                    iprint=int(
                        os.environ.get("XMVB_BLOCK2_D3G_IPRINT", "0")
                    ),
                    source_scratch=rdm_scratch,
                    scratch_parent=scratch_parent,
                )
                print(
                    "[TNVB-BLOCK2-D3G-CONTRACTED] complete "
                    "backend=su2-native-complementary "
                    f"vee_expectation={vee_expectation:.12e} "
                    f"transition_trace={np.trace(direct_transition):.12e} "
                    f"trace_error={direct_trace_error:.3e} "
                    f"seconds={time.perf_counter() - stage_started:.3f}",
                    flush=True,
                )
            elif direct_backend == "npdm":
                (
                    direct_contracted_orth,
                    original_middle,
                    contracted_middle,
                ) = contracted_d3g_from_weighted_su2_npdm(
                    driver,
                    total,
                    norm,
                    g_orth,
                    cutoff=integral_cutoff,
                    iprint=int(
                        os.environ.get("XMVB_BLOCK2_D3G_IPRINT", "0")
                    ),
                )
                contracted_trace_reference = (nae - 2) * float(
                    np.einsum("faeb,abef", g_orth, dm2, optimize=True)
                )
                direct_trace_error = abs(
                    float(np.trace(direct_contracted_orth))
                    - contracted_trace_reference
                )
                print(
                    "[TNVB-BLOCK2-D3G-CONTRACTED] complete "
                    "backend=weighted-su2-npdm "
                    f"middle_original={original_middle} "
                    f"middle_contracted={contracted_middle} "
                    f"trace_error={direct_trace_error:.3e} "
                    f"seconds={time.perf_counter() - stage_started:.3f}",
                    flush=True,
                )
            elif direct_backend == "sz-bridge":
                bridge_tag = "TNVB-D3G-SZ-BRIDGE"
                sz_total, sz_bond, conversion_seconds = (
                    save_su2_singlet_as_sz_component(
                        driver, total, rdm_scratch, bridge_tag
                    )
                )
                print(
                    "[TNVB-BLOCK2-D3G-SZ-BRIDGE] converted "
                    f"expanded_bond={sz_bond} "
                    f"seconds={conversion_seconds:.3f}",
                    flush=True,
                )

                # DMRGDriver owns process-global Block2 stacks.  The bindings
                # explicitly require the previous driver to be discarded
                # before another symmetry type is initialized.  At this point
                # every SU(2) observable needed below is already a NumPy
                # array, so the switch is safe and one-way for this request.
                del states, identity_mpo, h_mpo, total, sz_total
                gc.collect()
                driver.finalize()
                del driver
                gc.collect()

                driver = DMRGDriver(
                    scratch=rdm_scratch,
                    symm_type=SymmetryTypes.SZ,
                    n_threads=rdm_threads,
                    stack_mem=rdm_stack_bytes,
                    clean_scratch=True,
                )
                driver.initialize_system(
                    n_sites=nao, n_elec=nae, spin=0, orb_sym=[0] * nao
                )
                total = driver.load_mps(bridge_tag)
                identity_mpo = driver.get_identity_mpo()
                sz_norm = float(driver.expectation(total, identity_mpo, total))
                norm_error = abs(sz_norm - norm)
                norm_tolerance = float(
                    os.environ.get("XMVB_BLOCK2_SZ_BRIDGE_NORM_TOL", "1e-10")
                )
                if norm_error > norm_tolerance:
                    raise RuntimeError(
                        "SU2-to-SZ bridge changed the total-state norm by "
                        f"{norm_error:.3e}; limit is {norm_tolerance:.3e}"
                    )
                measurement_started = time.perf_counter()
                (
                    direct_contracted_orth,
                    direct_transition,
                    direct_lower,
                    vee_expectation,
                    direct_trace_error,
                ) = contracted_d3g_from_sz_stacked_npdm(
                    driver,
                    total,
                    sz_norm,
                    g_orth,
                    dm2,
                    iprint=int(
                        os.environ.get("XMVB_BLOCK2_D3G_IPRINT", "0")
                    ),
                )
                measurement_seconds = time.perf_counter() - measurement_started
                print(
                    "[TNVB-BLOCK2-D3G-CONTRACTED] complete "
                    "backend=su2-sz-bridge "
                    f"expanded_bond={sz_bond} "
                    f"conversion_seconds={conversion_seconds:.3f} "
                    f"measurement_seconds={measurement_seconds:.3f} "
                    f"norm_error={norm_error:.3e} "
                    f"vee_expectation={vee_expectation:.12e} "
                    f"transition_trace={np.trace(direct_transition):.12e} "
                    f"trace_error={direct_trace_error:.3e} "
                    f"seconds={time.perf_counter() - stage_started:.3f}",
                    flush=True,
                )
            elif direct_backend == "sigma":
                (
                    direct_contracted_orth,
                    direct_transition,
                    direct_lower,
                    vee_expectation,
                    direct_trace_error,
                    sigma_diagnostics,
                ) = contracted_d3g_from_su2_sigma_mps(
                    driver,
                    total,
                    norm,
                    sum_bond,
                    g_orth,
                    dm2,
                )
                print(
                    "[TNVB-BLOCK2-D3G-CONTRACTED] complete "
                    "backend=su2-sigma-mps "
                    f"bond={int(os.environ.get('XMVB_BLOCK2_SIGMA_BOND', 2 * sum_bond))} "
                    f"overlap_error={sigma_diagnostics['overlap_error']:.3e} "
                    f"vee_expectation={vee_expectation:.12e} "
                    f"transition_trace={np.trace(direct_transition):.12e} "
                    f"trace_error={direct_trace_error:.3e} "
                    f"seconds={time.perf_counter() - stage_started:.3f}",
                    flush=True,
                )
            else:
                (
                    direct_contracted_orth,
                    direct_transition,
                    direct_lower,
                    vee_expectation,
                    direct_trace_error,
                ) = contracted_d3g_from_su2_explicit_product(
                    driver,
                    total,
                    norm,
                    g_orth,
                    dm2,
                    cutoff=integral_cutoff,
                    iprint=int(
                        os.environ.get("XMVB_BLOCK2_D3G_IPRINT", "0")
                    ),
                    source_scratch=rdm_scratch,
                    scratch_parent=scratch_parent,
                )
            direct_seconds = time.perf_counter() - stage_started
            if direct_backend == "explicit":
                print(
                    "[TNVB-BLOCK2-D3G-CONTRACTED] complete "
                    "backend=su2-explicit "
                    f"scalar_mpos={nao * (nao + 1) // 2} "
                    f"vee_expectation={vee_expectation:.12e} "
                    f"transition_trace={np.trace(direct_transition):.12e} "
                    f"trace_error={direct_trace_error:.3e} "
                    f"seconds={direct_seconds:.3f}",
                    flush=True,
                )
            if d3g_mode == "validate":
                transition_error = 0.0
                transition_check = "skipped"
                if direct_transition is not None:
                    reference_lower = np.einsum(
                        "abde,cabe->dc", g_orth, dm2, optimize=True
                    )
                    transition_reference = (
                        reference_lower + 0.5 * reference_contracted_orth
                    ).T
                    transition_error = float(
                        np.max(
                            np.abs(direct_transition - transition_reference)
                        )
                    )
                    transition_check = "performed"
                contraction_error = float(
                    np.max(
                        np.abs(
                            direct_contracted_orth
                            - reference_contracted_orth
                        )
                    )
                )
                direct_validation_error = max(
                    transition_error,
                    contraction_error,
                    direct_trace_error,
                    direct_overlap_error,
                )
                print(
                    "[TNVB-BLOCK2-D3G-VALIDATE] complete "
                    f"backend={direct_backend} "
                    f"transition_check={transition_check} "
                    f"transition_error={transition_error:.3e} "
                    f"contraction_error={contraction_error:.3e} "
                    f"overlap_error={direct_overlap_error:.3e}",
                    flush=True,
                )
                direct_tolerance = float(
                    os.environ.get("XMVB_BLOCK2_DIRECT_D3G_CHECK_TOL", "1e-9")
                )
                if direct_validation_error > direct_tolerance:
                    raise RuntimeError(
                        "contracted D3G validation failed: "
                        f"transition={transition_error:.3e}, "
                        f"contraction={contraction_error:.3e}, "
                        f"trace={direct_trace_error:.3e}, "
                        f"overlap={direct_overlap_error:.3e}, "
                        f"max={direct_validation_error:.3e}, "
                        f"limit={direct_tolerance:.3e}"
                    )
        print(
            f"[TNVB-BLOCK2-RDM-TIMING] dm1_seconds={dm1_seconds:.3f} "
            f"dm2_seconds={dm2_seconds:.3f} dm3_seconds={dm3_seconds:.3f} "
            f"contracted_d3g_seconds={direct_seconds:.3f} "
            f"d3g_mode={d3g_mode} npdm_algorithm={npdm_algorithm_name}",
            flush=True,
        )
        contracted_orth = (
            direct_contracted_orth
            if d3g_mode == "contracted"
            else reference_contracted_orth
        )
        if orthogonal_output:
            pmat, pit, contracted_d3 = orthogonal_reduced_data(dm1,dm2,contracted_orth)
        else:
            pmat, pit, contracted_d3 = transform_reduced_data_to_xmvb(
                rinv, dm1, dm2, contracted_orth)
        reconstructed = float(np.einsum("ij,ij", h_orth, dm1))
        reconstructed += 0.5 * float(np.einsum("ijkl,iklj", g_orth, dm2))
        trace_error = abs(float(np.trace(dm1)) - nae)
        contraction2_error = float(
            np.max(np.abs(np.einsum("ijja->ia", dm2) - (nae - 1) * dm1))
        )
        if dm3 is not None and not reduced_dm3:
            contraction3_error = float(
                np.max(
                    np.abs(
                        np.einsum("ijkkba->ijba", dm3) - (nae - 2) * dm2
                    )
                )
            )
        rdm_energy_error = abs(reconstructed - total_energy)
        identity_error = max(
            trace_error,
            contraction2_error,
            contraction3_error,
            direct_trace_error,
            direct_overlap_error,
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
        validation_summary = (
            f" validate_max={direct_validation_error:.3e}"
            if d3g_mode == "validate"
            else ""
        )
        log_summary(
            f"[TNVB-RDM] sum={sum_seconds:.3f}s dm1={dm1_seconds:.3f}s "
            f"dm2={dm2_seconds:.3f}s dm3={dm3_seconds:.3f}s "
            f"contracted={direct_seconds:.3f}s total={rdm_total_seconds:.3f}s "
            f"fit_dE={energy_error:.3e} RDM_dE={rdm_energy_error:.3e} "
            f"identity_max={identity_error:.3e} norm_error={abs(1.0 - norm):.3e}"
            f"{validation_summary}"
        )
        rdm_tolerance = float(os.environ.get("XMVB_BLOCK2_RDM_CHECK_TOL", "1e-7"))
        check_total_state_rdm(
            trace_error,
            contraction2_error,
            contraction3_error,
            rdm_energy_error,
            direct_trace_error,
            direct_overlap_error,
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
    request, root=None, sum_bond=None, d3g_mode=None, d3g_backend=None,
    sum_mode=0, ci_options=None, ci_result=None, retain_state=None,
    integral_frame=None,
):
    nstr, nao, nae, nel, tnbond, ntstr, sso, hho, packed_ggo = request
    started = time.perf_counter()
    if integral_frame is None:
        rmat, h_orth, g_orth = orthogonalize_integrals(sso, hho, packed_ggo)
    else:
        rmat, h_orth, g_orth = accept_orthogonal_integrals(sso,hho,packed_ggo,integral_frame)
    rinv = scipy.linalg.solve_triangular(
        rmat, np.eye(nao), lower=False, check_finite=True
    )

    symmetry = "SU2"
    if retain_state is not None and root is not None:
        raise ValueError("TNVB can retain forward states only for ENERGY")
    if root is not None and (sum_bond is None or sum_bond < tnbond):
        raise RuntimeError("TNVB RDM sum bond must be at least TNBOND")
    state_builder = "pair"

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
    if root is not None and symmetry == "SU2":
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
        symmetry == "SU2"
        and state_workers > 1
        and nstr > 1
        and parallel_states_setting in ("1", "true", "yes")
    )
    if not parallel_states:
        state_workers = 1

    d3g_name = "ENERGY" if root is None else contracted_d3g_mode(d3g_mode).upper()
    if root is None:
        backend_name = "NONE"
    elif d3g_name == "FULL":
        backend_name = "FULL"
    else:
        backend_name = contracted_d3g_backend(d3g_backend).upper()
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
        f"sum={('DIRECT' if sum_mode == 0 else 'FIT') if root is not None else '-'} "
        f"d3g={d3g_name} backend={backend_name} "
        f"integrals={'ORTHOGONAL' if integral_frame is not None else 'RAW'}"
    )
    log_summary(
        f"[TNVB-PARALLEL] state={state_workers}x{state_threads} "
        f"hs={hs_workers}x{hs_threads} rdm=1x{rdm_threads_preview} "
        f"stack={rdm_stack_preview_gib:.3f}GiB"
    )

    print(
        f"[TNVB-BLOCK2] configuration symmetry={symmetry} "
        f"state_builder={state_builder} TNBOND={tnbond} "
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
            symm_type=SymmetryTypes.SU2 if symmetry == "SU2" else SymmetryTypes.SZ,
            n_threads=state_threads,
            stack_mem=stack_memory_bytes(nao, tnbond),
            clean_scratch=True,
        )
        driver.initialize_system(n_sites=nao, n_elec=nae, spin=0, orb_sym=[0] * nao)
        identity_mpo = driver.get_identity_mpo()
        if symmetry == "SU2":
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
        else:
            states = build_sz_states(
                driver, ntstr, nstr, nao, nae, nel, tnbond, rmat
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
                root, hmat, smat, rinv, h_orth, g_orth, d3g_mode, d3g_backend,
                sum_mode, warm_read_directory=warm_read_directory,
                warm_write_directory=warm_write_directory,
                selected_eigenpair=(eigenvalue, coefficient),
                orthogonal_output=integral_frame is not None)
        if warm_root is not None:
            commit_warm_cache(warm_root, warm_write_directory, warm_manifest_data)
        if retain_state is not None:
            from forward_state import export_context
            export_context(retain_state, scratch, state_tags, rinv, h_orth, g_orth)
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
            rinv,
            h_orth,
            g_orth,
            d3g_mode,
            d3g_backend,
            sum_mode,
            warm_read_directory=warm_read_directory,
            warm_write_directory=warm_write_directory,
            orthogonal_output=integral_frame is not None,
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
        export_context(retain_state, scratch, state_tags, rinv, h_orth, g_orth)
    shutil.rmtree(scratch, ignore_errors=True)
    if rdm_result is None:
        return hmat, smat
    return (hmat, smat) + rdm_result
