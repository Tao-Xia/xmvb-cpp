#!/usr/bin/env python3
"""Independent C++ TNVB protocol: singlet SU(2), fixed D, DIRECT, FULL only.

Energy requests construct structure MPSs and solve their H/S problem.
Gradient requests additionally construct the DIRECT total state and its full RDMs.
"""

import os
import struct
import sys
import time
import traceback

# This process (and its spawned workers) owns the numerical runtime.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["GOTO_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["XMVB_BLOCK2_RDM_CHECK_TOL"] = "1e-7"

import numpy as np
import scipy.linalg
import pair_core as core

REQUEST_MAGIC = b"CPTNREQ3"
RESPONSE_MAGIC = b"CPTNRES3"
VERSION = 3


def read_array(stream, count, dtype, shape=None):
    itemsize = np.dtype(dtype).itemsize
    value = np.frombuffer(core.read_exact(stream, count * itemsize), dtype=dtype)
    return value.copy() if shape is None else value.reshape(shape, order="F").copy()


def read_request(path):
    with open(path, "rb") as stream:
        if core.read_exact(stream, 8) != REQUEST_MAGIC:
            raise ValueError("invalid C++ TNVB request magic")
        version, nstr, norb, nelec, total_elec, bond, sum_bond, mode, solver, nguess = struct.unpack(
            "<10i", core.read_exact(stream, 40))
        if version != VERSION or mode not in (0, 1) or solver not in (0, 1) or nguess not in (0, 1):
            raise ValueError("unsupported C++ TNVB request")
        if min(nstr, norb, nelec, bond) <= 0 or nelec % 2 or total_elec < nelec:
            raise ValueError("invalid C++ TNVB dimensions")
        if (total_elec - nelec) % 2 or (mode and sum_bond < bond):
            raise ValueError("invalid inactive electron count or total-state bond")
        energy_tol, gradient_tol = struct.unpack("<2d", core.read_exact(stream, 16))
        if not all(np.isfinite(t) and t > 0 for t in (energy_tol, gradient_tol)):
            raise ValueError("invalid TNVB structure-solve accuracy")
        initial = read_array(stream, nstr, "<f8") if nguess else None
        if initial is not None and not np.all(np.isfinite(initial)):
            raise ValueError("invalid TNVB initial CI coefficients")
        options = (dict(energy_tolerance=energy_tol, gradient_tolerance=gradient_tol,
                        initial=initial) if solver else None)
        structures = read_array(stream, nstr * total_elec, "<i4", (total_elec, nstr))
        overlap = read_array(stream, norb * norb, "<f8", (norb, norb))
        frame = read_array(stream, norb * norb, "<f8", (norb, norb))
        one_electron = read_array(stream, norb * norb, "<f8", (norb, norb))
        npair = norb * (norb + 1) // 2
        eri = read_array(stream, npair * (npair + 1) // 2, "<f8")
        if stream.read(1):
            raise ValueError("trailing C++ TNVB request data")
        if not all(np.all(np.isfinite(v)) for v in (overlap, frame, one_electron, eri)):
            raise ValueError("nonfinite C++ TNVB integrals")
    return (nstr, norb, nelec, total_elec, bond, structures,
            overlap, one_electron, eri), sum_bond, mode, options, frame


def dense_roots(hamiltonian, overlap):
    core.validate_structure_metric(hamiltonian, overlap)
    norms = np.sqrt(np.diag(overlap))
    h = hamiltonian / norms[:, None] / norms[None, :]
    s = overlap / norms[:, None] / norms[None, :]
    energy, vector = scipy.linalg.eigh(h, s, driver="gvd", check_finite=True)
    coefficients = vector / norms[:, None]
    return energy, coefficients


def evaluate(request, sum_bond, mode, ci_options=None, retain_state=None, integral_frame=None):
    if integral_frame is None:
        raise ValueError("C++ TNVB requires its orthogonal integral frame")
    # The C++ paired-CSF expansion is invariant under exchanging the two
    # members of an individual singlet pair. The reused XMVB core attaches an
    # oriented-pair phase instead. Canonicalize pair members at this adapter
    # boundary so H/S and reported coefficients follow the C++ convention;
    # preserve pair order, inactive occupations and the caller's raw input.
    nstr, norb, nelec, total_elec, bond, structures, overlap, h1, eri = request
    structures = structures.copy()
    first_active = total_elec - nelec
    for istr in range(nstr):
        active_pairs = structures[first_active:, istr].reshape((-1, 2))
        structures[first_active:, istr] = np.sort(active_pairs, axis=1).ravel()
    request = (nstr, norb, nelec, total_elec, bond, structures, overlap, h1, eri)
    started = time.perf_counter()
    ci_result = {}
    if mode == 0:
        hmat, smat = core.build_hs(request, ci_options=ci_options, ci_result=ci_result,
                                  retain_state=retain_state, integral_frame=integral_frame)
        reduced = None
        fit_error = identity_error = rdm_seconds = 0.0
    else:
        result = core.build_hs(request, root=0, sum_bond=sum_bond,
                               ci_options=ci_options, ci_result=ci_result,
                               integral_frame=integral_frame)
        hmat, smat, fit_error, root_energy, root_coefficient, p, pi, d3g, checks = result
        reduced = (p, pi, d3g)
        identity_error = checks["identity_error"]
        rdm_seconds = checks["total_seconds"]
    if ci_options is None:
        energy, coefficients = dense_roots(hmat, smat)
    else:
        energy = np.asarray([ci_result["energy"]])
        coefficients = ci_result["coefficient"][:, None]
    if mode:
        if abs(energy[0] - root_energy) > 1e-10:
            raise RuntimeError("RDM and H/S root energies disagree")
        # Preserve exactly the coefficients actually used to construct the RDM.
        coefficients[:, 0] = root_coefficient
    hs_seconds = max(0.0, time.perf_counter() - started - rdm_seconds)
    return (hmat, smat, energy, coefficients, reduced,
            (fit_error, identity_error, hs_seconds, rdm_seconds),
            ci_result.get("diagnostics"))


def write_response(path, request, mode, result):
    nstr, norb = request[:2]
    h, s, energy, coefficients, reduced, diagnostics, davidson = result
    with open(path, "wb") as stream:
        stream.write(RESPONSE_MAGIC)
        stream.write(struct.pack("<8i", VERSION, 0, nstr, norb, mode, len(energy),
                                 int(davidson is not None), 0))
        for value in (h, s, energy, coefficients):
            if not np.all(np.isfinite(value)):
                raise RuntimeError("nonfinite C++ TNVB response")
            stream.write(np.asarray(value, dtype="<f8").ravel(order="F").tobytes())
        stream.write(struct.pack("<4d", *diagnostics))
        if davidson is not None:
            stream.write(struct.pack("<3i2d", davidson["iterations"], davidson["block_actions"],
                davidson["peak_subspace_dimension"], davidson["maximum_relative_residual"],
                davidson["maximum_stationarity_residual"]))
        if mode:
            for value in reduced:
                if not np.all(np.isfinite(value)):
                    raise RuntimeError("nonfinite C++ TNVB density response")
                stream.write(np.asarray(value, dtype="<f8").ravel(order="F").tobytes())


def main(argv):
    if len(argv) not in (3,5) or (len(argv)==5 and argv[3] not in ('--retain','--resume')):
        raise ValueError("usage: evaluate.py REQUEST RESPONSE [--retain|--resume STATE]")
    request, sum_bond, mode, options, frame = read_request(argv[1])
    retain=None
    if len(argv)==5:
        import forward_state
        if argv[3]=='--resume':
            if mode!=0:
                raise ValueError('TNVB handoff requires its original ENERGY request')
            result=forward_state.resume(argv[4],argv[1],request,sum_bond)
            mode=1
        else:
            if mode!=0:
                raise ValueError('TNVB handoff retention requires ENERGY')
            retain=dict(directory=argv[4])
            result=evaluate(request,sum_bond,mode,options,retain,frame)
    else:
        result=evaluate(request,sum_bond,mode,options,integral_frame=frame)
    write_response(argv[2], request, mode, result)
    if retain is not None:
        forward_state.publish(argv[4],argv[1],argv[2],retain)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except Exception:
        traceback.print_exc()
        raise SystemExit(99)
