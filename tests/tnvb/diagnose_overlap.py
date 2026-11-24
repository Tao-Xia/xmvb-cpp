#!/usr/bin/env python3
"""Small-system overlap oracle; explicit expansion is TEST ONLY (K<=14).

Independent batched LU determinants distinguish finite-D paired-state errors
from the C++ cofactor reference, without changing either production backend.
Run with the configured Block2 Python/runtime and an exported C++ request.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np


def spin_string(occupied):
    sign = (-1) ** sum(a > b for i, a in enumerate(occupied)
                      for b in occupied[i + 1:])
    if len(set(occupied)) != len(occupied):
        raise ValueError("test oracle requires valid paired occupations")
    return sorted(occupied), sign


def expansion(pairs):
    choices = [(0,) if a == b else (0, 1) for a, b in pairs]
    alpha, beta, coefficients = [], [], []
    for selection in itertools.product(*choices):
        a, sa = spin_string([p[c] for p, c in zip(pairs, selection)])
        b, sb = spin_string([p[1 - c] for p, c in zip(pairs, selection)])
        alpha.append(a)
        beta.append(b)
        coefficients.append(sa * sb)
    return np.asarray(alpha), np.asarray(beta), np.asarray(coefficients)


def exact_overlap(structures, overlap, nelec, total_elec):
    inactive = (total_elec - nelec) // 2
    terms = [expansion((s[2 * inactive:] - inactive - 1).reshape((-1, 2)))
             for s in structures.T]
    result = np.empty((len(terms), len(terms)))
    for i, (ai, bi, ci) in enumerate(terms):
        for j, (aj, bj, cj) in enumerate(terms[:i + 1]):
            av = overlap[ai[:, None, :, None], aj[None, :, None, :]]
            bv = overlap[bi[:, None, :, None], bj[None, :, None, :]]
            value = np.linalg.det(av) * np.linalg.det(bv)
            result[i, j] = result[j, i] = ci @ value @ cj
    norms = np.sqrt(np.diag(result))
    return result / norms[:, None] / norms[None, :]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request")
    parser.add_argument("worker")
    parser.add_argument("report")
    parser.add_argument("--bond", type=int, help="test-only bond sensitivity; original input is not changed")
    parser.add_argument("--pair-order", choices=["original", "right-endpoint"], default="original",
                        help="test commuting pair-product order without changing the CSF")
    args = parser.parse_args()
    content = Path(args.request).read_bytes()
    if content[:8] != b"CPTNREQ3":
        raise ValueError("bad C++ request")
    version, n, k, nae, nel, d, s, mode, solver, nguess = struct.unpack("<10i", content[8:48])
    if version != 3 or solver or nguess or not (0 < k <= 14) or nae % 2 or nel < nae:
        raise ValueError("oracle restricted to valid small singlet requests")
    if args.bond is not None:
        if not 1 <= args.bond <= 400:
            raise ValueError("diagnostic bond must lie in [1,400]")
        d = args.bond
    structures = np.frombuffer(content, dtype="<i4", count=n * nel,
                               offset=64).reshape((nel, n), order="F")
    overlap = np.frombuffer(content, dtype="<f8", count=k * k,
                           offset=64 + 4 * n * nel).reshape((k, k), order="F")
    reference = exact_overlap(structures, overlap, nae, nel)
    if args.pair_order == "right-endpoint":
        structures = structures.copy()
        first_active = nel - nae
        for istr in range(n):
            pairs = np.sort(structures[first_active:, istr].reshape((-1, 2)), axis=1)
            order = np.lexsort((pairs[:, 0], pairs[:, 1]))
            structures[first_active:, istr] = pairs[order].ravel()
        content = (content[:64] + np.asarray(structures, dtype="<i4").ravel(order="F").tobytes()
                   + content[64 + 4 * n * nel:])
    with tempfile.TemporaryDirectory(prefix="tnvb-overlap-oracle-") as directory:
        directory = Path(directory)
        request, response = directory / "request.bin", directory / "response.bin"
        request.write_bytes(content[:8] + struct.pack("<10i", version, n, k,
                            nae, nel, d, s, 0, 0, 0) + content[48:])
        env = os.environ.copy()
        env["XMVB_BLOCK2_SCRATCH_PARENT"] = str(directory)
        env["XMVB_BLOCK2_WARM_CACHE"] = str(directory / "warm")
        subprocess.run([sys.executable, str(Path(args.worker).resolve()),
                        str(request), str(response)], check=True, env=env, timeout=180)
        data = response.read_bytes()
    if data[:8] != b"CPTNRES3" or struct.unpack("<8i", data[8:40]) != (3, 0, n, k, 0, n, 0, 0):
        raise ValueError("invalid ENERGY response")
    tensor = np.frombuffer(data, dtype="<f8", count=n * n,
                           offset=40 + 8 * n * n).reshape((n, n), order="F")
    norms = np.sqrt(np.diag(tensor))
    tensor = tensor / norms[:, None] / norms[None, :]
    difference = float(np.max(np.abs(tensor - reference)))
    report = dict(S_error=difference, reference_S=reference.tolist(),
                  tnvb_S=tensor.tolist(), structure_bond=d,
                  pair_order=args.pair_order,
                  pair_settings={name: os.environ.get(name, default) for name, default in
                      [("XMVB_BLOCK2_SU2_SWEEPS", "4"),
                       ("XMVB_BLOCK2_SU2_TOL", "1e-10"),
                       ("XMVB_BLOCK2_SU2_CUTOFF", "1e-14")]},
                  oracle="explicit paired CSF / batched LU (test only)")
    Path(args.report).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
