#!/usr/bin/env python3
"""Test-only bridge to a specified frozen old worker; never used in production.

Use the configured Block2 Python and runtime for both subprocesses. The input
is exported by export_tnvb_request. This test adapter converts the orthogonal
request and density response to the frozen worker's raw-orbital convention.
"""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np


def array(stream, count):
    data = stream.read(count * 8)
    if len(data) != count * 8:
        raise ValueError("truncated interop response")
    return np.frombuffer(data, dtype="<f8").copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request")
    parser.add_argument("new_worker")
    parser.add_argument("old_worker")
    parser.add_argument("report")
    args = parser.parse_args()
    content = Path(args.request).read_bytes()
    if content[:8] != b"CPTNREQ3":
        raise ValueError("bad exported request")
    version, n, k, nae, nel, d, s, mode, solver, nguess = struct.unpack("<10i", content[8:48])
    if version != 3 or mode != 1 or solver or nguess:
        raise ValueError("requires a gradient request")
    start=64+4*n*nel
    overlap=np.frombuffer(content,dtype='<f8',count=k*k,offset=start).reshape((k,k),order='F')
    frame=np.frombuffer(content,dtype='<f8',count=k*k,offset=start+8*k*k).reshape((k,k),order='F')
    h=np.frombuffer(content,dtype='<f8',count=k*k,offset=start+16*k*k).reshape((k,k),order='F')
    packed=np.frombuffer(content,dtype='<f8',offset=start+24*k*k)
    def pair(i,j):
        a,b=max(i,j),min(i,j)
        return a*(a+1)//2+b
    g=np.empty((k,)*4)
    for i,j,a,b in np.ndindex((k,)*4):g[i,j,a,b]=packed[pair(pair(i,j),pair(a,b))]
    raw_g=np.einsum('ia,jb,kc,ld,ijkl->abcd',frame,frame,frame,frame,g,optimize=True)
    raw_packed=np.empty_like(packed)
    for i,j,a,b in np.ndindex((k,)*4):raw_packed[pair(pair(i,j),pair(a,b))]=raw_g[i,j,a,b]
    raw_payload=content[64:start]
    for value in (overlap,frame.T@h@frame,raw_packed):
        raw_payload+=np.asarray(value,dtype='<f8').ravel(order='F').tobytes()
    with tempfile.TemporaryDirectory(prefix="cpp-tnvb-interop-") as folder:
        folder = Path(folder)
        old_request = folder / "old.request"
        old_request.write_bytes(b"XMB2RD1\0" + struct.pack("<11i", 7, n, k, nae, nel, d, s, 0, 0, 0, 0) + raw_payload)
        responses = [folder / "new.response", folder / "old.response"]
        env = os.environ.copy()
        env["XMVB_BLOCK2_RDM_CHECK_TOL"] = "1e-8"
        env["XMVB_BLOCK2_WORKERS"] = "1"
        for worker, request, response in zip(
                [args.new_worker, args.old_worker], [args.request, old_request], responses):
            # Per-worker scratch/cache isolation, including separate total MPSs.
            with tempfile.TemporaryDirectory(prefix="worker-", dir=folder) as scratch:
                env["XMVB_BLOCK2_SCRATCH_PARENT"] = scratch
                env["XMVB_BLOCK2_WARM_CACHE"] = str(Path(scratch) / "warm")
                subprocess.run([sys.executable, str(Path(worker).resolve()), str(Path(request).resolve()), str(response)], env=env, check=True)
        with responses[0].open("rb") as stream:
            if stream.read(8) != b"CPTNRES3":
                raise ValueError("bad new response")
            header = struct.unpack("<8i", stream.read(32))
            if header != (3, 0, n, k, 1, n, 0, 0):
                raise ValueError("bad new response dimensions")
            hn, sn = array(stream, n*n), array(stream, n*n)
            en, cn = array(stream, n), array(stream, n*n).reshape((n,n), order="F")[:,0]
            diagnostics = array(stream, 4)
            pn, pin, gn = array(stream, k*k), array(stream, k**4), array(stream, k*k)
            if stream.read(1):
                raise ValueError("trailing new response")
        q=np.linalg.inv(frame)
        pn=(q@pn.reshape((k,k),order='F')@q.T).ravel(order='F')
        pin=np.einsum('ia,jb,kc,ld,abcd->ijkl',q,q,q,q,
                      pin.reshape((k,)*4,order='F'),optimize=True).ravel(order='F')
        gn=(q@gn.reshape((k,k),order='F')@q.T).ravel(order='F')
        with responses[1].open("rb") as stream:
            if stream.read(8) != b"XMB2RO1\0" or struct.unpack("<2i", stream.read(8)) != (7, 0):
                raise ValueError("bad old response")
            old_fit = array(stream, 1)[0]
            ho, so = array(stream, n*n), array(stream, n*n)
            eo, co = array(stream, 1)[0], array(stream, n)
            po, pio, go = array(stream, k*k), array(stream, k**4), array(stream, k*k)
            if stream.read(1):
                raise ValueError("trailing old response")
        if np.dot(cn, co) < 0:
            cn = -cn
        error = {name: float(np.max(np.abs(a-b))) for name, a, b in [
            ("H",hn,ho), ("S",sn,so), ("coefficients",cn,co),
            ("P",pn,po), ("Pi",pin,pio), ("D3G",gn,go)]}
        error["energy"] = float(abs(en[0]-eo))
        limits = dict(H=1e-8, S=1e-10, coefficients=1e-8, P=1e-8, Pi=1e-8, D3G=1e-7, energy=1e-9)
        report = dict(passed=all(np.isfinite(value) and value <= limits[key] for key, value in error.items()),
                      errors=error, limits=limits, new_fit=float(diagnostics[0]), old_fit=float(old_fit),
                      old_worker=str(Path(args.old_worker).resolve()))
        Path(args.report).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, allow_nan=False))
        return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
