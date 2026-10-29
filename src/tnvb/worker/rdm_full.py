"""Bounded-workspace reductions and optional profiling of the full SU(2) 3-PDM.

The native six-index density is evaluated for the supplied finite-bond MPS.
Only its two required reductions are normalized, so neither normalization nor
contraction copies the K**6 array.
"""
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np


def reduce_full_3pdm(density, integrals, norm):
    """Return C[d,c] and sum_k Gamma3[i,j,k,k,b,a], divided by norm.

    C[d,c] = sum_abef integrals[f,a,e,b] density[a,b,c,d,e,f]/norm.
    Inputs are read-only. Additional workspace is O(K**4), including a packed
    integral tensor and the particle-contraction result. Noncontiguous input
    uses a stride-aware einsum instead of copying the six-index array.
    """
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("full 3-PDM requires a positive finite state norm")
    k = integrals.shape[0]
    if (integrals.shape != (k,) * 4 or density.shape != (k,) * 6 or
            density.dtype != np.float64 or integrals.dtype != np.float64):
        raise ValueError("invalid full 3-PDM/integral dimensions or dtype")
    started = time.perf_counter()
    if density.flags.c_contiguous:
        # ab and ef are contracted; cd is the output. Each matrix-vector
        # product reads a contiguous K**4 slab, with no six-index transpose.
        weights = np.ascontiguousarray(integrals.transpose(1, 3, 2, 0))
        weights = weights.reshape(k * k, k * k)
        slabs = density.reshape(k * k, k * k, k * k)
        output = np.zeros(k * k, dtype=np.float64)
        for ab in range(k * k):
            output += slabs[ab] @ weights[ab]
        contracted = output.reshape(k, k).T.copy()
    else:
        contracted = np.einsum("faeb,abcdef->dc", integrals, density,
                               optimize=False)
    contracted /= norm
    integral_seconds = time.perf_counter() - started
    started = time.perf_counter()
    particle = np.einsum("ijkkba->ijba", density, optimize=False)
    particle /= norm
    particle_seconds = time.perf_counter() - started
    return contracted, particle, dict(integral_seconds=integral_seconds,
                                      particle_seconds=particle_seconds)


def get_profiled_3pdm(driver, state, algorithm, log):
    """Profile native phases only on explicit request; keep normal output short."""
    directory = os.environ.get("XMVB_BLOCK2_RDM_PROFILE", "").strip()
    if not directory:
        return driver.get_3pdm(state, algo_type=algorithm, iprint=0)
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="dm3-", suffix=".json", dir=path)
    os.close(descriptor)
    # Pybind methods in this Block2 build are invisible to cProfile's C-call
    # events. Time the enclosing Python source lines instead; native calls
    # remain unmodified and their wall times are charged to their call sites.
    function = driver.get_npdm.__func__
    code = function.__code__
    source, first = inspect.getsourcelines(function)
    lines = {first + index: value.strip() for index, value in enumerate(source)}
    timings = {}
    previous = [None, None]

    def trace(frame, event, arg):
        if frame.f_code is not code:
            return None
        now = time.perf_counter()
        if previous[0] is not None:
            number = previous[0]
            timings[number] = timings.get(number, 0.0) + now - previous[1]
        previous[:] = [frame.f_lineno, now] if event == "line" else [None, None]
        return trace

    previous_trace = sys.gettrace()
    started = time.perf_counter()
    sys.settrace(trace)
    try:
        return driver.get_3pdm(state, algo_type=algorithm, iprint=0)
    finally:
        sys.settrace(previous_trace)
        elapsed = time.perf_counter() - started
        stages = {}
        for label, text in (("graph", "pmpo.build("),
                            ("environment", "pme.init_environments("),
                            ("sweep", "expect.solve("),
                            ("assemble", "list(expect.get_npdm(")):
            stages[label] = sum(seconds for line, seconds in timings.items()
                                if text in lines.get(line, ""))
        stages["other"] = max(0.0, elapsed - sum(stages.values()))
        rows = [dict(line=line, source=lines.get(line, ""), seconds=seconds)
                for line, seconds in timings.items()]
        rows.sort(key=lambda row: row["seconds"], reverse=True)
        with open(name, "w") as stream:
            json.dump(dict(seconds=elapsed, phases=stages, source=code.co_filename,
                           lines=rows), stream, indent=2)
        log("[TNVB-3RDM-PROFILE] " + " ".join(
            f"{key}={value:.3f}s" for key, value in stages.items()) +
            f" file={name}")
