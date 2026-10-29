"""Lowest real generalized eigenpair using only joint H/S actions.

The metric must be positive definite on the *entire* supplied structure space.
No structure removal, dense Hamiltonian fallback or finite-bond sigma fitting.
Only the small Ritz problem is diagonalized; restart retains low Ritz vectors.
"""
import numpy as np
import scipy.linalg


def solve(action, hdiag, sdiag, energy_tolerance=1e-7,
          gradient_tolerance=1e-3, initial=None, max_space=32,
          max_iterations=1000):
    hdiag, sdiag = np.asarray(hdiag, float), np.asarray(sdiag, float)
    n = hdiag.size
    if (hdiag.shape != (n,) or sdiag.shape != (n,) or n == 0 or
            not np.all(np.isfinite(hdiag)) or not np.all(np.isfinite(sdiag)) or
            np.any(sdiag <= 0)):
        raise ValueError('invalid Davidson diagonals')
    if not all(np.isfinite(t) and t > 0 for t in (energy_tolerance, gradient_tolerance)):
        raise ValueError('invalid Davidson accuracy')
    if max_space < 3 or max_iterations < 1:
        raise ValueError('invalid Davidson workspace/iteration limit')
    max_space = min(n, max_space)
    # Internal CI accuracy is tighter than the orbital stopping conditions.
    # It does not change those conditions. Measure in normalized-structure
    # coordinates so independent structure rescaling cannot relax accuracy.
    tolerance = min(1e-10, energy_tolerance, gradient_tolerance)
    norms = np.sqrt(sdiag)
    calls, restarts, peak = 0, 0, 0

    def apply(v):
        nonlocal calls
        hv, sv = action(v)[:2]
        hv, sv = np.asarray(hv, float), np.asarray(sv, float)
        calls += 1
        if hv.shape != (n,) or sv.shape != (n,) or not (
                np.all(np.isfinite(hv)) and np.all(np.isfinite(sv))):
            raise RuntimeError('Davidson action returned invalid data')
        return hv, sv

    v, hv, sv = [], [], []

    def append(candidate):
        x = np.asarray(candidate, float).copy()
        if x.shape != (n,) or not np.all(np.isfinite(x)):
            raise ValueError('invalid Davidson initial/correction vector')
        # Scale before applying a potentially ill-conditioned metric.
        scale = np.linalg.norm(norms*x)
        if scale == 0:
            return False
        x /= scale
        hx, sx = apply(x)
        initial_norm = float(x@sx)
        if initial_norm <= 0:
            raise RuntimeError('Davidson encountered a nonpositive metric')
        for _ in range(2):
            if v:
                weights = np.column_stack(v).T@sx
                x -= np.column_stack(v)@weights
                hx -= np.column_stack(hv)@weights
                sx -= np.column_stack(sv)@weights
        norm = float(x@sx)
        if norm < -1e-12*initial_norm:
            raise RuntimeError('Davidson metric orthogonalization became indefinite')
        if norm <= 1e-24*initial_norm:
            return False
        factor = np.sqrt(norm)
        v.append(x/factor); hv.append(hx/factor); sv.append(sx/factor)
        return True

    if initial is not None:
        append(initial)
    seed = np.zeros(n); seed[int(np.argmin(hdiag/sdiag))] = 1.0
    if len(v) < max_space:
        append(seed)
    # Avoid an exactly invariant excited-state seed, including a warm start.
    rng = np.random.default_rng(187923)
    if len(v) < max_space:
        append(rng.normal(size=n)/norms)
    for iteration in range(1, max_iterations+1):
        V, H, S = map(np.column_stack, (v, hv, sv))
        peak = max(peak, len(v))
        hp, sp = V.T@H, V.T@S
        scale = max(1.0, float(np.max(np.abs(hp))))
        if (np.max(np.abs(hp-hp.T)) > 1e-10*scale or
                np.max(np.abs(sp-sp.T)) > 1e-10):
            raise RuntimeError('Davidson action is not self-adjoint')
        values, y = scipy.linalg.eigh((hp+hp.T)*.5, (sp+sp.T)*.5)
        energy = float(values[0])
        c, hc, sc = V@y[:,0], H@y[:,0], S@y[:,0]
        residual = hc-energy*sc
        stationarity = float(np.linalg.norm(residual/norms))
        relative = stationarity/max(1.0, np.linalg.norm(hc/norms)+abs(energy)*np.linalg.norm(sc/norms))
        if stationarity <= tolerance and relative <= tolerance:
            # A fresh native action catches cache/gauge/nonlinear inconsistencies.
            hc, sc = apply(c)
            normalization = float(c@sc)
            if abs(normalization-1.0) > 1e-9:
                raise RuntimeError('Davidson final metric norm failed')
            energy = float(c@hc/normalization)
            stationarity = float(np.linalg.norm((hc-energy*sc)/norms))
            relative = stationarity/max(1.0, np.linalg.norm(hc/norms)+abs(energy)*np.linalg.norm(sc/norms))
            if stationarity > tolerance or relative > tolerance:
                raise RuntimeError('Davidson fresh-action residual failed')
            return energy, c, dict(iterations=iteration, block_actions=calls,
                peak_subspace_dimension=peak, maximum_relative_residual=relative,
                maximum_stationarity_residual=stationarity, restarts=restarts)
        if len(v) == n:
            raise RuntimeError('Davidson full subspace residual failed; check numerical conditioning')
        denominator = hdiag-energy*sdiag
        floor = max(1e-8, 1e-4*float(np.max(np.abs(denominator/sdiag))))*sdiag
        denominator = np.where(np.abs(denominator)<floor,
                               np.where(denominator<0, -floor, floor), denominator)
        correction = -residual/denominator
        if len(v) >= max_space:
            keep = min(4, max(1, max_space//2))
            # Cached H/S columns are transformed together, preserving linearity.
            v = list((V@y[:,:keep]).T.copy())
            hv = list((H@y[:,:keep]).T.copy())
            sv = list((S@y[:,:keep]).T.copy())
            restarts += 1
        if not append(correction):
            if not append(rng.normal(size=n)/norms):
                raise RuntimeError('Davidson correction space stagnated')
    raise RuntimeError(f'Davidson failed to converge in {max_iterations} iterations')
