# TNHVP model and accuracy audit

Date: 2026-09-20. Audited source: `1664152`.

This is an evidence-backed design review, not a claim that the proposed repairs
have been implemented. The intended method remains matrix-free orbital Newton
correction of an orbital-block-preconditioned L-BFGS predictor, with variational
structure coefficients eliminated through their response equations.

## 1. What the multi-system experiment establishes

Slurm job `246953`, Hanhai25 `anode001`, 32 OpenMP threads per run, one BLAS
thread, exact LIBCINT integrals, Davidson, and common outer tolerances
$\tau_g=10^{-3}$ and $\tau_E=10^{-7}\ E_h$. No 30-step cutoff was imposed.
These are single-run diagnostics, not repeated publication timings. Two method
jobs shared a node; shared-node timing interference has not been excluded.

| System | TNHVP accepted steps/status | Standard L-BFGS steps | TNHVP SCF time / s | L-BFGS SCF time / s |
|---|---:|---:|---:|---:|
| 240 | 6, converged | 55 | 9.57 | 17.79 |
| 241 | 3, curvature-symmetry failure | 94 | 0.37 | 4.17 |
| MnF2 | 18, converged | 479 | 86.23 | 25.28 |
| FeCl2 | 5, converged | 13 | 15.47 | 3.68 |
| 7963 | 6, converged | 61 | 1.79 | 4.52 |
| 7975 | 4, curvature-symmetry failure | 57 | 0.86 | 4.11 |
| YAMSAI | 6, converged | 60 | 1.22 | 3.20 |
| CERRAS | 4, curvature-symmetry failure | 129 | 25.38 | 137.26 |
| LOFLEA | 6, curvature-symmetry failure | 60 | 539.82 | 228.78 |

Remote records are under
`/home/guqqgroup/taoxia/xmvb-runs/final-uncapped-1664152/results/`;
the audit copy is under `build/diagnostics/final_uncapped_audit/results/`.
Failure times include work performed after the last accepted iterate. Accepted
trace counters do not account for the final failed solve.

The comparison does **not** isolate exact-curvature enrichment. In
`optimization/backends/lbfgs.cpp`, standalone L-BFGS uses `ScaledIdentity`;
`backends/truncated_newton.cpp` uses `OrbitalBlock`. Commit `d0e07e3` deliberately
introduced this distinction. Its isolated results changed F2 from 8 to 26
steps and 7975 from 9 to 57 by changing the initial inverse alone. A third
experimental arm, orbital-block L-BFGS without HVP correction, is necessary.

For MnF2, steps 12, 13, 16, and 17 consume 64.26 s of 86.15 s recorded outer
iteration time and 503 of 779 HVP block calls. Their recorded KKT residuals
meet the requested forcing threshold. Thus the observed expense is not proof
that the solver is stuck below an unattainable floating-point residual floor.
The full run spends 54.02 s inside HVPs, 0.10 s in accepted-point setup, and
0.99 s in trial objectives. Approximately 31.04 s remains outside those
counters; more detailed profiling is required to attribute that remainder.

LOFLEA's fifth and sixth accepted steps cost 166.43 and 278.96 s, with sampled
dimensions 27 and 63. Both meet the implemented KKT threshold, while the sixth
step increases the gradient norm. The final curvature-symmetry failure is a
separate event during the next solve.

## 2. One step must have one model

Let $g$ be the reduced gradient covector, $G$ the physical tangent metric, and
$M$ the transported orbital-block L-BFGS inverse action. The predictor is

$$
p_B=-Mg.
$$

A quadratic step model with Hessian action $\widetilde H$ predicts

$$
\operatorname{pred}(p)=-g^Tp-\frac12p^T\widetilde Hp,
\qquad p^TGp\leq\Delta^2.
$$

The current baseline Armijo branch instead records $-g^Tp$ as the predicted
decrease, then feeds its ratio into the Newton trust-radius update. The
baseline is not constrained by that radius. For an exact quadratic and its
Newton minimizer, actual decrease is $-g^Tp/2$: the linear ratio is $1/2$,
whereas the correct quadratic agreement ratio is one. Even a perfect quadratic
model can therefore be treated as inaccurate by this mixed rule.

Measured examples: MnF2 step 6 contracts the radius from 0.454 to 0.0104 after
a baseline step; steps 13 and 17 contract it from 0.735 to 0.00306 and from
0.0920 to 0.00398 after a rejected Newton trial followed by a baseline step.
FeCl2 step 1 contracts it from one to $4.63\times10^{-4}$. These observations
implicate the control rule; an A/B repair is still needed to measure causality.

The rejected Newton trial and subsequent accepted baseline also share one
mutable trial record. The latter overwrites the former before updating the
radius. Preserve both records. Update quadratic-model trust using the actual
quadratic trial and its numerical uncertainty. A pure Armijo step provides
descent evidence but does not by itself measure quadratic-model fidelity.
After a rejected curvature trial, use the retained same-point subspace to test
the revised radius before rebuilding expensive HVP information.

Current curvature admission uses the *previous* nonlinear gradient contraction.
A successful Newton step can therefore switch curvature off at the next point.
That policy does not certify the current predictor, nor establish consecutive
local Newton convergence. Current-point defect information is required whenever
a Newton certificate is claimed. A cheap symmetric approximate model can
provide that information without an independent accurate response for every
sampled direction.

## 3. Response residual, HVP error, and symmetry

Work in the gauge-fixed structure-response domain. Absorb normalization and
overlap derivative terms into the consistent orbital and coupling blocks.
With symmetric invertible response operator $D$ on this domain,

$$
H=A-B^TD^{-1}B,
\qquad Dz(p)=-Bp.
$$

An approximate response $\widetilde z$ has true residual

$$
e_c=D\widetilde z+Bp,
\qquad
\widetilde Hp=Ap+B^T\widetilde z,
\qquad
\widetilde Hp-Hp=B^TD^{-1}e_c.
$$

A small relative backward residual for $D$ does not independently bound the
orbital HVP error. The transfer through $D^{-1}$ and $B^T$ matters. Currently
`core/contracts/eigensolver.hpp` derives a response tolerance from outer energy
and gradient tolerances, without this transfer estimate. In
`derivatives/hessian/context/response_cache.cpp`, a requested tolerance is
combined with the default using `max`, so it cannot request a more accurate
response than the default. The orbital KKT certificate contains no response
error allowance.

Finite-tolerance, independently stopped response solves with mutable recycled
starting spaces do not define one fixed linear self-adjoint map. Current
recycling accelerates independent solves; it is not a common response model.
For sample columns $s_i$, put $b_i=Bs_i$ and
$D\widetilde z_i=-b_i+e_i$. Then

$$
b_i^T\widetilde z_j-b_j^T\widetilde z_i
=\widetilde z_j^Te_i-\widetilde z_i^Te_j.
$$

This identity gives a direct diagnostic: measure the skew, predict the part
explained by the true response residuals, and examine any remaining error in
the adjoint coupling, coordinate pullback, or fixed-state Hessian. Compare
repeated calls, column permutations, block/scalar calls, additivity, and dense
response at the *same failing accepted point*. Current logs do not contain
the failed block's skew or residuals. Response truncation is a plausible
mechanism, not yet a demonstrated explanation of all four failures.

For exact block inverse-BFGS secants $MY=S$ with symmetric $M$, symmetry of
$S^TY$ is necessary: $S^TY=Y^TMY$. The current trust projection symmetrizes
$Q^TY$, but positive Ritz extraction retains the raw images $Yv$. Feeding
these into the block update reintroduces their incompatibility. Merely relaxing
the assertion or symmetrizing the small curvature matrix does not make the
raw secants satisfy the defining block identity.

### A coherent response model

For a common response space $W$ frozen over an orbital block, define

$$
H_W=A-B^TW(W^TDW)^{-1}W^TB.
$$

This is a symmetric linear operator if the component adjoints and projected
solve are consistent. Its images may be used together in a block update.
Near selected-state degeneracies the response domain must exclude internal
gauge rotations; equal-weight state averaging must retain its correct
selected-subspace formulation. An indefinite projected operator requires a
symmetric-indefinite solve, not an unjustified Cholesky factorization.

For enrichment $W_+=[W,U]$, define

$$
K=W^TDW,\quad C=W^TDU,\quad F=U^TDU-C^TK^{-1}C,
\quad \widehat B=U^TB-C^TK^{-1}W^TB.
$$

For nonsingular projected blocks, the exact update is

$$
H_{W_+}=H_W-\widehat B^TF^{-1}\widehat B.
$$

Old sampled images can be refreshed by this low-rank correction. They must not
be mixed across response-model versions without refresh. This reduces repeated
response solves only when the required response rank remains small; it does
not lower the cost of every other orbital-derivative kernel automatically.
Full response-space matrices and the orbital Hessian remain unassembled.
Memory is $O(n_s r+n_q m)$ for response rank $r$ and orbital rank $m$; these
ranks must be measured, since allowing either to reach full dimension loses
the intended bounded-memory advantage.

As a small algebraic check, shared-model symmetry and the enrichment identity
were verified with exact rational arithmetic for both a positive-definite and
an indefinite three-dimensional response operator, two orbital coordinates,
and a one-column enrichment. This checks the identities, not their molecular
accuracy or performance.

## 4. Different certificates for globalization and local Newton convergence

Let $\widetilde r=g+\widetilde Hp+\lambda Gp$. If
$\epsilon_H(p)$ bounds the action error in a covector norm, then

$$
\|g+Hp+\lambda Gp\|_*
\leq \|\widetilde r\|_*+\epsilon_H(p).
$$

Uncertainty in the computed gradient adds another term. A local Newton
certificate must apply to this total, not only the stored linear combination
of approximate images. For $p=Qz$ and image errors $E$, the residual gap is
$Ez$; changing response precision requires refreshing/certifying the relevant
images or evaluating the combined action on $p$ independently.

The metric-dual norm is $\|r\|_{G^{-1}}$. A practical covariant alternative is
the fixed SPD inverse-model norm $\|r\|_M=(r^TMr)^{1/2}$, provided equivalence
and conditioning are monitored. No dense global metric whitening is required.
Covectors become search vectors through an inverse/Riesz map; appending the
raw coordinate residual as a tangent vector is coordinate dependent.

During globalization, a useful trust step needs certified sufficient model
decrease and actual energy acceptance; it need not minimize the whole
trust-region subproblem to the accuracy of an interior Newton equation.
A preconditioned Cauchy/predictor decrease supplies the reference. This is the
standard sufficient-decrease distinction in trust-region theory [1]. HVP
uncertainty affects model decrease by at most

$$
\epsilon_q(p)\leq\frac12\|p\|_G\epsilon_H(p)
$$

when $\epsilon_H$ uses the metric-dual norm. It must be smaller than the
decision margin used for acceptance; an uncertain decision calls for numerical
refinement, not arbitrarily many new orbital directions.

In the local interior regime, $\lambda=0$ and the untruncated candidate obeys

$$
\|g+\widetilde Hp\|_*+\epsilon_H(p)
\leq\eta_k\|g\|_*.
$$

With the usual local smoothness, nonsingular positive Hessian on the physical
quotient, consistent retraction/pullback, and sufficiently accurate gradients,
$\eta_k\to0$ supports superlinear convergence;
$\eta_k=O(\|g_k\|_*)$ supports quadratic convergence [2]. The gradient
linearization error, computed after consistent covector transport, should
inform forcing. Distance traveled from the initial gradient alone does not
establish that the current model is accurate.

The existing forcing contains a positive floor
$\sqrt{\tau_g/\|g_0\|_2}$ at fixed requested tolerance. Consequently its
formula is not a vanishing-forcing proof. Moreover, the previous-step
contraction admission can alternate Newton and uncertified baseline steps.
The currently advertised quadratic-convergence argument is incomplete.

For boundary correction the actual operator is $H+\lambda G$, whereas the
current inverse enrichment samples $H$. Shift-aware correction preconditioning
is needed when $\lambda$ dominates the sampled curvatures. Good preconditioning
and sufficient global decrease address the cost without a molecule-dependent
HVP cap. A small shifted stationarity residual is not a global second-order
certificate: unsampled negative curvature and dual feasibility remain separate
questions. A minimum sampled Ritz value is not a certified global lower bound.

## 5. Remove repeated dense work inside a matrix-free algorithm

Every two-column expansion currently reconstructs the projected Gram matrices
and performs full projected eigendecompositions for both the trust solve and
positive-secant selection. For final sampled rank $m$, fixed block width, and
reduced dimension $n_q$, this entails

$$
\sum_{j\leq m}O(n_qj^2)=O(n_qm^3),
\qquad
\sum_{j\leq m}O(j^3)=O(m^4).
$$

Incremental Gram updates reduce their cumulative work to $O(n_qm^2)$.
Spectral information and factorizations should be reused, with the local SPD
phase using an iterative correction rather than repeated full spectral solves.
Such changes must preserve the action-error and trust certificates above.
The measured unaccounted solver time is consistent with this source-level
cost, but component profiling is required before assigning all of it here.

Trace repair is also required: `minimum_ritz_value` is currently fabricated
from minus the spectral radius whenever negative curvature is flagged. Record
the actual minimum and the inner stop reason. Distinguish the rejected Newton
model certificate from the accepted baseline step, and retain failure-point
diagnostics rather than only accepted-step counters.

## 6. Acceptance and remaining uncertainties

The next tests must isolate: baseline quality, quadratic globalization,
response-operator consistency, and local convergence order. First reproduce
the failing accepted point and check the skew identity. Then change one
interface at a time and rerun the same panel. Include a dense small-system
Newton oracle, nonorthogonal coordinate changes, a nontrivial SPD quadratic,
and an indefinite boundary problem. Directly test no false radius contraction
for a consistent exact quadratic and no precision-history dependence of a
frozen response model.

Identical adjacent-energy and gradient thresholds do not mathematically bound
the error in the final total energy. Soft curvatures or distinct stationary
points can give different energies. For MnF2 and FeCl2 the present endpoint
differences are about $1.3\times10^{-5}$ and $1.1\times10^{-5}\ E_h$; diagnose
both endpoints with common objective/gradient evaluation and cross-restarts
before assigning these differences to a tolerance or an energy bug. Likewise,
LOFLEA's lower but unconverged TNHVP energy is not a validated improved minimum.

Success requires fewer outer steps and competitive total wall time against
the *orbital-block* L-BFGS baseline, failure-free convergence across the panel,
and an explicit explanation of endpoint differences. Same-point dominance of
a curvature candidate cannot prove fewer total outer iterations, since the
subsequent paths and secant histories diverge. No universal speedup or a fixed
ten-step convergence claim is justified.

## 9. Orbital-block baseline qualification

Commit `4e118d0` exposes `--lbfgs-initial-inverse orbital-block` without
duplicating the L-BFGS backend or changing its default scalar initialization.
The ablation and TNHVP predictor use the same inverse-action enum, local
orbital blocks, and transported secants. The ablation performs no HVPs.

Hanhai25 Slurm job `246984` used 32 OpenMP threads, one BLAS thread,
Davidson, exact integrals, and the same outer thresholds as Section 1.
Each case had an isolated input/output directory and an explicit 2000-step
limit. All nine process exits were zero and all reported dual-tolerance
convergence. These are single-run SCF timings, not publication benchmarks.

| System | Accepted steps | SCF time / s | Final energy / $E_h$ |
|---|---:|---:|---:|
| 240 | 10 | 2.970417 | -343.446302350844 |
| 241 | 24 | 1.346254 | -230.720590368221 |
| MnF2 | 102 | 6.861466 | -1348.893352807651 |
| FeCl2 | 7 | 3.119502 | -2181.617634827448 |
| 7963 | 15 | 1.139692 | -285.731743343953 |
| 7975 | 9 | 0.639360 | -285.733616926419 |
| YAMSAI | 11 | 0.593832 | -305.561919466566 |
| CERRAS | 26 | 33.716539 | -397.093434999630 |
| LOFLEA | 24 | 72.605682 | -399.106646148841 |

The comparison is now correctly defined, but TNHVP acceptance remains open.
In particular, its step savings alone do not establish a wall-time benefit
over this stronger baseline. Endpoint agreement still requires the checks in
Section 8. Logs reside under
`build/diagnostics/consistency-3460c35/ablation/`; remote provenance is
`/home/guqqgroup/taoxia/xmvb-runs/consistency-3460c35/`.

## References

1. Conn, A. R.; Gould, N. I. M.; Toint, P. L. *Trust Region Methods*,
   Chapter 7, The Trust-Region Subproblem.
   https://doi.org/10.1137/1.9780898719857.ch7
2. Eisenstat, S. C.; Walker, H. F. Choosing the Forcing Terms in an Inexact
   Newton Method. *SIAM J. Sci. Comput.* **1996**, *17*, 16--32.
   https://doi.org/10.1137/0917003
3. Simoncini, V.; Szyld, D. B. Theory of Inexact Krylov Subspace Methods and
   Applications to Scientific Computing. *SIAM J. Sci. Comput.* **2003**,
   *25*, 454--477. https://doi.org/10.1137/S1064827502406415

The shared response-model and image-refresh formulas above are algebraic
derivations for this implementation. Their performance benefit still requires
measurement; the references support the surrounding convergence/error framework.
