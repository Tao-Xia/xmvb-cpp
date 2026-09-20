# TNHVP model and accuracy audit

Date: 2026-09-20. Audited source: `1664152`.

Sections 1--6 record the original design review. Sections 7 onward record
subsequent implementation and validation; uncompleted repairs remain explicit.
The intended method remains matrix-free orbital Newton
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

## 7. Orbital-block baseline qualification

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
Section 6. Logs reside under
`build/diagnostics/consistency-3460c35/ablation/`; remote provenance is
`/home/guqqgroup/taoxia/xmvb-runs/consistency-3460c35/`.

## 8. Fixed-point response diagnosis

The unmodified optimizer was reproduced in Hanhai25 job `246977`: Davidson
failed after 3 accepted steps for 241 and after 4 for 7975, whereas dense
converged in 9 and 6 steps, respectively. Orbital snapshots from the Davidson
trajectories were then held fixed in job `246988`.

The diagnostic `audit_newton_step --audit-operator true` records each HVP
block even if the subproblem subsequently rejects its block secants. It
replays the same directions cold/warm, in one block, and in reverse order;
tests additivity; and isolates the structure-response contribution.
`--audit-dense-reference true` explicitly opts into an additional dense
calculation at the **same orbitals and on the same reduced directions and
chart**. This is a diagnostic, never a production dense fallback. Failed
subproblem exits remain nonzero after printing the diagnostics.

At the captured 241 point, the audit generated three directions. Its raw
block was rejected with skew norm $1.07370\times10^{-6}$ versus the actual
test limit $4.72808\times10^{-7}$. The following are direct observations:

| Quantity | Measured value |
|---|---:|
| Cold replay relative projected skew | $1.01517\times10^{-7}$ |
| Cold/warm HVP relative difference | $2.27261\times10^{-7}$ |
| Warm replay relative projected skew | $1.22963\times10^{-15}$ |
| Original block segmentation vs. one-block HVP difference | $6.42291\times10^{-7}$ |
| One-block full projected skew norm | $6.45785\times10^{-7}$ |
| Same block, structure response disabled, skew norm | $7.56146\times10^{-14}$ |
| Isolated structure-response skew norm | $6.45786\times10^{-7}$ |
| Dense full response on identical directions, skew norm | $1.99896\times10^{-14}$ |
| Cold additivity defect, relative | $4.14958\times10^{-6}$ |
| Davidson directional finite-difference HVP defect, relative | $3.94505\times10^{-6}$ |
| Dense directional finite-difference HVP defect, relative | $6.08507\times10^{-9}$ |

Finite differences used a step of $10^{-4}$ in the recorded reduced chart.
The cold response's maximum relative equation residual was
$5.39777\times10^{-6}$. That scalar is **not** an HVP error bound; the
response-to-orbital error amplification in Section 3 still applies. The
warm diagnostic's residual maximum is cumulative and must not be described
as the final warm block's residual. Reversing the single block gave no
measurable difference in this experiment: history and block segmentation,
not column order, are the demonstrated dependencies.

The 7975 fixed-point audit stopped with a one-direction subspace, so its
zero projected skew cannot test symmetry. It nevertheless showed a cold
additivity defect of $4.39676\times10^{-7}$ and a Davidson finite-difference
defect of $3.22842\times10^{-7}$, versus $1.18208\times10^{-9}$ in the
separate dense audit. Do not describe that audit as reproducing the exact
7975 production secant failure: it used the same accepted point but not
the full production secant history/forcing sequence.

These observations isolate the 241 incompatibility to the inexact structure
response, rather than showing an intrinsic asymmetry of the orbital
derivatives. They support the common symmetric response-model construction
in Section 3, not loosening the strict block-BFGS check or claiming universal
derivative correctness from two probes. Logs are under
`build/diagnostics/consistency-3460c35/audit/`.

## 9. Separating line-search acceptance from quadratic model trust

The predictor line search observes the actual decrease
$a_B=f(x)-f(R_x(\alpha p_B))$ and the linear decrease
$\ell_B=-\alpha g^T p_B$. Armijo requires $a_B\ge c_1\ell_B$.
Neither that inequality nor the value of $\ell_B$ supplies the unmeasured
quadratic curvature $p_B^THp_B$.

The implementation now represents these observations separately. Only a
Newton trial with measured prediction

$$
q_N=-g^Tp_N-\frac12p_N^THp_N
$$

can update the quadratic trust radius. If no such trial was evaluated,
$\Delta_{k+1}=\Delta_k$. If a Newton trial is rejected and the predictor
is subsequently accepted, the radius decision still uses the **rejected
Newton trial's** actual decrease and prediction. The predictor never
overwrites those data. A failed Armijo search is attempted only once per
accepted point, because changing the Newton radius cannot change its
radius-independent search direction or backtracking sequence.

For $f(x)=50x^2$ at $x=0.01$, a backtracked predictor step $p=-1/64$
satisfies Armijo with $a_B=0.00341796875$ and $\ell_B=0.015625$.
Mislabeling $\ell_B$ as the quadratic prediction falsely contracts the
radius despite the objective being an exact quadratic. The new regression
preserves the radius when this is the only observation. It also checks
that the exact Newton step $p_N=-0.01$ has $a_N=q_N=0.005$, not a
quadratic trust ratio of one half, and that predictor acceptance cannot
erase a preceding Newton rejection.

Trace fields now distinguish `accepted_newton_step`, `newton_trial_evaluated`,
and the last Newton trial's actual/predicted decreases and norm. For an
accepted Armijo-only step, the accepted `predicted_decrease`, `trust_ratio`,
and `accepted_trial_radius` are zero (unavailable/not applicable), rather
than inventing a quadratic prediction or boundary classification. Krylov
residuals and spectral fields describe the last attempted Newton model,
not necessarily the accepted predictor.

This repair does not alter the forcing rule, response tolerances, curvature
admission gate, or strict block-secant symmetry check. In particular, the
structure-response defect established in Section 8 remains to be repaired.

### Initial qualification of the trust-observation repair

Commit `d023aba` passed all **46/46** tests on Hanhai25 in job `247009`,
including the new end-to-end F2 trace regression. The preceding setup job
`247002` did not build excluded developer targets, so those tests were
reported **Not Run**; it is not counted as a passing suite. Building the
explicit `xmvb_dev_tools` target resolved that test setup issue.

The same 32-thread, exact-integral Davidson panel has the following completed
results. Timings are single-run SCF measurements. The table is deliberately
partial while the remaining calculation runs.

| System | Previous TNHVP steps / s | Repaired TNHVP steps / s |
|---|---:|---:|
| 240 | 6 / 9.565958 | 6 / 9.471276 |
| 241 | failed after 3 / 0.371610 | failed after 3 / 0.370574 |
| MnF2 | 18 / 86.225576 | 18 / 58.018364 |
| FeCl2 | 5 / 15.469963 | 3 / 1.718220 |
| 7963 | 6 / 1.793981 | 6 / 1.776911 |
| 7975 | failed after 4 / 0.864279 | failed after 4 / 0.865893 |
| YAMSAI | 6 / 1.220511 | 6 / 1.212238 |
| CERRAS | failed after 4 / 25.383004 | failed after 4 / 24.842835 |
| LOFLEA | failed after 6 / 539.819579 | running, 5 accepted steps |

The three completed failures remain strict block-curvature symmetry failures,
not numerical exceptions suppressed by this repair. Recorded trace checks
confirmed that every no-Newton-observation step preserved its source radius
and that every accepted Armijo predictor had no fabricated quadratic ratio.
MnF2 exercised five rejected-Newton/accepted-predictor transitions, so this
separation was also exercised outside the synthetic tests.

For MnF2, HVP block actions decreased from 779 to 592, structure-response
actions from 21905 to 17942, and cumulative HVP time from 54.02 to 39.81 s.
Its final energy changed from $-1348.893353187962$ to
$-1348.893353223406\ E_h$. Nevertheless, the repaired 58.02 s remains much
slower than orbital-block L-BFGS's 6.86 s. This is not overall algorithm
acceptance.

FeCl2 needs an explicit accuracy caveat: the repaired endpoint is
$-2181.617635688955\ E_h$, versus the previous
$-2181.617645274734\ E_h$. Its projected gradient infinity norm is
$4.63601\times10^{-4}$ instead of $7.06846\times10^{-5}$. Both satisfy the
requested outer thresholds, but their energy difference is
$9.58578\times10^{-6}\ E_h$. Thus the apparent time improvement must not
be advertised as an equal-endpoint-accuracy speedup. Job `247010` repeats
old TNHVP, repaired TNHVP, and orbital-block L-BFGS with a common diagnostic
$\tau_g=10^{-4}$, $\tau_E=10^{-9}\ E_h$; production defaults are unchanged.
That diagnostic is still running. Both jobs have 20-minute scheduler limits
and a 2000-step outer limit, not a 30-step convergence cutoff.

LOFLEA's fifth repaired iteration is already more expensive than its earlier
counterpart. Preserving a mathematically meaningful trust radius alone does
not repair the global subproblem stopping rule. No improvement is claimed
for LOFLEA before the calculation completes. Next gates remain the symmetric
response model and distinct global/local subproblem accuracy contracts.

Current run logs are under `build/diagnostics/consistency-3460c35/trust/`
and `.../endpoints/FeCl2/`. The tested executable SHA-256 is
`fb162f9696ed88335c3279ed708e563d8d3feaea0e88e9fcfe41edb2c6fcf05a`.

## 10. Common response model and versioned Hessian samples

The consistency repair separates **enrichment** from **frozen application**.
On the gauge-fixed response domain of Section 3, let $W$ contain the retained
orthonormal response vectors and let $DW$ contain their operator images. Define

$$
K_W=W^TDW,\qquad R_W=W K_W^{\dagger}W^T,\qquad
H_W=A-B^TR_WB.
$$

The symmetric spectral inverse retains the signs of resolved eigenvalues;
negative eigenvalues are not removed. Its numerical-rank threshold depends on
the frozen projected operator, not on the current right-hand side. Consequently
$R_W^T=R_W$ and $H_W^T=H_W$, assuming consistent component adjoints. The
normalization/overlap derivatives and selected-space multipliers remain part of
the response and pullback; they are not discarded when changing its solver.

The old recycled `guess` was only a warm-start policy: it discarded a Galerkin
solution unless the Euclidean residual improved relative to a zero guess. Such
a right-hand-side-dependent decision is not a linear inverse action, including
for positive-definite response matrices. The new unconditional
`galerkin_apply` replaces the removed `guess` interface. Only the enrichment
solver may choose whether to use its output as an initial guess, after
screening against the full bordered residual. That warm-start decision is
not part of the frozen inverse action.

For an incoming orbital block, independent MINRES solves first enrich the
accepted-point response spaces. All columns are then evaluated in the final
common space, including equal-weight directions previously processed earlier
in the block. The common responses must meet the requested **true bordered**
residual before they are published as new samples. Further enrichment is
driven by that residual and actual space growth, not a fixed retry count.
An explicit positive response tolerance now overrides the default in either
direction; zero retains the default. The previous `max(default, requested)`
incorrectly prevented a caller from requesting tighter accuracy.

Each accepted independent enrichment advances the response-model revision.
For retained orbital directions $Q=[q_1,\ldots,q_m]$, the solver must store

$$
Y=H_WQ
$$

at **one revision** before building the projected Hessian, positive Ritz
secants, inverse-BFGS update, predicted decrease, or KKT certificate. On model
change all retained images are refreshed. Same-point trust-radius retries
reuse them only when the revisions match. A failed solve after enrichment
cannot return a step certified against the previous revision. The affine
L-BFGS/Newton diagnostic also rebuilds its baseline image and right-hand side
if its response model changes during the correction solve.

The first production repair replays the complete frozen HVP for retained
directions so that no gauge, multiplier, or orbital pullback term is omitted.
This replay streams one direction at a time: it does not allocate AO-matrix
intermediates for the entire orbital Krylov basis. It introduces no full
structure H/S or orbital Hessian assembly. Replacing this replay with a proven
low-rank update is a separate performance task, not a condition hidden behind
the correctness repair.

Frozen application reports its true residual without changing the model. In
particular, enrichment can change the residual of an earlier right-hand side;
the old sample is refreshed consistently, not claimed to retain its previous
residual certificate. This repair establishes a coherent model, **not** a
bound on the difference between that model and the exact relaxed Hessian.
Residual-transfer estimates and error-aware Newton forcing remain separate
open work. Section 10.2 corrects the finite-Ritz gauge forcing and multiplier
recovery; it does not replace those missing orbital-HVP error estimates.

Validation includes indefinite projected inverses, gauge-preserving isolated
and equal-weight responses, changing-model Newton sample refresh, stale
trust-radius retry invalidation, and molecular frozen-model identities at the
captured 241 and 7975 points. Molecular fixtures are portable decimal orbital
tables under `tests/vbscf/fixtures/response/`; their provenance is recorded
there. Fixed-model linearity, symmetry, segmentation, column permutation and
repeatability are asserted numerically. Equality between different enriched
models is deliberately not an invariant.

### 10.1 First shared-model qualification at the captured failure points

Before the finite-Ritz corrections below, Hanhai25 Slurm job `247101` passed
all 48 tests, including the original
exact/RI and isolated-root/equal-weight finite-difference tests. The new
fixed-point checks use three independent orbital probes even when the Newton
solver would stop with a one-dimensional subspace. The measured absolute
Frobenius-norm defects of the frozen models are:

| Quantity | 241 | 7975 |
|---|---:|---:|
| Projected curvature skew | $6.45\times10^{-15}$ | $3.95\times10^{-15}$ |
| HVP linear-combination defect | $1.12\times10^{-13}$ | $1.20\times10^{-13}$ |
| Cached Newton-step image versus frozen direct application | $8.16\times10^{-14}$ | $2.01\times10^{-16}$ |
| Scalar/block, segmentation, permutation, repeat, and zero defects | 0 | 0 |

Adding the third training direction changes the first two images by
$8.48\times10^{-7}$ for 241 and $1.93\times10^{-6}$ for 7975. Thus the
revision/refresh check is not vacuous: the same-model identities pass after
actual changes to previously stored samples. The production curvature guard
has not been loosened.

The same job ran both structure eigensolvers from the original input guesses
with 32 CPU threads, exact integrals, TNHVP, and common outer thresholds
$\tau_g=10^{-3}$ and $\tau_E=10^{-7}\ E_h$. The outer limit was 2000,
not 30 steps. All four runs converged:

| System | Eigensolver | Accepted steps | Final total energy / $E_h$ | SCF wall time / s | Peak RSS / KiB |
|---|---|---:|---:|---:|---:|
| 241 | Davidson | 9 | $-230.720590391790$ | 9.618 | 991312 |
| 241 | dense | 9 | $-230.720590391792$ | 2.827 | 998516 |
| 7975 | Davidson | 6 | $-285.733616934240$ | 3.680 | 1548364 |
| 7975 | dense | 6 | $-285.733616934243$ | 1.480 | 1659544 |

These single-run timings establish successful execution, not a speedup:
Davidson remains slower here. The replay overhead and repeated projected
factorizations remain explicit optimization opportunities. Convergence on
these two cases also does not establish that TNHVP outperforms orbital-block
L-BFGS or that its inexact-Hessian error budget is complete.

Local logs are under `build/diagnostics/response-model-20260920/`; remote logs
are under `/home/guqqgroup/taoxia/xmvb-runs/response-model-20260920/`.
The tested executable SHA-256 is
`7f07ea08f33890918ada3e97f6aeb1f426ab562dcb2676fa14281f340d5834c2`.
The remaining seven-system Davidson panel ran as array `247105`. It exposed
an additional bordered-response failure in CERRAS after four accepted steps;
the first shared-model qualification alone therefore does not close the repair.
240, FeCl2, 7963, and YAMSAI converged. MnF2 and LOFLEA were cancelled while
still running to replace this incomplete patch with the full repair; they are
not counted as convergence successes or failures.

### 10.2 Finite-Ritz response and the normalization constraint

An approximate selected root has a nonzero Ritz residual. For one state define

$$
A=H-ES,\qquad l=Sc,\qquad n=c^TSc,\qquad r=Ac,
$$

$$
f=(\delta H-E\delta S)c,\qquad q=-\frac12c^T\delta S c.
$$

The symmetric bordered response equation is

$$
\begin{pmatrix} A&l\\l^T&0\end{pmatrix}
\begin{pmatrix}\delta c\\-\delta E\end{pmatrix}
=\begin{pmatrix}-f\\q\end{pmatrix}.
$$

Multiplication of its first row by $c^T$ gives

$$
\delta E=\frac{c^Tf+r^T\delta c}{n}.
$$

The previous isolated-root solver fixed $\delta E=c^Tf/n$, which discards the
second term. It cannot generally satisfy the full bordered equation when
$r\ne0$, even after an arbitrarily accurate projected solve.

Use the Euclidean projector onto the normalization tangent,

$$
P_l=I-\frac{ll^T}{l^Tl},\qquad
\delta c=\frac{q}{n}c+z,\qquad l^Tz=0.
$$

The correct projected system is

$$
P_l A P_l z=-P_l\left(f+\frac{q}{n}r\right).
$$

For a frozen $W$ with $W^Tl=0$, its Galerkin solution is

$$
z=-W(W^TAW)^{\dagger}W^T\left(f+\frac{q}{n}r\right).
$$

Thus the projector, gauge forcing, and recovered multiplier must change
together. This is a restriction of the original symmetric bordered matrix,
not an empirical correction to its residual threshold. In particular, with
$s=c/n$, $K=W^TAW$, $k=W^TAs$, and $d=s^TAs$, the restricted matrix is

$$
\begin{pmatrix}
K&k&0\\k^T&d&1\\0&1&0
\end{pmatrix}.
$$

It is symmetric without the assumption $r=0$. The corresponding structure
contribution for two orbital directions has the reciprocal bilinear form
$2f_u^T\delta c_v+2q_u\delta E_v$. Omitting either the gauge forcing or the
multiplier correction destroys that reciprocity at finite Ritz residual.

For an equal-weight selected cluster, define $L=SC$, $M=C^TSC$,
$Q=-\tfrac12 C^T\delta S C$, and $R=HC-SC\mathcal E$. The particular gauge
response is $X_0=CM^{-1}Q$. The existing $L^\perp$ projected equations also
require the forcing correction $P_L R M^{-1}Q$, while their full
selected-space multiplier recovery is retained. The accepted-point residual
$R$ can be obtained from the already computed selected H/S images; no extra
accepted-point H/S action or dense assembly is required.

A separate stopping-rule issue must not be confused with finite Ritz drift:
a projected tolerance can be larger than the requested full bordered
tolerance. Passing the former while missing the latter does not establish
an uncorrectable residual. Qualification must report both residual components
and both targets rather than misclassifying that situation as breakdown.
Moreover, with the recovered multiplier the full top residual is an oblique
reconstruction of the projected residual $e\in l^\perp$:

$$
e_{\mathrm{full}}=\left(I-\frac{lc^T}{n}\right)e,\qquad
\|e_{\mathrm{full}}\|_2\le\gamma\|e\|_2,\qquad
\gamma=\frac{\|l\|_2\|c\|_2}{n}.
$$

This factor is the exact norm on $l^\perp$ (for a nonempty complement).
Consequently, a sufficient projected target is the requested bordered target
divided by $\gamma$, followed by the actual full-residual check. For a cluster,
the corresponding factor is $\gamma_C=1/\sigma_{\min}(Q_L^TQ_C)$, where
$Q_L,Q_C$ are thin orthonormal bases of $SC,C$. It requires only a
selected-state-sized singular-value decomposition. This is a norm conversion,
not a fitted convergence parameter or a relaxation of the bordered tolerance.
The independent target relative to the *projected* right-hand side is removed:
projection may nearly cancel that vector while the original bordered equation
still has a finite accuracy scale. Imposing both relative tolerances would
introduce an unrelated overprecision requirement. The isolated-root solver
accepts a candidate as soon as its true full residual passes; the sufficient
bound is not a necessary acceptance condition. The cluster solver still uses
the conservative projected bound before its final full check, leaving a
possible efficiency improvement without weakening the certified equation.

For CERRAS, diagnostic job `247124` measured a full residual of
$6.65046\times10^{-5}$, projected residual of $6.65045\times10^{-5}$,
projected target $6.71277\times10^{-5}$, and bordered target
$6.57759\times10^{-5}$. The selected-root residual component was only
$-3.14591\times10^{-8}$. Hence that specific failure is demonstrably a
misclassified, still-correctable residual, **not** the independently identified
finite-Ritz obstruction. In job `247139`, the finite-Ritz isolated and
equal-weight cold, recycled, and frozen responses agree with their explicit
bordered references to at most $1.44\times10^{-17}$. That intermediate job
still fails the gauge-induced conditioning fixture discussed below, so it is
not final qualification of the full repair.

### 10.3 Gauge-stabilized response preconditioning

Changing the eliminated direction from $c$ to $Sc$ also changes which entries
are represented in the projected equation. The former raw diagonal inverse
can assign an artificial $1/\epsilon_{\mathrm{mach}}$ weight to an entry
whose shifted diagonal vanishes only because of the selected-root gauge.
For example, with $c=e_1$, $Sc=(1,0.9,0)^T$, and

$$
A=\begin{pmatrix}0&0&0\\0&1&0.4\\0&0.4&3\end{pmatrix},
$$

the new tangent is not the coordinate plane that discards the first entry.
A raw reciprocal of $A_{11}=0$ makes an otherwise solvable two-dimensional
projected problem artificially ill-conditioned.

Let $Q_L$ be the orthonormal normal basis and $P_L=I-Q_LQ_L^T$. A gauge lift
satisfies

$$
P_L(A+\sigma Q_LQ_L^T)P_L=P_LAP_L.
$$

The positive Jacobi surrogate uses

$$
d_i=|A_{ii}|+\sigma (Q_LQ_L^T)_{ii},\qquad
\sigma=\max_i|A_{ii}|.
$$

It adds magnitudes, rather than taking the magnitude after addition, to avoid
sign cancellation for indefinite shifts. The remaining reciprocal floor is
only the floating-point scale. No H/S action or dense matrix is needed to
form the projector diagonal. The scale is determined by the current operator,
not fitted to a molecule. This changes the preconditioner, not the response
equation, frozen inverse, or accepted residual tolerance; genuine physical
ill-conditioning can still remain.

A cold or recycled initial response must also be checked against the original
bordered equation before entering MINRES. Near cancellation, that initial
response can already be accurate enough even when a relative comparison to
the small projected right-hand side would reject it. Conversely, an inexact
response seed does not guarantee that its later Galerkin projection meets the
same Euclidean tolerance without refinement. Regression tests must certify
the equation and reference solution rather than hard-code a particular
iteration trajectory.

### 10.4 Final shared-model and finite-Ritz qualification

Job `247155` passes all 48 tests after the complete repair, including the
noncommuting-RHS recycling fixture, the gauge-conditioned repeated-root
problem, isolated/equal-weight finite-Ritz cold/recycled/frozen responses,
and exact/RI molecular finite differences. The repeated-root problem now
converges in two iterations per column with a bordered relative residual of
$1.93\times10^{-16}$, instead of stopping with a residual near 0.126 after
four iterations under the unstabilized preconditioner.

At the stored molecular failure points, the final frozen-model skew is
$7.49\times10^{-15}$ for 241 and $5.41\times10^{-15}$ for 7975. Their
linearity defects are respectively $1.13\times10^{-13}$ and
$1.19\times10^{-13}$; cached-step image defects are $1.12\times10^{-13}$
and $2.20\times10^{-16}$. The new response direction changes previously
stored images by $2.29\times10^{-6}$ and $5.48\times10^{-6}$, so these
tests exercise actual model enrichment and image refresh.

The 32-thread convergence reruns retain the same exact integrals and outer
thresholds as Section 10.1. The four completed reference comparisons are:

| System | Eigensolver | Accepted steps | Final total energy / $E_h$ | SCF wall time / s | Peak RSS / KiB |
|---|---|---:|---:|---:|---:|
| 241 | Davidson | 9 | $-230.720590391789$ | 9.264 | 996140 |
| 241 | dense | 9 | $-230.720590391792$ | 2.688 | 981404 |
| 7975 | Davidson | 6 | $-285.733616934238$ | 3.560 | 1638352 |
| 7975 | dense | 6 | $-285.733616934243$ | 1.376 | 1627540 |

The matching executable in array `247148` also converges 240 (6 steps),
FeCl2 (3), 7963 (6), YAMSAI (6), and CERRAS (8). CERRAS reaches
$-397.093434952605\ E_h$ with projected gradient infinity norm
$1.4834\times10^{-4}$ and last energy change $-3.05\times10^{-8}\ E_h$.
Its SCF wall time is 767.943 s and process peak RSS is 2066556 KiB.
MnF2 and LOFLEA remain pending. No uncompleted calculation is counted as converged.
Timings are single runs on allocated cluster CPUs, not repeated performance
measurements; the full frozen-HVP replay remains a known cost.

The executable SHA-256 for job `247155` and array `247148` is
`c1f75db276a0d483c8e9506a4c895bfe0d5590b414e7e2d93ce14e3953562ce9`.
After removing the unused residual-gated `guess` API, job `247173` again passes
all 48 tests and repeats the four Davidson/dense convergence comparisons with
identical accepted-step counts and reported final energies. The production
executable SHA-256 after that cleanup is
`99474663e121912118f74b60762aa55d4e660c3dba8b6db7438eaa7b91e40186`.
Job `247174` additionally reruns all 48 tests after strengthening the molecular
audit to compare the actual scalar `apply_reduced` entry point with the block
entry point. Both scalar/block defects are zero. All other recorded frozen-model
defects are unchanged. The audit executable SHA-256 is
`b4f8fbfad723ad4f82da021f9c325524175b46df653eeb4772a363b9c0613a1f`.
Local logs are under
`build/diagnostics/response-model-20260920/runs/qualified/` and
`.../qualified-fixed/`, with cleanup reruns under `.../runs/final/`
and final frozen-model diagnostics under `.../final-fixed/`.
Remote run logs are under
`/home/guqqgroup/taoxia/xmvb-runs/response-model-20260920/qualified/`
and its sibling `final/`.
The separate response-residual-to-orbital-HVP error budget and global/local
Newton stopping contract remain open; successful consistency qualification
does not establish overall TNHVP performance acceptance.

### 10.5 Cost limitation of the correctness-first refresh

The production refresh currently reevaluates the complete HVP for every retained
orbital direction after each response-model revision. This recomputes unchanged
direct-core, fixed-upstream, and local-active terms as well as the changed
structure response. If the orbital subspace grows in blocks of width $b$ to
dimension $m$, and each block changes the response model, the cumulative replay
count is

$$
\sum_{j=1}^{m/b} jb = O(m^2/b),
$$

in addition to the $m$ new sampled directions. For dimensions $1,3,\ldots,63$,
the expansions after the first direction replay 1023 full HVPs. This is a
confirmed implementation cost, not a measured attribution of every second in
the long molecular runs.

Likewise, each frozen Galerkin application currently rebuilds $W^TDW$ and its
spectral inverse, even when the model revision has not changed. With response
rank $r$, these repeated operations cost $O(n_{\mathrm{str}}r^2+r^3)$ per RHS.
The next implementation should cache this factorization by revision and update
only the response-dependent HVP contribution, while preserving the shared-model
identities. This repair intentionally does not claim that optimization complete.

The current trace field `exact_hvp_block_actions` counts batched calls, not the
scalar frozen replay calls. Total HVP/response timings include the replay, but
that field alone is not a complete work counter. Dedicated replay/rank counters
are needed before quantitatively attributing the observed slowdown.

The completed CERRAS trace gives a concrete warning: step 7 retains 77 orbital
directions and consumes 588.34 s, including 581.11 s in HVPs and 545.39 s in
outer response. These are nested times, not additive categories. Its 39
reported batched HVP calls exclude scalar replay; the 3383 reported response
actions do not isolate projected factorization time. Thus CERRAS convergence
qualifies the repaired equations on that case, not acceptable performance.

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
