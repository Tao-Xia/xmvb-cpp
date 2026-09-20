# TODO: Exact-Curvature Enrichment of Block-LBFGS

## Primary objective

The production TNHVP optimizer must be redesigned as a genuine curvature-enriched
block-LBFGS method:

> Retain block-LBFGS as the global, inexpensive inverse-Hessian model, and use
> exact matrix-free Hessian-vector products (HVPs) to replace or enrich its
> curvature only in directions where the current Newton residual demonstrates
> that additional second-order information is needed.

The intended result is fewer accepted outer iterations than block-LBFGS, while
preserving matrix-free memory scaling and minimizing total wall time. A small
fixed projected-Newton subspace initialized by an L-BFGS step is not the final
algorithm.

This document is the implementation TODO and acceptance contract. Do not replace
the derivation below with system-specific iteration budgets, molecule-specific
thresholds, or empirical switches.

## Execution rule

Advance the implementation checklist strictly one item at a time. For every
item:

1. implement only the scoped mathematical or software change;
2. add and run the corresponding focused correctness tests;
3. compare the relevant numerical and performance diagnostics with the previous
   accepted state;
4. remove code made obsolete by that item;
5. mark the item complete and create a dedicated Git commit before proceeding.

Do not start a later item while an earlier item is mathematically unresolved or
failing its tests. Benchmark-only work may run concurrently, but it must not
change the production algorithm out of sequence.

## Confirmed state of the current implementation

The current production path does the following:

1. Construct an inverse L-BFGS model $M_k$ with an orbital-block initial inverse.
2. Form the baseline step

   $$
   p_{\mathrm B}=-M_k g_k.
   $$

3. Use $p_{\mathrm B}$ only as the first direction of a small exact-HVP
   projected trust-region model.
4. Solve

   $$
   \min_z\; g_k^TQz+\frac{1}{2}z^TQ^TH_kQz,
   \qquad \|Qz\|_{G_k}\leq\Delta_k.
   $$

Thus, the production method is currently an exact projected-Newton/GLTR method
seeded and preconditioned by block-LBFGS. It is not yet an algebraic
block-LBFGS model enriched by exact curvature. Outside $\operatorname{span}(Q)$,
the projected model does not preserve the block-LBFGS curvature model.

The name `block-LBFGS` also needs to be interpreted carefully: the present code
uses an ordinary transported L-BFGS two-loop recursion with a block-structured
initial inverse. It does not yet perform a multi-vector block secant update.

Relevant production code:

- `src/vbscf/optimization/backends/truncated_newton.cpp`
- `src/vbscf/optimization/trust_region/truncated_newton.cpp`
- `src/vbscf/optimization/quasi_newton/transported_lbfgs.cpp`

## What is and is not duplicated

### No direct curvature double counting is currently present

Accepted L-BFGS secants are

$$
s_i=x_{i+1}-x_i,
\qquad
y_i=g_{i+1}-g_i.
$$

Because the structure coefficients are relaxed, $y_i$ approximates a path
average of the reduced orbital Hessian,

$$
y_i=\int_0^1\bar H(x_i+t s_i)s_i\,dt,
$$

where, formally,

$$
\bar H=A-B^TD^{-1}B.
$$

An exact HVP $H_kq$ supplies current-point local curvature. These two sources
contain related information, but the current implementation does not add an
L-BFGS Hessian and the exact projected Hessian together. L-BFGS only generates
and preconditions directions. Therefore, the current defect is not arithmetic
double counting.

### Confirmed avoidable repeated work

- At a fixed accepted orbital point, repeated generalized-eigen response solves
  use the same response operator but restart MINRES from zero for every new HVP
  right-hand side.
- Exact pairs $(q,H_kq)$ are discarded after the small projected solve. Only the
  accepted displacement-gradient secant is retained by L-BFGS.
- Trust-radius retries at an unchanged accepted point can rebuild the same HVP
  subspace even though the low-level solver supports an initial subspace.
- The current production cap of four curvature-space dimensions can terminate
  correction even when the full Newton/KKT residual does not satisfy the forcing
  condition.
- The RI block-HVP implementation still processes admitted columns separately in
  part of the exact-HVP path.

Work that is already correctly shared and must not be misidentified as
duplication:

- candidate directions are orthogonalized before an HVP is evaluated;
- $(q,H_kq)$ is stored within one projected solve;
- admitted directions are sent through block HVP interfaces;
- direct-core and outer-response channels share accepted-point intermediates;
- a final true response residual evaluation is required for certification.

## Required mathematical formulation

### Metric-consistent coordinates

The accepted-point reduced coordinates are locally prewhitened for independent
orbital normalization, but they are not globally Euclidean in the coupled
inactive-subspace/active-ray metric. The full reduced metric is

$$
G_k=U_k^TM_{\mathrm{phys},k}U_k\neq I
$$

in general. It must not be assembled or factorized merely to implement a
quasi-Newton update.

Instead, preserve the distinction between tangent vectors and gradient
covectors. Under an invertible coordinate change $p=A\widehat p$,

$$
\widehat g=A^Tg,
\qquad
\widehat H=A^THA,
\qquad
\widehat G=A^TGA,
\qquad
\widehat M=A^{-1}MA^{-T}.
$$

Here $H$ and $G$ map tangent vectors to covectors, whereas the inverse model
$M$ maps covectors to tangent vectors. The contractions $s^Ty$, $g^Tp$, and
$p^THp$ are the natural vector-covector pairings and are coordinate invariant.
Consequently, BFGS and block inverse-BFGS require no full-space metric
whitening when their arguments retain these types.

The physical metric enters the trust constraint and shifted KKT equation:

$$
p^TG_kp\leq\Delta_k^2,
\qquad
g_k+H_kp+\lambda G_kp=0.
$$

For a sampled basis $Q$, assemble only

$$
G_Q=Q^TG_kQ,
$$

and whiten this small matrix by Cholesky factorization in the projected
trust-region solve. This is already the production trust-region strategy and
retains matrix-free memory scaling. Euclidean orthogonalization may choose a
numerically convenient basis for the same sampled span; it must never replace
$G_Q$ by the identity in the trust-region model.

### Baseline inverse model

Let

$$
M_k\approx H_k^{-1}
$$

be the transported block-LBFGS inverse model from reduced gradient covectors to
reduced tangent vectors. Start from

$$
p_0=-M_kg_k.
$$

Evaluate one exact current-point HVP,

$$
w_0=H_kp_0,
$$

and form the true Newton residual

$$
r_0=g_k+w_0.
$$

For an interior step, no further exact curvature is needed when

$$
\|r_0\|\leq\eta_k\|g_k\|.
$$

For a trust-region boundary step, use the shifted KKT residual

$$
r(p,\lambda)=g_k+H_kp+\lambda G_kp
$$

together with primal feasibility and complementarity. Only after projected
metric whitening does this shift become $\lambda p$.

### Positive-curvature block enrichment

Collect linearly independent exact samples

$$
S=[s_1,\ldots,s_b],
\qquad
Y=H_kS.
$$

After removing nonpositive and numerically dependent modes, require

$$
S^TY\succ0.
$$

With

$$
R=(S^TY)^{-1},
$$

form the block inverse-BFGS enrichment

$$
M_k^+
=
(I-SRY^T)M_k(I-YRS^T)
+
SRS^T.
$$

The defining condition is

$$
M_k^+Y=S.
$$

Consequently, exact HVP curvature replaces the previous inverse-model action in
the sampled positive-curvature subspace. It is not added a second time. Outside
that subspace, the inexpensive block-LBFGS model remains active.

The implementation should use stable factorizations of the symmetrized small
matrix $S^TY$, not form an explicit inverse.

### Residual-driven expansion

If the certified Newton/KKT residual is still too large, construct the next
candidate from the unresolved defect, for example

$$
s_{j+1}=-M_k^{(j)}r_j.
$$

Orthogonalize it in the correct metric, evaluate $H_ks_{j+1}$, enrich the model,
and recompute the step. Expansion stops because the certified residual satisfies
the forcing condition, not because an empirical four-dimensional limit has been
reached.

A resource limit may remain solely as a failure guard. Reaching it must be
reported as an uncertified inner solve and must not be presented as normal
convergence.

### Negative curvature

Negative or nearly zero curvature modes must not be inserted into the
positive-definite block inverse-BFGS update. Retain such modes explicitly in a
small trust-region subspace and use them to construct a boundary step. Thus:

- positive modes enrich $M_k$ through exact block secants;
- negative modes remain explicit trust-region directions;
- the final step is certified with the full shifted KKT residual.

## Response-space recycling

This is a wall-time optimization, not a substitute for the curvature-enrichment
algorithm.

At one accepted orbital point, let $\mathcal D_k$ denote the fixed projected
structure-response operator. Retain a response basis $W$ and its image
$\mathcal D_kW$. For each new right-hand side $b$:

1. compute a Galerkin solution in $\operatorname{span}(W)$;
2. evaluate the true residual;
3. expand only the unresolved component;
4. retain newly useful response directions for later HVPs at the same point.

Recycle this space across exact-HVP directions and trust-radius retries at the
same accepted point. Invalidate or transport it explicitly when the accepted
orbital point, horizontal chart, structure state, or response operator changes.

## Persistence of exact curvature information

Within one accepted point, retain all valid $(S,Y)$ pairs and reuse them across
trust-radius retries. Do not recompute an HVP for an unchanged direction.

Across accepted points, transported pairs are no longer exact current-point
HVPs. They may be retained as quasi-Newton history after correct horizontal
transport, but they must be labelled as approximate history. Current-point
Newton/KKT certification must always use current-point HVPs.

## Implementation sequence

- [x] Establish metric-consistent reduced algebra. The coordinates are not
      globally Euclidean; vector-covector quasi-Newton algebra remains
      coordinate covariant, while only the small projected trust metric is
      whitened. A nonorthogonal coordinate-change regression verifies the
      generalized trust-region solution.
- [x] Add a block inverse-BFGS update operating on matrix-free inverse actions.
      The standalone component supports vector and block actions without an
      ambient inverse matrix and rejects nonpositive sampled curvature.
- [x] Add algebraic tests for symmetry, positive definiteness, block secant
      exactness, and invariance under a change of basis within $\operatorname{span}(S)$.
      The tests also cover general vector-covector coordinate covariance,
      complete-space inverse recovery, vector/block action agreement, and
      rejection of nonpositive curvature.
- [x] Replace the fixed four-dimensional production correction with
      residual-driven enrichment. The backend now permits the full natural
      reduced dimension, while the solver stops as soon as the certified KKT
      forcing condition is met. Positive exact Ritz curvature enriches the
      block-LBFGS inverse used to precondition each unresolved defect.
- [ ] Separate positive-curvature enrichment from explicit negative-curvature
      trust-region handling.
- [ ] Return and retain $(S,H_kS)$ from the step solver.
- [ ] Reuse orbital HVP samples across trust-radius retries at the same accepted
      point.
- [ ] Add same-point generalized-eigen response-space recycling.
- [ ] Batch the remaining columnwise RI HVP work where mathematically shared
      intermediates exist.
- [ ] Remove the obsolete fixed-cap and unused correction/recycling code once the
      replacement is validated; do not retain a fallback implementation.
- [ ] Update the theory article with the final accepted formulation and measured
      complexity.

## Required correctness tests

### Small synthetic Hessians

- Compare enriched inverse actions against explicitly assembled dense Hessians.
- Verify $M_k^+Y=S$ to numerical precision.
- Verify symmetry and positive definiteness for $S^TY\succ0$.
- Verify that rejected negative modes do not contaminate the SPD update.
- Verify convergence to the dense trust-region Newton step as the sampled space
  becomes complete.

### VBSCF derivative tests

- Check the gradient by finite differences in the same nonredundant chart.
- Check every sampled HVP by directional finite differences of the fully relaxed
  gradient.
- Check consistency among the horizontal lift, metric, vector transport,
  retraction, and pullback.
- Compare dense and Davidson structure solvers at identical accepted orbital
  points.
- Verify identical converged energies for block-LBFGS and TNHVP under identical
  physical and convergence settings.

### Regression systems

Include at least F2, 240, 241, MnF2, FeCl2, 7963, 7975, YAMSAI, CERRAS, and
LOFLEA, covering sparse HAO, full-AO OEO, small structure spaces, large
unique-string spaces, and ill-conditioned pair overlaps.

## Performance acceptance criteria

The new method is accepted only if the measurements separate convergence quality
from per-step cost. Record:

- accepted outer iterations;
- rejected trials;
- exact HVP directions and block calls;
- structure-response operator actions;
- Newton/KKT residual and forcing threshold at every accepted step;
- time in direct-core response, outer response, H/S action, and total HVP;
- total wall time and peak RSS;
- final energy, projected-gradient infinity norm, and adjacent-step energy
  change.

The main comparison is against the current block-LBFGS baseline. TNHVP should
demonstrate a reproducible reduction in accepted outer iterations from genuinely
new second-order information. The additional HVP cost must then be assessed by
total wall time, not hidden behind iteration counts.

No system-specific tuning is an acceptable success criterion. Stopping and
admission decisions must follow the Newton/KKT residual, trust-region model
agreement, curvature spectrum, and measured reusable-work cost.

## Non-goals

- Do not reintroduce an explicitly assembled orbital Hessian.
- Do not require a fully coupled orbital-structure Newton solve merely to enrich
  block-LBFGS; structure response remains a matrix-free Schur-complement action.
- Do not perform outer response unconditionally when the baseline step already
  satisfies the required Newton accuracy or when the certified cost-benefit rule
  rejects it.
- Do not optimize only the number of outer iterations while ignoring wall time
  and peak memory.
- Do not keep obsolete fallback algorithms after the replacement passes the
  correctness and regression tests.
