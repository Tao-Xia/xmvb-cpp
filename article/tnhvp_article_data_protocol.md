# TNHVP Method Freeze and Article Data Protocol

## 1. Purpose

This document defines the algorithmic identity, validation hierarchy, and data
collection protocol for the matrix-free VBSCF orbital optimizer. Its purpose is
to prevent three common sources of ambiguity: changing the algorithm while
collecting publication data, treating a nearly converged input as evidence of
fast nonlinear convergence, and comparing wall times obtained from different
software states or hardware configurations.

The production method is denoted **TNHVP** in the article. The command-line
backend name is currently `nonredundant_truncated_newton`; the article should
not use that implementation name as the method name.

## 2. Frozen mathematical definition

The production optimization variables are accepted-point quotient coordinates
for either strictly sparse HAO orbitals or full-AO OEO orbitals. The reduced
gradient and reduced Hessian action are

$$
\mathbf g_k=\mathbf U_k^{\mathrm T}\mathbf g_{\mathbf x,k},
\qquad
\mathbf H_k\mathbf v
=\mathbf U_k^{\mathrm T}
\left[\nabla^2_{\mathbf x}E(\mathbf x_k)\right]
\mathbf U_k\mathbf v,
$$

where the accepted-point basis is prewhitened in the inexpensive local
pullback metric of the per-orbital normalization maps,

$$
\mathbf U_k^{\mathrm T}\mathbf M_k^{\mathrm{loc}}\mathbf U_k=\mathbf I,
\qquad
\mathbf M_k^{\mathrm{loc}}
=
\bigoplus_p
\mathbf J_{N,p}^{\mathrm T}\mathbf S_p\mathbf J_{N,p}.
$$

This local prewhitening is not the trust-region metric. Denote by
$\mathbf G_k$ the physical inactive-subspace/projected-active-ray metric of
eq 31 in the theory note. The optimizer applies $\mathbf G_k$ through
matrix-free metric-vector actions. Only its projection
onto the current Newton subspace is factored; no full orbital Hessian or full
reduced physical metric is assembled. The trial retraction remains additive
on the immutable sparse supports, and the trust radius bounds the physical
length of its accepted-point tangent.

Here, the normalization Jacobians act on the stored sparse coefficient slots;
the raw Euclidean identity is not the trust-region metric. The columns of the
quotient basis remove all support-admissible inactive-orbital and scaling gauge
directions. Strict sparsity is preserved by the retraction; OEO is represented
by full AO support, not by a separate optimizer.

The Hessian is never assembled in production. Each action differentiates the
complete accepted-point computation graph, including orbital normalization,
the inactive projector, one- and two-electron integral transformations, the
generalized VB eigenproblem, structure response, and the final gradient
pullback. The trust-region model is

$$
m_k(\mathbf s)
=E_k+\mathbf g_k^{\mathrm T}\mathbf s
+\frac{1}{2}\mathbf s^{\mathrm T}\mathbf H_k\mathbf s,
\qquad
\mathbf s^{\mathrm T}\mathbf G_k\mathbf s\leq\Delta_k^2.
$$

An inner solution is certified with the shifted KKT residual

$$
\mathbf r_k
=\mathbf g_k+\mathbf H_k\mathbf s_k+
\lambda_k\mathbf G_k\mathbf s_k,
\qquad
\frac{\lVert\mathbf r_k\rVert_2}{\lVert\mathbf g_k\rVert_2}
\leq\eta_k.
$$

The accepted energy must be evaluated with the complete VBSCF objective. No
local-only, frozen-response, finite-difference, or energy-polishing fallback is
part of the production algorithm.

## 3. Current production realization

The implementation at the start of the article campaign contains:

1. the exact support-constrained quotient basis for HAO and OEO orbitals;
2. an analytic exact-context Hessian-vector product;
3. a trust-region subproblem solved in a reorthogonalized inner subspace;
4. a positive local orbital-curvature block preconditioner;
5. transported accepted-step L-BFGS secants and current-point exact block
   inverse-BFGS enrichment used only as preconditioning information;
6. residual-driven block-HVP expansion without a default iteration or
   subspace cap;
7. accepted-point curvature and generalized-eigen response recycling;
8. exact objective evaluation before every accepted outer step; and
9. per-accepted-step records of HVP work, KKT residual, model agreement,
   spectrum, trust radius, and curvature events.

The finite-difference HVP mode is diagnostic code and must not appear in timing
or convergence comparisons. An explicitly assembled reduced Hessian is allowed
only as a small-system reference oracle.

## 4. Historical baseline before the final solver revision

The following single-run results were regenerated with four OpenMP threads and
one BLAS thread on `node45` before the residual-driven uncapped solver was
completed. They are retained only as historical diagnostics and must not be
used as current or publication-quality timing statistics.

| System | Orbital chart | Outer steps | HVP directions | Final projected gradient 2-norm | SCF time / s |
|---|---|---:|---:|---:|---:|
| F2 | strict sparse HAO | 5 | 11 | 5.33625e-7 | 0.0230 |
| benzene (`241_VBSCF`) | strict sparse HAO | 10 | 39 | 2.55850e-7 | 15.8584 |
| MnF2 | strict sparse HAO | 15 | 183 | 4.15456e-5 | 37.74 |
| FeCl2 | full-AO OEO | 6 | 192 | 2.62862e-4 | 52.99 |

The historical traces showed why a fixed 32-direction budget was not a valid
convergence rule: several steps exhausted it while retaining unresolved KKT
residuals. That budget has been removed. Current runs terminate the inner solve
from the shifted KKT certificate, outer-accuracy condition, numerical
dependence, or algebraic completion.

FeCl2 starts only 1.34e-5 hartree above the final energy. It is a useful OEO
derivative, memory, and endpoint-conditioning test, but it is not by itself a
valid test of the nonlinear convergence basin or convergence order.

## 5. Production curvature-correction space

TNHVP starts every outer iteration from the transported, orbital-block-
preconditioned L-BFGS predictor

$$
\mathbf p_k^{\mathrm B}=-\mathbf B_k\mathbf g_k.
$$

The standard L-BFGS comparison does not use this orbital block; it uses
$H_k^{(0)}=\gamma_k I$ as specified in `tnhvp_lbfgs_benchmark.md`.

Exact curvature is admitted by an a posteriori forcing test rather than by a
fixed secant count. For the preceding accepted step, define the measured
nonlinear contraction

$$
q_{k-1}=\frac{\lVert\mathbf g_k\rVert_2}
              {\lVert\mathbf g_{k-1}\rVert_2}.
$$

If $q_{k-1}\leq\eta_{k-1}$, the inexpensive predictor has already delivered
the contraction requested of the inexact Newton solve, and the next step
remains on the cheap block-L-BFGS path. If $q_{k-1}>\eta_{k-1}$, TNHVP applies
the current-point exact relaxed Hessian first to $\mathbf p_k^{\mathrm B}$.
The predictor is retained as a fixed affine origin; its coefficient is not
reoptimized by the exact-curvature solver.  Its initial Newton defect is

$$
\mathbf r_k^{\mathrm B}=
\mathbf g_k+\mathbf H_k\mathbf p_k^{\mathrm B}.
$$

If

$$
\lVert\mathbf r_k^{\mathrm B}\rVert_2
\leq\eta_k\lVert\mathbf g_k\rVert_2,
$$

that single HVP certifies the predictor. Otherwise the correction space is
expanded by the raw and preconditioned Newton defects

$$
-\mathbf r_k^{\mathrm B},
\qquad
-\mathbf B_k\mathbf r_k^{\mathrm B}.
$$

Every admitted basis vector receives a current-point exact relaxed HVP,
including the structure response. With correction basis $\mathbf Q_k$, the
final step is restricted to the affine space

$$
\mathbf p_k=\mathbf p_k^{\mathrm B}+\mathbf Q_k\mathbf z_k,
$$

and minimizes

$$
m_k(\mathbf p_k)=
\mathbf g_k^{\mathrm T}\mathbf p_k+
\frac{1}{2}\mathbf p_k^{\mathrm T}\mathbf H_k\mathbf p_k
$$

subject to

$$
\mathbf p_k^{\mathrm T}\mathbf G_k\mathbf p_k\leq\Delta_k^2.
$$

Thus the projected linear term is

$$
\mathbf Q_k^{\mathrm T}
\left(\mathbf g_k+\mathbf H_k\mathbf p_k^{\mathrm B}\right),
$$

not $\mathbf Q_k^{\mathrm T}\mathbf g_k$. Exact HVP information therefore
corrects the Newton defect of block-LBFGS instead of re-solving the whole
quadratic model in a linear space that happens to contain the predictor.
Positive
Ritz curvature enriches the inverse preconditioner through an exact block
secant update, while nonpositive Ritz modes remain explicit in the projected
trust model.

The admission condition contains no molecule name, basis-size threshold,
orbital-type branch, outer-iteration index, or required number of secants. It
asks whether the measured nonlinear contraction met the same forcing target
used to certify the inner Newton equation. There is no default correction-rank
limit; the reduced coordinate dimension is the finite algebraic completion
bound, and failure to meet the residual target is never labelled as a
converged Newton solve.

## 6. Validation hierarchy

### 6.1 Derivative correctness

For F2 in both HAO and OEO representations, compare the analytic action with a
centered finite difference in one fixed accepted-point chart:

$$
\mathbf H_k\mathbf v
\approx
\frac{\mathbf g_k(+\epsilon\mathbf v)
      -\mathbf g_k(-\epsilon\mathbf v)}{2\epsilon}.
$$

Report maximum and relative action errors over normalized random directions,
the symmetry defect of an explicitly assembled small reduced Hessian, and the
energy agreement between the HAO and OEO stationary solutions where the two
representations describe the same physical state.

### 6.2 Local Newton reference

On the small F2 HAO and OEO spaces, assemble the reduced Hessian only inside a
diagnostic executable by applying the matrix-free operator to basis vectors.
Compare TNHVP with the global solution of the same dense trust-region model.
Required quantities are step angle, step-norm ratio, predicted decrease,
shifted KKT residual, accepted energy, and the number of exact HVP directions.
This test establishes proximity to exact second-order behavior without adding
a dense Hessian path to production.

### 6.3 Nonlinear convergence

Use the same unmodified input for TNHVP and L-BFGS. In addition,
construct deterministic quotient-space perturbations at several common,
dimensionless tangent norms. The perturbation generator, seed, norm, and
resulting input checksum must be archived. A method is not credited with a
better convergence basin when it succeeds only from an already optimized
starting point.

Report the projected gradient norm and energy error at every accepted outer
iteration. Near a nonsingular stationary point, estimate the observed local
order only over steps that remain above numerical noise and below the onset of
the asymptotic plateau. Do not infer quadratic convergence from total iteration
counts alone.

### 6.4 Performance and memory

Collect at least five measured repetitions after one untimed warm-up on one
exclusive compute node. Pin OpenMP threads, use one BLAS thread unless a
separate scaling experiment is being performed, and keep the compiler,
libraries, executable checksum, and input checksum fixed. Report the median and
interquartile range for wall time. Report maximum resident memory from the
external process measurement.

The principal efficiency quantities are:

- accepted outer iterations;
- objective-and-gradient evaluations and energy-only trial evaluations;
- exact HVP directions and block calls;
- time inside exact HVP actions;
- total SCF and end-to-end time;
- maximum resident memory;
- KKT success rate and relative residual by accepted step; and
- rejection, boundary, negative-curvature, and rescue events.

Large integral storage must be reported separately from optimizer working
memory. In particular, the approximately 10 GB resident set observed for the
benzene input cannot be attributed to Hessian storage without a component-level
memory measurement.

## 7. System matrix

The existing inputs define the initial validation matrix:

| Role | Systems | Evidence |
|---|---|---|
| analytic derivative oracle | F2 HAO, F2 OEO | action error, symmetry, dense reduced reference |
| compact molecular convergence | F2, benzene (`241_VBSCF`) | outer convergence, HVP cost, L-BFGS comparison |
| difficult open-shell sparse case | MnF2 | late-stage conditioning and convergence basin |
| full-AO OEO stress case | FeCl2 | OEO correctness, conditioning, memory, HVP cost |
| large structure-response case | `10698_VBSCF` | structure-response scaling and wall time |

This set is sufficient for software regression but not for the final article.
The publication set should add chemically independent examples spanning
closed-shell and open-shell bonding, multiple active-space sizes, several
numbers of VB structures, HAO sparsity ratios, and at least two full-AO OEO
cases. Systems must be selected before the final timing campaign and retained
whether their results favor TNHVP or L-BFGS.

## 8. Reproducible data layout

The canonical driver is `tools/run_optimizer_benchmarks.py`. Each immutable
campaign directory contains:

- `manifest.json` with the Git revision, dirty-worktree state, executable and
  input checksums, host, thread settings, and requested tolerances;
- `summary.tsv` with one row per independent run;
- raw standard output, standard error, external time, and command files; and
- one accepted-step `.tnhvp.tsv` file for every TNHVP run.

Raw output is the source of record. Article tables and plots must be generated
from campaign directories; values must not be copied manually from terminal
output. The historical `benchmarks/local_baseline_20260907` data predate the
corrected quotient coordinates and must not be combined with the final
campaign.

## 9. Method-freeze gate

Publication data collection begins only after:

1. the correction-space solver decision is complete;
2. all derivative and trust-region tests pass;
3. the production code contains one TNHVP path and no experimental runtime
   policies;
4. deterministic perturbed inputs are generated and checksummed;
5. the explicit small-system Newton reference is automated; and
6. the complete benchmark system list, compiler, node type, and thread policy
   are frozen.

After this gate, any algorithmic change requires a new campaign directory and
a new Git revision. Data from different revisions may be shown as development
evidence but may not be pooled into one final performance table.
