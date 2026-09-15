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

where the accepted-point basis is whitened in the pullback metric of the
per-orbital normalization maps,

$$
\mathbf U_k^{\mathrm T}\mathbf M_k\mathbf U_k=\mathbf I,
\qquad
\mathbf M_k
=
\bigoplus_p
\mathbf J_{N,p}^{\mathrm T}\mathbf S_p\mathbf J_{N,p}.
$$

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
\lVert\mathbf s\rVert\leq\Delta_k.
$$

An inner solution is certified with the shifted KKT residual

$$
\mathbf r_k
=\mathbf g_k+\mathbf H_k\mathbf s_k+\lambda_k\mathbf s_k,
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
5. transported accepted-step and positive Ritz secants used only as
   preconditioning information;
6. exact objective evaluation before every accepted outer step; and
7. per-accepted-step records of HVP work, KKT residual, model agreement,
   spectrum, trust radius, and curvature events.

The finite-difference HVP mode is diagnostic code and must not appear in timing
or convergence comparisons. An explicitly assembled reduced Hessian is allowed
only as a small-system reference oracle.

## 4. Baseline state before the final solver revision

The following single-run results were regenerated with four OpenMP threads and
one BLAS thread on `node45`. They verify the current numerical state but are not
publication-quality timing statistics.

| System | Orbital chart | Outer steps | HVP directions | Final projected gradient 2-norm | SCF time / s |
|---|---|---:|---:|---:|---:|
| F2 | strict sparse HAO | 5 | 11 | 5.33625e-7 | 0.0230 |
| benzene (`241_VBSCF`) | strict sparse HAO | 10 | 39 | 2.55850e-7 | 15.8584 |
| MnF2 | strict sparse HAO | 15 | 183 | 4.15456e-5 | 37.74 |
| FeCl2 | full-AO OEO | 6 | 192 | 2.62862e-4 | 52.99 |

The accepted-step trace identifies the present limiting behavior more clearly
than aggregate counts. The final three MnF2 steps each consume the full
32-direction safety budget. Their actual-to-predicted decrease ratios remain
close to unity, but their KKT residuals do not satisfy the requested inner
accuracy. All six FeCl2 steps also consume 32 HVPs; their model ratios are close
to unity, with no rejection and no negative-curvature event, while their KKT
relative residuals remain between approximately 0.31 and 0.98. Increasing the
fixed work limit would therefore attack the symptom rather than the spectral
conditioning problem.

FeCl2 starts only 1.34e-5 hartree above the final energy. It is a useful OEO
derivative, memory, and endpoint-conditioning test, but it is not by itself a
valid test of the nonlinear convergence basin or convergence order.

## 5. Final algorithmic milestone

The remaining solver milestone is an operator-aligned correction-space method,
not another molecule-dependent Krylov budget. The working decomposition is

$$
\mathbf H_k=\mathbf A_k+\mathbf R_k,
$$

where the first operator contains the direct and fixed-upstream orbital response
and the second contains the relaxed structure response. Both remain available
only as matrix-free actions. The existing local orbital blocks provide the
lowest-cost positive approximation to the first operator.

The next solver candidate should use a Davidson-type trust-region correction
space. Its orthonormal basis and exact images satisfy

$$
\mathbf Y_j=\mathbf H_k\mathbf Q_j.
$$

It solves the small symmetric
trust-region problem, forms the full KKT residual, and generates a new
preconditioned correction direction. The basis expansion is driven by observed
residual components and Ritz information, rather than by a fixed number of
iterations. This differs from the previously rejected transported block-HVP
experiment: old-point Ritz vectors are not accepted as a new-point model, and
every column entering the current projected Hessian receives a current-point
exact HVP.

The implementation is eligible for production only if it satisfies all of the
following conditions:

- no system name, basis size, orbital type, or iteration index appears in a
  decision rule;
- the HAO and OEO finite-difference HVP regressions remain unchanged;
- F2 and benzene do not require more HVP directions or outer iterations;
- both MnF2 and FeCl2 show lower HVP counts and lower wall time, not merely one
  of those systems;
- accepted energies agree with the current reference within the converged
  optimization tolerance;
- improvement persists across deterministic perturbed starting points; and
- every reported current-point projected model uses exact current-point HVP
  images.

Failure of any condition means that the candidate remains a diagnostic branch
and is not exposed as a production option.

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

Use the same unmodified input for TNHVP and nonredundant L-BFGS. In addition,
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
