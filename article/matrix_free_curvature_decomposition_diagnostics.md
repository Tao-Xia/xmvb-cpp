# Matrix-Free Curvature Decomposition and HVP Consistency Diagnostics

## Purpose and scope

This audit separates deficiencies of the local preconditioner from
cross-orbital coupling and the computational outer-response contribution.
It does not change the optimizer, trust-region radius, HVP definition,
transport history, or convergence tolerances. All matrices representing
orbital Hessians remain implicit. Only small sampled subspace matrices are
formed.

**Historical finding (before the OEO derivative corrections):** FeCl2 violates the expected linearity of the evaluated
HVP in the endpoint probe below. The matrix notation in the definitions
specifies the intended mathematical operators. Until this discrepancy is
resolved, sampled spectra and component splits are diagnostic observations,
not validated physical Hessian decompositions or a basis for replacing the
production solver.

The subsequent [full-AO OEO derivative correction](full_ao_oeo_derivative_validation.md)
localizes and corrects normalization, inactive-projector, and near-singular
cofactor derivative defects. The numerical data below remain a record of
the pre-correction implementation, not its current validation status.

An important distinction is that an implementation-stage decomposition need
not be a decomposition into individually symmetric physical Hessians.
Consequently, the diagnostic does not assume that an isolated stage can be
used as a Hessian in CG. The initial exploratory CG comparison was replaced
by a minimum-residual linear probe before drawing conclusions from component
solves.

## Operator definitions

All actions use the same accepted-point quotient chart: sparse support for
HAO inputs and full AO support for OEO inputs. Let
$\mathbf H$ denote the complete `exact_ctx` action. Define $\mathbf A$ by
enabling `direct_core_response` and `fixed_upstream_pullback` and disabling
`outer_response`. Define $\mathbf C$ by enabling only `outer_response`.
The additive identity to be checked numerically is

$$
\mathbf H\mathbf v=\mathbf A\mathbf v+\mathbf C\mathbf v.
\tag{1}
$$

These labels specify code paths, not a claim that $\mathbf A$ is the exact
fixed-CI Hessian or that $\mathbf C$ is solely a negative-semidefinite CI
relaxation correction. The outer path includes active-integral, structure,
and adjoint-response work. Symmetry must be measured, not assumed for each
component.

Let $\mathbf T_p$ be the orthogonal coordinate projector onto target orbital
$p$ in the reduced chart. The block-diagonal construction of the local
quotient bases gives

$$
\mathbf T_p^{\mathrm T}=\mathbf T_p,
\qquad
\mathbf T_p\mathbf T_q=\delta_{pq}\mathbf T_p,
\qquad
\sum_p\mathbf T_p=\mathbf I.
\tag{2}
$$

The implementation expands a reduced vector into packed coefficients, masks
all other target orbitals, and projects back into the same chart. It does
not construct the full reduced basis. Define the exact diagonal-target
part of the computational core action by

$$
\mathbf D\mathbf v=\sum_p\mathbf T_p\mathbf A\mathbf T_p\mathbf v.
\tag{3}
$$

This requires one core HVP for each nonzero orbital component of a probe;
it does not assemble the diagonal blocks themselves. Let $\mathbf B$ be the
production positive local preconditioning matrix, including inactive double
occupancy but excluding transported L-BFGS corrections. Then

$$
(\mathbf H-\mathbf B)\mathbf v
=\underbrace{(\mathbf D-\mathbf B)\mathbf v}_{\text{local model difference}}
+\underbrace{(\mathbf A-\mathbf D)\mathbf v}_{\text{cross-target core coupling}}
+\underbrace{\mathbf C\mathbf v}_{\text{outer response}}.
\tag{4}
$$

The reported component ratios divide each vector norm in eq 4 by
$\|\mathbf H\mathbf v\|_2$. They are **not percentages of an error budget**:
the vectors can cancel and their norms can exceed the denominator by large
factors. Signed Rayleigh contributions are reported separately. Because
$\mathbf B$ approximates the complete Hessian rather than just $\mathbf D$,
the labels in eq 4 describe an algebraic decomposition, not independent
causal errors that can be fixed one at a time.

## Direction selection and validation

At a fixed point, the full and core linear probes start from zero with
right-hand side $-\mathbf g$ and the same production base inverse
$\mathbf B^{-1}$. Each iteration expands the subspace using the
preconditioned residual. The new direction is orthonormalized before its
fresh HVP is evaluated. With accumulated columns $\mathbf Q$ and
$\mathbf W=\mathbf H\mathbf Q$ (or $\mathbf A\mathbf Q$), the diagnostic
computes

$$
\mathbf y=\operatorname*{arg\,min}_{\mathbf z}
  \|\mathbf g+\mathbf W\mathbf z\|_2,
\qquad
\mathbf s=\mathbf Q\mathbf y.
\tag{5}
$$

The least-squares solve uses pivoted QR, not normal equations. This probe
does not require operator symmetry or positive definiteness. It uses at
most the requested number of directions (32 in the reported runs), stopping
earlier at a square-root-machine-precision relative residual or loss of a
new independent direction. It is a diagnostic, not a proposed production
solver. It ignores the outer trust boundary and transported history, so its
residuals must not be equated to production inner-solver results.

The four unit-norm probes are the gradient, the full minimum-residual step,
its residual, and the lowest-eigenvalue direction of the symmetric part of
$\mathbf Q^{\mathrm T}\mathbf H\mathbf Q$. A normalized residual after an
essentially exact solve is noise-dominated and should not be interpreted as
a remaining physical difficulty. Generalized sampled eigenvalues compare
this symmetric part against $\mathbf Q^{\mathrm T}\mathbf B\mathbf Q$.
Neither spectrum certifies the full-space Hessian or global conditioning.

The audit reports relative skew norms before any symmetrization and verifies
eq 1 using an independently requested outer-only HVP. The soft-direction
HVP is also compared with central differences of the relaxed gradient at
steps $10^{-4}$, $10^{-5}$, and $10^{-6}$, always projecting into the fixed
accepted-point chart. If the sampled full skew exceeds a
square-root-machine-precision relative scale, the largest-skew direction
pair is additionally checked with gradient finite differences.
Separate operator instances are used for full, core-only, and outer-only
actions, ruling out component-mode switching within one instance as an
explanation for the reported discrepancy. Repeatability, soft-direction
linearity, and sign/scaling homogeneity are checked separately.

## FeCl2 findings

The endpoint was generated by the unchanged production optimizer: six outer
iterations and 192 HVP directions, energy -2181.617645346998 Eh, and printed
projected gradient norm $2.78093662\times10^{-4}$. Reloading the full coefficient
table for independent evaluation gives gradient norm
$2.78206962\times10^{-4}$; this small difference is recorded rather than
assuming bitwise identical accepted-point and reloaded evaluations.

### A consistency failure precedes preconditioner redesign

For the sampled soft direction $\mathbf v=\mathbf Q\mathbf z$, compare a
fresh action against the linear combination of the already evaluated basis
images:

$$
e_{\mathrm{lin}}=
\frac{\|\mathcal H(\mathbf Q\mathbf z)
       -[\mathcal H(\mathbf q_1),\ldots,\mathcal H(\mathbf q_k)]\mathbf z\|_2}
     {\|\mathcal H(\mathbf Q\mathbf z)\|_2}.
\tag{6}
$$

At the FeCl2 endpoint, $e_{\mathrm{lin}}=0.0122438601$, approximately 1.2%.
Repeating the same selected basis-direction action gives zero printed
discrepancy. The independent-instance run reproduces the shared-instance
audit, so neither random call-to-call noise nor switching component flags
within one instance explains this result.

Decomposing eq 6 by computational stage localizes the discrepancy. Both
stage errors below use the complete soft-direction HVP norm as denominator:

| Point | Complete action linearity error | Core contribution error | Outer contribution error |
|---|---:|---:|---:|
| F2 input | $9.24\times10^{-15}$ | $8.99\times10^{-15}$ | $1.35\times10^{-15}$ |
| 241 input | $3.70\times10^{-11}$ | $2.74\times10^{-13}$ | $3.69\times10^{-11}$ |
| MnF2 input | $1.69\times10^{-12}$ | $1.53\times10^{-12}$ | $5.88\times10^{-13}$ |
| FeCl2 endpoint | $1.224386\times10^{-2}$ | $6.44\times10^{-13}$ | $1.224386\times10^{-2}$ |

Changing the sign or doubling the soft direction gives zero printed
homogeneity discrepancy at these points. The defect is therefore not exposed
by these simple scaling checks; combining independently evaluated directions
is essential. This localizes the FeCl2 failure to the outer-response action,
without yet distinguishing an incorrect formula, direction-dependent branch,
or severe numerical cancellation inside that action. It is not evidence of
a new defect in the sparse quotient or proof that this discrepancy alone
explains all of FeCl2's slow convergence.

The nearest selected-root gap is approximately 0.00624404 Eh at the input
and 0.00625371 Eh at the endpoint (`FeCl2.initial.gap.log` and
`FeCl2.final.gap.log`). The selected root is therefore not exactly degenerate
at either audited point; exact root degeneracy is not an established
explanation for the observed failure of linearity.

The sampled complete-action skew norm relative to its projected matrix norm
is $8.1493\times10^{-6}$. For its largest-skew direction pair, the analytic
bilinear mismatch is $-7.2494\times10^{-5}$, whereas independent gradient
central differences give approximately $5.25\times10^{-7}$. The selected
direction's HVP/gradient-difference relative discrepancy stays near
$1.3154\times10^{-3}$ for steps $10^{-4}$, $10^{-5}$, and $10^{-6}$.
Its persistence over this range argues against ordinary finite-difference
step selection being the sole explanation. The soft direction by itself
has much smaller HVP/gradient-difference error, approximately
$1.20\times10^{-6}$ at step $10^{-4}$: passing one direction does not validate
the whole operator.

The input point also shows a discrepancy, but with a different pattern:
$e_{\mathrm{lin}}=4.5970\times10^{-5}$ and relative sampled skew
$3.9535\times10^{-6}$. There, the largest bilinear skew
($-2.89665\times10^{-5}$) is reproduced by gradient finite differences
($-2.89608\times10^{-5}$ at step $10^{-4}$). Thus the input-point and endpoint
observations should not be collapsed into a single asserted root cause.
Energy/gradient consistency and HVP linearization both require examination.

These checks expose a previously untested failure: agreement of the HVP with
one gradient finite difference does not establish linearity, self-adjointness,
or consistency with the scalar energy. The exact failing contraction or
branch has not yet been established, and no production fix is claimed here.

### Why a gradient-only response estimate was misleading

At the input point, the response norm on the gradient direction is only
approximately $4.79\times10^{-4}$ times the complete HVP norm. On the sampled
soft direction it is approximately 89 times that norm, while the core
cross-target contribution is approximately 83 times that norm. These large
contributions cancel strongly. At the endpoint, their corresponding ratios
are approximately 28 and 26. The endpoint local-model difference is also
large, approximately 24 times the complete HVP norm.

The endpoint signed soft-direction contributions illustrate this cancellation:

| Evaluated contribution | Directional curvature |
|---|---:|
| Local positive model | 0.161110 |
| Diagonal-target computational core | 0.155559 |
| Cross-target computational core | -0.114672 |
| Outer response | -0.037198 |
| Fresh complete action | 0.003689 |

Because of the consistency failure above, these numbers identify directions
and code paths requiring investigation; they do not validate a physical
coupled approximation. In particular, merely omitting response, adding
core coupling alone, or symmetrizing the small matrix does not repair an
underlying nonlinear HVP action.

The fixed-point minimum-residual probe reduces the initial FeCl2 relative
residual to approximately 0.0558 with 32 full actions. At the endpoint it
only reaches approximately 0.3558. Solving the core-only system within the
same direction allowance gives complete-model residuals of approximately
0.1080 and 0.4195, respectively. These are diagnostic linear-probe residuals,
not production CG counts, and the nonlinearity limits their interpretation.

### Cost remains concentrated in response

The independent-instance endpoint benchmark evaluates one warmup and three
timed repetitions per component, with four OpenMP threads and one OpenBLAS
thread. Full response remains the dominant runtime stage; a two-column block
does not provide a large speedup over two scalar calls. The independent run
measured approximately 0.265 s per full action, including 0.241 s in outer
response; core-only cost was approximately 0.088 s, and the two-column block
speedup was approximately 0.995. These are short measurements on a shared
host, not a statistically replicated timing study. Timing measures the
current implementation's cost, not its mathematical correctness. The
consistency failure should be resolved before changing which response terms
the optimizer uses.

## Reproduction

```bash
cmake --build build --target benchmark_exact_ctx_hvp test_curvature_decomposition -j 4
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build/src/test_curvature_decomposition
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 build/src/benchmark_exact_ctx_hvp INPUT.xmi --nonredundant-adapt true --curvature-audit-directions 32 --warmup 1 --repeats 3
```

To inspect a production endpoint, first copy the input to scratch because
the optimizer may write a requested Molden file beside it:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 build/src/xmvb-cpp.exe SCRATCH/FeCl2.xmi --optimizer-backend nonredundant_truncated_newton --dump-final-orbital-value-table-bin SCRATCH/FeCl2.final.bin
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 build/src/benchmark_exact_ctx_hvp SCRATCH/FeCl2.xmi --nonredundant-adapt true --orbital-value-table-bin SCRATCH/FeCl2.final.bin --curvature-audit-directions 32 --warmup 1 --repeats 3
```

The binary override must come from the same input support and adaptation
convention. The diagnostic checks its size and finiteness; raw binary data
does not carry a support identifier. The independent synthetic decomposition
test uses known dense operators only as a tiny test oracle and checks each
term of eq 4 separately, including cross-target blocks and a nonzero outer
response.

## Artifacts and verification status

All input copies, endpoint coefficients, executable snapshots, and logs are
under `/tmp/xmvb-curvature-audit.u4C2y4/`. The retained component-linearity
results use `audit-final.exe`, `FeCl2.final.component-linearity.log`, and
`F2.final-audit.log`, `241_VBSCF.final-audit.log`, `MnF2.final-audit.log`.
The input and endpoint independent-instance comparisons are
`FeCl2.independent.log` and `FeCl2.final.independent.log`. The exploratory
`*.cg-probe.log` files are superseded and must not be interpreted as valid
core-Hessian CG benchmarks. `FeCl2.optimization.log` and `FeCl2.final.bin`
reproduce the production endpoint used here.

Ten selected automated tests pass, including the new synthetic decomposition
test, an end-to-end F2 audit smoke test, and the previous eight regressions.
This does **not** certify FeCl2: the direction-resolved endpoint audit exposes
a failure outside the coverage of those tests. Production optimization and
HVP code were not modified during this diagnostic step. The next task is to
isolate and correct the outer-response linearity discrepancy before drawing
optimization conclusions from new coupled preconditioners.
