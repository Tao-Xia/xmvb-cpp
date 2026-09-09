# Full-AO OEO Orbital Derivatives: Corrections and Validation

## Scope

OEO orbitals have full AO support in the present input convention. They must
not be interpreted as strictly sparse HAO orbitals. Full support is the
identity-support special case of the coefficient-space formulation; it does
not imply that arbitrary coefficient perturbations preserve orthogonality.

This note records three independent derivative defects and their corrections.
The orbital Hessian remains implicit. The solver, trust-region policy,
convergence tolerances, and inner iteration budget are unchanged.

## 1. Normalization is part of the differentiated map

Let $\mathbf c_p$ be a raw AO coefficient vector and define

$$
n_p=(\mathbf c_p^{\mathrm T}\mathbf S\mathbf c_p)^{1/2},
\qquad
\mathbf q_p=\frac{\mathbf c_p}{n_p}.
\tag{1}
$$

For an upstream gradient $\mathbf b_p=\partial E/\partial\mathbf q_p$,
the raw-coefficient gradient is

$$
\frac{\partial E}{\partial\mathbf c_p}
=\frac{\mathbf b_p}{n_p}
-\frac{\mathbf S\mathbf c_p}{n_p^3}
 (\mathbf c_p^{\mathrm T}\mathbf b_p).
\tag{2}
$$

The old OEO-only return scattered $\mathbf b_p$ directly into the raw
coefficient slots. However, the forward orbital preparer normalizes both
OEO and HAO coefficients. The omitted chain rule therefore changed the
gradient rather than merely choosing a different coordinate representation.
The corresponding second-order pullback must differentiate Eq. (2), too.

Both OEO-only bypasses have been removed. An independent scalar objective
of the prepared active orbitals and inactive density tests every raw
coefficient of an unnormalized, nonorthogonal full-support example. It also
checks invariance under changing only the orbital-type label and verifies
$\mathbf c_p^{\mathrm T}\partial E/\partial\mathbf c_p=0$.

## 2. Orthogonality at one point is not an orthogonality constraint

For normalized inactive coefficients $\mathbf Q_{\mathrm I}$, define

$$
\mathbf M=\mathbf Q_{\mathrm I}^{\mathrm T}\mathbf S\mathbf Q_{\mathrm I},
\qquad
\mathbf A=\mathbf Q_{\mathrm I}\mathbf M^{-1},
\qquad
\mathbf P=\mathbf A\mathbf Q_{\mathrm I}^{\mathrm T}.
\tag{3}
$$

The exact directional derivatives include

$$
\begin{aligned}
\dot{\mathbf M}
&=\dot{\mathbf Q}_{\mathrm I}^{\mathrm T}\mathbf S\mathbf Q_{\mathrm I}
 +\mathbf Q_{\mathrm I}^{\mathrm T}\mathbf S\dot{\mathbf Q}_{\mathrm I},\\
\dot{\mathbf M}^{-1}
&=-\mathbf M^{-1}\dot{\mathbf M}\mathbf M^{-1},\\
\dot{\mathbf A}
&=\dot{\mathbf Q}_{\mathrm I}\mathbf M^{-1}
 +\mathbf Q_{\mathrm I}\dot{\mathbf M}^{-1}.
\end{aligned}
\tag{4}
$$

Even when $\mathbf M=\mathbf I$ at the accepted point,
$\dot{\mathbf M}^{-1}=-\dot{\mathbf M}$ is not generally zero. The former
OEO orthonormal-inactive shortcut incorrectly used constant-metric
directional formulas for unrestricted coefficient perturbations. It also
specialized a pullback before differentiating it. These shortcuts have been
removed; the same complete projector derivatives now apply to all supports.

The new `F2_OEO.xmi` regression starts from a full-AO OEO guess and exposes
this error. It has a different variational space from the existing HAO F2
benchmark and must not be used as a like-for-like performance comparison.

## 3. Polynomial cofactor actions replace unstable inverse identities

Let $\mathbf X$ be a spin-determinant overlap block, not the AO metric or
the orbital Hessian. Write its first cofactor matrix as

$$
\mathbf C^{(1)}(\mathbf X)=\nabla_{\mathbf X}\det\mathbf X.
\tag{5}
$$

For nonsingular $\mathbf X$, the identity
$\mathbf C^{(1)}=\det(\mathbf X)\mathbf X^{-\mathrm T}$ is exact.
Differentiating inverse-based expressions, however, introduces large
intermediate inverse powers whose cancellation is numerically unreliable
for nearly singular determinant-pair overlaps. Full-AO calculations can
encounter such overlaps even when the AO metric and the selected VB root
are well conditioned. This is not an OEO-specific mathematical formula.

The replacement uses an accepted-point SVD

$$
\mathbf X=\mathbf U\boldsymbol\Sigma\mathbf V^{\mathrm T},
\qquad
\eta=\det\mathbf U\det\mathbf V,
\qquad
p_I=\eta\prod_{k\notin I}\sigma_k.
\tag{6}
$$

The factors $\mathbf U$ and $\mathbf V$ are held fixed while evaluating
directional derivatives. No singular-vector differentiation, inverse
singular values, or rank threshold occurs in these cofactor actions.
Complementary products are formed directly, including at zero singular
values. For $\widetilde{\mathbf D}=\mathbf U^{\mathrm T}\mathbf D\mathbf V$,
the first cofactor derivative in this fixed diagonal chart is

$$
\begin{aligned}
[D\widetilde{\mathbf C}^{(1)}[\widetilde{\mathbf D}]]_{ii}
 &=\sum_{j\ne i}p_{\{i,j\}}\widetilde D_{jj},\\
[D\widetilde{\mathbf C}^{(1)}[\widetilde{\mathbf D}]]_{ij}
 &=-p_{\{i,j\}}\widetilde D_{ji},\qquad i\ne j.
\end{aligned}
\tag{7}
$$

The result is transformed back with $\mathbf U$ and $\mathbf V^{\mathrm T}$.
Mixed first-cofactor derivatives similarly use three-index complementary
products. Second cofactors are indexed by ordered pairs $i<j$ and $k<l$;
their transformations use the second compound matrices of $\mathbf U$ and
$\mathbf V$. Their contracted gradient directions use complementary
products with at most four excluded indices.

With the Frobenius inner product $\langle\mathbf A,\mathbf B\rangle
=\operatorname{tr}(\mathbf A^{\mathrm T}\mathbf B)$, a same-spin Hamiltonian
element has the polynomial representation

$$
H_\sigma
=\langle\mathbf h,\mathbf C^{(1)}(\mathbf X)\rangle
 +\langle\mathbf G,\mathbf C^{(2)}(\mathbf X)\rangle,
\tag{8}
$$

where $\mathbf G$ contains the antisymmetrized two-electron integrals on
the determinant's occupied pair indices. Differentiating Eq. (8) gives
the directional scalar, overlap gradient, and overlap-gradient direction
without forming an orbital Hessian. The same polynomial actions cover
regular and rank-deficient pairs in the matrix-form response path.

For the local opposite-spin contraction
$E_{\alpha\beta}=\langle\mathbf W,\mathbf C^{(1)}(\mathbf X)\rangle$,
the required overlap-gradient direction is

$$
D(\nabla_{\mathbf X}E_{\alpha\beta})
=D^2\mathbf C^{(1)}(\mathbf X)[\dot{\mathbf X},\mathbf W]
 +D\mathbf C^{(1)}(\mathbf X)[\dot{\mathbf W}].
\tag{9}
$$

This replaces the determinant/inverse/projection cancellation chain.
The computational label `outer_response` includes these local derivative
terms; the original failure was not evidence of a nonlinear CI eigensolver.

## 4. Independent checks and implementation scope

`test_cofactor_differential` compares first and second cofactor values and
derivatives against deleted-minor and replaced-column determinant
references. Tests cover dimensions zero through seven, nonsymmetric
matrices, small singular values, and rank deficiencies one through four.
Additional checks cover pushforward/pullback duality, directional linearity,
mixed-derivative symmetry, and finite differences of contracted gradients.
`test_oeo_normalization_pullback` tests Eq. (2) independently of the VB
Hamiltonian. The HAO and OEO F2 integration tests compare complete HVPs
against orbital-gradient differences.

The accepted overlap blocks are retained explicitly so that derivative
evaluation does not reconstruct an ill-conditioned block by inverting its
inverse. Memory accounting includes this additional determinant-local
storage. Temporary compound matrices are occupied-pair tensors, not the
orbital Hessian. There are no molecule-dependent thresholds or new solver
controls in this correction.

These changes target the matrix-form `exact_ctx` path exercised below.
They do not certify every legacy determinant-pair fallback or arbitrary
degenerate-root calculation. Correct derivatives also do not establish
quadratic convergence when the inner Newton equation remains inaccurate.

## 5. Fixed-point evidence and remaining performance work

The saved FeCl2 endpoint from the preceding optimizer is used to avoid
confounding derivative changes with a different orbital geometry. With
32 sampled directions, the staged results are:

| Implementation | Relative HVP linearity error | Relative projected skew |
|---|---:|---:|
| Before these corrections | $1.2244\times10^{-2}$ | $8.149\times10^{-6}$ |
| Normalization chain corrected | $5.7209\times10^{-3}$ | $1.1037\times10^{-5}$ |
| Inverse-free opposite-spin response | $4.9876\times10^{-7}$ | $2.2474\times10^{-9}$ |
| Polynomial same-spin and opposite-spin response | $1.8972\times10^{-11}$ | $8.7051\times10^{-15}$ |

The last row uses the fixed-factor, batched compound implementation, not
the slower repeated-minor reference prototype. Its soft-direction HVP
agrees with a central gradient difference to approximately
$1.37\times10^{-7}$ at step $10^{-4}$. Smaller steps increase cancellation
in the reference gradient difference; they do not improve this check.

These rows establish correctness independently of subsequent acceleration.
Accepted-point factor reuse and cross-module directional-pair reuse are
reported separately in
`article/matrix_free_hvp_performance_validation.md`. No convergence tolerance
or CG budget was adjusted to obtain the final kernel speedup.

Scratch logs and executable snapshots are in
`/tmp/xmvb-oeo-fix.Cdw3GW/` on the validation machine; they are not
repository-distributed benchmark assets.

### Final-build integration checks

All 13 configured standalone CTests passed. The additional full-AO F2
audit gives a relative projected skew of $2.35\times10^{-16}$ and a
linearity error of $2.31\times10^{-13}$. Before removal of the
orthonormal-inactive shortcut, that same input had a relative skew of
approximately $0.084$ and a soft-direction gradient-difference discrepancy
of approximately $0.55$.

At the FeCl2 initial point, a separate full-HVP direction test yields
relative gradient-difference errors $5.69\times10^{-7}$ and
$5.69\times10^{-9}$ for steps $10^{-4}$ and $10^{-5}$, respectively.
The approximately hundredfold reduction is consistent with second-order
central-difference truncation error. The $10^{-5}$ run passes the specified
$10^{-7}$ integration-test error limit; the larger-step run does not.

The original and corrected FeCl2 executables were rerun with four OpenMP
threads and one BLAS thread, without changing solver settings:

| FeCl2 run | Outer iterations | HVP directions | Final total energy / $E_{\mathrm h}$ | Wall time / s |
|---|---:|---:|---:|---:|
| Repeated old baseline | 6 | 192 | -2181.617645346998 | 55.66 |
| Corrected final build | 6 | 192 | -2181.617645350183 | 76.98 |

The corrected final projected gradient norm is $2.63\times10^{-4}$.
Both runs meet the existing outer stopping rule, but neither reaches the
inner KKT target in any of the six solves. Thus this correction does **not**
yet improve outer iteration counts or demonstrate quadratic convergence.
The observed wall-time overhead is approximately 38%; accepted-factor and
response-intermediate reuse remains necessary. These are single-run timing
observations, not statistically controlled performance measurements.

The HAO regressions with the new polynomial response kernels gave:

| Input | Old iterations / HVPs | Corrected iterations / HVPs | Corrected final total energy / $E_{\mathrm h}$ |
|---|---:|---:|---:|
| F2 HAO | 5 / 11 | 5 / 11 | -198.751155830527 |
| 241 HAO | 10 / 39 | 10 / 39 | -230.720590392895 |
| MnF2 HAO | 14 / 165 | 14 / 166 | -1348.893353224033 |

These HAO runs used `spectral-polynomial.exe` before deletion of the
OEO-only orthonormal-inactive shortcut; that deletion does not change the
HAO path. FeCl2 was additionally rerun with the final executable.
The MnF2 final gradient infinity norm is $2.97\times10^{-5}$, versus
$6.07\times10^{-6}$ in the preceding run: identical outer stopping
criteria do not imply identical final residuals or trajectories.

The later optimized production results superseding the timing rows in this
correctness note are reported in
`article/matrix_free_hvp_performance_validation.md`.

Representative reproduction commands, executed from the repository root:

```sh
ctest --test-dir build --output-on-failure
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build/src/benchmark_exact_ctx_hvp testdata/vbscf/F2_OEO.xmi --nonredundant-adapt true --curvature-audit-directions 6
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 build/src/check_exact_ctx_hvp test/FeCl2.xmi --step 1e-5 --probe full --nonredundant-adapt true --max-rel-error 1e-7
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 build/src/benchmark_exact_ctx_hvp test/FeCl2.xmi --nonredundant-adapt true --orbital-value-table-bin /tmp/xmvb-oeo-fix.Cdw3GW/FeCl2.final.bin --curvature-audit-directions 32
```
