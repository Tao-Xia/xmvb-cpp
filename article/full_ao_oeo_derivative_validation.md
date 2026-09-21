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

## 3. Rank-aware exact cofactor actions

Let $\mathbf X$ be a spin-determinant overlap block, not the AO metric or
the orbital Hessian. Write its first cofactor matrix as

$$
\mathbf C^{(1)}(\mathbf X)=\nabla_{\mathbf X}\det\mathbf X.
\tag{5}
$$

For a nonsingular $\mathbf X$, define

$$
\mathcal D=\det(\mathbf X),
\qquad
\mathbf R=\mathbf X^{-1}.
\tag{5a}
$$

Then the first cofactor and its directional derivative are

$$
\begin{aligned}
\mathbf C^{(1)}
&=\mathcal D\mathbf R^{\mathrm T},\\
D\mathbf C^{(1)}[\boldsymbol\Delta]
&=\mathcal D\left[
\operatorname{tr}(\mathbf R\boldsymbol\Delta)\mathbf R^{\mathrm T}
-\left(\mathbf R\boldsymbol\Delta\mathbf R\right)^{\mathrm T}
\right].
\end{aligned}
\tag{5b}
$$

The production implementation uses these Jacobi identities only for
numerically regular overlap blocks. Blindly applying them to nearly singular
determinant pairs introduces large inverse powers whose cancellation is
unreliable. Full-AO calculations can encounter such pairs even when the AO
metric and selected VB root are well conditioned.

The replacement uses an accepted-point SVD

$$
\mathbf X=\mathbf U\boldsymbol\Sigma\mathbf V^{\mathrm T},
\qquad
\eta=\det\mathbf U\det\mathbf V,
\qquad
p_I=\eta\prod_{k\notin I}\sigma_k.
\tag{6}
$$

For ill-conditioned full-rank blocks and all rank-deficient blocks, the
factors $\mathbf U$ and $\mathbf V$ are held fixed while evaluating the
inverse-free low-order cofactor polynomials. No singular-vector
differentiation or inverse singular value enters these expressions.
Complementary products are formed directly, including at zero singular
values. For
$\widetilde{\boldsymbol\Delta}
=\mathbf U^{\mathrm T}\boldsymbol\Delta\mathbf V$,
the first cofactor derivative in this fixed diagonal chart is

$$
\begin{aligned}
[D\widetilde{\mathbf C}^{(1)}[\widetilde{\boldsymbol\Delta}]]_{ii}
 &=\sum_{j\ne i}p_{\{i,j\}}\widetilde\Delta_{jj},\\
[D\widetilde{\mathbf C}^{(1)}[\widetilde{\boldsymbol\Delta}]]_{ij}
 &=-p_{\{i,j\}}\widetilde\Delta_{ji},\qquad i\ne j.
\end{aligned}
\tag{7}
$$

The result is transformed back with $\mathbf U$ and $\mathbf V^{\mathrm T}$.
Mixed first-cofactor derivatives similarly use three-index complementary
products. These expressions are exact polynomials, not compatibility
fallbacks. Section 3.2 gives the exact interpolation used to avoid the much
larger fourth-order complementary-product representation required by the
second-cofactor response.

### 3.1 Direct exterior-algebra contraction for regular pairs

Let $i<k$ and $j<l$ index occupied-orbital pairs. For a regular overlap block,
the second cofactor is

$$
C^{(2)}_{(i,k),(j,l)}
=\mathcal D\left(R_{ji}R_{lk}-R_{jk}R_{li}\right).
\tag{7a}
$$

Define the exterior-square kernel and its contraction with an
antisymmetrized two-electron weight matrix $\mathbf W$ by

$$
\begin{aligned}
A_{(i,k),(j,l)}(\mathbf R)
&=R_{ji}R_{lk}-R_{jk}R_{li},\\
F(\mathbf R,\mathbf W)
&=\langle\mathbf W,\mathbf A(\mathbf R)\rangle,\\
\mathbf Q(\mathbf R,\mathbf W)
&=\frac{\partial F}{\partial\mathbf R}.
\end{aligned}
\tag{7b}
$$

Since

$$
\dot{\mathcal D}=\mathcal D\tau,
\qquad
\tau=\operatorname{tr}(\mathbf R\boldsymbol\Delta),
\qquad
\dot{\mathbf R}=-\mathbf R\boldsymbol\Delta\mathbf R,
\tag{7c}
$$

the overlap gradient of the contracted second cofactor is

$$
\mathbf G_{\mathbf X}
=\nabla_{\mathbf X}\left[\mathcal D F(\mathbf R,\mathbf W)\right]
=\mathcal D\left[
F\mathbf R^{\mathrm T}
-\mathbf R^{\mathrm T}\mathbf Q\mathbf R^{\mathrm T}
\right].
\tag{7d}
$$

For simultaneous directions $\boldsymbol\Delta$ and $\dot{\mathbf W}$,

$$
\begin{aligned}
\dot F
&=\langle\dot{\mathbf W},\mathbf A(\mathbf R)\rangle
 +\langle\mathbf Q,\dot{\mathbf R}\rangle,\\
\dot{\mathbf Q}
&=\mathbf Q(\dot{\mathbf R},\mathbf W)
 +\mathbf Q(\mathbf R,\dot{\mathbf W}),\\
\dot{\mathbf G}_{\mathbf X}
&=\mathcal D\Bigl\{
\tau\left(F\mathbf R^{\mathrm T}
-\mathbf R^{\mathrm T}\mathbf Q\mathbf R^{\mathrm T}\right)
+\dot F\mathbf R^{\mathrm T}
+F\dot{\mathbf R}^{\mathrm T}\\
&\hspace{2.4em}
-\dot{\mathbf R}^{\mathrm T}\mathbf Q\mathbf R^{\mathrm T}
-\mathbf R^{\mathrm T}\dot{\mathbf Q}\mathbf R^{\mathrm T}
-\mathbf R^{\mathrm T}\mathbf Q\dot{\mathbf R}^{\mathrm T}
\Bigr\}.
\end{aligned}
\tag{7e}
$$

Equations (7b)--(7e) are evaluated by direct four-index contraction. They do
not materialize the second compound rotations or the second cofactor when
only a scalar contraction or overlap gradient is required.

### 3.2 Asymptotic reduction and numerical admission

For $n$ same-spin electrons, let $p=n(n-1)/2$. The former regular-pair path
formed two $p\times p$ compound rotations and transformed a $p\times p$
second-cofactor object. Its dense pair-local leading costs were

$$
T_{\mathrm{compound}}=O(p^3)=O(n^6),
\qquad
M_{\mathrm{compound}}=O(p^2)=O(n^4).
\tag{7f}
$$

The direct exterior contraction requires

$$
T_{\mathrm{exterior}}=O(p^2)=O(n^4),
\qquad
M_{\mathrm{exterior}}=O(n^2).
\tag{7g}
$$

The regular formula contains at most four inverse factors. It is admitted
when the cached inverse exists, the numerical nullity is zero, and

$$
\kappa_{\infty}(\mathbf X)
=\lVert\mathbf X\rVert_{\infty}
 \lVert\mathbf X^{-1}\rVert_{\infty}
\leq \epsilon_{\mathrm{mach}}^{-1/8},
\tag{7h}
$$

which bounds the leading inverse amplification
$\kappa_{\infty}^4\epsilon_{\mathrm{mach}}$ by
$\sqrt{\epsilon_{\mathrm{mach}}}$. This admission rule depends only on
numerical conditioning and machine precision; it contains no molecule- or
input-specific parameter. Nonregular blocks with at least four same-spin
electrons enter the interpolation construction below. Smaller blocks remain
on the exact polynomial path because compound-space work is then negligible.

For the remaining blocks, define

$$
K_*=\epsilon_{\mathrm{mach}}^{-1/8},
\qquad
\tau=\frac{n\sigma_{\max}}{K_*},
\qquad
I_{\mathrm d}=\{i:\sigma_i<\tau\},
\qquad
q=|I_{\mathrm d}|.
\tag{7i}
$$

The indices in $I_{\mathrm d}$ span only the dangerous singular subspace.
For every sign vector $\mathbf s\in\{-1,+1\}^q$, construct the regular node

$$
\mathbf X_{\mathbf s}
=\mathbf U\operatorname{diag}(z_1^{(\mathbf s)},\ldots,
z_n^{(\mathbf s)})\mathbf V^{\mathrm T},
\qquad
z_i^{(\mathbf s)}=
\begin{cases}
s_i\tau,&i\in I_{\mathrm d},\\
\sigma_i,&i\notin I_{\mathrm d}.
\end{cases}
\tag{7j}
$$

Every deleted-minor cofactor and every directional derivative used here is
multi-affine in the dangerous singular coordinates. Tensor-product two-point
interpolation is consequently exact:

$$
\mathcal C(\mathbf X)
=\sum_{\mathbf s\in\{-1,+1\}^q}
\lambda_{\mathbf s}\mathcal C(\mathbf X_{\mathbf s}),
\qquad
\lambda_{\mathbf s}
=\prod_{i\in I_{\mathrm d}}
\frac{1+s_i\sigma_i/\tau}{2}.
\tag{7k}
$$

Because $0\leq\sigma_i/\tau<1$ on the dangerous subspace, the weights are
nonnegative and sum to one. Thus the interpolation has Lebesgue constant one:
it is a convex combination rather than an extrapolation. Each node satisfies

$$
\kappa_2(\mathbf X_{\mathbf s})
\leq\frac{\sigma_{\max}}{\tau}
=\frac{K_*}{n},
\qquad
\kappa_\infty(\mathbf X_{\mathbf s})\leq K_*.
\tag{7l}
$$

The node inverse and determinant-weight product are evaluated without
reconstructing or refactorizing the node:

$$
\begin{aligned}
\mathbf X_{\mathbf s}^{-1}
&=\sum_{i\notin I_{\mathrm d}}\sigma_i^{-1}
  \mathbf v_i\mathbf u_i^{\mathrm T}
 +\sum_{i\in I_{\mathrm d}}\frac{s_i}{\tau}
  \mathbf v_i\mathbf u_i^{\mathrm T},\\
\lambda_{\mathbf s}\det(\mathbf X_{\mathbf s})
&=\eta\left(\prod_{i\notin I_{\mathrm d}}\sigma_i\right)
  \prod_{i\in I_{\mathrm d}}\frac{\sigma_i+s_i\tau}{2}.
\end{aligned}
\tag{7m}
$$

Interpolation is admitted when

$$
2^q\leq n^2,
\tag{7n}
$$

which compares its $O(2^q n^4)$ direct-node work with the former
$O(n^6)$ compound-space action. This is an asymptotic, molecule-independent
rule. The implementation deliberately retains the $O(n^3)$ polynomial
formulas for the value, first cofactor, and mixed first-cofactor derivative;
interpolating those quantities would add a needless factor $2^q$ to the
opposite-spin path. Exact interpolation is used for the second-cofactor
series, where it removes the fourth-order persistent tensor.

For $P_{\mathrm r}$ regular pairs, $P_{\mathrm i}$ admitted interpolation
pairs, and $P_{\mathrm p}$ residual polynomial pairs, the leading pair-local
costs are therefore

$$
\begin{aligned}
T
&=O\left(P_{\mathrm r}n^4
 +P_{\mathrm i}2^q n^4
 +P_{\mathrm p}n^6\right),\\
M
&=O\left(P_{\mathrm r}n^2
 +P_{\mathrm i}n^3
 +P_{\mathrm p}n^4\right).
\end{aligned}
\tag{7o}
$$

The $O(n^3)$ interpolated-pair storage is the complementary-product data
still shared by the low-order and opposite-spin kernels; the second-order
exterior payload itself is only $O(n^2)$. The residual polynomial branch is
retained only when the exact node count fails Eq. (7n).

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
without forming an orbital Hessian. The rank-aware exact actions cover both
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

### Complete-CAS OEO quotient and independent CASSCF comparison

For a complete-CAS full-AO OEO wavefunction, nonsingular active--active
transformations are gauge transformations. The accepted-point chart therefore
removes the entire occupied span from every active-orbital tangent. For the
F2 CASSCF(2,2) regression with 30 Cartesian cc-pVDZ functions, eight inactive
orbitals, and two active orbitals, the orbital dimension changes from 218 to

$$
8(30-8)+2(30-8-2)=216.
$$

The same chart builder is used by TNHVP, NEO, and L-BFGS. A truncated VB
structure subspace retains its active--active variables because it is not, in
general, invariant under active-orbital transformations.

An independent PySCF 2.11.0 calculation used Cartesian cc-pVDZ functions,
no point-group constraint, and CASSCF(2,2). Different initial active-orbital
pairs converge to distinct CASSCF stationary points:

| Method and stationary point | Total energy / $E_{\mathrm h}$ |
|---|---:|
| PySCF CASSCF, local initial active pair | -198.689800047006 |
| xmvb-cpp TNHVP, 5 iterations | -198.689799876580 |
| PySCF CASSCF, lower stationary point | -198.761111550672 |
| xmvb-cpp NEO, 6 iterations | -198.761111541670 |

Thus TNHVP and NEO agree with the corresponding independently converged
CASSCF solutions to $1.70\times10^{-7}\ E_{\mathrm h}$ and
$9.00\times10^{-9}\ E_{\mathrm h}$, respectively. The lower NEO energy is
not a violation of the CASSCF variational space: NEO crosses out of the local
active-space basin reached by the supplied initial orbitals. A symmetry-locked
PySCF calculation remains on the higher stationary branch and is therefore
not a valid test of the lower solution.

Before finite active-subspace canonicalization, the NEO trajectory reached the
same energy only after 19 iterations. The smallest eigenvalue of the projected
active overlap fell from 1 to approximately $4.5\times10^{-7}$, while its
condition number grew to approximately $2.2\times10^6$. This was gauge drift,
not physical ill-conditioning. Applying eqs 27o--27p after every trial keeps a
stable complete-CAS representative. The final three projected gradient
infinity norms are

$$
5.23\times10^{-3},\qquad
1.78\times10^{-5},\qquad
1.81\times10^{-10},
$$

which exhibits the expected local quadratic contraction. No trust-radius or
Krylov tolerance was changed.

Representative reproduction commands, executed from the repository root:

```sh
ctest --test-dir build --output-on-failure
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build/src/benchmark_exact_ctx_hvp testdata/vbscf/F2_OEO.xmi --nonredundant-adapt true --curvature-audit-directions 6
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 build/src/check_exact_ctx_hvp testdata/vbscf/FeCl2.xmi --step 1e-5 --probe full --nonredundant-adapt true --max-rel-error 1e-7
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 build/src/benchmark_exact_ctx_hvp testdata/vbscf/FeCl2.xmi --nonredundant-adapt true --orbital-value-table-bin /tmp/xmvb-oeo-fix.Cdw3GW/FeCl2.final.bin --curvature-audit-directions 32
```
