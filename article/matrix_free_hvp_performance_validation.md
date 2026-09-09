# Performance Validation of the Exact Matrix-Free VBSCF Orbital Hessian Action

## 1. Scope

This note records performance changes made after correcting the sparse HAO
quotient coordinates and the full-AO OEO derivative chain. The optimized
operator remains an exact Hessian-vector product (HVP): no orbital Hessian is
formed, no response term is dropped, and no molecule-dependent solver
parameter is introduced.

For a reduced nonredundant direction $\mathbf v$, the implemented operation is

$$
\mathbf y = \mathbf H_{\mathrm{red}}\mathbf v
= \mathbf U_p^{\mathrm T}
D\mathbf g_{\mathbf x}[\mathbf U_p\mathbf v],
\tag{1}
$$

where $\mathbf U_p$ is the orthonormal basis of the physical sparse quotient
space and $\mathbf g_{\mathbf x}$ is the packed-coordinate orbital gradient.
All optimizations below change the evaluation schedule of Eq. (1), not its
mathematical definition.

## 2. Accepted-point factor reuse

For an occupied-orbital overlap block $\mathbf X$, define the first- and
second-order polynomial cofactor tensors

$$
\mathbf C^{(1)}(\mathbf X)
= \frac{\partial\det\mathbf X}{\partial\mathbf X},
\qquad
\mathbf C^{(2)}(\mathbf X)
= \frac{\partial^2\det\mathbf X}{\partial\mathbf X\,\partial\mathbf X}.
\tag{2}
$$

The singular-vector factors, compound rotations, cofactor values, and
accepted one- and two-electron occupied blocks depend only on the accepted
orbital point. They are therefore constructed once per unique same-spin pair.
The directional actions

$$
D\mathbf C^{(1)}(\mathbf X)[\dot{\mathbf X}],
\qquad
D\mathbf C^{(2)}(\mathbf X)[\dot{\mathbf X}]
\tag{3}
$$

reuse those factors without recomputing an SVD. Complement products excluding
two, three, or four singular values are also cached once. This remains
division-free and is valid at arbitrary rank deficiency.

## 3. One directional pair response per HVP

Previously, the projected structure action, the same-spin local adjoint, and
the opposite-spin local adjoint independently reconstructed overlapping parts
of the same directional pair response. For every ordered unique-spin pair,
the HVP now builds one shared payload

$$
\mathcal P_{LR}[\mathbf v]
=\left\{
\dot d_{LR},
\dot H_{LR}^{\sigma},
\mathbf C_{LR}^{(1)},
\dot{\mathbf C}_{LR}^{(1)},
\dot{\mathbf G}_{LR}^{S}
\right\},
\tag{4}
$$

where $\dot d_{LR}$ is the determinant-overlap response,
$\dot H_{LR}^{\sigma}$ is the same-spin Hamiltonian response, and
$\dot{\mathbf G}_{LR}^{S}$ is its overlap-gradient response. Equation (4) is
shared across the structure and adjoint modules for the lifetime of one HVP.
The reverse ordered pair is obtained by the exact transpose relation rather
than by a second factor action.

The projected structure builder also memoizes direction-dependent packed-pair
projections across OpenMP workers. Hence a determinant pair first encountered
by one structure block is not recomputed when another block uses it.

## 4. Fixed-endpoint HVP timings

The fixed corrected FeCl2 OEO endpoint was used to separate kernel performance
from changes in the nonlinear trajectory. Measurements used four OpenMP
threads and one OpenBLAS thread on one AMD EPYC 7K62 node. The final value is a
five-application mean after one warmup; earlier stages are single-application
diagnostics and should be interpreted as indicative rather than statistically
controlled timings.

| Exact HVP implementation | Time per HVP / s | Change from corrected reference |
|---|---:|---:|
| Correct polynomial response | 0.39155 | reference |
| Accepted cofactor/SVD reuse | 0.30169 | -22.9% |
| Same-spin directional reuse | 0.27102 | -30.8% |
| Shared pair response, final | 0.25012 | -36.1% |

At the final endpoint, the optimized HVP retained the following numerical
audits:

$$
\epsilon_{\mathrm{skew}} = 6.38\times10^{-15},
\qquad
\epsilon_{\mathrm{linear}} = 3.00\times10^{-13},
\tag{5}
$$

and the central gradient-difference discrepancy at step $10^{-4}$ was

$$
\epsilon_{\mathrm{FD}} = 1.00\times10^{-8}.
\tag{6}
$$

Thus the speedup is not obtained by relaxing the HVP definition.

## 5. Production convergence cases

The following runs use the exact-context HVP, the existing maximum inner
dimension of 32, four OpenMP threads, and one OpenBLAS thread.

| Input | Orbital type | Outer iterations | HVP directions | Final total energy / hartree | Final gradient 2-norm | End-to-end time / s |
|---|---|---:|---:|---:|---:|---:|
| F2 | sparse HAO | 5 | 11 | -198.751155830527 | $5.34\times10^{-7}$ | 0.060 |
| 241 | sparse HAO | 10 | 39 | -230.720590392896 | $2.56\times10^{-7}$ | 18.486 |
| MnF2 | sparse HAO | 15 | 183 | -1348.893353224116 | $4.15\times10^{-5}$ | 36.515 |
| FeCl2 | full-AO OEO | 6 | 192 | -2181.617645350186 | $2.63\times10^{-4}$ | 53.540 |

For FeCl2, the first fully corrected production build required 76.98 s with
the same 6 outer iterations and 192 HVP directions. The final implementation
therefore reduces end-to-end time by approximately 30%, while preserving the
corrected energy and orbital-gradient trajectory to the reported precision.

MnF2 requires one additional outer iteration relative to the earlier
14-iteration run, but reaches a smaller final gradient norm and remains
slightly faster in wall time. Both trajectories satisfy the unchanged outer
stopping rule. This difference is retained rather than hidden by a case-specific
iteration budget.

## 6. Inner-solve sensitivity is not an algorithm

FeCl2 was also run with fixed inner dimensions solely as a diagnostic:

| Maximum inner dimension | Outer iterations | Total HVPs | Final gradient 2-norm | SCF iteration time / s |
|---:|---:|---:|---:|---:|
| 8 | 20 | 160 | $6.69\times10^{-4}$ | 49.40 |
| 16 | 10 | 160 | $5.71\times10^{-4}$ | 45.70 |
| 24 | 7 | 168 | $2.74\times10^{-4}$ | 46.34 |
| 32 | 6 | 192 | $2.63\times10^{-4}$ | 52.02 |

Reducing the fixed budget lowers the cost per outer iteration but degrades the
outer convergence rate and, for the smaller spaces, the final residual reached
under the energy stopping criterion. Therefore 16 or 24 is not adopted as a
new global default. A defensible next method must terminate or enlarge the
subspace using the spectrum, the trust-region KKT residual, and observed
actual-to-predicted reduction, rather than the identity of a molecule or a
fixed empirical budget.

For a step $\mathbf s_k$, the relevant tests are

$$
\left\|
\mathbf g_k + \mathbf H_k\mathbf s_k + \lambda_k\mathbf s_k
\right\|
\leq \eta_k\left\|\mathbf g_k\right\|,
\tag{7}
$$

and

$$
\rho_k
=
\frac{
E(\mathbf x_k)-E(R_{\mathbf x_k}(\mathbf s_k))
}{
-\mathbf g_k^{\mathrm T}\mathbf s_k
-\tfrac12\mathbf s_k^{\mathrm T}\mathbf H_k\mathbf s_k
}.
\tag{8}
$$

These quantities support an adaptive recycled block-Newton method without
forming the Hessian. The present width-two block kernel does not yet provide a
speedup on FeCl2, because the structure response remains direction-local;
production block Krylov should therefore follow, not precede, batched
structure-response evaluation.

## 7. Explicit reduced-Hessian reference and block-Newton decision

For a small reduced space, the matrix-free operator now provides an explicit
gold-standard construction used only by the diagnostic executable. With
$\{\mathbf e_i\}_{i=1}^{n_r}$ denoting the canonical reduced-coordinate
basis, the reference matrix is assembled as

$$
\mathbf H_{\mathrm{ref}}
=
\begin{bmatrix}
\mathcal H[\mathbf e_1] &
\mathcal H[\mathbf e_2] &
\cdots &
\mathcal H[\mathbf e_{n_r}]
\end{bmatrix}.
\tag{9}
$$

Coordinate vectors are submitted in blocks, but every column is produced by
the same production HVP used by the optimizer. Thus this reference isolates
the inner solver from derivative implementation. It is never called by the
production optimizer and incurs the expected storage

$$
M_{\mathrm{dense}}=8n_r^2\ \mathrm{bytes}.
\tag{10}
$$

The diagnostic fails if either the relative skew norm or an independent action
reconstruction exceeds the square root of machine precision. At the initial
F2 points, the measured results are

| Orbital chart | $n_r$ | Relative skew | Independent action error | Spectral interval / hartree |
|---|---:|---:|---:|---:|
| strict sparse HAO | 42 | $4.53\times10^{-16}$ | $4.44\times10^{-16}$ | $[0.6144,79.5178]$ |
| full-AO OEO | 218 | $3.18\times10^{-16}$ | $5.59\times10^{-16}$ | $[-0.08016,121.6543]$ |

The OEO reference is especially important: it demonstrates that the
normalization pullback and the full-AO quotient produce one symmetric reduced
operator even when the accepted Hessian is indefinite.

A recycled residual-expanded block trust-region solver was then tested against
the retained positive-conjugate inner solve. The candidate enforced the full
KKT residual after every projected solve and re-evaluated transported positive
Ritz directions at the new accepted point. It increased F2 from 11 to 30 HVPs
without reducing its five outer iterations, and increased MnF2 from 183 to 246
HVPs without reducing its 15 outer iterations. A second variant activated
recycling only when the observed residual contraction predicted exhaustion of
the remaining inner budget; this preserved F2 but changed MnF2 to 28 outer
iterations and 213 HVPs. Neither variant is retained in production.

This negative result establishes an algorithmic boundary rather than a tuning
failure. Transported Ritz pairs are useful as positive preconditioning secants,
but their old-point model decrease does not certify a useful new-point exact
trust-region subspace under a nonlinear sparse-orbital retraction. Moreover,
the current width-two FeCl2 block action has speedup $0.994<1$ over two scalar
actions because structure response is still direction-local. Production
block-Newton is therefore gated on a genuinely batched structure-response
kernel and on improvement relative to eq 9, not merely on the existence of an
`apply_batch` interface.

The block benchmark was subsequently generalized to arbitrary width and now
compares every returned column with an independent scalar HVP. At the corrected
FeCl2 endpoint the width-four relative discrepancy is zero at stored precision.
A direction-parallel structure prototype produced a nominal width-four speedup
of 1.010, but its structure stage required 0.456 s versus 0.440 s for four
scalar structure actions. After removing that prototype, a repeated width-four
measurement gives speedup 1.000. Hence the nominal gain was scheduling noise,
and the direction-parallel implementation is not retained. A useful batch
kernel must instead traverse each determinant pair once and contract all
directional overlap, one-electron, and two-electron channels while its accepted
cofactor and coefficient data remain resident.

## 8. Reproducibility

The configured test suite contains 15 tests, including independent polynomial
cofactor derivatives and complete HAO/OEO HVP finite differences. All 15 pass
after the performance changes. The two additional tests validate block basis
assembly at the utility and molecular-integration levels; the orthonormal-basis
test was also extended to verify single-call block admission. The benchmark
logs used in this note are kept
under `/tmp/xmvb-oeo-fix.Cdw3GW/` on the validation machine and are not treated
as repository test assets.

The explicit references can be reproduced with

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  ./build/src/benchmark_exact_ctx_hvp src/test_molecule/F2.xmi \
  --repeats 1 --dense-reference-block-width 4

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  ./build/src/benchmark_exact_ctx_hvp src/tools/test_inputs/F2_OEO.xmi \
  --repeats 1 --nonredundant-adapt true \
  --dense-reference-block-width 8
```

No performance-selection environment variable remains in the same-spin,
opposite-spin, packed-gradient, structure-tile, or selected-state sparse/dense
decision paths. Sparse/dense selected-state contraction is chosen from the
estimated operation counts of the actual support data.
