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

## 8. Rank-aware cofactor exterior contraction

The regular determinant-pair cofactor representation now contracts the
second exterior power directly. For $n$ same-spin electrons and
$p=n(n-1)/2$, this changes the regular pair-local leading work and persistent
storage from

$$
O(p^3)=O(n^6)
\quad\hbox{and}\quad
O(p^2)=O(n^4)
$$

to

$$
O(p^2)=O(n^4)
\quad\hbox{and}\quad
O(n^2),
$$

respectively. Ill-conditioned pairs retain the inverse-free polynomial form,
so the optimization does not trade numerical stability for coverage. The
complete formulas and the machine-precision-derived admission condition are
given in
[Full-AO OEO derivative validation](full_ao_oeo_derivative_validation.md).

A same-node, separate-process A/B test used the CERRAS input, 32 OpenMP
threads, one BLAS thread, and one `local` HVP action. Both runs returned the
same response infinity norm, $0.277727397311$.

| Quantity | Polynomial-only baseline | Rank-aware representation | Change |
|---|---:|---:|---:|
| Local HVP action / s | 3.125673 | 2.822618 | $-9.70\%$ |
| Local active-gradient stage / s | 1.848025 | 1.672079 | $-9.52\%$ |
| Whole process wall time / s | 26.97 | 25.82 | $-4.26\%$ |
| Peak RSS / KiB | 5,835,356 | 5,644,808 | $-3.27\%$ |

All 213,444 ordered CERRAS unique-string pairs were numerically full rank,
but only 29,423 pairs, or $13.8\%$, satisfied the conservative regular-form
condition. These pairs account for the measured reduction; the remaining
184,021 pairs preserve the stable polynomial representation. The accepted
cofactor dynamic storage is 1,252,096,600 bytes after the rewrite. The modest
whole-process RSS reduction is expected because the AO-pair graph and other
accepted-point objects remain larger than the eliminated compound matrices.

The initial prototype used `BDCSVD` for the small ill-conditioned overlap
blocks. Its workspace increased peak RSS and erased the gain. That prototype
was removed; the retained polynomial branch reuses an accepted SVD when one
exists and otherwise uses the original small-matrix Jacobi SVD.

The corresponding 32-thread LOFLEA measurement gives

| Quantity | Polynomial-only baseline | Rank-aware representation | Change |
|---|---:|---:|---:|
| Local HVP action / s | 10.617550 | 9.920676 | $-6.56\%$ |
| Local active-gradient stage / s | 5.978685 | 5.790816 | $-3.14\%$ |
| Whole process wall time / s | 30.98 | 29.65 | $-4.29\%$ |
| Peak RSS / KiB | 16,589,768 | 15,890,972 | $-4.21\%$ |

The LOFLEA response infinity norm is unchanged at $0.282004273478$.
Of 853,776 ordered pairs, 107,180 use the regular exterior form, 746,582
full-rank pairs use the polynomial form, and 14 nullity-one pairs use the same
polynomial form. The accepted cofactor dynamic storage is 5,072,888,032 bytes;
the separate-process peak RSS decreases by approximately 682 MiB.

### 8.1 Exact interpolation over the dangerous singular subspace

The next implementation stage replaces the second-cofactor polynomial object
for nonregular pairs by exact tensor-product interpolation over only the
dangerous singular coordinates. The regular-only implementation at commit
`952303f` is the A/B baseline. Both executables were compiled in separate
worktrees and run as separate processes on the same Slurm node with 32 OpenMP
threads and one BLAS thread. The local-action values below are medians of two
runs in interleaved baseline--candidate--candidate--baseline order.

| System | Quantity | Regular-only baseline | Exact interpolation | Change |
|---|---|---:|---:|---:|
| CERRAS | Local HVP action / s | 2.821783 | 1.397666 | $-50.47\%$ |
| CERRAS | Local active-gradient stage / s | 1.723514 | 1.007430 | $-41.55\%$ |
| CERRAS | Whole process / s | 26.355 | 23.900 | $-9.32\%$ |
| CERRAS | Peak RSS / KiB | 5,644,308 | 4,714,872 | $-16.47\%$ |
| LOFLEA | Local HVP action / s | 9.675633 | 4.973933 | $-48.59\%$ |
| LOFLEA | Local active-gradient stage / s | 5.586146 | 3.679822 | $-34.13\%$ |
| LOFLEA | Whole process / s | 28.895 | 20.920 | $-27.60\%$ |
| LOFLEA | Peak RSS / KiB | 15,890,126 | 12,074,242 | $-24.01\%$ |

The local HVP speedups are $2.019\times$ for CERRAS and $1.945\times$ for
LOFLEA. Cofactor dynamic storage decreases from 1,252,096,600 to 289,298,728
bytes for CERRAS ($-76.89\%$), and from 5,072,888,032 to 1,166,697,760 bytes
for LOFLEA ($-77.00\%$). The corresponding peak-RSS reductions are about
0.89 GiB and 3.64 GiB. The response infinity norms remain identical at the
printed precision: $0.277727397311$ and $0.282004273478$, respectively.

The pair classification confirms that the new representation covers every
nonregular pair in these systems:

| System | Regular | Interpolated | Residual polynomial | Dangerous-mode distribution |
|---|---:|---:|---:|---|
| CERRAS | 29,423 | 184,021 | 0 | $q=1$: 79,393; $q=2$: 88,340; $q=3$: 15,856; $q=4$: 432 |
| LOFLEA | 107,180 | 746,596 | 0 | $q=1$: 273,430; $q=2$: 375,401; $q=3$: 93,009; $q=4$: 4,712; $q=5$: 44 |

A second A/B measurement evaluated the complete exact HVP, including the
structure response rather than only its local component:

| System | Quantity | Regular-only baseline | Exact interpolation | Change |
|---|---|---:|---:|---:|
| CERRAS | Complete exact HVP / s | 6.762652 | 4.788595 | $-29.19\%$ |
| CERRAS | Whole process / s | 29.88 | 27.40 | $-8.30\%$ |
| CERRAS | Peak RSS / KiB | 5,643,780 | 4,722,404 | $-16.33\%$ |
| LOFLEA | Complete exact HVP / s | 45.574557 | 35.595521 | $-21.90\%$ |
| LOFLEA | Whole process / s | 64.98 | 51.59 | $-20.61\%$ |
| LOFLEA | Peak RSS / KiB | 17,541,768 | 13,670,296 | $-22.07\%$ |

The complete-HVP response infinity norms are unchanged at
$17.2965082974$ for CERRAS and $16.1201446991$ for LOFLEA. Thus the measured
gain is not produced by weakening the response model or omitting a derivative
channel. The smaller complete-HVP speedups, $1.412\times$ and $1.280\times$,
are consistent with Amdahl's law because structure response and other HVP
stages are unchanged.

### 8.2 Symmetry-preserving exact AO integral storage

The exact LIBCINT path previously expanded the canonical permutationally
unique AO integrals into a symmetric directed AO-pair CSR operator.  For
$n_{\mathrm{bf}}$ AO basis functions, the packed AO-pair dimension is

$$
N_{\mathrm{pair}}
=
\frac{n_{\mathrm{bf}}(n_{\mathrm{bf}}+1)}{2},
$$

and the maximum number of unique pair-pair integrals is

$$
N_{\mathrm{ERI}}^{\mathrm{unique}}
=
\frac{N_{\mathrm{pair}}(N_{\mathrm{pair}}+1)}{2}.
$$

This count already includes the eightfold AO-integral permutation symmetry.
The symmetric CSR expansion nevertheless introduces

$$
N_{\mathrm{edge}}
=
2N_{\mathrm{ERI}}^{\mathrm{stored}}-N_{\mathrm{diag}}
$$

directed edges.  Consequently, a valid symmetry-compressed integral list can
still overflow a signed 32-bit CSR edge offset.  The revised representation
retains the canonical tuple $(p,q,K_{pq})$ exactly once whenever
$N_{\mathrm{edge}}>\mathrm{INT\_MAX}$ and applies both symmetric
contributions directly during a block action:

$$
Y_p \mathrel{+}=K_{pq}X_q,
\qquad
Y_q \mathrel{+}=K_{pq}X_p
\quad (p\ne q).
$$

With 32-bit pair indices and double-precision values, the leading persistent
storage is therefore reduced from

$$
M_{\mathrm{CSR}}
\simeq
12N_{\mathrm{edge}}+8N_{\mathrm{ERI}}^{\mathrm{stored}}
$$

bytes to

$$
M_{\mathrm{unique}}
=
16N_{\mathrm{ERI}}^{\mathrm{stored}}
$$

bytes.  Large integral materializations use a deterministic two-pass
count-and-fill algorithm.  The first pass counts retained integrals per outer
shell; the second writes directly into disjoint, exactly sized ranges.  This
removes the former simultaneous residency of all per-shell buffers and their
complete aggregate while preserving the canonical integral order and the
original screening threshold.

For the 322-AO state-averaged test, $N_{\mathrm{pair}}=52{,}003$ and
$N_{\mathrm{ERI}}^{\mathrm{stored}}=1{,}108{,}932{,}855$.  The old CSR path
failed while constructing approximately $2.218\times10^9$ directed edges.
The canonical-stream path completed the initial objective and a subsequent
orbital update.  On the same 32-core Slurm node, the measured process peak RSS
decreased from 36,054,456 KiB to 19,224,224 KiB, a reduction of $46.68\%$.

### 8.3 Factor-native RI Hessian action

The TNHVP operator also supports the resolution-of-the-identity (RI)
two-electron representation without reconstructing AO four-index integrals.
Let $\mathbf L$ contain the metric-whitened three-index factors in packed AO
pair space and let $\mathbf Q(\mathbf C)$ denote the quadratic packed-pair map
from AO coefficients $\mathbf C$ to active-orbital pairs.  The accepted active
pair factors and active two-electron kernel are

$$
\mathbf B(\mathbf C)=\mathbf L\mathbf Q(\mathbf C),
\qquad
\mathbf G(\mathbf C)=\mathbf B(\mathbf C)^{\mathrm T}
                     \mathbf B(\mathbf C).
$$

For an orbital direction $\mathbf D$, their directional derivatives are

$$
\dot{\mathbf B}
=
\mathbf L\mathbf Q'(\mathbf C)[\mathbf D],
$$

$$
\dot{\mathbf G}
=
\dot{\mathbf B}^{\mathrm T}\mathbf B
+
\mathbf B^{\mathrm T}\dot{\mathbf B}.
$$

If $\boldsymbol\lambda$ is the packed active-integral adjoint, define the
symmetric pair matrix $\mathbf W(\boldsymbol\lambda)$ using the canonical
pair-of-pairs multiplicities,

$$
W_{PQ}
=
\begin{cases}
2\lambda_{PP}, & P=Q,\\
\lambda_{\{P,Q\}}, & P\ne Q.
\end{cases}
$$

The fixed-adjoint RI two-electron Hessian action is then evaluated as

$$
\dot{\mathbf g}_{\mathbf C}^{(2e)}
=
\mathbf Q'(\mathbf C)^{*}
\!\left[
\mathbf L^{\mathrm T}\dot{\mathbf B}\mathbf W
\right]
+
\mathbf Q'(\mathbf D)^{*}
\!\left[
\mathbf L^{\mathrm T}\mathbf B\mathbf W
\right].
$$

The first term is the response of the RI factor adjoint; the second is the
derivative of the quadratic orbital pair map.  The structure response produces
a directional adjoint $\dot{\boldsymbol\lambda}$ and contributes only

$$
\mathbf g_{\mathbf C,\mathrm{outer}}^{(2e)}
=
\mathbf Q'(\mathbf C)^{*}
\!\left[
\mathbf L^{\mathrm T}\mathbf B
\mathbf W(\dot{\boldsymbol\lambda})
\right],
$$

because the fixed-adjoint terms are already included in the direct core HVP.
The same packed-$\dot{\mathbf G}$ interface is consumed by the local
same-spin, opposite-spin, direct-CI, and generalized-eigenvector response
paths, so those physical response equations are independent of the AO
integral representation.

For the inactive-density contribution, each whitened factor is interpreted as
a symmetric AO matrix $\mathbf L_A$.  The RI Coulomb--exchange map is

$$
\mathcal G_{\mathrm{RI}}[\mathbf X]
=
2\sum_A
\langle\mathbf L_A,\mathbf X\rangle_F\mathbf L_A
-
\sum_A\mathbf L_A\mathbf X\mathbf L_A.
$$

This linear map is self-adjoint.  A fused auxiliary-factor sweep computes both
the forward $\dot{\mathbf F}$ and its transpose pullback without forming the
AO-pair Gram matrix when the AO inputs are dense.  Orbital-response inputs,
however, are normally numerically low rank.  For a symmetric input, the
retained signed spectral representation is

$$
\mathbf X
=
\mathbf U_{+}\mathbf U_{+}^{\mathrm T}
-
\mathbf U_{-}\mathbf U_{-}^{\mathrm T}.
$$

Define the auxiliary-transformed factors

$$
Z_{A k,\mu}
=
\sum_{\nu}L_{A,\mu\nu}U_{\nu k}.
$$

The exchange action is then a pair of level-3 symmetric rank contractions,

$$
-\sum_A\mathbf L_A\mathbf X\mathbf L_A
=
-\mathbf Z_{+}^{\mathrm T}\mathbf Z_{+}
+\mathbf Z_{-}^{\mathrm T}\mathbf Z_{-},
$$

and the Coulomb projection is recovered from the same transformed tensor,

$$
q_A
=
\sum_{\mu k}s_k U_{\mu k}Z_{Ak,\mu},
\qquad
s_k\in\{+1,-1\}.
$$

This replaces many auxiliary-local matrix products by AO-slice GEMMs followed
by large SYRK contractions.  Its leading work and workspace are

$$
O\!\left(N_{\mathrm{aux}}N_{\mathrm{bf}}^2r\right),
\qquad
O\!\left(N_{\mathrm{aux}}N_{\mathrm{bf}}r\right),
$$

where $r$ is the retained signed rank.  AO slices and final rank contractions
are partitioned across OpenMP workers; nested calls reduce to one worker to
avoid oversubscription.  The production builder consumes the complete
symmetric RI result directly; only the reverse-mode density gradient is
encoded in the canonical lower-triangular storage with one-half diagonal
weights.

Rank alone does not determine wall time: the dense fused kernel has fewer
launches and the spectral kernel has less arithmetic.  At each accepted
orbital point, the first HVP therefore evaluates both kernels on the same
forward and adjoint inputs, verifies their agreement, and retains the faster
kernel for the remaining Krylov actions at that point.  This measured
admission rule has no molecule, HAO/OEO, or input-file-specific threshold.

The leading factor contraction is

$$
O\!\left(
N_{\mathrm{aux}}N_{\mathrm{AO-pair}}N_{\mathrm{active-pair}}
\right),
$$

with no AO four-index tensor.  Random-matrix tests independently verify
$\dot{\mathbf B}$, $\dot{\mathbf G}$, the packed adjoint identity, and the
fixed-adjoint HVP by centered finite differences.  End-to-end F$_2$ HAO, OEO,
two-state dense, and two-state Davidson checks give maximum relative gradient
finite-difference errors between $2.13\times10^{-8}$ and
$2.80\times10^{-8}$; the tests explicitly remove the exact AO-pair graph so
that an exact-integral fallback cannot pass unnoticed.

For sufficiently large active spaces, explicitly building the directional
packed pair map is no longer the preferred contraction.  At a fixed auxiliary
index,

$$
\dot{\mathbf B}_A
=
\mathbf C^{\mathrm T}\mathbf L_A\mathbf D
+
\mathbf D^{\mathrm T}\mathbf L_A\mathbf C.
$$

The direct implementation first forms $\mathbf L_A\mathbf C$ and
$\mathbf L_A\mathbf D$ by contiguous auxiliary-major AO-slice GEMMs and then
performs the second AO-to-active transformation.  Its asymptotic work is

$$
O\!\left[
N_{\mathrm{aux}}N_{\mathrm{bf}}N_{\mathrm{act}}
\left(N_{\mathrm{bf}}+N_{\mathrm{act}}\right)
\right],
$$

instead of

$$
O\!\left(
N_{\mathrm{aux}}N_{\mathrm{AO-pair}}N_{\mathrm{active-pair}}
\right).
$$

The implementation compares the two leading floating-point counts directly
and selects the lower-count contraction. The admission rule therefore depends
only on the AO and active dimensions; it contains no molecule, orbital-type,
or fitted crossover parameter. A synthetic test in the admitted regime
compares every transformed factor against the packed-pair contraction.

For a block of $b$ HVP directions, define

$$
\Delta\mathbf Q
=
\left[
Q'(C)[D_1]\;\cdots\;Q'(C)[D_b]
\right].
$$

The packed path now evaluates

$$
\Delta\mathbf B=L\Delta\mathbf Q
$$

as one wide matrix product. It then contracts all integral responses through

$$
\left[
B^{\mathrm T}\delta B_1\;\cdots\;B^{\mathrm T}\delta B_b
\right]
=B^{\mathrm T}\Delta B,
$$

followed by the exact symmetric completion
$\delta G_d=B^{\mathrm T}\delta B_d+\delta B_d^{\mathrm T}B$. This preserves
the scalar leading-order work but removes repeated RI-factor reads and BLAS
launches.

In the direct-transform path, the accepted $L_A C$ transform is shared across
the complete block. The leading work changes from

$$
4bN_{\mathrm{aux}}N_{\mathrm{bf}}N_{\mathrm{act}}
\left(N_{\mathrm{bf}}+N_{\mathrm{act}}\right)
$$

for $b$ independent scalar actions to

$$
2(b+1)N_{\mathrm{aux}}N_{\mathrm{bf}}^2N_{\mathrm{act}}
+4bN_{\mathrm{aux}}N_{\mathrm{bf}}N_{\mathrm{act}}^2.
$$

The block schedule also admits all directional structure responses to one
block generalized-eigen action instead of routing RI HVPs through the scalar
driver. For a three-direction F$_2$ block on four CPU threads, the block result
agrees with three independent RI HVPs to relative errors of
$2.29\times10^{-14}$ (single state) and $2.68\times10^{-12}$ (two-state equal
average). Over 20 repeated block calls, the corresponding measured speedups
are $1.03$ and $1.09$, respectively. These compact tests are correctness and
scheduling checks, not large-system scaling claims. A prototype that widened
the AO one-electron RI exchange sweep was removed: it duplicated
$O(t b N_{\mathrm{bf}}^2)$ thread-local storage while sharing only one of the
two exchange multiplications, and did not improve the steady-state compact
benchmark.

The RI path also uses the same factor representation during closed-shell RHF
initialization.  For the spinless occupied-orbital projector
$\mathbf D=\mathbf C_{\mathrm{occ}}\mathbf C_{\mathrm{occ}}^{\mathrm T}$,
the initial Fock operator is

$$
\mathbf F_{\mathrm{RI}}
=
\mathbf H_{\mathrm{core}}
+2\mathbf J_{\mathrm{RI}}[\mathbf D]
-\mathbf K_{\mathrm{RI}}[\mathbf D].
$$

Thus exact and RI optimizations start from equivalent RHF-quality orbitals;
RI does not silently replace the two-electron Fock contribution by a core-only
guess.  With cc-pVDZ/cc-pVDZ-JKFIT, the F$_2$ RI calculation starts at
$-198.4881856608\ E_h$ and converges in six TNHVP iterations, matching the
exact path's iteration count.  Its final energy is $-198.7509765327\ E_h$,
while the remaining difference from the exact result is the controlled RI
integral approximation.

### 8.4 Direction-local NEO structure response

The NEO response workspace now solves structure equations only for the current
candidate step and an optional hard-case curvature direction.  It no longer
closes

$$
\mathbf C\mathbf Z=-\mathbf B\mathbf Q_o
$$

for every retained orbital basis vector.  Continuous basis/image matrices and
incrementally updated projected blocks remove the repeated reconstruction of
$\mathbf Q_o$, $\mathbf A\mathbf Q_o$, $\mathbf B\mathbf Q_o$, and their
Gram matrices.  Independent unresolved response directions are
orthogonalized and submitted to one memory-bounded structure H/S block action.

Hanhai25 Slurm array job `250378` used 32 CPU cores, exact Libcint integrals,
Davidson structure diagonalization, the one-electron orbital preconditioner,
and common convergence thresholds of $10^{-3}$ for the projected gradient and
$10^{-7}\ E_h$ for the energy change.  No outer-iteration truncation was used.

| case | outer steps | orbital actions | structure actions | total coupled actions | wall time / s | peak RSS / MiB |
|---|---:|---:|---:|---:|---:|---:|
| MnF2 | 11 | 199 | 26 | 225 | 8.73 | 539.7 |
| FeCl2 | 4 | 72 | 0 | 72 | 5.96 | 425.5 |
| 241 | 6 | 9 | 0 | 9 | 1.26 | 933.5 |
| 240 | 5 | 10 | 5 | 15 | 9.32 | 1811.6 |

The zero entries are residual decisions, not molecule labels or a disabled
response implementation.  In FeCl2 and 241, the fixed-structure candidate
already satisfied the complete coupled KKT forcing condition.  MnF2 admitted
14 and 12 structure directions in outer iterations 7 and 8, respectively;
240 admitted one and four directions in iterations 3 and 4.  Relative to the
immediately preceding response-closure implementation, MnF2 structure actions
decreased from 325 to 26 and FeCl2 structure actions from 194 to zero.  The
corresponding wall times changed from 8.64 to 8.73 s and from 7.95 to 5.96 s.
Thus direction-local allocation removes most response work, although the
MnF2 tail is now dominated by 199 orbital actions and remains a separate
preconditioning problem.

## 9. Reproducibility

The configured test suite contains 61 tests, including independent polynomial
cofactor derivatives, complete exact-integral HAO/OEO HVP finite differences,
factor-native RI response tests, and RI HAO/OEO/state-averaged HVP finite
differences. All 44 pass after the performance changes. The block-basis tests
validate assembly at the utility and molecular-integration levels; the
orthonormal-basis test also verifies single-call block admission. The benchmark
logs used in this note are kept
under `/tmp/xmvb-oeo-fix.Cdw3GW/` and
`/tmp/xmvb-hvp-cofactor-direct/`. The interpolation A/B logs are under
`/tmp/xmvb-interpolation-ab/` on the validation machine. These logs are not
treated as repository test assets.

The explicit references can be reproduced with

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  ./build/src/benchmark_exact_ctx_hvp testdata/vbscf/F2.xmi \
  --repeats 1 --dense-reference-block-width 4

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  ./build/src/benchmark_exact_ctx_hvp testdata/vbscf/F2_OEO.xmi \
  --repeats 1 \
  --dense-reference-block-width 8
```

No performance-selection environment variable remains in the same-spin,
opposite-spin, packed-gradient, structure-tile, or selected-state sparse/dense
decision paths. Sparse/dense selected-state contraction is chosen from the
estimated operation counts of the actual support data.  Hanhai25 Slurm job
`250388` rebuilt all developer targets and passed all 61 tests, including the
241/7975 fixed-point response models, the F2 PySCF NEO reference, HAO/OEO,
exact/RI, Davidson, and finite-difference HVP checks.
