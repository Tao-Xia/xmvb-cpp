# Streamed Exact Two-Electron Hessian-Vector Products

## 1. Scope

This document specifies the memory-bounded exact two-electron response used by
the TNHVP orbital optimizer.  The objective is to preserve the exact AO
integrals and the exact Hessian-vector product while removing accepted-point
and per-direction matrices whose storage grows as

$$
O(N_{\mathrm{AO\mbox{-}pair}}N_{\mathrm{active\mbox{-}pair}}).
$$

No density fitting, Cholesky decomposition, integral screening, or truncated
response is introduced.  Sparse HAO and full-AO OEO orbitals must use the same
mathematical operator.

## 2. Pair-space notation

Let

$$
N = \frac{n_{\mathrm{bf}}(n_{\mathrm{bf}}+1)}{2},
\qquad
A = \frac{n_{\mathrm{act}}(n_{\mathrm{act}}+1)}{2},
$$

be the numbers of packed AO pairs and packed active-orbital pairs.  The
accepted active-orbital coefficient matrix and an orbital direction are

$$
C,D \in \mathbb{R}^{n_{\mathrm{bf}}\times n_{\mathrm{act}}}.
$$

For the packed AO pair $r=(\mu,\nu)$ and active pair $i=(p,q)$, define the
pair-product map

$$
B(C)_{r i}
=
C_{\mu p}C_{\nu q}
+
(1-\delta_{\mu\nu})C_{\nu p}C_{\mu q},
$$

and its directional derivative

$$
M(C;D)_{r i}
=
D_{\mu p}C_{\nu q}
+C_{\mu p}D_{\nu q}
+(1-\delta_{\mu\nu})
\left(
D_{\nu p}C_{\mu q}
+C_{\nu p}D_{\mu q}
\right).
$$

The exact AO electron-repulsion integrals define a symmetric linear operator

$$
K\in\mathbb{R}^{N\times N},
\qquad K^{\mathsf T}=K.
$$

The active-space two-electron integrals are

$$
G(C)=B(C)^{\mathsf T}K B(C).
$$

Only the lower triangle of $G$ is stored by the public active-space integral
interface.  All pair-space equations below use the corresponding symmetric
matrix representation.

### 2.1 Sparsity boundary

Strict sparsity belongs to the nonredundant HAO parameter coordinates $U_p$.
It does not imply that the accepted AO-by-active matrix $C$ is sparse. In
particular, removal of the inactive component gives

$$
C=(I-\Gamma_{\mathrm{inact}}S)\Phi_{\mathrm{act}},
$$

which is generally dense even when the input orbitals and $U_p$ supports are
sparse. Therefore the exact pair kernel must use the actual coefficients in
$C$ and cannot assume a sparse $B(C)$. Full-AO OEO and HAO share this rule.

## 3. Exact directional derivative

At an accepted point, write

$$
B=B(C),
\qquad
M=M(C;D),
\qquad
Q=KB,
\qquad
\dot Q=KM.
$$

The exact integral direction is

$$
\dot G
=
M^{\mathsf T}KB+B^{\mathsf T}KM
=
M^{\mathsf T}Q+Q^{\mathsf T}M.
$$

The second equality follows from $K^{\mathsf T}=K$.  This identity is an
implementation invariant: a streamed implementation may change the order of
contractions, but it must return the same packed $\dot G$.

## 4. Exact adjoint Hessian action

Let

$$
W\in\mathbb{R}^{A\times A}
$$

denote the accepted adjoint of the packed active-space two-electron integrals.
Let $J_C$ be the Jacobian of $B(C)$ and $J_C^{*}$ its Euclidean adjoint.  The
two-electron orbital gradient is

$$
g_C^{(2e)}=J_C^{*}(QW).
$$

For a fixed accepted adjoint $W$, its exact directional derivative is

$$
\dot g_{C,\mathrm{fixed}}^{(2e)}
=
J_D^{*}(QW)
+J_C^{*}(\dot QW).
$$

The active-space structure response produces an additional adjoint direction
$\dot W$.  Its orbital pullback is

$$
\dot g_{C,\mathrm{outer}}^{(2e)}
=
J_C^{*}(Q\dot W).
$$

Therefore the complete exact two-electron contribution is

$$
\boxed{
\dot g_C^{(2e)}
=
J_D^{*}(QW)
+J_C^{*}(\dot QW+Q\dot W)
}.
$$

These equations also define the finite-difference and explicit-Hessian
reference tests.  Streaming is correct only if it preserves every term in the
boxed expression.

## 5. Why the original storage was not scalable

The original accepted cache stored three dense pair matrices:

$$
B,\;Q,\;QW\in\mathbb{R}^{N\times A}.
$$

The original directional and HVP workspaces could additionally store

$$
M,\;\dot Q,\;MW,\;\dot QW
\in\mathbb{R}^{N\times A}.
$$

One such matrix requires

$$
8NA\ \text{bytes}.
$$

Because

$$
N=O(n_{\mathrm{bf}}^2),
\qquad
A=O(n_{\mathrm{act}}^2),
$$

each resident matrix scales as

$$
O(n_{\mathrm{bf}}^2n_{\mathrm{act}}^2).
$$

This is avoidable storage.  The persistent mathematical state consists of
$C$, $W$, pair indices, and the AO-pair operator $K$; $B$, $Q$, and $QW$ are
derived quantities.

## 6. Canonical streamed operator

The implementation exposes one exact pair-space operator with bounded
tiles.  A tile is a view of a contiguous interval of AO-pair rows or
active-pair columns, not a second approximate algorithm.

For an AO-row tile $I$, the operator can generate

$$
B_I,
\qquad
M_I,
\qquad
Q_I=(KB)_I,
\qquad
\dot Q_I=(KM)_I,
$$

and immediately consume those rows in one of three sinks:

1. accumulate $M_I^{\mathsf T}Q_I$ into the mandatory $A\times A$ integral
   direction;
2. apply $J_D^{*}$ or $J_C^{*}$ to a pair-gradient tile;
3. accumulate a diagnostic norm or checksum.

For an active-pair column tile $J$, associativity gives

$$
(KB)W = K(BW),
\qquad
(KM)W = K(MW).
$$

This permits adjoint contractions to be formed and consumed without a full
$N\times A$ result.  The tile interfaces must preserve the contraction order
required by the directional integral and must never create a hidden full-size
temporary through an Eigen expression.

The canonical implementation may let a tile span the complete small problem.
That is still the same code path.  It is not a dense compatibility fallback.

## 7. Dependency imposed by the structure response

The structure-response adjoint $\dot W$ is not known until $\dot G$ has been
formed and the selected-state response equation has been solved:

$$
D
\longrightarrow
\dot G
\longrightarrow
\dot W
\longrightarrow
J_C^{*}(Q\dot W).
$$

Consequently, removing the resident $Q$ matrix has an unavoidable exactness
tradeoff: $Q$ must either be recomputed after $\dot W$ becomes available or be
retained somewhere.  No contraction reordering can remove this dependency for
an arbitrary $\dot W$.

The implementation therefore uses an explicit memory--work policy:

- the complete accepted $Q$ is retained only when the combined dense $B,Q$
  construction workspace is no larger than the AO-pair graph plus the packed
  active tensor;
- otherwise $Q_I$ is recomputed exactly in bounded row tiles when consumed;
- the decision depends only on dimensions and representation storage, never on
  molecule names, orbital type, iteration number, or environment variables;
- changing the memory regime may change time and memory, but not numerical
  results.

This policy is part of the exact algorithm rather than a fallback path.

## 8. Memory invariant

Let $T_r$ and $T_a$ be the AO-row and active-pair tile extents.  Apart from the
AO integral graph and mandatory accepted state, the pair-space working memory
must satisfy

$$
M_{\mathrm{pair\ workspace}}
=
O(T_rA+NT_a+A^2),
$$

with

$$
T_r<N,
\qquad
T_a<A
$$

whenever a full $N\times A$ matrix exceeds the workspace capacity.  No vector
of directions may multiply this bound by the Krylov block size.  Block HVP
directions are streamed through the same bounded workspace.

For diagnostics, the implementation will report the peak number of resident
pair elements.  Tests must be able to assert

$$
N_{\mathrm{resident\ pair\ elements}}
< cNA
$$

for a forced multi-tile calculation, with $c<1$ chosen by the test capacity
rather than embedded in the scientific algorithm.

## 9. Parallel execution

Parallelism follows ownership, so accumulation does not require atomics in the
inner contraction:

- AO-row tiles own disjoint output rows during $KX$;
- thread-local $A\times A$ matrices accumulate the directional integral and
  are reduced once per tile or parallel region;
- dense orbital-gradient rows are partitioned by AO index during $J_C^{*}$;
- a tile is reused for all consumers before it is released.

The tile extent must be large enough for vectorization and Eigen microkernels,
but it is constrained by the workspace capacity.  Scheduling is based on AO
graph edge counts rather than row counts because exact-integral rows have
unequal work.

## 10. Implementation stages

### Stage A: remove proven redundant accepted buffers

1. Remove persistent $B$; it is used only to build other accepted data.
2. Remove persistent $QW$; form and consume it through the canonical tile
   operator.
3. Eliminate duplicate ownership of $Q$ between the accepted integral result
   and the exact-HVP cache.
4. Add storage diagnostics and numerical regression tests.

This stage reduces memory without changing the number of AO-pair operator
applications.

### Stage B: introduce bounded tile primitives

1. Add range-based builders for $B_I$ and $M_I$.
2. Add range-based AO-pair operator application.
3. Add range-based $J_C^{*}$ accumulation.
4. Replace whole-matrix Eigen expressions with tile-local contractions.

The whole-matrix routines are deleted after all consumers migrate; they are
not retained as compatibility implementations.

### Stage C: stream the directional integral and fixed adjoint

Fuse the tile lifetime so that $M_I$, $Q_I$, and $\dot Q_I$ are consumed by
$\dot G$ and the fixed-adjoint HVP before release.  Batch HVPs iterate over
directions inside the same capacity bound instead of concatenating
$N\times A$ matrices.

### Stage D: adaptive accepted-state storage

Retain the accepted $Q$ only when admitted by the intrinsic storage rule.
Otherwise recompute its bounded row tiles for the post-structure-response
$\dot W$ pullback. Record the selected regime, resident pair elements, and
tile rows.

### Stage E: delete obsolete paths

After the streamed operator passes all validation gates, delete the old full
pair-matrix response implementation and diagnostic code that depends on its
internal buffers.

## 11. Validation gates

Every stage must satisfy all of the following:

1. **Pair-map derivative**

$$
\frac{B(C+\varepsilon D)-B(C-\varepsilon D)}{2\varepsilon}
\approx M(C;D).
$$

2. **Integral directional derivative**

$$
\frac{G(C+\varepsilon D)-G(C-\varepsilon D)}{2\varepsilon}
\approx \dot G.
$$

3. **Adjoint consistency**

$$
\langle J_C D,Y\rangle
=
\langle D,J_C^{*}Y\rangle.
$$

4. **HVP finite difference**

$$
H(C)D
\approx
\frac{g(C+\varepsilon D)-g(C-\varepsilon D)}{2\varepsilon}.
$$

5. **Explicit reduced-Hessian comparison** for small systems.

6. **Tile invariance:** one-tile and forced multi-tile results agree to the
   expected floating-point tolerance.

7. **Orbital coverage:** both strictly sparse HAO and full-AO OEO inputs pass.

8. **Optimization trajectory:** accepted energies and convergence criteria
   remain consistent for F2, 241, MnF2, and FeCl2.

9. **Memory scaling:** forced streamed tests demonstrate bounded resident
   pair storage as $N$ and $A$ grow.

10. **Performance:** report wall time, AO-graph traversals, peak RSS, and peak
    resident pair elements.  A memory reduction is not presented as a speedup
    unless the measured wall time also improves.

## 12. Acceptance criterion

The milestone is complete only when the production exact TNHVP path no longer
requires a resident $N\times A$ accepted-point matrix or a resident
$N\times A$ per-direction matrix for large problems, while all numerical gates
remain satisfied.  Small-problem full tiles and large-problem partial tiles
must be two capacities of the same exact implementation.

## 13. Implementation status and initial measurements

Status on 2026-09-13:

1. Persistent $B$ and $QW$ matrices have been removed from the exact-HVP
   cache.
2. The accepted $Q$ matrix is borrowed directly from the forward active-space
   result and is no longer copied into the HVP operator.
3. The accepted active coefficient matrix $C$ is also borrowed; the operator
   and its exact-2e cache no longer own two additional copies.
4. When $Q$ is not retained, $Q_I$ and $\dot Q_I$ are generated from the AO
   pair graph in bounded AO-row tiles.
5. The same streamed pass forms $\dot G$ and the fixed-adjoint HVP.  The outer
   $\dot W$ pullback requires one subsequent exact $Q$ pass.
6. Scalar and block TNHVP applications use the same bounded path; block
   directions are not concatenated into full $N\times A$ buffers.
7. The old fixed cutoff $A\le 64$ has been deleted.  The forward builder keeps
   dense pair products only when the combined $B,Q$ construction workspace is
   no larger than the storage already required by the AO-pair graph and packed
   active tensor.

Forced multi-tile finite-difference tests give relative errors

$$
1.13\times 10^{-9}\quad\text{(F2 HAO)},
\qquad
5.58\times 10^{-9}\quad\text{(F2 full-AO OEO)}.
$$

The fixed-adjoint streamed and resident implementations agree to

$$
6.7\times10^{-16}\quad\text{(HAO)},
\qquad
1.3\times10^{-15}\quad\text{(OEO)}.
$$

The following single-sample timings used four OpenMP threads and one BLAS
thread.  They are engineering measurements, not publication benchmarks.

| system | resident $Q$ elements | resident full HVP / s | forced-stream full HVP / s | block/scalar relative error |
| --- | ---: | ---: | ---: | ---: |
| 241 | 152460 | 0.205 | 2.04 | $2.5\times10^{-16}$ |
| MnF2 | 279000 | 0.145 | 1.85 | $1.2\times10^{-14}$ |
| FeCl2 | 335376 | 0.258 | 1.51 | $2.6\times10^{-17}$ |

These accepted matrices occupy only approximately 1.22, 2.23, and 2.68 MB,
respectively.  Retaining them is therefore the correct memory--work decision
for the present test cases.  The forced-stream results quantify the cost paid
only after a future problem crosses the storage invariant; they must not be
reported as a speedup.
