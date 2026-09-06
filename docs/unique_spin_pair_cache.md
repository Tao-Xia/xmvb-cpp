# Unique Same-Spin Pair Cache For Full Determinant Matrices

## Goal

When the expanded full-determinant space is close to a Cartesian product of a
small alpha determinant set and a small beta determinant set, the current
full-pair loop performs the same alpha-pair and beta-pair work many times.

The optimization in this note does **not** cache full determinant pairs.
Instead, it caches the reusable same-spin kernels for unique alpha and beta
determinant pairs and then reconstructs the full determinant-pair matrix
element from those cached spin-resolved quantities.

## Notation

Let a full determinant be indexed as

$$
D_p = \left(\alpha_{i(p)}, \beta_{j(p)}\right),
$$

where $\alpha_{i(p)}$ is the alpha occupied-string of full determinant $p$ and
$\beta_{j(p)}$ is the beta occupied-string.

Define the same-spin determinant-pair quantities

$$
S^\alpha_{ij}, \quad H^\alpha_{ij}, \quad
S^\beta_{kl}, \quad H^\beta_{kl},
$$

where $S$ denotes the determinant overlap and $H$ denotes the same-spin
Hamiltonian matrix element for that spin block.

## Exact factorization of the full determinant pair

For two full determinants

$$
D_p = \left(\alpha_{i(p)}, \beta_{j(p)}\right), \qquad
D_q = \left(\alpha_{i(q)}, \beta_{j(q)}\right),
$$

the full overlap factorizes exactly as

$$
S_{pq} =
S^\alpha_{i(p)i(q)}
S^\beta_{j(p)j(q)}.
$$

The full Hamiltonian matrix element can be written as

$$
H_{pq} =
H^\alpha_{i(p)i(q)} S^\beta_{j(p)j(q)}
+ S^\alpha_{i(p)i(q)} H^\beta_{j(p)j(q)}
+ C^{\alpha\beta}_{pq},
$$

where $C^{\alpha\beta}_{pq}$ is the opposite-spin Coulomb contribution. The
first two terms depend only on the unique alpha pair and the unique beta pair.

This means the expensive same-spin determinant kernels can be evaluated once
per unique spin pair and reused across many full determinant pairs.

## What gets cached

For each unique alpha pair $(i,j)$ and beta pair $(k,l)$, cache the compact
same-spin result

$$
\mathcal{K}^\sigma_{ab} =
\left(
S^\sigma_{ab},
H^\sigma_{ab},
\text{nullity}^\sigma_{ab},
\text{overlap/cofactor data needed later}
\right),
\qquad \sigma \in \{\alpha,\beta\}.
$$

In code, the reusable payload is the existing
`SpinDeterminantPairEvaluation`. This keeps the overlap determinant, nullity,
and the overlap-factorization data needed to reconstruct first cofactors for
the opposite-spin term.

For regular same-spin pairs with overlap submatrix \(M\), the first deleted
minor matrix is

$$
C^{(1)} = \det(M)\, M^{-T}.
$$

For rank-\((n-1)\) pairs with singular value decomposition

$$
M = U \Sigma V^T,
$$

the same first deleted minor matrix remains available in closed form:

$$
C^{(1)} =
\operatorname{parity}(U,V)
\left(\prod_{k=1}^{n-1} \sigma_k\right)
u_n v_n^T,
$$

where \(u_n\) and \(v_n\) are the right/left null singular vectors in the
internal row/column convention.

This matrix is reused in three places:

$$
\text{same-spin } H^\sigma_{ab},
\qquad
C^{\alpha\beta}_{pq},
\qquad
\nabla \text{active-space integrals}.
$$

So the production payload should cache \(C^{(1)}\) once inside the overlap
result instead of rebuilding it independently in the forward Hamiltonian path,
the opposite-spin combine path, and the backward gradient path.

The production cache must preserve the **left/right orientation** of the
same-spin pair. Although the scalar same-spin matrix elements are symmetric,
the cached overlap factorization is not orientation-free: rows correspond to
the right determinant and columns correspond to the left determinant. Because
the opposite-spin Coulomb term reuses that factorization, the cache stores the
directed kernels

$$
\mathcal{K}^\alpha_{ij}
\quad \text{and} \quad
\mathcal{K}^\alpha_{ji}
$$

as distinct entries whenever $i \ne j$, and likewise for beta pairs.

The full determinant pair is then reconstructed as

$$
H_{pq} =
\mathcal{H}\!\left(
\mathcal{K}^\alpha_{i(p)i(q)},
\mathcal{K}^\beta_{j(p)j(q)}
\right),
$$

with the same exact formulas already used by the current full-pair evaluator.

When the reverse orientation is needed later in the gradient pass, the payload
must be transposed consistently:

$$
\left(C^{(1)}_{ij}\right)^T = C^{(1)}_{ji}.
$$

That means the backward path can either cache both directed kernels directly or
transpose the cached overlap payload lazily when it processes the swapped
unordered determinant pair contribution.

## Complexity reduction

Let

$$
N_{\text{full}} = \text{number of unique full determinants},
$$

$$
N_\alpha = \text{number of unique alpha determinants},
\qquad
N_\beta = \text{number of unique beta determinants}.
$$

The current same-spin work scales with the number of unordered full pairs:

$$
N_{\text{same-spin, direct}}
=
2 \cdot \frac{N_{\text{full}}(N_{\text{full}}+1)}{2}.
$$

With the unique same-spin cache, the reusable kernel count becomes

$$
N_{\text{same-spin, cached}}
=
N_\alpha^2 + N_\beta^2.
$$

The ideal reuse factor for the same-spin kernel is therefore

$$
\rho
=
\frac{
2 \cdot \frac{N_{\text{full}}(N_{\text{full}}+1)}{2}
}{
N_\alpha^2 + N_\beta^2
}.
$$

For the observed `C6H6.full.xmi` / `7675_VBSCF` pattern,

$$
N_{\text{full}} = 400, \qquad N_\alpha = N_\beta = 20,
$$

so

$$
N_{\text{same-spin, direct}} = 160400,
\qquad
N_{\text{same-spin, cached}} = 800,
$$

and the same-spin kernel reuse factor is about

$$
\rho \approx 200.5.
$$

## Memory scaling

Caching full determinant pairs would cost

$$
O\!\left(N_{\text{full}}^2\right),
$$

which is not acceptable for large full determinant spaces.

Caching only unique alpha and beta same-spin pairs costs

$$
O\!\left(N_\alpha^2 + N_\beta^2\right),
$$

which is much smaller whenever the full determinant space comes from combining
a modest number of unique alpha strings with a modest number of unique beta
strings.

This is why the implementation should cache **spin-pair kernels**, not
full-pair results.

## Opposite-spin term

The opposite-spin Coulomb contribution is not stored as a full-pair cache in
this design. It is still evaluated per full determinant pair, but it reuses the
cached overlap-factorization data from the alpha and beta same-spin caches.

That keeps the implementation exact while still removing the dominant repeated
same-spin determinant factorizations.

## Backward reuse

The same ordered alpha/beta kernel cache should be used not only during the
forward structure build, but also during the active-space backward pass.

The reverse-mode implementation revisits the same unordered full determinant
pairs and recomputes

$$
\mathcal{K}^\alpha_{i(p)i(q)},
\qquad
\mathcal{K}^\beta_{j(p)j(q)}
$$

from scratch before accumulating overlap, one-electron, and two-electron
adjoints. If the full determinant space is close to a Cartesian product,
backward sees the same combinatorial reuse as forward, so it should reconstruct
each full determinant pair from the cached same-spin kernels with the same
combination routine:

$$
\text{pair payload}_{pq}
\leftarrow
\mathcal{H}\!\left(
\mathcal{K}^\alpha_{i(p)i(q)},
\mathcal{K}^\beta_{j(p)j(q)}
\right).
$$

This keeps the algebra exact and removes repeated overlap resolution, repeated
same-spin Hamiltonian evaluation, and repeated first-cofactor construction from
the reverse pass as well.

## Fallback behavior

Small or sparse determinant spaces may have little or no alpha/beta reuse. In
that regime, the builder should keep the direct full-pair path instead of
forcing the cache path.

The production implementation also needs a memory guard. Even when the cache
reduces arithmetic, it should be disabled if the estimated same-spin cache
footprint becomes too large relative to the available memory budget. In the
current code path, the cache is enabled only when both

$$
N_\alpha^2 + N_\beta^2
<
N_{\text{full}}(N_{\text{full}}+1)
$$

and the estimated cache payload stays below a configurable memory cap.

By default the implementation uses a conservative same-spin cache budget of
$512\,\text{MiB}$, and the environment variable
`XMVB_SAME_SPIN_PAIR_CACHE_MB` can be used to tighten or relax that limit.

The per-pair payload estimate should include the dominant matrix data for

$$
M^{-1},
\qquad
U, V, \sigma,
\qquad
C^{(1)},
$$

because the cofactor cache is now part of the reusable same-spin kernel.

## Future extension

If a future workload has very large $N_\alpha$ or $N_\beta$, the same-spin
cache can be block-streamed:

$$
\text{alpha-pair block} \times \text{beta-pair sweep}
$$

so the memory ceiling stays proportional to one spin-pair block rather than the
entire unique spin-pair table.
