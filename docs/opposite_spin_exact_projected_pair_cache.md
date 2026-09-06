# Exact Opposite-Spin Projected Pair Cache

## Goal

The current same-spin cache removes repeated alpha-alpha and beta-beta
determinant kernels, but the opposite-spin Coulomb term is still evaluated
pair-by-pair.

For one full determinant pair

$$
D_p = \left(\alpha_{i(p)}, \beta_{j(p)}\right), \qquad
D_q = \left(\alpha_{i(q)}, \beta_{j(q)}\right),
$$

the remaining exact opposite-spin contribution is

$$
C^{\alpha\beta}_{pq}
=
\sum_{a_L,a_R,b_L,b_R}
C^\alpha_{a_R a_L}
C^\beta_{b_R b_L}

\left(
\beta_{b_R}^{R}\beta_{b_L}^{L}
\middle|
\alpha_{a_R}^{R}\alpha_{a_L}^{L}
\right),
$$

where \(C^\alpha\) and \(C^\beta\) are the first deleted-minor matrices of the
same-spin determinant overlaps.

The key observation is that this contraction does **not** need to stay in the
four-index occupied-orbital loop. It can be rewritten exactly in the packed
active-pair space already used by the production `eri_act` storage.

## Packed active-pair representation

Let

$$
M = \frac{n_{\mathrm{act}}(n_{\mathrm{act}} + 1)}{2}
$$

be the number of packed unordered active-orbital pairs.

Define the packed-pair index

$$
P = \operatorname{pack}(r, l),
$$

using the same unordered pair convention as `TwoElectronIndexer::packed_pair_index`.

For one same-spin determinant pair, define the exact packed-pair cofactor vector

$$
u^\sigma_P
=
\sum_{c,r}
C^\sigma_{r c}\,
\mathbf{1}\!\left[
P = \operatorname{pack}\!\left(o^{\sigma,R}_r, o^{\sigma,L}_c\right)
\right],
\qquad
\sigma \in \{\alpha,\beta\}.
$$

For regular determinant pairs, define the packed-pair inverse-overlap vector

$$
x^\sigma_P
=
\sum_{c,r}
\left(S_\sigma^{-1}\right)_{c r}\,
\mathbf{1}\!\left[
P = \operatorname{pack}\!\left(o^{\sigma,R}_r, o^{\sigma,L}_c\right)
\right].
$$

These vectors are exact regroupings of the current occupied-orbital loops. If
multiple matrix entries map to the same unordered active pair, their
coefficients are summed.

## Dense packed-pair kernel

The packed exact two-electron tensor can be viewed as a dense symmetric kernel
on packed active pairs:

$$
G_{PQ}
=
\left(
q_R q_L
\middle|
p_R p_L
\right),
$$

with storage taken directly from `eri_act` through the existing packed
pair-of-pairs index.

For the current active spaces of interest, \(M\) is still modest:

$$
n_{\mathrm{act}} = 6, 8, 10
\quad \Longrightarrow \quad
M = 21, 36, 55.
$$

So materializing one dense \(M \times M\) kernel once per active-space
evaluation is cheap.

## Exact forward contraction

With the packed-pair vectors above, the opposite-spin Coulomb term becomes

$$
C^{\alpha\beta}_{pq}
=
\left(u^\alpha\right)^T
G
u^\beta.
$$

If evaluated directly, this still costs \(O(M^2)\) per full determinant pair.
That is not yet the right production shape.

Instead, for each unique same-spin determinant pair cache entry, precompute the
projected vector

$$
v^\sigma = G u^\sigma.
$$

Then every full determinant pair reconstruction uses only a sparse-dense dot:

$$
C^{\alpha\beta}_{pq}
=
\left(u^\alpha\right)^T v^\beta
=
\left(u^\beta\right)^T v^\alpha.
$$

This keeps the contraction exact while reducing the per-full-pair work to

$$
O\!\left(\operatorname{nnz}(u^\alpha)\right)
\quad \text{or} \quad
O\!\left(\operatorname{nnz}(u^\beta)\right),
$$

instead of the original

$$
O\!\left(n_\alpha^2 n_\beta^2\right).
$$

## Exact backward contraction

The same idea applies to the regular-pair opposite-spin `phi` used by the
active-space gradient:

$$
\phi^{\alpha\beta}
=
\left(x^\alpha\right)^T
G
x^\beta.
$$

Precompute

$$
y^\sigma = G x^\sigma.
$$

Then

$$
\phi^{\alpha\beta}
=
\left(x^\alpha\right)^T y^\beta.
$$

More importantly, the inverse-overlap gradients collapse to simple packed-pair
lookups:

$$
\frac{\partial \phi^{\alpha\beta}}
{\partial \left(S_\alpha^{-1}\right)_{c r}}
=
y^\beta_{\operatorname{pack}(o^{\alpha,R}_r, o^{\alpha,L}_c)},
$$

and likewise for the beta block with \(y^\alpha\).

So the current fourfold opposite-spin `phi` loop in backward becomes:

1. one sparse-dense dot for \(\phi^{\alpha\beta}\),
2. one \(n_\alpha^2\) lookup pass for the alpha inverse-gradient matrix,
3. one \(n_\beta^2\) lookup pass for the beta inverse-gradient matrix.

## Two-electron gradient

The exact opposite-spin two-electron gradient remains an outer product in the
packed-pair space:

$$
\frac{\partial C^{\alpha\beta}_{pq}}{\partial G_{PQ}}
=
u^\alpha_P u^\beta_Q.
$$

This does **not** require the dense \(M^2\) outer product. The cached sparse
packed-pair vectors can be used directly:

$$
O\!\left(
\operatorname{nnz}(u^\alpha)
\operatorname{nnz}(u^\beta)
\right)
\text{ storage,}
\qquad
O\!\left(
\operatorname{nnz}(u^\alpha)\operatorname{nnz}(u^\beta)
\right)
\text{ work.}
$$

That matches the current occupied-orbital scaling in the worst case, but it
reuses the already cached pair regrouping and benefits whenever several
occupied-orbital products collapse onto the same packed pair.

## Production payload

For one ordered same-spin determinant pair, cache:

$$
\mathcal{U}^\sigma = \left\{(P, u^\sigma_P)\right\},
\qquad
\mathcal{V}^\sigma = v^\sigma,
$$

for nullity \(\le 1\), and additionally

$$
\mathcal{X}^\sigma = \left\{(P, x^\sigma_P)\right\},
\qquad
\mathcal{Y}^\sigma = y^\sigma,
$$

for regular pairs with nullity \(= 0\).

Here:

- \(\mathcal{U}^\sigma\) and \(\mathcal{X}^\sigma\) are sparse packed-pair
  vectors stored as `(packed_pair_index, value)` lists,
- \(\mathcal{V}^\sigma\) and \(\mathcal{Y}^\sigma\) are dense length-\(M\)
  projected vectors used for fast full-pair reconstruction and backward
  inverse-gradient lookups.

## Where it connects in code

This exact optimization should be applied only inside the ordered unique
same-spin pair cache path:

1. Build one dense packed-pair kernel \(G\) from `eri_act` once per active
   space evaluation.
2. For each cached unique alpha pair and beta pair:
   compute and store \(\mathcal{U}^\sigma\), \(\mathcal{V}^\sigma\), and, when
   available, \(\mathcal{X}^\sigma\), \(\mathcal{Y}^\sigma\).
3. In forward full-pair reconstruction, replace the explicit opposite-spin
   occupied-orbital loops with the sparse-dense dot
   \(\left(u^\alpha\right)^T v^\beta\).
4. In backward opposite-spin `phi` and inverse-overlap gradient construction,
   replace the explicit fourfold loops with \(\left(x^\alpha\right)^T y^\beta\)
   and packed-pair lookups.
5. In backward opposite-spin ERI gradients, replace the occupied-orbital loops
   with the sparse packed-pair outer product.

This keeps the implementation exact and makes the opposite-spin path follow the
same reuse pattern as the already deployed unique same-spin cache.

## Relation to RI

This document intentionally stays on the exact packed-ERI path.

If later work enables RI/DF production mode with

$$
G_{PQ} \approx \sum_A L_{A,P} L_{A,Q},
$$

the same packed-pair vectors \(u^\sigma\) and \(x^\sigma\) remain the right
interface. Only the projected vectors change from

$$
v^\sigma = G u^\sigma,
\qquad
y^\sigma = G x^\sigma
$$

to RI projections through the auxiliary factors \(L\).
