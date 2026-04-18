# Exact Raw-VB Overlap: Metric-Aware Graph And DP Design

## 1. Purpose

This note sharpens the only remaining exact-compression direction that still
looks mathematically credible after the failure of the single-AGP route.

The target is no longer:

- generic worst-case polynomial evaluation of exact raw-VB overlap.

That target is currently unsupported.

The realistic target is instead:

- exact raw-VB overlap;
- but with the exponential dependence moved from the total number of covalent
  bonds to a smaller **metric-aware structural width** when the orbital metric
  has exploitable locality.

In other words, the goal is to replace

```math
O(2^m n^3)
```

by something closer to

```math
O(\operatorname{poly}(n)\,\gamma^{w}),
```

where:

- `m` is the total number of covalent bond labels in the structure pair;
- `w` is a separator width or treewidth of a metric-aware interaction graph;
- `w << m` is hoped for in localized-orbital chemistry.

This note is a **design document**, not a finished proof.

Relevant files:

- `src/pfaffian_vbscf/raw_vb_exact_overlap_generating_function.md`
- `src/pfaffian_vbscf/raw_vb_overlap_coefficient_extraction_analysis.md`
- `legacy_pair_generating_function.md`
- `src/tools/check_union_graph_overlap_blocks.cpp`
- `src/vb/matrices/union_graph_screening.hpp`
- `src/vb/matrices/union_graph_rank_predictor.hpp`

---

## 2. Starting Exact Formula

For one closed-shell raw-VB structure pair `(X,Y)`, the exact overlap is

```math
S_{XY}
=
[\mathbf{x}^{0}\mathbf{y}^{0}]\,
\det K^\alpha_{XY}(\mathbf{x},\mathbf{y})\,
\det K^\beta_{XY}(\mathbf{x},\mathbf{y}),
```

where

```math
K^\alpha_{XY}(\mathbf{x},\mathbf{y}) = U_X(\mathbf{x}) S U_Y(\mathbf{y})^{T},
```

```math
K^\beta_{XY}(\mathbf{x},\mathbf{y}) = V_X(\mathbf{x}) S V_Y(\mathbf{y})^{T}.
```

Equivalently, because each covalent label enters with exponent in
`{-1,0,+1}`, the exact coefficient can be extracted by hypercube averaging:

```math
S_{XY}
=
\frac{1}{2^m}
\sum_{\lambda \in \{\pm 1\}^{m}}
\det K^\alpha_{XY}(\lambda)\,
\det K^\beta_{XY}(\lambda),
```

where `m = c_X + c_Y` is the total number of covalent bond labels.

This formula is already numerically verified in
`scripts/verify_raw_vb_overlap_generating_function.py`.

So the exact overlap problem has now been cleanly reduced to:

- a polynomial-cost determinant kernel for one fixed label assignment;
- an exact sum over `2^m` assignments.

The only exact-compression question that remains is:

- can this `2^m` label sum be reorganized so that the true exponent depends on
  a smaller structural width.

---

## 3. Why A Graph Should Enter At All

The determinant kernel for one fixed assignment is dense in general, but it is
not arbitrary.

Each row and column of `K^\alpha` or `K^\beta` comes from one bond selector:

- left bond `i` contributes one row;
- right bond `j` contributes one column;
- an entry depends only on the small orbital-overlap subblock connecting the
  two bonds.

So the dense `n x n` kernel is assembled from many small bond-to-bond
interaction blocks.

This suggests that exact compression, if it exists, should not be phrased in
terms of individual determinants or individual orbitals alone, but rather in
terms of:

- bonds;
- groups of bonds;
- and the metric couplings between their orbital supports.

The pair-union graph already captures part of this story, but it ignores the
actual overlap metric `S`. That is why union-graph topology alone cannot give
exact factorization.

So the correct structural object must be **metric-aware**.

---

## 4. Three Graph Levels

To avoid confusion, it is useful to separate three related graphs.

### 4.1 Orbital metric graph

Let `G_orb(S)` be the graph whose vertices are active orbitals and whose edges
are the nonzero entries of the overlap matrix:

```math
(p,q) \in E(G_{\mathrm{orb}}) \iff S_{pq} \neq 0.
```

For approximate screening or empirical diagnostics, one may replace exact
nonzero by a threshold on `|S_{pq}|`.

This graph tells us which orbitals are directly coupled by the metric.

### 4.2 Pair-union graph

Let `G_union(X,Y)` be the graph formed by overlaying the two pairings.

Its connected components are the objects already exposed by the existing
codepath:

- doubled edges;
- alternating cycles;
- more general components.

This graph is useful, but it is not sufficient for exact factorization because
metric couplings can connect different union-graph components.

### 4.3 Metric-aware bond interaction graph

This is the graph that matters for exact compression.

There are two natural versions.

#### Version A: bond-label graph

Vertices are the covalent bond labels of the left and right structures:

- left label vertices `x_1,\dots,x_{c_X}`;
- right label vertices `y_1,\dots,y_{c_Y}`.

An edge is placed between two labels if their orbital supports interact through
the metric strongly enough that their contributions cannot be separated.

For example, between left bond `x_i = (p_i,q_i)` and right bond `y_j = (r_j,s_j)`,
one may define an exact interaction criterion

```math
\max\{
|S_{p_i r_j}|, |S_{p_i s_j}|, |S_{q_i r_j}|, |S_{q_i s_j}|
\} > 0.
```

This is the finest-grained graph and the most natural one for sign-label DP.
After coarse graining, additional same-side supernode couplings may also
appear through shared separators. So the exact separator graph need not remain
strictly bipartite even if the finest label graph starts that way.

#### Version B: union-component graph

Collapse each pair-union component into one supernode. Then connect two
supernodes if the support-overlap cross-block between them is nonzero.

This is the exact graph-level interpretation of the current diagnostics already
available in:

- `src/tools/check_union_graph_overlap_blocks.cpp`
- `src/vb/matrices/union_graph_screening.hpp`

This coarser graph is more practical as an initial diagnostic and as a first
candidate for block or separator decompositions.

The likely research path is:

- use Version B first for empirical structure discovery;
- if promising, refine to Version A or a hybrid separator graph for exact DP.

---

## 5. What Exact Factorization Already Looks Like

Suppose the support-overlap matrix can be permuted into exact block-diagonal
form with blocks aligned to a partition of the bond interaction graph:

```math
S = \operatorname{diag}(S^{(1)},\dots,S^{(B)}).
```

Assume also that every bond lies fully within one block.

Then:

- both selector matrices split by block;
- both determinant kernels split by block;
- the exact constant-term extraction splits by block as well.

So the overlap factors exactly:

```math
S_{XY} = \prod_{b=1}^{B} S_{XY}^{(b)}.
```

This is the ideal exact-compression case.

The key lesson is not merely that factorization can happen, but rather:

- the correct decomposition criterion is block structure in the metric-aware
  support matrix, not pair-union topology alone.

This is the first reason a metric-aware graph is the right object.

---

## 6. Separator Elimination View

The next step beyond exact block factorization is separator decomposition.

Assume the bond interaction graph is partitioned into:

- one left subproblem `L`;
- one separator `B`;
- one right subproblem `R`;

such that all couplings between `L` and `R` pass through `B`.

For one fixed bond-label assignment `\lambda`, the determinant kernel can be
reordered into block form:

```math
K^\sigma(\lambda) =
\begin{pmatrix}
K^\sigma_{LL} & K^\sigma_{LB} & 0 \\
K^\sigma_{BL} & K^\sigma_{BB} & K^\sigma_{BR} \\
0 & K^\sigma_{RB} & K^\sigma_{RR}
\end{pmatrix},
\qquad
\sigma \in \{\alpha,\beta\}.
```

Then block elimination gives

```math
\det K^\sigma(\lambda)
=
\det K^\sigma_{LL}
\det K^\sigma_{RR}
\det \widetilde{K}^\sigma_{BB}(\lambda),
```

where the exact separator Schur complement is

```math
\widetilde{K}^\sigma_{BB}
=
K^\sigma_{BB}
- K^\sigma_{BL}(K^\sigma_{LL})^{-1}K^\sigma_{LB}
- K^\sigma_{BR}(K^\sigma_{RR})^{-1}K^\sigma_{RB}.
```

The important point is structural:

- the interior of `L` and `R` can be compressed into boundary data on `B`.

This is the exact algebraic mechanism behind any future dynamic program.

So the right abstraction for a subtree is not just a scalar overlap
contribution. It is:

- a scalar determinant prefactor;
- plus an effective separator kernel seen from that subtree.

That observation is crucial. A scalar-only message would in general lose
information.

---

## 7. Candidate Exact DP State

This leads to the following candidate exact message for one subtree `T`.

Let:

- `I_T` be the interior bond labels of the subtree;
- `B_T` be the boundary bond labels touching the separator.

For each assignment of the boundary labels `\lambda_{B_T}`, define a message

```math
\mathcal{M}_T(\lambda_{B_T})
=
\Big(
d^\alpha_T(\lambda_{B_T}),
d^\beta_T(\lambda_{B_T}),
\widetilde{K}^\alpha_T(\lambda_{B_T}),
\widetilde{K}^\beta_T(\lambda_{B_T})
\Big),
```

where:

- `d^\alpha_T` and `d^\beta_T` are determinant prefactors obtained after exact
  summation over interior labels and elimination of interior rows or columns;
- `\widetilde{K}^\alpha_T` and `\widetilde{K}^\beta_T` are effective separator
  kernels of size `|B_T| x |B_T|`.

The corresponding partial exact sum is

```math
\mathcal{M}_T(\lambda_{B_T})
=
\sum_{\lambda_{I_T}}
\Bigl(
\text{eliminate subtree interior exactly}
\Bigr).
```

In words:

- interior labels are summed exactly inside the subtree;
- the result is compressed to a boundary object indexed only by the separator
  labels.

When two child subtrees are merged across a separator, their messages combine
through block determinant identities and another exact summation over the
labels internal to the merge.

This is the matrix-valued exact analogue of junction-tree or transfer-matrix
dynamic programming.

---

## 8. Two State-Compression Levels

There are two natural levels of ambition for the DP state.

### 8.1 Sign-only boundary state

The simplest possibility is that, after a suitable derivation, each subtree can
be summarized by a scalar table over boundary sign assignments:

```math
f_T(\lambda_{B_T}).
```

Then the cost would be roughly of the form

```math
O(\operatorname{poly}(n)\,2^{w}),
```

where `w` is the maximum number of boundary labels.

This would be the cleanest outcome, but it is not yet justified.

### 8.2 Matrix-valued separator state

The more realistic exact possibility is the message defined above:

- one scalar prefactor per boundary label assignment;
- one small effective separator matrix per assignment.

If the separator size is `s` and the boundary-label width is `w`, then one
merge step would have a cost more like

```math
O(2^{w}\,s^3)
```

or, depending on the exact merge rule,

```math
O(\gamma^{w}\,s^3),
```

with `gamma` between `2` and `3`.

This is less elegant, but still potentially far better than `2^m n^3` when
`w << m`.

This note treats the matrix-valued separator state as the more credible exact
target.

---

## 9. Relation To Existing Union-Graph Diagnostics

The current code already computes several objects that are directly relevant to
this exact-compression route:

- support orbitals of a structure pair;
- pair-union connected components;
- block-diagonalized support-overlap matrices;
- off-block overlap matrices;
- singular values and numerical ranks of cross-block couplings.

These diagnostics currently support a heuristic rank-screening discussion.

For the present exact-DP program, they should be reinterpreted as:

- empirical probes of whether the metric-aware component graph has small
  separators;
- empirical probes of whether off-block couplings are low-rank enough to make
  matrix-valued separator messages compact;
- empirical probes of whether exact block factorization happens often enough in
  localized orbitals to be chemically useful.

So the current union-graph tooling is not obsolete. It is an exploratory front
end for the more exact metric-aware DP route.

---

## 10. Exact Low-Rank Boundary Idea

There is one additional refinement worth making explicit.

Suppose that after splitting into blocks or separators, the cross-block metric
coupling has low rank:

```math
E = A B^{T},
\qquad
\operatorname{rank}(E)=r.
```

Then for one fixed label assignment, the determinant kernels can be rewritten
using matrix determinant lemma style formulas so that the subtree interaction
passes through only an `r`-dimensional boundary.

This suggests a refined message type:

```math
\mathcal{M}_T(\lambda_{B_T})
=
\Big(
d^\alpha_T,
d^\beta_T,
L^\alpha_T,
L^\beta_T
\Big),
```

where `L^\sigma_T` is a low-rank boundary representation instead of a dense
separator matrix.

If exact low-rank boundary compression is available, then the practical cost
could become closer to

```math
O(\gamma^{w} r^3)
```

than to `O(\gamma^{w} s^3)`.

This does not remove the exponential in `w`, but it could drastically improve
the constant for realistic systems.

This is the second reason the current cross-block singular-value diagnostics
are important.

---

## 11. What Is Already Solid And What Is Not

### Solid

The following points are already on firm ground:

- exact raw-VB overlap is a bond-labeled generating-function problem;
- exact coefficient extraction can be written as a `2^m` hypercube average;
- exact block factorization occurs when the support-overlap metric is exactly
  block diagonal in a bond-compatible ordering;
- separator elimination is the correct algebraic lens for any future exact DP.

### Not yet derived

The following points remain open:

- the exact matrix-valued separator recurrence;
- the smallest sufficient boundary state for exact merging;
- whether a sign-only state ever closes exactly in the generic nonorthogonal
  case;
- whether realistic localized-orbital structure pairs actually have small
  separator widths.

So the DP route is currently a serious design direction, not a completed
algorithm.

---

## 12. Concrete Next Steps

The development path should be incremental.

### Step 1: exact graph definitions and diagnostics

Add an explicit **metric-aware component graph** builder that starts from the
existing union-graph components and adds weighted edges from the support-overlap
cross blocks.

For each structure pair, report:

- component count;
- separator candidates from graph partitioning;
- maximum component covalent count;
- approximate treewidth surrogate;
- cross-block ranks and norms.

### Step 2: exact block-factorized overlap

Implement the trivial exact special case:

- if the metric-aware graph disconnects exactly, evaluate each connected block
  independently and multiply the results.

This gives an exact acceleration in the easiest favorable cases and serves as a
sanity check.

### Step 3: exact hypercube kernel on graph blocks

Reimplement the current exact hypercube overlap using the metric-aware block
partition, so that the code path is already organized by blocks even before the
full DP exists.

### Step 4: derive matrix-valued separator message

For a one-separator split, derive the exact Schur-complement message update for

- `K^\alpha`;
- `K^\beta`;
- and the combined overlap contribution.

Do this first for one tree split, not for a full general graph.

### Step 5: test width empirically

On small but nontrivial molecules, measure whether the metric-aware separator
width remains significantly below the total number of covalent bonds.

If it does not, the exact-DP route is probably not worth pursuing further.

If it does, then this becomes the strongest remaining exact-compression line.

---

## 13. Bottom Line

If there is still hope for exact raw-VB compression beyond naive `2^n`, it is
not in another single-state collapse.

It is here:

- keep the exact bond-labeled generating object;
- define a metric-aware interaction graph;
- compress exact evaluation by block factorization, separator elimination, and
  possibly low-rank boundary messages.

This does **not** promise generic worst-case polynomial scaling.

What it does promise, if it works, is something more realistic:

- exact raw-VB overlap whose exponential depends on a smaller chemical-structure
  width rather than on the total number of covalent bonds.
