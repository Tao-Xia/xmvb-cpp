# Exact Separator Boundary Message Data Design

> Status note (2026-04-05):
> this document remains the design target for the final optimized recursive
> boundary-message layout. The current production `one_leaf_star` path is
> already exact, and the generic rooted-tree Hamiltonian path is now connected
> through a typed recursive Hamiltonian bundle. However, the active bundle is
> still stored in a more generic payload form than the explicit
> boundary-sector / family layout sketched here.

## 1. Purpose

This note turns the closure formulas into concrete message objects that can be
implemented in `component_tree.cpp` without ambiguity.

The main conclusion is:

1. the exact rooted-tree message must be organized by **boundary sector**;
2. within one sector, the exact content must be organized by **degree family**
   and **support-space basis type**;
3. opposite-spin and same-spin require different higher-order exact objects;
4. the current payloads
   `JointDeletionPayload`,
   `SameSpinBoundaryPayload`,
   `HamiltonianBoundaryPayload`
   are not the final exact representation.

This document is the data-structure companion to
[root_closure.md](./root_closure.md).

---

## 2. Boundary Sector Index

For one outgoing spin boundary \(B^\sigma\), a boundary sector is still the
selected-mask pair

\[
s = (R,C),
\]

where:

- \(R\) is the selected right/row boundary mask;
- \(C\) is the selected left/column boundary mask.

This is already implemented by `BoundarySectorIndexer`.

For exact degree-`0/1/2` messages we do **not** change the exponential object.
The exponential object remains only the boundary sector index.

Everything else is polynomial support-space payload attached to that sector.

---

## 3. Family Delta

For one spin block \(X\), define the local family delta

\[
\delta(X) := n_r(X) - n_c(X).
\]

In exact degree-`0/1/2` boundary algebra, the admissible family range is

\[
-2 \le \delta \le 2.
\]

The meaning is:

- \(\delta = 0\): balanced family;
- \(\delta = +1\): one extra row family;
- \(\delta = -1\): one extra column family;
- \(\delta = +2\): two extra rows family;
- \(\delta = -2\): two extra columns family.

This family structure is not optional. It is the exact statement of which
deleted-minor sectors can appear under degree-`2` merge.

The one-leaf code already hints at this through
`BoundarySectorFamilyIndexerRange` and the family range `-2 .. +2`.

---

## 4. Exact One-Spin Degree-0/1/2 Basis

The exact one-spin message is **not** just:

\[
(S,\; r,\; c,\; C,\; \Gamma).
\]

That shorthand is useful conceptually, but it is not the full family-complete
implementation basis.

The actual exact one-spin degree-`\le 2` basis is:

### 4.1 Family \(\delta = 0\)

Balanced square family:

\[
O,
\qquad
C(q,p),
\qquad
\Gamma(q_1,q_2;p_1,p_2).
\]

These correspond to:

- closed overlap scalar;
- first cofactor `(1,1)`;
- second cofactor `(2,2)`.

### 4.2 Family \(\delta = +1\)

One extra row family:

\[
u(q),
\qquad
P(q_1,q_2;p).
\]

These correspond to:

- row-open minors `(1,0)`;
- mixed degree-`2` minors `(2,1)`.

### 4.3 Family \(\delta = -1\)

One extra column family:

\[
v(p),
\qquad
Q(q;p_1,p_2).
\]

These correspond to:

- col-open minors `(0,1)`;
- mixed degree-`2` minors `(1,2)`.

### 4.4 Family \(\delta = +2\)

Two extra rows family:

\[
U(q_1,q_2),
\]

corresponding to `(2,0)` deleted minors.

### 4.5 Family \(\delta = -2\)

Two extra columns family:

\[
V(p_1,p_2),
\]

corresponding to `(0,2)` deleted minors.

So the full exact one-spin degree-`\le 2` basis is

\[
\mathcal B^\sigma_{\le 2}
=
\{
O,\;
u,\;
v,\;
C,\;
P,\;
Q,\;
\Gamma,\;
U,\;
V
\}.
\]

That is the exact family-complete object required for same-spin closure.

---

## 5. Packed Polynomial Indices

Within each family, the payload should be stored in packed support-space basis
indices.

### 5.1 Degree-1 packers

Use:

- single row index
  \[
  q
  \]
  for \(u(q)\);
- single column index
  \[
  p
  \]
  for \(v(p)\);
- support pair index
  \[
  \mu_C = (q,p)
  \]
  for \(C(q,p)\).

### 5.2 Degree-2 packers

Use:

- row pair index
  \[
  \rho = (q_1,q_2),\qquad q_1 < q_2
  \]
  for \(U(q_1,q_2)\);
- column pair index
  \[
  \kappa = (p_1,p_2),\qquad p_1 < p_2
  \]
  for \(V(p_1,p_2)\);
- pair-single index
  \[
  \pi_P = (\rho,p)
  \]
  for \(P(q_1,q_2;p)\);
- single-pair index
  \[
  \pi_Q = (q,\kappa)
  \]
  for \(Q(q;p_1,p_2)\);
- pair-pair index
  \[
  \nu = (\rho,\kappa)
  \]
  for \(\Gamma(q_1,q_2;p_1,p_2)\).

These are all polynomial-size support-space indices.

The message remains exponential only in the boundary sector count.

---

## 6. Exact One-Spin Message Layout

The clean exact one-spin message should be typed by family.

### 6.1 Degree-1 one-spin message

The exact degree-`\le 1` message is:

```cpp
struct BoundaryDegree1SpinMessage {
  BoundarySectorIndexer family_neg1_indexer;  // delta = -1
  Eigen::MatrixXd col_open_values;            // rows = p, cols = sector

  BoundarySectorIndexer family_0_indexer;     // delta = 0
  Eigen::VectorXd overlap_values;             // cols = sector
  Eigen::MatrixXd cofactor_values;            // rows = packed (q,p), cols = sector

  BoundarySectorIndexer family_pos1_indexer;  // delta = +1
  Eigen::MatrixXd row_open_values;            // rows = q, cols = sector
};
```

This object is exactly the implementation form of

\[
\bar U^\sigma.
\]

### 6.2 Degree-2 one-spin message

The exact degree-`\le 2` message extension is:

```cpp
struct BoundaryDegree2SpinMessage {
  BoundarySectorIndexer family_neg2_indexer;  // delta = -2
  Eigen::MatrixXd col_pair_values;            // rows = packed (p1,p2), cols = sector

  BoundarySectorIndexer family_neg1_indexer;  // delta = -1
  Eigen::MatrixXd q_values;                   // rows = packed (q,p1,p2), cols = sector

  BoundarySectorIndexer family_0_indexer;     // delta = 0
  Eigen::MatrixXd gamma_values;               // rows = packed ((q1,q2),(p1,p2)), cols = sector

  BoundarySectorIndexer family_pos1_indexer;  // delta = +1
  Eigen::MatrixXd p_values;                   // rows = packed ((q1,q2),p), cols = sector

  BoundarySectorIndexer family_pos2_indexer;  // delta = +2
  Eigen::MatrixXd row_pair_values;            // rows = packed (q1,q2), cols = sector
};
```

This object is exactly the implementation form of

\[
G^\sigma.
\]

### 6.3 Important clarification

For implementation:

- `BoundaryDegree1SpinMessage` is not a replacement for overlap;
- `BoundaryDegree2SpinMessage` is not only \(\Gamma\).

It is the full typed family basis needed for exact closure.

If the implementation stores only `cofactor_values` and `gamma_values`, it is
still incomplete.

---

## 7. Exact Mixed Opposite-Spin Object

The exact opposite-spin message is not another one-spin object. It is a mixed
alpha/beta second-order moment on the degree-`1` basis.

Let:

\[
\mu \in \mathcal B^\alpha_{\le 1},
\qquad
\nu \in \mathcal B^\beta_{\le 1}.
\]

Then the exact mixed object is

\[
X_{\mu\nu}
=
\sum_t w_t\, a^\alpha_{t\mu} a^\beta_{t\nu}.
\]

### 7.1 Exact storage layout

Because alpha and beta each still carry boundary sectors, the exact message
must be indexed by the **joint** alpha/beta sector:

\[
s_{\alpha\beta} = (s_\alpha, s_\beta).
\]

The natural exact storage is block-typed by family pair:

```cpp
struct BoundaryMixedDegree1MomentBlock {
  int alpha_family_delta = 0;
  int beta_family_delta = 0;
  BoundarySectorIndexer alpha_indexer;
  BoundarySectorIndexer beta_indexer;
  Eigen::MatrixXd values;  // rows = packed (mu,nu), cols = packed (s_alpha,s_beta)
};

struct BoundaryMixedDegree1MomentMessage {
  std::vector<BoundaryMixedDegree1MomentBlock> blocks;
};
```

Only family pairs

\[
(\delta_\alpha,\delta_\beta) \in \{-1,0,+1\}^2
\]

appear here, because opposite-spin is degree `1` in each spin channel.

### 7.2 Why this object is necessary

This object is the exact carrier of alpha/beta correlation after local term
aggregation.

Without it, one can reconstruct the marginals

\[
\bar U^\alpha,\qquad \bar U^\beta
\]

and therefore the exact one-electron channel, but not the exact
opposite-spin channel.

---

## 8. Family Arithmetic Under Merge

The family picture gives a simple exact bookkeeping rule.

### 8.1 Degree-1

For degree-`1`,

\[
\delta_{\text{out}} = \delta_{\text{left}} + \delta_{\text{right}},
\]

with the admissible outputs restricted to

\[
\delta_{\text{out}} \in \{-1,0,+1\}.
\]

That is the family form of the exact `row-open / col-open / cofactor` merge.

### 8.2 Degree-2

For degree-`2`,

\[
\delta_{\text{out}} = \delta_{\text{left}} + \delta_{\text{right}},
\]

with admissible outputs

\[
\delta_{\text{out}} \in \{-2,-1,0,+1,+2\}.
\]

This is the family form of the exact `U/V/P/Q/Gamma` merge.

So the exact structure tensor

\[
\mathcal S^\sigma_{\nu i m}
\]

is block sparse by family:

\[
\delta(\nu) = \delta(i) + \delta(m).
\]

This is the key implementation simplification for same-spin.

---

## 9. Exact Frontier/Root Closure Objects

In the current `component_tree` architecture, the exact aggregated closure
should act on the following objects:

\[
\bar U_F^\alpha,\quad
\bar U_F^\beta,\quad
G_F^\alpha,\quad
G_F^\beta,\quad
X_F,
\]

for the frontier side, and

\[
\bar U_R^\alpha,\quad
\bar U_R^\beta,\quad
G_R^\alpha,\quad
G_R^\beta,\quad
X_R,
\]

for the root/local side.

Then:

- degree-`1` one-spin closure uses only \(\bar U_F^\sigma,\bar U_R^\sigma\);
- same-spin closure uses \(G_F^\sigma,G_R^\sigma\) together with
  \(\bar U_F^\sigma,\bar U_R^\sigma\);
- opposite-spin closure uses \(X_F,X_R\).

This is the exact content the current root closure is missing.

---

## 10. Exact Production Message Bundle

The clean exact production message for one subtree should therefore be:

```cpp
struct ExactBoundaryMessageBundle {
  BoundaryDegree1SpinMessage alpha_degree1;
  BoundaryDegree1SpinMessage beta_degree1;
  BoundaryDegree2SpinMessage alpha_degree2;
  BoundaryDegree2SpinMessage beta_degree2;
  BoundaryMixedDegree1MomentMessage opposite_spin_mixed;
};
```

This bundle is the exact boundary-only analogue of the full subtree state.

It is sufficient to reconstruct:

1. overlap;
2. one-electron;
3. opposite-spin;
4. same-spin;
5. total electronic Hamiltonian.

No full subtree determinant-state table is needed above this layer.

---

## 11. What This Means For The Current Refactor

The exact next implementation target should be:

1. replace the current degree-`1` ad hoc payload by the typed
   `BoundaryDegree1SpinMessage`;
2. add the exact mixed object `BoundaryMixedDegree1MomentMessage` for
   opposite-spin;
3. add the exact typed degree-`2` object `BoundaryDegree2SpinMessage` for
   same-spin;
4. make root closure act on these exact objects directly.

Only after these message objects exist does it make sense to optimize
contraction order, sparsity layout, or truncation.

That is the exact representation-level closure needed for a real
`2^n \to 2^m` implementation.
