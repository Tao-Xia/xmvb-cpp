# Exact Separator Boundary Message Implementation Design

## 1. Purpose

This note continues
[boundary_repr.md](./boundary_repr.md).

That previous document answered the conceptual question:

> what exact object replaces the current full-state representation if we want
> the exponential dependence to move from subtree combinatorics to separator
> width?

The answer there was:

$$
M_v^\sigma(R,C)
$$

and its degree-1 / degree-2 derivative extensions.

This note answers the next question:

> how should those exact boundary messages actually be represented and merged
> in code?

The goals are:

1. keep the exact formalism clear;
2. expose the boundary-only state basis explicitly;
3. derive exact merge formulas that can become production code;
4. identify the exact point where `SVD` truncation should be inserted for
   chemical-accuracy compression.

This note is still design-only. It does not change code.

## 2. Node Model And Boundary Sets

Consider one rooted tree node $v$.

Its children are

$$
c = 1,\dots,t.
$$

For one spin channel $\sigma$, split the row/column labels seen at node $v$
into:

1. node-local internal labels that will be eliminated inside the node;
2. child-interface boundary labels that connect to each child;
3. the outgoing parent-facing boundary labels of the whole subtree.

Write:

$$
X_{c,R}^\sigma,\quad X_{c,L}^\sigma
$$

for the right/left boundary labels connecting child $c$ to node $v$, and

$$
B_{v,R}^\sigma,\quad B_{v,L}^\sigma
$$

for the outgoing right/left boundary labels of the merged subtree rooted at
$v$.

Define the full node boundary as the disjoint union

$$
\Gamma_{v,R}^\sigma
=
\Bigl(\bigsqcup_{c=1}^t X_{c,R}^\sigma\Bigr)
\sqcup
B_{v,R}^\sigma,
$$

$$
\Gamma_{v,L}^\sigma
=
\Bigl(\bigsqcup_{c=1}^t X_{c,L}^\sigma\Bigr)
\sqcup
B_{v,L}^\sigma.
$$

The entire node-local merge problem is:

1. integrate node-local internal labels out exactly;
2. contract child messages across the interface blocks
   $X_c^\sigma$;
3. produce a new outgoing message on $B_v^\sigma$ only.

## 3. Boundary Sector Basis

## 3.1 Deleted-set sector

For any boundary block with row set $Y_R$ and column set $Y_L$, define one
sector by deleted subsets

$$
R \subseteq Y_R,
\qquad
C \subseteq Y_L,
\qquad
|R|=|C|.
$$

We call

$$
d = |R| = |C|
$$

the sector degree.

The exact deleted-set sector count is

$$
N_{\mathrm{sec}}(Y_R,Y_L)
=
\sum_{d=0}^{\min(|Y_R|,|Y_L|)}
\binom{|Y_R|}{d}\binom{|Y_L|}{d}.
$$

This is the exact one-spin boundary state count.

## 3.2 Sector indexer

The implementation should use one reusable indexer:

```cpp
struct BoundarySector {
  uint32_t row_mask = 0;
  uint32_t col_mask = 0;
  int degree = 0;
};

class BoundarySectorIndexer {
 public:
  int row_count() const;
  int col_count() const;
  int sector_count() const;

  const BoundarySector& sector(int index) const;
  int index(uint32_t row_mask, uint32_t col_mask) const;
  int complement_index(int index) const;

  const std::vector<int>& degree_offsets() const;
  const std::vector<int>& degree_sizes() const;
};
```

with the following rules:

1. `row_mask` and `col_mask` encode deleted boundary rows/columns;
2. only equal-popcount pairs are admissible;
3. sectors are grouped by degree;
4. `complement_index` returns the sector
   $$
   (Y_R \setminus R,\; Y_L \setminus C),
   $$
   which is needed in exact child-parent contraction.

The degree grouping matters because:

1. it reduces unnecessary traversal;
2. degree compatibility is the first exact pruning rule;
3. it makes truncation and low-rank compression more stable when done
   block-by-block by degree.

## 3.3 Concatenated sector over multiple child interfaces

At one node, the local kernel lives on the concatenated boundary

$$
\Gamma_v^\sigma
=
\Bigl(\bigsqcup_c X_c^\sigma\Bigr)\sqcup B_v^\sigma.
$$

So the node-local sector indexer should support blockwise concatenation:

$$
S = S_1 \sqcup S_2 \sqcup \cdots \sqcup S_t \sqcup S_B.
$$

Concretely, if the child-interface boundary blocks are disjoint in block order,
then the concatenated sector masks are just shifted bit masks.

This is the clean way to define the exact local kernel on "all child
interfaces plus outgoing boundary" without inventing a second indexing scheme.

## 4. Exact Message Objects

For one spin channel $\sigma$, one subtree message should be stored as three
degrees.

## 4.1 Degree 0: overlap message

$$
M_v^{(0),\sigma}(R,C).
$$

Implementation:

```cpp
struct BoundaryMessageDegree0 {
  BoundarySectorIndexer indexer;
  Eigen::VectorXd values;  // size = sector_count
};
```

This is the exact overlap minor message.

## 4.2 Degree 1: one-electron / first-cofactor message

For exact one-electron we need the first cofactor:

$$
M_v^{(1),\sigma}(q,p;R,C),
$$

where $(q,p)$ is one inserted row/column label pair in support-space
coordinates.

Implementation should flatten support-space pair labels:

$$
\mu = (q,p),
\qquad
0 \le \mu < n_{\mathrm{sup}}^2.
$$

Then store

```cpp
struct BoundaryMessageDegree1 {
  BoundarySectorIndexer indexer;
  Eigen::MatrixXd values;  // rows = support pair index μ, cols = sector index
};
```

So:

$$
\texttt{values}(\mu,s)
=
M_v^{(1),\sigma}(q,p;R_s,C_s).
$$

This is already exact and boundary-only:

1. the exponential index is only `sector index`;
2. the support-space pair index is polynomial.

## 4.3 Degree 2: same-spin message

For exact same-spin we need the second-cofactor object:

$$
M_v^{(2),\sigma}(q_1,q_2;p_1,p_2;R,C).
$$

The right representation is an antisymmetrized pair basis.

Define packed row-pair and column-pair indices:

$$
\rho = (q_1,q_2),\quad q_1 < q_2,
$$

$$
\kappa = (p_1,p_2),\quad p_1 < p_2.
$$

Then define the combined packed pair-pair index

$$
\nu = (\rho,\kappa).
$$

The exact storage is:

```cpp
struct BoundaryMessageDegree2 {
  BoundarySectorIndexer indexer;
  Eigen::MatrixXd values;  // rows = antisymmetrized support pair-pair index ν
                           // cols = sector index
};
```

So:

$$
\texttt{values}(\nu,s)
=
M_v^{(2),\sigma}(q_1,q_2;p_1,p_2;R_s,C_s).
$$

This is the exact same-spin message.

### Why the antisymmetrized pair basis is important

Because same-spin always contracts against

$$
(q_1 p_1 \mid q_2 p_2) - (q_1 p_2 \mid q_2 p_1),
$$

so the message should live in the same antisymmetric basis. This avoids
duplicated storage and keeps sign handling localized.

## 4.4 One spin message bundle

The natural exact object is:

```cpp
struct BoundarySpinMessage {
  BoundaryMessageDegree0 degree0;
  BoundaryMessageDegree1 degree1;
  BoundaryMessageDegree2 degree2;
};
```

and a node carries one such bundle for alpha and one for beta.

## 5. Exact Node-Local Kernel

Before merging child messages, node $v$ must integrate out its own local
internal orbitals.

Define the exact node-local kernel on the full boundary

$$
\Gamma_v^\sigma
=
\Bigl(\bigsqcup_c X_c^\sigma\Bigr)\sqcup B_v^\sigma.
$$

This kernel is itself a degree-0/1/2 message:

$$
L_v^{(0),\sigma},
\qquad
L_v^{(1),\sigma},
\qquad
L_v^{(2),\sigma}.
$$

It has the same storage layout as the subtree message, except its sector
indexer is built on the concatenated boundary $\Gamma_v^\sigma$ rather than
only the outgoing boundary $B_v^\sigma$.

This gives:

```cpp
struct LocalBoundaryKernel {
  BoundarySpinMessage alpha;
  BoundarySpinMessage beta;
};
```

Construction:

1. regular square internal block:
   use exact Schur elimination to a boundary matrix;
2. singular or rectangular internal block:
   use exact rank-revealing minor builder;
3. in either case, output the degree-0/1/2 kernel on $\Gamma_v^\sigma$.

This is the object that replaces the current "full-state aggregate builder"
inside one node.

## 6. Exact Merge Formula For Degree 0

## 6.1 Sector notation

Let:

- $s_B = (R_B,C_B)$ be an outgoing sector on $B_v^\sigma$;
- $s_c = (R_c,C_c)$ be a local kernel sector on child interface $X_c^\sigma$;
- $\bar s_c$ denote its complement on that interface:
  $$
  \bar s_c = (X_{c,R}^\sigma \setminus R_c,\; X_{c,L}^\sigma \setminus C_c).
  $$

Let

$$
s_\Gamma = s_1 \sqcup \cdots \sqcup s_t \sqcup s_B
$$

be the concatenated local-kernel sector.

## 6.2 Exact overlap merge

Then the exact degree-0 outgoing message is

$$
M_v^{(0),\sigma}(s_B)
=
\sum_{s_1,\dots,s_t}
(-1)^{\Pi_v^\sigma(s_\Gamma)}
\,
L_v^{(0),\sigma}(s_\Gamma)
\prod_{c=1}^t
M_c^{(0),\sigma}(\bar s_c).
$$

This is the exact boundary-only analogue of the current frontier/root mask
merge:

1. the local kernel contributes deleted-boundary amplitudes on all child
   interfaces plus the outgoing boundary;
2. each child contributes the complementary deleted-boundary sector;
3. the sign $\Pi_v^\sigma$ is the exact block-order parity induced by the
   merge convention.

This is already the full exact overlap recurrence.

## 6.3 Why complement appears

On each child interface, a deleted boundary row or column must be resolved
exactly once when the child subtree is glued to the local kernel.

Therefore:

1. if the local kernel deletes a particular interface row, the child must keep
   it;
2. if the child deletes it, the local kernel must keep it.

That is why the merge matches each local-kernel interface sector with the
complementary child sector.

This complement rule should be explicit in code via
`BoundarySectorIndexer::complement_index(...)`.

## 7. Exact Merge Formula For Degree 1

The degree-1 message obeys the Leibniz rule: one insertion can land either in
the local kernel or in one child.

For one support-space pair label $\mu=(q,p)$,

$$
M_v^{(1),\sigma}(\mu;s_B)
=
\sum_{s_1,\dots,s_t}
(-1)^{\Pi_v^\sigma(s_\Gamma)}
\Bigg[
L_v^{(1),\sigma}(\mu;s_\Gamma)
\prod_{c=1}^t M_c^{(0),\sigma}(\bar s_c)
$$

$$
\qquad\qquad
+
\sum_{a=1}^t
L_v^{(0),\sigma}(s_\Gamma)\,
M_a^{(1),\sigma}(\mu;\bar s_a)
\prod_{c\ne a} M_c^{(0),\sigma}(\bar s_c)
\Bigg].
$$

So the exact first-cofactor / one-electron message is a sum of:

1. insertion in the local kernel;
2. insertion in exactly one child.

This is exactly what a degree-1 message should mean.

## 8. Exact Merge Formula For Degree 2

The degree-2 message follows the second-order Leibniz rule.

Let $\nu$ denote the packed antisymmetrized support pair-pair label.

Then:

$$
M_v^{(2),\sigma}(\nu;s_B)
=
\sum_{s_1,\dots,s_t}
(-1)^{\Pi_v^\sigma(s_\Gamma)}
\Big[
\mathrm{A}
+
\mathrm{B}
+
\mathrm{C}
+
\mathrm{D}
\Big],
$$

where:

### A. both insertions in the local kernel

$$
\mathrm{A}
=
L_v^{(2),\sigma}(\nu;s_\Gamma)
\prod_{c=1}^t M_c^{(0),\sigma}(\bar s_c).
$$

### B. one insertion in the local kernel and one in one child

$$
\mathrm{B}
=
\sum_{a=1}^t
\mathcal{S}\!\left[
L_v^{(1),\sigma}(\cdot;s_\Gamma),
M_a^{(1),\sigma}(\cdot;\bar s_a)
\right]_\nu
\prod_{c\ne a} M_c^{(0),\sigma}(\bar s_c).
$$

Here $\mathcal{S}$ is the exact antisymmetrized merge from two degree-1
insertions to one degree-2 pair-basis index $\nu$.

### C. both insertions in the same child

$$
\mathrm{C}
=
\sum_{a=1}^t
L_v^{(0),\sigma}(s_\Gamma)\,
M_a^{(2),\sigma}(\nu;\bar s_a)
\prod_{c\ne a} M_c^{(0),\sigma}(\bar s_c).
$$

### D. one insertion in one child and one in another child

$$
\mathrm{D}
=
\sum_{1\le a<b\le t}
L_v^{(0),\sigma}(s_\Gamma)\,
\mathcal{S}\!\left[
M_a^{(1),\sigma}(\cdot;\bar s_a),
M_b^{(1),\sigma}(\cdot;\bar s_b)
\right]_\nu
\prod_{c\ne a,b} M_c^{(0),\sigma}(\bar s_c).
$$

This is the exact same-spin merge.

The whole current same-spin recurrence should ultimately collapse to this
formula.

## 9. Opposite-Spin And Total Hamiltonian

For opposite spin, alpha and beta remain separate one-spin messages.

The exact opposite-spin energy is reconstructed only at the final contraction:

$$
H_{\mathrm{opp}}
=
\sum_{p,q,r,s}
(q p \mid s r)\,
C_\alpha^{(1)}(q,p)\,
C_\beta^{(1)}(s,r).
$$

So opposite spin does not require a mixed alpha-beta boundary tensor.

That means the message hierarchy stays:

1. one alpha degree-0/1/2 message;
2. one beta degree-0/1/2 message;
3. the spin coupling appears only in the final Hamiltonian contraction.

This is important for keeping message size under control.

## 10. Implementation Skeleton

The natural code-level split is:

```cpp
class BoundarySectorIndexer;

struct BoundaryMessageDegree0;
struct BoundaryMessageDegree1;
struct BoundaryMessageDegree2;
struct BoundarySpinMessage;
struct LocalBoundaryKernel;

LocalBoundaryKernel build_local_boundary_kernel(...);

BoundarySpinMessage merge_boundary_spin_messages(
    const LocalBoundaryKernel& kernel,
    const std::vector<const BoundarySpinMessage*>& child_messages,
    const BoundarySectorIndexer& outgoing_indexer,
    const MergeParityTables& parity_tables);

void compress_boundary_spin_message(...);  // optional approximate step
```

And the dataflow per node is:

1. gather child messages;
2. build exact local kernel on all child interfaces plus outgoing boundary;
3. contract them with complement matching and exact parity;
4. output the exact outgoing boundary message;
5. optionally compress it.

## 11. Complexity Of Exact Boundary Merge

Let

$$
N_B = N_{\mathrm{sec}}(B_{v,R}^\sigma, B_{v,L}^\sigma),
$$

and

$$
N_c = N_{\mathrm{sec}}(X_{c,R}^\sigma, X_{c,L}^\sigma).
$$

Then the naive exact degree-0 merge is

$$
O\!\left(
N_B \prod_{c=1}^t N_c
\right).
$$

This is still exponential, but now exponential only in boundary widths:

$$
N_B,\ N_c
=
2^{O(m)}.
$$

Crucially, it is no longer exponential in the full subtree state count.

For degree-1 and degree-2:

$$
T^{(1)} = O(n_{\mathrm{sup}}^2)\, T^{(0)},
$$

$$
T^{(2)} = O(n_{\mathrm{pair}}^2)\, T^{(0)},
$$

where:

$$
n_{\mathrm{pair}} = \binom{n_{\mathrm{sup}}}{2}.
$$

So the support-space operator labels add polynomial overhead, while the
exponential factor remains a boundary factor.

This is exactly the theoretical form we want.

## 12. Where `SVD` Truncation Should Be Applied

The correct place is:

> after an exact outgoing boundary message has been formed, and before that
> message is passed upward to the parent.

Not before exact subtree elimination, and not on the original full-state
tables.

## 12.1 Compression target

The message should be compressed along the **boundary sector dimension**.

For example:

### Degree 0

Treat the degree-0 message as one vector or as grade-block matrices.

### Degree 1

Treat

$$
M_v^{(1),\sigma}
\in
\mathbb{R}^{n_{\mathrm{sup}}^2 \times N_B}
$$

and compute truncated `SVD`

$$
M_v^{(1),\sigma}
\approx
U \Sigma V^{\mathsf T}.
$$

### Degree 2

Treat

$$
M_v^{(2),\sigma}
\in
\mathbb{R}^{n_{\mathrm{pair}}^2 \times N_B},
$$

or compress it blockwise by grade and spin channel.

The main point is:

1. the boundary sector dimension is the exponential dimension;
2. the support-side dimension is only polynomial;
3. so truncating the boundary rank is the right attack.

## 12.2 Grade-block truncation

Do not compress all sector degrees together by default.

Instead compress each degree block

$$
d = 0,1,\dots,m
$$

separately:

$$
M^{(r)}_d \approx U_d \Sigma_d V_d^{\mathsf T}.
$$

This is numerically safer because:

1. different deleted-set degrees can differ strongly in scale;
2. degree mixing can blur exact combinatorial meaning;
3. the degree decomposition is already exact and natural.

## 12.3 Error control

For one node $v$, choose a local tolerance $\varepsilon_v$ and require

$$
\|M_v - \tilde M_v\|_F \le \varepsilon_v.
$$

Then with downstream sensitivity factor $\Lambda_v$,

$$
|\delta E_v|
\le
\Lambda_v \varepsilon_v.
$$

So a global chemical-accuracy budget can be distributed as

$$
\sum_v \Lambda_v \varepsilon_v \le \varepsilon_{\mathrm{chem}}.
$$

This is the correct place to enforce chemical accuracy.

## 13. Recommended Development Order

The implementation order should be:

1. build `BoundarySectorIndexer`;
2. build exact degree-0 boundary message only;
3. validate exact overlap merge against current exact reference;
4. add exact degree-1 message;
5. validate one-electron and opposite-spin reconstructions;
6. add exact degree-2 message;
7. validate same-spin reconstruction;
8. only then add optional `SVD` truncation on outgoing messages.

This is the safest route because:

1. degree 0 proves the boundary basis is correct;
2. degree 1 proves the first derivative algebra is correct;
3. degree 2 is the same-spin difficulty and should be isolated.

## 14. What This Changes Relative To Current Code

The current production code does:

1. compress orientation terms into component state operators;
2. build exact full-state aggregates on root-state / leaf-state products;
3. contract those tables.

The proposed boundary-message implementation does instead:

1. eliminate node-local interior to a boundary kernel;
2. summarize each child by a boundary deleted-set message;
3. merge children and local kernel entirely in that boundary basis;
4. optionally truncate the outgoing message rank.

So the exponential object changes from:

$$
\text{unique full-state count } Q
$$

to:

$$
\text{boundary sector count } N_{\mathrm{sec}} = 2^{O(m)}.
$$

That is exactly the missing `2^n \to 2^m` step.

## 15. Bottom Line

The correct code design for a boundary-only separator is:

1. index every subtree state by deleted boundary row/column sets, not by full
   occupied lists;
2. store degree-0/1/2 messages in that basis;
3. merge exact subtree messages by complement-matched sector contraction with
   explicit parity;
4. apply `SVD` truncation only to the outgoing boundary message, because that
   is the true exponential object.

So the central implementation change is not another local micro-optimization.
It is this:

> replace full-state aggregate tables by degree-0/1/2 boundary message
> tensors and make subtree merge operate entirely in that boundary basis.

That is the implementation path that can genuinely move the separator from a
full-state exponential to a boundary-width exponential.
