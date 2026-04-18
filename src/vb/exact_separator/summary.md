# Exact Recursive Separator Hamiltonian Summary

## 1. Purpose

This note summarizes the exact rooted-tree separator algorithm that is
currently working in `src/vb/exact_separator`, and explains:

1. how overlap and Hamiltonian matrix elements are computed by the recursive
   boundary-message algorithm;
2. what has already been validated exactly;
3. why the current implementation is still slower than determinant-pair
   contraction on the present benchmark set;
4. which structural optimizations should be implemented next.

The goal is to record the **current successful exact closure** clearly, without
mixing it with earlier partial designs that are no longer the main story.

---

## 2. Current Status In One Sentence

The exact rooted-tree Hamiltonian recurrence is now closed at the
representation level:

- the specialized `one_leaf_star` production path remains exact;
- the generic rooted-tree Hamiltonian path is now also wired through an exact
  recursive typed boundary bundle;
- synthetic non-star recursive trees already match determinant-space exact
  results to numerical noise;
- but the current payload layout is still too expensive, so performance has
  not yet beaten the determinant-pair algorithm on the present star-heavy
  molecular benchmarks.

---

## 3. Rooted-Tree Problem Statement

Let the component interaction graph already be reduced to a rooted tree

$$
\mathcal{T} = (V, E),
$$

with root \(r\). Each node \(v \in V\) stores one local component with its
left and right orientation-term expansions.

For a fixed global bra/ket structure pair, the exact matrix element is

$$
\langle L | \hat{H} | R \rangle
=
H^{(1)} + H^{(2)},
$$

with

$$
S = \langle L | R \rangle
$$

for overlap,

$$
H^{(1)}
=
\sum_{\sigma \in \{\alpha,\beta\}}
\sum_{p,q}
h_{pq} C^\sigma(q,p),
$$

and

$$
H^{(2)} = H_{\alpha\alpha} + H_{\beta\beta} + H_{\alpha\beta}.
$$

The separator idea is to avoid explicit Cartesian products over all subtree
orientation terms, and instead propagate exact **boundary-only messages** along
tree edges.

---

## 4. Exact Boundary State: What Is Exponential And What Is Polynomial

For one subtree message on edge \(v \to p\), the exponential object is only the
selected parent-interface boundary mask:

$$
m =
\big(
m^\alpha_r,
m^\alpha_c,
m^\beta_r,
m^\beta_c
\big).
$$

This is the `2^m` part. Here \(m\) means separator width, not subtree size.

For each mask, the message carries polynomial payload. In the current generic
Hamiltonian implementation this polynomial payload is stored by
`HamiltonianBoundaryPayload`, which contains:

1. `overlap`: the closed scalar sector;
2. `alpha_sectors`: exact alpha active deleted-minor sectors up to degree 2,
   already weighted by closed beta overlap;
3. `beta_sectors`: the symmetric beta active sectors;
4. `mixed_sectors`: exact mixed alpha/beta degree-1 moments needed by the
   opposite-spin channel.

Mathematically this is already the exact recursive Hamiltonian bundle, even
though the current storage still uses sparse deleted-label sectors rather than
the final packed family-block layout.

---

## 5. Exact Message Definition

For one node \(v\) with parent \(p\), define the subtree \(T_v\). The exact
subtree message is indexed by the parent boundary mask \(m\) and stores the
fully summed contribution of all local term assignments inside \(T_v\).

The conceptual scalar deleted-minor form is

$$
\mathcal{M}_{v \to p}(m; K^\alpha, K^\beta)
=
\sum_{\Omega(T_v)}
w(\Omega)
(-1)^{\Pi(\Omega, m, K)}
\Delta^\alpha_{T_v}(m; K^\alpha)
\Delta^\beta_{T_v}(m; K^\beta),
$$

where:

- \(\Omega(T_v)\) runs over all local orientation-term choices in the subtree;
- \(K^\sigma\) is a deleted-minor key of rank at most 2;
- \(\Delta^\sigma_{T_v}\) is the exact deleted minor of the spin-\(\sigma\)
  frontier matrix;
- \(\Pi\) is the exact block-order sign.

The current implementation does not use this raw joint payload as the final
production closure object. Instead, it projects onto the typed Hamiltonian
bundle described above, which is sufficient to recover:

1. overlap;
2. one-electron;
3. same-spin two-electron;
4. opposite-spin two-electron;
5. total electronic Hamiltonian.

---

## 6. Leaf/Base Case

For a leaf subtree, the message is built exactly from the parent-selected
interface orbitals and the leaf local term pair.

At the one-spin level, the exact leaf separator object already exists in the
specialized one-leaf implementation:

- `build_one_leaf_spin_boundary_message(...)`
- `contract_one_leaf_spin_boundary_message(...)`

For the generic rooted-tree recursion, the leaf/base case in
`component_tree.cpp` builds the exact interface payload
directly, so the subtree interior is already fully integrated out before the
message is returned to the parent.

This is the first exact `2^n -> 2^m` step: the parent sees only boundary mask
states, not full leaf determinant states.

---

## 7. Internal Recursive Merge

Suppose node \(v\) has children

$$
\mathrm{ch}(v) = \{c_1, \dots, c_t\}.
$$

For a fixed local term pair on \(v\), the recursion proceeds as follows.

### 7.1 Child Messages

For each child \(c_i\), build the exact child message

$$
\mathcal{M}_{c_i \to v}.
$$

This is conditioned on the chosen local term pair of \(v\), because the child
boundary masks act on the occupied-orbital lists of the parent node.

### 7.2 Frontier Accumulation

Children are merged one by one into a running frontier payload

$$
F_v^{(i)}.
$$

Initialization:

$$
F_v^{(0)} = 1.
$$

Recurrence:

$$
F_v^{(i)} = F_v^{(i-1)} \star \mathcal{M}_{c_i \to v},
$$

where \(\star\) is the exact deleted-minor merge law with the correct block
sign convention.

In code, the exact Hamiltonian path uses

- `merge_hamiltonian_boundary_payloads_limited(...)`

to carry out this typed merge.

### 7.3 Local Interface Closure At The Node

After all children are merged, the remaining local orbitals of node \(v\)
become the node-local complement block. This local complement is merged with
the frontier payload to form the outgoing node-to-parent message.

Because the natural local block order is

$$
[\text{child frontier}, \text{parent interface}, \text{local remainder}],
$$

but the external child-message convention expects

$$
[\text{parent interface}, \text{child frontier}, \text{local remainder}],
$$

an additional exact interface reordering sign is required.

In code, that is handled by

- `transform_hamiltonian_interface_block_to_front(...)`.

---

## 8. Root Closure

At the root there is no outgoing parent edge. The final matrix element is
obtained by:

1. recursively building and merging all child messages into one root frontier
   payload;
2. building the root local-complement payload;
3. merging these two exact bundles once;
4. reading out overlap and Hamiltonian channels by final linear contractions.

The current internal driver is

- `evaluate_rooted_component_tree_hamiltonian_bundle_collapsed_internal(...)`.

### 8.1 Overlap

The overlap is the closed sector of the final bundle:

$$
S = \text{closed overlap sector}.
$$

### 8.2 One-Electron

The one-electron value is the contraction of the first-order sectors against
the one-electron matrix:

$$
H^{(1)}
=
\sum_{p,q}
h_{pq} C^\alpha(q,p)
+
\sum_{p,q}
h_{pq} C^\beta(q,p).
$$

In code this is read by

- `contract_total_one_electron_first_sectors(...)`.

### 8.3 Same-Spin Two-Electron

For one spin channel \(\sigma\), the same-spin contribution is the linear
contraction of exact second-order sectors:

$$
H_{\sigma\sigma}
=
\sum_{p_1 < p_2}
\sum_{q_1 < q_2}
\Big[
(q_1 p_1 \mid q_2 p_2)
-
(q_1 p_2 \mid q_2 p_1)
\Big]
\Gamma^\sigma(q_1, q_2; p_1, p_2).
$$

In code this is read by

- `contract_same_spin_second_sectors(...)`.

### 8.4 Opposite-Spin Two-Electron

Opposite-spin does **not** close on separate alpha and beta first-cofactor
marginals after subtree aggregation. It requires the exact mixed degree-1
moment:

$$
X_{\mu\nu}
=
\sum_t w_t
U_t^\alpha(\mu)
U_t^\beta(\nu).
$$

Then the opposite-spin value is a final linear contraction:

$$
H_{\alpha\beta}
=
\sum_{p,a,q,b}
(q b \mid p a)
X_{(p,a),(q,b)}.
$$

In code this is read by

- `contract_opposite_spin_first_sectors(...)`.

This mixed object is exactly why the recursive Hamiltonian closure now works
for opposite-spin, while older "sum alpha and beta separately, then multiply"
closures were not exact.

---

## 9. Why Exactness Closes Now

The exact recursive closure now works because the payload carries all channels
that are mathematically required:

1. closed overlap scalar;
2. exact one-spin degree-1 sectors, not only the balanced `(1,1)` first
   cofactor;
3. exact one-spin degree-2 sectors for same-spin;
4. exact mixed alpha/beta degree-1 moment for opposite-spin.

This is the essential difference between the current exact recursive Hamiltonian
bundle and earlier incomplete closures.

---

## 10. Current Validation Evidence

The currently relevant validation results are:

1. `validate_rooted_component_tree_exact_hamiltonian
   test_molecule/C6H6_full.xmi --max-pairs 5 --top-examples 5 --edge-threshold 0`
   gives exact agreement on all 5 tested tree pairs;
2. `debug_rooted_component_tree_star_pair` remains exact on the known
   `C6H6` star-pair example after the generic Hamiltonian closure was rewired;
3. `check_exact_separator_two_electron_component_tree` now shows exact
   agreement on:
   - `chain_rooted_subtree`
   - `branched_rooted_subtree`
   - `one_leaf_star_rooted_subtree`
   - `star_rooted_subtree`

So the generic recursive exact Hamiltonian closure is no longer only a design
target. It is already working on both star and non-star rooted-tree checks.

---

## 11. What `2^n -> 2^m` Means Here

The structural goal is:

$$
T \sim \mathrm{poly}(n)\, 2^{O(m)},
$$

where:

- \(n\) is subtree internal combinatorial size;
- \(m\) is separator width.

The current implementation already achieves this at the **representation**
level:

- the exponential state is boundary-mask enumeration;
- subtree interiors are integrated out recursively.

However, practical speed depends not only on the exponential object, but also
on the polynomial prefactor and constants.

That is exactly where the current implementation is still weak.

---

## 12. Why Performance Is Still Poor

On current closed-shell molecular benchmarks such as
`benchmark_rooted_component_tree_pairs test_molecule/C6H6_full.xmi`,
the exact separator is still much slower than determinant-pair contraction.

This does **not** mean the recursive exact closure is mathematically wrong.
It means the current implementation still pays too much overhead.

The main reasons are the following.

### 12.1 Star-Heavy Benchmark Set

For the current `C6H6_full` benchmark, the connected tree-like cases are still
all star trees.

That matters because:

1. the separator width is not especially small relative to the active support;
2. determinant-pair exact contraction is already very efficient on these small
   star cases;
3. the current separator implementation carries large exact payload overhead
   before it can benefit from better asymptotics.

So the present benchmark is a hard case for winning on constants.

### 12.2 Generic Sparse Deleted-Label Payload Storage

`HamiltonianBoundaryPayload` is exact, but it is still stored as sparse sector
lists keyed by deleted labels:

- `vector<SameSpinDeletedSectorValue>`
- `vector<HamiltonianMixedDeletedSectorValue>`

This causes:

1. repeated key concatenation;
2. canonicalization work;
3. branch-heavy rank checks;
4. poor memory locality;
5. no direct block contraction on packed family coordinates.

This is the largest remaining representation-level bottleneck.

### 12.3 Outer Mask Combinatorics

The child merge and root closure still enumerate many disjoint mask
combinations explicitly.

Even though this is boundary-only combinatorics, the constant factor is still
too high on star cases.

### 12.4 Too Many Small Exact Subdeterminant Builds

The current exact payload construction still rebuilds many local exact
deleted-minor objects separately across masks and term pairs.

This is much better than full determinant-pair fallback, but it is still not
the final best reuse pattern.

---

## 13. Exact-Preserving Optimization Roadmap

The next optimization pass should preserve exactness and focus on structure,
not cache tricks.

### 13.1 Replace Generic Deleted-Label Storage By Family-Block Storage

This is the most important next step.

Instead of storing sectors as deleted-label lists, store them directly in the
explicit boundary family basis:

$$
\{O, u, v, C, U, V, P, Q, \Gamma\}.
$$

For one spin, organize data by family delta

$$
\delta \in \{-2, -1, 0, +1, +2\}.
$$

For opposite-spin, organize by family-pair blocks

$$
(\delta_\alpha, \delta_\beta) \in \{-1,0,+1\}^2.
$$

Expected benefits:

1. no deleted-label map logic in the hot path;
2. no repeated canonicalization;
3. dense or semi-dense packed linear algebra inside each family block;
4. much better data locality;
5. direct root contraction on packed blocks.

This is the natural next representation after the current exact closure.

### 13.2 Reduce Outer Mask DP Cost

The second major bottleneck is explicit disjoint-mask recursion.

This should be reduced by:

1. grouping messages by sector family and sector index before merge;
2. using disjoint-mask subset-style DP rather than repeated generic child-loop
   branching;
3. reusing complement-sector relations where available;
4. aggregating identical local state families before full payload construction.

This is the main way to shrink the remaining large constant in the current
`2^m` frontier enumeration.

### 13.3 Stronger One-Leaf / Star Reuse

The present `one_leaf_star` path is exact but still expensive.

It should be optimized by:

1. stronger reuse across repeated one-spin state families;
2. direct rectangular-family kernels for degree-1 and degree-2 boundary blocks;
3. eliminating repeated per-mask local payload builds when the underlying
   family block is the same.

This is especially important because current molecular validations are
dominated by one-leaf stars.

### 13.4 Batch Final Contractions

Once the bundle is stored in packed family blocks, root closure should switch
from sparse-entry loops to block contractions:

1. one-electron as matrix-block contractions;
2. opposite-spin as pre-applied ERI operators on packed degree-1 blocks;
3. same-spin as packed antisymmetrized degree-2 contractions.

This can reduce both instruction overhead and repeated index arithmetic.

---

## 14. If Chemical Accuracy Is Allowed

If exact arithmetic is no longer mandatory and energy errors of roughly

$$
10^{-3} \text{ to } 10^{-4}
$$

are acceptable, then there is additional room for improvement.

The natural approximation is **low-rank truncation on the boundary message**:

1. SVD or rank-revealing compression of degree-1 family blocks;
2. low-rank compression of degree-2 family blocks;
3. truncation of mixed alpha/beta degree-1 moment blocks;
4. adaptive dropping of weak singular directions with direct energy control.

This can reduce the polynomial payload size dramatically.

Important caveat:

- this does not automatically reduce the exact boundary mask count;
- but it can reduce the cost per boundary sector so much that separator
  methods become faster on systems where exact arithmetic is still too
  expensive.

So for exact work, the priority is still exact family-block redesign.
For approximate work, low-rank boundary compression is the next major lever.

---

## 15. Bottom Line

The main conclusion is:

1. the exact recursive rooted-tree Hamiltonian closure is now working;
2. overlap, one-electron, same-spin, and opposite-spin are all represented
   exactly by the current recursive boundary bundle;
3. the remaining gap is no longer exactness, but performance;
4. the most important next optimization is to replace the current generic
   deleted-label payload storage by the explicit boundary-sector / family-block
   layout and then reduce outer mask combinatorics on top of it.

That is the path most likely to turn the current exact recursive closure into
an implementation that is not only correct, but also actually faster than the
determinant-pair algorithm.
