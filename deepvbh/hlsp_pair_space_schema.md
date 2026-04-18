# HLSP Pair-Space Representation for DeepVBH

## 1. Motivation

The current `deepvbh` prototype uses two ordered structure channels derived from
oriented orbital-pair adjacency matrices. That encoding is useful as an
AGP-style auxiliary representation, but it is not the most natural primary
representation for HLSP structure states.

For HLSP matrix-element prediction, the design goal is:

1. represent each structure as a discrete HLSP object rather than as a general
   AGP coefficient matrix;
2. preserve the full active-space two-electron information rather than only
   reduced Coulomb/exchange summaries;
3. enforce strict structure-exchange symmetry
   $$
   H_{IJ} = H_{JI}
   $$
   without requiring two forward passes.

This note proposes a pair-space schema that satisfies those requirements.

## 2. Pair Space

Let the active-space orbitals be indexed by
$$
i,j,k,l \in \{1,\dots,n\},
$$
where $n$ is the number of active spatial orbitals.

Define the pair set
$$
\mathcal{P}_n = \{(i,j) \mid 1 \le i \le j \le n\}.
$$

The number of packed pairs is
$$
P(n) = \frac{n(n+1)}{2}.
$$

Each pair $p \in \mathcal{P}_n$ is assigned a deterministic packed index
$$
p = p(i,j), \qquad i \le j.
$$

In the repository implementation, zero-based local orbital indices use
$$
p_0(a,b)
=
\min(a,b) + \frac{\max(a,b)\,(\max(a,b)+1)}{2},
\qquad 0 \le a,b < n,
$$
which is equivalent to the one-based ordering
$$
(1,1),\ (1,2),\ (2,2),\ (1,3),\ (2,3),\ (3,3),\ \dots,\ (n,n).
$$

This pair indexing should stay consistent with the repository's packed
two-electron storage convention in
[/pool1/home/xiatao/project/xmvb-cpp/src/vb/matrices/two_electron_indexer.hpp](/pool1/home/xiatao/project/xmvb-cpp/src/vb/matrices/two_electron_indexer.hpp).

## 3. Closed-Shell HLSP Representation

### 3.1 Primary Structure Label

For a closed-shell HLSP structure $X$, define its pair-occupancy vector
$$
b_X(p) \in \{0,1\}, \qquad p \in \mathcal{P}_n.
$$

Interpretation:

1. if $p=(i,j)$ with $i<j$, then $b_X(i,j)=1$ means the structure contains the
   covalent singlet pair $(i,j)$;
2. if $p=(i,i)$, then $b_X(i,i)=1$ means the structure contains a doubly
   occupied orbital $i$.

This representation is discrete. It is not a free AGP amplitude tensor.

### 3.2 Occupation Recovery

The spatial-orbital occupation number recovered from $b_X$ is
$$
\operatorname{occ}_X(i)
=
2\,b_X(i,i)
+
\sum_{j<i} b_X(j,i)
+
\sum_{j>i} b_X(i,j).
$$

For a valid closed-shell HLSP structure, the following must hold:
$$
\operatorname{occ}_X(i) \in \{0,1,2\},
$$
and
$$
\sum_{i=1}^n \operatorname{occ}_X(i) = N_{\mathrm{active\ electrons}}.
$$

The number of occupied pairs is
$$
\sum_{p \in \mathcal{P}_n} b_X(p)
=
\frac{N_{\mathrm{active\ electrons}}}{2}.
$$

### 3.3 Matching Constraint

The vector $b_X$ is only valid if it satisfies the HLSP matching constraint:

1. every orbital participates in at most one non-diagonal pair;
2. if $b_X(i,i)=1$, then all off-diagonal pairs incident on $i$ must vanish.

Equivalently, $b_X$ must encode a valid partial matching with optional diagonal
self-pairs for double occupation.

### 3.4 Example: `1-2 3-4`

For $n=4$, choose the packed pair ordering
$$
(1,1),\ (1,2),\ (2,2),\ (1,3),\ (2,3),\ (3,3),\ (1,4),\ (2,4),\ (3,4),\ (4,4).
$$

Then the structure
$$
X = (1,2)(3,4)
$$
is encoded by
$$
b_X = [0,1,0,0,0,0,0,0,1,0].
$$

Similarly,
$$
(1,3)(2,4)
$$
is encoded by
$$
[0,0,0,1,0,0,0,1,0,0],
$$
and
$$
(1,1)(3,4)
$$
is encoded by
$$
[1,0,0,0,0,0,0,0,1,0].
$$

## 4. State Decoding and Phase Convention

Pair occupancy is complete for closed-shell HLSP only after fixing a canonical
decoding rule.

For $i<j$, define the standard singlet-pair creator
$$
g_{ij}^{\dagger}
=
a_{i\alpha}^{\dagger} a_{j\beta}^{\dagger}
-
a_{j\alpha}^{\dagger} a_{i\beta}^{\dagger}.
$$

For a doubly occupied orbital, define
$$
d_i^{\dagger}
=
a_{i\alpha}^{\dagger} a_{i\beta}^{\dagger}.
$$

For each packed pair $p$, define
$$
G_p^{\dagger}
=
\begin{cases}
g_{ij}^{\dagger}, & p=(i,j),\ i<j, \\
d_i^{\dagger}, & p=(i,i).
\end{cases}
$$

Then, after sorting all selected pairs in canonical packed order, the HLSP
state is
$$
|X\rangle
=
\prod_{p \in \mathcal{P}_n,\ b_X(p)=1}
G_p^{\dagger}\,|0\rangle.
$$

Under this convention, the tuple
$$
\bigl(b_X,\ \text{packed pair order},\ \text{fixed singlet sign rule}\bigr)
$$
uniquely specifies the closed-shell HLSP basis state.

## 5. Relation to AGP

The distinction between HLSP and AGP is not whether one uses a matrix or a
vector. The distinction is the meaning and admissible values of the object.

An AGP state is parameterized by a generally continuous pairing-amplitude
object. In pair-space language, this is a set of amplitudes
$$
f_X(p) \in \mathbb{R}
$$
or a more general coefficient tensor.

By contrast, HLSP uses a constrained discrete selection:
$$
b_X(p) \in \{0,1\},
$$
subject to the matching constraints above.

Therefore:

1. a closed-shell HLSP structure can be embedded into the larger AGP
   representation space as a special sparse point;
2. but HLSP should not be modeled as a generic free AGP coefficient matrix.

The proposed representation is therefore HLSP-native, not AGP-native.

## 6. Optional Matrix Form

If a matrix form is needed for diagnostics or visualization, define the
symmetric pair-selection matrix
$$
M_X \in \{0,1\}^{n \times n}
$$
by
$$
M_X(i,j)
=
\begin{cases}
b_X(i,j), & i \le j, \\
b_X(j,i), & i > j.
\end{cases}
$$

For the structure $(1,2)(3,4)$,
$$
M_X =
\begin{pmatrix}
0 & 1 & 0 & 0 \\
1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 1 & 0
\end{pmatrix}.
$$

This matrix is only a repacked view of the discrete HLSP structure. It is not
to be interpreted as a free AGP coefficient matrix.

## 7. Open-Shell Extension

For open-shell HLSP structures, pair occupancy alone is not complete. Extend
the representation by adding single-occupation masks:
$$
s_X^{\alpha}(i) \in \{0,1\}, \qquad
s_X^{\beta}(i) \in \{0,1\}.
$$

The recovered occupation count becomes
$$
\operatorname{occ}_X(i)
=
2\,b_X(i,i)
+
\sum_{j<i} b_X(j,i)
+
\sum_{j>i} b_X(i,j)
+
s_X^{\alpha}(i)
+
s_X^{\beta}(i).
$$

The active-electron constraint becomes
$$
2 \sum_{p \in \mathcal{P}_n} b_X(p)
+
\sum_{i=1}^n s_X^{\alpha}(i)
+
\sum_{i=1}^n s_X^{\beta}(i)
=
N_{\mathrm{active\ electrons}}.
$$

For fixed-$M_S$ open-shell calculations, the tuple
$$
\bigl(b_X,\ s_X^{\alpha},\ s_X^{\beta}\bigr)
$$
is the minimal recommended structure representation.

If the basis later distinguishes different spin-coupling patterns with the same
orbital occupancy pattern, an additional spin-coupling label may be required.

## 8. Complete Pair-Compressed ERI

The full active-space two-electron information should be represented in pair
space, not reduced to a small number of $(n \times n)$ summary channels.

Let
$$
p = p(i,j), \qquad q = p(k,l),
$$
with $i \le j$ and $k \le l$.

Define the complete pair-compressed two-electron tensor
$$
G_{pq}.
$$

This is the pair-space matrix view of the full active-space two-electron
integrals. In the repository, the corresponding packed data is already exported
as
[/pool1/home/xiatao/project/xmvb-cpp/src/tools/run_cpp_vbscf.cpp](/pool1/home/xiatao/project/xmvb-cpp/src/tools/run_cpp_vbscf.cpp).

The current prototype only uses two reduced matrices derived from this full
object:
$$
J_{ij} = (ii|jj), \qquad K_{ij} = (ij|ji),
$$
which loses substantial information. The proposed schema keeps the complete
pair-space object $G_{pq}$.

## 9. Recommended Tensor Schema

For a closed-shell structure pair $(I,J)$, define pair-space tokens indexed by
$p \in \mathcal{P}_n$.

### 9.1 Pair Features

For each pair token $p=(i,j)$, define a fixed-width local feature vector
$$
x_p =
\bigl[
h_{ij},\ S_{ij},\ \mathbf{1}_{i=j}
\bigr].
$$

Additional scalar local invariants may be added later if needed, but the full
two-electron coupling should remain in pair-space form via $G_{pq}$ rather than
being collapsed into a few handcrafted channels.

### 9.2 Structure Features

For matrix-element prediction between structures $I$ and $J$, do not feed the
two structures as ordered channels. Instead use strict exchange-invariant
structure features:
$$
b_{+}(p) = b_I(p) + b_J(p),
$$
$$
b_{-}(p) = \left| b_I(p) - b_J(p) \right|.
$$

For open-shell cases, also define
$$
s_{+}^{\alpha}(i) = s_I^{\alpha}(i) + s_J^{\alpha}(i), \qquad
s_{-}^{\alpha}(i) = \left| s_I^{\alpha}(i) - s_J^{\alpha}(i) \right|,
$$
and analogously for $\beta$ singles.

### 9.3 Pair Relation Matrix

The pair-token relation matrix is the complete pair-compressed ERI:
$$
R_{pq} = G_{pq}.
$$

This matrix should be treated as the pair-token interaction object in the model
rather than as a fixed set of input channels.

### 9.4 Masks

Because $P(n)$ depends on the active-space size, batching should use:

1. `pair_mask[p]` for valid pair tokens;
2. optional `orbital_mask[i]` if orbital-level auxiliary features remain;
3. padding to bucket-specific or global maximum pair counts.

## 10. Strict Exchange Symmetry

If the model input depends only on
$$
\bigl(x_p,\ b_{+}(p),\ b_{-}(p),\ R_{pq}\bigr),
$$
then the full input is invariant under exchanging $I$ and $J$.

Therefore the predicted matrix element automatically satisfies
$$
\hat{H}_{IJ} = \hat{H}_{JI}
$$
without requiring
$$
\frac{1}{2}\bigl(f(I,J) + f(J,I)\bigr)
$$
and without requiring two inference passes.

This is the preferred symmetry mechanism for HLSP matrix-element prediction.

## 11. Practical Model Direction

The recommended next architecture is a pair-token model:

1. token index: packed pair $p=(i,j),\ i \le j$;
2. token features: local one-body and overlap features, plus symmetric
   structure-pair features;
3. token-token relation: complete pair-compressed ERI $G_{pq}$;
4. masks: `pair_mask`;
5. readout pooling: masked mean over valid pair tokens so the pooled state does
   not scale trivially with $P(n)$.

Compared with the current orbital-space adjacency design, this approach:

1. preserves the full active-space two-electron information;
2. uses an HLSP-native structure representation;
3. enforces strict structure-exchange symmetry by construction;
4. handles variable active-space sizes through pair-token masking.

## 12. Implementation Order

The recommended implementation sequence is:

1. replace ordered structure adjacency channels with HLSP pair occupancy;
2. decode `packed_active_two_electron_integrals` into pair-space matrix form
   $G_{pq}$;
3. change the model from orbital-space tokens to pair-space tokens;
4. extend to open-shell structure tensors after the closed-shell path is stable.

This sequence minimizes conceptual drift while directly addressing the main
representation and symmetry issues in the current prototype.
