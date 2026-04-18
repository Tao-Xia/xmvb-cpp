# Exact Separator Boundary-Only Message Representation

## 1. Purpose

This note answers the question that now matters most:

> in the current exact separator implementation, what is the actual
> exponential object, and what exact representation is needed to change the
> scaling from a full-state `2^n`-type behavior to a boundary-width
> `2^m`-type behavior?

Here:

- `n` means the internal combinatorial size of a subtree or component pair;
- `m` means the separator or boundary width seen by the rest of the tree.

The main conclusion is:

1. the current exact one-leaf implementation is **not** yet boundary-only;
2. its exponential object is still the number of unique full one-spin states;
3. to obtain true separator-width scaling, the subtree must be represented by
   a message indexed only by **boundary deleted sets**;
4. if exact boundary messages are still too large, then controlled
   `SVD`/rank-truncation on those boundary messages is the natural
   chemical-accuracy route.

This document is derivation-only. It does not change code.

## 2. What The Current Exponential Object Really Is

The current one-leaf exact path no longer enumerates raw term quadruples and
no longer enumerates separator masks in the hot path. That part is already
fixed.

But the current state key is still the **full occupied list**:

- [leaf_coefficient_operator.hpp](./leaf_coefficient_operator.hpp)

Specifically:

$$
\texttt{SpinPairStateKey} = (\texttt{left\_occ}, \texttt{right\_occ}).
$$

So the current one-spin aggregate tables are built over the Cartesian product
of root states and leaf states:

- [spin_state_aggregate.cpp](./spin_state_aggregate.cpp)

Schematically,

$$
Q_\sigma = A_R^\sigma A_L^\sigma,
$$

where:

- $A_R^\sigma$ is the number of unique root one-spin states;
- $A_L^\sigma$ is the number of unique leaf one-spin states.

The current one-leaf one-electron and two-electron paths then build one direct
aggregate for every such full state:

- [one_electron.cpp](./one_electron.cpp)
- [two_electron_opposite_spin.cpp](./two_electron_opposite_spin.cpp)

So the current exact separator is of the form

$$
T_{\mathrm{current}}
\sim
Q \cdot \mathrm{poly}(k),
$$

not

$$
T_{\mathrm{ideal}}
\sim
\mathrm{poly}(n)\, 2^{O(m)}.
$$

This is why the current implementation can still fail to beat exact
determinant-pair on cases where the unique full-state count does not collapse.

## 3. Why `Q` Is Not Yet A Boundary Message

The key defect is structural:

1. the current state key remembers the full occupied-orbital pattern of the
   subtree piece;
2. that means the interior of the subtree is still explicit in the state
   label;
3. so the state count can still scale with the full subtree combinatorics.

To obtain genuine separator-width scaling, the interior must be **integrated
out**, leaving only a message on the separator.

That message must depend only on:

1. which boundary rows are left open;
2. which boundary columns are left open;
3. for Hamiltonian work, low-order derivative payloads attached to those same
   boundary open sets.

That is the missing step.

## 4. Subtree Partition And Notation

Fix one spin channel $\sigma$ and one subtree $v$.

Partition the occupied-orbital labels seen by that subtree into:

- internal left orbitals $I_L^\sigma(v)$;
- internal right orbitals $I_R^\sigma(v)$;
- boundary left orbitals $B_L^\sigma(v)$;
- boundary right orbitals $B_R^\sigma(v)$.

Here "boundary" means the orbitals whose overlap couplings cross from subtree
$v$ to the rest of the tree.

Order rows and columns in subtree block order:

$$
\text{rows} = [I_R^\sigma,\, B_R^\sigma],
\qquad
\text{cols} = [I_L^\sigma,\, B_L^\sigma].
$$

Then the one-spin overlap block has the partition

$$
S_v^\sigma
=
\begin{pmatrix}
A_v^\sigma & U_v^\sigma \\
V_v^\sigma & W_v^\sigma
\end{pmatrix},
$$

with:

$$
\begin{aligned}
A_v^\sigma &\in \mathbb{R}^{|I_R^\sigma|\times |I_L^\sigma|}, \\
U_v^\sigma &\in \mathbb{R}^{|I_R^\sigma|\times |B_L^\sigma|}, \\
V_v^\sigma &\in \mathbb{R}^{|B_R^\sigma|\times |I_L^\sigma|}, \\
W_v^\sigma &\in \mathbb{R}^{|B_R^\sigma|\times |B_L^\sigma|}.
\end{aligned}
$$

Define:

$$
m_R^\sigma = |B_R^\sigma|,
\qquad
m_L^\sigma = |B_L^\sigma|.
$$

In the balanced one-spin case we write simply

$$
m^\sigma = m_R^\sigma = m_L^\sigma.
$$

## 5. Exact Boundary-Only Message For Overlap

## 5.1 Boundary deleted-set definition

For one subtree $v$, one spin $\sigma$, and subsets

$$
R \subseteq B_R^\sigma,
\qquad
C \subseteq B_L^\sigma,
$$

with

$$
|R| = |C|,
$$

define the exact boundary minor message

$$
M_v^\sigma(R,C)
:=
\det
S_v^\sigma\!\left[
I_R^\sigma \cup (B_R^\sigma \setminus R),\,
I_L^\sigma \cup (B_L^\sigma \setminus C)
\right].
$$

This is the determinant of the subtree overlap block after deleting the
boundary rows in $R$ and the boundary columns in $C$.

Equivalently, in block notation,

$$
M_v^\sigma(R,C)
=
\det
\begin{pmatrix}
A_v^\sigma & U_v^\sigma[:,\, \bar C] \\
V_v^\sigma[\bar R,\, :] & W_v^\sigma[\bar R,\, \bar C]
\end{pmatrix},
$$

where

$$
\bar R = B_R^\sigma \setminus R,
\qquad
\bar C = B_L^\sigma \setminus C.
$$

This is already the exact transformation from a full-state representation to a
boundary-only representation:

1. all interior orbitals are integrated out inside the determinant;
2. the resulting label depends only on boundary deleted sets $R,C$.

## 5.2 Why this is the right exact message

The collection

$$
\mathcal{M}_v^\sigma
=
\left\{
M_v^\sigma(R,C)
\;:\;
R \subseteq B_R^\sigma,\,
C \subseteq B_L^\sigma,\,
|R|=|C|
\right\}
$$

is the exact one-spin boundary message of subtree $v$.

It is exact because the subtree interior enters only through the determinant of
the internal-plus-surviving-boundary block. No information about the interior
is lost.

The environment never needs the full internal determinant list again. It only
needs these boundary deleted-set amplitudes.

## 5.3 Exact contraction with the complement

Let $\bar v$ denote the complementary environment side sharing the same
boundary.

With a consistent block-order sign convention, the full determinant can be
written as a bilinear contraction over boundary deleted sets:

$$
\det S_{\mathrm{full}}^\sigma

=
\sum_{\substack{
R \subseteq B_R^\sigma,\,
C \subseteq B_L^\sigma \\
|R|=|C|
}}
(-1)^{\chi_v^\sigma(R,C)}
\,
M_v^\sigma(R,C)\,
M_{\bar v}^\sigma(B_R^\sigma \setminus R,\,
B_L^\sigma \setminus C).
$$

The exact sign $\chi_v^\sigma(R,C)$ depends only on the chosen block-order
convention. The important structural point is not the sign, but the form:

1. subtree $v$ contributes only its boundary message;
2. the complement contributes only its boundary message;
3. the full determinant is their exact boundary contraction.

So once the subtree has been summarized by $\mathcal{M}_v^\sigma$, its
internal combinatorics is gone from the recursion.

## 5.4 Exact regular-case reduction to a Schur boundary kernel

If the internal block $A_v^\sigma$ is square and nonsingular, then define the
exact boundary Schur kernel

$$
K_v^\sigma
:=
W_v^\sigma - V_v^\sigma (A_v^\sigma)^{-1} U_v^\sigma.
$$

Then for every boundary deleted-set pair $(R,C)$,

$$
M_v^\sigma(R,C)
=
\det(A_v^\sigma)\,
\det K_v^\sigma[\bar R,\bar C].
$$

This is immediate from the Schur complement formula applied to the deleted
submatrix.

So in the regular case, the exact boundary message can be computed in two
stages:

1. eliminate the full internal block once into $\det(A_v^\sigma)$ and
   $K_v^\sigma$;
2. store only the deleted minors of the boundary matrix $K_v^\sigma$.

This is the clean algebraic statement of "internal elimination".

## 5.5 Singular or rectangular case

If $A_v^\sigma$ is singular or rectangular, the Schur formula above does not
apply directly.

But the exact boundary-only message still exists and is still given by the
definition

$$
M_v^\sigma(R,C)
=
\det
S_v^\sigma\!\left[
I_R^\sigma \cup (B_R^\sigma \setminus R),\,
I_L^\sigma \cup (B_L^\sigma \setminus C)
\right].
$$

So the exact object is the same. Only the preferred construction changes:

1. regular square internal block: Schur complement;
2. singular or rectangular internal block: exact minor/rank-revealing local
   solver;
3. both give the same boundary message tensor.

This is important because it separates the two issues:

1. **representation**: boundary deleted-set tensor;
2. **construction**: Schur, `SVD`, rank-revealing factorization, or exact
   local minor logic.

The representation is the real step from `2^n`-type states to `2^m`-type
states. The construction method is secondary.

## 6. Exact Boundary Message Size And The `2^m` Statement

For one spin, the exact number of boundary deleted-set amplitudes is

$$
N_{\mathrm{msg}}^\sigma(m_R^\sigma,m_L^\sigma)
=
\sum_{d=0}^{\min(m_R^\sigma,m_L^\sigma)}
\binom{m_R^\sigma}{d}\binom{m_L^\sigma}{d}.
$$

In the balanced case

$$
m_R^\sigma = m_L^\sigma = m^\sigma,
$$

we get

$$
N_{\mathrm{msg}}^\sigma(m^\sigma)
=
\sum_{d=0}^{m^\sigma}\binom{m^\sigma}{d}^2
=
\binom{2m^\sigma}{m^\sigma}
=
\Theta\!\left(\frac{4^{m^\sigma}}{\sqrt{m^\sigma}}\right).
$$

So the exact one-spin boundary message is not literally `2^{m^\sigma}`; it is
more precisely

$$
2^{\,2m^\sigma + o(m^\sigma)}.
$$

But this is still the right separator-width statement:

1. the exponential depends only on boundary width;
2. it does **not** depend on the full subtree combinatorics.

If one uses $m_{\mathrm{tot}}$ to denote the full boundary bit count across
left/right selectors, then the scaling is exactly of the informal
`2^{O(m_{\mathrm{tot}})}` type.

That is the correct mathematical version of the desired `2^n \to 2^m`.

## 7. How This Extends To One-Electron And Two-Electron

Overlap alone is not enough. We also need exact one-electron and exact
same-spin two-electron data.

The clean way to derive those is to introduce a source matrix.

## 7.1 Source-matrix generating determinant

For one spin channel, introduce a source matrix $J^\sigma$ of the same shape as
the subtree overlap block and define

$$
Z_v^\sigma(J^\sigma)
:=
\det\!\bigl(S_v^\sigma + J^\sigma\bigr).
$$

Then the exact first and second cofactors are the derivatives at zero:

$$
C_v^{(1),\sigma}(q,p)
=
\left.
\frac{\partial Z_v^\sigma}{\partial J^\sigma_{q p}}
\right|_{J^\sigma=0},
$$

$$
C_v^{(2),\sigma}(q_1,q_2;p_1,p_2)
=
\left.
\frac{\partial^2 Z_v^\sigma}
{\partial J^\sigma_{q_1 p_1}\,\partial J^\sigma_{q_2 p_2}}
\right|_{J^\sigma=0}.
$$

These are, up to the standard sign convention, the exact first- and
second-cofactor tensors.

## 7.2 Boundary-only derivative messages

Now apply the same deleted-boundary construction before setting $J^\sigma=0$.

Define

$$
M_v^\sigma(R,C;J^\sigma)
:=
\det\!\Bigl(
S_v^\sigma[
I_R^\sigma \cup (B_R^\sigma\setminus R),\,
I_L^\sigma \cup (B_L^\sigma\setminus C)
]

+
J^\sigma[
I_R^\sigma \cup (B_R^\sigma\setminus R),\,
I_L^\sigma \cup (B_L^\sigma\setminus C)
]
\Bigr).
$$

Then define the boundary derivative messages:

$$
M_v^{(0),\sigma}(R,C)
:=
M_v^\sigma(R,C;0),
$$

$$
M_v^{(1),\sigma}(q,p;R,C)
:=
\left.
\frac{\partial M_v^\sigma(R,C;J^\sigma)}
{\partial J^\sigma_{q p}}
\right|_{J^\sigma=0},
$$

$$
M_v^{(2),\sigma}(q_1,q_2;p_1,p_2;R,C)
:=
\left.
\frac{\partial^2 M_v^\sigma(R,C;J^\sigma)}
{\partial J^\sigma_{q_1 p_1}\,\partial J^\sigma_{q_2 p_2}}
\right|_{J^\sigma=0}.
$$

These are exact subtree messages indexed only by:

1. boundary deleted sets $(R,C)$;
2. at most two inserted row/column labels for operator response.

This is the exact boundary-only analogue of the current overlap /
first-cofactor / second-cofactor payloads.

## 7.3 Reconstruction of one-electron and same-spin channels

Given the exact first-cofactor tensor,

$$
H_v^{(1),\sigma}
=
\sum_{p,q}
h_{q p}\,
C_v^{(1),\sigma}(q,p).
$$

Given the exact second-cofactor tensor,

$$
H_{v,\mathrm{same}}^{(2),\sigma}
=
\frac{1}{2}
\sum_{p_1,p_2,q_1,q_2}
\bigl[
(q_1 p_1 \mid q_2 p_2)
-
(q_1 p_2 \mid q_2 p_1)
\bigr]
C_v^{(2),\sigma}(q_1,q_2;p_1,p_2).
$$

So exact one-electron and same-spin two-electron can be reconstructed from the
same boundary message hierarchy:

1. degree `0` message for overlap;
2. degree `1` message for one-electron;
3. degree `2` message for same-spin.

The decisive point is that the **exponential index remains the boundary
deleted-set index**, while the operator insertion labels contribute only
polynomial overhead.

## 8. What Is Still Missing In The Current Code

The current code already has:

1. compressed component coefficient operators;
2. direct full-state aggregates;
3. exact overlap / first-cofactor / same-spin scalars per full state.

But it still does **not** have:

1. a state basis indexed only by separator boundary deleted sets;
2. an exact merge routine that contracts those boundary messages across the
   tree;
3. a degree-0/1/2 boundary message algebra replacing the current full-state
   tables.

So the gap is not "more clever local factorization".

The gap is:

> replace the full-state key by a boundary deleted-set message key.

Only that can move the exponential variable from the subtree combinatorics to
the separator width.

## 9. Approximate Chemical-Accuracy Route

If exact boundary messages are still too large, then the right approximation is
not to truncate full-state tables directly, but to truncate the **boundary
message**.

That is the approximation which attacks the correct exponential object.

## 9.1 Exact message first, then low-rank compression

After forming the exact boundary message tensor, flatten it into a matrix or a
small collection of matrices. For example, for one spin and one message degree:

$$
M_v \in \mathbb{R}^{N_{\mathrm{left}}\times N_{\mathrm{right}}},
$$

where the row and column indices enumerate ordered boundary sector labels.

Then apply truncated `SVD`:

$$
M_v
\approx
U_v \Sigma_v V_v^{\mathsf T},
$$

retaining rank $r_v(\varepsilon)$ such that the discarded norm is below a
local tolerance:

$$
\|M_v - M_{v,r}\|_F \le \varepsilon_v
\qquad
\text{or}
\qquad
\|M_v - M_{v,r}\|_2 \le \varepsilon_v.
$$

Then the effective message complexity becomes

$$
r_v(\varepsilon)
\ll
N_{\mathrm{msg}}^\sigma(m_R^\sigma,m_L^\sigma)
$$

when the singular spectrum decays rapidly.

This is the natural route to chemical-accuracy separator compression.

## 9.2 Why this is better than only using `RI`

`RI`/`DF`/Cholesky reduces the local polynomial cost of one matrix element.
Exact determinant-pair can use those techniques too.

So by itself, `RI` does **not** change the decisive comparison:

$$
P \cdot C
\longrightarrow
P \cdot C_{\mathrm{RI}},
\qquad
Q \cdot C
\longrightarrow
Q \cdot C_{\mathrm{RI}}.
$$

If $Q$ is still large, both methods simply become faster together.

Boundary-message compression is different. It attacks the state count itself:

$$
Q
\longrightarrow
r(\varepsilon,m).
$$

That is exactly the kind of reduction needed to approach true
separator-width scaling.

## 9.3 Error-budget view

If the final energy is obtained by a bilinear contraction

$$
E = \langle L, M \rangle,
$$

then the truncation error obeys

$$
|E - \tilde E|
\le
\|L\|_F \,\|M - \tilde M\|_F.
$$

For a whole tree, this becomes a sum of local truncation contributions
amplified by downstream contraction norms:

$$
|\delta E|
\le
\sum_v \Lambda_v \,\varepsilon_v,
$$

where $\Lambda_v$ is the sensitivity of the remaining tree contraction to the
message at node $v$.

This is the right framework for targeting chemical accuracy such as:

$$
10^{-3}
\;\text{to}\;
10^{-4}\ \text{Hartree}.
$$

## 10. Practical Consequence

If the goal is truly to move from `2^n`-type behavior to `2^m`-type behavior,
the implementation priority should be:

1. define the exact boundary deleted-set message basis;
2. express overlap, first-cofactor, and second-cofactor as degree-0/1/2
   boundary messages;
3. implement exact subtree merge as contraction of those boundary messages;
4. only then add `SVD`/rank truncation for chemical-accuracy compression.

By contrast:

1. rectangular family exact reuse;
2. Schur-complement local speedups;
3. same-spin per-state constant-factor improvements;
4. `RI` alone;

are all worthwhile, but they do **not** by themselves change the exponential
variable from subtree size to separator width.

## 11. Bottom Line

The mathematically correct answer is:

1. the current exact separator still indexes states by full occupied lists, so
   its exponential object is still the unique full-state count $Q$;
2. the exact boundary-only replacement is the deleted-boundary message tensor
   $M_v^\sigma(R,C)$ and its degree-1/2 derivative extensions;
3. this replaces a subtree state space depending on interior combinatorics by a
   state space depending only on boundary width;
4. the exact one-spin message dimension is
   $$
   \sum_d \binom{m_R}{d}\binom{m_L}{d},
   $$
   which is exponential only in separator width;
5. if that exact boundary message is still too large, then truncated `SVD` on
   the boundary message, not on the original full-state table, is the right
   chemical-accuracy approximation.

So the single most important missing algorithmic step is:

> replace full-state tables by exact boundary deleted-set messages.

That is the step that can finally make the separator scale exponentially in
boundary width instead of exponentially in full subtree combinatorics.
