# One-Leaf Boundary-Message Low-Rank Truncation Design

## 1. Purpose

This note answers the next practical question for the separator project:

> if the exact value kernel is already close to algorithmically closed, but
> star-dominated cases such as `C6H6` still do not beat determinant-pair wall
> clock, what approximation should we add next if chemical accuracy is
> acceptable?

The short answer is:

1. for the current exact value kernel, the main remaining gap is no longer
   correctness but universal speed;
2. on star-dominated inputs, especially `one_leaf_star`, exact separator can
   still lose because its structural advantage is small while its bookkeeping
   constant is larger;
3. if we want speedups on almost all systems, including many star-like cases,
   then we likely need a controlled approximation;
4. the right approximation is **not** to truncate full-state tables directly,
   and not to rely on `RI` alone;
5. the right approximation is to compress the **boundary message** after exact
   local elimination and before the parent-side contraction.

This document is design-only. It does not change code.

## 2. Current Status Of The Exact Kernel

At this point, for the value kernel itself, the exact separator is close to
closed-loop:

1. overlap is exact;
2. one-electron is exact;
3. same-spin is exact;
4. opposite-spin is exact;
5. the total electronic Hamiltonian is exact.

The current implementation goal of numerical equality with the
determinant-pair reference is therefore already mostly achieved for supported
value paths. The major missing production piece is still the gradient path.

So the next question is no longer:

> can the exact separator reproduce determinant-pair values?

but rather:

> can it do so faster across the cases we actually care about?

For `C6H6`, the latest benchmark makes the current limitation very clear:

1. all connected tree-like pairs are either `one_node` or `one_leaf_star`;
2. there are no `nonstar_tree` pairs in that input;
3. therefore the newly added rooted-tree boundary-message structure has almost
   no room to pay off;
4. the dominant practical bottleneck becomes the `one_leaf_star` path.

So if we want `C6H6`-type systems to accelerate as well, we should assume that
exact arithmetic alone will not be enough.

## 3. Why Exact Separator Still Loses On Some Star Cases

For a star-dominated case, the separator does not yet enjoy a large recursive
tree advantage.

In a `one_leaf_star` pair:

1. there is only one separator edge;
2. the total support is often small;
3. determinant-pair can contract the exact value with a relatively small
   constant;
4. separator still pays for state aggregation, deleted-minor bookkeeping,
   channel organization, and sign-safe message assembly.

So even if the exact separator is mathematically in the right form, the actual
wall-clock comparison can still be

$$
T_{\mathrm{separator,exact}}
>
T_{\mathrm{detpair,exact}}
$$

for small or weakly structured star cases.

This is not primarily a bug in implementation quality. It is a consequence of
the fact that, in these cases, the exact separator has little asymptotic room
to exploit.

That is why further exact constant-factor work alone is unlikely to produce a
universal win.

## 4. What Kind Of Approximation Is Actually Worthwhile

There are many possible approximations, but most of them do not attack the
right scaling variable.

### 4.1 Why direct full-state truncation is not the right core design

Suppose we still index states by the current unique full one-spin state label

$$
q = (\text{left\_occ}, \text{right\_occ}).
$$

Then truncating or screening those full-state tables may reduce constants, but
it does not fundamentally change the fact that the internal combinatorics of
the subtree are still present in the state label.

So direct full-state truncation is not the best long-term design.

### 4.2 Why `RI` alone is not enough

`RI`/density-fitting/Cholesky reduces the local polynomial cost of one matrix
element, but determinant-pair can use the same idea too.

So `RI` alone changes

$$
Q \cdot C
\longrightarrow
Q \cdot C_{\mathrm{RI}},
$$

but it does not change the state-count variable.

For separator, the strategically important reduction is

$$
Q
\longrightarrow
N_B
\longrightarrow
r(\varepsilon),
$$

where:

1. $Q$ is a full-state count;
2. $N_B$ is the exact boundary-sector count;
3. $r(\varepsilon)$ is an approximate low-rank boundary dimension.

Only the second arrow attacks the actual exponential object.

### 4.3 The right approximation target

The correct approximation target is:

> exact boundary message first, low-rank compression second.

This is exactly the continuation of
[boundary_repr.md](./boundary_repr.md)
and
[boundary_impl.md](./boundary_impl.md).

## 5. Exact One-Leaf Boundary Form

This section restates the one-leaf case in the boundary-message language,
because that is the object we will truncate.

Consider one separator edge between a root component $R$ and a leaf component
$L$. Fix one spin channel $\sigma$.

Let the interface boundary have deleted-set sector labels

$$
\gamma = (R_\gamma, C_\gamma),
\qquad
|R_\gamma| = |C_\gamma|.
$$

Let the exact degree-$d$ one-spin message for node $v \in \{R,L\}$ be

$$
M_{v,d}^{\sigma}(:, \gamma),
$$

where:

1. the column index $\gamma$ is a boundary deleted-set sector;
2. the row index enumerates the support-space operator labels for that degree.

For the three degrees of interest:

$$
d = 0,1,2,
$$

we have:

1. degree $0$: overlap message;
2. degree $1$: one-electron / first-cofactor message;
3. degree $2$: same-spin / second-cofactor message.

Let

$$
N_B^\sigma
=
\sum_{d=0}^{m^\sigma}
\binom{m_R^\sigma}{d}
\binom{m_L^\sigma}{d}
$$

denote schematically the one-spin boundary-sector count.

The exact sector-matching between the two sides is a complement-and-sign map,
which can be represented by a sparse signed permutation matrix

$$
P^\sigma \in \mathbb{R}^{N_B^\sigma \times N_B^\sigma}.
$$

Then the exact one-spin overlap channel can be written as

$$
S^\sigma
=
\left(M_{R,0}^{\sigma}\right)^{\mathsf T}
P^\sigma
M_{L,0}^{\sigma}.
$$

The exact one-electron and same-spin channels are analogous bilinear
contractions involving degree-$1$ and degree-$2$ messages against fixed
support-space operator tensors.

The important point is structural:

> once the exact one-leaf kernel is written in this form, the only exponential
> index left is the boundary-sector dimension $N_B^\sigma$.

That is the object to compress.

## 6. Common Low-Rank Boundary Basis

The approximation should not truncate degree-$0$, degree-$1$, and degree-$2$
messages independently with unrelated bases.

That would be dangerous because:

1. overlap, one-electron, same-spin, and opposite-spin share the same exact
   boundary-sector meaning;
2. independent truncations can break cancellation patterns between channels;
3. using unrelated bases makes later gradient design much harder.

Instead, for each spin channel $\sigma$ and each one-leaf interface, define
one stacked exact snapshot matrix

$$
X_v^\sigma
=
\begin{bmatrix}
\omega_0 M_{v,0}^{\sigma} \\
\omega_1 M_{v,1}^{\sigma} \\
\omega_2 M_{v,2}^{\sigma}
\end{bmatrix}
\in
\mathbb{R}^{n_{\mathrm{stack}}^\sigma \times N_B^\sigma},
$$

where:

1. $\omega_0,\omega_1,\omega_2$ are channel-balancing weights;
2. $M_{v,0}^{\sigma}$ is the degree-$0$ overlap block;
3. $M_{v,1}^{\sigma}$ is the degree-$1$ first-cofactor block;
4. $M_{v,2}^{\sigma}$ is the degree-$2$ same-spin block.

Then compute a thin singular value decomposition

$$
X_v^\sigma
=
U_v^\sigma \Sigma_v^\sigma \left(Q_v^\sigma\right)^{\mathsf T}.
$$

Choose rank $r_v^\sigma$ and keep only the leading singular vectors:

$$
Q_{v,r}^\sigma
\in
\mathbb{R}^{N_B^\sigma \times r_v^\sigma}.
$$

This defines a common compressed boundary basis for all exact channels on that
interface and spin.

## 7. Compressed One-Leaf Messages

Project each exact degree block into the retained basis:

$$
\widehat M_{v,d}^{\sigma}
=
M_{v,d}^{\sigma} Q_{v,r}^{\sigma}.
$$

In the general two-sided case, the compressed complement/sign operator is

$$
\widehat P^\sigma
=
\left(Q_{R,r}^\sigma\right)^{\mathsf T}
P^\sigma
Q_{L,r}^\sigma.
$$

If we deliberately enforce one shared basis on both sides, then

$$
Q_{R,r}^\sigma = Q_{L,r}^\sigma = Q_r^\sigma,
$$

and the compressed sign map reduces to

$$
\widehat P^\sigma
=
\left(Q_r^\sigma\right)^{\mathsf T}
P^\sigma
Q_r^\sigma.
$$

The approximate overlap contraction becomes

$$
\widetilde S^\sigma
=
\left(\widehat M_{R,0}^{\sigma}\right)^{\mathsf T}
\widehat P^\sigma
\widehat M_{L,0}^{\sigma}.
$$

The same two-sided reduced-basis pattern applies to degree-$1$ and
degree-$2$ channels:

1. project the exact message columns into the reduced boundary basis;
2. keep the operator-label dimension exact;
3. perform all final contractions in the reduced boundary space.

So the compressed one-leaf design is:

1. exact local elimination in operator space;
2. exact assembly of boundary messages;
3. low-rank projection only along the boundary-sector dimension;
4. exact final contractions inside the reduced basis.

This keeps the chemistry-bearing support-space operator labels exact while
compressing only the true exponential dimension.

## 8. Reusable Basis Instead Of Per-Pair SVD

If we build an `SVD` from scratch for every single pair evaluation, small
cases such as `C6H6` may still fail to speed up because the compression setup
itself becomes a dominant constant.

So the design should not be:

> build one fresh low-rank basis independently for every one-leaf contraction.

Instead, the basis should be reused across repeated evaluations.

The natural reuse levels are:

1. same one-leaf component pair within one SCF iteration;
2. same component-family/interface type across all determinant-structure pairs
   in one geometry;
3. same geometry across many SCF macro-iterations;
4. optionally, neighboring geometries in a scan or optimizer run.

Mathematically, this means we should build the boundary basis from a snapshot
ensemble

$$
\mathcal X^\sigma
=
\begin{bmatrix}
X_1^\sigma \\
X_2^\sigma \\
\cdots \\
X_T^\sigma
\end{bmatrix},
$$

and compress that aggregate snapshot matrix once:

$$
\mathcal X^\sigma
\approx
\mathcal U^\sigma
\mathcal \Sigma^\sigma
\left(Q_{\mathrm{shared}}^\sigma\right)^{\mathsf T}.
$$

Then every individual one-leaf message uses the same shared basis

$$
Q_{\mathrm{shared}}^\sigma.
$$

This is important for two reasons:

1. it amortizes the basis-build cost;
2. it keeps the approximate separator numerically smooth across repeated SCF
   evaluations.

Without reuse, the small-case speed story is much weaker.

## 9. Error Control

Once we accept truncation, we should stop talking about machine-precision
equality and instead target chemical accuracy.

Let the projection error for node $v$, spin $\sigma$, and stacked snapshot
matrix be

$$
E_v^\sigma
=
X_v^\sigma
-
X_v^\sigma
Q_r^\sigma
\left(Q_r^\sigma\right)^{\mathsf T}.
$$

Choose the retained rank so that

$$
\left\|E_v^\sigma\right\|_F
\le
\varepsilon_v^\sigma
$$

or, equivalently, the discarded singular-value mass obeys

$$
\sum_{j > r_v^\sigma} \left(\sigma_j^\sigma\right)^2
\le
\left(\varepsilon_v^\sigma\right)^2.
$$

If the final energy is a bilinear contraction of the compressed messages, then
its local error obeys a sensitivity bound of the form

$$
|\delta E_v^\sigma|
\le
\Lambda_v^\sigma \varepsilon_v^\sigma,
$$

where $\Lambda_v^\sigma$ is a downstream contraction sensitivity factor.

So a practical global target such as

$$
\varepsilon_{\mathrm{chem}}
\in
[10^{-4}, 10^{-3}] \ \text{Hartree}
$$

can be enforced by distributing a local budget:

$$
\sum_{v,\sigma}
\Lambda_v^\sigma \varepsilon_v^\sigma
\le
\varepsilon_{\mathrm{chem}}.
$$

In code, the first production version does not need a perfect theoretical
estimate of every $\Lambda_v^\sigma$. A pragmatic route is:

1. use a simple relative singular-value threshold;
2. validate the resulting energy error against the exact kernel on calibration
   cases;
3. tighten or loosen thresholds until the desired error band is achieved.

## 10. Complexity Picture

### 10.1 Exact one-leaf boundary form

Once the exact one-leaf kernel is expressed in the boundary basis, the
dominant exact state dimension is

$$
N_B^\sigma
=
\sum_{d=0}^{m^\sigma}
\binom{m_R^\sigma}{d}
\binom{m_L^\sigma}{d},
$$

or, more precisely for balanced interfaces, a quantity exponential only in the
boundary width $m^\sigma$.

So the exact boundary-only one-leaf scaling is of the form

$$
T_{\mathrm{one\text{-}leaf,exact}}
\sim
\mathrm{poly}(k)\, N_B.
$$

This is already much better than a full-state exponential in the subtree
combinatorics, but it can still lose on small star cases because the constant
factor is nontrivial.

### 10.2 Truncated one-leaf form

After low-rank compression,

$$
N_B^\sigma
\longrightarrow
r^\sigma(\varepsilon),
\qquad
r^\sigma(\varepsilon) \ll N_B^\sigma
$$

when the singular spectrum decays.

Then the leading contraction cost becomes

$$
T_{\mathrm{one\text{-}leaf,trunc}}
\sim
\mathrm{poly}(k)\, r(\varepsilon)
$$

plus a lower-order reduced-basis contraction such as

$$
O(r^2)
$$

or blockwise variants.

So the intended scaling transition is

$$
2^n
\longrightarrow
2^m
\longrightarrow
r(\varepsilon).
$$

This is the approximate version of the structural reduction we actually want.

## 11. Can This Beat Determinant-Pair?

### 11.1 Cases where the answer can be yes

The truncated one-leaf design can plausibly beat determinant-pair when:

1. the exact boundary spectrum decays rapidly;
2. the retained rank is much smaller than the exact boundary-sector count;
3. the basis is reused many times across SCF iterations or many pair
   contractions;
4. the support-side operator dimensions stay exact but the exponential
   boundary dimension is genuinely compressed.

This is the regime where separator gets a structural advantage that
determinant-pair does not naturally share.

### 11.2 Cases where the answer may still be no

It is important to be precise here.

This design does **not** guarantee a speed win on every tiny system.

In particular, it may still fail to beat determinant-pair when:

1. the interface boundary is already very small;
2. the singular spectrum is flat, so the retained rank is close to full rank;
3. the case is so small that basis setup dominates;
4. the exact determinant-pair contraction is already extremely cheap.

So for `C6H6`-type small star cases, low-rank boundary compression is the
right new algorithmic direction, but it is not a mathematical guarantee of
victory on every benchmark. It is the only direction that has a realistic
chance to make separator competitive there without abandoning the separator
framework entirely.

## 12. Recommended Implementation Plan

The implementation should proceed in the following order.

### Step 1: expose exact one-leaf boundary messages

Do not start from the current full-state aggregate tables.

Instead, expose the exact one-leaf degree-$0/1/2$ boundary messages as
first-class data objects using the boundary notation already established in
the design notes.

### Step 2: build a shared basis from stacked exact snapshots

For each interface and spin channel:

1. stack degree-$0/1/2$ exact messages;
2. apply balancing weights;
3. compute a thin `SVD`;
4. retain a rank based on a local truncation tolerance.

### Step 3: project all exact channels into that basis

The approximate one-leaf production path should never truncate channels
independently.

Instead, it should use one common basis per spin/interface and store:

1. compressed degree-$0$ overlap message;
2. compressed degree-$1$ one-electron / opposite-spin message;
3. compressed degree-$2$ same-spin message;
4. compressed complement/sign map.

### Step 4: benchmark basis reuse, not only one-shot contractions

A one-shot microbenchmark is not enough.

We should separately measure:

1. exact message build time;
2. basis-build time;
3. compressed contraction time;
4. amortized cost over repeated SCF evaluations.

This matters because small star cases will only benefit if reuse is real.

### Step 5: validate energy error against the exact kernel

The exact value kernel we already have should become the oracle for the new
approximate path.

For each benchmark family, record:

1. overlap error;
2. one-electron error;
3. same-spin error;
4. opposite-spin error;
5. total energy error;
6. speedup versus exact determinant-pair;
7. speedup versus exact separator.

## 13. Bottom Line

The correct design conclusion is:

1. the exact value kernel is already close to closed-loop, so the next major
   problem is universal speed, not exactness;
2. `C6H6`-type star-dominated inputs expose the practical limit of the exact
   path, because there is little rooted-tree structure to exploit;
3. if chemical accuracy is acceptable, then the next real algorithmic step is
   one-leaf **boundary-message low-rank truncation**;
4. the truncation must act on the boundary message after exact local
   elimination, not on the original full-state tables;
5. the degree-$0/1/2$ channels should share one common compressed boundary
   basis per spin/interface;
6. the real scaling target is
   $$
   2^n
   \to
   2^m
   \to
   r(\varepsilon),
   $$
   where $r(\varepsilon)$ is a chemically controlled effective boundary rank.

So if the project goal is now:

> keep the separator framework, accept chemical accuracy instead of machine
> precision, and make even star-like systems substantially faster,

then this is the right next design step.
