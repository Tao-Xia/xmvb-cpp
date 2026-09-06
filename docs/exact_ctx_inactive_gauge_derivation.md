# Exact-Context Inactive-Space Gauge Derivation

## Goal

This note derives the inactive-projector and active-auxiliary formulas used by
the current `exact_ctx` / TNHVP implementation, and clarifies which
orthogonality relations are valid in the current chart.

The main question is whether we can exploit

1. orthogonality inside the inactive block,
2. orthogonality between inactive and active orbitals,
3. nonorthogonality only inside the active block,

to obtain a more fundamental Hessian-vector-product speedup.

The conclusion is:

1. The current implementation already enforces inactive-active orthogonality in
   the auxiliary gauge.
2. The current implementation does **not** enforce block
   $S$-orthonormality inside the inactive physical block.
3. Therefore the simple projector formula
   $$
   dP_i = dQ_i Q_i^T + Q_i dQ_i^T
   $$
   is **not** directly valid in the current variables.
4. That formula becomes valid only after a genuine gauge / chart change to an
   $S$-orthonormal inactive block.

## Code Mapping

Current code paths:

- [`active_space_orbital_preparer.cpp`](/pool1/home/xiatao/project/xmvb-cpp/src/vb/orbital/active_space_orbital_preparer.cpp)
  builds the accepted-point inactive density and active auxiliary orbitals.
- [`exact_orbital_second_order_operator.cpp`](/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/exact_orbital_second_order_operator.cpp)
  differentiates those objects in the exact-context HVP.

The key formulas correspond to:

- inactive overlap inverse:
  [`active_space_orbital_preparer.cpp#L343`](/pool1/home/xiatao/project/xmvb-cpp/src/vb/orbital/active_space_orbital_preparer.cpp#L343)
- inactive density:
  [`active_space_orbital_preparer.cpp#L354`](/pool1/home/xiatao/project/xmvb-cpp/src/vb/orbital/active_space_orbital_preparer.cpp#L354)
- active auxiliary orbitals:
  [`active_space_orbital_preparer.cpp#L370`](/pool1/home/xiatao/project/xmvb-cpp/src/vb/orbital/active_space_orbital_preparer.cpp#L370)
- directional inactive projector derivative:
  [`exact_orbital_second_order_operator.cpp#L1570`](/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/exact_orbital_second_order_operator.cpp#L1570)
- fixed-upstream pullback through the same objects:
  [`exact_orbital_second_order_operator.cpp#L1757`](/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/exact_orbital_second_order_operator.cpp#L1757)

## Notation

Let:

- $S \in \mathbb{R}^{N \times N}$ be the AO overlap matrix.
- $C_i \in \mathbb{R}^{N \times n_i}$ be the inactive physical orbitals.
- $C_a \in \mathbb{R}^{N \times n_a}$ be the active physical orbitals.

In the current code, each orbital is individually normalized, but the inactive
block is not block-orthonormalized. Therefore define
$$
M = C_i^T S C_i \in \mathbb{R}^{n_i \times n_i},
$$
which is symmetric positive definite in a valid accepted point, but in general
$$
M \neq I.
$$

Define the inactive dual / auxiliary block
$$
A_i = C_i M^{-1},
$$
and the inactive density / projector matrix
$$
P_i = A_i C_i^T = C_i M^{-1} C_i^T.
$$

Also define the metric projector
$$
\Pi_i = P_i S.
$$

The current active auxiliary orbitals are
$$
T_a = (I - P_i S) C_a = (I - \Pi_i) C_a.
$$

This is exactly what the preparer builds in the code.

## What Orthogonality Is Actually True

### Inactive-active orthogonality in the auxiliary gauge

We have
$$
C_i^T S T_a
= C_i^T S (I - P_i S) C_a
= C_i^T S C_a - C_i^T S C_i M^{-1} C_i^T S C_a
= C_i^T S C_a - M M^{-1} C_i^T S C_a
= 0.
$$

Equivalently,
$$
A_i^T S T_a
= M^{-1} C_i^T S T_a
= 0.
$$

So the current implementation already gives exact inactive-active
$S$-orthogonality between the inactive span and the active auxiliary span.

### Inactive block is not internally orthonormal

Inside the inactive block,
$$
C_i^T S C_i = M,
$$
and in general
$$
M \neq I.
$$

This is the main obstruction to using the simplest orthonormal-projector
derivative formulas in the current chart.

### Active block remains nonorthogonal

The active auxiliary overlap is
$$
K_a = T_a^T S T_a,
$$
and in general
$$
K_a \neq I.
$$

This matches the intended mixed gauge:

- inactive-active overlap vanishes,
- inactive-inactive uses $M$,
- active-active remains nonorthogonal.

## Projector Algebra in the Current Chart

Because
$$
P_i = C_i M^{-1} C_i^T,
$$
we have
$$
\Pi_i^2 = (P_i S)(P_i S)
= C_i M^{-1} C_i^T S C_i M^{-1} C_i^T S
= C_i M^{-1} M M^{-1} C_i^T S
= P_i S
= \Pi_i.
$$

So $\Pi_i$ is the metric projector onto the inactive span.

Also define the complementary projector
$$
Q_i = I - P_i S = I - \Pi_i.
$$

Then
$$
Q_i C_i = 0,
$$
and
$$
T_a = Q_i C_a.
$$

## First Derivative of the Inactive Projector

Since
$$
M = C_i^T S C_i,
$$
its directional derivative is
$$
dM = dC_i^T S C_i + C_i^T S dC_i.
$$

Therefore
$$
dM^{-1} = -M^{-1} (dM) M^{-1}.
$$

For the inactive dual block,
$$
A_i = C_i M^{-1},
$$
so
$$
dA_i = dC_i M^{-1} + C_i dM^{-1}.
$$

Substituting $dM^{-1}$ gives
$$
dA_i
= dC_i M^{-1}
- C_i M^{-1}(dC_i^T S C_i + C_i^T S dC_i)M^{-1}.
$$

This can be rearranged into a useful mixed-gauge form:
$$
dA_i
= (I - P_i S) dC_i M^{-1} - A_i dC_i^T S A_i
= Q_i dC_i M^{-1} - A_i dC_i^T S A_i.
$$

Now differentiate
$$
P_i = A_i C_i^T.
$$

Then
$$
dP_i = dA_i C_i^T + A_i dC_i^T.
$$

Substituting the expression for $dA_i$ yields
$$
dP_i
= Q_i dC_i A_i^T + A_i dC_i^T Q_i^T.
$$

Since $Q_i^T = I - S P_i$, this is equivalent to the longer formula
$$
dP_i
= dC_i M^{-1} C_i^T
+ C_i M^{-1} dC_i^T
- C_i M^{-1}(dC_i^T S C_i + C_i^T S dC_i)M^{-1} C_i^T.
$$

This is the exact current-chart formula implemented in the HVP code.

### Structural consequence

Even though $dP_i$ is dense as an AO matrix, it has rank at most $2n_i$:
$$
dP_i = Q_i dC_i A_i^T + A_i dC_i^T Q_i^T.
$$

This is why low-rank treatment of $dP_i$ is mathematically justified in the
current implementation, even without changing gauge.

## First Derivative of the Active Auxiliary Orbitals

Recall
$$
T_a = Q_i C_a = (I - P_i S) C_a.
$$

Differentiate:
$$
dT_a = -dP_i S C_a + Q_i dC_a.
$$

Using the low-rank form of $dP_i$,
$$
dP_i S C_a
= Q_i dC_i (A_i^T S C_a) + A_i dC_i^T Q_i^T S C_a.
$$

Because
$$
Q_i^T S C_a = S Q_i C_a = S T_a,
$$
we get
$$
dT_a
= Q_i dC_a - Q_i dC_i (A_i^T S C_a) - A_i dC_i^T S T_a.
$$

This formula is important:

- the inactive-active coupling is not gone,
- it appears through the small matrix
  $$
  B_{ia} = A_i^T S C_a = M^{-1} C_i^T S C_a,
  $$
- and through the metric overlap with the active auxiliary block,
  $$
  S T_a.
  $$

So the current chart does exploit inactive-active orthogonality in the
auxiliary gauge, but it still carries the inactive block metric $M$.

## Fixed-Upstream Pullback in the Current Chart

The exact-context fixed-upstream pullback differentiates the orbital
backpropagation map while holding the accepted-point upstream adjoints fixed.

Introduce accepted-point upstream matrices:

- $G_A \in \mathbb{R}^{N \times n_a}$ for the active auxiliary block,
- $G_P \in \mathbb{R}^{N \times N}$ for the inactive density block.

The active contribution is
$$
L_A = \langle G_A, T_a \rangle.
$$

Using
$$
T_a = Q_i C_a,
$$
the pullback to the physical active block is
$$
\bar C_a = Q_i^T G_A = (I - S P_i) G_A.
$$

This matches the code path
$$
\bar C_a = G_A - S P_i G_A.
$$

Now define the effective inactive upstream matrix
$$
G_{\mathrm{fix}} = G_P - G_A (S C_a)^T.
$$

This is exactly the combination formed in the code before differentiating the
inactive projector contribution.

Only the symmetric part of $G_{\mathrm{fix}}$ contributes through $P_i$:
$$
G_{\mathrm{sym}} = G_{\mathrm{fix}} + G_{\mathrm{fix}}^T.
$$

Also define the small inactive-space contraction
$$
X = C_i^T G_{\mathrm{fix}} C_i,
$$
and
$$
\nabla_M = -M^{-1} X M^{-1}.
$$

Then the current-chart inactive pullback is
$$
\bar C_i = G_{\mathrm{sym}} C_i M^{-1} + S C_i (\nabla_M + \nabla_M^T).
$$

This is the formula implemented in the fixed-upstream code.

## Directional Derivative of the Fixed-Upstream Pullback

For a Hessian-vector product we need the directional derivative of the
pullback itself.

The fixed upstreams $G_A$ and $G_P$ are held constant, but
$$
G_{\mathrm{fix}} = G_P - G_A (S C_a)^T
$$
still depends on the current orbital direction through $C_a$.

Therefore
$$
dG_{\mathrm{fix}} = -G_A (S dC_a)^T.
$$

Its symmetric part is
$$
dG_{\mathrm{sym}} = dG_{\mathrm{fix}} + dG_{\mathrm{fix}}^T.
$$

For the small inactive contraction,
$$
X = C_i^T G_{\mathrm{fix}} C_i,
$$
so
$$
dX =
dC_i^T G_{\mathrm{fix}} C_i
+ C_i^T dG_{\mathrm{fix}} C_i
+ C_i^T G_{\mathrm{fix}} dC_i.
$$

Since
$$
\nabla_M = -M^{-1} X M^{-1},
$$
we obtain
$$
d\nabla_M =
-dM^{-1} X M^{-1}
-M^{-1} dX M^{-1}
-M^{-1} X dM^{-1}.
$$

Finally,
$$
\bar C_i = G_{\mathrm{sym}} C_i M^{-1} + S C_i (\nabla_M + \nabla_M^T),
$$
so
$$
d\bar C_i
= dG_{\mathrm{sym}} C_i M^{-1}
+ G_{\mathrm{sym}} dC_i M^{-1}
+ G_{\mathrm{sym}} C_i dM^{-1}
+ S dC_i (\nabla_M + \nabla_M^T)
+ S C_i (d\nabla_M + d\nabla_M^T).
$$

This is the formula behind the long `fixed_upstream` directional code. The
computationally expensive parts are exactly the terms involving

- $dM^{-1}$,
- $dX$,
- $d\nabla_M$.

Those terms disappear only if we move to a chart where the inactive block is
represented as an explicitly $S$-orthonormal frame.

## Why the Simple Orthonormal-Projector Formula Does Not Apply Yet

Suppose we define an $S$-orthonormal inactive frame
$$
Q_i = C_i R,
$$
where $R$ is chosen so that
$$
Q_i^T S Q_i = I.
$$

For example, one may take $R = M^{-1/2}$.

Then the projector becomes
$$
P_i = Q_i Q_i^T.
$$

In an orthonormal gauge with admissible tangent $\Delta_i$ satisfying
$$
Q_i^T S \Delta_i + \Delta_i^T S Q_i = 0,
$$
the projector derivative simplifies to
$$
dP_i = \Delta_i Q_i^T + Q_i \Delta_i^T.
$$

This is the attractive formula that would remove explicit differentiation of
$M^{-1}$.

However, this formula is valid only if the working variables are the
orthonormal frame $Q_i$ and the tangent is taken in that gauge.

In the current implementation, the working variables are the physical inactive
coefficients $C_i$. Then
$$
dQ_i = dC_i R + C_i dR,
$$
and the extra term
$$
C_i dR
$$
contains exactly the information that is currently carried by
$$
dM^{-1}
$$
or equivalently by differentiating the inactive block metric.

So one cannot simply replace the present formulas by the orthonormal-gauge
formula while keeping the current chart unchanged.

## Practical Consequence for Optimization

### What is valid in the current chart

The current chart legitimately supports:

1. low-rank treatment of $dP_i$,
2. reuse of accepted-point contractions involving $P_i$, $A_i$, and $S C_a$,
3. low-rank evaluation of products like $dP_i (S C_a)$ and $S dP_i G$.

These are local algebraic optimizations that do not change the variables.

### What requires a real gauge change

A more fundamental reduction of work requires changing the inactive block
parameterization so that the optimization variables themselves satisfy
$$
Q_i^T S Q_i = I.
$$

Only then do the expensive chain terms involving $M^{-1}$ become removable at
the formulation level rather than merely hidden inside cached contractions.

That means a true next step is not "drop the $M^{-1}$ terms in the current
code", but rather:

1. define an inactive-block orthonormal chart,
2. rewrite the accepted-point orbital preparer in that chart,
3. rewrite the nonredundant TN tangent map and pullback in the same chart,
4. re-derive the exact-context HVP formulas in those variables.

## Bottom Line

The user intuition is directionally right:

- there is real structure coming from inactive-active orthogonality,
- and a properly orthonormal inactive gauge should admit simpler formulas.

But in the **current** accepted-point variables, the mathematically correct
projector is
$$
P_i = C_i (C_i^T S C_i)^{-1} C_i^T,
$$
not
$$
P_i = Q_i Q_i^T
$$
with a direction-independent orthonormal $Q_i$.

Therefore:

1. the current code cannot safely drop the differentiation of the inactive
   metric,
2. the clean simplification requires a genuine gauge / chart redesign,
3. any future "fundamental optimization" should begin with that derivation,
   not with local code rewrites.

## Orthornormal Inactive Gauge

This section derives the corresponding formulas after a genuine inactive-gauge
change.

Assume we replace the physical inactive block $C_i$ by an
$S$-orthonormal frame
$$
Q_i \in \mathbb{R}^{N \times n_i},
\qquad
Q_i^T S Q_i = I.
$$

One concrete realization is
$$
Q_i = C_i M^{-1/2},
\qquad
M = C_i^T S C_i,
$$
but once the new chart is adopted, $Q_i$ is treated as the primary variable,
not as a derived quantity that must be rebuilt by differentiating $M^{-1/2}$ on
every Hessian-vector product.

Define the orthonormal-gauge inactive density
$$
P_i = Q_i Q_i^T,
$$
and the metric projector
$$
\Pi_i = P_i S = Q_i Q_i^T S.
$$

The complementary projector is
$$
R_i = I - \Pi_i = I - Q_i Q_i^T S.
$$

The active auxiliary block is still
$$
T_a = R_i C_a.
$$

### Basic orthogonality relations

Because $Q_i^T S Q_i = I$, we immediately have
$$
\Pi_i^2 = \Pi_i,
$$
and
$$
Q_i^T S T_a = Q_i^T S (I - Q_i Q_i^T S) C_a = 0.
$$

So the orthonormal gauge preserves the desired inactive-active orthogonality,
but now the inactive block itself is also block $S$-orthonormal.

## Tangent Structure in the Orthornormal Gauge

Differentiating
$$
Q_i^T S Q_i = I
$$
gives the generalized Stiefel tangent constraint
$$
Q_i^T S \Delta_i + \Delta_i^T S Q_i = 0,
$$
where
$$
\Delta_i = dQ_i.
$$

If we additionally choose the horizontal gauge
$$
Q_i^T S \Delta_i = 0,
$$
then automatically
$$
\Delta_i^T S Q_i = 0.
$$

This horizontal gauge is attractive because it removes pure inactive-inactive
gauge rotations from the TN chart and keeps only physically relevant directions.

## Projector Derivative in the Orthornormal Gauge

Now
$$
P_i = Q_i Q_i^T,
$$
so its derivative is simply
$$
dP_i = \Delta_i Q_i^T + Q_i \Delta_i^T.
$$

Therefore
$$
d\Pi_i = dP_i \, S
= \Delta_i Q_i^T S + Q_i \Delta_i^T S.
$$

This is the main simplification compared with the current chart:

- there is no inactive metric
  $$
  M = C_i^T S C_i,
  $$
- no inverse
  $$
  M^{-1},
  $$
- and therefore no
  $$
  dM^{-1}.
  $$

The entire inactive-projector derivative is encoded directly by the tangent
frame $\Delta_i$.

## Active Auxiliary Derivative in the Orthornormal Gauge

Recall
$$
T_a = R_i C_a = (I - Q_i Q_i^T S) C_a.
$$

Differentiate:
$$
dT_a = -d\Pi_i \, C_a + R_i dC_a.
$$

Substituting the projector derivative gives
$$
dT_a
= R_i dC_a - \Delta_i (Q_i^T S C_a) - Q_i (\Delta_i^T S C_a).
$$

Define the small inactive-active overlap block
$$
B_{ia} = Q_i^T S C_a \in \mathbb{R}^{n_i \times n_a}.
$$

Then
$$
dT_a
= R_i dC_a - \Delta_i B_{ia} - Q_i (\Delta_i^T S C_a).
$$

If we decompose the physical active block as
$$
C_a = Q_i B_{ia} + T_a,
$$
then under the horizontal gauge $Q_i^T S \Delta_i = 0$ we get
$$
\Delta_i^T S C_a = \Delta_i^T S T_a,
$$
so the formula simplifies to
$$
dT_a
= R_i dC_a - \Delta_i B_{ia} - Q_i (\Delta_i^T S T_a).
$$

This is the orthonormal-gauge analogue of the current-chart formula
$$
dT_a
= Q_i dC_a - Q_i dC_i (A_i^T S C_a) - A_i dC_i^T S T_a,
$$
with the mixed-gauge dual block $A_i = C_i M^{-1}$ replaced by the
orthonormal frame $Q_i$.

### Consequence

Inactive-active coupling does not disappear entirely: the small block
$$
B_{ia} = Q_i^T S C_a
$$
still appears.

What disappears is the whole chain coming from differentiating the inactive
metric inverse.

## Fixed-Upstream Pullback in the Orthornormal Gauge

Keep the same accepted-point upstream quantities:

- $G_A \in \mathbb{R}^{N \times n_a}$ for the active auxiliary contribution,
- $G_P \in \mathbb{R}^{N \times N}$ for the inactive-density contribution.

As before,
$$
T_a = R_i C_a,
$$
so
$$
\langle G_A, dT_a \rangle
= \langle R_i^T G_A, dC_a \rangle - \langle G_A, d\Pi_i C_a \rangle.
$$

Therefore the pullback to the physical active block is
$$
\bar C_a = R_i^T G_A = (I - S Q_i Q_i^T) G_A.
$$

This is the direct orthonormal-gauge analogue of the current formula
$$
\bar C_a = (I - S P_i) G_A.
$$

For the inactive block, combine the $P_i$- and $T_a$-dependent terms into
$$
G_{\mathrm{fix}} = G_P - G_A (S C_a)^T.
$$

Then the $Q_i$-dependent scalar contribution is
$$
L_i = \langle G_{\mathrm{fix}}, Q_i Q_i^T \rangle.
$$

Only the symmetric part contributes:
$$
G_{\mathrm{sym}} = G_{\mathrm{fix}} + G_{\mathrm{fix}}^T.
$$

Differentiating
$$
L_i = \langle G_{\mathrm{fix}}, Q_i Q_i^T \rangle
$$
gives
$$
dL_i = \langle G_{\mathrm{sym}} Q_i, \Delta_i \rangle.
$$

So the coefficient-level pullback to the inactive orthonormal frame is simply
$$
\bar Q_i = G_{\mathrm{sym}} Q_i.
$$

If the optimizer uses a horizontal generalized-Stiefel tangent chart, an
additional tangent-space projection of $\bar Q_i$ is needed. But crucially, the
raw pullback contains no $M^{-1}$, no small inactive-block linear solve, and no
metric-inverse derivative chain.

### Important Caveat for Option A $(Q_i, C_a)$

The simplification
$$
\bar Q_i = G_{\mathrm{sym}} Q_i
$$
is only valid when the physical active block is already $S$-orthogonal to the
inactive frame, namely
$$
B_{ia} = Q_i^T S C_a = 0.
$$

In the current exact-context implementation we use Option A, where the primary
variables are $(Q_i, C_a)$ and
$$
T_a = (I - Q_i Q_i^T S) C_a.
$$
For this chart,
$$
dT_a
= R_i dC_a - \Delta_i B_{ia} - Q_i (\Delta_i^T S C_a),
$$
so the inactive pullback still contains the inactive-active coupling carried by
$B_{ia}$.

Therefore the orthonormal inactive shortcut is not valid for the fixed-upstream
pullback at a general open-shell accepted point. In practice, even when
$$
Q_i^T S Q_i = I,
$$
the implementation must keep the full mixed-gauge fixed-upstream formula unless
it can additionally guarantee
$$
Q_i^T S C_a = 0.
$$

### Raw-Coefficient Chart Actually Used by the Current Code

The current optimizer and exact-context HVP do not parameterize the inactive
block by a constrained generalized-Stiefel tangent variable $dQ_i$.

Instead, after the accepted point is orthonormalized, the code still
differentiates with respect to unconstrained raw inactive coefficients
$$
C_i,
\qquad
C_i^T S C_i = I \quad \text{at the accepted point only}.
$$

So the correct exact raw-chart identity is
$$
A_i = C_i (C_i^T S C_i)^{-1} = C_i
$$
at the accepted point, but
$$
dA_i \neq dC_i.
$$

Because
$$
M_i = C_i^T S C_i,
$$
we still have
$$
dM_i = dC_i^T S C_i + C_i^T S dC_i,
$$
and therefore
$$
dA_i = dC_i - C_i dM_i.
$$

For the fixed-upstream inactive pullback, if we define
$$
G_{\mathrm{sym}} = G_{\mathrm{fix}} + G_{\mathrm{fix}}^T,
$$
then the exact raw-coefficient pullback at an accepted point with
$$
C_i^T S C_i = I
$$
is
$$
\bar C_i
= G_{\mathrm{sym}} C_i
- S C_i \left(C_i^T G_{\mathrm{sym}} C_i\right).
$$

Its directional derivative is
$$
d\bar C_i
= dG_{\mathrm{sym}} C_i
+ G_{\mathrm{sym}} dC_i
- S dC_i K_i
- S C_i dK_i,
$$
with
$$
K_i = C_i^T G_{\mathrm{sym}} C_i
$$
and
$$
dK_i
= dC_i^T G_{\mathrm{sym}} C_i
+ C_i^T dG_{\mathrm{sym}} C_i
+ C_i^T G_{\mathrm{sym}} dC_i.
$$

These formulas are still much cheaper than the fully general mixed-gauge
expressions because they avoid any inactive inverse or inverse-derivative work,
but they remain exact for the raw coefficient chart used by the current code.

## Directional Derivative of the Fixed-Upstream Pullback in the Orthornormal Gauge

For the exact-context HVP, the upstreams remain fixed but
$$
G_{\mathrm{fix}} = G_P - G_A (S C_a)^T
$$
still depends on the current active orbitals.

Hence
$$
dG_{\mathrm{fix}} = -G_A (S dC_a)^T,
$$
and
$$
dG_{\mathrm{sym}} = -G_A (S dC_a)^T - S dC_a G_A^T.
$$

### Active pullback derivative

Since
$$
\bar C_a = (I - S Q_i Q_i^T) G_A,
$$
we get
$$
d\bar C_a
= -S (\Delta_i Q_i^T + Q_i \Delta_i^T) G_A.
$$

Equivalently,
$$
d\bar C_a
= -S \Delta_i (Q_i^T G_A) - S Q_i (\Delta_i^T G_A).
$$

This depends only on AO-by-inactive and inactive-by-active contractions.

### Inactive pullback derivative

Since
$$
\bar Q_i = G_{\mathrm{sym}} Q_i,
$$
we get
$$
d\bar Q_i = dG_{\mathrm{sym}} Q_i + G_{\mathrm{sym}} \Delta_i.
$$

Substituting $dG_{\mathrm{sym}}$ gives
$$
d\bar Q_i
= -G_A (S dC_a)^T Q_i - S dC_a G_A^T Q_i + G_{\mathrm{sym}} \Delta_i.
$$

So
$$
d\bar Q_i
= G_{\mathrm{sym}} \Delta_i
- G_A (dC_a^T S Q_i)
- S dC_a (G_A^T Q_i).
$$

This is the key simplification:

- no inactive overlap matrix,
- no inverse,
- no $dM^{-1}$,
- no intermediate
  $$
  X = C_i^T G_{\mathrm{fix}} C_i,
  $$
- no
  $$
  \nabla_M = -M^{-1} X M^{-1},
  $$
- no
  $$
  d\nabla_M.
  $$

All remaining terms are direct contractions of accepted-point matrices with the
current tangent directions.

## Comparison of Algebraic Complexity

Let:

- $N$ be the AO dimension,
- $n_i$ be the number of inactive doubly occupied orbitals,
- $n_a$ be the number of active orbitals.

The exact asymptotic wall time of the full `exact_ctx` HVP also includes AO
integral contraction work and outer-response work. Those parts are not changed
by the inactive-gauge redesign. The comparison below isolates only the
inactive-projector / active-auxiliary algebra.

### Current chart

Per directional evaluation, the current chart requires:

1. building
   $$
   M = C_i^T S C_i,
   $$
   which costs roughly
   $$
   O(N n_i^2)
   $$
   after reusing $S C_i$;
2. forming and factorizing $M$, plus applying $M^{-1}$, which costs roughly
   $$
   O(n_i^3);
   $$
3. building
   $$
   dM = dC_i^T S C_i + C_i^T S dC_i,
   $$
   again at
   $$
   O(N n_i^2);
   $$
4. forming
   $$
   dM^{-1} = -M^{-1} dM M^{-1},
   $$
   which costs another
   $$
   O(n_i^3);
   $$
5. in fixed-upstream, building
   $$
   X = C_i^T G_{\mathrm{fix}} C_i,
   \qquad
   dX,
   \qquad
   \nabla_M,
   \qquad
   d\nabla_M,
   $$
   which adds more small inactive-block cubic work and several AO-by-inactive
   contractions.

So the current formulation carries an intrinsic extra cost of order
$$
O(N n_i^2 + n_i^3)
$$
in both orbital preparation and fixed-upstream differentiation, in addition to
the unavoidable dense AO contractions.

### Orthornormal gauge

In the orthonormal gauge:

1. there is no directional build of
   $$
   M,
   $$
2. no application or differentiation of
   $$
   M^{-1},
   $$
3. no
   $$
   X, \nabla_M, d\nabla_M
   $$
   chain in the fixed-upstream derivative.

What remains are contractions like
$$
Q_i^T S C_a,
\qquad
S T_a,
\qquad
Q_i^T G_A,
\qquad
G_{\mathrm{sym}} Q_i,
\qquad
G_{\mathrm{sym}} \Delta_i.
$$

These cost roughly
$$
O(N n_i n_a)
\quad\text{or}\quad
O(N^2 n_i),
$$
depending on whether the upstream matrix is AO-dense.

Therefore the orthonormal gauge removes the entire inactive-block-specific
overhead
$$
O(N n_i^2 + n_i^3),
$$
but it does **not** remove the AO-dense contractions already present in the
exact-context operator.

## What This Means for Expected Speedup

The orthonormal inactive gauge is a real formulation-level simplification, but
its wall-time impact depends on which stage dominates:

1. If `fixed_upstream` and orbital-preparation differentiation are a large
   fraction of the HVP cost, the gauge change can produce a meaningful speedup.
2. If AO effective one-electron contractions or outer response dominate, the
   net wall-time gain may be modest even though the inactive formulas become
   much cleaner.

So the orthonormal gauge is mathematically the right place to look for a
fundamental optimization, but it is not automatically a guarantee of large
end-to-end speedup. The expected benefit is "remove an entire class of
inactive-block chain rules", not "remove all expensive exact-context work".

## Implication for the Next Derivation Step

If we want to pursue this direction seriously, the next derivation should not
start from the current `C_i` variables anymore. It should define a full
accepted-point chart in terms of

1. an orthonormal inactive frame $Q_i$,
2. a compatible active variable, either the physical $C_a$ or the already
   projected active auxiliary block $T_a$,
3. a tangent map that preserves
   $$
   Q_i^T S Q_i = I
   $$
   by construction.

Only after that is fixed does it make sense to redesign the TNHVP
implementation around the simplified formulas above.

## Toward a Compatible Nonredundant TN Chart

This section makes the previous discussion concrete. The goal is to describe a
nonredundant optimizer chart that is compatible with the orthonormal inactive
gauge, rather than leaving the gauge change at the level of projector algebra
alone.

For reference, the current nonredundant chart is described in
[docs/nonredundant_orbital_parameterization.md](/pool1/home/xiatao/project/xmvb-cpp/docs/nonredundant_orbital_parameterization.md)
and is implemented by block-local inactive-active plus occupied-virtual
rotation coordinates in
[`nonredundant_orbital_space.cpp`](/pool1/home/xiatao/project/xmvb-cpp/src/vb/orbital/nonredundant_orbital_space.cpp).

At present the code builds a block rotation generator with:

- inactive-active amplitudes,
- occupied-virtual amplitudes,

and applies those directions to the current occupied block in a chart built on
top of the physical occupied orbitals. See in particular
[`nonredundant_orbital_space.cpp#L484`](/pool1/home/xiatao/project/xmvb-cpp/src/vb/orbital/nonredundant_orbital_space.cpp#L484)
and
[`nonredundant_orbital_space.cpp#L1046`](/pool1/home/xiatao/project/xmvb-cpp/src/vb/orbital/nonredundant_orbital_space.cpp#L1046).

The question is how to redesign those reduced coordinates after replacing the
inactive physical block $C_i$ by an $S$-orthonormal frame $Q_i$.

## Desired Variables

The cleanest formulation is to separate:

1. the inactive occupied subspace, represented by an orthonormal frame
   $$
   Q_i \in \mathbb{R}^{N \times n_i},
   \qquad
   Q_i^T S Q_i = I,
   $$
2. the active information, represented either by the physical active block
   $$
   C_a \in \mathbb{R}^{N \times n_a},
   $$
   or by the active auxiliary block
   $$
   T_a = (I - Q_i Q_i^T S) C_a.
   $$

These two choices lead to two distinct chart designs.

## Option A: Use $(Q_i, C_a)$ as Primary Variables

This option is the closest to the current implementation.

### State variables

The accepted-point state is:
$$
(Q_i, C_a),
\qquad
Q_i^T S Q_i = I.
$$

The active auxiliary block remains a derived quantity:
$$
T_a = (I - Q_i Q_i^T S) C_a.
$$

### Tangent variables

Let the inactive tangent be
$$
\Delta_i = dQ_i,
$$
subject to the generalized Stiefel constraint
$$
Q_i^T S \Delta_i + \Delta_i^T S Q_i = 0.
$$

Choose the horizontal gauge
$$
Q_i^T S \Delta_i = 0.
$$

Then $\Delta_i$ lives entirely in the $S$-orthogonal complement of the
inactive span. Introduce an $S$-orthonormal complement basis
$$
W = [W_a \; W_v],
$$
where:

- $W_a$ spans the occupied-active complement directions,
- $W_v$ spans the occupied-virtual complement directions.

Then every horizontal inactive tangent can be written as
$$
\Delta_i = W K_i,
$$
with
$$
K_i \in \mathbb{R}^{(n_a + n_v) \times n_i}.
$$

Split
$$
K_i =
\begin{bmatrix}
K_{ai} \\
K_{vi}
\end{bmatrix},
$$
where:

- $K_{ai} \in \mathbb{R}^{n_a \times n_i}$ encodes inactive-active rotations,
- $K_{vi} \in \mathbb{R}^{n_v \times n_i}$ encodes inactive-virtual rotations.

This is the orthonormal-gauge analogue of the current inactive-active and
inactive-virtual reduced coordinates.

For the active physical block, use an unconstrained tangent
$$
\Delta_a = dC_a.
$$

Then decompose $\Delta_a$ relative to the accepted-point decomposition
$$
\mathbb{R}^N = \operatorname{span}(Q_i) \oplus \operatorname{span}(T_a) \oplus \operatorname{span}(W_v),
$$
as
$$
\Delta_a = Q_i U_{ia} + T_a U_{aa} + W_v U_{va}.
$$

Here:

- $U_{ia} \in \mathbb{R}^{n_i \times n_a}$ changes inactive-active mixing
  through the active coordinates,
- $U_{aa} \in \mathbb{R}^{n_a \times n_a}$ changes the active internal gauge,
- $U_{va} \in \mathbb{R}^{n_v \times n_a}$ gives active-virtual directions.

### Nonredundant reduction

Not all of these blocks are physically independent:

1. $K_{ai}$ and $U_{ia}$ both affect inactive-active coupling,
2. $U_{aa}$ is mostly active internal gauge / reparameterization,
3. only certain combinations map to distinct first-order changes in
   $(Q_i, T_a)$.

The natural reduced choice is:

1. keep $K_{ai}$ as the unique inactive-active block,
2. keep $K_{vi}$ as the inactive-virtual block,
3. keep only the projected active-virtual block $U_{va}$,
4. eliminate $U_{aa}$ as internal active gauge,
5. eliminate $U_{ia}$ because inactive-active coupling is already represented
   by $K_{ai}$.

So the reduced coordinate vector remains structurally similar to the current
TN chart:
$$
\xi =
\begin{bmatrix}
\xi_{ia} \\
\xi_{iv} \\
\xi_{av}
\end{bmatrix},
$$
but now these amplitudes live in a chart attached to $(Q_i, C_a)$ rather than
to the legacy physical occupied block.

### Advantage

This option preserves the current conceptual split:

- inactive block defines the projector,
- active block is still the physical object from which $T_a$ is built.

So the orbital preparation pipeline changes less conceptually.

### Disadvantage

The derivative formulas still contain $dC_a$ explicitly, and the active
internal gauge must be managed carefully to avoid redundancy.

## Option B: Use $(Q_i, T_a)$ as Primary Variables

This option is more radical but mathematically cleaner for `exact_ctx`.

### State variables

Take
$$
(Q_i, T_a)
$$
as the primary orbital variables, with constraints
$$
Q_i^T S Q_i = I,
\qquad
Q_i^T S T_a = 0.
$$

The active overlap matrix
$$
K_a = T_a^T S T_a
$$
is allowed to remain nonorthogonal.

### Tangent variables

Let
$$
\Delta_i = dQ_i,
\qquad
\Theta_a = dT_a.
$$

The constraints imply
$$
Q_i^T S \Delta_i + \Delta_i^T S Q_i = 0,
$$
and
$$
\Delta_i^T S T_a + Q_i^T S \Theta_a = 0.
$$

Under the horizontal inactive gauge
$$
Q_i^T S \Delta_i = 0,
$$
the second constraint becomes
$$
Q_i^T S \Theta_a = -\Delta_i^T S T_a.
$$

So $\Theta_a$ is not completely free: its inactive component is determined by
$\Delta_i$.

Decompose
$$
\Theta_a = Q_i Y_{ia} + T_a Y_{aa} + W_v Y_{va}.
$$

Then the orthogonality constraint fixes
$$
Y_{ia} = -\Delta_i^T S T_a.
$$

Therefore the independent active tangent variables reduce to:

- an internal active block $Y_{aa}$,
- an active-virtual block $Y_{va}$.

Again, the internal active block is mostly gauge-like. So the clean reduced
choice is:

1. inactive-active directions carried by $\Delta_i$,
2. inactive-virtual directions carried by $\Delta_i$,
3. active-virtual directions carried by the $W_v$ component of $\Theta_a$,
4. drop active-internal gauge directions from the reduced TN coordinates.

### Why this option is attractive

All exact-context formulas are already naturally written in terms of:

- the inactive projector,
- the active auxiliary block,
- active-space matrices built from that auxiliary block.

So if $T_a$ itself becomes a primary variable, the orbital-preparation map is
shorter:

1. build projector from $Q_i$,
2. treat $T_a$ directly as the active block entering active-space integrals,
3. avoid repeatedly differentiating the map
   $$
   T_a = (I - Q_i Q_i^T S) C_a.
   $$

### Main drawback

The rest of the codebase currently still interprets active orbitals partly
through the physical block $C_a$. So adopting $(Q_i, T_a)$ as primary variables
would require a larger API and data-layout redesign than Option A.

## Which Option Is Better?

For a first serious redesign, Option A is probably lower risk:

1. it preserves the current preparer concept of "physical active block" plus
   "derived active auxiliary block",
2. it changes mainly the inactive representation,
3. it should slot into the existing runtime / packed sparse chart with less
   invasive change.

For a more ambitious exact-context-focused redesign, Option B is probably the
cleaner endpoint:

1. the HVP formulas become closest to the actual maintained objects,
2. the active-space integral builder can consume $T_a$ directly,
3. some current orbital-preparation derivative logic disappears entirely.

So a practical roadmap is:

1. derive and prototype Option A first,
2. use that to validate the orthonormal inactive gauge numerically,
3. only then consider whether promoting $T_a$ itself to a primary variable is
   worth the larger code change.

## Proposed Reduced Coordinates for a New TN Chart

A concrete reduced coordinate layout compatible with Option A is:
$$
\xi =
\begin{bmatrix}
\operatorname{vec}(K_{ai}) \\
\operatorname{vec}(K_{vi}) \\
\operatorname{vec}(U_{va})
\end{bmatrix},
$$
with sizes
$$
\dim(\xi_{ai}) = n_a n_i,
\qquad
\dim(\xi_{iv}) = n_v n_i,
\qquad
\dim(\xi_{av}) = n_v n_a.
$$

This is noteworthy because it matches the current nonredundant counting:

- inactive-active,
- inactive-virtual,
- active-virtual.

So the **dimension** of the reduced chart need not change. What changes is the
meaning of those coordinates and the basis vectors they generate.

In particular:

1. the current chart builds those directions from physical occupied orbitals and
   virtual complements of the occupied span;
2. the new chart should build them from the orthonormal inactive frame $Q_i$,
   the active block representation, and an $S$-orthonormal virtual complement.

This is encouraging: the optimizer-level trust-region and Krylov machinery may
not need a conceptual rewrite, only a new block-basis construction.

## Recommended Next Mathematical Step

Before any code change, the next derivation should write the explicit
first-order map
$$
\xi \mapsto (\Delta_i, \Delta_a)
$$
for Option A, block by block, in the same level of detail that the current
code uses for its local direction scatter.

Concretely, that means deriving:

1. how $\xi_{ai}$ maps to the horizontal inactive tangent $\Delta_i$,
2. how $\xi_{iv}$ maps to the inactive-virtual tangent,
3. how $\xi_{av}$ maps to the active tangent $\Delta_a$,
4. how those tangents are scattered back into the packed sparse orbital chart,
5. whether the resulting chart can still use the existing block-local union AO
   domains.

Only after that derivation is written down does it make sense to start
designing a replacement for
[`NonredundantOrbitalSpace`](/pool1/home/xiatao/project/xmvb-cpp/src/vb/orbital/nonredundant_orbital_space.cpp).

## Block-Local First-Order Map for Option A

This section pushes Option A down to the same level of detail as the current
block-local reduced chart.

The current `NonredundantOrbitalSpace` works block by block on a union AO
domain, builds a block-local occupied basis and an $S$-orthonormal virtual
complement, then stores reduced coordinates for:

1. inactive-active rotations,
2. occupied-virtual rotations.

See
[`nonredundant_orbital_space.cpp#L617`](/pool1/home/xiatao/project/xmvb-cpp/src/vb/orbital/nonredundant_orbital_space.cpp#L617),
[`nonredundant_orbital_space.cpp#L679`](/pool1/home/xiatao/project/xmvb-cpp/src/vb/orbital/nonredundant_orbital_space.cpp#L679),
and
[`nonredundant_orbital_space.cpp#L1024`](/pool1/home/xiatao/project/xmvb-cpp/src/vb/orbital/nonredundant_orbital_space.cpp#L1024).

The goal here is to derive the analogue for the orthonormal inactive gauge.

## Block-Local State

Consider one orbital block $b$ with union AO domain $\Omega_b$, exactly as in
the current implementation.

Restrict all quantities to that AO domain. Let:

- $Q_b \in \mathbb{R}^{|\Omega_b| \times n_{i,b}}$ be the inactive orthonormal
  frame in the block,
- $C_{a,b} \in \mathbb{R}^{|\Omega_b| \times n_{a,b}}$ be the physical active
  block in the same AO domain,
- $T_{a,b} = (I - Q_b Q_b^T S_b) C_{a,b}$ be the active auxiliary block,
- $V_b \in \mathbb{R}^{|\Omega_b| \times n_{v,b}}$ be an $S_b$-orthonormal
  virtual complement,

where $S_b$ is the AO overlap matrix on $\Omega_b$.

The desired block-local orthogonality relations are
$$
Q_b^T S_b Q_b = I,
\qquad
Q_b^T S_b T_{a,b} = 0,
\qquad
V_b^T S_b V_b = I,
$$
and $V_b$ spans the complement of the occupied block in the AO metric.

For a clean first derivation we additionally assume that the active auxiliary
columns are linearly independent and define
$$
K_{a,b} = T_{a,b}^T S_b T_{a,b}.
$$

Unlike the inactive block, $K_{a,b}$ is not assumed to be the identity.

## Block-Local Tangent Basis

We want a first-order map from reduced block coefficients to the pair
$$
(\Delta Q_b, \Delta C_{a,b}).
$$

Following Option A, use three reduced blocks:
$$
\xi_b =
\begin{bmatrix}
\xi_{ai,b} \\
\xi_{iv,b} \\
\xi_{av,b}
\end{bmatrix}.
$$

Equivalently, reshape them into matrices:
$$
K_{ai,b} \in \mathbb{R}^{n_{a,b} \times n_{i,b}},
\qquad
K_{vi,b} \in \mathbb{R}^{n_{v,b} \times n_{i,b}},
\qquad
U_{va,b} \in \mathbb{R}^{n_{v,b} \times n_{a,b}}.
$$

### Why use $T_{a,b}$ instead of $C_{a,b}$ inside the basis?

Even though Option A still keeps $C_a$ as a primary variable globally, the
block-local tangent basis is cleaner if inactive-active directions are
constructed against the active auxiliary block $T_{a,b}$ rather than the raw
physical block $C_{a,b}$.

The reason is that
$$
Q_b^T S_b T_{a,b} = 0
$$
already holds, so the coupled inactive-active first-order updates preserve the
horizontal inactive gauge more transparently.

Therefore define the block-local tangent basis using:

- $Q_b$ for inactive columns,
- $T_{a,b}$ for active-occupied columns,
- $V_b$ for virtual columns.

This is the analogue of the current code using:

- occupied block columns,
- virtual block columns.

## Inactive-Active Reduced Directions

For one inactive index $i$ and one active index $a$, define the reduced
coefficient matrix
$$
K_{ai,b} = e_a e_i^T.
$$

The induced first-order inactive update is
$$
\Delta Q_b^{(ai)} = T_{a,b} e_a e_i^T.
$$

Equivalently, for a general coefficient matrix $K_{ai,b}$,
$$
\Delta Q_b^{(ai)} = T_{a,b} K_{ai,b}.
$$

This automatically satisfies the horizontal condition:
$$
Q_b^T S_b \Delta Q_b^{(ai)}
= Q_b^T S_b T_{a,b} K_{ai,b}
= 0.
$$

To avoid double-counting the same physical coupling through the active block,
choose the associated active-block update to be
$$
\Delta C_{a,b}^{(ai)} = -Q_b K_{ai,b}^T.
$$

This is the direct orthonormal-gauge analogue of the current antisymmetric
inactive-active occupied rotation.

### Effect on the active auxiliary block

At first order,
$$
\Delta T_{a,b}^{(ai)}
= -\Delta Q_b^{(ai)} B_{ia,b}
- Q_b (\Delta Q_b^{(ai)})^T S_b T_{a,b}
+ (I - Q_b Q_b^T S_b)\Delta C_{a,b}^{(ai)},
$$
where
$$
B_{ia,b} = Q_b^T S_b C_{a,b}.
$$

Because $\Delta Q_b^{(ai)} = T_{a,b} K_{ai,b}$, this becomes
$$
\Delta T_{a,b}^{(ai)}
= -T_{a,b} K_{ai,b} B_{ia,b}
- Q_b K_{ai,b}^T K_{a,b}
- (I - Q_b Q_b^T S_b) Q_b K_{ai,b}^T.
$$

The last term vanishes since
$$
(I - Q_b Q_b^T S_b) Q_b = 0.
$$

So
$$
\Delta T_{a,b}^{(ai)}
= -T_{a,b} K_{ai,b} B_{ia,b}
- Q_b K_{ai,b}^T K_{a,b}.
$$

This shows two things:

1. inactive-active reduced coordinates can indeed be carried entirely by the
   pair $(\Delta Q_b, \Delta C_{a,b})$ above;
2. the resulting $\Delta T_{a,b}$ stays inside the span of $(Q_b, T_{a,b})$,
   which is exactly what one wants for a block-internal occupied rotation.

## Inactive-Virtual Reduced Directions

For a general coefficient matrix
$$
K_{vi,b} \in \mathbb{R}^{n_{v,b} \times n_{i,b}},
$$
define
$$
\Delta Q_b^{(iv)} = V_b K_{vi,b}.
$$

Again this is horizontal because
$$
Q_b^T S_b \Delta Q_b^{(iv)}
= Q_b^T S_b V_b K_{vi,b}
= 0.
$$

Choose no direct physical active update:
$$
\Delta C_{a,b}^{(iv)} = 0.
$$

Then the induced auxiliary variation is
$$
\Delta T_{a,b}^{(iv)}
= -\Delta Q_b^{(iv)} B_{ia,b}
- Q_b (\Delta Q_b^{(iv)})^T S_b T_{a,b}.
$$

Substituting $\Delta Q_b^{(iv)} = V_b K_{vi,b}$ gives
$$
\Delta T_{a,b}^{(iv)}
= -V_b K_{vi,b} B_{ia,b}
- Q_b K_{vi,b}^T (V_b^T S_b T_{a,b}).
$$

So inactive-virtual coordinates also stay in the span of
$$
Q_b \oplus T_{a,b} \oplus V_b.
$$

This is the orthonormal-gauge analogue of the current occupied-virtual update
applied to inactive occupied columns.

## Active-Virtual Reduced Directions

For a coefficient matrix
$$
U_{va,b} \in \mathbb{R}^{n_{v,b} \times n_{a,b}},
$$
define
$$
\Delta Q_b^{(av)} = 0,
\qquad
\Delta C_{a,b}^{(av)} = V_b U_{va,b}.
$$

Then
$$
\Delta T_{a,b}^{(av)}
= (I - Q_b Q_b^T S_b) V_b U_{va,b}.
$$

Since $V_b$ is orthogonal to the occupied block,
$$
Q_b^T S_b V_b = 0.
$$

So if $V_b$ is also chosen orthogonal to the active auxiliary span, then
$$
(I - Q_b Q_b^T S_b) V_b = V_b.
$$

More generally, even without that stronger choice, the projector simply removes
any inactive component and the active-virtual direction remains a clean
block-external excitation.

Thus the natural first-order formula is just
$$
\Delta C_{a,b}^{(av)} = V_b U_{va,b}.
$$

This is the direct analogue of the current active-virtual block.

## Compact Block-Local Map

Collecting the three reduced pieces, the Option-A first-order block map is:
$$
\Delta Q_b = T_{a,b} K_{ai,b} + V_b K_{vi,b},
$$
and
$$
\Delta C_{a,b} = -Q_b K_{ai,b}^T + V_b U_{va,b}.
$$

This is the simplest block-local formula that:

1. preserves the horizontal inactive gauge
   $$
   Q_b^T S_b \Delta Q_b = 0,
   $$
2. carries inactive-active, inactive-virtual, and active-virtual directions,
3. avoids duplicating inactive-active coupling across both inactive and active
   coordinate blocks.

It is the orthonormal-gauge analogue of the current block-local combination
formula used by `accumulate_block_candidate_combination(...)`.

## Relation to Current Candidate Counts

The sizes are:
$$
\#(K_{ai,b}) = n_{a,b} n_{i,b},
$$
$$
\#(K_{vi,b}) = n_{v,b} n_{i,b},
$$
$$
\#(U_{va,b}) = n_{v,b} n_{a,b}.
$$

Summing gives
$$
n_{i,b} n_{a,b} + n_{v,b} n_{i,b} + n_{v,b} n_{a,b}
= n_{i,b} n_{a,b} + n_{v,b} (n_{i,b} + n_{a,b})
= n_{i,b} n_{a,b} + n_{v,b} n_{occ,b}.
$$

This is exactly the current block-local reduced dimension.

So the proposed chart change does not enlarge the reduced Krylov dimension.

## Scatter Back to the Packed Sparse Chart

The current code stores per-orbital scatter data:

- block-row to packed-index maps for dense full-support blocks,
- masked local matrices for sparse blocks.

That machinery can still be reused if the new chart produces explicit block
matrices for:

$$
\Delta Q_b \in \mathbb{R}^{|\Omega_b| \times n_{i,b}},
\qquad
\Delta C_{a,b} \in \mathbb{R}^{|\Omega_b| \times n_{a,b}}.
$$

The key observation is that both are already written in the block AO basis
$(Q_b, T_{a,b}, V_b)$ restricted to the same union AO domain $\Omega_b$.

Therefore, for a dense full-support block:

1. assemble the occupied block variation matrix
   $$
   \Delta C_{occ,b}
   =
   \begin{bmatrix}
   \Delta Q_b & \Delta C_{a,b}
   \end{bmatrix},
   $$
2. scatter it row by row through the existing block-row to packed-index maps.

For a sparse block:

1. restrict each occupied column of $\Delta C_{occ,b}$ to the explicit sparse
   support of that orbital,
2. write those local coefficients through the existing masked scatter maps.

So the packed sparse chart does not prevent the new gauge. What changes is the
construction of the block-local direction matrices, not the existence of a
scatter path.

## What Still Needs To Be Proved

The formulas above give a plausible first-order block map, but before any code
change they still need three mathematical checks.

### 1. Compatibility with finite retraction

The current optimizer eventually needs a finite trial-point map, not just an
infinitesimal tangent.

So one must define a retraction that updates:
$$
(Q_b, C_{a,b})
\mapsto
(Q_b', C_{a,b}')
$$
while preserving
$$
(Q_b')^T S_b Q_b' = I.
$$

The obvious candidate is a generalized-Stiefel Cayley or QR-based retraction on
$Q_b$, combined with a linear or coupled update on $C_{a,b}$.

### 2. Block locality under union AO domains

The derivation assumes that all block-local basis vectors
$$
Q_b,\; T_{a,b},\; V_b
$$
live on the same union domain $\Omega_b$.

This matches the current implementation, but the numerical quality of the new
chart will depend on whether the orthonormalized inactive frame stays well
localized on those same domains.

### 3. Gauge consistency across accepted-point rebuilds

Because the inactive frame is now orthonormalized, one must choose a stable
accepted-point gauge:

- either by canonicalizing $Q_b$,
- or by transporting the previous accepted frame to the new point.

Without that, the exact-context cache and the TN model may see artificial
frame jumps between iterations.

## Immediate Next Derivation

The next mathematical task should be to derive the **finite-step retraction**
corresponding to the infinitesimal map
$$
\Delta Q_b = T_{a,b} K_{ai,b} + V_b K_{vi,b},
\qquad
\Delta C_{a,b} = -Q_b K_{ai,b}^T + V_b U_{va,b}.
$$

That is the point where we can decide whether the new chart is merely elegant,
or actually implementable inside the current TN optimizer with acceptable
complexity.

## Finite-Step Retraction for the New Chart

This section addresses the remaining missing piece: given the infinitesimal
block-local map
$$
\Delta Q_b = T_{a,b} K_{ai,b} + V_b K_{vi,b},
\qquad
\Delta C_{a,b} = -Q_b K_{ai,b}^T + V_b U_{va,b},
$$
how should we build a finite trial point?

The current implementation uses a block Cayley-style retraction in the local
occupied/virtual coordinates, then samples the resulting occupied block back to
the packed sparse chart. See
[`nonredundant_orbital_space.cpp#L523`](/pool1/home/xiatao/project/xmvb-cpp/src/vb/orbital/nonredundant_orbital_space.cpp#L523)
and
[`nonredundant_orbital_space.cpp#L1277`](/pool1/home/xiatao/project/xmvb-cpp/src/vb/orbital/nonredundant_orbital_space.cpp#L1277).

The new chart needs an analogue that preserves
$$
(Q_b')^T S_b Q_b' = I
$$
at finite step size.

## Retraction Design Requirements

For one block $b$, a retraction should map a reduced step
$$
(K_{ai,b}, K_{vi,b}, U_{va,b})
$$
to updated block variables
$$
(Q_b', C_{a,b}')
$$
such that:

1. $Q_b'$ is exactly $S_b$-orthonormal,
2. $Q_b'$ remains in the same block AO domain $\Omega_b$,
3. the first-order expansion matches
   $$
   \Delta Q_b = T_{a,b} K_{ai,b} + V_b K_{vi,b},
   $$
   $$
   \Delta C_{a,b} = -Q_b K_{ai,b}^T + V_b U_{va,b},
   $$
4. the update can still be scattered back into the packed sparse chart.

There are two obvious candidates:

1. a generalized-Stiefel QR / polar retraction on $Q_b$ plus an explicit
   update for $C_{a,b}$,
2. a block Cayley transform on a larger mixed basis that includes
   $Q_b$, $T_{a,b}$, and $V_b$.

## Candidate 1: Generalized-Stiefel QR Retraction

Start from the linear trial
$$
\widetilde{Q}_b
= Q_b + T_{a,b} K_{ai,b} + V_b K_{vi,b},
$$
$$
\widetilde{C}_{a,b}
= C_{a,b} - Q_b K_{ai,b}^T + V_b U_{va,b}.
$$

This already matches the required first-order variation.

Now enforce exact inactive orthonormality by re-orthonormalizing
$\widetilde{Q}_b$ in the $S_b$ metric. A natural retraction is
$$
Q_b' =
\widetilde{Q}_b
\left(\widetilde{Q}_b^T S_b \widetilde{Q}_b\right)^{-1/2}.
$$

Equivalently, one can compute a generalized QR using the Cholesky factor of
$S_b$.

### Why this is a valid retraction

At zero step, $\widetilde{Q}_b = Q_b$ and
$$
\widetilde{Q}_b^T S_b \widetilde{Q}_b = I + O(\|\xi_b\|^2)
$$
because the tangent is horizontal:
$$
Q_b^T S_b \Delta Q_b = 0.
$$

Therefore
$$
\left(\widetilde{Q}_b^T S_b \widetilde{Q}_b\right)^{-1/2}
= I + O(\|\xi_b\|^2),
$$
and hence
$$
Q_b' = Q_b + \Delta Q_b + O(\|\xi_b\|^2).
$$

So the first-order map is preserved.

### What to do with the active block

The simplest companion update is just
$$
C_{a,b}' = \widetilde{C}_{a,b}.
$$

Then rebuild the active auxiliary block from the updated inactive frame:
$$
T_{a,b}' = (I - Q_b' {Q_b'}^T S_b) C_{a,b}'.
$$

This yields an accepted-point state consistent with Option A.

### Pros

1. very simple derivation,
2. clearly preserves the desired first-order tangent,
3. exactly enforces inactive orthonormality,
4. does not require building a large antisymmetric generator.

### Cons

1. the active block is not updated by one coupled orthogonality-preserving
   group action,
2. the accepted-point gauge of $Q_b'$ must be stabilized carefully,
3. metric QR / inverse square root is more expensive than a pure linear update,
   although still only on an $n_{i,b} \times n_{i,b}$ matrix.

## Candidate 2: Block Cayley Retraction on $(Q_b, T_{a,b}, V_b)$

The current code already uses a Cayley transform on a block-local
occupied/virtual basis. So it is natural to ask whether the new chart can keep
the same style.

Construct the block-local basis
$$
Y_b = [Q_b \;\; \widehat{T}_{a,b} \;\; V_b],
$$
where $\widehat{T}_{a,b}$ is an $S_b$-orthonormal basis of the active
auxiliary span. One convenient choice is
$$
\widehat{T}_{a,b} = T_{a,b} K_{a,b}^{-1/2},
\qquad
K_{a,b} = T_{a,b}^T S_b T_{a,b}.
$$

Then
$$
Y_b^T S_b Y_b = I.
$$

Now define a skew-symmetric generator
$$
\Omega_b =
\begin{bmatrix}
0 & -A_b^T & -B_b^T \\
A_b & 0 & -C_b^T \\
B_b & C_b & 0
\end{bmatrix},
$$
with block sizes:

- $A_b \in \mathbb{R}^{n_{a,b} \times n_{i,b}}$,
- $B_b \in \mathbb{R}^{n_{v,b} \times n_{i,b}}$,
- $C_b \in \mathbb{R}^{n_{v,b} \times n_{a,b}}$.

Apply the Cayley transform
$$
U_b =
\left(I - \frac{1}{2}\Omega_b\right)^{-1}
\left(I + \frac{1}{2}\Omega_b\right).
$$

Then the updated block basis is
$$
Y_b' = Y_b U_b.
$$

Since $U_b$ is orthogonal and $Y_b$ is $S_b$-orthonormal, we get
$$
{Y_b'}^T S_b Y_b' = I.
$$

### First-order expansion

To first order,
$$
U_b = I + \Omega_b + O(\|\Omega_b\|^2).
$$

Therefore
$$
Y_b' = Y_b + Y_b \Omega_b + O(\|\Omega_b\|^2).
$$

The inactive block changes as
$$
\Delta Q_b = \widehat{T}_{a,b} A_b + V_b B_b.
$$

To match the Option-A infinitesimal map
$$
\Delta Q_b = T_{a,b} K_{ai,b} + V_b K_{vi,b},
$$
we must choose
$$
A_b = K_{a,b}^{1/2} K_{ai,b},
\qquad
B_b = K_{vi,b}.
$$

The orthonormal active-auxiliary basis changes as
$$
\Delta \widehat{T}_{a,b} = -Q_b A_b^T + V_b C_b.
$$

So $C_b$ is naturally the orthonormal-gauge active-virtual block.

### Relation to the physical active block

This retraction updates the orthonormal basis $\widehat{T}_{a,b}$, not directly
the physical block $C_{a,b}$.

To recover Option A, one must reconstruct a physical active block satisfying
$$
T_{a,b}' = (I - Q_b' {Q_b'}^T S_b) C_{a,b}'.
$$

One possibility is to preserve the accepted active internal metric
$K_{a,b}$ and set
$$
T_{a,b}' = \widehat{T}_{a,b}' K_{a,b}^{1/2},
$$
then choose
$$
C_{a,b}' = T_{a,b}' + Q_b' B_{ia,b}',
$$
for a suitable updated inactive-active coefficient block $B_{ia,b}'$.

But this is already more involved than Candidate 1.

### Pros

1. it is structurally close to the current block Cayley implementation,
2. all updated basis blocks remain inside one coupled $S_b$-orthonormal frame,
3. the inactive-active / inactive-virtual / active-virtual amplitudes appear as
   one skew generator.

### Cons

1. it naturally parameterizes $(Q_b, \widehat{T}_{a,b})$, not $(Q_b, C_{a,b})$,
2. reconstructing the physical active block is nontrivial,
3. this pushes the implementation closer to Option B than to Option A.

## Which Retraction Matches Option A Best?

For Option A specifically, Candidate 1 is the cleaner match:

1. update $Q_b$ through a generalized-Stiefel retraction,
2. update $C_{a,b}$ directly with the derived first-order map,
3. rebuild $T_{a,b}$ afterward from $(Q_b', C_{a,b}')$.

This preserves the meaning of the primary variables and avoids introducing an
auxiliary orthonormal active basis into the main optimizer state.

Candidate 2 is elegant, but it effectively wants to optimize a coupled
orthonormal basis for $(Q_b, T_{a,b})$. That is closer to Option B.

So the right pairing is:

- Option A + generalized-Stiefel QR/polar retraction,
- Option B + coupled Cayley retraction on an orthonormal occupied/virtual basis.

## Recommended Finite-Step Formula for Option A

The most practical finite-step retraction candidate is therefore:

### Step 1: form linear trial blocks
$$
\widetilde{Q}_b
= Q_b + T_{a,b} K_{ai,b} + V_b K_{vi,b},
$$
$$
\widetilde{C}_{a,b}
= C_{a,b} - Q_b K_{ai,b}^T + V_b U_{va,b}.
$$

### Step 2: re-orthonormalize the inactive block
$$
Q_b' =
\widetilde{Q}_b
\left(\widetilde{Q}_b^T S_b \widetilde{Q}_b\right)^{-1/2}.
$$

### Step 3: keep the physical active block update
$$
C_{a,b}' = \widetilde{C}_{a,b}.
$$

### Step 4: rebuild the active auxiliary block
$$
T_{a,b}' = (I - Q_b' {Q_b'}^T S_b) C_{a,b}'.
$$

This is the simplest finite-step map that is consistent with all previous
first-order derivations.

## First-Order Verification

Let
$$
\widetilde{Q}_b = Q_b + \Delta Q_b.
$$

Because the tangent is horizontal,
$$
\widetilde{Q}_b^T S_b \widetilde{Q}_b
= I + \Delta Q_b^T S_b \Delta Q_b
= I + O(\|\Delta Q_b\|^2).
$$

Hence
$$
\left(\widetilde{Q}_b^T S_b \widetilde{Q}_b\right)^{-1/2}
= I + O(\|\Delta Q_b\|^2),
$$
which implies
$$
Q_b' = Q_b + \Delta Q_b + O(\|\xi_b\|^2).
$$

Similarly,
$$
C_{a,b}' = C_{a,b} + \Delta C_{a,b},
$$
exactly at first order.

Therefore
$$
T_{a,b}'
= (I - Q_b' {Q_b'}^T S_b) C_{a,b}'
= T_{a,b} + \Delta T_{a,b} + O(\|\xi_b\|^2),
$$
with $\Delta T_{a,b}$ equal to the previously derived Option-A tangent.

So this retraction is first-order consistent with the desired exact-context
HVP formulas.

## Expected Implementation Burden

Relative to the current `NonredundantOrbitalSpace::retract_step(...)`, the new
Option-A retraction would require:

1. storing block-local inactive orthonormal frames $Q_b$,
2. storing or rebuilding block-local active auxiliary blocks $T_{a,b}$,
3. building the linear trial blocks $(\widetilde{Q}_b, \widetilde{C}_{a,b})$,
4. performing a small generalized-Stiefel orthonormalization of each
   $\widetilde{Q}_b$,
5. scattering $(Q_b', C_{a,b}')$ back into the sparse orbital table.

This is more invasive than the current Cayley path, but it is still a
block-local construction. It does not require changing the optimizer trust
region, Krylov machinery, or reduced dimension formulas.

## Final Assessment

At this point the mathematical picture is:

1. the orthonormal inactive gauge admits a clean first-order chart,
2. it also admits a viable finite-step retraction compatible with Option A,
3. the most practical retraction is not a direct transplant of the current
   Cayley formula, but a generalized-Stiefel retraction for the inactive block
   plus a direct update for the physical active block.

So the remaining uncertainty is no longer "is there a mathematically coherent
retraction?", but rather "is the implementation cost and numerical behavior of
that retraction justified by the expected wall-time savings?".

That is now a code-design and benchmarking question, not a missing-formula
question.

## Engineering Design Recommendation

The derivation above is now detailed enough to support an implementation-level
decision.

The key question is no longer whether an orthonormal inactive gauge exists, but
which version of it is the right engineering target for this repository.

## Recommended Path: Option A First

The recommended implementation path is:

1. adopt the orthonormal inactive gauge,
2. keep the physical active block $C_a$ as the primary active variable,
3. keep the active auxiliary block
   $$
   T_a = (I - Q_i Q_i^T S) C_a
   $$
   as a derived maintained object,
4. use the generalized-Stiefel retraction derived above for finite steps.

In short:

$$
\text{Recommended first implementation} = \text{Option A} + \text{QR/polar retraction on } Q_i.
$$

### Why Option A is the right first target

Option A preserves the current architectural split used by the code:

1. the optimizer variables still correspond to physical orbital coefficients,
2. the active auxiliary block is still built by the orbital preparer,
3. the exact-context operator still consumes that prepared active auxiliary
   block and its derivatives.

This means the implementation changes remain concentrated in the orbital
parameterization and preparer layers, rather than forcing an immediate
repository-wide change in the meaning of "active orbitals".

## Why Option B Should Wait

Option B, with $(Q_i, T_a)$ as primary variables, is mathematically cleaner for
the exact-context formulas. But it should be treated as a second-stage design,
not the first implementation target.

The reason is that Option B would force a broader semantic rewrite:

1. the runtime and preparer APIs would have to treat active auxiliary orbitals
   as the primary optimizer variables,
2. several places that currently assume access to physical active orbitals
   would need redesign,
3. the retraction and chart logic would shift from "physical occupied block"
   semantics to "mixed orthonormal/auxiliary block" semantics.

That is likely too much simultaneous change for a first implementation of the
new gauge.

So the design recommendation is:

1. prove the idea numerically with Option A,
2. only consider Option B if Option A validates the expected speedup but still
   leaves too much algebraic overhead in the active block.

## Module Impact

The gauge redesign does **not** affect all modules equally.

### Modules that would need direct redesign

#### 1. `src/vb/orbital/nonredundant_orbital_space.*`

This is the main chart / reduced-space layer.

It would need:

1. new block-local basis construction using
   $$
   Q_b,\; T_{a,b},\; V_b
   $$
   instead of the current physical occupied block,
2. new reduced-direction assembly using
   $$
   \Delta Q_b = T_{a,b} K_{ai,b} + V_b K_{vi,b},
   $$
   $$
   \Delta C_{a,b} = -Q_b K_{ai,b}^T + V_b U_{va,b},
   $$
3. a new finite-step retraction that orthonormalizes the inactive block,
4. stable accepted-point gauge handling for $Q_b$.

This is the most intrusive required change.

#### 2. `src/vb/orbital/active_space_orbital_preparer.*`

This module would need a new accepted-point representation:

1. store the orthonormal inactive frame $Q_i$ explicitly,
2. rebuild the inactive projector as
   $$
   P_i = Q_i Q_i^T,
   $$
3. keep building the active auxiliary block from
   $$
   T_a = (I - Q_i Q_i^T S) C_a,
   $$
4. decide whether to preserve a recoverable physical inactive block for
   diagnostics / legacy compatibility.

This is where the formulation-level simplification first enters production
data structures.

#### 3. `src/vb/scf/exact_orbital_second_order_operator.*`

This module would need the exact-context formulas rewritten in the new chart:

1. replace all current inactive-metric chains involving
   $$
   M,\; M^{-1},\; dM^{-1}
   $$
   by the orthonormal-gauge formulas,
2. rewrite the fixed-upstream pullback in terms of
   $$
   Q_i,\; \Delta_i,\; T_a,
   $$
3. update accepted-point cache contents accordingly.

This is the performance-critical target module, but it should be changed only
after the new chart is available.

### Modules that likely need only adaptation, not redesign

#### 1. `src/vb/scf/cpp_vb_scf_optimizer.cpp`

The optimizer-level trust region, Krylov solver, and reduced-space interfaces
should mostly stay intact because:

1. the reduced dimension can be kept the same,
2. the optimizer still consumes reduced gradients and reduced HVP actions,
3. the trust-region logic is agnostic to the detailed orbital gauge.

The optimizer would mainly need:

1. to accept the new `NonredundantOrbitalSpace`,
2. to use the updated retraction path,
3. to keep any diagnostics consistent with the new gauge.

#### 2. Integral and outer-response code

Modules that operate on already prepared active-space matrices or AO
contractions should need little or no conceptual redesign, provided the
preparer still exports the same downstream objects:

1. active auxiliary orbitals,
2. inactive density / projector,
3. active-space overlap / one-electron / two-electron intermediates.

That is an important reason to keep Option A first: it preserves downstream
contracts more effectively.

## Implementation Order

The safest implementation sequence is:

### Phase 1: Infrastructure and accepted-point state

1. extend the preparer data structures to represent the orthonormal inactive
   frame $Q_i$,
2. define a stable accepted-point gauge convention,
3. add diagnostics that compare the old and new inactive projectors and active
   auxiliary blocks at the same accepted point.

Goal:

- verify that the new representation reproduces the same prepared physical
  objects to numerical precision.

### Phase 2: New nonredundant chart

1. implement the block-local basis construction for Option A,
2. implement the reduced tangent assembly
   $$
   \xi \mapsto (\Delta Q_b, \Delta C_{a,b}),
   $$
3. implement the generalized-Stiefel retraction for each inactive block,
4. keep a reference path that reconstructs the old physical occupied block for
   comparison.

Goal:

- verify that finite steps are stable and that reduced directions map to the
  expected first-order orbital changes.

### Phase 3: Exact-context HVP port

1. port orbital-preparation directional formulas to the new chart,
2. port fixed-upstream pullback formulas,
3. keep cached vs uncached validation hooks,
4. re-run closed-shell, open-shell, and `241_VBSCF.xmi` exact-context checks.

Goal:

- preserve exactness before pursuing further tuning.

### Phase 4: Performance validation

1. add maintained exact-context stage timing breakdown,
2. benchmark the old chart and new chart on the same accepted point,
3. compare:
   - total `exact_ctx` wall time,
   - fixed-upstream time,
   - orbital-preparation directional time,
   - overall truncated-Newton iteration time.

Goal:

- decide whether the orthonormal inactive gauge delivers enough real speedup to
  justify permanent adoption.

## Main Risks

The derivation resolves the formula question, but three engineering risks
remain.

### 1. Gauge drift and accepted-point instability

If the orthonormal inactive frame is not canonically chosen, the accepted-point
cache may see frame rotations unrelated to physical orbital changes.

This could:

1. destabilize exact-context caching,
2. degrade finite-difference comparisons,
3. create noisy reduced directions across accepted iterations.

So stable gauge selection is mandatory.

### 2. Retraction cost vs. saved HVP cost

The new formulation removes inactive-metric derivatives inside `exact_ctx`, but
it adds a generalized-Stiefel orthonormalization in the finite-step map.

That tradeoff is likely favorable because retraction happens much less often
than HVP matvecs inside inner Krylov solves, but it still needs measurement.

### 3. Block-local sparsity degradation

Even if the chart is mathematically cleaner, the orthonormalized inactive frame
may be less sparse or less localized than the current physical block.

If that significantly widens effective support inside union AO domains, some of
the expected savings could be eaten by denser local block algebra.

## Decision Summary

Based on the current derivation, the design recommendation is:

1. **Do not** keep searching for local formula hacks in the current chart.
2. **Do** treat the orthonormal inactive gauge as the only mathematically
   credible route to a more fundamental simplification.
3. **Implement Option A first**, not Option B.
4. **Use generalized-Stiefel QR/polar retraction**, not a forced transplant of
   the current Cayley formula.
5. **Port the exact-context operator only after the new chart exists**.

This is the narrowest implementation path that still tests the real underlying
idea.
