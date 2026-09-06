# Block-Local Inactive-Orthogonal Gauge Derivation

## 1. Scope

This note derives a unified block-local formulation for an inactive-orthogonal
working gauge that can cover both of the orbital representations discussed in
the current code base:

1. `full-AO`: each orbital is stored directly on a full block-local AO domain,
2. `sparse`: each orbital is stored only on its explicit support, but is
   embedded into the same block-local AO domain before the orbital-preparation
   and HVP formulas are applied.

The goal is not to describe the current implementation exactly. The goal is to
settle the formulas that would be needed if the inactive block were genuinely
rewritten into an $S$-orthogonal chart.

The main conclusion is:

$$
\boxed{
\text{the simplification is real only if the working inactive variables are }
Q_i \text{ themselves, not } C_i \text{ followed by } Q_i = C_i M^{-1/2}.
}
$$

If one keeps $C_i$ as the real optimization variable and differentiates through
$M^{-1/2}$ afterward, then the complicated chain rule simply reappears in a new
place.

## 2. Unified Block-Local Physical Representation

Consider one orbital block $b$. Let

$$
\Omega_b = \{\mu_1, \mu_2, \dots, \mu_{N_b}\}
$$

be the block-local AO domain, where $N_b = |\Omega_b|$, and let

$$
S_b \in \mathbb{R}^{N_b \times N_b}
$$

be the AO overlap matrix restricted to that domain.

### 2.1 Per-orbital embedding

For each occupied orbital $p$ in block $b$, introduce a physical coefficient
vector

$$
x_{b,p} \in \mathbb{R}^{m_{b,p}},
$$

and an embedding matrix

$$
U_{b,p} \in \mathbb{R}^{N_b \times m_{b,p}},
$$

such that the block-local AO coefficient vector is

$$
c_{b,p} = U_{b,p} x_{b,p}.
$$

This single formula covers both cases:

1. `full-AO` case:
   $$
   m_{b,p} = N_b,
   \qquad
   U_{b,p} = I_{N_b},
   \qquad
   c_{b,p} = x_{b,p}.
   $$
2. `sparse` case:
   $$
   m_{b,p} = |J_{b,p}|,
   $$
   where $J_{b,p} \subseteq \Omega_b$ is the explicit support of orbital $p$,
   and $U_{b,p}$ is the column-selector / injection matrix that places those
   sparse coefficients into the rows of $\Omega_b$.

Therefore the forward and reverse local mappings are

$$
\delta c_{b,p} = U_{b,p} \, \delta x_{b,p},
\qquad
\bar x_{b,p} = U_{b,p}^{\mathrm T} \bar c_{b,p}.
$$

The first relation is the local gather / embed map for a direction, and the
second relation is the local scatter / restriction map for a gradient or HVP
adjoint.

### 2.2 Block-local occupied matrices

Collect the inactive and active occupied orbitals in block $b$ as

$$
C_{i,b}
=
\bigl[
c_{b,i_1}, c_{b,i_2}, \dots, c_{b,i_{n_{i,b}}}
\bigr]
\in
\mathbb{R}^{N_b \times n_{i,b}},
$$

$$
C_{a,b}
=
\bigl[
c_{b,a_1}, c_{b,a_2}, \dots, c_{b,a_{n_{a,b}}}
\bigr]
\in
\mathbb{R}^{N_b \times n_{a,b}}.
$$

All formulas below are written only on $(\Omega_b, S_b)$ and therefore apply
equally to `full-AO` and `sparse`, as long as the orbitals are first embedded
into this common block-local AO domain.

## 3. Inactive-Orthogonal Working Gauge

Define the block-local inactive overlap matrix

$$
M_b = C_{i,b}^{\mathrm T} S_b C_{i,b}
\in
\mathbb{R}^{n_{i,b} \times n_{i,b}}.
$$

Assume $M_b$ is symmetric positive definite at a valid accepted point. Then
define the inactive-orthogonal working orbitals

$$
Q_{i,b} = C_{i,b} M_b^{-1/2}.
$$

By construction,

$$
Q_{i,b}^{\mathrm T} S_b Q_{i,b} = I.
$$

Now define the inactive density matrix and metric projector

$$
P_{i,b} = Q_{i,b} Q_{i,b}^{\mathrm T},
\qquad
\Pi_{i,b} = P_{i,b} S_b = Q_{i,b} Q_{i,b}^{\mathrm T} S_b.
$$

The complementary metric projector is

$$
R_{i,b} = I - \Pi_{i,b}
=
I - Q_{i,b} Q_{i,b}^{\mathrm T} S_b.
$$

The corresponding active auxiliary orbitals are

$$
T_{a,b} = R_{i,b} C_{a,b}.
$$

This immediately gives

$$
Q_{i,b}^{\mathrm T} S_b T_{a,b} = 0.
$$

So in this chart:

1. the inactive block is internally $S_b$-orthonormal,
2. the inactive block and the active auxiliary block are exactly $S_b$-orthogonal,
3. the active block generally remains nonorthogonal:
   $$
   T_{a,b}^{\mathrm T} S_b T_{a,b} \neq I.
   $$

## 4. What Changes Relative to the Current Mixed Gauge

The current mixed gauge uses

$$
P_{i,b}^{\text{current}}
=
C_{i,b}
\left(
C_{i,b}^{\mathrm T} S_b C_{i,b}
\right)^{-1}
C_{i,b}^{\mathrm T},
$$

so the formulas for the pullback and the HVP must keep the entire
$M_b^{-1}$-chain.

In contrast, once $Q_{i,b}$ becomes the genuine working variable, one has

$$
P_{i,b} = Q_{i,b} Q_{i,b}^{\mathrm T},
$$

and the projector derivative simplifies to

$$
dP_{i,b} = dQ_{i,b} Q_{i,b}^{\mathrm T} + Q_{i,b} dQ_{i,b}^{\mathrm T}.
$$

This simplification is genuine only in the $Q_{i,b}$ chart itself. If one
keeps $C_{i,b}$ as the true variable and writes

$$
Q_{i,b} = C_{i,b} M_b^{-1/2},
$$

then the derivative $d(M_b^{-1/2})$ reintroduces a nontrivial chain rule.

## 5. First Derivatives in the Orthogonal Inactive Chart

To simplify notation, drop the block label $b$ below and write

$$
Q_i = Q_{i,b},
\qquad
C_a = C_{a,b},
\qquad
S = S_b,
\qquad
R_i = I - Q_i Q_i^{\mathrm T} S.
$$

Also define the inactive-active overlap matrix in this chart:

$$
B_{ia} = Q_i^{\mathrm T} S C_a
\in
\mathbb{R}^{n_i \times n_a}.
$$

### 5.1 Inactive projector derivative

Since

$$
P_i = Q_i Q_i^{\mathrm T},
$$

we have

$$
dP_i = dQ_i Q_i^{\mathrm T} + Q_i dQ_i^{\mathrm T}.
$$

For the metric projector

$$
\Pi_i = P_i S = Q_i Q_i^{\mathrm T} S,
$$

the derivative is

$$
d\Pi_i = dQ_i Q_i^{\mathrm T} S + Q_i dQ_i^{\mathrm T} S.
$$

### 5.2 Active auxiliary derivative

Because

$$
T_a = (I - Q_i Q_i^{\mathrm T} S) C_a = R_i C_a,
$$

we obtain

$$
dT_a
=
R_i \, dC_a
- dQ_i \, (Q_i^{\mathrm T} S C_a)
- Q_i \, dQ_i^{\mathrm T} S C_a.
$$

Equivalently,

$$
\boxed{
dT_a
=
R_i \, dC_a
- dQ_i B_{ia}
- Q_i dQ_i^{\mathrm T} S C_a.
}
$$

Compared with the current mixed gauge, the entire $M^{-1}$ and $dM^{-1}$ chain
has disappeared from this expression.

## 6. Reverse-Mode Pullback on the Orthogonal Inactive Chart

Suppose the energy depends on the occupied block through

$$
E = E(T_a, P_i).
$$

Introduce the accepted-point upstream gradients

$$
G_T = \frac{\partial E}{\partial T_a}
\in
\mathbb{R}^{N_b \times n_a},
$$

$$
G_P = \frac{\partial E}{\partial P_i}
\in
\mathbb{R}^{N_b \times N_b}.
$$

Under the Frobenius inner product

$$
\langle X, Y \rangle = \operatorname{tr}(X^{\mathrm T} Y),
$$

the first variation is

$$
\delta E
=
\langle G_T, \delta T_a \rangle
+
\langle G_P, \delta P_i \rangle.
$$

### 6.1 Active physical block pullback

Substitute the $R_i \delta C_a$ part of $\delta T_a$:

$$
\langle G_T, R_i \delta C_a \rangle
=
\langle R_i^{\mathrm T} G_T, \delta C_a \rangle.
$$

Therefore

$$
\boxed{
G_{C_a}
=
R_i^{\mathrm T} G_T
=
\left(I - S Q_i Q_i^{\mathrm T}\right) G_T.
}
$$

### 6.2 Effective inactive-density gradient

Define

$$
\widetilde G_P = G_P - G_T (S C_a)^{\mathrm T}.
$$

Then the remaining two contributions involving $\delta Q_i$ combine with the
$\delta P_i$ term into a single symmetric pullback:

$$
\boxed{
G_{Q_i}
=
\left(
\widetilde G_P + \widetilde G_P^{\mathrm T}
\right) Q_i.
}
$$

This is the ambient block-local pullback with respect to the orthogonal
inactive variables.

### 6.3 Tangent projection if the optimizer works directly on the constraint

If one chooses to optimize $Q_i$ directly under the constraint

$$
Q_i^{\mathrm T} S Q_i = I,
$$

then the admissible tangent directions satisfy

$$
Q_i^{\mathrm T} S \, \delta Q_i
+
\delta Q_i^{\mathrm T} S Q_i
=
0.
$$

The ambient gradient $G_{Q_i}$ should then be projected to the tangent space:

$$
\operatorname{grad}_{Q_i}^{\mathrm{tan}}
=
G_{Q_i}
- Q_i \, \operatorname{sym}\!\left(Q_i^{\mathrm T} S G_{Q_i}\right),
$$

where

$$
\operatorname{sym}(X) = \frac{1}{2}(X + X^{\mathrm T}).
$$

## 7. Physical-Parameter Gather / Scatter

The previous section gives gradients with respect to the block-local AO
matrices $Q_i$ and $C_a$. To recover gradients on the original physical
parameterization, use the same embedding operators $U_{b,p}$ introduced in
Section 2.

### 7.1 Active orbitals

For an active orbital $a$ in block $b$, let

$$
\bar c_{b,a}
$$

be the corresponding column of $G_{C_a}$. Then the gradient on the physical
parameter vector is simply

$$
\boxed{
\bar x_{b,a}
=
U_{b,a}^{\mathrm T} \bar c_{b,a}.
}
$$

This formula is valid for both `full-AO` and `sparse`.

### 7.2 Inactive orbitals

If the inactive working variable is truly chosen to be $Q_i$, then the same
gather / scatter rule applies:

$$
\boxed{
\bar x_{b,i}^{(Q)}
=
U_{b,i}^{\mathrm T} \bar q_{b,i}.
}
$$

However, this should not be confused with the old physical inactive
coefficients in the current mixed gauge. In the orthogonal chart, the internal
inactive orbitals are no longer represented by the original per-orbital sparse
supports after block mixing by $M_b^{-1/2}$.

Therefore:

1. in `full-AO`, the new inactive chart is still naturally represented,
2. in `sparse`, the new inactive chart remains block-local on $\Omega_b$, but
   generally no longer preserves the original per-orbital support pattern.

This is why the orthogonal inactive chart is compatible with block sparsity,
but not generally with the original fixed sparse slot pattern for each
individual inactive orbital.

## 8. Fixed-Upstream Pullback Derivative in the Orthogonal Inactive Chart

The previous section gives the accepted-point pullback itself. For a Hessian
vector product we also need its directional derivative while the upstream
adjoints are held fixed.

Again drop the block label and treat

$$
G_A \in \mathbb{R}^{N_b \times n_a},
\qquad
G_P \in \mathbb{R}^{N_b \times N_b}
$$

as fixed accepted-point upstream objects.

Define

$$
\widetilde G_P
=
G_P - G_A (S C_a)^{\mathrm T},
\qquad
G_{\mathrm{sym}}
=
\widetilde G_P + \widetilde G_P^{\mathrm T}.
$$

Then the accepted-point pullback is

$$
\bar C_a
=
\left(I - S Q_i Q_i^{\mathrm T}\right) G_A,
$$

$$
\bar Q_i
=
G_{\mathrm{sym}} Q_i.
$$

### 8.1 Directional derivative of the active pullback

Because $G_A$ is fixed,

$$
\delta \bar C_a
=
- S \, \delta Q_i \, Q_i^{\mathrm T} G_A
- S \, Q_i \, \delta Q_i^{\mathrm T} G_A.
$$

### 8.2 Directional derivative of the inactive pullback

Since only $C_a$ varies inside $\widetilde G_P$,

$$
\delta \widetilde G_P
=
- G_A (S \delta C_a)^{\mathrm T}
=
- G_A \delta C_a^{\mathrm T} S.
$$

Therefore

$$
\delta G_{\mathrm{sym}}
=
- G_A \delta C_a^{\mathrm T} S
- S \delta C_a G_A^{\mathrm T}.
$$

Using $\bar Q_i = G_{\mathrm{sym}} Q_i$, we obtain

$$
\delta \bar Q_i
=
\delta G_{\mathrm{sym}} \, Q_i
+ G_{\mathrm{sym}} \, \delta Q_i,
$$

that is,

$$
\boxed{
\delta \bar Q_i
=
G_{\mathrm{sym}} \, \delta Q_i
- G_A \delta C_a^{\mathrm T} S Q_i
- S \delta C_a G_A^{\mathrm T} Q_i.
}
$$

This is the orthogonal-chart analogue of the current fixed-upstream formulas.
The important difference is that no $M^{-1}$ or $dM^{-1}$ terms survive.

## 9. Direction Transport from Physical Space to the Orthogonal Working Chart

For HVP use, the physical direction still enters through the block-local
embedded orbital directions

$$
\delta c_{b,p} = U_{b,p} \delta x_{b,p}.
$$

If active orbitals remain physical variables, then

$$
\delta C_{a,b}
$$

is obtained by stacking those embedded active directions.

If inactive orbitals are reparameterized directly in the orthogonal chart, then
the working direction is already

$$
\delta Q_{i,b}.
$$

In that case, the accepted-point second-order action in block $b$ can be
assembled entirely from the formulas in Sections 5 and 8, then scattered back
to whichever external parameter vector is chosen for the optimizer.

## 10. Consequence for Sparse-Orbital Mode

The orthogonal inactive gauge is mathematically compatible with `sparse`
orbitals only in the following sense:

1. each block still lives on a finite union support $\Omega_b$,
2. all inactive orthogonalization and HVP formulas remain local to that block,
3. but the inactive working columns generally become dense on $\Omega_b$ after
   multiplication by $M_b^{-1/2}$.

So the correct interpretation is:

$$
\boxed{
\text{the orthogonal inactive chart preserves block sparsity, but not
per-orbital fixed support sparsity in general.}
}
$$

This is not a contradiction. It only means that the optimizer chart must be
changed from the old "each orbital keeps its own fixed sparse slots" view to a
new block-local working chart.

## 11. Practical Summary

If one wants the orthogonal inactive gauge to produce a real TNHVP
simplification, then the implementation should follow this logic:

1. gather both `full-AO` and `sparse` physical orbitals into the common
   block-local AO domain $\Omega_b$,
2. use $Q_{i,b}$, not $C_{i,b}$, as the inactive working variable,
3. build
   $$
   T_{a,b}
   =
   \left(I - Q_{i,b} Q_{i,b}^{\mathrm T} S_b\right) C_{a,b},
   $$
4. apply the pullback and fixed-upstream derivative formulas above,
5. only at the outer interface scatter gradients or steps back to the chosen
   physical parameterization.

If step 2 is skipped, then the apparent projector simplification is only
superficial.
