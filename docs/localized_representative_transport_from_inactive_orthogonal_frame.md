# Localized Representative Transport From An Inactive-Orthogonal Working Frame

## 1. Goal

This note derives the exact coordinate relations needed if TNHVP uses an
inactive-orthogonal **internal working frame**, while the optimizer and Molden
output still need a **localized physical representative**.

The guiding constraint is specific to VB:

$$
\boxed{
\text{the internal chart may change, but the occupied-orbital picture seen by
the user must stay on a localized representative.}
}
$$

Therefore the real question is not merely whether the inactive block can be
orthogonalized. The real question is:

1. which representative changes are exact gauge changes of the current OEO
   occupied chart,
2. how to reconstruct a localized physical representative from the internal
   working frame,
3. how gradients and accepted-point history should be transported across that
   chart change.

## 2. Occupied Representative Changes That Are Exactly Harmless

Let

$$
C_i \in \mathbb{R}^{N \times n_i},
\qquad
C_a \in \mathbb{R}^{N \times n_a},
\qquad
S \in \mathbb{R}^{N \times N},
$$

and define the current OEO inactive projector

$$
P_i = C_i (C_i^{\mathrm T} S C_i)^{-1} C_i^{\mathrm T}.
$$

The active auxiliary block is

$$
T_a = (I - P_i S) C_a.
$$

Now consider the two-parameter representative change

$$
\boxed{
\hat C_i = C_i U,
\qquad
\hat C_a = C_a + C_i \Lambda,
}
$$

with

$$
U \in GL(n_i),
\qquad
\Lambda \in \mathbb{R}^{n_i \times n_a}.
$$

### 2.1 Inactive projector invariance

Because

$$
\hat C_i^{\mathrm T} S \hat C_i
=
U^{\mathrm T} (C_i^{\mathrm T} S C_i) U,
$$

we obtain

$$
\hat P_i
=
\hat C_i
\left(
\hat C_i^{\mathrm T} S \hat C_i
\right)^{-1}
\hat C_i^{\mathrm T}
=
C_i U
\left(
U^{\mathrm T} (C_i^{\mathrm T} S C_i) U
\right)^{-1}
U^{\mathrm T} C_i^{\mathrm T}
=
P_i.
$$

So the inactive span may be reparameterized by any invertible right transform
without changing the projector.

### 2.2 Active auxiliary invariance

Since $\hat P_i = P_i$,

$$
(I - \hat P_i S)\hat C_a
=
(I - P_i S)(C_a + C_i \Lambda)
=
(I - P_i S) C_a + (I - P_i S) C_i \Lambda.
$$

But

$$
(I - P_i S) C_i = 0,
$$

therefore

$$
\boxed{
(I - \hat P_i S)\hat C_a = T_a.
}
$$

So two kinds of representative freedom are exact in the present OEO occupied
chart:

1. an internal right transform of the inactive block,
2. adding inactive components onto the active physical orbitals.

This is the exact chart-change structure that can be used to preserve a
localized occupied-orbital picture without changing the physical point seen by
the VB objective.

## 3. Internal Working Frame And External Localized Representative

Introduce the inactive-orthogonal working frame

$$
Q_i = C_i R_i,
\qquad
R_i = (C_i^{\mathrm T} S C_i)^{-1/2}.
$$

Then

$$
Q_i^{\mathrm T} S Q_i = I,
$$

and the inactive projector becomes

$$
P_i = Q_i Q_i^{\mathrm T}.
$$

Define the internal active auxiliary block

$$
T_a = (I - Q_i Q_i^{\mathrm T} S) C_a.
$$

The key point is that **the working frame** $(Q_i, T_a)$ and **the physical
localized representative** $(C_i, C_a)$ should be treated as different
coordinate descriptions of the same accepted point.

### 3.1 Reconstruction form

Given $(Q_i, T_a)$, every physical representative of the same OEO occupied
point can be written as

$$
\boxed{
C_i = Q_i U_i,
\qquad
C_a = T_a + Q_i K_a,
}
$$

with

$$
U_i \in GL(n_i),
\qquad
K_a \in \mathbb{R}^{n_i \times n_a}.
$$

This is just the same exact gauge freedom from Section 2, rewritten around the
inactive-orthogonal frame.

### 3.2 Proof of completeness

First, any pair $(U_i, K_a)$ gives a valid physical representative because

$$
P_i
=
C_i (C_i^{\mathrm T} S C_i)^{-1} C_i^{\mathrm T}
=
Q_i U_i
\left(
U_i^{\mathrm T} Q_i^{\mathrm T} S Q_i U_i
\right)^{-1}
U_i^{\mathrm T} Q_i^{\mathrm T}
=
Q_i Q_i^{\mathrm T},
$$

and

$$
(I - P_i S) C_a
=
(I - Q_i Q_i^{\mathrm T} S)(T_a + Q_i K_a)
=
T_a.
$$

Conversely, if $(C_i, C_a)$ is any physical representative with inactive
projector $P_i = Q_i Q_i^{\mathrm T}$ and active auxiliary block $T_a$, then
$C_i$ spans the same inactive space as $Q_i$, so there exists an invertible
$U_i$ such that $C_i = Q_i U_i$. Also

$$
C_a - T_a \in \operatorname{span}(Q_i),
$$

so there exists $K_a$ such that

$$
C_a = T_a + Q_i K_a.
$$

Hence the parametrization above is exact.

## 4. What Can And Cannot Be Used To Preserve Localized Orbitals

The decomposition

$$
C_a = T_a + Q_i K_a
$$

shows that the active physical picture has at least two different issues:

1. the inactive-null component $Q_i K_a$,
2. the internal active basis inside $T_a$ itself.

Only the first one is covered by the exact representative freedom derived
above. That is:

$$
\boxed{
K_a \text{ may be reset exactly without changing } T_a,
\text{ but general active-active mixing inside } T_a
\text{ is not included here.}
}
$$

This is important for VB. Localized active orbitals should not be entrusted to
an unconstrained active-active gauge drift and then "repaired" only at export
time.

## 5. Deterministic Representative Selection Map

To keep the physical occupied orbitals localized while the internal working
frame may drift, introduce a deterministic representative selector

$$
\Sigma(Q_i, T_a; \mathcal{R}) = (U_i, K_a),
$$

where $\mathcal{R}$ is reference data such as:

1. original support metadata,
2. initial localized accepted point,
3. a support-aware or localization-based target.

Then the physical representative is reconstructed as

$$
C_i^{\mathrm{loc}} = Q_i U_i,
\qquad
C_a^{\mathrm{loc}} = T_a + Q_i K_a.
$$

If $\Sigma$ is locally differentiable and its Jacobian is known, then
accepted-point canonicalization is a true chart change. If $\Sigma$ is treated
only as a postprocessing rule, then the optimizer history remains in the wrong
chart.

## 6. Norm-Preserving Choice For The Active Inactive-Null Component

One practical requirement is that the physical active orbitals should keep
their usual metric normalization.

For one active orbital, write

$$
c_a = t_a + Q_i k_a,
$$

where

$$
t_a^{\mathrm T} S t_a = \|t_a\|_S^2,
\qquad
Q_i^{\mathrm T} S Q_i = I,
\qquad
Q_i^{\mathrm T} S t_a = 0.
$$

Then

$$
\|c_a\|_S^2
=
c_a^{\mathrm T} S c_a
=
\|t_a\|_S^2 + \|k_a\|_2^2.
$$

So unit normalization requires

$$
\boxed{
\|k_a\|_2^2 = 1 - \|t_a\|_S^2.
}
$$

Now suppose a localized reference orbital $c_a^{\mathrm{ref}}$ is available.
Project its inactive component onto the current orthogonal inactive frame:

$$
r_a = Q_i^{\mathrm T} S c_a^{\mathrm{ref}}.
$$

If $r_a \neq 0$, a simple metric-preserving representative rule is

$$
\boxed{
k_a
=
\alpha_a r_a,
\qquad
\alpha_a
=
\sqrt{
\frac{1 - \|t_a\|_S^2}{r_a^{\mathrm T} r_a}
}.
}
$$

This keeps the current auxiliary orbital $t_a$ exactly fixed while steering
the physical inactive component toward the reference localized direction.

This is the orthogonal-frame analogue of the current
`build_metric_preserving_inactive_repaired_active_physical_orbitals(...)`
logic in
[cpp_vb_scf_optimizer.cpp](/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/cpp_vb_scf_optimizer.cpp).

## 7. Exact Differential Relations For Pure Representative Changes

Now derive the coordinate transport formulas needed when the representative is
changed but the physical point is not.

### 7.1 Physical representative form

Take the exact representative change

$$
\hat C_i = C_i U,
\qquad
\hat C_a = C_a + C_i \Lambda,
$$

with fixed $U$ and $\Lambda$ at the accepted point.

Then the corresponding tangent transport is

$$
\boxed{
d\hat C_i = dC_i \, U,
\qquad
d\hat C_a = dC_a + dC_i \, \Lambda.
}
$$

### 7.2 Gradient transport in the physical representative chart

Let the old-chart gradients be $G_{C_i}$ and $G_{C_a}$, and the new-chart
gradients be $\hat G_{C_i}$ and $\hat G_{C_a}$. Requiring

$$
\langle G_{C_i}, dC_i \rangle + \langle G_{C_a}, dC_a \rangle
=
\langle \hat G_{C_i}, d\hat C_i \rangle
+ \langle \hat G_{C_a}, d\hat C_a \rangle
$$

for all $(dC_i, dC_a)$ gives

$$
\boxed{
\hat G_{C_a} = G_{C_a},
}
$$

and

$$
\boxed{
\hat G_{C_i}
=
\left(
G_{C_i} - G_{C_a} \Lambda^{\mathrm T}
\right)
U^{-{\mathrm T}}.
}
$$

This is the exact inactive-plus-active-null chart transport formula.

### 7.3 Special cases

If only the inactive representative is rotated,

$$
\hat C_i = C_i U,
\qquad
\hat C_a = C_a,
$$

then

$$
\hat G_{C_i} = G_{C_i} U^{-{\mathrm T}},
\qquad
\hat G_{C_a} = G_{C_a}.
$$

This is exactly the rule already used in
[support_aware_mo_gauge_fix.hpp](/pool1/home/xiatao/project/xmvb-cpp/src/vb/orbital/support_aware_mo_gauge_fix.hpp).

If only the active inactive-null component is reset,

$$
\hat C_i = C_i,
\qquad
\hat C_a = C_a + C_i \Lambda,
$$

then

$$
\hat G_{C_a} = G_{C_a},
\qquad
\hat G_{C_i} = G_{C_i} - G_{C_a} \Lambda^{\mathrm T}.
$$

So even an "active representative repair" feeds back into the inactive
gradient block if it is treated as a true chart change.

## 8. Differential Relations Between The Internal Frame And The Localized Representative

Now return to the internal-frame reconstruction

$$
C_i = Q_i U_i,
\qquad
C_a = T_a + Q_i K_a.
$$

If $U_i$ and $K_a$ are frozen at the accepted point, then the differential
relation is

$$
\boxed{
dC_i = dQ_i \, U_i,
\qquad
dC_a = dT_a + dQ_i \, K_a.
}
$$

### 8.1 Forward gradient map: physical representative to internal frame

Let the localized physical representative gradients be $G_{C_i}$ and
$G_{C_a}$. Then

$$
\delta E
=
\langle G_{C_i}, dQ_i U_i \rangle
+
\langle G_{C_a}, dT_a + dQ_i K_a \rangle.
$$

Therefore

$$
\boxed{
G_{T_a} = G_{C_a},
}
$$

and

$$
\boxed{
G_{Q_i}
=
G_{C_i} U_i^{\mathrm T}
+ G_{C_a} K_a^{\mathrm T}.
}
$$

So the internal inactive-frame gradient receives contributions from both the
localized inactive representative and the localized active inactive-null
components.

### 8.2 Backward gradient map: internal frame to localized representative

Conversely, if $G_{Q_i}$ and $G_{T_a}$ are first computed in the internal
frame, then the corresponding localized physical representative gradients are

$$
\boxed{
G_{C_a} = G_{T_a},
}
$$

and

$$
\boxed{
G_{C_i}
=
\left(
G_{Q_i} - G_{T_a} K_a^{\mathrm T}
\right)
U_i^{-{\mathrm T}}.
}
$$

This is the direct transport formula needed if TNHVP acts on $(Q_i, T_a)$ but
the optimizer history and output chart are stored in localized physical
orbitals.

## 9. Accepted-Point Chart Reset And History Transport

Suppose that after an accepted step the internal frame $(Q_i, T_a)$ is kept,
but the localized selector changes from $(U_i, K_a)$ to
$(\hat U_i, \hat K_a)$.

Then the physical representative changes as

$$
\hat C_i = Q_i \hat U_i = C_i U_i^{-1} \hat U_i,
$$

$$
\hat C_a = T_a + Q_i \hat K_a
=
C_a + C_i U_i^{-1} (\hat K_a - K_a).
$$

Therefore the exact accepted-point representative transport is obtained by the
specialization

$$
\boxed{
U_{\mathrm{rep}} = U_i^{-1} \hat U_i,
\qquad
\Lambda_{\mathrm{rep}} = U_i^{-1} (\hat K_a - K_a).
}
$$

Once $U_{\mathrm{rep}}$ and $\Lambda_{\mathrm{rep}}$ are known, the gradient
transport is exactly

$$
\hat G_{C_a} = G_{C_a},
$$

$$
\boxed{
\hat G_{C_i}
=
\left(
G_{C_i} - G_{C_a} \Lambda_{\mathrm{rep}}^{\mathrm T}
\right)
U_{\mathrm{rep}}^{-{\mathrm T}}.
}
$$

Likewise, accepted-step displacements in the localized physical chart satisfy

$$
d\hat C_i = dC_i U_{\mathrm{rep}},
\qquad
d\hat C_a = dC_a + dC_i \Lambda_{\mathrm{rep}}.
$$

So transported secant pairs must be updated with the same Jacobian; they
cannot simply be reused unchanged if the localized representative is reset.

## 10. What Is Still Missing For A Full Strict Solution

The formulas in this note settle the exact transport structure for:

1. inactive internal gauge changes,
2. active physical representative changes that only modify the inactive-null
   component.

They do **not** settle a strict chart transport for general active-active
changes inside $T_a$. That part remains a separate covariance question.

So the current state of the problem is:

$$
\boxed{
\text{inactive gauge transport is exact;}
\quad
\text{inactive-null active representative transport is exact;}
\quad
\text{general active-active representative transport is still unresolved.}
}
$$

## 11. Practical Consequence For TNHVP

If the future implementation wants both of the following:

1. use an inactive-orthogonal working frame to simplify the TNHVP formulas,
2. preserve the localized VB occupied-orbital picture during optimization,

then the clean route is:

1. run TNHVP internally on $(Q_i, T_a)$,
2. store a deterministic localized selector $(U_i, K_a)$,
3. reconstruct localized physical orbitals by
   $$
   C_i = Q_i U_i,
   \qquad
   C_a = T_a + Q_i K_a,
   $$
4. transport gradients and accepted-point history with the formulas in
   Sections 7 to 9 whenever the selector changes.

Without that transport, "internal orthogonal frame + external localized
orbitals" is only a visualization trick, not a true optimizer chart.

## 12. Mapping To The Current Code

This section records how the abstract objects in this note already map onto the
present C++ implementation, and which pieces are still missing.

### 12.1 Accepted-point internal frame objects already cached

The accepted-point orbital preparer already stores the main internal-frame
objects needed by this formulation:

1. $C_i$:
   `physical_orbital_frame.inactive_physical_orbital_matrix`
2. $Q_i$:
   `physical_orbital_frame.inactive_orthonormal_orbital_matrix`
3. $R_i = (C_i^T S C_i)^{-1/2}$:
   `physical_orbital_frame.inactive_orthonormal_gauge_transform`
4. $C_a$:
   `physical_orbital_frame.active_physical_orbital_matrix`
5. $T_a$:
   the occupied-active block of
   `orbital_preparation_result.auxiliary_orbital_matrix`

So the internal inactive-orthogonal frame itself is not hypothetical anymore.
Its accepted-point state is already materialized by
`ActiveSpaceOrbitalPreparer`.

### 12.2 Current exact inactive chart transport

The file
`src/vb/orbital/support_aware_mo_gauge_fix.cpp`
already implements the special case

$$
\hat C_i = C_i U,
\qquad
\hat C_a = C_a,
$$

together with the exact gradient transport

$$
\hat G_{C_i} = G_{C_i} U^{-T}.
$$

In the code this is represented by

1. `right_transform`, which stores $U$,
2. `inverse_transpose_right_transform`, which stores $U^{-T}$.

Therefore the inactive part of the present accepted-point chart-reset
machinery already matches the theory in Section 7.

### 12.3 Current active representative reset corresponds to changing $K_a$

The current OEO active repair path in
`cpp_vb_scf_optimizer.cpp`
does not rotate the active auxiliary block itself. Instead, it keeps the
current auxiliary orbitals fixed and reconstructs new physical active orbitals
by adding an inactive component chosen from a localized reference.

That is exactly the representative structure

$$
C_a = T_a + Q_i K_a,
$$

or, in the current mixed-gauge implementation,

$$
C_a = T_a + C_i \Lambda.
$$

So the present accepted-point active repair is already operating in the
"inactive-null representative" sector described in this note.

### 12.4 What is still missing in the current active path

Even though the present active representative repair changes the physical
occupied chart, it is not yet implemented as a full exact chart transport.

The current accepted-point canonicalization does the following:

1. overwrite the sparse orbital parameters,
2. repack the parameter vector,
3. refresh some accepted-point orbital caches,
4. refresh the displayed gradient norm.

However, unlike the inactive chart-reset path, it does **not** yet apply an
analytic Jacobian transport to:

1. `last_gradient_result_.sparse_orbital_energy_gradient`,
2. `last_gradient_result_.sparse_orbital_reference_energy_gradient`,
3. the packed secant history used by the transported TN preconditioner.

So at present:

$$
\boxed{
\text{inactive chart reset is already an exact transported chart change,}
\quad
\text{active representative reset is not yet.}
}
$$

This is the main implementation gap exposed by the formulas in Sections 7 to
9.

### 12.5 Where the future Jacobian must act

The TN transport history is currently stored in the ambient packed chart as

$$
(\Delta x_{\mathrm{pack}}, \Delta g_{\mathrm{pack}}),
$$

through the structure

`PackedSecantPair { packed_step, packed_projected_gradient_change }`.

So once the representative selector changes from $(U_i, K_a)$ to
$(\hat U_i, \hat K_a)$, the future implementation must apply the corresponding
packed-chart Jacobian to:

1. the current packed parameter vector,
2. the current packed gradient,
3. every stored packed secant pair before it is reprojected into the current
   nonredundant space.

Otherwise the transported preconditioner will continue to replay history from a
different physical representative chart.

## 13. Immediate Implementation Implications

The code mapping above suggests the following order of work.

### 13.1 First missing object: an explicit localized selector state

The accepted-point result already stores $Q_i$, $R_i$, $C_i$, $C_a$, and
$T_a$, but it does not yet store the representative selector explicitly as

$$
(U_i, K_a).
$$

For future exact chart transport, these should become accepted-point state
objects rather than temporary reconstruction byproducts.

### 13.2 Second missing step: active representative Jacobian transport

Once $(U_i, K_a)$ is explicit, the active representative reset can be upgraded
from a parameter overwrite to a true chart change by applying

$$
\hat G_{C_i}
=
\left(
G_{C_i} - G_{C_a} \Lambda_{\mathrm{rep}}^T
\right)
U_{\mathrm{rep}}^{-T},
\qquad
\hat G_{C_a} = G_{C_a},
$$

and the corresponding packed-chart map.

### 13.3 Third missing step: transported secant-pair update

Only after the active Jacobian is available does it make sense to transport the
stored TN secant pairs across accepted-point representative resets.

So the practical dependency is

$$
\boxed{
\text{selector state}
\;\Longrightarrow\;
\text{gradient transport}
\;\Longrightarrow\;
\text{secant-history transport}.
}
$$
