# `orbtyp=oeo` Public Orbital-Gradient Derivation

## 1. Scope

This note is only about the shared public `orbtyp=oeo` orbital-preparation and
orbital-gradient path:

1. the paper
   `Nonorthogonal orbital based N-body reduced density.pdf`,
2. legacy XMVB
   `OrbPrep_6.F90`, `gradient_rdm.F90`, `getOOgrad.F90`, `Iopt6.F90`,
3. the current C++ implementation
   `active_space_orbital_preparer.cpp`,
   `active_space_orbital_backpropagator.cpp`,
   `cpp_orbital_gradient_evaluator.cpp`.

The immediate goal is not to fix code here. The goal is to settle the formulas
first, so the later debug target is unambiguous.

The main conclusion of this note is:

$$
\boxed{
\text{the active pullback }
G_{C_a} = A_1^T G_{T_a}
\text{ is correct if }
G_{T_a}
\text{ is already the AO-coefficient gradient of the projected active block}
}
$$

Therefore the real open question is not the projector-chain rule by itself, but
whether the current C++ upstream object called
`active_auxiliary_gradient` is exactly the same mathematical object as the
legacy auxiliary-orbital coefficient gradient that appears after the
`A5` / `getOOgrad` transforms.

## 2. Notation

Throughout this note, orbitals are stored as columns of AO coefficient
matrices.

| Symbol | Dimension | Meaning |
| --- | --- | --- |
| $N$ | scalar | number of AO basis functions |
| $n_i$ | scalar | number of inactive doubly occupied orbitals |
| $n_a$ | scalar | number of active occupied orbitals |
| $n_v$ | scalar | number of virtual orbitals |
| $S$ | $N \times N$ | AO overlap matrix |
| $C_i$ | $N \times n_i$ | original physical inactive occupied orbitals |
| $C_a$ | $N \times n_a$ | original physical active occupied orbitals |
| $T_a$ | $N \times n_a$ | projected active auxiliary orbitals |
| $T_v$ | $N \times n_v$ | virtual auxiliary orbitals |
| $M_{ii}$ | $n_i \times n_i$ | inactive overlap, $M_{ii} = C_i^T S C_i$ |
| $P_i$ | $N \times N$ | inactive projector in AO coefficients, $P_i = C_i M_{ii}^{-1} C_i^T$ |
| $O$ | $N \times N$ | occupied-space projector, $O = I - P_i S$ |
| $A_i$ | $N \times n_i$ | inactive dual block, $A_i = C_i M_{ii}^{-1}$ |
| $G_{T_a}$ | $N \times n_a$ | gradient with respect to the AO coefficients of $T_a$ |
| $G_P$ | $N \times N$ | gradient with respect to $P_i$ |

The Frobenius inner product convention is

$$
\langle X, Y \rangle = \operatorname{tr}(X^T Y).
$$

Under this convention,

$$
\delta E = \langle G_X, \delta X \rangle
$$

defines the matrix gradient $G_X$.

## 3. Paper-Level AuO Construction

The paper states two invariances for a VB determinant built from occupied
orbitals:

1. adding inactive components onto active orbitals leaves the determinant
   invariant,
2. any nonsingular transformation inside the inactive and virtual subspaces
   leaves the determinant invariant.

This is the key reason why the auxiliary orbital construction is legal.

The paper introduces auxiliary orbitals $\bar \phi_q$ by

$$
\bar \phi_q = \sum_p B_{p q} \phi_p,
$$

with the inactive and active blocks chosen so that

$$
\bar s_{ij} = \delta_{ij},
\qquad
\bar s_{ia} = 0,
\qquad
\bar s_{it} = 0,
$$

while the active-active block remains nonorthogonal:

$$
\bar s_{tu} \neq \delta_{tu}
$$

in general.

This point matters later: the paper does **not** eliminate the active-active
gradient block. Its Eq. (48) is precisely the active-to-active orbital gradient
$\xi_{x \cdot t}$.

So the theory says:

$$
\boxed{
\text{active-active is present in the auxiliary-gradient theory}
}
$$

even if a later reduced optimizer may choose not to use active-active directions
as independent step variables.

## 4. Legacy XMVB Formulation

### 4.1 `OrbPrep6`: occupied auxiliary orbitals

Legacy `OrbPrep6` first orthonormalizes the inactive block:

$$
\bar C_i = C_i M_{ii}^{-1/2}.
$$

Then it projects the active physical orbitals out of the inactive subspace:

$$
T_a
=
C_a - C_i M_{ii}^{-1} C_i^T S C_a
=
(I - P_i S) C_a.
$$

This is exactly the projected active auxiliary block used later in the code.

In the occupied space, the legacy transform may be written as

$$
\bar C_{\mathrm{occ}}
=

[C_i, C_a]
\begin{bmatrix}
M_{ii}^{-1/2} & -M_{ii}^{-1/2} V_{ia} \\
0 & I
\end{bmatrix},
$$

where

$$
V_{ia} = M_{ii}^{-1/2} C_i^T S C_a.
$$

This produces

$$
\bar C_{\mathrm{occ}} = [\bar C_i, T_a].
$$

### 4.2 `gradient_rdm.F90`: the cached tensors `A1` to `A5`

Legacy `gradient_rdm.F90` builds the following objects:

$$
A_1 = I - P_i S,
$$

$$
A_3 = C_i M_{ii}^{-1} = A_i,
$$

$$
A_2 = A_3^T S C_a = M_{ii}^{-1} C_i^T S C_a,
$$

$$
A_4 = A_1^T S C_a = (I - S P_i) S C_a,
$$

$$
A_5 = T_{\mathrm{aux}}^{-1}.
$$

The current C++ code stores the same objects under different names:

| Legacy | Current C++ |
| --- | --- |
| `A1` | `occupied_space_projector` |
| `A3` | `inactive_auxiliary_transform` |
| `A2` | `inactive_active_overlap_matrix` |
| `A4` | `projected_active_overlap_matrix` |
| `A5` | `auxiliary_orbital_inverse_matrix` |

The useful identity

$$
S C_a = A_4 + S C_i A_2
$$

is also exactly what the current C++ code reconstructs as

$$
\texttt{basis\_overlap\_times\_active}
=
A_4 + (S C_i) A_2.
$$

### 4.3 `Grdori`: pull back from auxiliary orbitals to original occupied orbitals

Legacy `Grdori` does two logically separate things.

First, it converts the legacy auxiliary-orbital excitation gradients into an
AO-coefficient gradient on the active auxiliary block:

$$
G_{T_a}^{(\text{legacy aux-coef})}

=

A_5[\text{active rows}]^T G_{aa}

+

A_5[\text{virtual rows}]^T G_{va}.
$$

In the Fortran code this is the construction of `Grdaux`.

Second, it applies the chain rule from the mixed objects

$$
T_a = (I - P_i S) C_a,
\qquad
P_i = C_i M_{ii}^{-1} C_i^T
$$

back to the original physical orbitals $C_a$ and $C_i$. That produces the
legacy `Grdbas`.

So:

$$
\boxed{
\texttt{Grdori}

=

\text{(convert excitation gradient to coefficient gradient)}

\circ

\text{(pull back through } T_a, P_i \text{)}
}
$$

This separation is important, because the current C++ code performs the second
part explicitly, but usually bypasses the first part by differentiating one
layer deeper.

### 4.4 `getOOgrad`: contravariant active-active treatment

Legacy `getOOgrad.F90` builds

$$
S_{tu} = T_a^T S T_a,
\qquad
T_a^{\mathrm{ct}} = T_a S_{tu}^{-1},
$$

and then forms a current-state inverse-like transform `ootran2`.

The purpose of this step is not to introduce new physics. The purpose is to
convert an orbital-excitation gradient written in the auxiliary orbital basis,
whose active-active block is nonorthogonal, into the coefficient-gradient form
needed by the optimizer.

This is why explicit $S_{tu}^{-1}$ appears in the legacy path.

## 5. Current C++ Formulation

### 5.1 The current preparer uses a mixed gauge

The current C++ `ActiveSpaceOrbitalPreparer` does **not** store the legacy full
auxiliary basis $[\bar C_i, T_a, T_v]$ with orthonormal inactive columns.

Instead, it uses the mixed representation

$$
[C_i, T_a, T_v],
$$

while separately storing the dual inactive block

$$
A_i = C_i M_{ii}^{-1}.
$$

That is why the C++ preparer builds

$$
P_i = A_i C_i^T = C_i M_{ii}^{-1} C_i^T,
$$

and then forms the active auxiliary block as

$$
T_a = (I - P_i S) C_a.
$$

This mixed gauge is mathematically valid because the energy only needs

1. the inactive projector $P_i$,
2. the projected active block $T_a$,
3. the virtual complement orthogonal to the occupied auxiliary block.

### 5.2 The current C++ upstream gradient object

The current C++ active-space backpropagators do **not** start from the legacy
orbital-excitation gradient tensor.

Instead, they differentiate the active-space matrices directly with respect to
the AO coefficients of $T_a$:

$$
S_{aa} = T_a^T S T_a,
$$

$$
H_{aa} = T_a^T F_{11} T_a,
$$

$$
g_{tuvw}
=
\sum_{\mu \nu \lambda \sigma}
T_{\mu t} T_{\nu u} T_{\lambda v} T_{\sigma w}
(\mu \nu | \lambda \sigma).
$$

So the current C++ object named `active_auxiliary_gradient` is intended to be

$$
G_{T_a}
=
\frac{\partial E}{\partial T_a},
$$

that is, the AO-coefficient gradient of the projected active auxiliary block.

If that statement is true, then the later `A5`-style conversion has already been
absorbed upstream, and the downstream pullback should only differentiate through
$T_a$ and $P_i$.

## 6. Derivation of the Current Public Pullback

This section derives the formulas implemented in
`active_space_orbital_backpropagator.cpp`.

### 6.1 Chain rule for the active block

The public mixed-gauge variables are

$$
T_a = O C_a,
\qquad
O = I - P_i S.
$$

Therefore

$$
\delta T_a = O \, \delta C_a - \delta P_i \, S C_a.
$$

Assume the energy depends on the orbitals through

$$
E = E(T_a, P_i).
$$

Then

$$
\delta E
=
\langle G_{T_a}, \delta T_a \rangle
+
\langle G_P, \delta P_i \rangle.
$$

Substitute $\delta T_a$:

$$
\delta E
=
\langle G_{T_a}, O \, \delta C_a \rangle
-
\langle G_{T_a}, \delta P_i S C_a \rangle
+
\langle G_P, \delta P_i \rangle.
$$

The first term gives the active physical-orbital gradient:

$$
\langle G_{T_a}, O \, \delta C_a \rangle
=
\langle O^T G_{T_a}, \delta C_a \rangle,
$$

so

$$
\boxed{
G_{C_a} = O^T G_{T_a} = A_1^T G_{T_a}.
}
$$

This is exactly the current C++ line

$$
\texttt{original\_active\_gradient}
=
\texttt{occupied\_space\_projector}^T
\texttt{active\_auxiliary\_gradient}.
$$

The second term may be absorbed into an effective projector gradient:

$$
\langle G_{T_a}, \delta P_i S C_a \rangle
=
\langle G_{T_a} (S C_a)^T, \delta P_i \rangle.
$$

Hence

$$
\delta E
=
\langle G_{C_a}, \delta C_a \rangle
+
\langle \widetilde G_P, \delta P_i \rangle,
$$

with

$$
\boxed{
\widetilde G_P = G_P - G_{T_a} (S C_a)^T.
}
$$

Using the cached identity

$$
S C_a = A_4 + S C_i A_2,
$$

the code reconstructs this quantity as

$$
\widetilde G_P
=
G_P
-
G_{T_a}
\left(
A_4 + S C_i A_2
\right)^T.
$$

This is exactly the current
`effective_inactive_density_gradient`.

### 6.2 Chain rule for the inactive block

Now treat the projector dependence

$$
P_i = C_i M_{ii}^{-1} C_i^T,
\qquad
M_{ii} = C_i^T S C_i.
$$

Its variation is

$$
\delta P_i
=
\delta C_i M_{ii}^{-1} C_i^T
+
C_i M_{ii}^{-1} \delta C_i^T
+
C_i \, \delta(M_{ii}^{-1}) \, C_i^T.
$$

The inverse-overlap variation is

$$
\delta(M_{ii}^{-1})
=
-M_{ii}^{-1}
\left(
\delta C_i^T S C_i + C_i^T S \delta C_i
\right)
M_{ii}^{-1}.
$$

Insert this into

$$
\delta E_i = \langle \widetilde G_P, \delta P_i \rangle.
$$

The two explicit $\delta C_i$ terms give

$$
\langle \widetilde G_P + \widetilde G_P^T, \delta C_i M_{ii}^{-1} C_i^T \rangle
=
\left\langle
(\widetilde G_P + \widetilde G_P^T) C_i M_{ii}^{-1},
\delta C_i
\right\rangle.
$$

Define

$$
G_{M^{-1}}
=
-M_{ii}^{-1} C_i^T \widetilde G_P C_i M_{ii}^{-1}.
$$

Then the $\delta(M_{ii}^{-1})$ term contributes

$$
\left\langle
S C_i \left(G_{M^{-1}} + G_{M^{-1}}^T\right),
\delta C_i
\right\rangle.
$$

Therefore

$$
\boxed{
G_{C_i}
=
(\widetilde G_P + \widetilde G_P^T) C_i M_{ii}^{-1}
+
S C_i \left(G_{M^{-1}} + G_{M^{-1}}^T\right).
}
$$

This is exactly the formula implemented in the current C++ code:

$$
\texttt{effective\_inactive\_density\_gradient\_symmetric} \cdot A_i
+
(S C_i)\left(
\texttt{inactive\_overlap\_gradient}
+
\texttt{inactive\_overlap\_gradient}^T
\right).
$$

### 6.3 Orthogonal inactive gauge special case

If one uses the orthonormal inactive chart

$$
Q_i = C_i M_{ii}^{-1/2},
\qquad
P_i = Q_i Q_i^T,
$$

then

$$
\delta P_i = \delta Q_i Q_i^T + Q_i \delta Q_i^T.
$$

So

$$
\delta E_i
=
\langle \widetilde G_P + \widetilde G_P^T, \delta Q_i Q_i^T \rangle
=
\left\langle
(\widetilde G_P + \widetilde G_P^T) Q_i,
\delta Q_i
\right\rangle.
$$

Hence

$$
\boxed{
G_{Q_i} = (\widetilde G_P + \widetilde G_P^T) Q_i.
}
$$

This is the current orthonormal-inactive fast path.

## 7. Relation to Paper Eq. (54)

The paper’s Eq. (54) is the transformation from auxiliary-orbital gradients back
to gradients with respect to original nonorthogonal orbital coefficients.

In matrix form, its structure is

$$
G_C^{(\text{paper})}
=
\bar T^{-T} \, \bar \Xi \, B^T,
$$

where

1. $\bar \Xi$ is the orbital gradient expressed in the auxiliary-orbital
   excitation basis,
2. $\bar T^{-1}$ converts from excitation parameters to AO coefficient
   gradients of the auxiliary orbitals,
3. $B$ converts from auxiliary orbitals back to original VBO columns.

This is the theoretical origin of the legacy `A5` / `OOTran` /
`ootran2` machinery.

However, the current C++ code usually does **not** construct $\bar \Xi$ first.
It directly differentiates the active-space matrices with respect to the AO
coefficients of $T_a$.

So if the current C++ upstream quantity is already

$$
G_{T_a}
=
\frac{\partial E}{\partial T_a},
$$

then the paper Eq. (54) conversion has already happened implicitly inside the
upstream backpropagation, and the remaining public pullback is only

$$
G_{C_a} = A_1^T G_{T_a},
$$

together with the projector term for $C_i$.

Therefore:

$$
\boxed{
\text{Eq. (54) is not missing if the upstream object is already a coefficient gradient.}
}
$$

But:

$$
\boxed{
\text{Eq. (54) \emph{is} missing if the upstream object is still an excitation-basis gradient.}
}
$$

This is the exact theoretical fork that must now be checked numerically.

## 8. Nonredundant Variables Versus Full Gradient Theory

Legacy `OrbDivide.F90` uses reduced nonredundant variables only for

1. inactive-active,
2. inactive-virtual,
3. active-virtual

directions.

That fact should **not** be misread as saying active-active gradients vanish.

The correct statement is:

$$
\boxed{
\text{active-active exists in the full auxiliary-gradient theory,}
}
$$

but the legacy nonredundant optimizer chooses not to keep active-active as an
independent reduced step variable.

These are different statements.

## 9. What Is Already Settled

The following statements are now on solid ground.

1. The paper and the legacy code both support the projected active block

   $$
   T_a = (I - P_i S) C_a.
   $$

2. The paper does not justify deleting active-active from the gradient theory.
   Its Eq. (48) is the active-active block.

3. The current C++ formulas

   $$
   G_{C_a} = A_1^T G_{T_a}
   $$

   and

   $$
   \widetilde G_P = G_P - G_{T_a}(S C_a)^T
   $$

   are the correct chain rule for the mixed variables
   $E(T_a, P_i)$.

4. The current inactive pullback formula is the correct derivative of

   $$
   P_i = C_i (C_i^T S C_i)^{-1} C_i^T.
   $$

5. The absence of explicit `A5` in the current C++ public pullback is not, by
   itself, a proof of a bug. It depends entirely on what object the upstream
   C++ backpropagators are returning.

## 10. What Is Still Not Proved

The remaining unresolved question is now very specific:

$$
\boxed{
\text{Is the current C++ }
\texttt{total\_active\_auxiliary\_gradient}
\text{ exactly the same object as the legacy }
G_{T_a}^{(\text{legacy aux-coef})}?
}
$$

This is the only high-value theoretical/debug gap left in the public path.

If the answer is yes, then the public pullback formulas are probably not the
root cause of the FeCl / TiCl open-shell orbital mismatch.

If the answer is no, then the bug is upstream of
`active_space_orbital_backpropagator.cpp`, in the semantic layer that should
match legacy `Grdaux` / `getOOgrad`.

## 11. Immediate Debug Target

The next debug step should be numerical, not conceptual.

For the same accepted orbital state, compare:

1. the legacy auxiliary coefficient gradient after
   `Grdori` / `getOOgrad`,
2. the current C++ `total_active_auxiliary_gradient`,
3. the current C++ final physical-orbital gradient after the public pullback.

The decision tree is then:

1. if items 1 and 2 already differ, the bug is in the current upstream
   active-space backprop semantics;
2. if items 1 and 2 match but item 3 differs, the bug is in the public pullback
   or sparse-slot mapping;
3. if all three match, the orbital mismatch must come from the optimizer chart
   or parameterization, not from the public gradient formulas themselves.

