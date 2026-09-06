# Nonredundant Orbital Parameterization in `xmvb-cpp`

## 1. Scope

This document describes the **current** nonredundant orbital parameterization
implemented in:

- `src/vb/orbital/nonredundant_orbital_space.hpp`
- `src/vb/orbital/nonredundant_orbital_space.cpp`
- `src/runtime/cpp_block_guess_builder.cpp`
- `src/vb/scf/cpp_vb_scf_optimizer.cpp`

The emphasis here is on the method that is actually used by the present
nonredundant optimizers. In particular:

1. The optimizer input is currently forwarded **unchanged** from the original
   legacy sparse chart.
2. The reduced nonredundant space is built on top of that original sparse
   chart.
3. The current nonredundant optimizer path uses `expand_step()` plus linear
   addition in packed sparse parameters for finite trial points.
4. The class also provides a Cayley-style `retract_step()` helper, but that is
   **not** the finite-step map used in the current main optimization loop.

## 2. Global Parameterization

Let the legacy sparse orbital coefficient table be viewed as a packed parameter
vector

$$
x \in \mathbb{R}^{n_{\mathrm{pack}}},
$$

where `SparseOrbitalParameterView` packs exactly the explicit sparse
coefficient slots appearing in `orbital_value_table`.

We denote by:

- \(n_{\mathrm{inact}}\): number of inactive doubly occupied orbitals,
- \(n_{\mathrm{act}}\): number of active orbitals,
- \(n_{\mathrm{occ}} = n_{\mathrm{inact}} + n_{\mathrm{act}}\): number of
  occupied orbitals used to build the nonredundant space.

The nonredundant orbital space introduces reduced coordinates

$$
\xi \in \mathbb{R}^{n_{\mathrm{nr}}},
$$

with a linear embedding

$$
\delta x = Q \, \delta \xi,
$$

where \(Q\) is never formed explicitly as one dense global matrix, but is
represented block by block.

## 3. Block Partition

### 3.1 Block detection

The orbital blocks are obtained by `detect_orbital_blocks()`:

1. If legacy block metadata is available, that metadata is used directly.
2. Otherwise, orbitals are grouped only when their explicit sparse AO supports
   are exactly equal.

Therefore, **partial support overlap does not merge orbitals into one block by
itself**.

If we denote the final block partition by

$$
\mathcal{B}_1, \mathcal{B}_2, \dots, \mathcal{B}_{N_{\mathrm{blk}}},
$$

then each \(\mathcal{B}_b\) is the block membership used by the current
nonredundant space.

### 3.2 Block AO domain

For a block \(\mathcal{B}_b\), the local AO domain is **not** the support of a
single representative orbital. Instead, the code uses the union support

$$
\Omega_b = \bigcup_{p \in \mathcal{B}_b} J_p,
$$

where \(J_p\) is the explicit AO support of orbital \(p\).

This is an important detail: the **block partition** is preserved, but the
**block-local AO rows** are built from the union support of all member
orbitals. This is how the current implementation handles partially overlapping
supports inside one legacy block without rewriting the legacy sparse chart.

## 4. AO-Metric Normalization of Sparse Orbitals

Before building the reduced space, each sparse orbital coefficient vector is
normalized in the AO overlap metric.

Let \(c_p\) denote the sparse AO coefficient vector of orbital \(p\), embedded
into its explicit AO support, and let \(S\) be the AO overlap matrix. The code
rescales each orbital so that

$$
\tilde{c}_p
=
\frac{c_p}{\sqrt{c_p^{\mathsf{T}} S c_p}},
\qquad
\tilde{c}_p^{\mathsf{T}} S \tilde{c}_p = 1.
$$

These normalized occupied orbitals are then used to build the block-local
nonredundant space.

## 5. Block-Local Occupied and Virtual Spaces

Consider one block \(b\). Restrict the occupied orbitals in that block to the
union AO domain \(\Omega_b\), and assemble

$$
C_b \in \mathbb{R}^{|\Omega_b| \times n_{\mathrm{occ},b}}.
$$

Here:

- \(n_{\mathrm{occ},b}\) is the number of occupied orbitals in block \(b\),
- \(n_{\mathrm{inact},b}\) is the number of inactive occupied orbitals in block
  \(b\),
- \(n_{\mathrm{act},b} = n_{\mathrm{occ},b} - n_{\mathrm{inact},b}\).

Let \(S_b\) be the AO overlap submatrix on \(\Omega_b\).

The current code constructs the block-local virtual complement using the
projector

$$
P_b
=
I
-
C_b
\left(C_b^{\mathsf{T}} S_b C_b\right)^{-1}
C_b^{\mathsf{T}} S_b.
$$

Then it diagonalizes

$$
P_b^{\mathsf{T}} S_b P_b
$$

and keeps the non-null eigenvectors to obtain an \(S_b\)-orthonormal virtual
basis

$$
V_b \in \mathbb{R}^{|\Omega_b| \times n_{\mathrm{vir},b}},
$$

with

$$
n_{\mathrm{vir},b} = |\Omega_b| - n_{\mathrm{occ},b}.
$$

## 6. Candidate Orbital-Rotation Directions

The current implementation retains the following block-local rotation classes:

1. inactive-active,
2. inactive-virtual,
3. active-virtual.

Equivalently, the code stores them as:

1. inactive-active amplitudes,
2. occupied-virtual amplitudes,

where the occupied-virtual part includes both inactive-virtual and
active-virtual rotations.

### 6.1 Local sampling and scattering

For each occupied orbital \(p\) in block \(b\), define:

- \(R_{b,p}\): restriction from a vector on \(\Omega_b\) to the explicit sparse
  coefficient slots of orbital \(p\), in exactly the storage order used by
  `orbital_value_table`,
- \(E_{b,p}\): scatter from the sparse slots of orbital \(p\) back into the
  global packed parameter vector.

These operators are represented in code by the per-orbital
`occupied_masked`, `virtual_masked`, and `packed_indices` data.

### 6.2 Inactive-active directions

For an inactive orbital \(i\) and an active orbital \(a\) in the same block,
the candidate direction is

$$
d_{ia}^{(b)}
=
E_{b,i} R_{b,i} C_b(:,a)
-
E_{b,a} R_{b,a} C_b(:,i).
$$

This is the usual antisymmetric occupied-occupied rotation written directly in
the packed sparse-coefficient chart.

### 6.3 Occupied-virtual directions

For an occupied orbital \(p\) and a virtual orbital \(u\) in block \(b\), the
candidate direction is

$$
d_{pu}^{(b)}
=
E_{b,p} R_{b,p} V_b(:,u).
$$

This covers both inactive-virtual and active-virtual couplings.

### 6.4 Candidate matrix

Collect all candidate directions in block \(b\) into

$$
D_b
=
\left[
\{d_{ia}^{(b)}\}_{i \in \mathcal{I}_b,\, a \in \mathcal{A}_b},
\{d_{pu}^{(b)}\}_{p \in \mathcal{O}_b,\, u \in \mathcal{V}_b}
\right]
\in
\mathbb{R}^{n_{\mathrm{pack}} \times m_b},
$$

with

$$
m_b
=
n_{\mathrm{inact},b} n_{\mathrm{act},b}
+
n_{\mathrm{occ},b} n_{\mathrm{vir},b}.
$$

This matches the code path

$$
n_{\mathrm{inact},b} n_{\mathrm{act},b}
+
n_{\mathrm{inact},b} n_{\mathrm{vir},b}
+
n_{\mathrm{act},b} n_{\mathrm{vir},b},
$$

because

$$
n_{\mathrm{occ},b} n_{\mathrm{vir},b}
=
n_{\mathrm{inact},b} n_{\mathrm{vir},b}
+
n_{\mathrm{act},b} n_{\mathrm{vir},b}.
$$

## 7. Gram-Based Removal of Redundancy

The candidate directions are generally linearly dependent or badly scaled in
the packed sparse chart. The current implementation removes this redundancy
through a block-local Gram decomposition.

For each block,

$$
G_b = D_b^{\mathsf{T}} D_b.
$$

The code diagonalizes

$$
G_b = U_b \Lambda_b U_b^{\mathsf{T}},
$$

discards eigenvalues below a numerical threshold, and defines the orthonormal
reduced basis

$$
Q_b = D_b U_b \Lambda_b^{-1/2}.
$$

The global nonredundant basis is the blockwise concatenation

$$
Q = [Q_1 \; Q_2 \; \cdots \; Q_{N_{\mathrm{blk}}}].
$$

Because all operations are implemented blockwise, the code never needs to
assemble one dense global \(Q\) explicitly.

## 8. Projection and Expansion

### 8.1 Projecting a packed vector into reduced coordinates

Given any packed vector \(g \in \mathbb{R}^{n_{\mathrm{pack}}}\), the reduced
coordinates are

$$
g_{\mathrm{nr}} = Q^{\mathsf{T}} g.
$$

Blockwise, this is implemented by first computing the candidate overlaps

$$
s_b = D_b^{\mathsf{T}} g,
$$

then transforming them into orthonormal reduced coordinates

$$
g_b^{\mathrm{nr}}
=
\Lambda_b^{-1/2} U_b^{\mathsf{T}} s_b.
$$

This is exactly what `project_vector()` computes.

### 8.2 Expanding a reduced step back to packed sparse parameters

Given a reduced step \(\delta \xi_b\) in block \(b\), the corresponding
candidate coefficients are

$$
a_b = U_b \Lambda_b^{-1/2} \delta \xi_b,
$$

and the lifted packed tangent step is

$$
\delta x_b = D_b a_b = Q_b \delta \xi_b.
$$

Summing over all blocks gives

$$
\delta x = Q \, \delta \xi.
$$

This is the map implemented by `expand_step()`.

## 9. Reduced Curvature Diagonal

When `ao_effective_h1e` is available, the code constructs a cheap reduced-space
curvature diagonal for preconditioning.

### 9.1 Raw candidate curvature

Let \(F_b\) be the block-local AO effective one-electron matrix. The occupied
and virtual orbital energies are approximated by

$$
\varepsilon_p = c_p^{\mathsf{T}} F_b c_p,
\qquad
\varepsilon_u = v_u^{\mathsf{T}} F_b v_u.
$$

Then the raw candidate curvature is defined as

$$
h_{ia}^{\mathrm{raw}}
=
\max(\eta_{\min}, |\varepsilon_a - \varepsilon_i|),
$$

$$
h_{pu}^{\mathrm{raw}}
=
\max(\eta_{\min}, |\varepsilon_u - \varepsilon_p|),
$$

with a small positive floor \(\eta_{\min}\).

### 9.2 Reduced diagonal

If

$$
Q_b = D_b U_b \Lambda_b^{-1/2},
$$

then the diagonal approximation used by the code is the diagonal of

$$
Q_b^{\mathsf{T}}
\operatorname{diag}(h_b^{\mathrm{raw}})
Q_b
=
\Lambda_b^{-1/2}
U_b^{\mathsf{T}}
\operatorname{diag}(h_b^{\mathrm{raw}})
U_b
\Lambda_b^{-1/2}.
$$

For reduced coordinate \(k\),

$$
h_{b,k}^{\mathrm{red}}
=
\max\!\left(
\eta_{\min}^{\mathrm{red}},
\frac{1}{\lambda_{b,k}}
\sum_{j=1}^{m_b} U_{b,jk}^2 h_{b,j}^{\mathrm{raw}}
\right).
$$

The code then normalizes this blockwise diagonal by its mean value and clamps
it into a bounded interval. This diagonal is used only as a cheap reduced-space
preconditioner, not as an exact Hessian model.

## 10. How the Current Optimizer Uses the Nonredundant Space

This is the most important implementation detail to keep straight.

### 10.1 What is preserved

The current helper `build_nonredundant_optimizer_input()` returns the original
input unchanged. Therefore:

1. the accepted-point sparse coefficient chart remains the legacy chart,
2. the block partition remains the original block partition,
3. the reduced nonredundant basis is built on top of that original chart.

### 10.2 What the main optimizer actually does

In the current nonredundant Armijo line search, the reduced direction
\(\delta \xi\) is first expanded into a packed tangent step

$$
\delta x = Q \, \delta \xi.
$$

The directional derivative is evaluated as

$$
g^{\mathsf{T}} \delta x.
$$

For a trial step length \(\alpha\), the finite trial point is currently
constructed as

$$
x_{\mathrm{trial}} = x_{\mathrm{current}} + \alpha \, \delta x.
$$

That is the behavior of
`try_build_nonredundant_lifted_trial_parameters()`.

So, in the **current production optimizer path**, the reduced coordinates are
used to generate better tangent directions, but the accepted finite point still
lives in the original packed sparse parameter chart via direct addition.

## 11. Cayley Retraction Helper Provided by the Space Object

Although the current optimizer does not use it, `NonredundantOrbitalSpace`
also provides a block-local finite-step helper `retract_step()`.

For one block, let:

- \(A_b\) be the inactive-active coefficient matrix,
- \(B_b\) be the occupied-virtual coefficient matrix.

The code builds an antisymmetric generator

$$
K_b =
\begin{bmatrix}
0      & -A_b^{\mathsf{T}} & -B_{I,b}^{\mathsf{T}} \\
A_b    & 0                 & -B_{A,b}^{\mathsf{T}} \\
B_{I,b} & B_{A,b}          & 0
\end{bmatrix},
$$

where \(B_b = [B_{I,b} \; B_{A,b}]\) is split by inactive and active occupied
columns.

It then applies a Cayley transform

$$
U_b
=
\left(I - \frac{1}{2} K_b\right)^{-1}
\left(I + \frac{1}{2} K_b\right),
$$

and keeps the first \(n_{\mathrm{occ},b}\) columns as the finite occupied
coordinates. These rotated occupied orbitals are finally sampled back onto each
orbital's explicit sparse rows.

This is a geometrically cleaner finite-step map in the block orbital chart, but
it is currently an auxiliary interface only; the main optimizer path does not
call it.

## 12. Summary

The current nonredundant orbital parameterization in `xmvb-cpp` can be
summarized as follows:

1. Preserve the original legacy sparse coefficient chart and block partition.
2. For each block, build the local AO domain from the union of all member
   supports.
3. Normalize the sparse occupied orbitals in the AO metric.
4. Construct a block-local virtual complement from the AO-metric projector.
5. Build inactive-active and occupied-virtual candidate directions directly in
   the packed sparse coefficient chart.
6. Remove linear redundancy by diagonalizing the candidate Gram matrix and
   forming an orthonormal reduced basis \(Q_b\).
7. Use `project_vector()` and `expand_step()` to move between packed sparse
   parameters and reduced nonredundant coordinates.
8. Optionally build a cheap reduced curvature diagonal from AO effective
   one-electron energy gaps.
9. In the current optimizer implementation, finite trial points are generated
   by linear addition in packed sparse parameters, not by the Cayley
   `retract_step()` helper.

This is the nonredundant parameterization currently used by the
`nonredundant_projected_gradient`, `nonredundant_lbfgspp`, and
`nonredundant_truncated_newton` backends.
