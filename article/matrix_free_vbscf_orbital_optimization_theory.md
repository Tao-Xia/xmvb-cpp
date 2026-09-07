# Matrix-Free Second-Order Orbital Optimization for Strictly Sparse Nonorthogonal VBSCF Wave Functions

## Scope and status

This document establishes the mathematical formulation underlying a general matrix-free second-order orbital optimizer for valence-bond self-consistent-field (VBSCF) wave functions. Particular attention is paid to three features that distinguish the present problem from conventional orthogonal molecular-orbital optimization:

1. the variational orbitals are nonorthogonal;
2. every orbital has an immutable, strictly sparse atomic-orbital (AO) support; and
3. the VB structure coefficients are relaxed at every orbital geometry.

The central objective is to obtain Newton-like local convergence without constructing or storing the orbital Hessian. The derivation below separates the physical orbital manifold, its gauge redundancies, the relaxed electronic Hessian, and its matrix-free action. Statements concerning the current implementation are collected separately in Section 11 and should not be interpreted as part of the formal theory.

## 1. Variational VBSCF energy

Let $\mathbf S$ denote the AO overlap matrix and let

$$
\mathbf C = \left(\mathbf C_{\mathrm I},\mathbf C_{\mathrm A}\right)
\tag{1}
$$

collect the inactive and active orbital coefficients. The inactive block $\mathbf C_{\mathrm I}\in\mathbb R^{n_{\mathrm{AO}}\times n_{\mathrm I}}$ contains the orbitals that are doubly occupied in every VB structure, whereas $\mathbf C_{\mathrm A}\in\mathbb R^{n_{\mathrm{AO}}\times n_{\mathrm A}}$ contains the active VB orbitals.

For a fixed orbital set, the VB structure coefficients $\mathbf a$ satisfy the generalized eigenvalue problem

$$
\mathbf H_{\mathrm{VB}}(\mathbf C)\mathbf a
=
E(\mathbf C)\mathbf S_{\mathrm{VB}}(\mathbf C)\mathbf a,
\tag{2}
$$

with normalization

$$
\mathbf a^{\mathrm T}\mathbf S_{\mathrm{VB}}(\mathbf C)\mathbf a=1.
\tag{3}
$$

Here, $\mathbf H_{\mathrm{VB}}$ and $\mathbf S_{\mathrm{VB}}$ are the Hamiltonian and overlap matrices in the nonorthogonal VB structure basis. For an isolated state, the orbitally relaxed energy is

$$
\mathcal E(\mathbf C)
=
\min_{\mathbf a}
\frac{
\mathbf a^{\mathrm T}\mathbf H_{\mathrm{VB}}(\mathbf C)\mathbf a
}{
\mathbf a^{\mathrm T}\mathbf S_{\mathrm{VB}}(\mathbf C)\mathbf a
}.
\tag{4}
$$

For a state-averaged calculation with normalized weights $w_s$,

$$
\mathcal E_{\mathrm{SA}}(\mathbf C)
=
\sum_s w_s E_s(\mathbf C),
\qquad
w_s\geq 0,
\qquad
\sum_s w_s=1.
\tag{5}
$$

The orbital gradient and Hessian discussed below are derivatives of the relaxed energy in eq 4 or eq 5, not derivatives evaluated at fixed VB structure coefficients.

## 2. Strictly sparse orbital parameterization

Each orbital $p$ has a fixed AO support $\mathcal S_p$. Let $\mathbf R_p$ inject its stored sparse coefficient vector $\mathbf x_p\in\mathbb R^{m_p}$ into the full AO space:

$$
\mathbf c_p=\mathbf R_p\mathbf x_p,
\qquad
\operatorname{supp}(\mathbf c_p)\subseteq\mathcal S_p.
\tag{6}
$$

The complete packed parameter vector is

$$
\mathbf x
=
\left(
\mathbf x_1^{\mathrm T},\ldots,\mathbf x_{n_{\mathrm{orb}}}^{\mathrm T}
\right)^{\mathrm T}
\in\mathbb R^m,
\qquad
m=\sum_p m_p.
\tag{7}
$$

Because the supports are immutable, the feasible raw tangent space is linear:

$$
\mathcal T_{\mathrm{sp}}
=
\left\{
\delta\mathbf C:
\operatorname{supp}(\delta\mathbf c_p)\subseteq\mathcal S_p
\text{ for every }p
\right\}.
\tag{8}
$$

Strict sparsity is therefore not a numerical truncation and must not be imposed by thresholding a dense step. It is part of the definition of the variational manifold.

Individual orbitals may be normalized before integral construction. With

$$
n_p=\left(\mathbf c_p^{\mathrm T}\mathbf S\mathbf c_p\right)^{1/2},
\qquad
\boldsymbol\phi_p=\frac{\mathbf c_p}{n_p},
\tag{9}
$$

the exact first-order variation is

$$
\delta\boldsymbol\phi_p
=
\frac{1}{n_p}
\left(
\mathbf I-\boldsymbol\phi_p\boldsymbol\phi_p^{\mathrm T}\mathbf S
\right)
\delta\mathbf c_p.
\tag{10}
$$

Equation 10 annihilates the radial direction $\delta\mathbf c_p=\alpha_p\mathbf c_p$, as required by orbital-scale invariance.

## 3. Physical orbital variables and gauge invariance

### 3.1 Inactive subspace

Define the inactive metric, dual orbitals, density representation, and complementary projector as

$$
\mathbf M_{\mathrm I}
=
\mathbf C_{\mathrm I}^{\mathrm T}\mathbf S\mathbf C_{\mathrm I},
\tag{11}
$$

$$
\widetilde{\mathbf C}_{\mathrm I}
=
\mathbf C_{\mathrm I}\mathbf M_{\mathrm I}^{-1},
\tag{12}
$$

$$
\mathbf P_{\mathrm I}
=
\mathbf C_{\mathrm I}\mathbf M_{\mathrm I}^{-1}
\mathbf C_{\mathrm I}^{\mathrm T},
\tag{13}
$$

and

$$
\mathbf O_{\mathrm I}
=
\mathbf I-\mathbf P_{\mathrm I}\mathbf S.
\tag{14}
$$

The operator $\mathbf P_{\mathrm I}\mathbf S$ is the $\mathbf S$-orthogonal projector onto the inactive orbital span. The physical inactive variable is consequently the subspace $\operatorname{span}(\mathbf C_{\mathrm I})$, not a particular matrix representation of that subspace. Indeed,

$$
\mathbf C_{\mathrm I}\longrightarrow \mathbf C_{\mathrm I}\mathbf A,
\qquad
\mathbf A\in GL(n_{\mathrm I}),
\tag{15}
$$

leaves both $\mathbf P_{\mathrm I}$ and $\mathbf O_{\mathrm I}$ invariant.

Equations 11--14 may be evaluated with either the raw inactive columns or their individually normalized representatives. Column normalization is an invertible diagonal transformation and therefore leaves the projector unchanged. This observation allows the gauge analysis to be carried out directly in the strictly sparse raw coefficients, even when the integral code normalizes each orbital first.

### 3.2 Projected active orbital rays

The active orbitals entering the active-space integral transformation are represented computationally as

$$
\mathbf b_p
=
\mathbf O_{\mathrm I}\boldsymbol\phi_{\mathrm A,p}.
\tag{16}
$$

Because $\boldsymbol\phi_{\mathrm A,p}$ differs from the raw $\mathbf c_{\mathrm A,p}$ only by a nonzero scalar, both $\mathbf O_{\mathrm I}\boldsymbol\phi_{\mathrm A,p}$ and $\mathbf O_{\mathrm I}\mathbf c_{\mathrm A,p}$ define the same projective ray.

Under the standard VBSCF assumption that every structure contains the same doubly occupied inactive orbitals, adding an inactive component to an active orbital does not change its projected representative:

$$
\boldsymbol\phi_{\mathrm A,p}
\longrightarrow
\boldsymbol\phi_{\mathrm A,p}+\mathbf C_{\mathrm I}\boldsymbol\ell_p,
\qquad
\mathbf O_{\mathrm I}\mathbf C_{\mathrm I}\boldsymbol\ell_p=\mathbf 0.
\tag{17}
$$

Furthermore, a nonzero scaling of $\mathbf b_p$ changes only the representative of the same orbital ray. The physical orbital variables may therefore be identified locally as

$$
\Phi(\mathbf C)
=
\left(
\operatorname{span}(\mathbf C_{\mathrm I}),
[\mathbf b_1],\ldots,[\mathbf b_{n_{\mathrm A}}]
\right),
\tag{18}
$$

where $[\mathbf b_p]$ denotes a projective ray. Additional active--active redundancies may exist only when the chosen VB structure space is invariant under the corresponding active-orbital transformation. Such redundancies must be derived from the structure model and must not be assumed for a selected or incomplete VB structure space.

### 3.3 Infinitesimal gauge transformations

For the generic selected-structure case, infinitesimal transformations that leave eq 18 invariant have the form

$$
\delta\mathbf C_{\mathrm I}=\mathbf C_{\mathrm I}\mathbf K,
\tag{19}
$$

and

$$
\delta\mathbf C_{\mathrm A}
=
\mathbf C_{\mathrm I}\mathbf L
+
\mathbf C_{\mathrm A}\mathbf D,
\tag{20}
$$

where $\mathbf K$ and $\mathbf L$ are arbitrary matrices before support constraints are imposed, and $\mathbf D$ is diagonal. Equation 19 changes only the basis used to represent the inactive subspace. The first term in eq 20 adds inactive components to the active orbitals, whereas the second independently rescales the active orbital rays.

Strict sparsity restricts the admissible gauge transformations. The exact sparse gauge space is

$$
\mathcal G_{\mathrm{sp}}
=
\left\{
(\delta\mathbf C_{\mathrm I},\delta\mathbf C_{\mathrm A})
\text{ satisfying eqs 19 and 20}:
\operatorname{supp}(\delta\mathbf c_p)\subseteq\mathcal S_p
\right\}.
\tag{21}
$$

The nonredundant tangent space is the quotient

$$
\mathcal T_{\mathrm{phys}}
=
\mathcal T_{\mathrm{sp}}/\mathcal G_{\mathrm{sp}}.
\tag{22}
$$

Crucially, $\mathcal G_{\mathrm{sp}}$ is generally a global space. When different orbitals have different supports, it need not decompose into a direct sum of independent per-orbital gauge spaces.

### 3.4 Characterization of the physical null space

The gauge statement above can be made precise without reference to a particular coordinate construction.

**Proposition 1.** Let $\mathbf S$ be positive definite, let $\mathbf C_{\mathrm I}$ have full column rank, and let every projected active orbital satisfy $\mathbf b_p\neq\mathbf 0$. Assume that the selected VB structure space has no additional continuous active--active invariance. Then the kernel of the differential of the physical map in eq 18, restricted to the strictly sparse tangent space, is exactly the space in eq 21:

$$
\ker D\Phi(\mathbf C)\big|_{\mathcal T_{\mathrm{sp}}}
=
\mathcal G_{\mathrm{sp}}.
\tag{22a}
$$

**Proof.** First consider a variation $\delta\mathbf C_{\mathrm I}=\mathbf C_{\mathrm I}\mathbf K$. Differentiation of eq 11 gives

$$
\delta\mathbf M_{\mathrm I}
=
\mathbf K^{\mathrm T}\mathbf M_{\mathrm I}
+
\mathbf M_{\mathrm I}\mathbf K.
\tag{22b}
$$

Substitution of eq 22b into the derivative of eq 13 shows by direct cancellation that $\delta\mathbf P_{\mathrm I}=\mathbf 0$. For an active variation of the form in eq 20, it follows that

$$
\delta\mathbf b_p
=
\mathbf O_{\mathrm I}
\left(
\mathbf C_{\mathrm I}\boldsymbol\ell_p
+
\alpha_p\mathbf c_{\mathrm A,p}
\right)
=
\alpha_p\mathbf b_p.
\tag{22c}
$$

Thus, the inactive projector and every projected active ray are unchanged. This proves $\mathcal G_{\mathrm{sp}}\subseteq\ker D\Phi|_{\mathcal T_{\mathrm{sp}}}$.

For the converse, suppose that a feasible variation lies in $\ker D\Phi$. The projector identity

$$
\mathbf P_{\mathrm I}\mathbf S\mathbf C_{\mathrm I}
=
\mathbf C_{\mathrm I}
\tag{22d}
$$

may be differentiated. Since $\delta\mathbf P_{\mathrm I}=\mathbf 0$, one obtains

$$
\mathbf O_{\mathrm I}\delta\mathbf C_{\mathrm I}
=
\mathbf 0.
\tag{22e}
$$

The kernel of $\mathbf O_{\mathrm I}$ is $\operatorname{span}(\mathbf C_{\mathrm I})$; hence there exists a matrix $\mathbf K$ such that $\delta\mathbf C_{\mathrm I}=\mathbf C_{\mathrm I}\mathbf K$. Invariance of the active ray implies that, for every $p$, there exists a scalar $\alpha_p$ satisfying $\delta\mathbf b_p=\alpha_p\mathbf b_p$. Because $\delta\mathbf P_{\mathrm I}=\mathbf 0$,

$$
\mathbf O_{\mathrm I}
\left(
\delta\mathbf c_{\mathrm A,p}
-
\alpha_p\mathbf c_{\mathrm A,p}
\right)
=
\mathbf 0.
\tag{22f}
$$

Therefore, $\delta\mathbf c_{\mathrm A,p}-\alpha_p\mathbf c_{\mathrm A,p}$ belongs to the inactive span, and

$$
\delta\mathbf c_{\mathrm A,p}
=
\mathbf C_{\mathrm I}\boldsymbol\ell_p
+
\alpha_p\mathbf c_{\mathrm A,p}.
\tag{22g}
$$

Finally, restriction to $\mathcal T_{\mathrm{sp}}$ imposes the fixed-support conditions in eq 21. Hence $\ker D\Phi|_{\mathcal T_{\mathrm{sp}}}\subseteq\mathcal G_{\mathrm{sp}}$, completing the proof.

Proposition 1 distinguishes representation redundancies from accidental zero curvature. Spatial symmetry, a small orbital Hessian eigenvalue, or a degeneracy at a particular stationary point does not by itself define a gauge direction. Only directions in the kernel of the physical parameterization should be removed from the variational coordinates.

## 4. Algebraic construction of the global sparse quotient

Let $\boldsymbol\theta$ collect the entries of $\mathbf K$, $\mathbf L$, and the diagonal of $\mathbf D$. Equations 19 and 20 define a linear gauge-action operator $\mathcal A$:

$$
\operatorname{vec}(\delta\mathbf C)=\mathcal A\boldsymbol\theta.
\tag{23}
$$

Let $\mathbf Z_{\perp}$ select all forbidden, off-support AO coefficients. Admissible gauge parameters satisfy

$$
\mathbf Z_{\perp}^{\mathrm T}\mathcal A\boldsymbol\theta=\mathbf 0.
\tag{24}
$$

If $\mathbf N$ spans the null space of the constraint operator in eq 24, then the allowed gauge directions in the packed sparse coordinates are

$$
\mathbf G
=
\mathcal R^{\dagger}\mathcal A\mathbf N,
\tag{25}
$$

where $\mathcal R$ is the global sparse injection and $\mathcal R^{\dagger}$ gathers allowed AO coefficients into the packed representation. A nonredundant coordinate matrix $\mathbf U\in\mathbb R^{m\times n_{\mathrm{red}}}$ must span a complement of $\operatorname{range}(\mathbf G)$. For a positive-definite ambient metric $\mathbf W$, it may be chosen to satisfy

$$
\mathbf G^{\mathrm T}\mathbf W\mathbf U=\mathbf 0,
\qquad
\mathbf U^{\mathrm T}\mathbf W\mathbf U=\mathbf I.
\tag{26}
$$

The exact reduced dimension is

$$
n_{\mathrm{red}}
=
m-\operatorname{rank}(\mathbf G).
\tag{27}
$$

Thus, the number of redundant variables follows from the actual support-constrained gauge rank; it should not be prescribed by a per-orbital dimension rule.

For numerical robustness, eqs 24--27 should be implemented using a rank-revealing QR factorization or singular-value decomposition. The numerical rank criterion must scale with the operator norm and machine precision rather than with molecule-specific thresholds.

## 5. A natural quotient metric

Euclidean distances between raw sparse coefficients are not invariant to AO scaling or to the choice of orbital representatives. A physically meaningful trust-region norm should instead measure changes in the inactive subspace and projected active rays.

For an inactive variation, define its horizontal component as

$$
\delta\mathbf C_{\mathrm I,h}
=
\mathbf O_{\mathrm I}\delta\mathbf C_{\mathrm I}.
\tag{28}
$$

For an active orbital, first differentiate eq 16:

$$
\delta\mathbf b_p
=
\mathbf O_{\mathrm I}\delta\boldsymbol\phi_{\mathrm A,p}
-
\delta\mathbf P_{\mathrm I}\mathbf S
\boldsymbol\phi_{\mathrm A,p},
\tag{29}
$$

and then remove the radial component of the projected orbital:

$$
\delta\mathbf b_{p,h}
=
\delta\mathbf b_p
-
\mathbf b_p
\frac{
\mathbf b_p^{\mathrm T}\mathbf S\delta\mathbf b_p
}{
\mathbf b_p^{\mathrm T}\mathbf S\mathbf b_p
}.
\tag{30}
$$

One natural local metric is

$$
\begin{aligned}
\lVert\delta\mathbf C\rVert_{\mathrm{phys}}^2
={}&
\operatorname{tr}\left[
\mathbf M_{\mathrm I}^{-1}
\delta\mathbf C_{\mathrm I,h}^{\mathrm T}
\mathbf S
\delta\mathbf C_{\mathrm I,h}
\right]
\\
&+
\sum_{p=1}^{n_{\mathrm A}}
\frac{
\delta\mathbf b_{p,h}^{\mathrm T}
\mathbf S
\delta\mathbf b_{p,h}
}{
\mathbf b_p^{\mathrm T}\mathbf S\mathbf b_p
}.
\end{aligned}
\tag{31}
$$

Equation 31 vanishes on the gauge directions of eqs 19 and 20. Subject to local identifiability of the physical orbital variables, it induces a positive-definite metric on the quotient in eq 22. If $\mathbf U$ is orthonormal in this induced metric, then the Euclidean norm of the reduced coordinate vector has an immediate physical meaning:

$$
\delta\mathbf x=\mathbf U\mathbf d,
\qquad
\lVert\delta\mathbf C\rVert_{\mathrm{phys}}
=
\lVert\mathbf d\rVert_2.
\tag{32}
$$

This construction makes the trust radius, Krylov orthogonalization, and convergence tests independent of arbitrary raw coefficient scaling.

## 6. Directional derivatives of the orbital-preparation map

The inactive metric derivative is

$$
\delta\mathbf M_{\mathrm I}
=
\delta\mathbf C_{\mathrm I}^{\mathrm T}\mathbf S\mathbf C_{\mathrm I}
+
\mathbf C_{\mathrm I}^{\mathrm T}\mathbf S\delta\mathbf C_{\mathrm I}.
\tag{33}
$$

Consequently,

$$
\delta\mathbf M_{\mathrm I}^{-1}
=
-\mathbf M_{\mathrm I}^{-1}
\delta\mathbf M_{\mathrm I}
\mathbf M_{\mathrm I}^{-1},
\tag{34}
$$

and

$$
\begin{aligned}
\delta\mathbf P_{\mathrm I}
={}&
\delta\mathbf C_{\mathrm I}\mathbf M_{\mathrm I}^{-1}
\mathbf C_{\mathrm I}^{\mathrm T}
+
\mathbf C_{\mathrm I}\mathbf M_{\mathrm I}^{-1}
\delta\mathbf C_{\mathrm I}^{\mathrm T}
\\
&-
\mathbf C_{\mathrm I}\mathbf M_{\mathrm I}^{-1}
\delta\mathbf M_{\mathrm I}
\mathbf M_{\mathrm I}^{-1}
\mathbf C_{\mathrm I}^{\mathrm T}.
\end{aligned}
\tag{35}
$$

Equivalently, with $\widetilde{\mathbf C}_{\mathrm I}=\mathbf C_{\mathrm I}\mathbf M_{\mathrm I}^{-1}$,

$$
\delta\mathbf P_{\mathrm I}
=
\delta\widetilde{\mathbf C}_{\mathrm I}\mathbf C_{\mathrm I}^{\mathrm T}
+
\widetilde{\mathbf C}_{\mathrm I}\delta\mathbf C_{\mathrm I}^{\mathrm T}.
\tag{36}
$$

Equation 36 shows that $\delta\mathbf P_{\mathrm I}$ has rank no greater than $2n_{\mathrm I}$. It should therefore remain in low-rank factor form throughout a matrix-free Hessian application.

Combining eqs 10, 29, and 35 gives the complete first-order response of the projected active orbitals. Written to expose its structured form,

$$
\begin{aligned}
\delta\mathbf B
={}&
\mathbf O_{\mathrm I}\delta\boldsymbol\Phi_{\mathrm A}
-
\delta\mathbf C_{\mathrm I}\mathbf M_{\mathrm I}^{-1}
\mathbf C_{\mathrm I}^{\mathrm T}\mathbf S\boldsymbol\Phi_{\mathrm A}
\\
&-
\mathbf C_{\mathrm I}\mathbf M_{\mathrm I}^{-1}
\delta\mathbf C_{\mathrm I}^{\mathrm T}\mathbf S\boldsymbol\Phi_{\mathrm A}
\\
&+
\mathbf C_{\mathrm I}\mathbf M_{\mathrm I}^{-1}
\delta\mathbf M_{\mathrm I}
\mathbf M_{\mathrm I}^{-1}
\mathbf C_{\mathrm I}^{\mathrm T}\mathbf S\boldsymbol\Phi_{\mathrm A},
\end{aligned}
\tag{37}
$$

where $\boldsymbol\Phi_{\mathrm A}$ contains the normalized active orbitals and $\mathbf B=\mathbf O_{\mathrm I}\boldsymbol\Phi_{\mathrm A}$. The first term originates from strictly sparse orbital directions; the remaining terms have ranks controlled by $n_{\mathrm I}$. Materializing eq 37 as an unrestricted dense AO-by-active matrix discards exploitable sparse-plus-low-rank structure.

## 7. Directional derivatives of active-space integrals

The active overlap and effective one-electron matrices are

$$
S^{\mathrm A}_{pq}=\mathbf b_p^{\mathrm T}\mathbf S\mathbf b_q
\tag{38}
$$

and

$$
h^{\mathrm A}_{pq}=\mathbf b_p^{\mathrm T}\mathbf F[\mathbf P_{\mathrm I}]\mathbf b_q,
\tag{39}
$$

respectively. Their directional derivatives are

$$
\delta S^{\mathrm A}_{pq}
=
\delta\mathbf b_p^{\mathrm T}\mathbf S\mathbf b_q
+
\mathbf b_p^{\mathrm T}\mathbf S\delta\mathbf b_q,
\tag{40}
$$

and

$$
\begin{aligned}
\delta h^{\mathrm A}_{pq}
={}&
\delta\mathbf b_p^{\mathrm T}\mathbf F\mathbf b_q
+
\mathbf b_p^{\mathrm T}\delta\mathbf F\mathbf b_q
+
\mathbf b_p^{\mathrm T}\mathbf F\delta\mathbf b_q,
\end{aligned}
\tag{41}
$$

where $\delta\mathbf F$ includes the response to $\delta\mathbf P_{\mathrm I}$.

An active two-electron integral is

$$
(pq|rs)
=
\sum_{\mu\nu\lambda\sigma}
b_{\mu p}b_{\nu q}
(\mu\nu|\lambda\sigma)
b_{\lambda r}b_{\sigma s}.
\tag{42}
$$

Its exact directional derivative is

$$
\delta(pq|rs)
=
(\delta p\,q|rs)
+
(p\,\delta q|rs)
+
(pq|\delta r\,s)
+
(pq|r\,\delta s).
\tag{43}
$$

Equivalently, let $\mathcal B(\mathbf B)$ denote the AO-pair to active-pair transformation and let $\mathbf K_{2e}$ denote the AO two-electron kernel. Then

$$
\mathbf G_{\mathrm A}
=
\mathcal B(\mathbf B)^{\mathrm T}
\mathbf K_{2e}
\mathcal B(\mathbf B).
\tag{44}
$$

With

$$
\mathcal D
=
\mathcal B'(\mathbf B)[\delta\mathbf B],
\tag{45}
$$

the two-electron response is

$$
\delta\mathbf G_{\mathrm A}
=
\mathcal D^{\mathrm T}\mathbf K_{2e}\mathcal B
+
\mathcal B^{\mathrm T}\mathbf K_{2e}\mathcal D.
\tag{46}
$$

For a symmetric two-electron kernel,

$$
\delta\mathbf G_{\mathrm A}
=
\mathbf A+\mathbf A^{\mathrm T},
\qquad
\mathbf A=\mathcal D^{\mathrm T}\mathbf K_{2e}\mathcal B.
\tag{47}
$$

Equations 37 and 45 imply that $\mathcal D$ inherits sparse-plus-low-rank structure. A fundamental matrix-free implementation should apply $\mathcal D$, $\mathcal D^{\mathrm T}$, and $\mathbf K_{2e}\mathcal D$ as structured operators rather than constructing dense AO-pair intermediates.

## 8. Relaxed orbital Hessian

For one target state, introduce the stationary Lagrangian

$$
\mathscr L(\mathbf x,\mathbf a,E)
=
\mathbf a^{\mathrm T}\mathbf H_{\mathrm{VB}}(\mathbf x)\mathbf a
-
E\left[
\mathbf a^{\mathrm T}\mathbf S_{\mathrm{VB}}(\mathbf x)\mathbf a-1
\right].
\tag{48}
$$

Let $\mathbf z$ denote the independent structure-response variables, including the normalization constraint. At a stationary VB solution,

$$
\mathscr L_{\mathbf z}=\mathbf 0.
\tag{49}
$$

Eliminating the first-order structure response gives the relaxed orbital Hessian as the Schur complement

$$
\mathbf H_{\mathrm{rel}}
=
\mathscr L_{\mathbf x\mathbf x}
-
\mathscr L_{\mathbf x\mathbf z}
\mathscr L_{\mathbf z\mathbf z}^{\dagger}
\mathscr L_{\mathbf z\mathbf x}.
\tag{50}
$$

The pseudoinverse in eq 50 is evaluated in the projected response space after removal of the reference-state normalization mode. A matrix-free application first solves

$$
\delta\mathbf z
=
-\mathscr L_{\mathbf z\mathbf z}^{\dagger}
\mathscr L_{\mathbf z\mathbf x}\mathbf v,
\tag{51}
$$

and then forms

$$
\mathbf H_{\mathrm{rel}}\mathbf v
=
\mathscr L_{\mathbf x\mathbf x}\mathbf v
+
\mathscr L_{\mathbf x\mathbf z}\delta\mathbf z.
\tag{52}
$$

For the generalized eigenproblem in eq 2, the directional energy derivative is

$$
\delta E
=
\mathbf a^{\mathrm T}
\left(
\delta\mathbf H_{\mathrm{VB}}
-
E\,\delta\mathbf S_{\mathrm{VB}}
\right)
\mathbf a.
\tag{53}
$$

The structure-coefficient response satisfies

$$
\left(
\mathbf H_{\mathrm{VB}}-E\mathbf S_{\mathrm{VB}}
\right)
\delta\mathbf a
=
-\left(
\delta\mathbf H_{\mathrm{VB}}
-
\delta E\,\mathbf S_{\mathrm{VB}}
-
E\,\delta\mathbf S_{\mathrm{VB}}
\right)
\mathbf a,
\tag{54}
$$

with differentiated normalization condition

$$
\mathbf a^{\mathrm T}\mathbf S_{\mathrm{VB}}\delta\mathbf a
=
-\frac{1}{2}
\mathbf a^{\mathrm T}\delta\mathbf S_{\mathrm{VB}}\mathbf a.
\tag{55}
$$

If $\{\mathbf a_j\}$ is an $\mathbf S_{\mathrm{VB}}$-orthonormal eigenbasis, the nonreference part of the response may formally be written

$$
\delta\mathbf a_{\perp}
=
-\sum_{j\ne 0}
\mathbf a_j
\frac{
\mathbf a_j^{\mathrm T}
\left(
\delta\mathbf H_{\mathrm{VB}}
-E\,\delta\mathbf S_{\mathrm{VB}}
\right)
\mathbf a
}{E_j-E}.
\tag{56}
$$

Equation 56 is unsuitable near degeneracies. A general implementation should instead solve the projected block response equation or an equivalent Sylvester equation for a near-degenerate state subspace. This avoids unstable division by individual energy gaps and provides a consistent foundation for state-averaged orbital optimization.

## 9. Reduced pullback gradient and Hessian

At a fixed accepted point, let $\mathbf U$ be the quotient basis from Section 4 and define the local raw sparse chart

$$
\mathbf x(\mathbf d)=\mathbf x_0+\mathbf U\mathbf d.
\tag{57}
$$

The reduced gradient is the covector pullback

$$
\mathbf g_{\mathrm{red}}
=
\mathbf U^{\mathrm T}\mathbf g_{\mathbf x}.
\tag{58}
$$

Because $\mathbf U$ is fixed within this local chart, the exact reduced Hessian action is

$$
\mathbf H_{\mathrm{red}}\mathbf v
=
\mathbf U^{\mathrm T}
D\mathbf g_{\mathbf x}(\mathbf x_0)
[\mathbf U\mathbf v].
\tag{59}
$$

All normalization, inactive-projector, integral, and structure-response terms must be differentiated inside $D\mathbf g_{\mathbf x}$. No derivative of $\mathbf U$ is required within a single local Newton model.

For a general nonlinear retraction $\mathcal R_{\mathbf x_0}(\mathbf d)$, the pullback Hessian contains an additional retraction-curvature contribution:

$$
\begin{aligned}
D^2(\mathcal E\circ\mathcal R_{\mathbf x_0})(\mathbf 0)
[\mathbf u,\mathbf v]
={}&
D^2\mathcal E(\mathbf x_0)
[\mathbf J_{\mathcal R}\mathbf u,
 \mathbf J_{\mathcal R}\mathbf v]
\\
&+
D\mathcal E(\mathbf x_0)
[D^2\mathcal R_{\mathbf x_0}(\mathbf 0)
[\mathbf u,\mathbf v]].
\end{aligned}
\tag{60}
$$

For the additive sparse chart in eq 57, the second term in eq 60 vanishes. Orbital normalization performed downstream remains part of $\mathcal E(\mathbf x)$ and must still be differentiated exactly.

## 10. Matrix-free trust-region Newton equation

The local second-order model is

$$
m_k(\mathbf d)
=
\mathcal E_k
+
\mathbf g_k^{\mathrm T}\mathbf d
+
\frac{1}{2}\mathbf d^{\mathrm T}
\mathbf H_k\mathbf d.
\tag{61}
$$

With the physically orthonormal quotient coordinates of eq 32, the trust-region subproblem is

$$
\min_{\mathbf d}
\left(
\mathbf g_k^{\mathrm T}\mathbf d
+
\frac{1}{2}\mathbf d^{\mathrm T}\mathbf H_k\mathbf d
\right),
\qquad
\lVert\mathbf d\rVert_2\leq\Delta_k.
\tag{62}
$$

The Hessian is accessed only through the operator chain

$$
\begin{aligned}
\mathbf v
&\xrightarrow{\ \mathbf U\ }
\delta\mathbf x
\xrightarrow{\ \text{orbital preparation}\ }
(\delta\mathbf P_{\mathrm I},\delta\mathbf B)
\\
&\xrightarrow{\ \text{integral response}\ }
(\delta\mathbf H_{\mathrm{VB}},\delta\mathbf S_{\mathrm{VB}})
\xrightarrow{\ \text{structure response}\ }
\delta\mathbf a
\\
&\xrightarrow{\ \text{gradient pullback}\ }
\delta\mathbf g_{\mathbf x}
\xrightarrow{\ \mathbf U^{\mathrm T}\ }
\mathbf H_{\mathrm{red}}\mathbf v.
\end{aligned}
\tag{63}
$$

An inexact Newton step should satisfy a residual condition of the form

$$
\left\lVert
\mathbf H_k\mathbf d_k+\mathbf g_k
\right\rVert
\leq
\eta_k\lVert\mathbf g_k\rVert,
\tag{64}
$$

where $\eta_k\rightarrow 0$ as convergence is approached. Under the usual smoothness and nonsingularity assumptions, $\eta_k=O(\lVert\mathbf g_k\rVert)$ is sufficient to recover asymptotically quadratic convergence. Away from the solution, $\eta_k$ should be selected from model agreement rather than from molecule-specific iteration limits.

A recycled block-HVP method may construct a metric-orthonormal basis $\mathbf Q$, evaluate

$$
\mathbf Y=\mathbf H_k\mathbf Q,
\qquad
\mathbf T=\mathbf Q^{\mathrm T}\mathbf Y,
\tag{65}
$$

and solve the reduced trust-region problem in $\operatorname{span}(\mathbf Q)$. Residual directions, preconditioned residuals, and transported Ritz vectors may be added in blocks. Recycling changes only how the Newton equation is solved; it does not alter the Hessian operator defined by eqs 50--63.

The step acceptance ratio is

$$
\rho_k
=
\frac{
\mathcal E(\mathbf x_k)-\mathcal E(\mathbf x_k+\mathbf U_k\mathbf d_k)
}{
-\mathbf g_k^{\mathrm T}\mathbf d_k
-\tfrac{1}{2}\mathbf d_k^{\mathrm T}\mathbf H_k\mathbf d_k
}.
\tag{66}
$$

Both the trust radius and the required inner accuracy should be adapted from $\rho_k$, the Newton residual, and the observed spectral information. Fixed system-dependent HVP or CG budgets are not part of the mathematical algorithm.

## 11. Implications for the present implementation

The current exact-context HVP differentiates orbital normalization, the inactive projector, active-space integrals, and the outer VB structure response. Its agreement with directional finite differences is evidence that the HVP is consistent with the present raw-coordinate computational graph.

However, this agreement does not validate the physical quotient coordinates. The present nonredundant-space construction requires revision for the following reasons:

1. **The inactive gauge is global.** Treating the restriction of another inactive orbital to the support of orbital $p$ as a local gauge vector is generally invalid. A support-restricted orbital is not, in general, a member of the original inactive span.
2. **Inactive additions to active orbitals are redundant.** Directions of the form $\delta\mathbf c_{\mathrm A,p}=\mathbf C_{\mathrm I}\boldsymbol\ell_p$ are annihilated by the projector in eq 14, up to an irrelevant active-orbital scaling induced by normalization. Retaining these directions introduces exact or near-zero modes.
3. **The existing rank diagnostics are not independent validation.** They test rank and intersection properties using the same per-orbital gauge model employed to construct the basis. They can therefore pass even if the assumed gauge space is physically incorrect.
4. **Euclidean local orthogonalization is coordinate dependent.** Although it can define a valid algebraic complement in special cases, it does not provide the physical norm required for a transferable trust-region method.

The immediate algorithmic priority is therefore to construct and test the global support-constrained quotient in eqs 21--27. Only after this correction should the Krylov or recycled block solver be judged, because its spectrum and convergence behavior depend directly on whether redundant zero modes have been removed correctly.

### 11.1 Initial global-gauge audit

An independent diagnostic was constructed using two routes: (i) the support-constrained algebraic gauge action in eqs 19--25 and (ii) the null space of the explicitly differentiated physical map in eq 18. The following results were obtained for the optimizer-adapted input representations. The column labeled "retained gauge" is the nullity of the physical Jacobian after restriction to the current reduced basis; "missing physical" is the rank deficit of its physical image.

| System | Packed dimension | Current reduced dimension | Exact quotient dimension | Retained gauge | Missing physical | Relative gauge-annihilation residual | Maximum principal-angle sine |
|---|---:|---:|---:|---:|---:|---:|---:|
| F$_2$ | 60 | 46 | 42 | 4 | 0 | $1.40\times10^{-16}$ | $2.11\times10^{-8}$ |
| 241 | 480 | 456 | 432 | 24 | 0 | $1.21\times10^{-16}$ | $4.47\times10^{-8}$ |
| MnF$_2$ | 1140 | 1019 | 957 | 62 | 0 | $5.04\times10^{-15}$ | $6.99\times10^{-8}$ |
| FeCl$_2$ | 4488 | 3855 | 3655 | 200 | 0 | $2.55\times10^{-16}$ | $1.47\times10^{-7}$ |

For all four systems, the algebraic gauge rank equals the independently determined physical-Jacobian nullity. The corresponding subspaces agree to numerical precision. The current coordinate space spans the complete physical image in these tests, but retains a system-dependent number of redundant directions. The observed excess dimensions are consistent with inactive additions to active orbitals that are retained by the present per-orbital construction.

## 12. Verification requirements

A revised implementation should satisfy the following system-independent tests.

### 12.1 Gauge annihilation

For every computed gauge direction $\mathbf g_j$,

$$
D\Phi(\mathbf x)[\mathbf g_j]=\mathbf 0
\tag{67}
$$

to numerical precision. Equivalently, directional changes in $\mathbf P_{\mathrm I}$ and in every projected active ray must vanish.

### 12.2 Quotient completeness

The combined gauge and nonredundant bases must span the entire sparse tangent space:

$$
\operatorname{rank}\left[\mathbf G\ \mathbf U\right]=m,
\qquad
\operatorname{range}(\mathbf G)\cap
\operatorname{range}(\mathbf U)=\{\mathbf 0\}.
\tag{68}
$$

### 12.3 Gradient gauge invariance

The packed orbital gradient must annihilate all gauge directions:

$$
\mathbf G^{\mathrm T}\mathbf g_{\mathbf x}=\mathbf 0.
\tag{69}
$$

Failure of eq 69 indicates either an incorrect gauge model or an inconsistency in the orbital-gradient pullback.

### 12.4 Hessian gauge behavior

At a stationary point, the unreduced Hessian must map gauge directions to zero up to numerical precision:

$$
\mathbf H_{\mathbf x}\mathbf G\approx\mathbf 0.
\tag{70}
$$

Away from stationarity, chart-curvature and gradient terms must be accounted for before interpreting this test.

### 12.5 Reduced HVP finite differences

For a nonredundant direction $\mathbf v$,

$$
\mathbf H_{\mathrm{red}}\mathbf v
\approx
\frac{
\mathbf g_{\mathrm{red}}(\epsilon\mathbf v)
-
\mathbf g_{\mathrm{red}}(-\epsilon\mathbf v)
}{2\epsilon}.
\tag{71}
$$

The comparison must use a common accepted-point quotient chart. Rebuilding unrelated bases at the two displaced points introduces coordinate-transport terms and does not test eq 59.

### 12.6 Representation invariance

Whenever a gauge transformation preserves the specified strict supports, the energy, physical gradient, and reduced Hessian spectrum must be invariant under that transformation. This is the decisive test that the optimizer operates on physical orbital variables rather than on arbitrary orbital representatives.

## 13. Central design principle

The desired method is not an approximation to an explicit orbital Hessian. It is an exact application of the same relaxed second-order operator in a correctly defined sparse quotient space:

$$
\boxed{
\text{strictly sparse quotient coordinates}
+
\text{exact relaxed HVP}
+
\text{adaptive block Newton solve}
}
\tag{72}
$$

An explicit Hessian and a matrix-free Hessian differ only in representation. If the quotient geometry, structure response, and directional integral transformations are exact, the matrix-free method can reproduce the local convergence of an explicit Newton method while avoiding quadratic Hessian storage and unnecessary Hessian construction.
