# Matrix-Free Second-Order Orbital Optimization for Strictly Sparse Nonorthogonal VBSCF Wave Functions

## Scope and status

This document establishes the mathematical formulation underlying a general matrix-free second-order orbital optimizer for valence-bond self-consistent-field (VBSCF) wave functions. Particular attention is paid to three features that distinguish the present problem from conventional orthogonal molecular-orbital optimization:

1. the variational orbitals are nonorthogonal;
2. every orbital has an immutable, strictly sparse atomic-orbital (AO) support; and
3. the VB structure coefficients are relaxed at every orbital geometry.

The central objective is to obtain Newton-like local convergence without constructing or storing the orbital Hessian. The derivation below separates the physical orbital manifold, its gauge redundancies, the relaxed electronic Hessian, and its matrix-free action. Statements concerning the current implementation are collected separately in Section 11 and should not be interpreted as part of the formal theory.

Full-AO OEO orbitals correspond to identity support maps, rather than
localized HAO supports. Their normalization and inactive-projector
derivatives, together with stable determinant-cofactor HVP actions, are
documented in [Full-AO OEO derivative corrections](full_ao_oeo_derivative_validation.md).

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

Here, $\mathbf H_{\mathrm{VB}}$ and $\mathbf S_{\mathrm{VB}}$ are the Hamiltonian and overlap matrices in the nonorthogonal VB structure basis. For the lowest state, the orbitally relaxed energy is

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

For an isolated excited state, one follows the corresponding stationary eigenvalue branch of eq 2 rather than the unconstrained minimum in eq 4. For a state-averaged calculation with normalized weights $w_s$,

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

The gauge sources depend on the **complete inactive span**, including orbitals assigned to other storage blocks. Nevertheless, under the fixed, independently specified column supports of eq 8, the constraints separate by target column. Consequently, the gauge in eq 21 **does** decompose into per-target spaces, provided each is obtained by enforcing off-support cancellation on the full source orbitals. Restricting the source orbitals first and declaring the restricted vectors to be gauge is not equivalent. This corrects the earlier claim that a dense global quotient was intrinsically necessary.

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

### 4.1 Exact factorization by target orbital

Let $\mathbf F_p$ collect the gauge source columns for target orbital $p$:
$\mathbf F_p=\mathbf C_{\mathrm I}$ for an inactive target, and
$\mathbf F_p=[\mathbf C_{\mathrm I}\ \mathbf c_p]$ for an active target.
Let $\mathbf R_p$ inject only **differentiable** slots. Stored but fixed slots are
treated as forbidden variations, just like off-support entries. Then define

$$
\mathbf N_p=\operatorname{null}\!\left[
(\mathbf I-\mathbf R_p\mathbf R_p^{\mathrm T})\mathbf F_p
\right],
\qquad
\mathbf G_p=\mathbf R_p^{\mathrm T}\mathbf F_p\mathbf N_p,
\qquad
\mathbf U_p=\operatorname{orth}\!\left(\ker\mathbf G_p^{\mathrm T}\right).
\tag{27a}
$$

Columns of $\mathbf K$ and $\mathbf L$, and the individual diagonal entries of
$\mathbf D$, act on distinct target columns. The off-support constraints do not
couple different targets. Thus, up to the parameter enumeration,

$$
\mathcal G_{\mathrm{sp}}=\bigoplus_p\operatorname{range}(\mathbf G_p),
\qquad
\mathbf U=\operatorname{blockdiag}(\mathbf U_1,\ldots,\mathbf U_{n_{\mathrm{orb}}}),
\qquad
n_{\mathrm{red}}=\sum_p\left[m_p-\operatorname{rank}(\mathbf G_p)\right].
\tag{27b}
$$

This proves equivalence to the global constraint construction in eqs 23--27.
Each $\mathbf U_p$ is the full Euclidean orthogonal complement of the admissible
gauge, so it contains no gauge direction and loses no direction modulo gauge.
A complete inactive source combination may cancel outside a target support even
when none of its individual source columns lies within that support. Equation
27a retains such combinations; a column-containment heuristic would miss them.

The Euclidean complement is an algebraic construction, not the final
optimization coordinate system. Let $\mathbf c_p$ contain every stored
coefficient of orbital $p$, including fixed coefficients, let $\mathbf S_p$
be the AO-overlap submatrix on that stored support, and define

$$
n_p=(\mathbf c_p^{\mathrm T}\mathbf S_p\mathbf c_p)^{1/2},
\qquad
\bar{\mathbf c}_p=\frac{\mathbf c_p}{n_p}.
\tag{27c}
$$

The Jacobian of metric normalization is

$$
\mathbf J_p
=
\frac{1}{n_p}
\left[
\mathbf I-
\bar{\mathbf c}_p
(\mathbf S_p\bar{\mathbf c}_p)^{\mathrm T}
\right].
\tag{27d}
$$

Let $\mathbf V_p$ embed the differentiable Euclidean quotient basis from eq
27a into the complete stored support, placing zero rows on fixed
coefficients. The positive-definite quotient metric and the whitened basis are

$$
\mathbf M_p
=
\mathbf V_p^{\mathrm T}
\mathbf J_p^{\mathrm T}\mathbf S_p\mathbf J_p
\mathbf V_p,
\qquad
\widetilde{\mathbf U}_p
=
\mathbf V_p\mathbf M_p^{-1/2}.
\tag{27e}
$$

Consequently,

$$
\widetilde{\mathbf U}_p^{\mathrm T}
\mathbf J_p^{\mathrm T}\mathbf S_p\mathbf J_p
\widetilde{\mathbf U}_p
=\mathbf I.
\tag{27f}
$$

Fixed coefficients enter $n_p$, $\mathbf J_p$, and $\mathbf S_p$ even though
their rows in $\mathbf V_p$ vanish. Omitting them changes the physical metric
and is incorrect.

The production implementation uses eq 27a and stores only local dense blocks,
requiring $O(\sum_p m_p n_{\mathrm{red},p})$ basis storage instead of
$O(m n_{\mathrm{red}})$. Its gradient pullback and step expansion remain exact
adjoints. It applies the local normalized-orbital whitening in eqs 27c--27f;
therefore the trust-region norm is invariant to independent rescaling of raw
orbital representatives.

After whitening, vector coordinates and gradient coordinates are different
linear maps. For a packed tangent $\mathbf v$ in the range of
$\widetilde{\mathbf U}$ and a packed gradient covector $\mathbf g$,

$$
\mathbf d
=
(\widetilde{\mathbf U}^{\mathrm T}\widetilde{\mathbf U})^{-1}
\widetilde{\mathbf U}^{\mathrm T}\mathbf v,
\qquad
\mathbf g_{\mathrm{red}}
=
\widetilde{\mathbf U}^{\mathrm T}\mathbf g.
\tag{27g}
$$

A reduced covector is lifted back to the Euclidean gauge-orthogonal packed
representative by

$$
\mathbf g_{\mathrm{pack}}
=
\widetilde{\mathbf U}
(\widetilde{\mathbf U}^{\mathrm T}\widetilde{\mathbf U})^{-1}
\mathbf g_{\mathrm{red}},
\qquad
\widetilde{\mathbf U}^{\mathrm T}\mathbf g_{\mathrm{pack}}
=\mathbf g_{\mathrm{red}}.
\tag{27h}
$$

Using the transpose pullback to recover vector coordinates is valid only for
an unwhitened Euclidean-orthonormal basis and is incorrect after eq 27e.

### 4.2 Support-preserving inactive representative

Removing the vertical tangent space does not by itself guarantee a numerically
usable representative of the inactive occupied subspace. Let

$$
\mathbf M_{\mathrm I}=\mathbf C_{\mathrm I}^{\mathrm T}
\mathbf S\mathbf C_{\mathrm I},
$$

and let $\mathbf Z_{\bar{\mathcal S}_p}^{\mathrm T}$ select AO rows outside
the prescribed support of inactive orbital $p$. A right-transform column
$\mathbf t_p$ preserves that support exactly when

$$
\mathbf Z_{\bar{\mathcal S}_p}^{\mathrm T}
\mathbf C_{\mathrm I}\mathbf t_p=\mathbf 0.
$$

Let $\mathbf A_p$ span this null space. For fixed columns
$\mathbf T_{-p}$ of the current right transform, define

$$
\mathbf R_{-p}
=
\mathbf M_{\mathrm I}
-
\mathbf M_{\mathrm I}\mathbf T_{-p}
(\mathbf T_{-p}^{\mathrm T}\mathbf M_{\mathrm I}\mathbf T_{-p})^{-1}
\mathbf T_{-p}^{\mathrm T}\mathbf M_{\mathrm I}.
$$

The support-preserving coordinate update is the generalized Rayleigh problem

$$
\max_{\mathbf z\ne\mathbf 0}
\frac{
\mathbf z^{\mathrm T}\mathbf A_p^{\mathrm T}
\mathbf R_{-p}\mathbf A_p\mathbf z
}{
\mathbf z^{\mathrm T}\mathbf A_p^{\mathrm T}
\mathbf M_{\mathrm I}\mathbf A_p\mathbf z
},
\qquad
\mathbf t_p=\mathbf A_p\mathbf z.
\tag{27i}
$$

With the other columns fixed, eq 27i maximizes the squared metric distance of
the new representative from their span. Equivalently, it maximizes the
normalized Gram determinant by a coordinate step. Cyclic updates therefore
remove avoidable near-linear dependence without filling forbidden AO rows or
changing $\operatorname{range}(\mathbf C_{\mathrm I})$. The resulting
$\mathbf T$ must be nonsingular and satisfies, at the selected point,

$$
\mathbf C_{\mathrm I}'=\mathbf C_{\mathrm I}\mathbf T.
\tag{27j}
$$

For unequal strict supports, eq 27j is not a linear coordinate transformation
on a neighborhood: a supported tangent $\delta\mathbf C_{\mathrm I}$ need not
make $\delta\mathbf C_{\mathrm I}\mathbf T$ satisfy the target supports.
Consequently, multiplying a support-restricted gradient by
$\mathbf T^{-\mathrm T}$ and discarding forbidden rows is not a valid covector
transport. The production path canonicalizes each trial representative
**before** evaluating its energy, gradient, and accepted-point HVP cache. This
direct evaluation avoids an unavailable off-support gradient and preserves the
actual computational graph. Null-space ranks are determined from matrix scale
and machine precision, not from chemical-system thresholds.

Production balancing is activated when the prescribed supports must be
reconstructed or when the standard roundoff estimate
$\epsilon n_{\mathrm I}\kappa(\mathbf M_{\mathrm I})$ exceeds
$\sqrt{\epsilon}$. Once activated, eq 27i defines a section of the gauge bundle
that is maintained after every accepted orbital step. Applying it only at the
initial point is insufficient: subsequent additive retractions can drift back
into the ill-conditioned vertical coordinates even when the occupied Gram
matrix has not yet become numerically singular.

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

Equation 31 vanishes on the gauge directions of eqs 19 and 20. On a regular stratum where the gauge rank is constant, it defines a positive-definite metric on the quotient tangent space at a fixed representative. If $\mathbf U$ is orthonormal in this induced metric, then the Euclidean norm of the reduced coordinate vector has an immediate physical meaning:

$$
\delta\mathbf x=\mathbf U\mathbf d,
\qquad
\lVert\delta\mathbf C\rVert_{\mathrm{phys}}
=
\lVert\mathbf d\rVert_2.
\tag{32}
$$

The metric removes dependence on independent orbital scalings, but full invariance under active additions from a moving inactive span requires care: differentiating $\mathbf c_p\mapsto\mathbf c_p+\mathbf C_{\mathrm I}\boldsymbol\ell_p$ also changes the active tangent by $\delta\mathbf C_{\mathrm I}\boldsymbol\ell_p$. The production code now implements the per-orbital normalized metric of eqs 27c--27f. Equation 31 is the stronger coupled quotient metric: it additionally accounts for motion of the inactive projector and projected active rays. Full representative invariance under moving inactive additions still requires that coupled metric or an equivalent connection, so the present local whitening must not be described as a complete invariant Riemannian Newton method.

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

If $\mathbf z=(\mathbf a,E)$ includes the Lagrange multiplier, $\mathscr L_{\mathbf z\mathbf z}$ is the bordered KKT matrix. For a simple eigenvalue and a positive-definite structure overlap it is nonsingular, and the dagger denotes its ordinary inverse. Alternatively, eliminating the normalization constraint gives a projected inverse on the nonreference response space. These two formulations must not be mixed. A matrix-free application first solves

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

Equation 56 exposes the genuine conditioning problem near degeneracies. A projected or block response solve does not remove the physical inverse-gap sensitivity of an isolated state. At an exact crossing an individual ordered eigenvalue may cease to be differentiable. A smoothly isolated cluster with equal weights can instead be treated by a subspace response or Sylvester equation; internal cluster rotations then cancel from the averaged objective. Unequal weights or state-specific tracking require an explicit differentiability assumption.

The linear response accuracy cannot be identified with the outer orbital-gradient
threshold. The response is a first-order quantity entering a second-order
energy model, and loose response residuals can corrupt a Rayleigh curvature
when large fixed-adjoint and relaxation terms cancel. Let
$\epsilon_E$ be the requested absolute Ritz-energy accuracy,
$\epsilon_g$ the outer gradient threshold, and
$E_{\mathrm{scale}}=\max(1,\max_i|E_i|)$ for the selected roots. The
production response solve uses the scale-derived relative residual target

$$
\tau_{\mathrm{resp}}
=
\min\left[
\epsilon_g,
\sqrt{\frac{\epsilon_E}{E_{\mathrm{scale}}}}
\right].
\tag{56a}
$$

Thus the response accuracy follows the accepted Ritz backward-error scale
without introducing a molecule-specific tolerance or forcing every HVP to
machine precision. The true bordered-equation residual is checked explicitly;
a recursive Krylov residual estimate alone is not accepted.

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

In the current physically whitened quotient chart, basis construction must preserve the
identity $\mathbf Y=\mathbf H_k\mathbf Q$ numerically, not only in exact
arithmetic. For a candidate direction $\mathbf p$, first orthogonalize and
normalize the direction itself:

$$
\mathbf a=\mathbf Q^{\mathrm T}\mathbf p,
\qquad
\beta=\left\lVert\mathbf p-\mathbf Q\mathbf a\right\rVert,
\qquad
\mathbf q=\frac{\mathbf p-\mathbf Q\mathbf a}{\beta},
\qquad
\mathbf y=\mathbf H_k\mathbf q.
\tag{65a}
$$

Reorthogonalization and preliminary scaling of $\mathbf p$ are used in finite
precision. The original search-direction action is then reconstructed as

$$
\mathbf H_k\mathbf p=\mathbf Y\mathbf a+\beta\mathbf y.
\tag{65b}
$$

This uses one HVP per admitted direction and keeps each cached image a direct
application to its normalized basis vector. Computing $\mathbf H_k\mathbf p$
first and then forming $(\mathbf H_k\mathbf p-\mathbf Y\mathbf a)/\beta$
instead can amplify cancellation when $\beta$ is small. Independently updating
both reduced basis vectors and packed tangents by projection recurrences can
also destroy their mutual consistency. The implementation now regenerates each
packed tangent from its admitted reduced vector. A small-gradient molecular
regression that exposed both failures is documented in the validation record.

For the current physically whitened quotient chart, the small trust-region problem must
be solved as a constrained quadratic problem, including singular and indefinite
models. With $\mathbf h=\mathbf Q^{\mathrm T}\mathbf g_k$ and
$\mathbf T=\mathbf Q^{\mathrm T}\mathbf H_k\mathbf Q$, its global optimality
conditions are

$$
(\mathbf T+\lambda\mathbf I)\mathbf z=-\mathbf h,
\qquad \mathbf T+\lambda\mathbf I\succeq\mathbf 0,
\qquad \lambda\geq 0,
\qquad \lVert\mathbf z\rVert\leq\Delta,
\qquad \lambda(\lVert\mathbf z\rVert-\Delta)=0.
\tag{65c}
$$

These conditions certify a minimum of the **projected** model. To see
sufficiency, let $m(\mathbf z)=\mathbf h^{\mathrm T}\mathbf z+
\tfrac12\mathbf z^{\mathrm T}\mathbf T\mathbf z$. For any feasible
$\mathbf w$,

$$
m(\mathbf w)-m(\mathbf z)
=\frac12(\mathbf w-\mathbf z)^{\mathrm T}
 (\mathbf T+\lambda\mathbf I)(\mathbf w-\mathbf z)
+\frac{\lambda}{2}(\lVert\mathbf z\rVert^2-\lVert\mathbf w\rVert^2)
\geq 0.
\tag{65d}
$$

In the eigenbasis of $\mathbf T$, let $\theta_j$ and $\widehat h_j$ be the
eigenvalues and gradient components. At
$\lambda_0=\max(0,-\theta_{\min})$, a singular denominator is not a solver
failure: the endpoint is evaluated by a pseudoinverse if its null-space
gradient vanishes; otherwise the secular root lies above that endpoint. In the
indefinite hard case, the pseudoinverse solution is completed along a
minimum-eigenvalue direction to reach the boundary. The implementation scales
to a unit ball and keeps the excess shift $\lambda-\lambda_0$ separate to avoid
losing a small positive denominator through cancellation.

The accuracy of the actual returned step, $\mathbf s=\mathbf Q\mathbf z$, can
be measured using its full reduced-coordinate KKT residual. This is not in
general the residual of an intermediate CG accumulation:

$$
\mathbf r=\mathbf g_k+\mathbf H_k\mathbf s+\lambda\mathbf s,
\qquad
\lVert\mathbf r\rVert_2\leq\eta_k\lVert\mathbf g_k\rVert_2.
\tag{65e}
$$

The production method expands the subspace with the residual block

$$
\mathbf P_k
=
\left[
-\mathbf M_k^{-1}\mathbf r,
-\mathbf r
\right],
\tag{65f}
$$

after two-pass orthogonalization against the current basis. All admitted
columns are evaluated by one block-HVP call, and eq 65c is solved again in the
enlarged space. The raw residual preserves an unpreconditioned correction when
the local positive preconditioner distorts a strongly indefinite mode; the
preconditioned residual accelerates the regular case. The preconditioner never
replaces the Hessian in the quadratic model.

Negative curvature is retained in $\mathbf T$ rather than terminating at the
first search direction. Reaching the boundary certifies only the minimum of
the current projected problem in eq 65c; it does not certify stationarity of
the full trust-region problem. Consequently, boundary and interior solutions
both continue subspace expansion until the shifted full-space KKT residual in
eq 65e meets the forcing condition or the finite resource guard is reached.
The guard is never interpreted as satisfying eq 65e. This distinction is
essential in an indefinite model: an early two-dimensional boundary solution
can omit a strongly descending direction outside the sampled space.

Hitting the HVP safety limit alone must not be interpreted as meeting the
inner residual target. A small residual does not exclude negative curvature
outside the sampled subspace. In particular, first-order outer stopping does
not certify a local minimum at a saddle with zero gradient. These limitations
must be distinguished from the exact small-model guarantee in eq 65c.

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

In particular, a positive actual decrease alone does not establish that the
quadratic model resolved the step. Write the predicted and actual decreases as
$p_k>0$ and $a_k$, and define the measured model error

$$
e_k=\lvert a_k-p_k\rvert.
$$

The production acceptance test requires

$$
a_k>0,
\qquad
e_k\leq a_k.
$$

Thus an overpredicting model is accepted only when the observed decrease is at
least as large as its model error; equivalently, $\rho_k\geq 1/2$ in that
case. An underpredicting model with $p_k>0$ satisfies the second inequality.
Rejected trials contract the radius and resolve the projected problem in the
already evaluated accepted-point Krylov space. Negative curvature is handled
by the trust-region boundary conditions in eq 65c; it is not replaced by an
unrelated first-order line-search step.

For an accepted boundary step of length $\lVert\mathbf s_k\rVert$, the observed
quadratic-model remainder $e_k$ also supplies a scale for radius growth. Under
a locally Lipschitz Hessian, the leading Taylor remainder is cubic. Requiring
the extrapolated error to remain no larger than the observed decrease gives

$$
\Delta_{k+1}
=
\lVert\mathbf s_k\rVert
\max\left[
1,
\left(\frac{a_k}{\max(e_k,e_{\mathrm{round}})}\right)^{1/3}
\right],
\qquad
e_{\mathrm{round}}
=16\epsilon\max(a_k,p_k).
$$

This update uses the measured nonquadratic error in the direction actually
taken. A cap derived from the largest Ritz magnitude can instead freeze the
radius because of a stiff mode unrelated to the accepted boundary direction.

### 10.1 Coordinate-consistent local preconditioning model

The inexpensive one-electron model used for preconditioning must be
distinguished from the exact relaxed VBSCF Hessian. For a target orbital, let
$\mathbf c$ contain **all stored coefficients**, including any fixed entries,
and let $\mathbf F$ and $\mathbf S$ denote its real symmetric local effective
one-electron and overlap matrices. In the current implementation these are the
inactive-projected pullbacks of eq 66i, not raw AO principal submatrices.
Freeze these matrices while defining the surrogate

$$
\varepsilon(\mathbf c)
=\frac{\mathbf c^{\mathrm T}\mathbf F\mathbf c}
       {\mathbf c^{\mathrm T}\mathbf S\mathbf c},
\qquad
\omega=\mathbf c^{\mathrm T}\mathbf S\mathbf c>0,
\qquad
\mathbf a=(\mathbf F-\varepsilon\mathbf S)\mathbf c,
\qquad
\mathbf b=\mathbf S\mathbf c.
\tag{66a}
$$

Direct differentiation gives

$$
\nabla_{\mathbf c}\varepsilon=\frac{2\mathbf a}{\omega},
\qquad
\nabla^2_{\mathbf c}\varepsilon
=\frac{2}{\omega}(\mathbf F-\varepsilon\mathbf S)
-\frac{4}{\omega^2}
 (\mathbf a\mathbf b^{\mathrm T}+\mathbf b\mathbf a^{\mathrm T}).
\tag{66b}
$$

Let $\mathbf V$ embed the local quotient basis into stored-coefficient space,
with zero rows on fixed coefficients, so that the local additive chart is
$\mathbf c(\mathbf d)=\mathbf c+\mathbf V\mathbf d$. Its curvature is

$$
\mathbf C
=\mathbf V^{\mathrm T}
  \nabla^2_{\mathbf c}\varepsilon\,\mathbf V.
\tag{66c}
$$

Euclidean orthogonality to orbital scaling does not imply
$\mathbf V^{\mathrm T}\mathbf S\mathbf c=\mathbf 0$. Consequently, the
second term in eq 66b cannot generally be omitted in the present chart.
Fixed coefficients also contribute to $\omega$, $\varepsilon$, $\mathbf a$,
and $\mathbf b$, despite having zero variations. With $\mathbf V$ held fixed,
rescaling all stored coefficients by a nonzero scalar $t$ gives
$\mathbf C(t\mathbf c)=t^{-2}\mathbf C(\mathbf c)$; replacing the quotient
basis by $\mathbf V\mathbf R$ gives $\mathbf R^{\mathrm T}\mathbf C\mathbf R$.
These identities and independent gradient finite differences are automated
tests of the implemented surrogate, not assumptions about its fidelity to the
many-electron Hessian.

Only small per-orbital blocks are formed. The inactive blocks carry the
double-occupancy weight derived in section 10.5; active blocks retain the unit
Rayleigh surrogate. Their existing positive spectral
regularization and inverse action define the base preconditioner, augmented
by transported positive-curvature L-BFGS secants. Negative curvature is not
removed from the exact HVP operator or from the projected trust-region model.
The surrogate omits the variation of the effective one-electron matrix and
many-electron response, and is not a physical-metric whitening or a proof of
invariance under arbitrary inactive-orbital mixing.

### 10.2 Stable inner conjugate directions

In the physically whitened quotient chart, the reduced Euclidean norm is the
local normalized-orbital AO norm of eq 27f. The production forcing function
takes $\lVert\mathbf g_k\rVert_2$, and the CG residual is checked in the same
norm. This removes independent raw-orbital scale dependence. It does not yet
implement the fully coupled metric of eq 31 or make the rule invariant under
energy-unit rescaling. The outer stopping criterion uses the corresponding
reduced gradient infinity norm.

The three-term preconditioned-CG recurrence can lose conjugacy in finite
precision even when its Euclidean HVP cache remains consistent. Let
$\mathbf M^{-1}$ be the fixed positive preconditioner for one inner solve,
$\mathbf r_j=-\mathbf g_k-\mathbf H_k\mathbf s_j$, and
$\mathbf z_j=\mathbf M^{-1}\mathbf r_j$. For previously admitted positive-
curvature directions, restore conjugacy by two passes of

$$
\mathbf p_j\leftarrow\mathbf z_j,
\qquad
\mathbf p_j\leftarrow\mathbf p_j-
\sum_{i<j}\mathbf p_i
\frac{(\mathbf H_k\mathbf p_i)^{\mathrm T}\mathbf p_j}
     {\mathbf p_i^{\mathrm T}\mathbf H_k\mathbf p_i}.
\tag{66d}
$$

The summands are applied sequentially within each pass. In exact arithmetic,
mutual conjugacy and positive curvature make these projections well-defined.
The line-minimizing update is

$$
\alpha_j=
\frac{\mathbf r_j^{\mathrm T}\mathbf p_j}
     {\mathbf p_j^{\mathrm T}\mathbf H_k\mathbf p_j},
\qquad
\mathbf s_{j+1}=\mathbf s_j+\alpha_j\mathbf p_j,
\qquad
\mathbf r_{j+1}=-\mathbf g_k-\mathbf H_k\mathbf s_{j+1}.
\tag{66e}
$$

The implementation reconstructs the residual from the accumulated step image
instead of subtracting successive residual updates. Reorthogonalization uses
already cached images and introduces no additional HVPs. Direction and image
storage is linear in the reduced dimension times the inner subspace dimension;
no dense orbital Hessian is assembled. Positive-definite quadratic tests verify
conjugacy, a fresh residual, and one HVP per admitted direction at condition
numbers $10^2$ and $10^6$. These tests do not assert stability for arbitrarily
ill-conditioned models. Negative-curvature and boundary exits continue to use
the spectral trust-region solver; reaching the inner work limit still does not
certify the final-step residual target.

### 10.3 Recycling evaluated curvature into the preconditioner

Discarding the inner HVP subspace after each accepted step loses information
that can be useful without being treated as a new-point Hessian. Diagonalize
the small projected matrix from eq 65 and retain its soft positive modes:

$$
\mathbf T\mathbf z_i=\theta_i\mathbf z_i,
\qquad
\mathbf s_i=\mathbf Q\mathbf z_i,
\qquad
\mathbf y_i=\mathbf Y\mathbf z_i=\mathbf H_k\mathbf s_i,
\qquad \theta_i>0.
\tag{66f}
$$

The **full images** $\mathbf Y\mathbf z_i$ must be used. Replacing them by
$\theta_i\mathbf s_i$ discards the component outside the sampled subspace.
In the fixed chart, these pairs satisfy
$\mathbf s_i^{\mathrm T}\mathbf y_j=\theta_i\delta_{ij}$. For a positive
inverse preconditioner $\mathbf B$, the inverse-BFGS update is

$$
\mathbf B^+
=\left(\mathbf I-\frac{\mathbf s_i\mathbf y_i^{\mathrm T}}
                              {\mathbf s_i^{\mathrm T}\mathbf y_i}\right)
 \mathbf B
 \left(\mathbf I-\frac{\mathbf y_i\mathbf s_i^{\mathrm T}}
                              {\mathbf s_i^{\mathrm T}\mathbf y_i}\right)
 +\frac{\mathbf s_i\mathbf s_i^{\mathrm T}}
             {\mathbf s_i^{\mathrm T}\mathbf y_i}.
\tag{66g}
$$

Positive curvature preserves positive definiteness. Mutual conjugacy preserves
previous secant equations within this set, so a complete positive-definite
quadratic model recovers its inverse after all independent pairs have been
applied. This identity is tested on a small synthetic model; production uses
the limited-memory two-loop action, not an assembled inverse matrix.

Sampled directions and HVP covectors are retained as approximate
preconditioning data only when consecutive accepted points remain in the same
sparse coefficient chart. If support-preserving canonicalization changes the
representative, the history is cleared. Equation 27j alone cannot transport a
strict-sparse tangent or covector, because its differential contains the
support-constrained response of $\mathbf T$; silently truncating
$\delta\mathbf C_{\mathrm I}\mathbf T$ is incorrect. Deriving that connection
would allow safe recycling across a reset, but the present implementation does
not substitute a heuristic map. Every new-point quadratic model uses fresh
exact HVPs regardless of whether approximate history is retained.

The existing history capacity is unchanged. At most one fewer than that
capacity is filled with positive Ritz pairs, reserving space for the actual
accepted-step secant when it is admissible. Soft modes are appended last;
negative and numerically zero Ritz values are excluded only from the positive
preconditioner, not from the Newton model. This recycling performs no additional
HVP evaluations. Its benefit depends on curvature persistence between points;
neither recycling nor the unchanged inner work cap guarantees quadratic
convergence.

### 10.4 Inactive-projected local surrogate

Normalization alone does not make the preconditioner consistent with the
physical variables in section 3. For an inactive target $p$, let $\mathbf B_p$
contain all other inactive columns. For an active target, let $\mathbf B_p$
contain the entire inactive space. Hold this excluded span fixed and define

$$
\mathbf G_p=\mathbf B_p^{\mathrm T}\mathbf S_{\mathrm{AO}}\mathbf B_p,
\qquad
\mathbf R_p=\mathbf I-\mathbf B_p\mathbf G_p^{-1}
                         \mathbf B_p^{\mathrm T}\mathbf S_{\mathrm{AO}}.
\tag{66h}
$$

For an empty excluded span, $\mathbf R_p=\mathbf I$. The nonempty span must
have full rank in the AO-overlap metric. These are global inactive spans, not
spans truncated to the target's storage block. Let $\mathbf L_p$ inject the stored
strictly sparse coefficients into AO space. The local matrices supplied to
eqs 66a--66c are

$$
\mathbf F_p=(\mathbf R_p\mathbf L_p)^{\mathrm T}
              \mathbf F_{\mathrm{AO}}(\mathbf R_p\mathbf L_p),
\qquad
\mathbf S_p=(\mathbf R_p\mathbf L_p)^{\mathrm T}
              \mathbf S_{\mathrm{AO}}(\mathbf R_p\mathbf L_p).
\tag{66i}
$$

The projected representatives may be dense in AO space; this does not enlarge
the optimization variables or alter their exact sparse support. Inactive
directions are projected out before constructing the normalized local model,
not by modifying the exact relaxed HVP afterward.

For an inactive target, the surrogate has an independent projector-trace
interpretation. Write $\mathbf x=\mathbf L_p\mathbf c$ and
$\mathbf P(\mathbf A)=\mathbf A(\mathbf A^{\mathrm T}\mathbf S_{\mathrm{AO}}
\mathbf A)^{-1}\mathbf A^{\mathrm T}$, with $\mathbf P(\varnothing)=\mathbf 0$,
in the convention of section 3. With
$\mathbf x$ independent of the excluded span,

$$
\begin{aligned}
\mathbf P([\mathbf B_p,\mathbf x])-\mathbf P(\mathbf B_p)
&=\frac{(\mathbf R_p\mathbf x)(\mathbf R_p\mathbf x)^{\mathrm T}}
        {(\mathbf R_p\mathbf x)^{\mathrm T}\mathbf S_{\mathrm{AO}}
         (\mathbf R_p\mathbf x)},
\\
\operatorname{Tr}\!\left[
 \mathbf F_{\mathrm{AO}}\bigl(\mathbf P([\mathbf B_p,\mathbf x])-
                              \mathbf P(\mathbf B_p)\bigr)\right]
&=\frac{\mathbf c^{\mathrm T}\mathbf F_p\mathbf c}
       {\mathbf c^{\mathrm T}\mathbf S_p\mathbf c}.
\end{aligned}
\tag{66j}
$$

The first identity follows by an invertible column operation replacing
$\mathbf x$ with $\mathbf R_p\mathbf x$, whose overlap with $\mathbf B_p$
is zero. The resulting Gram matrix is block diagonal; its inverse gives the
rank-one difference directly. Thus eq 66b gives the exact frozen-target
curvature of this projector-trace surrogate. For active targets it describes
the normalized inactive-projected ray. Equation 66j is an unweighted identity;
the production inactive block includes the physical double-occupancy factor
derived below. Active blocks remain unit-weight ray models, not exact
active-space occupation or many-electron response models.

A nonsingular change of basis within the fixed excluded span leaves
$\mathbf R_p$, $\mathbf F_p$, and $\mathbf S_p$ unchanged. Adding an excluded
inactive component to the target also leaves the projected representative
unchanged whenever that addition is compatible with the specified support.
Tests verify these identities and differentiate an independently evaluated
full projector-trace energy. The raw, unprojected normalized model fails the
same fixed-span gauge-annihilation fixture. Fixed stored coefficients remain
in the normalized representative while having zero tangent rows.

The implementation solves the small excluded-space Gram system and forms
only the projected support columns and local surrogate blocks. It constructs
neither the global orbital Hessian nor its inverse. Couplings among changing
targets, changes of the effective Fock matrix, and relaxed structure response
remain absent from the preconditioner and present in the exact HVP. Fixed-span
invariance does not imply covariance of the entire block-diagonal approximation
under arbitrary simultaneous mixing of target and excluded orbitals.

### 10.5 Double occupancy of the inactive reference

The unweighted projector identity must be distinguished from the reference
energy convention used in the code. Let $\mathbf P_{\mathrm I}$ be the
inactive projector density of section 3, without an occupation factor, and
let $\mathcal G$ be the linear Coulomb--exchange map. This map is self-adjoint
under the matrix trace pairing. With the core Hamiltonian $\mathbf h$,

$$
\begin{aligned}
\mathbf F_{11}&=\mathbf h+\mathcal G(\mathbf P_{\mathrm I}),\\
E_{11}&=\operatorname{Tr}\!\left[
 \mathbf P_{\mathrm I}(\mathbf h+\mathbf F_{11})\right],\\
\mathrm d E_{11}&=2\operatorname{Tr}
 \left[\mathbf F_{11}\,\mathrm d\mathbf P_{\mathrm I}\right].
\end{aligned}
\tag{66k}
$$

The last equality follows by differentiating both occurrences of the density
in the quadratic interaction term and using self-adjointness of $\mathcal G$.
Along an additive target-orbital path, dots denote derivatives at the accepted
point. A second differentiation gives

$$
\frac{\mathrm d^2 E_{11}}{\mathrm d t^2}
=2\operatorname{Tr}
  \left[\mathbf F_{11}\ddot{\mathbf P}_{\mathrm I}\right]
 +2\operatorname{Tr}
  \left[\dot{\mathbf P}_{\mathrm I}
             \mathcal G(\dot{\mathbf P}_{\mathrm I})\right].
\tag{66l}
$$

For an inactive target with the other inactive orbitals fixed, eq 66j implies
that the first term is exactly twice the directional curvature of the local
Rayleigh surrogate with the accepted $\mathbf F_{11}$ frozen. Consequently,
the local matrices entering positive spectral regularization are

$$
\widehat{\mathbf C}_p=w_p\mathbf C_p,
\qquad
w_p=\begin{cases}
2,&p\in\mathrm I,\\
1,&p\in\mathrm A.
\end{cases}
\tag{66m}
$$

Here the inactive value is fixed by double occupancy, not fitted to any
molecule, residual, or iteration count. The active value preserves the
existing unit-ray approximation; it is not a statement that all active
orbitals have physical occupation one. The field-response term in eq 66l,
active-density couplings, cross-target curvature, and relaxed structure
response still belong to the exact HVP, not this local approximation. Thus
eq 66m improves the reference-energy weighting without claiming an exact
VBSCF block Hessian.

Independent tests differentiate a projector-density quadratic energy with a
self-adjoint linear interaction map and recover both terms of eq 66l.
A separate production-space test checks the stationary one-electron gap
curvature and its inverse for inactive and active targets; removing the
inactive factor makes that test fail. No HVP formula, quotient basis, stopping
tolerance, or inner budget is changed by this weighting.

## 11. Implications for the present implementation

The current exact-context HVP differentiates orbital normalization, the inactive projector, active-space integrals, and the outer VB structure response. Its agreement with directional finite differences is evidence that the HVP is consistent with the present raw-coordinate computational graph.

However, this agreement does not validate the physical quotient coordinates. The pre-fix nonredundant-space construction required revision for the following reasons:

1. **Gauge sources must use the global inactive span.** Treating the restriction of another inactive orbital to the support of orbital $p$ as a local gauge vector is generally invalid. A support-restricted orbital is not, in general, a member of the original inactive span.
2. **Inactive additions to active orbitals are redundant.** Directions of the form $\delta\mathbf c_{\mathrm A,p}=\mathbf C_{\mathrm I}\boldsymbol\ell_p$ are annihilated by the projector in eq 14, up to an irrelevant active-orbital scaling induced by normalization. Retaining these directions introduces exact or near-zero modes.
3. **The existing rank diagnostics are not independent validation.** They test rank and intersection properties using the same per-orbital gauge model employed to construct the basis. They can therefore pass even if the assumed gauge space is physically incorrect.
4. **Euclidean local orthogonalization is coordinate dependent.** Orthogonalizing against the correct admissible gauge defines a valid algebraic complement, but does not by itself provide a scale-invariant physical norm for the trust-region method. Equations 27c--27g now supply that local normalized-orbital metric while preserving the exact quotient span.

The corrected production construction uses the exact factorization in eqs 27a
and 27b followed by the physical whitening in eqs 27c--27f. The previous
occupied/virtual generator, its empirical rank cutoffs, and the raw-coordinate
trust norm have been removed. The independent full global construction remains
a diagnostic oracle. Vector recovery and covector pullback are tested
separately according to eq 27g.

### 11.1 Initial global-gauge audit

An independent diagnostic was constructed using two routes: (i) the support-constrained algebraic gauge action in eqs 19--25 and (ii) the null space of the explicitly differentiated physical map in eq 18. The following pre-fix results were obtained at the initial optimizer-adapted input representations. They are finite-precision rank observations at those points, not a proof of constant rank along all optimization trajectories. The column labeled "retained gauge" is the nullity of the physical Jacobian after restriction to the current reduced basis; "missing physical" is the rank deficit of its physical image.

| System | Packed dimension | Current reduced dimension | Exact quotient dimension | Retained gauge | Missing physical | Relative gauge-annihilation residual | Maximum principal-angle sine |
|---|---:|---:|---:|---:|---:|---:|---:|
| F$_2$ | 60 | 46 | 42 | 4 | 0 | $1.40\times10^{-16}$ | $2.11\times10^{-8}$ |
| 241 | 480 | 456 | 432 | 24 | 0 | $1.21\times10^{-16}$ | $4.47\times10^{-8}$ |
| MnF$_2$ | 1140 | 1019 | 957 | 62 | 0 | $5.04\times10^{-15}$ | $6.99\times10^{-8}$ |
| FeCl$_2$ | 4488 | 3855 | 3655 | 200 | 0 | $2.55\times10^{-16}$ | $1.47\times10^{-7}$ |

At these four initial points, the algebraic gauge rank equals the independently determined physical-Jacobian nullity. The subspaces agree within the reported numerical resolution. The large audit uses the Gram matrix of the physical Jacobian; this squares conditioning and cannot reliably classify arbitrarily small nonzero singular values. Synthetic regression tests additionally use a direct SVD and finite differences of the physical map. The pre-fix coordinate space spans the complete physical image in these tests, but retains a system-dependent number of redundant directions. The observed excess dimensions are consistent with inactive additions to active orbitals retained by the old per-orbital construction.

### 11.2 Corrected construction and regression results

The implementation of eq 27a removes all retained gauge directions at the four
audited initial points, giving reduced dimensions 42, 432, 957, and 3655,
respectively. The independent physical-Jacobian audit finds no missing physical
directions. Seven synthetic tests additionally exercise unequal supports,
off-support cancellation, frozen stored coefficients, full support, inactive
basis changes, absent inactive or active spaces, and a zero-dimensional quotient.
Directional finite differences of the relaxed HVP pass on all four molecular
inputs. A larger 1764-structure diagnostic provides a more discriminating
test of the stabilized coordinate construction. Its packed dimension is 630.
The support-constrained gauge has rank 72, giving a 558-dimensional quotient;
an independent physical-map Jacobian has rank 558 and nullity 72. The current
reduced basis retains no gauge direction and misses no physical direction. The
relative gauge-annihilation residual is $3.76\times10^{-16}$, and the maximum
principal-angle sine between the algebraic gauge space and the physical
Jacobian null space is $4.71\times10^{-8}$.

At the same initial point, central energy differences along the six largest
reduced-gradient coordinates agree with the analytic pullback to relative
errors between $2.45\times10^{-8}$ and $1.88\times10^{-7}$; the largest
absolute error is $3.95\times10^{-7}$. For a generic full-space probe, the
relaxed HVP agrees with a central difference of relaxed gradients to a maximum
relative error of $2.11\times10^{-4}$ when the structure-response relative
residual is $9.91\times10^{-6}$. The analytic and finite-difference directional
curvatures are 1.726458 and 1.726393, respectively. The core HVP component has
a relative error of $3.06\times10^{-8}$, localizing the remaining discrepancy
to the iteratively solved structure response rather than to the quotient
coordinates or orbital-gradient pullback.

The residual-driven block subspace of eq 65f retains negative Ritz directions
and solves the projected indefinite trust-region problem. For the
1764-structure diagnostic, a 30-step, 32-thread run lowers the energy from
$-298.2403673953$ to $-343.4463014459$ hartree and the projected-gradient
infinity norm to $6.28\times10^{-4}$. The final actual-to-predicted decrease
ratio is 0.99997, demonstrating an accurate local quadratic model. The energy
change, $5.15\times10^{-6}$ hartree, does not yet satisfy the
$10^{-7}$-hartree termination threshold, so this run is not recorded as
converged. Twenty-four of the 30 accepted steps lie on the trust-region
boundary and 18 encounter negative curvature; only four steps reach the
32-direction subspace limit. Thus the remaining iteration-count problem is
primarily the transition from indefinite globalized motion to the local
Newton region, not an incorrect gradient or a uniformly insufficient fixed
subspace limit.

At the common production tolerances of $10^{-3}$ for the projected-gradient
infinity norm and $10^{-7}$ hartree for the accepted energy change, the
corrected implementation gives the following complete 32-thread runs. Times
are end-to-end wall times on the same workstation and are intended as
regression data rather than machine-independent benchmarks.

| Input | Reduced dimension | Iterations | HVP directions | Final energy / hartree | Final projected gradient infinity norm | Wall time / s | Peak RSS / MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| F$_2$ sparse | 42 | 6 | 16 | -198.751155830455 | $1.60\times10^{-5}$ | 0.14 | 34.5 |
| F$_2$ full AO | 218 | 15 | 50 | -198.689799086735 | $1.78\times10^{-4}$ | 0.35 | 36.0 |
| 241 | 432 | 10 | 38 | -230.720590362167 | $7.49\times10^{-5}$ | 4.24 | 1396.9 |
| MnF$_2$ | 957 | 22 | 206 | -1348.893352227972 | $8.48\times10^{-4}$ | 10.81 | 675.1 |
| FeCl$_2$ full AO | 3655 | 6 | 32 | -2181.617636472959 | $8.43\times10^{-4}$ | 6.18 | 568.3 |
| C$_6$H$_6$ sparse | 1320 | 6 | 22 | -230.634508518379 | $5.36\times10^{-5}$ | 2.90 | 909.3 |
| C$_6$H$_6$ full | 1320 | 5 | 20 | -230.777284187920 | $2.02\times10^{-4}$ | 2.93 | 908.4 |
| 10698 | 722 | 14 | 56 | -422.611824582135 | $8.03\times10^{-5}$ | 24.10 | 7441.7 |

No molecule-dependent budgets or thresholds are used. Reproduction commands,
earlier coordinate audits, and additional limitations are recorded in
[Sparse quotient correction: validation record](sparse_quotient_fix_validation.md)
and
[Matrix-free curvature decomposition diagnostics](matrix_free_curvature_decomposition_diagnostics.md).

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

Whenever a gauge transformation preserves the specified strict supports, the energy and transported physical derivatives must agree. Raw coordinate Hessian eigenvalues are not generally invariant under a nonorthogonal coordinate transformation, and away from stationarity a Hessian also acquires a gradient-dependent chart-curvature term. Spectral comparisons require a common physical metric and consistent tangent transport; they are not valid for unrelated Euclidean coordinate bases.

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

Implementation-level correctness and performance evidence are recorded in
`article/full_ao_oeo_derivative_validation.md` and
`article/matrix_free_hvp_performance_validation.md`, respectively.
The latter also defines the small-system explicit reduced-Hessian reference
used to compare matrix-free actions and future subproblem solvers without
allowing dense Hessian assembly to enter the production optimization path.
