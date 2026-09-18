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

The public state-averaged implementation currently specializes eq 5 to the
consecutive lowest $m$ roots with equal, geometry-independent weights,

$$
w_s=\frac{1}{m},
\qquad
s=0,\ldots,m-1.
$$

The input keyword `NSTATE=m` selects this objective.  Because the weights are
constant, differentiation commutes with the finite state sum:

$$
\mathbf g_{\mathrm{SA}}
=
\frac{1}{m}\sum_{s=0}^{m-1}\mathbf g_s,
\qquad
\mathbf H_{\mathrm{SA}}\mathbf v
=
\frac{1}{m}\sum_{s=0}^{m-1}\mathbf H_s\mathbf v.
$$

The implementation applies the same normalized weights to the determinant-pair
Hamiltonian and overlap adjoints, their selected-state response, and the exact
matrix-free HVP.  On the two-state F2 regression, the directly evaluated
state-average agrees with the arithmetic average of independent state
calculations to $2.84\times10^{-14}$ hartree in energy,
$3.26\times10^{-14}$ in the gradient infinity norm, and
$2.89\times10^{-14}$ in the HVP infinity norm.  The analytic state-averaged
HVP agrees with a central difference of the state-averaged gradient to a
relative error of $3.24\times10^{-9}$ for both dense and Davidson structure
solvers.

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
adjoints. It applies the local normalized-orbital prewhitening in eqs 27c--27f;
the actual trust-region norm uses the coupled quotient metric of eq 31 and
annihilates support-admissible first-order gauge directions.

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

### 4.3 Support-preserving active representative

An algebraically nonredundant tangent space can still be poorly conditioned
when a raw active representative contains a large inactive component. For an
active target $p$, let $\mathbf N_p$ span the kernel of the forbidden-row
constraint on the complete inactive columns, including stored but fixed
coefficient rows, and define the support-admissible addition basis

$$
\mathbf N_p
=\operatorname{null}(\mathbf Z_{\perp,p}^{\mathrm T}\mathbf C_{\mathrm I}),
\qquad
\mathbf Z_p=\mathbf C_{\mathrm I}\mathbf N_p.
\tag{27k}
$$

On a regular stratum, $\mathbf Z_p^{\mathrm T}\mathbf S\mathbf Z_p$ is positive
definite. The unique minimum-$\mathbf S$-norm representative of the coset
$\mathbf c_p+\operatorname{range}(\mathbf Z_p)$ is

$$
\mathbf c_p^{\star}
=\mathbf c_p
-\mathbf Z_p
 (\mathbf Z_p^{\mathrm T}\mathbf S\mathbf Z_p)^{-1}
 \mathbf Z_p^{\mathrm T}\mathbf S\mathbf c_p,
\qquad
\mathbf Z_p^{\mathrm T}\mathbf S\mathbf c_p^{\star}=\mathbf 0.
\tag{27l}
$$

The subtraction in eq 27l lies exactly in the inactive span and satisfies
every strict-support constraint. Consequently,

$$
\mathbf O_{\mathrm I}\mathbf c_p^{\star}
=\mathbf O_{\mathrm I}\mathbf c_p,
\tag{27m}
$$

so the projected active ray, the VB variational space, and its relaxed energy
are unchanged. A nonzero scalar normalization of $\mathbf c_p^{\star}$ is
applied only if every stored but nondifferentiable active coefficient is zero;
otherwise active scaling is not an admissible variation and the fixed
coefficients are preserved exactly. Both energy-only screening and accepted
gradient/HVP evaluation use the same representative section.

This choice removes avoidable cancellation in the accepted raw coefficients;
it does **not** regularize a genuine loss of transversality between the fixed
support and the inactive subspace. Such a loss is diagnosed by the smallest
nonzero singular value of
$\mathbf O_{\mathrm I}\vert_{\mathcal S_p}$ in the AO metric, not by the
dimension of the algebraic quotient. A complete-CAS full-AO OEO orbital space
has additional active--active gauge and requires an active-subspace, rather
than per-orbital-ray, representative construction.

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

The metric removes dependence on independent orbital scalings, but full invariance under active additions from a moving inactive span requires care: differentiating $\mathbf c_p\mapsto\mathbf c_p+\mathbf C_{\mathrm I}\boldsymbol\ell_p$ also changes the active tangent by $\delta\mathbf C_{\mathrm I}\boldsymbol\ell_p$. The production chart retains the support-admissible algebraic horizontal lift of eqs 27a--27b and the inexpensive per-orbital prewhitening of eqs 27c--27f. Its trust-region constraint, however, now uses the complete coupled metric of eq 31, including motion of the inactive projector and projected active rays. The horizontal lift is a representative section of the quotient, not a claim of an invariant Levi-Civita connection or an intrinsic Riemannian Hessian.

Let $\mathbf U_{\rm loc}$ denote the accepted-point sparse quotient lift and let $\mathbf M_{\rm phys}$ denote the pullback metric in eq 31. No full reduced metric is stored. For a reduced direction $\mathbf v$, the optimizer evaluates its metric covector through the analytic matrix-free action

$$
\mathbf M_{\rm red}\mathbf v
=\mathbf U_{\rm loc}^{\rm T}\mathbf M_{\rm phys}
\mathbf U_{\rm loc}\mathbf v,
\qquad
\lVert\mathbf v\rVert_{\rm phys}^{2}
=\mathbf v^{\rm T}\mathbf M_{\rm red}\mathbf v.
\tag{32a}
$$

For a Newton subspace $\mathbf Q$ of dimension $k$, the only assembled metric is $\mathbf G_k=\mathbf Q^{\rm T}\mathbf M_{\rm red}\mathbf Q$; its Cholesky factor converts the $k$-dimensional generalized trust-region problem to a Euclidean one. In the ambient reduced space, the shifted residual is $\mathbf g+\mathbf H\mathbf s+\lambda\mathbf M_{\rm red}\mathbf s$, not $\mathbf g+\mathbf H\mathbf s+\lambda\mathbf s$. Thus metric assembly and factorization scale with $k$, not with the potentially thousands-dimensional orbital quotient. The trust radius measures the tangent length at the accepted point; it is not asserted to equal the finite geodesic distance to the trial orbital frame.

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

### 9.1 Raw-coefficient update and the orbital-preparation pullback

The finite optimizer step updates the stored representative, not an
orthonormal MO frame:

$$
\mathbf x_{k+1}^{\mathrm{trial}}
=
\mathbf x_k+\mathbf U_k\mathbf d_k.
\tag{60a}
$$

Every row of the embedded basis in eq 60a that lies outside the prescribed
support is identically zero. The orbital-preparation map then constructs the
physical quantities used by the VB energy. In particular,

$$
\mathbf n_p(\mathbf c_p)
=
\frac{\mathbf c_p}
{\sqrt{\mathbf c_p^{\mathrm T}\mathbf S\mathbf c_p}},
\qquad
\mathbf P_{\mathrm I}
=
\mathbf C_{\mathrm I}
\left(\mathbf C_{\mathrm I}^{\mathrm T}\mathbf S\mathbf C_{\mathrm I}\right)^{-1}
\mathbf C_{\mathrm I}^{\mathrm T},
\tag{60b}
$$

and the active orbitals enter through their inactive-projected representatives,

$$
\mathbf b_p
=
\left(\mathbf I-\mathbf P_{\mathrm I}\mathbf S\right)\mathbf n_p.
\tag{60c}
$$

Denote the complete preparation map and physical energy composition by

$$
\boldsymbol\Phi=\boldsymbol\Phi(\mathbf x),
\qquad
E(\mathbf x)=\mathcal F(\boldsymbol\Phi(\mathbf x)).
$$

The raw-coordinate Hessian differentiated by the exact HVP is therefore

$$
\nabla_{\mathbf x}^2
\left(\mathcal F\circ\boldsymbol\Phi\right)
=
\mathbf J_{\boldsymbol\Phi}^{\mathrm T}
\nabla_{\boldsymbol\Phi}^2\mathcal F
\mathbf J_{\boldsymbol\Phi}
+
\sum_{\alpha}
\frac{\partial\mathcal F}{\partial\Phi_{\alpha}}
\nabla_{\mathbf x}^2\Phi_{\alpha}.
\tag{60d}
$$

The first term contains the integral and relaxed structure responses. The
second term is the fixed-upstream pullback curvature generated by orbital
normalization, the inactive projector, and the projected active orbitals. It
must not be dropped merely because the raw lift in eq 60a is linear. The
production HVP evaluates both terms before applying the accepted-point
covector pullback:

$$
\mathbf H_{k}\mathbf v
=
\mathbf U_k^{\mathrm T}
\nabla_{\mathbf x}^2
\left(\mathcal F\circ\boldsymbol\Phi\right)
\mathbf U_k\mathbf v.
\tag{60e}
$$

### 9.2 Why a CASSCF orbital rotation is not used

The familiar CASSCF update

$$
\mathbf C' = \mathbf C\exp(\boldsymbol\kappa),
\qquad
\boldsymbol\kappa^{\mathrm T}=-\boldsymbol\kappa,
\tag{60f}
$$

is a unitary-frame parameterization. Its redundancy arguments rely on an
orthonormal orbital frame and on orbital subspaces under which the many-electron
expansion is closed. General VBSCF active orbitals are labeled,
nonorthogonal rays. Arbitrary right mixing changes those rays and therefore
changes the VB ansatz unless the structure space and its coefficients are
transformed consistently. For strict-sparse HAO orbitals, the same mixing also
generally violates the independently prescribed column supports.

Nonorthogonality does not forbid all orbital transformations. Nonsingular
right transformations within the inactive block leave its occupied projector
unchanged, individual orbital rescalings are ray gauges, and additions of the
inactive span to an active orbital are removed by eq 60c whenever the
transformed column remains support admissible. These are precisely the local
gauge directions removed in Section 4. The remaining directions are physical
orbital variations and are represented directly by eq 60a; an orthogonal
CASSCF rotation is neither required nor generally valid.

The present additive chart is consequently a mathematically consistent local
parameterization, provided eq 60d is evaluated in full. It is not yet the most
intrinsic possible global geometry. The current quotient complement is chosen
algebraically and then whitened with a per-orbital normalization metric. Away
from stationarity, a different representative can therefore produce a
different finite chart curve even though the exact HVP remains correct for the
chosen curve. A future improvement should construct the coupled horizontal
lift associated with eq 31 and pair it with a support-preserving second-order
retraction. Such a change is valid only if its additional second derivative in
eq 60 is included in the HVP; changing the finite orbital update alone would
make the trust-region model inconsistent.

### 9.3 Exact rank-aware determinant-cofactor action

The local same-spin two-electron response also remains matrix free. For a
regular determinant-overlap block $\mathbf X$, let
$\mathcal D=\det(\mathbf X)$ and $\mathbf R=\mathbf X^{-1}$. Its second cofactor is the
exterior-square action

$$
C^{(2)}_{(i,k),(j,l)}
=\mathcal D\left(R_{ji}R_{lk}-R_{jk}R_{li}\right),
\qquad i<k,\quad j<l.
\tag{60g}
$$

For $\mathbf A(\mathbf R)=\mathbf R\wedge\mathbf R$,
$F=\langle\mathbf W,\mathbf A(\mathbf R)\rangle$, and
$\mathbf Q=\partial F/\partial\mathbf R$, its exact overlap gradient is

$$
\nabla_{\mathbf X}\left[\mathcal D F\right]
=\mathcal D\left[
F\mathbf R^{\mathrm T}
-\mathbf R^{\mathrm T}\mathbf Q\mathbf R^{\mathrm T}
\right].
\tag{60h}
$$

Direct contraction of eqs 60g--60h reduces the regular pair-local time from
$O(n^6)$ to $O(n^4)$ and persistent memory from $O(n^4)$ to $O(n^2)$, where
$n$ is the same-spin electron count. For an ill-conditioned or rank-deficient
pair, let $I_{\mathrm d}$ contain the $q$ singular values below
$\tau=n\sigma_{\max}\epsilon_{\mathrm{mach}}^{1/8}$. Since deleted minors are
multi-affine in these singular coordinates, their second-cofactor response is
recovered exactly from the $2^q$ regular nodes

$$
z_i^{(\mathbf s)}=s_i\tau,
\qquad
\lambda_{\mathbf s}
=\prod_{i\in I_{\mathrm d}}
\frac{1+s_i\sigma_i/\tau}{2},
\qquad
\mathcal C(\mathbf X)
=\sum_{\mathbf s\in\{-1,+1\}^q}
\lambda_{\mathbf s}\mathcal C(\mathbf X_{\mathbf s}).
\tag{60i}
$$

The weights are nonnegative and sum to one, while every node obeys the same
inverse-amplification bound used for a regular pair. Exact interpolation is
admitted when $2^q\leq n^2$, reducing its second-order response from
$O(n^6)$ time and $O(n^4)$ storage to $O(2^q n^4)$ time and $O(n^3)$ complete
hybrid storage. The lower-order and opposite-spin kernels retain their faster
inverse-free $O(n^3)$ polynomial representation. Only a pair failing the node
admission test retains the full polynomial second-cofactor representation.
All switches are determined by matrix conditioning, electron count, and
machine precision, not by a molecular label. The full directional derivation,
stability bound, and mixed-representation complexity are given in
[Full-AO OEO derivative validation](full_ao_oeo_derivative_validation.md).

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

In the locally prewhitened quotient chart, basis construction must preserve the
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

For the coupled physical metric, the small trust-region problem must
be solved as a constrained quadratic problem, including singular and indefinite
models. Let $\mathbf G=\mathbf Q^{\mathrm T}\mathbf M_{\rm red}\mathbf Q
=\mathbf R^{\mathrm T}\mathbf R$ and $\widehat{\mathbf Q}=\mathbf Q
\mathbf R^{-1}$. With $\widehat{\mathbf h}=\widehat{\mathbf Q}^{\mathrm T}
\mathbf g_k$ and $\widehat{\mathbf T}=\widehat{\mathbf Q}^{\mathrm T}
\mathbf H_k\widehat{\mathbf Q}$, its global optimality conditions in the
whitened *small subspace* are

$$
(\widehat{\mathbf T}+\lambda\mathbf I)\mathbf z=-\widehat{\mathbf h},
\qquad \widehat{\mathbf T}+\lambda\mathbf I\succeq\mathbf 0,
\qquad \lambda\geq 0,
\qquad \lVert\mathbf z\rVert\leq\Delta,
\qquad \lambda(\lVert\mathbf z\rVert-\Delta)=0.
\tag{65c}
$$

These conditions certify a minimum of the **projected** model. To see
sufficiency, let $m(\mathbf z)=\widehat{\mathbf h}^{\mathrm T}\mathbf z+
\tfrac12\mathbf z^{\mathrm T}\widehat{\mathbf T}\mathbf z$. For any feasible
$\mathbf w$,

$$
m(\mathbf w)-m(\mathbf z)
=\frac12(\mathbf w-\mathbf z)^{\mathrm T}
 (\widehat{\mathbf T}+\lambda\mathbf I)(\mathbf w-\mathbf z)
+\frac{\lambda}{2}(\lVert\mathbf z\rVert^2-\lVert\mathbf w\rVert^2)
\geq 0.
\tag{65d}
$$

In the eigenbasis of $\widehat{\mathbf T}$, let $\theta_j$ and $\widehat h_j$ be the
eigenvalues and gradient components. At
$\lambda_0=\max(0,-\theta_{\min})$, a singular denominator is not a solver
failure: the endpoint is evaluated by a pseudoinverse if its null-space
gradient vanishes; otherwise the secular root lies above that endpoint. In the
indefinite hard case, the pseudoinverse solution is completed along a
minimum-eigenvalue direction to reach the boundary. The implementation scales
to a unit ball and keeps the excess shift $\lambda-\lambda_0$ separate to avoid
losing a small positive denominator through cancellation.

The accuracy of the actual returned step, $\mathbf s=\widehat{\mathbf Q}\mathbf z$, can
be measured using its full reduced-coordinate KKT residual. This is not in
general the residual of an intermediate CG accumulation:

$$
\mathbf r=\mathbf g_k+\mathbf H_k\mathbf s+
\lambda\mathbf M_{\rm red}\mathbf s,
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

The outer projected-gradient threshold must not be reused as an absolute
componentwise inner-residual threshold.  In particular, the condition

$$
\lVert\mathbf r_k\rVert_\infty\leq\tau_g
$$

does not imply eq 65e and does not provide a vanishing forcing sequence.  Once
the residual components happen to fall below the fixed outer tolerance
$\tau_g$, such a shortcut can leave $\lVert\mathbf r_k\rVert_2$ comparable to
$\lVert\mathbf g_k\rVert_2$ and reduce the local iteration to linear
convergence.  The production KKT test therefore uses eq 65e in one consistent
norm.  Inner expansion may also stop when the current outer gradient already
satisfies $\lVert\mathbf g_k\rVert_\infty\leq\tau_g$ and the resolved projected
model decrease is no larger than the requested energy accuracy $\tau_E$.
This second test limits work that cannot affect the outer accuracy decision; it
does not declare convergence.  The finite trial energy and gradient are still
evaluated, accepted, and tested against both outer tolerances.

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

Acceptance of a physical step and trust in its quadratic model are distinct.
Write the predicted and actual decreases as $p_k>0$ and $a_k$, and define the
measured model error

$$
e_k=\lvert a_k-p_k\rvert.
$$

The production acceptance test first requires a numerically resolved monotone
trial,

$$
a_k>e_{\mathrm{acc},k},
\qquad
e_{\mathrm{acc},k}=16\epsilon p_k.
$$

For a full or directionally exact Hessian model it additionally requires

$$
e_k\leq a_k+e_{\mathrm{acc},k}.
$$

For a core-only candidate this additional certificate is relaxed only when
the measured work model declares full response asymptotically more expensive
than the core action,

$$
\mathcal W_k^{\mathrm{out}}>\mathcal W_k^{\mathrm{core}}.
$$

The exact trial energy therefore protects monotonicity, whereas $e_k$ controls
the validity radius of the model that proposed the step.  This separation is
important for an approximate core Hessian with unaffordable response:
rejecting an energy-lowering step solely because $e_k>a_k$ repeatedly resolves
the same systematically deficient model at the old point.  Accepting it
instead supplies the exact new gradient and a transported secant that corrects
the next Hessian model.  If full response is affordable, poor agreement instead
signals that the nonlinear step is too large and the strict certificate is
retained.  Negative curvature is still handled by the trust-region boundary
conditions in eq 65c; it is not replaced by an unrelated first-order
line-search step.

For an accepted boundary step of length $\lVert\mathbf s_k\rVert$, the observed
model remainder $e_k$ also supplies a scale for radius growth.  The order of
that remainder depends on the fidelity of the Hessian action that actually
generated the trial.  Let

$$
\nu_k=
\begin{cases}
2, & \mathbf B_k=\mathbf H_k^{\mathrm c}
     \text{ or another approximate Hessian},\\
3, & \mathbf B_k\mathbf s_k=\mathbf H_k\mathbf s_k
     \text{ on the accepted direction}.
\end{cases}
$$

Indeed, Taylor expansion gives

$$
p_k-a_k
=\frac12\mathbf s_k^{\mathrm T}
  (\mathbf H_k-\mathbf B_k)\mathbf s_k
+\mathcal O(\lVert\mathbf s_k\rVert^3).
$$

The missing outer response therefore leaves a quadratic model defect even
when the core HVP itself is evaluated exactly.  Only a full HVP or an exact
directional outer-response correction removes that term and exposes the cubic
Taylor remainder.  For a model-resolved accepted step,
$e_k\leq a_k+e_{\mathrm{round}}$, requiring the extrapolated error to remain no
larger than the observed decrease gives

$$
\Delta_{k+1}
=
\lVert\mathbf s_k\rVert
\max\left[
1,
\left(\frac{a_k}{\max(e_k,e_{\mathrm{round}})}\right)^{1/\nu_k}
\right],
\qquad
e_{\mathrm{round}}
=16\epsilon\max(a_k,p_k).
$$

If an unaffordable-response core step is monotone but model-inaccurate,
$e_k>a_k+e_{\mathrm{round}}$, it is accepted but the same measured remainder
contracts the next validity radius:

$$
\Delta_{k+1}
=
\lVert\mathbf s_k\rVert
\left(
\frac{a_k}{\max(e_k,e_{\mathrm{round}})}
\right)^{1/\nu_k}.
$$

Thus a poor trust ratio cannot authorize another step of the same scale, but
it also cannot discard a useful monotone displacement and its exact gradient.
This update has no molecule-dependent acceptance ratio or contraction factor.

For a nonmonotone or numerically unresolved trial, the measured energy also
determines a direct
one-dimensional interpolation.  Define the positive linear decrease

$$
\ell_k=-\mathbf g_k^{\mathrm T}\mathbf s_k.
$$

The quadratic ray model that matches the derivative at the accepted point and
the measured full-step decrease is

$$
\widehat a_k(\alpha)
=\alpha\ell_k-\alpha^2(\ell_k-a_k).
$$

When $\ell_k>0$, $\ell_k-a_k>0$, and its maximizer lies strictly inside the
trial, the next radius is

$$
\Delta_{k+1}
=\lVert\mathbf s_k\rVert
\frac{\ell_k}{2(\ell_k-a_k)}.
$$

This replaces a sequence of weak contractions by the scale inferred from the
already evaluated energy.  If the interpolation is invalid or noncontracting,
the rejected trial instead uses the same fidelity-dependent model order when
scaling its retained trustworthy fraction.  The combined update prevents a
core-only step from being treated as though it had the cubic remainder of an
exact Newton model.  A cap derived from the largest Ritz magnitude can instead
freeze the radius because of a stiff mode unrelated to the accepted boundary
direction.

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

The sparse quotient chart is locally prewhitened by eq 27f, whereas the
trust-region norm and shifted KKT equation now use the coupled metric of eq
31 through eq 32a. The production forcing function still uses the local
coordinate $\lVert\mathbf g_k\rVert_2$, and the residual is checked in that
same norm. This is a coordinate-consistent inexact-Newton test, but not yet
the fully coupled dual norm and not invariant under energy-unit rescaling.
The outer stopping criterion uses the corresponding reduced-gradient
infinity norm.

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

The current implementation stores accepted secants and sampled positive Ritz
pairs in the common packed strict-sparse coefficient embedding. Let
$\mathcal P_k^{\mathrm v}$ and $\mathcal P_k^{\ast}$ denote, respectively, the
current-chart vector and covector projections defined by the local quotient
basis and its raw-tangent Gram matrix. A packed historical pair
$(\overline{\mathbf s}_i,\overline{\mathbf y}_i)$ is reused at point $k$ as

$$
\mathbf s_i^{(k)}
=\mathcal P_k^{\mathrm v}\overline{\mathbf s}_i,
\qquad
\mathbf y_i^{(k)}
=\mathcal P_k^{\ast}\overline{\mathbf y}_i.
\tag{66h}
$$

For an accepted step, the packed pair is the actual displacement between the
two canonical representatives and their packed gradient difference. For a
Ritz pair, the old-chart reduced direction and its **full** HVP image are first
expanded into the same packed embedding. Equation 66h is an extrinsic
projection transport used only to construct a positive approximate inverse
preconditioner. It is not used as a current-point HVP, and every new-point
quadratic model continues to use fresh exact Hessian actions. Pairs whose
current-chart projected curvature is not positive are rejected. If the
quotient rank signature changes, the history is cleared.

This construction avoids the incorrect dense occupied-orbital transform
followed by off-support truncation: no coefficient outside the declared sparse
support is ever introduced. It is a first-order vector transport in the
embedded representation, not an exact parallel transport for the quotient
connection. Its role is therefore restricted to preconditioning; convergence
and acceptance never depend on a transported secant being exact.

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
\tag{66i}
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
\tag{66j}
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
\tag{66k}
$$

The first identity follows by an invertible column operation replacing
$\mathbf x$ with $\mathbf R_p\mathbf x$, whose overlap with $\mathbf B_p$
is zero. The resulting Gram matrix is block diagonal; its inverse gives the
rank-one difference directly. Thus eq 66b gives the exact frozen-target
curvature of this projector-trace surrogate. For active targets it describes
the normalized inactive-projected ray. Equation 66k is an unweighted identity;
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
\tag{66l}
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
\tag{66m}
$$

For an inactive target with the other inactive orbitals fixed, eq 66k implies
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
\tag{66n}
$$

Here the inactive value is fixed by double occupancy, not fitted to any
molecule, residual, or iteration count. The active value preserves the
existing unit-ray approximation; it is not a statement that all active
orbitals have physical occupation one. The field-response term in eq 66m,
active-density couplings, cross-target curvature, and relaxed structure
response still belong to the exact HVP, not this local approximation. Thus
eq 66n improves the reference-energy weighting without claiming an exact
VBSCF block Hessian.

Independent tests differentiate a projector-density quadratic energy with a
self-adjoint linear interaction map and recover both terms of eq 66m.
A separate production-space test checks the stationary one-electron gap
curvature and its inverse for inactive and active targets; removing the
inactive factor makes that test fail. No HVP formula, quotient basis, stopping
tolerance, or inner budget is changed by this weighting.

### 10.6 Cost-aware dynamic outer-response accuracy

Solving every trust-region subproblem to the most accurate available relaxed
Hessian residual does not in general minimize the time to convergence.  Split
the reduced Hessian at accepted point $k$ into the inexpensive core action and
the relaxed outer response,

$$
\mathbf H_k=\mathbf H_k^{\mathrm c}+\mathbf R_k,
\tag{66o}
$$

where neither term is explicitly assembled.  If $n_{c,k}$ core actions,
$n_{r,k}$ response actions, and $n_{f,k}$ exact trial evaluations are used at
that point, the relevant computational objective is

$$
T_{\mathrm{total}}
=\sum_k\left(
n_{c,k}t_{c,k}+n_{r,k}t_{r,k}+n_{f,k}t_{f,k}
\right),
\tag{66p}
$$

not the number of outer iterations or the residual of one inner solve in
isolation.  The measured costs $t_{c,k}$, $t_{r,k}$, and $t_{f,k}$ are allowed
to vary with the accepted point.  No molecule name, active-space-size cutoff,
or fixed response budget enters eq 66p.

For a candidate $\mathbf s_k^{\mathrm c}$ obtained from the core action, the
omitted directional response is

$$
\mathbf d_k=\mathbf R_k\mathbf s_k^{\mathrm c}.
\tag{66q}
$$

One exact response action evaluates this vector.  It corrects the quadratic
decrease without constructing or interpolating $\mathbf R_k$,

$$
p_k
=-\mathbf g_k^{\mathrm T}\mathbf s_k^{\mathrm c}
-\frac12(\mathbf s_k^{\mathrm c})^{\mathrm T}
\left(\mathbf H_k^{\mathrm c}\mathbf s_k^{\mathrm c}+\mathbf d_k\right),
\tag{66r}
$$

and gives the complete shifted KKT residual on the same candidate,

$$
\mathbf r_k^{\mathrm{probe}}
=\mathbf g_k
+\mathbf H_k^{\mathrm c}\mathbf s_k^{\mathrm c}
+\mathbf d_k
+\lambda_k^{\mathrm c}\mathbf M_k\mathbf s_k^{\mathrm c}.
\tag{66s}
$$

The exact directional image also permits minimization of the corrected
quadratic model on the already sampled ray without another HVP. Define

$$
\ell_k=-\mathbf g_k^{\mathrm T}\mathbf s_k^{\mathrm c},
\qquad
q_k=(\mathbf s_k^{\mathrm c})^{\mathrm T}
\left(\mathbf H_k^{\mathrm c}\mathbf s_k^{\mathrm c}+\mathbf d_k\right).
\tag{66sa}
$$

When $\ell_k>0$ and $q_k>0$, the ray minimizer retained inside the sampled
step is

$$
\alpha_k=
\min\left(1,\frac{\ell_k}{q_k}\right),
\qquad
\widetilde{\mathbf s}_k=\alpha_k\mathbf s_k^{\mathrm c}.
\tag{66sb}
$$

For nonpositive directional curvature the original trust-region step is
retained. This scalar minimization uses the exact sampled curvature, preserves
the matrix-free representation, and prevents a response correction from
submitting a point beyond the minimum of its own corrected ray model.

The probe is a certificate, not a rank-one Hessian model.  It certifies an
inexact-Newton step when its corrected decrease is positive and its complete
shifted residual satisfies

$$
\lVert\mathbf r_k^{\mathrm{probe}}\rVert_2
\leq\eta_k\lVert\mathbf g_k\rVert_2,
\tag{66t}
$$

using the same forcing term as the complete Newton solve. If this condition
fails, the core subspace is not fitted with arbitrary orbital-space secants,
and one sampled direction does not authorize complete response actions on all
subsequent Krylov directions. Such a promotion would multiply the cost of one
directional certificate by an unknown final subspace dimension while inferring
global response curvature from one sample.

Before the first probe is evaluated, its unique-string contraction scale is
compared with the leading AO-pair/active-pair scale of one core action.  For
$N_\alpha$ and $N_\beta$ unique spin strings and $N_{\mathrm{st}}$ selected
states, the dense response-work proxy is

$$
W_R^{\mathrm{dense}}
=N_{\mathrm{st}}N_\alpha N_\beta(N_\alpha+N_\beta).
\tag{66ua}
$$

When the selected-state coefficient support is trimmed, the implementation
uses the smaller of eq 66ua and the operation count of the sparse contraction
actually selected by the production kernel.  The corresponding leading core
pair-contraction count is

$$
W_C
=\frac{N_{\mathrm{AO}}(N_{\mathrm{AO}}+1)}{2}
 \frac{N_{\mathrm{act}}(N_{\mathrm{act}}+1)}{2}.
\tag{66ub}
$$

If the current core solve has already evaluated $m_k$ HVP directions, one
unsampled response probe is admitted only when

$$
W_R\leq m_k W_C.
\tag{66uc}
$$

Thus a response direction may be more expensive than one core direction but
must still fit inside the work already invested in the complete core
candidate.  This is an asymptotic algorithmic comparison, not an
active-space-size threshold, and it does not authorize full outer response on
all $m_k$ Krylov directions.  Once a response has been sampled, it must
additionally satisfy the measured affordability condition

$$
t_{r,k}\leq T_k^{\mathrm{core\ candidate}},
\tag{66u}
$$

where $T_k^{\mathrm{core\ candidate}}$ is the measured wall time already
invested in constructing and solving the current core candidate.  This is the
cost of the complete cheap alternative, not only its HVP kernel.  The most
recent measured $t_{r,k}$ is retained across accepted points.  A requested
probe is skipped before evaluation whenever that measured response cost
exceeds the work invested in the new core candidate; it becomes eligible
again only if accumulated core work reaches the same cost. Thus an expensive
calibration is not repeated blindly at every accepted point.

When the exact probe fails eq 66t but the corrected decrease in eq 66r remains
positive, the direction may still be submitted as a trust-region trial. Its
acceptance is guarded by the exact relaxed energy and gradient; it is not
reported as a converged Newton solve. A rejected trial contracts the radius
and returns to the core model at the same accepted point. Consequently, the
bounded-response implementation uses at most one exact outer-response
direction per accepted point and never incurs
$\mathcal O(m_k C_{R,k})$ response work merely because one probe exposed an
unresolved residual. Recovering a complete Newton certificate without that
multiplication requires the response-space construction of Section 10.7,
rather than an unconstrained orbital-space secant fit.

During trust-region globalization, the first core candidate does not require
a precise Newton equation.  It can use zero response actions because the exact
trial energy and gradient still guard acceptance.  An interior candidate,
negative curvature, a rejected trial, or failure of an accepted boundary step
to continue expanding the trust radius while reducing the gradient requests a
response certificate at the next solve.  These events are properties of the
optimization model; they introduce no molecular or active-space cutoff.

Across accepted steps, the directly observed wall-time progress measure

$$
\Pi_k
=\frac{-\log\!\left(
\lVert\mathbf g_{k+1}\rVert_2/\lVert\mathbf g_k\rVert_2
\right)}{T_k}
\tag{66v}
$$

compares response fidelities in the quantity actually being optimized. A
nonpositive numerator records stagnation rather than being clipped into an
apparent speedup. The global-to-local transition
requires no fixed outer-iteration count.  In the absence of a cost deferral, a
core step remains in the inexpensive globalization branch only while

$$
\lVert\mathbf s_k\rVert_{\mathbf M_k}=\Delta_k,
\qquad
\lVert\mathbf g_{k+1}\rVert_2<\lVert\mathbf g_k\rVert_2,
\qquad
\Delta_{k+1}>\Delta_k,
\tag{66w}
$$

and no negative-curvature direction was detected.  When any part of eq 66w
fails, the next candidate requests a response certificate. If eq 66u defers
that probe and the exact trial still reduces the gradient, the core branch is
continued; loss of gradient reduction requests a new certificate. Thus the
cheap branch is continued only while it makes measurable progress.

This dynamic-accuracy construction deliberately permits a few extra accepted
globalization steps when they are cheaper than one relaxed response action.
As the gradient decreases and the step enters the interior Newton regime,
eq 66t determines whether a candidate possesses a complete Newton
certificate. Fixed response counts and system-specific activation thresholds
are excluded from the production algorithm. The bounded-response production
path does not claim asymptotically quadratic convergence when eq 66t fails;
that stronger property is the target of the certified low-rank Schur model
below.

#### 10.6.1 Transported accepted-step correction of the core Hessian

When eq 66ua--66ub excludes exact outer response, repeatedly solving with
$\mathbf H_k^{\mathrm c}$ alone generally gives linear local convergence
because the omitted curvature does not vanish with the gradient.  The core
model remains unchanged during successful globalization; a rejected trial is
the parameter-free event that activates the missing-curvature model. Exact
accepted-point gradients then provide full secant information without an
additional structure-response solve.  After transporting an accepted
displacement and gradient difference into the current quotient chart, define

$$
\mathbf s_i=\mathcal T_{i\rightarrow k}\Delta\mathbf d_i,
\qquad
\mathbf y_i=\mathcal T_{i\rightarrow k}\Delta\mathbf g_i,
\qquad
\mathbf z_i=\mathbf y_i-\mathbf H_k^{\mathrm c}\mathbf s_i.
\tag{66wa}
$$

Thus $\mathbf z_i$ samples the curvature missing from the current core model,
including the relaxed response to first order and the finite change of the
core Hessian between accepted points.  Let $\mathbf B_i$ be the accumulated
symmetric correction and

$$
\mathbf u_i=\mathbf z_i-\mathbf B_i\mathbf s_i.
\tag{66wb}
$$

The Powell-symmetric-Broyden least-change update is

$$
\mathbf B_{i+1}
=\mathbf B_i
+\frac{\mathbf u_i\mathbf s_i^{\mathrm T}
       +\mathbf s_i\mathbf u_i^{\mathrm T}}
      {\mathbf s_i^{\mathrm T}\mathbf s_i}
-\frac{\mathbf u_i^{\mathrm T}\mathbf s_i}
      {(\mathbf s_i^{\mathrm T}\mathbf s_i)^2}
 \mathbf s_i\mathbf s_i^{\mathrm T}.
\tag{66wc}
$$

Equation 66wc is symmetric, satisfies the newest transported residual secant,
and permits indefinite curvature.  Its action is retained as two vectors per
accepted pair; neither $\mathbf B_i$ nor the orbital Hessian is materialized.
The large-space model applied by the Krylov solver is therefore

$$
\widetilde{\mathbf H}_k\mathbf v
=\mathbf H_k^{\mathrm c}\mathbf v+\mathbf B_k\mathbf v.
\tag{66wd}
$$

For a fixed accepted point, construction of $\mathbf B_k$ requires the single
block action $\mathbf H_k^{\mathrm c}\mathbf S_k$, where the columns of
$\mathbf S_k$ are the transported secant displacements.  Trust-radius retries
change neither the chart, the accepted-point core Hessian, nor the transported
history.  The correction is therefore built once and reused until a trial is
accepted.  With $q_k$ retained secants and $r_k$ rejected trials, rebuilding
the same correction would consume

$$
(r_k+1)q_k
\tag{66we}
$$

core-HVP directions, whereas accepted-point caching requires only $q_k$.
This removes retry-dependent HVP work without changing eq 66wd or any trial
step.

The same accepted-point lifetime applies to the RI response intermediates,
orbital pullback cache, physical retraction metric, and transported
preconditioner.  Denoting their common construction cost by
$C_k^{\mathrm{setup}}$ and all actual directional actions by
$C_k^{\mathrm{act}}$, the retry cost changes from

$$
C_k^{\mathrm{uncached}}
=(r_k+1)C_k^{\mathrm{setup}}+C_k^{\mathrm{act}}
\quad\hbox{to}\quad
C_k^{\mathrm{cached}}
=C_k^{\mathrm{setup}}+C_k^{\mathrm{act}}.
\tag{66wf}
$$

This cache is invalidated immediately after acceptance or a quotient-rank
change; it is never transported to a different orbital point.

Every finite step remains guarded by the exact relaxed energy and gradient and
by the trust-region acceptance test.  This correction is a quasi-Newton model,
not an exact outer-response certificate; eq 66t is asserted only after an
admitted exact response action.  Under the usual Dennis--Mor\'e secant
condition, however, the correction can recover superlinear local behavior
without paying the unique-string response cost on every HVP direction.

### 10.7 Residual-certified low-rank Schur response

Cost-aware scheduling cannot change the asymptotic cost of one exact relaxed
response.  In a large active space, the structure dimension, the number of
unique string pairs, and the active integral-response dimension can all grow
rapidly.  If a Krylov solve uses $m_k$ complete HVP directions, its relaxed
part costs

$$
T_k^{\mathrm{response}}=
\mathcal O\!\left(m_k C_{R,k}\right),
\tag{66x}
$$

where $C_{R,k}$ denotes the full cost of one directional integral response,
structure response solve, and backward orbital contraction.  Batching changes
constants and memory traffic but not the factor $m_k$.

The algebraic target for a lower-cost method is the Schur term in eq 50.  At
one accepted point, write

$$
\mathbf R_k
=-\mathbf B_k^{\mathrm T}\mathbf A_k^{\dagger}\mathbf B_k,
\qquad
\mathbf A_k=\mathscr L_{\mathbf z\mathbf z},
\qquad
\mathbf B_k=\mathscr L_{\mathbf z\mathbf x}.
\tag{66y}
$$

Neither $\mathbf A_k$, $\mathbf B_k$, nor $\mathbf R_k$ should be assembled.
For a small orthonormal set of orbital directions
$\mathbf U_k\in\mathbb R^{n_x\times r_k}$, exact response probes provide

$$
\mathbf W_k=\mathbf R_k\mathbf U_k.
\tag{66z}
$$

The samples should not be treated as arbitrary orbital secants.  Their Schur
origin supplies response-space vectors and a small projected response matrix,

$$
\mathbf Z_k=\mathbf A_k^{\dagger}\mathbf B_k\mathbf U_k,
\qquad
\mathbf K_k=\mathbf Z_k^{\mathrm T}\mathbf A_k\mathbf Z_k
=-\mathbf U_k^{\mathrm T}\mathbf W_k.
\tag{66aa}
$$

The corresponding Galerkin--Schur approximation is

$$
\widehat{\mathbf R}_k
=-\mathbf B_k^{\mathrm T}\mathbf Z_k
  \mathbf K_k^{\dagger}
  \mathbf Z_k^{\mathrm T}\mathbf B_k
=-\mathbf W_k\mathbf K_k^{\dagger}\mathbf W_k^{\mathrm T}.
\tag{66ab}
$$

For an exactly solved self-adjoint response, $\mathbf K_k$ is symmetric and
eq 66ab satisfies
$\widehat{\mathbf R}_k\mathbf U_k=\mathbf W_k$ on the resolved range of
$\mathbf K_k$.  Its rank is at most $r_k$,
its application costs $\mathcal O(n_x r_k)$, and only the orbital-space
matrices $\mathbf W_k$ and $\mathbf K_k$ are required for application.
The directions $\mathbf U_k$ are retained for enrichment and transport.
Directional
structure vectors of length $n_{\mathrm{str}}$ are transient and can be
discarded after the backward contraction.  Thus this construction does not
replace a dense structure-space object by another persistent
$n_{\mathrm{str}}\times r_k$ cache.  A rank-revealing factorization of
$\mathbf K_k$ must use the certified structure-response residual and floating
point backward error; a fitted eigenvalue cutoff would reintroduce a
system-dependent parameter.

The low-rank operator is a subproblem accelerator, not a convergence
certificate.  If $\widehat{\mathbf s}_k$ solves the approximate trust problem,
one exact probe forms

$$
\mathbf e_k
=\mathbf R_k\widehat{\mathbf s}_k
 -\widehat{\mathbf R}_k\widehat{\mathbf s}_k.
\tag{66ac}
$$

The candidate is accepted as an inexact Newton solution only when the exact
corrected shifted residual satisfies eq 66t.  A useful sufficient allocation
of the residual budget is

$$
\lVert\mathbf e_k\rVert_2
\leq \theta_k\eta_k\lVert\mathbf g_k\rVert_2,
\qquad 0<\theta_k<1,
\tag{66ad}
$$

with the remaining budget assigned to the approximate subproblem residual.
The parameter $\theta_k$ need not be a fitted molecular constant: it can be
chosen from the ratio of the measured two residual components so that neither
one dominates the certified total.  When eq 66ad fails, the normalized part of
$\widehat{\mathbf s}_k$ outside $\operatorname{span}(\mathbf U_k)$ is appended
to $\mathbf U_k$, the already computed exact probe becomes its new response
sample, and the approximate Krylov solve is restarted.  Restarting is required
because changing $\widehat{\mathbf R}_k$ inside one nominal Krylov solve would
invalidate its recurrence and residual certificate.

If the numerical response rank remains $r_k\ll m_k$, the resulting work is

$$
T_k^{\mathrm{response,LR}}
=\mathcal O\!\left(r_k C_{R,k}+m_k n_x r_k\right),
\qquad
M_k^{\mathrm{LR}}=\mathcal O(n_x r_k),
\tag{66ae}
$$

apart from the transient memory of one exact response action.  This is an
algorithmic reduction in the number of expensive responses, not yet a lower
asymptotic bound for $C_{R,k}$ itself.  Lowering that remaining factor requires
the implicit maps $\mathbf B_k$ and $\mathbf B_k^{\mathrm T}$ to act directly
through unique-string-pair factorizations and requires the projected solve
with $\mathbf A_k$ to preserve the same residual certificate.  Consequently,
the implementation should proceed in two independently testable stages:
residual-certified response-rank compression first, followed by factorized
lower-scaling exact probes.  A symmetric orbital-space multisecant formula by
itself is insufficient: it can interpolate sampled products while missing the
important range of $\mathbf A_k^{-1}\mathbf B_k$, and rejected trials can then
force repeated expensive enrichment.  Direct interpolation of isolated
vectors $\mathbf R_k\mathbf s$ without the Schur factorization and an
independent exact certificate is therefore excluded.

### 10.8 Exact pair-domain decomposition of one response probe

Low-rank Schur compression reduces the number of exact response probes, whereas
the cost of every retained probe must still be minimized without changing its
algebra. In the unique-string representation, the active-space adjoint is a
sum of independent ordered-pair contributions. For a generic active-space
gradient component,

$$
\delta\mathbf g_{\mathrm{act}}
=
\sum_{I,J\in\mathcal U_\alpha}
\mathcal A^{\alpha}_{IJ}
+
\sum_{K,L\in\mathcal U_\beta}
\mathcal A^{\beta}_{KL}.
\tag{66af}
$$

For any disjoint partition of either pair domain,

$$
\delta\mathbf g_{\mathrm{act}}
=
\sum_{t=1}^{n_{\mathrm{thread}}}
\delta\mathbf g_{\mathrm{act}}^{(t)},
\qquad
\delta\mathbf g_{\mathrm{act}}^{(t)}
=
\sum_{(I,J)\in\mathcal P_t}
\mathcal A_{IJ}.
\tag{66ag}
$$

Thread-local active one-electron, overlap, and packed two-electron adjoints are
reduced only after all pair tiles have been consumed. Equation 66ag changes
the evaluation order but neither truncates the pair domain nor alters the HVP.
It also removes the previous three-channel concurrency ceiling: the packed
opposite-spin contraction and the two overlap pullbacks each use the complete
worker team rather than competing as three internally serial tasks.

The directional structure images admit an analogous channel decomposition.
Writing the opposite-spin part in packed active-pair channels gives

$$
\delta\mathbf H_{\mathrm{os}}\mathbf C
=
\sum_P
\left[
\delta\mathbf A_P\mathbf C\mathbf B_P^{\mathrm T}
+
\mathbf A_P\mathbf C\delta\mathbf B_P^{\mathrm T}
\right],
\tag{66ah}
$$

so the packed-pair index can be distributed independently and accumulated into
thread-local structure images. This is an exact block contraction and retains
the direct unique-string-product-to-structure projection; no determinant-space
intermediate is introduced.

For a closed-shell unique-string space, the alpha and beta accepted and
directional pair kernels are identical after the shared string indexing,

$$
\mathcal U_\alpha=\mathcal U_\beta,
\qquad
\mathbf P_{IJ}^{\alpha}=\mathbf P_{IJ}^{\beta},
\qquad
\delta\mathbf P_{IJ}^{\alpha}=\delta\mathbf P_{IJ}^{\beta}.
\tag{66ai}
$$

The beta pair projection graph and directional pair payload can consequently
alias the alpha objects. This identity removes duplicate construction and
storage; it is not a spin approximation and is enabled only when the accepted
same-spin cache certifies the shared representation.

On the 12-electron, 12-orbital LOFLEA calculation with 924 unique strings per
spin, one 32-thread full exact HVP decreased from 86.56 to 30.24 s while its
infinity norm remained 16.1201446991. The peak resident set decreased from
19.47 to 18.63 GB. The largest individual reductions were the directional
structure construction, from 42.56 to 13.01 s, and the active-gradient
response, from 34.57 to 7.65 s. These changes reduce the constant multiplying
one exact probe in eq 66ae; they do not remove the quadratic accepted-point
unique-pair cache, which remains the principal memory-scaling target.

## 11. Implications for the present implementation

The current exact-context HVP differentiates orbital normalization, the inactive projector, active-space integrals, and the outer VB structure response. Its agreement with directional finite differences is evidence that the HVP is consistent with the present raw-coordinate computational graph.

However, this agreement does not validate the physical quotient coordinates. The pre-fix nonredundant-space construction required revision for the following reasons:

1. **Gauge sources must use the global inactive span.** Treating the restriction of another inactive orbital to the support of orbital $p$ as a local gauge vector is generally invalid. A support-restricted orbital is not, in general, a member of the original inactive span.
2. **Inactive additions to active orbitals are redundant.** Directions of the form $\delta\mathbf c_{\mathrm A,p}=\mathbf C_{\mathrm I}\boldsymbol\ell_p$ are annihilated by the projector in eq 14, up to an irrelevant active-orbital scaling induced by normalization. Retaining these directions introduces exact or near-zero modes.
3. **The existing rank diagnostics are not independent validation.** They test rank and intersection properties using the same per-orbital gauge model employed to construct the basis. They can therefore pass even if the assumed gauge space is physically incorrect.
4. **Euclidean local orthogonalization is coordinate dependent.** Orthogonalizing against the correct admissible gauge defines a valid algebraic complement, but does not by itself provide a scale-invariant physical norm for the trust-region method. Equations 27c--27g supply inexpensive local prewhitening, while eqs 31 and 32a now supply the coupled physical trust norm without a global metric factorization.

The corrected production construction uses the exact factorization in eqs 27a
and 27b followed by the local prewhitening in eqs 27c--27f. The previous
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

The same trajectory also exposed an implementation-level loss of curvature
history. Support-preserving canonicalization changed the representative at
every one of the first ten accepted points, and the previous implementation
therefore cleared the nominal eight-pair history at every step. Applying the
packed-space projection transport of eq 66h lowers the tenth-step energy from
$-341.8212742060$ to $-342.4821293288$ hartree and the corresponding projected
gradient two-norm from 1.91086 to 1.35965, without increasing the first-ten-step
HVP count. Over 30 accepted steps, the HVP direction count decreases from 540
to 418, the number of boundary steps from 24 to 20, and the number of steps
that encounter negative curvature from 18 to 15. The 32-thread wall time falls
from 453.45 to 364.67 s. The transported run satisfies both stopping criteria
at step 30 with

$$
E=-343.446302094450\ E_{\mathrm h},
\qquad
\lVert\mathbf g_{\mathrm{proj}}\rVert_{\infty}
=4.98\times10^{-4},
\qquad
|\Delta E|=9.54\times10^{-8}\ E_{\mathrm h}.
$$

These data do not imply that the full trajectory is quadratically convergent.
At the tenth point of the earlier no-transport trajectory, explicit assembly
of the 558-dimensional reduced Hessian gives eigenvalue bounds
$-0.48566$ and $90.3696\ E_{\mathrm h}$. At the distinct tenth point of the
transported trajectory, a deterministic matrix-free spectral probe first
detects negative curvature at dimension 24 and reaches a lowest Ritz value of
$-0.41990\ E_{\mathrm h}$ at dimension 32. The ordinary four-dimensional
gradient-residual subspace at that point does not contain this weakly coupled
mode. However, augmenting the local model with the 32-direction spectral
subspace at the existing radius predicts a decrease of $0.97590\ E_{\mathrm h}$
but produces an actual increase of $0.43479\ E_{\mathrm h}$. Its trust ratio is
$-0.4455$, so a correct trust-region method must reject that hard-case trial.
Thus negative curvature hidden from the gradient Krylov space is real, but
following it at the current large radius is not a shortcut to the local basin.
The approximately 30 accepted steps measure a nonconvex globalization phase
followed by local Newton convergence; they cannot be interpreted as 30 local
Newton steps.

At the common production tolerances of $10^{-3}$ for the projected-gradient
infinity norm and $10^{-7}$ hartree for the accepted energy change, the
current implementation gives the following complete 32-thread runs. Times
are end-to-end wall times on the same workstation and are intended as
regression data rather than machine-independent benchmarks.

| Input | Reduced dimension | Iterations | HVP directions | Final energy / hartree | Final projected gradient infinity norm | Wall time / s | Peak RSS / MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| F$_2$ sparse | 42 | 6 | 18 | -198.751155830527 | $1.03\times10^{-6}$ | 0.08 | 34.8 |
| benzene (`241`) | 432 | 8 | 46 | -230.720590392560 | $1.00\times10^{-5}$ | 3.36 | 1360.4 |
| MnF$_2$ | 957 | 13 | 234 | -1348.893351212391 | $9.11\times10^{-4}$ | 10.54 | 678.7 |
| FeCl$_2$ full AO | 3655 | 3 | 66 | -2181.617643403439 | $6.65\times10^{-4}$ | 7.00 | 569.4 |
| `240` | 558 | 28 | 438 | -343.446302349539 | $1.17\times10^{-4}$ | 368.62 | 2569.3 |

The preceding revision used an absolute infinity-norm shortcut in the inner
KKT test.  It required 12 accepted steps for benzene and 30 for `240`, with
final projected gradient infinity norms of $6.21\times10^{-5}$ and
$4.98\times10^{-4}$, respectively.  The norm-consistent rule removes the
linear local tails: benzene requires 8 accepted steps, while `240` requires 28
and reaches a lower energy and smaller gradient.  The `240` HVP count increases
from 418 to 438 and its wall time remains essentially unchanged, showing that
its first 26 steps are a nonconvex globalization problem rather than an inner
accuracy problem.  FeCl$_2$ similarly trades 34 additional HVP directions for
three fewer accepted steps and a lower stationary energy; improving the
full-AO preconditioner is therefore a separate performance task.

Earlier full-AO F$_2$, alternate C$_6$H$_6$, and `10698` timings were produced
by older solver revisions and are not mixed into this current-code table.

The fidelity-dependent acceptance rule was also tested on the larger
524-structure, two-state-average OEO/RI input with a 11670-dimensional reduced
orbital space.  At this point the work estimator classifies full outer response
as unaffordable.  Over the same first ten accepted steps, replacing the
unconditional model-agreement certificate by the conditional core-model rule
above reduces rejected trials from 18 to 4 and wall time from 500.42 to
493.02 s.  The projected-gradient infinity norm decreases from
$5.18\times10^{-3}$ to $3.63\times10^{-3}$, and the energy decreases from
$-457.0975983163$ to $-457.0976648047\ E_{\mathrm h}$.  HVP directions increase
from 211 to 234 and outer-response directions from 3 to 6, while peak RSS
remains approximately 2.4 GiB.  The increase in sampled curvature is therefore
repaid by substantially better outer progress rather than by a large short-run
wall-time reduction.  Neither ten-step run satisfies the projected-gradient
threshold, so these data establish improved globalization efficiency rather
than local quadratic convergence.  Affordable-response F$_2$, benzene, and
MnF$_2$ retain the strict trajectory; their regression iteration counts and
first ten MnF$_2$ steps are unchanged.

No molecule-dependent budgets or thresholds are used. Reproduction commands,
earlier coordinate audits, and additional limitations are recorded in
[Sparse quotient correction: validation record](sparse_quotient_fix_validation.md)
and
[Matrix-free curvature decomposition diagnostics](matrix_free_curvature_decomposition_diagnostics.md).

### 11.3 State-averaged response cancellation in the 524-structure case

The difficult 524-structure, two-state-average OEO/RI case was used to test
whether the transported sequential PSB correction was itself responsible for
the large sampled curvature and slow outer convergence.  A rank-revealing
simultaneous block multisecant prototype formed the transported displacement
and missing-curvature blocks

$$
\mathbf S=[\mathbf s_1,\ldots,\mathbf s_m],
\qquad
\mathbf Z=[\mathbf y_1-\mathbf H_k^{\mathrm c}\mathbf s_1,
            \ldots,
            \mathbf y_m-\mathbf H_k^{\mathrm c}\mathbf s_m],
$$

and used the thin singular-value decomposition

$$
\mathbf S=\mathbf Q\boldsymbol\Sigma\mathbf V^{\mathrm T},
\qquad
\mathbf T=\mathbf Z\mathbf V\boldsymbol\Sigma^{-1}.
$$

With

$$
\mathbf A=\operatorname{sym}(\mathbf Q^{\mathrm T}\mathbf T),
\qquad
\mathbf T_\perp=(\mathbf I-\mathbf Q\mathbf Q^{\mathrm T})\mathbf T,
$$

the simultaneous symmetric correction was

$$
\mathbf B
=\mathbf Q\mathbf A\mathbf Q^{\mathrm T}
+\mathbf T_\perp\mathbf Q^{\mathrm T}
+\mathbf Q\mathbf T_\perp^{\mathrm T}.
$$

Numerical rank was determined from the floating-point backward-error scale,
not from a molecular threshold.  The prototype preserved the important third
accepted-step decrease, and all 42 regression tests passed.  It nevertheless
gave a worse ten-step trajectory: the final energy was
$-457.097610990672\ E_{\mathrm h}$ and the projected-gradient infinity norm
was $4.53\times10^{-3}$, compared with $-457.097664804657\ E_{\mathrm h}$ and
$3.63\times10^{-3}$ for sequential PSB.  Wall time decreased from 498.6 to
434.9 s because the altered trajectory requested only three rather than six
outer-response directions; the lower cost did not compensate for the lost
outer progress.  Sampled spectral radii of approximately
$1.2$--$1.6\times10^4$ remained at accepted steps 4, 6, and 9.  At step 4 the
relative block compatibility defect was only 0.0124, excluding accumulated
sequential incompatibility as the primary origin of that spectral scale.  The
prototype was therefore removed rather than retained as an alternative path.

A separate fixed-point audit then exposed the physical source of the difficult
geometry.  The audit tool previously hard-coded a single selected state even
for a state-averaged input; it was corrected to use the consecutive selected
roots and equal weights parsed from `WSTATE`.  The corrected initial-point
audit has projected-gradient two-norm 0.1686088, identical to the production
trajectory.  The complete and core HVPs are symmetric on the six-direction
sampled space to relative errors $2.83\times10^{-12}$ and
$2.96\times10^{-13}$, respectively.  The soft-direction HVP linearity error is
$3.56\times10^{-8}$.  Thus neither a nonlinear HVP action nor a large
antisymmetric implementation defect explains this trajectory.

For a unit direction $\mathbf v$, decompose the complete action as

$$
\mathbf H\mathbf v
=\mathbf D\mathbf v
+(\mathbf H^{\mathrm c}-\mathbf D)\mathbf v
+\mathbf H^{\mathrm o}\mathbf v,
$$

where $\mathbf D$ is the target-orbital block diagonal of the computational
core, the second term is cross-target core coupling, and
$\mathbf H^{\mathrm o}$ is outer response.  The measured signed Rayleigh
contributions are

| Direction | Diagonal core | Cross-target core | Outer response | Complete |
|---|---:|---:|---:|---:|
| projected gradient | 12.5911 | 5.63026 | -0.0166605 | 18.2047 |
| six-direction minimum-residual step | 0.0118130 | 0.00363556 | -0.00982474 | 0.00562386 |
| softest sampled Ritz direction | 0.00593426 | 0.0108592 | -0.0159734 | 0.000820048 |

Outer response is small on the gradient direction, with norm only 0.0207 times
the complete HVP norm.  It is not small on the directions that determine a
Newton step: the corresponding norm ratios are 2.00 for the sampled Newton
step and 6.95 for the softest Ritz direction.  In the latter direction two
positive core contributions are almost cancelled by negative relaxation
curvature.  A six-direction complete minimum-residual solve reaches relative
residual 0.207, whereas the core solve reaches 0.675 and its step has complete
residual 0.660.  A gradient-direction response test therefore systematically
misses the important physics.

The outer action was further separated into the fixed-selected-state
active-gradient derivative and the structure-state derivative,

$$
\mathbf H^{\mathrm o}\mathbf v
=\mathbf H^{\mathrm{loc}}\mathbf v
+\mathbf H^{\mathrm{str}}\mathbf v.
$$

On the same six-direction subspace, the relative skew norms of
$\mathbf H^{\mathrm{loc}}$ and $\mathbf H^{\mathrm{str}}$ are
$1.04\times10^{-13}$ and $2.11\times10^{-11}$, respectively, and the
projected additivity defect is $1.18\times10^{-14}$.  Thus both components are
linear and self-adjoint to numerical precision.  Nevertheless, on the softest
sampled direction,

$$
\frac{\lVert\mathbf H^{\mathrm{loc}}\mathbf v\rVert}
     {\lVert\mathbf H\mathbf v\rVert}=17.91,
\qquad
\frac{\lVert\mathbf H^{\mathrm{str}}\mathbf v\rVert}
     {\lVert\mathbf H\mathbf v\rVert}=18.27,
$$

while their signed Rayleigh contributions are

$$
\mathbf v^{\mathrm T}\mathbf H^{\mathrm{loc}}\mathbf v
=-1.59338\times10^{-2},
\qquad
\mathbf v^{\mathrm T}\mathbf H^{\mathrm{str}}\mathbf v
=-3.96054\times10^{-5}.
$$

The small structure Rayleigh quotient therefore does not imply a negligible
structure action: two large response vectors cancel in components orthogonal
to $\mathbf v$.  Any approximation that retains only one response component
destroys this vector cancellation even if it reproduces the scalar curvature
along one direction.

Single-direction timings on 32 CPU cores quantify the associated trade-off.
The computational core, a fused core plus local-active action, and the complete
HVP require 2.19, 3.38, and 3.83 s, respectively.  Replacing the core operator
by the fused core plus local-active action is therefore only 12% cheaper than
the complete HVP and 54% more expensive than the core.  More importantly, its
first three accepted steps leave the projected-gradient infinity norm at
$2.15\times10^{-2}$ after 3 min 42 s, essentially reproducing the initial
core-only stagnation.  This experimental path was removed.  The result rules
out component deletion as a useful response approximation.

A complementary forced-complete-HVP experiment separates the outer optimizer
from the fidelity policy.  The first exact subproblem uses six HVP directions
in three width-two calls.  Its relative KKT residual is 0.316, below the
inexact-Newton forcing term 0.411, and the trial is accepted without rejection
with trust ratio 0.956.  In one outer step the energy decreases by
$1.25703\times10^{-3}\ E_{\mathrm h}$ and the projected-gradient infinity
norm decreases from $2.17\times10^{-2}$ to $5.08\times10^{-3}$.  A second
complete-HVP step reaches $2.00\times10^{-3}$.  The core-only production path,
by contrast, remains at $3.63\times10^{-3}$ after ten accepted steps.  This
establishes that the trust-region/Krylov framework can generate the required
near-Newton direction when the complete response curvature is present.

The direct complete-HVP route is not yet competitive in wall time.  Two steps
use 28 HVP directions in 14 width-two calls, 324.6 s of HVP time, 289.5 s of
outer-response time, and 370.1 s end to end; peak RSS is 2.22 GiB.  The first
step alone costs 71.9 s in HVP actions, of which 63.9 s is outer response.
For this RI input the current block interface evaluates its columns through
the scalar factor-native path.  A same-direction width-two check is exact to
reported precision and gives only a 1.21-fold wall-time speedup, from 4.04 s
per scalar action to 3.34 s per direction in the block.  The substantially
higher average cost on residual Krylov directions must therefore be profiled
separately; it cannot be attributed to a numerical block-action discrepancy.

The apparent direction dependence was traced to a diagnostic mismatch rather
than to the orbital direction.  The fixed-point benchmark initially called an
evaluator overload that defaulted to dense structure diagonalization and
therefore retained the complete eigenspectrum.  It ignored the input keyword
`eigensolver=davidson`.  After the benchmark was corrected to propagate the
requested eigensolver, one Davidson complete HVP costs 13.05 s, including
9.22 s in the eigensystem response.  The shifted response equation takes 84
linear iterations and 87 H/S block actions to reach relative residual
$6.76\times10^{-6}$.  The corresponding dense/full-spectrum response takes
approximately 0.11 s and no iterative response solve.  Thus Davidson is not
intrinsically slower for the ground-state eigenproblem; the expensive stage is
the repeatedly overconverged directional shifted solve used inside the HVP.

The nested solves should obey one accuracy hierarchy.  Let $\eta_k$ be the
inexact-Newton forcing term at accepted orbital point $k$, and let
$\tau_{\mathrm{final}}$ denote the response tolerance implied by the final
structure energy and gradient requirements.  The implemented response target
is

$$
\tau_{\mathrm{resp},k}
=\max\!\left(\tau_{\mathrm{final}},\eta_k^2\right).
$$

Because the current forcing sequence satisfies
$\eta_k=O(\sqrt{\lVert\mathbf g_k\rVert})$ away from its numerical bounds,
the directional structure solve has error
$O(\lVert\mathbf g_k\rVert)$ far from convergence and automatically tightens
to the requested final accuracy near stationarity.  No molecular dimension,
element identity, or iteration number enters this rule.  Strict response
accuracy is retained for isolated candidate-certification probes; the relaxed
tolerance is used only inside an inexact complete-HVP Krylov solve.

On the 524-structure case the first-step requested tolerance is 0.1686.  The
HVP time decreases from 69.6 to 26.3 s, eigensystem-response time from 53.5 to
10.5 s, outer-response time from 61.1 to 17.8 s, and end-to-end time from
104.7 to 58.5 s.  The accepted energy decrease changes only from
$1.25703\times10^{-3}$ to $1.25613\times10^{-3}\ E_{\mathrm h}$, and the
projected-gradient infinity norm changes from $5.08\times10^{-3}$ to
$5.12\times10^{-3}$.  After two steps the tolerance has automatically
tightened to 0.0541; wall time is 184.9 s rather than 370.1 s, while the
projected-gradient infinity norm is $1.94\times10^{-3}$ rather than
$2.00\times10^{-3}$.  The relaxed nested solve therefore preserves the outer
trajectory while approximately halving the complete-HVP time over the first
two accepted steps.

The five-step forced-complete trajectory clarifies the remaining limitation.
At step 3 the projected-gradient infinity norm reaches
$7.28\times10^{-4}$ and the energy is $-457.0979184159\ E_{\mathrm h}$, but
the energy change is still $5.65\times10^{-5}\ E_{\mathrm h}$.  Subsequent
steps lower the energy further to $-457.0980471425\ E_{\mathrm h}$, far below
the ten-step core value $-457.0976648047\ E_{\mathrm h}$, while the fifth-step
gradient is $1.12\times10^{-3}$.  The run is therefore exploring a materially
lower valley rather than merely polishing the old core trajectory.

Steps 3 and 4 both use 34 directions, including the two-direction interior
pilot beyond the nominal 32-direction work tranche.  Their KKT relative
residuals are 0.399 and 0.916, compared with forcing terms 0.135 and 0.0936;
both stop by the subspace work budget.  They consume 198 and 224 s of HVP time,
respectively.  Hence the remaining loss of local Newton behavior is no longer
caused by the accuracy of an individual Davidson response solve.  It is caused
by an orbital Krylov space that becomes too large before resolving the coupled
Newton residual.  Simply removing the work limit would improve the formal
subproblem residual but would not be a competitive algorithm: the five-step
run already costs 1068 s.  The next optimization target is therefore a
response-aware preconditioner or coupled orbital--structure block solve that
reduces the required subspace dimension, not a larger fixed Krylov budget.

An auxiliary three-root Davidson solve places the unselected boundary root
approximately $0.205\ E_{\mathrm h}$ above the second selected root at the
initial point.  The cancellation is consequently not attributable to a
second/third-root near degeneracy or simple root flipping.  It is a collective
orbital--structure relaxation effect: fixed-structure core curvature makes the
mode appear hard, while response makes the complete mode soft.  A core-only
Krylov space followed by one exact action on its final candidate can certify
that the candidate is inadequate, but it cannot generate a direction that is
soft only after response is included.  Cross-point PSB learns such curvature
only after accepted motion and becomes progressively less transferable as the
response-softened subspace rotates.

The resulting algorithmic target is therefore a same-point,
response-informed Krylov enrichment, not a more aggressive cross-point secant
fit.  Exact outer actions must be selected by the unresolved complete Newton
residual and used to create new search directions.  A practical method must
bound this enrichment by measured wall time and reuse block structure-response
work; simply applying the complete HVP to every Krylov vector remains too
expensive for the large-active-space regime.

Equivalently, let $\mathbf p$ denote the orbital step and $\mathbf z$ the
combined tangent response of the selected structure states.  The coupled
Newton equation has the schematic saddle-point form

$$
\begin{bmatrix}
\mathbf A & \mathbf B^{\mathrm T}\\
\mathbf B & \mathbf C
\end{bmatrix}
\begin{bmatrix}
\mathbf p\\
\mathbf z
\end{bmatrix}
=-
\begin{bmatrix}
\mathbf g\\
\mathbf 0
\end{bmatrix},
$$

where $\mathbf A$ contains the fixed-state orbital curvature, $\mathbf B$ is
the orbital--structure coupling, and $\mathbf C$ is the projected
structure-state Hessian.  Explicit elimination gives the relaxed orbital
operator

$$
\mathbf H_{\mathrm{rel}}
=\mathbf A-\mathbf B^{\mathrm T}\mathbf C^{\dagger}\mathbf B.
$$

The present complete HVP realizes this Schur-complement physics through a new
directional response calculation for every orbital Krylov vector.  A coupled
matrix-free block solve is the principled route to retain the cancellation
without repeatedly converging an eliminated response problem.  It must be
constructed in the normalized structure tangent space, preserve the
state-average weights, and use direct H/S actions so that neither a dense
structure Hessian nor a dense orbital Hessian is formed.

### 11.4 Matrix-free coupled orbital--structure Newton equation

For an equal-weight cluster containing $m$ selected structure states, collect
the accepted generalized eigenvectors in
$\mathbf C\in\mathbb R^{n_s\times m}$ and their energies in the diagonal
matrix $\boldsymbol\Lambda$.  They satisfy

$$
\mathbf H\mathbf C=\mathbf S\mathbf C\boldsymbol\Lambda,
\qquad
\mathbf C^{\mathrm T}\mathbf S\mathbf C=\mathbf I.
$$

An orbital direction $\mathbf p$ induces the structure-matrix images

$$
\mathbf F(\mathbf p)
=D\mathbf H[\mathbf p]\mathbf C
-D\mathbf S[\mathbf p]\mathbf C\boldsymbol\Lambda,
\qquad
\mathbf N(\mathbf p)
=\frac{1}{2}\mathbf C^{\mathrm T}
D\mathbf S[\mathbf p]\mathbf C.
$$

The selected-state tangent $\mathbf Z$ is defined in the symmetric horizontal
gauge

$$
\mathbf C^{\mathrm T}\mathbf S\mathbf Z
=-\frac{1}{2}\mathbf C^{\mathrm T}
D\mathbf S[\mathbf p]\mathbf C.
$$

Consequently, a full matrix $\mathbf M\in\mathbb R^{m\times m}$ is required
as the multiplier of the horizontal constraint.  Replacing $\mathbf M$ by one
scalar per state is valid only for isolated one-state clusters; for a
multistate cluster it reintroduces selected--selected inverse gaps that are
pure gauge for an equal-weight average.  Define the matrix-free response block

$$
\mathcal C
\begin{bmatrix}\mathbf Z\\\mathbf M\end{bmatrix}
=
\begin{bmatrix}
\mathbf H\mathbf Z
-\mathbf S\mathbf Z\boldsymbol\Lambda
+\mathbf S\mathbf C\mathbf M\\
\mathbf C^{\mathrm T}\mathbf S\mathbf Z
\end{bmatrix}.
$$

For a matrix-free projected solution, both the horizontal constraint and the
multiplier image select the Euclidean complement of
$\operatorname{range}(\mathbf S\mathbf C)$.  If
$\mathbf Q_{SC}$ is an orthonormal basis for this range, the symmetric
projector is

$$
\mathbf P_{SC}
=
\mathbf I-\mathbf Q_{SC}\mathbf Q_{SC}^{\mathrm T},
$$

and the external response for column $i$ is obtained from

$$
\mathbf P_{SC}
(\mathbf H-E_i\mathbf S)
\mathbf P_{SC}\mathbf z_i
=
-\mathbf P_{SC}\mathbf f_i.
$$

Projecting against $\operatorname{range}(\mathbf C)$ instead is incorrect in
a nonorthogonal structure basis: the residual is then left in
$\operatorname{range}(\mathbf C)$, whereas the multiplier can absorb only an
$\mathbf S\mathbf C$ component.  The two spaces coincide only in special
commuting or orthogonal cases.  The $\mathbf P_{SC}$ construction preserves a
symmetric projected operator and enforces
$\mathbf C^{\mathrm T}\mathbf S\mathbf Z_{\mathrm{ext}}=\mathbf 0$ directly.

For a per-state weight $w$, the symmetric coupled coordinates are
$\mathbf q=\sqrt{2w}\,\operatorname{vec}(\mathbf Z,\mathbf M)$.  The same
factor multiplies the orbital-to-response forcing
$\operatorname{vec}(\mathbf F,\mathbf N)$.  This scaling makes the two mixed
actions exact Euclidean adjoints; it is not a numerical tuning parameter.
Different state weights define different clusters, because rotations between
states of unequal weight are physical rather than gauge degrees of freedom.

At a trust-region shift $\lambda$, the coupled Newton equation is

$$
\begin{bmatrix}
\mathbf A+\lambda\mathbf G & \mathbf B^{\mathrm T}\\
\mathbf B & \mathbf C
\end{bmatrix}
\begin{bmatrix}\mathbf p\\\mathbf q\end{bmatrix}
=-
\begin{bmatrix}\mathbf g\\\mathbf 0\end{bmatrix},
\qquad
\|\mathbf p\|_{\mathbf G}\leq\Delta.
$$

Here $\mathbf G$ is the nonredundant orbital metric.  The shift and the trust
constraint act only on $\mathbf p$: the structure tangent is an induced
first-order response, not an independently bounded physical displacement.
Eliminating $\mathbf q$ gives

$$
\left(
\mathbf A-\mathbf B^{\mathrm T}\mathbf C^{-1}\mathbf B
+\lambda\mathbf G
\right)\mathbf p=-\mathbf g,
$$

which is exactly the relaxed matrix-free orbital Newton equation.  The coupled
form therefore changes the linear-algebra realization, not the energy model.
Its advantage is that MINRES can reduce the joint residual using inexpensive
H/S block actions without converging a separate structure-response problem for
every orbital Krylov vector.

The trust-region multiplier is accepted only when the explicit coupled KKT
residual satisfies the inexact-Newton forcing condition and either
$\lambda=0$ with an interior orbital step or

$$
\|\mathbf p\|_{\mathbf G}=\Delta,
\qquad \lambda>0.
$$

For an indefinite reduced Hessian, the shifted reduced operator must also be
positive semidefinite on the orbital tangent space.  A norm match alone is not
a sufficient trust-region certificate.  This condition is obtained from the
same Lanczos spectral information used by the coupled Krylov solve, rather
than from a fixed iteration count or a molecule-dependent shift.

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
