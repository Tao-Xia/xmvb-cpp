# Matrix-Free Norm-Extended Optimization for VBSCF

## 1. Purpose and scope

This note derives a norm-extended optimization (NEO) method for the VBSCF
energy used in `xmvb-cpp`.  The derivation is adapted to four properties that
distinguish the present problem from a conventional orthonormal-orbital
CASSCF implementation:

1. the VB orbitals are generally nonorthogonal and may have immutable, strict
   AO supports;
2. the orbital variables live in a nonredundant quotient tangent space rather
   than in an unrestricted orbital-rotation space;
3. the VB structure coefficients solve a generalized eigenvalue problem with
   an orbital-dependent overlap matrix; and
4. the Hessian is available only through matrix-free Hessian-vector products
   (HVPs).

The objective is a genuine restricted-step second-order method.  L-BFGS
history may be used to precondition the iterative eigensolver, but it is not
part of the NEO Hessian and cannot replace the NEO residual certificate.

Two related formulations are derived below:

- **relaxed-orbital NEO**, in which the structure problem is solved at every
  orbital point and eliminated before the trust-region problem is formed;
- **coupled orbital--structure NEO**, which is the direct analogue of the
  original second-order MCSCF NEO method.

These formulations have the same unshifted Newton limit but are not identical
away from that limit.  Their response equations must therefore not be mixed.

## 2. VBSCF stationary problem

Let $\mathbf x$ collect the stored, strictly sparse orbital coefficients.  At
a fixed orbital point, the VB structure coefficients solve

$$
\mathbf H(\mathbf x)\mathbf a_s
=
E_s(\mathbf x)\mathbf S(\mathbf x)\mathbf a_s,
\qquad
\mathbf a_r^{\mathrm T}\mathbf S(\mathbf x)\mathbf a_s
=
\delta_{rs}.
\tag{1}
$$

For one state, the orbitally relaxed objective is the corresponding stationary
eigenvalue.  For a state-averaged calculation it is

$$
\mathcal E(\mathbf x)
=
\sum_{s\in\mathcal A}w_s E_s(\mathbf x),
\qquad
w_s>0,
\qquad
\sum_{s\in\mathcal A}w_s=1.
\tag{2}
$$

The present implementation uses equal weights unless stated otherwise.  Equal
weights are important: rotations within an isolated averaged eigenspace do not
change eq 2 and are structure-space gauge directions.  Response variables
must exclude normalization, rotations within the equally weighted selected
space, and any exact null directions.

For a single state, introduce the stationary Lagrangian

$$
\mathscr L_s(\mathbf x,\mathbf a_s,E_s)
=
\mathbf a_s^{\mathrm T}\mathbf H(\mathbf x)\mathbf a_s
-
E_s
\left[
\mathbf a_s^{\mathrm T}\mathbf S(\mathbf x)\mathbf a_s-1
\right].
\tag{3}
$$

The generalized Hellmann--Feynman derivative at a stationary structure vector
is

$$
D E_s[\delta\mathbf x]
=
\mathbf a_s^{\mathrm T}
\left(
D\mathbf H[\delta\mathbf x]
-E_sD\mathbf S[\delta\mathbf x]
\right)
\mathbf a_s.
\tag{4}
$$

Equation 4 is only a first derivative.  The second derivative contains the
response of $\mathbf a_s$ and is the source of orbital--structure coupling.

## 3. Nonredundant sparse orbital coordinates

At accepted point $k$, let $\mathbf U_k$ be the support-preserving horizontal
lift from reduced orbital coordinates to packed raw coefficients.  The local
additive chart is

$$
\mathbf x(\mathbf p)
=
\mathbf x_k+\mathbf U_k\mathbf p.
\tag{5}
$$

The columns of $\mathbf U_k$ span the strictly sparse tangent space modulo the
inactive-subspace and orbital-ray gauge directions.  They are held fixed
inside one local model.  Consequently,

$$
\mathbf g_o
=
\mathbf U_k^{\mathrm T}\mathbf g_x,
\tag{6}
$$

and an exact reduced orbital HVP is

$$
\mathbf H_o\mathbf v
=
\mathbf U_k^{\mathrm T}
D\mathbf g_x(\mathbf x_k)[\mathbf U_k\mathbf v].
\tag{7}
$$

All derivatives of orbital normalization, the moving inactive projector,
active-orbital projection, integral transformation, and structure response
belong inside $D\mathbf g_x$.  No derivative of $\mathbf U_k$ appears in eq 7
because eq 5 is the pullback chart used for that accepted-point model.

The quotient tangent has a physical positive-definite metric

$$
\mathbf M_o
=
\mathbf U_k^{\mathrm T}\mathbf M_{\mathrm{phys}}\mathbf U_k,
\qquad
\|\mathbf p\|_o^2
=
\mathbf p^{\mathrm T}\mathbf M_o\mathbf p.
\tag{8}
$$

The NEO step restriction must use eq 8.  A Euclidean norm of arbitrary packed
coefficients would depend on the chosen sparse representative and would not be
a well-defined physical step length.

## 4. Structure tangent coordinates

For one isolated state, choose an accepted-point structure tangent basis
$\mathbf W_s$ satisfying

$$
\mathbf W_s^{\mathrm T}\mathbf S\mathbf W_s=\mathbf I,
\qquad
\mathbf W_s^{\mathrm T}\mathbf S\mathbf a_s=\mathbf 0.
\tag{9}
$$

For an equally weighted state average, $\mathbf W_s$ is additionally projected
outside the complete selected eigenspace.  This removes energetically
irrelevant internal rotations.  A local structure variation is

$$
\delta\mathbf a_s=\mathbf W_s\mathbf q_s
+\text{the normalization component}.
\tag{10}
$$

With eq 9, the structure tangent metric is Euclidean.  More generally it will
be denoted by $\mathbf M_s$ so that the derivation remains valid for a
nonorthonormal tangent representation.

For a fixed accepted orbital point, the simple-state structure Hessian in the
orthonormal tangent basis is, up to the convention used for the quadratic
coefficient,

$$
\mathbf C_s
=
2\mathbf W_s^{\mathrm T}
\left(\mathbf H-E_s\mathbf S\right)
\mathbf W_s.
\tag{11}
$$

The corresponding orbital-to-structure coupling action contains

$$
\mathbf B_s\mathbf p
=
2\mathbf W_s^{\mathrm T}
\left(
D\mathbf H[\mathbf U_k\mathbf p]
-E_sD\mathbf S[\mathbf U_k\mathbf p]
\right)
\mathbf a_s
+\text{chart and constraint terms}.
\tag{12}
$$

The final phrase in eq 12 is essential.  In production code, $\mathbf B_s$ is
defined as the exact mixed derivative of the stationary Lagrangian, not by
using the displayed leading term alone.  This definition automatically
includes the chosen normalization chart and the orbital-dependent generalized
metric.

For the state average, define weighted blocks directly by

$$
\mathbf A
=
\sum_s w_s\mathbf A_s,
\qquad
\mathbf B
=
\begin{pmatrix}
w_1\mathbf B_1\\
\vdots\\
w_m\mathbf B_m
\end{pmatrix},
\qquad
\mathbf C
=
\operatorname{diag}
\left(w_1\mathbf C_1,\ldots,w_m\mathbf C_m\right).
\tag{13}
$$

Here $\mathbf A_s$ is the fixed-structure orbital Hessian of state $s$.  This
weighting gives

$$
\mathbf B^{\mathrm T}\mathbf C^{\dagger}\mathbf B
=
\sum_s w_s
\mathbf B_s^{\mathrm T}\mathbf C_s^{\dagger}\mathbf B_s.
\tag{14}
$$

For a nearly degenerate selected cluster, separate state-by-state inverses are
not a stable definition.  Equations 11--14 must then be interpreted as the
projected block or Sylvester response of the isolated averaged subspace.

## 5. The coupled quadratic model

Collect the orbital and structure tangent variables as

$$
\mathbf z
=
\begin{pmatrix}
\mathbf p\\
\mathbf q
\end{pmatrix},
\qquad
\mathbf g
=
\begin{pmatrix}
\mathbf g_o\\
\mathbf 0
\end{pmatrix}.
\tag{15}
$$

The zero structure-gradient block follows from solving eq 1 at the accepted
point.  The exact second-order local model is

$$
m(\mathbf z)
=
\mathcal E_k
+\mathbf g^{\mathrm T}\mathbf z
+\frac12\mathbf z^{\mathrm T}\mathbf K\mathbf z,
\tag{16}
$$

with

$$
\mathbf K
=
\begin{pmatrix}
\mathbf A & \mathbf B^{\mathrm T}\\
\mathbf B & \mathbf C
\end{pmatrix},
\qquad
\mathbf M
=
\begin{pmatrix}
\mathbf M_o & \mathbf 0\\
\mathbf 0 & \mathbf M_s
\end{pmatrix}.
\tag{17}
$$

The full Newton equation is

$$
\mathbf K\mathbf z_N=-\mathbf g.
\tag{18}
$$

If $\mathbf C$ is invertible on the projected response space, eliminating
$\mathbf q$ gives

$$
\mathbf q_N=-\mathbf C^{\dagger}\mathbf B\mathbf p_N,
\tag{19}
$$

and

$$
\mathbf H_{\mathrm{rel}}
=
\mathbf A
-\mathbf B^{\mathrm T}\mathbf C^{\dagger}\mathbf B,
\qquad
\mathbf H_{\mathrm{rel}}\mathbf p_N=-\mathbf g_o.
\tag{20}
$$

Equation 20 is the relaxed VBSCF orbital Hessian.  It is not an empirical
curvature correction and it must not be added on top of another approximation
that already contains the same response contribution.

## 6. Metric trust-region equations

The coupled trust-region problem is

$$
\min_{\mathbf z}
\left[
\mathbf g^{\mathrm T}\mathbf z
+\frac12\mathbf z^{\mathrm T}\mathbf K\mathbf z
\right]
\quad\text{subject to}\quad
\mathbf z^{\mathrm T}\mathbf M\mathbf z\leq\Delta^2.
\tag{21}
$$

Its global optimality conditions are

$$
\left(\mathbf K+\lambda\mathbf M\right)\mathbf z=-\mathbf g,
\tag{22a}
$$

$$
\mathbf K+\lambda\mathbf M\succeq\mathbf 0,
\qquad
\lambda\geq0,
\tag{22b}
$$

$$
\mathbf z^{\mathrm T}\mathbf M\mathbf z\leq\Delta^2,
\qquad
\lambda
\left(\mathbf z^{\mathrm T}\mathbf M\mathbf z-\Delta^2\right)=0.
\tag{22c}
$$

The positive semidefiniteness condition in eq 22b is what resolves negative
curvature.  Merely solving a shifted linear equation without checking eq 22b
does not certify a minimum of the quadratic model.

## 7. NEO as a generalized augmented-Hessian eigenproblem

Introduce the gradient-scaling parameter $\alpha>0$ and the norm-extended
pencil

$$
\mathbf L_\alpha
=
\begin{pmatrix}
0 & \alpha\mathbf g^{\mathrm T}\\
\alpha\mathbf g & \mathbf K
\end{pmatrix},
\qquad
\mathbf G
=
\begin{pmatrix}
1 & \mathbf 0^{\mathrm T}\\
\mathbf 0 & \mathbf M
\end{pmatrix}.
\tag{23}
$$

Let the lowest generalized eigenpair satisfy

$$
\mathbf L_\alpha
\begin{pmatrix}
\beta\\
\mathbf y
\end{pmatrix}
=
\mu
\mathbf G
\begin{pmatrix}
\beta\\
\mathbf y
\end{pmatrix},
\qquad
\beta\neq0.
\tag{24}
$$

Define the physical step by the projective ratio

$$
\mathbf z(\alpha)
=
\frac{\mathbf y}{\alpha\beta}.
\tag{25}
$$

The lower block of eq 24 gives

$$
\alpha\beta\mathbf g+\mathbf K\mathbf y
=
\mu\mathbf M\mathbf y.
\tag{26}
$$

Substituting eq 25 yields

$$
\left(\mathbf K-\mu\mathbf M\right)\mathbf z
=
-\mathbf g.
\tag{27}
$$

The upper block gives

$$
\alpha\mathbf g^{\mathrm T}\mathbf y=\mu\beta,
\tag{28}
$$

and therefore

$$
\mu=\alpha^2\mathbf g^{\mathrm T}\mathbf z.
\tag{29}
$$

For a descent step, $\mathbf g^{\mathrm T}\mathbf z<0$, so $\mu<0$.
Identifying

$$
\lambda=-\mu
\tag{30}
$$

turns eq 27 into the trust-region stationarity equation 22a.  Because $\mu$
is the lowest generalized eigenvalue, generalized eigenvalue interlacing gives

$$
\mathbf K-\mu\mathbf M\succeq\mathbf 0,
\tag{31}
$$

which is exactly eq 22b.  Thus the lowest root, rather than an arbitrary root
of the augmented problem, is essential for ground-state minimization.

The scale $\alpha$ is chosen so that

$$
\|\mathbf z(\alpha)\|_{\mathbf M}
=
\sqrt{\mathbf z(\alpha)^{\mathrm T}
\mathbf M\mathbf z(\alpha)}
=
\Delta
\tag{32}
$$

when the boundary is active.  If the unshifted Newton step is inside the trust
region and the relevant Hessian is positive semidefinite, the $\alpha\to0$
limit gives $\mu\to0$ and recovers eq 18.  No fixed iteration count is part of
this definition.

The assumption $\beta\neq0$ in eq 24 describes the regular case.  If the
gradient is orthogonal to a lowest-curvature eigenspace, the lowest augmented
root can have $\beta=0$.  This is the trust-region hard case, not numerical
convergence.  Let $\theta_{min}$ be the lowest generalized eigenvalue of
$(\mathbf K,\mathbf M)$ and set

$$
\lambda_*=max(0,-\theta_{min}).
\tag{32a}
$$

The minimum-norm particular solution is

$$
\mathbf z_p
=
-\left(\mathbf K+\lambda_*\mathbf M\right)^{\dagger}\mathbf g.
\tag{32b}
$$

It is completed by a vector $\mathbf u_{min}$ in the null space of
$\mathbf K+\lambda_*\mathbf M$,

$$
\mathbf z
=
\mathbf z_p+\tau\mathbf u_{min},
\qquad
\mathbf u_{min}^{\mathrm T}\mathbf M\mathbf u_{min}=1,
\qquad
\|\mathbf z\|_{\mathbf M}=\Delta,
\tag{32c}
$$

with the sign of $\tau$ chosen to minimize the quadratic model.  A robust NEO
implementation must detect this case from $\beta$, the eigengap, and the true
KKT residual rather than dividing by a small $\beta$.

## 8. Matrix-free NEO action

The augmented operator acts on a trial vector without constructing either the
Hessian or the augmented matrix:

$$
\mathbf L_\alpha
\begin{pmatrix}
\beta\\
\mathbf y
\end{pmatrix}
=
\begin{pmatrix}
\alpha\mathbf g^{\mathrm T}\mathbf y\\
\alpha\beta\mathbf g+\mathbf K\mathbf y
\end{pmatrix}.
\tag{33}
$$

For $\mathbf y=(\mathbf y_o,\mathbf y_s)^{\mathrm T}$, the coupled Hessian
action is

$$
\mathbf K\mathbf y
=
\begin{pmatrix}
\mathbf A\mathbf y_o+\mathbf B^{\mathrm T}\mathbf y_s\\
\mathbf B\mathbf y_o+\mathbf C\mathbf y_s
\end{pmatrix}.
\tag{34}
$$

Equation 34 requires only four operator kernels:

1. fixed-structure orbital curvature $\mathbf A\mathbf y_o$;
2. forward orbital--structure coupling $\mathbf B\mathbf y_o$;
3. adjoint coupling $\mathbf B^{\mathrm T}\mathbf y_s$; and
4. a direct structure sigma action $\mathbf C\mathbf y_s$.

The forward and adjoint coupling actions must be an exact transpose pair in
the metrics used by eq 17.  This is both a mathematical requirement and a
directly testable implementation invariant.

An iterative generalized Davidson or Lanczos-type solver may find the lowest
root of eq 24.  “Davidson” here denotes the matrix-free eigensolver for the NEO
microproblem; it is independent of whether dense diagonalization or Davidson
is used to solve the accepted-point VB structure eigenproblem in eq 1.

## 9. Relaxed-orbital NEO versus coupled NEO

This distinction is central to the VBSCF implementation.

### 9.1 Relaxed-orbital NEO

If eq 1 is solved to the required accuracy at every orbital point, the outer
objective is already the relaxed function $\mathcal E(\mathbf x)$.  Its exact
local Hessian is eq 20.  The orbital-only trust problem is

$$
\min_{\mathbf p}
\left[
\mathbf g_o^{\mathrm T}\mathbf p
+\frac12\mathbf p^{\mathrm T}
\mathbf H_{\mathrm{rel}}\mathbf p
\right]
\quad\text{subject to}\quad
\mathbf p^{\mathrm T}\mathbf M_o\mathbf p\leq\Delta_o^2.
\tag{35}
$$

Its stationarity equation is

$$
\left(
\mathbf H_{\mathrm{rel}}+\lambda\mathbf M_o
\right)\mathbf p
=
-\mathbf g_o.
\tag{36}
$$

The corresponding NEO pencil is

$$
\begin{pmatrix}
0 & \alpha\mathbf g_o^{\mathrm T}\\
\alpha\mathbf g_o & \mathbf H_{\mathrm{rel}}
\end{pmatrix}
\begin{pmatrix}
\beta\\
\mathbf y_o
\end{pmatrix}
=
\mu
\begin{pmatrix}
1 & \mathbf 0^{\mathrm T}\\
\mathbf 0 & \mathbf M_o
\end{pmatrix}
\begin{pmatrix}
\beta\\
\mathbf y_o
\end{pmatrix}.
\tag{37}
$$

Here the unshifted structure inverse in eq 20 is correct: it is part of the
derivative of the already relaxed objective.  The trust-region shift is applied
after structure elimination.

### 9.2 Coupled orbital--structure NEO

The classic MCSCF-style formulation applies the trust norm to the joint step
$(\mathbf p,\mathbf q)$.  Eliminating $\mathbf q$ from eq 22a gives

$$
\mathbf q(\lambda)
=
-\left(\mathbf C+\lambda\mathbf M_s\right)^{-1}
\mathbf B\mathbf p,
\tag{38}
$$

and

$$
\left[
\mathbf A+\lambda\mathbf M_o
-\mathbf B^{\mathrm T}
\left(\mathbf C+\lambda\mathbf M_s\right)^{-1}
\mathbf B
\right]\mathbf p
=
-\mathbf g_o.
\tag{39}
$$

Therefore the coupled NEO contains a **shift-dependent response operator**.
Replacing the inverse in eq 39 by $\mathbf C^{\dagger}$ while retaining the
joint trust norm does not solve the coupled trust-region problem.

Equations 36 and 39 are nevertheless both valid because they constrain
different models:

- eq 36 constrains only the orbital displacement of the exactly relaxed
  energy surface;
- eq 39 constrains the simultaneous orbital and structure displacement.

They become identical in the Newton limit $\lambda\to0$, after the structure
step is eliminated.  Away from that limit, one formulation must be selected
and used consistently from the operator through the norm, residual, predicted
reduction, and acceptance test.

### 9.3 Projected response without a relaxed-HVP oracle

The current macroiteration retracts only the orbital step and solves the
structure eigenproblem again at every trial point.  Its finite trial path is
therefore the relaxed orbital energy in eq 35.  Consequently, the production
trust norm must contain only $\mathbf p^{\mathrm T}\mathbf M_o\mathbf p$.
The structure response remains an internal variable satisfying the unshifted
stationarity equation in eq 19; the trust-region multiplier must not be added
to $\mathbf C$.

The action of $\mathbf H_{\mathrm{rel}}$ need not be evaluated by solving a
full structure-response equation after every orbital HVP.  Let
$\mathbf Q_o$ and $\mathbf Q_s$ be independently expanded orthonormal bases
for the orbital and horizontal structure spaces.  Define

$$
\begin{aligned}
\mathbf A_r&=\mathbf Q_o^{\mathrm T}\mathbf A\mathbf Q_o,
&\mathbf B_r&=\mathbf Q_s^{\mathrm T}\mathbf B\mathbf Q_o,\\
\mathbf C_r&=\mathbf Q_s^{\mathrm T}\mathbf C\mathbf Q_s,
&\mathbf G_r&=\mathbf Q_o^{\mathrm T}\mathbf M_o\mathbf Q_o,\\
\mathbf g_r&=\mathbf Q_o^{\mathrm T}\mathbf g_o.
\end{aligned}
\tag{40}
$$

For an orbital coefficient vector $\mathbf x$, static condensation inside the
projected structure space gives

$$
\mathbf y=-\mathbf C_r^{\dagger}\mathbf B_r\mathbf x,
\qquad
\mathbf H_{\mathrm{rel},r}
=\mathbf A_r-\mathbf B_r^{\mathrm T}\mathbf C_r^{\dagger}\mathbf B_r.
\tag{41}
$$

The reduced NEO problem is then

$$
\min_{\mathbf x}
\left[
\mathbf g_r^{\mathrm T}\mathbf x
+\frac12\mathbf x^{\mathrm T}
\mathbf H_{\mathrm{rel},r}\mathbf x
\right]
\quad\text{subject to}\quad
\mathbf x^{\mathrm T}\mathbf G_r\mathbf x\leq\Delta_o^2.
\tag{42}
$$

After reconstruction,

$$
\mathbf p=\mathbf Q_o\mathbf x,
\qquad
\mathbf q=\mathbf Q_s\mathbf y,
\tag{43}
$$

the two independent full-space residuals are

$$
\mathbf r_o
=\mathbf g_o+\mathbf A\mathbf p
+\mathbf B^{\mathrm T}\mathbf q
+\lambda\mathbf M_o\mathbf p,
\qquad
\mathbf r_s=\mathbf B\mathbf p+\mathbf C\mathbf q.
\tag{44}
$$

$\mathbf r_o$ expands $\mathbf Q_o$ and $\mathbf r_s$ expands $\mathbf Q_s$.
Thus the response rank is selected by the equations themselves rather than by
a coefficient-magnitude threshold.  The projected operator is fixed during
each micro-solve, symmetric by construction, and uses only the direct block
actions $\mathbf A\mathbf p$, $\mathbf B\mathbf p$,
$\mathbf B^{\mathrm T}\mathbf q$, and $\mathbf C\mathbf q$.

Static condensation is well posed for an isolated stationary state or state
cluster when

$$
\mathbf C=\mathbf C^{\mathrm T},
\qquad
\operatorname{Null}(\mathbf C)
\subseteq\operatorname{Null}(\mathbf B^{\mathrm T}),
\tag{45}
$$

on the horizontal response space.  Equivalently,
$\operatorname{Range}(\mathbf B)\subseteq
\operatorname{Range}(\mathbf C)$.  The structure equation is a stationarity
condition obtained by differentiating the selected eigenpair, not a separate
minimization over $\mathbf q$.  Therefore $\mathbf C$ may be indefinite for a
root-followed excited state; its nonzero eigenvalues retain their signs in
$\mathbf C^{\dagger}$.  For the lowest root, or an equally weighted cluster of
the lowest consecutive roots projected outside the complete selected span,
$\mathbf C$ is positive semidefinite as a special case.  A true null direction
coupled to the orbitals makes the response and Schur complement undefined and
must not be hidden by a level shift or empirical pseudoinverse cutoff.

### 9.4 Recommended mathematical definition

In the present code, **VBSCF-NEO** denotes the relaxed-orbital trust problem in
eqs 35--37 evaluated through the residual-controlled projected response
construction in eqs 40--45.  This definition matches the actual finite trial:
only the orbital component is retracted and the structure problem is then
solved to stationarity.

The fully coupled problem in eqs 21--34 remains a distinct, mathematically
valid formulation.  It should be used only with a simultaneous finite
orbital--structure retraction and its joint trust norm.  Naming the two
formulations explicitly prevents an unshifted relaxed response from being
combined with the shift-dependent coupled equations in eqs 38--39.

## 10. Residual and model certificates

For a computed generalized eigenpair $\widehat\mu$, $\widehat\beta$,
$\widehat{\mathbf y}$, define the augmented residual

$$
\mathbf r_{\mathrm{NEO}}
=
\mathbf L_\alpha
\begin{pmatrix}
\widehat\beta\\
\widehat{\mathbf y}
\end{pmatrix}
-
\widehat\mu\mathbf G
\begin{pmatrix}
\widehat\beta\\
\widehat{\mathbf y}
\end{pmatrix}.
\tag{46}
$$

The lower block, divided by $\alpha\widehat\beta$, is the trust-region Newton
residual

$$
\mathbf r_N
=
\mathbf g
+\left(\mathbf K+\widehat\lambda\mathbf M\right)
\widehat{\mathbf z},
\qquad
\widehat\lambda=-\widehat\mu.
\tag{47}
$$

Thus the NEO microiteration should stop from a residual forcing condition such
as

$$
\|\mathbf r_N\|_{\mathbf M^{-1}}
\leq
\eta_k\|\mathbf g\|_{\mathbf M^{-1}},
\qquad
0\leq\eta_k<1,
\tag{48}
$$

together with the norm equation and a lowest-root check.  A hard cap on HVP
count is only a resource safeguard; it is not evidence that the Newton
microproblem has been solved.

The residual preconditioner must approximate the shifted KKT operator rather
than the unshifted Hessian alone.  In the present local orbital blocks, the
quotient bases are whitened in their normalized-orbital metrics, so the
shift-aware block preconditioner is

$$
\mathbf P_\lambda^{-1}
=
\operatorname{blockdiag}_i
\left(\mathbf P_i+\lambda\mathbf I_i\right)^{-1}.
\tag{48a}
$$

For an interior Newton step, $\lambda=0$ and eq 48a reduces exactly to the
existing orbital-block inverse.  For a boundary or negative-curvature step,
including the same NEO shift used in the KKT residual prevents the
preconditioner from amplifying directions that the shifted equation has
regularized.  This changes only Davidson subspace generation; the converged
step and its residual certificate remain defined by the exact matrix-free
operator.

For a trial step, the quadratic predicted reduction is

$$
\Delta m_k
=
-\mathbf g^{\mathrm T}\mathbf z
-\frac12\mathbf z^{\mathrm T}\mathbf K\mathbf z.
\tag{49}
$$

The agreement ratio is

$$
\rho_k
=
\frac{
\mathcal E_k-\mathcal E(\mathbf x_k^{\mathrm{trial}})
}{
\Delta m_k
}.
\tag{50}
$$

The accepted-point structure problem must be solved consistently before the
numerator of eq 50 is evaluated.  The trust radius is then adjusted from
$\rho_k$ and whether the boundary was active.  Molecule-specific iteration
budgets or energy heuristics are not part of the mathematical method.

## 11. Retraction and accepted-point update

The orbital component uses the existing strict-support retraction

$$
\mathbf x_{k+1}^{\mathrm{trial}}
=
\mathbf x_k+\mathbf U_k\mathbf p.
\tag{51}
$$

Orbital normalization and inactive projection occur downstream and are part
of the differentiated energy map.  No orthogonal orbital rotation is assumed.

For the current macroiteration design, the structure component $\mathbf q$
is an internal variable of the coupled NEO model.  At the trial orbital point,
eq 1 is solved again and its converged eigenvectors define the accepted
structure coefficients.  This avoids committing a finite linear update to a
nonorthogonal generalized-eigenvector manifold.  A future simultaneous
orbital--structure retraction would require explicit normalization,
state-subspace tracking, and a consistent finite structure metric; it is not
implied by the infinitesimal derivation above.

## 12. Local convergence

Assume that:

1. the selected state or equally weighted state cluster remains isolated;
2. the quotient chart has constant rank near the solution;
3. the exact relaxed Hessian is nonsingular on the physical tangent space;
4. HVP and structure-response errors satisfy a tightening inexact-Newton
   forcing condition; and
5. the trust region eventually becomes inactive.

Then $\lambda_k\to0$, the NEO step approaches the exact Newton step, and the
usual local Newton convergence result applies.  Exact inner solves give
quadratic convergence; a forcing sequence with $\eta_k\to0$ gives
superlinear convergence, with quadratic convergence recovered when
$\eta_k=O(\|\mathbf g_k\|)$.  NEO supplies robust global step restriction and
correct treatment of negative curvature, but it cannot create quadratic
convergence from an inconsistent gradient, HVP, metric, response solve, or
retraction.

## 13. Required numerical identities before implementation

The following tests are mathematical prerequisites rather than optional
performance diagnostics:

1. **Gradient test**

$$
\frac{
\mathcal E(\mathbf x+h\mathbf U\mathbf v)
-\mathcal E(\mathbf x-h\mathbf U\mathbf v)
}{2h}
=
\mathbf g_o^{\mathrm T}\mathbf v+O(h^2).
\tag{52}
$$

2. **Relaxed HVP test**

$$
\frac{
\mathbf g_o(\mathbf x+h\mathbf U\mathbf v)
-\mathbf g_o(\mathbf x-h\mathbf U\mathbf v)
}{2h}
=
\mathbf H_{\mathrm{rel}}\mathbf v+O(h^2).
\tag{53}
$$

3. **Hessian symmetry**

$$
\mathbf u^{\mathrm T}\mathbf H_{\mathrm{rel}}\mathbf v
=
\mathbf v^{\mathrm T}\mathbf H_{\mathrm{rel}}\mathbf u.
\tag{54}
$$

4. **Coupling adjointness**

$$
\mathbf q^{\mathrm T}\mathbf B\mathbf p
=
\mathbf p^{\mathrm T}\mathbf B^{\mathrm T}\mathbf q.
\tag{55}
$$

5. **Coupled/reduced Newton equivalence** at $\lambda=0$

$$
\mathbf p_{\mathrm{coupled}}
=
\mathbf p_{\mathrm{relaxed}}
\tag{56}
$$

to the requested linear-solve accuracy, with $\mathbf q$ satisfying eq 19.

6. **NEO/KKT equivalence**: the step reconstructed from eq 25 must satisfy
the appropriate form of eq 22 and the explicit residual in eq 47 using
$\lambda=-\mu$.

7. **Representation invariance**: allowed inactive-basis changes and active
orbital rescalings must not change the physical step or predicted reduction.

8. **Dense reference**: for a small problem, an explicitly assembled Hessian
must reproduce the matrix-free lowest NEO root, step, and prediction.

## 14. Minimal implementation decomposition

The eventual implementation should preserve a small, explicit dependency
graph:

- a physical metric action;
- a coupled Hessian action or a relaxed-HVP action;
- a norm-extended operator that implements only eq 33;
- a lowest generalized-eigenpair solver;
- an $\alpha$ controller for eq 32;
- a step certificate implementing eqs 46--50; and
- an outer driver that performs eq 51 and the accepted-point structure solve.

L-BFGS data, if retained, belongs only to eigensolver preconditioning and
initial-subspace construction.  The converged NEO step must be defined by the
current-point operators and residuals in this note.  No compatibility branch,
system-specific fallback, or duplicated optimizer model is required.

Within one nonlinear macroiteration, the accepted Davidson direction from one
exact-gradient keyframe may initialize the next keyframe subspace after vector
transport into the new orbital chart.  This is an initial-subspace choice only:
the new accepted-point Hessian images and residual certificate are recomputed.
The direction is discarded at the next macroiteration, so no secant curvature
or stale Hessian action enters the NEO model.

## 15. Unified VBSCF Hessian-diagonal preconditioner

Both HAO and OEO orbitals are nonorthogonal.  They differ in their AO support:
HAO orbitals are strictly sparse, whereas OEO orbitals have full AO support.
Consequently, an orthogonal-CASSCF rotation diagonal is not a general VBSCF
preconditioner and must not be substituted for the curvature of either raw
coefficient representation.

Let $U$ be the accepted-point nonredundant horizontal lift defined in section
7.  The coupled NEO equation contains the orbital block

$$
A
=
H_{\mathrm{direct}}
+H_{\mathrm{fixed\ pullback}}
+H_{\mathrm{local\ active}},
\tag{57}
$$

together with the explicit orbital--structure coupling $B$ and structure
block $C$.  The unified Hessian diagonal is defined in the same reduced
coordinates as the NEO step:

$$
d_i
=
e_i^{\mathrm T}Ae_i
=
\left[U^{\mathrm T}H_{CC}U\right]_{ii}.
\tag{58}
$$

The relaxed Schur term $-BC^{\dagger}B^{\mathrm T}$ is deliberately absent
from eq 58.  Structure amplitudes are independent variables in the coupled
NEO equation; including their eliminated response in $d_i$ would count the
same coupling twice.

Equation 58 is evaluated from the production analytic block-HVP rather than
from a separately derived CASSCF expression.  For the natural reduced block
$I_p$ belonging to orbital $p$, define the coordinate selector

$$
E_p
=
\begin{bmatrix}
e_{i_1}&e_{i_2}&\cdots&e_{i_{m_p}}
\end{bmatrix},
\qquad i_k\in I_p.
\tag{59}
$$

One fused matrix-free action produces

$$
Y_p=AE_p,
\qquad
d_{i_k}=(Y_p)_{i_k k}.
\tag{60}
$$

Only the entries in eq 60 are retained.  The full reduced Hessian is never
assembled or stored.  Grouping directions by their physical orbital block
removes any empirical batch-width parameter and bounds the working memory by
the largest local HAO/OEO quotient block rather than by the full reduced
dimension.

The exact diagonal can be indefinite.  A positive absolute-curvature model is
used only for preconditioning:

$$
\widetilde d_i
=
\max\!\left(
|d_i|,
\sqrt{\epsilon_{\mathrm{mach}}}\max_j|d_j|
\right),
\qquad
z_i=\frac{r_i}{\widetilde d_i+\lambda}.
\tag{61}
$$

This replacement does not alter $A$, its negative curvature, the NEO Ritz
values, or the KKT certificate.  It changes only the metric used to solve the
shifted iterative subproblem.  Because eqs 58--60 use the actual VBSCF HVP and
the actual accepted-point chart, the same implementation applies without a
coordinate reinterpretation to strict-sparse HAO, full-AO OEO, complete
structure spaces, and selected structure subspaces.

## 16. References

1. H. J. Aa. Jensen and P. Jørgensen, “A direct approach to second-order
   MCSCF calculations using a norm extended optimization scheme,” *J. Chem.
   Phys.* **80**, 1204--1214 (1984),
   [doi:10.1063/1.446797](https://doi.org/10.1063/1.446797).
2. H. J. Aa. Jensen and H. Ågren, “A direct, restricted-step, second-order MC
   SCF program for large scale ab initio calculations,” *Chem. Phys.* **104**,
   229--250 (1986),
   [doi:10.1016/0301-0104(86)80169-0](https://doi.org/10.1016/0301-0104(86)80169-0).
3. T. Nottoli, J. Gauss, and F. Lipparini, “Second-Order CASSCF Algorithm with
   the Cholesky Decomposition of the Two-Electron Integrals,” *J. Chem. Theory
   Comput.* **17**, 6819--6831 (2021),
   [doi:10.1021/acs.jctc.1c00327](https://doi.org/10.1021/acs.jctc.1c00327).
4. B. Helmich-Paris, “A trust-region augmented Hessian implementation for
   state-specific and state-averaged CASSCF wave functions,” *J. Chem. Phys.*
   **156**, 204104 (2022),
   [doi:10.1063/5.0090447](https://doi.org/10.1063/5.0090447).
