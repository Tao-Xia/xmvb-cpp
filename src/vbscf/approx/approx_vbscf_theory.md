# VB Functional Theory And Approximate VBSCF

## 0. Thesis

The central idea is stronger than a neural surrogate for VBSCF.

The target is a density-functional-theory-like formulation of valence-bond
electronic structure: replace the exponentially large nonorthogonal VB
structure-coefficient problem by a functional of compact, chemically meaningful
VB observables.

The working name in this note is VB Functional Theory, abbreviated as VBFT.  An
approximate self-consistent implementation of this idea is denoted aVBSCF.

The intended exact-to-approximate hierarchy is:

$$
\text{exact VBSCF}
\quad \Longleftrightarrow \quad
\text{exact VB constrained-search functional}
\quad \longrightarrow \quad
\text{practical approximate VB functional}.
$$

Neural-network correction is optional.  It can be useful later as a residual
model, but it is not the foundation of the theory.

The key claim should be:

> VBFT seeks an exact functional formulation of VBSCF in compact VB
> observables, and aVBSCF is a controlled approximation to that functional.

This is analogous in spirit to DFT, but it should not be presented as a direct
copy of the Hohenberg-Kohn theorem.  The rigorous route available immediately
is closer to a Levy-Lieb constrained-search construction.

## 1. Exact VBSCF Reference

At fixed localized nonorthogonal VB orbitals $R$, exact VBSCF writes the active
wavefunction as

$$
|\Psi(R,c)\rangle
=
\sum_{I=1}^{N_s}
c_I |\Phi_I(R)\rangle,
$$

where $|\Phi_I(R)\rangle$ are nonorthogonal VB structures and $N_s$ is the
number of structures.

The coefficients solve

$$
H(R)c
=
E_{\mathrm{VB}}(R) S(R)c,
\qquad
c^\mathrm T S(R)c = 1.
$$

The total VBSCF energy is

$$
E_{\mathrm{VBSCF}}^{\mathrm{tot}}(R)
=
E_{\mathrm{ref}}(R)
+
E_{\mathrm{VB}}(R)
+
E_{\mathrm{nuc}}.
$$

Orbital optimization has the nonorthogonal response structure

$$
\frac{\partial E_{\mathrm{VBSCF}}^{\mathrm{tot}}}{\partial R_k}
=
\frac{\partial E_{\mathrm{ref}}}{\partial R_k}
+
c^\mathrm T
\left(
\frac{\partial H}{\partial R_k}
-
E_{\mathrm{VB}}
\frac{\partial S}{\partial R_k}
\right)c.
$$

The computational obstacles are therefore:

1. $N_s$ grows combinatorially with the active space.
2. Nonorthogonal orbitals require both $H$ and $S$.
3. Orbital optimization requires the full $dH-E\,dS$ response.

Any scalable VB theory must avoid production construction of the full
$N_s\times N_s$ Hamiltonian and overlap matrices.

## 2. DFT Analogy And The Rigorous Boundary

DFT uses the electron density $\rho(\mathbf r)$ as a basic variable.  The
Hohenberg-Kohn theorem states, under its assumptions, that the ground-state
density determines the external potential up to a constant.  The Levy-Lieb
formulation then defines a universal constrained-search functional.

For VBFT, the analogous basic variables are not the real-space density alone.
They are compact VB observables such as pair populations, pair-pair
correlations, resonance coherences, and nonorthogonal metric descriptors.

Let $\mathcal V_R$ be the exact VB wavefunction manifold spanned by all allowed
VB structures at fixed localized orbitals $R$.  Let

$$
\mathcal M_R:
|\Psi\rangle
\mapsto
\xi
$$

be a descriptor map from a normalized VB wavefunction to compact VB observables.
Typical components of $\xi$ are

$$
\xi
=
\{n_p,\Gamma_{pq},\rho_\alpha,\mu_\beta,\ldots\}.
$$

The representable domain is

$$
\mathcal D_R
=
\left\{
\xi
\mid
\exists |\Psi\rangle\in\mathcal V_R,
\quad
\langle\Psi|\Psi\rangle=1,
\quad
\mathcal M_R[\Psi]=\xi
\right\}.
$$

Define the exact VB constrained-search functional

$$
F_R^{\mathrm{VB}}[\xi]
=
\min_{
\substack{
|\Psi\rangle\in\mathcal V_R \\
\langle\Psi|\Psi\rangle=1 \\
\mathcal M_R[\Psi]=\xi
}
}
\langle\Psi|\hat H_{\mathrm{act}}(R)|\Psi\rangle.
$$

Then the exact VB energy at fixed $R$ is

$$
E_{\mathrm{VB}}(R)
=
\min_{\xi\in\mathcal D_R}
F_R^{\mathrm{VB}}[\xi].
$$

The full orbital-optimized VBSCF energy is therefore

$$
E_{\mathrm{VBSCF}}^{\mathrm{tot}}
=
\min_R
\left[
E_{\mathrm{ref}}(R)
+
E_{\mathrm{nuc}}
+
\min_{\xi\in\mathcal D_R}
F_R^{\mathrm{VB}}[\xi]
\right].
$$

This statement is exact by construction.  The proof is simply that the
constrained-search functional partitions the full VB wavefunction manifold by
the descriptor value $\xi$:

$$
\min_{|\Psi\rangle\in\mathcal V_R}
\langle\Psi|\hat H_{\mathrm{act}}(R)|\Psi\rangle
=
\min_{\xi\in\mathcal D_R}
\min_{\mathcal M_R[\Psi]=\xi}
\langle\Psi|\hat H_{\mathrm{act}}(R)|\Psi\rangle.
$$

This is the rigorous VBFT foundation.

What is not yet proven is an HK-like uniqueness theorem saying that a small
compact $\xi$ uniquely determines the VB Hamiltonian or the VB ground state.
For practical compact descriptors, such uniqueness is generally too strong and
should not be claimed without a separate theorem.

The correct rigor statement is therefore:

> An exact VB functional exists by constrained search.  Practical aVBSCF
> approximates that exact functional in a compact representable variable space.

## 3. Basic VB Variables

The exact descriptor map $\mathcal M_R$ can be chosen at different levels of
completeness.

The complete but non-scalable choice is effectively the full VB density matrix
in structure space.  For a real single-state expansion,

$$
D_{IJ}
=
c_I c_J.
$$

Together with the nonorthogonal metric and Hamiltonian matrices, this contains
the exact VB state.  It is not useful for large active spaces because it has
$O(N_s^2)$ size.

The scalable VBFT choice is a hierarchy of chemically meaningful reduced
observables.

### 3.1 Pair Populations

Let $p=(i,j)$ denote a local VB electron-pair pattern over active localized
orbitals.  A compact pair population is denoted

$$
n_p.
$$

In small exact calculations, a population-style diagnostic can be obtained from
nonorthogonal structure weights

$$
W_I
=
c_I(Sc)_I,
\qquad
\sum_I W_I = 1,
$$

and an incidence value $b_I(p)$:

$$
n_p^{\mathrm{exact}}
=
\sum_I W_I b_I(p).
$$

For a rigorous functional theory, it is better to view $n_p$ as the expectation
value of a chosen VB pair observable:

$$
n_p
=
\langle\Psi|\hat B_p(R)|\Psi\rangle.
$$

This avoids over-interpreting nonorthogonal Mulliken-like populations as
positive probabilities.  The population formula remains useful for validation
and chemical diagnostics.

### 3.2 Pair-Pair Correlations

Pair populations alone do not determine resonance and cooperative bonding.
The next descriptor layer is

$$
\Gamma_{pq}
=
\langle\Psi|\hat B_{pq}(R)|\Psi\rangle.
$$

In the structure-population diagnostic form,

$$
\Gamma_{pq}^{\mathrm{exact}}
=
\sum_I W_I b_I(p)b_I(q).
$$

The connected pair correlation is

$$
C_{pq}
=
\Gamma_{pq}
-
n_p n_q.
$$

The current first aVB-2 code step implements the ability to evaluate the energy
with explicit $\Gamma_{pq}$ labels, while the production pair-SCF path still
uses the mean-field factorization

$$
\Gamma_{pq}
\approx
n_p n_q.
$$

### 3.3 Resonance Coherences

VB resonance is not only population transfer.  It involves coherent mixing
between chemically related structures.  Introduce resonance observables

$$
\rho_\alpha
=
\langle\Psi|\hat R_\alpha(R)|\Psi\rangle,
$$

where $\alpha$ labels local resonance motifs:

- covalent-ionic mixing on a bond;
- adjacent pair exchange;
- Kekule exchange in conjugated rings;
- donor-acceptor rearrangement;
- spin recoupling motif.

Without $\rho_\alpha$, a compact VB model can describe which pairs are
occupied, but it cannot fully describe resonance stabilization.

### 3.4 Metric Descriptors

Nonorthogonal VB theory is not a theory of $H$ alone.  It is a theory of
$H$ and $S$ together.  Therefore the compact variable set should include metric
descriptors

$$
\mu_\beta
=
\langle\Psi|\hat M_\beta(R)|\Psi\rangle,
$$

or an equivalent cumulant representation of the overlap denominator.

The practical state variable is therefore

$$
\xi
=
\{n_p,\Gamma_{pq},\rho_\alpha,\mu_\beta\}.
$$

## 4. Representability

The exact functional is only defined on representable descriptors:

$$
\xi\in\mathcal D_R.
$$

For closed-shell pair variables, basic constraints include

$$
\sum_{p\in\mathcal P} n_p
=
N_{\mathrm{pair}},
\qquad
0\le n_p\le 1
$$

when a positive pair-population representation is used.

The active orbital population derived from pair populations is

$$
q_i
=
2n_{ii}
+
\sum_{j<i} n_{ji}
+
\sum_{j>i} n_{ij},
$$

with

$$
0\le q_i\le 2.
$$

For pair-pair descriptors,

$$
\Gamma_{pp}=n_p,
\qquad
0\le \Gamma_{pq}\le \min(n_p,n_q)
$$

are natural constraints in a positive ensemble representation.

However, exact nonorthogonal VB populations can be signed or quasi-probabilistic
depending on the chosen population analysis.  Therefore there are two
representability levels:

1. Exact operator representability:

$$
\xi=\mathcal M_R[\Psi]
\quad
\text{for some normalized }
|\Psi\rangle\in\mathcal V_R.
$$

2. Positive diagnostic representability:

$$
P(I)\ge 0,
\qquad
\sum_I P(I)=1,
$$

with

$$
n_p=\sum_I P(I)b_I(p),
\qquad
\Gamma_{pq}=\sum_I P(I)b_I(p)b_I(q).
$$

The exact theory should be based on operator representability.  The positive
diagnostic representation is useful for stable approximate optimization and
chemical interpretation.

## 5. Exact Functional Versus Practical Approximation

The exact constrained-search functional

$$
F_R^{\mathrm{VB}}[\xi]
$$

contains all missing structure-space information implicitly.  If $\xi$ is
small, the functional is complicated because it must encode everything not
explicitly represented by $\xi$.

The practical aVBSCF approximation replaces this exact functional by a
tractable ansatz:

$$
F_R^{\mathrm{VB}}[\xi]
\approx
F_R^{\mathrm{aVB}}[\xi].
$$

A useful nonorthogonal form is

$$
F_R^{\mathrm{aVB}}[\xi]
=
\frac{
\mathcal H_{\mathrm{aVB}}(R,\xi)
}{
\mathcal S_{\mathrm{aVB}}(R,\xi)
}.
$$

The total approximate VBSCF energy is

$$
E_{\mathrm{aVBSCF}}(R,\xi)
=
E_{\mathrm{ref}}(R)
+
E_{\mathrm{nuc}}
+
F_R^{\mathrm{aVB}}[\xi].
$$

This gives a DFT-like division:

$$
\text{exact functional}
\quad
F_R^{\mathrm{VB}}[\xi]
\quad
\text{exists formally},
$$

but practical calculation uses

$$
\text{approximate functional}
\quad
F_R^{\mathrm{aVB}}[\xi].
$$

This is directly analogous to how practical DFT uses LDA, GGA, hybrid, or other
approximations to the exact exchange-correlation functional.

## 6. Approximate Functional Form

The approximate numerator can be decomposed as

$$
\mathcal H_{\mathrm{aVB}}
=
E_{\mathrm{loc}}
+
E_{\mathrm{corr}}
+
E_{\mathrm{res}}
+
E_{\mathrm{lr}}
+
E_{\mathrm{rem}}.
$$

### 6.1 Local Pair Energy

The local term is

$$
E_{\mathrm{loc}}(R,n)
=
\sum_{p\in\mathcal P}
n_p \epsilon_p(R).
$$

For a doubly occupied ionic pair $p=(i,i)$,

$$
\epsilon_{ii}(R)
=
2h_{ii}^{\mathrm{act}}(R)
+
(ii|ii)_R
+
\epsilon_{ii}^{\mathrm{ion}}.
$$

For a covalent pair $p=(i,j)$ with $i\ne j$,

$$
\epsilon_{ij}(R)
=
h_{ii}^{\mathrm{act}}(R)
+
h_{jj}^{\mathrm{act}}(R)
+
(ii|jj)_R
-
(ij|ij)_R
+
\epsilon_{ij}^{\mathrm{spin}}.
$$

### 6.2 Pair-Correlation Energy

The correlation term is

$$
E_{\mathrm{corr}}(R,\Gamma)
=
\frac{1}{2}
\sum_{p,q}
\Gamma_{pq}V_{pq}(R).
$$

For scalable calculation, $V_{pq}$ should be local, screened, or low rank:

$$
V_{pq}(R)
\approx
\sum_{\ell=1}^{r}
L_{p\ell}(R)L_{q\ell}(R),
\qquad
r\ll P.
$$

The first code step toward this level is explicit evaluation with
$\Gamma_{pq}$, but the physical form of $V_{pq}$ still needs improvement.

### 6.3 Resonance Energy

The resonance term is

$$
E_{\mathrm{res}}(R,\rho)
=
\sum_\alpha
\rho_\alpha T_\alpha(R).
$$

This term is essential for systems such as benzene, where the dominant VB
physics is not captured by pair populations alone.

### 6.4 Nonorthogonal Metric Functional

The denominator should approximate the nonorthogonal structure metric:

$$
\log \mathcal S_{\mathrm{aVB}}(R,\xi)
=
\sum_p n_p a_p(R)
+
\frac{1}{2}\sum_{p,q}\Gamma_{pq}b_{pq}(R)
+
\sum_\alpha \rho_\alpha d_\alpha(R)
+
\sum_\beta \mu_\beta m_\beta(R).
$$

The exact VB orbital gradient contains

$$
dH-E\,dS.
$$

Therefore aVBSCF must preserve the approximate counterpart

$$
\frac{\partial E_{\mathrm{aVBSCF}}}{\partial R_k}
=
\frac{\partial E_{\mathrm{ref}}}{\partial R_k}
+
\frac{1}{\mathcal S_{\mathrm{aVB}}}
\left[
\frac{\partial \mathcal H_{\mathrm{aVB}}}{\partial R_k}
-
\varepsilon_{\mathrm{aVB}}
\frac{\partial \mathcal S_{\mathrm{aVB}}}{\partial R_k}
\right],
$$

where

$$
\varepsilon_{\mathrm{aVB}}
=
\frac{\mathcal H_{\mathrm{aVB}}}{\mathcal S_{\mathrm{aVB}}}.
$$

Without this metric response, the method is only a pair-energy model, not an
approximate VBSCF.

## 7. Self-Consistent Principle

The practical aVBSCF variational problem is

$$
\min_{R,\xi\in\mathcal C_R}
E_{\mathrm{aVBSCF}}(R,\xi).
$$

With constraints $C_a(R,\xi)=0$, define

$$
\mathcal L(R,\xi,\lambda)
=
E_{\mathrm{aVBSCF}}(R,\xi)
+
\sum_a \lambda_a C_a(R,\xi).
$$

The stationarity equations are

$$
\nabla_R \mathcal L = 0,
\qquad
\nabla_\xi \mathcal L = 0,
\qquad
C_a(R,\xi)=0.
$$

This is the VBFT analogue of Kohn-Sham self-consistency:

1. The compact state variables $\xi$ are optimized.
2. The localized VB orbitals $R$ are optimized.
3. The functional derivative defines the effective VB equations.

The current code only performs the first part in a simplified mean-field
pair-space form.  It is therefore an aVB prototype, not yet a full aVBSCF.

## 8. Physical Observables

The theory should preserve the physical outputs that make VBSCF valuable.

### 8.1 Structure Importance

Exact nonorthogonal VB structure weights are often reported as

$$
W_I
=
c_I(Sc)_I.
$$

In a scalable theory, $W_I$ cannot be a fundamental variable because $I$ ranges
over an exponentially large structure set.

Structure importance should be treated as a projection diagnostic.  Given a
candidate structure pool $\mathcal A$, define

$$
A_{pI}
=
b_I(p).
$$

The projected structure weights solve

$$
\widetilde W
=
\arg\min_W
\left\|
AW-n
\right\|_2^2,
$$

subject to

$$
W_I\ge 0,
\qquad
\sum_I W_I=1.
$$

With pair-pair and resonance descriptors, the projection becomes

$$
\widetilde W
=
\arg\min_W
\left[
\left\|AW-n\right\|_{M_n}^2
+
\left\|BW-\Gamma\right\|_{M_\Gamma}^2
+
\left\|CW-\rho\right\|_{M_\rho}^2
\right].
$$

This gives approximate structure ranking without making full structure
enumeration part of the production energy calculation.

### 8.2 Charge, Spin, Bond Order, And Resonance

Pair variables give active-orbital populations such as

$$
q_i
=
2n_{ii}
+
\sum_{j<i}n_{ji}
+
\sum_{j>i}n_{ij}.
$$

Pair correlations $\Gamma_{pq}$ add cooperative bonding and ionic-covalent
correlation.  Resonance variables $\rho_\alpha$ add coherent VB mixing.

The target correspondence is

$$
\mathcal O_{\mathrm{aVBSCF}}(R,\xi)
\approx
\mathcal O_{\mathrm{VBSCF}}(R,c)
$$

for the VB observables used in chemical analysis.

## 9. Computational Scaling

Exact VBSCF scales with the number of structures:

$$
\mathrm{cost}_{\mathrm{exact}}
\sim
O(N_s^2 C_{\mathrm{mat}})
+
O(N_s^3).
$$

The compact pair space has size

$$
P
=
\frac{n_a(n_a+1)}{2}.
$$

For a $30$-$30$ active space,

$$
n_a=30,
\qquad
P=465,
\qquad
P^2=216225.
$$

Therefore a practical aVB functional should scale as

$$
\mathrm{cost}_{\mathrm{aVB}}
\sim
O(P^2)
$$

or, with locality or low rank,

$$
\mathrm{cost}_{\mathrm{aVB}}
\sim
O(Pr)
\quad
\text{or}
\quad
O(Pz),
\qquad
r,z\ll P.
$$

The production method must satisfy

$$
\text{no production construction of }
N_s\times N_s
\text{ matrices}.
$$

## 10. Open Theoretical Gaps

To make VBFT both computable and physically rigorous, the theory still needs
several missing pieces.  These are more important than neural correction.

### 10.1 Choice Of Basic Variables

The first unresolved question is the minimal descriptor set:

$$
\xi
=
\{n_p,\Gamma_{pq},\rho_\alpha,\mu_\beta,\ldots\}.
$$

If $\xi$ is too small, the exact constrained-search functional still exists,
but it becomes highly nonlocal and hard to approximate.  If $\xi$ is too large,
the method drifts back toward full VB structure space.

The theory needs a clear hierarchy:

- Level 1: $n_p$ only;
- Level 2: $n_p$ and $\Gamma_{pq}$;
- Level 3: $n_p$, $\Gamma_{pq}$, and resonance variables $\rho_\alpha$;
- Level 4: add metric cumulants $\mu_\beta$;
- exact limit: full VB reduced density matrix or equivalent structure-space
  information.

The key theorem or practical criterion should identify which level is sufficient
for each class of chemistry.  For example, benzene cannot be expected to work
with $n_p$ alone because aromatic stabilization is a coherence effect.

### 10.2 Representability Conditions

The constrained-search definition requires

$$
\xi\in\mathcal D_R.
$$

For a practical optimizer, this is not enough.  We need computable constraints
or parameterizations that keep $\xi$ inside a physically meaningful domain.

The missing theory is:

1. exact operator representability for nonorthogonal VB observables;
2. positive ensemble representability for stable approximate populations;
3. constraints linking $n_p$, $\Gamma_{pq}$, $\rho_\alpha$, and $\mu_\beta$;
4. a projection or parameterization that is differentiable and scalable.

Without this, minimization over $\xi$ may find unphysical pair populations or
correlations that no VB wavefunction can produce.

### 10.3 Universal Versus Orbital-Dependent Functional

DFT separates the universal functional from the external potential.  VBFT has a
subtle difference: the localized nonorthogonal orbitals $R$ are part of the
state representation.

The current exact functional is written as

$$
F_R^{\mathrm{VB}}[\xi].
$$

This is rigorous, but it is orbital-dependent.  The theory still needs to
clarify whether there is a more universal object such as

$$
F^{\mathrm{VB}}[\xi,R]
$$

or whether orbital dependence should be accepted as part of the VB functional,
similar in spirit to orbital-dependent functionals in generalized Kohn-Sham
theory.

This matters because a self-consistent method must define both

$$
\frac{\delta F_R^{\mathrm{VB}}}{\delta \xi}
$$

and

$$
\frac{\partial F_R^{\mathrm{VB}}}{\partial R}.
$$

### 10.4 Nonorthogonal Metric Functional

This is probably the largest missing theoretical component.

Exact VBSCF is governed by

$$
\frac{c^\mathrm T H c}{c^\mathrm T S c},
$$

not by $c^\mathrm T Hc$ alone.  Therefore the approximate functional must define

$$
\mathcal S_{\mathrm{aVB}}(R,\xi)
$$

with enough accuracy to reproduce the metric response

$$
d\mathcal H_{\mathrm{aVB}}
-
\varepsilon_{\mathrm{aVB}}d\mathcal S_{\mathrm{aVB}}.
$$

The missing work is to derive a controlled expansion of
$\log\mathcal S_{\mathrm{aVB}}$ in terms of $n_p$, $\Gamma_{pq}$,
$\rho_\alpha$, and $\mu_\beta$, and to decide which terms are required for
chemical accuracy.

### 10.5 Resonance Functional

Pair populations and pair correlations are not enough for VB chemistry.  The
functional needs explicit resonance variables:

$$
E_{\mathrm{res}}
=
\sum_\alpha \rho_\alpha T_\alpha(R).
$$

The missing theory is:

1. how to define a complete but local motif set $\alpha$;
2. how to compute $T_\alpha(R)$ without enumerating all structures;
3. how to constrain $\rho_\alpha$ together with $n_p$ and $\Gamma_{pq}$;
4. how to report resonance energies in a way that corresponds to VBSCF
   interpretation.

This is essential if the theory is meant to preserve the physical insight of
VBSCF, not only approximate the energy.

### 10.6 Euler Equations And Effective VB Potentials

A self-consistent theory requires explicit stationarity equations.  For

$$
E_{\mathrm{aVBSCF}}(R,\xi)
=
E_{\mathrm{ref}}(R)
+
E_{\mathrm{nuc}}
+
F_R^{\mathrm{aVB}}[\xi],
$$

the Euler equations are

$$
\frac{\delta F_R^{\mathrm{aVB}}}{\delta \xi_a}
+
\sum_b \lambda_b
\frac{\partial C_b}{\partial \xi_a}
=
0,
$$

and

$$
\frac{\partial E_{\mathrm{ref}}}{\partial R_k}
+
\frac{\partial F_R^{\mathrm{aVB}}[\xi]}{\partial R_k}
+
\sum_b \lambda_b
\frac{\partial C_b}{\partial R_k}
=
0.
$$

The theory still needs the explicit form of these functional derivatives and
their physical interpretation as effective VB potentials or fields conjugate to
$n_p$, $\Gamma_{pq}$, $\rho_\alpha$, and metric descriptors.

### 10.7 Physical Observable Reconstruction

The theory must define how every important VBSCF observable is obtained from
$\xi$:

- energy;
- localized orbitals;
- structure importance;
- ionic and covalent weights;
- bond order;
- charge;
- spin coupling;
- resonance energy;
- pair and pair-pair correlations.

At present, structure importance can be defined as a projection diagnostic, but
charge, spin, bond order, and resonance still need exact or approximate
reconstruction formulas.

### 10.8 Size Consistency And Locality

A practical functional must satisfy a separability condition for noninteracting
fragments $A$ and $B$:

$$
E_{\mathrm{aVB}}(A+B)
=
E_{\mathrm{aVB}}(A)
+
E_{\mathrm{aVB}}(B)
$$

when the localized orbitals and descriptors factorize.  This requirement will
strongly constrain the allowed forms of $E_{\mathrm{corr}}$,
$E_{\mathrm{res}}$, and $\mathcal S_{\mathrm{aVB}}$.

Without size consistency, the theory may look good on small molecules but fail
systematically on large active spaces.

### 10.9 Exact Limit And Error Decomposition

The approximate hierarchy should have a clear exact limit.  There should be a
sequence of descriptor spaces

$$
\xi^{(1)}
\subset
\xi^{(2)}
\subset
\cdots
\subset
\xi^{(\infty)}
$$

such that

$$
F_R^{\mathrm{aVB},k}[\xi^{(k)}]
\rightarrow
F_R^{\mathrm{VB}}[\xi^{(\infty)}].
$$

The error should be decomposed into:

- descriptor incompleteness error;
- approximate functional error;
- representability/projection error;
- orbital self-consistency error;
- numerical optimization error.

This is the analogue of separating basis-set error, functional error, and SCF
error in DFT.

### 10.10 Calibration Without Losing Theory

Even without neural networks, the practical functional may contain parameters
or fitted local terms.  The theory needs a calibration principle:

$$
F_R^{\mathrm{aVB}}[\xi;\omega]
$$

should be fitted or constrained so that it respects:

- exact one-pair and two-pair limits;
- dissociation limits;
- spin and particle-number constraints;
- nonorthogonal metric response;
- size consistency;
- small-system exact VBSCF benchmarks.

This keeps the method as a physical VB functional rather than an empirical
energy model.

## 11. Non-Neural Development Plan

The non-neural path to a complete aVBSCF implementation has five stages.

### 11.1 Stage 1: Explicit Pair-Pair Variables

Implemented first step:

$$
E(n,\Gamma)
=
\sum_p n_p\epsilon_p
+
\frac{1}{2}
\sum_{p,q}
\Gamma_{pq}V_{pq}.
$$

Small exact VBSCF calculations can now map one exact state to both $n_p$ and
$\Gamma_{pq}$ for validation.

### 11.2 Stage 2: Nonorthogonal Metric Functional

Implement

$$
\mathcal S_{\mathrm{aVB}}(R,\xi)
$$

and the corresponding metric response

$$
d\mathcal H_{\mathrm{aVB}}
-
\varepsilon_{\mathrm{aVB}}d\mathcal S_{\mathrm{aVB}}.
$$

This is the highest-priority theoretical step because nonorthogonality is one
of the two original VBSCF bottlenecks.

The detailed metric derivation is in
[vbft_metric_functional_derivation.md](vbft_metric_functional_derivation.md).

### 11.3 Stage 3: Resonance Motif Variables

Implement selected $\rho_\alpha$ variables and motif amplitudes $T_\alpha(R)$.
This is required for aromatic, conjugated, and strongly resonance-stabilized
systems.

### 11.4 Stage 4: Joint $R,\xi$ Self-Consistency

Move from $\xi$-only optimization to

$$
\min_{R,\xi}
E_{\mathrm{aVBSCF}}(R,\xi).
$$

This requires analytic orbital gradients of the approximate functional.

### 11.5 Stage 5: Physical Output And Benchmarks

The method must report:

- total energy;
- self-consistent localized orbitals;
- $n_p$ pair weights;
- $\Gamma_{pq}$ pair correlations;
- $\rho_\alpha$ resonance contributions;
- charge, spin, and bond-order diagnostics;
- projected structure importance.

Validation should include:

- F2;
- C6H6 $6$-$6$ full space;
- active spaces where exact VBSCF is still possible;
- large active spaces where exact VBSCF is impossible and only scaling can be
  tested.

## 12. Optional Machine Learning Residual

Machine learning is not required for the formal VBFT idea.  If used, it should
correct the residual of a physically constrained functional:

$$
E_{\mathrm{ML\text{-}aVBSCF}}
=
E_{\mathrm{aVBSCF}}
+
\Delta E_\theta(R,\xi).
$$

It should not replace the exact constrained-search logic or the nonorthogonal
VB metric response.  A non-neural analytic aVB functional would be more
fundamental and more valuable if it reaches useful accuracy.

## 13. Acceptance Criteria

The theory should be considered well defined only if the following statements
are true.

1. The exact constrained-search functional $F_R^{\mathrm{VB}}[\xi]$ is clearly
   defined.
2. The representable domain $\mathcal D_R$ is defined.
3. The approximate functional $F_R^{\mathrm{aVB}}[\xi]$ is explicitly separated
   from the exact functional.
4. The method preserves nonorthogonal metric response.
5. The method optimizes both $R$ and $\xi$.
6. The method avoids production construction of full structure-space $H$ and
   $S$ matrices.
7. The method reports the same kinds of physical observables that make VBSCF
   useful.
8. The method has small-system exact VBSCF validation and large-system scaling
   validation.

The final scientific message should be:

> VBFT is a DFT-like constrained-search theory for valence-bond observables.
> aVBSCF is its practical approximate self-consistent realization.
