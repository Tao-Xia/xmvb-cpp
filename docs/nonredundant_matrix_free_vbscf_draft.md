# Nonredundant Matrix-Free Second-Order Orbital Optimization for VBSCF

## Title Options

### Option A
Nonredundant Matrix-Free Second-Order Orbital Optimization for Valence-Bond Self-Consistent Field Theory

### Option B
A Matrix-Free Nonredundant Orbital Optimizer for General VBSCF Wave Functions

### Option C
Block-Structured Nonredundant Truncated-Newton Orbital Optimization in VBSCF

---

## Abstract

Valence-bond self-consistent field (VBSCF) theory remains one of the most
physically transparent multiconfigurational frameworks for describing chemical
bonding, resonance, and strong static correlation. Its practical use,
however, is often limited by the difficulty of orbital optimization in a
nonorthogonal and highly redundant parameter space. Conventional first-order or
quasi-Newton updates in raw sparse orbital coefficients can require many outer
iterations, while full-Hessian Newton methods become prohibitively expensive as
the orbital space grows. In this work, we present a nonredundant matrix-free
second-order orbital optimizer for VBSCF implemented in `xmvb-cpp`. The method
combines three ingredients: block-local nonredundant orbital coordinates,
matrix-free reduced Hessian-vector products, and a trust-region truncated-Newton
solver with transported secant preconditioning. In contrast to explicit reduced
Hessian approaches, the present method never forms the full orbital Hessian.
Instead, it builds a compact reduced orbital space from occupied orbitals and
block-local virtual complements, evaluates curvature information only through
matrix-free probes, and solves the reduced trust-region subproblem iteratively.
To broaden applicability beyond exact-support block partitions, we further
introduce a union-support construction for partially overlapping orbital blocks,
allowing the nonredundant optimizer to operate on cases that previously failed
at the reduced-space setup stage. Preliminary calculations show that the method
retains the fast convergence of second-order orbital optimization while avoiding
the dominant cost of explicit Hessian assembly. A systematic benchmark over
diverse VBSCF wave functions will be reported separately in the final version.

---

## 1. Introduction

Valence-bond methods provide a chemically intuitive representation of
electronic structure in terms of localized orbitals and resonance structures.
Because of this direct connection to bonding pictures, VBSCF remains attractive
for mechanistic analysis, spin coupling problems, and systems with strong
near-degeneracy. At the same time, the orbital optimization problem in VBSCF is
numerically difficult. Unlike orthogonal molecular orbital methods, the VBSCF
energy depends on nonorthogonal orbitals, and the resulting orbital parameter
space contains substantial redundancy. As a consequence, naive optimization in
raw orbital coefficients is often poorly conditioned.

Historically, two broad directions have been used to improve VBSCF orbital
optimization. The first is to retain raw orbital parameters and rely on
first-order or quasi-Newton updates. This route is easy to implement and is
often robust, but it spends optimization effort in redundant coordinates. The
second is to identify nonredundant orbital variables and solve a reduced Newton
problem. This approach is geometrically better motivated and can converge in
fewer iterations, but explicit reduced Hessian construction and dense linear
algebra rapidly become expensive for larger orbital spaces.

For a modern C++ VBSCF implementation, the natural goal is therefore not to
reproduce a legacy full-Hessian Newton optimizer exactly, but to preserve the
correct variable geometry while removing the dominant explicit-curvature cost.
This suggests the following design principle:

$$
\text{nonredundant orbital coordinates}
\;+\;
\text{matrix-free inexact second-order optimization}.
$$

The present work follows this principle. We construct a block-local
nonredundant orbital space directly from the sparse AO support structure of the
current occupied orbitals. Within each block, we generate local virtual
directions from the AO-metric orthogonal complement of the occupied span, then
retain only the physically relevant inactive-active and occupied-virtual
rotation classes. The reduced-space optimization problem is solved with a
matrix-free truncated-Newton (TN) method, where the reduced Hessian is queried
only through Hessian-vector products generated on demand. A trust-region model
controls step acceptance, and transported secant information is used to enrich
the reduced-space preconditioner without leaving the matrix-free setting.

An additional issue arises in realistic VBSCF inputs: orbital blocks are not
always exact-support partitions. In many systems, different orbitals within the
same overlap-connected region share AO support only partially. A reduced-space
optimizer that assumes one representative support per block then fails before
any optimization step is taken. To address this, we generalize the block-local
nonredundant space to partially overlapping sparse orbital blocks by defining
the block AO domain as the union of all member supports. This removes a major
generality restriction and allows the nonredundant optimizer to operate on a
broader class of VBSCF wave functions.

The intended contribution of this work is therefore methodological rather than
purely implementation-oriented. We do not claim that Newton-like methods are
universally new in VB optimization. Rather, the contribution is the
combination of:

1. a block-structured nonredundant orbital parameterization suitable for sparse
   VB orbitals;
2. a matrix-free second-order optimizer in that reduced space;
3. a union-support treatment of partially overlapping orbital blocks that
   broadens applicability beyond exact-support cases.

The remainder of the paper is organized as follows. Section 2 defines the
reduced nonredundant orbital space and the associated matrix-free second-order
model. Section 3 describes the trust-region truncated-Newton algorithm and its
transported secant preconditioner. Section 4 explains the union-support
generalization for partially overlapping orbital blocks. Section 5 summarizes
the implementation in `xmvb-cpp`. Section 6 outlines the benchmark protocol and
reports preliminary examples. Section 7 discusses limitations and future
directions.

---

## 2. Theory

### 2.1 Relaxed VBSCF Orbital Objective

Let

$$
E(\theta)
$$

denote the fully relaxed VBSCF objective after all dependent wave function
quantities, including selected-state coefficients or state-averaged weights
when relevant, have been optimized for the current orbitals. Here

$$
\theta \in \mathbb{R}^{n_{\mathrm{raw}}}
$$

is the raw sparse orbital parameter vector associated with the explicit orbital
coefficient slots stored by the current program.

Direct optimization in $\theta$-space is unattractive for two reasons. First,
the raw coefficient representation contains redundancy induced by the
nonorthogonal orbital parameterization. Second, the local curvature in these
coordinates is often highly anisotropic, which slows first-order or quasi-Newton
methods.

### 2.2 Block-Local Nonredundant Coordinates

We introduce reduced orbital coordinates

$$
x \in \mathbb{R}^{n_{\mathrm{nr}}},
\qquad
n_{\mathrm{nr}} \ll n_{\mathrm{raw}},
$$

and interpret a reduced step through a linear embedding

$$
\delta \theta = P \, \delta x.
$$

The reduced gradient and reduced Hessian are therefore

$$
g_{\mathrm{nr}} = P^{\mathsf T} g_{\mathrm{raw}},
\qquad
H_{\mathrm{nr}} = P^{\mathsf T} H_{\mathrm{raw}} P.
$$

The central task is to define $P$ so that it spans only physically meaningful
orbital directions while remaining cheap to construct.

Orbitals are partitioned into local blocks according to sparse AO support.
Within one block $b$, let

$$
C_b \in \mathbb{R}^{n_b^{\mathrm{AO}} \times n_b^{\mathrm{occ}}}
$$

be the occupied orbital coefficient matrix restricted to the block AO domain,
and let

$$
S_b \in \mathbb{R}^{n_b^{\mathrm{AO}} \times n_b^{\mathrm{AO}}}
$$

be the corresponding AO overlap matrix.

The block-local virtual complement is constructed from the $S_b$-orthogonal
projector

$$
Q_b
=
I
- C_b
\left(C_b^{\mathsf T} S_b C_b\right)^{-1}
C_b^{\mathsf T} S_b.
$$

We then diagonalize

$$
Q_b^{\mathsf T} S_b Q_b
$$

and retain the non-null eigenvectors to obtain an $S_b$-orthonormal local
virtual basis

$$
V_b.
$$

Within each block, we retain only the following orbital couplings:

1. inactive $\leftrightarrow$ active,
2. inactive $\leftrightarrow$ virtual,
3. active $\leftrightarrow$ virtual.

Inactive-inactive, active-active, and virtual-virtual rotations are omitted in
the present production method because they are redundant or less relevant to
the dominant relaxation directions.

### 2.3 Candidate Direction Gram Matrix

The raw candidate directions generated from the couplings above are not
orthonormal in the packed sparse coefficient coordinates. Let

$$
D_b = \left[d_{b,1}, d_{b,2}, \ldots, d_{b,m_b}\right]
$$

collect all candidate block directions expressed in packed raw parameters. The
block candidate Gram matrix is

$$
G_b = D_b^{\mathsf T} D_b.
$$

Diagonalization of $G_b$ yields

$$
G_b = U_b \Lambda_b U_b^{\mathsf T}.
$$

After discarding numerically null eigenvalues, the orthonormal reduced basis is

$$
Q_b = D_b U_b \Lambda_b^{-1/2}.
$$

Concatenating all blocks gives the full nonredundant embedding matrix

$$
P = \left[Q_1 \; Q_2 \; \cdots \; Q_{n_{\mathrm{blocks}}}\right].
$$

### 2.4 Matrix-Free Local Quadratic Model

At the current reduced-space iterate $x_k$, we define the quadratic model

$$
m_k(p) = E_k + g_k^{\mathsf T} p + \frac{1}{2} p^{\mathsf T} \widehat H_k p,
$$

where $E_k = E(x_k)$ and $g_k$ is the reduced gradient. The operator

$$
\widehat H_k
$$

is never assembled explicitly. Instead, products $\widehat H_k v$ are obtained
through matrix-free reduced Hessian-vector probes.

In practice, a reduced-space direction $v$ is first expanded back to a packed
raw parameter displacement, a perturbed VBSCF gradient is evaluated, and the
directional curvature is recovered by finite difference. This yields a
matrix-free Hessian-vector product suitable for Krylov subspace methods.

---

## 3. Trust-Region Truncated-Newton Algorithm

### 3.1 Reduced Trust-Region Subproblem

At each outer iteration, we approximately solve

$$
\min_{\|p\| \le \Delta_k} \;
g_k^{\mathsf T} p + \frac{1}{2} p^{\mathsf T} \widehat H_k p,
$$

where $\Delta_k$ is the current trust radius. The solution is computed with a
truncated conjugate-gradient procedure in the reduced nonredundant space. The
inner iteration terminates when one of the following conditions is met:

1. the residual is sufficiently small,
2. negative or nonpositive curvature is encountered,
3. the candidate step reaches the trust-region boundary,
4. a preset Krylov budget is exhausted.

Because the reduced dimension is already much smaller than the raw parameter
dimension, the resulting TN subproblem is substantially cheaper than a
full-space second-order solve.

### 3.2 Step Acceptance and Trust-Radius Update

Let the predicted decrease from the reduced quadratic model be

$$
\Delta E_k^{\mathrm{pred}},
$$

and let the actual decrease be

$$
\Delta E_k^{\mathrm{act}} = E(x_k) - E(x_k + p_k).
$$

We define the trust ratio

$$
\rho_k
=
\frac{\Delta E_k^{\mathrm{act}}}{\Delta E_k^{\mathrm{pred}}}.
$$

The step is accepted when $\rho_k$ exceeds a minimum threshold. Otherwise the
trust radius is reduced and a new reduced-space step is computed. Large values
of $\rho_k$, especially for boundary steps, trigger trust-radius expansion.

This procedure provides a more stable globalization strategy than direct line
search on approximate Newton directions, particularly when matrix-free HVPs are
computed from finite-difference gradient probes.

### 3.3 Transported Secant Preconditioning

Although the TN subproblem is solved in reduced coordinates, curvature
information from previous accepted steps remains useful. To exploit this
information without forming a dense Hessian, we transport a short history of
accepted reduced-space secant pairs into the current reduced basis and combine
them with a cheap diagonal curvature model derived from block-local
one-electron energy differences.

The resulting preconditioner is not intended to approximate the exact reduced
Hessian accurately in every direction. Instead, it improves the scaling of the
Krylov subproblem enough to reduce the number of HVP applications required per
accepted step.

---

## 4. Partial-Overlap Orbital Blocks

### 4.1 Limitation of Exact-Support Block Assumptions

A naive block-local nonredundant construction assumes that all orbitals inside
one block share the same AO support pattern. Under this assumption, a single
representative orbital can define the block AO domain, and all other orbitals
can be embedded in that same local coordinate system.

This assumption is too restrictive for realistic VBSCF inputs. In many cases,
orbitals belong to the same overlap-connected region while sharing only part of
their AO supports. Such partially overlapping blocks violate the representative
support assumption and cause a reduced-space setup failure before optimization
begins.

### 4.2 Union-Support Construction

To remove this restriction, we define the block AO domain as the union of all
explicit AO basis functions appearing in the member orbitals:

$$
\Omega_b
=
\bigcup_{i \in b} \mathrm{supp}(\phi_i).
$$

The occupied block matrix $C_b$ is then assembled on $\Omega_b$, with
coefficients outside each orbital's native sparse support treated as zero.
Local overlap matrices, virtual complements, and candidate directions are all
built on this union-support domain.

This construction has two consequences. First, partially overlapping orbitals
can now be treated in the same nonredundant framework as exact-support blocks.
Second, exact-support blocks become a special case of the more general
union-support formalism, so no separate algorithmic branch is required in the
reduced-space solver.

### 4.3 Practical Significance

The union-support construction is not merely a compatibility patch. It expands
the class of VBSCF wave functions accessible to the nonredundant second-order
optimizer and removes a major obstacle to routine application. In the present
implementation, systems that previously terminated at iteration zero because of
partially overlapping block metadata can now enter the normal optimization
loop.

---

## 5. Implementation in `xmvb-cpp`

The method is implemented in the C++ VBSCF code path of `xmvb-cpp`. The
nonredundant truncated-Newton backend operates on the exact VBSCF
energy-and-gradient evaluator already used by the full-space optimizers. The
main implementation components are:

1. sparse-orbital packing and unpacking for differentiable raw coefficients;
2. block-local nonredundant space construction from occupied and virtual
   orbitals;
3. matrix-free reduced Hessian-vector products generated from relaxed orbital
   gradients;
4. trust-region TN stepping with transported secant preconditioning.

The optimizer is fully integrated into the existing VBSCF infrastructure rather
than implemented as a standalone prototype. This allows direct comparison with
existing full-space L-BFGS and nonredundant L-BFGS backends under identical
energy, gradient, and structure-expansion pathways.

The present production implementation also preserves compatibility with sparse
orbital inputs and legacy runtime metadata. The partially overlapping block
generalization is incorporated directly into the nonredundant orbital-space
construction so that the optimizer can operate on the original sparse orbital
layout.

---

## 6. Results

### 6.1 Benchmark Design

This section will report the systematic benchmark in the final version. The
benchmark set should include:

1. small and medium closed-shell systems,
2. open-shell systems,
3. different VB structure-space choices,
4. inputs with partially overlapping orbital blocks,
5. cases where full-space optimization is known to require many outer
   iterations.

The primary comparisons will be made against:

1. full-space L-BFGS in raw orbital coordinates,
2. nonredundant L-BFGS in the reduced orbital space,
3. the present nonredundant truncated-Newton method,
4. legacy reduced-Hessian or legacy Newton-style references when available.

The key reported metrics will be:

1. number of accepted outer iterations,
2. number of objective/gradient evaluations,
3. number of reduced HVP applications,
4. total wall time,
5. final energy differences relative to a common reference tolerance.

### 6.2 Preliminary Examples

Even before the full benchmark is assembled, several representative behaviors
are already clear from the current implementation.

First, the nonredundant truncated-Newton optimizer converges in very few outer
iterations on standard test systems. For example, the current implementation
reaches convergence in three iterations for F$_2$ and in six iterations for
benzene in the tested full-structure setup. These examples do not constitute a
complete benchmark, but they are consistent with the intended role of the
matrix-free second-order method: reducing the number of expensive relaxed VBSCF
orbital steps.

Second, the partially overlapping block generalization removes a structural
failure mode rather than merely improving asymptotic efficiency. In earlier
versions of the nonredundant optimizer, some realistic inputs terminated before
the first optimization step because partially overlapping block metadata could
not be mapped into an exact-support reduced space. With the union-support
construction, these inputs now proceed into the normal optimization loop.

### 6.3 Placeholder Table Structure

The final manuscript should include at least one table of the following form.

| System | Structure space | Optimizer | Iterations | Grad/Energy calls | HVP calls | Final energy (Eh) | Wall time (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| F$_2$ | full | raw L-BFGS | TBD | TBD | 0 | TBD | TBD |
| F$_2$ | full | nonredundant L-BFGS | TBD | TBD | 0 | TBD | TBD |
| F$_2$ | full | nonredundant TN | 3 | TBD | TBD | TBD | TBD |
| C$_6$H$_6$ | full | raw L-BFGS | TBD | TBD | 0 | TBD | TBD |
| C$_6$H$_6$ | full | nonredundant L-BFGS | TBD | TBD | 0 | TBD | TBD |
| C$_6$H$_6$ | full | nonredundant TN | 6 | TBD | TBD | TBD | TBD |

The partially overlapping cases should be summarized separately with an
"old behavior vs new behavior" table.

---

## 7. Discussion

The present method targets a specific regime of the VBSCF optimization problem.
It is most useful when full-space orbital optimization suffers from slow
convergence in redundant coordinates, but full explicit reduced Hessians are too
expensive to build and solve routinely.

The method should not be interpreted as a universal replacement for every
possible optimizer. For very small reduced spaces, explicit second-order
methods may still be competitive. For very noisy or aggressively truncated
energy-and-gradient evaluations, full quasi-Newton updates may remain more
robust. The main contribution of the present work is instead the practical
middle ground it opens: second-order-quality orbital updates in a compact
nonredundant space without explicit Hessian construction.

The partial-overlap treatment is especially important from a generality
perspective. A reduced-space algorithm that works only for exact-support blocks
is difficult to use as a production optimizer in realistic VBSCF workflows.
By moving to a union-support block formalism, the present method removes one of
the main reasons the nonredundant optimizer would previously fail before any
scientific comparison could even be made.

Several extensions remain possible. A more sophisticated block-local curvature
model could improve the preconditioner further. Adaptive inner Krylov
tolerances may reduce unnecessary HVP probes late in convergence. Finally, a
more intrinsic reduced-space treatment of sparse orbital manifolds could avoid
some of the residual dependence on raw sparse coefficient slots. These
directions, however, are improvements to an already usable framework rather
than prerequisites for the method described here.

---

## 8. Conclusions

We have presented a nonredundant matrix-free second-order orbital optimizer for
VBSCF theory. The method combines block-local nonredundant orbital coordinates,
matrix-free reduced Hessian-vector products, and a trust-region
truncated-Newton solver with transported secant preconditioning. The optimizer
avoids explicit reduced Hessian construction while retaining the rapid outer
convergence expected from second-order orbital optimization.

A further contribution is the extension from exact-support orbital blocks to
partially overlapping sparse blocks through a union-support reduced-space
construction. This removes a major setup restriction and broadens the range of
VBSCF inputs accessible to the nonredundant optimizer.

Taken together, these developments establish a practical route toward
production-quality second-order orbital optimization in modern VBSCF
implementations. A complete benchmark analysis will determine the final
performance envelope, but the present theory and preliminary calculations
already show that matrix-free reduced-space second-order optimization is a
promising and general strategy for VBSCF orbital relaxation.

---

## Notes for Finalization

1. Replace the title with the final journal-targeted version.
2. Add literature citations throughout:
   - VBSCF background
   - legacy XMVB orbital optimization
   - nonorthogonal orbital Hessian work
   - trust-region / truncated-Newton references
3. Fill in the benchmark protocol in Section 6.1.
4. Replace preliminary examples in Section 6.2 with full quantitative tables.
5. Add one algorithm box for the nonredundant TN outer loop.
6. Add one schematic figure for:
   - sparse orbital blocks,
   - union-support partial-overlap construction,
   - reduced-space TN workflow.
