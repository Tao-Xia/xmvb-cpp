# Exact Separator Gradient Design

> Status note (2026-04-05):
> this note is the current design target for exact-separator gradients.
> The immediate goal is not another performance pass. The immediate goal is to
> finish an exact active-space gradient path that is structurally aligned with
> the current exact value kernel, so later performance work can optimize both
> forward and backward together.

## 1. Purpose

The current exact-separator code already has:

- an exact rooted-tree value kernel for overlap and Hamiltonian channels in
  [component_tree.cpp](./component_tree.cpp);
- a determinant-pair active-space gradient implementation in
  [cpp_active_space_gradient_evaluator.cpp](../scf/cpp_active_space_gradient_evaluator.cpp);
- exact one-leaf / rooted-tree first-cofactor aggregates exposed in the current
  separator code.

What is still missing is the exact-separator analogue of the active-space
gradient kernel.

This note answers four questions:

1. what should be reused directly from the determinant-pair gradient code;
2. what exact active-space derivatives can already be read off from the current
   separator forward kernel;
3. what new reverse-mode message algebra is required for the overlap gradient;
4. how the implementation should be staged so gradient work finishes before the
   next performance pass.

The scope here is the **active-space gradient**

$$
\frac{\partial \mathcal L}{\partial S_{\mathrm{act}}},
\qquad
\frac{\partial \mathcal L}{\partial h^{(1)}_{\mathrm{act}}},
\qquad
\frac{\partial \mathcal L}{\partial g_{\mathrm{act}}},
$$

not the later AO / orbital backpropagation. Once these three exact active-space
derivatives exist, the existing orbital backpropagation stack can be reused.

---

## 2. Core Design Decision

We **should** reference the determinant-pair gradient method, but only at the
right abstraction level.

We should **reuse**:

- the state-averaged generalized-eigenproblem adjoint;
- the structure-pair weighting logic;
- the validation oracle against exact determinant-pair gradients.

We should **not** reuse:

- determinant-pair enumeration as the local gradient kernel;
- determinant-pair-local inverse-overlap loops as the production separator
  backward path.

The separator value kernel already replaced local determinant-pair enumeration
by exact message algebra. The gradient must follow the same structural change.
Otherwise the value path becomes separator-based while the backward path falls
back to determinant pairs, and the eventual forward/backward performance model
stays inconsistent.

So the correct split is:

1. outer structure-matrix adjoint: reuse determinant-pair logic exactly;
2. local structure-pair derivative kernel: replace determinant-pair loops by an
   exact separator forward/backward message kernel.

---

## 3. Outer Adjoint Reused From Determinant-Pair Method

The current determinant-pair gradient implementation already computes the
correct state-averaged adjoints of the structure Hamiltonian and overlap
matrices. In code this is
[determinant_pair_structure_adjoints(...)](../scf/cpp_active_space_gradient_evaluator.cpp).

For selected states $k \in \mathcal K$ with normalized weights $\omega_k$,
eigenvalues $E_k$, and generalized eigenvectors $c^{(k)}$, define the
upper-triangle multiplicity

$$
\eta_{IJ} =
\begin{cases}
1, & I = J, \\
2, & I \ne J.
\end{cases}
$$

Then the exact structure-pair adjoints are

$$
W^H_{IJ}
=
\eta_{IJ}
\sum_{k \in \mathcal K}
\omega_k\,
c^{(k)}_I c^{(k)}_J,
$$

$$
W^S_{IJ}
=
-
\eta_{IJ}
\sum_{k \in \mathcal K}
\omega_k E_k\,
c^{(k)}_I c^{(k)}_J.
$$

Therefore for any active-space parameter block $\theta$,

$$
\frac{\partial \mathcal L}{\partial \theta}
=
\sum_{I \le J}
\left(
W^H_{IJ}\,
\frac{\partial H_{IJ}}{\partial \theta}
+
W^S_{IJ}\,
\frac{\partial S_{IJ}}{\partial \theta}
\right).
$$

This layer is completely independent of whether $H_{IJ}$ and $S_{IJ}$ are
computed by determinant pairs or by separator messages. So it should be reused
unchanged.

---

## 4. Current Exact-Separator Forward Object

For one rooted structure pair $(I,J)$, the current generic exact value kernel
returns a typed Hamiltonian boundary bundle

$$
P_{IJ}
=
\left(
O_{IJ},
A^\alpha_{IJ},
A^\beta_{IJ},
\Gamma^\alpha_{IJ},
\Gamma^\beta_{IJ},
M_{IJ}
\right),
$$

where:

- $O_{IJ}$ is the closed overlap scalar;
- $A^\alpha_{IJ}(r,c)$ is the exact alpha first-order deleted-minor sector
  already multiplied by closed beta overlap;
- $A^\beta_{IJ}(r,c)$ is the symmetric beta channel;
- $\Gamma^\alpha_{IJ}\big((r_1,r_2),(c_1,c_2)\big)$ is the exact alpha
  degree-2 deleted-minor sector already multiplied by closed beta overlap;
- $\Gamma^\beta_{IJ}$ is the symmetric beta channel;
- $M_{IJ}((r,c),(q,b))$ is the exact mixed alpha/beta degree-1 product
  channel used by opposite-spin contraction.

In the current code this bundle is represented by
`HamiltonianBoundaryPayload` in [component_tree.cpp](./component_tree.cpp), and
the final root contractions are:

$$
S_{IJ} = O_{IJ},
$$

$$
H^{(1)}_{IJ}
=
\sum_{r,c}
h[c,r]\,
A^\alpha_{IJ}(r,c)
+
\sum_{q,b}
h[b,q]\,
A^\beta_{IJ}(q,b),
$$

$$
H^{(2,\alpha\alpha)}_{IJ}
=
\sum_{r_1<r_2}
\sum_{c_1<c_2}
\left(
g[(r_1,c_1),(r_2,c_2)]
-
g[(r_1,c_2),(r_2,c_1)]
\right)
\Gamma^\alpha_{IJ}\big((r_1,r_2),(c_1,c_2)\big),
$$

$$
H^{(2,\beta\beta)}_{IJ}
=
\sum_{q_1<q_2}
\sum_{b_1<b_2}
\left(
g[(q_1,b_1),(q_2,b_2)]
-
g[(q_1,b_2),(q_2,b_1)]
\right)
\Gamma^\beta_{IJ}\big((q_1,q_2),(b_1,b_2)\big),
$$

$$
H^{(2,\alpha\beta)}_{IJ}
=
\sum_{r,c,q,b}
g[(q,b),(r,c)]\,
M_{IJ}((r,c),(q,b)).
$$

These formulas are exactly what the current root contractions in
[component_tree.cpp](./component_tree.cpp) implement:

- one-electron via `contract_spin_one_electron_first_sectors(...)`;
- same-spin via `contract_same_spin_second_sectors(...)`;
- opposite-spin via `contract_opposite_spin_first_sectors(...)`.

---

## 5. Exact Active-Space Gradients Already Exposed By The Current Forward Kernel

The important consequence of Section 4 is that the current value kernel already
contains enough exact information to compute two of the three active-space
gradient blocks directly.

### 5.1 One-Electron Gradient

Since the one-electron contraction is linear in $h$,

$$
\frac{\partial \mathcal L}{\partial h[c,r]}
=
\sum_{I \le J}
W^H_{IJ}\,
\left(
A^\alpha_{IJ}(r,c)
+
A^\beta_{IJ}(r,c)
\right).
$$

This is exactly the same object as the accumulated support-space first-cofactor
matrices already exposed by the rooted-tree debug path:

- `alpha_first_cofactor`;
- `beta_first_cofactor`;

in [component_tree.hpp](./component_tree.hpp) and
[component_tree.cpp](./component_tree.cpp).

So the exact-separator one-electron active-space gradient is not a new theory
problem. It is already present in the forward payload.

### 5.2 Two-Electron Gradient

The same is true for the active-space two-electron tensor because the root
contractions are linear in the packed ERIs.

For same-spin:

$$
\frac{\partial \mathcal L}{\partial g[(r_1,c_1),(r_2,c_2)]}
\mathrel{+}=
\sum_{I \le J}
W^H_{IJ}\,
\Gamma^\alpha_{IJ}\big((r_1,r_2),(c_1,c_2)\big),
$$

$$
\frac{\partial \mathcal L}{\partial g[(r_1,c_2),(r_2,c_1)]}
\mathrel{-}=
\sum_{I \le J}
W^H_{IJ}\,
\Gamma^\alpha_{IJ}\big((r_1,r_2),(c_1,c_2)\big),
$$

and analogously for beta.

For opposite-spin:

$$
\frac{\partial \mathcal L}{\partial g[(q,b),(r,c)]}
\mathrel{+}=
\sum_{I \le J}
W^H_{IJ}\,
M_{IJ}((r,c),(q,b)).
$$

So just like the one-electron case, the exact-separator two-electron
active-space gradient is already carried by the current forward bundle.

### 5.3 Immediate Consequence

The exact-separator active-space gradient problem is **not** uniformly hard.

- $ \partial \mathcal L / \partial h_{\mathrm{act}} $ is straightforward.
- $ \partial \mathcal L / \partial g_{\mathrm{act}} $ is straightforward.
- $ \partial \mathcal L / \partial S_{\mathrm{act}} $ is the real new work.

That is the correct priority ordering.

---

## 6. Why The Overlap Gradient Is The Real New Work

The determinant-pair gradient code computes the overlap derivative through the
inverse overlap matrix.

For one non-singular spin-overlap block

$$
X \in \mathbb R^{n \times n},
\qquad
D = \det X,
\qquad
K = X^{-1},
$$

the first cofactor matrix is

$$
C = D K^{\mathsf T}.
$$

The determinant-pair active-space gradient code uses exactly this structure.
Its overlap pullback in [spin_pair_utils.cpp](../matrices/spin_pair_utils.cpp)
is

$$
\frac{\partial \mathcal L}{\partial X}
=
\bar D\,
D K^{\mathsf T}
-
D K^{\mathsf T}\,
\bar K\,
K^{\mathsf T},
$$

where:

- $\bar D$ is the scalar adjoint for the determinant;
- $\bar K$ is the adjoint for the inverse overlap matrix.

In the determinant-pair code, these are assembled as follows:

$$
\bar D^\alpha
=
W^S_{IJ} D^\beta
+
W^H_{IJ} D^\beta
\left(
\Phi^\alpha
+
\Phi^\beta
+
\Phi^{\alpha\beta}
\right),
$$

$$
\bar K^\alpha
=
W^H_{IJ} D^\beta
\left(
\nabla_{K^\alpha}\Phi^\alpha
+
\nabla_{K^\alpha}\Phi^{\alpha\beta}
\right),
$$

and symmetrically for beta.

This is exactly what the current determinant-pair gradient code does around
[cpp_active_space_gradient_evaluator.cpp](../scf/cpp_active_space_gradient_evaluator.cpp).

### 6.1 Why We Should Not Reuse This Local Kernel Directly

If we simply reintroduce determinant-pair-local inverse-overlap loops inside
the separator gradient path, then:

1. the value kernel uses exact separator messages;
2. the backward kernel falls back to determinant pairs;
3. future performance work cannot optimize forward and backward together.

So the correct gradient design is:

- reuse the outer structure adjoints;
- replace the local determinant-pair gradient kernel by a separator reverse
  pass.

---

## 7. Exact Root Seeds For Separator Reverse Mode

Let the total exact electronic contribution for one rooted structure pair be

$$
H^{\mathrm{el}}_{IJ}
=
H^{(1)}_{IJ}
+
H^{(2,\alpha\alpha)}_{IJ}
+
H^{(2,\beta\beta)}_{IJ}
+
H^{(2,\alpha\beta)}_{IJ}.
$$

The separator reverse pass starts from the root bundle $P_{IJ}$. The exact
root seeds are:

$$
\bar O_{IJ} = W^S_{IJ},
$$

$$
\bar A^\alpha_{IJ}(r,c)
\mathrel{+}=
W^H_{IJ}\, h[c,r],
\qquad
\bar A^\beta_{IJ}(q,b)
\mathrel{+}=
W^H_{IJ}\, h[b,q],
$$

$$
\bar \Gamma^\alpha_{IJ}\big((r_1,r_2),(c_1,c_2)\big)
\mathrel{+}=
W^H_{IJ}
\left(
g[(r_1,c_1),(r_2,c_2)]
-
g[(r_1,c_2),(r_2,c_1)]
\right),
$$

$$
\bar \Gamma^\beta_{IJ}\big((q_1,q_2),(b_1,b_2)\big)
\mathrel{+}=
W^H_{IJ}
\left(
g[(q_1,b_1),(q_2,b_2)]
-
g[(q_1,b_2),(q_2,b_1)]
\right),
$$

$$
\bar M_{IJ}((r,c),(q,b))
\mathrel{+}=
W^H_{IJ}\, g[(q,b),(r,c)].
$$

These are the exact reverse-mode seeds for the current forward closure.

Everything after that is pure reverse-mode differentiation of the separator
message algebra.

---

## 8. The Right Overlap-Gradient Object: Sector Adjoint, Not Determinant-Pair Re-Enumeration

The current separator forward algebra is written in terms of exact deleted
minors, not in terms of explicit inverse overlap matrices. The backward pass
should use the same object.

For a local overlap block $X$, define one deleted-minor sector

$$
m_{R,C}(X)
=
s(R,C)\,
\det X[\bar R,\bar C],
$$

where:

- $R$ is the deleted row-label set;
- $C$ is the deleted column-label set;
- $X[\bar R,\bar C]$ is the surviving minor;
- $s(R,C)$ is the exact separator sign convention already used in the forward
  payload.

Let $\bar m_{R,C}$ be the reverse-mode adjoint on that sector after the full
message reverse pass.

Then the exact local overlap-block gradient is

$$
\frac{\partial \mathcal L}{\partial X_{ij}}
=
\sum_{R,C}
\bar m_{R,C}\,
\frac{\partial m_{R,C}(X)}{\partial X_{ij}}.
$$

If $i \in R$ or $j \in C$, the derivative is zero. Otherwise let
$\rho_R(i)$ and $\kappa_C(j)$ be the compressed row/column indices of
$i$ and $j$ inside the surviving minor $X[\bar R,\bar C]$. Then

$$
\frac{\partial m_{R,C}(X)}{\partial X_{ij}}
=
s(R,C)\,
\operatorname{Cof}
\left(
X[\bar R,\bar C]
\right)_{\rho_R(i),\kappa_C(j)}.
$$

So the exact leaf pullback is

$$
\frac{\partial \mathcal L}{\partial X_{ij}}
=
\sum_{\substack{R,C \\ i \notin R,\; j \notin C}}
\bar m_{R,C}\,
s(R,C)\,
\operatorname{Cof}
\left(
X[\bar R,\bar C]
\right)_{\rho_R(i),\kappa_C(j)}.
$$

This formula is the key reason the separator gradient should be written as a
reverse pass on deleted-minor messages.

### 8.1 Why This Is Better Than Reusing Determinant-Pair Inverse Formulas

This leaf pullback:

- is exactly aligned with the forward payload algebra;
- does not force a determinant-pair loop to reappear in the separator kernel;
- is compatible with singular cases, because deleted minors and their
  cofactors remain meaningful even when the full block is singular.

That last point matters. The current determinant-pair active-space gradient
code assumes non-singular spin overlaps for the overlap derivative, while the
separator value kernel is already formulated in deleted-minor language.
Designing the separator backward pass around deleted-minor adjoints avoids
hard-coding the determinant-pair nullity restriction into the new path.

---

## 9. Reverse Merge Prototype: One-Spin Degree-1

The current production rooted-tree Hamiltonian path uses a larger typed bundle,
but the exact reverse structure is already visible in the degree-1 one-spin
merge.

Suppose

$$
m_A^{(1)} = (S_A, r_A, c_A, C_A),
\qquad
m_B^{(1)} = (S_B, r_B, c_B, C_B),
$$

and the forward exact merge is

$$
S = S_A S_B,
$$

$$
r
=
\alpha_r S_B r_A
+
\beta_r S_A r_B,
$$

$$
c
=
\alpha_c S_B c_A
+
\beta_c S_A c_B,
$$

$$
C
=
S_B C_A
+
\sigma S_A C_B
-
\tau r_A c_B^{\mathsf T}
-
\upsilon r_B c_A^{\mathsf T},
$$

with exact sign factors
$\alpha_r,\beta_r,\alpha_c,\beta_c,\sigma,\tau,\upsilon \in \{\pm 1\}$.

Let the reverse adjoint on the merged payload be

$$
\bar m = (\bar S,\bar r,\bar c,\bar C).
$$

Then the exact reverse merge is

$$
\bar C_A \mathrel{+}= S_B \bar C,
\qquad
\bar C_B \mathrel{+}= \sigma S_A \bar C,
$$

$$
\bar r_A \mathrel{+}= \alpha_r S_B \bar r - \tau \bar C c_B,
\qquad
\bar r_B \mathrel{+}= \beta_r S_A \bar r - \upsilon \bar C c_A,
$$

$$
\bar c_A \mathrel{+}= \alpha_c S_B \bar c - \upsilon \bar C^{\mathsf T} r_B,
\qquad
\bar c_B \mathrel{+}= \beta_c S_A \bar c - \tau \bar C^{\mathsf T} r_A,
$$

$$
\bar S_A \mathrel{+}=
S_B \bar S
+
\beta_r \langle \bar r, r_B \rangle
+
\beta_c \langle \bar c, c_B \rangle
+
\sigma \langle \bar C, C_B \rangle_F,
$$

$$
\bar S_B \mathrel{+}=
S_A \bar S
+
\alpha_r \langle \bar r, r_A \rangle
+
\alpha_c \langle \bar c, c_A \rangle
+
\langle \bar C, C_A \rangle_F.
$$

This is the pattern that matters.

For the current typed Hamiltonian bundle, the same rule applies channel by
channel:

- every forward term of the form
  $z \mathrel{+}= \gamma x y$
  becomes
  $\bar x \mathrel{+}= \gamma \bar z y$,
  $\bar y \mathrel{+}= \gamma \bar z x$;
- every exact sign and parity factor is reused unchanged;
- every canonicalization / aggregation step in the forward payload has a
  matching accumulation step in reverse.

So the separator backward pass is not a new mathematical object. It is the
transpose of the existing exact multilinear message recurrence.

---

## 10. Root Multi-Child Boundary Reverse Closure

Sections 7--9 explain the generic reverse-mode principle, but they still stop
one step short of the missing rooted-tree formula.

That missing step is the following:

- the current public rooted-tree overlap-gradient path is exact only because it
  temporarily flattens the whole root subtree into one synthetic leaf;
- that flattening is numerically correct, but it is **not** yet the structural
  $2^n \to 2^m$ reverse closure we actually want;
- the real missing formula is the exact multi-child root reverse written
  directly on boundary messages.

This section derives that closure.

### 10.1 Exact Bundle Carried By One Subtree

For one subtree block $X$ and one spin channel $\sigma \in \{\alpha,\beta\}$,
write the exact boundary bundle as

$$
B_X^\sigma
=
\left(
s_X^\sigma,\;
u_X^\sigma,\;
G_X^\sigma
\right),
$$

where:

- $s_X^\sigma$ is the degree-$0$ closed overlap scalar;
- $u_X^\sigma(i)$ is the exact degree-$1$ boundary message on the one-spin
  basis index $i$;
- $G_X^\sigma(\nu)$ is the exact degree-$2$ same-spin boundary message on the
  antisymmetrized pair-pair basis index $\nu$.

For opposite-spin we also need the mixed second-order moment

$$
X_X(i,j)
=
\sum_{t \in X}
w_t\,
u_t^\alpha(i)\,
u_t^\beta(j),
$$

where $t$ indexes exact local states and $w_t$ is the exact weight already
produced by the separator forward algebra.

So the full exact rooted-tree message is

$$
\mathcal B_X
=
\left(
B_X^\alpha,\;
B_X^\beta,\;
X_X
\right).
$$

The direct $2^n \to 2^m$ root reverse exists if and only if the root can be
written and differentiated using only these objects, without flattening the
whole root subtree into one exact determinant-pair-like leaf.

### 10.2 Binary Exact Merge On Boundary Bundles

Let $X$ and $Y$ be two exact blocks being merged across one separator edge.
Denote the merged block by

$$
Z = X \star Y.
$$

The exact separator recurrence is associative, so a multi-child root can be
built from this binary merge.

For overlap, the forward closure is simply

$$
s_Z^\sigma
=
s_X^\sigma s_Y^\sigma.
$$

For the degree-$1$ channel, let

$$
T^\sigma_{k i m}
$$

be the exact sparse structure tensor that inserts one deleted index into the
merged boundary basis with the correct separator sign convention. Then

$$
u_Z^\sigma(k)
=
\sum_{i,m}
T^\sigma_{k i m}\,
u_X^\sigma(i)\,
u_Y^\sigma(m).
$$

This is the exact degree-$1$ closure already used implicitly by the current
one-electron / first-cofactor rooted-tree path.

For opposite-spin, the exact mixed closure is

$$
X_Z(k,\ell)
=
\sum_{i,m,j,n}
T^\alpha_{k i m}\,
T^\beta_{\ell j n}\,
X_X(i,j)\,
X_Y(m,n).
$$

This is the exact opposite-spin closure derived in
[root_closure.md](./root_closure.md).

The same-spin degree-$2$ channel is the crucial missing piece. In full basis
form, the exact merge is

$$
G_Z^\sigma(\nu)
=
\sum_{\rho}
P^\sigma_{\nu\rho}(Y)\,
G_X^\sigma(\rho)
+
\sum_{\lambda}
Q^\sigma_{\nu\lambda}(X)\,
G_Y^\sigma(\lambda)
+
\sum_{i,m}
\mathcal S^\sigma_{\nu i m}\,
u_X^\sigma(i)\,
u_Y^\sigma(m).
$$

Here:

- $P^\sigma_{\nu\rho}(Y)$ is the exact signed insertion map that places a
  degree-$2$ object from $X$ into the merged basis in the presence of the
  complementary block $Y$;
- $Q^\sigma_{\nu\lambda}(X)$ is the symmetric insertion map for a degree-$2$
  object from $Y$;
- $\mathcal S^\sigma_{\nu i m}$ is the exact sparse structure tensor that
  combines two degree-$1$ objects into one degree-$2$ object.

This is the precise mathematical statement of the same-spin root closure that
the current rooted-tree reverse path is still missing.

If the message basis is chosen so that the insertion signs are absorbed into
the basis itself, then the same formula simplifies to

$$
G_Z^\sigma
=
s_Y^\sigma\, G_X^\sigma
+
s_X^\sigma\, G_Y^\sigma
+
\mathcal S^\sigma\!\left(u_X^\sigma, u_Y^\sigma\right).
$$

This is the cleanest exact production target:

1. carry the full exact degree-$2$ boundary vector $G^\sigma$;
2. never collapse same-spin to one scalar before the final root contraction;
3. let the root scalar be one final linear functional of $G^\sigma$.

### 10.3 Root Multi-Child Forward Closure

Let the root have children

$$
c_1,\dots,c_d,
$$

with exact child bundles

$$
\mathcal B_1,\dots,\mathcal B_d,
$$

and let $\mathcal R$ denote the exact local root-complement bundle.

Define the prefix bundles

$$
\mathcal P_0 = \mathcal I,
\qquad
\mathcal P_k = \mathcal P_{k-1} \star \mathcal B_k
\quad (1 \le k \le d),
$$

where $\mathcal I$ is the exact neutral element of the merge.

Then the total root bundle is

$$
\mathcal T
=
\mathcal P_d \star \mathcal R.
$$

Equivalently, for any child $k$,

$$
\mathcal T
=
\mathcal P_{k-1}
\star
\mathcal B_k
\star
\mathcal Q_{k+1},
$$

where the suffix context is defined by

$$
\mathcal Q_{d+1} = \mathcal R,
\qquad
\mathcal Q_k = \mathcal B_k \star \mathcal Q_{k+1}.
$$

This identity is the whole reason a direct root reverse exists: every child
only interacts with a left context and a right context, both of which are
themselves exact boundary bundles.

So the multi-child root closure is not a special new combinatorial object. It
is repeated application of the same binary exact merge.

### 10.4 Final Root Contractions As Linear Functionals

Once the total root bundle $\mathcal T$ has been formed, the final scalars are
linear functionals of its channels:

$$
S_{IJ} = s_{\mathcal T},
$$

$$
H^{(1)}_{IJ}
=
\sum_{r,c}
h[c,r]\,
u_{\mathcal T}^\alpha(r,c)
+
\sum_{q,b}
h[b,q]\,
u_{\mathcal T}^\beta(q,b),
$$

$$
H^{(2,\sigma\sigma)}_{IJ}
=
\sum_{\nu}
\Lambda^\sigma(\nu)\,
G_{\mathcal T}^\sigma(\nu),
$$

$$
H^{(2,\alpha\beta)}_{IJ}
=
\sum_{i,j}
\Omega(i,j)\,
X_{\mathcal T}(i,j),
$$

where:

$$
\Lambda^\alpha\big((r_1,r_2),(c_1,c_2)\big)
=
g[(r_1,c_1),(r_2,c_2)]
-
g[(r_1,c_2),(r_2,c_1)],
$$

and analogously for beta, while

$$
\Omega((r,c),(q,b))
=
g[(q,b),(r,c)].
$$

Therefore the exact root reverse seeds can be written directly on bundle
channels:

$$
\bar s_{\mathcal T}
\mathrel{+}=
W^S_{IJ},
$$

$$
\bar u_{\mathcal T}^\alpha(r,c)
\mathrel{+}=
W^H_{IJ}\, h[c,r],
\qquad
\bar u_{\mathcal T}^\beta(q,b)
\mathrel{+}=
W^H_{IJ}\, h[b,q],
$$

$$
\bar G_{\mathcal T}^\alpha(\nu)
\mathrel{+}=
W^H_{IJ}\, \Lambda^\alpha(\nu),
\qquad
\bar G_{\mathcal T}^\beta(\nu)
\mathrel{+}=
W^H_{IJ}\, \Lambda^\beta(\nu),
$$

$$
\bar X_{\mathcal T}(i,j)
\mathrel{+}=
W^H_{IJ}\, \Omega(i,j).
$$

These are exactly the root seeds we need for a direct bundle reverse. They are
the bundle form of Section 7.

### 10.5 Exact Reverse Of The Binary Merge

The reverse pass is now just the transpose of the exact merge algebra.

For overlap,

$$
\bar s_X^\sigma
\mathrel{+}=
\bar s_Z^\sigma\, s_Y^\sigma,
\qquad
\bar s_Y^\sigma
\mathrel{+}=
\bar s_Z^\sigma\, s_X^\sigma.
$$

For the degree-$1$ channel,

$$
\bar u_X^\sigma(i)
\mathrel{+}=
\sum_{k,m}
\bar u_Z^\sigma(k)\,
T^\sigma_{k i m}\,
u_Y^\sigma(m),
$$

$$
\bar u_Y^\sigma(m)
\mathrel{+}=
\sum_{k,i}
\bar u_Z^\sigma(k)\,
T^\sigma_{k i m}\,
u_X^\sigma(i).
$$

For opposite-spin,

$$
\bar X_X(i,j)
\mathrel{+}=
\sum_{k,\ell,m,n}
\bar X_Z(k,\ell)\,
T^\alpha_{k i m}\,
T^\beta_{\ell j n}\,
X_Y(m,n),
$$

$$
\bar X_Y(m,n)
\mathrel{+}=
\sum_{k,\ell,i,j}
\bar X_Z(k,\ell)\,
T^\alpha_{k i m}\,
T^\beta_{\ell j n}\,
X_X(i,j).
$$

For same-spin, starting from

$$
G_Z^\sigma(\nu)
=
\sum_{\rho}
P^\sigma_{\nu\rho}(Y)\,
G_X^\sigma(\rho)
+
\sum_{\lambda}
Q^\sigma_{\nu\lambda}(X)\,
G_Y^\sigma(\lambda)
+
\sum_{i,m}
\mathcal S^\sigma_{\nu i m}\,
u_X^\sigma(i)\,
u_Y^\sigma(m),
$$

the exact reverse formulas are

$$
\bar G_X^\sigma(\rho)
\mathrel{+}=
\sum_{\nu}
\bar G_Z^\sigma(\nu)\,
P^\sigma_{\nu\rho}(Y),
$$

$$
\bar G_Y^\sigma(\lambda)
\mathrel{+}=
\sum_{\nu}
\bar G_Z^\sigma(\nu)\,
Q^\sigma_{\nu\lambda}(X),
$$

$$
\bar u_X^\sigma(i)
\mathrel{+}=
\sum_{\nu,\lambda}
\bar G_Z^\sigma(\nu)\,
\frac{\partial Q^\sigma_{\nu\lambda}(X)}
{\partial u_X^\sigma(i)}\,
G_Y^\sigma(\lambda)
+
\sum_{\nu,m}
\bar G_Z^\sigma(\nu)\,
\mathcal S^\sigma_{\nu i m}\,
u_Y^\sigma(m),
$$

$$
\bar u_Y^\sigma(m)
\mathrel{+}=
\sum_{\nu,\rho}
\bar G_Z^\sigma(\nu)\,
\frac{\partial P^\sigma_{\nu\rho}(Y)}
{\partial u_Y^\sigma(m)}\,
G_X^\sigma(\rho)
+
\sum_{\nu,i}
\bar G_Z^\sigma(\nu)\,
\mathcal S^\sigma_{\nu i m}\,
u_X^\sigma(i).
$$

These are the exact reverse formulas the current root multi-child same-spin
path still lacks.

In the simplified absorbed-sign basis,

$$
G_Z^\sigma
=
s_Y^\sigma\, G_X^\sigma
+
s_X^\sigma\, G_Y^\sigma
+
\mathcal S^\sigma\!\left(u_X^\sigma, u_Y^\sigma\right),
$$

and the reverse simplifies to

$$
\bar G_X^\sigma
\mathrel{+}=
s_Y^\sigma\, \bar G_Z^\sigma,
\qquad
\bar G_Y^\sigma
\mathrel{+}=
s_X^\sigma\, \bar G_Z^\sigma,
$$

$$
\bar s_X^\sigma
\mathrel{+}=
\left\langle \bar G_Z^\sigma, G_Y^\sigma \right\rangle,
\qquad
\bar s_Y^\sigma
\mathrel{+}=
\left\langle \bar G_Z^\sigma, G_X^\sigma \right\rangle,
$$

$$
\bar u_X^\sigma
\mathrel{+}=
\left(
\mathcal S^\sigma(\,\cdot\,, u_Y^\sigma)
\right)^\ast
\bar G_Z^\sigma,
\qquad
\bar u_Y^\sigma
\mathrel{+}=
\left(
\mathcal S^\sigma(u_X^\sigma, \,\cdot\,)
\right)^\ast
\bar G_Z^\sigma.
$$

This is the exact direct root reverse we want in production.

### 10.6 Multi-Child Root Reverse By Prefix/Suffix Contexts

Let the adjoint on the total root bundle be

$$
\bar{\mathcal T}.
$$

For child $k$, define

$$
\mathcal C_k = \mathcal P_{k-1} \star \mathcal B_k.
$$

Then the reverse through

$$
\mathcal T = \mathcal C_k \star \mathcal Q_{k+1}
$$

gives the intermediate adjoint

$$
\bar{\mathcal C}_k
=
D_1(\star)^\ast_{(\mathcal C_k,\mathcal Q_{k+1})}
\left[
\bar{\mathcal T}
\right],
$$

and the reverse through

$$
\mathcal C_k = \mathcal P_{k-1} \star \mathcal B_k
$$

gives the child adjoint

$$
\bar{\mathcal B}_k
=
D_2(\star)^\ast_{(\mathcal P_{k-1},\mathcal B_k)}
\left[
\bar{\mathcal C}_k
\right].
$$

Here $D_1(\star)^\ast$ and $D_2(\star)^\ast$ are the transposed Jacobians of
the exact binary merge with respect to its left and right arguments.

So the whole multi-child root reverse is:

1. one forward prefix scan;
2. one forward suffix scan;
3. one local binary reverse per child using its left and right contexts.

No full enumeration over all root child mask tuples is required.

This is the precise structural replacement for the current root flattening
shortcut.

### 10.7 Why This Is The Missing $2^n \to 2^m$ Step

The current exact flattening path is numerically correct, but it temporarily
reintroduces the whole root subtree into one exact leaf-like object before the
reverse pass.

By contrast, the formulas above depend only on:

- child boundary bundles $\mathcal B_k$;
- left/right context bundles $\mathcal P_{k-1}, \mathcal Q_{k+1}$;
- fixed sparse structure tensors $T^\sigma$, $\mathcal S^\sigma$ and the
  signed insertion maps $P^\sigma,Q^\sigma$.

Therefore the reverse exponential part lives only in the boundary message
basis, not in the full internal root subtree.

If the separator width is $m$, and the exact bundle dimensions are

$$
N_0(m),\quad N_1(m),\quad N_2(m),\quad N_{\alpha\beta}(m),
$$

then the direct root reverse cost is governed by contractions in these bundle
dimensions and by the number of children $d$, not by re-expanding all internal
root states on the full subtree size $n$.

That is exactly the last algorithmic block still missing before the
rooted-tree reverse really becomes structural $2^n \to 2^m$.

---

## 11. Recommended Implementation Order

To keep the work exact and coherent, the gradient implementation should be done
in this order.

### Phase G1: Exact $h_{\mathrm{act}}$ and $g_{\mathrm{act}}$ gradients from the current forward kernel

Implement an exact-separator active-space gradient kernel that:

1. reuses the outer structure adjoints $W^H_{IJ}, W^S_{IJ}$;
2. calls the current rooted-tree exact value kernel;
3. accumulates:
   - one-electron gradient from root first-cofactor aggregates,
   - two-electron gradient from root degree-2 / mixed sectors.

This phase should already cover:

- rooted trees through [component_tree.cpp](./component_tree.cpp);
- one-leaf star through the specialized one-leaf exact path.

This phase does **not** yet require the full separator overlap backward pass.

### Phase G2: Reverse-mode message algebra for the overlap gradient

Add a reverse bundle with the same channel layout as the forward exact payload.

For the current rooted-tree Hamiltonian path, that means:

- reverse seeds at the root from Sections 7 and 10.4;
- reverse merge by transposing the exact bundle recurrence of Section 10;
- leaf pullback using the deleted-minor cofactor formula of Section 8.

This phase produces the exact active-space overlap gradient
$\partial \mathcal L / \partial S_{\mathrm{act}}$.

### Phase G3: Integrate into the existing orbital backprop stack

Once the exact-separator active-space gradient kernel returns:

- `active_orbital_overlap_gradient`,
- `active_one_electron_gradient`,
- `packed_active_two_electron_gradient`,

the existing AO / orbital backpropagation code can be reused.

So this phase is mostly wiring, not new mathematics.

---

## 12. Validation Plan

The new exact-separator gradient path should be validated against the current
determinant-pair active-space gradient implementation, not against finite
differences first.

For small exact cases:

1. compare exact-separator and determinant-pair active-space gradients for
   overlap-only;
2. compare them for one-electron-only;
3. compare them for two-electron-only;
4. compare them for full electronic Hamiltonian;
5. only after that use finite differences as a secondary sanity check.

The validation order matters because determinant-pair gradients already encode
the repository conventions for:

- upper-triangle structure weights;
- storage order of $h_{\mathrm{act}}$ and packed ERIs;
- sign conventions for same-spin exchange;
- swapped off-diagonal structure-pair symmetry handling.

---

## 13. What This Means For The Next Coding Step

The next coding step should **not** be another performance optimization pass.

The next coding step should be:

1. add an exact-separator active-space gradient interface;
2. implement exact $h_{\mathrm{act}}$ and $g_{\mathrm{act}}$ gradients from
   the current rooted-tree value kernel;
3. implement the overlap reverse pass on top of deleted-minor message adjoints.

Only after that should we revisit performance, because then forward and
backward can be optimized together around one shared message algebra.
