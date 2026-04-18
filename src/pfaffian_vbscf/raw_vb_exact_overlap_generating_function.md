# Exact Raw-VB Overlap As A Bond-Labeled Generating Function

## 1. Purpose

After the failure of the `single AGP = raw-VB exact` route, the next question
is:

- what is the correct exact algebraic object for one raw-VB structure pair;
- and where exactly does the internal combinatorics live.

This note answers that question for the overlap only.

The main point is:

- exact raw-VB overlap is naturally a **bond-labeled coefficient-extraction**
  problem;
- that exact object can be written in a structured determinant form, and also
  in a formal bond-labeled Pfaffian form;
- but it is **not** the same as the current single-kernel,
  single-variable ` [t^n] det(I + t M) ` closure.

So this note gives the mathematically honest replacement problem after the
single-AGP route was ruled out.

The follow-up complexity discussion is recorded in
`src/pfaffian_vbscf/raw_vb_overlap_coefficient_extraction_analysis.md`.

Relevant files:

- `src/pfaffian_vbscf/raw_vb_exact_single_agp_failure.md`
- `legacy_pair_generating_function.md`
- `scripts/example_closed_shell_vb_overlap.py`
- `scripts/verify_closed_shell_raw_vb_vs_oriented_agp.py`
- `src/vb/matrices/full_structure_expander.cpp`
- `src/tools/check_geminal_structure_overlap.cpp`

---

## 2. Scope And Phase Convention

There are two closely related objects that can be discussed:

1. the exact closed-shell structure basis used by the current
   `xmvb-cpp.exe` determinant oracle;
2. the more textbook singlet-bond convention written with explicit local
   minus signs.

For the present code-comparison problem, item 1 is the relevant exact target.

The current `xmvb-cpp` closed-shell determinant oracle is most faithfully
described by the "legacy" pair convention already summarized in
`legacy_pair_generating_function.md`:

- each pair contributes one alpha orbital and one beta orbital;
- one first chooses the alpha-side orbital for each pair;
- the beta-side orbital is then the complementary partner;
- the final determinant coefficient comes from canonical sorting of the alpha
  and beta strings.

That convention is the exact object reproduced by
`src/vb/matrices/full_structure_expander.cpp`.

The formulas below are written for that exact code-level target. The same
algorithmic conclusion also applies to the antisymmetric singlet convention:

- the local bond matrices change;
- but the need for one label per bond does not go away.

So the core conclusion is convention-independent:

- exact raw-VB keeps bond labels;
- the current single-variable AGP projector does not.

---

## 3. Exact Determinant-Sum Oracle

Consider one left closed-shell structure `X` and one right closed-shell
structure `Y`.

Let the left structure contain `n_X` active pairs:

```math
X = \{(p_1,q_1),\dots,(p_{n_X},q_{n_X})\},
```

and the right structure contain `n_Y` active pairs:

```math
Y = \{(r_1,s_1),\dots,(r_{n_Y},s_{n_Y})\}.
```

For overlap to be nonzero in the fixed-electron sector, we need

```math
n_X = n_Y = n.
```

For one covalent pair `(p_k,q_k)`, define one orientation bit

```math
\sigma_k \in \{0,1\},
```

with

```math
a_k^X(\sigma_k) =
\begin{cases}
p_k,& \sigma_k=0,\\
q_k,& \sigma_k=1,
\end{cases}
\qquad
b_k^X(\sigma_k) =
\begin{cases}
q_k,& \sigma_k=0,\\
p_k,& \sigma_k=1.
\end{cases}
```

Then one determinant term of `X` is specified by the two occupied strings

```math
A_X(\sigma) = \bigl(a_1^X(\sigma_1),\dots,a_n^X(\sigma_n)\bigr),
\qquad
B_X(\sigma) = \bigl(b_1^X(\sigma_1),\dots,b_n^X(\sigma_n)\bigr).
```

After canonical sorting of both strings, let the corresponding sign be

```math
c_X(\sigma).
```

Similarly, for the right structure:

```math
A_Y(\tau), \qquad B_Y(\tau), \qquad c_Y(\tau).
```

With `S` the active-space spatial overlap matrix, the exact determinant oracle
is

```math
S_{XY}
=
\sum_{\sigma,\tau \in \{0,1\}^n}
c_X(\sigma)c_Y(\tau)\,
\det S[A_X(\sigma),A_Y(\tau)]\,
\det S[B_X(\sigma),B_Y(\tau)].
```

This is the exact raw-VB overlap computed by determinant expansion.

The important fact is that the hidden combinatorics is entirely carried by the
bond-orientation labels `sigma` and `tau`.

---

## 4. Exact Bond-Labeled Determinant Generating Function

### 4.1 Covalent pairs

Introduce one formal variable `x_k` for each left bond and one formal variable
`y_l` for each right bond.

Define the left selector matrices

```math
U_X(\mathbf{x})_{k,m} = \delta_{m,p_k} + x_k\,\delta_{m,q_k},
```

```math
V_X(\mathbf{x})_{k,m} = \delta_{m,q_k} + x_k^{-1}\,\delta_{m,p_k},
```

and similarly for the right structure

```math
U_Y(\mathbf{y}), \qquad V_Y(\mathbf{y}).
```

Here:

- `U` chooses the alpha-side orbital from each bond;
- `V` chooses the complementary beta-side orbital from the same bond.

By Cauchy-Binet,

```math
\det\!\bigl(U_X(\mathbf{x}) S U_Y(\mathbf{y})^{T}\bigr)
=
\sum_{\sigma,\tau}
s_X^\alpha(\sigma)s_Y^\alpha(\tau)\,
\mathbf{x}^{\sigma}\mathbf{y}^{\tau}\,
\det S[A_X(\sigma),A_Y(\tau)],
```

with

```math
\mathbf{x}^{\sigma} = \prod_{k=1}^{n} x_k^{\sigma_k},
\qquad
\mathbf{y}^{\tau} = \prod_{l=1}^{n} y_l^{\tau_l}.
```

Similarly,

```math
\det\!\bigl(V_X(\mathbf{x}) S V_Y(\mathbf{y})^{T}\bigr)
=
\sum_{\sigma,\tau}
s_X^\beta(\sigma)s_Y^\beta(\tau)\,
\mathbf{x}^{-\sigma}\mathbf{y}^{-\tau}\,
\det S[B_X(\sigma),B_Y(\tau)].
```

Multiplying the two determinants and extracting the constant term forces the
same left orientation `sigma` and the same right orientation `tau` to be used
on the alpha and beta sides. Therefore

```math
S_{XY}
=
[\mathbf{x}^{0}\mathbf{y}^{0}]\,
\det\!\bigl(U_X(\mathbf{x}) S U_Y(\mathbf{y})^{T}\bigr)\,
\det\!\bigl(V_X(\mathbf{x}) S V_Y(\mathbf{y})^{T}\bigr).
```

This is an exact generating-function form of the determinant oracle.

### 4.2 Ionic pairs

If a bond is ionic, `(p_k,q_k) = (i,i)`, then there is no orientation freedom.
That bond contributes a fixed row

```math
e_i^{T}
```

to both the alpha selector and the beta selector, and no bond variable is
needed for that pair.

So ionic bonds do not change the structure of the exact formula:

- covalent bonds carry bond labels;
- ionic bonds are fixed selectors.

Therefore the true combinatorial burden is the number of covalent bonds whose
orientation must be projected exactly.

---

## 5. Formal Bond-Labeled Pfaffian Version

The determinant form above is already exact. But because our broader project
grew out of Pfaffian ideas, it is useful to also state the corresponding
bond-labeled Pfaffian object as a **formal packaging** of the same projection
idea.

For each left bond `mu`, define a `2M x 2M` antisymmetric bond matrix

```math
P_\mu^X,
```

and for each right bond `nu`, define

```math
Q_\nu^Y.
```

These are the single-bond spin-orbital pairing matrices. For example:

- an ionic bond `(i,i)` contributes the usual local alpha-beta doubly occupied
  block;
- a covalent bond `(i,j)` contributes the local two-orbital singlet-pair block
  with the convention chosen for the exact target basis.

Now introduce one formal label per bond:

```math
\eta_1,\dots,\eta_n,
\qquad
\xi_1,\dots,\xi_n.
```

The cleanest exact statement uses **commuting nilpotent labels**:

```math
\eta_\mu^2 = 0,
\qquad
\xi_\nu^2 = 0.
```

Define the bond-labeled left and right pairing matrices

```math
L_X(\eta) = \sum_{\mu=1}^{n} \eta_\mu P_\mu^X,
\qquad
R_Y(\xi) = \sum_{\nu=1}^{n} \xi_\nu Q_\nu^Y.
```

Let

```math
\Sigma = \operatorname{diag}(S,S)
```

be the spin-orbital metric built from the spatial overlap matrix.

Then the exact raw-VB overlap is formally represented by

```math
S_{XY}
=
[\eta_1\cdots\eta_n\,\xi_1\cdots\xi_n]\,
\operatorname{Pf}
\begin{pmatrix}
L_X(\eta) & \Sigma \\
-\Sigma^{T} & R_Y(\xi)
\end{pmatrix}.
```

This formula should be interpreted carefully:

- the Pfaffian generates all possible selections of bond matrices from the left
  and right;
- the nilpotent bond labels kill any repeated use of the same bond;
- the final coefficient extraction selects the term in which every bond is used
  exactly once.

This is the Pfaffian analogue of the determinant constant-term formula.

For the present project, the determinant constant-term expression remains the
most concrete exact formula. The Pfaffian form is best viewed as a compact
formal restatement of the same bond-labeled selection principle, consistent
with small-scale bond-resolved projection experiments such as
`scripts/example_closed_shell_vb_overlap.py`.

The essential point is that this object is **bond-labeled**. That is the
algebraic feature that the failed single-AGP route threw away.

---

## 6. Why The Current Single-Kernel Closure Fails

The current fast closed-shell AGP-style kernel is based on a single matrix
`M_{XY}` and a single scalar counting variable `t`, schematically

```math
[t^n]\det(I+tM_{XY}).
```

That construction counts only the **total number of selected pairs**.

Exact raw-VB requires a stronger condition:

- each bond of the left structure must be used exactly once;
- each bond of the right structure must be used exactly once.

That is not a one-variable counting problem. It is a labeled projection
problem.

The correct schematic object is therefore

```math
[x_1\cdots x_n\, y_1\cdots y_n]\,
\mathcal{G}_{XY}(x_1,\dots,x_n,y_1,\dots,y_n),
```

or, in the determinant version,

```math
[\mathbf{x}^{0}\mathbf{y}^{0}]\, \mathcal{D}_{XY}(\mathbf{x},\mathbf{y}),
```

or, in the Pfaffian version,

```math
[\eta_1\cdots\eta_n\,\xi_1\cdots\xi_n]\, \mathcal{P}_{XY}(\eta,\xi).
```

All three say the same thing:

- exact raw-VB overlap is a **multivariable projection**;
- the single-variable AGP projector is too coarse;
- identifying all bond labels with one scalar variable is precisely what
  reintroduces spurious repeated-bond terms.

So the single-kernel collapse fails not because Pfaffians are useless, but
because the exact projector lives in a larger labeled algebra.

---

## 7. Complexity Consequence

This reformulation is exact, but it is not automatically a low-scaling
algorithm.

What it does achieve is something more basic and more important:

- it isolates the true source of the combinatorics;
- it tells us exactly what any future exact acceleration must compress.

The generic worst-case difficulty is now visible:

- the overlap is not just a function of `n`, but of `n` distinct bond labels;
- exact projection means enforcing one-use-per-bond constraints;
- absent additional structure, that multivariable extraction can still carry an
  exponential burden in the number of active covalent bonds.

So after the single-AGP failure, the realistic question is no longer

- "can we keep the current `O(M^4)` kernel unchanged?"

but rather

- "can we compress the bond-labeled exact projector better than naive
  determinant expansion?"

That is a different mathematical problem.

---

## 8. Where Exact Acceleration Could Still Come From

Although the generic worst-case remains difficult, this bond-labeled form
suggests several more honest acceleration directions.

### 8.1 Interaction-graph dynamic programming

Define a bond-interaction graph whose vertices are bonds and whose edges measure
whether two bonds couple strongly through the orbital overlap metric `S`.

If that graph has small treewidth, pathwidth, or nearly disconnected
components, then one can hope to evaluate the bond-labeled projection by
dynamic programming instead of full `2^n` enumeration.

### 8.2 Block structure in the overlap metric

If the active-space overlap matrix can be permuted to a nearly block-diagonal
form, then the determinant factors in the exact generating function may split
into graph-local pieces plus controlled inter-block corrections.

This is close in spirit to the old union-graph intuition, but now expressed in
the correct exact object.

### 8.3 Low-rank inter-block correction

Even when exact factorization fails, the off-block couplings may be low rank or
screenable. Then the exact generating function could be reorganized into:

- block-local exact kernels;
- plus a low-rank correction series.

### 8.4 Selected-VB or screened structure spaces

If the full structure space is not required, then bond-labeled exact kernels
may still be practical in selected subspaces where:

- the number of active covalent bonds is moderate;
- or the interaction graph remains sparse.

This is a more realistic route to chemical applications than insisting on a
generic worst-case `O(M^4)` theorem immediately.

---

## 9. Immediate Implication For The Current Project

The current project should now distinguish three layers clearly.

### Layer A: determinant oracle

This is the current exact reference:

- raw structure definition;
- determinant expansion;
- determinant-pair overlap or Hamiltonian assembly.

### Layer B: current Pfaffian-VBSCF implementation

This is a different basis-state model:

- one structure surrogate per basis state;
- one pair matrix per state;
- one-variable coefficient extraction.

It may still be useful as its own variational model, but it is not raw-VB
exact structure by structure.

### Layer C: future exact bond-labeled theory

If raw-VB exactness remains the target, then the correct next object is the
bond-labeled generating function described in this note, not the current
single-kernel AGP closure.

This means:

- overlap should be rederived in bond-labeled form first;
- one-body and two-body matrix elements should then be derived as source
  derivatives of that same bond-labeled object;
- only after that should one ask whether any reusable fast substructure
  survives.

---

## 10. Bottom Line

There is still "hope", but the nature of the problem has changed.

What survives:

- exact raw-VB overlap does have a compact algebraic description;
- it can be written as a structured coefficient-extraction problem rather than
  only as an explicit determinant double sum;
- this gives a principled target for future exact acceleration.

What does not survive:

- one raw structure equals one single AGP;
- exact raw-VB overlap collapses to the current single-variable
  ` [t^n] det(I+tM) ` kernel;
- exactness can be recovered by only changing the current basis encoding.

So the correct next step is not to keep forcing the old fast kernel, but to
study the exact bond-labeled generating object and ask whether its projector
can be compressed by graph structure, low-rank structure, or selected-space
constraints.
