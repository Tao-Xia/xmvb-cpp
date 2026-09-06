# Latest Exact Support-Sparse Matrix-Element Formulas

## 1. Scope

This note summarizes the **current exact unique-spin implementation with
support-sparse local contractions**. The scope is:

$$
\text{exact integrals} \quad (\texttt{int=libcint}),
$$

with the following production paths:

$$
\texttt{full\_structure\_builder.cpp}
$$

for forward structure-basis matrix elements, and

$$
\texttt{same\_spin\_matrix\_backward.cpp}, \qquad
\texttt{opposite\_spin\_matrix\_backward.cpp}
$$

for backward contractions on selected states.

This document does **not** discuss RI approximations. The key point is that the
new sparsity is **not** a sparsity of the global same-spin Gram matrices.
Instead, it comes from the fact that each structure or selected state only
occupies a small support in the unique-spin product space.

---

## 2. Notation

Let

$$
N_\alpha = \text{number of unique alpha strings}, \qquad
N_\beta = \text{number of unique beta strings}.
$$

For each structure \( I \), define its coefficient matrix in the unique-spin
product basis:

$$
\mathbf{C}_I \in \mathbb{R}^{N_\alpha \times N_\beta}.
$$

The entry \( (\mathbf{C}_I)_{ab} \) is the total coefficient of all full
determinants belonging to structure \( I \) whose unique alpha string is
\( a \) and unique beta string is \( b \).

Define the support sets

$$
A_I = \left\{ a \,\middle|\, \exists b,\; (\mathbf{C}_I)_{ab} \neq 0 \right\},
\qquad
B_I = \left\{ b \,\middle|\, \exists a,\; (\mathbf{C}_I)_{ab} \neq 0 \right\}.
$$

Their sizes are

$$
r_I = |A_I|, \qquad c_I = |B_I|.
$$

The corresponding local coefficient block is

$$
\mathbf{C}_I^{\mathrm{loc}}
=
\mathbf{C}_I[A_I, B_I]
\in \mathbb{R}^{r_I \times c_I}.
$$

For a matrix defined on the unique alpha space, we write

$$
\mathbf{M}^\alpha[I,J]
=
\mathbf{M}^\alpha[A_I, A_J]
\in \mathbb{R}^{r_I \times r_J}.
$$

For a matrix defined on the unique beta space, we write

$$
\mathbf{M}^\beta[I,J]
=
\mathbf{M}^\beta[B_I, B_J]
\in \mathbb{R}^{c_I \times c_J}.
$$

We use the Frobenius inner product

$$
\left\langle \mathbf{X}, \mathbf{Y} \right\rangle_{\mathrm{F}}
=
\operatorname{Tr}\!\left( \mathbf{X}^{\mathrm{T}} \mathbf{Y} \right).
$$

---

## 3. Global Unique-Spin Kernels

The current exact same-spin cache builds the following dense symmetric matrices:

$$
\mathbf{S}^\alpha \in \mathbb{R}^{N_\alpha \times N_\alpha}, \qquad
\mathbf{S}^\beta \in \mathbb{R}^{N_\beta \times N_\beta},
$$

$$
\mathbf{h}^\alpha \in \mathbb{R}^{N_\alpha \times N_\alpha}, \qquad
\mathbf{h}^\beta \in \mathbb{R}^{N_\beta \times N_\beta},
$$

and the same-spin total matrices

$$
\mathbf{T}^\alpha \in \mathbb{R}^{N_\alpha \times N_\alpha}, \qquad
\mathbf{T}^\beta \in \mathbb{R}^{N_\beta \times N_\beta}.
$$

Here,

$$
\mathbf{T}^\sigma
=
\mathbf{h}^\sigma + \mathbf{g}^{\sigma\sigma},
\qquad
\sigma \in \{ \alpha, \beta \},
$$

where \( \mathbf{g}^{\sigma\sigma} \) denotes the same-spin two-electron part in
the unique-spin basis.

For the exact opposite-spin channel, the implementation uses packed-pair
channels. For each packed pair index \( P \), we define

$$
\mathbf{U}_P^\alpha \in \mathbb{R}^{N_\alpha \times N_\alpha}, \qquad
\widetilde{\mathbf{U}}_P^\beta \in \mathbb{R}^{N_\beta \times N_\beta}.
$$

In the current code path:

$$
\mathbf{U}_P^\alpha
$$

is stored as a sparse matrix, while

$$
\widetilde{\mathbf{U}}_P^\beta
$$

is stored as a dense projected image matrix.

The important point is:

$$
\mathbf{S}^\alpha,\; \mathbf{S}^\beta,\; \mathbf{h}^\alpha,\; \mathbf{h}^\beta,\;
\mathbf{T}^\alpha,\; \mathbf{T}^\beta
$$

are typically **dense**, whereas

$$
\mathbf{C}_I
$$

is usually highly sparse in the unique-spin product space.

Therefore the natural optimization is to keep the global kernels dense, but to
restrict each structure-pair contraction to the local supports \( A_I, B_I \).

---

## 4. Forward Structure-Basis Matrix Elements

### 4.1 Overlap

The exact global matrix-form formula is

$$
S_{IJ}
=
\left\langle
\mathbf{C}_I,\;
\mathbf{S}^\alpha \mathbf{C}_J (\mathbf{S}^\beta)^{\mathrm{T}}
\right\rangle_{\mathrm{F}}.
$$

In index form,

$$
S_{IJ}
=
\sum_{a=1}^{N_\alpha}
\sum_{a'=1}^{N_\alpha}
\sum_{b=1}^{N_\beta}
\sum_{b'=1}^{N_\beta}
(\mathbf{C}_I)_{ab}\,
S^\alpha_{aa'}\,
(\mathbf{C}_J)_{a'b'}\,
S^\beta_{bb'}.
$$

Because \( \mathbf{C}_I \) and \( \mathbf{C}_J \) are nonzero only on their
supports, this contracts exactly to

$$
S_{IJ}
=
\left\langle
\mathbf{C}_I^{\mathrm{loc}},\;
\mathbf{S}^\alpha[I,J]\,
\mathbf{C}_J^{\mathrm{loc}}\,
\left( \mathbf{S}^\beta[I,J] \right)^{\mathrm{T}}
\right\rangle_{\mathrm{F}}.
$$

This is the exact formula used by the support-sparse forward path.

### 4.2 One-Electron Hamiltonian

The exact global formula is

$$
H_{IJ}^{1\mathrm{e}}
=
\left\langle
\mathbf{C}_I,\;
\mathbf{h}^\alpha \mathbf{C}_J (\mathbf{S}^\beta)^{\mathrm{T}}
+
\mathbf{S}^\alpha \mathbf{C}_J (\mathbf{h}^\beta)^{\mathrm{T}}
\right\rangle_{\mathrm{F}}.
$$

Its support-restricted exact form is

$$
H_{IJ}^{1\mathrm{e}}
=
\left\langle
\mathbf{C}_I^{\mathrm{loc}},\;
\mathbf{h}^\alpha[I,J]\,
\mathbf{C}_J^{\mathrm{loc}}\,
\left( \mathbf{S}^\beta[I,J] \right)^{\mathrm{T}}
+
\mathbf{S}^\alpha[I,J]\,
\mathbf{C}_J^{\mathrm{loc}}\,
\left( \mathbf{h}^\beta[I,J] \right)^{\mathrm{T}}
\right\rangle_{\mathrm{F}}.
$$

### 4.3 Same-Spin Total Contribution

The exact same-spin total contribution is

$$
H_{IJ}^{\mathrm{ss}}
=
\left\langle
\mathbf{C}_I,\;
\mathbf{T}^\alpha \mathbf{C}_J (\mathbf{S}^\beta)^{\mathrm{T}}
+
\mathbf{S}^\alpha \mathbf{C}_J (\mathbf{T}^\beta)^{\mathrm{T}}
\right\rangle_{\mathrm{F}}.
$$

After restricting to the structure supports, we obtain

$$
H_{IJ}^{\mathrm{ss}}
=
\left\langle
\mathbf{C}_I^{\mathrm{loc}},\;
\mathbf{T}^\alpha[I,J]\,
\mathbf{C}_J^{\mathrm{loc}}\,
\left( \mathbf{S}^\beta[I,J] \right)^{\mathrm{T}}
+
\mathbf{S}^\alpha[I,J]\,
\mathbf{C}_J^{\mathrm{loc}}\,
\left( \mathbf{T}^\beta[I,J] \right)^{\mathrm{T}}
\right\rangle_{\mathrm{F}}.
$$

This is the precise support-sparse matrix-element formula for the latest
same-spin exact forward implementation.

### 4.4 Opposite-Spin Contribution

For the exact opposite-spin channel, the global matrix-form expression is

$$
H_{IJ}^{\alpha\beta}
=
\sum_{P=1}^{N_{\mathrm{pair}}}
\left\langle
\mathbf{C}_I,\;
\mathbf{U}_P^\alpha \mathbf{C}_J
\left( \widetilde{\mathbf{U}}_P^\beta \right)^{\mathrm{T}}
\right\rangle_{\mathrm{F}}.
$$

In index form,

$$
H_{IJ}^{\alpha\beta}
=
\sum_{P=1}^{N_{\mathrm{pair}}}
\sum_{a=1}^{N_\alpha}
\sum_{a'=1}^{N_\alpha}
\sum_{b=1}^{N_\beta}
\sum_{b'=1}^{N_\beta}
(\mathbf{C}_I)_{ab}\,
U_{P,aa'}^\alpha\,
(\mathbf{C}_J)_{a'b'}\,
\widetilde{U}_{P,bb'}^\beta.
$$

The exact support-sparse local formula is

$$
H_{IJ}^{\alpha\beta}
=
\sum_{P=1}^{N_{\mathrm{pair}}}
\left\langle
\mathbf{C}_I^{\mathrm{loc}},\;
\mathbf{U}_P^\alpha[I,J]\,
\mathbf{C}_J^{\mathrm{loc}}\,
\left( \widetilde{\mathbf{U}}_P^\beta[I,J] \right)^{\mathrm{T}}
\right\rangle_{\mathrm{F}}.
$$

If we define the active channel set

$$
\mathcal{P}_{IJ}
=
\left\{
P \,\middle|\,
\mathbf{U}_P^\alpha[I,J] \neq 0
\right\},
$$

then the current implementation is effectively

$$
H_{IJ}^{\alpha\beta}
=
\sum_{P \in \mathcal{P}_{IJ}}
\left\langle
\mathbf{C}_I^{\mathrm{loc}},\;
\mathbf{U}_P^\alpha[I,J]\,
\mathbf{C}_J^{\mathrm{loc}}\,
\left( \widetilde{\mathbf{U}}_P^\beta[I,J] \right)^{\mathrm{T}}
\right\rangle_{\mathrm{F}}.
$$

That is, the sparse alpha channel is screened first; the beta projected-image
subblock is gathered only after the alpha subblock is confirmed to have support.

### 4.5 Total Hamiltonian

Combining all parts, the full exact structure-basis Hamiltonian element is

$$
H_{IJ}
=
\left\langle
\mathbf{C}_I^{\mathrm{loc}},\;
\mathbf{T}^\alpha[I,J]\,
\mathbf{C}_J^{\mathrm{loc}}\,
\left( \mathbf{S}^\beta[I,J] \right)^{\mathrm{T}}
+
\mathbf{S}^\alpha[I,J]\,
\mathbf{C}_J^{\mathrm{loc}}\,
\left( \mathbf{T}^\beta[I,J] \right)^{\mathrm{T}}
\right\rangle_{\mathrm{F}}
+
\sum_{P \in \mathcal{P}_{IJ}}
\left\langle
\mathbf{C}_I^{\mathrm{loc}},\;
\mathbf{U}_P^\alpha[I,J]\,
\mathbf{C}_J^{\mathrm{loc}}\,
\left( \widetilde{\mathbf{U}}_P^\beta[I,J] \right)^{\mathrm{T}}
\right\rangle_{\mathrm{F}}.
$$

The overlap element is still

$$
S_{IJ}
=
\left\langle
\mathbf{C}_I^{\mathrm{loc}},\;
\mathbf{S}^\alpha[I,J]\,
\mathbf{C}_J^{\mathrm{loc}}\,
\left( \mathbf{S}^\beta[I,J] \right)^{\mathrm{T}}
\right\rangle_{\mathrm{F}}.
$$

These two formulas are the central support-sparse matrix-element formulas for
the current exact unique-spin code path.

---

## 5. Selected-State Support-Sparse Backward Formulas

The latest implementation also uses trimmed local supports in backward
contractions. Let selected state \( n \) have weight \( w_n \), energy
\( E_n \), support sets \( A_n, B_n \), and local coefficient matrix

$$
\mathbf{C}_n^{\mathrm{loc}}
=
\mathbf{C}_n[A_n, B_n].
$$

### 5.1 Same-Spin Backward Weight Matrices

Define the local subblocks

$$
\mathbf{S}_n^\alpha = \mathbf{S}^\alpha[A_n, A_n], \qquad
\mathbf{S}_n^\beta = \mathbf{S}^\beta[B_n, B_n],
$$

$$
\mathbf{T}_{n,\mathrm{reg}}^\alpha = \mathbf{T}_{\mathrm{reg}}^\alpha[A_n, A_n], \qquad
\mathbf{T}_{n,\mathrm{reg}}^\beta = \mathbf{T}_{\mathrm{reg}}^\beta[B_n, B_n].
$$

Then the support-sparse exact same-spin backward assembles

$$
\mathbf{W}_{\alpha,H}
=
\sum_n
w_n\,
\mathbf{C}_n^{\mathrm{loc}}\,
\mathbf{S}_n^\beta\,
\left( \mathbf{C}_n^{\mathrm{loc}} \right)^{\mathrm{T}},
$$

$$
\mathbf{W}_{\alpha,S}
=
\sum_n
\left( - w_n E_n \right)\,
\mathbf{C}_n^{\mathrm{loc}}\,
\mathbf{S}_n^\beta\,
\left( \mathbf{C}_n^{\mathrm{loc}} \right)^{\mathrm{T}},
$$

$$
\mathbf{T}_{\alpha,\mathrm{partner}}
=
\sum_n
w_n\,
\mathbf{C}_n^{\mathrm{loc}}\,
\mathbf{T}_{n,\mathrm{reg}}^\beta\,
\left( \mathbf{C}_n^{\mathrm{loc}} \right)^{\mathrm{T}}.
$$

Similarly,

$$
\mathbf{W}_{\beta,H}
=
\sum_n
w_n\,
\left( \mathbf{C}_n^{\mathrm{loc}} \right)^{\mathrm{T}}
\,
\mathbf{S}_n^\alpha\,
\mathbf{C}_n^{\mathrm{loc}},
$$

$$
\mathbf{W}_{\beta,S}
=
\sum_n
\left( - w_n E_n \right)\,
\left( \mathbf{C}_n^{\mathrm{loc}} \right)^{\mathrm{T}}
\,
\mathbf{S}_n^\alpha\,
\mathbf{C}_n^{\mathrm{loc}},
$$

$$
\mathbf{T}_{\beta,\mathrm{partner}}
=
\sum_n
w_n\,
\left( \mathbf{C}_n^{\mathrm{loc}} \right)^{\mathrm{T}}
\,
\mathbf{T}_{n,\mathrm{reg}}^\alpha\,
\mathbf{C}_n^{\mathrm{loc}}.
$$

Each local image is then scattered back into the full unique alpha or beta
spaces.

### 5.2 Opposite-Spin Pair Accumulation

For a beta-channel pair matrix

$$
\mathbf{B}^\beta \in \mathbb{R}^{N_\beta \times N_\beta},
$$

the support-sparse alpha accumulation is

$$
\mathbf{M}_\alpha
=
\sum_n
w_n\,
\mathbf{C}_n^{\mathrm{loc}}\,
\mathbf{B}^\beta[B_n, B_n]\,
\left( \mathbf{C}_n^{\mathrm{loc}} \right)^{\mathrm{T}}.
$$

For an alpha-channel pair matrix

$$
\mathbf{B}^\alpha \in \mathbb{R}^{N_\alpha \times N_\alpha},
$$

the support-sparse beta accumulation is

$$
\mathbf{M}_\beta
=
\sum_n
w_n\,
\left( \mathbf{C}_n^{\mathrm{loc}} \right)^{\mathrm{T}}
\,
\mathbf{B}^\alpha[A_n, A_n]\,
\mathbf{C}_n^{\mathrm{loc}}.
$$

These are the exact local formulas used in the current backward implementation.

---

## 6. Cost Model

### 6.1 Forward Structure-Pair Contraction

For a fixed structure pair \( (I,J) \), the dense global matrix-form cost for a
single same-spin-like kernel is, more explicitly,

$$
O\!\left( N_\alpha^2 N_\beta + N_\alpha N_\beta^2 \right)
$$

if one writes the two matrix multiplications directly on the full spaces.

With support restriction, the local cost becomes

$$
O\!\left( r_I r_J c_J + r_I c_J c_I \right).
$$

The first term corresponds to

$$
\mathbf{K}^\alpha[I,J] \mathbf{C}_J^{\mathrm{loc}},
$$

and the second term corresponds to multiplication by the beta-side local block
and the final Frobenius contraction.

For the opposite-spin forward term, the local cost is

$$
O\!\left(
|\mathcal{P}_{IJ}|\,
\left( r_I r_J c_J + r_I c_J c_I \right)
\right).
$$

### 6.2 Selected-State Backward

For a selected state \( n \), the dense global cost scales like

$$
O\!\left( N_\alpha^2 N_\beta + N_\alpha N_\beta^2 \right),
$$

whereas the support-sparse local contraction scales like

$$
O\!\left( r_n^2 c_n + r_n c_n^2 \right).
$$

Therefore the benefit appears when

$$
r_n \ll N_\alpha, \qquad c_n \ll N_\beta.
$$

This is exactly the regime encountered when each structure or selected state
touches only a small subset of unique alpha and beta strings.

---

## 7. Implementation Notes

The current implementation follows the following principles.

First, the global same-spin matrices remain dense and symmetric:

$$
\mathbf{S}^\alpha,\;
\mathbf{S}^\beta,\;
\mathbf{h}^\alpha,\;
\mathbf{h}^\beta,\;
\mathbf{T}^\alpha,\;
\mathbf{T}^\beta.
$$

This is the correct design because these objects are Gram- or Hamiltonian-like
matrices on the unique-spin basis and are generally not sparse.

Second, the sparse object is the local coefficient representation:

$$
(A_I, B_I, \mathbf{C}_I^{\mathrm{loc}}).
$$

Third, the support-sparse forward path and the support-sparse backward path are
both exact. They do **not** change the mathematics; they only replace global
dense contractions by local dense contractions on trimmed supports.

Finally, the exact opposite-spin forward path already uses one level of channel
screening:

$$
\mathbf{U}_P^\alpha[I,J] = 0
\quad \Longrightarrow \quad
\text{skip channel } P \text{ for structure pair } (I,J).
$$

This is why the current support-sparse exact path is more than a cosmetic
rewrite: it preserves the unique-spin separation while additionally exploiting
the coefficient-block sparsity of each structure and each selected state.
