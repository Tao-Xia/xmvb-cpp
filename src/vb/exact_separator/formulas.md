# Exact Separator Exact Formulas

## 1. Purpose

This note collects the current exact formulas used by the exact-separator
implementation for:

1. overlap;
2. one-electron matrix elements;
3. two-electron matrix elements.

The goal is not to propose a new approximation, but to write the existing
exact algebra in one place with consistent notation. In particular:

- determinant-pair formulas are written first;
- then the one-leaf separator recurrence is derived from them;
- for same-spin two-electron terms, both the exact target formula and the
  current exact-but-slow deleted-overlap recurrence are recorded explicitly.

This file uses the same row/column convention as the C++ implementation:

- row index = ket/right occupied orbital;
- column index = bra/left occupied orbital.

## 2. Notation

For one fixed bra determinant term and one fixed ket determinant term, let

$$
S^\alpha,\qquad S^\beta
$$

be the alpha and beta overlap matrices. Their determinants are

$$
O^\alpha = \det S^\alpha,\qquad
O^\beta = \det S^\beta.
$$

The full determinant-pair overlap is

$$
O = O^\alpha O^\beta.
$$

For one spin block $S^\sigma$, define:

- the first cofactor
  $$
  C^\sigma_{rc}
  =
  (-1)^{r+c}\det S^\sigma[\widehat r,\widehat c];
  $$
- the second cofactor
  $$
  \Gamma^\sigma_{r_1 r_2,\, c_1 c_2}
  =
  (-1)^{r_1+r_2+c_1+c_2}
  \det S^\sigma[\widehat{r_1},\widehat{r_2};\widehat{c_1},\widehat{c_2}],
  $$
  where $r_1 < r_2$ and $c_1 < c_2$.

These formulas are exact even when the overlap matrix is singular. In the
regular case they can be rewritten using the inverse:

$$
C^\sigma_{rc} = O^\sigma (S^\sigma)^{-1}_{cr},
$$

and

$$
\Gamma^\sigma_{r_1 r_2,\, c_1 c_2}
=
O^\sigma
\Big(
(S^\sigma)^{-1}_{c_1 r_1}(S^\sigma)^{-1}_{c_2 r_2}
-
(S^\sigma)^{-1}_{c_2 r_1}(S^\sigma)^{-1}_{c_1 r_2}
\Big).
$$

Let $h_{rc}$ be the active-space one-electron matrix, and let

$$
(r_1 c_1 \mid r_2 c_2)
$$

be the packed two-electron integral in the same active-orbital basis.

For compactness define the contractions

$$
\langle h, C \rangle
=
\sum_{r,c} h_{rc} C_{rc},
$$

$$
\mathcal B(A,B)
=
\sum_{r_\alpha,c_\alpha}
\sum_{r_\beta,c_\beta}
(r_\beta c_\beta \mid r_\alpha c_\alpha)
A_{r_\alpha c_\alpha}
B_{r_\beta c_\beta},
$$

and

$$
\mathcal A(\Gamma)
=
\sum_{r_1<r_2}
\sum_{c_1<c_2}
\Big[
(r_1 c_1 \mid r_2 c_2)
-
(r_1 c_2 \mid r_2 c_1)
\Big]
\Gamma_{r_1 r_2,\, c_1 c_2}.
$$

## 3. Determinant-Pair Exact Formulas

### 3.1 Overlap

The exact determinant-pair overlap factorizes by spin:

$$
O = O^\alpha O^\beta.
$$

This is the quantity accumulated by the exact determinant-pair baseline and is
also the quantity that the separator recurrence must reconstruct exactly.

### 3.2 One-Electron Matrix Element

For a spin-independent one-electron operator

$$
\hat H^{(1)} = \sum_{r,c} h_{rc}\,
\big(
\hat a^\dagger_{r\alpha}\hat a_{c\alpha}
+
\hat a^\dagger_{r\beta}\hat a_{c\beta}
\big),
$$

the exact determinant-pair matrix element is

$$
H^{(1)}
=
\langle h, C^\alpha \rangle\, O^\beta
+
O^\alpha\, \langle h, C^\beta \rangle.
$$

Written out explicitly,

$$
H^{(1)}
=
\sum_{r,c} h_{rc}\, C^\alpha_{rc}\, O^\beta
+
\sum_{r,c} h_{rc}\, O^\alpha\, C^\beta_{rc}.
$$

So the one-electron problem only needs:

- one overlap scalar for each spin;
- one first-cofactor matrix for each spin.

### 3.3 Two-Electron Matrix Element

The exact determinant-pair pure two-electron matrix element splits naturally
into three channels:

$$
H^{(2)}
=
H_{\alpha\alpha}^{(2)}
+
H_{\beta\beta}^{(2)}
+
H_{\alpha\beta}^{(2)}.
$$

#### Opposite-Spin Channel

The opposite-spin channel is bilinear in first cofactors:

$$
H_{\alpha\beta}^{(2)}
=
\mathcal B(C^\alpha, C^\beta)
=
\sum_{r_\alpha,c_\alpha}
\sum_{r_\beta,c_\beta}
(r_\beta c_\beta \mid r_\alpha c_\alpha)\,
C^\alpha_{r_\alpha c_\alpha}\,
C^\beta_{r_\beta c_\beta}.
$$

This formula is exact for both regular and singular overlaps because cofactors
are defined through deleted minors rather than through the inverse.

#### Same-Spin Channel

For alpha-alpha,

$$
H_{\alpha\alpha}^{(2)}
=
O^\beta\, \mathcal A(\Gamma^\alpha)
=
O^\beta
\sum_{r_1<r_2}
\sum_{c_1<c_2}
\Big[
(r_1 c_1 \mid r_2 c_2)
-
(r_1 c_2 \mid r_2 c_1)
\Big]
\Gamma^\alpha_{r_1 r_2,\, c_1 c_2}.
$$

Similarly,

$$
H_{\beta\beta}^{(2)}
=
O^\alpha\, \mathcal A(\Gamma^\beta).
$$

So the exact two-electron problem needs:

- overlap $O^\sigma$;
- first cofactor $C^\sigma$;
- second cofactor $\Gamma^\sigma$.

## 4. One-Leaf Separator Setup

We now specialize to the current exact one-leaf separator recurrence used for a
root component plus one leaf component.

For one spin $\sigma \in \{\alpha,\beta\}$, let

$$
R_L^\sigma,\quad R_R^\sigma
$$

be the bra/ket occupied-orbital lists from the root term, and let

$$
L_L^\sigma,\quad L_R^\sigma
$$

be the corresponding lists from the leaf term.

For a mask state $m^\sigma$, select:

- a row subset $I^\sigma \subseteq R_R^\sigma$;
- a column subset $J^\sigma \subseteq R_L^\sigma$.

Their complements are

$$
\bar I^\sigma,\qquad \bar J^\sigma.
$$

The frontier lists are the concatenations

$$
F_R^\sigma = I^\sigma \cup L_R^\sigma,\qquad
F_L^\sigma = J^\sigma \cup L_L^\sigma,
$$

and the root-complement lists are

$$
Q_R^\sigma = \bar I^\sigma,\qquad
Q_L^\sigma = \bar J^\sigma.
$$

For this mask, define:

1. the frontier matrix
   $$
   \widetilde S_F^\sigma(m^\sigma),
   $$
   built from $F_R^\sigma$ and $F_L^\sigma$, with the
   selected-root/selected-root top-left block zeroed;
2. the root-complement matrix
   $$
   S_R^\sigma(\bar m^\sigma),
   $$
   built from $Q_R^\sigma$ and $Q_L^\sigma$.

The block-order sign induced by moving the selected root orbitals in front of
the remainder root orbitals is denoted by

$$
\eta^\sigma(m^\sigma) \in \{\pm 1\}.
$$

For a fixed orientation-term quadruple, the separator recurrence sums over all
valid alpha and beta mask states.

## 5. Overlap Exact Formula

### 5.1 One Mask

For one spin channel and one mask,

$$
O^\sigma_{\mathrm{full}}(m^\sigma)
=
\eta^\sigma(m^\sigma)\,
O_F^\sigma(m^\sigma)\,
O_R^\sigma(\bar m^\sigma),
$$

where

$$
O_F^\sigma = \det \widetilde S_F^\sigma,\qquad
O_R^\sigma = \det S_R^\sigma.
$$

This is exact because after the block-order permutation the selected-root and
root-remainder rows/columns separate into a frontier block and a root block,
and the full determinant becomes the signed product of those two local
determinants.

### 5.2 One Spin Aggregate

Summing over all valid mask states gives the exact spin overlap:

$$
\bar O^\sigma
=
\sum_{m^\sigma}
\eta^\sigma(m^\sigma)\,
O_F^\sigma(m^\sigma)\,
O_R^\sigma(\bar m^\sigma).
$$

### 5.3 Fixed Orientation-Term Quadruple

For a fixed root-left/root-right/leaf-left/leaf-right term quadruple, the full
overlap is

$$
O_{\mathrm{quad}}
=
\bar O^\alpha \bar O^\beta.
$$

## 6. One-Electron Exact Formula

### 6.1 Local First-Order Payload

For one local spin block, the exact first-order payload consists of:

- a closed overlap scalar $O$;
- a row-open vector $u$, made of signed $(1,0)$ deleted minors;
- a col-open vector $v$, made of signed $(0,1)$ deleted minors;
- a first-cofactor matrix $C$, made of signed $(1,1)$ deleted minors.

This is the same local object already constructed by the current
`PartialSpinPayload` builder.

### 6.2 One-Mask Full First-Cofactor Merge

Let

$$
f_r^\sigma,\; f_c^\sigma
$$

be the frontier row/column counts, and let

$$
r_r^\sigma,\; r_c^\sigma
$$

be the root-complement row/column counts.

For one mask, the exact first cofactor of the full spin overlap matrix is

$$
\begin{aligned}
C_{\mathrm{full}}^\sigma(m^\sigma)
=\;&
O_R^\sigma\, C_F^\sigma
+
(-1)^{f_r^\sigma+f_c^\sigma}\, O_F^\sigma\, C_R^\sigma \\
&+
(-1)^{f_r^\sigma+f_c^\sigma+r_c^\sigma}\,
u_F^\sigma (v_R^\sigma)^{\mathsf T}
+
(-1)^{f_r^\sigma+f_c^\sigma+r_r^\sigma}\,
u_R^\sigma (v_F^\sigma)^{\mathsf T}.
\end{aligned}
$$

This identity comes from classifying a full cofactor entry by where its
deleted row and deleted column land:

1. both in the frontier block: $O_R C_F$;
2. both in the root block: $O_F C_R$, with the block-order parity;
3. one in each block: product of a row-open minor and a col-open minor.

### 6.3 One Spin Aggregate

Summing the exact one-mask full cofactor over mask states gives

$$
\bar C^\sigma
=
\sum_{m^\sigma}
\eta^\sigma(m^\sigma)\,
C_{\mathrm{full}}^\sigma(m^\sigma).
$$

Together with

$$
\bar O^\sigma
=
\sum_{m^\sigma}
\eta^\sigma(m^\sigma)\,
O_F^\sigma O_R^\sigma,
$$

this is the complete exact spin-local data needed by the one-electron
recurrence.

### 6.4 Fixed Orientation-Term Quadruple

For a fixed orientation-term quadruple,

$$
H^{(1)}_{\mathrm{quad}}
=
\langle h, \bar C^\alpha \rangle\, \bar O^\beta
+
\bar O^\alpha\, \langle h, \bar C^\beta \rangle.
$$

So the current one-electron exact separator recurrence is already genuinely
reduced: it only needs $\bar O^\sigma$ and $\bar C^\sigma$, not any
determinant-pair loop in the final contraction.

## 7. Two-Electron Exact Formula

### 7.1 Opposite-Spin Channel

The opposite-spin channel is the direct analogue of the determinant-pair
formula, but with the exact mask-summed first cofactors:

$$
H_{\alpha\beta,\mathrm{quad}}^{(2)}
=
\mathcal B(\bar C^\alpha, \bar C^\beta).
$$

Written out,

$$
H_{\alpha\beta,\mathrm{quad}}^{(2)}
=
\sum_{r_\alpha,c_\alpha}
\sum_{r_\beta,c_\beta}
(r_\beta c_\beta \mid r_\alpha c_\alpha)\,
\bar C^\alpha_{r_\alpha c_\alpha}\,
\bar C^\beta_{r_\beta c_\beta}.
$$

This is why the current opposite-spin separator path is already structurally
sound and relatively fast: it only needs the exact first-order payload.

### 7.2 Same-Spin Exact Target

The exact same-spin target should have the parallel form

$$
H_{\alpha\alpha,\mathrm{quad}}^{(2)}
=
\mathcal A(\bar \Gamma^\alpha)\, \bar O^\beta,
$$

$$
H_{\beta\beta,\mathrm{quad}}^{(2)}
=
\mathcal A(\bar \Gamma^\beta)\, \bar O^\alpha,
$$

where

$$
\bar \Gamma^\sigma
=
\sum_{m^\sigma}
\eta^\sigma(m^\sigma)\,
\Gamma^\sigma_{\mathrm{full}}(m^\sigma).
$$

So the true missing object is not another scalar, but the exact mask-summed
second-cofactor tensor $\bar \Gamma^\sigma$.

### 7.3 Local Degree-2 Deleted-Minor Payload

To derive $\Gamma^\sigma_{\mathrm{full}}(m^\sigma)$ exactly, the local
first-order payload

- closed overlap $O$,
- row-open minors $u$,
- col-open minors $v$,
- first cofactors $C$,

must be extended to degree $2$.

For one local block $X$ with row count $n_r$ and column count $n_c$, define:

- square block, $n_r = n_c$:
  $$
  O_X = \det X,
  $$
  $$
  C_X(r,c)
  =
  (-1)^{r+c}\det X[\widehat r,\widehat c],
  $$
  $$
  \Gamma_X(r_1,r_2;c_1,c_2)
  =
  (-1)^{r_1+r_2+c_1+c_2}
  \det X[\widehat{r_1},\widehat{r_2};\widehat{c_1},\widehat{c_2}] .
  $$
- row-open block, $n_r = n_c + 1$:
  $$
  u_X(r) = (-1)^r \det X[\widehat r,:],
  $$
  $$
  P_X(r_1,r_2;c)
  =
  (-1)^{r_1+r_2+c}
  \det X[\widehat{r_1},\widehat{r_2};\widehat c] .
  $$
- col-open block, $n_c = n_r + 1$:
  $$
  v_X(c) = (-1)^c \det X[:,\widehat c],
  $$
  $$
  Q_X(r;c_1,c_2)
  =
  (-1)^{r+c_1+c_2}
  \det X[\widehat r;\widehat{c_1},\widehat{c_2}] .
  $$
- row excess two, $n_r = n_c + 2$:
  $$
  U_X(r_1,r_2)
  =
  (-1)^{r_1+r_2}\det X[\widehat{r_1},\widehat{r_2};:] .
  $$
- col excess two, $n_c = n_r + 2$:
  $$
  V_X(c_1,c_2)
  =
  (-1)^{c_1+c_2}\det X[:;\widehat{c_1},\widehat{c_2}] .
  $$

All indices above are local block-order indices. These are the exact degree-2
deleted minors needed for the one-mask same-spin merge. In the bracket
notation, deleted rows are listed before the semicolon and deleted columns are
listed after it; a `:` means "keep all indices on this side".

### 7.4 One-Mask Full Second-Cofactor Derivation

For one mask $m^\sigma$, let the block-order full spin matrix be

$$
S^\sigma_{\mathrm{blk}}(m^\sigma)
=
\begin{pmatrix}
\widetilde S_F^\sigma(m^\sigma) & 0 \\
0 & S_R^\sigma(\bar m^\sigma)
\end{pmatrix},
$$

where the frontier block has size
$f_r^\sigma \times f_c^\sigma$ and the root block has size
$r_r^\sigma \times r_c^\sigma$.

Fix one full second-cofactor entry. Let:

- $I_F$ be the deleted frontier-row set;
- $J_F$ be the deleted frontier-col set;
- $I_R$ be the deleted root-row set;
- $J_R$ be the deleted root-col set.

Write

$$
a_r = |I_F|,\qquad a_c = |J_F|,\qquad
b_r = |I_R|,\qquad b_c = |J_R|.
$$

Because this is a second cofactor,

$$
a_r + b_r = 2,\qquad a_c + b_c = 2.
$$

The corresponding deleted minor is nonzero only if both residual blocks are
square:

$$
f_r^\sigma - a_r = f_c^\sigma - a_c,
\qquad
r_r^\sigma - b_r = r_c^\sigma - b_c.
$$

Let

$$
d^\sigma := f_r^\sigma - f_c^\sigma .
$$

Then the square-block condition is equivalent to

$$
a_r - a_c = d^\sigma .
$$

So for one mask the valid degree-2 sectors are determined entirely by the
frontier shape difference $d^\sigma \in \{-2,-1,0,1,2\}$.

Now let

$$
\Sigma_F = \sum_{r \in I_F} r + \sum_{c \in J_F} c,
\qquad
\Sigma_R = \sum_{r \in I_R} r + \sum_{c \in J_R} c
$$

be the local deleted-index sums inside the frontier and root blocks.

In global block order, each deleted root row is shifted by $f_r^\sigma$ and
each deleted root column is shifted by $f_c^\sigma$. Therefore the exact full
second-cofactor sign is

$$
(-1)^{\Sigma_F + \Sigma_R + b_r f_r^\sigma + b_c f_c^\sigma}.
$$

Since the residual matrix remains block diagonal after the deletions,

$$
\det
S^\sigma_{\mathrm{blk}}[\widehat I_F,\widehat I_R;\widehat J_F,\widehat J_R]
=
\det \widetilde S_F^\sigma[\widehat I_F,\widehat J_F]\,
\det S_R^\sigma[\widehat I_R,\widehat J_R].
$$

Therefore the exact one-mask full second cofactor factorizes as

$$
\Gamma^\sigma_{\mathrm{blk}}(I_F,I_R;J_F,J_R)
=
(-1)^{b_r f_r^\sigma + b_c f_c^\sigma}\,
\Delta_F^{(a_r,a_c)}(I_F,J_F)\,
\Delta_R^{(b_r,b_c)}(I_R,J_R),
$$

where $\Delta_F^{(a_r,a_c)}$ and $\Delta_R^{(b_r,b_c)}$ are the signed local
degree-2 deleted-minor sectors defined above.

This is the exact second-order analogue of the first-cofactor merge:

- the second cofactor still factorizes into frontier and root pieces;
- the only new ingredient is the larger family of valid degree-2 sectors;
- the extra sign comes entirely from the shift of deleted root indices past
  the frontier block in block order.

### 7.5 Explicit Sector Form Of The One-Mask Same-Spin Merge

The previous factorization can now be specialized by the frontier shape
difference $d^\sigma = f_r^\sigma - f_c^\sigma$.

For compactness, write $\wedge$ for the natural insertion of two local sectors
into the full block-order $(2,2)$ tensor. The exact component formulas are
listed immediately after each schematic tensor identity.

For example, if $A$ is a frontier $(1,1)$ sector and $B$ is a root $(1,1)$
sector, then

$$
(A \wedge B)(r_F,r_R;c_F,c_R)
:=
A(r_F,c_F)\, B(r_R,c_R).
$$

The other schematic products below are read in the same way: the frontier
indices fill the frontier slots, the root indices fill the root slots, and the
later canonical antisymmetrization over row-pair and column-pair order is
handled separately.

#### Case $d^\sigma = 0$

Both frontier and root are square. The valid deletion distributions are

1. $(a_r,a_c) = (2,2)$ and $(b_r,b_c) = (0,0)$;
2. $(a_r,a_c) = (1,1)$ and $(b_r,b_c) = (1,1)$;
3. $(a_r,a_c) = (0,0)$ and $(b_r,b_c) = (2,2)$.

So

$$
\Gamma^\sigma_{\mathrm{full}}(m^\sigma)
=
\Gamma_F^\sigma O_R^\sigma
+
(-1)^{f_r^\sigma + f_c^\sigma}\,
\big(C_F^\sigma \wedge C_R^\sigma\big)
+
O_F^\sigma \Gamma_R^\sigma .
$$

In component form, the mixed term is

$$
\Gamma^\sigma_{\mathrm{full}}(r_F,r_R;c_F,c_R)
=
(-1)^{f_r^\sigma + f_c^\sigma}\,
C_F^\sigma(r_F,c_F)\,
C_R^\sigma(r_R,c_R) .
$$

#### Case $d^\sigma = +1$

The frontier has one extra row and the root has one extra column. The valid
deletion distributions are

1. $(a_r,a_c) = (2,1)$ and $(b_r,b_c) = (0,1)$;
2. $(a_r,a_c) = (1,0)$ and $(b_r,b_c) = (1,2)$.

So

$$
\Gamma^\sigma_{\mathrm{full}}(m^\sigma)
=
(-1)^{f_c^\sigma}\,
\big(P_F^\sigma \wedge v_R^\sigma\big)
+
(-1)^{f_r^\sigma}\,
\big(u_F^\sigma \wedge Q_R^\sigma\big) .
$$

Componentwise,

$$
\Gamma^\sigma_{\mathrm{full}}(r_{F,1},r_{F,2};c_F,c_R)
=
(-1)^{f_c^\sigma}
P_F^\sigma(r_{F,1},r_{F,2};c_F)\,
v_R^\sigma(c_R),
$$

and

$$
\Gamma^\sigma_{\mathrm{full}}(r_F,r_R;c_{R,1},c_{R,2})
=
(-1)^{f_r^\sigma}
u_F^\sigma(r_F)\,
Q_R^\sigma(r_R;c_{R,1},c_{R,2}) .
$$

#### Case $d^\sigma = -1$

The frontier has one extra column and the root has one extra row. The valid
deletion distributions are

1. $(a_r,a_c) = (1,2)$ and $(b_r,b_c) = (1,0)$;
2. $(a_r,a_c) = (0,1)$ and $(b_r,b_c) = (2,1)$.

So

$$
\Gamma^\sigma_{\mathrm{full}}(m^\sigma)
=
(-1)^{f_r^\sigma}\,
\big(Q_F^\sigma \wedge u_R^\sigma\big)
+
(-1)^{f_c^\sigma}\,
\big(v_F^\sigma \wedge P_R^\sigma\big) .
$$

Componentwise,

$$
\Gamma^\sigma_{\mathrm{full}}(r_F,r_R;c_{F,1},c_{F,2})
=
(-1)^{f_r^\sigma}
Q_F^\sigma(r_F;c_{F,1},c_{F,2})\,
u_R^\sigma(r_R),
$$

and

$$
\Gamma^\sigma_{\mathrm{full}}(r_{R,1},r_{R,2};c_F,c_R)
=
(-1)^{f_c^\sigma}
v_F^\sigma(c_F)\,
P_R^\sigma(r_{R,1},r_{R,2};c_R) .
$$

#### Case $d^\sigma = +2$

The frontier has two extra rows and the root has two extra columns. The only
valid distribution is

$$
(a_r,a_c) = (2,0),\qquad (b_r,b_c) = (0,2),
$$

so

$$
\Gamma^\sigma_{\mathrm{full}}(m^\sigma)
=
U_F^\sigma \wedge V_R^\sigma ,
$$

with component formula

$$
\Gamma^\sigma_{\mathrm{full}}(r_{F,1},r_{F,2};c_{R,1},c_{R,2})
=
U_F^\sigma(r_{F,1},r_{F,2})\,
V_R^\sigma(c_{R,1},c_{R,2}) .
$$

#### Case $d^\sigma = -2$

The frontier has two extra columns and the root has two extra rows. The only
valid distribution is

$$
(a_r,a_c) = (0,2),\qquad (b_r,b_c) = (2,0),
$$

so

$$
\Gamma^\sigma_{\mathrm{full}}(m^\sigma)
=
V_F^\sigma \wedge U_R^\sigma ,
$$

with component formula

$$
\Gamma^\sigma_{\mathrm{full}}(r_{R,1},r_{R,2};c_{F,1},c_{F,2})
=
V_F^\sigma(c_{F,1},c_{F,2})\,
U_R^\sigma(r_{R,1},r_{R,2}) .
$$

These five cases exhaust the exact one-mask same-spin merge. The final
mask-summed second-cofactor tensor is therefore

$$
\bar \Gamma^\sigma
=
\sum_{m^\sigma}
\eta^\sigma(m^\sigma)\,
\Gamma^\sigma_{\mathrm{full}}(m^\sigma).
$$

When these entries are later embedded into support-orbital labels, any
reordering of the deleted row pair or deleted column pair to the chosen
canonical pair order contributes the usual antisymmetry sign. That is separate
bookkeeping and does not change the block-order merge formulas above.

### 7.6 Current Exact Same-Spin Implementation

The current code does not yet have an explicit one-mask merge formula for

$$
\Gamma^\sigma_{\mathrm{full}}(m^\sigma).
$$

Instead, it uses the defining deleted-minor formula directly.

Fix one spin channel and one full ordered overlap block in block order

$$
[R^\sigma,\;L^\sigma]
$$

on both rows and columns. For one global deleted pair

$$
r_1 < r_2,\qquad c_1 < c_2,
$$

split the deleted indices back into reduced root/leaf occupied-orbital lists by
removing those positions from the original root-plus-leaf ordering. Then the
exact aggregated second cofactor is

$$
\bar \Gamma^\sigma_{r_1 r_2,\, c_1 c_2}
=
(-1)^{r_1+r_2+c_1+c_2}\,
\bar O^\sigma_{\mathrm{reduced}}(r_1,r_2;c_1,c_2),
$$

where

$$
\bar O^\sigma_{\mathrm{reduced}}(r_1,r_2;c_1,c_2)
$$

is the exact one-leaf overlap recurrence evaluated on the reduced
occupied-orbital lists after deleting those two rows and two columns.

Therefore the current exact same-spin scalar is

$$
\begin{aligned}
H_{\alpha\alpha,\mathrm{quad}}^{(2)}
=\;&
\bar O^\beta
\sum_{r_1<r_2}
\sum_{c_1<c_2}
\Big[
(r_1 c_1 \mid r_2 c_2)
-
(r_1 c_2 \mid r_2 c_1)
\Big] \\
&\times
(-1)^{r_1+r_2+c_1+c_2}\,
\bar O^\alpha_{\mathrm{reduced}}(r_1,r_2;c_1,c_2).
\end{aligned}
$$

The beta-beta channel is identical after swapping alpha and beta.

This formula is exact, but it is also the current bottleneck: every global
$(2,2)$ entry triggers a fresh reduced one-leaf overlap recurrence.

### 7.7 Full Fixed-Quadruple Two-Electron Formula

Putting the three channels together,

$$
H^{(2)}_{\mathrm{quad}}
=
\mathcal A(\bar \Gamma^\alpha)\, \bar O^\beta
+
\mathcal A(\bar \Gamma^\beta)\, \bar O^\alpha
+
\mathcal B(\bar C^\alpha, \bar C^\beta).
$$

This is the exact formula already matched by the current implementation.

## 8. Root-Leaf Orientation-Term Sum

For a rooted star with one root component and one leaf component, let

$$
\lambda_q
=
c^{L}_{R,q}\, c^{R}_{R,q}\, c^{L}_{L,q}\, c^{R}_{L,q}
$$

be the signed coefficient of one bra-root / ket-root / bra-leaf / ket-leaf
orientation-term quadruple $q$.

Then the exact component-ordered matrix elements are:

### 8.1 Overlap

$$
O_{\mathrm{star}}
=
\sum_q \lambda_q\,
\bar O_q^\alpha\,
\bar O_q^\beta.
$$

### 8.2 One-Electron

$$
H^{(1)}_{\mathrm{star}}
=
\sum_q \lambda_q
\Big(
\langle h, \bar C_q^\alpha \rangle\, \bar O_q^\beta
+
\bar O_q^\alpha\, \langle h, \bar C_q^\beta \rangle
\Big).
$$

### 8.3 Two-Electron

$$
\begin{aligned}
H^{(2)}_{\mathrm{star}}
=
\sum_q \lambda_q
\Big(
&
\mathcal A(\bar \Gamma_q^\alpha)\, \bar O_q^\beta
+
\mathcal A(\bar \Gamma_q^\beta)\, \bar O_q^\alpha \\
&+
\mathcal B(\bar C_q^\alpha, \bar C_q^\beta)
\Big).
\end{aligned}
$$

These are exactly the formulas that the current exact one-leaf separator code
must reproduce.

## 9. What Is Already Reduced, and What Is Still Missing

The present status is:

1. overlap is already reduced to the exact mask-summed scalar $\bar O^\sigma$;
2. one-electron is already reduced to $\bar O^\sigma$ and $\bar C^\sigma$;
3. opposite-spin two-electron is already reduced to $\bar C^\alpha$ and
   $\bar C^\beta$;
4. same-spin two-electron is still exact, but not yet reduced, because the
   code still reconstructs each global second cofactor through a reduced
   overlap recurrence.

So the mathematical bottleneck is now very specific:

- the exact formula itself is known;
- the missing piece is the exact one-mask merge formula for
  $\Gamma^\sigma_{\mathrm{full}}(m^\sigma)$.

Once that second-order payload exists, same-spin can be collapsed in the same
spirit as overlap, one-electron, and opposite-spin two-electron.
