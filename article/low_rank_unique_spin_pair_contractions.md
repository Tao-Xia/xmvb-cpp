# Low-Rank, Low-Scaling Unique-Spin-String Pair Contractions for Nonorthogonal VBSCF

## Abstract

This note develops an exact low-scaling algorithm for matrix elements and
orbital derivatives between nonorthogonal unique-spin strings. The central
observation is that two adjacent strings can be ordered to differ by only one
occupied orbital. Their occupied-orbital overlap matrices are then related by
a rank-one row or column modification. Determinants, inverse overlap matrices,
one-electron transition contractions, and resolution-of-the-identity (RI)
two-electron contractions can consequently be updated by determinant-lemma
and Woodbury identities instead of being recomputed independently.

The same factorized formulation yields closed analytic pullbacks and
directional pullbacks. Therefore the energy, orbital gradient, and
matrix-free Hessian-vector product (HVP) can all be evaluated without forming
high-order reduced density matrices, high-order cofactor tensors, or
four-index active-space adjoints. The algorithm is exact for a fixed
factorized two-electron Hamiltonian. Its asymptotic pair-local cost is reduced
from an anchor cost proportional to

$$
O\!\left(N_{\mathrm{aux}} n^3\right)
$$

to an incremental cost proportional to

$$
O\!\left(N_{\mathrm{aux}} n^2\right),
$$

where $n$ is the number of occupied orbitals in one spin string. A
two-dimensional traversal of the left-string/right-string product graph
further reduces the number of full anchor factorizations from one per string
or string pair to one per connected component. Ill-conditioned pairs are
handled by residual-certified re-anchoring or exact contracted interpolation,
without changing the mathematical functional.

## 1. Scope and guiding principle

Consider a nonorthogonal VBSCF wave function whose determinant expansion has
been compressed into unique alpha and beta spin strings. Matrix elements are
naturally reused at the unique-spin-string-pair level. For a fixed spin
sector, the fundamental task is to evaluate quantities associated with every
ordered pair

$$
(\mathcal R,\mathcal L),
$$

where $\mathcal R$ is the right spin string and $\mathcal L$ is the left spin
string.

If every pair is evaluated independently, the occupied overlap matrix must be
factorized repeatedly and the same determinant/cofactor algebra is repeated
for strings that differ by only one occupied orbital. The low-rank algorithm
instead has three levels:

1. order the unique strings by single-site substitutions;
2. propagate pair-local algebra by rank-one or rank-$r$ updates; and
3. contract the required scalar, gradient, or HVP output before constructing
   any high-order RDM or cofactor tensor.

The algorithm does not approximate the nonorthogonal determinant algebra.
With an RI or Cholesky representation of the two-electron integrals, it is an
exact reassociation of finite sums and exact differentiation of that
reassociated functional.

The word *low-rank* in this note refers exclusively to the algebraic change
between adjacent occupied overlap and transition-integral matrices. It does
not refer to a truncated wave function, a low-rank Hessian approximation, or
a screened RDM.

## 2. Pair-local notation

### 2.1 Left and right spin strings

Let a right spin string contain the ordered occupied orbitals

$$
\mathcal R=(r_1,r_2,\ldots,r_n),
$$

and let a left spin string contain

$$
\mathcal L=(l_1,l_2,\ldots,l_n).
$$

The corresponding orbital matrices are denoted by

$$
R=\begin{pmatrix}r_1&r_2&\cdots&r_n\end{pmatrix},
\qquad
L=\begin{pmatrix}l_1&l_2&\cdots&l_n\end{pmatrix}.
$$

For real orbitals and AO overlap matrix $S$, define the occupied overlap
matrix

$$
X=R^{\mathrm T}SL,
\qquad
X_{ij}=\langle r_i|l_j\rangle.
\tag{1}
$$

At a regular pair,

$$
K=X^{-1},
\qquad
\Omega=\det X.
\tag{2}
$$

The unnormalized overlap between the two spin determinants is $\Omega$.

The ordered-slot convention in eq 1 is essential. If occupied orbitals are
canonically reordered after a substitution, the corresponding permutation
parity must be retained explicitly. Otherwise a nominal single-site change
can silently alter the determinant sign.

### 2.2 One-electron transition block

For a one-electron operator $h$, define its occupied transition block

$$
M^h=R^{\mathrm T}hL.
\tag{3}
$$

The normalized one-electron contraction and its unnormalized matrix element
are

$$
j_h=\operatorname{tr}(KM^h),
\tag{4}
$$

and

$$
H_1=\Omega j_h.
\tag{5}
$$

### 2.3 Factorized two-electron transition blocks

Assume the two-electron interaction is represented in a symmetric
factorized form,

$$
(pq|rs)=\sum_Q B^Q_{pq}B^Q_{rs}.
\tag{6}
$$

Equation 6 may be an RI or Cholesky representation. All derivations below are
exact with respect to the Hamiltonian defined by eq 6.

For each auxiliary index $Q$, define

$$
M^Q=R^{\mathrm T}B^Q L,
\tag{7}
$$

and

$$
A^Q=KM^Q,
\qquad
j_Q=\operatorname{tr}(A^Q).
\tag{8}
$$

The normalized same-spin two-electron contraction is

$$
\phi_{\mathrm{ss}}
=
\frac{1}{2}\sum_Q
\left[
j_Q^2-\operatorname{tr}\!\left((A^Q)^2\right)
\right].
\tag{9}
$$

Consequently, the unnormalized same-spin matrix element is

$$
V_{\mathrm{ss}}=\Omega\phi_{\mathrm{ss}}.
\tag{10}
$$

Equation 9 is the factorized form of the direct-minus-exchange contraction.
It eliminates the need to construct either a four-index two-particle
transition density or a second-cofactor tensor.

## 3. Single-site substitutions as low-rank modifications

### 3.1 Right-string substitution

Suppose only the occupied orbital in right slot $p$ changes. Then one row of
$X$ changes, and the new overlap matrix can be written as

$$
X'=X+uv^{\mathrm T},
\qquad
u=e_p,
\tag{11}
$$

where $e_p$ is the $p$th Cartesian unit vector and $v$ contains the overlap
changes against all left occupied orbitals.

The corresponding one-electron or auxiliary transition block changes in the
same row:

$$
M'=M+u\,\delta m^{\mathrm T}.
\tag{12}
$$

Define

$$
a=Ku,
\qquad
c=K^{\mathrm T}v,
\qquad
d=1+v^{\mathrm T}Ku.
\tag{13}
$$

The matrix determinant lemma gives

$$
\Omega'=\Omega d,
\tag{14}
$$

and the Sherman--Morrison identity gives

$$
K'=K-\frac{ac^{\mathrm T}}{d}.
\tag{15}
$$

These equations require $d\ne0$. Numerical certification and singular cases
are discussed in Section 10.

### 3.2 Exact update of the contracted auxiliary matrix

Let

$$
A=KM.
$$

Define the row-update residual

$$
t^{\mathrm T}
=
\delta m^{\mathrm T}-c^{\mathrm T}M.
\tag{16}
$$

Substituting eqs 12 and 15 gives the exact rank-one relation

$$
A'=A+\frac{at^{\mathrm T}}{d}.
\tag{17}
$$

Therefore

$$
j'=j+\frac{t^{\mathrm T}a}{d}.
\tag{18}
$$

For the unnormalized one-electron contraction, eqs 14 and 18 yield

$$
H_1'
=
dH_1+\Omega t_h^{\mathrm T}a.
\tag{19}
$$

For an RI auxiliary block, define

$$
s_Q=t_Q^{\mathrm T}a.
\tag{20}
$$

Using

$$
\operatorname{tr}\!\left((A^Q+at_Q^{\mathrm T}/d)^2\right)
=
\operatorname{tr}\!\left((A^Q)^2\right)
+\frac{2t_Q^{\mathrm T}A^Qa}{d}
+\frac{s_Q^2}{d^2},
\tag{21}
$$

the quadratic terms in $s_Q$ cancel exactly against the corresponding change
in $j_Q^2$. The same-spin two-electron update becomes

$$
V_{\mathrm{ss}}'
=
dV_{\mathrm{ss}}
+\Omega\sum_Q
t_Q^{\mathrm T}(j_QI-A^Q)a.
\tag{22}
$$

The cancellation in eq 22 is important: no four-index intermediate is needed
to update the full direct-minus-exchange energy.

### 3.3 Left-string substitution

If only left slot $p$ changes, one column changes:

$$
X'=X+uv^{\mathrm T},
\qquad
v=e_p,
\tag{23}
$$

and

$$
M'=M+\delta m\,v^{\mathrm T}.
\tag{24}
$$

Equations 14 and 15 remain valid with the definitions in eq 13. The update of
$A=KM$ can be evaluated as

$$
A'
=
A+bv^{\mathrm T}
-\frac{a\left(c^{\mathrm T}M+s v^{\mathrm T}\right)}{d},
\tag{25}
$$

where

$$
b=K\delta m,
\qquad
s=c^{\mathrm T}\delta m=v^{\mathrm T}b.
\tag{26}
$$

Equation 25 is a sum of at most two rank-one terms and therefore has the same
$O(n^2)$ asymptotic cost as the row update. Equivalently, one may propagate
$\widetilde A=MK$ for column updates and use the transpose analogue of eq 17.
The scalar invariants are unchanged because

$$
\operatorname{tr}(KM)=\operatorname{tr}(MK)
$$

and

$$
\operatorname{tr}\!\left((KM)^2\right)
=
\operatorname{tr}\!\left((MK)^2\right).
$$

### 3.4 Rank-$r$ and delayed updates

Several substitutions may be grouped into

$$
X'=X+UV^{\mathrm T},
\tag{27}
$$

where $U,V\in\mathbb R^{n\times r}$. The Woodbury identity gives

$$
K'
=
K-KU\left(I_r+V^{\mathrm T}KU\right)^{-1}V^{\mathrm T}K,
\tag{28}
$$

and

$$
\Omega'
=
\Omega\det\!\left(I_r+V^{\mathrm T}KU\right).
\tag{29}
$$

The corresponding transition-block modification may be represented as

$$
M'=M+PQ^{\mathrm T}.
\tag{30}
$$

Equations 28--30 provide an exact delayed-update formulation. Small batches of
rank-one substitutions can thus be accumulated and applied with matrix-matrix
operations. This improves arithmetic intensity without changing the
functional or introducing a truncation threshold.

## 4. Two-dimensional unique-string-pair traversal

### 4.1 Single-spin substitution graphs

For a fixed spin sector, define a graph

$$
G=(\mathcal V,\mathcal E),
$$

whose vertices are unique spin strings. Two vertices share an edge if their
ordered occupied lists differ by one site substitution. The edge stores:

1. the replaced slot;
2. the incoming and outgoing orbital labels;
3. the overlap row or column difference; and
4. the determinant permutation sign, if canonical reordering is used.

The graph can be constructed from occupation masks by enumerating valid
occupied-to-unoccupied substitutions and looking up the resulting mask. This
requires work proportional to the number of possible substitutions,

$$
O\!\left(U n(m-n)\right),
\tag{31}
$$

rather than all-pairs string comparison.

### 4.2 Product graph of ordered string pairs

Let $G_R$ and $G_L$ be the right- and left-string graphs. All ordered pair
matrix elements form the vertex set of the Cartesian product graph

$$
G_{RL}=G_R\square G_L.
\tag{32}
$$

An edge in $G_R$ changes one row of $X$, while an edge in $G_L$ changes one
column. Therefore every edge of $G_{RL}$ admits the low-rank updates in
Section 3.

If $G_R$ and $G_L$ have $c_R$ and $c_L$ connected components, respectively,
then $G_{RL}$ has

$$
c_{RL}=c_Rc_L
\tag{33}
$$

connected components. Only one independently factorized anchor is required
for each component. All remaining pair states can be reached through row or
column updates.

For a rectangular full-product space, a snake or Gray-like path may visit
every pair once. For an arbitrary selected subset, a spanning forest provides
the same exact construction. A depth-first traversal may move forward and
backward along tree edges; the reverse edge is itself another exact low-rank
update. Consequently, an arbitrary component can be streamed with bounded
state memory.

### 4.3 Algebraic exactness of the traversal

Let $F(X,\{M^Q\})$ denote any scalar defined by eqs 5, 9, or their linear
combinations. At each graph edge, eqs 14--18 and 22 are identities. By
induction along a path from the anchor to any vertex,

$$
F_{\mathrm{updated}}=F_{\mathrm{independent}}
\tag{34}
$$

in exact arithmetic. The traversal order therefore changes computational
cost only; it does not define a new physical approximation.

## 5. Opposite-spin contraction in auxiliary space

For spin sector $\sigma\in\{\alpha,\beta\}$, define the unnormalized
auxiliary projection

$$
g_Q^{\sigma}
=
\Omega_{\sigma}j_Q^{\sigma}.
\tag{35}
$$

For one alpha-string pair and one beta-string pair, the opposite-spin
two-electron matrix element is

$$
V_{\alpha\beta}
=
\sum_Q g_Q^{\alpha}g_Q^{\beta}.
\tag{36}
$$

Under a right-string row update in sector $\sigma$, eqs 14 and 18 give

$$
g_Q^{\sigma\prime}
=
d_{\sigma}g_Q^{\sigma}
+\Omega_{\sigma}t_Q^{\mathrm T}a.
\tag{37}
$$

Thus opposite-spin contractions require only one auxiliary vector for each
same-spin pair. A four-index alpha-beta active-pair tensor is unnecessary.

When the VB structure expansion is sparse, the vectors in eq 35 can be
contracted directly through the sparse structure-to-unique-string map. No
intermediate expansion to a redundant determinant representation is required.

## 6. Contract-first analytic differentiation

### 6.1 Differential identities

For an arbitrary variation of a regular pair,

$$
\delta K=-K\,\delta X\,K,
\tag{38}
$$

and

$$
\delta\Omega
=
\Omega\operatorname{tr}(K\delta X).
\tag{39}
$$

For each auxiliary block,

$$
\delta A^Q
=
\delta K M^Q+K\delta M^Q.
\tag{40}
$$

Define

$$
D^Q=A^QK=KM^QK,
\tag{41}
$$

and

$$
E^Q=A^QD^Q=(A^Q)^2K.
\tag{42}
$$

The differential of eq 9 is

$$
\delta\phi_{\mathrm{ss}}
=
\sum_Q
\operatorname{tr}
\left[
(j_QI-A^Q)\delta A^Q
\right].
\tag{43}
$$

### 6.2 Same-spin pullback

Let a scalar objective contain the weighted pair contribution

$$
\mathcal L_{\mathrm{ss}}
=
w\Omega\phi_{\mathrm{ss}},
\tag{44}
$$

where $w$ is an external coefficient independent of the local variables in
the present pullback. Define adjoints by

$$
\delta\mathcal L
=
\operatorname{tr}(\overline X^{\mathrm T}\delta X)
+\sum_Q
\operatorname{tr}\!\left((\overline M^Q)^{\mathrm T}\delta M^Q\right).
\tag{45}
$$

Substitution of eqs 38--43 yields

$$
\overline M^Q
=
w\Omega
\left(j_QK-D^Q\right)^{\mathrm T},
\tag{46}
$$

and

$$
\overline X
=
w\Omega
\left[
\phi_{\mathrm{ss}}K
-\sum_Q\left(j_QD^Q-E^Q\right)
\right]^{\mathrm T}.
\tag{47}
$$

Equations 46 and 47 are the complete pair-local same-spin pullback. They are
obtained by differentiating the contracted scalar in eq 9, not by forming and
then differentiating a four-index transition RDM.

### 6.3 One-electron pullback

For

$$
\mathcal L_1=w\Omega j_h,
$$

define

$$
A^h=KM^h,
\qquad
D^h=A^hK.
$$

The corresponding adjoints are

$$
\overline M^h=w\Omega K^{\mathrm T},
\tag{48}
$$

and

$$
\overline X
=
w\Omega\left(j_hK-D^h\right)^{\mathrm T}.
\tag{49}
$$

Overlap, one-electron, same-spin, and opposite-spin contributions can be
accumulated into the same $\overline X$ before the orbital pullback.

### 6.4 Opposite-spin pullback

When differentiating one spin sector of eq 36, regard the other sector as an
auxiliary weight

$$
\lambda_Q=g_Q^{\bar\sigma}.
$$

The local functional is

$$
\mathcal L_{\mathrm{os}}
=
\Omega\sum_Q\lambda_Qj_Q.
\tag{50}
$$

Define

$$
F=\sum_Q\lambda_Qj_Q.
\tag{51}
$$

Then

$$
\overline M^Q
=
\Omega\lambda_QK^{\mathrm T},
\tag{52}
$$

and

$$
\overline X
=
\Omega
\left[
FK-\sum_Q\lambda_QD^Q
\right]^{\mathrm T}.
\tag{53}
$$

If $\lambda_Q$ itself depends on the differentiated variables, its pullback is
handled symmetrically in the other spin sector. Equation 53 again avoids a
four-index alpha-beta adjoint.

## 7. Pullback to factorized integrals and orbitals

Because

$$
M^Q=R^{\mathrm T}B^Q L,
$$

the auxiliary-factor adjoint is directly

$$
\overline B^Q
=
R\overline M^Q L^{\mathrm T}.
\tag{54}
$$

There is no need to form an intermediate adjoint with four active-orbital
indices. The orbital contributions from $M^Q$ are

$$
\overline R_{M^Q}
=
B^Q L(\overline M^Q)^{\mathrm T},
\tag{55}
$$

and

$$
\overline L_{M^Q}
=
(B^Q)^{\mathrm T}R\overline M^Q.
\tag{56}
$$

Similarly, since

$$
X=R^{\mathrm T}SL,
$$

the overlap-block pullback contributes

$$
\overline R_X
=
SL\overline X^{\mathrm T},
\tag{57}
$$

and

$$
\overline L_X
=
S^{\mathrm T}R\overline X.
\tag{58}
$$

Equations 54--58 close the reverse-mode chain from the contracted pair scalar
to RI factors and orbital coefficients.

## 8. Directional pullback and matrix-free HVP

### 8.1 Tangent propagation

Let a dot denote the directional derivative along an orbital displacement.
The fundamental tangents are

$$
\dot K=-K\dot X K,
\tag{59}
$$

$$
\dot\Omega=\Omega\operatorname{tr}(K\dot X),
\tag{60}
$$

$$
\dot A^Q=\dot K M^Q+K\dot M^Q,
\tag{61}
$$

$$
\dot j_Q=\operatorname{tr}(\dot A^Q),
\tag{62}
$$

$$
\dot D^Q=\dot A^QK+A^Q\dot K,
\tag{63}
$$

and

$$
\dot E^Q=\dot A^QD^Q+A^Q\dot D^Q.
\tag{64}
$$

The tangent of the normalized same-spin contraction is

$$
\dot\phi_{\mathrm{ss}}
=
\sum_Q
\operatorname{tr}
\left[
(j_QI-A^Q)\dot A^Q
\right].
\tag{65}
$$

### 8.2 Directional same-spin adjoint

Define the untransposed adjoint kernels

$$
Z_M^Q=j_QK-D^Q,
\tag{66}
$$

and

$$
Z_X
=
\phi_{\mathrm{ss}}K
-\sum_Q(j_QD^Q-E^Q).
\tag{67}
$$

Their directional derivatives are

$$
\dot Z_M^Q
=
\dot j_QK+j_Q\dot K-\dot D^Q,
\tag{68}
$$

and

$$
\dot Z_X
=
\dot\phi_{\mathrm{ss}}K
+\phi_{\mathrm{ss}}\dot K
-\sum_Q
\left(
\dot j_QD^Q+j_Q\dot D^Q-\dot E^Q
\right).
\tag{69}
$$

If the external pair weight also has a directional response $\dot w$, then

$$
\dot{\overline M}^Q
=
(\dot w\,\Omega+w\dot\Omega)(Z_M^Q)^{\mathrm T}
+w\Omega(\dot Z_M^Q)^{\mathrm T},
\tag{70}
$$

and

$$
\dot{\overline X}
=
(\dot w\,\Omega+w\dot\Omega)Z_X^{\mathrm T}
+w\Omega\dot Z_X^{\mathrm T}.
\tag{71}
$$

Equations 59--71 provide the exact pair-local contribution to a matrix-free
HVP. They require only matrices with two occupied indices and one optional
auxiliary index. Third- and fourth-order cofactor derivatives never appear.

### 8.3 Block HVP

For $b$ trial directions, attach a direction index $a=1,\ldots,b$ to every
dotted quantity in eqs 59--71. The base quantities

$$
K,\quad \Omega,\quad A^Q,\quad D^Q,\quad E^Q
$$

are shared by the entire block. Consequently, one pair traversal can
propagate all $b$ tangents. The natural storage scales as

$$
O\!\left(N_{\mathrm{aux}}n^2\right)
+O\!\left(bN_{\mathrm{aux}}n^2\right),
\tag{72}
$$

or less when the auxiliary index is blocked. It does not require a matrix of
size $O(m^4b)$ in the active-orbital dimension $m$.

## 9. Complexity and bounded-memory execution

Let

$$
N_{\mathrm{pair}}=U_RU_L
$$

be the number of ordered unique-string pairs and let $c_{RL}$ be the number of
connected components of the product graph.

An independently evaluated RI pair requires, in the straightforward dense
occupied-block formulation,

$$
O\!\left(N_{\mathrm{aux}}n^3\right)
$$

work to form all $A^Q=KM^Q$. Repeating this for every pair costs

$$
O\!\left(
N_{\mathrm{pair}}N_{\mathrm{aux}}n^3
\right).
\tag{73}
$$

With product-graph traversal, anchors cost

$$
O\!\left(
c_{RL}N_{\mathrm{aux}}n^3
\right),
\tag{74}
$$

and the remaining edges cost

$$
O\!\left(
N_{\mathrm{pair}}N_{\mathrm{aux}}n^2
\right).
\tag{75}
$$

The total leading cost is therefore

$$
O\!\left(
c_{RL}N_{\mathrm{aux}}n^3
+N_{\mathrm{pair}}N_{\mathrm{aux}}n^2
\right).
\tag{76}
$$

For large connected string spaces, $c_{RL}\ll N_{\mathrm{pair}}$, and eq 76
removes one occupied-orbital factor from the dominant pair sweep.

The auxiliary index may be processed in blocks of width $q_b$. A streamed
implementation then requires pair-local storage

$$
O(q_bn^2)
$$

rather than

$$
O(N_{\mathrm{aux}}n^2).
$$

The overlap inverse $K$ is independent of $Q$ and may be regenerated along
the same graph traversal for each auxiliary block. Scalar energies and
$\overline X$ are accumulated across blocks. This trades a controlled amount
of $O(n^2)$ update work for an explicit memory bound.

Parallel execution should use a forest of independently anchored traversal
tiles. Within a tile, pair states are streamed sequentially; across tiles,
anchors and traversals are independent. The tile count is an execution
parameter determined by available threads and memory, not by molecule-specific
physical thresholds.

## 10. Ill-conditioned and singular pair treatment

### 10.1 Residual-certified propagation

Repeated rank-one updates accumulate floating-point error even though each
identity is exact algebraically. The inverse must therefore be certified by a
backward-error measure such as

$$
\eta_K
=
\frac{\|I-XK\|}
{1+\|X\|\,\|K\|}.
\tag{77}
$$

Re-anchoring is triggered when $\eta_K$ is inconsistent with the requested
accuracy. A fixed number of updates between factorizations has no mathematical
basis and is not required.

Likewise, the Sherman--Morrison denominator

$$
d=1+v^{\mathrm T}Ku
$$

must be assessed relative to the norms of its terms. A small or uncertain
$d$ indicates that the updated pair cannot be represented reliably by the
current inverse chart. The correct response is a certified factorization or
an exact singular-pair representation, not regularization by an empirical
constant.

For long traversals, determinants should be propagated in signed logarithmic
form,

$$
\det X=s_X\exp(\ell_X),
$$

with

$$
s_X\in\{-1,+1\},
\qquad
\ell_X=\log|\det X|,
$$

to avoid overflow and underflow. Scaled unnormalized contractions may be
propagated consistently with the same exponent.

### 10.2 Exact contracted interpolation

When $X$ is singular or too ill-conditioned for an inverse-based chart,
introduce a regularizing path

$$
X(z)=X+zUV^{\mathrm T}
\tag{78}
$$

chosen so that the evaluation nodes $z_k$ are regular. Unnormalized overlap,
one-electron, and two-electron determinant matrix elements are polynomials in
the entries of $X$ and the corresponding transition blocks. Along eq 78,
their contracted values are therefore finite-degree polynomials in $z$.

The required scalar, adjoint, or directional adjoint is evaluated at a
sufficient number of well-conditioned nodes and interpolated to $z=0$:

$$
F(0)=\sum_k \ell_k(0)F(z_k),
\tag{79}
$$

where $\ell_k$ are the Lagrange basis polynomials. The same relation applies
componentwise to $\overline X$, $\overline M^Q$, and their directional
derivatives.

This is a contract-first singular algorithm. It reconstructs only the final
objects required by the energy, gradient, or HVP. It does not reconstruct
generic third- or fourth-order cofactor arrays. When the nullity or required
regularizing rank is small, the number of interpolation nodes is also small,
which is the practically important regime.

## 11. Exact-integral path and factorization boundary

The determinant and inverse-overlap updates in Sections 3 and 4 are
independent of the integral representation. They therefore also accelerate
overlap and one-electron contractions in an exact four-index integral path.

However, a generic exact two-electron tensor still contains four active
orbital indices. Low-rank updates of $X^{-1}$ alone do not remove the
four-index contraction cost. The reduction in eqs 9, 22, 46, and 47 relies on
a factorized two-electron representation such as RI or Cholesky.

Accordingly, the two mathematical paths should remain explicit:

1. an exact four-index reference formulation for validation and small active
   spaces; and
2. a factorized low-scaling formulation for production calculations.

They should share the same scalar and derivative output contract but need not
share a four-index internal representation.

## 12. Correctness statements

The proposed algorithm rests on the following statements.

### Proposition 1: traversal invariance

For a regular connected component, any two traversal paths from the same
anchor to a given string pair produce the same $X^{-1}$, determinant, and
contracted matrix elements in exact arithmetic.

*Reason.* Every edge applies the exact determinant lemma and Woodbury identity
to the uniquely defined endpoint matrices. Path independence follows because
the endpoint matrices themselves are path independent.

### Proposition 2: contraction-before-differentiation equivalence

The adjoints in eqs 46--53 equal the contraction of the corresponding exact
transition RDM/cofactor derivatives with the same integral and external
weights.

*Reason.* Equations 9 and 36 are exact reassociations of the original finite
sums. Differentiation is linear and commutes with finite summation and matrix
trace operations.

### Proposition 3: RI-native HVP exactness

Equations 59--71 give the exact directional derivative of the RI-native
gradient for all regular pairs.

*Reason.* They are obtained by forward differentiation of the exact reverse
pullback in eqs 46 and 47. No finite-difference, secant, or truncated response
approximation is introduced.

### Proposition 4: singular interpolation exactness

If the interpolation degree bound is satisfied and the node evaluations are
exact, eq 79 reproduces the singular-pair contracted output exactly.

*Reason.* The unnormalized determinant matrix elements and their contracted
derivatives are finite-degree polynomials along the chosen regularization
path, and polynomial interpolation is unique.

## 13. Mathematical validation protocol

A complete validation should be independent of the traversal order and should
separate algebraic correctness from performance.

### 13.1 Pair-local value tests

For random small nonorthogonal orbital blocks, compare independent evaluation
and graph-updated evaluation of

$$
\Omega,
\qquad
H_1,
\qquad
V_{\mathrm{ss}},
\qquad
g_Q,
\qquad
V_{\alpha\beta}.
$$

The tests must include row updates, column updates, mixed paths, reversed
edges, and alternative paths to the same endpoint.

### 13.2 Gradient tests

For a scalar pair functional $F$, verify

$$
\frac{
F(X+\epsilon\Delta X,M+\epsilon\Delta M)
-F(X-\epsilon\Delta X,M-\epsilon\Delta M)
}{2\epsilon}
$$

against

$$
\operatorname{tr}(\overline X^{\mathrm T}\Delta X)
+\sum_Q
\operatorname{tr}\!\left((\overline M^Q)^{\mathrm T}\Delta M^Q\right).
$$

Complex-step differentiation may be used where the factorization and
determinant conventions support complex arithmetic.

### 13.3 HVP tests

Compare the analytic directional adjoint with the central difference of the
analytic gradient. The comparison must include both fixed external weights
and directional external weights $\dot w$, because the latter represent
selected-state or structure response.

### 13.4 Singular-pair tests

Construct matrices of controlled nullity and compare contracted interpolation
against explicit minor/cofactor reference expressions. Nullities zero, one,
and two should be covered, together with nearly singular regular pairs that
exercise the backward-error certificate.

### 13.5 Wave-function-level tests

The final validation set should cover:

1. arbitrary subsets of VB structures;
2. complete and incomplete unique-string spaces;
3. HAO and full-AO OEO orbitals;
4. single-state and state-averaged objectives;
5. exact-integral reference and factorized-integral calculations; and
6. permutations of the input unique-string order.

The last item verifies that graph order changes performance only and not the
computed wave function.

## 14. Summary

The low-scaling formulation replaces independent unique-spin-string-pair
evaluation by an exact traversal of the left/right product graph. A
single-site substitution produces a rank-one row or column modification of
the occupied overlap and transition-integral blocks. Determinants and inverse
overlaps follow from determinant-lemma and Woodbury updates, while RI
same-spin and opposite-spin energies are propagated through auxiliary-index
contractions.

The decisive step is to differentiate the already contracted RI functional.
This yields compact analytic adjoints and directional adjoints involving only
$K$, $A^Q$, $D^Q$, and $E^Q$. High-order transition RDMs, high-order cofactor
tensors, four-index active-space adjoints, and their directional counterparts
are not required in the regular production path.

For connected unique-string spaces, the resulting leading pair-sweep cost is

$$
O\!\left(
c_{RL}N_{\mathrm{aux}}n^3
+U_RU_LN_{\mathrm{aux}}n^2
\right),
$$

with explicitly bounded auxiliary-block memory. Residual-certified
re-anchoring and exact contracted interpolation extend the formulation to
ill-conditioned and singular pairs without empirical regularization or loss
of mathematical consistency.
