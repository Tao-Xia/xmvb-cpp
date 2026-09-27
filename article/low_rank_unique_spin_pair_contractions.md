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

The accepted same-spin response can be propagated by the same traversal. Define
the compact occupied-space aggregate

$$
W
=
\sum_Q\left[j_QA^Q-(A^Q)^2\right].
\tag{22a}
$$

Write the channel update as

$$
A^{Q\prime}=A^Q+u t_Q^{\mathrm T},
\qquad
u=\frac{a}{d},
\qquad
s_Q=t_Q^{\mathrm T}u.
\tag{22b}
$$

Direct expansion of eq 22a gives

$$
W'-W
=
\sum_Q
\left[
s_QA^Q
+j_Qut_Q^{\mathrm T}
-(A^Qu)t_Q^{\mathrm T}
-u(t_Q^{\mathrm T}A^Q)
\right].
\tag{22c}
$$

The nominal quadratic rank-one term cancels exactly between the two parts of
eq 22a. Consequently, updating the compact response aggregate requires only

$$
O(N_{\mathrm{aux}}n^2)
$$

work per single-site substitution. Moreover, the relation between the
transition block and contracted channel is

$$
M^Q=XA^Q.
$$

The derivative of the normalized same-spin functional with respect to the
inverse overlap is therefore

$$
\frac{\partial\phi_{\mathrm{ss}}}{\partial K}
=
W^{\mathrm T}X^{\mathrm T}.
\tag{22d}
$$

For the one- plus two-electron functional this becomes

$$
\frac{\partial\phi}{\partial K}
=
h^{\mathrm T}+W^{\mathrm T}X^{\mathrm T}.
\tag{22e}
$$

Thus the accepted response payload is obtained from one occupied-space square
aggregate;
the auxiliary channel matrices remain traversal-local and are not stored for
all unique-string pairs.

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

### 4.4 Arbitrary-order transition-density hierarchy

The product-graph update is not restricted to the one- and two-particle
objects needed by the Hamiltonian. It extends to an arbitrary transition-RDM
order. Let $I=(i_1<\cdots<i_q)$ and $J=(j_1<\cdots<j_q)$ be ordered occupied
index sets and define the $q$th compound matrix

$$
\mathcal C_q(K^{\mathrm T})_{I,J}
=
\det\!\left[(K^{\mathrm T})_{I,J}\right],
\qquad
\mathcal C_0=1.
\tag{34a}
$$

The occupied-index representation of the unnormalized $q$-particle
transition density is

$$
\Gamma^{(q)}_{I,J}
=
\Omega\,\mathcal C_q(K^{\mathrm T})_{I,J}.
\tag{34b}
$$

Scattering the occupied labels in $I$ and $J$ back to their active-orbital
labels gives the usual transition RDM. Equation 34b includes the overlap at
$q=0$ and the first cofactor at $q=1$.

Consider a generic rank-one inverse update

$$
K'=K+ab^{\mathrm T},
\qquad
K'^{\mathrm T}=K^{\mathrm T}+ba^{\mathrm T}.
\tag{34c}
$$

Multilinearity of the determinant implies that at most one column of a minor
can be taken from the rank-one term. Therefore

$$
\begin{aligned}
\mathcal C_q(K'^{\mathrm T})_{I,J}
={}&\mathcal C_q(K^{\mathrm T})_{I,J}\\
&+\sum_{\mu=1}^{q}\sum_{\nu=1}^{q}
(-1)^{\mu+\nu}
b_{i_\mu}a_{j_\nu}
\mathcal C_{q-1}(K^{\mathrm T})
_{I\setminus i_\mu,\,J\setminus j_\nu}.
\end{aligned}
\tag{34d}
$$

Define the insertion operator represented by the double sum as

$$
\mathscr I_q(\mathcal C_{q-1};b,a)_{I,J}.
\tag{34e}
$$

If the same edge changes the determinant by $\Omega'=d\Omega$, the
unnormalized transition RDM obeys

$$
\Gamma^{(q)\prime}
=
d\Gamma^{(q)}
+d\Omega\,
\mathscr I_q(\mathcal C_{q-1};b,a).
\tag{34f}
$$

Equations 34d--34f are exact for every $q$. All requested orders are advanced
together. An in-place implementation visits
$q=q_{\max},q_{\max}-1,\ldots,1$, so that the right-hand side always uses the
old $(q-1)$th level. No independent minor factorization is performed at the
new string pair.

Let

$$
N_q=\binom{n}{q}.
$$

Materializing a full $q$th-order occupied transition density necessarily
requires

$$
\Theta(N_q^2)
$$

storage and writes, because that is the size of the requested output. The
rank-one recurrence costs

$$
O(q^2N_q^2)
\tag{34g}
$$

per graph edge. For a fixed physical order $q$, this is output-linear up to a
small order-dependent factor. In contrast, independently evaluating every
minor adds a factorization or determinant cost to every output element. When
only a contraction of $\Gamma^{(q)}$ is required, the insertion in eq 34e
should be contracted immediately and the $N_q\times N_q$ matrix need not be
stored.

For a rank-$r$ inverse update, Cauchy--Binet gives a finite sum containing
$s=0,\ldots,\min(q,r)$ inserted update columns. The rank-one traversal is the
preferred production form because it terminates after $s=1$ and shares the
same string graph as the determinant and RI-channel recurrences.

### 4.5 Directional arbitrary-order densities

For any square matrix $A$, the directional derivative of its $q$th compound
matrix is

$$
\begin{aligned}
D\mathcal C_q(A)[\dot A]_{I,J}
=
\sum_{\mu=1}^{q}\sum_{\nu=1}^{q}
(-1)^{\mu+\nu}
\dot A_{i_\mu j_\nu}
\mathcal C_{q-1}(A)
_{I\setminus i_\mu,\,J\setminus j_\nu}.
\end{aligned}
\tag{34h}
$$

Consequently,

$$
\dot\Gamma^{(q)}
=
\dot\Omega\,\mathcal C_q(K^{\mathrm T})
+\Omega\,D\mathcal C_q(K^{\mathrm T})[\dot K^{\mathrm T}].
\tag{34i}
$$

More importantly, the tangent can be propagated along the same graph edge by
differentiating eq 34d:

$$
\begin{aligned}
\dot{\mathcal C}_q'={}&\dot{\mathcal C}_q
+\mathscr I_q(\dot{\mathcal C}_{q-1};b,a)\\
&+\mathscr I_q(\mathcal C_{q-1};\dot b,a)
+\mathscr I_q(\mathcal C_{q-1};b,\dot a).
\end{aligned}
\tag{34j}
$$

Thus the HVP of an arbitrary-order RDM contraction has the same
output-sensitive scaling as the accepted RDM itself. It does not require
reconstructing the density independently at every string pair.

### 4.6 Reverse pullback of an arbitrary-order contraction

Suppose a scalar objective contains

$$
F_q
=
\left\langle W^{(q)},\Gamma^{(q)}\right\rangle.
\tag{34k}
$$

The determinant adjoint is

$$
\overline\Omega
=
\left\langle
W^{(q)},\mathcal C_q(K^{\mathrm T})
\right\rangle.
\tag{34l}
$$

Writing $A=K^{\mathrm T}$, the matrix adjoint follows from the same
$(q-1)$th compound level:

$$
\overline A_{ij}
=
\Omega
\sum_{I\ni i}\sum_{J\ni j}
(-1)^{\operatorname{pos}_I(i)+\operatorname{pos}_J(j)}
W^{(q)}_{I,J}
\mathcal C_{q-1}(A)_{I\setminus i,\,J\setminus j}.
\tag{34m}
$$

With $\overline K=\overline A^{\mathrm T}$ and $K=X^{-1}$, the pullback to
the occupied overlap block is

$$
\overline X
=
\overline\Omega\,\Omega K^{\mathrm T}
-K^{\mathrm T}\overline K K^{\mathrm T}.
\tag{34n}
$$

Equations 34l--34n show that reverse differentiation of a $q$th-order RDM
contraction needs the already available $q$th and $(q-1)$th compound levels,
not an explicitly constructed $(q+1)$th-order density. The reverse of the
edge recurrence in eq 34d provides the equivalent streamed implementation and
allows each tile-local hierarchy to be released after its consumer finishes.

For singular pairs, eq 34b is replaced by the corresponding complementary
minors of $X$. The contracted interpolation in Section 10.2 applies
componentwise to eqs 34b, 34i, and 34k. Hence the arbitrary-order formulation
does not introduce an inverse-based approximation at ill-conditioned pairs.

### 4.7 Representation lower bounds, not the production algorithm

This section distinguishes an implicit density from an explicitly requested
RDM. The explicit-output discussion below is only a lower-bound argument for
why the production algorithm must not materialize high-order densities. It is
not included in the production complexity derived in Section 9.

The phrase "low-scaling arbitrary-order density" refers to three distinct
computational problems. They must not be assigned the same complexity.

Let

$$
n = \text{number of occupied orbitals in one spin string},
$$

$$
N_q=\binom{n}{q},
$$

and let $P$ denote the number of touched ordered unique-spin-string pairs.
The structure-to-string contraction determines $P$. In the dense worst case,

$$
P=U^2,
$$

but a selected structure space can give a substantially smaller touched-pair
support. Determinants appearing in a VB-structure expansion are not an
additional computational layer in the proposed algorithm: the contraction is
formed directly between unique spin strings.

#### 4.7.1 Implicit representation of every RDM order

For one regular pair, the complete hierarchy

$$
\Gamma^{(q)}=\Omega\mathcal C_q(K^{\mathrm T}),
\qquad q=0,1,\ldots,n,
$$

is determined exactly by only

$$
(\Omega,K).
$$

It is therefore unnecessary to materialize any compound level merely to
carry the pair state across the string graph. Under one site substitution,

$$
X'=X+uv^{\mathrm T},
$$

the determinant lemma and Sherman--Morrison relation give

$$
d=1+v^{\mathrm T}Ku,
$$

$$
\Omega'=d\Omega,
$$

$$
K'=K-\frac{(Ku)(v^{\mathrm T}K)}{d}.
$$

Maintaining an explicit dense $K$ therefore costs

$$
\Theta(n^2)
$$

per graph edge and requires

$$
\Theta(n^2)
$$

pair-local storage, independently of the requested RDM order. For $P$ touched
pairs, the regular-pair propagation target is consequently

$$
T_{\mathrm{implicit}}=\Theta(Pn^2),
\qquad
M_{\mathrm{implicit}}=\Theta(n^2).
\tag{34o}
$$

This is the first genuine scaling reduction: independent pair factorization
costs $O(Pn^3)$, whereas the graph traversal removes one power of $n$ for the
entire arbitrary-order hierarchy. The tangent pair

$$
(\dot\Omega,\dot K)
$$

and the reverse adjoints

$$
(\overline\Omega,\overline K)
$$

obey differentiated rank-one relations with the same $\Theta(n^2)$ edge
cost. Thus accepted densities, orbital gradients, and matrix-free HVPs can
share one asymptotic propagation bound.

Equation 34o assumes regular pairs. A production algorithm must maintain the
same target for ill-conditioned pairs by a certified rank-one update of a
rank-revealing factorization. Recomputing an SVD gives an $O(n^3)$ anchor;
rank-one SVD updating has an $O(n^2)$ target. A full refactorization remains a
numerical recovery operation, not the nominal pair cost.

#### 4.7.2 Explicit materialization has an unavoidable output bound

A full occupied-index order-$q$ transition density contains

$$
N_q^2=\binom{n}{q}^2
$$

numbers. Any algorithm that explicitly writes this object for every touched
pair therefore satisfies

$$
T_{\mathrm{explicit}}
\geq
\Omega(PN_q^2).
\tag{34p}
$$

No Woodbury, interpolation, RI, or parallel implementation can reduce this
lower bound while retaining the explicit output.

The insertion in eq 34e has the factorized exterior form

$$
\mathscr I_q(\mathcal C_{q-1};b,a)
=
L_q(b)\,\mathcal C_{q-1}\,L_q(a)^{\mathrm T},
\tag{34q}
$$

where every row of the exterior-insertion matrix $L_q$ has at most $q$
nonzero entries. Sparse--dense association reduces the materialization cost
from a literal $q^2$ sum for every output element to

$$
O\!\left(
qN_qN_{q-1}+qN_q^2
\right).
\tag{34r}
$$

For fixed physical order and $q\le n/2$, eq 34r is output-linear up to the
order-dependent factor $q$. Hence the best possible explicit algorithm is

$$
T_{\mathrm{explicit}}
=
\Theta(PN_q^2)
$$

with respect to $n$ at fixed $q$. This is still exponentially unfavorable as
$q$ approaches the middle of the occupied space because the requested output
itself is exponentially large.

The global active-orbital RDM has an analogous output bound. Its exact sum over
arbitrary dense pair weights is not generally compressible: a sum of
compound matrices need not remain one compound matrix. Consequently, an
explicit high-order global RDM should be formed only when it is itself the
requested observable.

#### 4.7.3 Contracted densities permit the qualitative reduction

Energy, gradient, and HVP evaluation usually need a scalar or a low-rank
adjoint contraction, not the explicit $N_q\times N_q$ density. For a general
unstructured weight tensor $W^{(q)}$, reading its local occupied block already
requires

$$
\Omega(N_q^2)
$$

work per pair. The exponent can be removed only when the operator or adjoint
also has a factorized representation.

For example, suppose the order-$q$ weight is a sum of $R_q$ decomposable
exterior products,

$$
W^{(q)}
=
\sum_{r=1}^{R_q}
\mathcal C_q(L_r)\,
\mathcal C_q(R_r)^{\mathrm T}.
\tag{34s}
$$

Cauchy--Binet gives the contraction without constructing the RDM,

$$
\left\langle
W^{(q)},\mathcal C_q(K^{\mathrm T})
\right\rangle
=
\sum_{r=1}^{R_q}
\det\!\left(L_r^{\mathrm T}K^{\mathrm T}R_r\right).
\tag{34t}
$$

For dense factors, eq 34t costs

$$
O\!\left[
R_q\left(qn^2+q^3\right)
\right]
\tag{34u}
$$

per independently contracted pair. At fixed $q$, this is

$$
O(R_qn^2),
$$

rather than $O(n^{2q})$. The same small matrices provide analytic forward and
reverse derivatives.

The two-electron case illustrates why integral factorization is essential.
With an unstructured exact four-index tensor, a general pair contraction
retains an $O(n^4)$ term. RI changes the consumer rather than the pair-density
identity and gives the graph-traversal target

$$
T_{\mathrm{RI},q=2}
=
O\!\left(
A N_{\mathrm{aux}}n^3
+
P N_{\mathrm{aux}}n^2
\right),
\tag{34v}
$$

where $A$ is the number of independently factorized anchors. The first term
builds anchor auxiliary channels; the second propagates them by rank-one
updates. COSX, THC, or an order-specific tensor factorization plays the same
role for other consumers. Without such a factorization, Woodbury alone cannot
remove the dense interaction-tensor exponent.

The production objective is therefore not to retain every compound matrix.
It is:

1. propagate the exact implicit state $(\Omega,K)$ and its tangent/adjoint at
   $O(n^2)$ per touched pair;
2. expose order-$q$ density elements only on demand;
3. fuse factorized Hamiltonian, gradient, and HVP consumers into the same tile
   traversal; and
4. materialize a high-order RDM only when that RDM is the requested output.

The compound hierarchy of Sections 4.4--4.6 is therefore a mathematical
identity, not a production data structure. The low-scaling implementation
stores only the implicit pair state and evaluates the factorized contraction
and its derivatives directly.

The Woodbury contraction algorithm is specifically a factorized-integral
algorithm. It applies to RI and to future COSX/THC-like representations. The
exact four-index integral implementation remains a numerical reference path;
it is not duplicated inside the Woodbury implementation and no reduced
asymptotic scaling is claimed for it.

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

### 5.1 Elimination of the repeated packed-pair projection

Let $P$ denote a packed active-orbital pair and let $B_{QP}$ be the
metric-whitened RI factor.  The packed first-cofactor projection is

$$
x_P
=
\sum_{r l:\,P=\operatorname{pack}(R_r,L_l)}
\Omega K_{lr}.
\tag{37a}
$$

A conventional implementation first forms $x$, then evaluates

$$
g=Bx,
\qquad
y=B^{\mathrm T}g.
\tag{37b}
$$

The first product is redundant when the accepted pair has already generated

$$
A^Q=K M^Q,
\qquad
M^Q_{rl}=B_{Q,\operatorname{pack}(R_r,L_l)}.
$$

Indeed,

$$
g_Q
=
\sum_{rl}
B_{Q,\operatorname{pack}(R_r,L_l)}\Omega K_{lr}
=
\Omega\operatorname{tr}(A^Q).
\tag{37c}
$$

Therefore the accepted traversal exports $g_Q$ while $A^Q$ is live and
computes only the required backprojection

$$
y_P
=
\sum_Q B_{QP}g_Q.
\tag{37d}
$$

This removes one $B x$ product for every regular pair.  The auxiliary vector
is row-local scratch and is released immediately after eq 37d; it is not
stored for a complete tile or for the $U^2$ pair table.  Consequently the
additional memory is

$$
O(WN_{\mathrm{aux}}),
$$

where $W$ is the number of live traversal workers, rather than
$O(T^2N_{\mathrm{aux}})$ or $O(U^2N_{\mathrm{aux}})$.  For nullity-one pairs,
eq 37c is unavailable because $Omega=0$ although the first cofactor can be
nonzero; those pairs retain the exact cofactor projection in eq 37a.

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

The auxiliary first-cofactor feature and its tangent follow from the same
accepted and directional channels:

$$
g_Q
=
\Omega\operatorname{tr}(A^Q),
\qquad
\dot g_Q
=
\dot\Omega\operatorname{tr}(A^Q)
+\Omega\operatorname{tr}(\dot A^Q).
\tag{65a}
$$

Consequently, the directional packed-pair image required by the
opposite-spin HVP is

$$
\dot y
=
B^{\mathrm T}\dot g
+\dot B^{\mathrm T}g.
\tag{65b}
$$

Equations 65a and 65b replace the three forward products
$Bx$, $\dot Bx$, and $B\dot x$ followed by two backprojections.  They are
evaluated while $A^Q$ and $\dot A^Q$ are already live in the regular-pair
same-spin response.  The final packed image occupies the same bounded tile
storage as the conventional route; no auxiliary feature table is retained.

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

For the full alpha/beta calculation, let $P_\sigma$ be the number of touched
ordered unique-string pairs, $a_\sigma$ the number of independently
factorized traversal anchors, and $n_\sigma$ the number of active electrons
of spin $\sigma$. Let $T_{\mathrm{str}}$ denote the structure-to-unique-string
matrix contractions, including the opposite-spin partner images. The
contract-first RI target is

$$
T_{\mathrm{RI}}
=
T_{\mathrm{str}}
+
\sum_{\sigma\in\{\alpha,\beta\}}
\left[
O\!\left(a_\sigma N_{\mathrm{aux}}n_\sigma^3\right)
+
O\!\left(P_\sigma N_{\mathrm{aux}}n_\sigma^2\right)
\right].
\tag{76a}
$$

The same bound applies to the direct RI pullback. For one same-spin pair,

$$
\overline M^Q
=
w\Omega
\left(j_QK-A^QK\right)^{\mathrm T}
$$

is scattered directly to the RI-factor adjoint. For an opposite-spin pair,

$$
\overline M^Q
=
\Omega\lambda_QK^{\mathrm T},
$$

where $\lambda_Q$ is obtained from the partner-spin structure contraction.
Neither expression contains a four-index density or a packed pair-pair
adjoint. Since a generic dense occupied RI block contains
$N_{\mathrm{aux}}n_\sigma^2$ numbers, the edge term in eq 76a is also the
information-theoretic lower bound for a generic RI consumer. The proposed
algorithm is therefore asymptotically optimal at the pair-kernel level unless
the auxiliary factors have additional sparsity or tensor factorization.

For a block of $b$ HVP directions, the base channels are shared and every
direction carries its own tangent channels. The corresponding target is

$$
T_{\mathrm{HVP}}(b)
=
T_{\mathrm{str}}(b)
+
O\!\left[
(b+1)
\sum_{\sigma}
\left(
a_\sigma N_{\mathrm{aux}}n_\sigma^3
+
P_\sigma N_{\mathrm{aux}}n_\sigma^2
\right)
\right].
\tag{76b}
$$

Higher cofactor orders that appear after formally differentiating a
transition RDM do not change eq 76b. The implementation differentiates the
already contracted scalar functional. More generally, propagating a Taylor
jet through derivative order $p$ requires only convolutions of the two-index
states

$$
\Omega,
\quad K,
\quad A^Q,
\quad D^Q,
\quad E^Q,
$$

and has the fixed-order bound

$$
O\!\left(
p^2P_\sigma N_{\mathrm{aux}}n_\sigma^2
\right).
\tag{76c}
$$

Thus gradient, HVP, and any fixed derivative order retain the same powers of
$n_\sigma$ and $N_{\mathrm{aux}}$. Third- and fourth-order RDMs are artifacts
of expanding the derivative before contraction and are never production
intermediates.

The corresponding streamed working memory for auxiliary block width $q_b$
is

$$
M_{\mathrm{work}}
=
O\!\left[
(b+1)q_b n_\sigma^2
\right]
$$

per traversal worker, in addition to the requested RI-factor/orbital adjoint
and bounded structure tiles. There is no $O(U^2N_{\mathrm{aux}}n^2)$ pair
cache and no $O(m^4)$ active pair-pair adjoint.

For comparison, an unstructured exact two-electron contraction retains

$$
\Theta(P_\sigma n_\sigma^4)
$$

work in the generic case. Updating $K$ by Woodbury does not change that
leading exponent. The exact-integral implementation is therefore excluded
from the low-scaling Woodbury algorithm and retained only as a numerical
reference.

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

### 9.1 Certified-component census at the initial orbital point

Topological connectivity alone does not determine the useful anchor count.
Inverse-based Woodbury propagation is admitted only when the endpoint overlap
inverse satisfies the same condition and backward-error certificate used by
the production pair derivatives.  Let

$$
P=P_{\mathrm r}+P_{\mathrm i}+P_{\mathrm s},
$$

where $P_{\mathrm r}$ is the number of certified regular pairs,
$P_{\mathrm i}$ is the number of full-rank pairs requiring the exact
interpolated representation, and $P_{\mathrm s}$ is the number of singular
pairs.  Let $A_{\mathrm r}$ be the number of connected components in the
subgraph induced by the $P_{\mathrm r}$ certified pairs.  This is the
topological lower bound on the number of independently factorized regular
anchors; it is not the number of re-anchors produced by an arbitrary linear
traversal.

Hanhai25 Slurm array job `260198`, using revision `e52eff0`, evaluated every
ordered same-spin pair at the input orbital point with the production
certificate.  Every tested unique-string space is the complete fixed-spin
space and its unfiltered substitution graph has one connected component.

| System | Spin | $U$ | $P_{\mathrm r}/P$ | $A_{\mathrm r}$ | Largest certified component | $P_{\mathrm s}$ |
|---|---:|---:|---:|---:|---:|---:|
| 241 | both | 20 | 242 / 400 | 1 | 242 | 0 |
| 7975 | both | 20 | 334 / 400 | 1 | 334 | 0 |
| 240 | both | 70 | 2,402 / 4,900 | 1 | 2,402 | 0 |
| MnF2 | alpha | 8 | 30 / 64 | 2 | 29 | 0 |
| MnF2 | beta | 28 | 500 / 784 | 1 | 500 | 0 |
| FeCl2 | alpha | 8 | 20 / 64 | 4 | 17 | 0 |
| FeCl2 | beta | 56 | 360 / 3,136 | 10 | 71 | 0 |
| YAMSAI | both | 35 | 723 / 1,225 | 1 | 723 | 0 |
| CERRAS | both | 462 | 29,423 / 213,444 | 3 | 21,256 | 0 |
| LOFLEA | both | 924 | 107,180 / 853,776 | 9 | 79,504 | 14 |

The one-electron F2 spin sectors contain only two strings.  A one-site change
has rank equal to the occupied-overlap dimension, so the production update
correctly treats all four ordered pairs as direct evaluations; Woodbury has no
algebraic advantage at $n=1$.

The census separates two effects that a linear path obscures.  First, the
certified regular subgraph remains extremely well connected: CERRAS requires
only three regular anchors and LOFLEA only nine.  Anchor proliferation is
therefore not the limiting issue inside the regular branch.  Second, only
$13.78\%$ of CERRAS pairs and $12.55\%$ of LOFLEA pairs belong to that branch.
Most pairs are full rank but too ill conditioned for inverse-based fourth-order
derivatives and must retain the exact interpolated representation.

The appropriate cost model is consequently

$$
T
=
A_{\mathrm r}C_{\mathrm A}^{\mathrm r}
+(P_{\mathrm r}-A_{\mathrm r})C_{\mathrm E}^{\mathrm r}
+P_{\mathrm i}C_{\mathrm i}
+P_{\mathrm s}C_{\mathrm s}.
$$

If only the regular branch is accelerated, if
$C_{\mathrm A}^{\mathrm r}/C_{\mathrm E}^{\mathrm r}\simeq n$, and if the
interpolated and singular costs are approximated by the direct-pair cost, the
best possible whole-pair speedup is bounded approximately by

$$
\mathcal S_{\mathrm{inverse\text{-}only}}
\lesssim
\left(1-f_{\mathrm r}+\frac{f_{\mathrm r}}{n}\right)^{-1},
\qquad
f_{\mathrm r}=\frac{P_{\mathrm r}}{P}.
$$

This gives only about $1.13\times$ for CERRAS and $1.12\times$ for LOFLEA,
before traversal and scheduling overhead.  It explains why improving only the
regular Woodbury path cannot produce a large end-to-end speedup for the large
HAO cases.  A low-scaling production algorithm must also reduce
$C_{\mathrm i}$, either by propagating a stable contracted representation
through the dangerous subspace or by lowering the cost of every exact
interpolation node with an integral factorization such as THC.

### 9.2 Dynamic Woodbury-core propagation census

The pointwise dangerous-mode count does not by itself prove that a small
inverse-free core can be maintained along the string-pair graph.  Replacing
each overlap block independently by an SVD-clipped stable completion may make
the difference between consecutive completions high rank and therefore loses
the graph advantage.  The relevant state is instead

$$
X=A+UV^{\mathrm T},
\qquad
K=A^{-1},
$$

where the base $A$ is certified regular and the columns of $U,V$ contain only
updates that cannot yet be absorbed safely.  A graph edge adds one rank-one
term.  After numerical rank compression, define the small Woodbury matrix

$$
G=I+V^{\mathrm T}KU.
$$

Let $G=P\Sigma Q^{\mathrm T}$ and partition the right singular vectors as
$Q=[Q_{\mathrm s},Q_{\mathrm d}]$.  For increasing retained dimension, form

$$
\begin{aligned}
A'&=A+(UQ_{\mathrm s})(VQ_{\mathrm s})^{\mathrm T},\\
U'&=UQ_{\mathrm d},\\
V'&=VQ_{\mathrm d}.
\end{aligned}
\tag{76a}
$$

Since $Q_{\mathrm s}Q_{\mathrm s}^{\mathrm T}+
Q_{\mathrm d}Q_{\mathrm d}^{\mathrm T}=I$, this split is exact:

$$
X=A'+U'V'^{\mathrm T}.
\tag{76b}
$$

The smallest retained dimension for which $A'$ and its block-Woodbury inverse
pass the production condition and backward-error certificate defines the
propagated dangerous-core rank.  No inverse of the retained core is formed.
This procedure is a diagnostic construction; its SVDs identify whether the
representation exists and are not the proposed production update.

Revision `7f022eb` replayed this state over exact fixed-weight combination
Gray paths on Hanhai25 in Slurm jobs `260235`--`260243`.  Every consecutive
overlap change had numerical rank one.  No traversal had a topology break or
required a full numerical re-anchor, and every propagated base retained the
production regularity certificate.

| System | Spin | Pairs | Mean propagated $q$ | Fraction $q\leq2$ | Fraction $q\leq3$ | Fraction $q\leq5$ | Maximum $q$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 241 | both | 400 | 0.443 | 1.000 | 1.000 | 1.000 | 2 |
| 7975 | both | 400 | 0.168 | 1.000 | 1.000 | 1.000 | 2 |
| 240 | both | 4,900 | 0.682 | 0.981 | 0.9996 | 1.000 | 4 |
| MnF2 | alpha | 64 | 0.875 | 0.922 | 0.984 | 1.000 | 4 |
| MnF2 | beta | 784 | 0.402 | 1.000 | 1.000 | 1.000 | 2 |
| FeCl2 | alpha | 64 | 1.188 | 0.859 | 0.984 | 1.000 | 4 |
| FeCl2 | beta | 3,136 | 1.520 | 0.892 | 1.000 | 1.000 | 3 |
| YAMSAI | both | 1,225 | 0.487 | 0.993 | 1.000 | 1.000 | 3 |
| CERRAS | both | 213,444 | 1.938 | 0.696 | 0.862 | 0.993 | 6 |
| LOFLEA | both | 853,776 | 2.042 | 0.656 | 0.840 | 0.996 | 6 |

The pointwise SVD census is more optimistic than dynamic propagation because
the latter preserves one evolving Woodbury base.  Nevertheless, over $99\%$
of the large-system paths require at most a five-dimensional retained core,
and the mean core dimension remains approximately two.  The production
kernel should therefore support a general small core rather than special
cases only for $q\leq3$.

After 213,443 rank-one edges, the maximum CERRAS reconstruction residual was
$1.19\times10^{-12}$.  After 853,775 edges, the maximum LOFLEA residual was
$5.32\times10^{-12}$.  The corresponding small cases remained near
$10^{-14}$.  These results establish representational coverage but do not yet
establish contraction accuracy or speed.  The next validation must compare
small-core Hamiltonian, gradient, and HVP contractions with the existing exact
interpolation reference, followed by an RI/THC timing comparison.

### 9.3 Inverse-free RI contraction and directional adjoint

The propagated representation is now used directly by the RI contraction,
not only as a coverage diagnostic. Write

$$
X=A+UV^{\mathrm T},
\qquad
K=A^{-1},
\qquad
G=I+V^{\mathrm T}KU,
$$

where only the certified base $A$ is inverted. The physical overlap $X$ and
the dangerous core $G$ may be singular. Their determinant and first cofactor
are evaluated as

$$
\det X=\det A\det G,
$$

$$
C_1(X)
=
\det A
\left[
\det(G)K
-KU C_1(G)^{\mathrm T}V^{\mathrm T}K
\right]^{\mathrm T}.
$$

For one RI transition block $M^Q$, define

$$
C^Q=KM^Q,
\qquad
P=KU,
\qquad
R^Q=V^{\mathrm T}C^Q,
$$

$$
B^Q=R^QP,
\qquad
D^Q=R^QC^QP.
$$

The unnormalized same-spin contribution is

$$
\begin{aligned}
F_Q
=\det A\Bigg[
&\det(G)\frac{\operatorname{tr}(C^Q)^2-
\operatorname{tr}((C^Q)^2)}{2}\\
&-\operatorname{tr}(C^Q)\langle C_1(G),B^Q\rangle
+\langle C_1(G),D^Q\rangle\\
&+\left\langle C_2(G),\wedge^2 B^Q\right\rangle
\Bigg].
\end{aligned}
$$

This expression contains cofactors only in the retained $q\times q$ core. It
is therefore well defined when $G$ is singular and never forms the physical
inverse $X^{-1}$ or a fourth-order occupied interaction tensor.

For a directional perturbation, the exact tangent begins with

$$
\dot K=-K\dot A K,
\qquad
\dot P=\dot K U,
\qquad
\dot G=V^{\mathrm T}\dot P,
$$

$$
\dot C^Q=\dot K M^Q+K\dot M^Q.
$$

The small-core polynomial is differentiated without an inverse of $G$:

$$
\dot{\det G}=\langle C_1(G),\dot G\rangle,
\qquad
\dot C_1(G)=D C_1(G)[\dot G],
$$

$$
\dot C_2(G)=D C_2(G)[\dot G].
$$

Forward differentiation of the contracted scalar followed by reverse
propagation gives the exact directional overlap adjoint
$\dot{\overline X}$ and RI-transition adjoint
$\dot{\overline M}^{Q}$. Thus the same core supplies the accepted energy,
gradient, and HVP quantities for regular, ill-conditioned, and singular
pairs. No finite-difference derivative or interpolation node is used in the
production contraction.

For fixed small $q$, the scalar graph update retains the target cost

$$
O(N_{\mathrm{aux}}n^2)
$$

per pair edge. The present full overlap-adjoint and directional-adjoint
implementation costs

$$
O(N_{\mathrm{aux}}n^3)
$$

per evaluated pair because it returns dense transition and overlap gradients,
but it removes the former occupied-pair interaction construction and its
$n^4$--$n^6$ polynomial-cofactor work. Further reduction requires contracting
these adjoints directly into their downstream orbital weights or replacing
the pair sweep by a biorthogonal action; it cannot be obtained by another
cache around the old high-order tensor.

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
