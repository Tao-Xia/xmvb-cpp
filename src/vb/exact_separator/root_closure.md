# Exact Separator Root Closure Formulas

> Status note (2026-04-05):
> the current production `one_leaf_star` path already uses the validated
> specialized one-leaf boundary-message implementation, and the generic
> rooted-tree Hamiltonian path is now wired through the typed recursive
> Hamiltonian bundle. The formulas in this note remain the mathematical
> reference for that recursive closure and for the next representation-level
> optimization pass.
>
> Investigation update (2026-04-06):
> two additional implementation hypotheses have now been ruled out on the
> concrete failing pair
> `test_molecule/C6H6_full.xmi (left_structure=1, right_structure=0)`.
>
> 1. Replacing the typed subtree Hamiltonian message by the exact generic joint
>    deleted-minor subtree message did **not** fix the numerical error. So the
>    main defect is not the child recursive message alone; the blocker remains
>    the root closure.
> 2. Replacing the current root `frontier + complement` merge by
>    `frontier + local-interface-payload + transform_interface_block_to_front`
>    made the result worse, including overlap. So the root fix is **not**
>    another copy of the subtree transform.
>
> The old append-only root merge has now been replaced in code by a
> root-specific closure that inserts the local root remainder block between the
> selected root interface and the frontier body before the final Hamiltonian
> contractions. This fixes the known failing one-leaf star case
> `test_molecule/C6H6_full.xmi (left_structure=1, right_structure=0)` down to
> machine precision.
>
> The formulas below remain the mathematical reference for understanding that
> closure and for future representation-level optimization passes:
>
> - degree-`1`: exact `U_F/U_R -> U_{F \otimes R}` closure;
> - opposite-spin: exact mixed `X_F/X_R -> X_{F \otimes R}` closure;
> - same-spin: exact degree-`2` `G_F/G_R` closure with the
>   antisymmetrized `degree-1 x degree-1 -> degree-2` map.

## 1. Goal

This note isolates the exact formulas that must be correct before any further
`component_tree` implementation work continues.

The current situation is now mathematically clear:

1. degree-`0` overlap closure is exact;
2. degree-`1` one-electron closure is exact only if the full one-spin
   degree-`1` boundary payload is kept, not just the `(1,1)` first-cofactor
   matrix;
3. opposite-spin is **not** determined by the summed alpha/beta first-cofactor
   matrices alone once local term pairs have been aggregated;
4. same-spin degree-`2` is **not** determined by a scalar-only closure; it
   requires the full one-spin degree-`2` boundary message.

So the remaining work is not "fix a sign in the old payload". The remaining
work is to represent the correct exact boundary objects.

---

## 2. One-Spin Degree-0/1/2 Objects

Consider one spin channel on an ordered block \(X\) with:

\[
n_r(X) = \text{number of right/row orbitals}, \qquad
n_c(X) = \text{number of left/column orbitals}.
\]

Write:

\[
S_X
\]

for the closed overlap determinant of the square block.

For degree `1`, the exact one-spin boundary data is not just one first-cofactor
matrix. The minimal exact payload is

\[
m_X^{(1)} = \left(S_X,\; r_X,\; c_X,\; C_X\right),
\]

where

\[
r_X(p)
\]

is the exact row-open minor amplitude with one deleted row orbital \(p\),

\[
c_X(q)
\]

is the exact col-open minor amplitude with one deleted column orbital \(q\),

and

\[
C_X(p,q)
\]

is the exact first-cofactor amplitude with deleted row orbital \(p\) and
deleted column orbital \(q\).

For degree `2`, the minimal exact one-spin boundary object is

\[
m_X^{(2)} = \left(S_X,\; r_X,\; c_X,\; C_X,\; \Gamma_X\right),
\]

where

\[
\Gamma_X\!\big((p_1,p_2),(q_1,q_2)\big)
\]

is the exact antisymmetric second-cofactor tensor.

The key point is:

- degree `1` requires `row-open + col-open + first-cofactor`;
- degree `2` requires `degree-1` plus the full second-cofactor object.

Keeping only the balanced `(1,1)` or `(2,2)` sectors is not exact under merge.

---

## 3. Exact Degree-1 One-Spin Merge

Let \(A\) and \(B\) be two ordered one-spin blocks merged in block order. In
the exact separator convention used by the validated dense one-leaf algebra,
the merged degree-`1` payload is

\[
m_{A \otimes B}^{(1)} =
\left(S_{A \otimes B},\; r_{A \otimes B},\; c_{A \otimes B},\; C_{A \otimes B}\right).
\]

The overlap is

\[
S_{A \otimes B} = S_A S_B.
\]

The row-open vector obeys

\[
r_{A \otimes B}
=
(-1)^{n_c(B)} S_B\, r_A
+
(-1)^{n_r(A)+n_c(A)} S_A\, r_B.
\]

The col-open vector obeys

\[
c_{A \otimes B}
=
(-1)^{n_r(B)} S_B\, c_A
+
(-1)^{n_r(A)+n_c(A)} S_A\, c_B.
\]

The first-cofactor matrix obeys

\[
C_{A \otimes B}
=
S_B\, C_A
+
(-1)^{n_r(A)+n_c(A)} S_A\, C_B
\]

\[
-
(-1)^{n_r(A)+n_c(A)+n_c(B)}\,
r_A c_B^{\mathsf T}
-
(-1)^{n_r(A)+n_c(A)+n_r(B)}\,
r_B c_A^{\mathsf T}.
\]

This is the exact degree-`1` Leibniz rule in the current block-order
convention.

### 3.1 Why the old root closure was wrong

The old root closure tried to use only

\[
C_{A \otimes B} \approx S_B C_A + S_A C_B.
\]

That drops the two outer-product terms

\[
r_A c_B^{\mathsf T}, \qquad r_B c_A^{\mathsf T},
\]

which are not optional. They are part of the exact first-cofactor.

This is exactly why the previous rooted-tree `degree1_first_cofactor_debug`
showed:

1. overlap exact;
2. one-electron overlap-root closure exact only in the scalar prototype;
3. first-cofactor entries missing or sign-wrong in the rooted-tree path.

After reinstating the full degree-`1` algebra, the reconstructed
first-cofactor matrix and the one-electron contraction match the exact
one-leaf result again.

---

## 4. Opposite-Spin Exact Formula

For one exact determinant pair, opposite-spin is bilinear in the alpha and
beta first-cofactor matrices:

\[
E_{\alpha\beta}
=
\sum_{p,a,q,b}
(q b \mid p a)\,
C^\alpha(p,a)\,
C^\beta(q,b).
\]

Equivalently, with the bilinear form \(B\),

\[
E_{\alpha\beta} = B(C^\alpha, C^\beta).
\]

If we had one single determinant pair, that would be enough.

The problem starts after aggregating many local term pairs.

Let \(t\) index the exact local term pairs and let \(w_t\) be the exact signed
coefficient of term \(t\). The exact total opposite-spin value is

\[
E_{\alpha\beta}^{\text{exact}}
=
\sum_t w_t\, B\!\left(C_t^\alpha, C_t^\beta\right).
\]

If one first sums the marginals

\[
\bar C^\alpha = \sum_t w_t C_t^\alpha,
\qquad
\bar C^\beta = \sum_t w_t C_t^\beta,
\]

and then contracts,

\[
B(\bar C^\alpha, \bar C^\beta)
=
\sum_t w_t^2 B(C_t^\alpha, C_t^\beta)
+
\sum_{t \neq u} w_t w_u B(C_t^\alpha, C_u^\beta),
\]

which is **not** the same object.

So after local aggregation, opposite-spin is **not** determined by the two
summed first-cofactor matrices alone.

### 4.1 Minimal exact mixed object

The minimal exact opposite-spin aggregate is the mixed tensor

\[
M_{(p,a),(q,b)}
=
\sum_t w_t\,
C_t^\alpha(p,a)\,
C_t^\beta(q,b).
\]

Then the exact opposite-spin value is linear in \(M\):

\[
E_{\alpha\beta}^{\text{exact}}
=
\sum_{p,a,q,b}
(q b \mid p a)\,
M_{(p,a),(q,b)}.
\]

This is the exact object that survives local aggregation.

### 4.2 Basis-indexed form

Let

\[
\{\Psi^\sigma_\mu\}_\mu
\]

be one exact degree-`1` boundary basis for one spin channel. This basis must
span all exact degree-`1` content:

1. row-open;
2. col-open;
3. first-cofactor.

For each exact local term pair \(t\), write

\[
m_t^{(1),\sigma}
=
\sum_\mu a^\sigma_{t\mu}\, \Psi^\sigma_\mu.
\]

The exact aggregated one-spin degree-`1` message is then

\[
A^\sigma_\mu
=
\sum_t w_t\, a^\sigma_{t\mu}.
\]

This is enough for overlap and one-electron, because they are linear in the
one-spin degree-`1` message.

But opposite-spin is not linear in one spin alone. It depends on the mixed
alpha/beta coefficient matrix

\[
X_{\mu\nu}
=
\sum_t w_t\,
a^\alpha_{t\mu}\,
a^\beta_{t\nu}.
\]

The exact opposite-spin value is then

\[
E_{\alpha\beta}^{\text{exact}}
=
\sum_{\mu,\nu}
K_{\mu\nu}^{\alpha\beta}\,
X_{\mu\nu},
\]

where

\[
K_{\mu\nu}^{\alpha\beta}
\]

is the ERI contraction kernel between the alpha degree-`1` basis function
\(\Psi^\alpha_\mu\) and the beta degree-`1` basis function \(\Psi^\beta_\nu\).

So in basis form:

- one-electron uses the marginal vector \(A^\sigma\);
- opposite-spin uses the mixed matrix \(X\).

### 4.2 Boundary-message version

Suppose the alpha and beta degree-`1` first-cofactor matrices are expanded in
boundary basis functions:

\[
C_t^\alpha(p,a) = \sum_\mu A^\alpha_{t\mu}\, \Phi^\alpha_\mu(p,a),
\qquad
C_t^\beta(q,b) = \sum_\nu A^\beta_{t\nu}\, \Phi^\beta_\nu(q,b).
\]

Then

\[
M_{(p,a),(q,b)}
=
\sum_{\mu,\nu}
X_{\mu\nu}\,
\Phi^\alpha_\mu(p,a)\,
\Phi^\beta_\nu(q,b),
\]

with

\[
X_{\mu\nu}
=
\sum_t w_t\, A^\alpha_{t\mu} A^\beta_{t\nu}.
\]

This shows the real exact opposite-spin boundary object:

\[
X_{\mu\nu},
\]

the mixed alpha/beta coefficient matrix over the degree-`1` boundary basis.

This is what the current rooted-tree implementation still does **not** carry.

### 4.3 Consequence

The current rooted-tree path can now rebuild the exact marginal first-cofactor
matrices \(\bar C^\alpha\) and \(\bar C^\beta\), so one-electron is correct.

But opposite-spin still misses the exact mixed coefficient object

\[
X_{\mu\nu}.
\]

Therefore:

- fixing the alpha first-cofactor alone is not enough;
- fixing the beta first-cofactor alone is not enough;
- fixing the final root contraction on \(\bar C^\alpha,\bar C^\beta\) is still
  not enough.

The mixed correlation must be represented explicitly.

### 4.4 Correction to the earlier simplified statement

Some earlier design notes said:

\[
\text{opposite-spin needs only } \bar C^\alpha \text{ and } \bar C^\beta.
\]

That statement is true only in the special regime where the exact local term
index \(t\) is still carried all the way to the final opposite-spin
contraction.

In the current rooted-tree / boundary-message direction, that is **not** what
we do. We aggregate local term pairs into subtree messages first. After that
aggregation, the exact opposite-spin channel is no longer recoverable from the
two marginals alone.

So for the actual intended `component_tree` implementation:

\[
\text{opposite-spin exactness } \Longrightarrow \text{ carry } X_{\mu\nu}.
\]

---

## 5. Same-Spin Exact Degree-2 Formula

For one spin channel \(\sigma\), the exact same-spin contribution is

\[
E_{\sigma\sigma}
=
\sum_{p_1 < p_2}
\sum_{q_1 < q_2}
\Big[
(p_1 q_1 \mid p_2 q_2)
-
(p_1 q_2 \mid p_2 q_1)
\Big]\,
\Gamma^\sigma\!\big((p_1,p_2),(q_1,q_2)\big).
\]

So same-spin is linear in the exact second-cofactor tensor \(\Gamma^\sigma\).

### 5.1 Exact merge law

For two one-spin blocks \(A\) and \(B\), the degree-`2` message satisfies

\[
m^{(2)}_{A \otimes B}
=
S_B\, m_A^{(2)}
+
S_A\, m_B^{(2)}
+
\mathcal{S}\!\left(m_A^{(1)}, m_B^{(1)}\right),
\]

where

\[
\mathcal{S}
\]

is the exact antisymmetrized degree-`1` to degree-`2` merge.

At the purely second-cofactor level this can be written schematically as

\[
\Gamma_{A \otimes B}
=
S_B\, \Gamma_A
+
S_A\, \Gamma_B
+
\operatorname{Alt}\!\left(C_A \boxtimes C_B\right),
\]

where \(\operatorname{Alt}\) means exact row/column antisymmetrization together
with the separator block-order parity.

The important point is not the outer notation but the structural content:

1. degree-`2` depends on the full degree-`1` payload;
2. degree-`2` cannot be recovered from one scalar per spin;
3. the correct same-spin message is a full degree-`2` boundary tensor.

### 5.2 Why the current same-spin closure is still wrong

The current rooted-tree same-spin path still tries to collapse too early.

It keeps:

- overlap;
- some first-order sectors;
- one final same-spin scalar contraction.

But the exact merge requires the full

\[
\Gamma^\sigma
\]

object, not just its final contracted scalar. In particular, the merge must be
able to represent all exact degree-`2` contributions generated from the
degree-`1` payload under

\[
\mathcal{S}\!\left(m_A^{(1)}, m_B^{(1)}\right).
\]

So the missing same-spin closure is not a numerical tweak. It is a missing
representation.

### 5.3 Basis-indexed degree-2 form

Let

\[
\{\Phi^\sigma_\nu\}_\nu
\]

be one exact antisymmetrized degree-`2` boundary basis for one spin channel.

For each exact local term pair \(t\), expand

\[
m_t^{(2),\sigma}
=
\sum_\nu g^\sigma_{t\nu}\, \Phi^\sigma_\nu.
\]

Then the exact aggregated degree-`2` message is the coefficient vector

\[
G^\sigma_\nu
=
\sum_t w_t\, g^\sigma_{t\nu}.
\]

The same-spin scalar is linear in \(G^\sigma\):

\[
E_{\sigma\sigma}^{\text{exact}}
=
\sum_\nu
L^\sigma_\nu\, G^\sigma_\nu,
\]

where \(L^\sigma_\nu\) is the antisymmetrized ERI contraction weight for
basis function \(\Phi^\sigma_\nu\).

The exact merge law in this basis is

\[
G^\sigma_{A \otimes B}
=
S_B\, G^\sigma_A
+
S_A\, G^\sigma_B
+
\mathcal{S}_{\nu\mu\lambda}\,
A^\sigma_{A,\mu}\,
A^\sigma_{B,\lambda},
\]

with:

\[
A^\sigma_{A,\mu}
\]

the degree-`1` coefficient vector on block \(A\),

\[
A^\sigma_{B,\lambda}
\]

the degree-`1` coefficient vector on block \(B\),

and

\[
\mathcal{S}_{\nu\mu\lambda}
\]

the exact antisymmetrized structure tensor mapping two degree-`1` basis
insertions into one degree-`2` basis insertion.

So in basis form, same-spin requires:

1. the exact degree-`1` coefficient vector \(A^\sigma\);
2. the exact degree-`2` coefficient vector \(G^\sigma\);
3. the exact bilinear map \(\mathcal{S}_{\nu\mu\lambda}\).

That is the minimum exact closure.

### 5.4 Practical implication for the current code

The current typed payload

\[
(\text{overlap},\; \text{first-order sectors},\; \text{final scalar})
\]

is still insufficient, because it skips the actual coefficient vector

\[
G^\sigma.
\]

The implementation can choose any concrete storage format for \(G^\sigma\), but
it must represent the full exact degree-`2` boundary message, not only its
final contracted scalar.

---

## 6. What Must Be Implemented Next

Before more performance work, the exact message algebra should be finalized on
paper as the following two objects:

### 6.1 Opposite-spin

Introduce an exact mixed alpha/beta boundary object:

\[
X_{\mu\nu}
=
\sum_t w_t\, A^\alpha_{t\mu} A^\beta_{t\nu},
\]

where \(\mu,\nu\) index the exact degree-`1` boundary basis.

Then opposite-spin is one final linear contraction of \(X\).

### 6.2 Same-spin

Introduce the full one-spin degree-`2` boundary message

\[
m^{(2)} = (S, r, c, C, \Gamma),
\]

or equivalently the pair of exact basis coefficient objects

\[
(A^\sigma,\; G^\sigma).
\]

Then same-spin is one final linear contraction of \(\Gamma\).

---

## 7. Implementation Reading Of The Formulas

These formulas imply the following coding rule:

1. `degree-1 one-electron` can be exact with a one-spin dense payload
   `(S, r, c, C)`;
2. `opposite-spin` needs an exact **mixed** alpha/beta aggregate after local
   term summation;
3. `same-spin` needs an exact one-spin `degree-2` message, not only a scalar.

So the next implementation should not start by "optimizing the current
payload".

It should start by replacing the current insufficient root/message objects by
the two exact objects above.

That is the mathematically correct next step.

---

## 8. Frontier/Root Closure In The Current `component_tree` Layout

The current rooted-tree hot path always reaches a closure of the following
form:

1. one aggregated `frontier` object built from all selected child messages;
2. one aggregated `root` or local-complement object;
3. one final merge between these two objects.

So the exact formula we actually need is not only a generic subtree merge. We
need the exact closure for one pair

\[
F \otimes R,
\]

where:

- \(F\) is the aggregated frontier side;
- \(R\) is the aggregated root/local side.

### 8.1 Exact factorized term structure

Let:

\[
f \in \mathcal F
\]

index the exact frontier term assignments, and

\[
r \in \mathcal R
\]

index the exact root/local term assignments.

Because the current root closure enumerates frontier and root independently,
the exact full term index is

\[
t = (f,r),
\]

with exact signed coefficient

\[
w_t = u_f\, v_r.
\]

This factorization is the key structural property that makes exact boundary
closure possible.

---

## 9. Exact Degree-1 Closure In Basis Form

Let

\[
U^\sigma
\]

denote the complete one-spin degree-`\le 1` boundary object, meaning one
coefficient vector that includes:

1. the closed overlap scalar;
2. all row-open basis components;
3. all col-open basis components;
4. all first-cofactor basis components.

For one exact term pair \(t\), write its coefficient vector as

\[
U_t^\sigma(i),
\]

where \(i\) runs over that complete degree-`\le 1` basis.

The exact merge of two blocks \(A\) and \(B\) is bilinear:

\[
U_{A \otimes B}^\sigma(k)
=
\sum_{i,m}
T^\sigma_{k i m}\,
U_A^\sigma(i)\,
U_B^\sigma(m).
\]

Here

\[
T^\sigma_{k i m}
\]

is the exact sparse structure tensor encoding the separator sign convention and
the degree-`0/1` Leibniz rule.

This one equation is the compact basis form of all the explicit formulas from
Section 3.

### 9.1 Aggregated closure

Define the aggregated frontier and root degree-`1` objects:

\[
\bar U_F^\sigma(i)
=
\sum_{f \in \mathcal F}
u_f\, U_f^\sigma(i),
\qquad
\bar U_R^\sigma(m)
=
\sum_{r \in \mathcal R}
v_r\, U_r^\sigma(m).
\]

Then the exact aggregated closure is

\[
\bar U_{F \otimes R}^\sigma(k)
=
\sum_{i,m}
T^\sigma_{k i m}\,
\bar U_F^\sigma(i)\,
\bar U_R^\sigma(m).
\]

This follows immediately from the factorization

\[
w_{(f,r)} = u_f v_r.
\]

So for degree `1`, the current root closure really can be exact with only the
two aggregated one-spin boundary objects

\[
\bar U_F^\sigma,
\qquad
\bar U_R^\sigma.
\]

That is exactly what the recent first-cofactor / one-electron fix confirmed in
code.

---

## 10. Exact Opposite-Spin Closure In The Current Layout

The opposite-spin channel does **not** close on the marginal vectors

\[
\bar U^\alpha,
\qquad
\bar U^\beta
\]

alone.

The exact aggregated object on one block \(X\) is

\[
X_X(i,j)
=
\sum_{t \in X}
w_t\,
U_t^\alpha(i)\,
U_t^\beta(j).
\]

This is the mixed alpha/beta second-order moment of the degree-`1` boundary
basis on block \(X\).

### 10.1 Exact frontier/root closure of the mixed object

For the merged block \(F \otimes R\),

\[
X_{F \otimes R}(k,\ell)
=
\sum_{t=(f,r)}
w_t\,
U_t^\alpha(k)\,
U_t^\beta(\ell).
\]

Using the bilinear degree-`1` merge,

\[
U_t^\alpha(k)
=
\sum_{i,m}
T^\alpha_{k i m}\,
U_f^\alpha(i)\,
U_r^\alpha(m),
\]

\[
U_t^\beta(\ell)
=
\sum_{j,n}
T^\beta_{\ell j n}\,
U_f^\beta(j)\,
U_r^\beta(n),
\]

we obtain

\[
X_{F \otimes R}(k,\ell)
=
\sum_{i,m,j,n}
T^\alpha_{k i m}\,
T^\beta_{\ell j n}\,
\Bigg[
\sum_f u_f\, U_f^\alpha(i) U_f^\beta(j)
\Bigg]
\Bigg[
\sum_r v_r\, U_r^\alpha(m) U_r^\beta(n)
\Bigg].
\]

Therefore

\[
X_{F \otimes R}(k,\ell)
=
\sum_{i,m,j,n}
T^\alpha_{k i m}\,
T^\beta_{\ell j n}\,
X_F(i,j)\,
X_R(m,n).
\]

This is the exact frontier/root closure formula for opposite-spin.

### 10.2 Why this is the right exact object

This formula proves two things at once:

1. the mixed object
   \[
   X
   \]
   is sufficient for exact opposite-spin closure under the current
   frontier/root factorization;
2. the marginal vectors
   \[
   \bar U^\alpha,\ \bar U^\beta
   \]
   are insufficient, because the closure acts on the mixed second-order moment,
   not on the marginals.

So the exact rooted-tree opposite-spin path should carry

\[
X_F,\qquad X_R
\]

and close them with the tensor formula above.

### 10.3 Final opposite-spin scalar contraction

Once

\[
X_{F \otimes R}(k,\ell)
\]

has been formed, the exact opposite-spin scalar is one linear contraction:

\[
E_{\alpha\beta}^{\text{exact}}
=
\sum_{k,\ell}
K^{\alpha\beta}_{k\ell}\,
X_{F \otimes R}(k,\ell),
\]

where

\[
K^{\alpha\beta}_{k\ell}
\]

is the ERI contraction kernel on the degree-`1` alpha/beta basis.

That is the exact root closure we actually need in
`evaluate_rooted_component_tree_opposite_spin_boundary_collapsed(...)`.

---

## 11. Exact Same-Spin Closure In The Current Layout

Let

\[
G_X^\sigma(\nu)
=
\sum_{t \in X}
w_t\, V_t^\sigma(\nu)
\]

be the exact aggregated degree-`2` boundary coefficient vector on block \(X\),
where

\[
V_t^\sigma(\nu)
\]

is the degree-`2` basis coefficient of term \(t\).

Let

\[
\bar U_X^\sigma(i)
\]

be the aggregated degree-`\le 1` vector from Section 9.

The exact one-term merge law is

\[
V_{A \otimes B}^\sigma(\nu)
=
\sum_{\rho}
P^\sigma_{\nu\rho}(B)\, V_A^\sigma(\rho)
+
\sum_{\lambda}
Q^\sigma_{\nu\lambda}(A)\, V_B^\sigma(\lambda)
+
\sum_{i,m}
\mathcal S^\sigma_{\nu i m}\,
U_A^\sigma(i)\,
U_B^\sigma(m).
\]

Here:

- \(P^\sigma_{\nu\rho}(B)\) is the exact linear map multiplying the degree-`2`
  payload of \(A\) by the degree-`0` content of \(B\) with the correct sign
  convention;
- \(Q^\sigma_{\nu\lambda}(A)\) is the symmetric map for the \(B\) side;
- \(\mathcal S^\sigma_{\nu i m}\) is the exact antisymmetrized bilinear map
  sending two degree-`\le 1` basis insertions to one degree-`2` basis
  insertion.

### 11.1 Aggregated frontier/root closure

Summing over \(t=(f,r)\) with \(w_t=u_f v_r\), the exact aggregated closure is

\[
G_{F \otimes R}^\sigma(\nu)
=
\sum_{\rho}
P^\sigma_{\nu\rho}(\bar U_R^\sigma)\, G_F^\sigma(\rho)
+
\sum_{\lambda}
Q^\sigma_{\nu\lambda}(\bar U_F^\sigma)\, G_R^\sigma(\lambda)
+
\sum_{i,m}
\mathcal S^\sigma_{\nu i m}\,
\bar U_F^\sigma(i)\,
\bar U_R^\sigma(m).
\]

In the simplest notation, this is exactly

\[
G_{F \otimes R}^\sigma
=
\bar S_R^\sigma\, G_F^\sigma
+
\bar S_F^\sigma\, G_R^\sigma
+
\mathcal S^\sigma(\bar U_F^\sigma,\bar U_R^\sigma),
\]

provided the basis and signs have already been absorbed into
\(\mathcal S^\sigma\).

### 11.2 Why the scalar-only same-spin closure cannot work

The exact same-spin scalar is

\[
E_{\sigma\sigma}^{\text{exact}}
=
\sum_\nu
L_\nu^\sigma\, G_{F \otimes R}^\sigma(\nu).
\]

So the final scalar depends linearly on the full exact vector

\[
G_{F \otimes R}^\sigma,
\]

not only on one number produced too early.

Therefore the rooted-tree same-spin path must carry the degree-`2` boundary
coefficient vector all the way to the final contraction.

That is the exact closure required in
`evaluate_rooted_component_tree_same_spin_boundary_collapsed(...)`.

---

## 12. Immediate Design Translation

The exact rooted-tree production message should ultimately carry:

\[
\Big(
\bar U^\alpha,\;
\bar U^\beta,\;
G^\alpha,\;
G^\beta,\;
X
\Big),
\]

where:

1. \(\bar U^\alpha\) is the aggregated alpha degree-`\le 1` boundary object;
2. \(\bar U^\beta\) is the aggregated beta degree-`\le 1` boundary object;
3. \(G^\alpha\) is the aggregated alpha degree-`2` boundary vector;
4. \(G^\beta\) is the aggregated beta degree-`2` boundary vector;
5. \(X\) is the exact mixed alpha/beta second-order moment on the degree-`1`
   basis.

Then:

- overlap is read from the degree-`0` component of \(\bar U^\alpha,\bar U^\beta\);
- one-electron is linear in \(\bar U^\alpha,\bar U^\beta\);
- same-spin is linear in \(G^\alpha,G^\beta\);
- opposite-spin is linear in \(X\).

This is the clean exact boundary-only closure for the current
`component_tree` architecture.
