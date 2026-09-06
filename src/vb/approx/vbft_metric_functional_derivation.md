# VBFT Metric Functional Derivation

## 1. Purpose

The purpose of this note is to derive a computable nonorthogonal metric
functional for VB Functional Theory.

At fixed localized active orbitals $R$, exact VBSCF is governed by the quotient

$$
E_{\mathrm{VB}}(R,c)
=
\frac{c^\mathrm T H(R)c}{c^\mathrm T S(R)c}.
$$

An approximate VB theory that keeps only a numerator energy model is therefore
not yet an approximate VBSCF.  It misses the metric response

$$
dH
-
E_{\mathrm{VB}}dS.
$$

The target is a compact denominator

$$
\mathcal S_{\mathrm{aVB}}(R,\xi)
$$

where

$$
\xi
=
\{n_p,\Gamma_{pq},\rho_\alpha,\mu_\beta,\ldots\}.
$$

The main result of this derivation is:

> The metric descriptors $\mu_\beta$ can be defined as log-overlap cumulants of
> local VB pair clusters, equivalently as closed-walk cumulants on the
> nonorthogonal active-orbital overlap graph.

This gives a concrete Stage 2 route: replace the current scalar overlap penalty
by an exponential metric denominator built from one-pair and two-pair metric
cumulants.

## 2. Exact Metric In The Current C++ Implementation

The current exact implementation reduces every structure overlap to determinant
overlaps.  A VB structure is expanded in alpha and beta determinant terms:

$$
|\Phi_I(R)\rangle
=
\sum_{a\in\mathcal D_I}
\eta_{Ia}
|D_{Ia}^{\alpha}(R)D_{Ia}^{\beta}(R)\rangle.
$$

For one spin channel, let $A$ be the ordered occupied-orbital list of the left
determinant and $B$ the ordered occupied-orbital list of the right determinant.
The determinant overlap is

$$
\Delta_\sigma(A,B;R)
=
\det M_\sigma(A,B;R),
$$

with

$$
\left[M_\sigma(A,B;R)\right]_{rs}
=
\langle\phi_{B_r}|\phi_{A_s}\rangle.
$$

This matches the code path in which `build_overlap_submatrix` builds a
right-occupied by left-occupied active-overlap submatrix and
`DeterminantOverlapResolver` returns its determinant.

The exact structure overlap is therefore

$$
S_{IJ}(R)
=
\sum_{a\in\mathcal D_I}
\sum_{b\in\mathcal D_J}
\eta_{Ia}\eta_{Jb}
\Delta_\alpha(A_{Ia}^{\alpha},A_{Jb}^{\alpha};R)
\Delta_\beta(A_{Ia}^{\beta},A_{Jb}^{\beta};R).
$$

The exact state metric is

$$
\mathcal S_{\mathrm{exact}}(R,c)
=
c^\mathrm T S(R)c
=
\sum_{I,J}c_Ic_JS_{IJ}(R).
$$

There is an important normalization convention.  In an exact generalized
eigenvalue calculation one can always choose

$$
c^\mathrm T S(R)c=1.
$$

In the compact quotient ansatz, $\mathcal S_{\mathrm{aVB}}$ should therefore be
viewed as the metric partition factor that normalizes an unnormalized compact
pair-cluster kernel, not as an additional observable of an already normalized
exact state.  The useful gauge is

$$
\mathcal S_{\mathrm{aVB}}(R_0,\xi)=1
$$

for the orthogonal active-overlap limit $R_0$.  With this gauge, only metric
ratios and metric derivatives enter the energy and orbital response.

The computational problem is that this expression requires the full structure
space.  The theoretical problem is to preserve its effect using compact VB
observables.

## 3. Determinant Log-Det Expansion

For one determinant transition, choose a permutation or matching matrix
$P_\sigma(A,B)$ that aligns the dominant right occupied orbitals with the
dominant left occupied orbitals.  Then write

$$
M_\sigma(A,B;R)
=
P_\sigma(A,B)
\left[I+K_\sigma(A,B;R)\right].
$$

The determinant becomes

$$
\Delta_\sigma(A,B;R)
=
\det P_\sigma(A,B)
\det\left[I+K_\sigma(A,B;R)\right].
$$

If the spectral radius of $K_\sigma$ is below one,

$$
\log\left|\det\left[I+K_\sigma\right]\right|
=
\mathrm{tr}\log(I+K_\sigma)
=
\sum_{m=1}^{\infty}
\frac{(-1)^{m+1}}{m}
\mathrm{tr}K_\sigma^m.
$$

For both spin channels,

$$
\ell(A,B;R)
=
\sum_{\sigma\in\{\alpha,\beta\}}
\log\left|\det\left[I+K_\sigma(A,B;R)\right]\right|.
$$

Thus

$$
\ell(A,B;R)
=
\sum_{\sigma}
\sum_{m=1}^{\infty}
\frac{(-1)^{m+1}}{m}
\mathrm{tr}K_\sigma^m.
$$

More exactly, the determinant transition has a sign:

$$
\Delta_\sigma(A,B;R)
=
\chi_\sigma(A,B;R)
\exp\ell_\sigma(A,B;R),
$$

where $\chi_\sigma$ contains the permutation sign and any sign change from the
determinant.  The log-det expansion controls the magnitude.  Signed
cancellations belong to the resonance/coherence part of the functional, and
near-zero determinant transitions must be screened or treated by a separate
singular metric rule.

Each trace is a closed-walk sum:

$$
\mathrm{tr}K_\sigma^m
=
\sum_{r_1,\ldots,r_m}
(K_\sigma)_{r_1r_2}
(K_\sigma)_{r_2r_3}
\cdots
(K_\sigma)_{r_mr_1}.
$$

For normalized orbitals and a self-like determinant sector, the diagonal of
$K_\sigma$ is zero.  The first nonzero term is then

$$
\ell_\sigma^{(2)}
=
-
\frac{1}{2}
\mathrm{tr}K_\sigma^2
=
-
\frac{1}{2}
\sum_{r\ne s}
(K_\sigma)_{rs}(K_\sigma)_{sr}.
$$

For real symmetric active overlaps, this becomes

$$
\ell_\sigma^{(2)}
=
-
\frac{1}{2}
\sum_{r\ne s}
s_{a_ra_s}^2.
$$

Therefore nonorthogonal determinant metric is not an arbitrary penalty.  Its
lowest same-spin Pauli correction is forced by the determinant identity

$$
d\log\det M
=
\mathrm{tr}(M^{-1}dM).
$$

## 4. Why Pair-Level Overlap Has The Opposite Sign

The determinant log-det term above is a same-spin Pauli metric.  A spin-adapted
VB pair also has an internal spin-coupling metric.

For a covalent pair $p=(i,j)$ with $i\ne j$, the legacy closed-shell pair
expansion contains the two determinant terms

$$
|i^\alpha j^\beta\rangle,
\qquad
|j^\alpha i^\beta\rangle.
$$

With normalized orbitals and $s_{ij}=\langle\phi_i|\phi_j\rangle$, the local
pair norm is proportional to

$$
1+s_{ij}^2.
$$

The convention-dependent constant at $s_{ij}=0$ can be divided out.  The
nontrivial one-pair metric cumulant is then

$$
a_{ij}(R)
=
\log(1+s_{ij}^2),
\qquad
i\ne j.
$$

For an ionic pair $p=(i,i)$,

$$
a_{ii}(R)
=
0
$$

under the same normalized convention.

This explains the current minimal implementation: its `overlap_response` term
uses $s_{ij}^2$, which is the leading term of

$$
\log(1+s_{ij}^2)
=
s_{ij}^2
-
\frac{1}{2}s_{ij}^4
+
O(s_{ij}^6).
$$

The issue is not the physical signal.  The issue is where it is placed.  It is
a metric contribution and should enter $\mathcal S_{\mathrm{aVB}}$, or a
metric-consistent numerator and denominator pair, rather than a fixed subtractive
energy penalty.

## 5. Pair-Cluster Metric Cumulants

The log metric can be written as a cluster expansion over local VB pairs.
This is the most direct computable skeleton for Stage 2.

For a one-pair cluster $p$, define its exact local metric by determinant
enumeration in the same convention as the C++ exact VB overlap code.  The local
state $|P_p(R)\rangle$ is the spin-adapted antisymmetrized VB pair state:

$$
S_p(R)
=
\langle P_p(R)|P_p(R)\rangle.
$$

Remove the convention constant by comparing with the orthogonal active-overlap
limit $R_0$:

$$
a_p(R)
=
\log
\frac{S_p(R)}{S_p(R_0)}.
$$

For a two-pair cluster $(p,q)$, define

$$
S_{pq}(R)
=
\langle P_p(R)P_q(R)|P_p(R)P_q(R)\rangle.
$$

Here $|P_pP_q\rangle$ means the antisymmetrized local VB cluster generated by
placing both pair patterns in the same determinant-expansion machinery.  The
definition applies only to representable pair clusters.  If two pair patterns
violate local particle or spin occupancy constraints, then $\Gamma_{pq}$ should
be constrained to zero and $b_{pq}$ should not be used as a physical metric
cumulant.

The connected two-pair metric cumulant is

$$
b_{pq}(R)
=
\log
\frac{S_{pq}(R)}{S_{pq}(R_0)}
-
a_p(R)
-
a_q(R).
$$

The three-pair cumulant would be

$$
c_{pqr}(R)
=
\log
\frac{S_{pqr}(R)}{S_{pqr}(R_0)}
-
\sum_{x\in\{p,q,r\}}a_x(R)
-
\sum_{x<y\in\{p,q,r\}}b_{xy}(R).
$$

For a structure $I$ with pair set $\mathcal P_I$, the exact structure self-metric
has the formal expansion

$$
\log
\frac{S_{II}(R)}{S_{II}(R_0)}
=
\sum_{p\in\mathcal P_I}a_p(R)
+
\sum_{p<q\in\mathcal P_I}b_{pq}(R)
+
\sum_{p<q<r\in\mathcal P_I}c_{pqr}(R)
+
\cdots.
$$

This expansion is exact if all cluster orders are retained.  Truncation at the
two-pair level gives the first scalable metric functional:

$$
\Omega_{\mathrm{pair2}}(R,n,\Gamma)
=
\sum_p n_p a_p(R)
+
\sum_{p<q}\Gamma_{pq}b_{pq}(R).
$$

The approximate denominator is

$$
\mathcal S_{\mathrm{aVB}}^{\mathrm{pair2}}(R,n,\Gamma)
=
\exp\left[
\Omega_{\mathrm{pair2}}(R,n,\Gamma)
\right].
$$

This formula is the clean replacement for the current minimal overlap response.

> Stage 2 should treat $a_p$ and $b_{pq}$ as metric cumulants, not as direct
> energy terms.

## 6. Relation Between Pair Cumulants And Closed Walks

The pair-cluster definition above is implementation-friendly because $S_p$ and
$S_{pq}$ are tiny exact VB overlap problems.  The log-det derivation explains
what these clusters contain.

For weak cross-pair overlaps, the leading two-pair connected term has the form

$$
b_{pq}^{(2)}(R)
=
-
\frac{1}{2}
\sum_{\sigma}
\sum_{i\ne j}
Q_{ij}^{\sigma}(p,q)
s_{ij}s_{ji}
+
\text{pair-exchange terms},
$$

where $Q_{ij}^{\sigma}(p,q)$ is the same-spin co-occupation weight generated
by the two-pair spin coupling.  The first part is the same-spin determinant
Pauli metric from $-\frac{1}{2}\mathrm{tr}K_\sigma^2$.  The pair-exchange terms
come from determinant transitions created by spin adaptation, such as the
one-pair $+\log(1+s_{ij}^2)$ contribution.

Higher terms have direct graph meanings:

$$
\frac{1}{3}\mathrm{tr}K^3
$$

counts triangular overlap cycles, and

$$
-
\frac{1}{4}\mathrm{tr}K^4
$$

counts four-step closed overlap walks.

This gives a concrete interpretation of the metric descriptor layer:

$$
\mu_m
=
\left\langle
\sum_{\sigma}\mathrm{tr}K_\sigma^m
\right\rangle_{\xi}.
$$

The log metric hierarchy is then

$$
\log\mathcal S_{\mathrm{aVB}}
\approx
\sum_{m=1}^{M}
\frac{(-1)^{m+1}}{m}
\mu_m
+
\text{spin-adapted pair-exchange cumulants}.
$$

The pair-cluster expansion and the closed-walk expansion are two views of the
same object:

- the cluster view is easier to implement and validate;
- the closed-walk view gives the analytic physical interpretation and gradient
  structure.

## 7. Off-Diagonal Resonance Metric

Exact VBSCF does not use only $S_{II}$.  The denominator contains

$$
\sum_{I\ne J}c_Ic_JS_{IJ}(R).
$$

These off-diagonal terms encode resonance overlap between different VB
structures.  They should not be hidden inside pair populations.

For a local resonance motif $\alpha$ connecting two local pairing patterns
$A_\alpha$ and $B_\alpha$, define the normalized local transition metric

$$
r_\alpha(R)
=
\frac{
S_{A_\alpha B_\alpha}(R)
}{
\sqrt{
S_{A_\alpha A_\alpha}(R)
S_{B_\alpha B_\alpha}(R)
}
}.
$$

The associated resonance coherence variable is

$$
\rho_\alpha
=
\langle\Psi|\hat R_\alpha|\Psi\rangle.
$$

At first order in local transition coherences, the resonance metric correction
can be written as

$$
\Omega_{\mathrm{res}}(R,\rho)
\approx
\sum_\alpha \rho_\alpha d_\alpha(R),
$$

with

$$
d_\alpha(R)
\sim
r_\alpha(R).
$$

For stronger resonance, the safer form is

$$
\Omega_{\mathrm{res}}(R,\rho)
=
\log
\left[
1+
\sum_\alpha \rho_\alpha r_\alpha(R)
+
\sum_{\alpha<\beta}\rho_{\alpha\beta}r_{\alpha\beta}(R)
+
\cdots
\right].
$$

This makes the theoretical boundary clear:

> Pair populations and pair-pair occupations can approximate diagonal metric
> physics.  Resonance coherences are required for off-diagonal metric physics.

## 8. Practical Stage 2 Metric Functional

The first useful Stage 2 functional should be

$$
\mathcal S_{\mathrm{aVB}}(R,\xi)
=
\exp\Omega(R,\xi),
$$

where

$$
\Omega(R,\xi)
=
\Omega_{\mathrm{pair2}}(R,n,\Gamma)
+
\Omega_{\mathrm{res}}(R,\rho)
+
\Omega_{\mathrm{walk}}^{(3+)}(R,\mu).
$$

The first implemented version can set

$$
\Omega_{\mathrm{res}}=0,
\qquad
\Omega_{\mathrm{walk}}^{(3+)}=0,
$$

and use

$$
\Omega(R,n,\Gamma)
=
\sum_p n_p a_p(R)
+
\sum_{p<q}\Gamma_{pq}b_{pq}(R).
$$

For a mean-field pair-SCF prototype,

$$
\Gamma_{pq}
\approx
n_pn_q.
$$

For exact-label validation on small systems, $\Gamma_{pq}$ should be mapped
from exact VBSCF structure weights:

$$
\Gamma_{pq}^{\mathrm{exact}}
=
\sum_I W_I b_I(p)b_I(q).
$$

This gives two validation modes:

1. exact-label aVB metric validation using exact $n_p$ and $\Gamma_{pq}$;
2. production aVBSCF validation using optimized $n_p$ and approximate
   $\Gamma_{pq}$.

## 9. Numerator And Denominator Consistency

The exact determinant-pair Hamiltonian kernels contain overlap factors.  A
metric denominator alone is therefore not sufficient unless the numerator is
defined consistently.

The approximate functional should remain a quotient:

$$
F_R^{\mathrm{aVB}}[\xi]
=
\frac{
\mathcal H_{\mathrm{aVB}}(R,\xi)
}{
\mathcal S_{\mathrm{aVB}}(R,\xi)
}.
$$

If

$$
\mathcal S_{\mathrm{aVB}}
=
\exp\Omega,
$$

then the orbital derivative is

$$
\frac{\partial F_R^{\mathrm{aVB}}}{\partial R_k}
=
\frac{1}{\mathcal S_{\mathrm{aVB}}}
\frac{\partial\mathcal H_{\mathrm{aVB}}}{\partial R_k}
-
F_R^{\mathrm{aVB}}
\frac{\partial\Omega}{\partial R_k}.
$$

The descriptor derivative is similarly

$$
\frac{\partial F_R^{\mathrm{aVB}}}{\partial \xi_a}
=
\frac{1}{\mathcal S_{\mathrm{aVB}}}
\frac{\partial\mathcal H_{\mathrm{aVB}}}{\partial \xi_a}
-
F_R^{\mathrm{aVB}}
\frac{\partial\Omega}{\partial \xi_a}.
$$

This is the approximate counterpart of

$$
dH
-
E\,dS.
$$

The current prototype has

$$
E_{\mathrm{proto}}
=
E_{\mathrm{pair}}
-
\sum_p n_p s_p^2.
$$

The Stage 2 form should instead be

$$
F_{\mathrm{stage2}}
=
\frac{
\mathcal H_{\mathrm{stage2}}(R,n,\Gamma)
}{
\exp\Omega_{\mathrm{stage2}}(R,n,\Gamma)
}.
$$

This does not mean the numerator can stay unchanged forever.  The numerator
must ultimately contain metric-weighted local Hamiltonian kernels, because
exact VB Hamiltonian matrix elements are not independent of determinant
overlaps.

## 10. Scaling

Let the active orbital count be $n_a$ and the active pair count be

$$
P
=
\frac{n_a(n_a+1)}{2}.
$$

The one-pair metric table costs

$$
O(P)
$$

small cluster evaluations.

The two-pair metric table costs

$$
O(P^2)
$$

small cluster evaluations.  Each cluster is a fixed-size determinant-overlap
problem, so this avoids the exponential structure count.

For a $30$-$30$ active space,

$$
P=465,
\qquad
P^2=216225.
$$

This is large but realistic, especially with locality screening.  It is not
comparable to exact VBSCF structure enumeration.

With locality, only pair pairs with non-negligible orbital overlap need
nonzero $b_{pq}$:

$$
|b_{pq}(R)|<\epsilon_{\mathrm{metric}}
\quad
\Rightarrow
\quad
b_{pq}(R)=0.
$$

The resulting cost is closer to

$$
O(Pz),
$$

where $z$ is the number of local metric neighbors per pair.

## 11. What This Derivation Solves

This derivation gives a real theoretical and implementation step:

1. It defines $\mu_\beta$ concretely as metric cumulants rather than vague
   learned descriptors.
2. It explains the existing $s_{ij}^2$ term as the leading one-pair metric
   cumulant $\log(1+s_{ij}^2)$.
3. It moves nonorthogonality from an energy penalty into the denominator
   $\mathcal S_{\mathrm{aVB}}$.
4. It gives an exact metric hierarchy: one-pair, two-pair, three-pair, and
   higher cluster cumulants.
5. It preserves the VB physical meaning because every term is derived from
   local spin-adapted VB overlap, not from black-box regression.

This is a plausible VBFT breakthrough at the metric level:

> The nonorthogonal VB metric can be approximated by a controlled local
> cumulant functional with polynomial scaling.

## 12. What Is Still Not Solved

The derivation does not yet prove a full HK-like theorem for VBFT.  It gives a
controlled metric hierarchy, not a universal proof that a small $\xi$ uniquely
determines the exact VB state.

The remaining theory gaps are:

1. off-diagonal resonance metric requires explicit $\rho_\alpha$ variables;
2. signed determinant overlaps and destructive cancellation need a stable
   treatment;
3. representability constraints must link $n_p$, $\Gamma_{pq}$, $\rho_\alpha$,
   and metric cumulants;
4. the numerator must be made consistent with the same metric approximation;
5. low-order cluster truncation needs numerical validation and possibly
   locality error bounds.

The exact cluster hierarchy suggests a route to an exact limit:

$$
\Omega^{(1)}
\rightarrow
\Omega^{(2)}
\rightarrow
\Omega^{(3)}
\rightarrow
\cdots
\rightarrow
\Omega^{(\infty)}.
$$

However, the convergence of low-order truncations is a physical locality
assumption, not yet a theorem.

## 13. Stage 2 Acceptance Criteria

The next code stage should be considered complete only when it satisfies the
following points.

1. Build $a_p(R)$ for every active pair using the exact one-pair local metric.
2. Build $b_{pq}(R)$ for every compatible active pair pair using the connected
   two-pair local metric.
3. Replace the subtractive overlap response energy by
   $\mathcal S_{\mathrm{aVB}}=\exp\Omega$.
4. Report $\Omega$, $\mathcal S_{\mathrm{aVB}}$, and separate one-pair and
   two-pair metric contributions.
5. Validate exact-label energies using exact VBSCF mapped $n_p$ and
   $\Gamma_{pq}$ on small systems.
6. Validate production pair-SCF energies using optimized $n_p$ and mean-field
   or constrained $\Gamma_{pq}$.
7. Compare against exact VBSCF for F2 and C6H6 $6$-$6$ full space.
8. Add a scaling benchmark that shows the metric build scales as $O(P^2)$ or
   $O(Pz)$ after screening, not with the number of VB structures.

Only after this denominator is working should neural correction be introduced.
If a neural residual is used later, the natural target is

$$
\Omega(R,\xi)
=
\Omega_{\mathrm{cluster}}(R,\xi)
+
\Delta\Omega_\theta(R,\xi)
$$

or a metric-consistent numerator residual, with exact one-pair and two-pair
limits preserved by construction.
