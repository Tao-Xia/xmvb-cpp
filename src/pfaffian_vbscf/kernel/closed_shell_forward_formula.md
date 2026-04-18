# Closed-Shell Pf-VBSCF Forward Formula Summary

## Scope

This note summarizes the analytic formulas currently used by the production
closed-shell Pf-VBSCF forward path under `src/pfaffian_vbscf/`.

The summary matches the current implementation in:

- `src/pfaffian_vbscf/kernel/pf_forward_kernel.cpp`
- `src/pfaffian_vbscf/kernel/pf_closed_shell_spatial_kernel.cpp`
- `src/pfaffian_vbscf/math/trace_projector.cpp`
- `src/pfaffian_vbscf/math/pf_matrix_polynomial.cpp`
- `src/pfaffian_vbscf/matrices/pf_pair_kernels.cpp`

Current scope:

- closed-shell only
- singlet `AB/BA` Pf basis
- direct Pfaffian/VB-structure matrix-element evaluation
- no determinant expansion in the production path

Throughout this note:

- `M` is the number of active spatial orbitals
- `n` is the number of closed-shell pairs
- `S` is the active-space spatial overlap matrix
- `h` is the active-space one-electron Hamiltonian matrix
- `g_{qp,sr}` is the packed active-space two-electron tensor

## 1. Pairwise Pfaffian Inputs

For one bra/ket state pair `(I, J)`, the code first forms

- left spin-pairing matrix `L`
- right spin-pairing matrix `R`
- spin metric
  `Sigma = diag(S, S)`

The spin-orbital kernel is

```text
K = L Sigma R Sigma^T
```

In the current closed-shell `AB/BA` basis, only a few spatial blocks are
needed. We define

```text
A = (L)_{beta,alpha}
B = (R)_{alpha,beta}
C = A S B
G = (K)_{alpha,alpha}
H = S C
```

In this basis, the spatial transition density lives entirely in the identical
`alpha/alpha` and `beta/beta` blocks, and the production two-electron formula
can be written only in terms of `A`, `B`, `C`, `G`, and `H`.

## 2. Trace Projection And Overlap Coefficients

Let

```text
t_p = tr(K^p),    p = 1, 2, ..., n
```

The projected overlap series is

```text
Omega(t) = sqrt(det(I + t K)) = sum_{k >= 0} c_k t^k
```

with recursion

```text
c_0 = 1
c_k = (1 / k) sum_{p = 1}^k [ (1/2) (-1)^(p+1) t_p c_{k-p} ]
```

The physical overlap matrix element is the `t^n` coefficient

```text
S_IJ = c_n
```

This is exactly what `TraceProjector::project(traces, n)` computes.

## 3. Projected Polynomial Coefficients

For any order `r >= 0`, define the projected coefficients

```text
pi_m^(r) = (-1)^m c_{r-m},    m = 0, 1, ..., r
```

and the corresponding matrix polynomial

```text
P_r(X; Y) = sum_{m = 0}^r pi_m^(r) X^m Y
```

The code also uses a right-applied version

```text
R_r(X; Y) = sum_{m = 0}^r pi_m^(r) Y X^m
```

and the Fréchet-type split kernel

```text
F_r(X; Y) = sum_{p = 1}^r pi_p^(r) sum_{a = 0}^{p-1} X^a Y X^(p-1-a)
```

In the current implementation these are built by the helpers

- `apply_left_matrix_polynomial`
- `apply_right_matrix_polynomial`
- `apply_left_matrix_polynomial_frechet`

using linear recurrences rather than explicit double summation.

## 4. Spatial 1-RDM And One-Electron Matrix Element

The closed-shell spatial transition 1-RDM is

```text
Gamma = 2 P_(n-1)(G; C)
      = 2 sum_{m = 0}^{n-1} pi_m^(n-1) G^m C
```

This is exactly the current `cache.one_rdm`.

The one-electron Hamiltonian matrix element is then

```text
H_IJ^(1) = sum_{q,p} h_{qp} Gamma_{qp}
```

which is implemented as the Frobenius inner product

```text
H_IJ^(1) = <Gamma, h>
```

## 5. Two-Electron Contraction Operators

The current closed-shell forward path uses the following spatial contraction
operators.

### 5.1 Exchange

```text
Ex(X, Y) = sum_{q,p,s,r} g_{qp,sr} X_{pr} Y_{sq}
```

### 5.2 Coulomb

```text
Co(X, Y) = sum_{q,p,s,r} g_{qp,sr} X_{rp} Y_{qs}
```

### 5.3 Direct

```text
Dir(X, Y) = sum_{q,p,s,r} g_{qp,sr} X_{qp} Y_{sr}
```

### 5.4 Same-Spin Separable

```text
Sep(X, Y) =
sum_{p<r, q<s} (g_{qp,sr} - g_{qr,sp}) X_{qp} Y_{sr}
```

### 5.5 Same-Spin Bridge

```text
Br(X, Y) =
sum_{p<r, q<s} (g_{qp,sr} - g_{qr,sp}) X_{qr} Y_{sp}
```

For compactness below, define

```text
EC(X, Y) = Ex(X, Y) + Co(X, Y)
XSep(X, Y) = 2 Sep(X, Y) + Dir(X, Y)
```

## 6. Closed-Shell Two-Electron Formula

The current production forward path splits the total closed-shell
two-electron matrix element into four pieces:

```text
H_IJ^(2) = E_pair + E_cross + E_same_spin_bridge + E_opposite_bridge
```

All four pieces are evaluated without determinant expansion.

### 6.1 Opposite-Spin Pair Term

First define the half-scaled coefficients

```text
rho_m^(n-1) = (1/2) pi_m^(n-1)
```

Then the pair-term matrix is

```text
M_pair =
rho_0^(n-1) A +
C [ sum_{m = 1}^{n-1} rho_m^(n-1) H^(m-1) ] S A
```

Equivalently,

```text
M_pair =
rho_0^(n-1) A + C Q_(n-1)(H; S A)
```

with

```text
Q_(n-1)(H; S A) = sum_{m = 1}^{n-1} rho_m^(n-1) H^(m-1) S A
```

The corresponding contribution is

```text
E_pair = EC(B, M_pair)
```

### 6.2 Cross Term

Define

```text
C_r = R_(n-2)(H; C)
F   = F_(n-2)(H; C)
U   = F H
```

Then the exact collapsed cross contribution is

```text
E_cross = XSep(C_r, C) + XSep(C, U)
```

or written out,

```text
E_cross =
2 Sep(C_r, C) + Dir(C_r, C) +
2 Sep(C, U)   + Dir(C, U)
```

This is the validated `h = S C` split replacement of the old `n-2` sequence
sum.

### 6.3 Same-Spin Bridge Correction

Using the same `C_r` and `U = F H`, the exact same-spin bridge correction is

```text
E_same_spin_bridge = -2 Br(C_r, C) - 2 Br(C, U)
```

This is the missing same-spin cumulant correction that has to be added on top
of the cross term.

### 6.4 Opposite-Spin Bridge Correction

Define

```text
D   = B H
D_r = R_(n-2)(H; D)
V   = F S A
```

Then the exact opposite-spin bridge correction is

```text
E_opposite_bridge = -(1/2) EC(D_r, A) -(1/2) EC(D, V)
```

or written out,

```text
E_opposite_bridge =
-(1/2) [ Ex(D_r, A) + Co(D_r, A) ]
-(1/2) [ Ex(D, V)   + Co(D, V)   ]
```

## 7. Final Hamiltonian Matrix Element

The final unnormalized pairwise Hamiltonian matrix element is

```text
H_IJ = H_IJ^(1) + H_IJ^(2)
```

with

```text
H_IJ^(1) = <Gamma, h>
H_IJ^(2) = E_pair + E_cross + E_same_spin_bridge + E_opposite_bridge
```

The corresponding overlap matrix element is

```text
S_IJ = c_n
```

These pairwise values are what the code assembles into the Pf-VBSCF overlap
and Hamiltonian matrices over the chosen VB-structure basis.

## 8. Current Computational Structure

At the current production level:

- overlap and projected coefficients come from the trace recursion
- the spatial 1-RDM is a projected matrix polynomial in `G`
- the total two-electron term is evaluated through one pair matrix
  `M_pair`, one right polynomial `R_(n-2)`, and one Fréchet kernel `F_(n-2)`
- the final tensor contractions are a fixed number of `O(M^4)` packed
  two-electron contractions

So the current closed-shell forward path avoids determinant expansion and has
the intended spatial-kernel structure:

```text
trace projection + matrix-polynomial collapse + O(M^4) tensor contractions
```

## 9. Implementation Mapping

- overlap coefficient recursion:
  `src/pfaffian_vbscf/math/trace_projector.cpp`
- 1-RDM polynomial:
  `src/pfaffian_vbscf/kernel/pf_forward_kernel.cpp`
- closed-shell two-electron collapsed formula:
  `src/pfaffian_vbscf/kernel/pf_closed_shell_spatial_kernel.cpp`
- final pairwise matrix-element assembly:
  `src/pfaffian_vbscf/matrices/pf_pair_kernels.cpp`

