# Closed-Shell Pf-VBSCF Two-Electron Adjoint Plan

## Scope

This note describes how the closed-shell Pf-VBSCF two-electron gradient should
be implemented after the forward path was collapsed to the current
`pair-density + h-split bridge` formula.

This document is intentionally limited to the current production forward path:

- closed-shell only
- singlet `AB/BA` Pf basis
- no determinant expansion in production
- only the two-electron adjoint is discussed here

## 1. Why The Old Adjoint Must Be Replaced

The current forward path is no longer the old exact sequence/Hankel expansion.
Production forward now evaluates the closed-shell two-electron matrix element
from the collapsed spatial formula in

- `src/pfaffian_vbscf/kernel/pf_closed_shell_spatial_kernel.cpp`

However, the current closed-shell adjoint still differentiates the old
sequence/Hankel expression:

- `src/pfaffian_vbscf/kernel/pf_adjoint_kernel.cpp`
- function `accumulate_closed_shell_two_electron_exact(...)`

So the correct fix is:

1. keep the existing packed tensor contraction adjoint primitives
2. add reverse helpers for the new matrix-polynomial building blocks
3. add reverse helpers for the projected overlap coefficient series
4. differentiate the new collapsed forward graph directly

We should not differentiate the removed forward representation and rely on
algebraic equivalence afterward.

## 2. Forward Variables

For one closed-shell bra/ket state pair, define:

```text
S  = spatial overlap matrix
A  = left BA block
B  = right AB block
C  = A S B
H  = S C
SA = S A
D  = B H
```

Projected overlap coefficients:

```text
sqrt(det(I + t K)) = sum_{k >= 0} c_k t^k
```

The closed-shell forward uses:

```text
rho_m^(n-1) = (1/2) (-1)^m c_{n-1-m}
pi_m^(n-2)  =       (-1)^m c_{n-2-m}
```

Define:

```text
W   = LeftPoly(H, rho_tail, SA)
M   = rho_0 A + C W
Cr  = RightPoly(H, pi, C)
F   = Frechet(H, pi, C)
U   = F H
Dr  = RightPoly(H, pi, D)
V   = F SA
```

The total closed-shell two-electron energy is:

```text
E2 = E_pair + E_cross + E_ss_bridge + E_os_bridge
```

with

```text
E_pair      = EC(B, M)
E_cross     = XSep(Cr, C) + XSep(C, U)
E_ss_bridge = -2 Br(Cr, C) - 2 Br(C, U)
E_os_bridge = -0.5 EC(Dr, A) - 0.5 EC(D, V)
```

where

```text
EC(X, Y)   = Ex(X, Y) + Co(X, Y)
XSep(X, Y) = 2 Sep(X, Y) + Dir(X, Y)
```

## 3. Reverse Graph Structure

The new adjoint should follow exactly the reverse of the graph above.

The final incoming scalar adjoint is `1`.

### 3.1 Tensor Contraction Layer

At the top level, each scalar tensor contraction contributes:

- operand adjoints for its left/right matrices
- packed `ggo` gradient via the existing outer-product helpers

This layer can reuse the current primitives in `PfTensorContractor`:

- `compute_exchange_operand_adjoints`
- `compute_coulomb_operand_adjoints`
- `compute_direct_operand_adjoints`
- `compute_same_spin_separable_operand_adjoints`
- `compute_same_spin_bridge_operand_adjoints`

and

- `add_exchange_outer_product`
- `add_coulomb_outer_product`
- `add_direct_outer_product`
- `add_same_spin_separable_outer_product`
- `add_same_spin_bridge_outer_product`

So `ggo_grad` does not need a new mathematical design. It only needs to be
assembled from the new collapsed operands.

### 3.2 Matrix Product Layer

After contraction backpropagation, the matrix chain rule is:

```text
U = F H
F_bar += U_bar H^T
H_bar += F^T U_bar
```

```text
V = F SA
F_bar  += V_bar SA^T
SA_bar += F^T V_bar
```

```text
D = B H
H_bar += B^T D_bar
```

```text
M = rho_0 A + C W
rho_0_bar += <M_bar, A>
C_bar     += M_bar W^T
W_bar     += C^T M_bar
```

```text
H = S C
S_bar += H_bar C^T
C_bar += S^T H_bar
```

```text
SA = S A
S_bar += SA_bar A^T
```

```text
C = A S B
S_bar += A^T C_bar B^T
```

Important:

- `A_bar` and `B_bar` are not needed for orbital/active-matrix optimization
- only `S_bar`, `ggo_grad`, and the trace-derived `kernel_bar` must be kept

### 3.3 Matrix Polynomial Layer

Three reverse helpers are needed.

#### Left polynomial

Forward:

```text
X_0 = source
X_m = H X_{m-1}
Y   = sum_m a_m X_m
```

Reverse:

```text
a_m_bar += <Y_bar, X_m>
X_m_bar += a_m Y_bar
X_{m-1}_bar += H^T X_m_bar
H_bar += X_m_bar X_{m-1}^T
```

#### Right polynomial

Forward:

```text
X_0 = source
X_m = X_{m-1} H
Y   = sum_m a_m X_m
```

Reverse:

```text
a_m_bar += <Y_bar, X_m>
X_m_bar += a_m Y_bar
X_{m-1}_bar += X_m_bar H^T
H_bar += X_{m-1}^T X_m_bar
```

#### Fréchet polynomial

The current forward recurrence is:

```text
R_1 = source
F_1 = source
R_p = R_{p-1} H
F_p = H F_{p-1} + R_p
Z   = sum_{p >= 1} a_p F_p
```

Reverse:

```text
a_p_bar += <Z_bar, F_p>
F_p_bar += a_p Z_bar
```

Then reverse the recurrence:

```text
from F_p = H F_{p-1} + R_p:
  H_bar      += F_p_bar F_{p-1}^T
  F_{p-1}_bar += H^T F_p_bar
  R_p_bar    += F_p_bar
```

```text
from R_p = R_{p-1} H:
  H_bar      += R_{p-1}^T R_p_bar
  R_{p-1}_bar += R_p_bar H^T
```

At the end:

```text
source_bar += F_1_bar + R_1_bar
```

These reverse helpers all scale as `O(n M^3)`.

## 4. Projected Coefficient Reverse

The new forward no longer depends only on `trace_weights`; it depends directly
on the whole coefficient sequence

```text
c_0, c_1, ..., c_n
```

stored in:

- `cache.projected_overlap_coefficients`

So `TraceProjector` needs a new reverse API that propagates adjoints of the
whole overlap coefficient series back to the trace series.

Recommended interface:

```cpp
static ScalarBuffer backpropagate_overlap_coefficient_adjoints(
    const ScalarBuffer& traces,
    int order,
    const ScalarBuffer& overlap_coefficient_adjoints);
```

The reverse is straightforward because the forward recursion is

```text
c_k = (1 / k) sum_{p = 1}^k s_p trace_p c_{k-p}
```

with

```text
s_p = (1/2) (-1)^(p+1)
```

Given incoming `c_bar`, the reverse sweep is:

```text
trace_p_bar += c_k_bar * (1/k) s_p c_{k-p}
c_{k-p}_bar += c_k_bar * (1/k) s_p trace_p
```

This new helper is the bridge from the collapsed coefficient adjoints back to
the trace series, and then to the kernel trace adjoint.

## 5. Coefficient Mapping Reverse

The closed-shell forward uses two projected coefficient families:

```text
rho_m^(n-1) = (1/2) (-1)^m c_{n-1-m}
pi_m^(n-2)  =       (-1)^m c_{n-2-m}
```

So reverse accumulation is simply:

```text
c_{n-1-m}_bar += (1/2) (-1)^m rho_m_bar
c_{n-2-m}_bar +=       (-1)^m pi_m_bar
```

The full coefficient adjoint of the overlap series is then backpropagated by
the new `TraceProjector` reverse helper.

## 6. Kernel And Sigma Backpropagation

The collapsed closed-shell H2 formula depends on `K` only through the projected
coefficients `c_k`.

So the kernel-trace path is now:

```text
coefficient_bar -> trace_bar -> kernel_bar
```

with

```text
kernel_bar += build_kernel_trace_adjoint(cache.kernel_powers, trace_bar)
```

The explicit spatial-overlap dependence enters separately through `S_bar`
coming from:

- `C = A S B`
- `H = S C`
- `SA = S A`

So the total sigma adjoint should be assembled from two pieces:

1. direct spatial-overlap contribution from the collapsed spatial formula
2. `backpropagate_sigma(...)` applied to the final `kernel_bar`

### Important implementation detail

The returned `PfAdjointResult` stores a spin-orbital `sigma_adjoint`, but the
closed-shell collapsed formula is written in terms of a single spatial matrix
`S`.

If a direct spatial contribution `S_bar` is obtained from the collapsed
formula, it should be written symmetrically into the spin blocks as

```text
sigma_bar(aa) += 0.5 * S_bar
sigma_bar(bb) += 0.5 * S_bar
```

so that

```text
collapse_spin_diagonal_blocks(sigma_bar) = S_bar
```

This avoids double counting when the final `spatial_density` is formed by block
collapse.

## 7. Replacement Plan

The new implementation order should be:

1. add a markdown summary of the adjoint graph
2. add `TraceProjector::backpropagate_overlap_coefficient_adjoints(...)`
3. add reverse helpers for
   - left polynomial
   - right polynomial
   - Fréchet polynomial
4. replace the current closed-shell H2 adjoint in
   `accumulate_closed_shell_two_electron_exact(...)`
   with a new collapsed reverse path
5. keep using existing packed tensor contraction primitives for `ggo_grad`

## 8. Complexity Target

With the collapsed reverse path:

- packed tensor contraction adjoints: `O(M^4)`
- matrix-polynomial reverse helpers: `O(n M^3)`
- trace projector reverse: lower order
- `n <= M`

So the total closed-shell H2 adjoint remains within the intended `O(M^4)`
scaling target.

