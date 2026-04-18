# Closed-Shell Two-Electron Fusion Plan

## Scope

This note describes how to reduce the current closed-shell exact two-electron
forward path from 12 primitive packed-integral contractions to 2 fused
`O(M^4)` passes over the packed active-space tensor `ggo`.

This is an implementation plan only. The analytic formula is unchanged. The
goal is to keep the current exact dense closed-shell forward result while
removing repeated traversals of the same packed two-electron tensor.

Relevant production files:

- `src/pfaffian_vbscf/kernel/pf_closed_shell_spatial_kernel.cpp`
- `src/pfaffian_vbscf/tensor/pf_tensor_contractor.cpp`
- `src/pfaffian_vbscf/tensor/pf_tensor_contractor.hpp`
- `src/pfaffian_vbscf/kernel/closed_shell_forward_formula.md`

## Current Cost Structure

The current production formula is

```text
H^(2) = E_pair + E_cross + E_same_spin_bridge + E_opposite_bridge
```

with

```text
E_pair = EC(B, M_pair)

E_cross = XSep(C_r, C) + XSep(C, U)

E_same_spin_bridge = -2 Br(C_r, C) - 2 Br(C, U)

E_opposite_bridge = -(1/2) EC(D_r, A) -(1/2) EC(D, V)
```

and contraction shorthands

```text
EC(X, Y)   = Ex(X, Y) + Co(X, Y)
XSep(X, Y) = 2 Sep(X, Y) + Dir(X, Y)
```

Using the current contractor decomposition, one pairwise matrix element
requires:

- `E_pair`: 2 contractions
- `E_cross`: 4 contractions
- `E_same_spin_bridge`: 2 contractions
- `E_opposite_bridge`: 4 contractions

Total:

```text
12 primitive O(M^4) contractions
```

The asymptotic order is already `O(M^4)`, so the remaining optimization target
is the constant factor, especially the number of repeated `ggo` traversals.

## Reorganization Into Two Families

Define the already available closed-shell intermediates:

```text
A   = closed_shell_left_ba
B   = closed_shell_right_ab
C   = closed_shell_pair_density_aa
M   = closed_shell_pair_term_matrix
X   = closed_shell_c_poly_h_n2
U   = closed_shell_frechet_h_c_h_n2
D   = closed_shell_d
D_r = closed_shell_d_poly_h_n2
V   = closed_shell_opposite_bridge_h_split_matrix
```

Then the total two-electron contribution can be regrouped as

```text
H^(2) = E_full + E_asym
```

where

```text
E_full =
EC(B, M)
-(1/2) EC(D_r, A)
-(1/2) EC(D, V)
+ Dir(X, C)
+ Dir(C, U)
```

and

```text
E_asym =
2 [ Sep(X, C) - Br(X, C) ]
+ 2 [ Sep(C, U) - Br(C, U) ]
```

This split is exact. No term has been dropped. The only change is to group
contractions by traversal pattern.

## Pass 1: Full Quartet Fused Kernel

### Mathematical Form

The first family uses the unrestricted packed quartet traversal:

```text
E_full = sum_{q,p,s,r} g_{qp,sr} W_{qp,sr}
```

with coefficient tensor

```text
W_{qp,sr} =
  B_{pr} M_{sq}
+ B_{rp} M_{qs}
- (1/2) D_r(pr) A_{sq}
- (1/2) D_r(rp) A_{qs}
- (1/2) D_{pr} V_{sq}
- (1/2) D_{rp} V_{qs}
+ X_{qp} C_{sr}
+ C_{qp} U_{sr}
```

This is just the explicit expansion of

```text
EC(B, M)
-(1/2) EC(D_r, A)
-(1/2) EC(D, V)
+ Dir(X, C)
+ Dir(C, U)
```

### Loop Shape

The natural implementation is one fused pass over all packed quartets:

```text
for q
  for p
    for s
      for r
        value += g(qp,sr) * W(q,p,s,r)
```

This replaces:

- 3 separate `EC` evaluations
- 2 separate `Dir` evaluations

That is, 8 primitive contractions become 1 fused full-quartet pass.

### Proposed API

Add a specialized contractor entry point:

```text
double contract_closed_shell_full_linear_combo(
    const ScalarBuffer& ggo,
    int n_active_orbitals,
    const ConstMatrixRef& b,
    const ConstMatrixRef& m_pair,
    const ConstMatrixRef& d_r,
    const ConstMatrixRef& a,
    const ConstMatrixRef& d,
    const ConstMatrixRef& v,
    const ConstMatrixRef& x,
    const ConstMatrixRef& c,
    const ConstMatrixRef& u);
```

The function returns the total scalar `E_full` directly.

This API is intentionally specialized. It keeps the final closed-shell forward
path explicit and avoids a generic runtime descriptor layer in the hot path.

## Pass 2: Same-Spin Antisymmetrized Fused Kernel

### Mathematical Form

The second family uses the restricted antisymmetrized pair-pair traversal:

```text
E_asym =
2 sum_{p<r, q<s} (g_{qp,sr} - g_{qr,sp}) Z_{qprs}
```

with

```text
Z_{qprs} =
  X_{qp} C_{sr} - X_{qr} C_{sp}
+ C_{qp} U_{sr} - C_{qr} U_{sp}
```

This comes directly from

```text
2 [ Sep(X, C) - Br(X, C) ]
+ 2 [ Sep(C, U) - Br(C, U) ]
```

The useful point is that both the separable and bridge pieces share the same
antisymmetrized integral weight

```text
g_{qp,sr} - g_{qr,sp}
```

so they should be accumulated together.

### Loop Shape

The natural implementation is

```text
for p in [0, n-2]
  for q in [0, n-2]
    for r in [p+1, n-1]
      for s in [q+1, n-1]
        interaction = g(qp,sr) - g(qr,sp)
        value += 2 * interaction * Z(q,p,r,s)
```

This replaces:

- 2 separate `Sep` evaluations
- 2 separate `Br` evaluations

That is, 4 primitive same-spin contractions become 1 fused antisymmetrized
pass.

### Proposed API

Add a second specialized contractor entry point:

```text
double contract_closed_shell_same_spin_asym_linear_combo(
    const ScalarBuffer& ggo,
    int n_active_orbitals,
    const ConstMatrixRef& x,
    const ConstMatrixRef& c,
    const ConstMatrixRef& u);
```

The function returns the total scalar `E_asym` directly.

## Resulting Forward Structure

After the fusion, the production forward kernel becomes

```text
if trace_order <= 0:
  return 0

value = contract_closed_shell_full_linear_combo(...)

if n_pairs >= 2:
  value += contract_closed_shell_same_spin_asym_linear_combo(...)

return value
```

So the two-electron forward path changes from

```text
12 primitive contractions
```

to

```text
2 fused O(M^4) passes
```

without changing the analytic result.

## Why Stop At Two Passes First

A one-pass mega-kernel is possible in principle. The total energy is linear in
`ggo`, so one could combine the unrestricted and antisymmetrized pieces into a
single traversal that conditionally adds the same-spin correction whenever
`p < r` and `q < s`.

However, the 2-pass version is the better first production target:

- it preserves a clean separation between unrestricted and antisymmetrized loop
  structures
- it is easier to verify numerically against the current implementation
- it is easier to differentiate later in the adjoint path
- it keeps the hot loops specialized without becoming unreadable

The 1-pass design can be revisited only after the 2-pass kernel is validated
and profiled.

## Expected Benefit

This fusion does not change the formal dense exact scaling:

```text
O(M^4)
```

What it changes is the number of repeated passes over the packed two-electron
tensor:

- current: 12 passes
- proposed first target: 2 passes
- theoretical dense exact lower limit: 1 pass

The practical speedup will be lower than `12 / 2 = 6` for the total pairwise
evaluation because:

- matrix-polynomial intermediates are still built outside the contraction
- cache construction remains unchanged
- the one-electron term is unchanged
- upper-level basis assembly still contributes to wall time

Still, if the active bottleneck is repeated `ggo` contraction, the 2-pass
version should be a major constant-factor improvement.

## Implementation Steps

1. Add the two fused contractor APIs to
   `src/pfaffian_vbscf/tensor/pf_tensor_contractor.hpp`.
2. Implement both fused kernels in
   `src/pfaffian_vbscf/tensor/pf_tensor_contractor.cpp`.
3. Rewrite
   `src/pfaffian_vbscf/kernel/pf_closed_shell_spatial_kernel.cpp`
   so it only calls the fused contractors.
4. Keep all existing matrix-polynomial and cache intermediates unchanged.
5. Verify exact equality against the current 12-contraction implementation
   before removing the old path.

## Verification Plan

The verification should proceed in three layers.

### Layer 1: Pairwise Kernel Equality

For a fixed cache and `ggo`, compare:

- old `evaluate_closed_shell_two_electron_spatial_exact`
- new fused 2-pass implementation

Target:

```text
abs(delta) <= 1e-12
```

for representative closed-shell state pairs.

### Layer 2: Full Pair Hamiltonian Equality

For each tested pair `(I, J)`, compare the total matrix element

```text
H_IJ = H_IJ^(1) + H_IJ^(2)
```

between old and new implementations.

The overlap and one-electron terms should remain identical.

### Layer 3: End-To-End Molecular Equality

Run the existing closed-shell molecular tests and compare:

- overlap matrix
- Hamiltonian matrix
- SCF energy trajectory
- final converged energy

No numerical change should appear beyond floating-point noise.

## Non-Goals

This plan does not attempt to:

- change the analytic closed-shell formula
- introduce RI, Cholesky, or THC
- change the matrix-polynomial staging
- generalize the closed-shell path to open-shell cases
- optimize memory layout beyond the fused traversal itself

Those are separate follow-up steps.

## Final Recommendation

The best next implementation target for the closed-shell exact dense forward
kernel is:

```text
current analytic formula
+ current cache intermediates
+ 2 specialized fused O(M^4) contraction passes
```

This gives a strong production-quality closed-shell kernel without changing the
mathematics and without overcomplicating the forward path.
