# `PfKernelCache` Field Semantics

This note explains what each `PfKernelCache` field actually means in the
current codebase. The main source of confusion is that the struct currently
stores two different layers of objects:

1. a generic spin-orbital Pfaffian kernel cache, and
2. a closed-shell spatial fast path used by the production forward/adjoint
   two-electron code.

These two layers coexist in the same struct, so names such as `sigma`,
`left`, `right`, `closed_shell_pair_core`, and `closed_shell_h` are easy
to misread if one assumes they all belong to one uniform notation.

## 1. High-Level Split

The cache should be read as two groups of fields.

### Group A: Generic spin-orbital cache

These fields belong to the original generic Pfaffian kernel

```text
K = L Sigma R Sigma^T
```

where `L`, `R`, and `Sigma` are all `2M x 2M` spin-orbital matrices.

Fields in this group:

- `left`
- `right`
- `sigma`
- `kernel`
- `right_sigma`
- `left_sigma_right`
- `kernel_powers`
- `kernel_power_times_left`
- `kernel_power_times_right`
- `kernel_power_times_left_sigma_right`
- `right_sigma_times_kernel_power_times_left_sigma_right`
- `traces`
- `trace_rdms`

### Group B: Closed-shell spatial fast path

These fields belong to the determinant-free closed-shell production formula.
They are `M x M` spatial objects.

Fields in this group:

- `closed_shell_spatial_overlap`
- `closed_shell_left_ba_block`
- `closed_shell_right_ab_block`
- `closed_shell_pair_core`
- `closed_shell_spatial_kernel`
- `closed_shell_sa`
- `closed_shell_h`
- `closed_shell_d`
- `closed_shell_traces`
- `closed_shell_trace_powers`
- `closed_shell_pair_coefficients`
- `closed_shell_pair_tail_coefficients`
- `closed_shell_coeff_n2`
- `closed_shell_pair_poly`
- `closed_shell_pair_term_matrix`
- `closed_shell_c_poly_h_n2`
- `closed_shell_frechet_h_c_n2`
- `closed_shell_frechet_h_c_h_n2`
- `closed_shell_d_poly_h_n2`
- `closed_shell_opposite_bridge_h_split_matrix`

## 2. Builder Entry Points

Three public builders populate different subsets of the cache.

### `PfForwardKernel::build_cache(...)`

Defined in:

- `src/pfaffian_vbscf/kernel/pf_forward_kernel.cpp`

Input:

- `left`, `sigma`, `right` as full `2M x 2M` matrices

Populates:

- generic spin-orbital fields
- closed-shell spatial fields

This is the only path that fully materializes the generic `kernel`,
`kernel_powers`, `trace_rdms`, and related auxiliary matrices.

### `PfForwardKernel::build_closed_shell_exact_cache(...)`

Input:

- `left`, `sigma`, `right` as full `2M x 2M` matrices

Populates:

- `left`, `right`, `sigma`
- closed-shell spatial fields
- overlap projection metadata

Does not populate the full generic `kernel` family. This is a closed-shell fast
path that still starts from full spin-orbital inputs.

### `PfForwardKernel::build_closed_shell_exact_spatial_cache(...)`

Input:

- `left_ba`, `spatial_overlap`, `right_ab` as `M x M` spatial blocks

Populates:

- closed-shell spatial fields
- overlap projection metadata

This path may leave `left`, `right`, `sigma`, and the generic `kernel` family
empty. It is intended for pure spatial closed-shell work where the full
spin-orbital cache is not needed.

## 3. Core Dimensions

The cache uses two dimensions:

- `n_active_orbitals = M`
- `n_spin_orbitals = 2M`

So:

- generic spin-orbital matrices are `(2M, 2M)`
- closed-shell spatial matrices are `(M, M)`

## 4. Generic Spin-Orbital Fields

### `left`, `right`

Meaning:

- full spin-orbital antisymmetric Pfaffian state matrices

Dimension:

- `(2M, 2M)`

Interpretation:

- these are not spatial pair matrices
- they are the full spin-resolved objects from which `AB` and `BA` blocks can
  be extracted

### `sigma`

Meaning:

- spin-orbital overlap metric `Sigma`

Dimension:

- `(2M, 2M)`

Important:

- this is not the spatial overlap matrix `S`
- in closed-shell work it is typically the block-diagonal embedding of the
  spatial overlap

```text
Sigma = diag(S, S)
```

### `kernel`

Meaning:

- generic spin-orbital Pfaffian kernel

Formula:

```text
kernel = left * sigma * right * sigma^T
```

Dimension:

- `(2M, 2M)`

### `left_sigma_right`

Meaning:

- the generic helper

```text
left_sigma_right = left * sigma * right
```

Dimension:

- `(2M, 2M)`

### `right_sigma`

Meaning:

- the generic helper

```text
right_sigma = right * sigma
```

Dimension:

- `(2M, 2M)`

### `kernel_powers`

Meaning:

- cached powers `[I, K, ..., K^(n-1)]` for the generic spin-orbital kernel

Dimension of each entry:

- `(2M, 2M)`

### `traces`

Meaning:

- generic trace series `[Tr(K), ..., Tr(K^n)]`

### `trace_rdms`

Meaning:

- generic trace-derived spin-orbital density helpers

Dimension of each entry:

- `(2M, 2M)`

## 5. Closed-Shell Spatial Fields

These are the fields that matter for the current production closed-shell
forward and adjoint path.

### `closed_shell_spatial_overlap`

Meaning:

- the spatial overlap matrix `S`

Dimension:

- `(M, M)`

This is the actual spatial overlap matrix. If you are reading the
closed-shell formula, this is the field that corresponds to `S`, not
`cache->sigma`.

### `closed_shell_left_ba_block`

Meaning:

- the `BetaAlpha` block extracted from the left spin-orbital Pfaffian matrix

Dimension:

- `(M, M)`

### `closed_shell_right_ab_block`

Meaning:

- the `AlphaBeta` block extracted from the right spin-orbital Pfaffian matrix

Dimension:

- `(M, M)`

Important:

- these are implementation-space block objects
- they are not yet the paper's final `A_I` / `B_J` symbols until one fixes the
  block-orientation convention

### `closed_shell_pair_core`

Meaning:

- the current code's core closed-shell spatial matrix

Construction:

```text
C_code = left_ba_block^T * S * right_ab_block^T
```

Dimension:

- `(M, M)`

Important:

- despite the name, this is not the final physical spatial 1-RDM
- it is the central pair-kernel object from which overlap, projected
  coefficients, and later spatial helpers are built

### `closed_shell_spatial_kernel`

Meaning:

- the current code's right-multiplied closed-shell spatial kernel

Construction:

```text
closed_shell_spatial_kernel = closed_shell_pair_core * S
```

Dimension:

- `(M, M)`

This is the matrix whose trace powers drive the closed-shell projection in the
current implementation.

### `closed_shell_traces`

Meaning:

- trace series of `closed_shell_spatial_kernel`

Important:

- these traces are multiplied by `2.0` in the code to account for the two
  identical spin-diagonal sectors in the closed-shell reduction

### `closed_shell_trace_powers`

Meaning:

- cached powers `[I, H, ..., H^(n-1)]` of `closed_shell_spatial_kernel`

Dimension of each entry:

- `(M, M)`

### `closed_shell_sa`

Meaning:

- internal helper constructed as

```text
closed_shell_sa = S * closed_shell_left_ba_block
```

Dimension:

- `(M, M)`

Important:

- the name comes from implementation history
- it should not be confused with a standalone mathematical symbol unless the
  block-orientation convention is stated explicitly

### `closed_shell_h`

Meaning:

- internal helper constructed as

```text
closed_shell_h = S * closed_shell_pair_core
```

Dimension:

- `(M, M)`

Important:

- this is not the one-electron Hamiltonian
- the name `h` here is historical and unfortunate
- it is a left-multiplied spatial kernel helper used in the projected
  polynomial/bridge formulas

### `closed_shell_d`

Meaning:

- internal helper constructed as

```text
closed_shell_d = closed_shell_right_ab_block * closed_shell_h
```

Dimension:

- `(M, M)`

### `closed_shell_pair_coefficients`

Meaning:

- projected coefficients for the `(n-1)` pair sector, including the `1/2`
  scaling used by the opposite-spin pair term

### `closed_shell_pair_tail_coefficients`

Meaning:

- `closed_shell_pair_coefficients` without the leading constant term

### `closed_shell_coeff_n2`

Meaning:

- projected coefficients for the `(n-2)` pair sector

### `closed_shell_pair_poly`

Meaning:

- projected polynomial applied to `closed_shell_sa`

### `closed_shell_pair_term_matrix`

Meaning:

- the final opposite-spin pair kernel matrix used in the closed-shell exact
  two-electron formula

Dimension:

- `(M, M)`

### `closed_shell_c_poly_h_n2`

Meaning:

- projected `(n-2)` right-polynomial density helper

### `closed_shell_frechet_h_c_n2`

Meaning:

- projected `(n-2)` Fréchet helper before multiplication by `closed_shell_h`

### `closed_shell_frechet_h_c_h_n2`

Meaning:

- product helper

```text
closed_shell_frechet_h_c_h_n2 =
    closed_shell_frechet_h_c_n2 * closed_shell_h
```

### `closed_shell_d_poly_h_n2`

Meaning:

- projected `(n-2)` right-polynomial applied to `closed_shell_d`

### `closed_shell_opposite_bridge_h_split_matrix`

Meaning:

- opposite-spin bridge helper

```text
closed_shell_opposite_bridge_h_split_matrix =
    closed_shell_frechet_h_c_n2 * closed_shell_sa
```

## 6. Most Common Misreadings

### Misreading 1: `sigma` means spatial overlap

Not in the generic cache.

- `sigma` means full spin-orbital metric `Sigma`, dimension `(2M, 2M)`
- `closed_shell_spatial_overlap` means spatial overlap `S`, dimension `(M, M)`

### Misreading 2: `left` and `right` are spatial pair matrices

Not in the generic cache.

- `left` and `right` are full spin-orbital antisymmetric Pfaffian state
  matrices, dimension `(2M, 2M)`
- `closed_shell_left_ba_block` and `closed_shell_right_ab_block` are the extracted spatial
  blocks used by the closed-shell fast path

### Misreading 3: `closed_shell_pair_core` is the final physical 1-RDM

It is not.

It is the core closed-shell spatial object `C_code` from which the actual
projected 1-RDM is built later through the coefficient recursion and matrix
polynomials.

### Misreading 4: `closed_shell_h` is the one-electron Hamiltonian

It is not.

`closed_shell_h` is only an internal spatial helper equal to

```text
S * closed_shell_pair_core
```

The one-electron Hamiltonian lives elsewhere in the SCF machinery.

## 7. Practical Reading Guide

If you are reading the current production closed-shell forward/adjoint path,
focus on these fields:

- `closed_shell_spatial_overlap`
- `closed_shell_left_ba_block`
- `closed_shell_right_ab_block`
- `closed_shell_pair_core`
- `closed_shell_spatial_kernel`
- `closed_shell_pair_term_matrix`
- `closed_shell_c_poly_h_n2`
- `closed_shell_frechet_h_c_n2`
- `closed_shell_frechet_h_c_h_n2`
- `closed_shell_d_poly_h_n2`
- `closed_shell_opposite_bridge_h_split_matrix`

If you are reading the older generic tensor-term framework, focus on:

- `left`
- `right`
- `sigma`
- `kernel`
- `kernel_powers`
- `trace_rdms`

Treat these as a separate layer.

## 8. Bottom Line

The current names are not fully wrong, but they do mix two notational worlds:

- a generic `2M x 2M` spin-orbital Pfaffian kernel world, and
- a closed-shell `M x M` spatial fast-path world

So the safest rule is:

- read `sigma/left/right/kernel` as generic spin-orbital objects
- read `closed_shell_*` as the actual production spatial closed-shell objects

Any future refactor should ideally separate these two layers more explicitly,
either by splitting the cache struct or by renaming the generic fields to
`spin_left`, `spin_right`, `spin_metric`, and similar names.
