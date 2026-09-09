# VBSCF module architecture

This directory is the canonical home of the C++ VBSCF implementation. The only
VBSCF-related files intentionally left under `src/vb` are the tracked DeepVBH
prototype and its forwarding headers. They form a
compatibility island and must not be used by new production code.

## Responsibility flow

The main evaluation flow is:

```text
input contracts
  -> orbital preparation + AO/active integral transformation
  -> determinant-pair kernels and caches
  -> structure Hamiltonian/overlap assembly
  -> energy/gradient evaluation
  -> matrix-free HVP responses
  -> orbital optimization
```

`core` owns aggregate input/result contracts, so it may reference value types
from lower numerical domains. `approx` is an optional side path and
`diagnostics` contains audits only. DeepVBH remains outside this tree under
`src/vb` and is not part of the matrix-free VBSCF optimizer. Its sources build
as the one-way dependent
`xmvb_cpp_deepvbh` compatibility library; `xmvb_cpp_vb` and
`xmvb_cpp_runtime` never link against it.

## Target layout

```text
vbscf/
  core/                 Aggregate input/result and algorithm contracts
  orbitals/
    charts/             HAO and full-AO OEO coordinate maps
    gauge/              Support-preserving gauge operations
  integrals/            AO and active-space integral transformations
  determinants/         Determinant overlap and Hamiltonian kernels
  structures/           VB structure expansion and state matrices
    reference/          Raw-expansion reference implementations
  derivatives/
    gradient/            Gradient evaluation and chart pullback
    hessian/             Matrix-free HVP orchestration and accepted contexts
      responses/         Orbital, same-spin, and opposite-spin responses
  optimization/
    krylov/              Matrix-free subspace primitives
    preconditioners/     Reduced-space preconditioners
    trust_region/        Trust-region model solvers and radius updates
  workflow/              End-to-end VBSCF evaluation and orchestration
  adaptive/              Adaptive structure-space algorithms
  diagnostics/           Numerical and coordinate audits
  approx/                Optional approximate models
```

## Naming rules

- Public C++ types omit the redundant `Cpp` prefix.
- Files describe one responsibility; generic `utils` files are not allowed in
  the canonical tree.
- `Context` is immutable data prepared at an accepted point.
- `Workspace` is mutable scratch storage scoped to an operation.
- `Operator` applies a mathematical linear or nonlinear action.
- `Evaluator` computes a complete observable or result.
- `Builder` constructs a reusable immutable object.
- `dense` describes storage/layout; `exact` describes the numerical model.
- HAO and full-AO OEO charts are named explicitly and are not selected by raw
  integer orbital-type flags inside optimization code.

The namespace remains `xmvb::vb` to preserve the existing public API while the
physical module boundaries are now expressed by directories and build-source
ownership.

## Compatibility policy

1. New code includes only `vbscf/...` headers.
2. Compatibility headers contain aliases only, never implementation.
3. A compatibility header must have a tracked consumer; unused aliases are
   removed instead of being kept as speculative API surface.
4. DeepVBH compatibility is isolated under `src/vb/scf` until that prototype
   is migrated or retired separately.

The canonical tree owns `optimization`, the single-step evaluator in
`workflow`, the complete orbital/chart/gauge layer, AO and active-space
`integrals`, `determinants`, `structures`, `derivatives/gradient`,
`derivatives/hessian`, diagnostics, and adaptive structure-space algorithms.
Algorithms retained for exact comparison are colocated with their owning
domain and named explicitly, for example
`structures/reference/raw_structure_overlap` and
`orbitals/gauge/legacy_jacobi_diagonalizer`.
