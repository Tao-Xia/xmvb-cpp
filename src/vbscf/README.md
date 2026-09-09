# VBSCF module architecture

This directory is the canonical home of the C++ VBSCF implementation. The only
VBSCF-related files intentionally left under `src/vb` are the tracked DeepVBH
prototype and the six forwarding headers required to compile it. They form a
compatibility island and must not be used by new production code.

## Dependency direction

Production modules follow this acyclic dependency chain:

```text
core
  -> orbitals + integrals
  -> determinants
  -> structures
  -> derivatives
  -> optimization
  -> workflow
```

`approx` and `legacy` are terminal modules. Production code must not include
from them. DeepVBH remains outside this tree under `src/vb` and is not part of
the matrix-free VBSCF optimizer. Its sources build as the one-way dependent
`xmvb_cpp_deepvbh` compatibility library; `xmvb_cpp_vb` and
`xmvb_cpp_runtime` never link against it.

## Target layout

```text
vbscf/
  core/                 Stable data types and dimensions
  orbitals/
    charts/             HAO and full-AO OEO coordinate maps
    gauge/              Support-preserving gauge operations
  integrals/            AO and active-space integral transformations
  determinants/         Determinant overlap and Hamiltonian kernels
  structures/           VB structure expansion and state matrices
  derivatives/
    gradient/            Gradient evaluation and chart pullback
    hessian/             Matrix-free exact HVP and response operators
  optimization/
    krylov/              Matrix-free subspace primitives
    preconditioners/     Reduced-space preconditioners
    trust_region/        Trust-region model solvers and radius updates
  workflow/              End-to-end VBSCF evaluation and orchestration
  adaptive/              Adaptive structure-space algorithms
  diagnostics/           Audits and memory/performance reporting
  approx/                Optional approximate models
  legacy/                Compatibility-only implementations
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

The namespace remains `xmvb::vb` during physical migration to avoid mixing an
ABI-wide namespace change with file ownership changes.

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
Legacy orbital and structure algorithms are isolated under `legacy`.
