# VBSCF module architecture

This directory is the canonical home of the C++ VBSCF implementation. The
legacy `src/vb` tree is being migrated incrementally so that file moves never
hide numerical changes. During the migration, `sources.cmake` records ownership
for files that have not moved yet.

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

`approx`, `legacy`, and `experimental` are terminal modules. Production code
must not include from them. In particular, DeepVBH belongs to `experimental`
and is not part of the matrix-free VBSCF optimizer.

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
  experimental/          DeepVBH and other research prototypes
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

## Migration policy

1. Move one coherent module at a time without changing numerical behavior.
2. Update all production includes to the canonical path.
3. Keep a thin forwarding header at the old public path for one migration
   window; forwarding headers contain no implementation.
4. Build and run the complete regression suite.
5. Commit the module migration with a clean worktree.
6. Remove forwarding headers after downstream users have migrated.

Migrated slices now include `optimization`, the single-step evaluator in
`workflow`, the complete orbital/chart/gauge layer, AO and active-space
`integrals`, and the independent chart audit. Legacy orbital algorithms are
isolated under `legacy/orbitals`. Remaining files listed in `sources.cmake`
retain their legacy paths but already have explicit logical ownership.
