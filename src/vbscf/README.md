# VBSCF module architecture

This directory is the canonical home of the VBSCF implementation. New code
must use this tree directly; obsolete and duplicate implementations are
deleted instead of being maintained in parallel.

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
from lower numerical domains. `diagnostics` contains audits only.

The standalone `runtime` is a one-way client of this module. Canonical
`vbscf/...` sources must not include `runtime/...` headers. Backend-generated
data enters through numerical value contracts or injected provider interfaces;
this rule is enforced when CMake configures the source manifest.

The exact active-space two-electron response is split by mathematical role:
`active/two_electron/transformation/pair_transforms.cpp` owns the
AO-pair/active-pair coordinate maps and their adjoint accumulation, while
`transformation/ao_pair_operator.cpp` owns the matrix-free action of the AO
two-electron integral operator. `response/directional.cpp` and
`response/adjoint.cpp` compose those primitives into the forward and
transpose-Jacobian response paths.

## Target layout

```text
vbscf/
  core/                 Aggregate input/result and algorithm contracts
  orbitals/
    charts/             HAO and full-AO OEO coordinate maps
    gauge/              Support-preserving gauge operations
    preparation/        Validated orbital frames and preparation contracts
    pullback/           Coordinate-map adjoint and its result contract
  integrals/
    ao/
      contracts/        Backend-neutral AO integral input
      libcint/          libcint input and validation
      one_electron/     Effective one-electron construction and pullback
      pairs/            Packed AO-pair indexing
      ri/               RI factorization contracts and cache
    active/
      matrix/           Generic active-matrix pullback
      one_electron/     Active one-electron construction
      preparation/      Prepared active-space state
      two_electron/
        construction/   Exact/RI builders, kernels, and indexing
        response/       Forward and transpose-Jacobian actions
        transformation/ AO-pair/active-pair coordinate operators
  determinants/         Determinant overlap and Hamiltonian kernels
  structures/
    assembly/           Hamiltonian/overlap matrices and contractions
    evaluation/         Complete VB structure-state evaluation
    expansion/          Structure expansion contracts and transformations
    reference/          Independent mathematical reference evaluations
    selection/
      subspace/         Linearly independent subspace selection
      union_graph/      Union-graph screening and rank prediction
  derivatives/
    gradient/            Gradient evaluation and chart pullback
    hessian/
      context/           Accepted-point state and reusable response caches
      exact/             Exact matrix-free HVP orchestration
      responses/
        active_space/    Integral and outer active-space responses
        orbital/         Orbital-preparation response
        structure/       VB structure directional response
        same_spin/       Same-spin determinant-pair response kernels
        opposite_spin/   Opposite-spin response kernels
  optimization/
    backends/            L-BFGS, projected-gradient, and TN step backends
    driver/              Optimizer lifecycle, contracts, and session state
    globalization/       Line-search globalization
    krylov/              Matrix-free subspace primitives
    objective/           VBSCF objective and reduced HVP action
    preconditioners/     Reduced-space preconditioners
    trust_region/        Trust-region model solvers and radius updates
  workflow/              End-to-end VBSCF evaluation and orchestration
  diagnostics/           Numerical and coordinate audits
```

## Naming rules

- Public types use domain names without implementation-language markers.
- A branch directory contains only functional subdirectories, apart from its
  module manifest or architecture document. Source files live in leaf
  directories. This keeps every directory level at one abstraction level.
- A file name does not repeat meaning already supplied by its parent path. For
  example, `orbitals/charts/layout.hpp` is preferred over
  `orbitals/charts/sparse_parameter_layout.hpp`; the public type may retain the
  mathematically explicit name `SparseParameterLayout`.
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

### Dimensions and orbital spaces

VBSCF uses both an AO basis-function space and a variational active-orbital
space, so a bare `n_ao` is not allowed to mean the AO basis dimension. The
following abbreviations have fixed meanings throughout new and refactored
code:

- `n_bf`: number of AO basis functions;
- `n_ao`: number of active orbitals (the established VBSCF abbreviation);
- `n_orb`: total number of variational orbitals;
- `n_bf_pairs`: number of packed AO basis-function pairs;
- `n_active_pairs`: number of packed active-orbital pairs.

The prefix `ao_` on a matrix or tensor still denotes representation in the
atomic-orbital basis, for example `ao_overlap` or `ao_h1e`. Active-space
objects use `active_`. Dimension variables must use the explicit conventions
above rather than inheriting the meaning of a nearby tensor prefix.

Mathematical state qualifiers are retained rather than shortened away:
`accepted_` identifies accepted-point invariants, `delta_` identifies a
directional derivative, and `packed_` or `dense_` identifies storage. Within a
short local scope, `h1e` and `eri` may replace `one_electron` and
`two_electron_integrals`; public types and functions should prefer the longer
form unless the shorter term is already the domain-standard name.

Operation names also carry fixed semantics: `directional` is a forward
Jacobian action, `adjoint` is a transpose-Jacobian action, `hvp` is a Hessian
action, and `pullback` is reserved for a coordinate-map adjoint.

The namespace is `xmvb::vb`; physical module boundaries are expressed by
directories and build-source ownership.

## Source policy

1. Each algorithm has one maintained implementation; compatibility aliases and
   inactive alternatives are deleted.
2. Reference implementations are retained only when they provide an
   independent mathematical correctness check used by the test suite.
3. Source names describe their mathematical or software responsibility, never
   an implementation language or a superseded program.
4. Unused source files, targets, options, and experimental branches are removed
   rather than preserved for possible future use.

The canonical tree owns `optimization`, the single-step evaluator in
`workflow`, the complete orbital/chart/gauge layer, AO and active-space
`integrals`, `determinants`, `structures`, `derivatives/gradient`,
`derivatives/hessian`, and diagnostics.
Algorithms retained for independent mathematical validation are colocated
with their owning domain and named explicitly, for example
`structures/reference/overlap`.
