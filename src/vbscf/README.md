# VBSCF module architecture

This directory is the canonical home of the C++ VBSCF implementation. New code
must use this tree directly; obsolete implementation and compatibility trees
are removed instead of being maintained in parallel.

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
`active_space_pair_transforms.cpp` owns the AO-pair/active-pair coordinate
maps and their adjoint accumulation, while `ao_pair_integral_operator.cpp`
owns the matrix-free action of the AO two-electron integral operator.
`active_space_two_electron_directional.cpp` and
`active_space_two_electron_adjoint.cpp` compose those primitives into the
forward and transpose-Jacobian response paths.

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
  diagnostics/           Numerical and coordinate audits
```

## Naming rules

- Public types use domain names without implementation-language markers.
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

The namespace remains `xmvb::vb` to preserve the existing public API while the
physical module boundaries are now expressed by directories and build-source
ownership.

## Compatibility policy

1. New code includes only `vbscf/...` headers.
2. Compatibility headers contain aliases only, never implementation.
3. A compatibility header must have a tracked consumer; unused aliases are
   removed instead of being kept as speculative API surface.

The canonical tree owns `optimization`, the single-step evaluator in
`workflow`, the complete orbital/chart/gauge layer, AO and active-space
`integrals`, `determinants`, `structures`, `derivatives/gradient`,
`derivatives/hessian`, and diagnostics.
Algorithms retained for exact comparison are colocated with their owning
domain and named explicitly, for example
`structures/reference/raw_structure_overlap` and
`orbitals/gauge/legacy_jacobi_diagonalizer`.
