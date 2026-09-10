# Code Organization and Naming

This document defines the production module boundaries and local naming rules.
The directory provides the context, so filenames should describe only the
operation or data owned by that directory.

## Production Modules

```text
src/
  core/     Small numerical and execution utilities shared across methods
  vbscf/    VBSCF mathematics, workflows, derivatives, and optimizers
  input/    XMI deck representation and construction of VBSCF inputs
  libcint/  AO integral providers and libcint-specific preparation
  guess/    Initial-orbital and restricted-HF construction
  output/   Molden and accepted-iteration serialization
  cli/      Command-line parsing, execution, and reporting
  tools/    Developer diagnostics and benchmarks
```

There is no `application` or `runtime` layer. The command-line interface is one
consumer of independently named production components, not the owner of their
scientific logic.

## Build Dependencies

The static-library dependency direction is:

```text
xmvb_vbscf
    ^
xmvb_libcint
    ^
xmvb_input_deck
    ^
xmvb_guess
    ^
xmvb_input
    ^
xmvb_output
```

Some components depend directly on more than the immediately preceding node;
all dependencies still point toward `xmvb_vbscf`. `xmvb_input_deck` is a
separate build target because guess construction consumes parsed deck data,
while complete input loading consumes the guess builder. This keeps the graph
acyclic without introducing an umbrella library.

The `xmvb` executable owns only files in `src/cli/`. Diagnostic executables are
excluded from the default build and live in `src/tools/`.

## Filenames

- Do not add a `cpp_` prefix; C++ is already the language of the repository.
- Do not encode obsolete implementations with `legacy`, `compat`, or
  `fallback` names. Delete retired paths.
- Avoid umbrella names such as `application`, `runtime`, `manager`, or `utils`
  when a scientific or I/O responsibility can be stated directly.
- Prefer a short noun for a data contract and a short operation name for an
  implementation, using the containing directories to supply context.

## Dimension Names

Use these local abbreviations consistently in numerical kernels:

| Name | Meaning |
| --- | --- |
| `n_bf` | number of AO basis functions |
| `n_ao` | number of active orbitals |
| `n_orb` | total number of orbitals |
| `n_occ` | number of occupied orbitals |
| `n_inact` | number of inactive orbitals |
| `n_det` | number of determinants |
| `n_str` | number of VB structures |

Public data contracts may retain explicit names such as
`n_basis_functions` and `n_active_orbitals`. Short forms are for local
expressions whose scope makes the meaning unambiguous. In particular, never use
`n_ao` for the AO basis dimension; use `n_bf` so it cannot be confused with the
active-orbital count.

Dense physical vectors and matrices are Eigen objects. `std::vector` remains
appropriate for ragged topology, discrete indices, packed integral records,
and raw external I/O buffers, but not as an implicit dense-matrix container.
