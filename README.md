# xmvb-cpp

A high-performance **Valence Bond Self-Consistent Field (VBSCF)** engine written in modern C++17.

Valence Bond theory provides direct chemical insight into bonding, reactivity, and electron correlation that molecular orbital methods cannot reveal. xmvb-cpp is an independent open-source VBSCF engine designed for computational chemistry researchers who need VB-level analysis of molecular electronic structure.

## Architecture

```
src/
  core/            Shared eigensolver and OpenMP execution helpers
  vbscf/           Canonical VBSCF implementation
    orbitals/      Sparse/full-AO charts, gauges, and pullbacks
    integrals/     AO and active-space integral operators
    determinants/  Determinant-pair algebra and caches
    structures/    VB structure expansion and state matrices
    derivatives/   Gradient and matrix-free Hessian-vector products
    optimization/  L-BFGS and trust-region Newton solvers
    workflow/      End-to-end VBSCF evaluation
  input/           XMI deck contracts and VBSCF input loading
  libcint/         libcint providers and AO integral preparation
  guess/           Initial orbitals and restricted-HF construction
  output/          Molden and accepted-iteration output
  cli/             Command-line entry point and reporting
  tools/           Benchmarking and diagnostic tools
basis/             Standard basis set library (Pople, Dunning, etc.)
testdata/vbscf/    Versioned VBSCF regression decks
tests/vbscf/       Unit and numerical-regression tests
```

The component dependency rules and numerical naming conventions are recorded
in [Code Organization and Naming](docs/code_organization.md).

## Dependencies

| Library | Purpose |
|---------|---------|
| [Eigen3](https://eigen.tuxfamily.org/) | Dense linear algebra |
| [OpenBLAS](https://www.openblas.net/) | BLAS/LAPACK routines |
| [libcint](https://github.com/sunqm/libcint) | Gaussian integral evaluation |
| [libxc](https://www.tddft.org/programs/libxc/) | Exchange-correlation functionals |

## Build

Use a conda environment providing cmake, ninja, eigen, openblas (with LAPACKE headers), libxc, and libcint, then build with CMake:

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

This produces `build/src/xmvb-cpp.exe`.

To enable optional diagnostic and test targets:

```bash
cmake -B build -G Ninja -DXMVB_BUILD_DEV_TARGETS=ON
cmake --build build
```

## Usage

```bash
OMP_NUM_THREADS=1 build/src/xmvb-cpp.exe <input.xmi>
```

Input files use the `.xmi` format. See `testdata/vbscf/` for the versioned
regression decks and `testdata/vbscf/F2.xmi` for the compact HAO smoke case.
Within `$CTRL`, `ISCF=5` selects the XMVB-compatible raw-coordinate L-BFGS,
`ISCF=7` selects TNHVP, `ISCF=8` selects nonredundant block-LBFGS, and
`ISCF=9` selects NEO. The project default when `ISCF` is omitted is
block-LBFGS.
`EIGENSOLVER=DAVIDSON` selects the default matrix-free structure solver,
while `EIGENSOLVER=DENSE` selects the explicit dense reference solver. An
explicit `--optimizer-backend` or `--eigensolver` command-line option overrides
the corresponding input keyword.

Block-LBFGS uses the same local orbital-block initial inverse and transported
secant recursion as the TNHVP predictor, without exact HVP correction. The
`ISCF=5` baseline instead follows XMVB 4.0: standard L-BFGS in the stored raw
orbital coefficients, 100 correction pairs, an initial step of 0.2, and a
Moré--Thuente strong-Wolfe line search. The corresponding command-line names
are `--optimizer-backend lbfgs` and `--optimizer-backend block_lbfgs`.

`NSTATE=n` selects an equal-weight average over the consecutive lowest `n`
VB states. Omitting it is equivalent to `NSTATE=1`. The optimizer uses the
same normalized weight `1/n` for the energy, analytic gradient, and TNHVP
Hessian-vector product. See `testdata/vbscf/F2_SA2.xmi` for a two-state input.

Davidson runs retain only the requested roots, the structure-overlap diagonal,
and the matrix-free products needed for exact Coulson--Chirgwin weights. Their
reports omit the full structure Hamiltonian/overlap matrices and the Lowdin and
inverse weights that require global overlap-matrix functions. Dense reference
runs retain the complete matrices, spectrum, and all four weight definitions.

## License

[MIT](LICENSE)
