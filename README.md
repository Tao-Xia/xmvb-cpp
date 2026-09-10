# xmvb-cpp

A high-performance **Valence Bond Self-Consistent Field (VBSCF)** engine written in modern C++17.

Valence Bond theory provides direct chemical insight into bonding, reactivity, and electron correlation that molecular orbital methods cannot reveal. xmvb-cpp is an open-source C++ implementation of the [XMVB](https://github.com/xmvb/xmvb) valence bond program, designed for computational chemistry researchers who need VB-level analysis of molecular electronic structure.

## Features

- **VBSCF optimization** with multiple backends (L-BFGS, Truncated Newton Hessian-vector products)
- **Hamiltonian and overlap matrix construction** for general VB structures
- **Active-space orbital optimization** with flexible orbital constraints
- **RI (Resolution-of-Identity)** integral approximation for two-electron terms
- **Pfaffian VBSCF** ansatz for open-shell and multi-reference systems
- **Hartree-Fock and DFT initial guess** (libcint + libxc)
- **OpenMP parallelization** for compute-intensive kernels
- **Molden output** for orbital visualization

## Architecture

```
src/
  core/            Shared numerical infrastructure
  vbscf/           Canonical C++ VBSCF implementation
    orbitals/      Sparse/full-AO charts, gauges, and pullbacks
    integrals/     AO and active-space integral operators
    determinants/  Determinant-pair algebra and caches
    structures/    VB structure expansion and state matrices
    derivatives/   Gradient and matrix-free Hessian-vector products
    optimization/  L-BFGS and trust-region Newton solvers
    workflow/      End-to-end VBSCF evaluation
  pfaffian_vbscf/  Pfaffian-based VB methods
  runtime/         Input parsing, molecule/basis setup, integral preparation
  tools/           Benchmarking and diagnostic tools
  third_party/     Vendored dependencies (LBFGSpp)
basis/             Standard basis set library (Pople, Dunning, etc.)
testdata/vbscf/    Versioned VBSCF regression decks
tests/vbscf/       Unit and numerical-regression tests
```

## Dependencies

| Library | Purpose |
|---------|---------|
| [Eigen3](https://eigen.tuxfamily.org/) | Dense linear algebra |
| [OpenBLAS](https://www.openblas.net/) | BLAS/LAPACK routines |
| [libcint](https://github.com/sunqm/libcint) | Gaussian integral evaluation |
| [libxc](https://www.tddft.org/programs/libxc/) | Exchange-correlation functionals |
| [LBFGSpp](https://github.com/yixuan/LBFGSpp) (vendored) | L-BFGS optimization |

## Build

Use a conda environment providing cmake, ninja, eigen, openblas (with LAPACKE headers), libxc, and libcint, then build with CMake:

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

This produces `build/src/xmvb-cpp.exe`.

To enable optional diagnostic and test targets:

```bash
cmake -B build -G Ninja -DXMVB_BUILD_DEV_TARGETS=ON
cmake --build build
```

## Usage

```bash
OMP_NUM_THREADS=1 build/src/xmvb-cpp.exe <input.xmi> --optimizer-backend lbfgspp
```

Input files use the `.xmi` format. See `testdata/vbscf/` for the versioned
regression decks and `testdata/vbscf/F2.xmi` for the compact HAO smoke case.

## License

[MIT](LICENSE)
