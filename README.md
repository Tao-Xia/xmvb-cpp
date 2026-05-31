# xmvb-cpp

A modern C++17 implementation of **Valence Bond Self-Consistent Field (VBSCF)** and related electronic structure methods for quantum chemistry.

Valence Bond theory provides direct chemical insight into bonding, reactivity, and electron correlation that molecular orbital methods cannot. xmvb-cpp is the standalone, open-source C++ rewrite of [XMVB](https://github.com/xmvb/xmvb), a widely used VB program in computational chemistry research.

## Features

- **VBSCF optimization** with multiple optimizer backends (L-BFGS, Truncated Newton Hessian-vector products)
- **Hamiltonian and overlap matrix construction** for general VB structures
- **Active-space orbital optimization** with flexible orbital constraints
- **RI (Resolution-of-Identity)** integral approximation for two-electron terms
- **Pfaffian VBSCF** ansatz support for open-shell and multi-reference systems
- **ML-assisted optimization** via DeepVBH (ONNX Runtime integration)
- **Hartree-Fock and DFT initial guess** (libcint + libxc)
- **OpenMP parallelization** for compute-intensive kernels
- **Molden output** for orbital visualization

## Project Structure

```
src/
  core/            # Linear algebra, shared data structures
  vb/
    matrices/      # Hamiltonian and overlap matrix builders
    model/         # VB model definitions
    orbital/       # Orbital optimization and active space
    scf/           # SCF optimizer implementations
  pfaffian_vbscf/  # Pfaffian-based VB methods
  runtime/         # Input parsing, molecule/basis setup, integral preparation
  tools/           # Benchmarking and diagnostic tools
  third_party/     # Vendored dependencies (LBFGSpp)
basis/             # Standard basis set library (Pople, Dunning, etc.)
```

## Dependencies

| Library | Purpose |
|---------|---------|
| [Eigen3](https://eigen.tuxfamily.org/) | Dense linear algebra |
| [OpenBLAS](https://www.openblas.net/) | BLAS/LAPACK routines |
| [libcint](https://github.com/sunqm/libcint) | Gaussian integral evaluation |
| [libxc](https://www.tddft.org/programs/libxc/) | Exchange-correlation functionals |
| [ONNX Runtime](https://onnxruntime.ai/) (optional) | DeepVBH ML inference |
| [LBFGSpp](https://github.com/yixuan/LBFGSpp) (vendored) | L-BFGS optimization |

## Build

Create the development environment with conda:

```bash
conda env create -f environment.yml
conda activate xmvb-cpp-dev
```

Build with the provided script:

```bash
./build.sh
```

This produces `build/src/xmvb-cpp.exe` using Ninja.

To enable optional diagnostic and test targets:

```bash
./build.sh build -DXMVB_CPP_BUILD_DEV_TARGETS=ON
```

## Usage

Run a VBSCF calculation:

```bash
OMP_NUM_THREADS=1 build/src/xmvb-cpp.exe src/test_molecule/F2.xmi --optimizer-backend lbfgspp
```

Input files use the `.xmi` format. See `src/test_molecule/` for examples.

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Contributions are welcome. Please open an issue or pull request on GitHub.
