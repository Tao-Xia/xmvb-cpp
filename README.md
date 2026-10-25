# xmvb-cpp

A high-performance **Valence Bond Self-Consistent Field (VBSCF)** engine written in C++23.

Valence Bond theory provides direct chemical insight into bonding, reactivity, and electron correlation that molecular orbital methods cannot reveal. xmvb-cpp is an independent open-source VBSCF engine designed for computational chemistry researchers who need VB-level analysis of molecular electronic structure.

## Architecture

```
src/
  core/            Pair indexing and lightweight timing
  linalg/          Eigensolvers, eigen response, metric solves, and Krylov algebra
  parallel/        Shared OpenMP execution and work partitioning
  integrals/       Shared exact/RI AO operators, active transforms and derivatives
  mcscf/           Shared direct-CI, state contracts, and CIAH/NEO solvers
  casscf/          Orthogonal CASCI model (orbital optimization not yet exposed)
  vbscf/           Canonical VBSCF implementation
    orbitals/      Sparse/full-AO charts, gauges, and pullbacks
    integrals/     VB orbital-preparation and integral-adjoint adapters
    determinants/  Determinant-pair algebra and caches
    structures/    VB structure expansion and state matrices
    derivatives/   Gradient and matrix-free Hessian-vector products
    optimization/  VB coordinates, physical preconditioners, and solver adapters
    workflow/      End-to-end VBSCF evaluation
  input/           XMI deck contracts and VBSCF input loading
  libcint/         Method-independent libcint providers (no VB dependency)
  guess/           Initial orbitals and restricted-HF construction
  scf/             Common restricted SCF/DIIS and Hartree-Fock evaluation
  dft/             Molecular grids, Libxc LDA/GGA, and RKS/UKS models
  output/          Molden and accepted-iteration output
  cli/             Command-line entry point and reporting
  tools/           Benchmarking and diagnostic tools
basis/             Standard basis set library (Pople, Dunning, etc.)
testdata/vbscf/    Versioned VBSCF regression decks
tests/vbscf/       Unit and numerical-regression tests
tests/mcscf/       Independent integrals, CI, CASCI, and common-solver tests
tests/linalg/      Linear-algebra tests with no electronic-model dependency
tests/parallel/    Shared-buffer and nested-team execution tests
```

Shared numerical modules do not depend on VB-specific model types. AO basis
counts use `n_bf`; active orbital counts use `n_active`.

## Dependencies

| Library | Purpose |
|---------|---------|
| [Eigen3](https://eigen.tuxfamily.org/) | Dense linear algebra |
| [OpenBLAS](https://www.openblas.net/) | BLAS/LAPACK routines |
| [libcint](https://github.com/sunqm/libcint) | Gaussian integral evaluation |
| [libxc](https://www.tddft.org/programs/libxc/) | Exchange-correlation functionals |

## Build

Use a C++23 compiler (tested with GCC14), CMake3.20 or newer, and an environment providing ninja, eigen, openblas (with LAPACKE headers), libxc, and libcint, then build with CMake:

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

## C++ and CUDA code style

Use clang-format and clang-tidy 20.1.0 with the repository's `.clang-format`
and `.clang-tidy`. The formatter uses four-space indentation, a 100-column
limit, and the formatter's `Latest` syntax mode for C++23 sources; it preserves
include order and comment wrapping.
Format the files you change rather than reformatting unrelated code:

```bash
clang-format --dry-run --Werror path/to/file.cpp path/to/kernel.cu
clang-format -i path/to/file.cpp path/to/kernel.cu
clang-tidy -p build path/to/file.cpp
```

Configure the matching build first so clang-tidy can use its exported
`compile_commands.json`. The enabled tidy checks target correctness; review
findings individually and do not apply automatic fixes to numerical code.
CUDA formatting uses the same rules; CUDA compilation remains C++20 as set
in CMake. Run builds and numerical validation through Slurm on compute nodes.

## Usage

```bash
OMP_NUM_THREADS=1 build/src/xmvb-cpp.exe <input.xmi>
```

Input files use the `.xmi` format. See `testdata/vbscf/` for the versioned
regression decks and `testdata/vbscf/F2.xmi` for the compact HAO smoke case.
Within `$CTRL`, `ISCF=7` selects TNHVP,
`ISCF=8` selects preconditioned L-BFGS (P-LBFGS), and
`ISCF=9` selects co-iterative augmented Hessian (CIAH). The project default
when `ISCF` is omitted is P-LBFGS.
`EIGENSOLVER=DAVIDSON` selects the default matrix-free structure solver,
while `EIGENSOLVER=DENSE` selects the explicit dense reference solver. An
explicit `--optimizer-backend` or `--eigensolver` command-line option overrides
the corresponding input keyword.

P-LBFGS uses the same local orbital-block initial inverse and transported
secant recursion as the TNHVP predictor, without exact HVP correction. The
native raw-coordinate L-BFGS backend and its Moré--Thuente line search have
been removed. `ISCF=5` and `--optimizer-backend lbfgs` are rejected, not
redirected to another method. Use `--optimizer-backend plbfgs` for P-LBFGS.
Use `--optimizer-backend ciah` and `--ciah-trace path.tsv` for CIAH. The retired
`neo`, `block_lbfgs`, and `--neo-trace` CLI spellings are not aliases; numeric
`ISCF=7/8/9` selections are unchanged.

`NSTATE=n` selects an equal-weight average over the consecutive lowest `n`
VB states. Omitting it is equivalent to `NSTATE=1`. The optimizer uses the
same normalized weight `1/n` for the energy, analytic gradient, and TNHVP
Hessian-vector product. See `testdata/vbscf/F2_SA2.xmi` for a two-state input.

Davidson runs retain only the requested roots, the structure-overlap diagonal,
and the matrix-free products needed for exact Coulson--Chirgwin weights. Their
reports omit the full structure Hamiltonian/overlap matrices and the Lowdin and
inverse weights that require global overlap-matrix functions. Dense reference
runs retain the complete matrices, spectrum, and all four weight definitions.

### Molecular DFT

Molecular RKS and collinear UKS support Libxc LDA/GGA and global-hybrid
functionals with exact AO Coulomb/exchange integrals, using the common SCF/DIIS core.
Select `DFT` rather than `VBSCF` in `$CTRL`:

```text
$CTRL
DFT BASIS=CC-PVDZ INT=LIBCINT
XC=LDA_X,LDA_C_VWN NRAD=80 NANG=590 SCF_TOL=1.0D-9
$END
$GEO
He 0.0 0.0 0.0
$END
```

The defaults are LDA exchange plus VWN correlation and an unpruned 80-by-590
atom-centered grid with Becke partitioning. `SCF_TOL` applies to the maximum
AO density change and commutator residual; `ITMAX` sets the iteration limit.
Angular grids of 302 or 590 points are supported. `XC` accepts `LDA`/`SVWN5`,
`PBE`, `BLYP`, `BP86`, `PW91`, `revPBE`, `PBEsol`, `PBE0`, `B3LYP`, and `B3LYP5`.
Explicit Libxc component names remain available, e.g. `XC=GGA_X_B88,GGA_C_LYP`.
Names are case-insensitive; output reports the resolved definition.

`PBE0` includes 25% exact exchange. `B3LYP` uses Libxc 402 (VWN-RPA), while
`B3LYP5` uses Libxc 475 (VWN5); both include 20% exact exchange. Exact exchange
is included consistently in energy, Fock and analytical density response.
The output distinguishes the semilocal XC energy from the total energy.
GGA density gradients and AO potentials use the same shared grid tiles as LDA;
all admitted components must provide analytical first and second derivatives.

`NMUL` fixes the alpha/beta populations: its default of 1 selects RKS, while
values greater than 1 select UKS. `KS=UKS` also allows an equal-population UKS
calculation; `KS=RKS` explicitly requires equal populations. For example, OH
uses `DFT NMUL=2 XC=PBE`. UKS fixes the spin projection, not a pure total-spin
eigenfunction; the output reports the determinant spin-square expectation.
Both spins share one Coulomb matrix and one coupled DIIS extrapolation.

See `testdata/dft/` for closed- and open-shell examples. Converged restricted
MO coefficients are written to `<input>.orb`; UKS writes `<input>.alpha.orb`
and `<input>.beta.orb`, all in fixed-point GUS-body format.

The initial guess diagonalizes the core Hamiltonian with the requested spin
occupations. Range-separated hybrids, meta-GGA, double hybrids, nonlocal VV10,
RI DFT, supplied guesses, nuclear derivatives and periodic DFT are not
implemented and are not silently substituted by another functional or HF.

## License

[MIT](LICENSE), with quadrature attribution in [NOTICE](NOTICE).
