# Running and building xmvb-cpp

## Run the binary distribution

Prebuilt artifacts currently target Linux x86-64 CPUs. Platform and backend are
explicit in canonical filenames and GHCR tags. Install Apptainer on the host.
A release consists of a reusable runtime SIF and a versioned program tarball.
The SIF supplies the operating-system and numerical runtime libraries. The program supplies the executable, launcher,
basis files, examples, licenses and build metadata.

Download the runtime and program with their independent GHCR tags.

```bash
apptainer pull runtime-linux-x86_64-cpu-ubuntu24.04-r1.sif \
  oras://ghcr.io/tao-xia/xmvb-cpp-release:runtime-linux-x86_64-cpu-ubuntu24.04-r1
oras pull ghcr.io/tao-xia/xmvb-cpp-release:xmvb-cpp-0.1.4-linux-x86_64-cpu
sha256sum -c runtime-linux-x86_64-cpu-ubuntu24.04-r1.sif.sha256
sha256sum -c xmvb-cpp-0.1.4-linux-x86_64-cpu.tar.gz.sha256
```

Extract the program into its own directory and select the matching runtime.

```bash
mkdir -p programs/xmvb-cpp-0.1.4-linux-x86_64-cpu
tar -xzf xmvb-cpp-0.1.4-linux-x86_64-cpu.tar.gz \
  -C programs/xmvb-cpp-0.1.4-linux-x86_64-cpu
export XMVB_RUNTIME_IMAGE=/absolute/path/runtime-linux-x86_64-cpu-ubuntu24.04-r1.sif
export XMVB_PROGRAM_DIR=/absolute/path/programs/xmvb-cpp-0.1.4-linux-x86_64-cpu
OMP_NUM_THREADS=4 "$XMVB_PROGRAM_DIR/bin/xmvb" input.xmi
```

The launcher mounts the selected program read-only and checks the runtime ID.
The native program and launcher honor explicit thread settings. Otherwise,
OpenMP uses `SLURM_CPUS_PER_TASK` within the available CPU affinity, or the
available CPUs outside Slurm; BLAS defaults to one worker. The launcher preserves
Slurm CPU/memory allocation variables and user binding settings across the
container boundary. Relative input paths and outputs use the
current working directory. Apptainer also accepts additional data mounts through
its standard `APPTAINER_BIND` setting.

Verify the downloaded runtime and program checksum files once after installation.
The program's `share/build-info.json` records its source revision, compiler,
CMake options, dependency versions, binary hash, required runtime ID and SIF
SHA-256. `SHA256SUMS` verifies the extracted program files. When an update uses
the same runtime ID, download the new program and change `XMVB_PROGRAM_DIR`.
The 0.1.4 program extends physical CI residual stopping to selected structure
spaces and the auxiliary-frame orbital preconditioner to truncated OEO and HAO,
while preserving the selected degrees of freedom and localization constraints.
Repeated CI transport solves reuse bounded single-spin overlap panels and
certified spin-exchange symmetry. The final SCF criteria and CIAH strategy are
unchanged. The program reuses the existing runtime byte for byte; download only
the new program archive if that runtime is already installed.
Use the architecture-qualified download entries above.

The supplied F₂ example uses CIAH (`ISCF=9`). This first packaged backend is CPU.

## Build from source

Use C, C++23 and Fortran compilers with OpenMP, CMake 3.20 or newer, Ninja,
Eigen, OpenBLAS with LAPACKE headers and symbols, libcint and libxc. The repository
provides a Conda environment definition.

```bash
conda env create -f environment.yml
conda activate xmvb-cpp-dev
cmake --preset cpu-release
cmake --build --preset cpu-release --parallel 4
build/cpu-release/src/xmvb-cpp.exe --help
```

The preset selects Release, C++23, CPU kernels and the Eigen BLAS/LAPACKE backend.
For another compiler installation, set `CC`, `CXX` and `FC` before configuring
in a fresh build directory. Dependencies default to the active `CONDA_PREFIX`.
Independent installations can set `LAPACK_ROOT_DIR`, `EIGEN3_ROOT_DIR`,
`CINT_ROOT_DIR` and `LIBXC_ROOT_DIR` as CMake cache paths.

`environment.yml` resolves compatible current dependencies. To reproduce a
release's dependency selection, use the exact package versions and build strings
in `share/build-info.json`. CMake writes `compile_commands.json` in the build tree.

Diagnostic targets and tests use a separate build directory.

```bash
cmake --preset cpu-release -B build/cpu-tests \
  -DXMVB_BUILD_DEV_TARGETS=ON -DBUILD_TESTING=ON
cmake --build build/cpu-tests --parallel 4
ctest --test-dir build/cpu-tests --output-on-failure
```

For a CUDA source build, configure a separate directory with
`-DXMVB_ENABLE_CUDA=ON` and the target `CMAKE_CUDA_ARCHITECTURES`. CUDA Toolkit
12.5 or newer is required.

## Maintain the split release

Run builds on a compute allocation. Export a committed source revision and keep
the dependency-complete Conda prefix and its package cache on the build host.
The first build freezes that prefix into a builder SIF. The runtime contains
only the linked numerical/compiler libraries and their licenses over the pinned
Ubuntu base. Basis files belong to the program package.

```bash
revision=$(git rev-parse HEAD)
mkdir -p /release-work/source
git archive "$revision" | tar -x -C /release-work/source
export XMVB_RELEASE_BASE_SIF=/persistent/ubuntu24.04.sif
export XMVB_RELEASE_DEPENDENCY_PREFIX=/path/to/xmvb-cpp-dev
export XMVB_RELEASE_CACHE_DIR=/persistent/xmvb-release-cache
bash /release-work/source/packaging/apptainer/build.sh \
  /release-work/source "$revision" 0.1.4 /release-work/build \
  /release-artifacts/runtime-linux-x86_64-cpu-ubuntu24.04-r1.sif \
  /release-artifacts/xmvb-cpp-0.1.4-linux-x86_64-cpu.tar.gz
```

Acquire and hash the Ubuntu 24.04 base SIF before the first build. Builds require
Apptainer, rsync, flock and Python 3 on the host. Images are assembled from
the base and prepared dependency files through an unprivileged sandbox build.
On Hanhai25, `APPTAINER_IGNORE_PROOT=1` avoids the site's ptrace restriction when
assembling SIF files.

Builder caches are keyed by host architecture, base SIF hash, dependency content
and builder recipe. Ninja caches are keyed by builder hash, backend and build
recipe. Source content is synchronized into stable paths, leaving unchanged
objects in place. Concurrent builds using the same cache are serialized.
`XMVB_RELEASE_BUILDER_SIF` can select an existing pinned builder directly.

An existing runtime is reused only when its identity and linked libraries match
the staged program. A dependency change requires a new runtime filename and tag.
The build writes separate runtime and program checksums and preserves configure,
compile, staging and linkage logs. Qualify the program with its runtime before
publishing, then download both from GHCR and compare SHA-256.

### Optional local-RI contraction adapter

`XMVB_ENABLE_LIBRI=ON` builds the experimental `xmvb_local_ri` library. It is
separate from the molecular executable and is disabled by default. It currently
contracts real, supplied local pair fits and auxiliary Coulomb blocks for
finite/already folded Gamma data; it does not yet generate NAO integrals or
enable periodic VBSCF.

Provide external LibRI, LibComm and cereal headers, together with MPI and an
LP64 BLAS. The dependency roots contain `include/`:

```bash
cmake -S . -B build-lri -DXMVB_ENABLE_LIBRI=ON -DXMVB_BUILD_DEV_TARGETS=ON \
  -DLIBRI_ROOT=/path/to/LibRI -DLIBCOMM_ROOT=/path/to/LibComm \
  -DCEREAL_ROOT=/path/to/cereal
cmake --build build-lri --target test_local_ri_exchange
ctest --test-dir build-lri -R '^local_ri_exchange$' --output-on-failure
```

The adapter uses MPI_COMM_SELF, with OpenMP inside LibRI. Its caller owns MPI
initialization (`MPI_THREAD_FUNNELED`) and finalization. No MPI initialization
or new dependency is introduced into the existing molecular execution path.
The adapter accepts signed symmetric response densities without numerical
screening. Local panel dimensions are checked against LibRI's LP64 BLAS limit.
LibRI and LibComm remain external LGPL-3.0 dependencies; their sources and
licenses are available from the ABACUS modeling repositories.

### Optional numerical AO preparation

Add `-DXMVB_ENABLE_NAO=ON -DABACUS_SOURCE_ROOT=/path/to/abacus-checkout`
to the local-RI configuration above, then build `test_nao_prepare` and run
`ctest --test-dir build-lri -R '^(nao_prepare|local_ri_exchange)$'`.
Only the unmodified `source/source_base/cubic_spline.cpp` numerical kernel
is compiled from ABACUS (tested revision
`4e9f1636b7ca27d1dc634711915c18087654b911`, LGPL-3.0). No ABACUS executable,
SCF driver or global input state is used. Retain upstream license notices
when distributing this optional linked component.

The reference generator accepts compact numerical radial functions through
`integrals/nao/prepare.hpp`, reuses the molecular Becke/Lebedev grid, and
produces overlap, Hartree-unit kinetic, auxiliary Coulomb blocks and
pair-local fits for LibRI. It supports real s through g functions with the
ABACUS real-harmonic phase convention. Radial functions and the auxiliary
basis must be supplied explicitly; there is no `.orb` importer or automatic
auxiliary-basis construction yet. The caller supplies quadrature resolution,
panel width and the local metric rank cutoff. Panel width only partitions
work; it does not screen integrals or switch algorithms.

This is a finite-system integration reference, not a periodic calculation
mode. It does not supply pseudopotentials, ionic energies, periodic Coulomb
sums or a new `.xmi` keyword. Fast two-center tables and the periodic
Hamiltonian remain separate follow-up work.
