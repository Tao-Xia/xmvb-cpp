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
oras pull ghcr.io/tao-xia/xmvb-cpp-release:xmvb-cpp-0.1.1-linux-x86_64-cpu
sha256sum -c runtime-linux-x86_64-cpu-ubuntu24.04-r1.sif.sha256
sha256sum -c xmvb-cpp-0.1.1-linux-x86_64-cpu.tar.gz.sha256
```

Extract the program into its own directory and select the matching runtime.

```bash
mkdir -p programs/xmvb-cpp-0.1.1-linux-x86_64-cpu
tar -xzf xmvb-cpp-0.1.1-linux-x86_64-cpu.tar.gz \
  -C programs/xmvb-cpp-0.1.1-linux-x86_64-cpu
export XMVB_RUNTIME_IMAGE=/absolute/path/runtime-linux-x86_64-cpu-ubuntu24.04-r1.sif
export XMVB_PROGRAM_DIR=/absolute/path/programs/xmvb-cpp-0.1.1-linux-x86_64-cpu
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
Architecture-qualified names are distribution labels: the renamed 0.1.1 archive
and runtime retain their original SHA-256 values and runtime identity. Original
build metadata retains its build-time filenames. Unqualified GHCR tags have been
removed; use the architecture-qualified download entries above.

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
  /release-work/source "$revision" 0.1.1 /release-work/build \
  /release-artifacts/runtime-linux-x86_64-cpu-ubuntu24.04-r1.sif \
  /release-artifacts/xmvb-cpp-0.1.1-linux-x86_64-cpu.tar.gz
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
