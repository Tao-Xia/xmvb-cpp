# Repository Guidelines

## Project Structure & Module Organization
`src/vb/` contains the C++ valence-bond kernel, split into `matrices/`, `orbital/`, `model/`, and `scf/`. `src/runtime/` and `src/runtime_c/local_runtime/` load `.xmi` inputs, prepare molecules and basis data, and bridge the standalone binary to the legacy-style runtime. `src/tools/run_cpp_vbscf.cpp` builds the CLI entry point. Public and legacy headers live in `include/`. Versioned runtime assets are stored in `basis/` and `data/`, and sample inputs live in `src/test_molecule/`. Treat `build/`, `compile_commands.json`, and `.vscode/settings.json` as generated artifacts.

## Build, Test, and Development Commands
Create the recommended toolchain with `conda env create -f environment.yml` and `conda activate xmvb-cpp-dev`.

`./build.sh` configures a Ninja build in `build/`, copies runtime assets, and produces `build/src/xmvb-cpp.exe`.

`./build.sh build -DXMVB_CPP_BUILD_DEV_TARGETS=ON` enables optional diagnostic and test targets declared in `src/CMakeLists.txt`.

`cmake --build build --target run_cpp_vbscf -j8` rebuilds incrementally after edits.

Smoke test:
```bash
OMP_NUM_THREADS=1 build/src/xmvb-cpp.exe src/test_molecule/F2.xmi --optimizer-backend lbfgspp
```

If the conda environment is not active, forward dependency roots through `build.sh` with flags such as `-DLAPACK_ROOT_DIR=/path/to/env`.

## Coding Style & Naming Conventions
Match the existing build settings: C99 for C and C++17 for C++. Follow the surrounding style: 2-space indentation, same-line braces, `snake_case` for files and functions, and `PascalCase` for C++ types. Keep file-local helpers in anonymous namespaces and keep related headers and implementations in the same module subtree, for example `src/vb/scf/` plus matching `*.hpp`. Reusable declarations belong in headers, not scattered across `*.cpp` files. Follow [docs/HPC_CPP_STYLE.md](/pool1/home/xiatao/project/xmvb-cpp/docs/HPC_CPP_STYLE.md) for the project-specific C++ rules, especially the direct `Eigen::MatrixXd` / `Eigen::VectorXd` preference, the repository-wide column-major policy, the ban on using `std::vector<double>` as a fake dense matrix/vector type, and the ban on ad hoc matrix aliases such as local `using Matrix = ...`. Use [docs/COL_MAJOR_MIGRATION.md](/pool1/home/xiatao/project/xmvb-cpp/docs/COL_MAJOR_MIGRATION.md) as the checklist for retiring remaining row-major storage contracts. No repository `clang-format` file is checked in; use `.clangd` and preserve nearby formatting.

For any nontrivial implementation, comments are required in the `*.cpp` file as well as the header. Function implementations should explain the mathematical role of the routine, the meaning of important local variables, and the expected matrix/vector dimensions and conventions. Do not rely on `*.hpp` comments alone when the implementation contains domain-specific formulas or ambiguous naming.

## Testing Guidelines
CMake expects unit tests under `src/tests/unit/` and support fixtures under `src/tests/support/`, with names such as `cpp_vb_scf_optimizer_test.cpp` and `*_diagnostic.cpp`. Some exports, including this one, may omit those sources. At minimum, run the F2 smoke test after changing runtime, SCF, or orbital code. When test sources are present, enable dev targets, build the relevant executable, and run it from `build/src/`.

## Commit & Pull Request Guidelines
Git history is not bundled in this snapshot, so follow a simple convention: short imperative commit subjects with a subsystem prefix, for example `vb/scf: tighten gradient termination`. Keep each commit focused on one behavior change. Pull requests should summarize numerical or runtime impact, list the exact build and test commands run, and include representative output deltas for solver changes instead of screenshots.
