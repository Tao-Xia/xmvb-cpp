# VBSCF Codebase Reduction Plan

## Goal

Reduce the maintenance surface of the C++ VBSCF implementation while preserving
the production path:

- standard VBSCF energy/gradient evaluation
- L-BFGS orbital optimization
- nonredundant truncated-Newton HVP optimization
- libcint-backed standalone input/runtime path

Removed or retired side branches should stay deleted unless they become a
measured blocker for the production path.

## Current Cleanup State

- Removed the biorthogonal VBSCF implementation and its diagnostic tools.
- Removed the VB-PDFT/grid/libxc implementation and its diagnostic tools.
- Removed the `exact_separator` experimental library and its diagnostic tools.
- Kept `PhysicalOrbitalFrame` because it is part of the current VBSCF/TNHVP
  orbital state, but moved it into `src/vb/orbital/`.

## Main Maintenance Problems

1. `cpp_vb_scf_optimizer.cpp` mixes optimizer orchestration, line search,
   truncated-Newton policy, Krylov solve, exact-ctx HVP wiring, diagnostics,
   environment parsing, and accepted-iteration tracing.
2. `exact_orbital_second_order_operator.cpp` is a large subsystem hidden behind
   a single operator name. It combines accepted-point cache access, orbital
   chart handling, direct HVP action, response terms, and diagnostics.
3. `nonredundant_orbital_space.cpp` mixes geometric space construction with
   metric application, preconditioning, projection, expansion, and retraction.
4. Small helper routines are duplicated across SCF, orbital, and model files,
   especially sparse-orbital indexing, small symmetric matrix powers,
   upper-triangle indexing, vector norms, and environment parsing.
5. Experimental diagnostics are too easy to keep alive after their algorithmic
   branch is no longer part of production, which increases build and review
   scope without improving the main solver.

## Reduction Rules

1. Prefer deletion over parking unused code.
2. Keep production entry points small and explicit.
3. Do not let core numerical kernels read environment variables directly.
4. Keep strategy selection separate from numerical kernels.
5. Prefer direct local code over cross-module helper layering. Only extract a
   helper when it removes a stable abstraction boundary rather than hiding one
   short mathematical routine behind another file.
6. Each cleanup step must build `run_cpp_vbscf` before moving to the next step.

## Phase 1: Local Simplification

Reduce repeated low-level code only when it does not add another layer of
indirection. Prefer one owning implementation inside the module that already
defines the dataflow, or direct local code when the routine is short and the
formula is easier to review in place.

- unify obviously duplicated indexing/counting logic
- delete dead branches and unused conversions
- standardize dense matrix/vector ownership on Eigen types
- keep short SPD/eigensystem formulas next to the orbital/HVP code that uses them

Expected result: fewer dead paths and less boilerplate without introducing extra
navigation cost for reviewers.

### Phase 1 Progress

- Done: unified sparse-orbital coefficient counting into
  `stored_sparse_orbital_coefficient_count(...)` and
  `differentiable_sparse_orbital_parameter_count(...)` in
  `src/vb/orbital/orbital_preparation_input.hpp`.
- Done: removed the duplicated `get_sparse_coefficient_count(...)` helper from
  the main `src/vb` / `src/runtime` production path.
- Done: converted the RHF result/Fock path to Eigen-owned dense types:
  `CppRestrictedHartreeFockResult` now stores
  `Eigen::MatrixXd` / `Eigen::VectorXd`, and
  `CppClosedShellFockBuilder` now returns `Eigen::MatrixXd`.
- Done: removed the duplicated `get_orbital_basis_count(...)` path from
  runtime/orbital code and switched the remaining production callers to
  `stored_sparse_orbital_coefficient_count(...)` so sparse-support length has
  one implementation and one meaning.
- Done: simplified the loader/guess front-end data flow:
  `hf_overlap_matrix` is now optional and falls back to
  `active_orbital_overlap_matrix` when absent, so the pure C++ loader no
  longer stores the same AO overlap matrix twice. The RHF AUTO block guess now
  passes `Eigen::MatrixXd` Fock matrices directly into the block guess path
  instead of flattening them back into temporary column-major vectors.
- Done: removed `copy_matrix_to_column_major_buffer(...)` from the loader path.
  `OrbitalPreparationInput` now stores the AO overlap matrices as
  `Eigen::MatrixXd`, so the standalone input loader keeps the overlap metric in
  Eigen form from libcint materialization through guess build, orbital
  preparation, optimizer setup, and matrix-backprop entry points.
- Done: simplified the libcint one-electron shell-pair path so the core
  Hamiltonian builder reuses the output buffer for the kinetic block and keeps
  only one extra temporary buffer for the nuclear block. This removes one
  dense shell-block allocation and one full extra write pass from every
  core-H shell evaluation without changing the interface.
- Done: simplified the materialized libcint ERI path so it writes canonical
  `(i,j,k,l,value)` tuples directly into per-`shell_i` buffers and then merges
  them once in shell order. The old `packed_integrals` staging vector, global
  sort, and packed-index inverse decode step were removed, which cuts both
  memory traffic and startup-only bookkeeping in the AO integral front-end.
- Done: merged the AO ERI index-cache preparation passes in
  `materialized_ao_integral_input_builder.cpp`. The symmetry-shift cache and
  AO effective-one-electron linear-index cache are now built in one parallel
  pass over `g2eidx` instead of rescanning the same large index table twice.
- Done: stopped building the exact_ctx-only AO-H1E sparse graph on the normal
  `run_cpp_vbscf` `lbfgspp` / non-exact_ctx load path. The loader now keeps
  that expensive graph build behind an explicit load option, and the CLI only
  enables it for `nonredundant_truncated_newton + exact_ctx`, so routine VBSCF
  runs no longer pay to materialize a cache they never use.
- Done: migrated the dense AO core Hamiltonian contract from flattened
  `std::vector<double>` storage to `Eigen::MatrixXd` across the materialized
  integral provider, AO integral input, RHF/guess path, AO effective
  one-electron builder, orbital-gradient backprop path, and exact-ctx zero-core
  probe path. This removes another fake dense-matrix buffer from the hot
  runtime chain and avoids rebuilding temporary `Eigen::Map` views around owned
  `std::vector<double>` storage.
- Done: migrated `AoEffectiveOneElectronResult` itself to Eigen-owned dense
  matrices and updated the downstream AO-H1E consumers on the production path
  (`ActiveSpaceOneElectronBuilder`, `ActiveSpaceMatrixBackpropagator`,
  prepared-active-space assembly, orbital gradient/HVP setup, and nonredundant
  reduced-curvature setup) to consume `Eigen::MatrixXd` directly instead of
  treating AO `F11/G11` as flattened vectors.
- Done: migrated `ActiveSpaceOneElectronResult::h1e_act` to
  `Eigen::MatrixXd` across the production determinant / structure chain:
  `FullDeterminantPairEvaluator`, `SameSpinPairCache`, `FullDeterminantStructureHamiltonianOverlapBuilder`,
  `CppActiveSpaceGradientEvaluator`, `SameSpinMatrixBackward`, and the exact
  orbital second-order operator now pass `HHO` as an Eigen matrix instead of a
  fake dense `std::vector<double>`.
- Done: removed the extra flattened `HHO` compatibility copy from the normal
  optimizer result / accepted-trace path. `CppOrbitalGradientResult` and
  `CppVbScfAcceptedIterationSnapshot` now keep `active_one_electron_integrals`
  as `Eigen::MatrixXd`, and only the final binary-export path flattens through
  `.data()` when it actually writes files.
- Dropped: cross-module self-adjoint helper extraction. It reduces line count
  locally but makes formula review harder by adding another helper layer.
- Next: clear the remaining compatibility-only flattened `HHO` exports in
  diagnostics, then continue deleting dead paths and simplifying large files in
  place before doing any further extraction.

## Phase 2: Optimizer Split

Keep `CppVbScfOptimizer` as the orchestration layer only. Move the following
responsibilities into separate files:

- `vb/scf/line_search.*`: Armijo search and steepest-descent fallback.
- `vb/scf/truncated_newton_solver.*`: Krylov basis, trust-region subproblem,
  and reduced-step solve.
- `vb/scf/truncated_newton_policy.*`: cheap/full exact-ctx policy, retry,
  follow-up, stall handling, and logging of policy decisions.
- `vb/scf/reduced_hvp_operator.*`: finite-difference and exact-ctx reduced HVP
  adapters.
- `vb/scf/accepted_iteration_trace.*`: snapshot construction, trace retention,
  and accepted-iteration callbacks.

Expected result: optimizer review can focus on one concern at a time instead of
reviewing a multi-thousand-line file for every strategy change.

## Phase 3: Exact-CTX HVP Split

Break the exact HVP subsystem along data-flow boundaries:

- `exact_ctx_cache.*`: accepted-point cache ownership and validation.
- `exact_ctx_orbital_chart.*`: physical/internal inactive chart conversion.
- `exact_ctx_direct_action.*`: direct orbital HVP action.
- `exact_ctx_response.*`: structure/eigen response contributions.
- `exact_ctx_diagnostics.*`: timing, availability, and debug reporting.

Expected result: direct-action formulas, response formulas, and diagnostic code
can be reviewed independently.

## Phase 4: Nonredundant Orbital Space Split

Separate geometry from numerical application:

- `nonredundant_orbital_blocks.*`: block-local orbital data and support layout.
- `nonredundant_candidate_layout.*`: reduced coordinate indexing.
- `nonredundant_candidate_metric.*`: metric diagonal/application/solve.
- `nonredundant_projection.*`: full-gradient to reduced-gradient projection.
- `nonredundant_retraction.*`: reduced step expansion and physical retraction.

Expected result: orbital chart changes do not require reviewing preconditioner
and solver code at the same time.

## Phase 5: Diagnostic Boundary

Only keep diagnostic tools that validate production behavior:

- gradient checks
- exact-ctx HVP checks
- benchmark tools for current optimizer backends
- input/runtime inspection tools that support standalone `.xmi` runs

Delete tools tied to retired branches immediately when the branch is removed.

## Verification Checklist

After each cleanup patch:

1. Run `rg` for deleted module names and retired keywords.
2. Build `cmake --build build --target run_cpp_vbscf -j8`.
3. When SCF/orbital logic changes, run at least the F2 smoke test.
4. For TNHVP changes, run the exact-ctx HVP diagnostic or sbatch benchmark used
   for current development.

## Near-Term Order

1. Finish retired-library deletion and keep CMake clean.
2. Consolidate duplicated helper functions.
3. Split `cpp_vb_scf_optimizer.cpp` without behavior changes.
4. Split exact-ctx HVP internals without behavior changes.
5. Split nonredundant orbital-space implementation without behavior changes.
