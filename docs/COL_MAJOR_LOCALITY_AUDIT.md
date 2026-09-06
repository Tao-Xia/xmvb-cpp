# Column-Major Locality Audit

This note records the `src/vb` and related RI boundary sites where
column-major `Eigen::MatrixXd` is still consumed or produced in a
row-oriented way, or where large row-buffer compatibility copies are still on
the hot path.

The goal is not to list every matrix loop. The goal is to identify the places
most likely to matter for wall time, especially on exact two-electron HVP /
TNHVP workloads.

Static inspection date: `2026-04-26`

## Reading Guide

- `P0`: highest-priority hot-path locality issue
- `P1`: real locality issue, but likely smaller or less frequent than `P0`
- `P2`: setup-time or compatibility-boundary issue; worth cleaning, but not the
  first place to spend effort

## Confirmed High-Priority Findings

### P0. Exact 2e AO-pair kernel over `Eigen::MatrixXd` scans logical rows of column-major matrices

References:

- `src/vb/orbital/active_space_two_electron_utils.cpp:4103`
- `src/vb/orbital/active_space_two_electron_utils.cpp:4128`
- `src/vb/orbital/active_space_two_electron_utils.cpp:4160`
- `src/vb/orbital/active_space_two_electron_utils.cpp:4195`

What happens:

- The exact AO-pair kernel materializes `pair_gradients` as
  `n_ao_pairs x n_active_pairs` `Eigen::MatrixXd`.
- The hot loops fix one AO-pair row and then sweep `active_pair_index` in the
  inner loop.
- Both the destination `pair_gradients(row, col)` and the source
  `transformed_pair_coefficients(row, col)` are column-major matrices, so this
  inner loop is strided instead of unit-stride.

Why this hurts:

- Inner-loop memory step is `outerStride()` rather than `1`.
- The kernel is on the exact 2e HVP path and can be hit many times inside one
  Krylov solve.
- The multi-thread fallback amplifies the problem by allocating full
  `Eigen::MatrixXd` partial buffers per thread and then reducing them.

Best fix direction:

- Reorder the contraction to work column-by-column or in active-pair blocks.
- Keep the AO-pair graph traversal, but accumulate a block of active-pair
  columns at a time.
- If a row-oriented contraction is required, keep that buffer row-oriented all
  the way through the kernel instead of forcing it through `MatrixXd`.

### P0. Exact 2e dense-active backprop reads `pair_gradients` row-wise and updates `dense_active_gradients` row-wise

References:

- `src/vb/orbital/active_space_two_electron_utils.cpp:4988`
- `src/vb/orbital/active_space_two_electron_utils.cpp:5020`
- `src/vb/orbital/active_space_two_electron_utils.cpp:5062`

What happens:

- The accepted-point HVP backprop path consumes `pair_gradients` as
  `Eigen::MatrixXd`.
- It again fixes one AO-pair row and sweeps active-pair columns.
- It also updates `dense_active_gradients(basis, active)` with active orbital as
  the inner loop while `dense_active_gradients` is column-major.

Why this hurts:

- The code pays the row-stride penalty once when building `pair_gradients` and
  then again when backpropagating it.
- This is one of the reasons the exact 2e materialized path is sensitive to
  storage order.

Best fix direction:

- Fuse or block the backprop so the active dimension is handled in column-major
  order.
- Prefer the cached-row / basis-owned formulations once their locality is fixed.

### P0. `delta_active_pair_matrix` assembly repeatedly extracts rows from column-major matrices with `.row(...).transpose()`

References:

- `src/vb/orbital/active_space_two_electron_utils.cpp:5873`
- `src/vb/orbital/active_space_two_electron_utils.cpp:5992`

What happens:

- Inside the AO-pair loop, the code builds four temporary `VectorXd`s from
  `.row(...).transpose()`.
- Each row access is strided because the source matrices are column-major.
- The transpose creates temporary contiguous vectors, so the loop pays both the
  strided load and the copy.

Why this hurts:

- This sits directly on the exact 2e directional-difference path.
- The loop is mathematically a sum of outer products and should be expressible
  as two GEMMs:

  - `DeltaM += DeltaC^T * P`
  - `DeltaM += C^T * DeltaP`

Best fix direction:

- Replace the AO-pair scalar loop with matrix products.
- Eliminate all `.row(...).transpose()` temporaries from this path.

## Other Real Locality Problems

### P1. Exact and RI active-pair coefficient builders write `MatrixXd` by logical row

References:

- `src/vb/orbital/active_space_two_electron_utils.cpp:2992`
- `src/vb/orbital/active_space_two_electron_utils.cpp:3057`
- `src/vb/orbital/active_space_two_electron_utils.cpp:3131`
- `src/vb/orbital/ri_active_space_two_electron_builder.cpp:76`

What happens:

- The builders materialize `n_ao_pairs x n_active_pairs` coefficient matrices as
  `Eigen::MatrixXd`.
- The loops fix one AO-pair row and sweep active-pair columns when writing the
  output.

Why this hurts:

- The destination writes are strided in column-major storage.
- The `build_mixed_ao_pair_to_active_pair_coefficients_from_cache(...)` variant
  is on the exact HVP path, so this is not just a one-time setup cost.

Best fix direction:

- Build by active-pair column, not by AO-pair row.
- Or keep a row-oriented compatibility buffer for the row-streaming kernels and
  only convert at the narrowest boundary.

### P1. Exact 2e fused cached-row path already has a column-major-aware inner accessor, but the default path still falls back to the worse materialized layout

References:

- `src/vb/orbital/active_space_two_electron_utils.cpp:128`
- `src/vb/orbital/active_space_two_electron_utils.cpp:555`
- `src/vb/orbital/active_space_two_electron_utils.cpp:623`
- `src/vb/orbital/active_space_two_electron_utils.cpp:2719`

What happens:

- The fused cached-row path includes `pair_source_column_stride` accessors that
  explicitly handle column-major pair-source matrices.
- That path is currently opt-in because the current direct formulation regressed
  badly on restored `241 exact_ctx` runs.

Why this matters:

- The code already contains the right idea: do not treat logical rows as
  contiguous.
- The practical issue is not conceptual correctness but poor locality in the
  current fused contraction schedule.

Best fix direction:

- Rework the fused kernel as a blocked contraction instead of abandoning it.
- This remains the best long-term path because it avoids materializing the full
  `pair_gradients` matrix.

### P1. Outer-response selected-state projection extracts eigensystem rows from column-major storage inside structure loops

References:

- `src/vb/scf/exact_orbital_second_order_operator.cpp:7178`
- `src/vb/scf/exact_orbital_second_order_operator.cpp:7582`
- `src/vb/scf/exact_orbital_second_order_operator.cpp:7587`
- `src/vb/scf/exact_orbital_second_order_operator.cpp:7620`

What happens:

- The eigensystem is mapped as column-major `Eigen::MatrixXd`.
- The outer-response projection loop uses
  `eigenvector_matrix.row(right_structure)` and
  `eigenvector_matrix.row(right_structure).transpose()` inside the structure
  sweep.

Why this hurts:

- Scalar indexing through a logical row is strided.
- The transposed row path may materialize a temporary vector per structure.
- This is not as dominant as exact 2e AO-pair contraction, but it is on a
  repeated response/HVP path.

Best fix direction:

- Cache selected columns once and read columns, not rows.
- Where a row is mathematically needed, consider block projection formulas that
  use `topRows(...).transpose()` and selected-column caches without per-row
  extraction.

### P1. RI provider shell-block scatter writes packed AO-pair factors by logical row of a column-major matrix

References:

- `src/runtime/libcint_ri_integral_provider.cpp:147`
- `src/runtime/libcint_ri_integral_provider.cpp:174`

What happens:

- `raw_ao_pair_factors(auxiliary_index, packed_pair_index)` is filled with
  fixed `auxiliary_index` and varying packed AO-pair index in the inner loops.
- That is row-wise write traffic into a column-major matrix.

Why this hurts:

- RI factor construction can be large.
- This is setup rather than per-HVP work, so it is not `P0`, but the write
  pattern is still poor.

Best fix direction:

- Scatter into a staging layout that matches the shell block traversal and then
  transpose/pack once.
- Or restructure the shell scatter so the innermost loop writes consecutive row
  indices inside one column-major column.

### P1. Optional RI dense lower-factor cache reads packed RI factor rows from a column-major matrix with non-unit stride

References:

- `src/runtime/libcint_ri_integral_provider.cpp:81`
- `src/runtime/libcint_ri_integral_provider.cpp:110`

What happens:

- `build_dense_lower_ao_factor_matrices(...)` walks `packed_factor_rows` with
  fixed auxiliary row and increasing packed pair index.
- The source comment correctly notes that packed-factor rows are not contiguous.

Why this hurts:

- This is an explicit row-wise read over a column-major source.
- The path is gated by `XMVB_CPP_ENABLE_RI_AO_FACTOR_MATRIX_CACHE`, so impact
  depends on runtime settings.

Best fix direction:

- If this cache becomes important, build it by source columns or pretranspose
  once.

## Compatibility-Boundary Costs

### P2. Exact / RI two-electron builder and backpropagator still convert large row buffers to and from `MatrixXd`

References:

- `src/vb/matrices/eigen_matrix_storage_utils.hpp:17`
- `src/vb/matrices/eigen_matrix_storage_utils.hpp:45`
- `src/vb/orbital/active_space_two_electron_builder.cpp:929`
- `src/vb/orbital/active_space_two_electron_builder.cpp:934`
- `src/vb/orbital/active_space_two_electron_builder.cpp:982`
- `src/vb/orbital/active_space_two_electron_builder.cpp:987`
- `src/vb/orbital/active_space_two_electron_backpropagator.cpp:531`
- `src/vb/orbital/active_space_two_electron_backpropagator.cpp:587`
- `src/vb/orbital/active_space_two_electron_backpropagator.cpp:684`
- `src/vb/orbital/active_space_two_electron_backpropagator.cpp:692`

What happens:

- Large `n_ao_pairs x n_active_pairs` tables are still copied between legacy
  row buffers and `Eigen::MatrixXd`.

Why this hurts:

- These are full-memory reorders, not views.
- The copy cost is not as bad as the `P0` hot loops, but it keeps the codebase
  paying layout-conversion tax at module boundaries.

Best fix direction:

- Migrate producers and consumers together so the compatibility buffer disappears
  rather than moving the copy around.

### P2. Some projector/gauge setup code copies logical rows of column-major matrices

References:

- `src/vb/orbital/nonredundant_orbital_space.cpp:1077`
- `src/vb/orbital/nonredundant_orbital_space.cpp:1109`
- `src/vb/orbital/support_aware_mo_gauge_fix.cpp:389`

What happens:

- These sites assign `matrix.row(...) = other.row(...)` repeatedly.

Why this is lower priority:

- These are setup / chart / masking operations, not the dominant double-electron
  contraction path.
- The access pattern is still row-oriented on column-major matrices, so it is
  worth cleaning once the heavy kernels are fixed.

Best fix direction:

- Replace repeated row copies with indexed gathers by column, or use a
  temporary row-oriented staging buffer if the operation is inherently row-based.

## Places That Look Fine For Now

- `src/vb/orbital/active_space_two_electron_utils.cpp:5709` and nearby RI
  projection code read `ri_active_pair_factors.col(...)`, which matches
  column-major storage well.
- `src/vb/scf/same_spin_matrix_backward.cpp:141` and
  `src/vb/scf/same_spin_matrix_backward.cpp:173` call BLAS with
  `CblasColMajor`; these are not row/column-major mismatch bugs.
- Small helper loops over tiny determinant submatrices, such as
  `spin_pair_utils.cpp`, are not the first targets unless profiling proves
  otherwise.

## Recommended Fix Order

1. `active_space_two_electron_utils.cpp:4103` exact `apply_exact_ao_pair_kernel(Eigen)`
2. `active_space_two_electron_utils.cpp:4988` exact dense-active backprop
3. `active_space_two_electron_utils.cpp:5873` and `:5992` `delta_active_pair_matrix`
   assembly
4. `active_space_two_electron_utils.cpp:3131` mixed pair-coefficient build on the
   HVP path
5. builder/backprop compatibility copies across `ActiveSpaceTwoElectronResult`
   boundaries
6. RI provider scatter/cache locality issues
7. outer-response eigensystem row extraction
8. setup-time row-copy cleanup in projector/gauge code

## Working Rule For Future Edits

For any dense `Eigen::MatrixXd` in this repository:

- prefer column-wise traversal and column-block GEMM/GEMV formulations
- avoid treating logical rows as contiguous
- avoid `.row(...).transpose()` in hot loops
- isolate unavoidable row-oriented compatibility buffers to narrow shims

