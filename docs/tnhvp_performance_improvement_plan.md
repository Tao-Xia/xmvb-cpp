# TNHVP Performance Improvement Plan

Status: active
Last updated: 2026-04-23

This document tracks concrete performance work for the current
`nonredundant_truncated_newton` + `exact_ctx` HVP path. It is intended to be
updated whenever a performance change is implemented or benchmarked.

Related background:

- [tnhvp_future_optimization_directions.md](./tnhvp_future_optimization_directions.md)
- [tnhvp_structural_optimization.md](./tnhvp_structural_optimization.md)
- [exact_ctx_post_tile_contraction_optimization_plan.md](./exact_ctx_post_tile_contraction_optimization_plan.md)
- [outer_response_linear_response_predecomposition.md](./outer_response_linear_response_predecomposition.md)

## Current Baseline

Command:

```bash
env OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  build/src/benchmark_exact_ctx_hvp test/241_VBSCF.xmi \
  --repeats 2 --warmup 1 --ao-integral-source libcint_cpp
```

Measured on 2026-04-23:

| component | avg wall time |
| --- | ---: |
| full cached HVP | 0.728 s |
| core-only cached HVP | 0.160 s |
| outer-only cached HVP | 0.485 s |
| full HVP outer-response part | 0.470 s |
| outer active-space integrals | 0.090 s |
| outer structure matrices | 0.292 s |
| outer active gradient | 0.009 s |
| outer orbital pullback | 0.078 s |

The current HVP kernel is no longer the only dominant issue. On `241_VBSCF`,
production optimizer logs show that accepted TN steps still cost about
`3-5 s` even when most inner HVP applies are cheap core-only probes. The
remaining wall time is now split between:

- expensive accepted-point trial objective evaluations;
- repeated trust-region trial and retry control flow;
- avoidable optimizer-side object copies and trial-state packing work.

For this case, the full HVP is still roughly `4.6x` the core-only HVP, but
real wall time is no longer explained by HVP cost alone.

## Optimization Roadmap

| ID | Priority | Status | Area | Target |
| --- | --- | --- | --- | --- |
| P1 | High | in progress | outer-response structure matrices | Reduce per-HVP structure-side contraction time |
| P2 | High | not started | TN hybrid/full retry policy | Avoid unnecessary full or outer-only HVP probes |
| P3 | High | not started | selected-state directional matrices | Cache accepted-point support/mapping and update values only |
| P4 | Medium | not started | outer-response orbital pullback | Workspace temporary matrices and reduce format conversions |
| P5 | Medium | in progress | exact active 2e HVP | Remove or fuse large AO-pair x active-pair intermediate layers |
| P6 | Medium | not started | Krylov storage | Avoid vector-to-matrix rebuilds and large subspace copies |
| P7 | High | in progress | trial step evaluation copies | Remove large `OrbitalObjective` and accepted-context copies from TN trial/fallback paths |
| P8 | Long term | design only | chart reformulation | Orthogonal inactive-frame or reduced-chart direct pullback |

## P1. Outer-Response Structure Matrices

### Problem

The largest measured full-HVP substage is
`outer_response_structure_matrices`. The hot path is
`build_selected_state_projected_directional_structure_matrices(...)` in
`src/vb/scf/exact_orbital_second_order_operator.cpp`. It repeatedly gathers
same-spin and opposite-spin local blocks, contracts structure-pair kernels, and
accumulates selected-state transformed columns.

The current implementation already avoids forming full directional structure
matrices, but it still pays many per-structure-pair gather and small dense
matrix costs.

### Plan

1. Add finer diagnostics for the structure-matrix stage: gather time, local
   kernel contraction time, selected-state projection time, and thread reduction
   time.
2. Reuse per-thread local block workspaces more aggressively and avoid repeated
   dynamic resizing in the inner pair loop.
3. Cache accepted-point support traversal metadata so each HVP only injects the
   directional values.
4. Revisit tile size and tile-cache defaults after measuring the current
   per-substage breakdown.
5. Consider a specialized single-selected-state path that removes the generic
   selected-state loop and matrix-column abstractions.

### Implemented So Far

1. Added `XMVB_CPP_EXACT_CTX_DIRECTIONAL_STRUCTURE_THREAD_DIVISOR` so this
   stage can be tuned independently in cluster benchmarks.
2. Changed the default directional-structure thread divisor from `96` to `48`.
3. Materialized the directional opposite-spin dense projected pair values
   (`G * delta U + delta G * U`) once per cached determinant-pair tile entry,
   instead of recomputing the same packed-pair projection repeatedly inside
   each structure-pair channel contraction.

The current heuristic was under-threading `241_VBSCF`-class cases. With the old
default, `n_structures = 175` gave only one thread to the structure-matrix
stage. The tuned divisor allows two or more threads earlier, while remaining
far below full oversubscription.

The new opposite-spin cache is the main implemented win. The structure stage
diagnostics showed that the dominant cost was the local pair-kernel work, not
the final projection or reduction. The previous code already cached the sparse
directional first-order projection, but it still rebuilt the dense directional
beta channel projection on every touched packed-pair channel. Moving that dense
`G * delta U + delta G * U` contraction to determinant-pair tile build time
lets the hot inner loop read a cached dense image instead of replaying the same
packed-pair kernel application.

### Measured Progress

All measurements below used `sbatch` with `OMP_NUM_THREADS=32`,
`OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`.

#### `test/241_VBSCF.xmi`

Correctness check:

- `test/chkctx241-opt3.1942237.out`
- `full_max_abs_diff = 4.75915862275e-09`
- `fixed_max_abs_diff = 3.83349146871e-09`
- `outer_max_abs_diff = 2.82713052968e-09`

Benchmark comparison:

| metric | before | after |
| --- | ---: | ---: |
| full cached HVP | 0.34044 s | 0.31766 s |
| outer-only cached HVP | 0.28611 s | 0.21812 s |
| full outer structure matrices | 0.08164 s | 0.04926 s |
| outer-only structure matrices | 0.07928 s | 0.04548 s |

Detailed structure-stage logging:

- previous pair-kernel time was about `0.198-0.207 s` per logged apply
- new pair-kernel time is about `0.064-0.066 s` on the fast applies

#### `test/10698_VBSCF.xmi`

Benchmark comparison:

| metric | before | after |
| --- | ---: | ---: |
| full cached HVP | 1.51093 s | 1.39842 s |
| outer-only cached HVP | 1.01414 s | 0.94473 s |
| full outer structure matrices | 0.08433 s | 0.03979 s |
| outer-only structure matrices | 0.08634 s | 0.04099 s |

For `10698_VBSCF`, the structure stage is now much smaller than
`outer_response_active_space_integrals` and no longer looks like the leading
outer-response hotspot.

### Current P1 Conclusion

The original `P1` target is now partially completed and validated. The
structure-matrix stage still matters, but it is no longer the dominant
outer-response bottleneck on the tested decks. The next highest-value work
should shift toward:

1. `outer_response_active_space_integrals`, especially on `10698_VBSCF`.
2. `outer_response_orbital_pullback`, which remains a visible substage after
   the structure-side reduction.
3. Accepted-point directional-metadata reuse for selected-state matrices
   (`P3`), if repeated HVP allocation pressure is still visible.

### Acceptance Criteria

- `outer_response_structure_matrices` time decreases on `test/241_VBSCF.xmi`.
- The result of `check_exact_ctx_hvp` remains within the current finite
  difference tolerance.
- No increase in nondeterminism across OpenMP thread counts.

## P2. TN Hybrid And Full-Retry Policy

### Problem

The optimizer can spend extra HVPs at the same accepted point:

- cheap core solve;
- optional full-model probe through outer-only correction;
- optional full-model refinement;
- optional full retry after cheap trial rejection;
- fallback predicted-decrease HVP.

Each extra full or outer-only HVP is expensive. On the current baseline,
outer-only is about 0.485 s, while core-only is about 0.160 s.

### Plan

1. Log per-accepted-iteration counts for core HVP, outer-only HVP, full HVP,
   full probes, full refinements, full retries, and fallback model evaluations.
2. Tighten full-probe admission so cheap steps are only probed when there is
   strong evidence of model mismatch or stagnation.
3. Cache and reuse `H s` values consistently when a step moves from cheap
   solve to full probe or retry.
4. Evaluate whether full retry should use a smaller capped CG budget than the
   cheap solve.
5. Prefer trust-radius shrink or descent fallback over full retry when the cheap
   step already failed for negative curvature or boundary reasons.

### Acceptance Criteria

- Fewer full/outer HVP calls per accepted step on the same input deck.
- Similar or lower wall time to reach the same gradient tolerance.
- No regression in convergence robustness on `F2`, `241_VBSCF`, and at least
  one larger sparse-orbital deck.

## P3. Selected-State Directional Matrices

### Problem

Every outer-response HVP rebuilds selected-state determinant matrices from
directional selected eigenvector columns. This allocates and fills:

- determinant coefficient vectors;
- dense unique-spin coefficient matrices;
- local support matrices;
- support index maps.

The sparsity pattern and determinant-to-unique-spin mapping are accepted-point
metadata and should not need full reconstruction on every HVP.

### Plan

1. Split selected-state determinant matrices into immutable accepted-point
   layout and mutable value buffers.
2. Cache determinant-to-structure scatter lists and touched unique-spin pair
   lists.
3. Add an update API that writes directional coefficients into existing buffers.
4. Keep the current builder as a validation/reference path behind a diagnostic
   flag.

### Acceptance Criteria

- Fewer heap allocations in outer-response active-gradient stage.
- `outer_response_active_gradient` and total outer-response time do not regress.
- Diagnostic comparison against the old builder is bitwise or near-bitwise
  identical for representative decks.

## P4. Outer-Response Orbital Pullback Workspace

### Problem

The current pullback path symmetrizes active gradients, copies Eigen matrices
into `std::vector<double>`, runs active-space matrix and two-electron
backpropagators, then backpropagates to full orbital values before gathering
into packed/reduced coordinates.

This is correct but allocation- and format-conversion-heavy.

### Plan

1. Replace per-call symmetric matrix vector construction with reusable
   workspace buffers.
2. Prefer `Eigen::Map` views over `std::vector<double>` copies where interfaces
   allow it.
3. Combine matrix and two-electron active auxiliary gradients directly into a
   caller-owned workspace.
4. Investigate a direct packed-chart pullback for sparse orbital charts.

### Acceptance Criteria

- Lower `outer_response_orbital_pullback` time.
- No change to projected HVP values.
- Reduced peak allocation count in profiling or memory diagnostics.

### Progress

Implemented a low-risk workspace change for the active SSO/HHO symmetrization
inside the outer-response orbital pullback. The code now writes the symmetric
active matrices directly into reusable operator-owned `std::vector<double>`
buffers instead of forming two temporary `Eigen::MatrixXd` objects and then
copying them into vectors.

The former internal inactive-chart experiment has since been removed: its HVP
did not represent the same physical coefficient chart as the optimizer and
failed the FeCl2 finite-difference check. Exact-CTX now has only the physical
strict-sparse chart path.

Validation:

- `test/chkctx241-pull.1942249.out`
- `full_max_abs_diff = 4.75915862275e-09`
- `outer_max_abs_diff = 2.82713052968e-09`

Performance note:

- The small workspace cleanup did not reduce
  `outer_response_orbital_pullback` in the `241_VBSCF` benchmark; that substage
  remained in the `0.05 s` range and appears dominated by the actual matrix /
  two-electron / AO backpropagation work rather than the removed temporary
  symmetrization copies.
- Keep this cleanup for lower allocation pressure, but do not count it as a
  completed performance win.

## P5. Exact Active 2e HVP Intermediate Reduction

### Problem

The exact active 2e path still copies directions into legacy row buffers and can
materialize large AO-pair by active-pair intermediate buffers. The structural
optimization target is to avoid fully materializing the mixed pair coefficient
layer when it can be generated on demand during AO-pair kernel traversal.

### Plan

1. Add timing around mixed-pair construction, pair-gradient transform, AO-pair
   kernel application, and final backprop.
2. Prototype an on-the-fly mixed-pair row generation path for small active
   spaces.
3. Fuse AO-pair kernel application and cached backprop where the graph layout
   permits deterministic accumulation.
4. Remove legacy row-buffer copies from the hot path when Eigen column-major
   storage can be consumed directly.

### Acceptance Criteria

- Lower `active_2e` time in both core-only and full HVP benchmarks.
- No increase in memory footprint.
- HVP checks pass with exact integral source.
- Benchmarks must include more than one deck family. A change should not be
  justified only by `241_VBSCF`; at minimum compare one compact selected-state
  deck, one larger exact deck, and one sparse/localized-orbital deck when those
  inputs are available.

### Experiment Log

Implemented a small-active-space specialization for the multithreaded AO-pair
graph matvec used by the default materialized exact-2e HVP path. The change
does not alter the strategy or formula; it dispatches triangular active-pair
counts `1, 3, 6, ..., 55` to fixed-size inner loops while preserving the same
row-parallel graph traversal. This targets active spaces with up to ten active
orbitals and falls back to the dynamic loop otherwise.

Cross-deck Slurm benchmarks with `OMP_NUM_THREADS=32`,
`OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`:

| deck | metric | before | after | note |
| --- | --- | ---: | ---: | --- |
| `241_VBSCF.xmi` | core-only cached HVP | 0.09974 s | 0.08817 s | clear cheap-core win |
| `241_VBSCF.xmi` | core-only active 2e | 0.06081 s | 0.04985 s | target stage improved |
| `241_VBSCF.xmi` | full cached HVP | 0.23894 s | 0.23592 s | small net win |
| `10698_VBSCF.xmi` | core-only cached HVP | 0.82584 s | 0.79079 s | modest cheap-core win |
| `10698_VBSCF.xmi` | core-only active 2e | 0.41611 s | 0.47957 s | noisy/regressed stage; total improved through lower H1E time |
| `10698_VBSCF.xmi` | full cached HVP | 1.39842 s | 1.41866 s | effectively neutral/slightly worse |
| `MnF2.xmi` | core-only cached HVP | pending comparable baseline | 0.07266 s | needs same-binary before/after comparison |

Correctness check:

- `test/chk-241-gpairopt.1942300.out`
- `full_max_abs_diff = 7.90796564343e-09`
- `fixed_max_abs_diff = 8.07047603746e-09`
- `outer_max_abs_diff = 6.71861998848e-09`

Current conclusion: keep this as a candidate only with broader validation. It
is a general small-active-space graph-kernel specialization, not a
`241_VBSCF`-specific branch, but `10698_VBSCF` shows that stage-level timings can
shift between active-2e and AO-H1E/outer-response noise. Do not stack further
optimizations on this assumption until a controlled before/after comparison is
available on at least `10698_VBSCF` and one localized-orbital deck.

Tried a more structural graph-layout rewrite for the default materialized HVP:
keep `transformed_pair_coefficients`, but stream each AO-pair graph row directly
into the dense active-gradient backpropagation and avoid materializing
`pair_gradients`. This was intended to remove one
`n_ao_pairs x n_active_pairs` intermediate while preserving row-graph locality.

The experiment was much slower and was reverted:

| deck | metric | stable path | experiment |
| --- | --- | ---: | ---: |
| `241_VBSCF.xmi` | core-only cached HVP | 0.08817 s | 0.57404 s |
| `241_VBSCF.xmi` | core-only active 2e | 0.04985 s | 0.52982 s |
| `241_VBSCF.xmi` | full cached HVP | 0.23592 s | 0.74697 s |

Conclusion: removing the pair-gradient intermediate by pushing packed
active-pair backprop into every AO graph edge is the wrong locality tradeoff.
The materialized `K * transformed` row kernel is much cheaper than repeating
the packed active-gradient backprop at graph-edge granularity. A future rewrite
should instead preserve row-level accumulation and look for a blocked
row-buffer strategy, not edge-level direct backprop.

Tried that row-buffer strategy next: keep the row-graph traversal, accumulate
one temporary packed `pair_gradient_row` per AO-pair row, then immediately
backpropagate that row to dense active coefficients instead of storing the full
`pair_gradients` matrix. This also did not pay off on `241_VBSCF`, and the
code was reverted:

| deck | metric | stable path | row-buffer trial |
| --- | --- | ---: | ---: |
| `241_VBSCF.xmi` | full cached HVP | 0.26148 s | 0.26026 s |
| `241_VBSCF.xmi` | core-only cached HVP | 0.09974 s | 0.11645 s |
| `241_VBSCF.xmi` | core-only active 2e | 0.04977 s | 0.05836 s |

Interpretation: removing the large `pair_gradients` buffer alone is not enough.
The current row-materialized `K * transformed` kernel is still cheaper than
rebuilding and immediately consuming one packed active-pair row per AO row.
Future exact-2e work should therefore avoid both:

- edge-level direct backprop
- naive AO-row buffered backprop

and look instead for a more global reduction in AO-kernel traffic or pair-space
transforms.

Tried routing the outer-response `delta GGO` directional derivative through the
accepted exact-2e HVP cache. This reused accepted AO-pair coefficient/product
buffers but was slower on `test/10698_VBSCF.xmi`:

| metric | before | experiment |
| --- | ---: | ---: |
| full cached HVP | 1.39842 s | 1.49365 s |
| full outer active-space integrals | 0.41026 s | 0.50583 s |
| outer-only cached HVP | 0.94473 s | 0.97854 s |
| outer-only active-space integrals | 0.39470 s | 0.51549 s |

The experiment was reverted. The likely reason is that the accepted exact-2e
HVP cache layout is tuned for adjoint-HVP application, while the current
outer-response directional `delta GGO` builder can reuse the accepted forward
`dense_ao_pair_products` from `ActiveSpaceTwoElectronResult` without paying the
same cache traversal cost. Future work should optimize the current directional
builder directly rather than blindly reusing the adjoint cache.

## P6. Krylov Storage And Subspace Reuse

### Problem

The truncated-Newton solver stores Krylov vectors as `std::vector<Eigen::VectorXd>`
and then rebuilds dense basis matrices during finalization or rescue. Returning
`TruncatedNewtonStepResult` can also copy whole Krylov subspaces.

This is not the top hotspot for `241_VBSCF`, but it becomes more visible when
the reduced dimension or inner iteration count grows.

### Plan

1. Store Krylov basis and HVP basis directly in preallocated `Eigen::MatrixXd`
   column blocks.
2. Return or move subspace objects instead of copying them.
3. Avoid rebuilding the reduced Hessian from scratch when appending one basis
   vector.
4. Add diagnostics for Krylov finalization time.

### Acceptance Criteria

- Lower allocation count during TN solves.
- No change in accepted steps or trust-region model values.

## P7. Trial Step Packing And Backtracking Copies

### Problem

The nonredundant trial builder repacks the current orbital input and constructs
a full trial `OrbitalPreparationInput` for every backtracking attempt. The
caller already has `current_parameters`, so part of this work is avoidable.

### Plan

1. Pass `current_parameters` into the lifted-trial helper instead of repacking.
2. Add a sparse in-place pack API for trial orbital inputs if the current
   `parameter_view.pack(...)` shows measurable cost.
3. Reuse trial buffers across Armijo attempts.

### Acceptance Criteria

- Lower cost in rejected-trial-heavy TNHVP iterations.
- No change in finite trial orbitals or accepted parameters.

## P8. Long-Term Chart Reformulation

### Problem

Some complexity in fixed-upstream and active auxiliary pullbacks comes from the
current inactive block chart. A true orthogonal inactive-frame formulation could
simplify several derivative chains, but it touches accepted-point semantics and
is higher risk.

### Plan

1. Keep this out of the short-term performance branch.
2. Derive the new chart independently.
3. Add finite-difference checks before replacing the production chart.

### Acceptance Criteria

- Only consider implementation after P1-P5 are measured.
- Must preserve nonorthogonal VB orbital semantics.

## Validation Matrix

Minimum checks after each nontrivial implementation:

```bash
cmake --build build --target run_cpp_vbscf check_exact_ctx_hvp benchmark_exact_ctx_hvp -j8

env OMP_NUM_THREADS=1 build/src/check_exact_ctx_hvp \
  src/test_molecule/F2.xmi --nonredundant-adapt true --ao-integral-source libcint_cpp

env OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  build/src/benchmark_exact_ctx_hvp test/241_VBSCF.xmi \
  --repeats 2 --warmup 1 --ao-integral-source libcint_cpp
```

For optimizer-level changes, also run at least one TNHVP end-to-end job through
`test/vbscf-cpp-tnhvp.sh` and record:

- accepted iterations;
- rejected trials;
- HVP counts by type;
- total wall time;
- final projected gradient.

## Progress Log

| Date | Change | Status | Benchmark / Validation |
| --- | --- | --- | --- |
| 2026-04-23 | Created this plan from current static inspection and `241_VBSCF` HVP benchmark. | baseline documented | `full_cached=0.728s`, `core_only=0.160s`, `outer_only=0.485s` on 32 threads |
| 2026-04-23 | Tuned directional-structure thread heuristic and exposed `XMVB_CPP_EXACT_CTX_DIRECTIONAL_STRUCTURE_THREAD_DIVISOR`. | implemented | `sbatch` benchmark on `241_VBSCF`: structure stage `0.2088s -> 0.0783s` (`96 -> 48`), full HVP `0.3846s -> 0.2684s` on the tested nodes. `10698_VBSCF`: structure stage `0.2073s -> 0.0843s`, full HVP roughly flat within node noise. |
| 2026-04-23 | Reused accepted-point selected-state generalized-eigen response metadata through the outer-response cache. | implemented | `sbatch` `check_exact_ctx_hvp test/241_VBSCF.xmi --ao-integral-source libcint_cpp`: `full_max_abs_diff=4.76e-09`, `fixed_max_abs_diff=3.83e-09`, `outer_max_abs_diff=2.83e-09`. |
| 2026-04-24 | Replaced singular same-spin second-order deleted-minor values in forward/backward/HVP with one shared SVD/null-space cofactor formula. Directional deleted-minor derivatives still use the generic path. | implemented, numerically validated | Local rebuild passed for `run_cpp_vbscf`, `check_exact_ctx_hvp`, and `benchmark_exact_ctx_hvp`. `F2` smoke stays at final energy `-198.751155825177`. On `6526Y`, `check_exact_ctx_hvp test/241_VBSCF.xmi --ao-integral-source libcint_cpp` (`1942375`) gives `full_max_abs_diff=3.57e-09`, `outer_max_abs_diff=2.23e-09`. `benchmark_exact_ctx_hvp test/241_VBSCF.xmi` (`1942374`) gives `full_cached=0.3413s`, `core_only=0.1161s`; this does not beat the recent `bench-241-current.1942293.out` baseline (`full_cached=0.2389s`, `core_only=0.1161s`), so `241_VBSCF` is still not limited by the singular second-order value path alone. |
| 2026-04-24 | Specialized singular same-spin degree-2 directional deleted-minor kernels to avoid generic deleted-index vectors, `std::find`, and generic minor scattering in the hot HVP/backward loops. | implemented, numerically validated | Local rebuild passed again for `run_cpp_vbscf`, `check_exact_ctx_hvp`, and `benchmark_exact_ctx_hvp`; `F2` smoke still ends at `-198.751155825177`. On `6526Y`, `benchmark_exact_ctx_hvp test/241_VBSCF.xmi` (`1942377`) improves the immediately previous singular-path trial from `full_cached=0.3413s` to `0.2614s`; active-2e drops `0.0766s -> 0.0560s`, and outer-response orbital-pullback drops `0.0507s -> 0.0242s`. Against the stronger historical `bench-241-current.1942293.out` baseline, `241` is still mixed (`full_cached 0.2614s` vs `0.2389s`), so the singular directional path is not the only remaining bottleneck. `MnF2` validation on `6526Y` also stays consistent: `check_exact_ctx_hvp test/MnF2.xmi` (`1942379`) gives `full_max_abs_diff=1.47e-4`, `max_rel_diff=6.16e-8`, and `benchmark_exact_ctx_hvp test/MnF2.xmi` (`1942378`) reports `full_cached=0.2359s`, `core_only=0.0845s`. |
| 2026-04-24 | Tightened `calc_same_spin_hamiltonian_impl` to use direct packed-`GGO` dispatch and hoist `1 / det(S)` out of the regular same-spin inner loop. A trial occupied-pair packed-index cache was benchmarked and discarded because it regressed `241_VBSCF`. | implemented, mixed perf impact | Final retained version is numerically consistent on `241_VBSCF`: `check_exact_ctx_hvp` (`1942388`) gives `full_max_abs_diff=5.55e-09`, `outer_max_abs_diff=3.27e-09`. On `6526Y`, `benchmark_exact_ctx_hvp test/241_VBSCF.xmi` (`1942387`) lands at `full_cached=0.2615s`, essentially flat against the immediately previous `1942377` run (`0.2614s`), with `core_only active_2e = 0.0498s` versus `0.05195s` there. The discarded occupied-pair cache trial (`1942385`) regressed badly to `full_cached=0.3311s`. |
## P7. Trial Step Evaluation Copies

### Problem

The TN optimizer was still copying `OrbitalObjective` in the accepted-point
trial path:

- once to initialize the accepted-trial holder;
- once for each trust-region trial step;
- once again in the nonredundant Armijo fallback path.

That object includes the current `CppVbInput`, the last full
`CppOrbitalGradientResult`, and the accepted-point second-order context.
Copying it on every rejected or accepted trial is much more expensive than a
small control object copy and is unrelated to the Hessian formula itself.

### Plan

1. Split trial evaluation into:
   - scratch evaluation without committing the accepted point;
   - explicit commit only after the trust-region trial is accepted.
2. Rebuild lightweight probe objectives from immutable inputs instead of
   copying the last accepted gradient result and second-order context.
3. Re-test `241_VBSCF` on Slurm with TN policy/HVP diagnostics enabled to see
   whether accepted-step wall time drops materially.

### Implemented So Far

1. Added `OrbitalObjective::TrialEvaluation` so TN trial steps can evaluate a
   scratch point and defer accepted-point commit.
2. Reworked `OrbitalObjective::operator()` to reuse the same scratch/commit
   path rather than maintaining a second copy-heavy implementation.
3. Changed `OrbitalObjective::make_probe_copy()` to reconstruct a fresh probe
   object from immutable inputs instead of copying the last accepted gradient
   result and second-order context.
4. Replaced the main TN trust-region trial path with
   `evaluate_trial_without_committing(...)`, so the accepted-point state is
   only updated after the step passes the trust-ratio test.
5. Switched the descent fallback path from `OrbitalObjective fallback = objective`
   to `objective.make_probe_copy()`.

### Status

- Code change implemented and validated on Slurm.

### Measured Progress

All runs below used `sbatch -c 32`, `OMP_NUM_THREADS=32`,
`OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`.

#### `test/241_VBSCF.xmi`

Main before/after comparison:

| backend / policy | log | iterations | SCF wall | end-to-end wall |
| --- | --- | ---: | ---: | ---: |
| TNHVP default, before trial-copy fix | `test/241_VBSCF.1942256.stdout.log` | 7 | 29.061628 s | 31.846091 s |
| TNHVP default, after trial-copy fix | `241_VBSCF.1942267.stdout.log` | 7 | 10.372709 s | 13.195751 s |
| LBFGS++ same environment | `241_VBSCF.1942271.stdout.log` | 74 | 10.966597 s | 13.771947 s |

The optimized TNHVP path now slightly beats LBFGS++ wall time on this deck,
but the margin is small.  The accepted-step times are roughly
`1.50, 1.45, 1.08, 1.08, 1.05, 0.79, 0.81 s`, so the current target should be
to make accepted TN steps consistently `~1 s` or below.

Policy experiments after removing trial-copy overhead:

| policy | log | iterations | SCF wall | result |
| --- | --- | ---: | ---: | --- |
| explicit `max_cg=6` | `241_VBSCF.1942268.stdout.log` | 9 | 10.479432 s | similar but not faster |
| explicit `max_cg=8` | `241_VBSCF.1942269.stdout.log` | 8 | 11.771201 s | slower |
| forced core-only outer-response off | `241_VBSCF.1942273.stdout.log` | 36 | 40.344074 s | much slower |
| lean policy: no startup full, no retry/full followup | `241_VBSCF.1942274.stdout.log` | 36 | 44.749127 s | much slower |

This invalidates the earlier suspicion that the default policy should simply
turn off outer-response for `241`.  The first two full startup solves are
important for iteration count.  The remaining performance work should focus on
the default path's cheap-core HVP count/cost, not on removing full startup
entirely.

## Next Structural Priorities

The current measurements suggest that most low-risk quick wins are already
exhausted. Future work should therefore be prioritized by expected broad
benefit across deck families, not by isolated wins on `241_VBSCF`.

### Priority 1. Active-2e Default Path Rewrite

- Area: `active_2e` default materialized HVP path
- Expected benefit: high
- Implementation cost: high
- Risk: medium to high
- Broad applicability: high for exact small-to-medium active spaces

Reasoning:

- `active_2e` remains a major cheap-core cost on both compact and larger exact
  decks.
- The current default path still pays for
  `mixed -> transformed -> pair_gradients -> dense_active_gradient`.
- Small loop optimizations can help, but the next real gain likely requires a
  true streaming or partially fused rewrite of the default materialized path.

Recommended scope:

1. Keep the accepted-point cache layout.
2. Reduce or eliminate one of the large AO-pair by active-pair intermediates.
3. Preserve the current default path as a reference during rollout.

### Priority 2. Outer-Response Active-Space Integrals

- Area: `outer_response_active_space_integrals`
- Expected benefit: high
- Implementation cost: high
- Risk: medium
- Broad applicability: high

Reasoning:

- On larger decks such as `10698_VBSCF`, this is the dominant outer-response
  stage.
- The current path still rebuilds `delta SSO`, `delta HHO`, and `delta GGO`
  from accepted-point metadata every HVP.
- This is structurally broader than `241` and therefore a better general
  target than further tuning one cheap-core kernel in isolation.

Recommended scope:

1. Split accepted-point layout metadata from per-direction values.
2. Cache traversal / scatter / support metadata aggressively.
3. Update directional values into preallocated accepted-point workspaces.

### Priority 3. Outer-Response Orbital Pullback

- Area: `outer_response_orbital_pullback`
- Expected benefit: medium
- Implementation cost: medium
- Risk: medium
- Broad applicability: high

Reasoning:

- This stage is no longer the largest hotspot, but it is still visible across
  multiple decks.
- The path still mixes symmetrization, legacy row-buffer conversion, active
  matrix backprop, two-electron backprop, and orbital pullback in separate
  steps.

Recommended scope:

1. Reuse workspace buffers more aggressively.
2. Reduce format crossings between Eigen matrices and legacy row buffers.
3. Combine consecutive pullback stages where they share the same accepted-point
   layout.

### Priority 4. AO-H1E Fused Fallback Path

- Area: `ao_h1e_fused`
- Expected benefit: medium
- Implementation cost: medium
- Risk: low to medium
- Broad applicability: medium

Reasoning:

- When graph fast paths are active, `ao_h1e_fused` is already much better than
  earlier baselines.
- Remaining work here is mostly about fallback-path workspace reuse and thread
  reduction cost, not a likely order-of-magnitude gain.

Recommended scope:

1. Reuse multithread partial buffers across applies instead of rebuilding local
   `std::vector<std::vector<double>>` storage.
2. Check whether graph-unavailable decks still pay a large fixed overhead.
3. Only pursue deeper refactors if a deck family actually lands on the fallback
   path in production.

### Priority 5. Further TN Control-Flow Optimization

- Area: trial / retry / trust-region orchestration
- Expected benefit: medium
- Implementation cost: medium
- Risk: high
- Broad applicability: medium

Reasoning:

- This area can still affect end-to-end wall time materially.
- However, the recent optimizer-copy fix already removed the largest obvious
  waste, and the default startup full-HVP policy is important for convergence.
- Additional work here is much more likely to trade robustness for speed.

Recommended scope:

1. Treat this as a second-pass optimization only after kernel-side work.
2. Require convergence comparisons, not wall time alone.
3. Avoid policy changes justified only by one deck.

### What Is No Longer A Good Bet

- Pure hand-loop tuning without reducing a real stage boundary.
- Disabling outer-response by default for cheap solves.
- Optimizations justified only by `241_VBSCF`.
- Reusing a cache built for one mathematical direction in another direction
  without measuring the traversal cost separately.

### Current Recommendation

If only one structural project is taken on next, it should be:

1. `active_2e` default-path rewrite, if the goal is faster cheap-core HVP.
2. `outer_response_active_space_integrals`, if the goal is broader full-HVP
   improvement across larger decks.

The first target is more aligned with reducing accepted TN step cost on the
current default strategy. The second target is more likely to generalize across
deck families.

### Static Bottleneck Map (2026-04-24)

This pass re-read the exact HVP call chain instead of chasing another small
kernel. The current code confirms that the next meaningful gains must come
from stage-boundary changes, not from more determinant-level tuning.

#### 1. `active_2e` default exact-HVP path is still the main cheap-core target

The accepted-point exact 2e apply in
`src/vb/orbital/active_space_two_electron_utils.cpp` still executes these
large stages in order:

1. copy `dense_active_direction` into the legacy row buffer
2. fixed accepted-adjoint backprop into dense active rows
3. build `mixed_pair_coefficients_buffer`
4. optional cached-row direct path
5. otherwise `mixed * active_pair_gradient_matrix`
6. optional fused AO-kernel backprop
7. otherwise materialize `pair_gradients_buffer`
8. final dense-active backprop

The important point is that the unresolved fallback still carries three large
`n_ao_pairs x n_active_pairs` buffers:

- `mixed_pair_coefficients_buffer`
- `transformed_pair_coefficients_buffer`
- `pair_gradients_buffer`

That means the remaining structural cost is not in one tiny inner loop. It is
the repeated AO-pair streaming across multiple full intermediates. As long as
that fallback exists in this shape, hand-tuning one determinant kernel or one
packing helper will only produce marginal wins.

Conclusion:

- Best next cheap-core target: remove one full AO-pair-by-active-pair
  intermediate from the materialized path.
- Bad target: more micro-optimizing `calc_same_spin_hamiltonian_impl`.

#### 2. `outer_response_active_space_integrals` is dominated by `delta GGO`

`build_active_space_directional_integrals(...)` itself is now split cleanly:

- `delta SSO`: one active-sized GEMM plus transpose
- `delta HHO`: a few active-sized GEMMs
- `delta GGO`: full exact directional builder

The first two are already cheap. The heavy part is
`compute_exact_packed_active_two_electron_integral_directional_derivative(...)`.
The current implementation still does:

1. build directional AO-pair-to-active-pair coefficients
2. run the AO pair kernel on that buffer
3. loop over every AO pair
4. accumulate two rank-1 outer products into a dense active-pair matrix
5. repack that matrix back to packed `GGO`

So the outer-response hotspot is not "active-space integrals" in general. It
is specifically the directional exact-2e builder inside that stage.

Conclusion:

- Best next outer-response target: rewrite the `delta GGO` builder, not the
  already-small `delta SSO/HHO` algebra.
- Bad target: touching accepted-point active-matrix GEMMs again.

#### 3. The accepted exact-2e cache is reusable, but not a free outer-response win

There is already a cached overload for directional `delta GGO` that reuses:

- accepted dense active coefficients
- accepted pair coefficients
- accepted base pair products

So in principle the cache can shorten the outer-response path. But the code and
previous measurements show an important limitation: simply routing
`outer_response_active_space_integrals` through the accepted exact-2e cache was
already tried and regressed. The reason is consistent with the current code:
the expensive part is still the AO-kernel sweep plus the AO-pair reduction into
`delta_active_pair_matrix`, not just rebuilding accepted metadata.

Conclusion:

- Do not retry the old "just reuse the accepted exact-2e cache" experiment.
- If this area is revisited, the rewrite must reduce AO-pair streaming or fuse
  the AO-pair reduction itself.

#### 4. `outer_response_orbital_pullback` is real, but second-tier

The outer-response consumer chain is:

1. directional structure matrices
2. directional eigensystem response
3. active-space gradient direction
4. orbital pullback

The `delta GGO` result is consumed by both the structure-matrix builder and the
outer-response backward pass, so it is not dead work. The orbital pullback path
still mixes symmetrization, matrix backprop, exact-2e backprop, and orbital
backprop, but at the moment it is a second-tier target behind the two exact-2e
stages above.

Conclusion:

- Keep this as a cleanup / reuse target.
- Do not treat it as the main reason `241` or larger decks are still slow.

#### Updated recommendation

If we only take one substantial optimization next, the most defensible order is:

1. `active_2e` materialized fallback rewrite
2. `outer_response_active_space_integrals` via `delta GGO` rewrite
3. `outer_response_orbital_pullback` cleanup

This ordering matches both the benchmark breakdowns and the actual code shape.
