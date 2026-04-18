# Exact Separator Performance Gap and `2^n -> 2^m` Roadmap

## 1. Purpose

This note records the current implementation status of
`src/vb/exact_separator` from a performance and representation point of
view.

The main questions are:

1. what is already exact in the current code;
2. why the current implementation is still slower than determinant-pair
   contraction on the present benchmark set;
3. why the code has still not reached the intended practical
   `2^n -> 2^m` transition;
4. which implementation steps are most likely to improve the production hot
   path first.

Here:

- `n` means subtree internal combinatorial size;
- `m` means separator / boundary width.

---

## 2. Short Answer

The current codebase is in a mixed but useful state.

1. the specialized `one_leaf_star` production path is still exact on the
   validated one-leaf synthetic case;
2. the current local workspace also contains a generic rooted-tree
   `branched_rooted_subtree` Hamiltonian regression, so generic exactness
   must be treated as "historically validated but currently recheck needed";
3. however, the dominant `one_leaf_star` production hot path is still not a
   true boundary-only message implementation;
4. the generic recursive path still uses sparse deleted-label payloads with
   expensive canonicalization and merge overhead;
5. therefore the code has not yet reached a clean practical
   `poly(n) * 2^{O(m)}` implementation, even though the exact recursive
   closure is no longer the only missing issue.

So the main structural gap is still representation and data layout, even
though the current workspace also needs generic exactness revalidation.

---

## 3. Current Code Paths

### 3.1 Rooted-tree Hamiltonian dispatch

The main production entry point is
[`evaluate_rooted_component_tree_hamiltonian_collapsed(...)`](./component_tree.cpp).

Its behavior is:

1. one-node trees dispatch to a local one-node path;
2. `one_leaf_star` trees dispatch to the specialized one-leaf production path;
3. all other trees dispatch to the generic recursive bundle path.

Relevant locations:

- [`component_tree.cpp`](./component_tree.cpp)
- [`two_electron_opposite_spin.cpp`](./two_electron_opposite_spin.cpp)

This distinction matters because the hottest benchmarked cases currently fall
into the specialized `one_leaf_star` branch, not into the fully generic
recursive branch.

The public
[`evaluate_rooted_component_tree_hamiltonian_bundle_boundary_collapsed(...)`](./component_tree.cpp)
entry now special-cases `one_node` and `one_leaf_star` trees instead of always
forcing the older generic recursive bundle path.

More precisely:

1. `one_node` still aliases the current production collapsed evaluator;
2. `one_leaf_star` now dispatches to a dedicated
   `evaluate_one_leaf_hamiltonian_boundary_bundle_collapsed(...)` helper;
3. only non-one-leaf trees still route to the generic recursive bundle path.

That matters because the `one_leaf_star_generic_bundle` benchmark label is now
backed by a real dense boundary-bundle one-leaf evaluator rather than by an alias
of the production direct-table kernel.

### 3.2 `one_leaf_star` production path

The one-leaf path currently builds:

1. one compressed root coefficient operator;
2. one compressed leaf coefficient operator;
3. one exact direct alpha table over root-state × leaf-state;
4. one exact direct beta table over root-state × leaf-state;
5. one fused contraction over the compressed coefficient operators.

The key implementation pieces are:

- [`leaf_coefficient_operator.cpp`](./leaf_coefficient_operator.cpp)
- [`spin_state_aggregate.cpp`](./spin_state_aggregate.cpp)
- [`two_electron_opposite_spin.cpp`](./two_electron_opposite_spin.cpp)

The crucial point is that the state basis here is still

`SpinPairStateKey = (left_occ, right_occ)`.

So the current exponential object in the one-leaf hot path is still the
number of unique full one-spin occupied-state pairs, not the number of
boundary sectors.

### 3.3 Generic rooted-tree recursive path

The generic recursive Hamiltonian path already propagates subtree messages
indexed by boundary sectors and merges them recursively.

Its central message/value types are:

- `BoundaryHamiltonianMessage`
- `HamiltonianBoundaryPayload`

The recursion and frontier merge live mainly in:

- [`component_tree.cpp`](./component_tree.cpp)

This path is already much closer to the intended separator formulation, but
its payload layout is still sparse deleted-label storage rather than the final
packed dense family-block bundle.

---

## 4. What Is Already Exact

For the current local workspace, the strongest validated statement is:

1. the specialized `one_leaf_star` production path is exact on the current
   one-leaf synthetic validator;
2. the dedicated overlap and one-electron boundary evaluators remain exact on
   the current non-star synthetic probes;
3. the historical design work and earlier validators still show that the
   generic rooted-tree Hamiltonian recurrence can be exact at the
   representation level;
4. but the present workspace must not be described as "fully exact again"
   until the current `branched_rooted_subtree` Hamiltonian regression is
   removed.

The important practical consequence is:

1. the structural `2^n -> 2^m` blocker is still representation overhead;
2. but structural migration should be staged carefully enough that the current
   generic regression is not used as evidence for or against one-leaf changes.

---

## 5. Why The Current Code Is Still Not A True `2^n -> 2^m` Implementation

## 5.1 The `one_leaf_star` hot path is still full-state, not boundary-only

The most important structural reason is simple:

the current one-leaf production path still builds exact full-state direct
aggregate tables over

`|root_states| * |leaf_states|`.

This is implemented by
[`build_indexed_one_leaf_direct_spin_aggregate_table(...)`](./spin_state_aggregate.cpp)
and consumed by
[`contract_one_leaf_direct_aggregate_channels(...)`](./spin_state_aggregate.cpp).

This means:

1. subtree interior information is still explicit in the state label;
2. the number of states can still scale with the full subtree combinatorics;
3. the implementation is therefore not yet the intended boundary-only
   separator message.

This is the main reason the hottest production branch has not yet completed
the `2^n -> 2^m` transition.

## 5.2 The generic recursive path is only partially migrated

The generic rooted-tree path is already boundary-indexed, but it still carries
payloads as sparse deleted-label sector lists:

- same-spin sectors keyed by `SpinDeletionKey`;
- mixed sectors keyed by `JointDeletionKey`.

So while the exponential object is now closer to the boundary width, the
polynomial payload attached to each sector is still represented in a generic
sparse format with high constant overhead.

This means the code is currently closer to:

`boundary-only outer state + expensive sparse exact payload`

than to:

`boundary-only outer state + packed dense polynomial payload`.

## 5.3 Existing dense bundle infrastructure is not yet on the production path

The repository already contains the main ingredients of the final packed
message design:

- [`bundle.hpp`](./bundle.hpp)
- [`bundle.cpp`](./bundle.cpp)
- [`bundle_merge.cpp`](./bundle_merge.cpp)

These files provide:

1. canonical dense bundle layouts;
2. exact forward merge on dense degree-0/1/2 payloads;
3. exact mixed alpha/beta bundle merge;
4. reverse merge support for later gradient work.

However, the current production rooted-tree driver is still not using
`BoundaryBundle` as its primary message type.

So the bundle design is present, but it is not yet the actual production
carrier of the subtree message algebra.

---

## 6. Current Performance Bottlenecks

## 6.1 Full-state one-leaf direct tables

The one-leaf production path builds exact direct aggregate tables in Cartesian
product order:

`root_state_index * leaf_state_count + leaf_state_index`.

This has two consequences:

1. work scales with unique full-state count rather than boundary width;
2. the final contraction still runs over root-entry × leaf-entry products.

This is the single most important reason the current one-leaf hot path does
not yet realize separator-width scaling.

## 6.2 Sparse deleted-label payload storage

The generic path stores polynomial payloads as sparse vectors of:

- `SameSpinDeletedSectorValue`
- `HamiltonianMixedDeletedSectorValue`

This introduces repeated overhead from:

1. key concatenation;
2. rank checks;
3. parity canonicalization;
4. sorting and compaction;
5. poor memory locality;
6. inability to use tight dense block contractions.

The relevant merge and cleanup code is concentrated in:

- [`component_tree.cpp`](./component_tree.cpp)

## 6.3 `std::map`-based frontier merge

The frontier merge currently uses

`std::map<FrontierHamiltonianMaskKey, HamiltonianBoundaryPayload>`.

This is structurally correct, but expensive in the hot path because it
combines:

1. dynamic node allocation;
2. ordered tree traversal;
3. repeated payload insertion and accumulation;
4. extra branch-heavy mask compatibility checks.

For exact recursion this is acceptable as a temporary implementation, but it
is not the right final data structure.

## 6.4 Repeated local payload construction inside recursive loops

In the generic recursive Hamiltonian loop, the code still repeatedly builds
local exact interface payloads inside nested loops over:

1. alpha parent sectors;
2. beta parent sectors;
3. frontier entries.

The repeated work includes:

- exact frontier spin payload build;
- Hamiltonian payload projection;
- interface-block reordering transform.

Even though the outer state is already boundary-only, the per-sector local
cost is still high.

## 6.5 Expensive deleted-minor builders

The scalar deleted-minor payload builders still rely on:

1. subset enumeration;
2. deleted-minor matrix construction;
3. determinant evaluation;
4. map insertion by deleted-label keys.

Some of this is unavoidable in exact arithmetic, but the current
implementation also pays avoidable overhead such as repeated small-object
construction and repeated `std::find` scans while building minors.

## 6.6 The current reusable Schur/block-family mode is not the active fix

The one-leaf direct aggregate builder does contain a reusable Schur/block
family mode. However, current local measurements show that simply enabling
that path in the production one-leaf kernel does not improve the hot
benchmark. On `C6H6_full` it increased subdeterminant counts and slightly
worsened wall-clock time.

So the present production path intentionally keeps that mode disabled.

The important conclusion is:

the main remaining one-leaf gap is not "we forgot to turn on an existing
reuse switch". The effective improvement came instead from changing the
representation and contraction pattern, especially avoiding unnecessary
support-space dense materialization in the two-electron path.

This means the main remaining blocker is still structural:

the production one-leaf path is full-state, not boundary-only.

## 6.7 The current benchmark set is star-heavy

On the present molecular benchmark set, especially `C6H6_full`, the connected
tree cases are dominated by star trees and especially one-leaf stars.

That matters because:

1. the production hot path is exactly the path that is still full-state;
2. star cases leave less room for recursive asymptotic advantage;
3. determinant-pair exact contraction is already strong on these small
   present-day cases.

So the current benchmark regime is almost the worst possible place to expect
the partially migrated separator implementation to win.

---

## 7. Local Validation And Benchmark Evidence

## 7.1 Exactness evidence

As of `2026-04-06`, the current local workspace no longer matches the earlier
"all synthetic cases exact" state.

Local run:

```bash
OMP_NUM_THREADS=1 build/src/check_exact_separator_two_electron_component_tree
```

Observed exact agreement to numerical noise on:

1. `chain_rooted_subtree`
2. `one_leaf_star_rooted_subtree`
3. `star_rooted_subtree`

Observed regression on `branched_rooted_subtree` in the current tree:

1. `collapsed_one_electron = 31.8395052396837` vs
   `exact_one_electron = 30.9138107018712`
2. `collapsed_opposite_spin = 168.014841354014` vs
   `exact_opposite_spin = 121.077925980514`
3. `collapsed_total_electronic = 203.131361416496` vs
   `exact_total_electronic = 154.889231723935`

At the same time, the dedicated overlap and one-electron boundary checks for
that same branched case are still exact to numerical noise.

So the current local conclusion is more limited than before:

1. the specialized `one_leaf_star` production path is still exact on the
   validated synthetic one-leaf case;
2. the current workspace still contains a generic rooted-tree Hamiltonian
   regression on `branched_rooted_subtree`;
3. that regression persisted after removing the recent local
   `leaf_coefficient_operator.cpp` state-compaction experiment, so it is not
   evidence that constant-factor one-leaf tweaks are the right place to work;
4. structural migration work should therefore stay isolated to the one-leaf
   hot path until the generic branch is revalidated.

## 7.2 Benchmark evidence on `C6H6_full`

Local run:

```bash
OMP_NUM_THREADS=1 build/src/benchmark_rooted_component_tree_pairs \
  test_molecule/C6H6_full.xmi \
  --max-pairs 5 \
  --repeat 5 \
  --edge-threshold 0
```

Observed summary on the current workspace:

1. `connected_tree_pairs = 5`
2. `star_tree_pairs = 5`
3. `one_leaf_star_pairs = 3`
4. `multi_leaf_star_pairs = 0`
5. `nonstar_tree_pairs = 0`

For the one-leaf-star subset:

1. `exact_total_seconds = 0.002423058`
2. `collapsed_total_seconds = 0.002739676`
3. `collapsed_speedup_vs_exact = 0.884432319734158`
4. `collapsed_avg_subdeterminant_evaluations = 128`
5. `collapsed_avg_subtree_message_state_count = 128`

For the one-leaf-star "generic bundle" replay:

1. `exact_total_seconds = 0.002360794`
2. `boundary_total_seconds = 0.164771427`
3. `boundary_speedup_vs_exact = 0.0143276904435622`
4. `boundary_avg_subtree_message_state_count = 2048`
5. `boundary_avg_subdeterminant_evaluations = 7552`

For the split one-leaf-star two-electron subset:

1. `two_electron_exact_total_seconds = 0.002373695`
2. `two_electron_collapsed_total_seconds = 0.002707624`
3. `two_electron_collapsed_speedup_vs_exact = 0.876670837605221`
4. `two_electron_collapsed_avg_subdeterminant_evaluations = 128`

This is consistent with the current structure:

1. the production one-leaf Hamiltonian path is exact;
2. the current benchmark set is still entirely star-dominated, so one-leaf is
   still the most important production optimization target;
3. the `one_leaf_star_generic_bundle` replay is now a real boundary-backed
   alternative and remains exact to numerical noise;
4. neither the present one-leaf collapsed path nor that boundary-backed replay
   reaches speedup `> 1`;
5. the production hot path still shows the old `128` full-state signature,
   while the boundary-backed replay now exposes a much larger
   `2048`-state / `7552`-subdeterminant scaffold;
6. that new counter profile is exactly what we expect from an intermediate
   representation change that still rebuilds explicit `root_states ×
   leaf_states` instead of contracting a final packed boundary bundle.

One additional local result is now clear enough to treat as a design
constraint.

A temporary production experiment switched the `one_leaf_star` collapsed path
from direct aggregate tables to a bundle-carrier implementation that rebuilt
exact one-spin boundary bundles for each state pair and then read back the
Hamiltonian channels from the accumulated `BoundaryBundle`.

That experiment remained exact on the synthetic one-leaf validator, but on
the same `C6H6_full` benchmark it regressed badly:

1. `collapsed_speedup_vs_exact = 0.0219395233878493`
2. `collapsed_avg_subtree_message_state_count = 2048`
3. `collapsed_avg_subdeterminant_evaluations = 7552`

The production switch was therefore reverted.

This matters because it rules out one tempting but insufficient rewrite:

simply replacing the one-leaf carrier by `BoundaryBundle` is not enough if
the implementation still constructs bundle data at explicit state-pair
granularity. The real optimization must change the construction and merge
granularity together, not only the final carrier type.

## 7.3 Current scaffold status after the April 6, 2026 rewrite pass

The repository now has three distinct one-leaf bundle-related pieces, and they
must not be conflated.

1. `leaf_boundary_bundle_table.cpp` is now an exact validation/debug scaffold.
   It builds exact one-spin `BoundarySpinBundle` objects for explicit
   root/leaf state pairs and is still useful for checking bundle algebra and
   debugging first-cofactor channels.
2. `one_leaf_packed_bundle.cpp` is the actual `one_leaf_star_generic_bundle`
   production replay path used by
   `evaluate_rooted_component_tree_hamiltonian_bundle_boundary_collapsed(...)`.
   It no longer exposes the old aggregate-table scaffold as the public
   carrier.
3. the generic rooted-tree recursion in `component_tree.cpp` is still carried
   by `HamiltonianBoundaryPayload`, not by a dense bundle-native message.

That distinction matters because only the second item is on the performance
critical benchmark path, while the third item is still the real blocker for the
full `2^n -> 2^m` migration.

During the current rewrite pass, the one-leaf packed bundle builder was changed
again:

1. it now caches only the one-spin root/leaf state pairs actually referenced by
   the sparse `ComponentSpinCoefficientOperator.entries`;
2. the old debug tool call sites were updated to the same sparse state-pair
   interface;
3. the generic recursive `BoundarySpinBundleLayout` construction now carries
   the real `support_size` instead of the placeholder `0`, so later dense
   degree-1/2 bundle work is no longer structurally blocked on missing layout
   dimensions.

However, that sparse state-pair rewrite did **not** materially improve the hot
benchmark.

On

`env OMP_NUM_THREADS=1 build/src/benchmark_rooted_component_tree_pairs test_molecule/C6H6_full.xmi --max-pairs 5 --repeat 5 --edge-threshold 0`

the current exact results are:

1. `one_leaf_star_generic_bundle.boundary_seconds_per_repeat = 0.0269340102`
2. `one_leaf_star_generic_bundle.boundary_speedup_vs_exact = 0.021459960685691`
3. `one_leaf_star_generic_bundle.boundary_avg_subtree_message_state_count = 32`
4. `one_leaf_star_generic_bundle.boundary_avg_subdeterminant_evaluations = 7552`

So the cache granularity is cleaner, but the dominant work is still unchanged.
The reason is that the packed path still constructs one full exact
`OneLeafBoundarySpinMessage` per referenced one-spin state pair. For the
current benchmark cases, the referenced state-pair set is already close enough
to dense that removing the unused Cartesian entries does not change the real
cost center.

## 7.4 Why the one-leaf packed path is still far from the final migration

The current one-leaf packed path is exact and structurally cleaner than the
removed aggregate scaffold, but it is still only an intermediate waypoint.

What it already does right:

1. it carries the final one-leaf Hamiltonian channels as
   `overlap + alpha bundle + beta bundle + mixed bundle`;
2. it contracts one-electron, same-spin, and opposite-spin totals directly
   from those dense bundle channels;
3. it no longer exposes a bespoke aggregate-only intermediate table.

What it still does wrong from a `2^n -> 2^m` perspective:

1. it builds one exact one-spin boundary message per referenced state pair;
2. that builder still enumerates the exact one-leaf sector families for each
   pair independently;
3. so the dominant work is still state-pair materialization, not one final
   boundary-only contraction.

This is why the sparse state-pair cache was necessary but insufficient.

The next one-leaf optimization cannot be another cache rearrangement. It has to
remove the per-state-pair exact message construction itself.

## 7.5 The generic rooted-tree blocker is now very specific

The generic rooted-tree code is no longer blocked by a vague "maybe the outer
state space is still too big" concern. The outer state space is already mostly
boundary/frontier-mask based. The real blocker is the **inner message carrier**.

The current forward chain is still payload-centric:

1. `BoundaryHamiltonianMessage` stores one `HamiltonianBoundaryPayload` per
   joint sector pair.
2. `FrontierHamiltonianMessage` stores one `HamiltonianBoundaryPayload` per
   frontier-mask combination.
3. `merge_frontier_hamiltonian_message_with_child(...)` still performs
   `frontier.entries × child_message.nonzero_entries` and then merges sparse
   deleted-label payloads.
4. `finalize_boundary_hamiltonian_message(...)` still walks the full joint
   sector grid and decides which sparse payloads survive cleanup.
5. root close/readout still consumes `HamiltonianBoundaryPayload`, not a dense
   Hamiltonian bundle carrier.

In other words, the current tree is closer to

`2^m outer masks + sparse exact payload algebra inside each mask`

than to

`2^m outer masks + fixed dense bundle contraction inside each mask`.

That is the real reason the codebase still has not completed the
`2^n -> 2^m` migration.

---

## 8. Updated Five-Step Structural Plan

The current implementation order should now be treated as five concrete steps.

## 8.1 Step 1: Keep the one-leaf packed path exact while removing obvious waste

This step is the work completed in the current pass.

Done:

1. the one-leaf packed path now caches only referenced one-spin state pairs;
2. the debug bundle-table tools were updated to the same sparse state-pair
   interface;
3. the recursive generic bundle layouts now carry the real `support_size`.

Result:

1. exactness is preserved on the one-leaf synthetic cases;
2. the pre-existing `branched_rooted_subtree` generic regression remains, but
   it was not introduced by this pass;
3. the hot benchmark remains much slower, so the next bottleneck is now
   isolated more clearly.

## 8.2 Step 2: Replace one-leaf per-state-pair message materialization

This is now the highest-value one-leaf optimization.

The target is **not** another table/cache tweak. The target is to stop calling
the full exact one-spin boundary-message builder independently for every
referenced state pair.

The right shape is:

1. build the final packed alpha/beta/mixed channels directly on dense boundary
   bundle coordinates;
2. reuse the same one-spin projected exact data across many state-pair
   contributions instead of materializing a whole sector-family message each
   time;
3. keep the current exact one-leaf bundle readout, but change the construction
   granularity under it.

If this step is not done, the one-leaf replay will continue to look
bundle-native while still paying state-pair exact-builder cost.

## 8.3 Step 3: Introduce a dense Hamiltonian forward carrier for generic trees

The next generic rewrite should not start by micro-optimizing
`HamiltonianBoundaryPayload`. It should replace that carrier.

The recommended new module is:

1. `hamiltonian_bundle.hpp`
2. `hamiltonian_bundle.cpp`

with one forward carrier shaped like:

1. `double overlap`
2. `BoundarySpinBundle alpha`
3. `BoundarySpinBundle beta`
4. `BoundaryMixedBundle mixed`

This is intentionally close to the already working one-leaf packed bundle
representation, so the one-leaf and generic code paths can converge on the
same dense Hamiltonian-channel semantics.

## 8.4 Step 4: Port the generic forward chain in the order leaf -> frontier -> root

The generic migration should not be attempted as one giant rewrite.

The practical order is:

1. convert `build_leaf_hamiltonian_subtree_message_values(...)` so leaf messages
   scatter directly into the dense Hamiltonian bundle carrier instead of back
   into `HamiltonianBoundaryPayload`;
2. convert `merge_frontier_hamiltonian_message_with_child(...)` so the fold is
   `dense bundle + dense bundle -> dense bundle`, not sparse payload merge;
3. convert root local build / root close / final readout to consume the same
   dense carrier.

The current reverse/gradient payload path can remain untouched until the
forward path is validated.

## 8.4a April 6 follow-up: what the current `hamiltonian_bundle` layer really is

One important architectural clarification came out of the latest rewrite pass:

the current [`hamiltonian_bundle.hpp`](./hamiltonian_bundle.hpp) carrier is
best understood as a **whole-message dense Hamiltonian channel carrier**, not
as a drop-in dense replacement for one per-entry
`HamiltonianBoundaryPayload`.

That distinction matters because the generic recursive message still has two
levels:

1. an outer alpha/beta boundary-sector table;
2. an inner payload carried at each table entry.

The present `HamiltonianBoundaryBundle` absorbs the boundary-sector basis into
the bundle itself through `flat_sector_index`, and its readout helper
[`contract_hamiltonian_boundary_bundle_channels(...)`](./hamiltonian_bundle.cpp)
already assumes final Hamiltonian-channel semantics.

So this layer is useful, but it does **not** mean the generic forward path can
simply replace

`std::vector<HamiltonianBoundaryPayload>`

by one `HamiltonianBoundaryBundle` without redesigning the per-entry carrier.

What the current pass did complete is the shared accumulation primitive layer:

1. `hamiltonian_bundle.cpp` now exposes
   `accumulate_weighted_spin_pair_into_hamiltonian_channels(...)`, so the
   common weighted alpha/beta/mixed accumulation law lives in one place;
2. the production one-leaf packed path now uses that shared dense Hamiltonian
   accumulation helper instead of duplicating the channel update logic
   locally.

One more structural cleanup is now in place as well:

3. [`bundle_root_helpers.cpp`](./bundle_root_helpers.cpp) is now part of the
   actual build, its mixed-channel parity helper uses the real alpha/beta
   bundle layouts instead of guessing sector counts from interface-label
   counts, and the module now exposes
   `reorder_root_hamiltonian_bundle(...)` for the dense Hamiltonian carrier.

This is intentionally a structural cleanup, not a claimed end-state
optimization. It removes one more fork between the one-leaf packed path and
the generic dense migration work, but it does **not** by itself finish the
generic `2^n -> 2^m` transition.

## 8.5 Step 5: Delete payload-era duplication only after the dense path runs

Dead-code removal is still required, but it must follow the structural switch,
not precede it.

Delete only after the dense forward path is active:

1. payload-only helpers that are no longer used by forward production;
2. placeholder bundle-root scaffolds that the final dense root close makes
   obsolete;
3. one-leaf debug or table code that is truly unreferenced after the dense
   path is validated.

Until then, validation scaffolds such as `leaf_boundary_bundle_table.cpp` still
have value and should not be removed prematurely.

## 8.6 Step 6: The forward generic dense payload path now closes structurally

The present pass did complete one important structural closure:

1. the generic non-one-leaf forward recursion in
   [`component_tree.cpp`](./component_tree.cpp) now has a real per-entry dense
   carrier,
   `HamiltonianEntryDensePayload`, instead of carrying
   `HamiltonianBoundaryPayload` through leaf build, frontier merge,
   interface transform, root close, and final readout;
2. that dense inner payload is **delta-aware** and explicitly keeps the open
   deleted-minor shapes required by the current exact algebra:
   `(1,0)`, `(0,1)`, `(2,1)`, `(1,2)`, `(2,0)`, `(0,2)`, as well as the
   square `(1,1)` and `(2,2)` channels;
3. the public forward wrappers for the generic Hamiltonian boundary path now
   dispatch to that dense forward recursion, while the old reverse / overlap
   gradient machinery remains on the legacy joint-payload path.

This is the first time the generic forward value path itself is no longer
payload-map based.

However, the benchmark result is also now clear: this closure is **not yet**
the final `2^n -> 2^m` answer.

On

`env OMP_NUM_THREADS=1 build/src/benchmark_rooted_component_tree_pairs test_molecule/C6H6_full.xmi --max-pairs 5 --repeat 5 --edge-threshold 0`

after the dense-forward switch, the current one-leaf generic benchmark is:

1. `one_leaf_star_generic_bundle.boundary_seconds_per_repeat = 0.0269340102`
2. `one_leaf_star_generic_bundle.boundary_speedup_vs_exact = 0.021459960685691`

So this pass closed the forward dense recursion structurally, but it did **not**
improve the hot benchmark. On this case it actually regressed slightly versus
the earlier sparse-typed generic path.

The reason is architectural, not just a bad constant:

1. the new dense carrier is still a **dense deleted-minor carrier**, not yet
   the final bundle-native `2^m` carrier;
2. every merge and interface transform still decodes deleted-minor basis
   elements, computes parity on those labels, and re-encodes them into the
   next dense payload;
3. that removes sparse maps from the forward value path, but it does **not**
   yet remove the label-aware per-entry deleted-minor algebra itself.

Validation after this pass is:

1. `chain_rooted_subtree` remains exact;
2. `one_leaf_star_rooted_subtree` remains exact;
3. `star_rooted_subtree` remains exact;
4. `branched_rooted_subtree` is still wrong, with the same generic correctness
   issue already present before this pass;
5. the forward dense rewrite did not introduce a new exactness regression on
   the previously passing cases.

---

## 8.7 April 6 audit: one failed fixed-key merge experiment was reverted, and the current workspace baseline is still not clean

The present pass also tested one more structural idea on the dense forward
inner algebra:

1. replace the current hot-path `decode -> deleted-label key -> parity ->
   re-encode` merge helpers by a fixed small-key dense helper layer inside
   [`component_tree.cpp`](./component_tree.cpp);
2. keep the outer frontier-mask DP unchanged, but try to make the inner
   dense Hamiltonian merge kernels less label-object driven.

That experiment did compile, but it failed the synthetic generic
two-electron validator immediately on multi-node cases, so it was **fully
reverted** in the same pass. No production merge code from that failed
experiment remains active.

This matters for two reasons.

First, it confirms something important about the remaining work:

1. the frontier merge layer is still the real structural bottleneck;
2. but it is also semantically delicate enough that a local kernel rewrite
   is not safe unless it is validated against the full generic exactness
   suite, not only one-leaf cases.

Second, re-running the current workspace baseline after that revert shows
that the generic Hamiltonian forward path is **still not in a clean
correctness state** on this checkout, even before another merge rewrite is
attempted.

On

`env OMP_NUM_THREADS=1 build/src/check_exact_separator_two_electron_component_tree`

the current workspace now reports:

1. `one_leaf_star_rooted_subtree`: still exact;
2. `chain_rooted_subtree`: now wrong in the Hamiltonian bundle collapsed path;
3. `star_rooted_subtree`: now wrong in the Hamiltonian bundle collapsed path;
4. `branched_rooted_subtree`: still wrong.

So at the moment the generic forward Hamiltonian path should be treated as
having a broader correctness regression than the earlier narrower
"branched-only" picture suggested.

The current benchmark recheck on

`env OMP_NUM_THREADS=1 build/src/benchmark_rooted_component_tree_pairs test_molecule/C6H6_full.xmi --max-pairs 5 --repeat 5 --edge-threshold 0`

is:

1. `one_leaf_star_generic_bundle.boundary_seconds_per_repeat = 0.0275436066`
2. `one_leaf_star_generic_bundle.boundary_speedup_vs_exact = 0.0205201231707978`

So this pass did **not** deliver a retained optimization. The failed fixed-key
merge experiment was reverted, the benchmark is still far below `1x`, and the
generic correctness baseline itself needs to be re-established before the next
true `2^m` frontier-merge rewrite can be trusted.

---

## 8.8 April 6 follow-up: the dense forward merge algebra was rebuilt exactly enough to restore the previous generic baseline

The next pass did not try another local fixed-key shortcut. Instead, it
replaced the broken dense-forward inner merge decomposition by a structurally
exact source-term merge for the carried Hamiltonian channels.

Concretely, inside
[`component_tree.cpp`](./component_tree.cpp):

1. the dense same-spin merge stopped assuming that `degree1 x degree1` always
   lands in `degree2`; it now enumerates all overlap / degree-1 / degree-2
   source terms and re-accumulates whatever exact `degree1` or `degree2`
   deleted sector the merge actually produces;
2. the dense mixed alpha-beta merge stopped approximating the typed algebra by
   a small set of overlap and cross-term cases; it now enumerates the exact
   overlap / alpha-first-order / beta-first-order / mixed-first-order source
   terms and applies the same merge law as the typed boundary payload;
3. the now-dead dense helper layer for the old incomplete overlap-only /
   `degree1 -> degree2` cross decomposition was deleted;
4. the production Hamiltonian bundle wrappers were switched back to the dense
   forward path by default, with the old typed internal kept only as a fallback
   under `XMVB_EXACT_SEPARATOR_FORCE_TYPED_HAMILTONIAN_PRODUCTION=1`.

This matters because the dense forward path is now back in the state we
actually needed before continuing the `2^m` work:

1. on the synthetic validator, `chain_rooted_subtree` is exact again;
2. `one_leaf_star_rooted_subtree` is exact again;
3. `star_rooted_subtree` is exact again;
4. `branched_rooted_subtree` is still wrong;
5. the dense forward recursive messages now match the typed boundary-message
   recurrence under
   `XMVB_DEBUG_EXACT_SEPARATOR_DENSE_HAMILTONIAN_COMPARE=1`, so the remaining
   `branched` mismatch is no longer a dense-only regression.

On

`env OMP_NUM_THREADS=1 XMVB_DEBUG_EXACT_SEPARATOR_DENSE_HAMILTONIAN_COMPARE=1 build/src/check_exact_separator_two_electron_component_tree`

the current default production path now reports the same old generic picture:

1. `chain_rooted_subtree`: exact to numerical noise;
2. `one_leaf_star_rooted_subtree`: exact to numerical noise;
3. `star_rooted_subtree`: exact to numerical noise;
4. `branched_rooted_subtree`: still wrong, with
   `one_electron_absolute_error = 0.925694537812465`,
   `same_spin_alpha_absolute_error = 0.1897598906242`,
   `same_spin_beta_absolute_error = 0.189759890624198`,
   and `opposite_spin_absolute_error = 46.9369153734999`.

That is a real structural recovery:

1. the dense forward generic path is no longer quarantined only as a debug
   curiosity;
2. the previously passing generic cases are restored on the new forward carrier;
3. the remaining `branched` failure is once again isolated as the shared old
   generic boundary-message issue.

On

`env OMP_NUM_THREADS=1 build/src/benchmark_rooted_component_tree_pairs test_molecule/C6H6_full.xmi --max-pairs 5 --repeat 5 --edge-threshold 0`

after restoring dense forward as the default production path, the current
one-leaf generic benchmark is:

1. `one_leaf_star_generic_bundle.boundary_seconds_per_repeat = 0.0242859112`
2. `one_leaf_star_generic_bundle.boundary_speedup_vs_exact = 0.0208399757304556`

So this pass did improve the absolute generic dense replay time versus the
earlier `0.0275s` quarantine baseline, but it still did **not** deliver the
`2^n -> 2^m` performance win. The path remains about fifty times slower than
exact on this hot case, because the inner carrier is still a dense
deleted-minor algebra rather than the final bundle-native `2^m` carrier.

---

## 9. Acceptance Criteria

The rewrite should now be considered successful only when all of the following
are true.

1. `one_leaf_star_generic_bundle` no longer spends its time building one full
   exact one-spin boundary message per state pair.
2. the generic forward subtree message carrier is no longer
   `HamiltonianBoundaryPayload`;
3. root close/readout works on the same dense Hamiltonian bundle carrier used
   by leaf and frontier messages;
4. `benchmark_rooted_component_tree_pairs test_molecule/C6H6_full.xmi`
   shows the one-leaf bundle replay closing the huge current gap to the exact
   baseline and eventually crossing speedup `> 1`;
5. the two-electron synthetic validator remains exact on the currently passing
   cases, and any remaining branched-tree mismatch is tracked as a separate
   generic correctness issue rather than hidden behind the performance work.

---

## 10. Bottom Line

After the current pass, the status is more precise than before:

1. the old one-leaf aggregate scaffold has already been replaced by a packed
   dense-bundle replay path;
2. that replay path is still slow because it is bundle-native only at the
   carrier level, not at the construction granularity;
3. the generic rooted-tree **forward** code is no longer sparse-payload based:
   it now runs through a forward-only dense per-entry carrier;
4. that new carrier is still a delta-aware dense deleted-minor basis, not yet
   the final bundle-native `2^m` carrier we actually want;
5. the dense forward closure is exact on the previously passing cases, but the
   old branched-tree generic correctness issue is still present;
6. the current forward dense rewrite still did not deliver the performance win:
   the default generic dense replay is now somewhat faster than the earlier
   quarantined dense baseline, but on the current one-leaf generic benchmark it
   is still far below `1x`;
7. the next real optimization step is therefore **not** another round of map
   cleanup or small-factor tuning, but replacing the current dense
   deleted-minor inner carrier by a true bundle-native `2^m` carrier whose
   merge / root-close kernels do not repeatedly decode and re-encode deleted
   labels.
