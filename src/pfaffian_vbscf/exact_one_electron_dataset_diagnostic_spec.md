# Exact One-Electron Dataset Diagnostic Specification

## 1. Purpose

This note turns the exact one-electron state-space idea into a concrete
dataset-level diagnostic program.

The purpose is not yet to implement the final exact recurrence.
The purpose is to answer one narrower but decisive question:

> across many realistic `6e6o full` structure pairs, is the exact local
> one-electron residual state space usually small enough to justify continued
> exact formula work?

If the answer is yes, then we continue pushing the exact route.
If the answer is no, then we should stop pretending that a small exact
state-space formula is likely to generalize.

---

## 2. Primary Dataset

Use the curated `6e6o full` pool:

- [data/training_xmi/6e6o_full](/pool1/home/xiatao/project/xmvb-cpp/data/training_xmi/6e6o_full)
- canonical sample list:
  [manifest.txt](/pool1/home/xiatao/project/xmvb-cpp/data/training_xmi/6e6o_full/manifest.txt)

This pool is preferable to benzene-only testing because:

- it contains many different `6e6o`, `str=full` systems;
- benzene is highly symmetric and may distort intuition;
- our question is about the typical behavior of the exact residual state
  dimension, not about one special molecule.

`C6H6_full` should still be kept as a known failure diagnostic, but it should
not be the only benchmark.

---

## 3. Diagnostic Unit

The natural analysis unit is:

```text
(molecule file, structure pair, root choice, decomposition)
```

More explicitly:

- one `.xmi` molecule file from `6e6o_full`
- one raw structure pair `(left_structure, right_structure)`
- one chosen root component
- one resulting component ordering / separator layout

This means we do **not** aggregate too early at the molecule level.
The first statistical object is the structure pair.

---

## 4. Scope For The First Pass

The first pass should remain narrow:

- singlet closed-shell only
- exact one-electron only
- star / near-star decompositions only
- no thresholding
- no two-electron terms
- no open-shell states

This is not because those topics are unimportant.
It is because the diagnostic must remain interpretable.

---

## 5. Existing Tool

The current main tool is:

- [analyze_star_separator_one_electron_dataset.cpp](/pool1/home/xiatao/project/xmvb-cpp/src/tools/analyze_star_separator_one_electron_dataset.cpp)

This tool already provides:

- exact overlap validation
- exact one-electron mismatch diagnostics
- width / root / star-graph information
- determinant-pair work counts
- explicit one-leaf residual-state diagnostics
- matrix-level residual span indicators on the known one-leaf cases

The current tool is already sufficient for a useful first dataset-level study,
even though its output is not yet ideal for machine aggregation.

---

## 6. Core Question To Measure

For one spin channel `\sigma` and one mask `\mu`, the key local object is

```math
K_{\rho,\mu}^{(\sigma)},
```

the exact leaf-level residual closed cofactor block under root context `\rho`.

The exact state-space question is:

```math
\mathcal{V}_{\mu,\sigma}
=
\mathrm{span}_{\rho}
\left\{
K_{\rho,\mu}^{(\sigma)}
\right\},
\qquad
m_{\mu,\sigma}
=
\dim \mathcal{V}_{\mu,\sigma}.
```

The dataset-level diagnostic must therefore estimate, directly or indirectly:

- whether `m_{\mu,\sigma}` is usually small;
- whether it correlates with graph width;
- whether it correlates with exact work reduction potential.

---

## 7. Required Output Columns

The following columns should be tracked for every analyzed structure pair.

### 7.1 Molecule metadata

- `input_file`
- `molecule_id`
  - initially this can just be the filename stem
- `n_active_orbitals`
  - expected to be `6` in this dataset
- `spin_multiplicity`

### 7.2 Structure-pair identity

- `left_structure`
- `right_structure`
- `node_count`
- `root_node`
- `width_upper_bound`
- `covered_star_pair`
  - whether the pair falls into the currently supported star-like exact path

### 7.3 Baseline exact work

- `reference_pair_count`
- `unique_reference_determinant_pair_count`
  - current tool reports this at run level, not per-example
- `reference_spin_over_unique_reference_spin_ratio`
  - run-level, useful for context

At minimum, the per-example field that must be retained is:

- `reference_pair_count`

### 7.4 Exact separator work proxies

- `local_term_pair_visits`
- `subdeterminant_evaluations`
- `dp_transitions`
- `unique_exact_separator_state_count`
  - currently run-level
- `unique_exact_leaf_message_state_count`
  - currently run-level
- `unique_exact_leaf_message_bundle_count`
  - currently run-level
- `unique_exact_merge_state_count`
  - currently run-level

These tell us whether any candidate exact reformulation is actually reducing
unique exact work.

### 7.5 Exactness checks

- `exact_overlap`
- `star_overlap`
- `abs_error`
- `exact_one_electron`
- `collapsed_one_electron`
- `one_electron_abs_error`
- `constructed_one_electron`
- `constructed_one_electron_abs_error`

These are basic sanity checks.
Any candidate exact state-space formula must preserve exactness.

### 7.6 Residual-state diagnostics

These are the most important fields for the present project.

- `explicit_one_leaf_captured`
- `explicit_one_leaf_mixed_bridge`
- `explicit_one_leaf_open_state_prototype_residual_closed_total`
- `explicit_one_leaf_max_open_state_prototype_residual_message_rank`
- `explicit_one_leaf_max_open_state_prototype_residual_leaf_leaf_rank`
- `explicit_one_leaf_alpha_matrix_target_span_rank`
- `explicit_one_leaf_beta_matrix_target_span_rank`
- `explicit_one_leaf_alpha_matrix_current_feature_fit_max_root_pair_frobenius_residual`
- `explicit_one_leaf_beta_matrix_current_feature_fit_max_root_pair_frobenius_residual`
- `explicit_one_leaf_alpha_matrix_full_feature_fit_max_root_pair_frobenius_residual`
- `explicit_one_leaf_beta_matrix_full_feature_fit_max_root_pair_frobenius_residual`

These fields are the current best proxies for:

- local exact state-space dimension
- adequacy of the current basis ansatz
- whether the residual looks structured enough to justify more formula work

---

## 8. Priority Statistics

For the first dataset pass, the most important aggregate statistics are:

### 8.1 State-space dimension statistics

- distribution of `explicit_one_leaf_alpha_matrix_target_span_rank`
- distribution of `explicit_one_leaf_beta_matrix_target_span_rank`
- joint distribution of `(width_upper_bound, target_span_rank)`

This is the primary answer to "is the exact one-electron state space usually
small?"

### 8.2 Basis-failure statistics

- distribution of matrix fit residuals under the current feature model
- distribution of matrix fit residuals under the fuller opposite-spin model

This tells us whether the present ansatz is almost right or fundamentally
wrong.

### 8.3 Exact-work statistics

- distribution of `reference_pair_count`
- distribution of `local_term_pair_visits`
- distribution of `subdeterminant_evaluations`
- ratios such as

```math
\frac{\text{local term visits}}{\text{reference pair count}}
```

and related exact-work proxies.

This is what prevents us from confusing state-space fitting with real
algorithmic improvement.

---

## 9. Benchmark Roles

The benchmark pool should be split into three roles.

### 9.1 Role A: failure-diagnostic molecules

These are molecules already known to expose specific exact one-electron
failures.

Current example:

- `C6H6_full`

Use these to quickly reject weak formulas.

### 9.2 Role B: diversity-screening molecules

Use many files from the `6e6o_full` manifest to estimate whether low exact
residual state-space dimension is common or rare.

The goal here is breadth, not deep study of one system.

### 9.3 Role C: representative benchmark subset

After Role B screening, choose a smaller subset with:

- low symmetry
- varied graph widths
- varied exact determinant-pair counts
- varied residual span ranks

This subset becomes the real exact one-electron benchmark suite.

---

## 10. Sampling Strategy

The sampling strategy should be staged.

### 10.1 Stage A: pilot scan

Select a modest number of molecules from the `6e6o_full` manifest and analyze:

- all star-covered structure pairs if cheap enough, or
- a capped random subset of structure pairs otherwise

The purpose is to see whether the residual state dimension distribution is
interesting at all.

### 10.2 Stage B: broad screening

Expand to a larger portion of the manifest and collect:

- structure-pair-level state-space rank data
- structure-pair-level exact-work data
- width correlations

The output of this stage should be a distribution over many pairs, not a few
hero examples.

### 10.3 Stage C: targeted deep diagnostics

Take the most informative pairs:

- smallest target span rank
- largest mismatch between scalar fit and matrix fit
- strongest width / rank outliers

and inspect them in detail when proposing new formulas.

---

## 11. Current Command Pattern

The current single-file diagnostic command is:

```bash
OMP_NUM_THREADS=1 build/src/analyze_star_separator_one_electron_dataset \
  data/training_xmi/6e6o_full/<FILE>.xmi \
  --pair-order lexicographic \
  --top-examples 12
```

For targeted inspection of one structure pair:

```bash
OMP_NUM_THREADS=1 build/src/analyze_star_separator_one_electron_dataset \
  data/training_xmi/6e6o_full/<FILE>.xmi \
  --left-structure I \
  --right-structure J \
  --max-pairs 1 \
  --top-examples 1
```

The current tool is text-oriented.
For a serious dataset pass, a machine-readable output mode such as CSV or
JSONL would be highly desirable.

---

## 12. What Should Count As Encouraging

The exact one-electron formula program should be considered encouraging if the
dataset shows:

- many structure pairs with small residual target span rank;
- a visible separation between residual state dimension and raw determinant
  pair count;
- evidence that the residual family dimension correlates more with graph width
  than with full determinant-pair scale;
- room for a small root-aware basis beyond the current mask-only basis.

This would justify continuing exact formula development.

---

## 13. What Should Count As Discouraging

The program should be considered discouraging if the dataset shows:

- residual target span rank grows rapidly toward raw determinant-pair scale;
- low-rank behavior appears only on highly symmetric molecules like benzene;
- the only successful fits remain post-summation scalar fits;
- any improved basis still fails to reduce exact work proxies.

This would mean we should stop over-investing in exact one-electron formulas.

---

## 14. Immediate Next Task

The next concrete step should be:

> run a pilot dataset scan over multiple molecules in
> `data/training_xmi/6e6o_full`, collect the structure-pair-level residual
> state-space diagnostics, and summarize the distribution of target span ranks
> and exact-work proxies.

That pilot scan will tell us whether continued exact formula exploration is
scientifically justified, or whether we are only overfitting one special
failure case.
