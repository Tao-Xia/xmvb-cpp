# Exact Raw-VB Hamiltonian Separator Exploration Summary

## 1. Purpose

This note summarizes the current exploration status for the following exact
goal:

- keep the **raw-VB determinant definition** unchanged;
- reduce the amount of exact Hamiltonian work below the baseline

```math
N_{\mathrm{det\text{-}pair}}^{\mathrm{uniq}};
```

- ideally by a separator / graph-width / layered-recurrence reformulation,
  rather than by changing the basis into AGP / pairing states.

The discussion below is only about the **exact raw-VB Hamiltonian** route.
It is separate from the current `pfaffian_vbscf` pairing-state method.

Relevant files:

- `src/tools/analyze_star_separator_one_electron_dataset.cpp`
- `src/pfaffian_vbscf/raw_vb_exact_layered_separator_overlap.md`
- `src/pfaffian_vbscf/raw_vb_exact_open_state_hamiltonian.md`

---

## 2. Baseline Question

For one raw-VB structure pair `(X,Y)`, the exact overlap and one-electron
Hamiltonian are computed from the determinant expansion

```math
\Phi_X = \sum_{D_L \in \mathcal{D}(X)} c_X(D_L)\, D_L,
```

```math
\Phi_Y = \sum_{D_R \in \mathcal{D}(Y)} c_Y(D_R)\, D_R.
```

The baseline exact one-electron matrix element is

```math
H^{(1)}_{XY}
=
\sum_{D_L,D_R}
c_X(D_L)\,c_Y(D_R)\,
\left[
H^{(1)}_\alpha(D_L,D_R)\,S_\beta(D_L,D_R)
+
H^{(1)}_\beta(D_L,D_R)\,S_\alpha(D_L,D_R)
\right].
```

Therefore the only meaningful success criterion is:

> can we reduce the number of **unique exact determinant-pair-like subproblems**
> needed for the Hamiltonian, not just for overlap?

Any proposal that does not reduce that exact work is only a reorganization,
not an accelerator.

---

## 3. What Was Successfully Established

### 3.1 Exact overlap admits a useful separator formulation

For star-like component graphs, the exact overlap can be reorganized into:

- exact leaf message bundles;
- exact root-remainder minors;
- layered merge states over boundary masks.

This does not prove a polynomial algorithm, but it does show that **exact
overlap** can be rewritten in a width-parameterized way and may reduce the
number of exact subproblems relative to naive determinant-pair enumeration.

This is the content of
`src/pfaffian_vbscf/raw_vb_exact_layered_separator_overlap.md`.

### 3.2 The overlap recurrence does not automatically transfer to Hamiltonian

The one-electron Hamiltonian carries first-cofactor information:

```math
H^{(1)}_\sigma
=
\sum_{r,c} h_{rc}\,C_{rc}(S_\sigma).
```

Unlike overlap, this depends on deleted-row / deleted-column structure.
Therefore an overlap-style separator state is not enough by itself.

This was the main reason to move from overlap-only messages toward
open-state / cofactor-state diagnostics.

---

## 4. What Failed

### 4.1 Reusing overlap keys plus scalar one-electron payloads fails

The first failed idea was:

```text
exact overlap separator key
+ local scalar H_alpha / H_beta payload
```

This fails because the exact one-electron cofactor contribution contains
cross-region terms of the form

```math
h_{rc}\,R_A(r)\,C_B(c),
```

where the unresolved row label `r` and column label `c` must both survive
until the final merge. If a leaf contracts

```math
\sum_{r,c \in \mathrm{local}} h_{rc} C_{rc}
```

too early, it destroys the information needed for exact cross-region
reconstruction.

### 4.2 Adding a simple mixed-bridge correction also fails

The next attempt was to keep the overlap-style closed part and add a
"bridge" correction for missing mixed terms. This also failed.

The reason is structural: the missing information is not one extra scalar.
It is a higher-dimensional open-state object.

---

## 5. Sharp Diagnostic: The Known Failing Example

The most useful failure case so far is:

- molecule: `test_molecule/C6H6_full.xmi`
- structure pair: `left_structure = 1`, `right_structure = 0`

Command:

```bash
OMP_NUM_THREADS=1 build/src/analyze_star_separator_one_electron_dataset \
  test_molecule/C6H6_full.xmi \
  --left-structure 1 \
  --right-structure 0 \
  --max-pairs 1 \
  --top-examples 1
```

The key outputs are:

```text
exact_overlap = -5.823336145753037
star_overlap = -5.823336145753037
exact_one_electron = 57.07506518045701
collapsed_one_electron = 78.26760007609747
constructed_one_electron = 118.6008120893094
```

So:

- the exact overlap recurrence is correct;
- the current one-electron collapse is not correct.

For this same pair, the explicit one-leaf diagnostic gives:

```text
explicit_one_leaf_captured = 78.26760007609751
explicit_one_leaf_mixed_bridge = -8.901363456790838
explicit_one_leaf_open_state_prototype_residual_closed_total = 80.66642402642394
```

This established a very important fact:

> after removing the captured part and the exact cross part, the remaining
> error is a **residual closed correction**, not merely a missing explicit
> cross term.

---

## 6. What The Residual Looks Like

For the same failing example, the aggregated three-block diagnostic produced

```text
explicit_one_leaf_open_state_prototype_residual_closed_blocks =
[[~0, ~0, 0],
 [~0, 80.66642402642394, 0],
 [0, 0, ~0]]
```

and

```text
explicit_one_leaf_max_full_cross_message_to_root_rank = 1
explicit_one_leaf_max_full_cross_root_to_message_rank = 1
explicit_one_leaf_max_open_state_prototype_residual_message_rank = 2
explicit_one_leaf_max_open_state_prototype_residual_leaf_leaf_rank = 2
```

Interpretation:

- the missing part is not dominated by a large message-to-root cross block;
- after aggregation, it looks concentrated in the **leaf-leaf closed block**;
- however, this aggregated picture alone is not enough to define a reusable
  exact reduced state.

This distinction turned out to matter a lot.

---

## 7. Why The Earlier Scalar Fit Was Misleading

An apparently encouraging result was:

```text
explicit_one_leaf_scalar_mask_closed_correction_fit_total =
57.07506518045705
explicit_one_leaf_scalar_mask_closed_correction_fit_abs_error =
4.263256414560601e-14
explicit_one_leaf_scalar_mask_closed_correction_fit_equation_count = 4
explicit_one_leaf_scalar_mask_closed_correction_fit_unknown_count = 4
```

At first sight, this suggested that a small per-mask scalar correction might
exist.

But this is not strong evidence, because:

- there are only `4` root-pair equations;
- there are also `4` scalar unknowns;
- therefore an exact scalar fit can occur simply because the system is
  square, not because a local reduced-state construction truly exists.

In other words:

> "post-summation scalar fit exists" is much weaker than
> "pre-summation separator payload exists".

This was one of the main lessons from the exploration.

---

## 8. Stronger Test: Leaf-Level Residual Matrix Payload

To remove the ambiguity above, the diagnostic tool was extended to expose the
exact **leaf-level residual closed cofactor matrix** before contraction with
the one-electron operator.

That stronger test asks:

> can the exact residual leaf-leaf cofactor family be represented by a small
> mask-indexed basis at the separator level?

Two feature models were tested for the same failing pair.

### 8.1 Current factorized opposite-spin feature model

For one spin channel, use per-mask coefficients proportional to

```math
\alpha_{\mathrm{root}} \beta_{\mathrm{root}} \beta_{\mathrm{leaf}}
```

or its beta-channel analogue.

### 8.2 Fuller opposite-spin feature model

Use instead

```math
\alpha_{\mathrm{root}} \beta_{\mathrm{full}}
```

or its beta-channel analogue, so that the opposite-spin weight is no longer
restricted to the factorized leaf-root form.

### 8.3 Result

Both models fail at the matrix level on the same example:

```text
explicit_one_leaf_alpha_matrix_current_feature_fit_max_root_pair_frobenius_residual
  = 0.4729727010139993
explicit_one_leaf_beta_matrix_current_feature_fit_max_root_pair_frobenius_residual
  = 0.4729727010139968

explicit_one_leaf_alpha_matrix_full_feature_fit_max_root_pair_frobenius_residual
  = 0.4729727010140015
explicit_one_leaf_beta_matrix_full_feature_fit_max_root_pair_frobenius_residual
  = 0.4729727010139975
```

and

```text
explicit_one_leaf_alpha_matrix_mask_closed_correction_fit_unknown_count = 2
explicit_one_leaf_beta_matrix_mask_closed_correction_fit_unknown_count = 2
explicit_one_leaf_alpha_matrix_target_span_rank = 4
explicit_one_leaf_beta_matrix_target_span_rank = 4
```

This is the strongest current negative result.

It means:

- the exact residual matrix family is not representable by only `2`
  mask-basis payloads per spin;
- replacing factorized opposite-spin weights by fuller opposite-spin weights
  does not solve the problem;
- the current mask-only bundle state is too weak for exact Hamiltonian
  collapse.

---

## 9. Current Best Interpretation

The most defensible interpretation at this point is:

1. **Exact overlap compression is real**, at least in the separator/layered
   sense already validated.
2. **Exact Hamiltonian compression is harder** because overlap keys alone do
   not carry enough cofactor information.
3. The exact one-electron residual does appear to live in a space much
   smaller than the full determinant-pair space, but
4. that space is **not** captured by the current root-independent
   mask-only leaf-bundle basis.

So the present status is neither:

- "exact collapse works", nor
- "no exact compression is possible".

The honest statement is:

> the currently tested exact reduced state is insufficient, but the observed
> residual family is still much lower dimensional than the raw determinant-pair
> space, so a richer exact state algebra may still exist.

---

## 10. Where Exact Progress Still Seems Possible

If the exact route is to survive, it likely needs one of the following.

### 10.1 Root-aware bundle states

The successful scalar fit happened only after root-pair aggregation.
This suggests that the right exact state may depend on more than leaf masks.

In practice this means:

- the payload may have to remember some root-term information;
- the true reduced state may live at a **bundle-with-root-context** layer,
  not at a purely local leaf-message layer.

### 10.2 Open-state tensor families, not scalar corrections

The note
`src/pfaffian_vbscf/raw_vb_exact_open_state_hamiltonian.md`
already points in this direction:

- closed state `Z`;
- row-open state `R`;
- column-open state `C`;
- cofactor kernel `Q`.

The new diagnostics support that direction. The missing information is not
well described as one scalar residual. It behaves more like a structured
cofactor payload.

### 10.3 Richer exact separator algebra

The matrix-span result

```text
target_span_rank = 4
```

for this failing pair is small enough to remain interesting.

It does **not** prove that a reusable exact algorithm exists, but it does say:

- the missing family is not exploding immediately to the full determinant
  dimension;
- there may still be a nontrivial exact algebraic compression if the right
  basis is chosen.

---

## 11. Why Thresholding Keeps Reappearing

The graph-width idea only helps if the **effective interaction graph** is not
too close to fully connected.

For nonorthogonal SCF orbitals, the metric-aware graph can easily become
dense. In that regime:

- separator width grows;
- message families grow;
- exact graph-based compression becomes weak.

This is why thresholding or screening repeatedly appears as a practical
alternative.

However, there is a strict distinction:

### 11.1 Thresholding as an exact heuristic

Use a thresholded graph only to choose:

- ordering;
- candidate separators;
- merge plans.

Then still evaluate the original exact Hamiltonian. This does **not** lose
exactness, but it also does not by itself guarantee a strong reduction.

### 11.2 Thresholding as an approximate accelerator

Use a threshold to actually remove weak couplings from the effective graph.
Then perform the separator recurrence on the reduced graph.

This can plausibly yield real speedups, but it is now an **approximate**
method rather than an exact raw-VB algorithm.

This distinction matters and should not be blurred.

---

## 12. Current Bottom Line

The current exploration supports the following conclusions.

### 12.1 What is already established

- Exact raw-VB overlap admits a meaningful separator/layered reformulation.
- Exact raw-VB Hamiltonian does **not** follow from the overlap recurrence by
  adding scalar one-electron payloads.
- The current mask-only residual correction idea is not exact.

### 12.2 What is currently false

- It is false that the known one-electron failure can be fixed by a simple
  scalar closed correction.
- It is false that the current `2`-state-per-spin mask basis is sufficient
  for the exact leaf-level residual cofactor family.

### 12.3 What remains open

- whether a richer exact open-state / root-aware separator algebra can reduce
  exact Hamiltonian work below `N_{\mathrm{det\text{-}pair}}^{\mathrm{uniq}}`;
- whether that richer algebra would still yield a real algorithmic gain, or
  only a repackaging of the same exact work;
- whether a controlled approximate thresholded graph route is the more
  productive path for practical acceleration.

---

## 13. Recommended Next Steps

If the priority is still the **exact** route, the next work should be:

1. identify a richer exact residual basis than the current mask-only basis;
2. test whether that basis can be constructed **before** full determinant-pair
   summation;
3. measure whether it reduces true exact subproblem counts, not only fitted
   dimensions after aggregation.

If the priority shifts to **practical acceleration**, the next work should be:

1. define a screened / thresholded effective graph;
2. quantify the reduction in separator width and exact work;
3. monitor the induced error in overlap and Hamiltonian matrix elements.

Those are two different projects and should be kept conceptually separate.
