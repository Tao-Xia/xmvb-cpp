# Strict `orbtyp=oeo` Active-Representative Reset

## 1. Goal

This note records the strict plan for the `orbtyp=oeo` active-orbital bug.

The target is:

1. the optimized active occupied orbitals must stay on a localized
   representative,
2. the total energy must remain exactly unchanged by the representative reset,
3. the reset must be treated as a chart change inside the optimizer,
4. the solution must not rely on "accepted step -> repair orbitals ->
   recompute energy/gradient".

So this note is explicitly **not** about an export-only Molden fix, and it is
also **not** about a safe-but-expensive reevaluation after each accepted step.

## 2. Problem Diagnosis

For `TiCl-lbfgs.molden` versus `xmvb-TiCl.molden`, the observed issue is not
primarily that the occupied-active subspace is wrong. The main issue is that
the active occupied orbitals drift to a different internal basis inside almost
the same active subspace.

The practical evidence is:

1. the active-auxiliary subspace singular values are all very close to `1`,
2. several active physical orbitals have visibly worse one-by-one overlap,
3. once the current active auxiliary block is optimally rotated inside its own
   subspace, the per-orbital overlaps become close to `1` again.

So the failure mode is:

$$
\text{same or nearly same active subspace}
\quad+\quad
\text{different active-active representative}.
$$

This is why the final orbitals can look more delocalized even when the energy
is close to the legacy result.

## 3. Current Code State

The relevant current code paths are:

1. `src/vb/scf/cpp_vb_scf_optimizer.cpp`
2. `src/vb/orbital/support_aware_mo_gauge_fix.cpp`
3. `src/vb/orbital/active_space_orbital_preparer.cpp`
4. `src/runtime/cpp_vb_input_loader.cpp`
5. `src/tools/compare_molden_active_auxiliary.cpp`

The current behavior is:

1. plain `lbfgspp` uses the full packed sparse-orbital chart,
2. it does not use `NonredundantOrbitalSpace`,
3. accepted-point chart reset exists only for the inactive MO gauge,
4. active representative repair currently exists only at final export time.

Therefore the optimizer is currently allowed to accumulate active-active drift
during the accepted iterations, and the final export repair is too late to
prevent that drift from polluting the optimization chart.

## 4. Mathematical Setup

Let:

- $S \in \mathbb{R}^{N \times N}$ be the AO overlap matrix,
- $C_i \in \mathbb{R}^{N \times n_i}$ be the inactive occupied physical block,
- $C_a \in \mathbb{R}^{N \times n_a}$ be the active occupied physical block.

Define the inactive projector in the current code chart:

$$
P_i = C_i \left(C_i^T S C_i\right)^{-1} C_i^T.
$$

The active auxiliary block is

$$
T_a = (I - P_i S) C_a.
$$

The observed drift is approximately

$$
C_a' \approx C_a U,
\qquad
T_a' \approx T_a U,
$$

for some invertible active-space mixing matrix

$$
U \in GL(n_a).
$$

The key point is that the current and legacy optimized states can agree on the
active subspace while disagreeing on the internal basis chosen inside that
subspace.

## 5. What A Strict Solution Actually Requires

The strict solution is not "repair the final orbitals". The strict solution is
to define a localized active representative as part of the optimizer chart.

In abstract form, after each accepted point we want an exact chart reset

$$
(x, \Psi, g, H^{-1}_{\mathrm{qn}})
\;\mapsto\;
(\hat{x}, \hat{\Psi}, \hat{g}, \hat{H}^{-1}_{\mathrm{qn}})
$$

such that:

1. $\hat{x}$ corresponds to a uniquely chosen localized active representative,
2. $\hat{\Psi}$ represents exactly the same physical VB wavefunction as
   $\Psi$,
3. the energy is unchanged exactly,
4. the gradient and quasi-Newton state are transformed analytically instead of
   being recomputed from scratch.

In other words, the active reset must be a true change of coordinates on the
same physical variational point.

## 6. Why Export-Time Repair Is Not Enough

Suppose we wait until the end and only replace the exported active orbitals by
another representative.

That does not solve the real problem, because:

1. the optimizer history was already accumulated in the delocalized chart,
2. the line search accepted points in the delocalized chart,
3. the L-BFGS secant pairs were built in the delocalized chart,
4. any active-active drift already affected the subsequent search directions.

So export-time repair can improve the appearance of Molden output, but it
cannot enforce localized active orbitals during optimization.

## 7. Why "Repair Then Reevaluate" Is Rejected Here

One safe implementation would be:

1. accept a step,
2. reset the active representative,
3. recompute energy and gradient at the repaired point,
4. continue optimization.

That approach is numerically safe, but it increases cost by adding an extra
full evaluation after accepted steps.

This note records the stricter target instead:

$$
\text{repair} = \text{exact chart change},
$$

not

$$
\text{repair} = \text{new physical evaluation point}.
$$

## 8. Exact Covariance Requirement

If the active physical block is changed by

$$
C_a \mapsto \hat{C}_a = C_a U,
$$

then the active-space tensors change as

$$
\hat{A} = U^T A U,
$$

$$
\hat{h} = U^T h U,
$$

and the active two-electron tensor changes as

$$
\hat{g}_{abcd}
=
\sum_{\alpha \beta \gamma \delta}
U_{\alpha a}
U_{\beta b}
U_{\gamma c}
U_{\delta d}
g_{\alpha \beta \gamma \delta}.
$$

Therefore an exact active representative reset is only possible if the current
VB ansatz is covariant under this active-basis transform. That means there must
exist an induced wavefunction-parameter transform

$$
\hat{c} = \Gamma(U)^{-1} c
$$

or an equivalent convention, such that the physical wavefunction is unchanged.

This is the core mathematical condition.

If this covariance fails for the current selected VB structure space, then a
generic active-active basis change is not a pure gauge transformation, and an
exact active chart reset is impossible.

## 9. Consequences Of The Covariance Question

There are three logically different cases.

### 9.1 Full active-basis covariance holds

If the current ansatz is covariant under a general invertible

$$
U \in GL(n_a),
$$

then a strict active representative reset is possible.

In that case the accepted-point reset can be implemented as a true chart
change, provided we derive:

1. the orbital-coordinate transform,
2. the structure-coefficient transform,
3. the gradient transform,
4. the quasi-Newton history transform.

### 9.2 Only a subgroup is exact

It may turn out that only a smaller subgroup is exact, for example:

1. permutations of active orbitals,
2. independent sign flips,
3. some restricted blockwise subgroup.

Then the strict active representative reset can use only that subgroup. In
that case the localization freedom is more limited, but the energy can still be
preserved exactly.

### 9.3 No sufficiently rich exact subgroup is available

If no useful exact subgroup exists, then "keep active orbitals localized while
preserving the exact energy" cannot be achieved by an accepted-point chart
reset alone.

In that case the localization condition has to be enforced directly in the
optimization variables, not imposed afterwards as if it were a pure gauge.

## 10. Accepted-Point Transport Requirement

Suppose an exact chart reset is available and, locally, the packed orbital
parameters transform with Jacobian

$$
J = \frac{\partial \hat{x}}{\partial x}.
$$

Then the first-order gradient transport is

$$
\hat{g} = J^{-T} g.
$$

For a strictly linear chart change, the natural secant-pair transport would be

$$
\hat{s} = J s,
\qquad
\hat{y} = J^{-T} y.
$$

However, if the reset depends nonlinearly on the current point, then cross-chart
transport of old L-BFGS history needs a separate derivation. This has to be
worked out before implementation; it should not be guessed.

## 11. Representative Selection Map

The strict active reset needs a deterministic map

$$
U_{\mathrm{loc}} = U_{\mathrm{loc}}(C_i, C_a; \mathcal{R}),
$$

where $\mathcal{R}$ is the reference data used to define the desired localized
representative.

Possible choices for $\mathcal{R}$ are:

1. the original `$orb` support pattern,
2. the legacy `readguess` support pattern,
3. a separate localization functional,
4. a blockwise target basis from the initial accepted point.

This choice matters because it determines whether the representative is merely
"stable" or genuinely "legacy localized".

## 12. Metadata Gap In The Current Loader

The current loader stores support-aware MO gauge metadata only for the inactive
repair path attached to `GUESS=MO`.

This is not yet enough for a strict active representative reset based on the
original active support layout.

In particular:

1. `mo_gauge_reference_orbital_basis_counts` and
   `mo_gauge_reference_orbital_basis_index_table` currently serve the inactive
   MO-gauge path,
2. `original_orbital_basis_counts` stores counts only,
3. counts alone are not enough to reconstruct the original active support
   index sets.

So if the final strict solution needs original active support data, the loader
must preserve explicit support indices for the active orbitals as well.

## 13. Planned Implementation Route If Exact Covariance Holds

If the covariance derivation succeeds, the implementation route should be:

1. preserve active support-reference metadata in the input object,
2. build a deterministic localized representative map for the active block,
3. add accepted-point active chart reset to plain `lbfgspp`,
4. transport gradient and quasi-Newton state analytically,
5. remove the need for export-only active representative repair.

This keeps the optimizer itself in the localized chart instead of producing a
delocalized optimization history and patching the output afterwards.

## 14. Immediate Questions To Resolve Before Coding

The unresolved questions are:

1. Is the current selected VB ansatz exactly covariant under general
   active-basis transforms?
2. If not, what is the largest exact subgroup?
3. What induced transform $\Gamma(U)$ acts on the VB structure coefficients or
   other wavefunction parameters?
4. How should old quasi-Newton secant pairs be transported across the active
   chart reset?
5. What reference data should define the localized active representative?

Until these questions are answered, the strict active representative reset
should be treated as a derivation task, not as a coding shortcut.

## 15. Short Summary

The current `TiCl` mismatch is best understood as active-representative drift,
not mainly as a wrong active subspace.

The strict target is:

$$
\text{localized active orbitals}
\;+\;
\text{exactly unchanged energy}
\;+\;
\text{no extra full reevaluation after accepted steps}.
$$

That target is achievable only if the active representative reset can be
derived as an exact chart change of the current VB ansatz. The next task is
therefore to prove the active-basis covariance structure, not to add another
post-processing repair.
