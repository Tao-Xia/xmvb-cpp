# Orthonormal Inactive Exact-Context Core Reduction

> **Retired design note.** The internal inactive chart described below was
> removed after failing the FeCl2 physical-coordinate HVP finite-difference
> check. Exact-CTX now acts only in the optimizer's physical (U_p) chart.

## 1. Goal

This note records the implementation-level conclusion behind the proposed
inactive-orthogonal speedup for the exact-context TNHVP path.

The main point is:

$$
\boxed{
\text{the real reduction is not an export-time representative repair.}
}
$$

The real reduction appears only if the accepted-point second-order operator
works in an internal chart

$$
(Q_i, T_a),
$$

with

$$
Q_i^T S Q_i = I,
\qquad
Q_i^T S T_a = 0,
$$

while the optimizer/output still reconstruct the localized physical occupied
representative

$$
C_i = Q_i U_i,
\qquad
C_a = T_a + Q_i K_a.
$$

That separation preserves the VB orbital picture while removing the mixed-gauge
inactive metric chain from the HVP core.

## 2. What Has Already Been Materialized In Code

The current accepted-point cache already stores the pieces needed for this
split:

1. the physical inactive block `C_i`,
2. the orthonormal inactive frame `Q_i`,
3. the physical active block `C_a`,
4. the projected active block `T_a`,
5. the selector matrices `(U_i, U_i^{-T}, K_a)`.

Concretely:

- [`physical_orbital_frame.hpp`](/pool1/home/xiatao/project/xmvb-cpp/src/vb/orbital/physical_orbital_frame.hpp)
- [`localized_representative_selector.cpp`](/pool1/home/xiatao/project/xmvb-cpp/src/vb/orbital/localized_representative_selector.cpp)
- [`active_space_orbital_preparer.cpp`](/pool1/home/xiatao/project/xmvb-cpp/src/vb/orbital/active_space_orbital_preparer.cpp)
- [`cpp_vb_scf_optimizer.cpp`](/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/cpp_vb_scf_optimizer.cpp)

So the representative transport machinery is no longer the blocker.

## 3. What The Core Simplification Actually Is

In the mixed gauge the inactive projector is

$$
P_i
=
C_i
\left(C_i^T S C_i\right)^{-1}
C_i^T,
$$

so the HVP direction must differentiate

$$
M^{-1},
\qquad
M = C_i^T S C_i,
$$

and therefore also

$$
\delta M,
\qquad
\delta M^{-1},
$$

throughout the pullback and fixed-upstream derivative.

If the internal working variable is changed to

$$
Q_i = C_i \left(C_i^T S C_i\right)^{-1/2},
$$

then at the accepted point

$$
P_i = Q_i Q_i^T,
\qquad
R_i = I - Q_i Q_i^T S,
\qquad
T_a = R_i C_a.
$$

The first directional derivatives become

$$
\delta P_i = \delta Q_i Q_i^T + Q_i \delta Q_i^T,
$$

and

$$
\delta T_a
=
R_i \, \delta C_a
- \delta Q_i \left(Q_i^T S C_a\right)
- Q_i \left(\delta Q_i^T S C_a\right).
$$

Because the accepted-point chart enforces

$$
Q_i^T S C_a = Q_i^T S T_a = 0,
$$

the first inactive-active coupling term vanishes:

$$
\delta T_a
=
R_i \, \delta C_a
- Q_i \left(\delta Q_i^T S C_a\right).
$$

That is the first real contraction reduction: one inactive-rank multiply is
removed from every directional projector action.

## 4. Where The Existing Code Already Reflects This Formula

The exact-context second-order operator already contains orthonormal-inactive
special cases:

- [`exact_orbital_second_order_operator.cpp`](/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/exact_orbital_second_order_operator.cpp)

The relevant helpers are:

1. `build_inactive_auxiliary(...)`
2. `apply_occupied_projector_to_orbitals(...)`
3. `apply_inactive_density_direction_to_basis_overlap_times_active(...)`
4. `build_delta_inactive_auxiliary_direction(...)`
5. `build_inactive_overlap_gradient(...)`
6. `build_inactive_overlap_gradient_direction(...)`
7. `build_identity_metric_inactive_projector_pullback_gradient(...)`
8. `build_identity_metric_inactive_projector_pullback_gradient_direction(...)`

Those branches already encode the intended simplification:

1. `A_i = Q_i` instead of `C_i M^{-1}`,
2. `\delta A_i = \delta Q_i` instead of
   `\delta C_i M^{-1} + C_i \delta M^{-1}`,
3. the inactive-overlap gradient chain vanishes because `M = I`,
4. the fixed-upstream pullback reduces to the identity-metric formula.

## 5. Why The Full Speedup Is Still Not Active

The current exact-context operator decides whether it can use the orthonormal
inactive chart by checking whether the accepted inactive overlap is already the
identity:

- [`exact_orbital_second_order_operator.cpp`](/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/exact_orbital_second_order_operator.cpp)

In other words, the optimized branch is presently a **dormant branch** for the
general OEO accepted point, because the optimizer still stores and varies the
raw physical inactive coefficients `C_i`, not the internal orthonormal chart
coordinates `Q_i`.

So the present code state is:

1. the formulas for the orthonormal inactive HVP core already exist in the
   operator,
2. the localized representative transport `(U_i, K_a)` already exists in the
   optimizer/cache,
3. but the accepted-point orbital chart still reaches exact-context as raw
   `C_i`, so the operator usually falls back to the mixed-gauge path.

## 6. The Actual Next Implementation Step

The next useful implementation is therefore not another export repair and not a
new active representative trick.

The actual next step is:

$$
\boxed{
\text{make exact\_ctx consume an internal } Q_i \text{ inactive chart while
the external optimizer/output continue to use } (C_i, C_a).
}
$$

That requires three coordinated changes.

### 6.1 Accepted-point exact-context chart state

At the accepted point, exact-context should treat

$$
Q_i
$$

as the inactive working block and

$$
T_a
$$

as the projected active block.

The physical representative remains available through

$$
C_i = Q_i U_i,
\qquad
C_a = T_a + Q_i K_a.
$$

### 6.2 Direction lift

The TNHVP direction must be lifted into

$$
(\delta Q_i, \delta C_a)
$$

or equivalently

$$
(\delta Q_i, \delta K_a)
$$

instead of the current raw

$$
(\delta C_i, \delta C_a)
$$

chart.

Without that lift change, the operator still has to rebuild the mixed-gauge
chain rule even if the accepted-point cache already stores `Q_i`.

### 6.3 Pullback scatter

After the exact-context HVP acts in the internal chart, the result must be
scattered back through the selector Jacobian to the external packed orbital
coordinates. That is conceptually the same representative-transport problem
already solved for accepted-point gradients and secant history, but now it must
be applied to the full HVP action.

## 7. Complexity Consequence

Relative to the current mixed-gauge exact-context path, the orthonormal
inactive internal chart removes the repeated need to build or differentiate

$$
M^{-1},
\qquad
\delta M^{-1},
\qquad
\delta\!\left(G_M\right),
$$

where all of those terms are small in dimension but occur in multiple places in
every HVP.

At the level of dominant inactive-block contractions, the change is:

1. mixed gauge:
   repeated
   $$
   O(N n_i^2) + O(n_i^3)
   $$
   work for inactive inverse / inverse-direction chains,
2. orthonormal inactive chart:
   those `n_i^3` inverse-update chains disappear from the core,
   and the active projector direction drops one inactive-rank multiply because
   `Q_i^T S T_a = 0`.

This does not change the determinant/RDM-side scaling. It only compresses the
orbital-response side of each HVP. So the speedup is meaningful exactly when
the current wall time is still visibly dominated by the orbital-response core.

## 8. Scope Boundary For Sparse Orbitals

For `orbtyp=oeo`, every occupied orbital already lives on a full-AO chart, so
the internal `Q_i` chart is compatible with the external storage.

For genuinely sparse per-orbital support charts, an inactive orthonormal block
mixes orbitals within the block and therefore no longer preserves the original
per-orbital sparse slot pattern. So the same idea still works mathematically,
but it becomes a new block-local chart rather than a trivial reinterpretation
of the old sparse slots.

Therefore the correct implementation order remains:

1. first make the `orbtyp=oeo` exact-context path truly use the internal
   orthonormal inactive chart,
2. only later generalize the same idea to sparse block-local charts.

## 9. Cleanup Status

The older export-time OEO representative repair path in

- [`cpp_vb_scf_optimizer.cpp`](/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/cpp_vb_scf_optimizer.cpp)

was only a downstream presentation repair. It did not activate the real
inactive-orthogonal HVP reduction, and once accepted-point canonicalization was
added it became redundant. That export-only path has therefore been removed so
the code now reflects the actual optimization strategy more cleanly.
