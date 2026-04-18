# Localized Selector State Implementation Plan

## 1. Scope

This plan translates the representative-transport derivations into a concrete
implementation route for the current code base.

The immediate target is intentionally narrow:

$$
\boxed{
\text{first implement the selector state and chart transport only for }
\texttt{orbtyp=oeo},
\text{ i.e. the current full-AO occupied chart.}
}
$$

This avoids mixing the inactive-orthogonal-frame work with the separate
problem of transporting representative changes through genuinely sparse
per-orbital support layouts.

## 2. Why `orbtyp=oeo` Is The Right First Target

The loader already records that `orbtyp=oeo` remains a full-AO chart:

- [`cpp_vb_input_loader.cpp`](/pool1/home/xiatao/project/xmvb-cpp/src/runtime/cpp_vb_input_loader.cpp)

That means:

1. every occupied orbital already has a full AO coefficient chart,
2. active representative resets do not need sparse-support truncation logic,
3. the Jacobian acts as dense column transforms on occupied blocks.

So the first exact implementation should be restricted to this regime.

## 3. New Accepted-Point State To Materialize

The present accepted-point cache already stores:

1. physical inactive orbitals $C_i$,
2. orthonormal inactive frame $Q_i$,
3. inactive gauge transform $R_i = (C_i^T S C_i)^{-1/2}$,
4. physical active orbitals $C_a$,
5. auxiliary active block $T_a$.

What is still missing is the **explicit selector state**

$$
\boxed{
C_i = Q_i U_i,
\qquad
C_a = T_a + Q_i K_a.
}
$$

The implementation should therefore add a new accepted-point cache object:

```cpp
struct LocalizedRepresentativeSelector {
  Eigen::MatrixXd inactive_right_transform;
  Eigen::MatrixXd inactive_inverse_transpose_right_transform;
  Eigen::MatrixXd active_inactive_coefficients;
};
```

with the meaning

$$
U_i = \texttt{inactive\_right\_transform},
\qquad
U_i^{-T} = \texttt{inactive\_inverse\_transpose\_right\_transform},
\qquad
K_a = \texttt{active\_inactive\_coefficients}.
$$

### Recommended storage location

The best home is:

- [`physical_orbital_frame.hpp`](/pool1/home/xiatao/project/xmvb-cpp/src/vb/pdft/physical_orbital_frame.hpp)

rather than `CppOrbitalGradientResult` directly, because the selector is part
of the accepted-point relation between the internal working frame and the
physical representative itself.

## 4. How To Build The Selector From Current Caches

At every accepted point, the selector can be built directly from the already
cached objects.

Let

$$
Q_i = \texttt{inactive\_orthonormal\_orbital\_matrix},
$$

$$
C_i = \texttt{inactive\_physical\_orbital\_matrix},
$$

$$
C_a = \texttt{active\_physical\_orbital\_matrix},
$$

$$
T_a = \texttt{auxiliary\_orbital\_matrix.middleCols(...)}.
$$

Then:

### 4.1 Inactive selector

Because $Q_i^T S Q_i = I$ and $C_i = Q_i U_i$,

$$
\boxed{
U_i = Q_i^T S C_i.
}
$$

This is equivalent to

$$
U_i = R_i^{-1},
$$

but the projection formula above is cleaner and self-checking.

### 4.2 Active inactive-null selector

Because

$$
C_a = T_a + Q_i K_a,
\qquad
Q_i^T S T_a = 0,
$$

we have

$$
\boxed{
K_a = Q_i^T S C_a.
}
$$

### 4.3 Validation identities

After constructing $(U_i, K_a)$, the preparer should optionally validate

$$
\|C_i - Q_i U_i\|_\infty,
\qquad
\|C_a - (T_a + Q_i K_a)\|_\infty
$$

against a tight accepted-point tolerance.

## 5. Files To Change In The First Coding Pass

The first implementation pass should touch only these files:

1. [`physical_orbital_frame.hpp`](/pool1/home/xiatao/project/xmvb-cpp/src/vb/pdft/physical_orbital_frame.hpp)
2. [`active_space_orbital_preparer.cpp`](/pool1/home/xiatao/project/xmvb-cpp/src/vb/orbital/active_space_orbital_preparer.cpp)
3. [`cpp_vb_scf_optimizer.cpp`](/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/cpp_vb_scf_optimizer.cpp)

No second-order kernels need to change in this pass.

## 6. Phase 1: Materialize Selector State Only

This phase is diagnostic and low risk.

### 6.1 `physical_orbital_frame.hpp`

Add the new selector struct and a member:

```cpp
LocalizedRepresentativeSelector localized_representative_selector;
```

### 6.2 `active_space_orbital_preparer.cpp`

After the existing physical and auxiliary frames are built:

1. compute $U_i = Q_i^T S C_i$,
2. compute $K_a = Q_i^T S C_a$,
3. cache $U_i^{-T}$,
4. store all three matrices in the selector state.

### 6.3 Expected result of Phase 1

After this phase, the accepted-point result should carry enough information to
express the localized physical representative as

$$
C_i = Q_i U_i,
\qquad
C_a = T_a + Q_i K_a
$$

without reconstructing those small matrices ad hoc inside the optimizer.

## 7. Phase 2: Upgrade The Current Active Representative Reset

The present OEO active reset already changes only the inactive-null component
of $C_a$. In the new notation, that means:

1. $Q_i$ stays fixed,
2. $T_a$ stays fixed,
3. only $K_a$ changes.

So the new selector state should be used as follows.

### 7.1 Old accepted-point selector

Read from the accepted-point cache:

$$
(U_i, K_a).
$$

### 7.2 Build the target selector

From the current $Q_i$, current $T_a$, and the stored localized reference
orbitals, build a target

$$
\hat K_a.
$$

For the first implementation, keep

$$
\hat U_i = U_i,
$$

because the present active reset does not change the inactive physical
representative.

### 7.3 Representative Jacobian

Then

$$
U_{\mathrm{rep}} = I,
\qquad
\Lambda_{\mathrm{rep}} = U_i^{-1} (\hat K_a - K_a).
$$

and the exact gradient transport becomes

$$
\boxed{
\hat G_{C_i} = G_{C_i} - G_{C_a} \Lambda_{\mathrm{rep}}^T,
\qquad
\hat G_{C_a} = G_{C_a}.
}
$$

This is the missing analytic step in the current active representative reset.

## 8. Where The Jacobian Must Be Applied In Phase 2

For the `orbtyp=oeo` first target, the Jacobian must be applied to:

1. `last_gradient_result_.sparse_orbital_energy_gradient`
2. `last_gradient_result_.sparse_orbital_reference_energy_gradient`
3. the current packed parameter vector after repacking

and later, after Phase 3, also to:

4. `PackedSecantPair::packed_step`
5. `PackedSecantPair::packed_projected_gradient_change`

The current code only does this transport for the inactive support-aware gauge
fix, not for the active representative reset.

## 9. Phase 3: Transport TN Secant History

Only after Phase 2 is correct should the transported TN history be upgraded.

The current TN preconditioner stores ambient packed secant pairs:

```cpp
struct PackedSecantPair {
  Eigen::VectorXd packed_step;
  Eigen::VectorXd packed_projected_gradient_change;
};
```

If the localized representative changes at an accepted point, then each stored
pair must be mapped by the same packed-chart Jacobian before it is reprojected
into the current nonredundant space.

For the first implementation target, because `orbtyp=oeo` is full-AO, this
Jacobian can be implemented first as a dense occupied-column transform on the
packed full-orbital chart.

## 10. Recommended Order Inside `cpp_vb_scf_optimizer.cpp`

Inside `canonicalize_orbital_chart_at_current_point(...)`, the order should be:

1. apply inactive support-aware gauge fix if needed,
2. update selector state if step 1 changed the inactive representative,
3. run active representative reset,
4. if the active selector changed, apply analytic gradient transport,
5. repack the parameter vector,
6. only then consider transported secant-history updates.

This keeps the accepted-point chart coherent before any packed chart is
reconstructed from the sparse input buffer.

## 11. What Should Explicitly Wait Until Later

The following should **not** be mixed into the first implementation pass:

1. sparse per-orbital support transport outside `orbtyp=oeo`,
2. general active-active chart transport inside $T_a$,
3. rewriting exact-core HVP formulas,
4. redesigning the nonredundant tangent basis.

Those are all larger tasks and should remain blocked until the selector-state
machinery is correct on the current OEO full-AO path.

## 12. Minimal Success Criteria

The first implementation should be considered successful if:

1. every accepted-point orbital result stores valid $(U_i, K_a)$,
2. the active representative reset updates that selector consistently,
3. the active reset no longer leaves `last_gradient_result_` in an
   untransported chart,
4. `orbtyp=oeo` Molden output remains localized without relying on a
   purely export-time repair.

Only after that should the code move on to transported secant-history updates
and then, later, to a true internal-$Q_i$ TNHVP implementation.
