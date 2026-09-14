#pragma once

#include "vbscf/orbitals/preparation/input.hpp"

namespace xmvb::vb {

bool orbital_input_has_support_preserving_gauge_reference(
    const OrbitalPreparationInput& orbital_preparation_input);

/**
 * @brief Balances inactive occupied orbitals within a sparse reference layout.
 *
 * The transformation rotates only within the inactive occupied manifold. Each
 * transformed column is constrained to its target sparse support, so the
 * represented subspace and strict support are preserved to backward error.
 * Within those exact constraints, cyclic determinant maximization selects a
 * normalized representative that removes avoidable near-linear dependence.
 * Balancing starts only when the occupied metric has lost more than half of
 * floating-point precision or the current and target supports differ. Once
 * selected, the same gauge section is maintained at later accepted points.
 *
 * @param reference_layout Sparse orbital layout whose inactive supports define
 *        the preferred gauge.
 * @param orbital_preparation_input In-place target whose inactive occupied
 *        orbitals will be rotated inside their current span.
 * @return Whether the stored orbital representative changed.
 */
bool apply_support_preserving_inactive_gauge(
    const OrbitalPreparationInput& reference_layout,
    OrbitalPreparationInput* orbital_preparation_input);

/**
 * @brief Balances inactive orbitals using their preferred sparse support.
 *
 * Embedded pre-expansion support metadata is used when present; otherwise the
 * current strict support is the target. This overload is used both for the
 * initial point and for accepted-point canonicalization.
 *
 * @return Whether the stored orbital representative changed.
 */
bool apply_support_preserving_inactive_gauge(
    OrbitalPreparationInput* orbital_preparation_input);

}  // namespace xmvb::vb
