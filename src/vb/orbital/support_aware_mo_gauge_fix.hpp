#pragma once

#include <vector>

#include "vb/orbital/orbital_preparation_input.hpp"

namespace xmvb::vb {

struct SupportAwareInactiveMoGaugeTransform {
  int n_inactive_orbitals = 0;
  bool chart_changed = false;
  std::vector<double> right_transform;
  std::vector<double> inverse_transpose_right_transform;
};

bool orbital_input_has_support_aware_mo_gauge_reference(
    const OrbitalPreparationInput& orbital_preparation_input);

/**
 * @brief Re-gauges inactive occupied GUESS=MO orbitals toward a sparse reference layout.
 *
 * This transformation only rotates the inactive occupied manifold internally.
 * It is applied only when the current sparse chart already differs from the
 * recorded legacy support layout; in that case the represented occupied
 * subspace is preserved exactly while the inactive gauge is steered toward the
 * original sparse supports. When the current chart already equals the legacy
 * sparse layout, the function returns a no-op instead of performing an
 * inexact sparse truncation.
 *
 * @param reference_layout Sparse orbital layout whose inactive supports define
 *        the preferred gauge.
 * @param orbital_preparation_input In-place target whose inactive occupied
 *        orbitals will be rotated inside their current span.
 */
SupportAwareInactiveMoGaugeTransform apply_support_aware_inactive_mo_gauge_fix(
    const OrbitalPreparationInput& reference_layout,
    OrbitalPreparationInput* orbital_preparation_input);

/**
 * @brief Re-gauges inactive occupied orbitals using embedded MO-gauge support metadata.
 *
 * This overload is intended for accepted-point canonicalization inside the
 * optimizer. It is a no-op unless the input carries the pre-expansion sparse
 * support metadata recorded for `GUESS=MO`.
 */
SupportAwareInactiveMoGaugeTransform apply_support_aware_inactive_mo_gauge_fix(
    OrbitalPreparationInput* orbital_preparation_input);

/**
 * @brief Applies the inactive gauge transform to a sparse full-gradient vector.
 *
 * The accepted-point chart reset rotates the inactive occupied orbitals by
 * `C <- C T`. The corresponding sparse orbital gradient transforms as
 * `G <- G T^{-T}` inside the inactive occupied block, while all other orbital
 * components remain unchanged.
 */
void transform_sparse_inactive_orbital_gradient(
    const SupportAwareInactiveMoGaugeTransform& transform,
    const OrbitalPreparationInput& orbital_preparation_input,
    std::vector<double>* sparse_orbital_gradient);

}  // namespace xmvb::vb
