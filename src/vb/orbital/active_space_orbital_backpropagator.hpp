#pragma once

#include <vector>

#include "vb/orbital/active_space_orbital_backpropagation_result.hpp"
#include "vb/orbital/orbital_preparation_input.hpp"

namespace xmvb::vb {

/**
 * @brief Reverse-mode backpropagator for `ActiveSpaceOrbitalPreparer`.
 *
 * The current implementation supports the no-inactive-orbital case, which is
 * the path exercised by the current C++ VBSCF real-case regression.
 * It propagates gradients from:
 * - the active auxiliary orbital matrix
 * - the active-orbital overlap matrix `SSO`
 *
 * back to the sparse orbital coefficient table.
 */
class ActiveSpaceOrbitalBackpropagator {
public:
  /**
   * @brief Backpropagates orbital-preparation adjoints.
   *
   * @param auxiliary_orbital_gradient Column-major gradient with respect to the full auxiliary orbital matrix.
   * @param active_orbital_overlap_gradient Column-major gradient with respect to `SSO`.
   * @param inactive_density_gradient Column-major gradient with respect to the inactive density matrix `P11`.
   * @param input Orbital preparation input.
   * @return ActiveSpaceOrbitalBackpropagationResult Gradient with respect to `orbital_value_table`.
   */
  ActiveSpaceOrbitalBackpropagationResult backpropagate(
      const std::vector<double>& auxiliary_orbital_gradient,
      const std::vector<double>& active_orbital_overlap_gradient,
      const std::vector<double>& inactive_density_gradient,
      const OrbitalPreparationInput& input) const;
};

}  // namespace xmvb::vb
