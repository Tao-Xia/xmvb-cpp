#pragma once

#include <Eigen/Core>

#include <vector>

#include "vb/orbital/active_space_orbital_backpropagation_result.hpp"
#include "vb/orbital/orbital_preparation_input.hpp"
#include "vb/orbital/orbital_preparation_result.hpp"

namespace xmvb::vb {

/**
 * @brief Reverse-mode backpropagator for `ActiveSpaceOrbitalPreparer`.
 *
 * This pullback consumes the active auxiliary block `dE / dT_active` together
 * with the inactive-density adjoint `dE / dP11` and returns the gradient with
 * respect to the sparse raw orbital coefficients in
 * `OrbitalPreparationInput::orbital_value_table`.
 *
 * The implementation reuses the cached mixed-gauge objects from
 * `OrbitalPreparationResult`: inactive duals `A3`, occupied projector `A1`,
 * inactive-active overlap block `A2`, projected active overlap block `A4`, and
 * the normalized physical orbital frame. This avoids rebuilding the inactive
 * density, occupied projector, and inactive-overlap inverse inside every
 * orbital pullback.
 *
 * The legacy `orbtyp=oeo` path is special. Historical XMVB normalizes the
 * physical orbitals inside `Orbprep` and then differentiates on that
 * normalized coefficient chart. For parity with the legacy open-shell OEO
 * representative, this backpropagator preserves that chart instead of pushing
 * the per-orbital normalization map all the way back to the raw coefficient
 * slots.
 */
class ActiveSpaceOrbitalBackpropagator {
public:
  /**
   * @brief Backpropagates orbital-preparation adjoints from matrix views.
   *
   * The `active_auxiliary_gradient` input is the adjoint of the active
   * auxiliary block after all `SSO`, `HHO`, and `GGO` contributions have
   * already been accumulated into it by the upstream backpropagators.
   *
   * @param active_auxiliary_gradient Column-major `n_basis x n_active` gradient with respect to `T_active`.
   * @param inactive_density_gradient Column-major `n_basis x n_basis` gradient with respect to `P11`.
   * @param input Orbital preparation input.
   * @param orbital_preparation_result Cached preparer result for the same orbital point.
   * @return ActiveSpaceOrbitalBackpropagationResult Gradient with respect to `orbital_value_table`.
   */
  ActiveSpaceOrbitalBackpropagationResult backpropagate(
      const Eigen::Ref<const Eigen::MatrixXd>& active_auxiliary_gradient,
      const Eigen::Ref<const Eigen::MatrixXd>& inactive_density_gradient,
      const OrbitalPreparationInput& input,
      const OrbitalPreparationResult& orbital_preparation_result) const;

  /**
   * @brief Backpropagates orbital-preparation adjoints from flat column-major buffers.
   *
   * @param active_auxiliary_gradient Column-major `n_basis x n_active` gradient with respect to `T_active`.
   * @param inactive_density_gradient Column-major `n_basis x n_basis` gradient with respect to `P11`.
   * @param input Orbital preparation input.
   * @param orbital_preparation_result Cached preparer result for the same orbital point.
   */
  ActiveSpaceOrbitalBackpropagationResult backpropagate(
      const std::vector<double>& active_auxiliary_gradient,
      const std::vector<double>& inactive_density_gradient,
      const OrbitalPreparationInput& input,
      const OrbitalPreparationResult& orbital_preparation_result) const;
};

}  // namespace xmvb::vb
