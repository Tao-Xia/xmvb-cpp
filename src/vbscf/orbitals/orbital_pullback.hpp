#pragma once

#include <Eigen/Core>

#include <vector>

#include "vbscf/orbitals/orbital_pullback_result.hpp"
#include "vbscf/orbitals/orbital_preparation_input.hpp"
#include "vbscf/orbitals/orbital_preparation_result.hpp"

namespace xmvb::vb {

struct ActiveSpaceOrbitalBackpropagationDiagnostics {
  Eigen::MatrixXd original_orbital_gradient;
  std::vector<double> orbital_value_gradient;
};

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
 * Full-AO OEO and support-constrained orbitals both use raw coefficients as
 * optimization variables. The normalization pullback is required for both;
 * a gradient on normalized coefficients is not interchangeable with this one.
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
   * @brief Returns the cached dense pullback stage and the final raw-slot gradient.
   *
   * This is a diagnostics hook for exact-context HVP validation. It exposes
   * the intermediate dense orbital gradient before the per-orbital
   * normalization pullback so callers can compare stage A/B of the orbital
   * backpropagation chain separately.
   */
  ActiveSpaceOrbitalBackpropagationDiagnostics compute_diagnostics(
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
