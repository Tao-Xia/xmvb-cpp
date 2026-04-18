#pragma once

#include <Eigen/Core>

#include <vector>

#include "runtime/libcint_ri_integral_provider.hpp"
#include "vb/orbital/ao_integral_input.hpp"
#include "vb/orbital/ao_effective_one_electron_backpropagation_result.hpp"
#include "vb/orbital/orbital_preparation_result.hpp"

namespace xmvb::vb {

/**
 * @brief Reverse-mode backpropagator for `AoEffectiveOneElectronBuilder`.
 *
 * This class propagates the gradient of the AO effective one-electron matrix
 * `F11` back to the inactive density matrix `P11`.
 */
class AoEffectiveOneElectronBackpropagator {
public:
  /**
   * @brief Backpropagates the AO effective one-electron adjoint.
   *
   * @param ao_effective_one_electron_gradient Column-major gradient with respect to `F11`.
   * @param ao_two_electron_integral_values Sparse AO two-electron integral values.
   * @param ao_two_electron_integral_indices Flattened AO two-electron index table.
   * @param n_basis_functions Number of AO basis functions.
   * @return AoEffectiveOneElectronBackpropagationResult Reverse-mode derivative.
   */
  AoEffectiveOneElectronBackpropagationResult backpropagate(
      const std::vector<double>& ao_effective_one_electron_gradient,
      const std::vector<double>& ao_two_electron_integral_values,
      const std::vector<int>& ao_two_electron_integral_indices,
      int n_basis_functions) const;

  AoEffectiveOneElectronBackpropagationResult backpropagate(
      const Eigen::Ref<const Eigen::MatrixXd>& ao_effective_one_electron_gradient,
      const std::vector<double>& ao_two_electron_integral_values,
      const std::vector<int>& ao_two_electron_integral_indices,
      int n_basis_functions) const;

  /**
   * @brief Backpropagates through the AO-side RI inactive-density contraction.
   */
  AoEffectiveOneElectronBackpropagationResult backpropagate(
      const std::vector<double>& ao_effective_one_electron_gradient,
      const LibcintRiIntegralProviderResult& ri_integral_provider_result,
      int n_basis_functions) const;

  AoEffectiveOneElectronBackpropagationResult backpropagate(
      const Eigen::Ref<const Eigen::MatrixXd>& ao_effective_one_electron_gradient,
      const LibcintRiIntegralProviderResult& ri_integral_provider_result,
      int n_basis_functions) const;

  /**
   * @brief Backpropagates the RI AO-H1E path from explicit orbital-space factors.
   *
   * The RI orbital gradient depends on the symmetric AO input
   *
   * `2 * P11 + T_active * (dE/dHHO + dE/dHHO^T) * T_active^T`.
   *
   * Both terms are naturally low-rank in orbital space, so this overload builds
   * the signed AO factors directly and bypasses the full-AO eigendecomposition
   * used by the generic matrix-based RI backpropagator.
   */
  AoEffectiveOneElectronBackpropagationResult backpropagate(
      const std::vector<double>& active_one_electron_gradient,
      const OrbitalPreparationResult& orbital_result,
      const LibcintRiIntegralProviderResult& ri_integral_provider_result,
      int n_basis_functions,
      int n_inactive_doubly_occupied_orbitals,
      int n_active_orbitals) const;

  /**
   * @brief Backpropagates through validated AO integral input caches.
   *
   * This overload is intended for the main runtime path after loader-side AO
   * index validation has already succeeded.
   */
  AoEffectiveOneElectronBackpropagationResult backpropagate(
      const std::vector<double>& ao_effective_one_electron_gradient,
      const AoIntegralInput& ao_integral_input) const;

  AoEffectiveOneElectronBackpropagationResult backpropagate(
      const Eigen::Ref<const Eigen::MatrixXd>& ao_effective_one_electron_gradient,
      const AoIntegralInput& ao_integral_input) const;
};

}  // namespace xmvb::vb
