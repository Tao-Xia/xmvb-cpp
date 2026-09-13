#pragma once

#include <Eigen/Core>

#include <vector>

#include "vbscf/integrals/ao/contracts/input.hpp"
#include "vbscf/integrals/ao/one_electron/backpropagation_result.hpp"
#include "vbscf/integrals/ao/ri/factorization.hpp"
#include "vbscf/orbitals/preparation/result.hpp"

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
   * @brief Backpropagates through the AO-side RI inactive-density contraction.
   */
  AoEffectiveOneElectronBackpropagationResult backpropagate(
      const std::vector<double>& ao_effective_one_electron_gradient,
      const RiAoFactorization& ri_factorization,
      int n_basis_functions) const;

  AoEffectiveOneElectronBackpropagationResult backpropagate(
      const Eigen::Ref<const Eigen::MatrixXd>& ao_effective_one_electron_gradient,
      const RiAoFactorization& ri_factorization,
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
      const RiAoFactorization& ri_factorization,
      int n_basis_functions,
      int n_inactive_doubly_occupied_orbitals,
      int n_active_orbitals) const;

  /**
   * @brief Backpropagates through the canonical exact AO operator.
   */
  AoEffectiveOneElectronBackpropagationResult backpropagate(
      const std::vector<double>& ao_effective_one_electron_gradient,
      const AoIntegralInput& ao_integral_input) const;

  AoEffectiveOneElectronBackpropagationResult backpropagate(
      const Eigen::Ref<const Eigen::MatrixXd>& ao_effective_one_electron_gradient,
      const AoIntegralInput& ao_integral_input) const;
};

}  // namespace xmvb::vb
