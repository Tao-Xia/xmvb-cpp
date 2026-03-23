#pragma once

#include <vector>

#include "vb/orbital/ao_effective_one_electron_backpropagation_result.hpp"

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
};

}  // namespace xmvb::vb
