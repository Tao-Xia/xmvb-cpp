#pragma once

#include <vector>

namespace xmvb::vb {

/**
 * @brief Reverse-mode derivative of the AO effective one-electron builder.
 */
struct AoEffectiveOneElectronBackpropagationResult {
  /**
   * @brief Column-major gradient with respect to the inactive density matrix `P11`.
   */
  std::vector<double> inactive_density_gradient;
};

}  // namespace xmvb::vb
