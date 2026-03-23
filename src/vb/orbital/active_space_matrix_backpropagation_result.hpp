#pragma once

#include <vector>

namespace xmvb::vb {

/**
 * @brief Reverse-mode derivatives through the active-space matrix builders.
 */
struct ActiveSpaceMatrixBackpropagationResult {
  /**
   * @brief Column-major gradient with respect to the full auxiliary orbital matrix.
   *
   * Only the active columns are populated by the current implementation.
   */
  std::vector<double> auxiliary_orbital_gradient;

  /**
   * @brief Column-major gradient with respect to the AO effective one-electron matrix `F11`.
   */
  std::vector<double> ao_effective_one_electron_gradient;
};

}  // namespace xmvb::vb
