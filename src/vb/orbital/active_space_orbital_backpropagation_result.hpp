#pragma once

#include <vector>

namespace xmvb::vb {

/**
 * @brief Reverse-mode derivatives through the active-space orbital preparer.
 */
struct ActiveSpaceOrbitalBackpropagationResult {
  /**
   * @brief Gradient with respect to the sparse orbital coefficient table.
   */
  std::vector<double> orbital_value_gradient;
};

}  // namespace xmvb::vb
