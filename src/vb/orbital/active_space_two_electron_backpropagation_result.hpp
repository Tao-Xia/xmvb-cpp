#pragma once

#include <vector>

namespace xmvb::vb {

/**
 * @brief Reverse-mode derivatives through the active-space two-electron builder.
 */
struct ActiveSpaceTwoElectronBackpropagationResult {
  /**
   * @brief Column-major gradient with respect to the full auxiliary orbital matrix.
   *
   * Only the active columns are populated by the current implementation.
   */
  std::vector<double> auxiliary_orbital_gradient;
};

}  // namespace xmvb::vb
