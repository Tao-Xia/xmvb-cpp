#pragma once

#include <vector>

namespace xmvb::vb {

/**
 * @brief Active-space effective one-electron matrices derived from AO data.
 */
struct ActiveSpaceOneElectronResult {
  /**
   * @brief Column-major active-space effective one-electron matrix `HHO`.
   */
  std::vector<double> active_one_electron_matrix;
};

}  // namespace xmvb::vb
