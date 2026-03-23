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
  std::vector<double> h1e_act;
};

}  // namespace xmvb::vb
