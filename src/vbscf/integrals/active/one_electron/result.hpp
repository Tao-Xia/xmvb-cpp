#pragma once

#include <Eigen/Core>

namespace xmvb::vb {

/**
 * @brief Active-space effective one-electron matrices derived from AO data.
 */
struct ActiveSpaceOneElectronResult {
  /**
   * @brief Column-major active-space effective one-electron matrix `HHO`.
   */
  Eigen::MatrixXd h1e_act;
};

}  // namespace xmvb::vb
