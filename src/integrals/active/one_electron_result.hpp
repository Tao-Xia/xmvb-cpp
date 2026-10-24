#pragma once

#include <Eigen/Core>

namespace xmvb::integrals {

/**
 * @brief Active-space effective one-electron matrices derived from AO data.
 */
struct ActiveSpaceOneElectronResult {
    /**
   * @brief Effective h in the same coefficient frame as the active 2e data.
   * VB preparation uses Cbar = C R^-1 for both h and G; an empty R denotes C.
   */
    Eigen::MatrixXd h1e_act;
    /** @brief Accepted F Cbar (or F C for raw inputs), shared by derivatives. */
    Eigen::MatrixXd ao_image;
};

} // namespace xmvb::integrals
