#pragma once

#include <Eigen/Core>

namespace xmvb::vb {

/** @brief Bounded diagnostic decomposition of a direct-CI orbital direction.
 * Raw structure images are F^T delta(h) y and the remaining basis-motion
 * contribution to delta(H C)-E delta(M C). The connection covector is
 * F^T delta(F) C. No assumption that the carrier Ritz residual vanishes is made.
 */
struct StructureConnectionDirection {
    Eigen::MatrixXd representation_forcing;
    Eigen::MatrixXd integral_forcing;
    Eigen::MatrixXd overlap;
    Eigen::VectorXd squared_norm;
};

} // namespace xmvb::vb
