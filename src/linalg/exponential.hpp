#pragma once

#include <Eigen/Core>

namespace xmvb::linalg {

/** @brief Matrix exponential of a real orbital generator. */
Eigen::MatrixXd exponential(const Eigen::MatrixXd& generator);

/** @brief Frobenius adjoint of the exponential Frechet derivative.
 * Returns L_exp(generator^T, cotangent), without one exponential per coordinate.
 */
Eigen::MatrixXd exponential_pullback(const Eigen::MatrixXd& generator,
                                     const Eigen::MatrixXd& cotangent);

} // namespace xmvb::linalg
