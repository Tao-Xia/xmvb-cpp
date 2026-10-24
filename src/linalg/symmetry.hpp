#pragma once

#include <limits>

#include <Eigen/Core>

namespace xmvb::linalg {

/**
 * @brief Frobenius roundoff bound for the skew of Q^T Y.
 *
 * Each length-n floating-point dot product has absolute error bounded by
 * gamma_n sum_k |Q_ki Y_kj|, where gamma_n = n u / (1 - n u) and u is
 * unit roundoff. The two dots and their subtraction contribute to each
 * skew entry. This bounds projection arithmetic, not errors inside Y.
 * Compute it only after the inexpensive symmetry check fails.
 */
inline double projection_skew_roundoff(const Eigen::Ref<const Eigen::MatrixXd>& basis,
                                       const Eigen::Ref<const Eigen::MatrixXd>& images,
                                       const Eigen::Ref<const Eigen::MatrixXd>& projected) {
    const double u = 0.5 * std::numeric_limits<double>::epsilon();
    const double nu = static_cast<double>(basis.rows()) * u;
    const double gamma = nu / (1.0 - nu);
    const Eigen::MatrixXd magnitude = basis.cwiseAbs().transpose() * images.cwiseAbs();
    // Correct the positive dot-product estimate for its own rounding error.
    const Eigen::MatrixXd error = (gamma / (1.0 - gamma)) * (magnitude + magnitude.transpose()) +
                                  u * (projected.cwiseAbs() + projected.transpose().cwiseAbs());
    return error.stableNorm();
}

} // namespace xmvb::linalg
