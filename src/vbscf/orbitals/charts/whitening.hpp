#pragma once

#include <Eigen/Core>

namespace xmvb::vb {

/** @brief Normalized orbital tangent and inverse of its raw AO-metric Gram. */
struct WhitenedTangent {
    Eigen::MatrixXd basis;
    Eigen::MatrixXd inverse_gram;
};

/** @brief Shared accepted-point whitening followed by exact radial updates.
 * The basis differs from per-orbital symmetric whitening by an orthogonal
 * reduced-coordinate transformation. Horizontal lift and physical metric are
 * unchanged. A shared quotient needs one factorization, not one per orbital.
 */
class TangentWhitening {
  public:
    TangentWhitening(const Eigen::Ref<const Eigen::MatrixXd>& quotient,
                     const Eigen::Ref<const Eigen::MatrixXd>& overlap);
    WhitenedTangent apply(const Eigen::Ref<const Eigen::VectorXd>& coefficients,
                          const Eigen::Ref<const Eigen::MatrixXd>& overlap,
                          bool sphere) const;

  private:
    Eigen::MatrixXd basis_;
    Eigen::MatrixXd metric_;
    Eigen::MatrixXd eigenvectors_;
    Eigen::VectorXd eigenvalues_;
};

} // namespace xmvb::vb
