#pragma once

#include <Eigen/Core>

namespace xmvb::vb {

/** @brief A quotient representative without a coordinate inverse square root. */
struct RawTangent {
    Eigen::MatrixXd basis;
    /** @brief Inverse of basis^T S basis for vector/covector duality. */
    Eigen::MatrixXd inverse_gram;
    Eigen::VectorXd projection, inverse_projection;
    double radial_gap = 1.0;
};

/** @brief Shared AO metric of a Euclidean-orthonormal quotient complement.
 * Normalization is a rank-one downdate. The inverse measures covectors and
 * shifts local preconditioners; it does not whiten the orbital tangent basis.
 */
class RawTangentMetric {
  public:
    RawTangentMetric(const Eigen::Ref<const Eigen::MatrixXd>& quotient,
                     const Eigen::Ref<const Eigen::MatrixXd>& overlap);
    RawTangent apply(const Eigen::Ref<const Eigen::MatrixXd>& quotient,
                     const Eigen::Ref<const Eigen::VectorXd>& coefficients,
                     const Eigen::Ref<const Eigen::MatrixXd>& overlap,
                     bool sphere) const;
    const Eigen::MatrixXd& metric() const noexcept {
        return metric_;
    }
    const Eigen::MatrixXd& inverse() const noexcept {
        return inverse_;
    }
    /** @brief Reference covector coordinates, not a whitened orbital basis.
   * The shared inverse root is formed lazily for the outer infinity criterion.
   */
    Eigen::VectorXd reference_covector(const Eigen::VectorXd& covector,
                                       const Eigen::VectorXd& projection) const;

  private:
    Eigen::MatrixXd metric_, inverse_;
    double lower_bound_ = 0.0, upper_bound_ = 0.0;
    mutable Eigen::MatrixXd reference_root_;
};

} // namespace xmvb::vb
