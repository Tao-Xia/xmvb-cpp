#pragma once

#include <vector>

#include <Eigen/Core>

namespace xmvb::vb {

/**
 * @brief Implicit arbitrary-order transition-density contractions.
 *
 * For a regular occupied overlap inverse `K` and transition block `M`, this
 * state represents the coefficients
 *
 * `e_q(K M) = [t^q] det(I + t K M)`.
 *
 * It stores matrix powers and the contracted moments `(K M)^j K`, not an
 * order-q RDM.  Both hierarchies admit exact quadratic-cost updates when `K`
 * and `K M` change by low rank along a unique-spin-string pair graph.
 */
class ContractedDensityState {
 public:
  ContractedDensityState(
      const Eigen::Ref<const Eigen::MatrixXd>& inverse_overlap,
      const Eigen::Ref<const Eigen::MatrixXd>& transition,
      int maximum_order);

  int dimension() const noexcept { return inverse_overlap_.rows(); }
  int maximum_order() const noexcept { return maximum_order_; }
  const Eigen::MatrixXd& inverse_overlap() const noexcept {
    return inverse_overlap_;
  }
  const Eigen::MatrixXd& channel() const noexcept { return channel_; }

  /** @brief Returns `e_order(K M)`. */
  double coefficient(int order) const;

  /** @brief Returns the unnormalized contracted transition density. */
  double contraction(int order, double overlap_determinant) const;

  /** @brief Derivative of the contraction with respect to `M`. */
  Eigen::MatrixXd transition_gradient(
      int order,
      double overlap_determinant) const;

  /** @brief Derivative of the contraction with respect to `X`, `K=X^-1`. */
  Eigen::MatrixXd overlap_gradient(
      int order,
      double overlap_determinant) const;

  /**
   * @brief Applies exact low-rank graph-edge changes.
   *
   * The supplied factors define
   *
   * `K_new = K + inverse_left inverse_right^T`,
   * `A_new = A + channel_left channel_right^T`, `A = K M`.
   *
   * Every stored order is updated without a dense matrix-matrix product.
   * For fixed order and ranks, the cost is quadratic in the occupied-string
   * dimension.
   */
  void update(
      const Eigen::Ref<const Eigen::MatrixXd>& inverse_left,
      const Eigen::Ref<const Eigen::MatrixXd>& inverse_right,
      const Eigen::Ref<const Eigen::MatrixXd>& channel_left,
      const Eigen::Ref<const Eigen::MatrixXd>& channel_right);

 private:
  void rebuild_coefficients();
  void validate_order(int order) const;

  int maximum_order_ = 0;
  Eigen::MatrixXd inverse_overlap_;
  Eigen::MatrixXd channel_;
  std::vector<Eigen::MatrixXd> powers_;
  std::vector<Eigen::MatrixXd> inverse_moments_;
  std::vector<double> coefficients_;
};

}  // namespace xmvb::vb
