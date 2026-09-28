#pragma once

#include <memory>

#include <Eigen/Core>

namespace xmvb::vb {

class CofactorDifferential;

struct WoodburyBaseUpdate {
  /**
   * `right` is the physical overlap-update factor `V`.  The inverse update is
   * `K_new = K_old + left * (right^T K_old)`.
   */
  Eigen::MatrixXd left;
  Eigen::MatrixXd right;
};

struct WoodburyContraction {
  double value = 0.0;
  /** Derivative with respect to the physical occupied overlap block. */
  Eigen::MatrixXd overlap_gradient;
  /** Derivative with respect to the contracted transition block. */
  Eigen::MatrixXd transition_gradient;
};

struct WoodburyContractionDirection {
  double value = 0.0;
  Eigen::MatrixXd overlap_gradient;
  Eigen::MatrixXd transition_gradient;
};

/**
 * @brief Stable inverse base plus an inverse-free low-rank overlap core.
 *
 * The represented occupied overlap is
 *
 * `X = A + U V^T`,
 *
 * where `A` has a certified inverse.  Cofactors of `X` are evaluated through
 * the small matrix `G = I + V^T A^-1 U`; no inverse of `G` is required.
 */
class WoodburyCore {
 public:
  explicit WoodburyCore(const Eigen::Ref<const Eigen::MatrixXd>& overlap);
  ~WoodburyCore();

  int dimension() const noexcept { return base_.rows(); }
  int rank() const noexcept { return core_left_.cols(); }
  int nullity() const noexcept { return core_nullity_; }
  double determinant() const noexcept;

  const Eigen::MatrixXd& base() const noexcept { return base_; }
  const Eigen::MatrixXd& inverse_base() const noexcept {
    return inverse_base_;
  }
  const Eigen::MatrixXd& core_left() const noexcept { return core_left_; }
  const Eigen::MatrixXd& core_right() const noexcept { return core_right_; }
  const Eigen::MatrixXd& overlap() const noexcept { return overlap_; }

  /**
   * @brief Appends an exact low-rank overlap update and absorbs safe modes.
   *
   * The returned factors update any base-inverse channel `C = K M` as
   * `C_new = C + update.left * (update.right^T C)` before applying a change
   * in `M` itself.
   */
  WoodburyBaseUpdate append(
      const Eigen::Ref<const Eigen::MatrixXd>& update_left,
      const Eigen::Ref<const Eigen::MatrixXd>& update_right);

  /** @brief Exact first cofactor of the represented overlap. */
  Eigen::MatrixXd first_cofactor() const;
  Eigen::MatrixXd first_cofactor_direction(
      const Eigen::Ref<const Eigen::MatrixXd>& overlap_direction) const;

  /** @brief Contracts one matrix with the exact first cofactor. */
  double first_contraction(
      const Eigen::Ref<const Eigen::MatrixXd>& transition) const;
  WoodburyContraction first_contraction_gradient(
      const Eigen::Ref<const Eigen::MatrixXd>& transition) const;
  WoodburyContractionDirection first_contraction_gradient_direction(
      const Eigen::Ref<const Eigen::MatrixXd>& transition,
      const Eigen::Ref<const Eigen::MatrixXd>& overlap_direction,
      const Eigen::Ref<const Eigen::MatrixXd>& transition_direction) const;
  double first_channel_contraction(
      const Eigen::Ref<const Eigen::MatrixXd>& channel) const;

  /**
   * @brief Contracts one RI factor with the exact second cofactor.
   *
   * The result is the unnormalized same-spin contribution
   *
   * `det(X) / 2 * {tr[(X^-1 M)]^2 - tr[(X^-1 M)^2]}`
   *
   * interpreted by polynomial continuation when `X` is singular.
   */
  double second_factor_contraction(
      const Eigen::Ref<const Eigen::MatrixXd>& transition) const;
  WoodburyContraction second_factor_contraction_gradient(
      const Eigen::Ref<const Eigen::MatrixXd>& transition) const;
  WoodburyContractionDirection second_factor_contraction_gradient_direction(
      const Eigen::Ref<const Eigen::MatrixXd>& transition,
      const Eigen::Ref<const Eigen::MatrixXd>& overlap_direction,
      const Eigen::Ref<const Eigen::MatrixXd>& transition_direction) const;
  WoodburyContraction second_channel_contraction_gradient(
      const Eigen::Ref<const Eigen::MatrixXd>& channel,
      const Eigen::Ref<const Eigen::MatrixXd>& transition) const;
  double second_channel_contraction(
      const Eigen::Ref<const Eigen::MatrixXd>& channel) const;

 private:
  double core_first_contraction(
      const Eigen::Ref<const Eigen::MatrixXd>& matrix) const;
  static Eigen::MatrixXd exterior_square(
      const Eigen::Ref<const Eigen::MatrixXd>& matrix);
  static Eigen::MatrixXd exterior_square_reverse(
      const Eigen::Ref<const Eigen::MatrixXd>& matrix,
      const Eigen::Ref<const Eigen::MatrixXd>& exterior_gradient);
  void rebuild_core_contraction();

  Eigen::MatrixXd overlap_;
  Eigen::MatrixXd base_;
  Eigen::MatrixXd inverse_base_;
  Eigen::MatrixXd core_left_;
  Eigen::MatrixXd core_right_;
  Eigen::MatrixXd inverse_base_core_left_;
  double base_determinant_ = 1.0;
  double core_determinant_ = 1.0;
  int core_nullity_ = 0;
  std::unique_ptr<const CofactorDifferential> core_cofactor_;
};

}  // namespace xmvb::vb
