#pragma once

#include <vector>

#include <Eigen/Core>

namespace xmvb::vb {

class ContractedDensityJet;

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
  friend class ContractedDensityJet;
  void rebuild_coefficients();
  void validate_order(int order) const;

  int maximum_order_ = 0;
  Eigen::MatrixXd inverse_overlap_;
  Eigen::MatrixXd channel_;
  std::vector<Eigen::MatrixXd> powers_;
  std::vector<Eigen::MatrixXd> inverse_moments_;
  std::vector<double> coefficients_;
};

/**
 * @brief Directional jet of the implicit contracted-density hierarchy.
 *
 * The jet propagates the tangent of every stored coefficient and adjoint
 * moment together with the accepted state.  A graph edge differentiates the
 * same low-rank recurrences used by `ContractedDensityState`; no higher-order
 * RDM and no dense matrix-matrix edge product is formed.
 */
class ContractedDensityJet {
 public:
  ContractedDensityJet(
      const Eigen::Ref<const Eigen::MatrixXd>& inverse_overlap,
      const Eigen::Ref<const Eigen::MatrixXd>& transition,
      const Eigen::Ref<const Eigen::MatrixXd>& inverse_direction,
      const Eigen::Ref<const Eigen::MatrixXd>& transition_direction,
      int maximum_order);

  const ContractedDensityState& value() const noexcept { return value_; }
  const Eigen::MatrixXd& inverse_direction() const noexcept {
    return inverse_direction_;
  }
  const Eigen::MatrixXd& channel_direction() const noexcept {
    return channel_direction_;
  }

  double coefficient_direction(int order) const;
  double contraction_direction(
      int order,
      double overlap_determinant,
      double overlap_determinant_direction) const;
  Eigen::MatrixXd transition_gradient_direction(
      int order,
      double overlap_determinant,
      double overlap_determinant_direction) const;
  Eigen::MatrixXd overlap_gradient_direction(
      int order,
      double overlap_determinant,
      double overlap_determinant_direction) const;

  /**
   * @brief Applies an accepted low-rank edge and its exact tangent.
   *
   * Directional factors describe derivatives of both low-rank products, e.g.
   * `delta(L R^T) = delta_L R^T + L delta_R^T`.
   */
  void update(
      const Eigen::Ref<const Eigen::MatrixXd>& inverse_left,
      const Eigen::Ref<const Eigen::MatrixXd>& inverse_right,
      const Eigen::Ref<const Eigen::MatrixXd>& inverse_left_direction,
      const Eigen::Ref<const Eigen::MatrixXd>& inverse_right_direction,
      const Eigen::Ref<const Eigen::MatrixXd>& channel_left,
      const Eigen::Ref<const Eigen::MatrixXd>& channel_right,
      const Eigen::Ref<const Eigen::MatrixXd>& channel_left_direction,
      const Eigen::Ref<const Eigen::MatrixXd>& channel_right_direction);

 private:
  void rebuild_directional_coefficients();

  ContractedDensityState value_;
  Eigen::MatrixXd inverse_direction_;
  Eigen::MatrixXd channel_direction_;
  std::vector<Eigen::MatrixXd> directional_powers_;
  std::vector<Eigen::MatrixXd> directional_inverse_moments_;
  std::vector<double> directional_coefficients_;
};

}  // namespace xmvb::vb
