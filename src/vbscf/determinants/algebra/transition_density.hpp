#pragma once

#include <Eigen/Core>

#include "vbscf/determinants/algebra/compound.hpp"

namespace xmvb::vb {

/**
 * @brief Arbitrary-order transition densities for one regular determinant pair.
 *
 * Given an invertible occupied-overlap matrix `X`, order q is represented as
 * `det(X) C_q(X^-T)`.  A rank-one change of `X` updates every retained order
 * through the determinant lemma and one compound rank-one insertion.  This
 * object is pair-local and is intended to be consumed inside a tile.
 * Ill-conditioned pairs must use the inverse-free complementary-minor path.
 */
class TransitionDensityHierarchy {
 public:
  TransitionDensityHierarchy(int n_electrons, int max_order);

  /** @brief Initialize from a certified determinant and inverse of `X`. */
  void assign(
      double overlap_determinant,
      const Eigen::Ref<const Eigen::MatrixXd>& inverse_overlap);

  /**
   * @brief Apply `X <- X + left * right.transpose()`.
   *
   * The caller must use this only while its occupied-overlap inverse remains
   * numerically certified.  A singular determinant-lemma denominator is
   * rejected so that the caller can rebuild with complementary minors.
   */
  bool apply_overlap_rank_one(
      const Eigen::Ref<const Eigen::VectorXd>& left,
      const Eigen::Ref<const Eigen::VectorXd>& right);

  double overlap_determinant() const noexcept { return determinant_; }
  const Eigen::MatrixXd& inverse_transpose() const noexcept {
    return inverse_transpose_;
  }
  const Eigen::MatrixXd& normalized_level(int order) const {
    return compounds_.level(order);
  }
  Eigen::MatrixXd level(int order) const;
  /** @brief Differentiate an order-q density along a change of `X`. */
  Eigen::MatrixXd directional_level(
      int order,
      const Eigen::Ref<const Eigen::MatrixXd>& overlap_direction) const;
  double contraction(
      int order, const Eigen::Ref<const Eigen::MatrixXd>& weights) const;
  /** @brief Pull an order-q density contraction back to `X`. */
  Eigen::MatrixXd contraction_gradient(
      int order, const Eigen::Ref<const Eigen::MatrixXd>& weights) const;

 private:
  int n_electrons_ = 0;
  double determinant_ = 0.0;
  Eigen::MatrixXd inverse_transpose_;
  CompoundHierarchy compounds_;
};

}  // namespace xmvb::vb
