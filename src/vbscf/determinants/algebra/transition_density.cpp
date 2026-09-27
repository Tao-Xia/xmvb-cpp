#include "vbscf/determinants/algebra/transition_density.hpp"

#include <cmath>
#include <stdexcept>

namespace xmvb::vb {

TransitionDensityHierarchy::TransitionDensityHierarchy(
    int n_electrons, int max_order)
    : n_electrons_(n_electrons),
      compounds_(n_electrons, max_order) {
  inverse_transpose_.resize(n_electrons, n_electrons);
}

void TransitionDensityHierarchy::assign(
    double overlap_determinant,
    const Eigen::Ref<const Eigen::MatrixXd>& inverse_overlap) {
  if (inverse_overlap.rows() != n_electrons_ ||
      inverse_overlap.cols() != n_electrons_ ||
      !std::isfinite(overlap_determinant) || !inverse_overlap.allFinite()) {
    throw std::invalid_argument("invalid regular transition-density anchor");
  }
  determinant_ = overlap_determinant;
  inverse_transpose_ = inverse_overlap.transpose();
  compounds_.assign(inverse_transpose_);
}

bool TransitionDensityHierarchy::apply_overlap_rank_one(
    const Eigen::Ref<const Eigen::VectorXd>& left,
    const Eigen::Ref<const Eigen::VectorXd>& right) {
  if (left.size() != n_electrons_ || right.size() != n_electrons_) {
    throw std::invalid_argument("transition-density update has the wrong size");
  }

  const Eigen::VectorXd inverse_left = inverse_transpose_.transpose() * left;
  const Eigen::VectorXd inverse_transpose_right = inverse_transpose_ * right;
  const double ratio = 1.0 + right.dot(inverse_left);
  const double updated_determinant = determinant_ * ratio;
  if (ratio == 0.0 || !std::isfinite(ratio) ||
      !std::isfinite(updated_determinant)) {
    return false;
  }

  const Eigen::VectorXd update_left = -inverse_transpose_right / ratio;
  const Eigen::MatrixXd updated_inverse_transpose =
      inverse_transpose_ + update_left * inverse_left.transpose();
  if (!updated_inverse_transpose.allFinite()) {
    return false;
  }
  compounds_.apply_rank_one(update_left, inverse_left);
  inverse_transpose_ = updated_inverse_transpose;
  determinant_ = updated_determinant;
  return true;
}

Eigen::MatrixXd TransitionDensityHierarchy::level(int order) const {
  return determinant_ * compounds_.level(order);
}

Eigen::MatrixXd TransitionDensityHierarchy::directional_level(
    int order,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_direction) const {
  if (overlap_direction.rows() != n_electrons_ ||
      overlap_direction.cols() != n_electrons_) {
    throw std::invalid_argument(
        "transition-density direction has the wrong size");
  }
  const Eigen::MatrixXd inverse_direction =
      -inverse_transpose_ * overlap_direction.transpose() *
      inverse_transpose_;
  const CompoundHierarchy compound_direction =
      compounds_.directional(inverse_direction);
  const double determinant_direction = determinant_ *
      (inverse_transpose_.transpose() * overlap_direction).trace();
  return determinant_direction * compounds_.level(order) +
      determinant_ * compound_direction.level(order);
}

double TransitionDensityHierarchy::contraction(
    int order, const Eigen::Ref<const Eigen::MatrixXd>& weights) const {
  const Eigen::MatrixXd& normalized = compounds_.level(order);
  if (weights.rows() != normalized.rows() ||
      weights.cols() != normalized.cols()) {
    throw std::invalid_argument(
        "transition-density contraction weights have the wrong size");
  }
  return determinant_ * (weights.array() * normalized.array()).sum();
}

Eigen::MatrixXd TransitionDensityHierarchy::contraction_gradient(
    int order, const Eigen::Ref<const Eigen::MatrixXd>& weights) const {
  const Eigen::MatrixXd& normalized = compounds_.level(order);
  if (weights.rows() != normalized.rows() ||
      weights.cols() != normalized.cols()) {
    throw std::invalid_argument(
        "transition-density contraction weights have the wrong size");
  }
  const double normalized_contraction =
      (weights.array() * normalized.array()).sum();
  const Eigen::MatrixXd inverse_gradient =
      compounds_.pullback(order, weights);
  return determinant_ *
      (normalized_contraction * inverse_transpose_ -
       inverse_transpose_ * inverse_gradient.transpose() *
           inverse_transpose_);
}

}  // namespace xmvb::vb
