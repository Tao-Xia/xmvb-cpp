#include "vbscf/optimization/preconditioners/block_inverse_bfgs.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <Eigen/Eigenvalues>

namespace xmvb::vb {

BlockInverseBfgs::BlockInverseBfgs(
    Eigen::MatrixXd directions,
    Eigen::MatrixXd hessian_images)
    : directions_(std::move(directions)),
      hessian_images_(std::move(hessian_images)) {
  if (directions_.rows() <= 0 || directions_.cols() <= 0 ||
      hessian_images_.rows() != directions_.rows() ||
      hessian_images_.cols() != directions_.cols() ||
      !directions_.allFinite() || !hessian_images_.allFinite()) {
    throw std::invalid_argument("invalid block inverse-BFGS secants");
  }

  const Eigen::MatrixXd raw_curvature =
      directions_.transpose() * hessian_images_;
  const double curvature_scale = std::max(
      std::numeric_limits<double>::min(),
      raw_curvature.stableNorm());
  const double symmetry_tolerance =
      std::sqrt(std::numeric_limits<double>::epsilon()) *
      static_cast<double>(std::max<Eigen::Index>(1, raw_curvature.rows())) *
      curvature_scale;
  if ((raw_curvature - raw_curvature.transpose()).stableNorm() >
      symmetry_tolerance) {
    throw std::invalid_argument(
        "block inverse-BFGS curvature is not symmetric");
  }
  curvature_ =
      0.5 * (raw_curvature + raw_curvature.transpose()).eval();

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(curvature_);
  if (spectrum.info() != Eigen::Success ||
      !spectrum.eigenvalues().allFinite()) {
    throw std::runtime_error(
        "failed to resolve block inverse-BFGS curvature");
  }
  const double maximum_curvature = spectrum.eigenvalues().maxCoeff();
  const double positivity_floor =
      std::numeric_limits<double>::epsilon() *
      static_cast<double>(std::max<Eigen::Index>(1, curvature_.rows())) *
      maximum_curvature;
  if (!(maximum_curvature > 0.0) ||
      spectrum.eigenvalues().minCoeff() <= positivity_floor) {
    throw std::invalid_argument(
        "block inverse-BFGS curvature is not numerically positive definite");
  }

  curvature_factor_.compute(curvature_);
  if (curvature_factor_.info() != Eigen::Success) {
    throw std::runtime_error(
        "failed to factor block inverse-BFGS curvature");
  }
}

int BlockInverseBfgs::dimension() const noexcept {
  return static_cast<int>(directions_.rows());
}

int BlockInverseBfgs::rank() const noexcept {
  return static_cast<int>(directions_.cols());
}

Eigen::MatrixXd BlockInverseBfgs::solve_curvature(
    const Eigen::MatrixXd& right_hand_side) const {
  Eigen::MatrixXd solution = curvature_factor_.solve(right_hand_side);
  if (curvature_factor_.info() != Eigen::Success || !solution.allFinite()) {
    throw std::runtime_error(
        "block inverse-BFGS curvature solve failed");
  }
  return solution;
}

Eigen::VectorXd BlockInverseBfgs::apply(
    const Eigen::VectorXd& covector,
    const VectorInverseAction& base_inverse) const {
  if (covector.size() != directions_.rows() || !covector.allFinite() ||
      !base_inverse) {
    throw std::invalid_argument("invalid block inverse-BFGS vector action");
  }

  const Eigen::VectorXd secant_coordinates =
      solve_curvature(directions_.transpose() * covector);
  const Eigen::VectorXd unresolved =
      covector - hessian_images_ * secant_coordinates;
  const Eigen::VectorXd base = base_inverse(unresolved);
  if (base.size() != covector.size() || !base.allFinite()) {
    throw std::runtime_error(
        "base inverse returned an invalid block inverse-BFGS vector");
  }
  const Eigen::VectorXd base_coordinates =
      solve_curvature(hessian_images_.transpose() * base);
  Eigen::VectorXd result =
      base + directions_ * (secant_coordinates - base_coordinates);
  if (!result.allFinite()) {
    throw std::runtime_error(
        "block inverse-BFGS vector action is non-finite");
  }
  return result;
}

Eigen::MatrixXd BlockInverseBfgs::apply_block(
    const Eigen::MatrixXd& covectors,
    const BlockInverseAction& base_inverse) const {
  if (covectors.rows() != directions_.rows() || !covectors.allFinite() ||
      !base_inverse) {
    throw std::invalid_argument("invalid block inverse-BFGS matrix action");
  }
  if (covectors.cols() == 0) {
    return Eigen::MatrixXd::Zero(directions_.rows(), 0);
  }

  const Eigen::MatrixXd secant_coordinates =
      solve_curvature(directions_.transpose() * covectors);
  const Eigen::MatrixXd unresolved =
      covectors - hessian_images_ * secant_coordinates;
  const Eigen::MatrixXd base = base_inverse(unresolved);
  if (base.rows() != covectors.rows() || base.cols() != covectors.cols() ||
      !base.allFinite()) {
    throw std::runtime_error(
        "base inverse returned an invalid block inverse-BFGS matrix");
  }
  const Eigen::MatrixXd base_coordinates =
      solve_curvature(hessian_images_.transpose() * base);
  Eigen::MatrixXd result =
      base + directions_ * (secant_coordinates - base_coordinates);
  if (!result.allFinite()) {
    throw std::runtime_error(
        "block inverse-BFGS matrix action is non-finite");
  }
  return result;
}

}  // namespace xmvb::vb
