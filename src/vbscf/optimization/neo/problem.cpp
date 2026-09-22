#include "vbscf/optimization/neo/problem.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace xmvb::vb {

NeoProblem::NeoProblem(
    Eigen::VectorXd gradient,
    NeoAction apply_hessian,
    NeoAction apply_metric,
    NeoPreconditioner apply_preconditioner,
    std::optional<double> hessian_lower_bound,
    Eigen::VectorXd initial_guess,
    NeoCombinedAction apply_hessian_metric)
    : gradient_(std::move(gradient)),
      apply_hessian_(std::move(apply_hessian)),
      apply_metric_(std::move(apply_metric)),
      apply_preconditioner_(std::move(apply_preconditioner)),
      hessian_lower_bound_(hessian_lower_bound),
      initial_guess_(std::move(initial_guess)),
      apply_hessian_metric_(std::move(apply_hessian_metric)) {
  if (gradient_.size() == 0 || !gradient_.allFinite()) {
    throw std::invalid_argument("NEO requires a finite nonempty gradient");
  }
  if (!apply_hessian_ || !apply_metric_) {
    throw std::invalid_argument("NEO requires Hessian and metric actions");
  }
  if (hessian_lower_bound_ && !std::isfinite(*hessian_lower_bound_)) {
    throw std::invalid_argument("NEO Hessian lower bound must be finite");
  }
  if (initial_guess_.size() != 0 &&
      (initial_guess_.size() != gradient_.size() ||
       !initial_guess_.allFinite())) {
    throw std::invalid_argument("NEO initial guess is invalid");
  }
}

Eigen::VectorXd NeoProblem::apply_checked(
    const NeoAction& action,
    const Eigen::VectorXd& vector,
    const char* name) const {
  if (vector.size() != size() || !vector.allFinite()) {
    throw std::invalid_argument("invalid vector passed to NEO operator");
  }
  Eigen::VectorXd result = action(vector);
  if (result.size() != size() || !result.allFinite()) {
    throw std::runtime_error(name);
  }
  return result;
}

Eigen::VectorXd NeoProblem::apply_hessian(
    const Eigen::VectorXd& direction) const {
  return apply_checked(
      apply_hessian_, direction, "NEO Hessian action returned an invalid vector");
}

Eigen::VectorXd NeoProblem::apply_metric(
    const Eigen::VectorXd& direction) const {
  return apply_checked(
      apply_metric_, direction, "NEO metric action returned an invalid vector");
}

NeoOperatorImages NeoProblem::apply_hessian_metric(
    const Eigen::VectorXd& direction) const {
  if (!apply_hessian_metric_) {
    return {apply_hessian(direction), apply_metric(direction)};
  }
  if (direction.size() != size() || !direction.allFinite()) {
    throw std::invalid_argument("invalid vector passed to NEO operator");
  }
  NeoOperatorImages images = apply_hessian_metric_(direction);
  if (images.hessian.size() != size() || images.metric.size() != size() ||
      !images.hessian.allFinite() || !images.metric.allFinite()) {
    throw std::runtime_error(
        "NEO combined Hessian/metric action returned invalid images");
  }
  return images;
}

Eigen::VectorXd NeoProblem::apply_preconditioner(
    const Eigen::VectorXd& covector,
    double shift) const {
  if (!apply_preconditioner_) {
    throw std::logic_error("NEO problem has no residual preconditioner");
  }
  if (!std::isfinite(shift)) {
    throw std::invalid_argument(
        "NEO augmented-Hessian eigenvalue must be finite");
  }
  if (covector.size() != size() || !covector.allFinite()) {
    throw std::invalid_argument("invalid vector passed to NEO preconditioner");
  }
  Eigen::VectorXd result = apply_preconditioner_(covector, shift);
  if (result.size() != size() || !result.allFinite()) {
    throw std::runtime_error(
        "NEO preconditioner returned an invalid vector");
  }
  return result;
}

}  // namespace xmvb::vb
