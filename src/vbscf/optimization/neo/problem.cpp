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
    Eigen::VectorXd initial_guess)
    : gradient_(std::move(gradient)),
      apply_hessian_(std::move(apply_hessian)),
      apply_metric_(std::move(apply_metric)),
      apply_preconditioner_(std::move(apply_preconditioner)),
      hessian_lower_bound_(hessian_lower_bound),
      initial_guess_(std::move(initial_guess)) {
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
