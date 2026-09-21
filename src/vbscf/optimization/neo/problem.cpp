#include "vbscf/optimization/neo/problem.hpp"

#include <stdexcept>
#include <utility>

namespace xmvb::vb {

NeoProblem::NeoProblem(
    Eigen::VectorXd gradient,
    NeoAction apply_hessian,
    NeoAction apply_metric,
    NeoAction apply_inverse_metric)
    : gradient_(std::move(gradient)),
      apply_hessian_(std::move(apply_hessian)),
      apply_metric_(std::move(apply_metric)),
      apply_inverse_metric_(std::move(apply_inverse_metric)) {
  if (gradient_.size() == 0 || !gradient_.allFinite()) {
    throw std::invalid_argument("NEO requires a finite nonempty gradient");
  }
  if (!apply_hessian_ || !apply_metric_ || !apply_inverse_metric_) {
    throw std::invalid_argument("NEO requires Hessian and metric actions");
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

Eigen::VectorXd NeoProblem::apply_inverse_metric(
    const Eigen::VectorXd& covector) const {
  return apply_checked(
      apply_inverse_metric_, covector,
      "NEO inverse-metric action returned an invalid vector");
}

}  // namespace xmvb::vb
