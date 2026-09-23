#include "vbscf/optimization/preconditioners/shifted_metric.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace xmvb::vb {
namespace {

Eigen::VectorXd checked_action(
    const PreconditionerAction& action,
    const Eigen::VectorXd& vector,
    const char* message) {
  Eigen::VectorXd image = action(vector);
  if (image.size() != vector.size() || !image.allFinite()) {
    throw std::runtime_error(message);
  }
  return image;
}

}  // namespace

Eigen::VectorXd apply_shifted_metric_preconditioner(
    const Eigen::VectorXd& covector,
    double shift,
    double residual_target,
    const PreconditionerAction& apply_model,
    const PreconditionerAction& apply_metric,
    const PreconditionerAction& apply_block_inverse) {
  if (covector.size() == 0 || !covector.allFinite() ||
      !(shift >= 0.0) || !std::isfinite(shift) ||
      !(residual_target >= 0.0) || !std::isfinite(residual_target) ||
      !apply_model || !apply_metric || !apply_block_inverse) {
    throw std::invalid_argument("invalid shifted-metric preconditioner data");
  }
  if (covector.isZero()) return Eigen::VectorXd::Zero(covector.size());
  if (shift == 0.0 || residual_target >= covector.norm()) {
    return checked_action(
        apply_block_inverse, covector,
        "shifted block inverse returned an invalid vector");
  }

  const auto apply_shifted = [&](const Eigen::VectorXd& vector) {
    Eigen::VectorXd image = checked_action(
        apply_model, vector, "local model returned an invalid vector");
    image.noalias() += shift * checked_action(
        apply_metric, vector, "orbital metric returned an invalid vector");
    return image;
  };

  // This solve only supplies a direction to the outer coupled Krylov space.
  // Use the geometric mean of its RHS norm and the outer KKT target: the
  // preconditioner becomes more accurate as the outer solve tightens, without
  // imposing the final KKT tolerance on every individual basis direction.
  const double target = std::max(
      std::sqrt(residual_target * covector.norm()),
      std::sqrt(std::numeric_limits<double>::epsilon()) * covector.norm());
  Eigen::VectorXd solution = Eigen::VectorXd::Zero(covector.size());
  Eigen::VectorXd residual = covector;
  Eigen::VectorXd preconditioned = checked_action(
      apply_block_inverse, residual,
      "shifted block inverse returned an invalid vector");
  Eigen::VectorXd direction = preconditioned;
  double residual_pairing = residual.dot(preconditioned);
  if (!(residual_pairing > 0.0) || !std::isfinite(residual_pairing)) {
    throw std::runtime_error(
        "shifted-metric block inverse is not positive definite");
  }

  Eigen::VectorXd best_solution = preconditioned;
  double best_residual =
      (covector - apply_shifted(best_solution)).norm();
  if (best_residual <= target) return best_solution;

  for (Eigen::Index iteration = 0;
       iteration < covector.size(); ++iteration) {
    const Eigen::VectorXd image = apply_shifted(direction);
    const double curvature = direction.dot(image);
    if (!(curvature > 0.0) || !std::isfinite(curvature)) {
      throw std::runtime_error(
          "shifted local metric model is not positive definite");
    }
    const double step = residual_pairing / curvature;
    if (!std::isfinite(step)) {
      throw std::runtime_error(
          "shifted-metric preconditioner produced a non-finite step");
    }
    solution.noalias() += step * direction;
    residual = covector - apply_shifted(solution);
    const double residual_norm = residual.norm();
    if (residual_norm < best_residual) {
      best_residual = residual_norm;
      best_solution = solution;
    }
    if (residual_norm <= target) return solution;

    preconditioned = checked_action(
        apply_block_inverse, residual,
        "shifted block inverse returned an invalid vector");
    const double next_pairing = residual.dot(preconditioned);
    if (!(next_pairing > 0.0) || !std::isfinite(next_pairing)) {
      throw std::runtime_error(
          "shifted-metric preconditioner lost positive definiteness");
    }
    direction = preconditioned +
        (next_pairing / residual_pairing) * direction;
    residual_pairing = next_pairing;
  }
  // Preconditioning is not an acceptance test. Return the best positive
  // model direction even if the outer target was unreachable here.
  return best_solution;
}

}  // namespace xmvb::vb
