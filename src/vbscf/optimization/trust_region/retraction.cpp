#include "vbscf/optimization/trust_region/retraction.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace xmvb::vb {

Eigen::VectorXd gather_nonredundant_retract_tangent(
    const OrbitalPreparationInput& orbital_preparation_input,
    const OrbitalChart& space,
    const SparseParameterLayout& parameter_view,
    const Eigen::VectorXd& reduced_step) {
  const Eigen::VectorXd full_tangent =
      space.expand_retract_input_tangent(
          orbital_preparation_input,
          reduced_step);
  Eigen::VectorXd packed_tangent = Eigen::VectorXd::Zero(
      static_cast<Eigen::Index>(parameter_view.size()));
  const auto& differentiable_indices =
      parameter_view.differentiable_parameter_indices();
  for (Eigen::Index packed_index = 0;
       packed_index < packed_tangent.size();
       ++packed_index) {
    packed_tangent[packed_index] =
        full_tangent[differentiable_indices[packed_index]];
  }
  return packed_tangent;
}

NonredundantRetractionMetric::NonredundantRetractionMetric(
    const OrbitalChart& space,
    const SparseParameterLayout& parameter_view,
    const OrbitalPreparationInput& input)
    : space_(space),
      parameter_view_(parameter_view),
      physical_metric_(input) {}

Eigen::VectorXd NonredundantRetractionMetric::tangent(
    const Eigen::VectorXd& reduced_step) const {
  return reduced_step;
}

Eigen::VectorXd NonredundantRetractionMetric::apply(
    const Eigen::VectorXd& reduced_step) const {
  return space_.project_reduced_gradient(
      physical_metric_.apply(
          parameter_view_, space_.expand_step(reduced_step)));
}

Eigen::VectorXd NonredundantRetractionMetric::solve(
    const Eigen::VectorXd& covector) const {
  const Eigen::Index dimension = space_.reduced_size();
  if (covector.size() != dimension) {
    throw std::runtime_error(
        "retraction metric solve received an incompatible covector");
  }
  if (!covector.allFinite()) {
    throw std::runtime_error(
        "retraction metric solve received a non-finite covector");
  }
  if (dimension == 0 || covector.isZero()) {
    return Eigen::VectorXd::Zero(dimension);
  }

  const double right_hand_side_norm = covector.norm();
  const double relative_tolerance =
      std::sqrt(std::numeric_limits<double>::epsilon());
  const double residual_tolerance =
      relative_tolerance * right_hand_side_norm;

  Eigen::VectorXd solution = Eigen::VectorXd::Zero(dimension);
  Eigen::VectorXd direction = covector;
  double residual_squared = covector.squaredNorm();
  double cycle_initial_residual_norm = right_hand_side_norm;
  Eigen::Index iteration_in_cycle = 0;
  while (true) {
    Eigen::VectorXd metric_direction = apply(direction);
    double curvature = direction.dot(metric_direction);
    if (!(curvature > 0.0) || !std::isfinite(curvature)) {
      // Finite-precision CG can lose conjugacy long before an SPD operator
      // loses positivity.  Restart from the explicitly recomputed residual;
      // only failure of that steepest direction is evidence of a singular
      // reduced metric.
      direction = covector - apply(solution);
      const double restarted_residual_norm = direction.norm();
      if (restarted_residual_norm <= residual_tolerance) {
        return solution;
      }
      metric_direction = apply(direction);
      curvature = direction.dot(metric_direction);
      if (!(curvature > 0.0) || !std::isfinite(curvature)) {
        throw std::runtime_error(
            "retraction metric is singular in reduced space");
      }
      residual_squared = direction.squaredNorm();
      cycle_initial_residual_norm = restarted_residual_norm;
      iteration_in_cycle = 0;
    }

    const double step = residual_squared / curvature;
    if (!std::isfinite(step)) {
      throw std::runtime_error(
          "retraction metric solve produced a non-finite CG step");
    }
    solution.noalias() += step * direction;

    // Recompute the true residual: recursively updated CG residuals can hide
    // loss of accuracy for an ill-conditioned pullback metric.
    const Eigen::VectorXd next_residual = covector - apply(solution);
    const double next_residual_norm = next_residual.norm();
    if (!std::isfinite(next_residual_norm)) {
      throw std::runtime_error(
          "retraction metric solve produced a non-finite residual");
    }
    if (next_residual_norm <= residual_tolerance) {
      return solution;
    }

    const double next_residual_squared = next_residual.squaredNorm();
    ++iteration_in_cycle;
    if (iteration_in_cycle == dimension) {
      // The n-step termination property of CG holds only in exact arithmetic.
      // Recomputed residuals deliberately trade exact conjugacy for a truthful
      // certificate, so an ill-conditioned but nonsingular metric can require
      // another Krylov cycle.  Continue only when the completed cycle made a
      // decrease resolvable in working precision.
      const double resolved_decrease =
          cycle_initial_residual_norm - next_residual_norm;
      const double roundoff_decrease =
          std::numeric_limits<double>::epsilon() *
          std::max(cycle_initial_residual_norm, right_hand_side_norm);
      if (!(resolved_decrease > roundoff_decrease)) {
        throw std::runtime_error(
            "retraction metric CG stagnated above its residual tolerance");
      }
      direction = next_residual;
      residual_squared = next_residual_squared;
      cycle_initial_residual_norm = next_residual_norm;
      iteration_in_cycle = 0;
      continue;
    }
    const double beta = next_residual_squared / residual_squared;
    if (!std::isfinite(beta)) {
      throw std::runtime_error(
          "retraction metric solve produced a non-finite CG recurrence");
    }
    direction = next_residual + beta * direction;
    residual_squared = next_residual_squared;
  }
}

double NonredundantRetractionMetric::dual_norm(
    const Eigen::VectorXd& covector) const {
  if (covector.size() == 0) return 0.0;
  const Eigen::VectorXd riesz_gradient = solve(covector);
  const double squared_norm = covector.dot(riesz_gradient);
  const double roundoff =
      64.0 * std::numeric_limits<double>::epsilon() *
      covector.norm() * riesz_gradient.norm();
  if (!std::isfinite(squared_norm) || squared_norm < -roundoff) {
    throw std::runtime_error(
        "retraction metric produced an invalid gradient dual norm");
  }
  return std::sqrt(std::max(0.0, squared_norm));
}

double NonredundantRetractionMetric::norm(
    const Eigen::VectorXd& reduced_step) const {
  const double squared_norm = physical_metric_.squared_norm(
      parameter_view_, space_.expand_step(reduced_step));
  const double tangent_norm = std::sqrt(squared_norm);
  return std::isfinite(tangent_norm) ? tangent_norm : 0.0;
}

Eigen::VectorXd NonredundantRetractionMetric::clip_to_radius(
    const Eigen::VectorXd& reduced_step,
    double trust_radius) const {
  if (!(trust_radius > 0.0) || !std::isfinite(trust_radius)) {
    return Eigen::VectorXd::Zero(reduced_step.size());
  }
  const double tangent_norm = norm(reduced_step);
  if (!(tangent_norm > 0.0) || !std::isfinite(tangent_norm)) {
    return Eigen::VectorXd::Zero(reduced_step.size());
  }
  if (tangent_norm <= trust_radius) {
    return reduced_step;
  }
  return (trust_radius / tangent_norm) * reduced_step;
}

Eigen::VectorXd shrink_reduced_step_inside_trust_radius(
    const Eigen::VectorXd& reduced_step,
    double trust_radius,
    const NonredundantRetractionMetric& metric) {
  if (!(trust_radius > 0.0) || !std::isfinite(trust_radius)) {
    return Eigen::VectorXd::Zero(reduced_step.size());
  }
  constexpr double kInitialStepSafetyFraction = 0.95;
  const double target_radius = kInitialStepSafetyFraction * trust_radius;
  if (!(target_radius > 0.0) || !std::isfinite(target_radius)) {
    return Eigen::VectorXd::Zero(reduced_step.size());
  }
  const double tangent_norm = metric.norm(reduced_step);
  if (!(tangent_norm > 0.0) || !std::isfinite(tangent_norm)) {
    return Eigen::VectorXd::Zero(reduced_step.size());
  }
  if (tangent_norm < target_radius) {
    return reduced_step;
  }
  return (target_radius / tangent_norm) * reduced_step;
}

}  // namespace xmvb::vb
