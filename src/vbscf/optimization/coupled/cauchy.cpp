#include "vbscf/optimization/coupled/cauchy.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace xmvb::vb {

RelaxedCauchyResult build_relaxed_cauchy_incumbent(
    const CoupledNewtonOperator& coupled_operator,
    const Eigen::Ref<const Eigen::VectorXd>& corrected_orbital_gradient,
    double trust_radius,
    const MinresOptions& response_options,
    const SymmetricOperatorAction& apply_inverse_orbital_metric,
    const SymmetricOperatorAction& apply_response_inverse_preconditioner) {
  const int n_orbitals = coupled_operator.n_orbital_coordinates();
  const int n_response = coupled_operator.n_response_coordinates();
  if (corrected_orbital_gradient.size() != n_orbitals ||
      !corrected_orbital_gradient.allFinite() ||
      !(trust_radius > 0.0) || !std::isfinite(trust_radius) ||
      !apply_inverse_orbital_metric) {
    throw std::invalid_argument("invalid relaxed Cauchy problem");
  }

  RelaxedCauchyResult result;
  result.orbital_step = Eigen::VectorXd::Zero(n_orbitals);
  result.response_step = Eigen::VectorXd::Zero(n_response);

  result.orbital_direction =
      -apply_inverse_orbital_metric(corrected_orbital_gradient);
  if (result.orbital_direction.size() != n_orbitals ||
      !result.orbital_direction.allFinite()) {
    result.stop_reason = RelaxedCauchyStopReason::InvalidInverseMetric;
    return result;
  }
  result.directional_derivative =
      corrected_orbital_gradient.dot(result.orbital_direction);
  if (!(result.directional_derivative < 0.0) ||
      !std::isfinite(result.directional_derivative)) {
    result.stop_reason = RelaxedCauchyStopReason::NonDescentDirection;
    return result;
  }

  result.orbital_metric_image = coupled_operator.apply_orbital_metric(
      result.orbital_direction).col(0);
  const double squared_orbital_norm =
      result.orbital_direction.dot(result.orbital_metric_image);
  if (!(squared_orbital_norm > 0.0) ||
      !std::isfinite(squared_orbital_norm)) {
    result.stop_reason = RelaxedCauchyStopReason::InvalidOrbitalMetric;
    return result;
  }
  result.orbital_direction_norm = std::sqrt(squared_orbital_norm);
  result.maximum_step_length =
      trust_radius / result.orbital_direction_norm;

  const Eigen::VectorXd response_rhs =
      -coupled_operator.apply_orbital_to_response(
          result.orbital_direction).col(0);
  result.response_linear_result = solve_symmetric_minres(
      [&coupled_operator](const Eigen::VectorXd& direction) {
        return coupled_operator.apply_response_hessian(direction).col(0).eval();
      },
      response_rhs,
      response_options,
      apply_response_inverse_preconditioner);
  result.response_direction = result.response_linear_result.solution;
  result.remaining_response_residual =
      -result.response_linear_result.residual;
  if (!result.response_linear_result.converged()) {
    result.stop_reason = RelaxedCauchyStopReason::ResponseLiftFailure;
    return result;
  }

  const Eigen::VectorXd relaxed_hessian_direction =
      coupled_operator.apply_orbital_hessian(
          result.orbital_direction).col(0) +
      coupled_operator.apply_response_to_orbital(
          result.response_direction).col(0);
  result.stationary_limit_curvature =
      result.orbital_direction.dot(relaxed_hessian_direction);
  result.coupled_ray_curvature =
      result.stationary_limit_curvature +
      result.response_direction.dot(result.remaining_response_residual);
  if (!std::isfinite(result.stationary_limit_curvature) ||
      !std::isfinite(result.coupled_ray_curvature)) {
    result.stop_reason =
        RelaxedCauchyStopReason::NonPositivePredictedDecrease;
    return result;
  }

  result.step_length = result.maximum_step_length;
  if (result.coupled_ray_curvature > 0.0) {
    result.step_length = std::min(
        result.maximum_step_length,
        -result.directional_derivative / result.coupled_ray_curvature);
  }
  result.predicted_decrease =
      -result.step_length * result.directional_derivative -
      0.5 * result.step_length * result.step_length *
          result.coupled_ray_curvature;
  if (!(result.step_length > 0.0) ||
      !std::isfinite(result.step_length) ||
      !(result.predicted_decrease > 0.0) ||
      !std::isfinite(result.predicted_decrease)) {
    result.stop_reason =
        RelaxedCauchyStopReason::NonPositivePredictedDecrease;
    return result;
  }

  result.orbital_step = result.step_length * result.orbital_direction;
  result.response_step = result.step_length * result.response_direction;
  result.stop_reason =
      RelaxedCauchyStopReason::CauchyDecreaseCertified;
  return result;
}

}  // namespace xmvb::vb
