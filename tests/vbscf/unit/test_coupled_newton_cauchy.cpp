#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/LU>

#include "vbscf/optimization/coupled/cauchy.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void require_close(
    const Eigen::Ref<const Eigen::VectorXd>& actual,
    const Eigen::Ref<const Eigen::VectorXd>& expected,
    double tolerance,
    const char* message) {
  if (actual.size() != expected.size() ||
      (actual - expected).stableNorm() >
          tolerance * std::max(1.0, expected.stableNorm())) {
    throw std::runtime_error(message);
  }
}

xmvb::vb::CoupledNewtonOperator make_operator(
    const Eigen::Matrix2d& orbital_hessian,
    const Eigen::Matrix2d& coupling,
    const Eigen::Matrix2d& response_hessian,
    const Eigen::Matrix2d& orbital_metric) {
  return xmvb::vb::CoupledNewtonOperator(
      2,
      xmvb::vb::SelectedSubspaceResponseLayout(
          1, {xmvb::vb::SelectedStateCluster{1, 1.0}}),
      xmvb::vb::CoupledNewtonActions{
          [orbital_hessian](const auto& directions) {
            return (orbital_hessian * directions).eval();
          },
          [coupling](const auto& directions) {
            return (coupling * directions).eval();
          },
          [coupling](const auto& directions) {
            return (coupling.transpose() * directions).eval();
          },
          [response_hessian](const auto& directions) {
            return (response_hessian * directions).eval();
          },
          [orbital_metric](const auto& directions) {
            return (orbital_metric * directions).eval();
          }});
}

}  // namespace

int main() {
  try {
    Eigen::Matrix2d orbital_hessian;
    orbital_hessian << 4.0, 0.2,
                       0.2, 2.0;
    Eigen::Matrix2d coupling;
    coupling << 0.3, -0.1,
                0.2,  0.4;
    Eigen::Matrix2d response_hessian;
    response_hessian << -2.0, 0.1,
                         0.1, 1.5;
    Eigen::Matrix2d metric;
    metric << 1.5, 0.0,
              0.0, 0.7;
    const auto coupled_operator = make_operator(
        orbital_hessian, coupling, response_hessian, metric);
    const Eigen::Vector2d corrected_gradient(0.8, -0.4);
    const double radius = 0.25;
    xmvb::vb::MinresOptions response_options;
    response_options.relative_residual_tolerance = 1.0e-12;
    const auto inverse_metric = [metric](const Eigen::VectorXd& vector) {
      return metric.diagonal().cwiseInverse().asDiagonal() * vector;
    };

    const xmvb::vb::RelaxedCauchyResult result =
        xmvb::vb::build_relaxed_cauchy_incumbent(
            coupled_operator,
            corrected_gradient,
            radius,
            response_options,
            inverse_metric);
    require(result.certified(),
            "relaxed Cauchy decrease was not certified");
    require(result.response_linear_result.converged() &&
                result.response_linear_result.residual_norm <=
                    result.response_linear_result.residual_target,
            "response lift lacks an explicit MINRES certificate");

    const Eigen::Vector2d reference_direction =
        -metric.inverse() * corrected_gradient;
    const Eigen::Vector2d response_rhs =
        -coupling * reference_direction;
    const Eigen::Vector2d reference_response =
        response_hessian.fullPivLu().solve(response_rhs);
    const double reference_directional_derivative =
        corrected_gradient.dot(reference_direction);
    const double reference_stationary_curvature = reference_direction.dot(
        orbital_hessian * reference_direction +
        coupling.transpose() * reference_response);
    const Eigen::Vector2d reference_response_residual =
        coupling * reference_direction +
        response_hessian * reference_response;
    const double reference_curvature =
        reference_stationary_curvature +
        reference_response.dot(reference_response_residual);
    const double reference_norm = std::sqrt(
        reference_direction.dot(metric * reference_direction));
    const double reference_maximum_length = radius / reference_norm;
    const double reference_length = reference_curvature <= 0.0
        ? reference_maximum_length
        : std::min(
              reference_maximum_length,
              -reference_directional_derivative / reference_curvature);
    const double reference_prediction =
        -reference_length * reference_directional_derivative -
        0.5 * reference_length * reference_length * reference_curvature;
    require_close(
        result.orbital_direction,
        reference_direction,
        2.0e-12,
        "inverse-metric Cauchy direction is inaccurate");
    require_close(
        result.response_direction,
        reference_response,
        2.0e-11,
        "matrix-free response lift is inaccurate");
    require_close(
        result.orbital_step,
        reference_length * reference_direction,
        2.0e-11,
        "relaxed Cauchy orbital step is inaccurate");
    require_close(
        result.response_step,
        reference_length * reference_response,
        2.0e-11,
        "relaxed Cauchy response step is inaccurate");
    require(std::abs(result.stationary_limit_curvature -
                     reference_stationary_curvature) <= 2.0e-12 &&
                std::abs(result.coupled_ray_curvature -
                         reference_curvature) <= 2.0e-12 &&
                std::abs(result.predicted_decrease - reference_prediction) <=
                    2.0e-12,
            "relaxed Cauchy model scalars are inaccurate");
    require(result.predicted_decrease > 0.0 &&
                result.orbital_step.dot(metric * result.orbital_step) <=
                    radius * radius,
            "certified Cauchy step is not feasible or decreasing");
    require_close(
        result.response_linear_result.residual,
        response_rhs - response_hessian * result.response_direction,
        2.0e-13,
        "reported response residual is not explicit");
    require_close(
        result.remaining_response_residual,
        coupling * result.orbital_direction +
            response_hessian * result.response_direction,
        2.0e-13,
        "remaining response-equation residual has the wrong sign");
    require(std::abs(
                result.coupled_ray_curvature -
                (result.stationary_limit_curvature +
                 result.response_direction.dot(
                     result.remaining_response_residual))) <= 2.0e-13,
            "finite-response curvature identity is inconsistent");

    xmvb::vb::MinresOptions insufficient_response_options = response_options;
    insufficient_response_options.maximum_iterations = 1;
    const auto uncertified =
        xmvb::vb::build_relaxed_cauchy_incumbent(
            coupled_operator,
            corrected_gradient,
            radius,
            insufficient_response_options,
            inverse_metric);
    require(!uncertified.certified() &&
                uncertified.stop_reason ==
                    xmvb::vb::RelaxedCauchyStopReason::ResponseLiftFailure &&
                uncertified.orbital_step.isZero(0.0) &&
                uncertified.response_step.isZero(0.0),
            "an uncertified response lift produced a Cauchy fallback step");

    xmvb::vb::MinresOptions finite_response_options = response_options;
    finite_response_options.relative_residual_tolerance = 0.9;
    finite_response_options.maximum_iterations = 1;
    const auto finite_response =
        xmvb::vb::build_relaxed_cauchy_incumbent(
            coupled_operator,
            corrected_gradient,
            radius,
            finite_response_options,
            inverse_metric);
    require(finite_response.certified() &&
                finite_response.response_linear_result.residual_norm > 0.0,
            "finite certified response residual was not exercised");
    const double explicit_finite_curvature =
        finite_response.orbital_direction.dot(
            orbital_hessian * finite_response.orbital_direction) +
        2.0 * finite_response.orbital_direction.dot(
            coupling.transpose() * finite_response.response_direction) +
        finite_response.response_direction.dot(
            response_hessian * finite_response.response_direction);
    require(std::abs(finite_response.coupled_ray_curvature -
                     explicit_finite_curvature) <= 2.0e-13,
            "finite-solve prediction omitted the response residual correction");

    // The gradient Krylov ray does not see the second orbital direction, whose
    // curvature is strongly negative. The layer must still return its positive
    // Cauchy decrease without claiming a global PSD or trust-region solution.
    Eigen::Matrix2d hidden_indefinite_hessian = Eigen::Matrix2d::Zero();
    hidden_indefinite_hessian.diagonal() << 1.0, -100.0;
    const Eigen::Matrix2d zero_coupling = Eigen::Matrix2d::Zero();
    const Eigen::Matrix2d identity = Eigen::Matrix2d::Identity();
    const auto hidden_indefinite_operator = make_operator(
        hidden_indefinite_hessian,
        zero_coupling,
        identity,
        identity);
    const auto hidden_result =
        xmvb::vb::build_relaxed_cauchy_incumbent(
            hidden_indefinite_operator,
            Eigen::Vector2d(0.1, 0.0),
            1.0,
            response_options,
            [](const Eigen::VectorXd& vector) { return vector; });
    require(hidden_result.certified() &&
                hidden_result.stop_reason ==
                    xmvb::vb::RelaxedCauchyStopReason::
                        CauchyDecreaseCertified &&
                hidden_result.predicted_decrease > 0.0,
            "hidden negative curvature destroyed the Cauchy guarantee");
    require(hidden_indefinite_hessian(1, 1) < 0.0 &&
                hidden_result.orbital_step[1] == 0.0,
            "synthetic negative curvature is not hidden from the Cauchy ray");

    std::cout << "relaxed coupled Cauchy incumbent: passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
