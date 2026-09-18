#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/LU>

#include "vbscf/optimization/coupled/model_reduction.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void require_close(
    double actual,
    double expected,
    double tolerance,
    const char* message) {
  if (std::abs(actual - expected) >
      tolerance * std::max({1.0, std::abs(actual), std::abs(expected)})) {
    throw std::runtime_error(message);
  }
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
    const Eigen::Matrix2d& response_hessian) {
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
          {}});
}

}  // namespace

int main() {
  try {
    Eigen::Matrix2d orbital_hessian;
    orbital_hessian << 2.4, 0.3,
                       0.3, 1.6;
    Eigen::Matrix2d coupling;
    coupling << 0.4, -0.2,
                0.1,  0.5;
    Eigen::Matrix2d response_hessian;
    response_hessian << -1.7, 0.25,
                         0.25, 1.3;
    const auto coupled_operator = make_operator(
        orbital_hessian, coupling, response_hessian);

    const Eigen::Vector2d accepted_gradient(0.8, -0.6);
    const Eigen::Vector2d structure_residual(0.3, -0.45);
    const Eigen::Vector2d exact_correction =
        -response_hessian.fullPivLu().solve(structure_residual);
    const Eigen::Vector2d response_correction =
        exact_correction + Eigen::Vector2d(0.12, -0.08);
    const Eigen::Vector2d remaining_residual =
        structure_residual + response_hessian * response_correction;
    const Eigen::Vector2d orbital_step(-0.11, 0.07);
    const Eigen::Vector2d response_step(0.05, 0.09);
    Eigen::Vector4d incremental_step;
    incremental_step << orbital_step, response_step;
    const Eigen::VectorXd incremental_image =
        coupled_operator.apply(incremental_step);

    const auto reduction = xmvb::vb::evaluate_coupled_model_reduction(
        coupled_operator,
        accepted_gradient,
        structure_residual,
        response_correction,
        remaining_residual,
        orbital_step,
        response_step,
        incremental_image);

    Eigen::Matrix4d coupled_hessian;
    coupled_hessian << orbital_hessian, coupling.transpose(),
                       coupling, response_hessian;
    const Eigen::Vector2d total_response =
        response_correction + response_step;
    Eigen::Vector4d total_step;
    total_step << orbital_step, total_response;
    Eigen::Vector4d accepted_coupled_gradient;
    accepted_coupled_gradient << accepted_gradient, structure_residual;
    const double direct_total_change =
        accepted_coupled_gradient.dot(total_step) +
        0.5 * total_step.dot(coupled_hessian * total_step);
    const double direct_baseline_change =
        structure_residual.dot(response_correction) +
        0.5 * response_correction.dot(
            response_hessian * response_correction);
    const Eigen::Vector2d corrected_gradient =
        accepted_gradient + coupling.transpose() * response_correction;
    const double direct_incremental_change =
        corrected_gradient.dot(orbital_step) +
        remaining_residual.dot(response_step) +
        0.5 * incremental_step.dot(coupled_hessian * incremental_step);

    require_close(
        reduction.corrected_orbital_gradient,
        corrected_gradient,
        2.0e-14,
        "defect-corrected orbital gradient is inaccurate");
    require_close(
        reduction.baseline_defect_model_change,
        direct_baseline_change,
        2.0e-14,
        "baseline defect model change is inaccurate");
    require_close(
        reduction.incremental_model_change,
        direct_incremental_change,
        2.0e-14,
        "incremental model change is inaccurate");
    require_close(
        reduction.total_model_change,
        direct_total_change,
        2.0e-14,
        "total model change does not equal the accepted-point quadratic");
    require_close(
        reduction.total_predicted_decrease,
        -direct_total_change,
        2.0e-14,
        "total predicted decrease has the wrong sign or baseline");
    require_close(
        reduction.remaining_structure_linear_change,
        remaining_residual.dot(response_step),
        2.0e-14,
        "finite structure residual contribution was omitted");

    // The stationary-response shortcuts are not exact for finite e0. Their
    // joint error contains both the baseline and incremental e0 terms.
    const double omitted_defect_total =
        0.5 * structure_residual.dot(response_correction) +
        corrected_gradient.dot(orbital_step) +
        reduction.incremental_quadratic_change;
    const double exact_omission_error =
        0.5 * response_correction.dot(remaining_residual) +
        remaining_residual.dot(response_step);
    require(std::abs(exact_omission_error) > 1.0e-4,
            "synthetic finite defect cannot detect omitted e0 terms");
    require_close(
        reduction.total_model_change - omitted_defect_total,
        exact_omission_error,
        2.0e-14,
        "finite e0 omission identity is inconsistent");

    const Eigen::Vector2d converged_residual =
        structure_residual + response_hessian * exact_correction;
    const auto converged = xmvb::vb::evaluate_coupled_model_reduction(
        coupled_operator,
        accepted_gradient,
        structure_residual,
        exact_correction,
        converged_residual,
        orbital_step,
        response_step,
        incremental_image);
    require(converged_residual.stableNorm() < 1.0e-14,
            "reference structure correction did not converge");
    require_close(
        converged.baseline_defect_model_change,
        0.5 * structure_residual.dot(exact_correction),
        2.0e-14,
        "converged defect baseline did not reduce to the stationary formula");
    require_close(
        converged.incremental_model_change,
        converged.corrected_orbital_gradient.dot(orbital_step) +
            converged.incremental_quadratic_change,
        2.0e-14,
        "converged incremental model retained a spurious defect term");

    bool dimension_rejected = false;
    try {
      const Eigen::VectorXd short_response = Eigen::VectorXd::Zero(1);
      (void)xmvb::vb::evaluate_coupled_model_reduction(
          coupled_operator,
          accepted_gradient,
          structure_residual,
          response_correction,
          remaining_residual,
          orbital_step,
          short_response,
          incremental_image);
    } catch (const std::invalid_argument&) {
      dimension_rejected = true;
    }
    require(dimension_rejected,
            "dimensionally inconsistent model inputs were accepted");

    bool nonfinite_rejected = false;
    try {
      Eigen::Vector2d nonfinite_step = response_step;
      nonfinite_step[0] = std::numeric_limits<double>::quiet_NaN();
      (void)xmvb::vb::evaluate_coupled_model_reduction(
          coupled_operator,
          accepted_gradient,
          structure_residual,
          response_correction,
          remaining_residual,
          orbital_step,
          nonfinite_step,
          incremental_image);
    } catch (const std::invalid_argument&) {
      nonfinite_rejected = true;
    }
    require(nonfinite_rejected,
            "non-finite model inputs were accepted");

    std::cout << "Coupled model-reduction tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Coupled model-reduction test failed: "
              << error.what() << '\n';
    return 1;
  }
}
