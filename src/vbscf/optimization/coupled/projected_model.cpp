#include "vbscf/optimization/coupled/projected_model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include <Eigen/LU>

namespace xmvb::vb {
namespace {

double roundoff(int dimension) {
  return std::numeric_limits<double>::epsilon() * std::max(1, dimension);
}

void validate_symmetric(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix,
    const char* label) {
  const double scale = std::max(1.0, matrix.stableNorm());
  if ((matrix - matrix.transpose()).stableNorm() >
      roundoff(matrix.rows()) * scale) {
    throw std::invalid_argument(
        std::string("projected ") + label + " is not symmetric");
  }
}

}  // namespace

CoupledProjectedModelResult solve_coupled_projected_model(
    const Eigen::Ref<const Eigen::MatrixXd>& orbital_hessian,
    const Eigen::Ref<const Eigen::MatrixXd>& response_hessian,
    const Eigen::Ref<const Eigen::MatrixXd>& coupling,
    const Eigen::Ref<const Eigen::MatrixXd>& orbital_metric,
    const Eigen::Ref<const Eigen::VectorXd>& orbital_gradient,
    const Eigen::Ref<const Eigen::VectorXd>& response_gradient,
    double trust_radius) {
  const int n_orbital = orbital_gradient.size();
  const int n_response = response_gradient.size();
  if (n_orbital <= 0 || n_response <= 0 ||
      orbital_hessian.rows() != n_orbital ||
      orbital_hessian.cols() != n_orbital ||
      response_hessian.rows() != n_response ||
      response_hessian.cols() != n_response ||
      coupling.rows() != n_response || coupling.cols() != n_orbital ||
      orbital_metric.rows() != n_orbital ||
      orbital_metric.cols() != n_orbital ||
      !orbital_hessian.allFinite() || !response_hessian.allFinite() ||
      !coupling.allFinite() || !orbital_metric.allFinite() ||
      !orbital_gradient.allFinite() || !response_gradient.allFinite() ||
      !(trust_radius > 0.0) || !std::isfinite(trust_radius)) {
    throw std::invalid_argument("invalid coupled projected model");
  }
  validate_symmetric(orbital_hessian, "orbital Hessian");
  validate_symmetric(response_hessian, "response Hessian");
  validate_symmetric(orbital_metric, "orbital metric");

  CoupledProjectedModelResult result;
  result.response_baseline = Eigen::VectorXd::Zero(n_response);
  result.response_lift = Eigen::MatrixXd::Zero(n_response, n_orbital);
  result.reduced_gradient = Eigen::VectorXd::Zero(n_orbital);
  result.reduced_hessian = Eigen::MatrixXd::Zero(n_orbital, n_orbital);
  result.orbital_coordinates = Eigen::VectorXd::Zero(n_orbital);
  result.response_coordinates = Eigen::VectorXd::Zero(n_response);
  result.orbital_kkt_residual = Eigen::VectorXd::Zero(n_orbital);
  result.response_kkt_residual = Eigen::VectorXd::Zero(n_response);

  const Eigen::MatrixXd symmetric_orbital_hessian =
      0.5 * (orbital_hessian + orbital_hessian.transpose());
  const Eigen::MatrixXd symmetric_response_hessian =
      0.5 * (response_hessian + response_hessian.transpose());
  const Eigen::MatrixXd symmetric_orbital_metric =
      0.5 * (orbital_metric + orbital_metric.transpose());
  Eigen::FullPivLU<Eigen::MatrixXd> response_factor(
      symmetric_response_hessian);
  if (!response_factor.isInvertible()) {
    result.status =
        CoupledProjectedModelStatus::ResponseProjectionSingular;
    return result;
  }

  Eigen::MatrixXd response_right_hand_sides(n_response, n_orbital + 1);
  response_right_hand_sides.col(0) = -response_gradient;
  response_right_hand_sides.rightCols(n_orbital) = -coupling;
  const Eigen::MatrixXd response_solutions =
      response_factor.solve(response_right_hand_sides);
  if (!response_solutions.allFinite()) return result;
  result.response_baseline = response_solutions.col(0);
  result.response_lift = response_solutions.rightCols(n_orbital);
  result.reduced_gradient =
      orbital_gradient + coupling.transpose() * result.response_baseline;
  result.reduced_hessian =
      symmetric_orbital_hessian + coupling.transpose() * result.response_lift;
  result.reduced_hessian =
      0.5 * (result.reduced_hessian + result.reduced_hessian.transpose());
  if (!result.response_baseline.allFinite() ||
      !result.response_lift.allFinite() ||
      !result.reduced_gradient.allFinite() ||
      !result.reduced_hessian.allFinite()) {
    return result;
  }

  result.orbital_solution = solve_projected_generalized_trust_region(
      result.reduced_hessian,
      symmetric_orbital_metric,
      result.reduced_gradient,
      trust_radius);
  if (!result.orbital_solution.converged()) {
    result.status =
        CoupledProjectedModelStatus::OrbitalTrustRegionFailure;
    return result;
  }

  result.orbital_coordinates = result.orbital_solution.coordinates;
  result.response_coordinates = result.response_baseline +
      result.response_lift * result.orbital_coordinates;
  const Eigen::VectorXd metric_image =
      symmetric_orbital_metric * result.orbital_coordinates;
  result.orbital_kkt_residual = orbital_gradient +
      symmetric_orbital_hessian * result.orbital_coordinates +
      coupling.transpose() * result.response_coordinates +
      result.orbital_solution.shift * metric_image;
  result.response_kkt_residual = response_gradient +
      coupling * result.orbital_coordinates +
      symmetric_response_hessian * result.response_coordinates;
  result.orbital_kkt_residual_norm =
      result.orbital_kkt_residual.stableNorm();
  result.response_kkt_residual_norm =
      result.response_kkt_residual.stableNorm();

  result.response_baseline_model_change =
      response_gradient.dot(result.response_baseline) +
      0.5 * result.response_baseline.dot(
          symmetric_response_hessian * result.response_baseline);
  result.reduced_incremental_model_change =
      result.reduced_gradient.dot(result.orbital_coordinates) +
      0.5 * result.orbital_coordinates.dot(
          result.reduced_hessian * result.orbital_coordinates);
  result.total_model_change = result.response_baseline_model_change +
      result.reduced_incremental_model_change;
  result.total_predicted_decrease = -result.total_model_change;
  if (!result.orbital_coordinates.allFinite() ||
      !result.response_coordinates.allFinite() ||
      !result.orbital_kkt_residual.allFinite() ||
      !result.response_kkt_residual.allFinite() ||
      !std::isfinite(result.orbital_kkt_residual_norm) ||
      !std::isfinite(result.response_kkt_residual_norm) ||
      !std::isfinite(result.response_baseline_model_change) ||
      !std::isfinite(result.reduced_incremental_model_change) ||
      !std::isfinite(result.total_model_change) ||
      !std::isfinite(result.total_predicted_decrease)) {
    result.status = CoupledProjectedModelStatus::NumericalFailure;
    return result;
  }
  result.status = CoupledProjectedModelStatus::Converged;
  return result;
}

}  // namespace xmvb::vb
