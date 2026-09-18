#include "vbscf/optimization/coupled/projected_trust.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

namespace xmvb::vb {
namespace {

double roundoff(int dimension) {
  return std::numeric_limits<double>::epsilon() *
      std::max(1, dimension);
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

struct DiagonalSolution {
  Eigen::VectorXd step;
  double shift = 0.0;
  ProjectedTrustStatus status =
      ProjectedTrustStatus::NumericalFailure;
};

DiagonalSolution solve_diagonal_problem(
    const Eigen::Ref<const Eigen::VectorXd>& eigenvalues,
    const Eigen::Ref<const Eigen::VectorXd>& gradient,
    double radius) {
  const int dimension = eigenvalues.size();
  DiagonalSolution result;
  result.step = Eigen::VectorXd::Zero(dimension);
  const double energy_scale = std::max(
      eigenvalues.cwiseAbs().maxCoeff(),
      gradient.stableNorm() / radius);
  if (energy_scale == 0.0) {
    result.status = ProjectedTrustStatus::InteriorGlobal;
    result.shift = 0.0;
    return result;
  }
  if (!std::isfinite(energy_scale)) return result;

  const Eigen::VectorXd values = eigenvalues / energy_scale;
  const Eigen::VectorXd scaled_gradient =
      gradient / (energy_scale * radius);
  Eigen::Index minimum_index = 0;
  const double minimum = values.minCoeff(&minimum_index);
  const double lower_shift = std::max(0.0, -minimum);
  const Eigen::VectorXd endpoint_denominators =
      values.array() + lower_shift;
  const double spectral_scale = std::max(
      1.0,
      values.cwiseAbs().maxCoeff() + lower_shift);
  const double spectral_tolerance =
      roundoff(dimension) * spectral_scale;
  const double gradient_tolerance =
      roundoff(dimension) *
      std::max(1.0, scaled_gradient.stableNorm());

  Eigen::VectorXd endpoint = Eigen::VectorXd::Zero(dimension);
  bool endpoint_compatible = true;
  for (int index = 0; index < dimension; ++index) {
    if (endpoint_denominators[index] <= spectral_tolerance) {
      if (std::abs(scaled_gradient[index]) > gradient_tolerance) {
        endpoint_compatible = false;
      }
    } else {
      endpoint[index] =
          -scaled_gradient[index] / endpoint_denominators[index];
    }
  }
  const double endpoint_norm = endpoint.stableNorm();
  const double norm_tolerance = roundoff(dimension);
  if (endpoint_compatible && endpoint_norm <= 1.0 + norm_tolerance) {
    if (lower_shift > 0.0) {
      const double remaining_squared = std::max(
          0.0,
          (1.0 - endpoint_norm) * (1.0 + endpoint_norm));
      const double boundary_component = std::sqrt(remaining_squared);
      endpoint[minimum_index] = scaled_gradient[minimum_index] > 0.0
          ? -boundary_component
          : boundary_component;
      result.status = ProjectedTrustStatus::HardCaseGlobal;
    } else {
      result.status = endpoint_norm >= 1.0 - norm_tolerance
          ? ProjectedTrustStatus::BoundaryGlobal
          : ProjectedTrustStatus::InteriorGlobal;
    }
    result.step = radius * endpoint;
    result.shift = energy_scale * lower_shift;
    return result;
  }

  auto solve_at_excess = [&](double excess, Eigen::VectorXd* step) {
    step->resize(dimension);
    for (int index = 0; index < dimension; ++index) {
      const double denominator = endpoint_denominators[index] + excess;
      if (!(denominator > 0.0) || !std::isfinite(denominator)) return false;
      (*step)[index] = -scaled_gradient[index] / denominator;
    }
    return step->allFinite();
  };

  double lower_excess = 0.0;
  double upper_excess = scaled_gradient.stableNorm();
  if (!(upper_excess > 0.0) || !std::isfinite(upper_excess)) return result;
  Eigen::VectorXd feasible;
  while (!solve_at_excess(upper_excess, &feasible) ||
         feasible.stableNorm() > 1.0) {
    upper_excess *= 2.0;
    if (!(upper_excess > 0.0) || !std::isfinite(upper_excess)) return result;
  }

  constexpr int numerical_limit =
      std::numeric_limits<double>::max_exponent -
      std::numeric_limits<double>::min_exponent +
      std::numeric_limits<double>::digits;
  for (int iteration = 0; iteration < numerical_limit; ++iteration) {
    const double feasible_norm = feasible.stableNorm();
    if (std::abs(feasible_norm - 1.0) <= norm_tolerance) break;
    const double trial_excess =
        lower_excess + 0.5 * (upper_excess - lower_excess);
    if (trial_excess == lower_excess || trial_excess == upper_excess) break;
    Eigen::VectorXd trial;
    if (!solve_at_excess(trial_excess, &trial) ||
        trial.stableNorm() > 1.0) {
      lower_excess = trial_excess;
    } else {
      upper_excess = trial_excess;
      feasible = std::move(trial);
    }
  }
  if (std::abs(feasible.stableNorm() - 1.0) >
      8.0 * norm_tolerance) {
    return result;
  }
  result.step = radius * feasible;
  result.shift = energy_scale * (lower_shift + upper_excess);
  result.status = ProjectedTrustStatus::BoundaryGlobal;
  return result;
}

}  // namespace

ProjectedTrustResult solve_projected_generalized_trust_region(
    const Eigen::Ref<const Eigen::MatrixXd>& projected_hessian,
    const Eigen::Ref<const Eigen::MatrixXd>& projected_metric,
    const Eigen::Ref<const Eigen::VectorXd>& projected_gradient,
    double trust_radius) {
  const int dimension = projected_gradient.size();
  if (dimension <= 0 || projected_hessian.rows() != dimension ||
      projected_hessian.cols() != dimension ||
      projected_metric.rows() != dimension ||
      projected_metric.cols() != dimension ||
      !projected_hessian.allFinite() || !projected_metric.allFinite() ||
      !projected_gradient.allFinite() || !(trust_radius > 0.0) ||
      !std::isfinite(trust_radius)) {
    throw std::invalid_argument(
        "invalid projected generalized trust-region problem");
  }
  validate_symmetric(projected_hessian, "Hessian");
  validate_symmetric(projected_metric, "orbital metric");

  ProjectedTrustResult result;
  result.coordinates = Eigen::VectorXd::Zero(dimension);
  const Eigen::MatrixXd symmetric_hessian =
      0.5 * (projected_hessian + projected_hessian.transpose());
  const Eigen::MatrixXd symmetric_metric =
      0.5 * (projected_metric + projected_metric.transpose());
  Eigen::LLT<Eigen::MatrixXd> metric_factor(symmetric_metric);
  if (metric_factor.info() != Eigen::Success) {
    result.status = ProjectedTrustStatus::InvalidMetric;
    return result;
  }
  const Eigen::MatrixXd inverse_lower =
      metric_factor.matrixL().solve(
          Eigen::MatrixXd::Identity(dimension, dimension));
  if (!inverse_lower.allFinite()) {
    result.status = ProjectedTrustStatus::InvalidMetric;
    return result;
  }
  Eigen::MatrixXd whitened_hessian =
      inverse_lower * symmetric_hessian * inverse_lower.transpose();
  whitened_hessian =
      0.5 * (whitened_hessian + whitened_hessian.transpose());
  const Eigen::VectorXd whitened_gradient =
      inverse_lower * projected_gradient;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectral_solver(
      whitened_hessian);
  if (spectral_solver.info() != Eigen::Success ||
      !spectral_solver.eigenvalues().allFinite() ||
      !spectral_solver.eigenvectors().allFinite()) {
    result.status = ProjectedTrustStatus::NumericalFailure;
    return result;
  }
  result.minimum_ritz_value = spectral_solver.eigenvalues().minCoeff();

  const Eigen::VectorXd spectral_gradient =
      spectral_solver.eigenvectors().transpose() * whitened_gradient;
  const DiagonalSolution diagonal = solve_diagonal_problem(
      spectral_solver.eigenvalues(),
      spectral_gradient,
      trust_radius);
  if (diagonal.status == ProjectedTrustStatus::NumericalFailure) return result;
  result.status = diagonal.status;
  result.shift = diagonal.shift;
  result.coordinates = inverse_lower.transpose() *
      spectral_solver.eigenvectors() * diagonal.step;
  if (!result.coordinates.allFinite() || !std::isfinite(result.shift)) {
    result.status = ProjectedTrustStatus::NumericalFailure;
    return result;
  }

  const Eigen::VectorXd metric_image =
      symmetric_metric * result.coordinates;
  const double metric_norm_squared = result.coordinates.dot(metric_image);
  if (!(metric_norm_squared >= 0.0) || !std::isfinite(metric_norm_squared)) {
    result.status = ProjectedTrustStatus::InvalidMetric;
    return result;
  }
  result.metric_norm = std::sqrt(metric_norm_squared);
  const Eigen::VectorXd stationarity =
      symmetric_hessian * result.coordinates + projected_gradient +
      result.shift * metric_image;
  result.stationarity_residual = stationarity.stableNorm();
  const double stationarity_scale =
      (symmetric_hessian * result.coordinates).stableNorm() +
      projected_gradient.stableNorm() +
      result.shift * metric_image.stableNorm();
  result.stationarity_backward_error = stationarity_scale > 0.0
      ? result.stationarity_residual / stationarity_scale
      : result.stationarity_residual == 0.0
          ? 0.0
          : std::numeric_limits<double>::infinity();
  result.feasibility_violation =
      std::max(0.0, result.metric_norm - trust_radius);
  result.complementarity_residual =
      result.shift * std::abs(result.metric_norm - trust_radius);
  result.minimum_shifted_ritz_value =
      result.minimum_ritz_value + result.shift;
  const double model_value =
      projected_gradient.dot(result.coordinates) +
      0.5 * result.coordinates.dot(
          symmetric_hessian * result.coordinates);
  result.predicted_decrease = -model_value;
  return result;
}

}  // namespace xmvb::vb
