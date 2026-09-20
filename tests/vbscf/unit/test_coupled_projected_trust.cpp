#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#include "vbscf/optimization/coupled/projected_trust.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void require_close(
    const Eigen::VectorXd& actual,
    const Eigen::VectorXd& reference,
    double tolerance,
    const char* message) {
  require(
      (actual - reference).stableNorm() <=
          tolerance * std::max(1.0, reference.stableNorm()),
      message);
}

void certify(
    const Eigen::MatrixXd& hessian,
    const Eigen::MatrixXd& metric,
    const Eigen::VectorXd& gradient,
    double radius,
    const xmvb::vb::ProjectedTrustResult& result,
    xmvb::vb::ProjectedTrustStatus expected_status) {
  require(result.converged() && result.status == expected_status,
          "projected trust-region status is wrong");
  require(result.scope ==
              xmvb::vb::ProjectedTrustScope::SuppliedProjectedSubspace,
          "projected solver overclaimed its global-optimality scope");
  const Eigen::VectorXd metric_image = metric * result.coordinates;
  const double norm = std::sqrt(result.coordinates.dot(metric_image));
  const Eigen::VectorXd stationarity =
      hessian * result.coordinates + gradient + result.shift * metric_image;
  const double scale = std::max({
      1.0,
      gradient.stableNorm(),
      (hessian.stableNorm() + result.shift * metric.stableNorm()) *
          result.coordinates.stableNorm()});
  require(result.shift >= 0.0 && norm <= radius * (1.0 + 1.0e-12),
          "projected trust-region solution is infeasible");
  require(stationarity.stableNorm() <= 2.0e-12 * scale,
          "projected trust-region stationarity is inaccurate");
  require(result.stationarity_residual <= 2.0e-12 * scale,
          "reported stationarity residual is inaccurate");
  require(result.stationarity_backward_error <= 2.0e-12,
          "reported stationarity backward error is too large");
  require(result.feasibility_violation <= 2.0e-12 * radius,
          "reported feasibility violation is inaccurate");
  require(result.shift == 0.0 ||
              std::abs(norm - radius) <= 2.0e-12 * radius,
          "projected trust-region complementarity is inaccurate");
  require(std::abs(result.metric_norm - norm) <= 1.0e-13,
          "reported projected metric norm is inaccurate");
  const double model = gradient.dot(result.coordinates) +
      0.5 * result.coordinates.dot(hessian * result.coordinates);
  require(std::abs(result.predicted_decrease + model) <= 1.0e-13 * scale,
          "reported projected predicted decrease is inaccurate");

  Eigen::LLT<Eigen::MatrixXd> metric_factor(metric);
  const Eigen::MatrixXd inverse_lower = metric_factor.matrixL().solve(
      Eigen::MatrixXd::Identity(metric.rows(), metric.cols()));
  const Eigen::MatrixXd shifted_whitened =
      inverse_lower * (hessian + result.shift * metric) *
      inverse_lower.transpose();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> shifted_solver(
      0.5 * (shifted_whitened + shifted_whitened.transpose()));
  require(shifted_solver.eigenvalues().minCoeff() >=
              -2.0e-12 * std::max(1.0, shifted_whitened.stableNorm()),
          "shifted projected Hessian is not positive semidefinite");
  require(std::abs(result.minimum_shifted_ritz_value -
                   shifted_solver.eigenvalues().minCoeff()) <=
              2.0e-12 * std::max(1.0, shifted_whitened.stableNorm()),
          "reported shifted projected minimum Ritz value is inaccurate");
}

}  // namespace

int main() {
  try {
    using xmvb::vb::ProjectedTrustStatus;
    using xmvb::vb::solve_projected_generalized_trust_region;

    Eigen::Matrix2d spd_hessian;
    spd_hessian << 2.0, 0.2,
                   0.2, 3.0;
    Eigen::Matrix2d metric;
    metric << 1.5, 0.1,
              0.1, 0.8;
    const Eigen::Vector2d gradient(0.2, -0.1);
    const auto interior = solve_projected_generalized_trust_region(
        spd_hessian,
        metric,
        gradient,
        2.0);
    certify(
        spd_hessian,
        metric,
        gradient,
        2.0,
        interior,
        ProjectedTrustStatus::InteriorGlobal);
    require_close(
        interior.coordinates,
        -spd_hessian.inverse() * gradient,
        1.0e-13,
        "SPD interior step differs from the unconstrained Newton step");

    const double small_radius = 0.04;
    const auto boundary = solve_projected_generalized_trust_region(
        spd_hessian,
        metric,
        gradient,
        small_radius);
    certify(
        spd_hessian,
        metric,
        gradient,
        small_radius,
        boundary,
        ProjectedTrustStatus::BoundaryGlobal);

    // The reduced chart is not globally whitened. Verify covariance under a
    // nonorthogonal coordinate change s = A z: vectors transform with A,
    // while the gradient, Hessian, and physical metric transform by pullback.
    Eigen::Matrix2d coordinate_map;
    coordinate_map << 1.3, -0.2,
                      0.4,  0.9;
    const Eigen::Matrix2d transformed_hessian =
        coordinate_map.transpose() * spd_hessian * coordinate_map;
    const Eigen::Matrix2d transformed_metric =
        coordinate_map.transpose() * metric * coordinate_map;
    const Eigen::Vector2d transformed_gradient =
        coordinate_map.transpose() * gradient;
    const auto transformed_boundary =
        solve_projected_generalized_trust_region(
            transformed_hessian,
            transformed_metric,
            transformed_gradient,
            small_radius);
    certify(
        transformed_hessian,
        transformed_metric,
        transformed_gradient,
        small_radius,
        transformed_boundary,
        ProjectedTrustStatus::BoundaryGlobal);
    require_close(
        coordinate_map * transformed_boundary.coordinates,
        boundary.coordinates,
        2.0e-12,
        "generalized trust step changed under a nonorthogonal coordinate map");
    require(std::abs(transformed_boundary.shift - boundary.shift) <=
                2.0e-12 * std::max(1.0, std::abs(boundary.shift)),
            "trust-region multiplier changed under a coordinate map");
    require(std::abs(transformed_boundary.predicted_decrease -
                     boundary.predicted_decrease) <= 2.0e-13,
            "predicted decrease changed under a coordinate map");

    Eigen::Matrix2d indefinite_hessian;
    indefinite_hessian << -2.0, 0.0,
                           0.0, 3.0;
    const Eigen::Matrix2d identity = Eigen::Matrix2d::Identity();
    const Eigen::Vector2d regular_gradient(1.0, 0.2);
    const auto regular = solve_projected_generalized_trust_region(
        indefinite_hessian,
        identity,
        regular_gradient,
        1.0);
    certify(
        indefinite_hessian,
        identity,
        regular_gradient,
        1.0,
        regular,
        ProjectedTrustStatus::BoundaryGlobal);
    require(regular.shift > 2.0,
            "regular indefinite solution did not pass its spectral pole");

    const Eigen::Vector2d hard_gradient(0.0, 1.0);
    const auto hard = solve_projected_generalized_trust_region(
        indefinite_hessian,
        identity,
        hard_gradient,
        1.0);
    certify(
        indefinite_hessian,
        identity,
        hard_gradient,
        1.0,
        hard,
        ProjectedTrustStatus::HardCaseGlobal);
    require(std::abs(hard.shift - 2.0) <= 1.0e-14,
            "hard case used a conservative shift instead of the spectral endpoint");
    require(std::abs(hard.coordinates[1] + 0.2) <= 1.0e-14 &&
                std::abs(std::abs(hard.coordinates[0]) - std::sqrt(0.96)) <=
                    1.0e-14,
            "hard case did not add minimum curvature to the pseudoinverse step");

    // A conservative over-shift cannot satisfy both stationarity and the
    // boundary in this hard case: its minimum-mode stationarity coefficient
    // is nonzero, forcing that component to vanish.
    const double conservative_shift = 2.1;
    const Eigen::Vector2d conservative_stationary_step(
        0.0,
        -1.0 / (3.0 + conservative_shift));
    require(conservative_stationary_step.norm() < 1.0 &&
                hard.metric_norm > conservative_stationary_step.norm(),
            "hard-case test does not distinguish the conservative over-shift");

    // The projected model sees only the first full-space direction. The
    // omitted second direction has negative curvature, so this result must be
    // advertised as projected-global and never as a full-space certificate.
    Eigen::Matrix<double, 1, 1> projected_hessian;
    Eigen::Matrix<double, 1, 1> projected_metric;
    Eigen::Matrix<double, 1, 1> projected_gradient;
    projected_hessian << 1.0;
    projected_metric << 1.0;
    projected_gradient << 0.25;
    const auto hidden_curvature = solve_projected_generalized_trust_region(
        projected_hessian,
        projected_metric,
        projected_gradient,
        1.0);
    require(hidden_curvature.status == ProjectedTrustStatus::InteriorGlobal &&
                hidden_curvature.scope ==
                    xmvb::vb::ProjectedTrustScope::SuppliedProjectedSubspace,
            "hidden full-space curvature led to an overclaimed status");
    const Eigen::Matrix2d omitted_full_hessian =
        (Eigen::Vector2d(1.0, -5.0)).asDiagonal();
    require(omitted_full_hessian.minCoeff() < 0.0,
            "hidden-curvature fixture lost its omitted negative direction");

    Eigen::Matrix2d invalid_metric;
    invalid_metric << 1.0, 0.0,
                      0.0, -1.0;
    const auto invalid = solve_projected_generalized_trust_region(
        spd_hessian,
        invalid_metric,
        gradient,
        1.0);
    require(invalid.status == ProjectedTrustStatus::InvalidMetric &&
                !invalid.converged(),
            "indefinite projected metric was not rejected");

    std::cout << "projected generalized trust-region solver: passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
