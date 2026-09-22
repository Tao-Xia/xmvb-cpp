#include <algorithm>
#include <cmath>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#include "vbscf/optimization/neo/problem.hpp"
#include "vbscf/optimization/neo/solver.hpp"

namespace {

using xmvb::vb::NeoOptions;
using xmvb::vb::NeoProblem;
using xmvb::vb::NeoResult;
using xmvb::vb::NeoStopReason;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void require_close(
    double actual,
    double reference,
    double tolerance,
    const std::string& message) {
  const double scale = std::max({1.0, std::abs(actual), std::abs(reference)});
  require(std::abs(actual - reference) <= tolerance * scale, message);
}

void require_close(
    const Eigen::VectorXd& actual,
    const Eigen::VectorXd& reference,
    double tolerance,
    const std::string& message) {
  const double scale = std::max(1.0, reference.stableNorm());
  require((actual - reference).stableNorm() <= tolerance * scale, message);
}

NeoProblem dense_problem(
    const Eigen::MatrixXd& hessian,
    const Eigen::MatrixXd& metric,
    const Eigen::VectorXd& gradient,
    std::optional<double> hessian_lower_bound = std::nullopt) {
  return NeoProblem(
      gradient,
      [hessian](const Eigen::VectorXd& vector) {
        return hessian * vector;
      },
      [metric](const Eigen::VectorXd& vector) { return metric * vector; },
      {},
      hessian_lower_bound);
}

void verify_kkt(
    const std::string& name,
    const Eigen::MatrixXd& hessian,
    const Eigen::MatrixXd& metric,
    const Eigen::VectorXd& gradient,
    double radius,
    const NeoResult& result) {
  require(result.converged(), name + ": solver did not converge");
  require(result.step.allFinite(), name + ": non-finite step");
  require(result.shift >= 0.0 && std::isfinite(result.shift),
          name + ": invalid trust-region shift");

  const Eigen::VectorXd hessian_step = hessian * result.step;
  const Eigen::VectorXd metric_step = metric * result.step;
  const Eigen::VectorXd residual =
      gradient + hessian_step + result.shift * metric_step;
  const double step_norm = std::sqrt(result.step.dot(metric_step));
  const double residual_norm = residual.stableNorm();
  const double residual_scale = std::max({
      1.0,
      gradient.stableNorm(),
      hessian_step.stableNorm(),
      result.shift * metric_step.stableNorm()});

  require(step_norm <= radius * (1.0 + 2.0e-10),
          name + ": step violates the physical trust region");
  require(residual_norm <= 2.0e-9 * residual_scale,
          name + ": KKT stationarity is not certified");
  require(result.curvature_residual_norm <=
              2.0 * result.curvature_residual_target,
          name + ": lowest-root Ritz residual is not converged");
  require_close(result.hessian_step, hessian_step, 2.0e-10,
                name + ": returned Hessian image is inconsistent");
  require_close(result.metric_step, metric_step, 2.0e-10,
                name + ": returned metric image is inconsistent");
  require_close(result.kkt_residual, residual, 2.0e-10,
                name + ": returned KKT residual is inconsistent");
  require_close(result.step_norm, step_norm, 2.0e-10,
                name + ": returned physical norm is inconsistent");
  require_close(result.residual_norm, residual_norm, 2.0e-9,
                name + ": returned residual norm is inconsistent");

  const double predicted =
      -gradient.dot(result.step) - 0.5 * result.step.dot(hessian_step);
  require_close(result.predicted_reduction, predicted, 2.0e-10,
                name + ": predicted reduction is inconsistent");
  require(predicted >= -2.0e-12 * residual_scale,
          name + ": NEO step increases its quadratic model");

  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> shifted_spectrum(
      hessian + result.shift * metric, metric);
  require(shifted_spectrum.info() == Eigen::Success,
          name + ": shifted generalized eigensystem failed");
  const double spectral_scale = std::max(
      1.0, shifted_spectrum.eigenvalues().cwiseAbs().maxCoeff());
  require(shifted_spectrum.eigenvalues().minCoeff() >=
              -2.0e-10 * spectral_scale,
          name + ": shifted Hessian is not positive semidefinite");

  if (result.shift > 2.0e-10 * spectral_scale) {
    require(std::abs(step_norm - radius) <= 2.0e-9 * radius,
            name + ": complementarity fails on a shifted step");
    require(result.boundary, name + ": boundary step was not classified");
  }
}

void check_positive_definite_newton_limit() {
  Eigen::Matrix2d hessian;
  hessian << 2.0, 0.5,
             0.5, 4.0;
  const Eigen::Matrix2d metric = Eigen::Matrix2d::Identity();
  const Eigen::Vector2d gradient(1.0, -2.0);
  constexpr double radius = 2.0;

  NeoOptions options;
  options.trust_radius = radius;
  options.relative_residual_tolerance = 1.0e-12;
  const NeoProblem problem = dense_problem(hessian, metric, gradient);
  require(!problem.has_preconditioner(),
          "dense reference unexpectedly installed a preconditioner");
  const NeoResult result = solve_neo(problem, options);
  verify_kkt("positive-definite Newton limit", hessian, metric, gradient,
             radius, result);

  const Eigen::Vector2d newton = -hessian.ldlt().solve(gradient);
  require_close(result.step, newton, 2.0e-10,
                "interior NEO step differs from the Newton step");
  require(result.shift <= 2.0e-10,
          "interior positive-definite problem has a nonzero shift");
  require(!result.boundary,
          "interior positive-definite step was classified as a boundary step");
  require(!result.augmented_certificate_valid,
          "interior Newton limit was reported as a finite-alpha certificate");
}

void check_generalized_metric_boundary() {
  Eigen::Matrix2d hessian;
  hessian << 3.5, 0.7,
             0.7, 2.0;
  Eigen::Matrix2d metric;
  metric << 2.0, 0.35,
            0.35, 0.8;
  const Eigen::Vector2d gradient(1.2, -0.9);
  constexpr double radius = 0.18;

  NeoOptions options;
  options.trust_radius = radius;
  options.relative_residual_tolerance = 1.0e-12;
  const NeoProblem problem = dense_problem(hessian, metric, gradient);
  require(!problem.has_preconditioner(),
          "generalized-metric reference unexpectedly requires a preconditioner");
  const NeoResult result = solve_neo(problem, options);
  verify_kkt("generalized-metric boundary", hessian, metric, gradient,
             radius, result);
  require(result.shift > 0.0,
          "small generalized trust region did not produce a shift");
  require(result.augmented_certificate_valid,
          "regular boundary step lacks its augmented certificate");

  // At the returned gradient scale, the NEO step must come from the lowest
  // root of the explicit augmented generalized eigenproblem.
  require(result.gradient_scale > 0.0,
          "boundary NEO result has no positive gradient scale");
  Eigen::Matrix3d augmented = Eigen::Matrix3d::Zero();
  augmented.block<1, 2>(0, 1) =
      result.gradient_scale * gradient.transpose();
  augmented.block<2, 1>(1, 0) = result.gradient_scale * gradient;
  augmented.bottomRightCorner<2, 2>() = hessian;
  Eigen::Matrix3d augmented_metric = Eigen::Matrix3d::Zero();
  augmented_metric(0, 0) = 1.0;
  augmented_metric.bottomRightCorner<2, 2>() = metric;
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(
      augmented, augmented_metric);
  require(eigensolver.info() == Eigen::Success,
          "explicit augmented eigensystem failed");
  require_close(result.augmented_eigenvalue,
                eigensolver.eigenvalues().minCoeff(), 2.0e-9,
                "NEO did not select the lowest augmented root");
  require_close(result.augmented_eigenvalue, -result.shift, 2.0e-9,
                "augmented root and KKT shift have inconsistent signs");

  Eigen::Vector3d reconstructed;
  reconstructed[0] = 1.0;
  reconstructed.tail<2>() = result.gradient_scale * result.step;
  const Eigen::Vector3d augmented_residual =
      augmented * reconstructed -
      result.augmented_eigenvalue * augmented_metric * reconstructed;
  require(augmented_residual.stableNorm() <=
              2.0e-9 * std::max(1.0, augmented.norm()),
          "returned step does not satisfy the augmented eigenproblem");
}

void check_indefinite_regular_boundary() {
  Eigen::Matrix2d hessian;
  hessian << -1.4, 0.25,
              0.25, 2.2;
  const Eigen::Matrix2d metric = Eigen::Matrix2d::Identity();
  const Eigen::Vector2d gradient(0.8, -0.45);
  constexpr double radius = 0.6;

  NeoOptions options;
  options.trust_radius = radius;
  options.relative_residual_tolerance = 1.0e-12;
  const NeoResult result = solve_neo(
      dense_problem(hessian, metric, gradient), options);
  verify_kkt("indefinite regular boundary", hessian, metric, gradient,
             radius, result);
  require(result.shift > 0.0,
          "indefinite problem did not stabilize its Hessian");
  require(!result.hard_case,
          "regular indefinite problem was classified as a hard case");
  require(result.augmented_certificate_valid,
          "regular indefinite boundary lacks its augmented certificate");
}

void check_hard_case() {
  Eigen::Matrix2d hessian;
  hessian << -2.0, 0.0,
              0.0, 3.0;
  const Eigen::Matrix2d metric = Eigen::Matrix2d::Identity();
  const Eigen::Vector2d gradient(0.0, 1.0);
  constexpr double radius = 1.0;

  NeoOptions options;
  options.trust_radius = radius;
  options.relative_residual_tolerance = 1.0e-12;
  const NeoResult result = solve_neo(
      dense_problem(hessian, metric, gradient), options);
  verify_kkt("hard case", hessian, metric, gradient, radius, result);
  require(result.hard_case, "hard case was not detected");
  require(!result.augmented_certificate_valid,
          "hard-case curvature vector was reported as a complete NEO step");
  require_close(result.shift, 2.0, 2.0e-10,
                "hard case has the wrong spectral shift");
  require_close(result.step[1], -0.2, 2.0e-10,
                "hard-case pseudoinverse component is wrong");
  require_close(std::abs(result.step[0]), std::sqrt(0.96), 2.0e-10,
                "hard-case minimum-mode component is wrong");
}

Eigen::Matrix3d hidden_curvature_hessian() {
  const Eigen::Vector3d visible(0.0, -4.0 / 3.0, 5.0 / 3.0);
  const Eigen::Vector3d hidden(0.0, 5.0 / 3.0, 4.0 / 3.0);
  Eigen::Vector3d gradient_mode = Eigen::Vector3d::Zero();
  gradient_mode[0] = 1.0;
  return 2.0 * gradient_mode * gradient_mode.transpose() +
      3.0 * visible * visible.transpose() / visible.squaredNorm() -
      100.0 * hidden * hidden.transpose() / hidden.squaredNorm();
}

void check_hidden_negative_curvature() {
  const Eigen::Matrix3d hessian = hidden_curvature_hessian();
  const Eigen::Matrix3d metric = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d gradient(1.0, 0.0, 0.0);

  NeoOptions budgeted_options;
  budgeted_options.trust_radius = 1.0;
  budgeted_options.relative_residual_tolerance = 1.0e-12;
  budgeted_options.maximum_subspace_dimension = 2;
  const NeoResult budgeted = solve_neo(
      dense_problem(hessian, metric, gradient), budgeted_options);
  require(budgeted.stop_reason == NeoStopReason::SubspaceLimit,
          "an incomplete subspace falsely certified hidden curvature");
  require(!budgeted.converged(),
          "a work-limited hidden-curvature solve reported convergence");
  require(budgeted.hessian_actions == 2,
          "the explicit subspace budget used the wrong number of actions");

  NeoOptions complete_options = budgeted_options;
  complete_options.maximum_subspace_dimension = 0;
  const NeoResult complete = solve_neo(
      dense_problem(hessian, metric, gradient), complete_options);
  verify_kkt("hidden negative curvature", hessian, metric, gradient, 1.0,
             complete);
  require(complete.hessian_actions == 3,
          "the full solve did not expose the third curvature mode");
  require_close(complete.shift, 100.0, 2.0e-10,
                "hidden negative curvature has the wrong shift");
  require(complete.hard_case,
          "gradient-orthogonal hidden curvature was not a hard case");
  require(complete.global_curvature_certified,
          "a complete basis lacks its global curvature certificate");
}

void check_certified_lower_bounds() {
  const Eigen::Matrix3d hessian = hidden_curvature_hessian();
  const Eigen::Matrix3d metric = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d gradient(1.0, 0.0, 0.0);

  NeoOptions options;
  options.trust_radius = 1.0 / 102.0;
  options.relative_residual_tolerance = 1.0e-12;
  options.maximum_subspace_dimension = 1;

  const NeoResult tight = solve_neo(
      dense_problem(hessian, metric, gradient, -100.0), options);
  verify_kkt("tight certified lower bound", hessian, metric, gradient,
             options.trust_radius, tight);
  require(tight.hessian_actions == 1,
          "a tight certified bound did not avoid the hidden-mode action");
  require_close(tight.shift, 100.0, 2.0e-10,
                "tight lower-bound solve has the wrong shift");

  const NeoResult conservative = solve_neo(
      dense_problem(hessian, metric, gradient, -101.0), options);
  require(conservative.stop_reason == NeoStopReason::SubspaceLimit,
          "an insufficient lower bound falsely certified global curvature");
  require(!conservative.converged(),
          "an insufficient lower bound reported convergence");
}

void check_scalable_lowest_root_convergence() {
  constexpr int dimension = 40;
  Eigen::MatrixXd hessian = Eigen::MatrixXd::Identity(dimension, dimension);
  hessian.diagonal().head(20).array() = 2.0;
  hessian.diagonal().tail(19).array() = 4.0;
  hessian(dimension - 1, dimension - 1) = -3.0;
  const Eigen::MatrixXd metric = Eigen::MatrixXd::Identity(
      dimension, dimension);
  Eigen::VectorXd gradient = Eigen::VectorXd::Zero(dimension);
  gradient.head(20).setOnes();

  NeoOptions options;
  options.trust_radius = 0.25;
  options.relative_residual_tolerance = 1.0e-11;
  options.maximum_subspace_dimension = 4;
  const NeoResult result = solve_neo(
      dense_problem(hessian, metric, gradient), options);
  verify_kkt("scalable lowest-root convergence", hessian, metric, gradient,
             options.trust_radius, result);
  require(result.hessian_actions < dimension,
          "lowest-root convergence unnecessarily built the full space");
  require(!result.global_curvature_certified,
          "an incomplete Davidson space claimed a rigorous global certificate");
}

void check_optional_curvature_certificate() {
  Eigen::Matrix3d hessian = Eigen::Matrix3d::Zero();
  hessian.diagonal() << 2.0, 3.0, 4.0;
  const Eigen::Matrix3d metric = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d gradient(1.0, 0.0, 0.0);

  NeoOptions kkt_options;
  kkt_options.trust_radius = 1.0;
  kkt_options.relative_residual_tolerance = 1.0e-12;
  kkt_options.require_curvature_certificate = false;
  const NeoResult kkt = solve_neo(
      dense_problem(hessian, metric, gradient), kkt_options);
  require(kkt.converged(),
          "KKT-only interior solve did not converge");
  require(kkt.hessian_actions == 1,
          "KKT-only interior solve evaluated unrelated curvature modes");
  require_close(kkt.step, Eigen::Vector3d(-0.5, 0.0, 0.0), 2.0e-10,
                "KKT-only interior solve returned the wrong Newton step");

  NeoOptions certified_options = kkt_options;
  certified_options.require_curvature_certificate = true;
  const NeoResult certified = solve_neo(
      dense_problem(hessian, metric, gradient), certified_options);
  verify_kkt("explicit curvature certificate", hessian, metric, gradient,
             certified_options.trust_radius, certified);
  require(certified.hessian_actions > kkt.hessian_actions,
          "explicit curvature certification did not inspect extra modes");
}

void check_boundary_requires_curvature_certificate() {
  Eigen::Matrix2d hessian;
  hessian << -1.0, 0.5,
              0.5, 2.0;
  const Eigen::Matrix2d metric = Eigen::Matrix2d::Identity();
  const Eigen::Vector2d gradient(1.0, 0.0);

  NeoOptions options;
  options.trust_radius = 0.1;
  options.relative_residual_tolerance = 1.0e-12;
  options.require_curvature_certificate = false;
  const NeoResult result = solve_neo(
      dense_problem(hessian, metric, gradient), options);
  verify_kkt("automatic boundary curvature certificate", hessian, metric,
             gradient, options.trust_radius, result);
  require(result.boundary,
          "negative-curvature test did not produce a boundary step");
  require(result.hessian_actions == 2,
          "boundary solve did not expand the curvature residual");
}

void check_nonsymmetric_hessian_rejected() {
  Eigen::Matrix2d hessian;
  hessian << 2.0, 1.0,
             0.0, 3.0;
  const Eigen::Matrix2d metric = Eigen::Matrix2d::Identity();
  const Eigen::Vector2d gradient(1.0, -0.5);
  NeoOptions options;
  options.relative_residual_tolerance = 1.0e-12;

  bool rejected = false;
  try {
    (void)solve_neo(dense_problem(hessian, metric, gradient), options);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected, "materially nonsymmetric Hessian action was accepted");
}

}  // namespace

int main() {
  try {
    check_positive_definite_newton_limit();
    check_generalized_metric_boundary();
    check_indefinite_regular_boundary();
    check_hard_case();
    check_hidden_negative_curvature();
    check_certified_lower_bounds();
    check_scalable_lowest_root_convergence();
    check_optional_curvature_certificate();
    check_boundary_requires_curvature_certificate();
    check_nonsymmetric_hessian_rejected();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
