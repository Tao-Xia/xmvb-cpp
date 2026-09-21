#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "vbscf/optimization/neo/response_solver.hpp"
#include "vbscf/optimization/trust_region/spectral.hpp"

namespace {

using xmvb::vb::NeoOptions;
using xmvb::vb::NeoStopReason;
using xmvb::vb::ResponseNeoDirection;
using xmvb::vb::ResponseNeoProblem;
using xmvb::vb::ResponseNeoResult;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void require_close(
    const Eigen::VectorXd& actual,
    const Eigen::VectorXd& reference,
    double tolerance,
    const std::string& message) {
  require((actual - reference).stableNorm() <=
              tolerance * std::max(1.0, reference.stableNorm()),
          message);
}

ResponseNeoProblem dense_problem(
    const Eigen::MatrixXd& a,
    const Eigen::MatrixXd& b,
    const Eigen::MatrixXd& c,
    const Eigen::MatrixXd& metric,
    const Eigen::VectorXd& gradient) {
  return ResponseNeoProblem(
      gradient,
      c.rows(),
      [a, b](const Eigen::VectorXd& p) {
        return ResponseNeoDirection{a * p, b * p};
      },
      [b, c](const Eigen::VectorXd& q) {
        return ResponseNeoDirection{b.transpose() * q, c * q};
      },
      [metric](const Eigen::VectorXd& p) { return metric * p; });
}

Eigen::VectorXd explicit_orbital_step(
    const Eigen::MatrixXd& relaxed,
    const Eigen::MatrixXd& metric,
    const Eigen::VectorXd& gradient,
    double radius) {
  Eigen::LLT<Eigen::MatrixXd> factor(metric);
  require(factor.info() == Eigen::Success, "explicit metric factor failed");
  const Eigen::MatrixXd whitening = factor.matrixU().solve(
      Eigen::MatrixXd::Identity(metric.rows(), metric.cols()));
  const Eigen::MatrixXd white_hessian =
      whitening.transpose() * relaxed * whitening;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(white_hessian);
  require(spectrum.info() == Eigen::Success, "explicit spectrum failed");
  const auto solution = xmvb::vb::solve_spectral_trust_region(
      spectrum.eigenvalues(),
      spectrum.eigenvectors().transpose() *
          whitening.transpose() * gradient,
      radius);
  return whitening * spectrum.eigenvectors() * solution.step;
}

void verify_coupled_residual(
    const Eigen::MatrixXd& a,
    const Eigen::MatrixXd& b,
    const Eigen::MatrixXd& c,
    const Eigen::MatrixXd& metric,
    const Eigen::VectorXd& gradient,
    double radius,
    const ResponseNeoResult& result) {
  require(result.converged(), "response NEO did not converge");
  const Eigen::VectorXd orbital = gradient +
      a * result.step.orbital + b.transpose() * result.step.structure +
      result.shift * metric * result.step.orbital;
  const Eigen::VectorXd structure =
      b * result.step.orbital + c * result.step.structure;
  const double residual = std::hypot(
      orbital.stableNorm(), structure.stableNorm());
  const double scale = std::max({
      1.0, gradient.stableNorm(),
      (a * result.step.orbital).stableNorm(),
      (b.transpose() * result.step.structure).stableNorm()});
  require(residual <= 2.0e-9 * scale,
          "full coupled KKT residual is not converged");
  require(structure.stableNorm() <= 2.0e-9 * scale,
          "structure stationarity is not converged");
  require(std::sqrt(result.step.orbital.dot(
              metric * result.step.orbital)) <=
              radius * (1.0 + 2.0e-10),
          "orbital step violates its trust region");
}

void check_spd_structure_and_explicit_schur() {
  Eigen::Matrix3d a;
  a << 3.0, 0.4, -0.2,
       0.4, 2.2, 0.3,
      -0.2, 0.3, 1.7;
  Eigen::Matrix<double, 2, 3> b;
  b << 0.7, -0.2, 0.4,
      -0.3, 0.5, 0.1;
  Eigen::Matrix2d c;
  c << 2.1, 0.35,
       0.35, 1.4;
  Eigen::Matrix3d metric;
  metric << 1.6, 0.1, 0.0,
            0.1, 0.9, 0.12,
            0.0, 0.12, 1.2;
  const Eigen::Vector3d gradient(0.9, -1.1, 0.6);
  constexpr double radius = 0.35;

  NeoOptions options;
  options.trust_radius = radius;
  options.relative_residual_tolerance = 1.0e-12;
  const ResponseNeoResult result = xmvb::vb::solve_response_neo(
      dense_problem(a, b, c, metric, gradient), options);
  verify_coupled_residual(a, b, c, metric, gradient, radius, result);

  const Eigen::Matrix3d relaxed =
      a - b.transpose() * c.ldlt().solve(b);
  const Eigen::Vector3d reference = explicit_orbital_step(
      relaxed, metric, gradient, radius);
  require_close(result.step.orbital, reference, 2.0e-9,
                "projected-response step differs from explicit Schur NEO");
  require_close(result.step.structure, -c.ldlt().solve(b * reference),
                2.0e-9,
                "reconstructed structure response is inconsistent");
}

void check_negative_relaxed_curvature() {
  Eigen::Matrix2d a;
  a << 0.4, 0.1,
       0.1, 1.8;
  Eigen::RowVector2d b;
  b << 1.2, 0.0;
  Eigen::Matrix<double, 1, 1> c;
  c << 1.0;
  const Eigen::Matrix2d metric = Eigen::Matrix2d::Identity();
  const Eigen::Vector2d gradient(0.3, -0.2);

  NeoOptions options;
  options.trust_radius = 0.5;
  options.relative_residual_tolerance = 1.0e-12;
  const ResponseNeoResult result = xmvb::vb::solve_response_neo(
      dense_problem(a, b, c, metric, gradient), options);
  verify_coupled_residual(
      a, b, c, metric, gradient, options.trust_radius, result);
  require(result.shift > 0.0,
          "negative relaxed curvature did not produce a spectral shift");
  require(result.boundary,
          "negative relaxed curvature did not reach the orbital boundary");
}

void check_budget_reports_subspace_limit() {
  const Eigen::Matrix3d a = Eigen::Matrix3d::Identity();
  Eigen::Matrix<double, 2, 3> b;
  b << 0.4, 0.2, 0.1,
      -0.1, 0.3, 0.5;
  Eigen::Matrix2d c;
  c << 1.5, 0.2,
       0.2, 1.0;
  const Eigen::Vector3d gradient(1.0, -0.4, 0.7);
  NeoOptions options;
  options.trust_radius = 0.2;
  options.relative_residual_tolerance = 1.0e-13;
  options.maximum_subspace_dimension = 1;
  const ResponseNeoResult result = xmvb::vb::solve_response_neo(
      dense_problem(
          a, b, c, Eigen::Matrix3d::Identity(), gradient), options);
  require(result.stop_reason == NeoStopReason::SubspaceLimit,
          "explicit response NEO budget was not reported");
}

void check_structure_contracts() {
  const Eigen::Matrix2d a = Eigen::Matrix2d::Identity();
  const Eigen::RowVector2d b(0.5, 0.0);
  const Eigen::Vector2d gradient(1.0, 0.2);
  NeoOptions options;
  options.trust_radius = 0.2;

  Eigen::Matrix<double, 1, 1> negative_c;
  negative_c << -1.0;
  const ResponseNeoResult excited_response = xmvb::vb::solve_response_neo(
      dense_problem(
          a, b, negative_c, Eigen::Matrix2d::Identity(), gradient),
      options);
  verify_coupled_residual(
      a, b, negative_c, Eigen::Matrix2d::Identity(), gradient,
      options.trust_radius, excited_response);

  Eigen::Matrix<double, 1, 1> singular_c;
  singular_c.setZero();
  bool rejected_range = false;
  try {
    (void)xmvb::vb::solve_response_neo(
        dense_problem(
            a, b, singular_c, Eigen::Matrix2d::Identity(), gradient),
        options);
  } catch (const std::runtime_error&) {
    rejected_range = true;
  }
  require(rejected_range, "coupling outside Range(C) was accepted");
}

}  // namespace

int main() {
  try {
    check_spd_structure_and_explicit_schur();
    check_negative_relaxed_curvature();
    check_budget_reports_subspace_limit();
    check_structure_contracts();
    std::cout << "response NEO solver tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "response NEO solver test failed: " << error.what() << '\n';
    return 1;
  }
}
