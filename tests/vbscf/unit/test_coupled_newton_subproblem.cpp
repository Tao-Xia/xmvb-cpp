#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#include "vbscf/optimization/coupled/operator.hpp"
#include "vbscf/optimization/coupled/subproblem.hpp"
#include "vbscf/optimization/trust_region/spectral.hpp"

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

xmvb::vb::CoupledNewtonOperator make_operator(
    const Eigen::Matrix2d& orbital_hessian,
    const Eigen::Matrix2d& coupling,
    const Eigen::Matrix2d& response_hessian,
    const Eigen::Matrix2d& metric) {
  using xmvb::vb::CoupledNewtonActions;
  using xmvb::vb::CoupledNewtonOperator;
  using xmvb::vb::SelectedStateCluster;
  using xmvb::vb::SelectedSubspaceResponseLayout;
  return CoupledNewtonOperator(
      2,
      SelectedSubspaceResponseLayout(1, {SelectedStateCluster{1, 0.5}}),
      CoupledNewtonActions{
          [orbital_hessian](const Eigen::Ref<const Eigen::MatrixXd>& x) {
            return orbital_hessian * x;
          },
          [coupling](const Eigen::Ref<const Eigen::MatrixXd>& x) {
            return coupling * x;
          },
          [coupling](const Eigen::Ref<const Eigen::MatrixXd>& x) {
            return coupling.transpose() * x;
          },
          [response_hessian](const Eigen::Ref<const Eigen::MatrixXd>& x) {
            return response_hessian * x;
          },
          [metric](const Eigen::Ref<const Eigen::MatrixXd>& x) {
            return metric * x;
          }});
}

}  // namespace

int main() {
  try {
    using xmvb::vb::CoupledSubproblemOptions;
    using xmvb::vb::solve_coupled_newton_subproblem;

    Eigen::Matrix2d a;
    a << 4.0, 0.2, 0.2, 3.0;
    Eigen::Matrix2d b;
    b << 0.4, 0.1, 0.2, -0.3;
    Eigen::Matrix2d c;
    c << -2.0, 0.0, 0.0, 1.0;
    Eigen::Matrix2d metric;
    metric << 1.5, 0.0, 0.0, 0.7;
    const auto op = make_operator(a, b, c, metric);
    const Eigen::Vector2d gradient(0.3, -0.2);
    CoupledSubproblemOptions options;
    options.minres.relative_residual_tolerance = 1.0e-12;
    options.boundary_relative_tolerance = 1.0e-9;

    const auto interior = solve_coupled_newton_subproblem(
        op, gradient, 2.0, options);
    require(interior.converged() && !interior.reached_boundary() &&
                interior.orbital_shift == 0.0,
            "coupled interior Newton step was not accepted");
    Eigen::Matrix4d full;
    full << a(0, 0), a(0, 1), b(0, 0), b(1, 0),
            a(1, 0), a(1, 1), b(0, 1), b(1, 1),
            b(0, 0), b(0, 1), c(0, 0), c(0, 1),
            b(1, 0), b(1, 1), c(1, 0), c(1, 1);
    const Eigen::Vector4d rhs(-gradient[0], -gradient[1], 0.0, 0.0);
    require_close(interior.step, full.fullPivLu().solve(rhs), 1.0e-11,
                  "coupled interior Newton step is inaccurate");
    require(interior.linear_result.residual_norm <=
                interior.linear_result.residual_target,
            "coupled interior KKT residual is uncertified");

    // Compare a non-Euclidean orbital boundary against the dense relaxed
    // Hessian reference. The response block remains outside the trust norm.
    const double radius = 0.04;
    const auto boundary = solve_coupled_newton_subproblem(
        op, gradient, radius, options,
        [full, metric](double shift) {
          Eigen::Matrix4d shifted = full;
          shifted.topLeftCorner<2, 2>() += shift * metric;
          const Eigen::Vector4d inverse_diagonal =
              shifted.diagonal().cwiseAbs().cwiseMax(0.5).cwiseInverse();
          return xmvb::vb::SymmetricOperatorAction(
              [inverse_diagonal](const Eigen::VectorXd& vector) {
                return inverse_diagonal.array() * vector.array();
              });
        });
    require(boundary.converged() && boundary.reached_boundary(),
            "coupled boundary Newton step did not converge");
    require(std::abs(boundary.orbital_norm - radius) <=
                options.boundary_relative_tolerance * radius,
            "orbital-only trust boundary is not certified");

    const Eigen::Matrix2d relaxed =
        a - b.transpose() * c.inverse() * b;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> metric_solver(metric);
    const Eigen::Matrix2d inverse_sqrt_metric =
        metric_solver.eigenvectors() *
        metric_solver.eigenvalues().cwiseSqrt().cwiseInverse().asDiagonal() *
        metric_solver.eigenvectors().transpose();
    const Eigen::Matrix2d transformed =
        inverse_sqrt_metric * relaxed * inverse_sqrt_metric;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> transformed_solver(
        transformed);
    const Eigen::Vector2d transformed_gradient =
        transformed_solver.eigenvectors().transpose() *
        inverse_sqrt_metric * gradient;
    const auto spectral = xmvb::vb::solve_spectral_trust_region(
        transformed_solver.eigenvalues(), transformed_gradient, radius);
    const Eigen::Vector2d reference_orbital =
        inverse_sqrt_metric * transformed_solver.eigenvectors() * spectral.step;
    require_close(boundary.orbital_step, reference_orbital, 2.0e-8,
                  "coupled secular boundary differs from dense reference");
    require(std::abs(boundary.orbital_shift - spectral.shift) < 2.0e-8,
            "coupled secular shift differs from dense reference");
    require(boundary.predicted_decrease > 0.0,
            "coupled model did not predict a decrease");

    // A supplied convexifying lower bound selects the regular branch of an
    // indefinite relaxed Hessian without imposing an empirical iteration cap.
    Eigen::Matrix2d indefinite_a;
    indefinite_a << -2.0, 0.0, 0.0, 3.0;
    const Eigen::Matrix2d zero = Eigen::Matrix2d::Zero();
    const Eigen::Matrix2d identity = Eigen::Matrix2d::Identity();
    const auto indefinite_op = make_operator(
        indefinite_a, zero, c, identity);
    CoupledSubproblemOptions indefinite_options = options;
    indefinite_options.convexifying_shift_lower_bound = 2.01;
    const auto indefinite = solve_coupled_newton_subproblem(
        indefinite_op,
        Eigen::Vector2d(1.0, 0.5),
        0.4,
        indefinite_options);
    require(indefinite.converged() && indefinite.reached_boundary() &&
                indefinite.orbital_shift >=
                    indefinite_options.convexifying_shift_lower_bound,
            "negative-curvature regular boundary was not resolved");
    require(std::abs(indefinite.orbital_norm - 0.4) <=
                indefinite_options.boundary_relative_tolerance * 0.4,
            "negative-curvature boundary norm is uncertified");

    const auto hard_case = solve_coupled_newton_subproblem(
        indefinite_op,
        Eigen::Vector2d(0.0, 0.5),
        0.4,
        indefinite_options);
    require(!hard_case.converged() &&
                hard_case.stop_reason ==
                    xmvb::vb::CoupledSubproblemStopReason::HardCaseUnresolved,
            "coupled solver hid an unresolved trust-region hard case");

    std::cout <<
        "coupled Newton subproblem: passed (interior, boundary, indefinite)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
