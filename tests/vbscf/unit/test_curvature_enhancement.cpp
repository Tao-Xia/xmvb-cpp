#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#include "vbscf/optimization/coupled/operator.hpp"
#include "vbscf/optimization/coupled/solver.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

double algebraic_tolerance(int dimension) {
  return 512.0 * std::numeric_limits<double>::epsilon() *
      std::max(1, dimension);
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

double model_decrease(
    const Eigen::Matrix3d& orbital_hessian,
    const Eigen::Matrix<double, 2, 3>& coupling,
    const Eigen::Matrix2d& response_hessian,
    const Eigen::Vector3d& orbital_gradient,
    const Eigen::Vector2d& response_residual,
    const Eigen::Vector3d& orbital_step,
    const Eigen::Vector2d& response_step) {
  const double change = orbital_gradient.dot(orbital_step) +
      response_residual.dot(response_step) +
      0.5 * orbital_step.dot(orbital_hessian * orbital_step) +
      orbital_step.dot(coupling.transpose() * response_step) +
      0.5 * response_step.dot(response_hessian * response_step);
  return -change;
}

xmvb::vb::CoupledNewtonOperator make_operator(
    const Eigen::Matrix3d& orbital_hessian,
    const Eigen::Matrix<double, 2, 3>& coupling,
    const Eigen::Matrix2d& response_hessian) {
  return xmvb::vb::CoupledNewtonOperator(
      3,
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
          [](const auto& directions) {
            return Eigen::MatrixXd(directions);
          }});
}

}  // namespace

int main() {
  try {
    Eigen::Matrix3d orbital_hessian;
    orbital_hessian << 4.0, 0.3, -0.2,
                       0.3, 3.0,  0.4,
                      -0.2, 0.4,  2.5;
    Eigen::Matrix<double, 2, 3> coupling;
    coupling << 0.8, -0.3, 0.2,
                0.1,  0.7, 0.5;
    Eigen::Matrix2d response_hessian;
    response_hessian << 2.0, 0.2,
                        0.2, 1.4;
    const Eigen::Vector3d gradient(0.9, -0.6, 0.4);
    const Eigen::Vector2d response_residual = Eigen::Vector2d::Zero();
    const double trust_radius = 10.0;
    const double tolerance = algebraic_tolerance(5);

    const Eigen::Matrix3d inverse_baseline =
        Eigen::Vector3d(0.18, 0.27, 0.22).asDiagonal();
    const Eigen::Vector3d baseline_orbital =
        -inverse_baseline * gradient;
    const Eigen::Vector2d baseline_response =
        -response_hessian.fullPivLu().solve(coupling * baseline_orbital);
    const double baseline_decrease = model_decrease(
        orbital_hessian,
        coupling,
        response_hessian,
        gradient,
        response_residual,
        baseline_orbital,
        baseline_response);
    require(baseline_decrease > 0.0,
            "curvature-enhancement fixture has no decreasing baseline");

    const auto coupled_operator = make_operator(
        orbital_hessian, coupling, response_hessian);
    xmvb::vb::CoupledSubspaceSolver solver(
        coupled_operator, gradient, response_residual);
    require(solver.append_orbital_block(baseline_orbital) == 1 &&
                solver.append_response_block(baseline_response) == 1,
            "baseline step was not admitted to the coupled subspaces");

    const Eigen::MatrixXd& orbital_basis = solver.cache().orbital_basis();
    const Eigen::MatrixXd& response_basis = solver.cache().response_basis();
    require_close(
        orbital_basis * (orbital_basis.transpose() * baseline_orbital),
        baseline_orbital,
        tolerance,
        "L-BFGS baseline is absent from the orbital trial subspace");
    require_close(
        response_basis * (response_basis.transpose() * baseline_response),
        baseline_response,
        tolerance,
        "induced baseline response is absent from the response trial subspace");

    const xmvb::vb::CoupledKktTolerances kkt_tolerances{
        8.0 * tolerance, 8.0 * tolerance};
    auto step = solver.solve(
        trust_radius, kkt_tolerances, baseline_decrease);
    require(step.status !=
                xmvb::vb::CoupledSubspaceStatus::InsufficientPredictedDecrease,
            "curvature-enhanced projection did not dominate its baseline");
    require(step.predicted_decrease + tolerance >= baseline_decrease,
            "projected model decrease is worse than the feasible baseline");

    const auto identity_inverse = [](const Eigen::VectorXd& vector) {
      return vector;
    };
    double preceding_decrease = step.predicted_decrease;
    while (!step.converged()) {
      const auto expansion = solver.expand(
          step, identity_inverse, identity_inverse);
      require(expansion.progressed(),
              "curvature-correction residual did not expand the subspaces");
      step = solver.solve(
          trust_radius, kkt_tolerances, baseline_decrease);
      require(step.status !=
                  xmvb::vb::CoupledSubspaceStatus::InsufficientPredictedDecrease,
              "nested curvature subspace lost baseline dominance");
      require(step.predicted_decrease + tolerance >= preceding_decrease,
              "expanding a nested subspace reduced predicted decrease");
      preceding_decrease = step.predicted_decrease;
    }

    const Eigen::Matrix3d reduced_hessian = orbital_hessian -
        coupling.transpose() *
            response_hessian.fullPivLu().solve(coupling);
    const Eigen::Vector3d newton_orbital =
        -reduced_hessian.fullPivLu().solve(gradient);
    const Eigen::Vector2d newton_response =
        -response_hessian.fullPivLu().solve(coupling * newton_orbital);
    require_close(
        step.orbital_step,
        newton_orbital,
        16.0 * tolerance,
        "complete curvature-enhancement space did not recover Newton");
    require_close(
        step.response_step,
        newton_response,
        16.0 * tolerance,
        "complete response space did not recover the Newton response");

    const Eigen::Vector3d core_only_step =
        -orbital_hessian.fullPivLu().solve(gradient);
    require((newton_orbital - core_only_step).stableNorm() >
                1.0e4 * tolerance,
            "fixture does not expose the outer-response Schur correction");
    xmvb::vb::CoupledSubspaceSolver exact_baseline_solver(
        coupled_operator, gradient, response_residual);
    exact_baseline_solver.append_orbital_block(newton_orbital);
    exact_baseline_solver.append_response_block(newton_response);
    const double exact_baseline_decrease = model_decrease(
        orbital_hessian,
        coupling,
        response_hessian,
        gradient,
        response_residual,
        newton_orbital,
        newton_response);
    const auto zero_correction = exact_baseline_solver.solve(
        trust_radius, kkt_tolerances, exact_baseline_decrease);
    require(zero_correction.converged(),
            "an exact baseline spuriously requested a curvature correction");
    require_close(
        zero_correction.orbital_step,
        newton_orbital,
        16.0 * tolerance,
        "zero curvature correction did not reproduce the baseline orbital step");
    require_close(
        zero_correction.response_step,
        newton_response,
        16.0 * tolerance,
        "zero curvature correction did not reproduce the baseline response");

    Eigen::Matrix3d indefinite_orbital_hessian = orbital_hessian;
    indefinite_orbital_hessian(0, 0) = -1.0;
    const Eigen::Matrix3d indefinite_reduced_hessian =
        indefinite_orbital_hessian - coupling.transpose() *
            response_hessian.fullPivLu().solve(coupling);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> indefinite_spectrum(
        indefinite_reduced_hessian);
    require(indefinite_spectrum.eigenvalues().minCoeff() < 0.0,
            "indefinite dominance fixture lost its negative curvature");
    const Eigen::Vector3d indefinite_baseline_orbital = -0.1 * gradient;
    const Eigen::Vector2d indefinite_baseline_response =
        -response_hessian.fullPivLu().solve(
            coupling * indefinite_baseline_orbital);
    const double indefinite_baseline_decrease = model_decrease(
        indefinite_orbital_hessian,
        coupling,
        response_hessian,
        gradient,
        response_residual,
        indefinite_baseline_orbital,
        indefinite_baseline_response);
    require(indefinite_baseline_decrease > 0.0,
            "indefinite fixture baseline is not decreasing");
    const auto indefinite_operator = make_operator(
        indefinite_orbital_hessian, coupling, response_hessian);
    xmvb::vb::CoupledSubspaceSolver indefinite_solver(
        indefinite_operator, gradient, response_residual);
    indefinite_solver.append_orbital_block(indefinite_baseline_orbital);
    indefinite_solver.append_response_block(indefinite_baseline_response);
    const auto indefinite_step = indefinite_solver.solve(
        0.5, kkt_tolerances, indefinite_baseline_decrease);
    require(indefinite_step.status !=
                xmvb::vb::CoupledSubspaceStatus::InsufficientPredictedDecrease &&
                indefinite_step.predicted_decrease + tolerance >=
                    indefinite_baseline_decrease,
            "negative curvature caused the projected model to lose its baseline");

    std::cout << "coupled curvature enhancement: passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
