#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/LU>

#include "vbscf/optimization/coupled/projected_model.hpp"

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

template <typename Actual, typename Expected>
void require_close(
    const Eigen::MatrixBase<Actual>& actual,
    const Eigen::MatrixBase<Expected>& expected,
    double tolerance,
    const char* message) {
  if (actual.rows() != expected.rows() || actual.cols() != expected.cols() ||
      (actual - expected).stableNorm() >
          tolerance * std::max(1.0, expected.stableNorm())) {
    throw std::runtime_error(message);
  }
}

double full_model_change(
    const Eigen::MatrixXd& orbital_hessian,
    const Eigen::MatrixXd& response_hessian,
    const Eigen::MatrixXd& coupling,
    const Eigen::VectorXd& orbital_gradient,
    const Eigen::VectorXd& response_gradient,
    const Eigen::VectorXd& orbital_coordinates,
    const Eigen::VectorXd& response_coordinates) {
  return orbital_gradient.dot(orbital_coordinates) +
      response_gradient.dot(response_coordinates) +
      0.5 * orbital_coordinates.dot(
          orbital_hessian * orbital_coordinates) +
      orbital_coordinates.dot(
          coupling.transpose() * response_coordinates) +
      0.5 * response_coordinates.dot(
          response_hessian * response_coordinates);
}

struct ConstructedProblem {
  Eigen::Matrix2d orbital_hessian;
  Eigen::Matrix2d response_hessian;
  Eigen::Matrix2d coupling;
  Eigen::Matrix2d metric;
  Eigen::Vector2d orbital_gradient;
  Eigen::Vector2d response_gradient;
};

ConstructedProblem problem_with_reduced_model(
    const Eigen::Matrix2d& reduced_hessian,
    const Eigen::Vector2d& reduced_gradient) {
  ConstructedProblem problem;
  problem.response_hessian << 1.4, 0.1,
                              0.1, 1.1;
  problem.coupling << 0.25, -0.15,
                      0.10,  0.30;
  problem.metric.setIdentity();
  problem.response_gradient << 0.2, -0.15;
  const Eigen::Vector2d baseline =
      -problem.response_hessian.fullPivLu().solve(
          problem.response_gradient);
  const Eigen::Matrix2d lift =
      -problem.response_hessian.fullPivLu().solve(problem.coupling);
  problem.orbital_hessian =
      reduced_hessian - problem.coupling.transpose() * lift;
  problem.orbital_gradient =
      reduced_gradient - problem.coupling.transpose() * baseline;
  return problem;
}

void verify_kkt_and_model(
    const ConstructedProblem& problem,
    double radius,
    const xmvb::vb::CoupledProjectedModelResult& result) {
  const Eigen::Vector2d response_residual = problem.response_gradient +
      problem.coupling * result.orbital_coordinates +
      problem.response_hessian * result.response_coordinates;
  const Eigen::Vector2d orbital_residual = problem.orbital_gradient +
      problem.orbital_hessian * result.orbital_coordinates +
      problem.coupling.transpose() * result.response_coordinates +
      result.orbital_solution.shift * problem.metric *
          result.orbital_coordinates;
  require_close(
      result.response_kkt_residual,
      response_residual,
      2.0e-14,
      "reported projected response KKT residual is inaccurate");
  require_close(
      result.orbital_kkt_residual,
      orbital_residual,
      2.0e-14,
      "reported projected orbital KKT residual is inaccurate");
  require_close(
      result.response_kkt_residual_norm,
      response_residual.stableNorm(),
      2.0e-14,
      "reported response KKT norm is inaccurate");
  require_close(
      result.orbital_kkt_residual_norm,
      orbital_residual.stableNorm(),
      2.0e-14,
      "reported orbital KKT norm is inaccurate");
  require(result.orbital_solution.metric_norm <= radius * (1.0 + 1.0e-12),
          "coupled projected solution violates the orbital trust region");

  const double direct_model = full_model_change(
      problem.orbital_hessian,
      problem.response_hessian,
      problem.coupling,
      problem.orbital_gradient,
      problem.response_gradient,
      result.orbital_coordinates,
      result.response_coordinates);
  require_close(
      result.total_model_change,
      direct_model,
      3.0e-14,
      "eliminated and full projected model changes disagree");
  require_close(
      result.total_predicted_decrease,
      -direct_model,
      3.0e-14,
      "full projected predicted decrease omitted its response constant");
  require_close(
      result.total_model_change,
      result.response_baseline_model_change +
          result.reduced_incremental_model_change,
      2.0e-14,
      "projected model baseline and increment do not sum exactly");
}

}  // namespace

int main() {
  try {
    using xmvb::vb::CoupledProjectedModelStatus;
    using xmvb::vb::ProjectedTrustStatus;
    using xmvb::vb::solve_coupled_projected_model;

    ConstructedProblem dense_problem;
    dense_problem.orbital_hessian << 3.0, 0.2,
                                     0.2, 2.0;
    dense_problem.response_hessian << 2.0, 0.1,
                                      0.1, 1.5;
    dense_problem.coupling << 0.4, -0.2,
                              0.1,  0.3;
    dense_problem.metric << 1.3, 0.1,
                            0.1, 0.9;
    dense_problem.orbital_gradient << 0.35, -0.25;
    dense_problem.response_gradient << 0.4, -0.3;
    const auto dense = solve_coupled_projected_model(
        dense_problem.orbital_hessian,
        dense_problem.response_hessian,
        dense_problem.coupling,
        dense_problem.metric,
        dense_problem.orbital_gradient,
        dense_problem.response_gradient,
        2.0);
    require(dense.converged() &&
                dense.orbital_solution.status ==
                    ProjectedTrustStatus::InteriorGlobal,
            "dense Schur fixture did not produce an interior solution");
    const Eigen::Vector2d reference_baseline =
        -dense_problem.response_hessian.fullPivLu().solve(
            dense_problem.response_gradient);
    const Eigen::Matrix2d reference_lift =
        -dense_problem.response_hessian.fullPivLu().solve(
            dense_problem.coupling);
    const Eigen::Vector2d reference_gradient =
        dense_problem.orbital_gradient +
        dense_problem.coupling.transpose() * reference_baseline;
    const Eigen::Matrix2d reference_hessian =
        dense_problem.orbital_hessian +
        dense_problem.coupling.transpose() * reference_lift;
    const Eigen::Vector2d reference_orbital =
        -reference_hessian.fullPivLu().solve(reference_gradient);
    const Eigen::Vector2d reference_response =
        reference_baseline + reference_lift * reference_orbital;
    require_close(dense.response_baseline, reference_baseline, 2.0e-14,
                  "projected response baseline is inaccurate");
    require_close(dense.response_lift, reference_lift, 2.0e-14,
                  "projected response lift is inaccurate");
    require_close(dense.reduced_gradient, reference_gradient, 2.0e-14,
                  "Schur-reduced gradient is inaccurate");
    require_close(dense.reduced_hessian, reference_hessian, 2.0e-14,
                  "Schur-reduced Hessian is inaccurate");
    require_close(dense.orbital_coordinates, reference_orbital, 2.0e-13,
                  "coupled orbital coordinates disagree with dense Schur solve");
    require_close(dense.response_coordinates, reference_response, 2.0e-13,
                  "coupled response coordinates disagree with dense solve");
    require_close(
        dense.response_baseline_model_change,
        dense_problem.response_gradient.dot(reference_baseline) +
            0.5 * reference_baseline.dot(
                dense_problem.response_hessian * reference_baseline),
        2.0e-14,
        "nonzero response-gradient baseline is inaccurate");
    verify_kkt_and_model(dense_problem, 2.0, dense);

    Eigen::Matrix2d indefinite_reduced_hessian;
    indefinite_reduced_hessian << -2.0, 0.0,
                                   0.0, 3.0;
    const ConstructedProblem regular_problem = problem_with_reduced_model(
        indefinite_reduced_hessian, Eigen::Vector2d(1.0, 0.2));
    const auto regular = solve_coupled_projected_model(
        regular_problem.orbital_hessian,
        regular_problem.response_hessian,
        regular_problem.coupling,
        regular_problem.metric,
        regular_problem.orbital_gradient,
        regular_problem.response_gradient,
        1.0);
    require(regular.converged() &&
                regular.orbital_solution.status ==
                    ProjectedTrustStatus::BoundaryGlobal &&
                regular.orbital_solution.shift > 2.0,
            "regular indefinite Schur problem was not solved at the boundary");
    require_close(regular.reduced_hessian, indefinite_reduced_hessian, 2.0e-14,
                  "regular indefinite reduced Hessian changed in elimination");
    verify_kkt_and_model(regular_problem, 1.0, regular);

    const ConstructedProblem hard_problem = problem_with_reduced_model(
        indefinite_reduced_hessian, Eigen::Vector2d(0.0, 1.0));
    const auto hard = solve_coupled_projected_model(
        hard_problem.orbital_hessian,
        hard_problem.response_hessian,
        hard_problem.coupling,
        hard_problem.metric,
        hard_problem.orbital_gradient,
        hard_problem.response_gradient,
        1.0);
    require(hard.converged() &&
                hard.orbital_solution.status ==
                    ProjectedTrustStatus::HardCaseGlobal,
            "coupled projected hard case was not preserved by elimination");
    require_close(hard.orbital_solution.shift, 2.0, 2.0e-14,
                  "coupled hard case did not use the exact spectral endpoint");
    require_close(hard.reduced_hessian, indefinite_reduced_hessian, 2.0e-14,
                  "hard-case reduced Hessian changed in elimination");
    verify_kkt_and_model(hard_problem, 1.0, hard);

    Eigen::Matrix2d singular_response;
    singular_response << 1.0, 0.0,
                         0.0, 0.0;
    const auto singular = solve_coupled_projected_model(
        dense_problem.orbital_hessian,
        singular_response,
        dense_problem.coupling,
        dense_problem.metric,
        dense_problem.orbital_gradient,
        dense_problem.response_gradient,
        1.0);
    require(singular.status ==
                CoupledProjectedModelStatus::ResponseProjectionSingular &&
                !singular.converged(),
            "singular projected response Hessian was regularized or hidden");

    std::cout << "coupled projected elimination: passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
