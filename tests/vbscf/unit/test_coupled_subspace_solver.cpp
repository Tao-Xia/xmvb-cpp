#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

#include <Eigen/Core>

#include "vbscf/optimization/coupled/operator.hpp"
#include "vbscf/optimization/coupled/solver.hpp"

namespace {

struct Counts {
  int a = 0;
  int b = 0;
  int bt = 0;
  int c = 0;
  int g = 0;
};

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

}  // namespace

int main() {
  try {
    Eigen::Matrix4d a;
    a << 3.0, 0.2, -0.1, 0.3,
         0.2, 2.0, 0.4, -0.2,
        -0.1, 0.4, 1.5, 0.1,
         0.3, -0.2, 0.1, 2.5;
    Eigen::Matrix4d g;
    g << 1.5, 0.1, 0.0, 0.1,
         0.1, 1.2, 0.1, 0.0,
         0.0, 0.1, 0.9, 0.05,
         0.1, 0.0, 0.05, 1.1;
    Eigen::Matrix<double, 3, 4> b;
    b << 0.3, -0.2, 0.1, 0.4,
         0.1, 0.5, -0.3, 0.2,
        -0.2, 0.1, 0.4, -0.1;
    Eigen::Matrix3d c;
    c << 2.0, 0.2, -0.1,
         0.2, 1.4, 0.3,
        -0.1, 0.3, 1.5;
    Eigen::Vector4d orbital_gradient(0.4, -0.3, 0.2, 0.1);
    Eigen::Vector3d response_residual(0.2, -0.1, 0.3);
    const auto counts = std::make_shared<Counts>();
    const xmvb::vb::CoupledNewtonOperator coupled(
        4,
        xmvb::vb::SelectedSubspaceResponseLayout(
            2,
            {xmvb::vb::SelectedStateCluster{1, 0.5}}),
        xmvb::vb::CoupledNewtonActions{
            [a, counts](const auto& x) {
              ++counts->a;
              return (a * x).eval();
            },
            [b, counts](const auto& x) {
              ++counts->b;
              return (b * x).eval();
            },
            [b, counts](const auto& x) {
              ++counts->bt;
              return (b.transpose() * x).eval();
            },
            [c, counts](const auto& x) {
              ++counts->c;
              return (c * x).eval();
            },
            [g, counts](const auto& x) {
              ++counts->g;
              return (g * x).eval();
            }});
    xmvb::vb::CoupledSubspaceSolver solver(
        coupled, orbital_gradient, response_residual);
    require(
        solver.solve(1.0, {1.0e-12, 1.0e-12}).status ==
            xmvb::vb::CoupledSubspaceStatus::EmptyOrbitalSpace,
        "empty coupled solve did not request an orbital space");

    require(solver.append_orbital_block(-orbital_gradient) == 1,
            "initial orbital descent direction was rejected");
    const auto orbital_only = solver.solve(1.0, {1.0e-12, 1.0e-12});
    require(
        orbital_only.status ==
            xmvb::vb::CoupledSubspaceStatus::ExpandResponseSpace ||
        orbital_only.status ==
            xmvb::vb::CoupledSubspaceStatus::ExpandBothSpaces,
        "orbital-only projection did not expose its omitted response equation");
    require(solver.append_response_block(response_residual) == 1,
            "initial response residual was rejected");
    const auto partial = solver.solve(1.0, {1.0e-12, 1.0e-12});
    require(!partial.converged(),
            "one-dimensional spaces unexpectedly solved the full problem");
    require(partial.orbital_kkt_residual.size() == 4 &&
                partial.response_kkt_residual.size() == 3,
            "full KKT residuals were not reconstructed");

    const auto identity_inverse = [](const Eigen::VectorXd& vector) {
      return vector;
    };
    auto complete = partial;
    while (!complete.converged()) {
      const int rank_before = solver.cache().orbital_subspace_size() +
          solver.cache().response_subspace_size();
      const auto expansion = solver.expand(
          complete,
          identity_inverse,
          identity_inverse);
      require(expansion.progressed(),
              "unresolved KKT residual failed to expand either space");
      const int rank_after = solver.cache().orbital_subspace_size() +
          solver.cache().response_subspace_size();
      require(rank_after > rank_before && rank_after <= 7,
              "residual expansion violated the algebraic rank bound");
      complete = solver.solve(1.0, {2.0e-13, 2.0e-13});
    }
    require(complete.converged(),
            "residual-driven spaces did not satisfy the full KKT certificate");
    require(complete.orbital_backward_error < 2.0e-13 &&
                complete.response_backward_error < 2.0e-13,
            "complete-space KKT backward error is too large");
    require_close(
        complete.model_change,
        complete.projected.total_model_change,
        2.0e-13,
        "full and projected coupled model changes disagree");
    const auto dominated = solver.solve(
        1.0,
        {2.0e-13, 2.0e-13},
        complete.predicted_decrease + 1.0);
    require(
        dominated.status ==
            xmvb::vb::CoupledSubspaceStatus::InsufficientPredictedDecrease,
        "projected step incorrectly dominated its certified incumbent");

    const Counts before_radius_change = *counts;
    const auto cached_counts_before_radius_change =
        solver.cache().action_counts();
    const auto smaller = solver.solve(0.35, {2.0e-13, 2.0e-13});
    require(smaller.converged(),
            "radius-only projected re-solve lost its KKT certificate");
    require(counts->a == before_radius_change.a &&
                counts->b == before_radius_change.b &&
                counts->bt == before_radius_change.bt &&
                counts->c == before_radius_change.c &&
                counts->g == before_radius_change.g,
            "radius-only re-solve evaluated a matrix-free action");
    require(solver.cache().action_counts() ==
                cached_counts_before_radius_change,
            "radius-only re-solve changed the production action counters");
    require(smaller.projected.orbital_solution.metric_norm <=
                0.35 * (1.0 + 1.0e-12),
            "radius-only re-solve violates the trust region");

    Eigen::Matrix3d saddle_c;
    saddle_c << 0.0, 1.0, 0.0,
                1.0, 0.0, 0.0,
                0.0, 0.0, 1.0;
    const xmvb::vb::CoupledNewtonOperator saddle_operator(
        4,
        xmvb::vb::SelectedSubspaceResponseLayout(
            2,
            {xmvb::vb::SelectedStateCluster{1, 0.5}}),
        xmvb::vb::CoupledNewtonActions{
            [a](const auto& x) { return (a * x).eval(); },
            [b](const auto& x) { return (b * x).eval(); },
            [b](const auto& x) { return (b.transpose() * x).eval(); },
            [saddle_c](const auto& x) { return (saddle_c * x).eval(); },
            [g](const auto& x) { return (g * x).eval(); }});
    xmvb::vb::CoupledSubspaceSolver saddle_solver(
        saddle_operator,
        orbital_gradient,
        Eigen::Vector3d::UnitX());
    saddle_solver.append_orbital_block(-orbital_gradient);
    saddle_solver.append_response_block(Eigen::Vector3d::UnitX());
    const auto inf_sup =
        saddle_solver.solve(1.0, {1.0e-12, 1.0e-12});
    require(
        inf_sup.status == xmvb::vb::CoupledSubspaceStatus::ExpandResponseSpace,
        "singular projected response block was treated as a terminal failure");
    require((inf_sup.response_expansion_candidate -
             Eigen::Vector3d::UnitY()).norm() <= 2.0e-14,
            "response inf-sup repair direction is inaccurate");

    const Eigen::Matrix<double, 3, 4> zero_b =
        Eigen::Matrix<double, 3, 4>::Zero();
    const xmvb::vb::CoupledNewtonOperator uncoupled_operator(
        4,
        xmvb::vb::SelectedSubspaceResponseLayout(
            2,
            {xmvb::vb::SelectedStateCluster{1, 0.5}}),
        xmvb::vb::CoupledNewtonActions{
            [a](const auto& x) { return (a * x).eval(); },
            [zero_b](const auto& x) { return (zero_b * x).eval(); },
            [zero_b](const auto& x) {
              return (zero_b.transpose() * x).eval();
            },
            [c](const auto& x) { return (c * x).eval(); },
            [g](const auto& x) { return (g * x).eval(); }});
    xmvb::vb::CoupledSubspaceSolver uncoupled_solver(
        uncoupled_operator,
        orbital_gradient,
        Eigen::Vector3d::Zero());
    uncoupled_solver.append_orbital_block(Eigen::Matrix4d::Identity());
    const auto uncoupled =
        uncoupled_solver.solve(1.0, {2.0e-13, 2.0e-13});
    require(uncoupled.converged() && uncoupled.response_step.size() == 3 &&
                uncoupled.response_step.isZero(0.0),
            "valid orbital-only coupled problem required a response basis");

    std::cout << "two-space coupled Newton coordinator: passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
