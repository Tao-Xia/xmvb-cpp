#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

#include <Eigen/Core>

#include "vbscf/optimization/coupled/workspace.hpp"

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

xmvb::vb::AcceptedPointCoupledModel make_model(
    const Eigen::Matrix4d& a,
    const Eigen::Matrix<double, 3, 4>& b,
    const Eigen::Matrix3d& c,
    const Eigen::Matrix4d& g,
    const Eigen::Vector3d& response_residual,
    const std::shared_ptr<Counts>& counts) {
  xmvb::vb::CoupledNewtonOperator coupled(
      4,
      xmvb::vb::SelectedSubspaceResponseLayout(
          2, {xmvb::vb::SelectedStateCluster{1, 0.5}}),
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
  return xmvb::vb::AcceptedPointCoupledModel{
      {}, std::move(coupled), response_residual,
      [](const Eigen::VectorXd& vector) { return vector; }};
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
    const Eigen::Vector4d gradient(0.4, -0.3, 0.2, 0.1);
    const Eigen::Vector3d residual(0.2, -0.1, 0.3);
    const auto identity_inverse = [](const Eigen::VectorXd& vector) {
      return vector;
    };

    const auto counts = std::make_shared<Counts>();
    xmvb::vb::AcceptedPointCoupledWorkspace workspace(
        make_model(a, b, c, g, residual, counts),
        gradient,
        identity_inverse);
    const auto complete = workspace.solve(
        1.0, xmvb::vb::CoupledKktTolerances{2.0e-13, 2.0e-13});
    require(complete.converged(),
            "accepted-point workspace did not reach its KKT certificate");
    require(complete.orbital_dimension <= 4 &&
                complete.response_dimension <= 3,
            "workspace exceeded the coupled algebraic dimensions");
    require(complete.step.orbital_backward_error <= 2.0e-13 &&
                complete.step.response_backward_error <= 2.0e-13,
            "workspace returned an uncertified full-space KKT residual");
    require(complete.step.predicted_decrease > 0.0,
            "workspace step has no positive exact model decrease");

    const Counts before_radius_change = *counts;
    const auto actions_before_radius_change = complete.action_counts;
    const auto smaller = workspace.solve(
        0.35, xmvb::vb::CoupledKktTolerances{2.0e-13, 2.0e-13});
    require(smaller.converged(),
            "cached radius re-solve lost its KKT certificate");
    require(counts->a == before_radius_change.a &&
                counts->b == before_radius_change.b &&
                counts->bt == before_radius_change.bt &&
                counts->c == before_radius_change.c &&
                counts->g == before_radius_change.g &&
                smaller.action_counts == actions_before_radius_change,
            "radius-only workspace solve evaluated a matrix-free action");
    require(smaller.step.projected.orbital_solution.metric_norm <=
                0.35 * (1.0 + 1.0e-12),
            "cached radius re-solve violated the physical trust metric");

    const auto recycled_counts = std::make_shared<Counts>();
    xmvb::vb::AcceptedPointCoupledWorkspace recycled_workspace(
        make_model(a, b, c, g, residual, recycled_counts),
        gradient,
        identity_inverse,
        complete.step.response_step);
    const auto recycled = recycled_workspace.solve(
        1.0, xmvb::vb::CoupledKktTolerances{2.0e-13, 2.0e-13});
    require(recycled.converged(),
            "recycled response direction lost the KKT certificate");
    require(recycled.step.orbital_backward_error <= 2.0e-13 &&
                recycled.step.response_backward_error <= 2.0e-13,
            "recycled workspace returned an uncertified step");

    const auto limited_counts = std::make_shared<Counts>();
    xmvb::vb::AcceptedPointCoupledWorkspace limited_workspace(
        make_model(a, b, c, g, residual, limited_counts),
        gradient,
        identity_inverse);
    const auto limited = limited_workspace.solve(
        1.0,
        xmvb::vb::CoupledKktTolerances{2.0e-13, 2.0e-13},
        xmvb::vb::CoupledWorkspaceLimits{1, 1});
    require(limited.status == xmvb::vb::CoupledWorkspaceStatus::WorkLimit &&
                !limited.converged(),
            "explicit subspace limit was misreported as convergence");
    require(limited.orbital_dimension == 1 &&
                limited.response_dimension == 1 &&
                (limited.step.orbital_kkt_residual.size() != 0 ||
                 limited.step.response_kkt_residual.size() != 0),
            "work-limited solve did not retain its unresolved certificate");

    const auto zero_counts = std::make_shared<Counts>();
    xmvb::vb::AcceptedPointCoupledWorkspace zero_workspace(
        make_model(
            a, b, c, g, Eigen::Vector3d::Zero(), zero_counts),
        Eigen::Vector4d::Zero(),
        identity_inverse);
    const auto zero = zero_workspace.solve(
        1.0, xmvb::vb::CoupledKktTolerances{0.0, 0.0});
    require(zero.converged() && zero.step.predicted_decrease == 0.0 &&
                zero.action_counts == xmvb::vb::CoupledActionCounts{},
            "stationary accepted point performed unnecessary actions");

    std::cout << "accepted-point coupled workspace: passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
