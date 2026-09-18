#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#include "vbscf/optimization/coupled/operator.hpp"
#include "vbscf/optimization/coupled/spectral.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void require_close(
    double actual,
    double reference,
    double tolerance,
    const char* message) {
  require(
      std::abs(actual - reference) <=
          tolerance * std::max({1.0, std::abs(actual), std::abs(reference)}),
      message);
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
    const Eigen::Matrix3d& orbital_hessian,
    const Eigen::Matrix<double, 2, 3>& coupling,
    const Eigen::Matrix2d& response_hessian,
    const Eigen::Matrix3d& metric) {
  return xmvb::vb::CoupledNewtonOperator(
      3,
      xmvb::vb::SelectedSubspaceResponseLayout(
          1,
          {xmvb::vb::SelectedStateCluster{1, 0.5}}),
      xmvb::vb::CoupledNewtonActions{
          [orbital_hessian](const Eigen::Ref<const Eigen::MatrixXd>& x) {
            return (orbital_hessian * x).eval();
          },
          [coupling](const Eigen::Ref<const Eigen::MatrixXd>& x) {
            return (coupling * x).eval();
          },
          [coupling](const Eigen::Ref<const Eigen::MatrixXd>& x) {
            return (coupling.transpose() * x).eval();
          },
          [response_hessian](const Eigen::Ref<const Eigen::MatrixXd>& x) {
            return (response_hessian * x).eval();
          },
          [metric](const Eigen::Ref<const Eigen::MatrixXd>& x) {
            return (metric * x).eval();
          }});
}

double leftmost_generalized_eigenvalue(
    const Eigen::Ref<const Eigen::MatrixXd>& hessian,
    const Eigen::Ref<const Eigen::MatrixXd>& metric) {
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(
      hessian,
      metric);
  require(solver.info() == Eigen::Success,
          "dense generalized reference eigensolve failed");
  return solver.eigenvalues()[0];
}

}  // namespace

int main() {
  try {
    Eigen::Matrix3d orbital_hessian;
    orbital_hessian << -0.4, 0.2, 0.5,
                         0.2, 1.0, 0.4,
                         0.5, 0.4, 2.0;
    Eigen::Matrix<double, 2, 3> coupling;
    coupling << 0.4, -0.1, 0.2,
                0.1,  0.3, -0.2;
    Eigen::Matrix2d response_hessian;
    response_hessian << 2.0, 0.2,
                        0.2, 1.5;
    Eigen::Matrix3d metric;
    metric << 1.5, 0.1, 0.0,
              0.1, 0.9, 0.05,
              0.0, 0.05, 1.2;
    const auto coupled_operator = make_operator(
        orbital_hessian,
        coupling,
        response_hessian,
        metric);
    const Eigen::Matrix3d relaxed = orbital_hessian -
        coupling.transpose() * response_hessian.inverse() * coupling;

    xmvb::vb::CoupledSpectralOptions options;
    options.response_solve.relative_residual_tolerance = 1.0e-13;
    options.response_solve.absolute_residual_tolerance = 1.0e-14;

    Eigen::Matrix<double, 3, 2> explored_basis;
    explored_basis << 1.0, 0.0,
                      0.0, 1.0,
                      0.0, 0.0;
    const xmvb::vb::CoupledResponseStationaryLifts built_lifts =
        xmvb::vb::build_response_stationary_lifts(
            coupled_operator,
            explored_basis,
            options);
    const auto explored = xmvb::vb::analyze_coupled_explored_spectrum(
        coupled_operator,
        explored_basis,
        built_lifts);
    require(
        explored.status ==
            xmvb::vb::CoupledSpectralStatus::ExploredSubspace,
        "partial trial basis was presented as a full-space result");
    const Eigen::Matrix2d projected_relaxed =
        explored_basis.transpose() * relaxed * explored_basis;
    const Eigen::Matrix2d projected_metric =
        explored_basis.transpose() * metric * explored_basis;
    require_close(
        explored.leftmost_ritz_value,
        leftmost_generalized_eigenvalue(
            projected_relaxed,
            projected_metric),
        1.0e-12,
        "explored-subspace Ritz value differs from the dense Schur reference");
    require(explored.response_stationarity_certified,
            "constructed response lift lacks a stationarity certificate");
    require(explored.negative_curvature_certified,
            "negative response-stationary curvature was not certified");
    require_close(
        explored.response_stationary_rayleigh_quotient,
        explored.leftmost_ritz_value,
        1.0e-12,
        "stationary Rayleigh quotient differs from its Ritz value");
    const Eigen::Vector3d dense_orbital_residual =
        relaxed * explored.leftmost_orbital_ritz_vector -
        explored.leftmost_ritz_value * metric *
            explored.leftmost_orbital_ritz_vector;
    require_close(
        explored.full_residual.head<3>(),
        dense_orbital_residual,
        1.0e-12,
        "matrix-free Ritz residual differs from the dense Schur residual");
    require(explored.orbital_residual_expansion.size() == 3,
            "nonzero Ritz residual did not provide expansion guidance");
    require((explored_basis.transpose() * metric *
                 explored.orbital_residual_expansion).norm() < 1.0e-12,
            "residual expansion is not G-orthogonal to the explored basis");
    require(explored.orbital_residual_expansion[2] > 0.0,
            "residual expansion sign convention is not deterministic");

    xmvb::vb::CoupledResponseStationaryLifts exact_lifts;
    exact_lifts.values =
        -response_hessian.inverse() * coupling * explored_basis;
    exact_lifts.residual_targets = Eigen::Vector2d::Constant(1.0e-13);
    exact_lifts.stop_reasons.assign(
        2,
        xmvb::vb::MinresStopReason::Converged);
    const auto consumed = xmvb::vb::analyze_coupled_explored_spectrum(
        coupled_operator,
        explored_basis,
        exact_lifts);
    require_close(
        consumed.leftmost_ritz_value,
        explored.leftmost_ritz_value,
        1.0e-12,
        "consumed exact lifts changed the explored Ritz value");
    require_close(
        consumed.full_residual,
        explored.full_residual,
        1.0e-12,
        "constructed and consumed response lifts disagree");

    xmvb::vb::CoupledResponseStationaryLifts unstationary_lifts;
    unstationary_lifts.values = Eigen::MatrixXd::Zero(2, 2);
    unstationary_lifts.residual_targets = Eigen::Vector2d::Zero();
    unstationary_lifts.stop_reasons.assign(
        2,
        xmvb::vb::MinresStopReason::Converged);
    const auto unstationary =
        xmvb::vb::analyze_coupled_explored_spectrum(
            coupled_operator,
            explored_basis,
            unstationary_lifts);
    require(!unstationary.response_stationarity_certified &&
                !unstationary.negative_curvature_certified,
            "nonstationary response lift produced a curvature certificate");

    xmvb::vb::CoupledSpectralOptions failed_options = options;
    failed_options.response_solve.maximum_iterations = 1;
    const xmvb::vb::CoupledResponseStationaryLifts failed_lifts =
        xmvb::vb::build_response_stationary_lifts(
            coupled_operator,
            explored_basis,
            failed_options);
    require(std::any_of(
                failed_lifts.stop_reasons.begin(),
                failed_lifts.stop_reasons.end(),
                [](xmvb::vb::MinresStopReason reason) {
                  return reason != xmvb::vb::MinresStopReason::Converged;
                }),
            "bounded response solve unexpectedly hid its termination state");
    const auto failed = xmvb::vb::analyze_coupled_explored_spectrum(
        coupled_operator,
        explored_basis,
        failed_lifts);
    require(!failed.response_stationarity_certified &&
                !failed.negative_curvature_certified,
            "nonconverged response solve was silently certified");

    const Eigen::Matrix3d full_basis = Eigen::Matrix3d::Identity();
    const auto full = xmvb::vb::analyze_coupled_explored_spectrum(
        coupled_operator,
        full_basis,
        options);
    require(
        full.status ==
            xmvb::vb::CoupledSpectralStatus::FullOrbitalSpace,
        "complete orbital basis was not identified as full-space coverage");
    require_close(
        full.leftmost_ritz_value,
        leftmost_generalized_eigenvalue(relaxed, metric),
        1.0e-12,
        "full-space matrix-free Ritz value differs from dense Schur reference");
    require(explored.leftmost_ritz_value >=
                full.leftmost_ritz_value - 1.0e-12,
            "explored Ritz value violates the dense variational reference");
    require(full.relative_full_residual < 1.0e-12,
            "full-space Ritz vector has a large matrix-free residual");
    require(full.orbital_residual_expansion.size() == 0,
            "full orbital space requested unnecessary expansion");

    std::cout << "coupled explored-subspace spectral analysis: passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
