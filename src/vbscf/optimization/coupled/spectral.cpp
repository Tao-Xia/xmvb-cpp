#include "vbscf/optimization/coupled/spectral.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <Eigen/Eigenvalues>
#include <Eigen/Cholesky>

namespace xmvb::vb {
namespace {

void validate_trial_basis(
    const CoupledNewtonOperator& coupled_operator,
    const Eigen::Ref<const Eigen::MatrixXd>& basis) {
  if (basis.rows() != coupled_operator.n_orbital_coordinates() ||
      basis.cols() <= 0 ||
      basis.cols() > coupled_operator.n_orbital_coordinates() ||
      !basis.allFinite()) {
    throw std::invalid_argument(
        "coupled spectral trial basis has invalid dimensions or values");
  }
}

double roundoff_floor(double scale, Eigen::Index dimension) {
  return std::numeric_limits<double>::epsilon() *
      std::max<Eigen::Index>(1, dimension) * std::max(1.0, scale);
}

Eigen::VectorXd deterministic_expansion(
    const CoupledNewtonOperator& coupled_operator,
    const Eigen::Ref<const Eigen::MatrixXd>& basis,
    const Eigen::Ref<const Eigen::VectorXd>& orbital_residual) {
  if (basis.cols() == basis.rows()) return Eigen::VectorXd();
  const Eigen::MatrixXd metric_basis =
      coupled_operator.apply_orbital_metric(basis);
  const Eigen::VectorXd metric_residual =
      coupled_operator.apply_orbital_metric(orbital_residual).col(0);
  const Eigen::MatrixXd basis_metric = 0.5 *
      (basis.transpose() * metric_basis +
       metric_basis.transpose() * basis);
  Eigen::LDLT<Eigen::MatrixXd> basis_metric_factor(basis_metric);
  if (basis_metric_factor.info() != Eigen::Success ||
      !basis_metric_factor.isPositive()) {
    throw std::runtime_error(
        "orbital trial basis metric factorization failed");
  }
  const Eigen::VectorXd projection = basis_metric_factor.solve(
      basis.transpose() * metric_residual);
  if (basis_metric_factor.info() != Eigen::Success ||
      !projection.allFinite()) {
    throw std::runtime_error(
        "orbital residual metric projection failed");
  }
  Eigen::VectorXd expansion = orbital_residual - basis * projection;
  const double residual_scale = orbital_residual.stableNorm();
  if (expansion.stableNorm() <=
      roundoff_floor(residual_scale, basis.rows())) {
    return Eigen::VectorXd();
  }
  const Eigen::VectorXd metric_image =
      coupled_operator.apply_orbital_metric(expansion).col(0);
  const double metric_norm_squared = expansion.dot(metric_image);
  if (!(metric_norm_squared > 0.0) ||
      !std::isfinite(metric_norm_squared)) {
    throw std::runtime_error(
        "orbital metric is not positive on the residual expansion");
  }
  expansion /= std::sqrt(metric_norm_squared);
  Eigen::Index pivot = 0;
  expansion.cwiseAbs().maxCoeff(&pivot);
  if (expansion[pivot] < 0.0) expansion = -expansion;
  return expansion;
}

}  // namespace

CoupledResponseStationaryLifts build_response_stationary_lifts(
    const CoupledNewtonOperator& coupled_operator,
    const Eigen::Ref<const Eigen::MatrixXd>& orbital_trial_basis,
    const CoupledSpectralOptions& options,
    const SymmetricOperatorAction& response_inverse_preconditioner) {
  validate_trial_basis(coupled_operator, orbital_trial_basis);
  const Eigen::MatrixXd forcing =
      coupled_operator.apply_orbital_to_response(orbital_trial_basis);
  CoupledResponseStationaryLifts lifts;
  lifts.values.resize(
      coupled_operator.n_response_coordinates(),
      orbital_trial_basis.cols());
  lifts.residual_targets.resize(orbital_trial_basis.cols());
  lifts.stop_reasons.reserve(
      static_cast<std::size_t>(orbital_trial_basis.cols()));
  for (Eigen::Index column = 0;
       column < orbital_trial_basis.cols();
       ++column) {
    const MinresResult solve = solve_symmetric_minres(
        [&](const Eigen::VectorXd& direction) {
          return coupled_operator.apply_response_hessian(direction)
              .col(0)
              .eval();
        },
        -forcing.col(column),
        options.response_solve,
        response_inverse_preconditioner);
    lifts.values.col(column) = solve.solution;
    lifts.residual_targets[column] = solve.residual_target;
    lifts.stop_reasons.push_back(solve.stop_reason);
    lifts.total_iterations += solve.iterations;
    lifts.total_operator_actions += solve.operator_actions;
    lifts.total_residual_checks += solve.residual_checks;
  }
  return lifts;
}

CoupledSpectralAnalysis analyze_coupled_explored_spectrum(
    const CoupledNewtonOperator& coupled_operator,
    const Eigen::Ref<const Eigen::MatrixXd>& orbital_trial_basis,
    const CoupledResponseStationaryLifts& response_lifts) {
  validate_trial_basis(coupled_operator, orbital_trial_basis);
  if (response_lifts.values.rows() !=
          coupled_operator.n_response_coordinates() ||
      response_lifts.values.cols() != orbital_trial_basis.cols() ||
      response_lifts.residual_targets.size() != orbital_trial_basis.cols() ||
      response_lifts.stop_reasons.size() !=
          static_cast<std::size_t>(orbital_trial_basis.cols()) ||
      !response_lifts.values.allFinite() ||
      !response_lifts.residual_targets.allFinite() ||
      (response_lifts.residual_targets.array() < 0.0).any()) {
    throw std::invalid_argument(
        "coupled response lifts have invalid dimensions or certificates");
  }

  const int n_orbitals = coupled_operator.n_orbital_coordinates();
  const int n_response = coupled_operator.n_response_coordinates();
  Eigen::MatrixXd full_basis(coupled_operator.size(), orbital_trial_basis.cols());
  full_basis.topRows(n_orbitals) = orbital_trial_basis;
  full_basis.bottomRows(n_response) = response_lifts.values;
  const Eigen::MatrixXd full_images =
      coupled_operator.apply_block(full_basis);
  const Eigen::MatrixXd metric_images =
      coupled_operator.apply_orbital_metric(orbital_trial_basis);

  CoupledSpectralAnalysis analysis;
  analysis.status = orbital_trial_basis.cols() == n_orbitals
      ? CoupledSpectralStatus::FullOrbitalSpace
      : CoupledSpectralStatus::ExploredSubspace;
  analysis.projected_relaxed_hessian = 0.5 *
      (orbital_trial_basis.transpose() * full_images.topRows(n_orbitals) +
       full_images.topRows(n_orbitals).transpose() * orbital_trial_basis);
  analysis.projected_orbital_metric = 0.5 *
      (orbital_trial_basis.transpose() * metric_images +
       metric_images.transpose() * orbital_trial_basis);

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> metric_solver(
      analysis.projected_orbital_metric);
  if (metric_solver.info() != Eigen::Success ||
      !metric_solver.eigenvalues().allFinite()) {
    throw std::runtime_error(
        "projected orbital metric eigensolve failed");
  }
  const double metric_scale =
      metric_solver.eigenvalues().cwiseAbs().maxCoeff();
  if (!(metric_solver.eigenvalues().minCoeff() >
        roundoff_floor(metric_scale, orbital_trial_basis.cols()))) {
    throw std::invalid_argument(
        "orbital trial basis is rank deficient in the orbital metric");
  }

  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> projected_solver(
      analysis.projected_relaxed_hessian,
      analysis.projected_orbital_metric);
  if (projected_solver.info() != Eigen::Success ||
      !projected_solver.eigenvalues().allFinite() ||
      !projected_solver.eigenvectors().allFinite()) {
    throw std::runtime_error(
        "projected relaxed generalized eigensolve failed");
  }
  analysis.leftmost_ritz_value = projected_solver.eigenvalues()[0];
  Eigen::VectorXd projected_vector = projected_solver.eigenvectors().col(0);
  Eigen::VectorXd orbital_vector = orbital_trial_basis * projected_vector;
  Eigen::VectorXd orbital_metric_image =
      metric_images * projected_vector;
  const double orbital_norm_squared =
      orbital_vector.dot(orbital_metric_image);
  if (!(orbital_norm_squared > 0.0) ||
      !std::isfinite(orbital_norm_squared)) {
    throw std::runtime_error(
        "leftmost Ritz vector has an invalid orbital metric norm");
  }
  const double inverse_norm = 1.0 / std::sqrt(orbital_norm_squared);
  projected_vector *= inverse_norm;
  orbital_vector *= inverse_norm;
  orbital_metric_image *= inverse_norm;
  Eigen::VectorXd response_vector =
      response_lifts.values * projected_vector;

  analysis.leftmost_orbital_ritz_vector = orbital_vector;
  analysis.leftmost_response_lift = response_vector;
  analysis.full_ritz_vector.resize(coupled_operator.size());
  analysis.full_ritz_vector.head(n_orbitals) = orbital_vector;
  analysis.full_ritz_vector.tail(n_response) = response_vector;
  const Eigen::VectorXd full_image = full_images * projected_vector;
  analysis.response_stationary_rayleigh_quotient =
      analysis.full_ritz_vector.dot(full_image);
  analysis.full_residual = full_image;
  analysis.full_residual.head(n_orbitals).noalias() -=
      analysis.leftmost_ritz_value * orbital_metric_image;
  analysis.full_residual_norm = analysis.full_residual.stableNorm();
  analysis.relative_full_residual = analysis.full_residual_norm /
      std::max({1.0,
                full_image.stableNorm(),
                std::abs(analysis.leftmost_ritz_value) *
                    orbital_metric_image.stableNorm()});

  const Eigen::MatrixXd stationarity_residuals =
      full_images.bottomRows(n_response);
  analysis.response_stationarity_residual_norms.resize(
      orbital_trial_basis.cols());
  for (Eigen::Index column = 0;
       column < orbital_trial_basis.cols();
       ++column) {
    analysis.response_stationarity_residual_norms[column] =
        stationarity_residuals.col(column).stableNorm();
  }
  analysis.left_response_stationarity_residual =
      analysis.full_residual.tail(n_response).stableNorm();
  analysis.left_response_stationarity_target =
      projected_vector.cwiseAbs().dot(response_lifts.residual_targets);
  const bool all_response_solves_converged = std::all_of(
      response_lifts.stop_reasons.begin(),
      response_lifts.stop_reasons.end(),
      [](MinresStopReason reason) {
        return reason == MinresStopReason::Converged;
      });
  analysis.response_stationarity_certified =
      all_response_solves_converged &&
      analysis.left_response_stationarity_residual <=
      analysis.left_response_stationarity_target;
  analysis.negative_curvature_certified =
      analysis.response_stationarity_certified &&
      analysis.response_stationary_rayleigh_quotient < 0.0;
  analysis.orbital_residual_expansion = deterministic_expansion(
      coupled_operator,
      orbital_trial_basis,
      analysis.full_residual.head(n_orbitals));
  return analysis;
}

CoupledSpectralAnalysis analyze_coupled_explored_spectrum(
    const CoupledNewtonOperator& coupled_operator,
    const Eigen::Ref<const Eigen::MatrixXd>& orbital_trial_basis,
    const CoupledSpectralOptions& options,
    const SymmetricOperatorAction& response_inverse_preconditioner) {
  return analyze_coupled_explored_spectrum(
      coupled_operator,
      orbital_trial_basis,
      build_response_stationary_lifts(
          coupled_operator,
          orbital_trial_basis,
          options,
          response_inverse_preconditioner));
}

}  // namespace xmvb::vb
