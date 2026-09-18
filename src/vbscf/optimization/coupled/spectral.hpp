#pragma once

#include <limits>
#include <vector>

#include <Eigen/Core>

#include "vbscf/optimization/coupled/operator.hpp"
#include "vbscf/optimization/krylov/minres.hpp"

namespace xmvb::vb {

/** @brief Coverage represented by a coupled spectral Ritz analysis. */
enum class CoupledSpectralStatus {
  /** The result describes only the supplied orbital trial subspace. */
  ExploredSubspace,
  /** The supplied basis spans the complete orbital coordinate space. */
  FullOrbitalSpace,
};

/**
 * @brief Response-stationary lifts associated with an orbital trial basis.
 *
 * Column @f$j@f$ approximates @f$q_j=-C^{-1}Bv_j@f$.
 * `residual_targets[j]` is an absolute certificate target for
 * @f$\|Bv_j+Cq_j\|_2@f$. Consumers always recompute the true residual; these
 * targets are not trusted as measured residuals.
 */
struct CoupledResponseStationaryLifts {
  Eigen::MatrixXd values;
  Eigen::VectorXd residual_targets;
  /** @brief Explicit MINRES outcome for every lift column. */
  std::vector<MinresStopReason> stop_reasons;
  int total_iterations = 0;
  int total_operator_actions = 0;
  int total_residual_checks = 0;
};

/** @brief Controls the matrix-free response-stationarity solves. */
struct CoupledSpectralOptions {
  MinresOptions response_solve;
};

/**
 * @brief Honest Ritz information from one explored orbital subspace.
 *
 * `leftmost_ritz_value` is a projected relaxed-Hessian Ritz value. It is not a
 * global spectral lower bound when `status == ExploredSubspace`.
 */
struct CoupledSpectralAnalysis {
  CoupledSpectralStatus status =
      CoupledSpectralStatus::ExploredSubspace;
  Eigen::VectorXd leftmost_orbital_ritz_vector;
  Eigen::VectorXd leftmost_response_lift;
  Eigen::VectorXd full_ritz_vector;
  Eigen::VectorXd full_residual;
  /** Deterministic orbital direction suggested for trial-space expansion. */
  Eigen::VectorXd orbital_residual_expansion;
  Eigen::VectorXd response_stationarity_residual_norms;
  /** @brief Exact projected quadratic form @f$X^TKX@f$, @f$X=(V,Q)@f$. */
  Eigen::MatrixXd projected_coupled_hessian;
  /** @brief Stationary-limit Schur projection @f$V^T(AV+B^TQ)@f$. */
  Eigen::MatrixXd projected_relaxed_hessian;
  Eigen::MatrixXd projected_orbital_metric;
  double leftmost_ritz_value =
      std::numeric_limits<double>::quiet_NaN();
  double response_stationary_rayleigh_quotient =
      std::numeric_limits<double>::quiet_NaN();
  double full_residual_norm = std::numeric_limits<double>::infinity();
  double relative_full_residual = std::numeric_limits<double>::infinity();
  double left_response_stationarity_residual =
      std::numeric_limits<double>::infinity();
  double left_response_stationarity_target = 0.0;
  bool response_stationarity_certified = false;
};

/**
 * @brief Constructs @f$q=-C^{-1}BV@f$ by matrix-free MINRES solves.
 *
 * The supplied inverse preconditioner acts only in response coordinates and
 * must be symmetric positive definite. There is no fixed iteration cap unless
 * one is explicitly supplied through `options`.
 */
CoupledResponseStationaryLifts build_response_stationary_lifts(
    const CoupledNewtonOperator& coupled_operator,
    const Eigen::Ref<const Eigen::MatrixXd>& orbital_trial_basis,
    const CoupledSpectralOptions& options = {},
    const SymmetricOperatorAction& response_inverse_preconditioner = {});

/**
 * @brief Analyzes the relaxed spectrum in an orbital trial subspace.
 *
 * The supplied lifts are independently checked against @f$BV+CQ=0@f$. The
 * projected generalized problem is
 *
 * @f[
 * V^T(A V+B^TQ)y=\theta V^TGVy.
 * @f]
 *
 * The returned full residual is evaluated by the original coupled operator,
 * not reconstructed from the projected matrices. A finite response solve
 * cannot certify the sign of the exact Schur curvature without an inverse
 * error bound for @f$C@f$; the coupled Rayleigh quotient is therefore retained
 * only as a diagnostic.
 */
CoupledSpectralAnalysis analyze_coupled_explored_spectrum(
    const CoupledNewtonOperator& coupled_operator,
    const Eigen::Ref<const Eigen::MatrixXd>& orbital_trial_basis,
    const CoupledResponseStationaryLifts& response_lifts);

/** @brief Convenience overload that first constructs stationary lifts. */
CoupledSpectralAnalysis analyze_coupled_explored_spectrum(
    const CoupledNewtonOperator& coupled_operator,
    const Eigen::Ref<const Eigen::MatrixXd>& orbital_trial_basis,
    const CoupledSpectralOptions& options,
    const SymmetricOperatorAction& response_inverse_preconditioner = {});

}  // namespace xmvb::vb
