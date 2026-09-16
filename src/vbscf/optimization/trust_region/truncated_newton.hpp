#pragma once

#include <limits>

#include <Eigen/Core>

#include "vbscf/optimization/preconditioners/transported_lbfgs.hpp"
#include "vbscf/optimization/objective/reduced_hvp.hpp"
#include "vbscf/optimization/trust_region/retraction.hpp"

namespace xmvb::vb {

struct TruncatedNewtonSubspace {
  Eigen::MatrixXd orthonormal_basis;
  Eigen::MatrixXd tangent_basis;
  Eigen::MatrixXd hessian_basis;
  // Full-space metric images, one per admitted Newton direction.
  Eigen::MatrixXd metric_basis;
  Eigen::MatrixXd reduced_hessian;
  // Only this small projected Gram matrix is factored by the trust solver.
  Eigen::MatrixXd reduced_metric;
  Eigen::VectorXd projected_gradient;
};

/** @brief Why the current-point Krylov expansion stopped. */
enum class TruncatedNewtonStopReason {
  None,
  ModelKktConverged,
  BelowOuterAccuracy,
  SubspaceLimit,
  InteriorPilotLimit,
  DependentDirections,
  InvalidProjectedStep,
  RadiusAdjusted,
  PreconditionedGradient,
};

struct TruncatedNewtonStepResult {
  Eigen::VectorXd reduced_step;
  Eigen::VectorXd reduced_hessian_times_step;
  Eigen::VectorXd reduced_metric_times_step;
  TruncatedNewtonSubspace subspace;
  double retract_tangent_norm = 0.0;
  bool reached_boundary = false;
  bool encountered_negative_curvature = false;
  int subspace_dimension = 0;
  double projected_model_gradient_norm = 0.0;
  double model_spectral_radius = 0.0;
  double trust_region_shift = 0.0;
  double predicted_decrease = 0.0;
  /** @brief Computed-model shifted KKT residual divided by Euclidean ||g||. */
  double model_kkt_relative_residual =
      std::numeric_limits<double>::infinity();
  /** @brief The computed shifted model meets its forcing condition. */
  bool model_kkt_converged = false;
  /** @brief An interior unshifted step meets the Newton forcing condition. */
  bool newton_forcing_converged = false;
  TruncatedNewtonStopReason stop_reason = TruncatedNewtonStopReason::None;
};

/** @brief Rechecks model stationarity after a step or its radius changes. */
void refresh_truncated_newton_step_certificate(
    const Eigen::VectorXd& reduced_gradient,
    TruncatedNewtonStepResult* step);

struct TruncatedNewtonTrialEvaluation {
  double actual_decrease = 0.0;
  double predicted_decrease = 0.0;
};

struct RejectedTruncatedNewtonStepCache {
  Eigen::VectorXd cached_step;

  bool has_cached_step(Eigen::Index expected_size) const;
  void clear();
  void update(
      const TruncatedNewtonStepResult& model_step,
      Eigen::Index expected_size,
      double trust_radius,
      const NonredundantRetractionMetric& metric);
};

double inexact_newton_forcing_term(double gradient_norm);

/** @brief Tests the norm-consistent inexact-Newton KKT condition. */
bool inexact_newton_residual_is_converged(
    double gradient_l2_norm,
    double residual_l2_norm);

/**
 * @brief Tests whether further inner work is below both outer accuracies.
 *
 * This is an inner-work stopping rule, not an outer convergence declaration;
 * the accepted trial must still satisfy the measured dual stopping test.
 */
bool truncated_newton_model_is_below_outer_accuracy(
    double gradient_inf_norm,
    double predicted_decrease,
    double gradient_tolerance,
    double energy_tolerance);

/**
 * @brief Accepts a trial only when its measured decrease resolves model error.
 *
 * A merely positive energy change is insufficient evidence for trusting a
 * quadratic step. This criterion requires the actual decrease to exceed the
 * absolute disagreement between actual and predicted decrease.
 */
bool truncated_newton_trial_is_acceptable(
    const TruncatedNewtonTrialEvaluation& trial);

void clamp_nonredundant_step_result_to_retract_tangent_radius(
    const OrbitalChart::ProjectionResult& current_projection,
    double trust_radius,
    const NonredundantRetractionMetric& metric,
    TruncatedNewtonStepResult* step);

bool truncated_newton_subspace_is_usable(
    const TruncatedNewtonSubspace& subspace,
    Eigen::Index reduced_size);

bool truncated_newton_step_is_usable(
    const TruncatedNewtonStepResult& step,
    const Eigen::VectorXd& reduced_gradient);

TruncatedNewtonStepResult solve_trust_region_in_subspace(
    const OrbitalChart::ProjectionResult& current_projection,
    double trust_radius,
    const NonredundantRetractionMetric& metric,
    const TruncatedNewtonSubspace& subspace);

Eigen::VectorXd build_nonredundant_preconditioned_reduced_gradient_step(
    const NonredundantRetractionMetric& retraction_metric,
    const OrbitalChart& space,
    const OrbitalChart::ProjectionResult& projection,
    double trust_radius,
    const TransportedReducedLbfgsPreconditioner* transported_preconditioner);

double update_nonredundant_truncated_newton_trust_radius(
    double trust_radius,
    double minimum_step_size,
    const TruncatedNewtonTrialEvaluation& trial,
    const TruncatedNewtonStepResult& model_step,
    bool accepted);

double estimate_nonredundant_reduced_model_decrease(
    const OrbitalChart::ProjectionResult& projection,
    const Eigen::VectorXd& reduced_step,
    ReducedHvp* hvp);

TruncatedNewtonStepResult solve_nonredundant_truncated_newton_step(
    const NonredundantRetractionMetric& retraction_metric,
    const OrbitalChart& current_space,
    const OrbitalChart::ProjectionResult& current_projection,
    double trust_radius,
    double energy_tolerance,
    double gradient_tolerance,
    int max_subspace_dimension,
    ReducedHvp* hvp,
    const TransportedReducedLbfgsPreconditioner* transported_preconditioner,
    const Eigen::VectorXd* initial_reduced_step = nullptr,
    const TruncatedNewtonSubspace* initial_subspace = nullptr);

}  // namespace xmvb::vb
