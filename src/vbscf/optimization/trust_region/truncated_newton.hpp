#pragma once

#include <Eigen/Core>

#include "vbscf/optimization/preconditioners/transported_lbfgs.hpp"
#include "vbscf/optimization/objective/reduced_hvp.hpp"
#include "vbscf/optimization/trust_region/retraction.hpp"

namespace xmvb::vb {

struct TruncatedNewtonKrylovSubspace {
  Eigen::MatrixXd orthonormal_basis;
  Eigen::MatrixXd tangent_basis;
  Eigen::MatrixXd hessian_basis;
  Eigen::MatrixXd reduced_hessian;
  Eigen::VectorXd projected_gradient;
};

struct TruncatedNewtonStepResult {
  Eigen::VectorXd reduced_step;
  Eigen::VectorXd reduced_hessian_times_step;
  TruncatedNewtonKrylovSubspace krylov_subspace;
  double retract_tangent_norm = 0.0;
  bool reached_boundary = false;
  bool encountered_negative_curvature = false;
  bool used_krylov_rescue = false;
  int cg_iterations = 0;
  double projected_model_gradient_norm = 0.0;
  double model_spectral_radius = 0.0;
  double trust_region_shift = 0.0;
  double predicted_decrease = 0.0;
};

struct TruncatedNewtonTrialEvaluation {
  double actual_decrease = 0.0;
  double predicted_decrease = 0.0;
};

struct RejectedTruncatedNewtonStepCache {
  Eigen::VectorXd cached_step;

  bool has_cached_step(Eigen::Index expected_size) const;
  void clear();
  void update(
      const OrbitalPreparationInput& orbital_preparation_input,
      const OrbitalChart& current_space,
      const SparseParameterLayout& parameter_view,
      const TruncatedNewtonStepResult& model_step,
      Eigen::Index expected_size,
      double trust_radius);
};

double inexact_newton_forcing_term(double gradient_norm);

void clamp_nonredundant_step_result_to_retract_tangent_radius(
    const OrbitalPreparationInput& orbital_preparation_input,
    const OrbitalChart& current_space,
    const SparseParameterLayout& parameter_view,
    const OrbitalChart::ProjectionResult& current_projection,
    double trust_radius,
    TruncatedNewtonStepResult* step);

bool truncated_newton_krylov_subspace_is_usable(
    const TruncatedNewtonKrylovSubspace& krylov_subspace,
    Eigen::Index reduced_size);

bool truncated_newton_step_is_usable(
    const TruncatedNewtonStepResult& step,
    const Eigen::VectorXd& reduced_gradient);

TruncatedNewtonStepResult solve_trust_region_in_krylov_subspace(
    const OrbitalChart::ProjectionResult& current_projection,
    double trust_radius,
    const TruncatedNewtonKrylovSubspace& krylov_subspace);

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
    double gradient_tolerance,
    int max_cg_iterations,
    ReducedHvp* hvp,
    const TransportedReducedLbfgsPreconditioner* transported_preconditioner,
    const Eigen::VectorXd* initial_reduced_step = nullptr);

}  // namespace xmvb::vb
