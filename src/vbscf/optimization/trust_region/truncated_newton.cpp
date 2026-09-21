#include "vbscf/optimization/trust_region/truncated_newton.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Eigenvalues>
#include <Eigen/Cholesky>

#include "vbscf/optimization/krylov/orthonormal_basis.hpp"
#include "vbscf/optimization/krylov/positive_ritz_secants.hpp"
#include "vbscf/optimization/preconditioners/block_inverse_bfgs.hpp"
#include "vbscf/optimization/trust_region/spectral.hpp"
#include "vbscf/optimization/driver/checks.hpp"

namespace xmvb::vb {

double inexact_newton_forcing_term(
    double gradient_l2_norm,
    double gradient_inf_norm,
    double initial_gradient_l2_norm,
    double gradient_tolerance) {
  if (!std::isfinite(gradient_l2_norm) || gradient_l2_norm < 0.0 ||
      !std::isfinite(gradient_inf_norm) || gradient_inf_norm < 0.0 ||
      !std::isfinite(initial_gradient_l2_norm) ||
      initial_gradient_l2_norm <= 0.0 ||
      !std::isfinite(gradient_tolerance) ||
      gradient_tolerance <= 0.0) {
    throw std::invalid_argument(
        "inexact Newton forcing requires a valid outer accuracy contract");
  }
  constexpr double kContractionLimit = 0.5;
  const double progress_ratio =
      gradient_l2_norm / initial_gradient_l2_norm;
  const double accuracy_floor =
      std::sqrt(gradient_tolerance / initial_gradient_l2_norm);
  double forcing = std::min(
      kContractionLimit,
      std::max(progress_ratio, accuracy_floor));
  // If even the maximum allowed inexact-Newton contraction would bring the
  // componentwise gradient below the requested outer threshold, solve the
  // current equation accurately enough to make that one-step target
  // attainable.  This is an accuracy contract, not a system-specific switch.
  if (kContractionLimit * gradient_inf_norm <= gradient_tolerance) {
    const double outer_target = gradient_tolerance /
        std::max(gradient_tolerance, gradient_l2_norm);
    forcing = std::min(forcing, outer_target);
  }
  return forcing;
}

bool observed_contraction_requires_newton_correction(
    double source_gradient_l2_norm,
    double accepted_gradient_l2_norm,
    double forcing_term) {
  if (!std::isfinite(source_gradient_l2_norm) ||
      !std::isfinite(accepted_gradient_l2_norm) ||
      !std::isfinite(forcing_term) ||
      source_gradient_l2_norm <= 0.0 ||
      accepted_gradient_l2_norm < 0.0 ||
      forcing_term < 0.0 || forcing_term >= 1.0) {
    return true;
  }
  return accepted_gradient_l2_norm >
      forcing_term * source_gradient_l2_norm;
}

void refresh_truncated_newton_step_certificate(
    const Eigen::VectorXd& reduced_gradient,
    TruncatedNewtonStepResult* step) {
  if (step == nullptr) return;
  step->model_kkt_relative_residual =
      std::numeric_limits<double>::infinity();
  step->model_kkt_converged = false;
  step->newton_forcing_converged = false;
  if (reduced_gradient.size() == 0 ||
      !reduced_gradient.allFinite() ||
      step->reduced_step.size() != reduced_gradient.size() ||
      step->reduced_hessian_times_step.size() != reduced_gradient.size() ||
      step->reduced_metric_times_step.size() != reduced_gradient.size() ||
      !step->reduced_step.allFinite() ||
      !step->reduced_hessian_times_step.allFinite() ||
      !step->reduced_metric_times_step.allFinite() ||
      !std::isfinite(step->trust_region_shift) ||
      step->trust_region_shift < 0.0) {
    return;
  }

  const double gradient_norm = reduced_gradient.stableNorm();
  if (!std::isfinite(step->target_kkt_relative_residual) ||
      step->target_kkt_relative_residual < 0.0 ||
      step->target_kkt_relative_residual >= 1.0) {
    return;
  }
  const Eigen::VectorXd residual = reduced_gradient +
      step->reduced_hessian_times_step +
      step->trust_region_shift * step->reduced_metric_times_step;
  const double residual_norm = residual.stableNorm();
  if (!std::isfinite(gradient_norm) || !std::isfinite(residual_norm)) return;
  step->model_kkt_relative_residual = gradient_norm > 0.0
      ? residual_norm / gradient_norm
      : (residual_norm == 0.0 ? 0.0
                              : std::numeric_limits<double>::infinity());
  step->model_kkt_converged = residual_norm <=
      step->target_kkt_relative_residual * gradient_norm;
  step->newton_forcing_converged =
      step->model_kkt_converged &&
      step->trust_region_shift == 0.0 &&
      !step->reached_boundary &&
      !step->encountered_negative_curvature;
}

bool truncated_newton_model_is_below_outer_accuracy(
    double gradient_inf_norm,
    double predicted_decrease,
    double gradient_tolerance,
    double energy_tolerance) {
  return std::isfinite(gradient_inf_norm) &&
      std::isfinite(predicted_decrease) &&
      std::isfinite(gradient_tolerance) &&
      std::isfinite(energy_tolerance) &&
      gradient_tolerance > 0.0 &&
      energy_tolerance > 0.0 &&
      gradient_inf_norm <= gradient_tolerance &&
      predicted_decrease >= 0.0 &&
      predicted_decrease <= energy_tolerance;
}

bool truncated_newton_trial_is_acceptable(
    const TruncatedNewtonTrialEvaluation& trial,
    bool accept_model_inaccurate_monotone) {
  if (!(trial.predicted_decrease > 0.0) ||
      !(trial.actual_decrease > 0.0) ||
      !std::isfinite(trial.predicted_decrease) ||
      !std::isfinite(trial.actual_decrease)) {
    return false;
  }
  const double roundoff =
      16.0 * std::numeric_limits<double>::epsilon() *
      trial.predicted_decrease;
  if (!(trial.actual_decrease > roundoff)) return false;
  if (accept_model_inaccurate_monotone) return true;
  const double model_error =
      std::abs(trial.actual_decrease - trial.predicted_decrease);
  return model_error <= trial.actual_decrease + roundoff;
}

double truncated_newton_step_effective_norm(
    const TruncatedNewtonStepResult& step) {
  return std::isfinite(step.retract_tangent_norm) &&
                 step.retract_tangent_norm > 0.0
             ? step.retract_tangent_norm
             : 0.0;
}

void clamp_nonredundant_step_result_to_retract_tangent_radius(
    const OrbitalChart::ProjectionResult& current_projection,
    double trust_radius,
    const NonredundantRetractionMetric& metric,
    TruncatedNewtonStepResult* step) {
  if (step == nullptr ||
      step->reduced_step.size() != current_projection.reduced_gradient.size() ||
      step->reduced_step.size() == 0 ||
      !step->reduced_step.allFinite()) {
    return;
  }

  const double tangent_norm =
      metric.norm(step->reduced_step);
  step->retract_tangent_norm = tangent_norm;
  if (!(trust_radius > 0.0) ||
      !std::isfinite(trust_radius) ||
      !(tangent_norm > 0.0) ||
      !std::isfinite(tangent_norm)) {
    return;
  }

  if (tangent_norm > trust_radius) {
    const double scale = trust_radius / tangent_norm;
    step->reduced_step *= scale;
    if (step->reduced_hessian_times_step.size() ==
            current_projection.reduced_gradient.size() &&
        step->reduced_hessian_times_step.allFinite()) {
      step->reduced_hessian_times_step *= scale;
      if (step->reduced_metric_times_step.size() == step->reduced_step.size()) {
        step->reduced_metric_times_step *= scale;
      }
      step->predicted_decrease =
          -current_projection.reduced_gradient.dot(step->reduced_step) -
          0.5 * step->reduced_step.dot(step->reduced_hessian_times_step);
    } else {
      step->predicted_decrease = 0.0;
    }
    step->retract_tangent_norm = trust_radius;
    step->reached_boundary = true;
    step->stop_reason = TruncatedNewtonStopReason::RadiusAdjusted;
    refresh_truncated_newton_step_certificate(
        current_projection.reduced_gradient, step);
    return;
  }

  step->reached_boundary =
      step->reached_boundary ||
      tangent_norm >= (1.0 - 1.0e-8) * trust_radius;
  refresh_truncated_newton_step_certificate(
      current_projection.reduced_gradient, step);
}

bool minimize_truncated_newton_step_on_ray(
    const OrbitalChart::ProjectionResult& current_projection,
    TruncatedNewtonStepResult* step) {
  if (step == nullptr || step->reduced_step.size() == 0 ||
      step->reduced_step.size() !=
          current_projection.reduced_gradient.size() ||
      step->reduced_hessian_times_step.size() !=
          step->reduced_step.size() ||
      step->reduced_metric_times_step.size() !=
          step->reduced_step.size() ||
      !step->reduced_step.allFinite() ||
      !step->reduced_hessian_times_step.allFinite() ||
      !step->reduced_metric_times_step.allFinite()) {
    return false;
  }
  const double linear_decrease =
      -current_projection.reduced_gradient.dot(step->reduced_step);
  const double directional_curvature =
      step->reduced_step.dot(step->reduced_hessian_times_step);
  if (!(linear_decrease > 0.0) || !(directional_curvature > 0.0) ||
      !std::isfinite(linear_decrease) ||
      !std::isfinite(directional_curvature)) {
    return false;
  }
  const double scale = linear_decrease / directional_curvature;
  if (!(scale > 0.0) || !(scale < 1.0) || !std::isfinite(scale)) {
    return false;
  }

  step->reduced_step *= scale;
  step->reduced_hessian_times_step *= scale;
  step->reduced_metric_times_step *= scale;
  step->retract_tangent_norm *= scale;
  step->trust_region_shift = 0.0;
  step->reached_boundary = false;
  step->predicted_decrease =
      -current_projection.reduced_gradient.dot(step->reduced_step) -
      0.5 * step->reduced_step.dot(
          step->reduced_hessian_times_step);
  step->stop_reason = TruncatedNewtonStopReason::RadiusAdjusted;
  refresh_truncated_newton_step_certificate(
      current_projection.reduced_gradient, step);
  return true;
}

bool truncated_newton_subspace_is_usable(
    const TruncatedNewtonSubspace& subspace,
    Eigen::Index reduced_size) {
  return
      reduced_size >= 0 &&
      subspace.orthonormal_basis.rows() == reduced_size &&
      subspace.orthonormal_basis.cols() > 0 &&
      subspace.orthonormal_basis.allFinite() &&
      subspace.tangent_basis.cols() == subspace.orthonormal_basis.cols() &&
      subspace.tangent_basis.allFinite() &&
      subspace.hessian_basis.rows() == reduced_size &&
      subspace.hessian_basis.cols() == subspace.orthonormal_basis.cols() &&
      subspace.hessian_basis.allFinite() &&
      subspace.metric_basis.rows() == reduced_size &&
      subspace.metric_basis.cols() == subspace.orthonormal_basis.cols() &&
      subspace.metric_basis.allFinite() &&
      subspace.reduced_hessian.rows() == subspace.orthonormal_basis.cols() &&
      subspace.reduced_hessian.cols() == subspace.orthonormal_basis.cols() &&
      subspace.reduced_hessian.allFinite() &&
      subspace.reduced_metric.rows() == subspace.orthonormal_basis.cols() &&
      subspace.reduced_metric.cols() == subspace.orthonormal_basis.cols() &&
      subspace.reduced_metric.allFinite() &&
      subspace.projected_gradient.size() == subspace.orthonormal_basis.cols() &&
      subspace.projected_gradient.allFinite();
}

bool truncated_newton_step_is_usable(
    const TruncatedNewtonStepResult& step,
    const Eigen::VectorXd& reduced_gradient) {
  return
      step.reduced_step.size() == reduced_gradient.size() &&
      step.reduced_step.allFinite() &&
      std::isfinite(step.reduced_step.stableNorm()) &&
      step.reduced_step.squaredNorm() > 0.0 &&
      step.reduced_hessian_times_step.size() == reduced_gradient.size() &&
      step.reduced_hessian_times_step.allFinite() &&
      std::isfinite(step.reduced_hessian_times_step.stableNorm()) &&
      step.reduced_metric_times_step.size() == reduced_gradient.size() &&
      step.reduced_metric_times_step.allFinite() &&
      std::isfinite(step.reduced_metric_times_step.stableNorm()) &&
      std::isfinite(step.trust_region_shift) &&
      std::isfinite(step.predicted_decrease) &&
      step.predicted_decrease > 0.0 &&
      reduced_gradient.dot(step.reduced_step) <= 0.0;
}

static TruncatedNewtonSubspace build_truncated_newton_subspace(
    const Eigen::VectorXd& reduced_gradient,
    const NonredundantRetractionMetric& metric,
    const std::vector<Eigen::VectorXd>& basis_vectors,
    const std::vector<Eigen::VectorXd>& tangent_basis_vectors,
    const std::vector<Eigen::VectorXd>& hessian_basis_vectors) {
  TruncatedNewtonSubspace subspace;
  if (basis_vectors.empty() ||
      basis_vectors.size() != tangent_basis_vectors.size() ||
      basis_vectors.size() != hessian_basis_vectors.size()) {
    return subspace;
  }

  const Eigen::Index reduced_size = reduced_gradient.size();
  const Eigen::Index basis_size =
      static_cast<Eigen::Index>(basis_vectors.size());
  subspace.orthonormal_basis.resize(reduced_size, basis_size);
  subspace.tangent_basis.resize(
      tangent_basis_vectors.front().size(),
      basis_size);
  subspace.hessian_basis.resize(reduced_size, basis_size);
  subspace.metric_basis.resize(reduced_size, basis_size);
  subspace.reduced_hessian.resize(basis_size, basis_size);
  subspace.reduced_metric.resize(basis_size, basis_size);
  subspace.projected_gradient.resize(basis_size);

  for (Eigen::Index column = 0; column < basis_size; ++column) {
    subspace.orthonormal_basis.col(column) =
        basis_vectors[column];
    subspace.tangent_basis.col(column) =
        tangent_basis_vectors[column];
    subspace.hessian_basis.col(column) =
        hessian_basis_vectors[column];
    subspace.metric_basis.col(column) = metric.apply(basis_vectors[column]);
    subspace.projected_gradient[column] =
        basis_vectors[column].dot(reduced_gradient);
  }

  for (Eigen::Index row = 0; row < basis_size; ++row) {
    for (Eigen::Index column = 0; column < basis_size; ++column) {
      subspace.reduced_hessian(row, column) =
          basis_vectors[row].dot(
              hessian_basis_vectors[column]);
      subspace.reduced_metric(row, column) =
          basis_vectors[row].dot(subspace.metric_basis.col(column));
    }
  }
  subspace.reduced_hessian =
      0.5 *
      (subspace.reduced_hessian +
       subspace.reduced_hessian.transpose()).eval();
  subspace.reduced_metric = 0.5 *
      (subspace.reduced_metric + subspace.reduced_metric.transpose()).eval();
  if (!truncated_newton_subspace_is_usable(
          subspace,
          reduced_size)) {
    return TruncatedNewtonSubspace();
  }
  return subspace;
}

static std::unique_ptr<BlockInverseBfgs> build_exact_positive_curvature_update(
    const std::vector<Eigen::VectorXd>& basis,
    const std::vector<Eigen::VectorXd>& hessian_basis) {
  if (basis.empty() || basis.size() != hessian_basis.size()) return nullptr;
  const Eigen::Index dimension = basis.front().size();
  const Eigen::Index rank = static_cast<Eigen::Index>(basis.size());
  Eigen::MatrixXd directions(dimension, rank);
  Eigen::MatrixXd images(dimension, rank);
  for (Eigen::Index column = 0; column < rank; ++column) {
    if (basis[column].size() != dimension ||
        hessian_basis[column].size() != dimension) {
      throw std::invalid_argument(
          "inconsistent exact-curvature basis dimensions");
    }
    directions.col(column) = basis[column];
    images.col(column) = hessian_basis[column];
  }
  const Eigen::MatrixXd projected = 0.5 *
      (directions.transpose() * images +
       images.transpose() * directions).eval();
  const std::vector<RitzSecant> positive = positive_ritz_secants(
      directions,
      images,
      projected,
      static_cast<int>(rank));
  if (positive.empty()) return nullptr;

  Eigen::MatrixXd positive_directions(dimension, positive.size());
  Eigen::MatrixXd positive_images(dimension, positive.size());
  for (std::size_t column = 0; column < positive.size(); ++column) {
    positive_directions.col(static_cast<Eigen::Index>(column)) =
        positive[column].direction;
    positive_images.col(static_cast<Eigen::Index>(column)) =
        positive[column].image;
  }
  return std::make_unique<BlockInverseBfgs>(
      std::move(positive_directions),
      std::move(positive_images));
}

TruncatedNewtonStepResult solve_affine_trust_region_in_subspace(
    const OrbitalChart::ProjectionResult& current_projection,
    double trust_radius,
    const NonredundantRetractionMetric& metric,
    const TruncatedNewtonSubspace& subspace,
    const Eigen::VectorXd& baseline_step,
    const Eigen::VectorXd& baseline_hessian_step,
    const Eigen::VectorXd& baseline_metric_step,
    double target_kkt_relative_residual) {
  TruncatedNewtonStepResult result;
  result.target_kkt_relative_residual = target_kkt_relative_residual;
  result.reduced_step =
      Eigen::VectorXd::Zero(current_projection.reduced_gradient.size());
  result.reduced_hessian_times_step =
      Eigen::VectorXd::Zero(current_projection.reduced_gradient.size());
  result.reduced_metric_times_step =
      Eigen::VectorXd::Zero(current_projection.reduced_gradient.size());
  if (!(trust_radius > 0.0) ||
      !std::isfinite(trust_radius) ||
      !truncated_newton_subspace_is_usable(
          subspace,
          current_projection.reduced_gradient.size()) ||
      baseline_step.size() != current_projection.reduced_gradient.size() ||
      baseline_hessian_step.size() != baseline_step.size() ||
      baseline_metric_step.size() != baseline_step.size() ||
      !baseline_step.allFinite() ||
      !baseline_hessian_step.allFinite() ||
      !baseline_metric_step.allFinite()) {
    return result;
  }

  const Eigen::MatrixXd reduced_hessian =
      0.5 *
      (subspace.reduced_hessian +
       subspace.reduced_hessian.transpose());
  const Eigen::MatrixXd reduced_metric = 0.5 *
      (subspace.reduced_metric + subspace.reduced_metric.transpose()).eval();
  Eigen::LLT<Eigen::MatrixXd> metric_factor(reduced_metric);
  if (metric_factor.info() != Eigen::Success) {
    throw std::runtime_error("projected orbital metric is not positive definite");
  }
  const double baseline_metric_norm_squared =
      baseline_step.dot(baseline_metric_step);
  if (!std::isfinite(baseline_metric_norm_squared) ||
      baseline_metric_norm_squared < 0.0) {
    return result;
  }
  const Eigen::VectorXd projected_metric_baseline =
      subspace.orthonormal_basis.transpose() * baseline_metric_step;
  const Eigen::VectorXd metric_center =
      metric_factor.solve(projected_metric_baseline);
  if (!projected_metric_baseline.allFinite() ||
      !metric_center.allFinite()) {
    return result;
  }
  // Complete the square in
  //   ||p_B + Q z||_G^2 = z^T M z + 2 c^T z + ||p_B||_G^2.
  // With u=z+M^{-1}c the affine trust region is a centered M-ball.
  const double affine_origin_norm_squared =
      baseline_metric_norm_squared -
      projected_metric_baseline.dot(metric_center);
  const double radius_roundoff =
      64.0 * std::numeric_limits<double>::epsilon() *
      std::max(1.0, trust_radius * trust_radius);
  const double effective_radius_squared =
      trust_radius * trust_radius - affine_origin_norm_squared;
  if (!std::isfinite(affine_origin_norm_squared) ||
      !std::isfinite(effective_radius_squared) ||
      effective_radius_squared <= radius_roundoff) {
    return result;
  }
  const double effective_radius =
      std::sqrt(std::max(0.0, effective_radius_squared));
  const Eigen::MatrixXd upper = metric_factor.matrixU();
  const Eigen::MatrixXd inverse_upper =
      upper.triangularView<Eigen::Upper>().solve(
          Eigen::MatrixXd::Identity(upper.rows(), upper.cols()));
  const Eigen::MatrixXd whitened_hessian = 0.5 *
      (inverse_upper.transpose() * reduced_hessian * inverse_upper +
       inverse_upper.transpose() * reduced_hessian.transpose() *
           inverse_upper).eval();
  const Eigen::VectorXd affine_projected_gradient =
      subspace.projected_gradient +
      subspace.orthonormal_basis.transpose() * baseline_hessian_step;
  const Eigen::VectorXd centered_projected_gradient =
      affine_projected_gradient - reduced_hessian * metric_center;
  const Eigen::VectorXd whitened_gradient =
      inverse_upper.transpose() * centered_projected_gradient;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(whitened_hessian);
  if (eigensolver.info() != Eigen::Success) {
    return result;
  }

  const Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();
  const Eigen::MatrixXd eigenvectors = eigensolver.eigenvectors();
  if (eigenvalues.size() == 0 ||
      !eigenvalues.allFinite() ||
      !eigenvectors.allFinite()) {
    return result;
  }
  const Eigen::VectorXd projected_gradient_in_eigenbasis =
      eigenvectors.transpose() * whitened_gradient;
  if (!projected_gradient_in_eigenbasis.allFinite()) {
    return result;
  }

  const auto spectral_solution = solve_spectral_trust_region(
      eigenvalues, projected_gradient_in_eigenbasis, effective_radius);
  const Eigen::VectorXd& eigen_coordinates = spectral_solution.step;
  const double trust_region_shift = spectral_solution.shift;
  const double minimum_eigenvalue = eigenvalues.minCoeff();
  const double spectral_scale = eigenvalues.cwiseAbs().maxCoeff();
  constexpr double kShiftToleranceFactor =
      64.0 * std::numeric_limits<double>::epsilon();

  const Eigen::VectorXd centered_coordinates =
      inverse_upper * eigenvectors * eigen_coordinates;
  Eigen::VectorXd subspace_coordinates =
      centered_coordinates - metric_center;
  Eigen::VectorXd reduced_step =
      baseline_step +
      subspace.orthonormal_basis * subspace_coordinates;
  double step_metric_norm = metric.norm(reduced_step);
  if (!(step_metric_norm > 0.0) || !std::isfinite(step_metric_norm)) {
    return result;
  }
  if (step_metric_norm >
      trust_radius * (1.0 + 1.0e-8)) {
    return result;
  }
  const Eigen::VectorXd reduced_hessian_times_step =
      baseline_hessian_step +
      subspace.hessian_basis * subspace_coordinates;
  const Eigen::VectorXd reduced_metric_times_step =
      baseline_metric_step +
      subspace.metric_basis * subspace_coordinates;
  if (reduced_hessian_times_step.size() !=
      current_projection.reduced_gradient.size() ||
      !reduced_hessian_times_step.allFinite() ||
      !reduced_metric_times_step.allFinite()) {
    return result;
  }

  const double predicted_decrease =
      -current_projection.reduced_gradient.dot(reduced_step) -
      0.5 * reduced_step.dot(reduced_hessian_times_step);
  if (!std::isfinite(predicted_decrease) ||
      predicted_decrease <= 0.0 ||
      current_projection.reduced_gradient.dot(reduced_step) > 0.0) {
    return result;
  }

  result.reduced_step = std::move(reduced_step);
  result.reduced_hessian_times_step = reduced_hessian_times_step;
  result.reduced_metric_times_step = reduced_metric_times_step;
  result.subspace = subspace;
  result.subspace_dimension =
      static_cast<int>(subspace.orthonormal_basis.cols());
  result.retract_tangent_norm = step_metric_norm;
  result.reached_boundary =
      step_metric_norm >= (1.0 - 1.0e-8) * trust_radius;
  result.encountered_negative_curvature =
      minimum_eigenvalue <= -kShiftToleranceFactor * spectral_scale;
  result.projected_model_gradient_norm =
      whitened_gradient.norm();
  result.model_spectral_radius = eigenvalues.cwiseAbs().maxCoeff();
  result.trust_region_shift = trust_region_shift;
  result.predicted_decrease = predicted_decrease;
  refresh_truncated_newton_step_certificate(
      current_projection.reduced_gradient, &result);
  return result;
}

TruncatedNewtonStepResult solve_trust_region_in_subspace(
    const OrbitalChart::ProjectionResult& current_projection,
    double trust_radius,
    const NonredundantRetractionMetric& metric,
    const TruncatedNewtonSubspace& subspace,
    double target_kkt_relative_residual) {
  const Eigen::Index reduced_size =
      current_projection.reduced_gradient.size();
  return solve_affine_trust_region_in_subspace(
      current_projection,
      trust_radius,
      metric,
      subspace,
      Eigen::VectorXd::Zero(reduced_size),
      Eigen::VectorXd::Zero(reduced_size),
      Eigen::VectorXd::Zero(reduced_size),
      target_kkt_relative_residual);
}

Eigen::VectorXd build_nonredundant_preconditioned_reduced_gradient_step(
    const NonredundantRetractionMetric& retraction_metric,
    const OrbitalChart& space,
    const OrbitalChart::ProjectionResult& projection,
    double trust_radius,
    const TransportedReducedLbfgsPreconditioner* transported_preconditioner) {
  const Eigen::VectorXd reduced_preconditioned_gradient =
      apply_nonredundant_truncated_newton_preconditioner(
          space,
          transported_preconditioner,
          projection.reduced_gradient);
  return retraction_metric.clip_to_radius(
      -reduced_preconditioned_gradient,
      trust_radius);
}

double update_nonredundant_truncated_newton_trust_radius(
    double trust_radius,
    double minimum_step_size,
    const std::optional<TruncatedNewtonTrialEvaluation>& observation,
    const TruncatedNewtonStepResult& model_step,
    TruncatedNewtonModelFidelity model_fidelity,
    bool accepted) {
  // A line-search predictor without an HVP supplies no quadratic-model
  // observation. Its Armijo decrease cannot determine the Newton radius.
  if (!observation.has_value()) return trust_radius;
  const auto& trial = *observation;
  // If B omits a finite Hessian contribution, the leading discrepancy is
  // 1/2 s^T(H-B)s = O(||s||^2).  A directionally exact Hessian instead leaves
  // the O(||s||^3) Taylor remainder.  Radius extrapolation must use the order
  // of the model that actually generated the trial.
  const auto scale_model_error_ratio = [model_fidelity](double ratio) {
    switch (model_fidelity) {
      case TruncatedNewtonModelFidelity::CoreApproximate:
        return std::sqrt(ratio);
      case TruncatedNewtonModelFidelity::DirectionallyExact:
        return std::cbrt(ratio);
    }
    throw std::invalid_argument("unknown truncated-Newton model fidelity");
  };
  const double step_norm = truncated_newton_step_effective_norm(model_step);
  const auto safe_radius_update = [&]() {
    if (accepted && step_norm > 0.0 && std::isfinite(step_norm)) {
      return std::max(minimum_step_size, step_norm);
    }
    return std::max(
        minimum_step_size,
        std::sqrt(minimum_step_size * trust_radius));
  };
  if (!(trust_radius > 0.0) || !std::isfinite(trust_radius) ||
      !(step_norm > 0.0) || !std::isfinite(step_norm) ||
      !(trial.predicted_decrease > 0.0) ||
      !std::isfinite(trial.predicted_decrease) ||
      !std::isfinite(trial.actual_decrease)) {
    return safe_radius_update();
  }

  if (!accepted) {
    // Fit the measured directional decrease by
    //   a(alpha) = alpha*l - alpha^2*(l-a),
    // where l=-g^T s and a is the full-step actual decrease.  Its positive
    // maximizer jumps directly to the scale supported by the observed energy,
    // avoiding a long sequence of nearly identical rejected retractions.
    // This is used only to contract the radius; invalid or noncontracting fits
    // fall through to the fidelity-aware model-error rule below.
    const double linear_decrease = trial.linear_decrease;
    const double fitted_curvature = linear_decrease - trial.actual_decrease;
    if (linear_decrease > 0.0 && fitted_curvature > 0.0 &&
        std::isfinite(linear_decrease) && std::isfinite(fitted_curvature)) {
      const double interpolated_scale =
          linear_decrease / (2.0 * fitted_curvature);
      if (interpolated_scale > 0.0 && interpolated_scale < 1.0 &&
          std::isfinite(interpolated_scale)) {
        return std::max(
            minimum_step_size,
            std::min(trust_radius, step_norm * interpolated_scale));
      }
    }

    // Estimate how much of the trial scale remains trustworthy from the
    // observed Taylor-model remainder.  This continuously contracts more for
    // worse disagreement instead of applying a fixed rejection multiplier.
    const double model_error =
        std::abs(trial.actual_decrease - trial.predicted_decrease);
    const double retained_model_fraction =
        trial.predicted_decrease /
        (trial.predicted_decrease + model_error);
    const double candidate_radius =
        step_norm * scale_model_error_ratio(retained_model_fraction);
    return std::max(
        minimum_step_size,
        std::min(trust_radius, candidate_radius));
  }

  // A model-inaccurate but energy-lowering step is useful: accepting it gives
  // the next point an exact gradient and hence a new transported secant.  Do
  // not repeatedly solve the deficient model at the old point.  Instead,
  // carry its measured error into the next point as a contracted validity
  // radius.  This separates monotone objective acceptance from model trust.
  const double model_error =
      std::abs(trial.actual_decrease - trial.predicted_decrease);
  const double roundoff =
      16.0 * std::numeric_limits<double>::epsilon() *
      std::max(trial.actual_decrease, trial.predicted_decrease);
  const double resolved_error = std::max(model_error, roundoff);
  if (resolved_error > trial.actual_decrease + roundoff) {
    const double candidate_radius =
        step_norm * scale_model_error_ratio(
                        trial.actual_decrease / resolved_error);
    if (!(candidate_radius > 0.0) || !std::isfinite(candidate_radius)) {
      return safe_radius_update();
    }
    return std::max(minimum_step_size, candidate_radius);
  }

  // An interior minimizer contains no evidence that the current radius is
  // restrictive once the measured model error is below the actual decrease.
  // Preserve it so local Newton convergence is not damped by harmless rho
  // fluctuations from the nonlinear retraction.
  if (!model_step.reached_boundary) {
    return std::max(minimum_step_size, trust_radius);
  }

  // Estimate the admissible next radius from the observed model remainder.
  // Keeping its extrapolated absolute error below the decrease just observed
  // gives Delta_next = ||s|| (actual/error)^(1/p), where p=2 for an
  // approximate Hessian and p=3 for a directionally exact Hessian. Unlike a
  // bound based on the largest Ritz value, this does not let an unrelated
  // stiff mode freeze a well-resolved boundary direction.
  const double radius_scale =
      scale_model_error_ratio(trial.actual_decrease / resolved_error);
  const double candidate_radius =
      step_norm * std::max(1.0, radius_scale);
  if (!(candidate_radius > 0.0) || !std::isfinite(candidate_radius)) {
    return safe_radius_update();
  }
  return std::max(minimum_step_size, candidate_radius);
}

double estimate_nonredundant_reduced_model_decrease(
    const OrbitalChart::ProjectionResult& projection,
    const Eigen::VectorXd& reduced_step,
    ReducedHvp* hvp) {
  const Eigen::VectorXd reduced_hessian_step =
      hvp->apply(reduced_step);
  return
      -projection.reduced_gradient.dot(reduced_step) -
      0.5 * reduced_step.dot(reduced_hessian_step);
}

TruncatedNewtonStepResult solve_nonredundant_truncated_newton_step(
    const NonredundantRetractionMetric& retraction_metric,
    const OrbitalChart& current_space,
    const OrbitalChart::ProjectionResult& current_projection,
    double trust_radius,
    double energy_tolerance,
    double gradient_tolerance,
    double target_kkt_relative_residual,
    ReducedHvp* hvp,
    const TransportedReducedLbfgsPreconditioner* transported_preconditioner,
    const Eigen::VectorXd* initial_reduced_step,
    const TruncatedNewtonSubspace* initial_subspace) {
  TruncatedNewtonStepResult result;
  result.target_kkt_relative_residual = target_kkt_relative_residual;
  if (!std::isfinite(target_kkt_relative_residual) ||
      target_kkt_relative_residual < 0.0 ||
      target_kkt_relative_residual >= 1.0) {
    throw std::invalid_argument("invalid Newton KKT residual target");
  }
  const Eigen::VectorXd preconditioned_gradient_step =
      build_nonredundant_preconditioned_reduced_gradient_step(
          retraction_metric,
          current_space,
          current_projection,
          trust_radius,
          transported_preconditioner);
  if (current_projection.reduced_gradient.size() == 0) {
    result.reduced_step = preconditioned_gradient_step;
    result.reduced_hessian_times_step.resize(0);
    result.stop_reason = TruncatedNewtonStopReason::PreconditionedGradient;
    return result;
  }

  const Eigen::VectorXd rhs = -current_projection.reduced_gradient;
  if (hvp == nullptr) {
    throw std::invalid_argument("Newton correction requires an HVP operator");
  }
  const int work_limit = static_cast<int>(rhs.size());
  TruncatedNewtonStopReason loop_stop_reason =
      TruncatedNewtonStopReason::CompleteSpace;
  // The inexact-Newton forcing term is an outer-iteration condition:
  // ||H s + g|| <= eta_k ||g||. A cached same-point trial changes the
  // initial residual but must not redefine the requested Newton accuracy.
  // The relative two-norm condition is the inexact-Newton contract.  An
  // absolute outer gradient threshold is not a valid substitute: once each
  // residual component falls below that threshold it can still be comparable
  // to the whole outer gradient, destroying the vanishing-forcing condition
  // and reducing the local method to a linearly convergent iteration.

  std::vector<Eigen::VectorXd> basis;
  std::vector<Eigen::VectorXd> tangent_basis;
  std::vector<Eigen::VectorXd> hessian_basis;
  basis.reserve(work_limit);
  tangent_basis.reserve(work_limit);
  hessian_basis.reserve(work_limit);

  // The strong block-L-BFGS predictor is the fixed origin of the affine
  // correction model. Exact curvature is used only to solve its Newton defect;
  // the projected trust solve must not re-optimize the predictor coefficient.
  Eigen::VectorXd baseline_step = preconditioned_gradient_step;
  if (initial_reduced_step != nullptr &&
      initial_reduced_step->size() == rhs.size() &&
      initial_reduced_step->allFinite() &&
      retraction_metric.norm(*initial_reduced_step) > 0.0) {
    baseline_step = retraction_metric.clip_to_radius(
        *initial_reduced_step, trust_radius);
  }
  const bool reuse_initial_subspace =
      initial_subspace != nullptr &&
      truncated_newton_subspace_is_usable(*initial_subspace, rhs.size()) &&
      initial_subspace->orthonormal_basis.cols() <= work_limit;
  Eigen::VectorXd baseline_hessian_step;
  if (reuse_initial_subspace &&
      initial_subspace->model_revision == hvp->model_revision()) {
    const Eigen::VectorXd coordinates =
        initial_subspace->orthonormal_basis.transpose() * baseline_step;
    const Eigen::VectorXd represented =
        initial_subspace->orthonormal_basis * coordinates;
    const double representation_tolerance =
        256.0 * std::numeric_limits<double>::epsilon() *
        std::max(1.0, baseline_step.norm());
    if ((represented - baseline_step).norm() <= representation_tolerance) {
      baseline_hessian_step = initial_subspace->hessian_basis * coordinates;
    }
  }
  if (baseline_hessian_step.size() == 0) {
    baseline_hessian_step = hvp->apply(baseline_step);
  }
  Eigen::VectorXd baseline_metric_step =
      retraction_metric.apply(baseline_step);
  if (!baseline_step.allFinite() ||
      !baseline_hessian_step.allFinite() ||
      !baseline_metric_step.allFinite()) {
    throw std::runtime_error("invalid affine Newton baseline images");
  }
  std::uint64_t image_revision = hvp->model_revision();
  const auto refresh_images = [&]() {
    Eigen::MatrixXd directions(rhs.size(), basis.size() + 1);
    Eigen::MatrixXd images(rhs.size(), basis.size() + 1);
    directions.col(0) = baseline_step;
    images.col(0) = baseline_hessian_step;
    for (std::size_t column = 0; column < basis.size(); ++column) {
      directions.col(static_cast<Eigen::Index>(column + 1)) = basis[column];
      images.col(static_cast<Eigen::Index>(column + 1)) =
          hessian_basis[column];
    }
    // New directions have already enriched the response space. Freeze that
    // model while refreshing the baseline and old correction images before
    // any Ritz/BFGS/KKT operation.
    const std::uint64_t revision = hvp->model_revision();
    if (!hvp->update_images(image_revision, directions, &images)) {
      images = hvp->apply_frozen_batch(directions);
    }
    if (images.rows() != directions.rows() ||
        images.cols() != directions.cols() || !images.allFinite() ||
        hvp->model_revision() != revision) {
      throw std::runtime_error("invalid refreshed Newton Hessian images");
    }
    baseline_hessian_step = images.col(0);
    for (std::size_t column = 0; column < basis.size(); ++column) {
      hessian_basis[column] =
          images.col(static_cast<Eigen::Index>(column + 1));
    }
    image_revision = hvp->model_revision();
  };

  result.reduced_step = baseline_step;
  result.reduced_hessian_times_step = baseline_hessian_step;
  result.reduced_metric_times_step = baseline_metric_step;
  result.retract_tangent_norm = retraction_metric.norm(baseline_step);
  result.reached_boundary =
      result.retract_tangent_norm >= (1.0 - 1.0e-8) * trust_radius;
  result.predicted_decrease =
      -current_projection.reduced_gradient.dot(baseline_step) -
      0.5 * baseline_step.dot(baseline_hessian_step);
  refresh_truncated_newton_step_certificate(
      current_projection.reduced_gradient, &result);
  if (truncated_newton_step_is_usable(
          result, current_projection.reduced_gradient)) {
    if (result.model_kkt_converged) {
      result.stop_reason = TruncatedNewtonStopReason::ModelKktConverged;
      return result;
    }
    if (truncated_newton_model_is_below_outer_accuracy(
            gradient_infinity_norm(current_projection.reduced_gradient),
            result.predicted_decrease,
            gradient_tolerance,
            energy_tolerance)) {
      result.stop_reason = TruncatedNewtonStopReason::BelowOuterAccuracy;
      return result;
    }
  }

  if (reuse_initial_subspace) {
    const Eigen::Index initial_dimension =
        initial_subspace->orthonormal_basis.cols();
    for (Eigen::Index column = 0; column < initial_dimension; ++column) {
      basis.push_back(initial_subspace->orthonormal_basis.col(column));
      tangent_basis.push_back(initial_subspace->tangent_basis.col(column));
      hessian_basis.push_back(initial_subspace->hessian_basis.col(column));
    }

    TruncatedNewtonSubspace current_subspace = *initial_subspace;
    image_revision = current_subspace.model_revision;
    if (image_revision != hvp->model_revision()) {
      refresh_images();
      current_subspace = build_truncated_newton_subspace(
          current_projection.reduced_gradient, retraction_metric,
          basis, tangent_basis, hessian_basis);
      current_subspace.model_revision = image_revision;
    }
    result = solve_affine_trust_region_in_subspace(
        current_projection,
        trust_radius,
        retraction_metric,
        current_subspace,
        baseline_step,
        baseline_hessian_step,
        baseline_metric_step,
        target_kkt_relative_residual);
    if (truncated_newton_step_is_usable(
            result,
            current_projection.reduced_gradient)) {
      if (result.model_kkt_converged) {
        result.stop_reason = TruncatedNewtonStopReason::ModelKktConverged;
        return result;
      }
      if (truncated_newton_model_is_below_outer_accuracy(
              gradient_infinity_norm(current_projection.reduced_gradient),
              result.predicted_decrease,
              gradient_tolerance,
              energy_tolerance)) {
        result.stop_reason = TruncatedNewtonStopReason::BelowOuterAccuracy;
        return result;
      }
    }
  }

  // Residual-driven block Davidson/GLTR correction. Negative curvature belongs
  // in the projected Hessian and must not terminate subspace construction. The
  // raw Newton defect and its positive preconditioned image expose complementary
  // missing-curvature directions in one fused block-HVP call.
  Eigen::VectorXd correction_rhs = -(
      current_projection.reduced_gradient +
      baseline_hessian_step);
  while (static_cast<int>(basis.size()) < work_limit) {
    const auto base_inverse =
        [&](const Eigen::VectorXd& vector) {
          return apply_nonredundant_truncated_newton_preconditioner(
              current_space,
              transported_preconditioner,
              vector);
        };
    std::vector<Eigen::VectorXd> sampled_directions;
    std::vector<Eigen::VectorXd> sampled_images;
    sampled_directions.reserve(basis.size() + 1);
    sampled_images.reserve(hessian_basis.size() + 1);
    sampled_directions.push_back(baseline_step);
    sampled_images.push_back(baseline_hessian_step);
    sampled_directions.insert(
        sampled_directions.end(), basis.begin(), basis.end());
    sampled_images.insert(
        sampled_images.end(), hessian_basis.begin(), hessian_basis.end());
    const std::unique_ptr<BlockInverseBfgs> curvature_update =
        build_exact_positive_curvature_update(
            sampled_directions, sampled_images);
    const Eigen::VectorXd preconditioned_correction =
        curvature_update != nullptr
            ? curvature_update->apply(correction_rhs, base_inverse)
            : base_inverse(correction_rhs);

    std::vector<Eigen::VectorXd> candidates;
    candidates.reserve(2);
    candidates.push_back(preconditioned_correction);
    candidates.push_back(correction_rhs);

    const int candidate_count = std::min(
        work_limit - static_cast<int>(basis.size()),
        static_cast<int>(candidates.size()));
    Eigen::MatrixXd candidate_block(rhs.size(), candidate_count);
    for (int column = 0; column < candidate_count; ++column) {
      candidate_block.col(column) = candidates[column];
    }

    const std::size_t previous_basis_size = basis.size();
    const int admitted = append_orthonormal_hvp_block(
        candidate_block,
        [&](const Eigen::Ref<const Eigen::MatrixXd>& directions) {
          return hvp->apply_batch(directions);
        },
        &basis,
        &hessian_basis);
    if (admitted == 0) {
      loop_stop_reason = TruncatedNewtonStopReason::DependentDirections;
      break;
    }
    if (image_revision != hvp->model_revision()) {
      refresh_images();
      // A failed solve of the revised projection must not return the previous
      // model's step, Hessian image, or convergence certificate.
      result = TruncatedNewtonStepResult();
      result.target_kkt_relative_residual = target_kkt_relative_residual;
    }
    for (std::size_t column = previous_basis_size;
         column < basis.size();
         ++column) {
      tangent_basis.push_back(retraction_metric.tangent(basis[column]));
    }

    TruncatedNewtonSubspace subspace =
        build_truncated_newton_subspace(
            current_projection.reduced_gradient,
            retraction_metric,
            basis,
            tangent_basis,
            hessian_basis);
    subspace.model_revision = image_revision;
    TruncatedNewtonStepResult candidate_step =
        solve_affine_trust_region_in_subspace(
            current_projection,
            trust_radius,
            retraction_metric,
            subspace,
            baseline_step,
            baseline_hessian_step,
            baseline_metric_step,
            target_kkt_relative_residual);
    if (!truncated_newton_step_is_usable(
            candidate_step,
            current_projection.reduced_gradient)) {
      loop_stop_reason = TruncatedNewtonStopReason::InvalidProjectedStep;
      break;
    }
    candidate_step.subspace_dimension = static_cast<int>(basis.size());
    result = std::move(candidate_step);

    const Eigen::VectorXd kkt_residual =
        current_projection.reduced_gradient +
        result.reduced_hessian_times_step +
        result.trust_region_shift * result.reduced_metric_times_step;
    if (result.model_kkt_converged) {
      result.stop_reason = TruncatedNewtonStopReason::ModelKktConverged;
      return result;
    }

    // Once the outer gradient condition already holds, resolving a model
    // decrease below the requested energy accuracy cannot change the outer
    // convergence decision.  The accepted trial is still evaluated exactly,
    // so an underestimated restricted-model decrease merely causes another
    // outer iteration; it cannot produce a false convergence declaration.
    if (truncated_newton_model_is_below_outer_accuracy(
            gradient_infinity_norm(current_projection.reduced_gradient),
            result.predicted_decrease,
            gradient_tolerance,
            energy_tolerance)) {
      result.stop_reason = TruncatedNewtonStopReason::BelowOuterAccuracy;
      return result;
    }

    // A boundary solution of the current projected model is not a full-space
    // trust-region convergence certificate. Continue expanding until the
    // shifted KKT residual meets the forcing condition or an explicit caller
    // resource guard is reached. Positive exact curvature enriches the
    // block-L-BFGS inverse used for the next defect direction; nonpositive
    // curvature remains explicit in the projected trust-region model.
    correction_rhs = -kkt_residual;
  }

  if (truncated_newton_step_is_usable(
          result,
          current_projection.reduced_gradient)) {
    result.stop_reason = loop_stop_reason;
    return result;
  }
  result.reduced_step = preconditioned_gradient_step;
  result.reduced_hessian_times_step.resize(0);
  result.predicted_decrease = 0.0;
  result.stop_reason = TruncatedNewtonStopReason::PreconditionedGradient;
  refresh_truncated_newton_step_certificate(
      current_projection.reduced_gradient, &result);
  return result;
}


}  // namespace xmvb::vb
