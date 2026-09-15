#include "vbscf/optimization/trust_region/truncated_newton.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include <Eigen/Eigenvalues>

#include "vbscf/optimization/krylov/orthonormal_basis.hpp"
#include "vbscf/optimization/trust_region/spectral.hpp"
#include "vbscf/optimization/driver/checks.hpp"

namespace xmvb::vb {

double inexact_newton_forcing_term(double gradient_norm) {
  if (!std::isfinite(gradient_norm) || gradient_norm <= 0.0) {
    return 0.5;
  }
  constexpr double kMinimumForcingTerm = 1.0e-3;
  constexpr double kMaximumForcingTerm = 0.5;
  return std::clamp(
      std::sqrt(gradient_norm),
      kMinimumForcingTerm,
      kMaximumForcingTerm);
}

bool inexact_newton_residual_is_converged(
    double gradient_l2_norm,
    double residual_l2_norm) {
  if (!std::isfinite(gradient_l2_norm) || gradient_l2_norm < 0.0 ||
      !std::isfinite(residual_l2_norm) || residual_l2_norm < 0.0) {
    return false;
  }
  return residual_l2_norm <=
      inexact_newton_forcing_term(gradient_l2_norm) * gradient_l2_norm;
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
    const TruncatedNewtonTrialEvaluation& trial) {
  if (!(trial.predicted_decrease > 0.0) ||
      !(trial.actual_decrease > 0.0) ||
      !std::isfinite(trial.predicted_decrease) ||
      !std::isfinite(trial.actual_decrease)) {
    return false;
  }
  const double model_error =
      std::abs(trial.actual_decrease - trial.predicted_decrease);
  const double roundoff =
      16.0 * std::numeric_limits<double>::epsilon() *
      std::max(trial.actual_decrease, trial.predicted_decrease);
  return model_error <= trial.actual_decrease + roundoff;
}

double truncated_newton_step_effective_norm(
    const TruncatedNewtonStepResult& step) {
  if (std::isfinite(step.retract_tangent_norm) &&
      step.retract_tangent_norm > 0.0) {
    return step.retract_tangent_norm;
  }
  const double reduced_norm = step.reduced_step.norm();
  return std::isfinite(reduced_norm) ? reduced_norm : 0.0;
}

void clamp_nonredundant_step_result_to_retract_tangent_radius(
    const OrbitalChart::ProjectionResult& current_projection,
    double trust_radius,
    TruncatedNewtonStepResult* step) {
  if (step == nullptr ||
      step->reduced_step.size() != current_projection.reduced_gradient.size() ||
      step->reduced_step.size() == 0 ||
      !step->reduced_step.allFinite()) {
    return;
  }

  const double tangent_norm =
      nonredundant_step_norm(step->reduced_step);
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
      step->predicted_decrease =
          -current_projection.reduced_gradient.dot(step->reduced_step) -
          0.5 * step->reduced_step.dot(step->reduced_hessian_times_step);
    } else {
      step->predicted_decrease = 0.0;
    }
    step->retract_tangent_norm = trust_radius;
    step->reached_boundary = true;
    return;
  }

  step->reached_boundary =
      step->reached_boundary ||
      tangent_norm >= (1.0 - 1.0e-8) * trust_radius;
}

bool RejectedTruncatedNewtonStepCache::has_cached_step(
    Eigen::Index expected_size) const {
  return finite_nonzero_vector_matches_size(cached_step, expected_size);
}

void RejectedTruncatedNewtonStepCache::clear() {
  cached_step.resize(0);
}

void RejectedTruncatedNewtonStepCache::update(
    const TruncatedNewtonStepResult& model_step,
    Eigen::Index expected_size,
    double trust_radius) {
  cached_step =
      finite_nonzero_vector_matches_size(model_step.reduced_step, expected_size)
          ? shrink_reduced_step_inside_trust_radius(
                model_step.reduced_step,
                trust_radius)
          : Eigen::VectorXd();
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
      subspace.reduced_hessian.rows() == subspace.orthonormal_basis.cols() &&
      subspace.reduced_hessian.cols() == subspace.orthonormal_basis.cols() &&
      subspace.reduced_hessian.allFinite() &&
      subspace.projected_gradient.size() == subspace.orthonormal_basis.cols() &&
      subspace.projected_gradient.allFinite();
}

bool truncated_newton_step_is_usable(
    const TruncatedNewtonStepResult& step,
    const Eigen::VectorXd& reduced_gradient) {
  return
      step.reduced_step.size() == reduced_gradient.size() &&
      step.reduced_step.allFinite() &&
      step.reduced_step.squaredNorm() > 0.0 &&
      std::isfinite(step.predicted_decrease) &&
      step.predicted_decrease > 0.0 &&
      reduced_gradient.dot(step.reduced_step) <= 0.0;
}

static TruncatedNewtonSubspace build_truncated_newton_subspace(
    const Eigen::VectorXd& reduced_gradient,
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
  subspace.reduced_hessian.resize(basis_size, basis_size);
  subspace.projected_gradient.resize(basis_size);

  for (Eigen::Index column = 0; column < basis_size; ++column) {
    subspace.orthonormal_basis.col(column) =
        basis_vectors[column];
    subspace.tangent_basis.col(column) =
        tangent_basis_vectors[column];
    subspace.hessian_basis.col(column) =
        hessian_basis_vectors[column];
    subspace.projected_gradient[column] =
        basis_vectors[column].dot(reduced_gradient);
  }

  for (Eigen::Index row = 0; row < basis_size; ++row) {
    for (Eigen::Index column = 0; column < basis_size; ++column) {
      subspace.reduced_hessian(row, column) =
          basis_vectors[row].dot(
              hessian_basis_vectors[column]);
    }
  }
  subspace.reduced_hessian =
      0.5 *
      (subspace.reduced_hessian +
       subspace.reduced_hessian.transpose()).eval();
  if (!truncated_newton_subspace_is_usable(
          subspace,
          reduced_size)) {
    return TruncatedNewtonSubspace();
  }
  return subspace;
}

TruncatedNewtonStepResult solve_trust_region_in_subspace(
    const OrbitalChart::ProjectionResult& current_projection,
    double trust_radius,
    const TruncatedNewtonSubspace& subspace) {
  TruncatedNewtonStepResult result;
  result.reduced_step =
      Eigen::VectorXd::Zero(current_projection.reduced_gradient.size());
  result.reduced_hessian_times_step =
      Eigen::VectorXd::Zero(current_projection.reduced_gradient.size());
  if (!(trust_radius > 0.0) ||
      !std::isfinite(trust_radius) ||
      !truncated_newton_subspace_is_usable(
          subspace,
          current_projection.reduced_gradient.size())) {
    return result;
  }

  const Eigen::MatrixXd reduced_hessian =
      0.5 *
      (subspace.reduced_hessian +
       subspace.reduced_hessian.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(reduced_hessian);
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
      eigenvectors.transpose() * subspace.projected_gradient;
  if (!projected_gradient_in_eigenbasis.allFinite()) {
    return result;
  }

  const auto spectral_solution = solve_spectral_trust_region(
      eigenvalues, projected_gradient_in_eigenbasis, trust_radius);
  const Eigen::VectorXd& eigen_coordinates = spectral_solution.step;
  const double trust_region_shift = spectral_solution.shift;
  const double minimum_eigenvalue = eigenvalues.minCoeff();
  const double spectral_scale = eigenvalues.cwiseAbs().maxCoeff();
  constexpr double kShiftToleranceFactor =
      64.0 * std::numeric_limits<double>::epsilon();

  Eigen::VectorXd subspace_coordinates =
      eigenvectors * eigen_coordinates;
  Eigen::VectorXd reduced_step =
      subspace.orthonormal_basis * subspace_coordinates;
  double step_metric_norm = subspace_coordinates.norm();
  if (!(step_metric_norm > 0.0) || !std::isfinite(step_metric_norm)) {
    return result;
  }
  if (step_metric_norm >
      trust_radius * (1.0 + 1.0e-8)) {
    const double scale = trust_radius / step_metric_norm;
    subspace_coordinates *= scale;
    reduced_step *= scale;
    step_metric_norm = subspace_coordinates.norm();
  }
  const Eigen::VectorXd reduced_hessian_times_step =
      subspace.hessian_basis * subspace_coordinates;
  if (reduced_hessian_times_step.size() !=
          current_projection.reduced_gradient.size() ||
      !reduced_hessian_times_step.allFinite()) {
    return result;
  }

  const Eigen::VectorXd reduced_model_hessian_step =
      reduced_hessian * subspace_coordinates;
  const double predicted_decrease =
      -subspace.projected_gradient.dot(subspace_coordinates) -
      0.5 * subspace_coordinates.dot(reduced_model_hessian_step);
  if (!std::isfinite(predicted_decrease) ||
      predicted_decrease <= 0.0 ||
      current_projection.reduced_gradient.dot(reduced_step) > 0.0) {
    return result;
  }

  result.reduced_step = std::move(reduced_step);
  result.reduced_hessian_times_step = reduced_hessian_times_step;
  result.subspace = subspace;
  result.subspace_dimension =
      static_cast<int>(subspace.orthonormal_basis.cols());
  result.retract_tangent_norm = step_metric_norm;
  result.reached_boundary =
      step_metric_norm >= (1.0 - 1.0e-8) * trust_radius;
  result.encountered_negative_curvature =
      minimum_eigenvalue <= -kShiftToleranceFactor * spectral_scale;
  result.projected_model_gradient_norm =
      subspace.projected_gradient.norm();
  result.model_spectral_radius = eigenvalues.cwiseAbs().maxCoeff();
  result.trust_region_shift = trust_region_shift;
  result.predicted_decrease = predicted_decrease;
  return result;
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
    const TruncatedNewtonTrialEvaluation& trial,
    const TruncatedNewtonStepResult& model_step,
    bool accepted) {
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
    // Estimate how much of the trial scale remains trustworthy from the
    // observed Taylor-model remainder.  This continuously contracts more for
    // worse disagreement instead of applying a fixed rejection multiplier.
    const double model_error =
        std::abs(trial.actual_decrease - trial.predicted_decrease);
    const double retained_model_fraction =
        trial.predicted_decrease /
        (trial.predicted_decrease + model_error);
    const double candidate_radius = step_norm * retained_model_fraction;
    return std::max(
        minimum_step_size,
        std::min(trust_radius, candidate_radius));
  }

  // An interior minimizer contains no evidence that the current radius is
  // restrictive.  Preserve it; repeatedly rescaling an interior radius by
  // rho would turn otherwise valid Newton convergence into tiny first-order
  // steps whenever the nonlinear retraction makes rho slightly smaller than
  // one.
  if (!model_step.reached_boundary) {
    return std::max(minimum_step_size, trust_radius);
  }

  // Estimate the admissible next radius from the observed quadratic-model
  // remainder. For a twice differentiable objective with locally Lipschitz
  // Hessian, the Taylor remainder is cubic in the step length. Keeping its
  // extrapolated absolute error below the decrease just observed gives
  // Delta_next = ||s|| (actual/error)^(1/3). Unlike a bound based on the
  // largest Ritz value, this does not let an unrelated stiff mode freeze a
  // well-resolved boundary direction.
  const double model_error =
      std::abs(trial.actual_decrease - trial.predicted_decrease);
  const double roundoff =
      16.0 * std::numeric_limits<double>::epsilon() *
      std::max(trial.actual_decrease, trial.predicted_decrease);
  const double resolved_error = std::max(model_error, roundoff);
  const double radius_scale =
      std::cbrt(trial.actual_decrease / resolved_error);
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
    int max_subspace_dimension,
    ReducedHvp* hvp,
    const TransportedReducedLbfgsPreconditioner* transported_preconditioner,
    const Eigen::VectorXd* initial_reduced_step,
    const TruncatedNewtonSubspace* initial_subspace) {
  TruncatedNewtonStepResult result;
  const Eigen::VectorXd preconditioned_gradient_step =
      build_nonredundant_preconditioned_reduced_gradient_step(
          retraction_metric,
          current_space,
          current_projection,
          trust_radius,
          transported_preconditioner);
  if (current_projection.reduced_gradient.size() == 0 ||
      max_subspace_dimension <= 0) {
    result.reduced_step = preconditioned_gradient_step;
    result.reduced_hessian_times_step.resize(0);
    return result;
  }

  const Eigen::VectorXd rhs = -current_projection.reduced_gradient;
  const double outer_gradient_norm =
      current_projection.reduced_gradient.stableNorm();
  // The inexact-Newton forcing term is an outer-iteration condition:
  // ||H s + g|| <= eta_k ||g||. A cached same-point trial changes the
  // initial residual but must not redefine the requested Newton accuracy.
  const auto residual_converged = [&](const Eigen::VectorXd& value) {
    return inexact_newton_residual_is_converged(
        outer_gradient_norm,
        value.stableNorm());
  };
  // The relative two-norm condition is the inexact-Newton contract.  An
  // absolute outer gradient threshold is not a valid substitute: once each
  // residual component falls below that threshold it can still be comparable
  // to the whole outer gradient, destroying the vanishing-forcing condition
  // and reducing the local method to a linearly convergent iteration.

  std::vector<Eigen::VectorXd> basis;
  std::vector<Eigen::VectorXd> tangent_basis;
  std::vector<Eigen::VectorXd> hessian_basis;
  basis.reserve(max_subspace_dimension);
  tangent_basis.reserve(max_subspace_dimension);
  hessian_basis.reserve(max_subspace_dimension);

  const bool reuse_initial_subspace =
      initial_subspace != nullptr &&
      truncated_newton_subspace_is_usable(*initial_subspace, rhs.size()) &&
      initial_subspace->orthonormal_basis.cols() <= max_subspace_dimension;
  if (reuse_initial_subspace) {
    const Eigen::Index initial_dimension =
        initial_subspace->orthonormal_basis.cols();
    for (Eigen::Index column = 0; column < initial_dimension; ++column) {
      basis.push_back(initial_subspace->orthonormal_basis.col(column));
      tangent_basis.push_back(initial_subspace->tangent_basis.col(column));
      hessian_basis.push_back(initial_subspace->hessian_basis.col(column));
    }

    result = solve_trust_region_in_subspace(
        current_projection,
        trust_radius,
        *initial_subspace);
    if (truncated_newton_step_is_usable(
            result,
            current_projection.reduced_gradient)) {
      const Eigen::VectorXd kkt_residual =
          current_projection.reduced_gradient +
          result.reduced_hessian_times_step +
          result.trust_region_shift * result.reduced_step;
      if (residual_converged(kkt_residual) ||
          truncated_newton_model_is_below_outer_accuracy(
              gradient_infinity_norm(current_projection.reduced_gradient),
              result.predicted_decrease,
              gradient_tolerance,
              energy_tolerance)) {
        return result;
      }
    }
  }

  // Residual-driven block Davidson/GLTR iteration. Negative curvature belongs
  // in the projected Hessian and must not terminate subspace construction.
  // The raw KKT residual and its positive preconditioned image expose two
  // complementary directions in one fused block-HVP call.
  Eigen::VectorXd correction_rhs = rhs;
  if (truncated_newton_step_is_usable(
          result,
          current_projection.reduced_gradient)) {
    correction_rhs = -(
        current_projection.reduced_gradient +
        result.reduced_hessian_times_step +
        result.trust_region_shift * result.reduced_step);
  }
  bool first_expansion = !reuse_initial_subspace;
  while (static_cast<int>(basis.size()) < max_subspace_dimension) {
    const Eigen::VectorXd preconditioned_correction =
        apply_nonredundant_truncated_newton_preconditioner(
            current_space,
            transported_preconditioner,
            correction_rhs);

    std::vector<Eigen::VectorXd> candidates;
    candidates.reserve(3);
    if (first_expansion && initial_reduced_step != nullptr &&
        initial_reduced_step->size() == rhs.size() &&
        initial_reduced_step->allFinite() &&
        retraction_metric.norm(*initial_reduced_step) > 0.0) {
      candidates.push_back(*initial_reduced_step);
    }
    candidates.push_back(preconditioned_correction);
    candidates.push_back(correction_rhs);

    const int candidate_count = std::min(
        max_subspace_dimension - static_cast<int>(basis.size()),
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
    if (admitted == 0) break;
    for (std::size_t column = previous_basis_size;
         column < basis.size();
         ++column) {
      tangent_basis.push_back(retraction_metric.tangent(basis[column]));
    }

    const TruncatedNewtonSubspace subspace =
        build_truncated_newton_subspace(
            current_projection.reduced_gradient,
            basis,
            tangent_basis,
            hessian_basis);
    TruncatedNewtonStepResult candidate_step =
        solve_trust_region_in_subspace(
            current_projection,
            trust_radius,
            subspace);
    if (!truncated_newton_step_is_usable(
            candidate_step,
            current_projection.reduced_gradient)) {
      break;
    }
    candidate_step.subspace_dimension = static_cast<int>(basis.size());
    result = std::move(candidate_step);

    const Eigen::VectorXd kkt_residual =
        current_projection.reduced_gradient +
        result.reduced_hessian_times_step +
        result.trust_region_shift * result.reduced_step;
    if (residual_converged(kkt_residual)) return result;

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
      return result;
    }

    // A boundary solution of the current projected model is not a full-space
    // trust-region convergence certificate. Continue expanding until the
    // shifted KKT residual meets the forcing condition or the subspace work
    // limit is reached.
    correction_rhs = -kkt_residual;
    first_expansion = false;
  }

  if (truncated_newton_step_is_usable(
          result,
          current_projection.reduced_gradient)) {
    return result;
  }
  result.reduced_step = preconditioned_gradient_step;
  result.reduced_hessian_times_step.resize(0);
  result.predicted_decrease = 0.0;
  return result;
}


}  // namespace xmvb::vb
