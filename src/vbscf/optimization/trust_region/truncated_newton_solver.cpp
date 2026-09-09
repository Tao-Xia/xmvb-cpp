#include "vbscf/optimization/trust_region/truncated_newton_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include <Eigen/Eigenvalues>

#include "vbscf/optimization/krylov/orthonormal_hvp_basis.hpp"
#include "vbscf/optimization/krylov/positive_conjugate_basis.hpp"
#include "vbscf/optimization/trust_region/spectral_trust_region.hpp"
#include "vbscf/optimization/vector_operations.hpp"

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

static double solve_trust_region_metric_boundary_tau(
    const Eigen::VectorXd& current_step_tangent,
    const Eigen::VectorXd& search_direction_tangent,
    double trust_radius) {
  if (current_step_tangent.size() != search_direction_tangent.size()) {
    return 0.0;
  }
  const double a = search_direction_tangent.squaredNorm();
  if (!(a > 0.0) || !std::isfinite(a) ||
      !(trust_radius > 0.0) || !std::isfinite(trust_radius)) {
    return 0.0;
  }
  // Solve ||J(s + tau p)||^2 = Delta^2 without materializing G = J^T J.
  const double b = current_step_tangent.dot(search_direction_tangent);
  const double c =
      current_step_tangent.squaredNorm() - trust_radius * trust_radius;
  const double discriminant = std::max(0.0, b * b - a * c);
  const double tau =
      (-b + std::sqrt(discriminant)) / a;
  if (!std::isfinite(tau)) {
    return 0.0;
  }
  return std::max(0.0, tau);
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
    const OrbitalPreparationInput& orbital_preparation_input,
    const OrbitalChart& current_space,
    const SparseParameterLayout& parameter_view,
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
      compute_nonredundant_retract_tangent_norm(
          orbital_preparation_input,
          current_space,
          parameter_view,
          step->reduced_step);
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
    const OrbitalPreparationInput& orbital_preparation_input,
    const OrbitalChart& current_space,
    const SparseParameterLayout& parameter_view,
    const TruncatedNewtonStepResult& model_step,
    Eigen::Index expected_size,
    double trust_radius) {
  cached_step =
      finite_nonzero_vector_matches_size(model_step.reduced_step, expected_size)
          ? shrink_nonredundant_reduced_step_inside_retract_tangent_radius(
                orbital_preparation_input,
                current_space,
                parameter_view,
                model_step.reduced_step,
                trust_radius)
          : Eigen::VectorXd();
}

bool truncated_newton_krylov_subspace_is_usable(
    const TruncatedNewtonKrylovSubspace& krylov_subspace,
    Eigen::Index reduced_size) {
  return
      reduced_size >= 0 &&
      krylov_subspace.orthonormal_basis.rows() == reduced_size &&
      krylov_subspace.orthonormal_basis.cols() > 0 &&
      krylov_subspace.orthonormal_basis.allFinite() &&
      krylov_subspace.tangent_basis.cols() ==
          krylov_subspace.orthonormal_basis.cols() &&
      krylov_subspace.tangent_basis.allFinite() &&
      krylov_subspace.hessian_basis.rows() == reduced_size &&
      krylov_subspace.hessian_basis.cols() ==
          krylov_subspace.orthonormal_basis.cols() &&
      krylov_subspace.hessian_basis.allFinite() &&
      krylov_subspace.reduced_hessian.rows() ==
          krylov_subspace.orthonormal_basis.cols() &&
      krylov_subspace.reduced_hessian.cols() ==
          krylov_subspace.orthonormal_basis.cols() &&
      krylov_subspace.reduced_hessian.allFinite() &&
      krylov_subspace.projected_gradient.size() ==
          krylov_subspace.orthonormal_basis.cols() &&
      krylov_subspace.projected_gradient.allFinite();
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

static TruncatedNewtonKrylovSubspace build_truncated_newton_krylov_subspace(
    const Eigen::VectorXd& reduced_gradient,
    const std::vector<Eigen::VectorXd>& basis_vectors,
    const std::vector<Eigen::VectorXd>& tangent_basis_vectors,
    const std::vector<Eigen::VectorXd>& hessian_basis_vectors) {
  TruncatedNewtonKrylovSubspace krylov_subspace;
  if (basis_vectors.empty() ||
      basis_vectors.size() != tangent_basis_vectors.size() ||
      basis_vectors.size() != hessian_basis_vectors.size()) {
    return krylov_subspace;
  }

  const Eigen::Index reduced_size = reduced_gradient.size();
  const Eigen::Index basis_size =
      static_cast<Eigen::Index>(basis_vectors.size());
  krylov_subspace.orthonormal_basis.resize(reduced_size, basis_size);
  krylov_subspace.tangent_basis.resize(
      tangent_basis_vectors.front().size(),
      basis_size);
  krylov_subspace.hessian_basis.resize(reduced_size, basis_size);
  krylov_subspace.reduced_hessian.resize(basis_size, basis_size);
  krylov_subspace.projected_gradient.resize(basis_size);

  for (Eigen::Index column = 0; column < basis_size; ++column) {
    krylov_subspace.orthonormal_basis.col(column) =
        basis_vectors[column];
    krylov_subspace.tangent_basis.col(column) =
        tangent_basis_vectors[column];
    krylov_subspace.hessian_basis.col(column) =
        hessian_basis_vectors[column];
    krylov_subspace.projected_gradient[column] =
        basis_vectors[column].dot(reduced_gradient);
  }

  for (Eigen::Index row = 0; row < basis_size; ++row) {
    for (Eigen::Index column = 0; column < basis_size; ++column) {
      krylov_subspace.reduced_hessian(row, column) =
          basis_vectors[row].dot(
              hessian_basis_vectors[column]);
    }
  }
  krylov_subspace.reduced_hessian =
      0.5 *
      (krylov_subspace.reduced_hessian +
       krylov_subspace.reduced_hessian.transpose()).eval();
  if (!truncated_newton_krylov_subspace_is_usable(
          krylov_subspace,
          reduced_size)) {
    return TruncatedNewtonKrylovSubspace();
  }
  return krylov_subspace;
}

TruncatedNewtonStepResult solve_trust_region_in_krylov_subspace(
    const OrbitalChart::ProjectionResult& current_projection,
    double trust_radius,
    const TruncatedNewtonKrylovSubspace& krylov_subspace) {
  TruncatedNewtonStepResult result;
  result.reduced_step =
      Eigen::VectorXd::Zero(current_projection.reduced_gradient.size());
  result.reduced_hessian_times_step =
      Eigen::VectorXd::Zero(current_projection.reduced_gradient.size());
  if (!(trust_radius > 0.0) ||
      !std::isfinite(trust_radius) ||
      !truncated_newton_krylov_subspace_is_usable(
          krylov_subspace,
          current_projection.reduced_gradient.size())) {
    return result;
  }

  const Eigen::MatrixXd reduced_hessian =
      0.5 *
      (krylov_subspace.reduced_hessian +
       krylov_subspace.reduced_hessian.transpose());
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
      eigenvectors.transpose() * krylov_subspace.projected_gradient;
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
      krylov_subspace.orthonormal_basis * subspace_coordinates;
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
      krylov_subspace.hessian_basis * subspace_coordinates;
  if (reduced_hessian_times_step.size() !=
          current_projection.reduced_gradient.size() ||
      !reduced_hessian_times_step.allFinite()) {
    return result;
  }

  const Eigen::VectorXd reduced_model_hessian_step =
      reduced_hessian * subspace_coordinates;
  const double predicted_decrease =
      -krylov_subspace.projected_gradient.dot(subspace_coordinates) -
      0.5 * subspace_coordinates.dot(reduced_model_hessian_step);
  if (!std::isfinite(predicted_decrease) ||
      predicted_decrease <= 0.0 ||
      current_projection.reduced_gradient.dot(reduced_step) > 0.0) {
    return result;
  }

  result.reduced_step = std::move(reduced_step);
  result.reduced_hessian_times_step = reduced_hessian_times_step;
  result.krylov_subspace = krylov_subspace;
  result.retract_tangent_norm = step_metric_norm;
  result.reached_boundary =
      step_metric_norm >= (1.0 - 1.0e-8) * trust_radius;
  result.encountered_negative_curvature =
      minimum_eigenvalue <= -kShiftToleranceFactor * spectral_scale;
  result.projected_model_gradient_norm =
      krylov_subspace.projected_gradient.norm();
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
  const auto geometric_fallback = [&]() {
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
    return geometric_fallback();
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

  if (model_step.used_krylov_rescue) {
    return std::max(minimum_step_size, step_norm);
  }

  // An interior minimizer contains no evidence that the current radius is
  // restrictive.  Preserve it; repeatedly rescaling an interior radius by
  // rho would turn otherwise valid Newton convergence into tiny first-order
  // steps whenever the nonlinear retraction makes rho slightly smaller than
  // one.
  if (!model_step.reached_boundary) {
    return std::max(minimum_step_size, trust_radius);
  }

  // The Ritz spectrum supplies a Cauchy-like length scale for the local
  // quadratic model.  On a boundary step, extrapolate only as far as both the
  // observed model agreement and that curvature length support.
  const double effective_curvature =
      model_step.model_spectral_radius + model_step.trust_region_shift;
  const double curvature_floor =
      std::numeric_limits<double>::epsilon() *
      model_step.model_spectral_radius;
  double spectral_radius = step_norm;
  if (model_step.projected_model_gradient_norm > 0.0 &&
      std::isfinite(model_step.projected_model_gradient_norm) &&
      effective_curvature > 0.0 &&
      std::isfinite(effective_curvature)) {
    spectral_radius =
        model_step.projected_model_gradient_norm /
        std::max(curvature_floor, effective_curvature);
  }
  const double trust_ratio = trial.actual_decrease / trial.predicted_decrease;
  const double overprediction_fraction = std::max(0.0, 1.0 - trust_ratio);
  const double agreement_denominator =
      std::max(
          std::sqrt(std::numeric_limits<double>::epsilon()),
          overprediction_fraction);
  const double agreement_radius = step_norm / agreement_denominator;
  const double curvature_radius = step_norm + spectral_radius;
  const double candidate_radius =
      std::max(
          step_norm,
          std::min(agreement_radius, curvature_radius));
  if (!(candidate_radius > 0.0) || !std::isfinite(candidate_radius)) {
    return geometric_fallback();
  }
  return std::max(minimum_step_size, candidate_radius);
}

double estimate_nonredundant_reduced_model_decrease(
    const OrbitalChart::ProjectionResult& projection,
    const Eigen::VectorXd& reduced_step,
    ReducedHvpOperator* hvp_operator) {
  const Eigen::VectorXd reduced_hessian_step =
      hvp_operator->apply(reduced_step);
  return
      -projection.reduced_gradient.dot(reduced_step) -
      0.5 * reduced_step.dot(reduced_hessian_step);
}

TruncatedNewtonStepResult solve_nonredundant_truncated_newton_step(
    const NonredundantRetractionMetric& retraction_metric,
    const OrbitalChart& current_space,
    const OrbitalChart::ProjectionResult& current_projection,
    double trust_radius,
    int max_cg_iterations,
    ReducedHvpOperator* hvp_operator,
    const TransportedReducedLbfgsPreconditioner* transported_preconditioner,
    const Eigen::VectorXd* initial_reduced_step) {
  TruncatedNewtonStepResult result;
  const Eigen::VectorXd fallback_step =
      build_nonredundant_preconditioned_reduced_gradient_step(
          retraction_metric,
          current_space,
          current_projection,
          trust_radius,
          transported_preconditioner);
  result.reduced_step =
      Eigen::VectorXd::Zero(current_projection.reduced_gradient.size());
  result.reduced_hessian_times_step =
      Eigen::VectorXd::Zero(current_projection.reduced_gradient.size());
  std::vector<Eigen::VectorXd> krylov_basis_vectors;
  std::vector<Eigen::VectorXd> krylov_tangent_basis_vectors;
  std::vector<Eigen::VectorXd> krylov_hessian_basis_vectors;
  krylov_basis_vectors.reserve(
      std::max(0, max_cg_iterations) + (initial_reduced_step != nullptr ? 1 : 0));
  krylov_tangent_basis_vectors.reserve(krylov_basis_vectors.capacity());
  krylov_hessian_basis_vectors.reserve(krylov_basis_vectors.capacity());
  auto append_and_apply = [&](const Eigen::VectorXd& direction,
                              Eigen::VectorXd* image) {
    // U_p^T U_p = I and the sparse retraction is additive. Thus reduced
    // Euclidean orthogonalization is the accepted-point retraction metric.
    // Recompute the packed tangent from the admitted q; never evolve it
    // independently through cancellation-prone projection recurrences.
    const bool admitted = append_orthonormal_hvp_direction(
        direction,
        [&](const Eigen::VectorXd& q) { return hvp_operator->apply(q); },
        &krylov_basis_vectors, &krylov_hessian_basis_vectors, image);
    if (admitted) {
      krylov_tangent_basis_vectors.push_back(
          retraction_metric.tangent(krylov_basis_vectors.back()));
    }
    return admitted;
  };
  auto finalize_result = [&]() -> TruncatedNewtonStepResult {
    result.krylov_subspace =
        build_truncated_newton_krylov_subspace(
            current_projection.reduced_gradient,
            krylov_basis_vectors,
            krylov_tangent_basis_vectors,
            krylov_hessian_basis_vectors);
    return result;
  };
  if (current_projection.reduced_gradient.size() == 0 ||
      max_cg_iterations <= 0) {
    result.reduced_step = fallback_step;
    result.reduced_hessian_times_step.resize(0);
    return finalize_result();
  }

  const Eigen::VectorXd rhs = -current_projection.reduced_gradient;
  Eigen::VectorXd residual = rhs;
  if (initial_reduced_step != nullptr) {
    const double initial_step_norm = retraction_metric.norm(*initial_reduced_step);
    if (std::isfinite(initial_step_norm) &&
        initial_step_norm > 0.0 &&
        initial_step_norm < trust_radius) {
      Eigen::VectorXd hessian_times_initial_step;
      if (append_and_apply(*initial_reduced_step, &hessian_times_initial_step)) {
        result.reduced_step = *initial_reduced_step;
        result.reduced_hessian_times_step = hessian_times_initial_step;
        result.predicted_decrease =
            rhs.dot(result.reduced_step) -
            0.5 * result.reduced_step.dot(hessian_times_initial_step);
        residual.noalias() -= hessian_times_initial_step;
      }
    }
  }

  Eigen::VectorXd preconditioned_residual =
      apply_nonredundant_truncated_newton_preconditioner(
          current_space,
          transported_preconditioner,
          residual);
  Eigen::VectorXd search_direction = preconditioned_residual;
  double residual_dot_search =
      residual.dot(preconditioned_residual);
  if (!std::isfinite(residual_dot_search) ||
      residual_dot_search <= 0.0) {
    if (result.reduced_step.squaredNorm() > 0.0 &&
        current_projection.reduced_gradient.dot(result.reduced_step) < 0.0 &&
        std::isfinite(result.predicted_decrease) &&
        result.predicted_decrease > 0.0) {
      return finalize_result();
    }
    result.reduced_step = fallback_step;
    result.reduced_hessian_times_step.resize(0);
    result.predicted_decrease = 0.0;
    return finalize_result();
  }

  const double initial_residual_norm = residual.stableNorm();
  const double outer_gradient_norm =
      current_projection.reduced_gradient.stableNorm();
  // The inexact-Newton forcing term is an outer-iteration condition:
  // ||H s + g|| <= eta_k ||g||. A cached same-point trial changes the
  // initial residual but must not redefine the requested Newton accuracy.
  const double residual_target =
      inexact_newton_forcing_term(outer_gradient_norm) * outer_gradient_norm;
  if (initial_residual_norm <= residual_target) {
    if (result.reduced_step.squaredNorm() > 0.0 &&
        current_projection.reduced_gradient.dot(result.reduced_step) < 0.0 &&
        std::isfinite(result.predicted_decrease) &&
        result.predicted_decrease > 0.0) {
      return finalize_result();
    }
    result.reduced_step = fallback_step;
    result.reduced_hessian_times_step.resize(0);
    result.predicted_decrease = 0.0;
    return finalize_result();
  }

  PositiveConjugateBasis conjugate_basis;
  constexpr double kCurvatureTolerance =
      64.0 * std::numeric_limits<double>::epsilon();
  for (int cg_iteration = 0;
       cg_iteration < max_cg_iterations;
       ++cg_iteration) {
    Eigen::VectorXd hessian_times_direction;
    if (!append_and_apply(search_direction, &hessian_times_direction)) {
      break;
    }
    const double curvature =
        search_direction.dot(hessian_times_direction);
    const Eigen::VectorXd current_step_tangent =
        retraction_metric.tangent(result.reduced_step);
    const Eigen::VectorXd search_direction_tangent =
        retraction_metric.tangent(search_direction);
    const double search_direction_metric_norm_squared =
        search_direction_tangent.squaredNorm();
    const double curvature_scale =
        search_direction.norm() * hessian_times_direction.norm();
    if (!std::isfinite(curvature) ||
        !std::isfinite(curvature_scale) ||
        curvature <= kCurvatureTolerance * curvature_scale) {
      result.encountered_negative_curvature = true;
      result.reached_boundary = true;
      if (search_direction_metric_norm_squared > 0.0 &&
          std::isfinite(search_direction_metric_norm_squared)) {
        const double tau = solve_trust_region_metric_boundary_tau(
            current_step_tangent,
            search_direction_tangent,
            trust_radius);
        result.predicted_decrease +=
            tau * residual.dot(search_direction) -
            0.5 * tau * tau * curvature;
        result.reduced_step.noalias() += tau * search_direction;
        result.reduced_hessian_times_step.noalias() +=
            tau * hessian_times_direction;
      }
      break;
    }

    const double alpha = residual_dot_search / curvature;
    if (!std::isfinite(alpha) || alpha <= 0.0) {
      result.reduced_step = fallback_step;
      result.reduced_hessian_times_step.resize(0);
      return finalize_result();
    }

    const Eigen::VectorXd candidate_step =
        result.reduced_step + alpha * search_direction;
    const double candidate_step_metric_norm =
        retraction_metric.norm(candidate_step);
    if (candidate_step_metric_norm >= trust_radius) {
      result.reached_boundary = true;
      const double tau = solve_trust_region_metric_boundary_tau(
          current_step_tangent,
          search_direction_tangent,
          trust_radius);
      result.predicted_decrease +=
          tau * residual.dot(search_direction) -
          0.5 * tau * tau * curvature;
      result.reduced_step.noalias() += tau * search_direction;
      result.reduced_hessian_times_step.noalias() +=
          tau * hessian_times_direction;
      break;
    }

    result.predicted_decrease +=
        alpha * residual.dot(search_direction) -
        0.5 * alpha * alpha * curvature;
    result.reduced_step = candidate_step;
    result.retract_tangent_norm = candidate_step_metric_norm;
    result.reduced_hessian_times_step.noalias() +=
        alpha * hessian_times_direction;
    residual = rhs - result.reduced_hessian_times_step;
    conjugate_basis.append(search_direction, hessian_times_direction);
    result.cg_iterations = cg_iteration + 1;
    if (residual.stableNorm() <= residual_target) {
      break;
    }

    preconditioned_residual =
        apply_nonredundant_truncated_newton_preconditioner(
            current_space,
            transported_preconditioner,
            residual);
    const double next_residual_dot_preconditioned =
        residual.dot(preconditioned_residual);
    if (!std::isfinite(next_residual_dot_preconditioned) ||
        next_residual_dot_preconditioned <= 0.0) {
      break;
    }
    search_direction =
        conjugate_basis.orthogonalize(preconditioned_residual);
    residual_dot_search = residual.dot(search_direction);

  }

  // U_p^T U_p = I makes this a Euclidean spherical subproblem. A CG line
  // truncated at the boundary need not minimize over the entire collected
  // subspace; solve that projected model explicitly, including negative modes.
  const TruncatedNewtonKrylovSubspace krylov_subspace =
      build_truncated_newton_krylov_subspace(
          current_projection.reduced_gradient,
          krylov_basis_vectors,
          krylov_tangent_basis_vectors,
          krylov_hessian_basis_vectors);
  auto metric_trust_region_step =
      solve_trust_region_in_krylov_subspace(
          current_projection,
          trust_radius,
          krylov_subspace);
  if (truncated_newton_step_is_usable(
          metric_trust_region_step,
          current_projection.reduced_gradient)) {
    metric_trust_region_step.encountered_negative_curvature =
        metric_trust_region_step.encountered_negative_curvature ||
        result.encountered_negative_curvature;
    metric_trust_region_step.cg_iterations = result.cg_iterations;
    return metric_trust_region_step;
  }

  const bool result_step_is_usable =
      std::isfinite(result.reduced_step.norm()) &&
      result.reduced_step.squaredNorm() > 0.0 &&
      current_projection.reduced_gradient.dot(result.reduced_step) < 0.0 &&
      std::isfinite(result.predicted_decrease) &&
      result.predicted_decrease > 0.0;
  if (!result_step_is_usable) {
    result.reduced_step = fallback_step;
    result.reduced_hessian_times_step.resize(0);
    result.predicted_decrease = 0.0;
  }
  return finalize_result();
}


}  // namespace xmvb::vb
