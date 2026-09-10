#include "vbscf/optimization/backends/truncated_newton.hpp"

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "vbscf/optimization/krylov/positive_ritz_secants.hpp"
#include "vbscf/optimization/globalization/line_search.hpp"
#include "vbscf/optimization/driver/checks.hpp"
#include "vbscf/optimization/driver/session.hpp"
#include "vbscf/optimization/driver/types.hpp"
#include "vbscf/optimization/preconditioners/transported_lbfgs.hpp"
#include "vbscf/optimization/objective/reduced_hvp.hpp"
#include "vbscf/optimization/trust_region/retraction.hpp"
#include "vbscf/optimization/trust_region/truncated_newton.hpp"
#include "vbscf/optimization/objective/function.hpp"
#include "vbscf/optimization/driver/options.hpp"
#include "vbscf/optimization/driver/result.hpp"
#include "vbscf/orbitals/charts/layout.hpp"

namespace xmvb::vb::optimizer_detail {

BackendRunResult run_truncated_newton_backend(
    VbScfObjective* objective,
    const SparseParameterLayout& parameter_view,
    const VbScfOptimizerOptions& options,
    const Eigen::VectorXd& initial_parameters,
    const Eigen::VectorXd& initial_gradient,
    double initial_energy,
    VbScfOptimizerResult* result) {
  BackendRunResult run_result;
  const int dimension = static_cast<int>(initial_parameters.size());
  double energy = initial_energy;
  double previous_energy = initial_energy;
  Eigen::VectorXd current_parameters = initial_parameters;
  Eigen::VectorXd current_gradient = initial_gradient;
  OrbitalChart current_space =
      build_orbital_chart(*objective, parameter_view);
  auto current_projection =
      current_space.project_gradient(current_gradient);
  double final_projected_gradient_inf_norm =
      gradient_infinity_norm(current_projection.reduced_gradient);
  double final_projected_gradient_l2_norm =
      current_projection.reduced_gradient.norm();
  std::vector<PackedSecantPair> packed_secant_history;
  packed_secant_history.reserve(
      std::max(
          0,
          options.nonredundant_truncated_newton_transport_history_size));
  double trust_radius =
      std::max(options.minimum_step_size, options.initial_step_size);
  int rejected_trial_step_count_for_current_point = 0;
  RejectedTruncatedNewtonStepCache rejected_step_cache;
  TruncatedNewtonKrylovSubspace cached_krylov_subspace;
  double initial_trust_radius_for_current_point = trust_radius;
  std::size_t initial_hvp_direction_count_for_current_point =
      result->matrix_free_hvp_direction_count;
  std::size_t initial_hvp_batch_count_for_current_point =
      result->matrix_free_hvp_batch_count;
  std::size_t initial_subproblem_count_for_current_point =
      result->matrix_free_subproblem_count;
  double initial_hvp_wall_time_for_current_point =
      result->matrix_free_hvp_wall_time_seconds;
  
  while (run_result.n_iterations < options.max_iterations) {
    if (current_space.reduced_size() == 0) {
      result->termination_reason = "nonredundant_space_empty";
      run_result.final_gradient_l2_norm = 0.0;
      break;
    }
  
    const double reduced_gradient_inf_norm =
        gradient_infinity_norm(current_projection.reduced_gradient);
    final_projected_gradient_inf_norm = reduced_gradient_inf_norm;
    final_projected_gradient_l2_norm =
        current_projection.reduced_gradient.norm();
    if (run_result.n_iterations == 0 &&
        reduced_gradient_inf_norm < options.gradient_tolerance) {
      result->converged = true;
      result->termination_reason =
          "nonredundant_truncated_newton_initial_tolerance";
      run_result.final_gradient_l2_norm = current_projection.reduced_gradient.norm();
      break;
    }
  
    if (!(trust_radius > 0.0) || !std::isfinite(trust_radius)) {
      result->termination_reason =
          "nonredundant_truncated_newton_invalid_trust_radius";
      run_result.final_gradient_l2_norm = current_projection.reduced_gradient.norm();
      break;
    }
  
    std::unique_ptr<ReducedHvpOperator> hvp_operator;
    switch (options.nonredundant_truncated_newton_hvp_mode) {
      case NonredundantTruncatedNewtonHvpMode::FullFiniteDifference:
        hvp_operator = std::make_unique<FullFiniteDifferenceReducedHvpOperator>(
            *objective,
            current_space,
            current_projection,
            objective->last_input().orbital_preparation_input,
            parameter_view,
            options.nonredundant_truncated_newton_hvp_step_size);
        break;
      case NonredundantTruncatedNewtonHvpMode::ExactContextDirectAction: {
        auto exact_ctx_hvp_operator =
            std::make_unique<ExactContextReducedHvpOperator>(
                *objective,
                current_space);
        if (!exact_ctx_hvp_operator->supports_analytic_core_model()) {
          throw std::runtime_error(
              build_exact_ctx_unavailable_message(*exact_ctx_hvp_operator));
        }
        hvp_operator = std::move(exact_ctx_hvp_operator);
        break;
      }
    }
    const int transport_history_size =
        choose_truncated_newton_transport_history_size(options);
    const auto transported_preconditioner =
        build_nonredundant_truncated_newton_preconditioner(
            current_space,
            packed_secant_history,
            transport_history_size);
    const int max_cg_iterations =
        choose_truncated_newton_max_cg_iterations(
            options,
            current_projection.reduced_gradient.size());
    const OrbitalPreparationInput current_orbital_input =
        objective->last_input().orbital_preparation_input;
    const NonredundantRetractionMetric retraction_metric(
        current_orbital_input,
        current_space,
        parameter_view);
    const auto admit_energy_only_trial_screen = [&]() {
      const auto& objective_time_history =
          objective->iteration_time_history_seconds();
      const double last_objective_seconds =
          objective_time_history.empty()
              ? 0.0
              : objective_time_history.back();
      // Measure the two available trial-evaluation paths directly.
      // After one sample, use energy-only screening exactly when its
      // observed cost is below a full objective-and-gradient call.
      if (objective->energy_only_call_count() == 0) {
        return true;
      }
      const double last_energy_only_seconds =
          objective->last_energy_only_wall_time_seconds();
      return
          std::isfinite(last_energy_only_seconds) &&
          last_energy_only_seconds > 0.0 &&
          last_energy_only_seconds < last_objective_seconds;
    };
    auto try_truncated_newton_trial_step =
        [&](const Eigen::VectorXd& candidate_reduced_step,
            double candidate_predicted_decrease,
            bool screen_with_energy_only,
            TruncatedNewtonTrialEvaluation* trial_evaluation,
            Eigen::VectorXd* accepted_packed_step,
            VbScfObjective::TrialEvaluation* accepted_trial_evaluation,
            Eigen::VectorXd* accepted_trial_parameters,
            Eigen::VectorXd* accepted_trial_gradient,
            double* accepted_trial_energy) -> bool {
          if (trial_evaluation != nullptr) {
            *trial_evaluation = TruncatedNewtonTrialEvaluation();
          }
          if (!std::isfinite(candidate_predicted_decrease) ||
              candidate_predicted_decrease <= 0.0) {
            return false;
          }
  
          Eigen::VectorXd candidate_trial_parameters(
              current_parameters.size());
          if (!try_build_nonredundant_lifted_trial_parameters(
                  current_orbital_input,
                  current_space,
                  parameter_view,
                  candidate_reduced_step,
                  &candidate_trial_parameters)) {
            return false;
          }
          const Eigen::VectorXd candidate_packed_step =
              candidate_trial_parameters - current_parameters;
          if (is_effectively_zero_step(
                  candidate_packed_step,
                  current_parameters)) {
            return false;
          }
  
          double effective_predicted_decrease =
              candidate_predicted_decrease;
          if (candidate_reduced_step.size() ==
                  current_projection.reduced_gradient.size()) {
            const double reduced_linear_decrease =
                -current_projection.reduced_gradient.dot(
                    candidate_reduced_step);
            const double packed_retraction_linear_decrease =
                -current_gradient.dot(candidate_packed_step);
            if (std::isfinite(reduced_linear_decrease) &&
                std::isfinite(packed_retraction_linear_decrease)) {
              const double curvature_decrease =
                  candidate_predicted_decrease -
                  reduced_linear_decrease;
              const double retraction_predicted_decrease =
                  packed_retraction_linear_decrease +
                  curvature_decrease;
              if (std::isfinite(retraction_predicted_decrease) &&
                  retraction_predicted_decrease > 0.0) {
                effective_predicted_decrease =
                    retraction_predicted_decrease;
              }
            }
          }
          if (!std::isfinite(effective_predicted_decrease) ||
              effective_predicted_decrease <= 0.0) {
            return false;
          }
          if (trial_evaluation != nullptr) {
            trial_evaluation->predicted_decrease =
                effective_predicted_decrease;
          }
  
          if (screen_with_energy_only) {
            const double candidate_trial_energy =
                objective->evaluate_energy_only(
                    candidate_trial_parameters);
            const double actual_decrease =
                energy - candidate_trial_energy;
            const double candidate_trust_ratio =
                actual_decrease / effective_predicted_decrease;
            if (trial_evaluation != nullptr) {
              trial_evaluation->actual_decrease = actual_decrease;
            }
            if (!std::isfinite(candidate_trial_energy) ||
                !std::isfinite(candidate_trust_ratio) ||
                actual_decrease <= 0.0) {
              return false;
            }
          }
          VbScfObjective::TrialEvaluation candidate_trial_evaluation =
              objective->evaluate_trial_without_committing(
                  candidate_trial_parameters);
          const double candidate_trial_energy =
              candidate_trial_evaluation.energy;
          const double actual_decrease =
              energy - candidate_trial_energy;
          const double candidate_trust_ratio =
              actual_decrease / effective_predicted_decrease;
          if (trial_evaluation != nullptr) {
            trial_evaluation->actual_decrease = actual_decrease;
          }
          if (!std::isfinite(candidate_trial_energy) ||
              !std::isfinite(candidate_trust_ratio) ||
              actual_decrease <= 0.0) {
            return false;
          }
  
          *accepted_packed_step = candidate_packed_step;
          *accepted_trial_evaluation =
              std::move(candidate_trial_evaluation);
          *accepted_trial_parameters = candidate_trial_parameters;
          *accepted_trial_gradient =
              std::move(accepted_trial_evaluation->gradient);
          *accepted_trial_energy = candidate_trial_energy;
          return true;
        };
    auto try_safeguarded_nonredundant_descent_step =
        [&](Eigen::VectorXd* accepted_packed_step,
            VbScfObjective* accepted_trial_objective,
            Eigen::VectorXd* accepted_trial_parameters,
            Eigen::VectorXd* accepted_trial_gradient,
            double* accepted_trial_energy) -> bool {
          Eigen::VectorXd descent_reduced_direction =
              -apply_nonredundant_truncated_newton_preconditioner(
                  current_space,
                  &transported_preconditioner,
                  current_projection.reduced_gradient);
          Eigen::VectorXd search_direction =
              gather_nonredundant_retract_tangent(
                  current_orbital_input,
                  current_space,
                  parameter_view,
                  descent_reduced_direction);
          double directional_derivative =
              current_gradient.dot(search_direction);
          if (!std::isfinite(directional_derivative) ||
              directional_derivative >= 0.0 ||
              is_effectively_zero_step(
                  search_direction,
                  current_parameters)) {
            descent_reduced_direction =
                -current_projection.reduced_gradient;
            search_direction =
                gather_nonredundant_retract_tangent(
                    current_orbital_input,
                    current_space,
                    parameter_view,
                    descent_reduced_direction);
            directional_derivative =
                current_gradient.dot(search_direction);
          }
          if (!std::isfinite(directional_derivative) ||
              directional_derivative >= 0.0 ||
              is_effectively_zero_step(
                  search_direction,
                  current_parameters)) {
            return false;
          }
  
          const double reduced_search_direction_norm =
              search_direction.norm();
          const double trust_radius_limited_initial_step =
              std::isfinite(reduced_search_direction_norm) &&
                      reduced_search_direction_norm > 0.0
                  ? trust_radius / reduced_search_direction_norm
                  : options.minimum_step_size;
          const double initial_descent_step =
              std::max(
                  options.minimum_step_size,
                  std::min(
                      std::min(
                          std::min(1.0, options.initial_step_size),
                          1.0 / std::max(1.0, reduced_gradient_inf_norm)),
                      trust_radius_limited_initial_step));
          // The safeguarded descent step is considered only after the current
          // accepted-point Newton model already failed to produce an
          // acceptable trust-region step. Starting the Armijo backtrack
          // from a reduced step that already fits inside the current
          // trust radius avoids burning many full objective evaluations
          // just to rediscover the same radius contraction.
          VbScfObjective descent_objective =
              objective->make_probe_copy();
          Eigen::VectorXd descent_parameters(current_parameters.size());
          Eigen::VectorXd descent_gradient(current_gradient.size());
          double descent_energy = energy;
          if (!try_armijo_backtracking_nonredundant_direction(
                  &descent_objective,
                  current_orbital_input,
                  current_space,
                  parameter_view,
                  current_parameters,
                  energy,
                  current_gradient,
                  descent_reduced_direction,
                  search_direction,
                  initial_descent_step,
                  options.minimum_step_size,
                  options.armijo_constant,
                  &descent_parameters,
                  &descent_gradient,
                  &descent_energy)) {
            return false;
          }
  
          *accepted_packed_step =
              descent_parameters - current_parameters;
          *accepted_trial_objective =
              std::move(descent_objective);
          *accepted_trial_parameters = std::move(descent_parameters);
          *accepted_trial_gradient = std::move(descent_gradient);
          *accepted_trial_energy = descent_energy;
          return true;
        };
    const Eigen::Index reduced_size =
        current_projection.reduced_gradient.size();
    TruncatedNewtonTrialEvaluation trial_evaluation_cache;
    const Eigen::VectorXd* initial_reduced_step_for_current_solve = nullptr;
    if (rejected_step_cache.has_cached_step(reduced_size)) {
      initial_reduced_step_for_current_solve =
          &rejected_step_cache.cached_step;
    }
  
    auto truncated_newton_step =
        solve_trust_region_in_krylov_subspace(
            current_projection,
            trust_radius,
            cached_krylov_subspace);
    const bool reused_krylov_subspace =
        truncated_newton_step_is_usable(
            truncated_newton_step,
            current_projection.reduced_gradient);
    if (!reused_krylov_subspace) {
      cached_krylov_subspace = TruncatedNewtonKrylovSubspace();
      truncated_newton_step =
          solve_nonredundant_truncated_newton_step(
              retraction_metric,
              current_space,
              current_projection,
              trust_radius,
              max_cg_iterations,
              hvp_operator.get(),
              &transported_preconditioner,
              initial_reduced_step_for_current_solve);
    }
    clamp_nonredundant_step_result_to_retract_tangent_radius(
        current_orbital_input,
        current_space,
        parameter_view,
        current_projection,
        trust_radius,
        &truncated_newton_step);
    if (!reused_krylov_subspace) {
      ++result->matrix_free_subproblem_count;
      if (truncated_newton_step.reduced_hessian_times_step.size() ==
          current_projection.reduced_gradient.size()) {
        const double gradient_norm = current_projection.reduced_gradient.stableNorm();
        const Eigen::VectorXd kkt_residual = current_projection.reduced_gradient +
            truncated_newton_step.reduced_hessian_times_step +
            truncated_newton_step.trust_region_shift * truncated_newton_step.reduced_step;
        if (kkt_residual.stableNorm() <=
            inexact_newton_forcing_term(gradient_norm) * gradient_norm) {
          ++result->matrix_free_residual_converged_count;
        }
      }
    }
    if (truncated_newton_krylov_subspace_is_usable(
            truncated_newton_step.krylov_subspace,
            current_projection.reduced_gradient.size())) {
      cached_krylov_subspace =
          truncated_newton_step.krylov_subspace;
    }
    const TruncatedNewtonStepResult model_step =
        truncated_newton_step;
    Eigen::VectorXd reduced_step = truncated_newton_step.reduced_step;
    if (reduced_step.size() != current_projection.reduced_gradient.size()) {
      result->termination_reason =
          "nonredundant_truncated_newton_invalid_step_dimension";
      run_result.final_gradient_l2_norm = current_projection.reduced_gradient.norm();
      break;
    }
    double predicted_decrease =
        truncated_newton_step.predicted_decrease;
    if (!std::isfinite(predicted_decrease) ||
        predicted_decrease <= 0.0) {
      reduced_step =
          build_nonredundant_preconditioned_reduced_gradient_step(
              retraction_metric,
              current_space,
              current_projection,
              trust_radius,
              &transported_preconditioner);
      predicted_decrease =
          estimate_nonredundant_reduced_model_decrease(
              current_projection,
              reduced_step,
              hvp_operator.get());
      truncated_newton_step.reduced_step = reduced_step;
      truncated_newton_step.reduced_hessian_times_step.resize(0);
      truncated_newton_step.retract_tangent_norm =
          compute_nonredundant_retract_tangent_norm(
              current_orbital_input,
              current_space,
              parameter_view,
              reduced_step);
      truncated_newton_step.reached_boundary =
          truncated_newton_step.retract_tangent_norm >=
          (1.0 - 1.0e-8) * trust_radius;
      truncated_newton_step.predicted_decrease = predicted_decrease;
    }
    if (const auto* exact_hvp_operator =
            dynamic_cast<const ExactContextReducedHvpOperator*>(
                hvp_operator.get())) {
      const auto hvp_diagnostics = exact_hvp_operator->diagnostics();
      result->matrix_free_hvp_direction_count +=
          hvp_diagnostics.apply_count;
      result->matrix_free_hvp_batch_count +=
          hvp_diagnostics.batch_apply_count;
      result->matrix_free_hvp_wall_time_seconds +=
          hvp_diagnostics.total_apply_wall_time_seconds;
    }
    TruncatedNewtonStepResult trial_step_for_current_trial =
        truncated_newton_step;
  
    Eigen::VectorXd packed_step(current_parameters.size());
    VbScfObjective::TrialEvaluation accepted_trial_evaluation;
    Eigen::VectorXd trial_parameters(current_parameters.size());
    Eigen::VectorXd trial_gradient(current_gradient.size());
    double trial_energy = energy;
    const bool screen_rejected_trials_with_energy_only =
        rejected_trial_step_count_for_current_point > 0 &&
        admit_energy_only_trial_screen();
    bool accepted_trial =
        try_truncated_newton_trial_step(
            reduced_step,
            predicted_decrease,
            screen_rejected_trials_with_energy_only,
            &trial_evaluation_cache,
            &packed_step,
            &accepted_trial_evaluation,
            &trial_parameters,
            &trial_gradient,
            &trial_energy);
    if (!accepted_trial &&
        rejected_trial_step_count_for_current_point == 0 &&
        reduced_gradient_inf_norm >=
            8.0 * options.gradient_tolerance &&
        model_step.encountered_negative_curvature) {
      if (try_safeguarded_nonredundant_descent_step(
              &packed_step,
              objective,
              &trial_parameters,
              &trial_gradient,
              &trial_energy)) {
        accepted_trial = true;
        const double safeguarded_actual_decrease = energy - trial_energy;
        trial_evaluation_cache.actual_decrease =
            safeguarded_actual_decrease;
        trial_evaluation_cache.predicted_decrease =
            safeguarded_actual_decrease;
        truncated_newton_step.used_krylov_rescue = true;
        truncated_newton_step.reached_boundary = false;
        truncated_newton_step.encountered_negative_curvature = false;
        truncated_newton_step.cg_iterations = 0;
        truncated_newton_step.reduced_step =
            current_space.project_vector(packed_step).reduced_gradient;
        truncated_newton_step.retract_tangent_norm = packed_step.norm();
        truncated_newton_step.predicted_decrease =
            std::max(options.energy_tolerance, energy - trial_energy);
      }
    }
    if (!accepted_trial) {
      ++rejected_trial_step_count_for_current_point;
      trust_radius =
          update_nonredundant_truncated_newton_trust_radius(
              trust_radius,
              options.minimum_step_size,
              trial_evaluation_cache,
              trial_step_for_current_trial,
              false);
      if (trust_radius <= options.minimum_step_size) {
        result->termination_reason =
            "nonredundant_truncated_newton_trust_radius_exhausted";
        run_result.final_gradient_l2_norm = current_projection.reduced_gradient.norm();
        break;
      }
      rejected_step_cache.update(
          current_orbital_input,
          current_space,
          parameter_view,
          model_step,
          reduced_size,
          trust_radius);
      continue;
    }
    if (accepted_trial_evaluation.valid) {
      objective->commit_trial_evaluation(
          std::move(accepted_trial_evaluation));
    }
    current_parameters = trial_parameters;
    current_gradient = std::move(trial_gradient);
    energy = trial_energy;
    // Cache evaluated curvature as approximate preconditioning data,
    // before canonicalization transports packed steps and covectors.
    // It is never reused as an exact Hessian action at the next point.
    if (transport_history_size > 1 &&
        truncated_newton_krylov_subspace_is_usable(
            cached_krylov_subspace, current_space.reduced_size())) {
      const auto pairs = positive_ritz_secants(
          cached_krylov_subspace.orthonormal_basis,
          cached_krylov_subspace.hessian_basis,
          cached_krylov_subspace.reduced_hessian,
          transport_history_size - 1);
      for (const auto& pair : pairs) {
        append_nonredundant_truncated_newton_secant_pair(
            current_space.expand_step(pair.direction),
            current_space.expand_step(pair.image),
            transport_history_size, &packed_secant_history);
      }
    }
    const bool accepted_point_chart_reset =
        objective->canonicalize_orbital_chart_at_current_point(
            &current_parameters,
            &current_gradient,
            &packed_secant_history);
    ++run_result.n_iterations;
    rejected_step_cache.clear();
    cached_krylov_subspace = TruncatedNewtonKrylovSubspace();
    sync_result_from_objective(*objective, result);
    run_result.final_gradient_l2_norm = current_gradient.norm();
    const double de = energy - previous_energy;
    previous_energy = energy;
    OrbitalChart next_space =
        build_orbital_chart(*objective, parameter_view);
    auto next_projection =
        next_space.project_gradient(current_gradient);
    const bool nonredundant_rank_changed =
        current_space.reduced_size() != next_space.reduced_size() ||
        current_space.rank_signature() != next_space.rank_signature();
    final_projected_gradient_inf_norm =
        gradient_infinity_norm(next_projection.reduced_gradient);
    final_projected_gradient_l2_norm =
        next_projection.reduced_gradient.norm();
    const double next_trust_radius =
        update_nonredundant_truncated_newton_trust_radius(
            trust_radius,
            options.minimum_step_size,
            trial_evaluation_cache,
            truncated_newton_step,
            true);
    TnhvpIterationRecord iteration_record;
    iteration_record.accepted_iteration_index = run_result.n_iterations;
    iteration_record.reduced_dimension = static_cast<int>(reduced_size);
    iteration_record.krylov_iterations = truncated_newton_step.cg_iterations;
    iteration_record.rejected_trial_count =
        rejected_trial_step_count_for_current_point;
    iteration_record.hvp_direction_count =
        result->matrix_free_hvp_direction_count -
        initial_hvp_direction_count_for_current_point;
    iteration_record.hvp_batch_count =
        result->matrix_free_hvp_batch_count -
        initial_hvp_batch_count_for_current_point;
    iteration_record.subproblem_count =
        result->matrix_free_subproblem_count -
        initial_subproblem_count_for_current_point;
    iteration_record.hvp_wall_time_seconds =
        result->matrix_free_hvp_wall_time_seconds -
        initial_hvp_wall_time_for_current_point;
    iteration_record.source_gradient_l2_norm =
        current_projection.reduced_gradient.stableNorm();
    iteration_record.accepted_gradient_l2_norm =
        next_projection.reduced_gradient.stableNorm();
    iteration_record.forcing_term =
        inexact_newton_forcing_term(iteration_record.source_gradient_l2_norm);
    if (truncated_newton_step.reduced_hessian_times_step.size() ==
            current_projection.reduced_gradient.size() &&
        truncated_newton_step.reduced_hessian_times_step.allFinite()) {
      const Eigen::VectorXd kkt_residual =
          current_projection.reduced_gradient +
          truncated_newton_step.reduced_hessian_times_step +
          truncated_newton_step.trust_region_shift *
              truncated_newton_step.reduced_step;
      const double residual_norm = kkt_residual.stableNorm();
      if (std::isfinite(residual_norm)) {
        iteration_record.has_kkt_residual = true;
        iteration_record.kkt_relative_residual =
            residual_norm /
            std::max(
                iteration_record.source_gradient_l2_norm,
                std::numeric_limits<double>::min());
      }
    }
    iteration_record.initial_trust_radius =
        initial_trust_radius_for_current_point;
    iteration_record.accepted_trial_radius = trust_radius;
    iteration_record.next_trust_radius = next_trust_radius;
    iteration_record.step_norm = truncated_newton_step.retract_tangent_norm;
    iteration_record.predicted_decrease =
        trial_evaluation_cache.predicted_decrease;
    iteration_record.actual_decrease = trial_evaluation_cache.actual_decrease;
    if (iteration_record.predicted_decrease > 0.0 &&
        std::isfinite(iteration_record.predicted_decrease)) {
      iteration_record.trust_ratio =
          iteration_record.actual_decrease /
          iteration_record.predicted_decrease;
    }
    iteration_record.model_spectral_radius =
        truncated_newton_step.model_spectral_radius;
    iteration_record.trust_region_shift =
        truncated_newton_step.trust_region_shift;
    iteration_record.reached_boundary = truncated_newton_step.reached_boundary;
    iteration_record.encountered_negative_curvature =
        model_step.encountered_negative_curvature;
    iteration_record.used_krylov_rescue =
        truncated_newton_step.used_krylov_rescue;
    iteration_record.reused_krylov_subspace = reused_krylov_subspace;
    result->tnhvp_iteration_trace.push_back(iteration_record);
    record_accepted_iteration_snapshot(
        objective,
        run_result.n_iterations,
        options,
        result,
        &iteration_record);
    trust_radius = next_trust_radius;
    rejected_trial_step_count_for_current_point = 0;
    initial_trust_radius_for_current_point = trust_radius;
    initial_hvp_direction_count_for_current_point =
        result->matrix_free_hvp_direction_count;
    initial_hvp_batch_count_for_current_point =
        result->matrix_free_hvp_batch_count;
    initial_subproblem_count_for_current_point =
        result->matrix_free_subproblem_count;
    initial_hvp_wall_time_for_current_point =
        result->matrix_free_hvp_wall_time_seconds;
    if (nonredundant_rank_changed) {
      packed_secant_history.clear();
    }
    if (!accepted_point_chart_reset && !nonredundant_rank_changed) {
      const Eigen::VectorXd packed_projected_gradient_change =
          next_projection.packed_projected_gradient -
          current_projection.packed_projected_gradient;
      append_nonredundant_truncated_newton_secant_pair(
          packed_step,
          packed_projected_gradient_change,
          transport_history_size,
          &packed_secant_history);
    }
    if (std::abs(de) < options.energy_tolerance &&
        gradient_infinity_norm(next_projection.reduced_gradient) <
            options.gradient_tolerance) {
      result->converged = true;
      result->termination_reason =
          "nonredundant_truncated_newton_dual_tolerance";
      run_result.final_gradient_l2_norm = next_projection.reduced_gradient.norm();
      break;
    }
  
    current_space = std::move(next_space);
    current_projection = std::move(next_projection);
  }
  result->final_projected_gradient_inf_norm =
      final_projected_gradient_inf_norm;
  result->final_projected_gradient_l2_norm =
      final_projected_gradient_l2_norm;
  run_result.final_projected_gradient_ready = true;
  return run_result;
}

}  // namespace xmvb::vb::optimizer_detail
