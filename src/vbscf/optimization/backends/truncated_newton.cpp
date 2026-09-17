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
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "vbscf/optimization/globalization/line_search.hpp"
#include "vbscf/optimization/driver/checks.hpp"
#include "vbscf/optimization/driver/session.hpp"
#include "vbscf/optimization/driver/types.hpp"
#include "vbscf/optimization/preconditioners/transported_lbfgs.hpp"
#include "vbscf/optimization/objective/reduced_hvp.hpp"
#include "vbscf/optimization/objective/secant_hvp.hpp"
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
  TruncatedNewtonSubspace cached_subspace;
  bool request_outer_response = false;
  bool request_secant_correction = false;
  bool outer_response_used_for_current_point = false;
  double last_outer_response_seconds = 0.0;
  double initial_trust_radius_for_current_point = trust_radius;
  std::size_t initial_hvp_direction_count_for_current_point =
      result->matrix_free_hvp_direction_count;
  std::size_t initial_hvp_batch_count_for_current_point =
      result->matrix_free_hvp_batch_count;
  std::size_t initial_subproblem_count_for_current_point =
      result->matrix_free_subproblem_count;
  double initial_hvp_wall_time_for_current_point =
      result->matrix_free_hvp_wall_time_seconds;
  std::size_t initial_core_hvp_direction_count_for_current_point =
      result->matrix_free_core_hvp_direction_count;
  std::size_t initial_outer_response_direction_count_for_current_point =
      result->matrix_free_outer_response_direction_count;
  double initial_outer_response_wall_time_for_current_point =
      result->matrix_free_outer_response_wall_time_seconds;
  auto accepted_point_start_time = std::chrono::steady_clock::now();
  
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
  
    ExactReducedHvp exact_hvp(*objective, current_space);
    if (!exact_hvp.supports_analytic_core_model()) {
      throw std::runtime_error(build_hvp_error(exact_hvp));
    }
    const auto response_scale_info = exact_hvp.diagnostics();
    const bool response_scale_is_affordable =
        outer_response_scale_is_affordable(response_scale_info);
    const int transport_history_size =
        choose_truncated_newton_transport_history_size(options);
    SecantCorrectedCoreHvp core_hvp(
        &exact_hvp,
        current_space,
        packed_secant_history,
        !response_scale_is_affordable && request_secant_correction
            ? transport_history_size
            : 0);
    const auto transported_preconditioner =
        build_nonredundant_truncated_newton_preconditioner(
            current_space,
            packed_secant_history,
            transport_history_size);
    const int max_subspace_dimension =
        choose_tnhvp_max_subspace_dimension(
            options,
            current_projection.reduced_gradient.size());
    const OrbitalPreparationInput current_orbital_input =
        objective->input().orbital_preparation_input;
    const NonredundantRetractionMetric retraction_metric(
        current_space, parameter_view, current_orbital_input);
    auto try_truncated_newton_trial_step =
        [&](const Eigen::VectorXd& candidate_reduced_step,
            double candidate_predicted_decrease,
            TruncatedNewtonTrialEvaluation* trial_evaluation,
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
  
          Eigen::VectorXd candidate_trial_parameters =
              build_nonredundant_lifted_trial_parameters(
                  current_orbital_input,
                  current_space,
                  parameter_view,
                  candidate_reduced_step);
          const Eigen::VectorXd candidate_packed_step =
              candidate_trial_parameters - current_parameters;
          if (is_effectively_zero_step(
                  candidate_packed_step,
                  current_parameters)) {
            return false;
          }
  
          const double effective_predicted_decrease =
              candidate_predicted_decrease;
          if (!std::isfinite(effective_predicted_decrease) ||
              effective_predicted_decrease <= 0.0) {
            return false;
          }
          if (trial_evaluation != nullptr) {
            trial_evaluation->predicted_decrease =
                effective_predicted_decrease;
          }
  
          VbScfObjective::TrialEvaluation candidate_trial_evaluation =
              objective->evaluate_trial(
                  candidate_trial_parameters,
                  true);
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
              !truncated_newton_trial_is_acceptable(
                  TruncatedNewtonTrialEvaluation{
                      actual_decrease,
                      effective_predicted_decrease})) {
            return false;
          }
  
          *accepted_trial_parameters = parameter_view.pack(
              candidate_trial_evaluation.orbital_preparation_input);
          *accepted_trial_evaluation =
              std::move(candidate_trial_evaluation);
          *accepted_trial_gradient =
              std::move(accepted_trial_evaluation->gradient);
          *accepted_trial_energy = candidate_trial_energy;
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
  
    bool reused_subspace =
        truncated_newton_subspace_is_usable(
            cached_subspace,
            current_projection.reduced_gradient.size());
    bool reused_subspace_for_trial = reused_subspace;
    bool candidate_has_exact_outer_response = false;
    bool response_probe_performed = false;
    double response_probe_relative_residual = 0.0;
    bool response_deferred_for_cost = false;
    ReducedHvp* active_hvp = &core_hvp;
    auto solve_subproblem =
        [&](ReducedHvp* operator_hvp,
            bool reuse,
            const Eigen::VectorXd* initial_step) {
          auto step = solve_nonredundant_truncated_newton_step(
              retraction_metric,
              current_space,
              current_projection,
              trust_radius,
              options.energy_tolerance,
              options.gradient_tolerance,
              inexact_newton_forcing_term(
                  current_projection.reduced_gradient.stableNorm()),
              max_subspace_dimension,
              operator_hvp,
              &transported_preconditioner,
              initial_step,
              reuse ? &cached_subspace : nullptr);
          clamp_nonredundant_step_result_to_retract_tangent_radius(
              current_projection,
              trust_radius,
              retraction_metric,
              &step);
          if (!reuse) {
            ++result->matrix_free_subproblem_count;
            if (!step.reached_boundary) {
              ++result->matrix_free_interior_subproblem_count;
            }
          }
          if (truncated_newton_subspace_is_usable(
                  step.subspace,
                  current_projection.reduced_gradient.size())) {
            cached_subspace = step.subspace;
          }
          return step;
        };

    TruncatedNewtonStepResult truncated_newton_step = solve_subproblem(
        active_hvp,
        reused_subspace,
        initial_reduced_step_for_current_solve);

    // The core model is sufficient while a successful boundary step is still
    // globalizing the orbitals. An interior step, detected negative curvature,
    // or a retry after deficient progress is certified with an exact outer
    // action. The exact image corrects the candidate's model decrease and KKT
    // certificate, but a single sampled direction must not authorize full
    // outer responses throughout the Krylov subspace. Thus outer work is
    // bounded by candidate samples rather than multiplied by subspace size.
    const double core_candidate_wall_time_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - accepted_point_start_time)
            .count();
    const auto response_cost_info = exact_hvp.diagnostics();
    const bool known_response_cost_is_affordable =
        response_scale_is_affordable &&
        (!(last_outer_response_seconds > 0.0) ||
         last_outer_response_seconds <= core_candidate_wall_time_seconds);
    const bool certify_core_candidate =
        request_outer_response &&
        !outer_response_used_for_current_point &&
        known_response_cost_is_affordable &&
        truncated_newton_step.reduced_step.size() == reduced_size &&
        truncated_newton_step.reduced_hessian_times_step.size() ==
            reduced_size &&
        truncated_newton_step.reduced_metric_times_step.size() ==
            reduced_size &&
        truncated_newton_step.reduced_step.allFinite() &&
        truncated_newton_step.reduced_hessian_times_step.allFinite() &&
        truncated_newton_step.reduced_metric_times_step.allFinite();
    if (request_outer_response &&
        !outer_response_used_for_current_point &&
        !known_response_cost_is_affordable) {
      response_deferred_for_cost = true;
    }
    if (certify_core_candidate) {
      response_probe_performed = true;
      const auto diagnostics_before_probe = exact_hvp.diagnostics();
      const double outer_response_seconds_before_probe =
          diagnostics_before_probe.outer_response_wall_time_seconds;
      const Eigen::VectorXd outer_response =
          exact_hvp.apply_outer(truncated_newton_step.reduced_step);
      truncated_newton_step.reduced_hessian_times_step += outer_response;
      const double probe_wall_time_seconds =
          exact_hvp.diagnostics().outer_response_wall_time_seconds -
          outer_response_seconds_before_probe;
      last_outer_response_seconds = probe_wall_time_seconds;
      truncated_newton_step.predicted_decrease =
          -current_projection.reduced_gradient.dot(
              truncated_newton_step.reduced_step) -
          0.5 * truncated_newton_step.reduced_step.dot(
              truncated_newton_step.reduced_hessian_times_step);
      refresh_truncated_newton_step_certificate(
          current_projection.reduced_gradient,
          &truncated_newton_step);
      minimize_truncated_newton_step_on_ray(
          current_projection,
          &truncated_newton_step);
      response_probe_relative_residual =
          truncated_newton_step.model_kkt_relative_residual;
      candidate_has_exact_outer_response = true;
      outer_response_used_for_current_point = true;

    }
    if (truncated_newton_step.newton_forcing_converged &&
        candidate_has_exact_outer_response) {
      ++result->matrix_free_residual_converged_count;
    }
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
      candidate_has_exact_outer_response = false;
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
              active_hvp);
      truncated_newton_step.reduced_step = reduced_step;
      truncated_newton_step.reduced_hessian_times_step.resize(0);
      truncated_newton_step.reduced_metric_times_step.resize(0);
      truncated_newton_step.retract_tangent_norm =
          retraction_metric.norm(reduced_step);
      truncated_newton_step.reached_boundary =
          truncated_newton_step.retract_tangent_norm >=
          (1.0 - 1.0e-8) * trust_radius;
      truncated_newton_step.predicted_decrease = predicted_decrease;
      truncated_newton_step.stop_reason =
          TruncatedNewtonStopReason::PreconditionedGradient;
      refresh_truncated_newton_step_certificate(
          current_projection.reduced_gradient, &truncated_newton_step);
    }
    const TruncatedNewtonStepResult model_step = truncated_newton_step;
    TruncatedNewtonStepResult trial_step_for_current_trial =
        truncated_newton_step;
  
    VbScfObjective::TrialEvaluation accepted_trial_evaluation;
    Eigen::VectorXd trial_parameters(current_parameters.size());
    Eigen::VectorXd trial_gradient(current_gradient.size());
    double trial_energy = energy;
    bool accepted_trial =
        try_truncated_newton_trial_step(
            reduced_step,
            predicted_decrease,
            &trial_evaluation_cache,
            &accepted_trial_evaluation,
            &trial_parameters,
            &trial_gradient,
            &trial_energy);
    if (!accepted_trial) {
      ++rejected_trial_step_count_for_current_point;
      request_outer_response = !outer_response_used_for_current_point;
      request_secant_correction =
          request_secant_correction || !response_scale_is_affordable;
      const double next_trust_radius =
          update_nonredundant_truncated_newton_trust_radius(
              trust_radius,
              options.minimum_step_size,
              trial_evaluation_cache,
              trial_step_for_current_trial,
              false);
      const auto hvp_diagnostics = exact_hvp.diagnostics();
      result->matrix_free_hvp_direction_count += hvp_diagnostics.apply_count;
      result->matrix_free_hvp_batch_count +=
          hvp_diagnostics.batch_apply_count;
      result->matrix_free_hvp_wall_time_seconds +=
          hvp_diagnostics.total_apply_wall_time_seconds;
      result->matrix_free_core_hvp_direction_count +=
          exact_hvp.core_direction_count();
      result->matrix_free_outer_response_direction_count +=
          exact_hvp.outer_response_direction_count();
      result->matrix_free_outer_response_wall_time_seconds +=
          hvp_diagnostics.outer_response_wall_time_seconds;
      trust_radius = next_trust_radius;
      if (trust_radius <= options.minimum_step_size) {
        result->termination_reason =
            "nonredundant_truncated_newton_trust_radius_exhausted";
        run_result.final_gradient_l2_norm = current_projection.reduced_gradient.norm();
        break;
      }
      rejected_step_cache.update(
          model_step,
          reduced_size,
          trust_radius,
          retraction_metric);
      continue;
    }
    const auto hvp_diagnostics = exact_hvp.diagnostics();
    result->matrix_free_hvp_direction_count += hvp_diagnostics.apply_count;
    result->matrix_free_hvp_batch_count +=
        hvp_diagnostics.batch_apply_count;
    result->matrix_free_hvp_wall_time_seconds +=
        hvp_diagnostics.total_apply_wall_time_seconds;
    result->matrix_free_core_hvp_direction_count +=
        exact_hvp.core_direction_count();
    result->matrix_free_outer_response_direction_count +=
        exact_hvp.outer_response_direction_count();
    result->matrix_free_outer_response_wall_time_seconds +=
        hvp_diagnostics.outer_response_wall_time_seconds;
    const bool accepted_point_chart_changed =
        accepted_trial_evaluation.chart_changed;
    const Eigen::VectorXd accepted_parameter_displacement =
        trial_parameters - current_parameters;
    const Eigen::VectorXd accepted_gradient_change =
        trial_gradient - current_gradient;
    objective->commit(std::move(accepted_trial_evaluation));
    current_parameters = trial_parameters;
    current_gradient = std::move(trial_gradient);
    energy = trial_energy;
    ++run_result.n_iterations;
    rejected_step_cache.clear();
    cached_subspace = TruncatedNewtonSubspace();
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
    const double source_gradient_l2_norm =
        current_projection.reduced_gradient.stableNorm();
    const double accepted_gradient_l2_norm =
        next_projection.reduced_gradient.stableNorm();
    const bool gradient_progressed =
        std::isfinite(source_gradient_l2_norm) &&
        std::isfinite(accepted_gradient_l2_norm) &&
        accepted_gradient_l2_norm < source_gradient_l2_norm;
    const double accepted_point_wall_time_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - accepted_point_start_time)
            .count();
    const double next_trust_radius =
        update_nonredundant_truncated_newton_trust_radius(
            trust_radius,
            options.minimum_step_size,
            trial_evaluation_cache,
            truncated_newton_step,
            true);
    const bool boundary_globalization_progressed =
        truncated_newton_step.reached_boundary &&
        !truncated_newton_step.encountered_negative_curvature &&
        gradient_progressed &&
        next_trust_radius > trust_radius;
    const bool continue_core_for_cost =
        response_deferred_for_cost && gradient_progressed;
    request_outer_response =
        !boundary_globalization_progressed && !continue_core_for_cost;
    TnhvpIterationRecord iteration_record;
    iteration_record.accepted_iteration_index = run_result.n_iterations;
    iteration_record.reduced_dimension = static_cast<int>(reduced_size);
    iteration_record.subspace_dimension =
        truncated_newton_step.subspace_dimension;
    iteration_record.secant_correction_size = core_hvp.correction_size();
    iteration_record.rejected_trial_count =
        rejected_trial_step_count_for_current_point;
    iteration_record.hvp_direction_count =
        result->matrix_free_hvp_direction_count -
        initial_hvp_direction_count_for_current_point;
    iteration_record.hvp_batch_count =
        result->matrix_free_hvp_batch_count -
        initial_hvp_batch_count_for_current_point;
    iteration_record.core_hvp_direction_count =
        result->matrix_free_core_hvp_direction_count -
        initial_core_hvp_direction_count_for_current_point;
    iteration_record.outer_response_direction_count =
        result->matrix_free_outer_response_direction_count -
        initial_outer_response_direction_count_for_current_point;
    iteration_record.subproblem_count =
        result->matrix_free_subproblem_count -
        initial_subproblem_count_for_current_point;
    iteration_record.hvp_wall_time_seconds =
        result->matrix_free_hvp_wall_time_seconds -
        initial_hvp_wall_time_for_current_point;
    iteration_record.outer_response_wall_time_seconds =
        result->matrix_free_outer_response_wall_time_seconds -
        initial_outer_response_wall_time_for_current_point;
    iteration_record.source_gradient_l2_norm = source_gradient_l2_norm;
    iteration_record.accepted_gradient_l2_norm = accepted_gradient_l2_norm;
    if (source_gradient_l2_norm > 0.0 &&
        accepted_gradient_l2_norm > 0.0 &&
        accepted_point_wall_time_seconds > 0.0 &&
        std::isfinite(accepted_point_wall_time_seconds)) {
      iteration_record.gradient_log_progress_per_second =
          -std::log(accepted_gradient_l2_norm / source_gradient_l2_norm) /
          accepted_point_wall_time_seconds;
    }
    iteration_record.used_outer_response =
        outer_response_used_for_current_point;
    iteration_record.response_probe_performed = response_probe_performed;
    iteration_record.response_probe_relative_residual =
        response_probe_relative_residual;
    if (response_cost_info.estimated_core_pair_contraction_work > 0.0) {
      iteration_record.response_work_ratio =
          response_cost_info.estimated_outer_string_contraction_work /
          response_cost_info.estimated_core_pair_contraction_work;
    }
    iteration_record.response_scale_affordable =
        response_scale_is_affordable;
    iteration_record.response_deferred_for_cost =
        response_deferred_for_cost;
    iteration_record.used_full_hvp = false;
    iteration_record.forcing_term =
        inexact_newton_forcing_term(iteration_record.source_gradient_l2_norm);
    if (candidate_has_exact_outer_response &&
        truncated_newton_step.reduced_hessian_times_step.size() ==
            current_projection.reduced_gradient.size() &&
        truncated_newton_step.reduced_hessian_times_step.allFinite()) {
      const Eigen::VectorXd kkt_residual =
          current_projection.reduced_gradient +
          truncated_newton_step.reduced_hessian_times_step +
          truncated_newton_step.trust_region_shift *
              truncated_newton_step.reduced_metric_times_step;
      const double residual_norm = kkt_residual.stableNorm();
      if (std::isfinite(residual_norm)) {
        iteration_record.has_kkt_residual = true;
        iteration_record.kkt_relative_residual =
            residual_norm /
            std::max(
                iteration_record.source_gradient_l2_norm,
                std::numeric_limits<double>::min());
        iteration_record.kkt_inf_norm =
            gradient_infinity_norm(kkt_residual);
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
    iteration_record.reused_subspace = reused_subspace_for_trial;
    iteration_record.chart_changed = accepted_point_chart_changed;
    result->tnhvp_iteration_trace.push_back(iteration_record);
    record_accepted_iteration_snapshot(
        objective,
        run_result.n_iterations,
        options,
        result,
        &iteration_record,
        &next_projection.reduced_gradient);
    trust_radius = next_trust_radius;
    rejected_trial_step_count_for_current_point = 0;
    outer_response_used_for_current_point = false;
    initial_trust_radius_for_current_point = trust_radius;
    initial_hvp_direction_count_for_current_point =
        result->matrix_free_hvp_direction_count;
    initial_hvp_batch_count_for_current_point =
        result->matrix_free_hvp_batch_count;
    initial_subproblem_count_for_current_point =
        result->matrix_free_subproblem_count;
    initial_hvp_wall_time_for_current_point =
        result->matrix_free_hvp_wall_time_seconds;
    initial_core_hvp_direction_count_for_current_point =
        result->matrix_free_core_hvp_direction_count;
    initial_outer_response_direction_count_for_current_point =
        result->matrix_free_outer_response_direction_count;
    initial_outer_response_wall_time_for_current_point =
        result->matrix_free_outer_response_wall_time_seconds;
    accepted_point_start_time = std::chrono::steady_clock::now();
    if (nonredundant_rank_changed) {
      packed_secant_history.clear();
      request_secant_correction = false;
    }
    if (!nonredundant_rank_changed &&
        transport_history_size > 0) {
      append_nonredundant_truncated_newton_secant_pair(
          accepted_parameter_displacement,
          accepted_gradient_change,
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
