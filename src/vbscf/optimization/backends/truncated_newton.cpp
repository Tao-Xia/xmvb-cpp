#include "vbscf/optimization/backends/truncated_newton.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "vbscf/optimization/globalization/line_search.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/optimization/coupled/workspace.hpp"
#include "vbscf/optimization/driver/checks.hpp"
#include "vbscf/optimization/driver/session.hpp"
#include "vbscf/optimization/driver/types.hpp"
#include "vbscf/optimization/preconditioners/transported_lbfgs.hpp"
#include "vbscf/derivatives/hessian/exact/operator.hpp"
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
  std::shared_ptr<ExactHvpOperator> accepted_point_operator;
  std::unique_ptr<TransportedReducedLbfgsPreconditioner>
      accepted_point_preconditioner;
  std::unique_ptr<NonredundantRetractionMetric> accepted_point_metric;
  std::unique_ptr<AcceptedPointCoupledWorkspace>
      accepted_point_coupled_workspace;
  int coupled_expansions_for_current_point = 0;
  double initial_trust_radius_for_current_point = trust_radius;
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
    const double newton_forcing_term = inexact_newton_forcing_term(
        final_projected_gradient_l2_norm);
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
  
    if (accepted_point_operator == nullptr) {
      accepted_point_operator = std::make_shared<ExactHvpOperator>(
          objective->second_order_context(),
          &objective->input(),
          parameter_view,
          &current_space);
    }
    ExactHvpOperator& exact_operator = *accepted_point_operator;
    if (!exact_operator.supports_analytic_core_model()) {
      throw std::runtime_error(
          "coupled Newton requires the analytic orbital Hessian action");
    }
    const int transport_history_size =
        choose_truncated_newton_transport_history_size(options);
    if (accepted_point_preconditioner == nullptr) {
      accepted_point_preconditioner =
          std::make_unique<TransportedReducedLbfgsPreconditioner>(
              build_nonredundant_truncated_newton_preconditioner(
                  current_space,
                  packed_secant_history,
                  transport_history_size));
    }
    const OrbitalPreparationInput current_orbital_input =
        objective->input().orbital_preparation_input;
    if (accepted_point_metric == nullptr) {
      accepted_point_metric =
          std::make_unique<NonredundantRetractionMetric>(
              current_space, parameter_view, current_orbital_input);
    }
    const NonredundantRetractionMetric& retraction_metric =
        *accepted_point_metric;
    auto try_truncated_newton_trial_step =
        [&](const Eigen::VectorXd& candidate_reduced_step,
            double candidate_predicted_decrease,
            double candidate_linear_decrease,
            bool accept_model_inaccurate_monotone,
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
            trial_evaluation->linear_decrease = candidate_linear_decrease;
          }
  
          VbScfObjective::TrialEvaluation candidate_trial_evaluation =
              objective->evaluate_trial_energy(
                  candidate_trial_parameters,
                  true);
          double candidate_trial_energy =
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
                      effective_predicted_decrease},
                  accept_model_inaccurate_monotone)) {
            return false;
          }

          objective->complete_trial(&candidate_trial_evaluation);
          candidate_trial_energy = candidate_trial_evaluation.energy;
  
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
    const bool reused_subspace_for_trial =
        accepted_point_coupled_workspace != nullptr;
    if (accepted_point_coupled_workspace == nullptr) {
      auto apply_orbital_metric =
          [&retraction_metric](
              const Eigen::Ref<const Eigen::MatrixXd>& directions) {
            Eigen::MatrixXd images(directions.rows(), directions.cols());
            for (Eigen::Index column = 0;
                 column < directions.cols();
                 ++column) {
              images.col(column) =
                  retraction_metric.apply(directions.col(column));
            }
            return images;
          };
      AcceptedPointCoupledModel coupled_model =
          make_accepted_point_coupled_model(
              *objective->second_order_context(),
              accepted_point_operator,
              static_cast<int>(reduced_size),
              std::move(apply_orbital_metric));
      const OrbitalChart* const accepted_chart = &current_space;
      const TransportedReducedLbfgsPreconditioner* const
          orbital_preconditioner = accepted_point_preconditioner.get();
      auto apply_inverse_orbital_preconditioner =
          [accepted_chart, orbital_preconditioner](
              const Eigen::VectorXd& vector) {
            return apply_nonredundant_truncated_newton_preconditioner(
                *accepted_chart,
                orbital_preconditioner,
                vector);
          };
      accepted_point_coupled_workspace =
          std::make_unique<AcceptedPointCoupledWorkspace>(
              std::move(coupled_model),
              current_projection.reduced_gradient,
              std::move(apply_inverse_orbital_preconditioner));
    }

    const AcceptedPointContext& accepted_context =
        *objective->second_order_context();
    double selected_energy_scale = 0.0;
    for (const double selected_energy :
         accepted_context.selected_state_energies) {
      selected_energy_scale = std::max(
          selected_energy_scale,
          std::abs(selected_energy));
    }
    const double response_backward_error_tolerance =
        accepted_context.structure_solve_accuracy
            .response_backward_error_tolerance(selected_energy_scale);
    const CoupledKktTolerances coupled_tolerances{
        newton_forcing_term,
        std::min(
            newton_forcing_term,
            response_backward_error_tolerance)};
    CoupledWorkspaceResult coupled_result =
        accepted_point_coupled_workspace->solve(
            trust_radius,
            coupled_tolerances);
    coupled_expansions_for_current_point += coupled_result.expansions;
    if (coupled_result.status == CoupledWorkspaceStatus::WorkLimit) {
      result->termination_reason =
          "coupled_newton_algebraic_work_limit";
      run_result.final_gradient_l2_norm =
          current_projection.reduced_gradient.norm();
      break;
    }
    if (coupled_result.status == CoupledWorkspaceStatus::NumericalFailure) {
      result->termination_reason = "coupled_newton_numerical_failure";
      run_result.final_gradient_l2_norm =
          current_projection.reduced_gradient.norm();
      break;
    }
    if (coupled_result.status == CoupledWorkspaceStatus::SubproblemFailure) {
      result->termination_reason = "coupled_newton_subproblem_failure";
      run_result.final_gradient_l2_norm =
          current_projection.reduced_gradient.norm();
      break;
    }

    const CoupledSubspaceStep& coupled_step = coupled_result.step;
    TruncatedNewtonStepResult trust_region_step;
    trust_region_step.retract_tangent_norm =
        coupled_step.projected.orbital_solution.metric_norm;
    trust_region_step.reached_boundary =
        coupled_step.projected.orbital_solution.status !=
        ProjectedTrustStatus::InteriorGlobal;
    const bool encountered_negative_curvature =
        coupled_step.projected.orbital_solution.minimum_ritz_value < 0.0;
    if (coupled_step.orbital_backward_error >
            coupled_tolerances.orbital ||
        coupled_step.response_backward_error >
            coupled_tolerances.response) {
      throw std::logic_error(
          "coupled Newton workspace returned an uncertified step");
    }
    const double coupled_linear_decrease =
        -current_projection.reduced_gradient.dot(coupled_step.orbital_step) -
        accepted_point_coupled_workspace->accepted_model()
            .structure_kkt_residual.dot(coupled_step.response_step);
    Eigen::VectorXd reduced_step = coupled_step.orbital_step;
    if (reduced_step.size() != current_projection.reduced_gradient.size()) {
      result->termination_reason =
          "nonredundant_truncated_newton_invalid_step_dimension";
      run_result.final_gradient_l2_norm = current_projection.reduced_gradient.norm();
      break;
    }
    double predicted_decrease = coupled_step.predicted_decrease;
    if (!std::isfinite(predicted_decrease) ||
        predicted_decrease <= 0.0) {
      result->termination_reason =
          "coupled_newton_nonpositive_predicted_decrease";
      run_result.final_gradient_l2_norm =
          current_projection.reduced_gradient.norm();
      break;
    }
    TruncatedNewtonStepResult trial_step_for_current_trial =
        trust_region_step;
  
    VbScfObjective::TrialEvaluation accepted_trial_evaluation;
    Eigen::VectorXd trial_parameters(current_parameters.size());
    Eigen::VectorXd trial_gradient(current_gradient.size());
    double trial_energy = energy;
    bool accepted_trial =
        try_truncated_newton_trial_step(
            reduced_step,
            predicted_decrease,
            coupled_linear_decrease,
            false,
            &trial_evaluation_cache,
            &accepted_trial_evaluation,
            &trial_parameters,
            &trial_gradient,
            &trial_energy);
    if (!accepted_trial) {
      ++rejected_trial_step_count_for_current_point;
      const double next_trust_radius =
          update_nonredundant_truncated_newton_trust_radius(
              trust_radius,
              options.minimum_step_size,
              trial_evaluation_cache,
              trial_step_for_current_trial,
              TruncatedNewtonModelFidelity::DirectionallyExact,
              false);
      trust_radius = next_trust_radius;
      if (trust_radius <= options.minimum_step_size) {
        result->termination_reason =
            "nonredundant_truncated_newton_trust_radius_exhausted";
        run_result.final_gradient_l2_norm = current_projection.reduced_gradient.norm();
        break;
      }
      continue;
    }
    const bool accepted_point_chart_changed =
        accepted_trial_evaluation.chart_changed;
    const Eigen::VectorXd accepted_parameter_displacement =
        trial_parameters - current_parameters;
    const Eigen::VectorXd accepted_gradient_change =
        trial_gradient - current_gradient;
    const int accepted_preconditioner_history_size =
        accepted_point_preconditioner->size();
    const int accepted_coupled_orbital_dimension =
        coupled_result.orbital_dimension;
    const int accepted_coupled_response_dimension =
        coupled_result.response_dimension;
    const CoupledActionCounts accepted_coupled_action_counts =
        coupled_result.action_counts;
    // Every callback retained by the coupled workspace represents this exact
    // accepted point and refers to its chart/metric.  Destroy it before the
    // objective commit changes that point; rejected radius retries retain it.
    accepted_point_coupled_workspace.reset();
    accepted_point_operator.reset();
    accepted_point_preconditioner.reset();
    accepted_point_metric.reset();
    objective->commit(std::move(accepted_trial_evaluation));
    current_parameters = trial_parameters;
    current_gradient = std::move(trial_gradient);
    energy = trial_energy;
    ++run_result.n_iterations;
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
    const double accepted_point_wall_time_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - accepted_point_start_time)
            .count();
    const double next_trust_radius =
        update_nonredundant_truncated_newton_trust_radius(
            trust_radius,
            options.minimum_step_size,
            trial_evaluation_cache,
            trust_region_step,
            TruncatedNewtonModelFidelity::DirectionallyExact,
            true);
    TnhvpIterationRecord iteration_record;
    iteration_record.accepted_iteration_index = run_result.n_iterations;
    iteration_record.reduced_dimension = static_cast<int>(reduced_size);
    iteration_record.coupled_orbital_subspace_dimension =
        accepted_coupled_orbital_dimension;
    iteration_record.coupled_response_subspace_dimension =
        accepted_coupled_response_dimension;
    iteration_record.coupled_expansion_count =
        coupled_expansions_for_current_point;
    iteration_record.coupled_orbital_hessian_block_actions =
        accepted_coupled_action_counts.orbital_hessian;
    iteration_record.coupled_orbital_to_response_block_actions =
        accepted_coupled_action_counts.orbital_to_response;
    iteration_record.coupled_response_to_orbital_block_actions =
        accepted_coupled_action_counts.response_to_orbital;
    iteration_record.coupled_response_hessian_block_actions =
        accepted_coupled_action_counts.response_hessian;
    iteration_record.coupled_orbital_metric_block_actions =
        accepted_coupled_action_counts.orbital_metric;
    iteration_record.preconditioner_history_size =
        accepted_preconditioner_history_size;
    iteration_record.rejected_trial_count =
        rejected_trial_step_count_for_current_point;
    iteration_record.outer_iteration_wall_time_seconds =
        accepted_point_wall_time_seconds;
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
    iteration_record.orbital_backward_error =
        coupled_step.orbital_backward_error;
    iteration_record.response_backward_error =
        coupled_step.response_backward_error;
    iteration_record.forcing_term = newton_forcing_term;
    iteration_record.initial_trust_radius =
        initial_trust_radius_for_current_point;
    iteration_record.accepted_trial_radius = trust_radius;
    iteration_record.next_trust_radius = next_trust_radius;
    iteration_record.step_norm = trust_region_step.retract_tangent_norm;
    iteration_record.linear_decrease =
        trial_evaluation_cache.linear_decrease;
    iteration_record.predicted_decrease =
        trial_evaluation_cache.predicted_decrease;
    iteration_record.actual_decrease = trial_evaluation_cache.actual_decrease;
    if (iteration_record.predicted_decrease > 0.0 &&
        std::isfinite(iteration_record.predicted_decrease)) {
      iteration_record.trust_ratio =
          iteration_record.actual_decrease /
          iteration_record.predicted_decrease;
    }
    iteration_record.minimum_ritz_value =
        coupled_step.projected.orbital_solution.minimum_ritz_value;
    iteration_record.minimum_shifted_ritz_value =
        coupled_step.projected.orbital_solution.minimum_shifted_ritz_value;
    iteration_record.trust_region_shift =
        coupled_step.projected.orbital_solution.shift;
    iteration_record.reached_boundary = trust_region_step.reached_boundary;
    iteration_record.encountered_negative_curvature =
        encountered_negative_curvature;
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
    coupled_expansions_for_current_point = 0;
    initial_trust_radius_for_current_point = trust_radius;
    accepted_point_start_time = std::chrono::steady_clock::now();
    if (nonredundant_rank_changed) {
      packed_secant_history.clear();
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
