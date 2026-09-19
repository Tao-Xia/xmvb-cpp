#include "vbscf/optimization/backends/truncated_newton.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "vbscf/optimization/globalization/line_search.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
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
namespace {

constexpr int kCurvatureBlockWidth = 2;
constexpr int kCurvatureExpansionCount = 2;
constexpr int kCurvatureModelDimension =
    kCurvatureBlockWidth * kCurvatureExpansionCount;

class AcceptedPointReducedHvp final : public ReducedHvp {
public:
  explicit AcceptedPointReducedHvp(const ExactHvpOperator* exact_operator)
      : exact_operator_(exact_operator) {
    if (exact_operator_ == nullptr) {
      throw std::invalid_argument("exact HVP operator must not be null");
    }
  }

  Eigen::VectorXd apply(const Eigen::VectorXd& direction) override {
    return exact_operator_->apply_reduced(direction);
  }

  Eigen::MatrixXd apply_batch(
      const Eigen::Ref<const Eigen::MatrixXd>& directions) override {
    return exact_operator_->apply_reduced_batch(directions);
  }

private:
  const ExactHvpOperator* exact_operator_;
};

}  // namespace

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
  const double initial_projected_gradient_l2_norm =
      final_projected_gradient_l2_norm;
  std::vector<PackedSecantPair> packed_secant_history;
  packed_secant_history.reserve(std::max(0, options.history_size));
  double trust_radius =
      std::max(options.minimum_step_size, options.initial_step_size);
  int rejected_trial_step_count_for_current_point = 0;
  std::shared_ptr<ExactHvpOperator> accepted_point_operator;
  std::unique_ptr<TransportedReducedLbfgsPreconditioner>
      accepted_point_preconditioner;
  std::unique_ptr<NonredundantRetractionMetric> accepted_point_metric;
  double initial_trust_radius_for_current_point = trust_radius;
  double accepted_point_setup_wall_time_seconds = 0.0;
  double trial_objective_wall_time_seconds = 0.0;
  bool curvature_correction_required = false;
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
    const double newton_forcing_term = inexact_newton_forcing_term(
        final_projected_gradient_l2_norm,
        reduced_gradient_inf_norm,
        initial_projected_gradient_l2_norm,
        options.gradient_tolerance);
  
    if (!(trust_radius > 0.0) || !std::isfinite(trust_radius)) {
      result->termination_reason =
          "nonredundant_truncated_newton_invalid_trust_radius";
      run_result.final_gradient_l2_norm = current_projection.reduced_gradient.norm();
      break;
    }
  
    // Use the same secant memory as standalone L-BFGS, but initialize its
    // inverse action with the positive orbital-curvature block. This is part
    // of the TNHVP preconditioner; standalone L-BFGS uses the conventional
    // scalar initialization so it remains an honest first-order baseline.
    const int transport_history_size = options.history_size;
    if (accepted_point_preconditioner == nullptr) {
      const auto setup_start = std::chrono::steady_clock::now();
      accepted_point_preconditioner =
          std::make_unique<TransportedReducedLbfgsPreconditioner>(
              build_transported_reduced_lbfgs_preconditioner(
                  current_space,
                  packed_secant_history,
                  transport_history_size,
                  LbfgsInitialInverse::OrbitalBlock));
      accepted_point_setup_wall_time_seconds +=
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - setup_start).count();
    }
    const OrbitalPreparationInput current_orbital_input =
        objective->input().orbital_preparation_input;
    if (accepted_point_metric == nullptr) {
      const auto setup_start = std::chrono::steady_clock::now();
      accepted_point_metric =
          std::make_unique<NonredundantRetractionMetric>(
              current_space, parameter_view, current_orbital_input);
      accepted_point_setup_wall_time_seconds +=
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - setup_start).count();
    }
    const NonredundantRetractionMetric& retraction_metric =
        *accepted_point_metric;
    // Build the primary L-BFGS direction before the coupled
    // workspace captures its inverse action.  If transported secants lose
    // descent through roundoff, discard them for both candidates so the
    // baseline and curvature model use one positive-definite preconditioner.
    Eigen::VectorXd baseline_reduced_direction =
        -accepted_point_preconditioner->apply(
            current_projection.reduced_gradient);
    if (!baseline_reduced_direction.allFinite() ||
        current_projection.reduced_gradient.dot(
            baseline_reduced_direction) >= 0.0) {
      packed_secant_history.clear();
      accepted_point_preconditioner =
          std::make_unique<TransportedReducedLbfgsPreconditioner>(
              build_transported_reduced_lbfgs_preconditioner(
                  current_space, packed_secant_history,
                  transport_history_size,
                  LbfgsInitialInverse::OrbitalBlock));
      baseline_reduced_direction =
          -accepted_point_preconditioner->apply(
              current_projection.reduced_gradient);
    }
    if (!baseline_reduced_direction.allFinite() ||
        current_projection.reduced_gradient.dot(
            baseline_reduced_direction) >= 0.0) {
      baseline_reduced_direction =
          -current_projection.reduced_gradient;
    }
    auto try_truncated_newton_trial_step =
        [&](const Eigen::VectorXd& candidate_reduced_step,
            double candidate_predicted_decrease,
            double candidate_linear_decrease,
            bool accept_model_inaccurate_monotone,
            TruncatedNewtonTrialEvaluation* trial_evaluation,
            VbScfObjective::TrialEvaluation* accepted_trial_evaluation,
            Eigen::VectorXd* accepted_trial_parameters,
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
          trial_objective_wall_time_seconds +=
              candidate_trial_evaluation.wall_time_seconds;
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

          *accepted_trial_parameters = parameter_view.pack(
              candidate_trial_evaluation.orbital_preparation_input);
          *accepted_trial_evaluation =
              std::move(candidate_trial_evaluation);
          *accepted_trial_energy = candidate_trial_energy;
          return true;
        };
    auto try_baseline_trial_step =
        [&](const Eigen::VectorXd& baseline_direction,
            TruncatedNewtonTrialEvaluation* trial_evaluation,
            VbScfObjective::TrialEvaluation* accepted_trial_evaluation,
            Eigen::VectorXd* accepted_trial_parameters,
            double* accepted_trial_energy,
            Eigen::VectorXd* accepted_reduced_step) -> bool {
          const double full_linear_decrease =
              -current_projection.reduced_gradient.dot(baseline_direction);
          if (!(full_linear_decrease > 0.0) ||
              !std::isfinite(full_linear_decrease)) {
            return false;
          }

          double scale = std::min(1.0, options.initial_step_size);
          while (scale >= options.minimum_step_size) {
            const Eigen::VectorXd candidate_reduced_step =
                scale * baseline_direction;
            Eigen::VectorXd candidate_trial_parameters =
                build_nonredundant_lifted_trial_parameters(
                    current_orbital_input,
                    current_space,
                    parameter_view,
                    candidate_reduced_step);
            if (is_effectively_zero_step(
                    candidate_trial_parameters - current_parameters,
                    current_parameters)) {
              scale *= 0.5;
              continue;
            }

            VbScfObjective::TrialEvaluation candidate_trial_evaluation =
                objective->evaluate_trial_energy(
                    candidate_trial_parameters,
                    true);
            trial_objective_wall_time_seconds +=
                candidate_trial_evaluation.wall_time_seconds;
            const double candidate_trial_energy =
                candidate_trial_evaluation.energy;
            const double actual_decrease = energy - candidate_trial_energy;
            const double linear_decrease =
                scale * full_linear_decrease;
            if (std::isfinite(candidate_trial_energy) &&
                actual_decrease >=
                    options.armijo_constant * linear_decrease) {
              *accepted_trial_parameters = parameter_view.pack(
                  candidate_trial_evaluation.orbital_preparation_input);
              *accepted_trial_energy =
                  candidate_trial_evaluation.energy;
              *accepted_trial_evaluation =
                  std::move(candidate_trial_evaluation);
              *accepted_reduced_step = candidate_reduced_step;
              *trial_evaluation = TruncatedNewtonTrialEvaluation{
                  actual_decrease,
                  linear_decrease,
                  linear_decrease};
              return true;
            }
            scale *= 0.5;
          }
          return false;
        };
    const Eigen::Index reduced_size =
        current_projection.reduced_gradient.size();
    TruncatedNewtonTrialEvaluation trial_evaluation_cache;
    // Certify the block-L-BFGS predictor with its exact Hessian image, then
    // expand only along the resulting Newton defect.  The forcing condition,
    // rather than a fixed amount of secant history, decides whether curvature
    // correction is required at this accepted point.
    TruncatedNewtonStepResult trust_region_step;
    bool newton_candidate_available = false;
    if (curvature_correction_required) {
      if (accepted_point_operator == nullptr) {
        const auto setup_start = std::chrono::steady_clock::now();
        accepted_point_operator = std::make_shared<ExactHvpOperator>(
            objective->second_order_context(),
            &objective->input(),
            parameter_view,
            &current_space);
        accepted_point_setup_wall_time_seconds +=
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - setup_start).count();
      }
      ExactHvpOperator& exact_operator = *accepted_point_operator;
      if (!exact_operator.supports_analytic_core_model()) {
        throw std::runtime_error(
            "TNHVP requires the analytic orbital Hessian action");
      }
      AcceptedPointReducedHvp reduced_hvp(&exact_operator);
      trust_region_step = solve_nonredundant_truncated_newton_step(
          retraction_metric,
          current_space,
          current_projection,
          trust_radius,
          options.energy_tolerance,
          options.gradient_tolerance,
          newton_forcing_term,
          std::min<int>(reduced_size, kCurvatureModelDimension),
          &reduced_hvp,
          accepted_point_preconditioner.get(),
          &baseline_reduced_direction);
      clamp_nonredundant_step_result_to_retract_tangent_radius(
          current_projection,
          trust_radius,
          retraction_metric,
          &trust_region_step);
      newton_candidate_available = truncated_newton_step_is_usable(
          trust_region_step,
          current_projection.reduced_gradient);
    }
    Eigen::VectorXd reduced_step = newton_candidate_available
        ? trust_region_step.reduced_step
        : Eigen::VectorXd::Zero(reduced_size);
    const double predicted_decrease = newton_candidate_available
        ? trust_region_step.predicted_decrease
        : 0.0;
    const double model_linear_decrease = newton_candidate_available
        ? -current_projection.reduced_gradient.dot(reduced_step)
        : 0.0;
    TruncatedNewtonStepResult trial_step_for_current_trial =
        trust_region_step;
  
    VbScfObjective::TrialEvaluation accepted_trial_evaluation;
    Eigen::VectorXd trial_parameters(current_parameters.size());
    double trial_energy = energy;
    const bool accepted_newton_trial = newton_candidate_available &&
        try_truncated_newton_trial_step(
            reduced_step,
            predicted_decrease,
            model_linear_decrease,
            false,
            &trial_evaluation_cache,
            &accepted_trial_evaluation,
            &trial_parameters,
            &trial_energy);
    bool accepted_baseline_trial = false;
    if (!accepted_newton_trial) {
      if (curvature_correction_required) {
        ++rejected_trial_step_count_for_current_point;
      }
      Eigen::VectorXd accepted_baseline_reduced_step;
      accepted_baseline_trial = try_baseline_trial_step(
          baseline_reduced_direction,
          &trial_evaluation_cache,
          &accepted_trial_evaluation,
          &trial_parameters,
          &trial_energy,
          &accepted_baseline_reduced_step);
      if (!accepted_baseline_trial) {
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
          run_result.final_gradient_l2_norm =
              current_projection.reduced_gradient.norm();
          break;
        }
        continue;
      }
      reduced_step = std::move(accepted_baseline_reduced_step);
      trial_step_for_current_trial.retract_tangent_norm =
          retraction_metric.norm(reduced_step);
      trial_step_for_current_trial.reached_boundary =
          trial_step_for_current_trial.retract_tangent_norm >=
          (1.0 - std::sqrt(std::numeric_limits<double>::epsilon())) *
              trust_radius;
      trial_step_for_current_trial.encountered_negative_curvature = false;
    }
    const double energy_only_wall_time_seconds =
        accepted_trial_evaluation.wall_time_seconds;
    objective->complete_trial(&accepted_trial_evaluation);
    trial_objective_wall_time_seconds +=
        accepted_trial_evaluation.wall_time_seconds -
        energy_only_wall_time_seconds;
    Eigen::VectorXd trial_gradient =
        accepted_trial_evaluation.gradient;
    trial_energy = accepted_trial_evaluation.energy;
    const bool accepted_point_chart_changed =
        accepted_trial_evaluation.chart_changed;
    const Eigen::VectorXd accepted_parameter_displacement =
        trial_parameters - current_parameters;
    const Eigen::VectorXd accepted_gradient_change =
        trial_gradient - current_gradient;
    const int accepted_preconditioner_history_size =
        accepted_point_preconditioner->size();
    const ExactHvpOperator::Diagnostics accepted_hvp_diagnostics =
        accepted_point_operator != nullptr
            ? accepted_point_operator->diagnostics()
            : ExactHvpOperator::Diagnostics{};
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
    curvature_correction_required =
        observed_contraction_requires_newton_correction(
            source_gradient_l2_norm,
            accepted_gradient_l2_norm,
            newton_forcing_term);
    const double accepted_point_wall_time_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - accepted_point_start_time)
            .count();
    const double next_trust_radius =
        update_nonredundant_truncated_newton_trust_radius(
            trust_radius,
            options.minimum_step_size,
            trial_evaluation_cache,
            trial_step_for_current_trial,
            accepted_baseline_trial
                ? TruncatedNewtonModelFidelity::CoreApproximate
                : TruncatedNewtonModelFidelity::DirectionallyExact,
            true);
    TnhvpIterationRecord iteration_record;
    iteration_record.accepted_iteration_index = run_result.n_iterations;
    iteration_record.reduced_dimension = static_cast<int>(reduced_size);
    iteration_record.curvature_subspace_dimension =
        trust_region_step.subspace_dimension;
    iteration_record.exact_hvp_block_actions =
        static_cast<int>(accepted_hvp_diagnostics.batch_apply_count);
    iteration_record.structure_response_block_actions =
        static_cast<int>(
            accepted_hvp_diagnostics.structure_response_block_actions);
    iteration_record.preconditioner_history_size =
        accepted_preconditioner_history_size;
    iteration_record.rejected_trial_count =
        rejected_trial_step_count_for_current_point;
    iteration_record.outer_iteration_wall_time_seconds =
        accepted_point_wall_time_seconds;
    iteration_record.accepted_point_setup_wall_time_seconds =
        accepted_point_setup_wall_time_seconds;
    iteration_record.exact_hvp_wall_time_seconds =
        accepted_hvp_diagnostics.total_apply_wall_time_seconds;
    iteration_record.outer_response_wall_time_seconds =
        accepted_hvp_diagnostics.outer_response_wall_time_seconds;
    iteration_record.trial_objective_wall_time_seconds =
        trial_objective_wall_time_seconds;
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
    iteration_record.model_kkt_relative_residual =
        trust_region_step.model_kkt_relative_residual;
    iteration_record.max_structure_response_relative_residual =
        accepted_hvp_diagnostics.max_structure_response_relative_residual;
    iteration_record.forcing_term = newton_forcing_term;
    iteration_record.initial_trust_radius =
        initial_trust_radius_for_current_point;
    iteration_record.accepted_trial_radius = trust_radius;
    iteration_record.next_trust_radius = next_trust_radius;
    iteration_record.step_norm =
        trial_step_for_current_trial.retract_tangent_norm;
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
        trust_region_step.encountered_negative_curvature
            ? -trust_region_step.model_spectral_radius
            : 0.0;
    iteration_record.minimum_shifted_ritz_value =
        iteration_record.minimum_ritz_value +
        trust_region_step.trust_region_shift;
    iteration_record.trust_region_shift =
        trust_region_step.trust_region_shift;
    iteration_record.reached_boundary =
        trial_step_for_current_trial.reached_boundary;
    iteration_record.encountered_negative_curvature =
        !accepted_baseline_trial &&
        trust_region_step.encountered_negative_curvature;
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
    initial_trust_radius_for_current_point = trust_radius;
    accepted_point_setup_wall_time_seconds = 0.0;
    trial_objective_wall_time_seconds = 0.0;
    accepted_point_start_time = std::chrono::steady_clock::now();
    if (nonredundant_rank_changed) {
      packed_secant_history.clear();
    }
    if (!nonredundant_rank_changed &&
        transport_history_size > 0) {
      append_projected_secant_pair(
          next_space,
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
