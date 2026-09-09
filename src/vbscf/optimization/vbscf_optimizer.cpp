#include "vbscf/optimization/vbscf_optimizer.hpp"

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
#include <optional>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <LBFGS.h>

#include "vbscf/orbitals/gauge/localized_representative.hpp"
#include "vbscf/orbitals/charts/support_layout_adapter.hpp"
#include "vbscf/orbitals/charts/orbital_chart.hpp"
#include "vbscf/orbitals/charts/sparse_parameter_layout.hpp"
#include "vbscf/orbitals/gauge/support_preserving_gauge.hpp"
#include "vbscf/optimization/vbscf_objective.hpp"
#include "vbscf/optimization/backends/lbfgs_backends.hpp"
#include "vbscf/optimization/backends/projected_gradient_backend.hpp"
#include "vbscf/optimization/optimizer_types.hpp"
#include "vbscf/optimization/optimizer_session.hpp"
#include "vbscf/optimization/line_search.hpp"
#include "vbscf/optimization/krylov/positive_ritz_secants.hpp"
#include "vbscf/optimization/preconditioners/transported_lbfgs_preconditioner.hpp"
#include "vbscf/optimization/reduced_hvp_operator.hpp"
#include "vbscf/optimization/trust_region/retraction_metric.hpp"
#include "vbscf/optimization/trust_region/truncated_newton_solver.hpp"
#include "vbscf/optimization/optimization_checks.hpp"

namespace xmvb::vb {

using optimizer_detail::build_orbital_chart;
using optimizer_detail::choose_truncated_newton_max_cg_iterations;
using optimizer_detail::choose_truncated_newton_transport_history_size;
using optimizer_detail::record_accepted_iteration_snapshot;
using optimizer_detail::sync_result_from_objective;
using optimizer_detail::uses_nonredundant_space;

VbScfOptimizer::VbScfOptimizer(
    VbScfOptimizerOptions options)
    : orbital_gradient_evaluator_(),
      scf_evaluator_(),
      options_(options) {}

VbScfOptimizer::VbScfOptimizer(
    OrbitalGradientEvaluator orbital_gradient_evaluator,
    VbScfEvaluator scf_evaluator,
    VbScfOptimizerOptions options)
    : orbital_gradient_evaluator_(std::move(orbital_gradient_evaluator)),
      scf_evaluator_(std::move(scf_evaluator)),
      options_(options) {}

VbScfOptimizerResult VbScfOptimizer::optimize(
    const VbScfInput& input,
    double nuclear_repulsion_energy) const {
  return optimize(input, {0}, {1.0}, nuclear_repulsion_energy);
}

VbScfOptimizerResult VbScfOptimizer::optimize(
    const VbScfInput& input,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    double nuclear_repulsion_energy) const {
  if (options_.max_iterations <= 0) {
    throw std::invalid_argument("max_iterations must be positive");
  }
  if (options_.gradient_tolerance <= 0.0 ||
      options_.energy_tolerance <= 0.0 ||
      options_.initial_step_size <= 0.0 ||
      options_.minimum_step_size <= 0.0) {
    throw std::invalid_argument("optimizer tolerances and step sizes must be positive");
  }
  if (options_.minimum_step_size > options_.initial_step_size) {
    throw std::invalid_argument("minimum_step_size must not exceed initial_step_size");
  }
  if (options_.history_size <= 0) {
    throw std::invalid_argument("history_size must be positive");
  }
  if (options_.nonredundant_truncated_newton_max_cg_iterations < 0) {
    throw std::invalid_argument(
        "nonredundant_truncated_newton_max_cg_iterations must be nonnegative");
  }
  if (options_.nonredundant_truncated_newton_hvp_step_size <= 0.0) {
    throw std::invalid_argument(
        "nonredundant_truncated_newton_hvp_step_size must be positive");
  }
  if (options_.nonredundant_truncated_newton_transport_history_size < 0) {
    throw std::invalid_argument(
        "nonredundant_truncated_newton_transport_history_size must be nonnegative");
  }
  VbScfOptimizerResult result;
  const auto optimization_start_time = std::chrono::steady_clock::now();

  // For `guess=mo`, numerical parity with the legacy VBSCF implementation is
  // more important than any temporary convergence-speed heuristic. Keep the
  // nonredundant optimizer on the original legacy sparse chart and exact-
  // support block partition so the reduced coordinates, projected gradients,
  // and exact-context orbital derivatives all live on the same variational
  // manifold as the reference `.xmo` calculation.
  std::optional<VbScfInput> adapted_optimizer_input;
  const VbScfInput* optimizer_input = &input;
  if (uses_nonredundant_space(options_.backend)) {
    adapted_optimizer_input = build_nonredundant_optimizer_input(input);
    optimizer_input = &adapted_optimizer_input.value();
  }
  const SparseParameterLayout parameter_view(
      optimizer_input->orbital_preparation_input);
  Eigen::VectorXd parameter_vector =
      parameter_view.pack(optimizer_input->orbital_preparation_input);
  Eigen::MatrixXd initial_normalized_orbital_matrix;

  VbScfObjective objective(
      *optimizer_input,
      parameter_view,
      selected_state_indices,
      state_average_weights,
      nuclear_repulsion_energy,
      &orbital_gradient_evaluator_,
      &scf_evaluator_);
  const int n = static_cast<int>(parameter_vector.size());
  int n_iterations = 0;
  double final_gradient_l2_norm = 0.0;
  bool final_projected_gradient_ready = false;

  try {
    Eigen::VectorXd gradient(parameter_vector.size());
    double energy = objective(parameter_vector, gradient);
    initial_normalized_orbital_matrix =
        objective.last_gradient_result()
            .orbital_preparation_result
            .physical_orbital_frame
            .normalized_orbital_matrix;
    sync_result_from_objective(objective, &result);
    record_accepted_iteration_snapshot(&objective, 0, options_, &result);
    result.initial_total_energy = energy;
    result.initial_one_electron_reference_energy =
        objective.last_gradient_result().scf_result.one_electron_reference_energy;
    double previous_energy = energy;
    final_gradient_l2_norm = gradient.norm();
    switch (options_.backend) {

      case VbScfOptimizerBackend::Lbfgspp: {
        const auto backend_result =
            optimizer_detail::run_full_space_lbfgs_backend(
                &objective,
                options_,
                parameter_vector,
                gradient,
                energy,
                &result);
        n_iterations = backend_result.n_iterations;
        final_gradient_l2_norm = backend_result.final_gradient_l2_norm;
        break;
      }

      case VbScfOptimizerBackend::NonredundantProjectedGradient: {
        const auto backend_result =
            optimizer_detail::run_projected_gradient_backend(
                &objective,
                parameter_view,
                options_,
                parameter_vector,
                gradient,
                energy,
                &result);
        n_iterations = backend_result.n_iterations;
        final_gradient_l2_norm = backend_result.final_gradient_l2_norm;
        break;
      }

      case VbScfOptimizerBackend::NonredundantLbfgspp: {
        const int history_size = options_.history_size;
        LBFGSpp::BFGSMat<double> inverse_hessian;
        inverse_hessian.reset(n, history_size);

        Eigen::VectorXd current_parameters = parameter_vector;
        Eigen::VectorXd current_gradient = gradient;
        OrbitalChart current_space =
            build_orbital_chart(objective, parameter_view);
        auto current_projection =
            current_space.project_gradient(current_gradient);
        Eigen::VectorXd previous_parameters(n);
        Eigen::VectorXd previous_gradient(n);
        Eigen::VectorXd previous_projected_gradient(n);
        constexpr double kCurvatureEpsilon = std::numeric_limits<double>::epsilon();

        for (int iteration = 0; iteration < options_.max_iterations; ++iteration) {
          if (current_space.reduced_size() == 0) {
            result.termination_reason = "nonredundant_space_empty";
            break;
          }

          const double reduced_gradient_inf_norm =
              gradient_infinity_norm(current_projection.reduced_gradient);
          if (iteration == 0 &&
              reduced_gradient_inf_norm < options_.gradient_tolerance) {
            result.converged = true;
            result.termination_reason =
                "nonredundant_lbfgspp_initial_tolerance";
            final_gradient_l2_norm = current_projection.reduced_gradient.norm();
            break;
          }

          const Eigen::VectorXd fallback_reduced_direction =
              -current_projection.reduced_gradient;
          const OrbitalPreparationInput previous_orbital_input =
              objective.last_input().orbital_preparation_input;
          const Eigen::VectorXd fallback_direction =
              gather_nonredundant_retract_tangent(
                  previous_orbital_input,
                  current_space,
                  parameter_view,
                  fallback_reduced_direction);
          Eigen::VectorXd search_direction;
          Eigen::VectorXd reduced_search_direction;
          inverse_hessian.apply_Hv(
              current_projection.packed_projected_gradient,
              -1.0,
              search_direction);
          {
            const auto search_projection =
                current_space.project_vector(search_direction);
            reduced_search_direction = search_projection.reduced_gradient;
            search_direction =
                gather_nonredundant_retract_tangent(
                    previous_orbital_input,
                    current_space,
                    parameter_view,
                    reduced_search_direction);
          }
          double directional_derivative =
              current_gradient.dot(search_direction);
          if (!std::isfinite(directional_derivative) ||
              directional_derivative >= 0.0 ||
              is_effectively_zero_step(search_direction, current_parameters)) {
            inverse_hessian.reset(n, history_size);
            search_direction = fallback_direction;
            reduced_search_direction = fallback_reduced_direction;
            directional_derivative = current_gradient.dot(search_direction);
          }
          if (!std::isfinite(directional_derivative) ||
              directional_derivative >= 0.0) {
            result.termination_reason =
                "nonredundant_lbfgspp_non_descent_direction";
            final_gradient_l2_norm = current_projection.reduced_gradient.norm();
            break;
          }

          previous_parameters = current_parameters;
          previous_gradient = current_gradient;
          previous_projected_gradient = current_projection.packed_projected_gradient;
          const Eigen::VectorXd previous_reduced_gradient =
              current_projection.reduced_gradient;
          const double reference_energy = energy;
          Eigen::VectorXd accepted_parameters(current_parameters.size());
          Eigen::VectorXd accepted_gradient(current_gradient.size());
          double accepted_energy = energy;
          if (!try_armijo_backtracking_nonredundant_direction(
                  &objective,
                  previous_orbital_input,
                  current_space,
                  parameter_view,
                  current_parameters,
                  energy,
                  current_gradient,
                  reduced_search_direction,
                  search_direction,
                  std::min(1.0, options_.initial_step_size),
                  options_.minimum_step_size,
                  options_.armijo_constant,
                  &accepted_parameters,
                  &accepted_gradient,
                  &accepted_energy)) {
            result.termination_reason =
                "nonredundant_lbfgspp_line_search_failed";
            final_gradient_l2_norm = current_projection.reduced_gradient.norm();
            break;
          }

          current_parameters = std::move(accepted_parameters);
          current_gradient = std::move(accepted_gradient);
          energy = accepted_energy;

          OrbitalChart next_space =
              build_orbital_chart(objective, parameter_view);
          auto next_projection =
              next_space.project_gradient(current_gradient);
          double next_reduced_gradient_inf_norm =
              gradient_infinity_norm(next_projection.reduced_gradient);

          Eigen::VectorXd parameter_step = current_parameters - previous_parameters;
          bool recovered_from_stall = false;
          const bool stalled_line_search =
              is_effectively_zero_step(parameter_step, previous_parameters) ||
              line_search_made_no_meaningful_progress(
                  reference_energy,
                  energy,
                  reduced_gradient_inf_norm,
                  next_reduced_gradient_inf_norm,
                  options_.gradient_tolerance);
          if (stalled_line_search &&
              reduced_gradient_inf_norm >= options_.gradient_tolerance) {
            const double fallback_initial_step =
                std::max(options_.minimum_step_size,
                         std::min(
                             std::min(1.0, options_.initial_step_size),
                             1.0 / std::max(1.0, reduced_gradient_inf_norm)));
            if (!try_armijo_backtracking_nonredundant_direction(
                    &objective,
                    previous_orbital_input,
                    current_space,
                    parameter_view,
                    previous_parameters,
                    reference_energy,
                    previous_gradient,
                    -previous_reduced_gradient,
                    gather_nonredundant_retract_tangent(
                        previous_orbital_input,
                        current_space,
                        parameter_view,
                        -previous_reduced_gradient),
                    fallback_initial_step,
                    options_.minimum_step_size,
                    options_.armijo_constant,
                    &current_parameters,
                    &current_gradient,
                    &energy)) {
              energy = objective(previous_parameters, current_gradient);
              current_parameters = previous_parameters;
              sync_result_from_objective(objective, &result);
              final_gradient_l2_norm = current_gradient.norm();
              result.termination_reason = "nonredundant_lbfgspp_line_search_stalled";
              break;
            }
            next_space = build_orbital_chart(objective, parameter_view);
            next_projection =
                next_space.project_gradient(current_gradient);
            next_reduced_gradient_inf_norm =
                gradient_infinity_norm(next_projection.reduced_gradient);
            parameter_step = current_parameters - previous_parameters;
            inverse_hessian.reset(n, history_size);
            recovered_from_stall = true;
          }

          const bool accepted_point_chart_reset =
              objective.canonicalize_orbital_chart_at_current_point(
                  &current_parameters,
                  &current_gradient);
          if (accepted_point_chart_reset) {
            next_space = build_orbital_chart(objective, parameter_view);
            next_projection =
                next_space.project_gradient(current_gradient);
            next_reduced_gradient_inf_norm =
                gradient_infinity_norm(next_projection.reduced_gradient);
          }

          ++n_iterations;
          sync_result_from_objective(objective, &result);
          record_accepted_iteration_snapshot(&objective, n_iterations, options_, &result);
          final_gradient_l2_norm = current_gradient.norm();
          const double de = energy - previous_energy;
          previous_energy = energy;
          if (std::abs(de) < options_.energy_tolerance &&
              next_reduced_gradient_inf_norm <
                  options_.gradient_tolerance) {
            result.converged = true;
            result.termination_reason =
                "nonredundant_lbfgspp_dual_tolerance";
            final_gradient_l2_norm = next_projection.reduced_gradient.norm();
            break;
          }

          if (accepted_point_chart_reset) {
            inverse_hessian.reset(n, history_size);
          } else {
            parameter_step = current_parameters - previous_parameters;
            const Eigen::VectorXd projected_gradient_step =
                next_projection.packed_projected_gradient -
                previous_projected_gradient;
            if (parameter_step.dot(projected_gradient_step) >
                kCurvatureEpsilon * projected_gradient_step.squaredNorm()) {
              inverse_hessian.add_correction(
                  parameter_step,
                  projected_gradient_step);
            } else if (recovered_from_stall) {
              inverse_hessian.reset(n, history_size);
            }
          }

          current_space = std::move(next_space);
          current_projection = std::move(next_projection);
        }

        break;
      }

      case VbScfOptimizerBackend::NonredundantTruncatedNewton: {
        Eigen::VectorXd current_parameters = parameter_vector;
        Eigen::VectorXd current_gradient = gradient;
        OrbitalChart current_space =
            build_orbital_chart(objective, parameter_view);
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
                options_.nonredundant_truncated_newton_transport_history_size));
        double trust_radius =
            std::max(options_.minimum_step_size, options_.initial_step_size);
        int rejected_trial_step_count_for_current_point = 0;
        RejectedTruncatedNewtonStepCache rejected_step_cache;
        TruncatedNewtonKrylovSubspace cached_krylov_subspace;

        while (n_iterations < options_.max_iterations) {
          if (current_space.reduced_size() == 0) {
            result.termination_reason = "nonredundant_space_empty";
            final_gradient_l2_norm = 0.0;
            break;
          }

          const double reduced_gradient_inf_norm =
              gradient_infinity_norm(current_projection.reduced_gradient);
          final_projected_gradient_inf_norm = reduced_gradient_inf_norm;
          final_projected_gradient_l2_norm =
              current_projection.reduced_gradient.norm();
          if (n_iterations == 0 &&
              reduced_gradient_inf_norm < options_.gradient_tolerance) {
            result.converged = true;
            result.termination_reason =
                "nonredundant_truncated_newton_initial_tolerance";
            final_gradient_l2_norm = current_projection.reduced_gradient.norm();
            break;
          }

          if (!(trust_radius > 0.0) || !std::isfinite(trust_radius)) {
            result.termination_reason =
                "nonredundant_truncated_newton_invalid_trust_radius";
            final_gradient_l2_norm = current_projection.reduced_gradient.norm();
            break;
          }

          std::unique_ptr<ReducedHvpOperator> hvp_operator;
          switch (options_.nonredundant_truncated_newton_hvp_mode) {
            case NonredundantTruncatedNewtonHvpMode::FullFiniteDifference:
              hvp_operator = std::make_unique<FullFiniteDifferenceReducedHvpOperator>(
                  objective,
                  current_space,
                  current_projection,
                  objective.last_input().orbital_preparation_input,
                  parameter_view,
                  options_.nonredundant_truncated_newton_hvp_step_size);
              break;
            case NonredundantTruncatedNewtonHvpMode::ExactContextDirectAction: {
              auto exact_ctx_hvp_operator =
                  std::make_unique<ExactContextReducedHvpOperator>(
                      objective,
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
              choose_truncated_newton_transport_history_size(options_);
          const auto transported_preconditioner =
              build_nonredundant_truncated_newton_preconditioner(
                  current_space,
                  packed_secant_history,
                  transport_history_size);
          const int max_cg_iterations =
              choose_truncated_newton_max_cg_iterations(
                  options_,
                  current_projection.reduced_gradient.size());
          const OrbitalPreparationInput current_orbital_input =
              objective.last_input().orbital_preparation_input;
          const NonredundantRetractionMetric retraction_metric(
              current_orbital_input,
              current_space,
              parameter_view);
          const auto admit_energy_only_trial_screen = [&]() {
            const auto& objective_time_history =
                objective.iteration_time_history_seconds();
            const double last_objective_seconds =
                objective_time_history.empty()
                    ? 0.0
                    : objective_time_history.back();
            // Measure the two available trial-evaluation paths directly.
            // After one sample, use energy-only screening exactly when its
            // observed cost is below a full objective-and-gradient call.
            if (objective.energy_only_call_count() == 0) {
              return true;
            }
            const double last_energy_only_seconds =
                objective.last_energy_only_wall_time_seconds();
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
                      objective.evaluate_energy_only(
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
                    objective.evaluate_trial_without_committing(
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
          auto try_nonredundant_descent_fallback_step =
              [&](Eigen::VectorXd* accepted_packed_step,
                  VbScfObjective* accepted_trial_objective,
                  Eigen::VectorXd* accepted_trial_parameters,
                  Eigen::VectorXd* accepted_trial_gradient,
                  double* accepted_trial_energy) -> bool {
                Eigen::VectorXd fallback_reduced_direction =
                    -apply_nonredundant_truncated_newton_preconditioner(
                        current_space,
                        &transported_preconditioner,
                        current_projection.reduced_gradient);
                Eigen::VectorXd search_direction =
                    gather_nonredundant_retract_tangent(
                        current_orbital_input,
                        current_space,
                        parameter_view,
                        fallback_reduced_direction);
                double directional_derivative =
                    current_gradient.dot(search_direction);
                if (!std::isfinite(directional_derivative) ||
                    directional_derivative >= 0.0 ||
                    is_effectively_zero_step(
                        search_direction,
                        current_parameters)) {
                  fallback_reduced_direction =
                      -current_projection.reduced_gradient;
                  search_direction =
                      gather_nonredundant_retract_tangent(
                          current_orbital_input,
                          current_space,
                          parameter_view,
                          fallback_reduced_direction);
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
                        : options_.minimum_step_size;
                const double initial_fallback_step =
                    std::max(
                        options_.minimum_step_size,
                        std::min(
                            std::min(
                                std::min(1.0, options_.initial_step_size),
                                1.0 / std::max(1.0, reduced_gradient_inf_norm)),
                            trust_radius_limited_initial_step));
                // The descent fallback is entered only after the current
                // accepted-point Newton model already failed to produce an
                // acceptable trust-region step. Starting the Armijo backtrack
                // from a reduced step that already fits inside the current
                // trust radius avoids burning many full objective evaluations
                // just to rediscover the same radius contraction.
                VbScfObjective fallback_objective =
                    objective.make_probe_copy();
                Eigen::VectorXd fallback_parameters(current_parameters.size());
                Eigen::VectorXd fallback_gradient(current_gradient.size());
                double fallback_energy = energy;
                if (!try_armijo_backtracking_nonredundant_direction(
                        &fallback_objective,
                        current_orbital_input,
                        current_space,
                        parameter_view,
                        current_parameters,
                        energy,
                        current_gradient,
                        fallback_reduced_direction,
                        search_direction,
                        initial_fallback_step,
                        options_.minimum_step_size,
                        options_.armijo_constant,
                        &fallback_parameters,
                        &fallback_gradient,
                        &fallback_energy)) {
                  return false;
                }

                *accepted_packed_step =
                    fallback_parameters - current_parameters;
                *accepted_trial_objective =
                    std::move(fallback_objective);
                *accepted_trial_parameters = std::move(fallback_parameters);
                *accepted_trial_gradient = std::move(fallback_gradient);
                *accepted_trial_energy = fallback_energy;
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
            ++result.matrix_free_subproblem_count;
            if (truncated_newton_step.reduced_hessian_times_step.size() ==
                current_projection.reduced_gradient.size()) {
              const double gradient_norm = current_projection.reduced_gradient.stableNorm();
              const Eigen::VectorXd kkt_residual = current_projection.reduced_gradient +
                  truncated_newton_step.reduced_hessian_times_step +
                  truncated_newton_step.trust_region_shift * truncated_newton_step.reduced_step;
              if (kkt_residual.stableNorm() <=
                  inexact_newton_forcing_term(gradient_norm) * gradient_norm) {
                ++result.matrix_free_residual_converged_count;
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
            result.termination_reason =
                "nonredundant_truncated_newton_invalid_step_dimension";
            final_gradient_l2_norm = current_projection.reduced_gradient.norm();
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
            result.matrix_free_hvp_direction_count +=
                hvp_diagnostics.apply_count;
            result.matrix_free_hvp_batch_count +=
                hvp_diagnostics.batch_apply_count;
            result.matrix_free_hvp_wall_time_seconds +=
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
                  8.0 * options_.gradient_tolerance &&
              model_step.encountered_negative_curvature) {
            if (try_nonredundant_descent_fallback_step(
                    &packed_step,
                    &objective,
                    &trial_parameters,
                    &trial_gradient,
                    &trial_energy)) {
              accepted_trial = true;
              const double fallback_actual_decrease = energy - trial_energy;
              trial_evaluation_cache.actual_decrease =
                  fallback_actual_decrease;
              trial_evaluation_cache.predicted_decrease =
                  fallback_actual_decrease;
              truncated_newton_step.used_krylov_rescue = true;
              truncated_newton_step.reached_boundary = false;
              truncated_newton_step.encountered_negative_curvature = false;
              truncated_newton_step.cg_iterations = 0;
              truncated_newton_step.reduced_step =
                  current_space.project_vector(packed_step).reduced_gradient;
              truncated_newton_step.retract_tangent_norm = packed_step.norm();
              truncated_newton_step.predicted_decrease =
                  std::max(options_.energy_tolerance, energy - trial_energy);
            }
          }
          if (!accepted_trial) {
            ++rejected_trial_step_count_for_current_point;
            trust_radius =
                update_nonredundant_truncated_newton_trust_radius(
                    trust_radius,
                    options_.minimum_step_size,
                    trial_evaluation_cache,
                    trial_step_for_current_trial,
                    false);
            if (trust_radius <= options_.minimum_step_size) {
              result.termination_reason =
                  "nonredundant_truncated_newton_trust_radius_exhausted";
              final_gradient_l2_norm = current_projection.reduced_gradient.norm();
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
            objective.commit_trial_evaluation(
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
              objective.canonicalize_orbital_chart_at_current_point(
                  &current_parameters,
                  &current_gradient,
                  &packed_secant_history);
          ++n_iterations;
          rejected_trial_step_count_for_current_point = 0;
          rejected_step_cache.clear();
          cached_krylov_subspace = TruncatedNewtonKrylovSubspace();
          sync_result_from_objective(objective, &result);
          record_accepted_iteration_snapshot(&objective, n_iterations, options_, &result);
          final_gradient_l2_norm = current_gradient.norm();
          const double de = energy - previous_energy;
          previous_energy = energy;
          OrbitalChart next_space =
              build_orbital_chart(objective, parameter_view);
          auto next_projection =
              next_space.project_gradient(current_gradient);
          const bool nonredundant_rank_changed =
              current_space.reduced_size() != next_space.reduced_size() ||
              current_space.rank_signature() != next_space.rank_signature();
          final_projected_gradient_inf_norm =
              gradient_infinity_norm(next_projection.reduced_gradient);
          final_projected_gradient_l2_norm =
              next_projection.reduced_gradient.norm();
          trust_radius =
              update_nonredundant_truncated_newton_trust_radius(
                  trust_radius,
                  options_.minimum_step_size,
                  trial_evaluation_cache,
                  truncated_newton_step,
                  true);
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
          if (std::abs(de) < options_.energy_tolerance &&
              gradient_infinity_norm(next_projection.reduced_gradient) <
                  options_.gradient_tolerance) {
            result.converged = true;
            result.termination_reason =
                "nonredundant_truncated_newton_dual_tolerance";
            final_gradient_l2_norm = next_projection.reduced_gradient.norm();
            break;
          }

          current_space = std::move(next_space);
          current_projection = std::move(next_projection);
        }
        result.final_projected_gradient_inf_norm =
            final_projected_gradient_inf_norm;
        result.final_projected_gradient_l2_norm =
            final_projected_gradient_l2_norm;
        final_projected_gradient_ready = true;
        break;
      }

    }
  } catch (const std::exception& error) {
    result.termination_reason = error.what();
  }

  if (result.total_energy_history.empty()) {
    if (!result.termination_reason.empty()) {
      throw std::runtime_error(
          std::string("optimizer did not evaluate the objective: ") +
          result.termination_reason);
    }
    throw std::runtime_error("optimizer did not evaluate the objective");
  }

  result.n_iterations = n_iterations;
  result.final_total_energy = result.total_energy_history.back();
  result.final_one_electron_reference_energy =
      result.scf_result.one_electron_reference_energy;
  result.final_gradient_inf_norm = result.gradient_inf_norm_history.back();
  result.final_gradient_l2_norm = final_gradient_l2_norm;
  if (uses_nonredundant_space(options_.backend) &&
      !final_projected_gradient_ready) {
    const OrbitalChart final_space =
        build_orbital_chart(objective, parameter_view);
    const Eigen::VectorXd final_packed_gradient =
        parameter_view.gather_from_full(
            objective.last_gradient_result().sparse_orbital_energy_gradient);
    const auto final_projection =
        final_space.project_gradient(final_packed_gradient);
    result.final_projected_gradient_inf_norm =
        gradient_infinity_norm(final_projection.reduced_gradient);
    result.final_projected_gradient_l2_norm =
        final_projection.reduced_gradient.norm();
  }
  result.optimized_input = objective.last_input();
  Eigen::MatrixXd final_normalized_orbital_matrix =
      objective.last_gradient_result()
          .orbital_preparation_result
          .physical_orbital_frame
          .normalized_orbital_matrix;
  if (final_normalized_orbital_matrix.size() != 0) {
    // The evaluator always works with the normalized physical orbital frame,
    // mirroring legacy `normalize(...)`. Store that same frame in the final
    // sparse slots before exporting so Molden / restart artifacts see the
    // actual accepted physical orbitals rather than a pre-normalization raw
    // parameter vector.
    //
    // The OEO representative is an export gauge, not an optimizer mutation.
    // Rebuild it once from the converged auxiliary block for Molden/restart
    // output without perturbing the accepted-point tangent chart.
    final_normalized_orbital_matrix =
        build_metric_preserving_oeo_repaired_normalized_orbital_matrix(
            result.optimized_input.orbital_preparation_input,
            objective.last_gradient_result().orbital_preparation_result,
            initial_normalized_orbital_matrix);
    overwrite_sparse_orbitals_from_dense_physical_frame(
        final_normalized_orbital_matrix,
        &result.optimized_input.orbital_preparation_input);
  }
  enforce_strict_sparse_orbital_support(
      &result.optimized_input.orbital_preparation_input);

  if (result.termination_reason.empty()) {
    if (result.n_iterations >= options_.max_iterations) {
      result.termination_reason = "max_iterations";
    } else {
      result.termination_reason = "stopped";
    }
  }

  const auto optimization_end_time = std::chrono::steady_clock::now();
  const std::chrono::duration<double> total_elapsed_seconds =
      optimization_end_time - optimization_start_time;
  result.total_wall_time_seconds = total_elapsed_seconds.count();

  return result;
}

}  // namespace xmvb::vb
