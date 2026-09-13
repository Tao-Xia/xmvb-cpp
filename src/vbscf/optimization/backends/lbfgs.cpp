#include "vbscf/optimization/backends/lbfgs.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

#include <LBFGS.h>

#include "vbscf/optimization/globalization/line_search.hpp"
#include "vbscf/optimization/driver/checks.hpp"
#include "vbscf/optimization/driver/session.hpp"
#include "vbscf/optimization/trust_region/retraction.hpp"
#include "vbscf/optimization/objective/function.hpp"
#include "vbscf/optimization/driver/options.hpp"
#include "vbscf/optimization/driver/result.hpp"
#include "vbscf/orbitals/charts/layout.hpp"

namespace xmvb::vb::optimizer_detail {

BackendRunResult run_full_space_lbfgs_backend(
    VbScfObjective* objective,
    const VbScfOptimizerOptions& options,
    const Eigen::VectorXd& initial_parameters,
    const Eigen::VectorXd& initial_gradient,
    double initial_energy,
    VbScfOptimizerResult* result) {
  constexpr double kLineSearchExpansionFactor = 10.0;
  constexpr double kCurvatureEpsilon = std::numeric_limits<double>::epsilon();
  constexpr double kTinyStepFactor = 10.0;
  constexpr double kRobustStepShrinkRatio = 0.1;
  constexpr double kSuspiciousPrimaryStepRatio = 0.1;

  BackendRunResult run_result;
  const int dimension = static_cast<int>(initial_parameters.size());
  LBFGSpp::LBFGSParam<double> parameters;
  parameters.m = options.history_size;
  parameters.epsilon = 0.0;
  parameters.epsilon_rel = 0.0;
  parameters.past = 0;
  parameters.delta = 0.0;
  parameters.max_iterations = 0;
  parameters.max_linesearch = 20;
  parameters.min_step = options.minimum_step_size;
  parameters.max_step =
      std::max(
          options.initial_step_size,
          options.initial_step_size * kLineSearchExpansionFactor);
  parameters.ftol = options.armijo_constant;
  parameters.wolfe = 0.9;
  parameters.linesearch =
      LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE;
  parameters.check_param();

  LBFGSpp::BFGSMat<double> inverse_hessian;
  inverse_hessian.reset(dimension, parameters.m);

  Eigen::VectorXd current_parameters = initial_parameters;
  Eigen::VectorXd current_gradient = initial_gradient;
  Eigen::VectorXd previous_parameters(dimension);
  Eigen::VectorXd previous_gradient(dimension);
  Eigen::VectorXd search_direction = -current_gradient;
  double energy = initial_energy;
  double previous_energy = initial_energy;
  double last_robust_step = std::min(1.0, options.initial_step_size);
  bool has_robust_step_history = false;
  bool last_iteration_used_steepest_descent = false;

  for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
    if (search_direction.dot(current_gradient) >= 0.0) {
      search_direction = -current_gradient;
    }
    double directional_derivative = current_gradient.dot(search_direction);
    if (directional_derivative >= 0.0) {
      result->termination_reason = "lbfgspp_non_descent_direction";
      break;
    }

    previous_parameters = current_parameters;
    previous_gradient = current_gradient;
    const double reference_energy = energy;
    const double previous_gradient_inf_norm =
        gradient_infinity_norm(previous_gradient);
    double step = std::min(1.0, options.initial_step_size);
    if (last_iteration_used_steepest_descent && has_robust_step_history) {
      step = std::max(
          parameters.min_step,
          std::min(last_robust_step, parameters.max_step));
    }
    bool used_steepest_descent = false;
    bool reset_inverse_hessian = false;
    std::string primary_line_search_error;

    try {
      LBFGSpp::LineSearchMoreThuente<double>::LineSearch(
          *objective,
          parameters,
          previous_parameters,
          search_direction,
          parameters.max_step,
          step,
          energy,
          current_gradient,
          directional_derivative,
          current_parameters);
    } catch (const std::exception& error) {
      primary_line_search_error = error.what();
    }

    Eigen::VectorXd parameter_step = current_parameters - previous_parameters;
    const double primary_gradient_inf_norm =
        gradient_infinity_norm(current_gradient);
    const bool stalled_line_search =
        primary_line_search_error.empty() &&
        (is_effectively_zero_step(parameter_step, previous_parameters) ||
         line_search_made_no_meaningful_progress(
             reference_energy,
             energy,
             previous_gradient_inf_norm,
             primary_gradient_inf_norm,
             options.gradient_tolerance));
    const bool suspicious_primary_step =
        primary_line_search_error.empty() &&
        has_robust_step_history &&
        step < kSuspiciousPrimaryStepRatio * last_robust_step;
    const bool should_try_steepest_descent =
        !primary_line_search_error.empty() ||
        (previous_gradient_inf_norm >= options.gradient_tolerance &&
         (stalled_line_search ||
          step <= kTinyStepFactor * parameters.min_step ||
          suspicious_primary_step));
    if (should_try_steepest_descent) {
      const Eigen::VectorXd primary_parameters = current_parameters;
      const Eigen::VectorXd primary_gradient = current_gradient;
      const double primary_energy = energy;
      double steepest_descent_step =
          has_robust_step_history
              ? last_robust_step
              : std::max(
                    parameters.min_step,
                    std::min(
                        std::min(1.0, options.initial_step_size),
                        parameters.max_step));
      Eigen::VectorXd steepest_descent_parameters;
      Eigen::VectorXd steepest_descent_gradient;
      double steepest_descent_energy = reference_energy;
      if (try_steepest_descent_armijo_step(
              objective,
              parameters,
              previous_parameters,
              previous_gradient,
              reference_energy,
              steepest_descent_step,
              &steepest_descent_parameters,
              &steepest_descent_gradient,
              &steepest_descent_energy,
              &steepest_descent_step)) {
        const bool should_replace_primary =
            !primary_line_search_error.empty() ||
            stalled_line_search ||
            step <= kTinyStepFactor * parameters.min_step ||
            suspicious_primary_step ||
            steepest_descent_energy < primary_energy;
        if (should_replace_primary) {
          current_parameters = std::move(steepest_descent_parameters);
          current_gradient = std::move(steepest_descent_gradient);
          energy = steepest_descent_energy;
          step = steepest_descent_step;
          parameter_step = current_parameters - previous_parameters;
          used_steepest_descent = true;
          reset_inverse_hessian = true;
        } else {
          current_parameters = primary_parameters;
          current_gradient = primary_gradient;
          energy = primary_energy;
        }
      } else if (!primary_line_search_error.empty() || stalled_line_search) {
        energy = (*objective)(previous_parameters, current_gradient);
        current_parameters = previous_parameters;
        sync_result_from_objective(*objective, result);
        run_result.final_gradient_l2_norm = current_gradient.norm();
        result->termination_reason =
            !primary_line_search_error.empty()
                ? std::string("lbfgspp_line_search: ") +
                      primary_line_search_error
                : "lbfgspp_line_search_stalled";
        break;
      }
    }

    if (step > kTinyStepFactor * parameters.min_step &&
        (!has_robust_step_history ||
         step >= kRobustStepShrinkRatio * last_robust_step)) {
      last_robust_step = step;
      has_robust_step_history = true;
    }
    last_iteration_used_steepest_descent = used_steepest_descent;

    const bool accepted_point_chart_reset =
        objective->canonicalize_chart(
            &current_parameters,
            &current_gradient);
    if (accepted_point_chart_reset) {
      reset_inverse_hessian = true;
    }

    ++run_result.n_iterations;
    sync_result_from_objective(*objective, result);
    record_accepted_iteration_snapshot(
        objective,
        run_result.n_iterations,
        options,
        result);
    run_result.final_gradient_l2_norm = current_gradient.norm();
    const double energy_change = energy - previous_energy;
    previous_energy = energy;
    if (std::abs(energy_change) < options.energy_tolerance &&
        run_result.final_gradient_l2_norm < options.gradient_tolerance) {
      result->converged = true;
      result->termination_reason = "lbfgspp_dual_tolerance";
      break;
    }

    Eigen::VectorXd gradient_step = current_gradient - previous_gradient;
    if (reset_inverse_hessian) {
      inverse_hessian.reset(dimension, parameters.m);
    }
    if (!reset_inverse_hessian &&
        parameter_step.dot(gradient_step) >
            kCurvatureEpsilon * gradient_step.squaredNorm()) {
      inverse_hessian.add_correction(parameter_step, gradient_step);
    }
    inverse_hessian.apply_Hv(current_gradient, -1.0, search_direction);
    if (used_steepest_descent && search_direction.dot(current_gradient) >= 0.0) {
      search_direction = -current_gradient;
    }
  }
  return run_result;
}

BackendRunResult run_nonredundant_lbfgs_backend(
    VbScfObjective* objective,
    const SparseParameterLayout& parameter_view,
    const VbScfOptimizerOptions& options,
    const Eigen::VectorXd& initial_parameters,
    const Eigen::VectorXd& initial_gradient,
    double initial_energy,
    VbScfOptimizerResult* result) {
  constexpr double kCurvatureEpsilon = std::numeric_limits<double>::epsilon();
  BackendRunResult run_result;
  const int dimension = static_cast<int>(initial_parameters.size());
  const int history_size = options.history_size;
  LBFGSpp::BFGSMat<double> inverse_hessian;
  inverse_hessian.reset(dimension, history_size);

  Eigen::VectorXd current_parameters = initial_parameters;
  Eigen::VectorXd current_gradient = initial_gradient;
  double energy = initial_energy;
  double previous_energy = initial_energy;
  OrbitalChart current_space = build_orbital_chart(*objective, parameter_view);
  auto current_projection = current_space.project_gradient(current_gradient);
  Eigen::VectorXd previous_parameters(dimension);
  Eigen::VectorXd previous_gradient(dimension);
  Eigen::VectorXd previous_projected_gradient(dimension);

  for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
    if (current_space.reduced_size() == 0) {
      result->termination_reason = "nonredundant_space_empty";
      break;
    }

    const double reduced_gradient_inf_norm =
        gradient_infinity_norm(current_projection.reduced_gradient);
    if (iteration == 0 &&
        reduced_gradient_inf_norm < options.gradient_tolerance) {
      result->converged = true;
      result->termination_reason = "nonredundant_lbfgspp_initial_tolerance";
      run_result.final_gradient_l2_norm =
          current_projection.reduced_gradient.norm();
      break;
    }

    const Eigen::VectorXd steepest_descent_reduced_direction =
        -current_projection.reduced_gradient;
    const OrbitalPreparationInput previous_orbital_input =
        objective->input().orbital_preparation_input;
    const Eigen::VectorXd steepest_descent_direction =
        gather_nonredundant_retract_tangent(
            previous_orbital_input,
            current_space,
            parameter_view,
            steepest_descent_reduced_direction);
    Eigen::VectorXd search_direction;
    inverse_hessian.apply_Hv(
        current_projection.packed_projected_gradient,
        -1.0,
        search_direction);
    const auto search_projection =
        current_space.project_vector(search_direction);
    Eigen::VectorXd reduced_search_direction =
        search_projection.reduced_gradient;
    search_direction = gather_nonredundant_retract_tangent(
        previous_orbital_input,
        current_space,
        parameter_view,
        reduced_search_direction);
    double directional_derivative = current_gradient.dot(search_direction);
    if (!std::isfinite(directional_derivative) ||
        directional_derivative >= 0.0 ||
        is_effectively_zero_step(search_direction, current_parameters)) {
      inverse_hessian.reset(dimension, history_size);
      search_direction = steepest_descent_direction;
      reduced_search_direction = steepest_descent_reduced_direction;
      directional_derivative = current_gradient.dot(search_direction);
    }
    if (!std::isfinite(directional_derivative) ||
        directional_derivative >= 0.0) {
      result->termination_reason =
          "nonredundant_lbfgspp_non_descent_direction";
      run_result.final_gradient_l2_norm =
          current_projection.reduced_gradient.norm();
      break;
    }

    previous_parameters = current_parameters;
    previous_gradient = current_gradient;
    previous_projected_gradient =
        current_projection.packed_projected_gradient;
    const Eigen::VectorXd previous_reduced_gradient =
        current_projection.reduced_gradient;
    const double reference_energy = energy;
    Eigen::VectorXd accepted_parameters(current_parameters.size());
    Eigen::VectorXd accepted_gradient(current_gradient.size());
    double accepted_energy = energy;
    if (!try_armijo_backtracking_nonredundant_direction(
            objective,
            previous_orbital_input,
            current_space,
            parameter_view,
            current_parameters,
            energy,
            current_gradient,
            reduced_search_direction,
            search_direction,
            std::min(1.0, options.initial_step_size),
            options.minimum_step_size,
            options.armijo_constant,
            &accepted_parameters,
            &accepted_gradient,
            &accepted_energy)) {
      result->termination_reason = "nonredundant_lbfgspp_line_search_failed";
      run_result.final_gradient_l2_norm =
          current_projection.reduced_gradient.norm();
      break;
    }

    current_parameters = std::move(accepted_parameters);
    current_gradient = std::move(accepted_gradient);
    energy = accepted_energy;

    OrbitalChart next_space = build_orbital_chart(*objective, parameter_view);
    auto next_projection = next_space.project_gradient(current_gradient);
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
            options.gradient_tolerance);
    if (stalled_line_search &&
        reduced_gradient_inf_norm >= options.gradient_tolerance) {
      const double steepest_descent_initial_step =
          std::max(
              options.minimum_step_size,
              std::min(
                  std::min(1.0, options.initial_step_size),
                  1.0 / std::max(1.0, reduced_gradient_inf_norm)));
      if (!try_armijo_backtracking_nonredundant_direction(
              objective,
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
              steepest_descent_initial_step,
              options.minimum_step_size,
              options.armijo_constant,
              &current_parameters,
              &current_gradient,
              &energy)) {
        energy = (*objective)(previous_parameters, current_gradient);
        current_parameters = previous_parameters;
        sync_result_from_objective(*objective, result);
        run_result.final_gradient_l2_norm = current_gradient.norm();
        result->termination_reason =
            "nonredundant_lbfgspp_line_search_stalled";
        break;
      }
      next_space = build_orbital_chart(*objective, parameter_view);
      next_projection = next_space.project_gradient(current_gradient);
      next_reduced_gradient_inf_norm =
          gradient_infinity_norm(next_projection.reduced_gradient);
      parameter_step = current_parameters - previous_parameters;
      inverse_hessian.reset(dimension, history_size);
      recovered_from_stall = true;
    }

    const bool accepted_point_chart_reset =
        objective->canonicalize_chart(
            &current_parameters,
            &current_gradient);
    if (accepted_point_chart_reset) {
      next_space = build_orbital_chart(*objective, parameter_view);
      next_projection = next_space.project_gradient(current_gradient);
      next_reduced_gradient_inf_norm =
          gradient_infinity_norm(next_projection.reduced_gradient);
    }

    ++run_result.n_iterations;
    sync_result_from_objective(*objective, result);
    record_accepted_iteration_snapshot(
        objective,
        run_result.n_iterations,
        options,
        result);
    run_result.final_gradient_l2_norm = current_gradient.norm();
    const double energy_change = energy - previous_energy;
    previous_energy = energy;
    if (std::abs(energy_change) < options.energy_tolerance &&
        next_reduced_gradient_inf_norm < options.gradient_tolerance) {
      result->converged = true;
      result->termination_reason = "nonredundant_lbfgspp_dual_tolerance";
      run_result.final_gradient_l2_norm = next_projection.reduced_gradient.norm();
      break;
    }

    if (accepted_point_chart_reset) {
      inverse_hessian.reset(dimension, history_size);
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
        inverse_hessian.reset(dimension, history_size);
      }
    }

    current_space = std::move(next_space);
    current_projection = std::move(next_projection);
  }
  return run_result;
}

}  // namespace xmvb::vb::optimizer_detail
