#include "vbscf/optimization/backends/lbfgs_backends.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

#include <LBFGS.h>

#include "vbscf/optimization/line_search.hpp"
#include "vbscf/optimization/optimization_checks.hpp"
#include "vbscf/optimization/optimizer_session.hpp"
#include "vbscf/optimization/vbscf_objective.hpp"
#include "vbscf/optimization/vbscf_optimizer_options.hpp"
#include "vbscf/optimization/vbscf_optimizer_result.hpp"

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
  bool last_iteration_used_fallback = false;

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
    if (last_iteration_used_fallback && has_robust_step_history) {
      step = std::max(
          parameters.min_step,
          std::min(last_robust_step, parameters.max_step));
    }
    bool used_fallback = false;
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
    const bool should_try_fallback =
        !primary_line_search_error.empty() ||
        (previous_gradient_inf_norm >= options.gradient_tolerance &&
         (stalled_line_search ||
          step <= kTinyStepFactor * parameters.min_step ||
          suspicious_primary_step));
    if (should_try_fallback) {
      const Eigen::VectorXd primary_parameters = current_parameters;
      const Eigen::VectorXd primary_gradient = current_gradient;
      const double primary_energy = energy;
      double fallback_step =
          has_robust_step_history
              ? last_robust_step
              : std::max(
                    parameters.min_step,
                    std::min(
                        std::min(1.0, options.initial_step_size),
                        parameters.max_step));
      Eigen::VectorXd fallback_parameters;
      Eigen::VectorXd fallback_gradient;
      double fallback_energy = reference_energy;
      if (try_steepest_descent_armijo_fallback(
              objective,
              parameters,
              previous_parameters,
              previous_gradient,
              reference_energy,
              fallback_step,
              &fallback_parameters,
              &fallback_gradient,
              &fallback_energy,
              &fallback_step)) {
        const bool should_replace_primary =
            !primary_line_search_error.empty() ||
            stalled_line_search ||
            step <= kTinyStepFactor * parameters.min_step ||
            suspicious_primary_step ||
            fallback_energy < primary_energy;
        if (should_replace_primary) {
          current_parameters = std::move(fallback_parameters);
          current_gradient = std::move(fallback_gradient);
          energy = fallback_energy;
          step = fallback_step;
          parameter_step = current_parameters - previous_parameters;
          used_fallback = true;
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
    last_iteration_used_fallback = used_fallback;

    const bool accepted_point_chart_reset =
        objective->canonicalize_orbital_chart_at_current_point(
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
    if (used_fallback && search_direction.dot(current_gradient) >= 0.0) {
      search_direction = -current_gradient;
    }
  }
  return run_result;
}

}  // namespace xmvb::vb::optimizer_detail
