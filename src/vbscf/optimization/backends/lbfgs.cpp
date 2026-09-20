#include "vbscf/optimization/backends/lbfgs.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "vbscf/optimization/globalization/line_search.hpp"
#include "vbscf/optimization/driver/checks.hpp"
#include "vbscf/optimization/driver/session.hpp"
#include "vbscf/optimization/trust_region/retraction.hpp"
#include "vbscf/optimization/objective/function.hpp"
#include "vbscf/optimization/preconditioners/transported_lbfgs.hpp"
#include "vbscf/optimization/driver/options.hpp"
#include "vbscf/optimization/driver/result.hpp"
#include "vbscf/orbitals/charts/layout.hpp"

namespace xmvb::vb::optimizer_detail {

BackendRunResult run_lbfgs_backend(
    VbScfObjective* objective,
    const SparseParameterLayout& parameter_view,
    const VbScfOptimizerOptions& options,
    const Eigen::VectorXd& initial_parameters,
    const Eigen::VectorXd& initial_gradient,
    double initial_energy,
    VbScfOptimizerResult* result) {
  BackendRunResult run_result;
  const int history_size = options.history_size;
  const LbfgsInitialInverse initial_inverse = options.lbfgs_initial_inverse;
  std::vector<PackedSecantPair> packed_secant_history;
  packed_secant_history.reserve(std::max(0, history_size));

  Eigen::VectorXd current_parameters = initial_parameters;
  Eigen::VectorXd current_gradient = initial_gradient;
  double energy = initial_energy;
  double previous_energy = initial_energy;
  OrbitalChart current_space = build_orbital_chart(*objective, parameter_view);
  auto current_projection = current_space.project_gradient(current_gradient);
  Eigen::VectorXd previous_parameters(initial_parameters.size());
  Eigen::VectorXd previous_gradient(initial_gradient.size());

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
      result->termination_reason = "lbfgs_initial_tolerance";
      run_result.final_gradient_l2_norm =
          current_projection.reduced_gradient.norm();
      break;
    }

    const auto build_reduced_search_direction = [&]() -> Eigen::VectorXd {
      const auto inverse_hessian =
          build_transported_reduced_lbfgs_preconditioner(
              current_space,
              packed_secant_history,
              history_size,
              initial_inverse);
      return -inverse_hessian.apply(current_projection.reduced_gradient);
    };
    const auto is_descent_direction = [&](const Eigen::VectorXd& direction) {
      return direction.allFinite() &&
          current_projection.reduced_gradient.dot(direction) < 0.0;
    };
    Eigen::VectorXd reduced_search_direction =
        build_reduced_search_direction();
    if (!is_descent_direction(reduced_search_direction)) {
      packed_secant_history.clear();
      reduced_search_direction = build_reduced_search_direction();
    }
    if (!is_descent_direction(reduced_search_direction)) {
      reduced_search_direction = -current_projection.reduced_gradient;
    }
    const OrbitalPreparationInput previous_orbital_input =
        objective->input().orbital_preparation_input;
    Eigen::VectorXd search_direction = gather_nonredundant_retract_tangent(
        previous_orbital_input,
        current_space,
        parameter_view,
        reduced_search_direction);
    double directional_derivative = current_gradient.dot(search_direction);
    if (!std::isfinite(directional_derivative) ||
        directional_derivative >= 0.0 ||
        is_effectively_zero_step(search_direction, current_parameters)) {
      packed_secant_history.clear();
      reduced_search_direction = build_reduced_search_direction();
      if (!is_descent_direction(reduced_search_direction)) {
        reduced_search_direction = -current_projection.reduced_gradient;
      }
      search_direction = gather_nonredundant_retract_tangent(
          previous_orbital_input,
          current_space,
          parameter_view,
          reduced_search_direction);
      directional_derivative = current_gradient.dot(search_direction);
    }
    if (!std::isfinite(directional_derivative) ||
        directional_derivative >= 0.0) {
      result->termination_reason =
          "lbfgs_non_descent_direction";
      run_result.final_gradient_l2_norm =
          current_projection.reduced_gradient.norm();
      break;
    }

    previous_parameters = current_parameters;
    previous_gradient = current_gradient;
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
      result->termination_reason = "lbfgs_line_search_failed";
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
            "lbfgs_line_search_stalled";
        break;
      }
      next_space = build_orbital_chart(*objective, parameter_view);
      next_projection = next_space.project_gradient(current_gradient);
      next_reduced_gradient_inf_norm =
          gradient_infinity_norm(next_projection.reduced_gradient);
      parameter_step = current_parameters - previous_parameters;
      packed_secant_history.clear();
    }

    ++run_result.n_iterations;
    sync_result_from_objective(*objective, result);
    record_accepted_iteration_snapshot(
        objective,
        run_result.n_iterations,
        options,
        result,
        nullptr,
        &next_projection.reduced_gradient);
    run_result.final_gradient_l2_norm = current_gradient.norm();
    const double energy_change = energy - previous_energy;
    previous_energy = energy;
    if (std::abs(energy_change) < options.energy_tolerance &&
        next_reduced_gradient_inf_norm < options.gradient_tolerance) {
      result->converged = true;
      result->termination_reason = "lbfgs_dual_tolerance";
      run_result.final_gradient_l2_norm = next_projection.reduced_gradient.norm();
      break;
    }

    const bool rank_changed =
        next_space.rank_signature() != current_space.rank_signature() ||
        next_space.reduced_size() != current_space.reduced_size();
    if (rank_changed) {
      packed_secant_history.clear();
    } else {
      append_projected_secant_pair(
          next_space,
          std::move(parameter_step),
          current_gradient - previous_gradient,
          history_size,
          &packed_secant_history);
    }

    current_space = std::move(next_space);
    current_projection = std::move(next_projection);
  }
  return run_result;
}

}  // namespace xmvb::vb::optimizer_detail
