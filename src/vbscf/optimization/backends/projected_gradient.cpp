#include "vbscf/optimization/backends/projected_gradient.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include "vbscf/optimization/globalization/line_search.hpp"
#include "vbscf/optimization/driver/checks.hpp"
#include "vbscf/optimization/driver/session.hpp"
#include "vbscf/optimization/trust_region/retraction.hpp"
#include "vbscf/optimization/objective/function.hpp"
#include "vbscf/optimization/driver/options.hpp"
#include "vbscf/optimization/driver/result.hpp"
#include "vbscf/orbitals/charts/layout.hpp"

namespace xmvb::vb::optimizer_detail {

BackendRunResult run_projected_gradient_backend(
    VbScfObjective* objective,
    const SparseParameterLayout& parameter_view,
    const VbScfOptimizerOptions& options,
    const Eigen::VectorXd& initial_parameters,
    const Eigen::VectorXd& initial_gradient,
    double initial_energy,
    VbScfOptimizerResult* result) {
  BackendRunResult run_result;
  Eigen::VectorXd current_parameters = initial_parameters;
  Eigen::VectorXd current_gradient = initial_gradient;
  double energy = initial_energy;
  double previous_energy = initial_energy;
  OrbitalChart current_space = build_orbital_chart(*objective, parameter_view);
  auto current_projection = current_space.project_gradient(current_gradient);

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
      result->termination_reason =
          "nonredundant_projected_gradient_initial_tolerance";
      run_result.final_gradient_l2_norm =
          current_projection.reduced_gradient.norm();
      break;
    }

    const Eigen::VectorXd reduced_search_direction =
        -current_space.apply_inverse_reduced_block_preconditioner(
            current_projection.reduced_gradient);
    const Eigen::VectorXd search_direction =
        gather_nonredundant_retract_tangent(
            objective->input().orbital_preparation_input,
            current_space,
            parameter_view,
            reduced_search_direction);
    const double directional_derivative =
        current_gradient.dot(search_direction);
    if (!std::isfinite(directional_derivative) ||
        directional_derivative >= 0.0) {
      result->termination_reason =
          "nonredundant_projected_gradient_non_descent_direction";
      run_result.final_gradient_l2_norm =
          current_projection.reduced_gradient.norm();
      break;
    }

    const OrbitalPreparationInput current_orbital_input =
        objective->input().orbital_preparation_input;
    Eigen::VectorXd accepted_parameters(current_parameters.size());
    Eigen::VectorXd accepted_gradient(current_gradient.size());
    double accepted_energy = energy;
    if (!try_armijo_backtracking_nonredundant_direction(
            objective,
            current_orbital_input,
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
      result->termination_reason =
          "nonredundant_projected_gradient_line_search_failed";
      run_result.final_gradient_l2_norm =
          current_projection.reduced_gradient.norm();
      break;
    }

    current_parameters = std::move(accepted_parameters);
    current_gradient = std::move(accepted_gradient);
    energy = accepted_energy;
    objective->canonicalize_chart(
        &current_parameters,
        &current_gradient);
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
    OrbitalChart next_space = build_orbital_chart(*objective, parameter_view);
    auto next_projection = next_space.project_gradient(current_gradient);
    if (std::abs(energy_change) < options.energy_tolerance &&
        gradient_infinity_norm(next_projection.reduced_gradient) <
            options.gradient_tolerance) {
      result->converged = true;
      result->termination_reason =
          "nonredundant_projected_gradient_dual_tolerance";
      run_result.final_gradient_l2_norm = next_projection.reduced_gradient.norm();
      break;
    }

    current_space = std::move(next_space);
    current_projection = std::move(next_projection);
  }
  return run_result;
}

}  // namespace xmvb::vb::optimizer_detail
