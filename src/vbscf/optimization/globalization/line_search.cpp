#include "vbscf/optimization/globalization/line_search.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "vbscf/optimization/driver/checks.hpp"

namespace xmvb::vb {

Eigen::VectorXd build_nonredundant_lifted_trial_parameters(
    const OrbitalPreparationInput& current_orbital_input,
    const OrbitalChart& current_space,
    const SparseParameterLayout& parameter_view,
    const Eigen::VectorXd& reduced_step) {
  const Eigen::VectorXd current_parameters =
      parameter_view.pack(current_orbital_input);
  // The reduced coordinates parameterize a local tangent vector on the
  // accepted sparse-orbital chart. Build finite trial points with the same
  // retraction used by the reduced HVP finite-difference operator, then pack
  // the resulting orbital table back into the optimizer coordinate vector.
  const OrbitalPreparationInput trial_orbital_input =
      current_space.retract_step(current_orbital_input, reduced_step);
  Eigen::VectorXd trial_parameters =
      parameter_view.pack(trial_orbital_input);
  if (trial_parameters.size() != current_parameters.size() ||
      !trial_parameters.allFinite()) {
    throw std::runtime_error(
        "nonredundant retraction returned invalid packed parameters");
  }
  return trial_parameters;
}

bool try_armijo_backtracking_nonredundant_direction(
    VbScfObjective* objective,
    const OrbitalPreparationInput& current_orbital_input,
    const OrbitalChart& current_space,
    const SparseParameterLayout& parameter_view,
    const Eigen::VectorXd& current_parameters,
    double current_energy,
    const Eigen::VectorXd& current_gradient,
    const Eigen::VectorXd& reduced_search_direction,
    const Eigen::VectorXd& packed_tangent_search_direction,
    double initial_step,
    double minimum_step,
    double armijo_constant,
    Eigen::VectorXd* accepted_parameters,
    Eigen::VectorXd* accepted_gradient,
    double* accepted_energy,
    bool* accepted_chart_changed) {
  if (accepted_chart_changed != nullptr) {
    *accepted_chart_changed = false;
  }
  const double directional_derivative =
      current_gradient.dot(packed_tangent_search_direction);
  if (!std::isfinite(directional_derivative) ||
      directional_derivative >= 0.0) {
    return false;
  }

  double step = std::max(minimum_step, initial_step);
  Eigen::VectorXd trial_parameters(current_parameters.size());
  // Keep the accepted-point state in the original sparse-coefficient chart,
  // but generate finite trial points with the nonredundant orbital-increment
  // lift rather than by adding the reduced direction directly in packed sparse
  // coordinates.
  while (step >= minimum_step) {
    trial_parameters = build_nonredundant_lifted_trial_parameters(
        current_orbital_input,
        current_space,
        parameter_view,
        step * reduced_search_direction);
    if (is_effectively_zero_step(
            trial_parameters - current_parameters,
            current_parameters)) {
      step *= 0.5;
      continue;
    }

    auto trial_evaluation =
        objective->evaluate_trial(trial_parameters, true);
    const double trial_energy = trial_evaluation.energy;
    const double armijo_upper_bound =
        current_energy + armijo_constant * step * directional_derivative;
    if (std::isfinite(trial_energy) && trial_energy <= armijo_upper_bound) {
      *accepted_parameters = parameter_view.pack(
          trial_evaluation.orbital_preparation_input);
      *accepted_gradient = trial_evaluation.gradient;
      *accepted_energy = trial_energy;
      if (accepted_chart_changed != nullptr) {
        *accepted_chart_changed = trial_evaluation.chart_changed;
      }
      objective->commit(std::move(trial_evaluation));
      return true;
    }
    step *= 0.5;
  }

  return false;
}

}  // namespace xmvb::vb
