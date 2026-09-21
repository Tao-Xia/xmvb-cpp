#include "vbscf/optimization/globalization/line_search.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "vbscf/optimization/driver/checks.hpp"

namespace xmvb::vb {
namespace {

bool update_more_thuente_interval(
    double* best_step,
    double* best_value,
    double* best_derivative,
    double* other_step,
    double* other_value,
    double* other_derivative,
    double* trial_step,
    double trial_value,
    double trial_derivative,
    bool* bracketed,
    double step_min,
    double step_max) {
  if ((*bracketed &&
       (*trial_step <= std::min(*best_step, *other_step) ||
        *trial_step >= std::max(*best_step, *other_step))) ||
      *best_derivative * (*trial_step - *best_step) >= 0.0 ||
      step_max < step_min) {
    return false;
  }

  const double opposite_sign =
      trial_derivative * (*best_derivative / std::abs(*best_derivative));
  bool bounded = false;
  double cubic_step = 0.0;
  double secant_step = 0.0;
  double safeguarded_step = 0.0;
  if (trial_value > *best_value) {
    bounded = true;
    const double theta =
        3.0 * (*best_value - trial_value) /
            (*trial_step - *best_step) +
        *best_derivative + trial_derivative;
    const double scale = std::max(
        {std::abs(theta), std::abs(*best_derivative),
         std::abs(trial_derivative)});
    double gamma = scale * std::sqrt(std::max(
        0.0,
        std::pow(theta / scale, 2) -
            (*best_derivative / scale) * (trial_derivative / scale)));
    if (*trial_step < *best_step) gamma = -gamma;
    const double p = gamma - *best_derivative + theta;
    const double q = gamma - *best_derivative + gamma + trial_derivative;
    cubic_step = *best_step + (p / q) * (*trial_step - *best_step);
    secant_step = *best_step +
        0.5 * (*best_derivative /
               ((*best_value - trial_value) /
                    (*trial_step - *best_step) +
                *best_derivative)) *
            (*trial_step - *best_step);
    safeguarded_step =
        std::abs(cubic_step - *best_step) <
                std::abs(secant_step - *best_step)
            ? cubic_step
            : cubic_step + 0.5 * (secant_step - cubic_step);
    *bracketed = true;
  } else if (opposite_sign < 0.0) {
    const double theta =
        3.0 * (*best_value - trial_value) /
            (*trial_step - *best_step) +
        *best_derivative + trial_derivative;
    const double scale = std::max(
        {std::abs(theta), std::abs(*best_derivative),
         std::abs(trial_derivative)});
    double gamma = scale * std::sqrt(std::max(
        0.0,
        std::pow(theta / scale, 2) -
            (*best_derivative / scale) * (trial_derivative / scale)));
    if (*trial_step > *best_step) gamma = -gamma;
    const double p = gamma - trial_derivative + theta;
    const double q = gamma - trial_derivative + gamma + *best_derivative;
    cubic_step = *trial_step + (p / q) * (*best_step - *trial_step);
    secant_step = *trial_step +
        (trial_derivative / (trial_derivative - *best_derivative)) *
            (*best_step - *trial_step);
    safeguarded_step =
        std::abs(cubic_step - *trial_step) >
                std::abs(secant_step - *trial_step)
            ? cubic_step
            : secant_step;
    *bracketed = true;
  } else if (std::abs(trial_derivative) <
             std::abs(*best_derivative)) {
    bounded = true;
    const double theta =
        3.0 * (*best_value - trial_value) /
            (*trial_step - *best_step) +
        *best_derivative + trial_derivative;
    const double scale = std::max(
        {std::abs(theta), std::abs(*best_derivative),
         std::abs(trial_derivative)});
    double gamma = scale * std::sqrt(std::max(
        0.0,
        std::pow(theta / scale, 2) -
            (*best_derivative / scale) * (trial_derivative / scale)));
    if (*trial_step > *best_step) gamma = -gamma;
    const double p = gamma - trial_derivative + theta;
    const double q = gamma + (*best_derivative - trial_derivative) + gamma;
    const double ratio = p / q;
    if (ratio < 0.0 && gamma != 0.0) {
      cubic_step = *trial_step + ratio * (*best_step - *trial_step);
    } else {
      cubic_step = *trial_step > *best_step ? step_max : step_min;
    }
    secant_step = *trial_step +
        (trial_derivative / (trial_derivative - *best_derivative)) *
            (*best_step - *trial_step);
    if (*bracketed) {
      safeguarded_step =
          std::abs(*trial_step - cubic_step) <
                  std::abs(*trial_step - secant_step)
              ? cubic_step
              : secant_step;
    } else {
      safeguarded_step =
          std::abs(*trial_step - cubic_step) >
                  std::abs(*trial_step - secant_step)
              ? cubic_step
              : secant_step;
    }
  } else {
    if (*bracketed) {
      const double theta =
          3.0 * (trial_value - *other_value) /
              (*other_step - *trial_step) +
          *other_derivative + trial_derivative;
      const double scale = std::max(
          {std::abs(theta), std::abs(*other_derivative),
           std::abs(trial_derivative)});
      double gamma = scale * std::sqrt(std::max(
          0.0,
          std::pow(theta / scale, 2) -
              (*other_derivative / scale) * (trial_derivative / scale)));
      if (*trial_step > *other_step) gamma = -gamma;
      const double p = gamma - trial_derivative + theta;
      const double q = gamma - trial_derivative + gamma + *other_derivative;
      cubic_step =
          *trial_step + (p / q) * (*other_step - *trial_step);
      safeguarded_step = cubic_step;
    } else {
      safeguarded_step = *trial_step > *best_step ? step_max : step_min;
    }
  }

  if (trial_value > *best_value) {
    *other_step = *trial_step;
    *other_value = trial_value;
    *other_derivative = trial_derivative;
  } else {
    if (opposite_sign < 0.0) {
      *other_step = *best_step;
      *other_value = *best_value;
      *other_derivative = *best_derivative;
    }
    *best_step = *trial_step;
    *best_value = trial_value;
    *best_derivative = trial_derivative;
  }

  safeguarded_step = std::clamp(safeguarded_step, step_min, step_max);
  *trial_step = safeguarded_step;
  if (*bracketed && bounded) {
    const double interval_guard =
        *best_step + 0.66 * (*other_step - *best_step);
    *trial_step = *other_step > *best_step
        ? std::min(interval_guard, *trial_step)
        : std::max(interval_guard, *trial_step);
  }
  return std::isfinite(*trial_step);
}

}  // namespace

bool try_xmvb_more_thuente_line_search(
    VbScfObjective* objective,
    const Eigen::VectorXd& parameters,
    double energy,
    const Eigen::VectorXd& gradient,
    const Eigen::VectorXd& direction,
    double initial_step,
    MoreThuenteResult* result) {
  if (objective == nullptr || result == nullptr ||
      parameters.size() != gradient.size() ||
      parameters.size() != direction.size()) {
    throw std::invalid_argument("invalid XMVB line-search arguments");
  }
  constexpr double sufficient_decrease = 1.0e-4;
  constexpr double curvature = 0.9;
  constexpr double interval_tolerance = 1.0e-16;
  constexpr double absolute_minimum_step = 1.0e-20;
  constexpr double maximum_step = 1.0e20;
  constexpr int maximum_evaluations = 20;
  constexpr double extrapolation = 4.0;

  const double initial_derivative = gradient.dot(direction);
  if (!(initial_step > 0.0) || !(initial_derivative < 0.0) ||
      !std::isfinite(energy) || !gradient.allFinite() ||
      !direction.allFinite()) {
    return false;
  }

  bool bracketed = false;
  bool first_stage = true;
  double step = initial_step;
  double best_step = 0.0;
  double best_value = energy;
  double best_derivative = initial_derivative;
  double other_step = 0.0;
  double other_value = energy;
  double other_derivative = initial_derivative;
  double width = maximum_step - absolute_minimum_step;
  double previous_width = 2.0 * width;
  const double derivative_test =
      sufficient_decrease * initial_derivative;

  for (int evaluation = 1; evaluation <= maximum_evaluations; ++evaluation) {
    const double interval_minimum = bracketed
        ? std::min(best_step, other_step)
        : best_step;
    const double interval_maximum = bracketed
        ? std::max(best_step, other_step)
        : step + extrapolation * (step - best_step);
    step = std::clamp(step, absolute_minimum_step, maximum_step);
    if ((bracketed &&
         (step <= interval_minimum || step >= interval_maximum)) ||
        (bracketed &&
         interval_maximum - interval_minimum <=
             interval_tolerance * interval_maximum)) {
      step = best_step;
    }

    auto trial = objective->evaluate_trial(parameters + step * direction);
    const double trial_value = trial.energy;
    const double trial_derivative = trial.gradient.dot(direction);
    const double armijo_bound = energy + step * derivative_test;
    if (std::isfinite(trial_value) && trial.gradient.allFinite() &&
        trial_value <= armijo_bound &&
        std::abs(trial_derivative) <=
            curvature * (-initial_derivative)) {
      result->parameters = parameters + step * direction;
      result->gradient = trial.gradient;
      result->energy = trial_value;
      result->step = step;
      result->evaluations = evaluation;
      objective->commit(std::move(trial));
      return true;
    }
    if (!std::isfinite(trial_value) || !trial.gradient.allFinite()) {
      return false;
    }
    if (evaluation == maximum_evaluations ||
        (bracketed &&
         interval_maximum - interval_minimum <=
             interval_tolerance * interval_maximum)) {
      return false;
    }

    if (first_stage && trial_value <= armijo_bound &&
        trial_derivative >=
            std::min(sufficient_decrease, curvature) *
                initial_derivative) {
      first_stage = false;
    }
    bool updated = false;
    if (first_stage && trial_value <= best_value &&
        trial_value > armijo_bound) {
      double modified_value = trial_value - step * derivative_test;
      double modified_best_value =
          best_value - best_step * derivative_test;
      double modified_other_value =
          other_value - other_step * derivative_test;
      double modified_derivative = trial_derivative - derivative_test;
      double modified_best_derivative =
          best_derivative - derivative_test;
      double modified_other_derivative =
          other_derivative - derivative_test;
      updated = update_more_thuente_interval(
          &best_step, &modified_best_value, &modified_best_derivative,
          &other_step, &modified_other_value, &modified_other_derivative,
          &step, modified_value, modified_derivative, &bracketed,
          interval_minimum, interval_maximum);
      best_value = modified_best_value + best_step * derivative_test;
      other_value = modified_other_value + other_step * derivative_test;
      best_derivative = modified_best_derivative + derivative_test;
      other_derivative = modified_other_derivative + derivative_test;
    } else {
      updated = update_more_thuente_interval(
          &best_step, &best_value, &best_derivative,
          &other_step, &other_value, &other_derivative,
          &step, trial_value, trial_derivative, &bracketed,
          interval_minimum, interval_maximum);
    }
    if (!updated) return false;
    if (bracketed) {
      if (std::abs(other_step - best_step) >= 0.66 * previous_width) {
        step = best_step + 0.5 * (other_step - best_step);
      }
      previous_width = width;
      width = std::abs(other_step - best_step);
    }
  }
  return false;
}

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
