#include "vbscf/optimization/coupled/subproblem.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace xmvb::vb {
namespace {

struct ShiftedSolution {
  double shift = 0.0;
  double orbital_norm = 0.0;
  Eigen::VectorXd metric_image;
  MinresResult linear;
  bool valid_metric = true;
};

double metric_norm(
    const Eigen::VectorXd& vector,
    const Eigen::VectorXd& metric_image,
    bool* valid) {
  const double squared_norm = vector.dot(metric_image);
  const double absolute_dot =
      vector.cwiseAbs().dot(metric_image.cwiseAbs());
  const double operations = 2.0 * static_cast<double>(vector.size());
  const double epsilon = std::numeric_limits<double>::epsilon();
  const double gamma = operations * epsilon < 1.0
      ? operations * epsilon / (1.0 - operations * epsilon)
      : std::numeric_limits<double>::infinity();
  *valid = std::isfinite(squared_norm) &&
      squared_norm >= -gamma * absolute_dot;
  return *valid ? std::sqrt(std::max(0.0, squared_norm)) : 0.0;
}

void accumulate_work(
    const MinresResult& linear,
    CoupledSubproblemResult* result) {
  ++result->shifted_linear_solves;
  result->total_minres_iterations += linear.iterations;
  result->total_operator_actions += linear.operator_actions;
  result->total_preconditioner_actions += linear.preconditioner_actions;
  result->total_residual_checks += linear.residual_checks;
}

}  // namespace

CoupledSubproblemResult solve_coupled_newton_subproblem(
    const CoupledNewtonOperator& coupled_operator,
    const Eigen::Ref<const Eigen::VectorXd>& orbital_gradient,
    double trust_radius,
    const CoupledSubproblemOptions& options,
    const ShiftedInversePreconditioner& make_inverse_preconditioner) {
  if (orbital_gradient.size() !=
          coupled_operator.n_orbital_coordinates() ||
      !orbital_gradient.allFinite() ||
      !(trust_radius > 0.0) || !std::isfinite(trust_radius) ||
      !(options.boundary_relative_tolerance > 0.0) ||
      !std::isfinite(options.boundary_relative_tolerance) ||
      !(options.convexifying_shift_lower_bound >= 0.0) ||
      !std::isfinite(options.convexifying_shift_lower_bound)) {
    throw std::invalid_argument("invalid coupled Newton subproblem");
  }

  CoupledSubproblemResult result;
  Eigen::VectorXd rhs = Eigen::VectorXd::Zero(coupled_operator.size());
  rhs.head(coupled_operator.n_orbital_coordinates()) = -orbital_gradient;

  auto solve_at = [&](double shift, const Eigen::VectorXd& initial_guess) {
    SymmetricOperatorAction preconditioner;
    if (make_inverse_preconditioner) {
      preconditioner = make_inverse_preconditioner(shift);
      if (!preconditioner) {
        throw std::runtime_error(
            "shifted inverse-preconditioner factory returned no action");
      }
    }
    ShiftedSolution shifted;
    shifted.shift = shift;
    shifted.linear = solve_symmetric_minres(
        [&](const Eigen::VectorXd& direction) {
          return coupled_operator.apply(direction, shift);
        },
        rhs,
        options.minres,
        preconditioner,
        initial_guess);
    accumulate_work(shifted.linear, &result);
    if (!shifted.linear.converged()) return shifted;
    const Eigen::VectorXd orbital = shifted.linear.solution.head(
        coupled_operator.n_orbital_coordinates());
    shifted.metric_image = coupled_operator.apply_orbital_metric(orbital).col(0);
    shifted.orbital_norm = metric_norm(
        orbital, shifted.metric_image, &shifted.valid_metric);
    return shifted;
  };

  auto finish = [&](ShiftedSolution shifted,
                    CoupledSubproblemStopReason reason) {
    result.linear_result = std::move(shifted.linear);
    result.step = result.linear_result.solution;
    result.orbital_step = result.step.head(
        coupled_operator.n_orbital_coordinates());
    result.response_step = result.step.tail(
        coupled_operator.n_response_coordinates());
    result.orbital_metric_image = std::move(shifted.metric_image);
    result.orbital_shift = shifted.shift;
    result.orbital_norm = shifted.orbital_norm;
    result.boundary_error = std::abs(result.orbital_norm - trust_radius);
    result.complementarity_residual =
        result.orbital_shift * result.boundary_error;
    result.unshifted_operator_image = result.linear_result.operator_image;
    if (result.unshifted_operator_image.size() == result.step.size() &&
        result.orbital_metric_image.size() == result.orbital_step.size()) {
      result.unshifted_operator_image.head(result.orbital_step.size()).noalias() -=
          result.orbital_shift * result.orbital_metric_image;
      const double model_value = orbital_gradient.dot(result.orbital_step) +
          0.5 * result.step.dot(result.unshifted_operator_image);
      result.predicted_decrease = -model_value;
    }
    result.stop_reason = reason;
    return result;
  };

  const double lower_bound = options.convexifying_shift_lower_bound;
  ShiftedSolution lower = solve_at(lower_bound, Eigen::VectorXd());
  if (!lower.linear.converged()) {
    result.linear_result = std::move(lower.linear);
    result.orbital_shift = lower_bound;
    result.stop_reason = CoupledSubproblemStopReason::LinearSolveFailure;
    return result;
  }
  if (!lower.valid_metric) {
    return finish(
        std::move(lower),
        CoupledSubproblemStopReason::InvalidOrbitalMetric);
  }

  const double boundary_tolerance =
      options.boundary_relative_tolerance * trust_radius;
  if (lower_bound == 0.0 && lower.orbital_norm <= trust_radius) {
    return finish(
        std::move(lower),
        CoupledSubproblemStopReason::InteriorConverged);
  }
  if (lower_bound > 0.0 && lower.orbital_norm < trust_radius) {
    if (trust_radius - lower.orbital_norm <= boundary_tolerance) {
      return finish(
          std::move(lower),
          CoupledSubproblemStopReason::BoundaryConverged);
    }
    return finish(
        std::move(lower),
        CoupledSubproblemStopReason::HardCaseUnresolved);
  }
  if (std::abs(lower.orbital_norm - trust_radius) <= boundary_tolerance) {
    return finish(
        std::move(lower),
        CoupledSubproblemStopReason::BoundaryConverged);
  }

  // The first shift increment has Hessian units and follows from the
  // directional Newton balance |g^T p| / (Delta ||p||_G). Subsequent growth
  // is geometric only to establish a secular bracket, not a work budget.
  double increment = std::abs(orbital_gradient.dot(
      lower.linear.solution.head(coupled_operator.n_orbital_coordinates()))) /
      (trust_radius * lower.orbital_norm);
  if (!(increment > 0.0) || !std::isfinite(increment)) {
    increment = std::nextafter(lower_bound,
                               std::numeric_limits<double>::infinity()) -
        lower_bound;
  }
  double upper_shift = lower_bound + increment;
  if (!(upper_shift > lower_bound) || !std::isfinite(upper_shift)) {
    upper_shift = std::nextafter(
        lower_bound, std::numeric_limits<double>::infinity());
  }

  ShiftedSolution upper;
  constexpr int numerical_limit = std::numeric_limits<double>::max_exponent -
      std::numeric_limits<double>::min_exponent +
      std::numeric_limits<double>::digits;
  bool bracketed = false;
  for (int expansion = 0; expansion < numerical_limit; ++expansion) {
    upper = solve_at(upper_shift, lower.linear.solution);
    if (!upper.linear.converged()) {
      result.linear_result = std::move(upper.linear);
      result.orbital_shift = upper_shift;
      result.stop_reason = CoupledSubproblemStopReason::LinearSolveFailure;
      return result;
    }
    if (!upper.valid_metric) {
      return finish(
          std::move(upper),
          CoupledSubproblemStopReason::InvalidOrbitalMetric);
    }
    if (upper.orbital_norm <= trust_radius) {
      bracketed = true;
      break;
    }
    lower = std::move(upper);
    increment *= 2.0;
    upper_shift = lower.shift + increment;
    if (!(upper_shift > lower.shift) || !std::isfinite(upper_shift)) break;
  }
  if (!bracketed) {
    result.linear_result = std::move(lower.linear);
    result.orbital_shift = lower.shift;
    result.stop_reason = CoupledSubproblemStopReason::ShiftBracketFailure;
    return result;
  }
  if (std::abs(upper.orbital_norm - trust_radius) <= boundary_tolerance) {
    return finish(
        std::move(upper),
        CoupledSubproblemStopReason::BoundaryConverged);
  }

  for (int refinement = 0; refinement < numerical_limit; ++refinement) {
    // On the convex branch, 1 / ||p(lambda)||_G is monotone and is exactly
    // affine for a one-mode secular equation. Its safeguarded secant therefore
    // avoids the endpoint stagnation of false position applied to ||p|| itself.
    const double inverse_lower_norm = 1.0 / lower.orbital_norm;
    const double inverse_upper_norm = 1.0 / upper.orbital_norm;
    const double inverse_radius = 1.0 / trust_radius;
    double trial_shift = lower.shift +
        (upper.shift - lower.shift) *
            (inverse_radius - inverse_lower_norm) /
            (inverse_upper_norm - inverse_lower_norm);
    if (!(trial_shift > lower.shift && trial_shift < upper.shift) ||
        !std::isfinite(trial_shift)) {
      trial_shift = lower.shift + 0.5 * (upper.shift - lower.shift);
    }
    if (trial_shift == lower.shift || trial_shift == upper.shift) break;

    const double interpolation =
        (trial_shift - lower.shift) / (upper.shift - lower.shift);
    const Eigen::VectorXd initial_guess =
        (1.0 - interpolation) * lower.linear.solution +
        interpolation * upper.linear.solution;
    ShiftedSolution trial = solve_at(trial_shift, initial_guess);
    if (!trial.linear.converged()) {
      result.linear_result = std::move(trial.linear);
      result.orbital_shift = trial_shift;
      result.stop_reason = CoupledSubproblemStopReason::LinearSolveFailure;
      return result;
    }
    if (!trial.valid_metric) {
      return finish(
          std::move(trial),
          CoupledSubproblemStopReason::InvalidOrbitalMetric);
    }
    if (std::abs(trial.orbital_norm - trust_radius) <= boundary_tolerance) {
      return finish(
          std::move(trial),
          CoupledSubproblemStopReason::BoundaryConverged);
    }
    if (trial.orbital_norm > trust_radius) {
      lower = std::move(trial);
    } else {
      upper = std::move(trial);
    }
  }

  // Adjacent floating-point shifts are the tightest representable secular
  // bracket. Return success only if its feasible endpoint meets the requested
  // orbital-boundary certificate.
  if (std::abs(upper.orbital_norm - trust_radius) <= boundary_tolerance) {
    return finish(
        std::move(upper),
        CoupledSubproblemStopReason::BoundaryConverged);
  }
  return finish(
      std::move(upper),
      CoupledSubproblemStopReason::BoundaryResolutionFailure);
}

}  // namespace xmvb::vb
