#include "vbscf/optimization/krylov/minres.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace xmvb::vb {
namespace {

Eigen::VectorXd apply_checked(
    const SymmetricOperatorAction& action,
    const Eigen::VectorXd& vector,
    const char* name,
    int* calls) {
  Eigen::VectorXd image = action(vector);
  ++*calls;
  if (image.size() != vector.size() || !image.allFinite()) {
    throw std::runtime_error(std::string(name) +
                             " returned an invalid vector");
  }
  return image;
}

double positive_preconditioned_norm_squared(
    const Eigen::VectorXd& residual,
    const Eigen::VectorXd& preconditioned_residual) {
  const double value = residual.dot(preconditioned_residual);
  const double absolute_dot =
      residual.cwiseAbs().dot(preconditioned_residual.cwiseAbs());
  const double operations = 2.0 * static_cast<double>(residual.size());
  const double epsilon = std::numeric_limits<double>::epsilon();
  const double gamma = operations * epsilon < 1.0
      ? operations * epsilon / (1.0 - operations * epsilon)
      : std::numeric_limits<double>::infinity();
  if (!std::isfinite(value) || value < -gamma * absolute_dot) {
    return value;
  }
  return std::max(0.0, value);
}

}  // namespace

MinresResult solve_symmetric_minres(
    const SymmetricOperatorAction& apply_operator,
    const Eigen::Ref<const Eigen::VectorXd>& rhs,
    const MinresOptions& options,
    const SymmetricOperatorAction& apply_inverse_preconditioner,
    const Eigen::VectorXd& initial_guess) {
  if (!apply_operator) {
    throw std::invalid_argument("MINRES requires an operator action");
  }
  if (rhs.size() == 0 || !rhs.allFinite()) {
    throw std::invalid_argument("MINRES requires a finite nonempty RHS");
  }
  if (!(options.relative_residual_tolerance >= 0.0) ||
      !std::isfinite(options.relative_residual_tolerance) ||
      !(options.absolute_residual_tolerance >= 0.0) ||
      !std::isfinite(options.absolute_residual_tolerance) ||
      (options.relative_residual_tolerance == 0.0 &&
       options.absolute_residual_tolerance == 0.0) ||
      options.maximum_iterations < 0) {
    throw std::invalid_argument("invalid MINRES convergence options");
  }
  if (initial_guess.size() != 0 &&
      (initial_guess.size() != rhs.size() || !initial_guess.allFinite())) {
    throw std::invalid_argument("invalid MINRES initial guess");
  }

  MinresResult result;
  result.solution = initial_guess.size() == 0
      ? Eigen::VectorXd::Zero(rhs.size())
      : initial_guess;
  result.rhs_norm = rhs.stableNorm();
  result.residual_target = std::max(
      options.absolute_residual_tolerance,
      options.relative_residual_tolerance * result.rhs_norm);
  const int maximum_iterations = options.maximum_iterations == 0
      ? static_cast<int>(rhs.size())
      : options.maximum_iterations;

  auto apply_preconditioner = [&](const Eigen::VectorXd& vector) {
    if (!apply_inverse_preconditioner) return vector;
    return apply_checked(
        apply_inverse_preconditioner,
        vector,
        "MINRES inverse preconditioner",
        &result.preconditioner_actions);
  };
  auto explicit_residual = [&]() -> Eigen::VectorXd {
    ++result.residual_checks;
    if (result.solution.isZero(0.0)) {
      result.operator_image = Eigen::VectorXd::Zero(rhs.size());
    } else {
      result.operator_image = apply_checked(
          apply_operator,
          result.solution,
          "MINRES operator",
          &result.operator_actions);
    }
    Eigen::VectorXd value = rhs;
    value.noalias() -= result.operator_image;
    return value;
  };

  Eigen::VectorXd residual = explicit_residual();
  result.residual_norm = residual.stableNorm();
  if (result.residual_norm <= result.residual_target) {
    result.residual = std::move(residual);
    result.stop_reason = MinresStopReason::Converged;
    return result;
  }

  Eigen::VectorXd v_old = Eigen::VectorXd::Zero(rhs.size());
  Eigen::VectorXd v = Eigen::VectorXd::Zero(rhs.size());
  Eigen::VectorXd v_new = residual;
  Eigen::VectorXd w = Eigen::VectorXd::Zero(rhs.size());
  Eigen::VectorXd w_new = apply_preconditioner(v_new);
  Eigen::VectorXd p_older = Eigen::VectorXd::Zero(rhs.size());
  Eigen::VectorXd p_old = Eigen::VectorXd::Zero(rhs.size());
  Eigen::VectorXd p = Eigen::VectorXd::Zero(rhs.size());

  double beta_new = 0.0;
  double beta_first = 0.0;
  double cosine = 1.0;
  double old_cosine = 1.0;
  double sine = 0.0;
  double old_sine = 0.0;
  double eta = 1.0;
  double restart_residual_norm = result.residual_norm;

  auto initialize_recurrence = [&]() -> bool {
    const double beta_squared = positive_preconditioned_norm_squared(
        v_new, w_new);
    if (!(beta_squared > 0.0) || !std::isfinite(beta_squared)) {
      result.stop_reason = MinresStopReason::NonPositivePreconditioner;
      return false;
    }
    beta_new = std::sqrt(beta_squared);
    beta_first = beta_new;
    result.estimated_preconditioned_residual_norm = beta_first;
    return true;
  };
  if (!initialize_recurrence()) {
    result.residual = std::move(residual);
    return result;
  }

  for (int iteration = 0; iteration < maximum_iterations; ++iteration) {
    const double beta = beta_new;
    if (!(beta > 0.0) || !std::isfinite(beta)) {
      result.stop_reason = MinresStopReason::LanczosBreakdown;
      break;
    }
    v_old = v;
    v_new /= beta;
    w_new /= beta;
    v = v_new;
    w = w_new;

    v_new = apply_checked(
        apply_operator, w, "MINRES operator", &result.operator_actions) -
        beta * v_old;
    const double alpha = v_new.dot(w);
    v_new.noalias() -= alpha * v;
    w_new = apply_preconditioner(v_new);

    const double beta_squared = positive_preconditioned_norm_squared(
        v_new, w_new);
    if (!std::isfinite(beta_squared) || beta_squared < 0.0) {
      result.stop_reason = MinresStopReason::NonPositivePreconditioner;
      result.iterations = iteration + 1;
      break;
    }
    beta_new = std::sqrt(beta_squared);

    const double r2 = sine * alpha + cosine * old_cosine * beta;
    const double r3 = old_sine * beta;
    const double r1_head =
        cosine * alpha - old_cosine * sine * beta;
    const double r1 = std::hypot(r1_head, beta_new);
    if (!(r1 > 0.0) || !std::isfinite(r1)) {
      result.stop_reason = MinresStopReason::LanczosBreakdown;
      result.iterations = iteration + 1;
      break;
    }
    old_cosine = cosine;
    old_sine = sine;
    cosine = r1_head / r1;
    sine = beta_new / r1;

    p_older = p_old;
    p_old = p;
    p.noalias() = (w - r2 * p_old - r3 * p_older) / r1;
    result.solution.noalias() += beta_first * cosine * eta * p;
    result.estimated_preconditioned_residual_norm *= std::abs(sine);
    result.iterations = iteration + 1;

    const double relative_trigger = std::min(
        1.0, result.residual_target / restart_residual_norm);
    const bool candidate =
        result.estimated_preconditioned_residual_norm <=
            relative_trigger * beta_first ||
        beta_new == 0.0;
    if (candidate) {
      residual = explicit_residual();
      result.residual_norm = residual.stableNorm();
      if (result.residual_norm <= result.residual_target) {
        result.residual = std::move(residual);
        result.stop_reason = MinresStopReason::Converged;
        return result;
      }
      if (beta_new == 0.0 && result.residual_norm > result.residual_target) {
        result.stop_reason = MinresStopReason::LanczosBreakdown;
        break;
      }

      // Reliable restart: the Givens estimate can drift below the true
      // residual in finite precision. Preserve x and restart from b - A*x.
      ++result.reliable_restarts;
      restart_residual_norm = result.residual_norm;
      v_old.setZero();
      v.setZero();
      v_new = residual;
      w.setZero();
      w_new = apply_preconditioner(v_new);
      p_older.setZero();
      p_old.setZero();
      p.setZero();
      cosine = 1.0;
      old_cosine = 1.0;
      sine = 0.0;
      old_sine = 0.0;
      eta = 1.0;
      if (!initialize_recurrence()) break;
      continue;
    }
    eta = -sine * eta;
  }

  // Every non-converged return still reports the true residual, so callers
  // can distinguish an exhausted work budget from a numerical breakdown.
  residual = explicit_residual();
  result.residual = std::move(residual);
  result.residual_norm = result.residual.stableNorm();
  if (result.residual_norm <= result.residual_target) {
    result.stop_reason = MinresStopReason::Converged;
  } else if (result.stop_reason != MinresStopReason::LanczosBreakdown &&
             result.stop_reason != MinresStopReason::NonPositivePreconditioner) {
    result.stop_reason = MinresStopReason::IterationLimit;
  }
  return result;
}

}  // namespace xmvb::vb
