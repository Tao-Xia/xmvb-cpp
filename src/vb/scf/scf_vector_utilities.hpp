#ifndef XMVB_VB_SCF_SCF_VECTOR_UTILITIES_HPP_
#define XMVB_VB_SCF_SCF_VECTOR_UTILITIES_HPP_

// Small numerical and validation helpers shared across the SCF optimizer
// line-search, truncated-Newton, and outer-loop code. Kept header-only and
// inline so that split TUs (vb/scf/line_search.cpp, vb/scf/orbital_objective.cpp,
// vb/scf/cpp_vb_scf_optimizer.cpp, ...) can call them without introducing a
// dedicated source file or another compatibility layer.

#include <cmath>
#include <limits>

#include <Eigen/Core>

namespace xmvb::vb {

// Infinity norm of a gradient-like vector. Returns +inf if any entry is
// non-finite so that callers can branch on a poisoned state without a second
// scan.
inline double gradient_infinity_norm(const Eigen::VectorXd& gradient) {
  double norm = 0.0;
  for (Eigen::Index index = 0; index < gradient.size(); ++index) {
    if (!std::isfinite(gradient[index])) {
      return std::numeric_limits<double>::infinity();
    }
    norm = std::max(norm, std::abs(gradient[index]));
  }
  return norm;
}

// Moré-Thuente returns the best trial point seen so far. When none of the
// trial evaluations satisfy the Wolfe conditions, that "best" point may be
// the original iterate itself, which appears as a zero update and can trap
// the outer L-BFGS loop in repeated null iterations.
inline bool is_effectively_zero_step(
    const Eigen::VectorXd& parameter_step,
    const Eigen::VectorXd& reference_parameters) {
  constexpr double kRelativeStepTolerance =
      128.0 * std::numeric_limits<double>::epsilon();
  const double step_inf_norm = gradient_infinity_norm(parameter_step);
  const double reference_inf_norm =
      std::max(1.0, gradient_infinity_norm(reference_parameters));
  return step_inf_norm <= kRelativeStepTolerance * reference_inf_norm;
}

// Predicate for detecting a Wolfe line search that returns nearly the same
// energy while leaving the gradient essentially unchanged. Such an iteration
// is treated as a line-search stall so the caller can fall back to Armijo
// steepest descent from the previous iterate.
inline bool line_search_made_no_meaningful_progress(
    double reference_energy,
    double trial_energy,
    double previous_gradient_inf_norm,
    double current_gradient_inf_norm,
    double gradient_tolerance) {
  if (!std::isfinite(reference_energy) || !std::isfinite(trial_energy) ||
      !std::isfinite(previous_gradient_inf_norm) ||
      !std::isfinite(current_gradient_inf_norm)) {
    return false;
  }

  const double energy_scale = std::max(1.0, std::abs(reference_energy));
  const double energy_change =
      std::abs(trial_energy - reference_energy);
  constexpr double kRelativeEnergyProgressTolerance =
      4096.0 * std::numeric_limits<double>::epsilon();
  const double effective_energy_tolerance =
      kRelativeEnergyProgressTolerance * energy_scale;
  if (energy_change > effective_energy_tolerance) {
    return false;
  }
  if (previous_gradient_inf_norm < gradient_tolerance) {
    return false;
  }

  return current_gradient_inf_norm >= 0.9 * previous_gradient_inf_norm;
}

inline bool finite_vector_matches_size(
    const Eigen::VectorXd& vector,
    Eigen::Index expected_size) {
  return vector.size() == expected_size &&
      vector.allFinite();
}

inline bool finite_nonzero_vector_matches_size(
    const Eigen::VectorXd& vector,
    Eigen::Index expected_size) {
  return finite_vector_matches_size(vector, expected_size) &&
      vector.squaredNorm() > 0.0;
}

}  // namespace xmvb::vb

#endif  // XMVB_VB_SCF_SCF_VECTOR_UTILITIES_HPP_
