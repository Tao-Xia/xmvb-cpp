#include "vbscf/optimization/neo/globalization.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace xmvb::vb {
namespace {

constexpr double kAcceptanceRatio = 0.1;
constexpr double kShrinkRatio = 0.25;
constexpr double kExpansionRatio = 0.75;
constexpr double kRadiusContraction = 0.25;
constexpr double kRadiusExpansion = 2.0;

void require_finite(double value, const char* name) {
  if (!std::isfinite(value)) {
    throw std::invalid_argument(std::string("NEO globalization requires finite ") +
                                name);
  }
}

}  // namespace

NeoGlobalizationResult globalize_neo_trial(
    double current_energy,
    double trial_energy,
    double gradient_dot_step,
    double step_dot_hessian_step,
    bool boundary,
    double current_radius,
    double minimum_radius) {
  require_finite(current_energy, "current energy");
  require_finite(trial_energy, "trial energy");
  require_finite(gradient_dot_step, "gradient-step product");
  require_finite(step_dot_hessian_step, "step-Hessian-step product");
  require_finite(current_radius, "current radius");
  require_finite(minimum_radius, "minimum radius");
  if (minimum_radius <= 0.0 || current_radius < minimum_radius) {
    throw std::invalid_argument(
        "NEO globalization requires 0 < minimum radius <= current radius");
  }

  NeoGlobalizationResult result;
  result.predicted_reduction =
      -gradient_dot_step - 0.5 * step_dot_hessian_step;
  result.actual_reduction = current_energy - trial_energy;
  require_finite(result.predicted_reduction, "predicted reduction");
  require_finite(result.actual_reduction, "actual reduction");

  if (result.predicted_reduction <= 0.0) {
    result.rho = -std::numeric_limits<double>::infinity();
  } else {
    result.rho = result.actual_reduction / result.predicted_reduction;
  }

  result.accepted = result.predicted_reduction > 0.0 &&
                    result.actual_reduction > 0.0 &&
                    result.rho >= kAcceptanceRatio;

  if (result.rho < kShrinkRatio) {
    result.next_radius =
        std::max(minimum_radius, kRadiusContraction * current_radius);
  } else if (result.rho > kExpansionRatio && boundary) {
    result.next_radius = kRadiusExpansion * current_radius;
    require_finite(result.next_radius, "expanded radius");
  } else {
    result.next_radius = current_radius;
  }
  return result;
}

}  // namespace xmvb::vb
