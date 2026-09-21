#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "vbscf/optimization/neo/globalization.hpp"

namespace {

using xmvb::vb::NeoGlobalizationResult;
using xmvb::vb::globalize_neo_trial;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void require_close(double actual, double expected, const std::string& message) {
  const double scale = std::max({1.0, std::abs(actual), std::abs(expected)});
  require(std::abs(actual - expected) <= 1.0e-14 * scale, message);
}

NeoGlobalizationResult evaluate(
    double actual,
    double predicted,
    bool boundary = true,
    double radius = 1.0,
    double minimum_radius = 1.0e-3) {
  // g^T p = -predicted and p^T H p = 0 gives the requested prediction.
  return globalize_neo_trial(
      0.0, -actual, -predicted, 0.0, boundary, radius, minimum_radius);
}

void check_accurate_boundary_step_expands_radius() {
  const auto result = evaluate(0.95, 1.0);
  require_close(result.predicted_reduction, 1.0,
                "predicted reduction is incorrect");
  require_close(result.actual_reduction, 0.95,
                "actual reduction is incorrect");
  require_close(result.rho, 0.95, "reduction ratio is incorrect");
  require(result.accepted, "accurate decreasing trial was rejected");
  require_close(result.next_radius, 2.0,
                "accurate boundary trial did not expand the radius");
}

void check_interior_step_does_not_expand_radius() {
  const auto result = evaluate(1.0, 1.0, false, 0.4);
  require(result.accepted, "exact interior trial was rejected");
  require_close(result.next_radius, 0.4,
                "interior trial unexpectedly expanded the radius");
}

void check_moderate_model_agreement_keeps_radius() {
  const auto result = evaluate(0.5, 1.0);
  require(result.accepted, "moderate model agreement was rejected");
  require_close(result.next_radius, 1.0,
                "moderate model agreement changed the radius");
}

void check_poor_model_agreement_retries_smaller() {
  const auto result = evaluate(0.05, 1.0);
  require(!result.accepted, "poor model agreement was accepted");
  require_close(result.next_radius, 0.25,
                "poor model agreement did not quarter the radius");
}

void check_nonpositive_actual_reduction_is_rejected() {
  const auto result = evaluate(-0.2, 1.0);
  require(!result.accepted, "energy-increasing trial was accepted");
  require(result.rho < 0.0, "energy-increasing trial has positive rho");
  require_close(result.next_radius, 0.25,
                "energy-increasing trial did not shrink the radius");
}

void check_nonpositive_prediction_is_rejected() {
  const auto result = evaluate(0.2, -1.0);
  require(!result.accepted, "model-increasing trial was accepted");
  require(std::isinf(result.rho) && result.rho < 0.0,
          "invalid prediction does not have the documented rho sentinel");
  require_close(result.next_radius, 0.25,
                "invalid model did not shrink the radius");
}

void check_minimum_radius_is_respected() {
  const auto result = evaluate(0.01, 1.0, true, 0.01, 0.004);
  require_close(result.next_radius, 0.004,
                "radius contraction crossed the minimum radius");
}

void check_invalid_input_is_rejected() {
  bool threw = false;
  try {
    evaluate(1.0, 1.0, true, 0.1, 0.2);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "inconsistent trust radii were accepted");

  threw = false;
  try {
    globalize_neo_trial(
        0.0,
        std::numeric_limits<double>::quiet_NaN(),
        -1.0,
        0.0,
        true,
        1.0,
        0.1);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "nonfinite trial energy was accepted");
}

}  // namespace

int main() {
  check_accurate_boundary_step_expands_radius();
  check_interior_step_does_not_expand_radius();
  check_moderate_model_agreement_keeps_radius();
  check_poor_model_agreement_retries_smaller();
  check_nonpositive_actual_reduction_is_rejected();
  check_nonpositive_prediction_is_rejected();
  check_minimum_radius_is_respected();
  check_invalid_input_is_rejected();
  return 0;
}
