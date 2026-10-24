/** @file Common-solver regression; links without the VBSCF library. */
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "mcscf/optimization/neo/globalization.hpp"

namespace {

using xmvb::mcscf::globalize_neo_trial;
using xmvb::mcscf::NeoGlobalizationResult;

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

void require_close(double actual, double expected, const std::string& message) {
    const double scale = std::max({1.0, std::abs(actual), std::abs(expected)});
    require(std::abs(actual - expected) <= 1.0e-14 * scale, message);
}

xmvb::mcscf::NeoGlobalizationResult evaluate(double actual,
                                             double predicted,
                                             bool boundary = true,
                                             double radius = 1.0,
                                             double minimum_radius = 1.0e-3) {
    // g^T p = -predicted and p^T H p = 0 gives the requested prediction.
    return xmvb::mcscf::globalize_neo_trial(
        0.0, -actual, -predicted, 0.0, boundary, radius, minimum_radius);
}

void check_accurate_boundary_step_expands_radius() {
    const auto result = evaluate(0.95, 1.0);
    require_close(result.predicted_reduction, 1.0, "predicted reduction is incorrect");
    require_close(result.actual_reduction, 0.95, "actual reduction is incorrect");
    require_close(result.rho, 0.95, "reduction ratio is incorrect");
    require(result.accepted, "accurate decreasing trial was rejected");
    require_close(result.next_radius, 2.0, "accurate boundary trial did not expand the radius");
}

void check_interior_step_does_not_expand_radius() {
    const auto result = evaluate(1.0, 1.0, false, 0.4);
    require(result.accepted, "exact interior trial was rejected");
    require_close(result.next_radius, 0.4, "interior trial unexpectedly expanded the radius");
}

void check_moderate_model_agreement_keeps_radius() {
    const auto result = evaluate(0.5, 1.0);
    require(result.accepted, "moderate model agreement was rejected");
    require_close(result.next_radius, 1.0, "moderate model agreement changed the radius");
}

void check_poor_model_agreement_retries_smaller() {
    const auto result = evaluate(0.05, 1.0);
    require(!result.accepted, "poor model agreement was accepted");
    require_close(result.next_radius, 0.25, "poor model agreement did not quarter the radius");
}

/** @brief Energy acceptance does not imply that the current radius is reliable. */
void check_accepted_trial_can_request_contraction() {
    const auto result = evaluate(0.15, 1.0, true, 0.25);
    require(result.accepted, "decreasing trial above the acceptance ratio was rejected");
    require_close(
        result.next_radius, 0.0625, "accepted low-agreement trial lost its contraction request");
}

void check_nonpositive_actual_reduction_is_rejected() {
    const auto result = evaluate(-0.2, 1.0);
    require(!result.accepted, "energy-increasing trial was accepted");
    require(result.rho < 0.0, "energy-increasing trial has positive rho");
    require_close(result.next_radius, 0.25, "energy-increasing trial did not shrink the radius");
}

void check_nonpositive_prediction_is_rejected() {
    const auto result = evaluate(0.2, -1.0);
    require(!result.accepted, "model-increasing trial was accepted");
    require(std::isinf(result.rho) && result.rho < 0.0,
            "invalid prediction does not have the documented rho sentinel");
    require_close(result.next_radius, 0.25, "invalid model did not shrink the radius");
}

void check_minimum_radius_is_respected() {
    const auto result = evaluate(0.01, 1.0, true, 0.01, 0.004);
    require_close(result.next_radius, 0.004, "radius contraction crossed the minimum radius");
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
        xmvb::mcscf::globalize_neo_trial(
            0.0, std::numeric_limits<double>::quiet_NaN(), -1.0, 0.0, true, 1.0, 0.1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "nonfinite trial energy was accepted");
}

void check_trial_phase_is_coordinate_invariant() {
    constexpr double tolerance = 1.0e-3;
    const double raw_gradient = 1.0e-4;
    const double raw_metric = 1.0e-8;
    const double scale = 1.0e4;
    const double transformed_gradient = scale * raw_gradient;
    const double transformed_metric = scale * scale * raw_metric;
    const double raw_norm = std::sqrt(raw_gradient * raw_gradient / raw_metric);
    const double transformed_norm =
        std::sqrt(transformed_gradient * transformed_gradient / transformed_metric);
    require(raw_gradient < tolerance && transformed_gradient > tolerance,
            "fixture does not expose coordinate-component phase misclassification");
    require_close(raw_norm, transformed_norm, "physical gradient norm changed under chart scaling");
    require(xmvb::mcscf::neo_trial_allowed(raw_norm, tolerance) &&
                xmvb::mcscf::neo_trial_allowed(transformed_norm, tolerance),
            "equivalent charts do not select the same nonlinear trial phase");
    require(!xmvb::mcscf::neo_trial_allowed(3.0 * tolerance, tolerance) &&
                !xmvb::mcscf::neo_trial_allowed(0.0, tolerance),
            "near-stationary solve unexpectedly enabled approximate trials");
    bool rejected = false;
    try {
        xmvb::mcscf::neo_trial_allowed(raw_norm, 0.0);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "trial phase accepted a nonpositive gradient tolerance");
}

void check_coordinate_trial_budget_cannot_hide_stagnation() {
    using xmvb::mcscf::neo_trial_residual_target;
    const double coordinate_gradient = 2.0e-4;
    const double physical_gradient = 4.0e-3;
    const double physical_target = 1.0e-3;
    const double target = xmvb::mcscf::neo_trial_residual_target(
        coordinate_gradient, physical_gradient, 1.0e-3, physical_target);
    require_close(target, 5.0e-5, "absolute coordinate floor hid an unresolved gradient");
    require_close(xmvb::mcscf::neo_trial_residual_target(
                      2.0 * coordinate_gradient, physical_gradient, 1.0e-3, physical_target),
                  2.0 * target,
                  "absolute-floor correction did not track compressed coordinates");
    require_close(xmvb::mcscf::neo_trial_residual_target(0.0045, 0.0078, 0.00102, 0.001),
                  0.00102,
                  "useful historical candidate budget was unnecessarily tightened");
    require_close(xmvb::mcscf::neo_trial_residual_target(1.0, 1.0, 0.1, 0.1),
                  0.1,
                  "trial safeguard loosened an existing tighter budget");
    require_close(xmvb::mcscf::neo_trial_residual_target(0.01, 1.0, 0.0001, 0.0),
                  0.0001,
                  "trial safeguard changed the relative-only candidate policy");
    require_close(xmvb::mcscf::neo_trial_residual_target(0.0, 0.0, 1.0, 1.0),
                  0.0,
                  "zero gradient admitted a nonzero residual budget");
}

} // namespace

int main() {
    check_accurate_boundary_step_expands_radius();
    check_interior_step_does_not_expand_radius();
    check_moderate_model_agreement_keeps_radius();
    check_poor_model_agreement_retries_smaller();
    check_accepted_trial_can_request_contraction();
    check_nonpositive_actual_reduction_is_rejected();
    check_nonpositive_prediction_is_rejected();
    check_minimum_radius_is_respected();
    check_invalid_input_is_rejected();
    check_trial_phase_is_coordinate_invariant();
    check_coordinate_trial_budget_cannot_hide_stagnation();
    return 0;
}
