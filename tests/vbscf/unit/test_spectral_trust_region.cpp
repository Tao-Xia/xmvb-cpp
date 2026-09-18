#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include "vbscf/optimization/trust_region/spectral.hpp"
#include "vbscf/optimization/trust_region/truncated_newton.hpp"

namespace {
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void check(const std::string& name, const Eigen::VectorXd& d,
           const Eigen::VectorXd& g, double radius, bool boundary,
           bool hard_case = false) {
  const auto result = xmvb::vb::solve_spectral_trust_region(d, g, radius);
  const double norm = result.step.stableNorm();
  const Eigen::VectorXd residual =
      ((d.array() + result.shift) * result.step.array()).matrix() + g;
  const double scale = std::max(g.stableNorm(),
      (d.cwiseAbs().maxCoeff() + result.shift) * radius);
  require(result.step.allFinite() && std::isfinite(result.shift), name + ": finite");
  require(norm <= radius * (1.0 + 1e-12), name + ": feasible");
  require(result.shift >= 0.0, name + ": nonnegative multiplier");
  require(d.minCoeff() + result.shift >= -1e-12 * d.cwiseAbs().maxCoeff(),
          name + ": shifted Hessian must be positive semidefinite");
  require(residual.stableNorm() <= 1e-12 * scale, name + ": stationarity");
  require(result.shift == 0.0 || std::abs(norm / radius - 1.0) < 1e-12,
          name + ": complementarity");
  require(result.boundary == boundary, name + ": boundary classification");
  require(result.hard_case == hard_case, name + ": hard-case classification");
  std::cout << name << ": passed\n";
}

void check_model_fidelity_radius_scaling() {
  using xmvb::vb::TruncatedNewtonModelFidelity;
  xmvb::vb::TruncatedNewtonStepResult boundary_step;
  boundary_step.retract_tangent_norm = 0.5;
  boundary_step.reached_boundary = true;

  const xmvb::vb::TruncatedNewtonTrialEvaluation accepted_trial{
      16.0, 15.0};
  const double approximate_radius =
      xmvb::vb::update_nonredundant_truncated_newton_trust_radius(
          0.5,
          1.0e-12,
          accepted_trial,
          boundary_step,
          TruncatedNewtonModelFidelity::CoreApproximate,
          true);
  const double exact_radius =
      xmvb::vb::update_nonredundant_truncated_newton_trust_radius(
          0.5,
          1.0e-12,
          accepted_trial,
          boundary_step,
          TruncatedNewtonModelFidelity::DirectionallyExact,
          true);
  require(
      std::abs(approximate_radius - 2.0) <= 1.0e-14,
      "quadratic-discrepancy radius did not use square-root scaling");
  require(
      std::abs(exact_radius - 0.5 * std::cbrt(16.0)) <= 1.0e-14,
      "cubic-remainder radius did not use cube-root scaling");
  require(
      approximate_radius > exact_radius,
      "model fidelities produced indistinguishable accepted-step scaling");

  const xmvb::vb::TruncatedNewtonTrialEvaluation rejected_trial{0.25, 1.0};
  const double retained_fraction = 1.0 / 1.75;
  const double approximate_contraction =
      xmvb::vb::update_nonredundant_truncated_newton_trust_radius(
          0.5,
          1.0e-12,
          rejected_trial,
          boundary_step,
          TruncatedNewtonModelFidelity::CoreApproximate,
          false);
  const double exact_contraction =
      xmvb::vb::update_nonredundant_truncated_newton_trust_radius(
          0.5,
          1.0e-12,
          rejected_trial,
          boundary_step,
          TruncatedNewtonModelFidelity::DirectionallyExact,
          false);
  require(
      std::abs(approximate_contraction -
               0.5 * std::sqrt(retained_fraction)) <= 1.0e-14,
      "quadratic-discrepancy rejection did not use square-root scaling");
  require(
      std::abs(exact_contraction -
               0.5 * std::cbrt(retained_fraction)) <= 1.0e-14,
      "cubic-remainder rejection did not use cube-root scaling");
  require(
      approximate_contraction < exact_contraction,
      "model fidelities produced indistinguishable rejected-step scaling");
}

void check_rejected_trial_interpolation() {
  using xmvb::vb::TruncatedNewtonModelFidelity;
  xmvb::vb::TruncatedNewtonStepResult boundary_step;
  boundary_step.retract_tangent_norm = 0.5;
  boundary_step.reached_boundary = true;

  const double interpolated_radius =
      xmvb::vb::update_nonredundant_truncated_newton_trust_radius(
          0.5,
          1.0e-12,
          {-9.0, 1.0, 1.0},
          boundary_step,
          TruncatedNewtonModelFidelity::CoreApproximate,
          false);
  require(
      std::abs(interpolated_radius - 0.025) <= 1.0e-14,
      "rejected energy-increasing trial did not use directional interpolation");

  const double positive_decrease_radius =
      xmvb::vb::update_nonredundant_truncated_newton_trust_radius(
          0.5,
          1.0e-12,
          {0.2, 1.0, 1.0},
          boundary_step,
          TruncatedNewtonModelFidelity::DirectionallyExact,
          false);
  require(
      std::abs(positive_decrease_radius - 0.3125) <= 1.0e-14,
      "rejected overpredicting trial did not minimize its measured ray model");
}

void check_accepted_radius_ray_cap() {
  using xmvb::vb::TruncatedNewtonModelFidelity;
  xmvb::vb::TruncatedNewtonStepResult boundary_step;
  boundary_step.retract_tangent_norm = 0.5;
  boundary_step.reached_boundary = true;

  const double radius =
      xmvb::vb::update_nonredundant_truncated_newton_trust_radius(
          0.5,
          1.0e-12,
          {0.9001, 0.9, 1.0},
          boundary_step,
          TruncatedNewtonModelFidelity::CoreApproximate,
          true);
  require(
      std::abs(radius - 2.5) <= 1.0e-12,
      "accepted-step expansion exceeded its directional quadratic minimizer");
}
}  // namespace

int main() {
  try {
    using V = Eigen::Vector2d;
    require(
        xmvb::vb::truncated_newton_model_is_below_outer_accuracy(
            9.0e-4, 9.0e-8, 1.0e-3, 1.0e-7),
        "outer-accuracy model stopping rule rejected resolved work");
    require(
        !xmvb::vb::truncated_newton_model_is_below_outer_accuracy(
            9.0e-4, 1.1e-7, 1.0e-3, 1.0e-7),
        "energy-unresolved model work was stopped");
    require(
        xmvb::vb::truncated_newton_trial_is_acceptable({0.8, 1.0}),
        "well-resolved model decrease was rejected");
    require(
        xmvb::vb::truncated_newton_trial_is_acceptable({1.5, 1.0}),
        "underpredicted model decrease was rejected");
    require(
        !xmvb::vb::truncated_newton_trial_is_acceptable({0.4, 1.0}),
        "model-error-dominated decrease was accepted");
    require(
        !xmvb::vb::truncated_newton_trial_is_acceptable({-0.1, 1.0}),
        "energy-increasing trial was accepted");
    check_model_fidelity_radius_scaling();
    check_rejected_trial_interpolation();
    check_accepted_radius_ray_cap();
    check("positive definite interior", V(2, 4), V(1, 2), 2, false);
    check("positive definite boundary", V(2, 4), V(1, 2), 0.1, true);
    check("zero-multiplier boundary", V(2, 4), V(2, 0), 1, true);
    check("semidefinite compatible", V(0, 2), V(0, 1), 1, false);
    check("semidefinite incompatible", V(0, 2), V(1, 1), 1, true);
    check("indefinite regular", V(-2, 3), V(1, 2), 1, true);
    check("indefinite hard case", V(-2, 3), V(0, 1), 1, true, true);
    check("pure negative curvature", V(-2, 3), V(0, 0), 1, true, true);
    check("zero model", V(0, 0), V(0, 0), 1, false);
    check("linear model", V(0, 0), V(1, 2), 1, true);
    check("repeated lowest eigenvalue", V(-2, -2), V(0, 0), 1, true, true);
    check("tiny positive mode", V(1e-60, 1), V(2e-60, 0.1), 1, true);
    check("large radius near spectral pole", V(-2, 3), V(1, 2), 1e20, true);
    check("small radius", V(-2, 3), V(1, 2), 1e-20, true);
    check("small radius with disparate scales", V(0, 1e200), V(1e-200, 0), 1e-200, true);
    for (double scale : {1e-100, 1e100}) {
      check("energy rescaling", scale * V(-2, 3), scale * V(1, 2), 1, true);
      const auto reference = xmvb::vb::solve_spectral_trust_region(V(-2, 3), V(1, 2), 1);
      const auto scaled = xmvb::vb::solve_spectral_trust_region(
          scale * V(-2, 3), scale * V(1, 2), 1);
      require((reference.step - scaled.step).norm() < 1e-12,
              "solution changed under energy rescaling");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
