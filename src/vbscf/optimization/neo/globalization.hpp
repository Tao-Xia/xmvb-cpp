#pragma once

namespace xmvb::vb {

/** @brief Outcome of comparing a NEO trial step with its quadratic model. */
struct NeoGlobalizationResult {
  /** Quadratic-model decrease, `-g^T p - 0.5 p^T H p`. */
  double predicted_reduction = 0.0;
  /** Evaluated energy decrease, `E_current - E_trial`. */
  double actual_reduction = 0.0;
  /** Actual-to-predicted reduction ratio; `-infinity` for an invalid model. */
  double rho = 0.0;
  /** Whether the trial replaces the current accepted point. */
  bool accepted = false;
  /** Radius for the next solve of the current or next quadratic model. */
  double next_radius = 0.0;
};

/**
 * @brief Globalizes a NEO trial by the trust-region reduction ratio.
 *
 * The quadratic prediction and evaluated decrease are
 *
 * `pred = -g^T p - 0.5 p^T H p`,
 * `ared = E_current - E_trial`, and `rho = ared / pred`.
 *
 * The constants are the conventional trust-region choices: a trial is
 * accepted for `rho >= 0.1`; the radius is quartered for `rho < 1/4` and
 * doubled for a boundary step with `rho > 3/4`.  They are dimensionless
 * numerical safeguards, not molecule- or method-dependent parameters.
 * Nonpositive predicted or actual reduction always rejects the trial.  A
 * rejected trial returns a smaller radius, so the caller can solve the same
 * accepted-point model again without rebuilding it.
 *
 * @param current_energy Energy at the accepted point.
 * @param trial_energy Evaluated energy at the trial point.
 * @param gradient_dot_step Scalar product `g^T p`.
 * @param step_dot_hessian_step Curvature `p^T H p`.
 * @param boundary Whether the step reached the current trust-region boundary.
 * @param current_radius Radius used to compute the trial step.
 * @param minimum_radius Smallest radius permitted by the outer optimizer.
 * @throws std::invalid_argument for nonfinite scalars or invalid radii.
 */
NeoGlobalizationResult globalize_neo_trial(
    double current_energy,
    double trial_energy,
    double gradient_dot_step,
    double step_dot_hessian_step,
    bool boundary,
    double current_radius,
    double minimum_radius);

}  // namespace xmvb::vb
