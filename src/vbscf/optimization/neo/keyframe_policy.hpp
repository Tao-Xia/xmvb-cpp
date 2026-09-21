#pragma once

namespace xmvb::vb {

/** @brief Model observations used to schedule an exact-gradient keyframe. */
struct NeoKeyframeObservation {
  double keyframe_gradient_norm = 0.0;
  double estimated_gradient_norm = 0.0;
  double reduction_ratio = 0.0;
  double trial_radius = 0.0;
  double next_radius = 0.0;
  bool reached_boundary = false;
};

/**
 * @brief PySCF-style frozen-Hessian/keyframe scheduling for backend NEO.
 *
 * PySCF mc1step uses a four-step keyframe interval and refreshes early when
 * its model gradient has fallen by a factor of three. NEO reuses one complete
 * quadratic model by enlarging its trust radius after a reliable boundary
 * trial. Every candidate is still checked with the true VBSCF energy; only
 * the exact gradient/Hessian rebuild is deferred to the selected keyframe.
 */
class NeoKeyframePolicy {
public:
  enum class Decision { Refresh, ContinueFrozenModel };

  Decision observe(const NeoKeyframeObservation& observation) {
    ++accepted_frozen_trials_;
    const bool estimated_gradient_requests_keyframe =
        observation.estimated_gradient_norm * kGradientReductionFactor <
        observation.keyframe_gradient_norm;
    const bool reliable_boundary_extension =
        observation.reached_boundary &&
        observation.reduction_ratio >= kReliableModelRatio &&
        observation.next_radius > observation.trial_radius;
    if (accepted_frozen_trials_ >= kMaximumFrozenTrials ||
        estimated_gradient_requests_keyframe ||
        !reliable_boundary_extension) {
      return Decision::Refresh;
    }
    return Decision::ContinueFrozenModel;
  }

  int accepted_frozen_trials() const noexcept {
    return accepted_frozen_trials_;
  }

private:
  /** PySCF mc1step default `kf_interval`. */
  static constexpr int kMaximumFrozenTrials = 4;
  /** PySCF mc1step default `kf_trust_region`. */
  static constexpr double kGradientReductionFactor = 3.0;
  /** Conventional trust-region ratio required before radius expansion. */
  static constexpr double kReliableModelRatio = 0.75;
  int accepted_frozen_trials_ = 0;
};

}  // namespace xmvb::vb
