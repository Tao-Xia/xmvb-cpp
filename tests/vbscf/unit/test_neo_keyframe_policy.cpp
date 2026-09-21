#include <iostream>

#include "vbscf/optimization/neo/keyframe_policy.hpp"

namespace {

using Decision = xmvb::vb::NeoKeyframePolicy::Decision;
using Observation = xmvb::vb::NeoKeyframeObservation;

bool reliable_boundary_trials_stop_at_the_fourth_candidate() {
  xmvb::vb::NeoKeyframePolicy policy;
  for (int trial = 1; trial <= 3; ++trial) {
    const Decision decision = policy.observe(
        Observation{1.0, 0.8, 0.9,
                    static_cast<double>(trial),
                    static_cast<double>(trial + 1), true});
    if (decision != Decision::ContinueFrozenModel) return false;
  }
  return policy.observe({1.0, 0.8, 0.9, 4.0, 5.0, true}) ==
             Decision::Refresh &&
         policy.accepted_frozen_trials() == 4;
}

bool unreliable_or_interior_trials_refresh_immediately() {
  xmvb::vb::NeoKeyframePolicy interior;
  xmvb::vb::NeoKeyframePolicy unreliable;
  xmvb::vb::NeoKeyframePolicy no_expansion;
  return interior.observe({1.0, 0.8, 0.9, 1.0, 2.0, false}) ==
             Decision::Refresh &&
         unreliable.observe({1.0, 0.8, 0.7, 1.0, 2.0, true}) ==
             Decision::Refresh &&
         no_expansion.observe({1.0, 0.8, 0.9, 1.0, 1.0, true}) ==
             Decision::Refresh;
}

bool strong_model_gradient_reduction_requests_a_keyframe() {
  xmvb::vb::NeoKeyframePolicy policy;
  return policy.observe({3.0, 0.9, 0.9, 1.0, 2.0, true}) ==
      Decision::Refresh;
}

}  // namespace

int main() {
  const bool interval =
      reliable_boundary_trials_stop_at_the_fourth_candidate();
  const bool reliability =
      unreliable_or_interior_trials_refresh_immediately();
  const bool gradient =
      strong_model_gradient_reduction_requests_a_keyframe();
  std::cout << "interval=" << interval
            << " reliability=" << reliability
            << " gradient=" << gradient << '\n';
  return interval && reliability && gradient ? 0 : 1;
}
