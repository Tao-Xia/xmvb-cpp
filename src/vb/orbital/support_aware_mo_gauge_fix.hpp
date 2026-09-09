#pragma once

// Compatibility header. New code should include
// "vbscf/orbitals/gauge/support_preserving_gauge.hpp".
#include "vbscf/orbitals/gauge/support_preserving_gauge.hpp"

namespace xmvb::vb {

using SupportAwareInactiveMoGaugeTransform = SupportPreservingGaugeTransform;

inline bool orbital_input_has_support_aware_mo_gauge_reference(
    const OrbitalPreparationInput& input) {
  return orbital_input_has_support_preserving_gauge_reference(input);
}

inline SupportAwareInactiveMoGaugeTransform
apply_support_aware_inactive_mo_gauge_fix(
    const OrbitalPreparationInput& reference_layout,
    OrbitalPreparationInput* input) {
  return apply_support_preserving_inactive_gauge(reference_layout, input);
}

inline SupportAwareInactiveMoGaugeTransform
apply_support_aware_inactive_mo_gauge_fix(OrbitalPreparationInput* input) {
  return apply_support_preserving_inactive_gauge(input);
}

}  // namespace xmvb::vb
