#include "vbscf/derivatives/gradient/active_space/backend.hpp"

namespace xmvb::vb {

ActiveSpaceGradientResult ElectronicStateBackend::evaluate_energy_gradient(
    const VbScfInput& input, const ElectronicStateRequest& request) const {
  return complete_gradient(input, evaluate_energy(input, request));
}

}  // namespace xmvb::vb
