#pragma once

#include <vector>

#include "vbscf/derivatives/hessian/responses/active_space_direction.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin_backward.hpp"
#include "vbscf/derivatives/hessian/structure_response_internal.hpp"

namespace xmvb::vb {

SelectedStateProjectedDirectionalMatrices
build_projected_structure_direction(
    const AcceptedOuterResponseContext& accepted,
    const ActiveSpaceIntegralDirectionView& direction,
    const SameSpinDirectionalPairCache& directional_pair_cache);

}  // namespace xmvb::vb
