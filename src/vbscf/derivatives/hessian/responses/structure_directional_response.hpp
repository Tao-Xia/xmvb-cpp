#pragma once

#include <vector>

#include "vbscf/derivatives/hessian/responses/same_spin_response.hpp"
#include "vbscf/derivatives/hessian/structure_response_internal.hpp"

namespace xmvb::vb {

SelectedStateProjectedDirectionalMatrices
build_projected_structure_direction(
    const AcceptedOuterResponseContext& accepted,
    const std::vector<double>& delta_ao_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals,
    const SameSpinDirectionalPairCache& directional_pair_cache);

}  // namespace xmvb::vb
