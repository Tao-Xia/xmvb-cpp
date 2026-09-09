#pragma once

#include <vector>

#include "vbscf/core/vbscf_input.hpp"
#include "vbscf/derivatives/hessian/accepted_point_context.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin_response.hpp"
#include "vbscf/derivatives/hessian/structure_response_internal.hpp"
#include "vbscf/structures/coefficient_blocks.hpp"

namespace xmvb::vb {

SelectedStateProjectedDirectionalMatrices
build_selected_state_projected_directional_structure_matrices(
    const VbScfInput& input,
    const AcceptedPointContext& accepted_point_context,
    const std::vector<StructureCoefficientBlock>& coefficient_blocks,
    const std::vector<double>& delta_ao_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals,
    const SameSpinDirectionalPairCache& directional_pair_cache);

SelectedStateGeneralizedEigenDirectionalResponse
build_selected_state_generalized_eigen_directional_response(
    const AcceptedPointContext& accepted_point_context,
    const SelectedStateProjectedDirectionalMatrices&
        projected_directional_structure_matrices);

}  // namespace xmvb::vb
