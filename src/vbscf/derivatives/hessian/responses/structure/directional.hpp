#pragma once

#include <memory>
#include <vector>

#include "vbscf/derivatives/hessian/responses/active_space/integral_direction.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/backward.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/pair_tile_stream_internal.hpp"
#include "vbscf/derivatives/hessian/context/response_internal.hpp"

namespace xmvb::vb {

std::shared_ptr<const AcceptedStructureResponseFactors>
build_accepted_structure_response_factors(
    const VbScfInput& input,
    const AcceptedPointContext& accepted_point,
    const Eigen::MatrixXd& selected_columns);

SelectedStateDirectionalStructureImages
build_selected_structure_direction(
    const AcceptedOuterResponseContext& accepted,
    const ActiveSpaceIntegralDirectionView& direction,
    const Eigen::MatrixXd* accepted_ri_active_pair_factors = nullptr,
    const Eigen::MatrixXd* directional_ri_active_pair_factors = nullptr);

/** Builds factorized structure images without a directional `U x U` cache. */
SelectedStateDirectionalStructureImages
build_selected_structure_direction_from_pair_tiles(
    const AcceptedOuterResponseContext& accepted,
    const ActiveSpaceIntegralDirectionView& direction,
    const Eigen::MatrixXd* accepted_ri_active_pair_factors = nullptr,
    const Eigen::MatrixXd* directional_ri_active_pair_factors = nullptr,
    const detail::DirectionalPairTileConsumer& additional_consumer = {});

}  // namespace xmvb::vb
