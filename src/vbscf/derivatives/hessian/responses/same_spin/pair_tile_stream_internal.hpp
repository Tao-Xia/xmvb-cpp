#pragma once

#include <functional>

#include "vbscf/derivatives/hessian/responses/opposite_spin/pair_response_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/pair_response_internal.hpp"

namespace xmvb::vb::detail {

/** One callback invocation consumes all directional data for a pair tile. */
using DirectionalPairTileConsumer = std::function<void(
    bool alpha_channel,
    bool beta_channel,
    const SameSpinDirectionalPairTileView& same_spin,
    const DirectionalOppositeSpinPairTileView* opposite_spin)>;

/**
 * @brief Streams bounded directional unique-spin pair tiles.
 *
 * Alpha and beta share one tile evaluation when their unique-spin spaces and
 * accepted pair cache are identical. Otherwise each spin table is streamed
 * once. The consumer must finish every use of a tile before returning.
 */
void stream_directional_pair_tiles(
    const SameSpinPairCacheContext& accepted_pair_cache,
    int n_active_orbitals,
    const std::vector<double>& accepted_active_overlap,
    const Eigen::MatrixXd& accepted_active_one_electron,
    const ActiveSpaceTwoElectronResult& accepted_two_electron,
    const ActiveSpaceIntegralDirectionView& direction,
    const Eigen::MatrixXd* accepted_ri_active_pair_factors,
    const Eigen::MatrixXd* directional_ri_active_pair_factors,
    bool include_opposite_spin,
    const DirectionalPairTileConsumer& consume);

}  // namespace xmvb::vb::detail
