#include "vbscf/derivatives/hessian/responses/same_spin/pair_tile_stream_internal.hpp"

#include "mcscf/ci/string_table.hpp"

#include <algorithm>
#include <optional>
#include <stdexcept>

#include "integrals/active/raw.hpp"
#include "integrals/active/ri_transform.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/tile_policy_internal.hpp"

namespace xmvb::vb::detail {
namespace {

void stream_spin_table(const xmvb::mcscf::StringTable& unique_determinants,
                       const AcceptedPairTileProvider& accepted_pair_provider,
                       bool alpha_channel,
                       bool beta_channel,
                       int n_active_orbitals,
                       const std::vector<double>& accepted_active_overlap,
                       const Eigen::MatrixXd& accepted_active_one_electron,
                       const xmvb::integrals::ActiveSpaceTwoElectronResult& accepted_two_electron,
                       const ActiveSpaceIntegralDirectionView& direction,
                       const Eigen::MatrixXd* accepted_ri_active_pair_factors,
                       const Eigen::MatrixXd* directional_ri_active_pair_factors,
                       bool include_opposite_spin,
                       int tile_extent,
                       const DirectionalPairTileConsumer& consume) {
    const int n_unique = static_cast<int>(unique_determinants.size());
    if (n_unique <= 0 || accepted_pair_provider.size() != n_unique) {
        throw std::invalid_argument(
            "directional pair tile stream requires a matching accepted pair provider");
    }
    tile_extent = std::max(1, std::min(n_unique, tile_extent));
    for (int left_begin = 0; left_begin < n_unique; left_begin += tile_extent) {
        const int left_end = std::min(n_unique, left_begin + tile_extent);
        // Ordered row traversal keeps every partner-action panel keyed by the
        // left tile live across the complete row. Visiting an upper tile and its
        // transpose consecutively would save one directional-pair construction,
        // but it invalidates that panel before the next column and consequently
        // repeats a complete accepted partner sweep for every pair tile.
        for (int right_begin = 0; right_begin < n_unique; right_begin += tile_extent) {
            const int right_end = std::min(n_unique, right_begin + tile_extent);
            const bool transposed = right_begin < left_begin;
            const int canonical_left_begin = transposed ? right_begin : left_begin;
            const int canonical_left_end = transposed ? right_end : left_end;
            const int canonical_right_begin = transposed ? left_begin : right_begin;
            const int canonical_right_end = transposed ? left_end : right_end;
            const AcceptedSpinPairTile canonical_accepted = accepted_pair_provider.build(
                canonical_left_begin,
                canonical_left_end,
                canonical_right_begin,
                canonical_right_end,
                accepted_active_overlap,
                accepted_active_one_electron,
                accepted_two_electron,
                AcceptedPairTileBuildOptions{.materialize_projected_pair_values = false,
                                             .populate_response_payload = true});
            SameSpinDirectionalPairTile same_spin = build_directional_pair_tile(
                unique_determinants,
                canonical_accepted,
                n_active_orbitals,
                direction,
                accepted_ri_active_pair_factors != nullptr ? &accepted_active_one_electron
                                                           : nullptr,
                accepted_ri_active_pair_factors,
                directional_ri_active_pair_factors);
            const SameSpinDirectionalPairTileView same_spin_view = same_spin.view(transposed);
            std::optional<DirectionalOppositeSpinPairTile> opposite_spin;
            if (include_opposite_spin) {
                opposite_spin.emplace(
                    build_directional_opposite_spin_pair_tile(unique_determinants,
                                                              canonical_accepted,
                                                              n_active_orbitals,
                                                              accepted_two_electron,
                                                              direction,
                                                              same_spin.view(),
                                                              accepted_ri_active_pair_factors,
                                                              directional_ri_active_pair_factors));
            }
            const std::optional<DirectionalOppositeSpinPairTileView> opposite_spin_view =
                opposite_spin ? std::optional<DirectionalOppositeSpinPairTileView>(
                                    opposite_spin->view(transposed))
                              : std::nullopt;
            std::optional<AcceptedSpinPairTile> ordered_accepted;
            if (transposed) {
                ordered_accepted.emplace(accepted_pair_provider.build(
                    left_begin,
                    left_end,
                    right_begin,
                    right_end,
                    accepted_active_overlap,
                    accepted_active_one_electron,
                    accepted_two_electron,
                    AcceptedPairTileBuildOptions{.materialize_projected_pair_values = false,
                                                 .populate_response_payload = true}));
            }
            const AcceptedSpinPairTile& accepted =
                transposed ? *ordered_accepted : canonical_accepted;
            consume(alpha_channel,
                    beta_channel,
                    accepted,
                    same_spin_view,
                    opposite_spin_view ? &*opposite_spin_view : nullptr);
        }
    }
}

} // namespace

void stream_directional_pair_tiles(
    const SameSpinPairCacheContext& accepted_pair_cache,
    int n_active_orbitals,
    const std::vector<double>& accepted_active_overlap,
    const Eigen::MatrixXd& accepted_active_one_electron,
    const xmvb::integrals::ActiveSpaceTwoElectronResult& accepted_two_electron,
    const ActiveSpaceIntegralDirectionView& direction,
    const Eigen::MatrixXd* accepted_ri_active_pair_factors,
    const Eigen::MatrixXd* directional_ri_active_pair_factors,
    bool include_opposite_spin,
    const DirectionalPairTileConsumer& consume,
    int n_response_states,
    std::size_t workspace_bytes) {
    if (!accepted_pair_cache.has_pair_providers() || !consume) {
        throw std::invalid_argument(
            "directional pair tile stream requires an enabled cache and consumer");
    }
    const Eigen::MatrixXd& upper = accepted_two_electron.orbital_transform;
    Eigen::MatrixXd raw_h1e;
    std::optional<xmvb::integrals::ActiveSpaceTwoElectronResult> raw_two_electron;
    std::vector<double> raw_one_electron_direction;
    std::vector<double> raw_packed_two_electron_direction;
    Eigen::MatrixXd raw_accepted_ri_factors;
    Eigen::MatrixXd raw_directional_ri_factors;
    std::optional<ActiveSpaceIntegralDirectionView> raw_direction;
    const Eigen::MatrixXd* streamed_h1e = &accepted_active_one_electron;
    const xmvb::integrals::ActiveSpaceTwoElectronResult* streamed_two_electron =
        &accepted_two_electron;
    const ActiveSpaceIntegralDirectionView* streamed_direction = &direction;
    const Eigen::MatrixXd* streamed_accepted_ri_factors = accepted_ri_active_pair_factors;
    const Eigen::MatrixXd* streamed_directional_ri_factors = directional_ri_active_pair_factors;
    if (upper.size() != 0) {
        raw_h1e =
            xmvb::integrals::materialize_raw_one_electron(accepted_active_one_electron, upper);
        streamed_h1e = &raw_h1e;
        raw_two_electron.emplace(
            xmvb::integrals::materialize_raw_active_integrals(accepted_two_electron));
        streamed_two_electron = &*raw_two_electron;

        const std::size_t active_matrix_size =
            static_cast<std::size_t>(n_active_orbitals) * n_active_orbitals;
        if (direction.one_electron.size() != active_matrix_size) {
            throw std::invalid_argument(
                "directional active one-electron matrix has incompatible dimensions");
        }
        const Eigen::Map<const Eigen::MatrixXd> framed_one_electron_direction(
            direction.one_electron.data(), n_active_orbitals, n_active_orbitals);
        const Eigen::MatrixXd raw_one_electron_direction_matrix =
            upper.transpose() * framed_one_electron_direction * upper;
        raw_one_electron_direction.assign(raw_one_electron_direction_matrix.data(),
                                          raw_one_electron_direction_matrix.data() +
                                              raw_one_electron_direction_matrix.size());
        if (!direction.packed_two_electron.empty()) {
            raw_packed_two_electron_direction =
                xmvb::integrals::materialize_raw_pair_kernel(direction.packed_two_electron, upper);
        }
        raw_direction.emplace(ActiveSpaceIntegralDirectionView{
            direction.overlap, raw_one_electron_direction, raw_packed_two_electron_direction});
        streamed_direction = &*raw_direction;

        if (accepted_ri_active_pair_factors != nullptr) {
            raw_accepted_ri_factors =
                xmvb::integrals::transform_ri_pair_factors(*accepted_ri_active_pair_factors, upper);
            streamed_accepted_ri_factors = &raw_accepted_ri_factors;
        }
        if (directional_ri_active_pair_factors != nullptr) {
            raw_directional_ri_factors = xmvb::integrals::transform_ri_pair_factors(
                *directional_ri_active_pair_factors, upper);
            streamed_directional_ri_factors = &raw_directional_ri_factors;
        }
    }
    const bool shared = accepted_pair_cache.shares_same_spin_pair_cache_between_spins();
    const PairTileExtents tile_extents = n_response_states > 0
                                             ? plan_local_response_tiles(accepted_pair_cache,
                                                                         n_active_orbitals,
                                                                         *streamed_two_electron,
                                                                         n_response_states,
                                                                         1,
                                                                         workspace_bytes)
                                                   .extents
                                             : plan_pair_tile_extents(accepted_pair_cache,
                                                                      n_active_orbitals,
                                                                      *streamed_two_electron,
                                                                      include_opposite_spin,
                                                                      workspace_bytes);
    stream_spin_table(accepted_pair_cache.alpha_reuse_table.unique_determinants,
                      accepted_pair_cache.alpha_provider(),
                      true,
                      shared,
                      n_active_orbitals,
                      accepted_active_overlap,
                      *streamed_h1e,
                      *streamed_two_electron,
                      *streamed_direction,
                      streamed_accepted_ri_factors,
                      streamed_directional_ri_factors,
                      include_opposite_spin,
                      tile_extents.alpha,
                      consume);
    if (shared) {
        return;
    }
    stream_spin_table(accepted_pair_cache.beta_reuse_table.unique_determinants,
                      accepted_pair_cache.beta_provider(),
                      false,
                      true,
                      n_active_orbitals,
                      accepted_active_overlap,
                      *streamed_h1e,
                      *streamed_two_electron,
                      *streamed_direction,
                      streamed_accepted_ri_factors,
                      streamed_directional_ri_factors,
                      include_opposite_spin,
                      tile_extents.beta,
                      consume);
}

} // namespace xmvb::vb::detail
