#include "vbscf/derivatives/hessian/responses/same_spin/pair_tile_stream_internal.hpp"

#include <algorithm>
#include <optional>
#include <stdexcept>

#include "vbscf/derivatives/hessian/responses/same_spin/tile_policy_internal.hpp"

namespace xmvb::vb::detail {
namespace {

void stream_spin_table(
    const std::vector<std::vector<int>>& unique_determinants,
    const AcceptedPairTileProvider& accepted_pair_provider,
    bool alpha_channel,
    bool beta_channel,
    int n_active_orbitals,
    const std::vector<double>& accepted_active_overlap,
    const Eigen::MatrixXd& accepted_active_one_electron,
    const ActiveSpaceTwoElectronResult& accepted_two_electron,
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
  for (int left_begin = 0; left_begin < n_unique;
       left_begin += tile_extent) {
    const int left_end = std::min(n_unique, left_begin + tile_extent);
    for (int right_begin = left_begin; right_begin < n_unique;
         right_begin += tile_extent) {
      const int right_end = std::min(n_unique, right_begin + tile_extent);
      const AcceptedSpinPairTile accepted_tile = accepted_pair_provider.build(
          left_begin,
          left_end,
          right_begin,
          right_end,
          accepted_active_overlap,
          accepted_active_one_electron,
          accepted_two_electron,
          AcceptedPairTileBuildOptions{
              .materialize_projected_pair_values = false,
              .populate_response_payload = true});
      SameSpinDirectionalPairTile same_spin = build_directional_pair_tile(
          unique_determinants,
          accepted_tile,
          n_active_orbitals,
          direction,
          accepted_ri_active_pair_factors != nullptr
              ? &accepted_active_one_electron
              : nullptr,
          accepted_ri_active_pair_factors,
          directional_ri_active_pair_factors);
      const SameSpinDirectionalPairTileView forward_same = same_spin.view();
      std::optional<DirectionalOppositeSpinPairTile> opposite_spin;
      if (include_opposite_spin) {
        opposite_spin.emplace(build_directional_opposite_spin_pair_tile(
            unique_determinants,
            accepted_tile,
            n_active_orbitals,
            accepted_two_electron,
            direction,
            forward_same,
            accepted_ri_active_pair_factors,
            directional_ri_active_pair_factors));
      }
      const std::optional<DirectionalOppositeSpinPairTileView>
          forward_opposite = opposite_spin
              ? std::optional<DirectionalOppositeSpinPairTileView>(
                    opposite_spin->view())
              : std::nullopt;
      consume(
          alpha_channel,
          beta_channel,
          forward_same,
          forward_opposite ? &*forward_opposite : nullptr);
      if (right_begin != left_begin) {
        const SameSpinDirectionalPairTileView reverse_same =
            same_spin.view(true);
        const std::optional<DirectionalOppositeSpinPairTileView>
            reverse_opposite = opposite_spin
                ? std::optional<DirectionalOppositeSpinPairTileView>(
                      opposite_spin->view(true))
                : std::nullopt;
        consume(
            alpha_channel,
            beta_channel,
            reverse_same,
            reverse_opposite ? &*reverse_opposite : nullptr);
      }
    }
  }
}

}  // namespace

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
    const DirectionalPairTileConsumer& consume) {
  if (!accepted_pair_cache.enabled() || !consume) {
    throw std::invalid_argument(
        "directional pair tile stream requires an enabled cache and consumer");
  }
  const bool shared =
      accepted_pair_cache.shares_same_spin_pair_cache_between_spins();
  const PairTileExtents tile_extents = plan_pair_tile_extents(
      accepted_pair_cache,
      n_active_orbitals,
      accepted_two_electron,
      include_opposite_spin);
  stream_spin_table(
      accepted_pair_cache.alpha_reuse_table.unique_determinants,
      accepted_pair_cache.alpha_provider(),
      true,
      shared,
      n_active_orbitals,
      accepted_active_overlap,
      accepted_active_one_electron,
      accepted_two_electron,
      direction,
      accepted_ri_active_pair_factors,
      directional_ri_active_pair_factors,
      include_opposite_spin,
      tile_extents.alpha,
      consume);
  if (shared) {
    return;
  }
  stream_spin_table(
      accepted_pair_cache.beta_reuse_table.unique_determinants,
      accepted_pair_cache.beta_provider(),
      false,
      true,
      n_active_orbitals,
      accepted_active_overlap,
      accepted_active_one_electron,
      accepted_two_electron,
      direction,
      accepted_ri_active_pair_factors,
      directional_ri_active_pair_factors,
      include_opposite_spin,
      tile_extents.beta,
      consume);
}

}  // namespace xmvb::vb::detail
