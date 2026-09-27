#include "vbscf/derivatives/hessian/responses/same_spin/pair_tile_stream_internal.hpp"

#include <algorithm>
#include <optional>
#include <stdexcept>

#include "vbscf/derivatives/hessian/responses/same_spin/tile_policy_internal.hpp"

namespace xmvb::vb::detail {
namespace {

void stream_spin_table(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<SpinDeterminantPairEvaluation>& accepted_pairs,
    bool alpha_channel,
    bool beta_channel,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& accepted_two_electron,
    const ActiveSpaceIntegralDirectionView& direction,
    const Eigen::MatrixXd* accepted_active_one_electron,
    const Eigen::MatrixXd* accepted_ri_active_pair_factors,
    const Eigen::MatrixXd* directional_ri_active_pair_factors,
    bool include_opposite_spin,
    const DirectionalPairTileConsumer& consume) {
  const int n_unique = static_cast<int>(unique_determinants.size());
  if (n_unique <= 0 ||
      accepted_pairs.size() !=
          static_cast<std::size_t>(n_unique) * n_unique) {
    throw std::invalid_argument(
        "directional pair tile stream requires a complete accepted pair table");
  }
  const int tile_extent = std::min(n_unique, kSameSpinTileExtent);
  for (int left_begin = 0; left_begin < n_unique;
       left_begin += tile_extent) {
    const int left_end = std::min(n_unique, left_begin + tile_extent);
    for (int right_begin = 0; right_begin < n_unique;
         right_begin += tile_extent) {
      const int right_end = std::min(n_unique, right_begin + tile_extent);
      SameSpinDirectionalPairTile same_spin = build_directional_pair_tile(
          unique_determinants,
          accepted_pairs,
          n_unique,
          n_active_orbitals,
          direction,
          left_begin,
          left_end,
          right_begin,
          right_end,
          accepted_active_one_electron,
          accepted_ri_active_pair_factors,
          directional_ri_active_pair_factors);
      std::optional<DirectionalOppositeSpinPairTile> opposite_spin;
      if (include_opposite_spin) {
        opposite_spin.emplace(build_directional_opposite_spin_pair_tile(
            unique_determinants,
            accepted_pairs,
            n_unique,
            n_active_orbitals,
            accepted_two_electron,
            direction,
            same_spin,
            accepted_ri_active_pair_factors,
            directional_ri_active_pair_factors));
      }
      consume(
          alpha_channel,
          beta_channel,
          same_spin,
          opposite_spin ? &*opposite_spin : nullptr);
    }
  }
}

}  // namespace

void stream_directional_pair_tiles(
    const SameSpinPairCacheContext& accepted_pair_cache,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& accepted_two_electron,
    const ActiveSpaceIntegralDirectionView& direction,
    const Eigen::MatrixXd* accepted_active_one_electron,
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
  stream_spin_table(
      accepted_pair_cache.alpha_reuse_table.unique_determinants,
      accepted_pair_cache.alpha_pair_cache_ref(),
      true,
      shared,
      n_active_orbitals,
      accepted_two_electron,
      direction,
      accepted_active_one_electron,
      accepted_ri_active_pair_factors,
      directional_ri_active_pair_factors,
      include_opposite_spin,
      consume);
  if (shared) {
    return;
  }
  stream_spin_table(
      accepted_pair_cache.beta_reuse_table.unique_determinants,
      accepted_pair_cache.beta_pair_cache_ref(),
      false,
      true,
      n_active_orbitals,
      accepted_two_electron,
      direction,
      accepted_active_one_electron,
      accepted_ri_active_pair_factors,
      directional_ri_active_pair_factors,
      include_opposite_spin,
      consume);
}

}  // namespace xmvb::vb::detail
