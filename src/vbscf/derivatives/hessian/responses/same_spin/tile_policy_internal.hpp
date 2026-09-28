#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "core/openmp.hpp"
#include "vbscf/determinants/pairs/accepted_tile.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "vbscf/integrals/active/two_electron/construction/kernel.hpp"

namespace xmvb::vb::detail {

/** Process-wide cap for one directional pair-tile sweep. */
inline constexpr std::size_t kPairTileWorkspaceBytes =
    256ULL * 1024ULL * 1024ULL;

struct PairTileExtents {
  int alpha = 1;
  int beta = 1;
};

/** Tile sizes and nested accepted-action workspaces for a local response. */
struct LocalResponseTilePlan {
  PairTileExtents extents;
  std::size_t alpha_partner_action_bytes = 1;
  std::size_t beta_partner_action_bytes = 1;
};

/** Outer and nested partner extents for opposite-spin backward sweeps. */
struct OppositeSpinTilePlan {
  PairTileExtents primary;
  PairTileExtents partner;
};

inline std::size_t accepted_scalar_pair_bytes(
    const AcceptedPairTileProvider& provider,
    int n_active_orbitals) {
  const std::size_t n_unique = static_cast<std::size_t>(
      std::max(1, provider.size()));
  const std::size_t pair_count = n_unique * n_unique;
  const std::size_t provider_bytes = estimate_same_spin_pair_cache_bytes(
      provider.unique_spin_strings(), n_active_orbitals);
  return std::max<std::size_t>(
      1,
      (provider_bytes + pair_count - 1) / pair_count +
          2 * sizeof(double));
}

inline int parallel_scalar_action_extent(
    const AcceptedPairTileProvider& provider) {
  const std::size_t workers = static_cast<std::size_t>(
      std::max(1, xmvb::effective_openmp_thread_count()));
  const int worker_extent = static_cast<int>(std::ceil(
      std::sqrt(static_cast<long double>(workers))));
  return std::min(
      std::max(1, provider.size()),
      worker_extent);
}

inline std::size_t parallel_scalar_action_bytes(
    const AcceptedPairTileProvider& provider,
    int n_active_orbitals) {
  const std::size_t n_unique = static_cast<std::size_t>(
      std::max(1, provider.size()));
  const std::size_t parallel_extent = static_cast<std::size_t>(
      parallel_scalar_action_extent(provider));
  const std::size_t parallel_pairs = std::min<std::size_t>(
      n_unique * n_unique,
      parallel_extent * parallel_extent);
  return accepted_scalar_pair_bytes(provider, n_active_orbitals) *
      parallel_pairs;
}

/**
 * @brief Chooses one square spin-pair tile from an explicit byte model.
 *
 * A streamed pair owns both the accepted-point evaluation and its directional
 * response.  The accepted payload includes the inverse/cofactor state,
 * same-spin response tensors, and sparse opposite-spin projection.  RI
 * additionally needs two auxiliary images and three packed-pair work vectors
 * during the block GEMM. A direct-RI Woodbury traversal also owns one
 * two `N_aux x n_electron^2` tables (`A_Q` and `A_Q K`) per live worker.
 * A consumer that
 * requests projected cofactor images additionally uses one bounded
 * `N_aux x tile_extent` row panel and its packed-pair image.  The workspace
 * bound therefore determines both the tile area and the number of
 * simultaneously live row states.
 */
inline int plan_pair_tile_extent(
    int n_unique,
    int n_electrons,
    int n_active_orbitals,
    int n_auxiliary,
    bool include_opposite_spin,
    std::size_t workspace_bytes = kPairTileWorkspaceBytes,
    std::size_t bytes_per_primary = 0,
    std::size_t reserved_bytes = 0,
    std::size_t additional_bytes_per_pair = 0,
    bool include_ri_channel_state = false) {
  if (n_unique <= 1) {
    return std::max(1, n_unique);
  }
  const std::size_t occupied_square =
      static_cast<std::size_t>(std::max(0, n_electrons)) *
      static_cast<std::size_t>(std::max(0, n_electrons));
  const std::size_t n_packed_pairs = static_cast<std::size_t>(
      packed_active_pair_count(std::max(0, n_active_orbitals)));

  const std::size_t occupied_pair_count =
      static_cast<std::size_t>(std::max(0, n_electrons)) *
      static_cast<std::size_t>(std::max(0, n_electrons - 1)) / 2;

  // Accepted-point pair payload. This is the same conservative model used by
  // `estimate_same_spin_pair_cache_bytes`, expressed per pair. It deliberately
  // includes projected vectors even when the current provider omits them, so
  // a future payload option cannot silently invalidate the workspace bound.
  std::size_t doubles_per_pair =
      9 * occupied_square + static_cast<std::size_t>(n_electrons) +
      2 * n_packed_pairs + 4 * occupied_pair_count * occupied_pair_count;
  std::size_t integers_per_pair = 4 * occupied_square;
  const std::size_t object_bytes_per_pair =
      sizeof(SpinDeterminantPairEvaluation);

  // Three scalar directional matrices plus two directional occupied blocks.
  // Opposite-spin storage adds the occupied overlap block, sparse channel
  // payload, and the raw/projected packed channels. The GEMM term accounts for
  // its largest simultaneous RI block rather than relying on a fixed extent.
  doubles_per_pair += 3 + 2 * occupied_square;
  if (include_opposite_spin) {
    doubles_per_pair += occupied_square + 2 * n_packed_pairs;
    integers_per_pair += occupied_square;
    if (n_auxiliary > 0) {
      doubles_per_pair += 3 * n_packed_pairs +
          2 * static_cast<std::size_t>(n_auxiliary);
    } else {
      doubles_per_pair += n_packed_pairs;
    }
  }
  const std::size_t bytes_per_pair = std::max<std::size_t>(
      1,
      doubles_per_pair * sizeof(double) +
          integers_per_pair * sizeof(int) + object_bytes_per_pair +
          additional_bytes_per_pair);

  const int workers = std::max(1, xmvb::effective_openmp_thread_count());
  std::size_t scratch_doubles = 6 * occupied_square + n_packed_pairs;
  if (include_ri_channel_state && n_auxiliary > 0) {
    scratch_doubles +=
        static_cast<std::size_t>(n_auxiliary) *
            (2 * occupied_square + 1) +
        occupied_square;
  }
  const std::size_t scratch_bytes_per_worker =
      std::max<std::size_t>(1, scratch_doubles) * sizeof(double);
  const std::size_t bounded_reserved_bytes =
      std::min(reserved_bytes, workspace_bytes);
  const auto fits = [&](int extent) {
    const long double size = static_cast<long double>(extent);
    const int live_workers = std::min(workers, extent);
    const long double ri_row_panel_bytes =
        include_ri_channel_state && n_auxiliary > 0
        ? static_cast<long double>(live_workers) * size *
            static_cast<long double>(n_auxiliary + n_packed_pairs) *
            sizeof(double)
        : 0.0L;
    const long double bytes =
        static_cast<long double>(bytes_per_pair) * size * size +
        static_cast<long double>(bytes_per_primary) * size +
        static_cast<long double>(bounded_reserved_bytes) +
        static_cast<long double>(live_workers) *
            scratch_bytes_per_worker +
        ri_row_panel_bytes;
    return bytes <= static_cast<long double>(workspace_bytes);
  };
  int lower = 1;
  int upper = n_unique;
  while (lower < upper) {
    const int middle = lower + (upper - lower + 1) / 2;
    if (fits(middle)) {
      lower = middle;
    } else {
      upper = middle - 1;
    }
  }
  return lower;
}

/** Chooses an extent when independent tiles, rather than pairs, are parallel. */
inline int plan_parallel_pair_tile_extent(
    int n_unique,
    int n_electrons,
    int n_active_orbitals) {
  const int memory_extent = plan_pair_tile_extent(
      n_unique,
      n_electrons,
      n_active_orbitals,
      0,
      false);
  const int workers = std::max(1, xmvb::effective_openmp_thread_count());
  const int tiles_per_axis = std::max(
      1,
      static_cast<int>(std::ceil(std::sqrt(
          static_cast<long double>(workers)))));
  const int parallel_extent = std::max(
      1, (n_unique + tiles_per_axis - 1) / tiles_per_axis);
  return std::min(memory_extent, parallel_extent);
}

inline PairTileExtents plan_pair_tile_extents(
    const SameSpinPairCacheContext& cache,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& two_electron,
    bool include_opposite_spin = true,
    std::size_t workspace_bytes = kPairTileWorkspaceBytes) {
  const bool direct_ri = uses_direct_ri_pair_factors(two_electron);
  const auto extent = [&](const std::vector<std::vector<int>>& strings) {
    const int n_electrons = strings.empty()
        ? 0
        : static_cast<int>(strings.front().size());
    return plan_pair_tile_extent(
        static_cast<int>(strings.size()),
        n_electrons,
        n_active_orbitals,
        two_electron.n_auxiliary_functions,
        include_opposite_spin,
        workspace_bytes,
        0,
        0,
        0,
        direct_ri);
  };
  return {
      extent(cache.alpha_reuse_table.unique_determinants),
      extent(cache.beta_reuse_table.unique_determinants)};
}

/**
 * @brief Bounds the complete nested local-response tile lifetime.
 *
 * A primary extent \c T keeps the directional pair payload, three dense
 * partner panels (input, S image, H image), and one accepted scalar pair tile
 * alive simultaneously.  The scalar tile reserves one pair per available
 * worker when possible; the remaining budget maximizes \c T because larger
 * primary panels reduce repeated partner-action sweeps.
 */
inline LocalResponseTilePlan plan_local_response_tiles(
    const SameSpinPairCacheContext& cache,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& two_electron,
    int n_states,
    int action_vector_sets = 1,
    std::size_t workspace_bytes = kPairTileWorkspaceBytes) {
  const bool direct_ri = uses_direct_ri_pair_factors(two_electron);
  const int state_count = std::max(1, n_states);
  const int vector_set_count = std::max(1, action_vector_sets);
  const int n_alpha = static_cast<int>(
      cache.alpha_reuse_table.unique_determinants.size());
  const int n_beta = static_cast<int>(
      cache.beta_reuse_table.unique_determinants.size());
  const std::size_t opposite_workspace_per_pair =
      2 * static_cast<std::size_t>(
              packed_active_pair_count(std::max(0, n_active_orbitals))) *
          sizeof(double);
  const std::size_t alpha_action_bytes = std::min(
      workspace_bytes,
      parallel_scalar_action_bytes(
          cache.beta_provider(), n_active_orbitals));
  const std::size_t beta_action_bytes = std::min(
      workspace_bytes,
      parallel_scalar_action_bytes(
          cache.alpha_provider(), n_active_orbitals));
  const auto extent = [&](
                          const std::vector<std::vector<int>>& strings,
                          int partner_size,
                          std::size_t action_bytes) {
    const int n_electrons = strings.empty()
        ? 0
        : static_cast<int>(strings.front().size());
    const std::size_t panel_bytes_per_primary =
        3 * static_cast<std::size_t>(vector_set_count) *
        static_cast<std::size_t>(state_count) *
        static_cast<std::size_t>(std::max(1, partner_size)) * sizeof(double);
    return plan_pair_tile_extent(
        static_cast<int>(strings.size()),
        n_electrons,
        n_active_orbitals,
        two_electron.n_auxiliary_functions,
        true,
        workspace_bytes,
        panel_bytes_per_primary,
        action_bytes,
        opposite_workspace_per_pair,
        direct_ri);
  };
  return {
      {extent(
           cache.alpha_reuse_table.unique_determinants,
           n_beta,
           alpha_action_bytes),
       extent(
           cache.beta_reuse_table.unique_determinants,
           n_alpha,
           beta_action_bytes)},
      alpha_action_bytes,
      beta_action_bytes};
}

/** Plans two simultaneously live spin-pair tiles and their channel images. */
inline OppositeSpinTilePlan plan_opposite_spin_backward_tiles(
    const SameSpinPairCacheContext& cache,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& two_electron,
    std::size_t workspace_bytes = kPairTileWorkspaceBytes) {
  const bool direct_ri = uses_direct_ri_pair_factors(two_electron);
  const std::size_t alpha_partner_bytes = std::min(
      workspace_bytes,
      parallel_scalar_action_bytes(
          cache.beta_provider(), n_active_orbitals));
  const std::size_t beta_partner_bytes = std::min(
      workspace_bytes,
      parallel_scalar_action_bytes(
          cache.alpha_provider(), n_active_orbitals));
  const std::size_t channel_bytes_per_primary_pair =
      2 * static_cast<std::size_t>(
              packed_active_pair_count(std::max(0, n_active_orbitals))) *
          sizeof(double);
  const auto extent = [&](
                          const std::vector<std::vector<int>>& strings,
                          std::size_t bytes,
                          std::size_t reserved,
                          std::size_t extra_per_pair) {
    const int n_electrons = strings.empty()
        ? 0
        : static_cast<int>(strings.front().size());
    return plan_pair_tile_extent(
        static_cast<int>(strings.size()),
        n_electrons,
        n_active_orbitals,
        two_electron.n_auxiliary_functions,
        true,
        bytes,
        0,
        reserved,
        extra_per_pair,
        direct_ri);
  };
  return {
      {extent(
           cache.alpha_reuse_table.unique_determinants,
           workspace_bytes,
           alpha_partner_bytes,
           channel_bytes_per_primary_pair),
       extent(
           cache.beta_reuse_table.unique_determinants,
           workspace_bytes,
           beta_partner_bytes,
           channel_bytes_per_primary_pair)},
      {parallel_scalar_action_extent(cache.alpha_provider()),
       parallel_scalar_action_extent(cache.beta_provider())}};
}

}  // namespace xmvb::vb::detail
