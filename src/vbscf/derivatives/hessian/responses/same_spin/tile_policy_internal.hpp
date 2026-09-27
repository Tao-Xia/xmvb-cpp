#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "core/openmp.hpp"
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

/**
 * @brief Chooses one square spin-pair tile from an explicit byte model.
 *
 * A directional pair owns its same-spin cofactor response and, when requested,
 * two packed opposite-spin channels. RI additionally needs two auxiliary
 * images and three packed-pair work vectors during the block GEMM. The
 * workspace bound therefore determines the tile area; the worker count only
 * raises the preferred area far enough to expose one pair per OpenMP worker
 * when the byte bound permits it.
 */
inline int plan_pair_tile_extent(
    int n_unique,
    int n_electrons,
    int n_active_orbitals,
    int n_auxiliary,
    bool include_opposite_spin,
    std::size_t workspace_bytes = kPairTileWorkspaceBytes) {
  if (n_unique <= 1) {
    return std::max(1, n_unique);
  }
  const std::size_t occupied_square =
      static_cast<std::size_t>(std::max(0, n_electrons)) *
      static_cast<std::size_t>(std::max(0, n_electrons));
  const std::size_t n_packed_pairs = static_cast<std::size_t>(
      packed_active_pair_count(std::max(0, n_active_orbitals)));

  // Three scalar tile matrices plus two directional occupied-block matrices.
  // Opposite-spin storage adds the occupied overlap block, sparse channel
  // payload, and the raw/projected packed channels. The GEMM term accounts for
  // its largest simultaneous RI block rather than relying on a fixed extent.
  std::size_t doubles_per_pair = 3 + 2 * occupied_square;
  std::size_t integers_per_pair = 0;
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
          integers_per_pair * sizeof(int));

  const int workers = std::max(1, xmvb::effective_openmp_thread_count());
  const std::size_t thread_scratch =
      static_cast<std::size_t>(workers) *
      std::max<std::size_t>(1, 6 * occupied_square + n_packed_pairs) *
      sizeof(double);
  const std::size_t pair_budget = workspace_bytes > thread_scratch
      ? workspace_bytes - thread_scratch
      : bytes_per_pair;
  const std::size_t maximum_pairs = std::max<std::size_t>(
      1, pair_budget / bytes_per_pair);
  int extent = static_cast<int>(std::sqrt(
      static_cast<long double>(maximum_pairs)));
  extent = std::max(1, std::min(n_unique, extent));

  const int parallel_extent = static_cast<int>(std::ceil(std::sqrt(
      static_cast<long double>(std::min<std::size_t>(
          static_cast<std::size_t>(workers),
          static_cast<std::size_t>(n_unique) * n_unique)))));
  if (static_cast<std::size_t>(parallel_extent) * parallel_extent <=
      maximum_pairs) {
    extent = std::max(extent, std::min(n_unique, parallel_extent));
  }
  return extent;
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
    bool include_opposite_spin = true) {
  const auto extent = [&](const std::vector<std::vector<int>>& strings) {
    const int n_electrons = strings.empty()
        ? 0
        : static_cast<int>(strings.front().size());
    return plan_pair_tile_extent(
        static_cast<int>(strings.size()),
        n_electrons,
        n_active_orbitals,
        two_electron.n_auxiliary_functions,
        include_opposite_spin);
  };
  return {
      extent(cache.alpha_reuse_table.unique_determinants),
      extent(cache.beta_reuse_table.unique_determinants)};
}

}  // namespace xmvb::vb::detail
