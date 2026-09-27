#include "vbscf/determinants/pairs/accepted_action.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "core/openmp.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"

namespace xmvb::vb {
namespace {

int plan_scalar_tile_extent(
    const AcceptedPairTileProvider& provider,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& active_two_electron,
    std::size_t workspace_bytes) {
  const int n_unique = provider.size();
  if (n_unique <= 1) {
    return std::max(1, n_unique);
  }
  const std::size_t full_bound = estimate_same_spin_pair_cache_bytes(
      provider.unique_spin_strings(), n_active_orbitals);
  const std::size_t pair_count =
      static_cast<std::size_t>(n_unique) * n_unique;
  const std::size_t pair_bytes = std::max<std::size_t>(
      1,
      (full_bound + pair_count - 1) / pair_count +
          2 * sizeof(double));
  const bool direct_ri = uses_direct_ri_pair_factors(active_two_electron);
  const int n_electrons = provider.unique_spin_strings().empty()
      ? 0
      : static_cast<int>(provider.unique_spin_strings().front().size());
  const std::size_t occupied_square =
      static_cast<std::size_t>(n_electrons) * n_electrons;
  const std::size_t state_bytes_per_worker = direct_ri
      ? (static_cast<std::size_t>(
             active_two_electron.n_auxiliary_functions) *
             (occupied_square + 1) +
         occupied_square) *
            sizeof(double)
      : 0;
  const int workers = std::max(1, xmvb::effective_openmp_thread_count());
  const auto fits = [&](int extent) {
    const long double size = static_cast<long double>(extent);
    const int live_workers = std::min(workers, extent);
    const long double bytes =
        static_cast<long double>(pair_bytes) * size * size +
        static_cast<long double>(live_workers) * state_bytes_per_worker;
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

}  // namespace

AcceptedSpinPairActionResult apply_accepted_spin_pair_action(
    const AcceptedPairTileProvider& provider,
    const std::vector<double>& active_overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& active_one_electron,
    const ActiveSpaceTwoElectronResult& active_two_electron,
    const Eigen::Ref<const Eigen::MatrixXd>& vectors,
    std::size_t workspace_bytes) {
  const int n_unique = provider.size();
  const int n_active_orbitals = static_cast<int>(active_one_electron.rows());
  if (n_unique <= 0 || vectors.rows() != n_unique || vectors.cols() <= 0 ||
      active_one_electron.cols() != n_active_orbitals) {
    throw std::invalid_argument(
        "accepted spin-pair action dimensions are inconsistent");
  }

  AcceptedSpinPairActionResult result{
      Eigen::MatrixXd::Zero(n_unique, vectors.cols()),
      Eigen::MatrixXd::Zero(n_unique, vectors.cols())};
  const int extent = plan_scalar_tile_extent(
      provider,
      n_active_orbitals,
      active_two_electron,
      workspace_bytes);
  for (int left_begin = 0; left_begin < n_unique; left_begin += extent) {
    const int left_end = std::min(n_unique, left_begin + extent);
    for (int right_begin = 0; right_begin < n_unique; right_begin += extent) {
      const int right_end = std::min(n_unique, right_begin + extent);
      const AcceptedSpinPairTile tile = provider.build(
          left_begin,
          left_end,
          right_begin,
          right_end,
          active_overlap,
          active_one_electron,
          active_two_electron,
          AcceptedPairTileBuildOptions{
              .materialize_projected_pair_values = false,
              .populate_response_payload = false,
              .populate_opposite_spin_projection = false});
      Eigen::MatrixXd overlap(tile.left_size, tile.right_size);
      Eigen::MatrixXd hamiltonian(tile.left_size, tile.right_size);
      for (int left = 0; left < tile.left_size; ++left) {
        for (int right = 0; right < tile.right_size; ++right) {
          const auto& pair = tile.pair(left, right);
          overlap(left, right) = pair.overlap_result.overlap_determinant;
          hamiltonian(left, right) = pair.total_hamiltonian;
        }
      }
      const auto input = vectors.middleRows(
          right_begin, tile.right_size);
      result.overlap.middleRows(left_begin, tile.left_size).noalias() +=
          overlap * input;
      result.hamiltonian.middleRows(left_begin, tile.left_size).noalias() +=
          hamiltonian * input;
    }
  }
  return result;
}

}  // namespace xmvb::vb
