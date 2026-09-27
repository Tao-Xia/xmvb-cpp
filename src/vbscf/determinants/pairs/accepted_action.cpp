#include "vbscf/determinants/pairs/accepted_action.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "vbscf/determinants/pairs/same_spin_cache.hpp"

namespace xmvb::vb {
namespace {

int plan_scalar_tile_extent(
    const AcceptedPairTileProvider& provider,
    int n_active_orbitals,
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
      1, (full_bound + pair_count - 1) / pair_count);
  const std::size_t maximum_pairs = std::max<std::size_t>(
      1, workspace_bytes / pair_bytes);
  return std::max(
      1,
      std::min(
          n_unique,
          static_cast<int>(std::sqrt(
              static_cast<long double>(maximum_pairs)))));
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
      provider, n_active_orbitals, workspace_bytes);
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
