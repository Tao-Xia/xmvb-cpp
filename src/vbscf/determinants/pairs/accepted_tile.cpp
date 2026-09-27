#include "vbscf/determinants/pairs/accepted_tile.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "core/openmp.hpp"
#include "vbscf/determinants/pairs/woodbury_overlap.hpp"

namespace xmvb::vb {

const SpinDeterminantPairEvaluation& AcceptedSpinPairTile::pair(
    int left_local,
    int right_local) const {
  if (left_local < 0 || left_local >= left_size ||
      right_local < 0 || right_local >= right_size) {
    throw std::out_of_range("accepted pair tile index is out of range");
  }
  return pairs[static_cast<std::size_t>(left_local) * right_size +
      right_local];
}

AcceptedPairTileProvider::AcceptedPairTileProvider(
    std::vector<std::vector<int>> unique_spin_strings,
    int n_active_orbitals,
    double linear_dependence_threshold)
    : unique_spin_strings_(std::move(unique_spin_strings)),
      n_active_orbitals_(n_active_orbitals),
      overlap_resolver_(linear_dependence_threshold),
      pair_evaluator_(overlap_resolver_,
                      DeterminantHamiltonianResolver(overlap_resolver_)) {}

AcceptedSpinPairTile AcceptedPairTileProvider::build(
    int left_begin,
    int left_end,
    int right_begin,
    int right_end,
    const std::vector<double>& active_overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& active_one_electron,
    const ActiveSpaceTwoElectronResult& active_two_electron,
    AcceptedPairTileBuildOptions options) const {
  if (left_begin < 0 || left_end <= left_begin || left_end > size() ||
      right_begin < 0 || right_end <= right_begin || right_end > size()) {
    throw std::invalid_argument("accepted pair tile bounds are invalid");
  }
  if (active_one_electron.rows() != n_active_orbitals_ ||
      active_one_electron.cols() != n_active_orbitals_ ||
      active_overlap.size() != static_cast<std::size_t>(n_active_orbitals_) *
          n_active_orbitals_) {
    throw std::invalid_argument(
        "accepted pair tile active-space dimensions differ");
  }

  AcceptedSpinPairTile tile;
  tile.left_begin = left_begin;
  tile.right_begin = right_begin;
  tile.left_size = left_end - left_begin;
  tile.right_size = right_end - right_begin;
  tile.pairs.resize(
      static_cast<std::size_t>(tile.left_size) * tile.right_size);

  const Eigen::Map<const Eigen::MatrixXd> overlap_map(
      active_overlap.data(), n_active_orbitals_, n_active_orbitals_);
  const int n_threads = std::max(
      1, std::min(xmvb::effective_openmp_thread_count(), tile.left_size));
#pragma omp parallel for schedule(static) if(n_threads > 1) num_threads(n_threads)
  for (int left_local = 0; left_local < tile.left_size; ++left_local) {
    for (int right_local = 0; right_local < tile.right_size; ++right_local) {
      const auto& occupied_left =
          unique_spin_strings_[left_begin + left_local];
      const auto& occupied_right =
          unique_spin_strings_[right_begin + right_local];
      SpinDeterminantPairEvaluation evaluation;
      const std::size_t pair_index =
          static_cast<std::size_t>(left_local) * tile.right_size + right_local;
      if (right_local == 0) {
        evaluation = pair_evaluator_.evaluate_same_spin_pair(
            occupied_left,
            occupied_right,
            active_overlap,
            active_one_electron,
            n_active_orbitals_,
            active_two_electron,
            true);
      } else {
        const std::size_t previous_index = pair_index - 1;
        auto updated_overlap = try_woodbury_right_overlap_update(
            occupied_left,
            unique_spin_strings_[right_begin + right_local - 1],
            occupied_right,
            overlap_map,
            tile.pairs[previous_index].overlap_result);
        if (updated_overlap.has_value()) {
          evaluation = pair_evaluator_.evaluate_same_spin_pair(
              occupied_left,
              occupied_right,
              std::move(*updated_overlap),
              active_one_electron,
              n_active_orbitals_,
              active_two_electron,
              true);
        } else {
          evaluation = pair_evaluator_.evaluate_same_spin_pair(
              occupied_left,
              occupied_right,
              active_overlap,
              active_one_electron,
              n_active_orbitals_,
              active_two_electron,
              true);
        }
      }
      if (options.populate_opposite_spin_projection ||
          options.populate_response_payload) {
        complete_same_spin_pair_evaluation(
            occupied_left,
            occupied_right,
            active_one_electron,
            n_active_orbitals_,
            active_two_electron,
            options.populate_opposite_spin_projection,
            options.materialize_projected_pair_values,
            options.populate_response_payload,
            &evaluation);
      }
      tile.pairs[pair_index] = std::move(evaluation);
    }
  }
  return tile;
}

}  // namespace xmvb::vb
