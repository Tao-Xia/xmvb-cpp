#include "vbscf/determinants/pairs/accepted_tile.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "core/openmp.hpp"
#include "vbscf/determinants/pairs/contractions.hpp"
#include "vbscf/determinants/pairs/ri_update.hpp"
#include "vbscf/determinants/pairs/traversal.hpp"
#include "vbscf/determinants/pairs/woodbury_overlap.hpp"
#include "vbscf/integrals/active/two_electron/construction/kernel.hpp"

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
  const bool direct_ri =
      active_two_electron.representation ==
          ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity &&
      active_two_electron.packed_active_two_electron_integrals.empty() &&
      active_two_electron.ri_active_pair_factors.size() != 0;
  if (direct_ri &&
      active_two_electron.ri_active_pair_factors.cols() !=
          packed_active_pair_count(n_active_orbitals_)) {
    throw std::invalid_argument(
        "RI active-pair factor dimensions are inconsistent");
  }
  const int n_threads = std::max(
      1, std::min(xmvb::effective_openmp_thread_count(), tile.left_size));
#pragma omp parallel for schedule(static) if(n_threads > 1) num_threads(n_threads)
  for (int left_local = 0; left_local < tile.left_size; ++left_local) {
    const int left_index = left_begin + left_local;
    RiPairUpdateState ri_state;
    const std::vector<int> right_traversal = build_pair_update_traversal(
        unique_spin_strings_, left_index, right_begin, right_end);
    for (int traversal_index = 0;
         traversal_index < tile.right_size;
         ++traversal_index) {
      const int right_index = right_traversal[traversal_index];
      const int right_local = right_index - right_begin;
      const auto& occupied_left =
          unique_spin_strings_[left_index];
      const auto& occupied_right =
          unique_spin_strings_[right_index];
      SpinDeterminantPairEvaluation evaluation;
      const std::size_t pair_index =
          static_cast<std::size_t>(left_local) * tile.right_size + right_local;
      DeterminantOverlapResult overlap_result;
      bool used_low_rank_overlap = false;
      int previous_right = -1;
      std::size_t previous_index = 0;
      if (traversal_index == 0) {
        overlap_result = overlap_resolver_.resolve_matrix(
            build_overlap_submatrix(
                occupied_left, occupied_right, overlap_map));
      } else {
        previous_right = right_traversal[traversal_index - 1];
        previous_index =
            static_cast<std::size_t>(left_local) * tile.right_size +
            previous_right - right_begin;
        auto updated_overlap = try_woodbury_right_overlap_update(
            occupied_left,
            unique_spin_strings_[previous_right],
            occupied_right,
            overlap_map,
            tile.pairs[previous_index].overlap_result);
        if (updated_overlap.has_value()) {
          overlap_result = std::move(*updated_overlap);
          used_low_rank_overlap = true;
        } else {
          overlap_result = overlap_resolver_.resolve_matrix(
              build_overlap_submatrix(
                  occupied_left, occupied_right, overlap_map));
        }
      }

      bool used_ri_update = false;
      if (direct_ri && overlap_result.nullity == 0 &&
          overlap_result.overlap_determinant != 0.0) {
        if (used_low_rank_overlap && ri_state.valid()) {
          used_ri_update = ri_state.update_right(
              occupied_left,
              unique_spin_strings_[previous_right],
              occupied_right,
              tile.pairs[previous_index].overlap_result,
              overlap_result,
              active_two_electron.ri_active_pair_factors);
        }
        if (!used_ri_update) {
          used_ri_update = ri_state.initialize(
              occupied_left,
              occupied_right,
              overlap_result,
              active_two_electron.ri_active_pair_factors);
        }
      } else {
        ri_state.reset();
      }

      if (used_ri_update) {
        evaluation = pair_evaluator_.evaluate_regular_same_spin_pair(
            occupied_left,
            occupied_right,
            std::move(overlap_result),
            active_one_electron,
            ri_state.two_electron_phi(),
            options.populate_response_payload);
      } else {
        evaluation = pair_evaluator_.evaluate_same_spin_pair(
            occupied_left,
            occupied_right,
            std::move(overlap_result),
            active_one_electron,
            n_active_orbitals_,
            active_two_electron,
            options.populate_response_payload);
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
