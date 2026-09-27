#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "vbscf/determinants/pairs/same_spin_cache.hpp"

namespace xmvb::vb {

struct AcceptedPairTileBuildOptions {
  bool materialize_projected_pair_values = false;
  bool populate_response_payload = false;
};

struct AcceptedPairTileStatistics {
  std::size_t anchors = 0;
  std::size_t woodbury_updates = 0;
  std::size_t certified_reanchors = 0;
};

/** Bounded accepted-point pair payload for one rectangular spin-string tile. */
struct AcceptedSpinPairTile {
  int left_begin = 0;
  int right_begin = 0;
  int left_size = 0;
  int right_size = 0;
  std::vector<SpinDeterminantPairEvaluation> pairs;
  AcceptedPairTileStatistics statistics;

  const SpinDeterminantPairEvaluation& pair(
      int left_local,
      int right_local) const;
};

/**
 * @brief Generates accepted same-spin pairs without persistent quadratic storage.
 *
 * The provider builds a single-substitution graph once. Within each requested
 * tile, roots are factorized independently and every certified regular child
 * overlap is obtained from its parent by a rank-one Woodbury update. Singular
 * or uncertified children are exact re-anchors. Integral contractions remain
 * representation-native: RI packed features are rebuilt directly rather than
 * carrying the slower full auxiliary matrix state.
 */
class AcceptedPairTileProvider {
 public:
  AcceptedPairTileProvider(
      std::vector<std::vector<int>> unique_spin_strings,
      int n_active_orbitals,
      double linear_dependence_threshold = 1.0e-12);

  int size() const noexcept {
    return static_cast<int>(unique_spin_strings_.size());
  }

  const std::vector<std::vector<int>>& unique_spin_strings() const noexcept {
    return unique_spin_strings_;
  }

  AcceptedSpinPairTile build(
      int left_begin,
      int left_end,
      int right_begin,
      int right_end,
      const std::vector<double>& active_overlap,
      const Eigen::Ref<const Eigen::MatrixXd>& active_one_electron,
      const ActiveSpaceTwoElectronResult& active_two_electron,
      AcceptedPairTileBuildOptions options = {}) const;

 private:
  std::vector<std::vector<int>> unique_spin_strings_;
  std::vector<std::vector<int>> substitution_graph_;
  int n_active_orbitals_ = 0;
  DeterminantOverlapResolver overlap_resolver_;
  DeterminantPairEvaluator pair_evaluator_;
};

}  // namespace xmvb::vb
