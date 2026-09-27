#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "vbscf/determinants/pairs/contractions.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"

namespace xmvb::vb {

struct AcceptedPairTileBuildOptions {
  bool materialize_projected_pair_values = false;
  bool populate_response_payload = false;
  bool populate_opposite_spin_projection = true;
};

/** RI integral direction consumed while accepted traversal channels are live. */
struct AcceptedPairRiDirectionView {
  const std::vector<double>& active_overlap;
  const std::vector<double>& active_one_electron;
  const Eigen::MatrixXd& active_pair_factors;
  bool project_first_cofactor = false;
};

/** Regular-pair directional sidecar with the same ordering as an accepted tile. */
struct AcceptedPairRiDirectionTile {
  std::vector<RegularRiSameSpinDirection> pairs;
  std::vector<unsigned char> ready;
};

/** Bounded accepted-point pair payload for one rectangular spin-string tile. */
struct AcceptedSpinPairTile {
  int left_begin = 0;
  int right_begin = 0;
  int left_size = 0;
  int right_size = 0;
  std::vector<SpinDeterminantPairEvaluation> pairs;

  const SpinDeterminantPairEvaluation& pair(
      int left_local,
      int right_local) const;
};

/**
 * @brief Generates accepted same-spin pairs without persistent quadratic storage.
 *
 * Each tile row starts from an exactly factorized pair and traverses the right
 * strings with certified Woodbury overlap updates. Singular, ill-conditioned,
 * or non-beneficial updates are factorized exactly and become the next anchor.
 * Tiling therefore bounds memory independently of update success.
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
      AcceptedPairTileBuildOptions options = {},
      const AcceptedPairRiDirectionView* direction = nullptr,
      AcceptedPairRiDirectionTile* direction_tile = nullptr) const;

 private:
  std::vector<std::vector<int>> unique_spin_strings_;
  int n_active_orbitals_ = 0;
  DeterminantOverlapResolver overlap_resolver_;
  DeterminantPairEvaluator pair_evaluator_;
};

}  // namespace xmvb::vb
