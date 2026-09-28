#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/responses/same_spin/backward.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/pair_response_internal.hpp"
#include "vbscf/determinants/pairs/accepted_tile.hpp"
#include "vbscf/determinants/pairs/evaluator.hpp"
#include "vbscf/integrals/active/two_electron/construction/result.hpp"

namespace xmvb::vb::detail {

class DirectionalOppositeSpinPairTileView;

/** Directional cofactor data for one ordered unique-spin pair. */
struct DirectionalOppositeSpinPairData {
  Eigen::MatrixXd delta_overlap_submatrix;
  OppositeSpinPackedPairProjection delta_first_order_cofactor_projection;
};

/** Directional opposite-spin data local to one unique-spin pair tile. */
struct DirectionalOppositeSpinPairTile {
  int left_begin = 0;
  int right_begin = 0;
  int left_size = 0;
  int right_size = 0;
  std::vector<DirectionalOppositeSpinPairData> pairs;
  /** Pair-major contiguous storage, with one packed channel per column. */
  Eigen::MatrixXd raw_channel_values;
  Eigen::MatrixXd projected_channel_values;

  const DirectionalOppositeSpinPairData& pair(
      int left_local,
      int right_local) const;

  DirectionalOppositeSpinPairTileView view(bool transposed = false) const;
};

/** Zero-copy orientation view of one canonical opposite-spin tile. */
class DirectionalOppositeSpinPairTileView {
public:
  using Stride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
  using ConstChannelMap = Eigen::Map<
      const Eigen::MatrixXd, Eigen::Unaligned, Stride>;

  DirectionalOppositeSpinPairTileView(
      const DirectionalOppositeSpinPairTile& storage,
      bool transposed) noexcept
      : storage_(&storage), transposed_(transposed) {}

  int left_begin() const noexcept;
  int right_begin() const noexcept;
  int left_size() const noexcept;
  int right_size() const noexcept;
  ConstChannelMap raw_channel(int packed_pair) const;
  ConstChannelMap projected_channel(int packed_pair) const;

  template <typename Consumer>
  void with_pair(
      int left_local,
      int right_local,
      Consumer&& consume) const {
    if (!transposed_) {
      consume(storage_->pair(left_local, right_local));
      return;
    }
    DirectionalOppositeSpinPairData pair =
        storage_->pair(right_local, left_local);
    pair.delta_overlap_submatrix.transposeInPlace();
    consume(pair);
  }

private:
  ConstChannelMap channel_view(
      const Eigen::MatrixXd& channels,
      int packed_pair) const;

  const DirectionalOppositeSpinPairTile* storage_ = nullptr;
  bool transposed_ = false;
};

/**
 * @brief Builds one opposite-spin response tile from its same-spin cofactor tile.
 *
 * The RI path batches packed-pair projections and evaluates
 * `K delta_X + delta_K X` with matrix products. Its temporary column block is
 * chosen so that the dense GEMM workspace is no larger than one accepted RI
 * factor matrix.
 */
DirectionalOppositeSpinPairTile build_directional_opposite_spin_pair_tile(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    int n_unique_determinants,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const ActiveSpaceIntegralDirectionView& direction,
    const SameSpinDirectionalPairTileView& same_spin_tile,
    const Eigen::MatrixXd* accepted_ri_active_pair_factors = nullptr,
    const Eigen::MatrixXd* directional_ri_active_pair_factors = nullptr);

DirectionalOppositeSpinPairTile build_directional_opposite_spin_pair_tile(
    const std::vector<std::vector<int>>& unique_determinants,
    const AcceptedSpinPairTile& accepted_pair_tile,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const ActiveSpaceIntegralDirectionView& direction,
    const SameSpinDirectionalPairTileView& same_spin_tile,
    const Eigen::MatrixXd* accepted_ri_active_pair_factors = nullptr,
    const Eigen::MatrixXd* directional_ri_active_pair_factors = nullptr);

}  // namespace xmvb::vb::detail
