#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/responses/same_spin/backward.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/pair_response_internal.hpp"
#include "vbscf/determinants/pairs/evaluator.hpp"
#include "vbscf/integrals/active/two_electron/construction/result.hpp"

namespace xmvb::vb::detail {

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

  const DirectionalOppositeSpinPairData& pair(
      int left_local,
      int right_local) const;
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
    const SameSpinDirectionalPairTile& same_spin_tile,
    const Eigen::MatrixXd* accepted_ri_active_pair_factors = nullptr,
    const Eigen::MatrixXd* directional_ri_active_pair_factors = nullptr);

std::vector<DirectionalOppositeSpinPairData>
build_directional_opposite_spin_pair_data(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    int n_unique_determinants,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const ActiveSpaceIntegralDirectionView& direction,
    const std::vector<SameSpinPolynomialDirectionalPairData>&
        precomputed_directional_pair_data,
    const Eigen::MatrixXd* accepted_ri_active_pair_factors = nullptr,
    const Eigen::MatrixXd* directional_ri_active_pair_factors = nullptr);

}  // namespace xmvb::vb::detail
