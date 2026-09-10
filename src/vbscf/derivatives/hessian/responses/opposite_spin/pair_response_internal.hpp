#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/responses/same_spin/backward.hpp"
#include "vbscf/determinants/determinant_pair_evaluator.hpp"
#include "vbscf/integrals/active/two_electron/construction/result.hpp"

namespace xmvb::vb::detail {

/** Directional cofactor data for one ordered unique-spin pair. */
struct DirectionalOppositeSpinPairData {
  Eigen::MatrixXd delta_overlap_submatrix;
  OppositeSpinPackedPairProjection delta_first_order_cofactor_projection;
};

std::vector<DirectionalOppositeSpinPairData>
build_directional_opposite_spin_pair_data(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    int n_unique_determinants,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const ActiveSpaceIntegralDirectionView& direction,
    const std::vector<SameSpinPolynomialDirectionalPairData>&
        precomputed_directional_pair_data);

}  // namespace xmvb::vb::detail
