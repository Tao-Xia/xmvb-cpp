#pragma once

#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "vb/matrices/full_determinant_pair_evaluator.hpp"
#include "vb/matrices/same_spin_pair_cache.hpp"

namespace xmvb::vb {

/**
 * @brief Matrix-form opposite-spin channels for one spin sector.
 *
 * Matrix dimensions and indexing conventions:
 * - Ordered unique determinant pairs are indexed by `(left_unique, right_unique)`.
 * - All dense/sparse matrices in this struct use:
 *   - row index = `left_unique`
 *   - column index = `right_unique`
 * - `k` in `*_by_packed_pair[k]` is the active packed-pair index.
 */
struct OppositeSpinPerSpinMatrixChannels {
  /**
   * @brief Number of unique determinants in this spin sector.
   */
  int n_unique_determinants = 0;

  /**
   * @brief Number of ordered unique determinant pairs (`n_unique^2`).
   */
  int n_ordered_unique_pairs = 0;

  /**
   * @brief Number of active packed pairs used by opposite-spin cache payloads.
   */
  int n_packed_active_pairs = 0;

  /**
   * @brief Dense overlap-determinant matrix over ordered unique pairs.
   *
   * Entry `(left_unique, right_unique)` stores
   * `pair_cache(left_unique, right_unique).overlap_result.overlap_determinant`.
   */
  Eigen::MatrixXd overlap_determinant_matrix;

  /**
   * @brief Sparse first-order coefficient matrices grouped by packed-pair index.
   *
   * For each packed-pair index `k`, `first_order_sparse_by_packed_pair[k]` is
   * a sparse matrix over ordered unique pairs:
   * `M_k(left_unique, right_unique) = u_{left,right}(k)`.
   */
  std::vector<Eigen::SparseMatrix<double, Eigen::ColMajor, int>> first_order_sparse_by_packed_pair;

  /**
   * @brief Sparse inverse-overlap coefficient matrices grouped by packed pair.
   *
   * For each packed-pair index `k`,
   * `inverse_overlap_sparse_by_packed_pair[k]` stores
   * `x_{left,right}(k)` over ordered unique pairs.
   */
  std::vector<Eigen::SparseMatrix<double, Eigen::ColMajor, int>> inverse_overlap_sparse_by_packed_pair;

  /**
   * @brief Dense inverse projected-image matrices grouped by packed pair.
   *
   * For each packed-pair index `k`,
   * `inverse_projected_image_by_packed_pair[k](left,right)` stores the dense
   * projected value `(G x_{left,right})(k)` from
   * `inverse_overlap_projection.projected_pair_values[k]`.
   */
  std::vector<Eigen::MatrixXd> inverse_projected_image_by_packed_pair;

  /**
   * @brief Ordered-pair availability mask for first-order projections.
   */
  std::vector<unsigned char> has_first_order_projection_by_ordered_pair;

  /**
   * @brief Ordered-pair availability mask for inverse projections.
   */
  std::vector<unsigned char> has_inverse_projection_by_ordered_pair;
};

/**
 * @brief Matrix-form opposite-spin channels for alpha and beta spin sectors.
 */
struct OppositeSpinMatrixChannels {
  OppositeSpinPerSpinMatrixChannels alpha;
  OppositeSpinPerSpinMatrixChannels beta;
};

/**
 * @brief Builds one spin-sector matrix-form opposite-spin channels.
 *
 * This routine consumes an ordered same-spin pair cache (`left/right` unique
 * determinant orientation) and reorganizes opposite-spin payloads into packed
 * pair indexed matrix families that are convenient for matrix-form backward
 * contractions.
 */
OppositeSpinPerSpinMatrixChannels build_opposite_spin_per_spin_matrix_channels(
    const std::vector<SpinDeterminantPairEvaluation>& ordered_spin_pair_cache,
    int n_unique_determinants);

/**
 * @brief Builds alpha/beta matrix-form opposite-spin channels from cache context.
 */
OppositeSpinMatrixChannels build_opposite_spin_matrix_channels(
    const SameSpinPairCacheContext& same_spin_pair_cache);

}  // namespace xmvb::vb
