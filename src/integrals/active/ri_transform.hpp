#pragma once

#include <cstddef>

#include <Eigen/Core>

#include "integrals/ao/ri_transform.hpp"

namespace xmvb::integrals {

/** @brief Transform symmetric RI pair factors by a square orbital frame.
 *
 * Each input row packs a symmetric n-by-n factor as (i,j), i >= j, at
 * i*(i+1)/2+j. The returned rows use the same convention for Q^T B Q.
 * Q need only be square; orthogonality is not assumed. Input and output are
 * outside the numerical-workspace allowance. The optional allowance bounds
 * two shared tile matrices, each with tile_size*n*n doubles. Products are
 * partitioned by tall matrix rows without auxiliary-wise small GEMM calls.
 *
 * @throws std::invalid_argument for inconsistent or empty dimensions.
 * @throws std::length_error if even one auxiliary row cannot fit in the
 *         requested numerical workspace.
 */
Eigen::MatrixXd transform_ri_pair_factors(const Eigen::Ref<const Eigen::MatrixXd>& factors,
                                          const Eigen::Ref<const Eigen::MatrixXd>& frame,
                                          std::size_t workspace_bytes = workspace::available());

/** @brief Differentiate the RI pair-factor transform in arbitrary directions.
 *
 * Returns the packed factors Q^T dB Q + dQ^T B Q + Q^T B dQ.
 * The two input factor matrices have identical packed layout and shape.
 * This implementation reuses B Q within each auxiliary tile. Its explicit
 * numerical workspace is three shared tile matrices.
 */
Eigen::MatrixXd
transform_ri_pair_factor_direction(const Eigen::Ref<const Eigen::MatrixXd>& factors,
                                   const Eigen::Ref<const Eigen::MatrixXd>& factor_direction,
                                   const Eigen::Ref<const Eigen::MatrixXd>& frame,
                                   const Eigen::Ref<const Eigen::MatrixXd>& frame_direction,
                                   std::size_t workspace_bytes = workspace::available());

/** @brief Differentiate from accepted, already-transformed RI factors.
 *
 * Given Bbar = Q^T B Q and dQ = -Q K, returns packed
 * Q^T dBraw Q - K^T Bbar - Bbar K. No resident raw B or zero-valued factor
 * direction is needed. The accepted Bbar is borrowed and remains unchanged.
 * Its explicit numerical workspace is two shared tile matrices.
 */
Eigen::MatrixXd transform_ri_pair_factor_direction_from_orthogonal(
    const Eigen::Ref<const Eigen::MatrixXd>& orthogonal_factors,
    const Eigen::Ref<const Eigen::MatrixXd>& raw_factor_direction,
    const Eigen::Ref<const Eigen::MatrixXd>& frame,
    const Eigen::Ref<const Eigen::MatrixXd>& frame_connection,
    std::size_t workspace_bytes = workspace::available());

} // namespace xmvb::integrals
