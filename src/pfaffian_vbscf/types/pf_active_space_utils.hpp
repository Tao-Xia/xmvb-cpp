#pragma once

#include "pfaffian_vbscf/types/eigen_types.hpp"

namespace xmvb::pfaffian_vbscf {

/**
 * @brief Flattens one Eigen dense matrix into the Pfaffian scalar-buffer layout.
 *
 * Pfaffian kernels consume contiguous scalar buffers rather than Eigen matrix
 * objects. This helper preserves Eigen's native column-major storage when
 * exporting one dense matrix into `ScalarBuffer`.
 */
inline ScalarBuffer flatten_column_major_matrix(
    const Eigen::MatrixXd& matrix) {
  return ScalarBuffer(matrix.data(), matrix.data() + matrix.size());
}

}  // namespace xmvb::pfaffian_vbscf
