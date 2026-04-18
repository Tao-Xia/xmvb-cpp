#pragma once

#include <stdexcept>
#include <string>

#include <Eigen/Core>

namespace xmvb::vb::biorthogonal_vbscf {

/**
 * @brief Validates one dense unique-spin coefficient matrix or local support block.
 *
 * The exact biorthogonal selected-state helpers may store either the full
 * unique-spin grid dimensions or one compressed local support block. This
 * check keeps all matrix-form contractions aligned with the expected row/column
 * counts before they enter the hot BLAS kernels.
 */
inline void validate_biorthogonal_state_matrix_shape(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix,
    int n_unique_alpha,
    int n_unique_beta,
    const char* label) {
  if (matrix.rows() != n_unique_alpha || matrix.cols() != n_unique_beta) {
    throw std::invalid_argument(
        std::string(label) + " shape does not match unique-spin dimensions");
  }
}

}  // namespace xmvb::vb::biorthogonal_vbscf
