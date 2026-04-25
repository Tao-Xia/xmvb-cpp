#pragma once

#include <stdexcept>
#include <vector>

#include <Eigen/Core>

namespace xmvb::vb {

/**
 * @brief Copies one legacy row-by-row dense buffer into `Eigen::MatrixXd`.
 *
 * Some migrated kernels still materialize compatibility buffers where one AO
 * row is stored contiguously. This helper isolates that boundary so the rest
 * of the code can keep using the repository-standard column-major Eigen type.
 */
inline Eigen::MatrixXd copy_legacy_row_buffer_to_matrix(
    const std::vector<double>& values,
    int n_rows,
    int n_cols) {
  if (n_rows < 0 || n_cols < 0) {
    throw std::invalid_argument("matrix dimensions must be non-negative");
  }
  if (values.size() != n_rows * n_cols) {
    throw std::invalid_argument("legacy row buffer size does not match matrix dimensions");
  }
  Eigen::MatrixXd matrix(n_rows, n_cols);
  for (int row = 0; row < n_rows; ++row) {
    const double* source_row =
        values.data() + row * n_cols;
    for (int col = 0; col < n_cols; ++col) {
      matrix(row, col) = source_row[col];
    }
  }
  return matrix;
}

/**
 * @brief Flattens one Eigen matrix into a legacy row-by-row compatibility buffer.
 *
 * New interfaces should prefer `Eigen::MatrixXd` directly. Use this only when
 * a still-migrating internal kernel explicitly expects one contiguous row at a
 * time.
 */
inline std::vector<double> copy_matrix_to_legacy_row_buffer(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
  const std::size_t row_count = matrix.rows();
  const std::size_t col_count = matrix.cols();
  std::vector<double> values(
      row_count * col_count,
      0.0);
  for (int row = 0; row < matrix.rows(); ++row) {
    double* target_row =
        values.data() + row * col_count;
    for (int col = 0; col < matrix.cols(); ++col) {
      target_row[col] = matrix(row, col);
    }
  }
  return values;
}

/**
 * @brief Flattens one column-major Eigen matrix into column-major storage.
 *
 * Use this only at compatibility boundaries that still consume raw contiguous
 * buffers but already follow the repository-wide column-major convention.
 */
inline std::vector<double> flatten_matrix_column_major(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
  return std::vector<double>(
      matrix.data(),
      matrix.data() + matrix.size());
}

}  // namespace xmvb::vb
