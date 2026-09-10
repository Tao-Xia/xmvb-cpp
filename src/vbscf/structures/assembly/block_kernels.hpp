#pragma once

#include <stdexcept>

namespace xmvb::vb {

/**
 * @brief Dense Frobenius inner product for one local structure block.
 *
 * All matrix-form structure builders in the codebase eventually contract a
 * left coefficient block with one intermediate image of the same shape. This
 * shared helper keeps the dense shape checks and dot-product semantics in one
 * place.
 */
template <typename MatrixType>
double dense_frobenius_inner_product(
    const MatrixType& left_matrix,
    const MatrixType& right_matrix) {
  if (left_matrix.rows() != right_matrix.rows() ||
      left_matrix.cols() != right_matrix.cols()) {
    throw std::invalid_argument(
        "matrix-form structure contraction shape mismatch");
  }
  if (left_matrix.size() == 0) {
    return 0.0;
  }
  return (left_matrix.array() * right_matrix.array()).sum();
}

/**
 * @brief Contracts one pair of same-spin kernel blocks through structure coefficients.
 *
 * Algebraically this computes the dense block contraction
 * `sum(C_L .* (alpha * (C_R * beta^T)))`, which is the common local kernel
 * used by the matrix-form structure builders.
 */
template <typename MatrixType>
double contract_dense_structure_pair_kernel(
    const MatrixType& left_coefficients,
    const MatrixType& right_coefficients,
    const MatrixType& alpha_kernel,
    const MatrixType& beta_kernel,
    MatrixType* beta_push,
    MatrixType* image) {
  if (beta_push == nullptr || image == nullptr) {
    throw std::invalid_argument("beta_push and image must not be null");
  }
  if (left_coefficients.rows() != alpha_kernel.rows() ||
      right_coefficients.rows() != alpha_kernel.cols() ||
      left_coefficients.cols() != beta_kernel.rows() ||
      right_coefficients.cols() != beta_kernel.cols()) {
    throw std::invalid_argument(
        "matrix-form structure contraction shape mismatch");
  }
  if (left_coefficients.size() == 0 || right_coefficients.size() == 0) {
    return 0.0;
  }

  beta_push->resize(right_coefficients.rows(), beta_kernel.rows());
  beta_push->noalias() = right_coefficients * beta_kernel.transpose();

  image->resize(alpha_kernel.rows(), beta_kernel.rows());
  image->noalias() = alpha_kernel * (*beta_push);

  return dense_frobenius_inner_product(left_coefficients, *image);
}

/**
 * @brief Contracts one pair of diagonal structure coefficient blocks.
 *
 * In close-shell expansions the structure coefficient matrices are diagonal on
 * the shared unique-spin support, so the local kernel reduces to
 *
 * `sum_{i,j} l_i * alpha(i,j) * r_j * beta(i,j)`.
 */
template <typename MatrixType>
double contract_diagonal_structure_pair_kernel(
    const std::vector<double>& left_diagonal_coefficients,
    const std::vector<double>& right_diagonal_coefficients,
    const MatrixType& alpha_kernel,
    const MatrixType& beta_kernel) {
  if (alpha_kernel.rows() != static_cast<int>(left_diagonal_coefficients.size()) ||
      alpha_kernel.cols() != static_cast<int>(right_diagonal_coefficients.size()) ||
      beta_kernel.rows() != static_cast<int>(left_diagonal_coefficients.size()) ||
      beta_kernel.cols() != static_cast<int>(right_diagonal_coefficients.size())) {
    throw std::invalid_argument(
        "diagonal matrix-form structure contraction shape mismatch");
  }
  double contraction = 0.0;
  for (int column = 0;
       column < static_cast<int>(right_diagonal_coefficients.size());
       ++column) {
    const double right_coefficient = right_diagonal_coefficients[column];
    if (right_coefficient == 0.0) {
      continue;
    }
    for (int row = 0;
         row < static_cast<int>(left_diagonal_coefficients.size());
         ++row) {
      const double left_coefficient = left_diagonal_coefficients[row];
      if (left_coefficient == 0.0) {
        continue;
      }
      contraction +=
          left_coefficient *
          alpha_kernel(row, column) *
          right_coefficient *
          beta_kernel(row, column);
    }
  }
  return contraction;
}

}  // namespace xmvb::vb
