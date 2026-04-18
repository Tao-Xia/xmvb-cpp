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
 * used by both the nonorthogonal and biorthogonal matrix-form structure
 * builders.
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

}  // namespace xmvb::vb
