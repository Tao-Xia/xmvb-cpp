#include "pfaffian_vbscf/matrices/pf_spin_adapted_matrix_projector.hpp"

#include <stdexcept>

namespace xmvb::pfaffian_vbscf {

namespace {

Matrix dense_mat(
    const ScalarBuffer& data,
    int dimension,
    const char* label) {
  const std::size_t expected_size =
      xmvb::to_size(dimension) * xmvb::to_size(dimension);
  if (data.size() != expected_size) {
    throw std::invalid_argument(std::string(label) + " size does not match the matrix dimension");
  }

  Matrix matrix = Matrix::Zero(dimension, dimension);
  for (int col = 0; col < dimension; ++col) {
    for (int row = 0; row < dimension; ++row) {
      matrix(row, col) = data[xmvb::to_size(col) * dimension + row];
    }
  }
  return matrix;
}

ScalarBuffer column_major_storage(const ConstMatrixRef& matrix) {
  ScalarBuffer data(xmvb::to_size(matrix.rows()) * matrix.cols(), 0.0);
  for (Eigen::Index col = 0; col < matrix.cols(); ++col) {
    for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
      data[xmvb::to_size(col) * matrix.rows() + row] = matrix(row, col);
    }
  }
  return data;
}

}  // namespace

PfMatrixBuildResult project_spin_adapted_matrices(
    const PfMatrixBuildResult& primitive_mats,
    const ConstMatrixRef& primitive_to_adapted_coefficients) {
  if (primitive_mats.n_states <= 0) {
    throw std::invalid_argument("primitive matrix dimension must be positive");
  }
  if (primitive_to_adapted_coefficients.rows() != primitive_mats.n_states ||
      primitive_to_adapted_coefficients.cols() <= 0) {
    throw std::invalid_argument(
        "primitive-to-adapted coefficient matrix does not match the primitive basis dimension");
  }

  const Matrix primitive_overlap =
      dense_mat(primitive_mats.s, primitive_mats.n_states, "primitive_mats.s");
  const Matrix primitive_hamiltonian =
      dense_mat(primitive_mats.h, primitive_mats.n_states, "primitive_mats.h");

  PfMatrixBuildResult projected;
  projected.n_states = primitive_to_adapted_coefficients.cols();
  projected.s = column_major_storage(
      primitive_to_adapted_coefficients.transpose() *
      primitive_overlap *
      primitive_to_adapted_coefficients);
  projected.h = column_major_storage(
      primitive_to_adapted_coefficients.transpose() *
      primitive_hamiltonian *
      primitive_to_adapted_coefficients);
  projected.pair_profile = primitive_mats.pair_profile;
  return projected;
}

Matrix lift_spin_adapted_matrix_gradient(
    const ConstMatrixRef& spin_adapted_gradient,
    const ConstMatrixRef& primitive_to_adapted_coefficients) {
  if (spin_adapted_gradient.rows() != spin_adapted_gradient.cols()) {
    throw std::invalid_argument("spin-adapted gradient matrix must be square");
  }
  if (primitive_to_adapted_coefficients.cols() != spin_adapted_gradient.rows()) {
    throw std::invalid_argument(
        "spin-adapted gradient dimension does not match the coefficient matrix");
  }
  return primitive_to_adapted_coefficients *
      spin_adapted_gradient *
      primitive_to_adapted_coefficients.transpose();
}

}  // namespace xmvb::pfaffian_vbscf
