#pragma once

#include <cmath>
#include <stdexcept>

#include "pfaffian_vbscf/math/dense_utils.hpp"

namespace xmvb::pfaffian_vbscf {

/**
 * @brief Returns the packed strict-upper-triangle size of an antisymmetric
 * matrix with the given dimension.
 */
inline int packed_antisymm_size(int dimension) {
  return packed_antisymmetric_size(dimension);
}

/**
 * @brief Decodes one packed antisymmetric matrix.
 */
inline Matrix decode_antisymm(
    const ScalarBuffer& packed_entries,
    int dimension) {
  return decode_antisymmetric_matrix(packed_entries, dimension);
}

/**
 * @brief Encodes a dense antisymmetric matrix into the strict upper triangle.
 */
inline ScalarBuffer encode_antisymm(const ConstMatrixRef& matrix) {
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument("matrix must be square");
  }

  const Eigen::Index dimension = matrix.rows();
  ScalarBuffer packed;
  packed.reserve(dimension * (dimension - 1) / 2);
  for (Eigen::Index col = 1; col < dimension; ++col) {
    for (Eigen::Index row = 0; row < col; ++row) {
      if (std::abs(matrix(row, row)) > 1.0e-12 ||
          std::abs(matrix(col, col)) > 1.0e-12 ||
          std::abs(matrix(row, col) + matrix(col, row)) > 1.0e-10) {
        throw std::invalid_argument("matrix is not antisymmetric");
      }
      packed.push_back(matrix(row, col));
    }
  }
  return packed;
}

}  // namespace xmvb::pfaffian_vbscf
