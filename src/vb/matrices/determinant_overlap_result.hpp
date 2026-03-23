#pragma once

#include <Eigen/Core>

namespace xmvb::vb {

/**
 * @brief Result of resolving a determinant-level overlap matrix.
 *
 * The matrix storage follows the Fortran-compatible column-major convention:
 * `matrix_data[column * dimension + row]`.
 */
struct DeterminantOverlapResult {
  /**
   * @brief Dimension of the square overlap submatrix.
   */
  int n_electrons = 0;

  /**
   * @brief Algebraic determinant of the overlap submatrix.
   *
   * This matches the `Det` quantity used by the legacy Fortran implementation.
   */
  double overlap_determinant = 0.0;

  /**
   * @brief Number of singular values below the linear-dependence threshold.
   *
   * This matches the `Nlt` quantity in the legacy implementation.
   */
  int nullity = 0;

  /**
   * @brief Singular values of the determinant overlap submatrix.
   *
   * The singular values are stored in the same descending order returned by
   * `Eigen::JacobiSVD`.
   */
  Eigen::VectorXd singular_values;

  /**
   * @brief Left singular vectors of the overlap submatrix.
   *
   * The overlap matrix uses the internal convention "rows = right determinant,
   * columns = left determinant", so `matrix_U` spans the right-determinant
   * occupied space.
   */
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> matrix_U;

  /**
   * @brief Right singular vectors of the overlap submatrix.
   *
   * `matrix_V` spans the left-determinant occupied space.
   */
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> matrix_V;

  /**
   * @brief Sign contribution from the orthogonal factors.
   *
   * This equals `det(U) * det(V)` and is always `+1.0` or `-1.0`.
   */
  double parity = 1.0;
};

}  // namespace xmvb::vb
