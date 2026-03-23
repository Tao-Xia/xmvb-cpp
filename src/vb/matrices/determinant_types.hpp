#pragma once

#include <vector>

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
  Eigen::MatrixXd matrix_U;

  /**
   * @brief Right singular vectors of the overlap submatrix.
   *
   * `matrix_V` spans the left-determinant occupied space.
   */
  Eigen::MatrixXd matrix_V;

  /**
   * @brief Sign contribution from the orthogonal factors.
   *
   * This equals `det(U) * det(V)` and is always `+1.0` or `-1.0`.
   */
  double parity = 1.0;
};

/**
 * @brief Determinant-level Hamiltonian and overlap quantities.
 */
struct DeterminantHamiltonianResult {
  /**
   * @brief Determinant overlap between the two determinants.
   */
  double overlap_determinant = 0.0;

  /**
   * @brief One-electron Hamiltonian matrix element.
   */
  double one_electron_hamiltonian = 0.0;

  /**
   * @brief Total Hamiltonian matrix element including one- and two-electron parts.
   */
  double total_hamiltonian = 0.0;

  /**
   * @brief Nullity of the determinant overlap submatrix.
   */
  int nullity = 0;
};

/**
 * @brief Input data for one determinant pair contribution.
 */
struct DeterminantPairInput {
  /**
   * @brief Zero-based determinant index on the left.
   */
  int determinant_index_left = 0;

  /**
   * @brief Zero-based determinant index on the right.
   */
  int determinant_index_right = 0;

  /**
   * @brief Zero-based occupied orbitals of the left determinant.
   */
  std::vector<int> occ_L;

  /**
   * @brief Zero-based occupied orbitals of the right determinant.
   */
  std::vector<int> occ_R;

  /**
   * @brief Column-major determinant overlap submatrix between occupied orbitals.
   */
  std::vector<double> det_ovlp_mat;
};

}  // namespace xmvb::vb
