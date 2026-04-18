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
   * @brief Cached inverse overlap submatrix for regular determinant pairs.
   *
   * When `nullity == 0`, the resolver may populate this matrix directly from a
   * fast LU path instead of storing a full SVD. Singular pairs leave this
   * matrix empty.
   */
  Eigen::MatrixXd inverse_overlap_submatrix;

  /**
   * @brief Cached first deleted-minor matrix for nullity-0/1 determinant pairs.
   *
   * This uses the internal convention "rows = right determinant, columns =
   * left determinant". The same matrix is reused by same-spin Hamiltonian
   * assembly, opposite-spin coupling, and active-space gradient accumulation.
   * Rank-deficient pairs with `nullity >= 2` leave this matrix empty.
   */
  Eigen::MatrixXd first_order_cofactor_matrix;

  /**
   * @brief Singular values of the determinant overlap submatrix.
   *
   * Singular pairs may require a full SVD. Regular pairs resolved through the
   * LU fast path leave this vector empty.
   */
  Eigen::VectorXd singular_values;

  /**
   * @brief Left singular vectors of the overlap submatrix.
   *
   * The overlap matrix uses the internal convention "rows = right determinant,
   * columns = left determinant", so `matrix_U` spans the right-determinant
   * occupied space. Regular pairs resolved through the LU fast path leave this
   * matrix empty.
   */
  Eigen::MatrixXd matrix_U;

  /**
   * @brief Right singular vectors of the overlap submatrix.
   *
   * `matrix_V` spans the left-determinant occupied space. Regular pairs
   * resolved through the LU fast path leave this matrix empty.
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

/**
 * @brief Sparse packed active-pair vector plus its dense projected image.
 *
 * `packed_pair_indices[k]` and `packed_pair_values[k]` store one nonzero entry
 * of the regrouped opposite-spin coefficient vector. `projected_pair_values`
 * stores the dense contraction of that sparse vector with the active-space
 * packed-pair kernel.
 */
struct OppositeSpinPackedPairProjection {
  std::vector<int> packed_pair_indices;
  std::vector<double> packed_pair_values;
  std::vector<double> projected_pair_values;
};

/**
 * @brief Exact packed-pair caches reused by opposite-spin forward/backward paths.
 *
 * `first_order_cofactor_projection` is available for determinant pairs with
 * `nullity <= 1`. `inverse_overlap_projection` is available only for regular
 * determinant pairs with `nullity == 0`.
 */
struct OppositeSpinPairCache {
  int n_packed_active_pairs = 0;
  OppositeSpinPackedPairProjection first_order_cofactor_projection;
  OppositeSpinPackedPairProjection inverse_overlap_projection;
};

}  // namespace xmvb::vb
