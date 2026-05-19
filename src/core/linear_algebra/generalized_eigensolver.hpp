#pragma once

#include <vector>

namespace xmvb::core {

/**
 * @brief Result of a symmetric generalized eigenvalue problem.
 *
 * All matrices use column-major storage:
 * `matrix_data[column * dimension + row]`.
 */
struct GeneralizedEigenResult {
  /**
   * @brief Eigenvalues in ascending order.
   */
  std::vector<double> eigenvalues;

  /**
   * @brief Column-major eigenvector matrix.
   *
   * Column `i` contains the eigenvector associated with `eigenvalues[i]`.
   */
  std::vector<double> eigenvector_matrix;
};

/**
 * @brief Solves the symmetric-definite problem `H C = S C E`.
 *
 * Both input matrices must use column-major storage and `overlap_matrix`
 * must be positive definite.
 */
class GeneralizedEigensolver {
public:
  /**
   * @brief Solves the generalized eigenvalue problem.
   *
   * @param hamiltonian_matrix Column-major symmetric Hamiltonian matrix.
   * @param overlap_matrix Column-major symmetric positive-definite overlap matrix.
   * @param dimension Matrix dimension.
   * @return GeneralizedEigenResult Eigenvalues and eigenvectors.
   */
  GeneralizedEigenResult solve(
      const std::vector<double>& hamiltonian_matrix,
      const std::vector<double>& overlap_matrix,
      int dimension) const;

  /**
   * @brief Solves only the generalized eigenvalues.
   *
   * This skips the full eigenvector matrix and is intended for energy-only
   * trial-point screening where only selected-state energies are needed.
   */
  std::vector<double> solve_eigenvalues_only(
      const std::vector<double>& hamiltonian_matrix,
      const std::vector<double>& overlap_matrix,
      int dimension) const;

  /**
   * @brief Davidson-iterative solve for the lowest n_roots eigenpairs.
   *
   * For large structure spaces (N > ~300), O(k N^2) vs O(N^3) for LAPACK.
   */
  GeneralizedEigenResult solve_davidson(
      const std::vector<double>& hamiltonian_matrix,
      const std::vector<double>& overlap_matrix,
      int dimension,
      int n_roots) const;
};

}  // namespace xmvb::core
