#pragma once

#include <functional>
#include <vector>

#include <Eigen/Core>

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
 * @brief Images returned by one block generalized-eigenvalue operator call.
 */
struct GeneralizedEigenActionResult {
  Eigen::MatrixXd hamiltonian;
  Eigen::MatrixXd overlap;
};

using GeneralizedEigenAction = std::function<GeneralizedEigenActionResult(
    const Eigen::Ref<const Eigen::MatrixXd>&)>;

/**
 * @brief Explicit numerical and memory budgets for block Davidson.
 */
struct DavidsonOptions {
  int n_roots;
  int max_iterations;
  int max_subspace_dimension;
  double residual_tolerance;
};

/**
 * @brief Converged Davidson eigenpairs and solver diagnostics.
 */
struct DavidsonResult {
  GeneralizedEigenResult eigenpairs;
  std::vector<double> relative_residual_norms;
  int iterations = 0;
  int block_actions = 0;
  int peak_subspace_dimension = 0;
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
   * @brief Solves the lowest generalized eigenpairs from block H/S actions.
   *
   * The basis, its Hamiltonian image, and its overlap image are updated
   * together, so metric orthogonalization never triggers redundant operator
   * calls. The projected Hamiltonian is extended incrementally, and only the
   * requested lowest Ritz pairs are solved unless a restart needs the full
   * subspace spectrum. Failure to meet the requested residual tolerance is
   * reported as an exception rather than returning unconverged Ritz pairs.
   */
  DavidsonResult solve_davidson(
      const GeneralizedEigenAction& action,
      const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_diagonal,
      const Eigen::Ref<const Eigen::VectorXd>& overlap_diagonal,
      const DavidsonOptions& options) const;
};

}  // namespace xmvb::core
