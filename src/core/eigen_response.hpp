#pragma once

#include <vector>

#include <Eigen/Core>

#include "core/eigensolver.hpp"

namespace xmvb::core {

/**
 * @brief Accuracy and work limit for matrix-free generalized-eigen response.
 */
struct EigenResponseOptions {
  int max_iterations;
  double relative_residual_tolerance;
};

struct EigenResponseRecycleGuess {
  Eigen::VectorXd solution;
  Eigen::VectorXd operator_image;
  bool used = false;
};

/**
 * @brief Same-operator Galerkin space recycled across response right-hand sides.
 *
 * The stored columns are Euclidean-orthonormal projected response vectors and
 * their exact projected operator images. The space is owned by one accepted
 * orbital point and one selected root; it must not survive an operator change.
 */
class EigenResponseRecycleSpace {
public:
  int dimension() const noexcept;
  int size() const noexcept;
  void clear();

  EigenResponseRecycleGuess guess(
      const Eigen::Ref<const Eigen::VectorXd>& right_hand_side) const;

  bool append(
      const Eigen::Ref<const Eigen::VectorXd>& solution,
      const Eigen::Ref<const Eigen::VectorXd>& operator_image);

private:
  Eigen::MatrixXd basis_;
  Eigen::MatrixXd operator_images_;
};

/**
 * @brief Selected-root response and numerical diagnostics.
 */
struct EigenResponseResult {
  Eigen::MatrixXd eigenvector_response;
  Eigen::VectorXd eigenvalue_response;
  Eigen::VectorXd relative_residual_norms;
  std::vector<int> iterations;
  int block_actions = 0;
};

/**
 * @brief Response of an equally weighted generalized-eigen subspace.
 *
 * `selected_matrix_response` is the selected-space effective-Hamiltonian
 * response in
 *
 * `H dC - S dC E - S C dE = -(dH C - dS C E)`.
 *
 * Unlike isolated-root response, `dE` is generally not diagonal.  Its
 * off-diagonal entries absorb the chosen rotation within the selected
 * subspace.  The returned vectors use the symmetric metric gauge
 *
 * `C^T S dC = -0.5 C^T dS C`.
 */
struct EigenSubspaceResponseResult {
  Eigen::MatrixXd eigenvector_response;
  Eigen::MatrixXd selected_matrix_response;
  Eigen::VectorXd relative_residual_norms;
  std::vector<int> iterations;
  int block_actions = 0;
};

/**
 * @brief Solves selected generalized-eigenpair response without a full spectrum.
 *
 * For each accepted pair `H c = E S c`, the solver applies preconditioned
 * MINRES to `H - E S` on the Euclidean complement of each known Ritz vector.
 * Removing the selected-root null mode before iteration keeps the metric gauge
 * out of the Krylov recurrence. The response along that root is recovered from
 * `c^T S dc = -0.5 c^T dS c`; the original symmetric bordered equation then
 * certifies the true residual, including Davidson Ritz-vector drift. All
 * selected roots advance together through one block H/S action per iteration.
 * The caller supplies accepted-point `S C`, reused by every direction.
 *
 * The returned residual measures this linear equation at the supplied Ritz
 * root, not the error of that Ritz root relative to an exact eigenpair.
 */
EigenResponseResult solve_generalized_eigen_response(
    const GeneralizedEigenAction& action,
    const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_diagonal,
    const Eigen::Ref<const Eigen::VectorXd>& overlap_diagonal,
    const Eigen::Ref<const Eigen::VectorXd>& selected_eigenvalues,
    const Eigen::Ref<const Eigen::MatrixXd>& selected_eigenvectors,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_hamiltonian_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_overlap_selected,
    const EigenResponseOptions& options,
    const std::vector<EigenResponseRecycleSpace*>& recycle_spaces = {});

/**
 * @brief Applies the complete generalized eigenspectrum to selected responses.
 *
 * Dense solves and memory-affordable Davidson bases retain the full spectrum.
 * This route evaluates its first-order response and certifies the bordered true
 * residual with the independent H/S action; it never forms an orbital Hessian.
 */
EigenResponseResult solve_generalized_eigen_response_from_full_spectrum(
    const GeneralizedEigenAction& action,
    const Eigen::Ref<const Eigen::VectorXd>& full_eigenvalues,
    const Eigen::Ref<const Eigen::MatrixXd>& full_eigenvectors,
    const std::vector<int>& selected_root_indices,
    const Eigen::Ref<const Eigen::VectorXd>& selected_eigenvalues,
    const Eigen::Ref<const Eigen::MatrixXd>& selected_eigenvectors,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_hamiltonian_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_overlap_selected,
    double relative_residual_tolerance);

/**
 * @brief Solves the response of one equally weighted selected-state cluster.
 *
 * All selected-selected rotations are gauge degrees of freedom for an equal
 * state average.  This routine therefore projects the linear systems against
 * the complete selected subspace and solves only selected-external response.
 * It remains well-defined when selected roots are degenerate; a degeneracy
 * with an omitted external root must instead be included in the cluster.
 *
 * The selected columns must be S-orthonormal generalized eigenvectors and
 * `delta_*_selected` must contain `dH C` and `dS C`, respectively.
 */
EigenSubspaceResponseResult
solve_equal_weight_generalized_eigen_subspace_response(
    const GeneralizedEigenAction& action,
    const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_diagonal,
    const Eigen::Ref<const Eigen::VectorXd>& overlap_diagonal,
    const Eigen::Ref<const Eigen::VectorXd>& selected_eigenvalues,
    const Eigen::Ref<const Eigen::MatrixXd>& selected_eigenvectors,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_hamiltonian_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_overlap_selected,
    const EigenResponseOptions& options,
    const std::vector<EigenResponseRecycleSpace*>& recycle_spaces = {});

/**
 * @brief Full-spectrum reference for equal-weight invariant-subspace response.
 */
EigenSubspaceResponseResult
solve_equal_weight_generalized_eigen_subspace_response_from_full_spectrum(
    const GeneralizedEigenAction& action,
    const Eigen::Ref<const Eigen::VectorXd>& full_eigenvalues,
    const Eigen::Ref<const Eigen::MatrixXd>& full_eigenvectors,
    const std::vector<int>& selected_root_indices,
    const Eigen::Ref<const Eigen::VectorXd>& selected_eigenvalues,
    const Eigen::Ref<const Eigen::MatrixXd>& selected_eigenvectors,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_hamiltonian_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_overlap_selected,
    double relative_residual_tolerance);

}  // namespace xmvb::core
