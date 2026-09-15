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
 * @brief Solves selected generalized-eigenpair response without a full spectrum.
 *
 * For each accepted pair `H c = E S c`, the solver applies preconditioned
 * MINRES to the symmetric bordered response equation. All selected roots
 * advance together, so each Krylov step uses one block H/S action. The caller
 * supplies the accepted-point product `S C` because it is invariant across
 * every directional response at the same accepted point.
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
    const EigenResponseOptions& options);

}  // namespace xmvb::core
