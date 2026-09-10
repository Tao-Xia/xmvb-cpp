#pragma once

#include <Eigen/Core>

#include <vector>

namespace xmvb::vb {

/**
 * @brief Reverse-mode derivatives through the active-space matrix builders.
 */
struct ActiveSpaceMatrixBackpropagationResult {
  /**
   * @brief Column-major gradient with respect to the active auxiliary block.
   *
   * The matrix uses the natural `n_basis_functions x n_active_orbitals`
   * layout consumed by `ActiveSpaceOrbitalBackpropagator` rather than
   * embedding the active block into a mostly zero `n_basis x n_basis`
   * auxiliary matrix.
   */
  Eigen::MatrixXd active_auxiliary_orbital_gradient;

  /**
   * @brief Column-major gradient with respect to the AO effective one-electron matrix `F11`.
   */
  std::vector<double> ao_effective_one_electron_gradient;
};

}  // namespace xmvb::vb
