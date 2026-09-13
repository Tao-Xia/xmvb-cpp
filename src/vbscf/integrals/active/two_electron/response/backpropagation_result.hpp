#pragma once

#include <Eigen/Core>

#include <vector>

namespace xmvb::vb {

/**
 * @brief Reverse-mode derivatives through the active-space two-electron builder.
 */
struct ActiveSpaceTwoElectronBackpropagationResult {
  /**
   * @brief Column-major gradient with respect to the active auxiliary block.
   *
   * The two-electron builder never touches the inactive or virtual auxiliary
   * columns, so returning the compact `n_basis_functions x n_active_orbitals`
   * block avoids repeated AO-sized allocations in the SCF hot path.
   */
  Eigen::MatrixXd active_auxiliary_orbital_gradient;
};

}  // namespace xmvb::vb
