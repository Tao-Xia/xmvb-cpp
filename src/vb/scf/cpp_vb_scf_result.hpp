#pragma once

#include <vector>

#include "vb/matrices/structure_types.hpp"

namespace xmvb::vb {

/**
 * @brief Result of a single C++ VBSCF matrix evaluation step.
 *
 * This object captures the complete state needed by downstream RDM and
 * gradient code:
 * - full structure Hamiltonian and overlap matrices
 * - generalized eigenvalues and eigenvectors
 * - selected-state averaging metadata
 */
struct CppVbScfResult {
  /**
   * @brief Number of VB structures.
   */
  int n_structures = 0;

  /**
   * @brief Nuclear repulsion energy supplied by the caller.
   */
  double nuclear_repulsion_energy = 0.0;

  /**
   * @brief Inactive-space reference energy `E11`.
   */
  double one_electron_reference_energy = 0.0;

  /**
   * @brief State-averaged electronic energy from the selected states.
   */
  double electronic_energy = 0.0;

  /**
   * @brief State-averaged total energy including nuclear repulsion.
   */
  double total_energy = 0.0;

  /**
   * @brief Mean of the diagonal structure overlap elements.
   */
  double average_structure_overlap = 0.0;

  /**
   * @brief All generalized eigenvalues in ascending order.
   *
   * These are electronic energies before adding nuclear repulsion.
   */
  std::vector<double> electronic_state_energies;

  /**
   * @brief Selected-state total energies including nuclear repulsion.
   */
  std::vector<double> selected_state_total_energies;

  /**
   * @brief Selected state indices used for state averaging.
   */
  std::vector<int> selected_state_indices;

  /**
   * @brief Normalized weights used for state averaging.
   */
  std::vector<double> state_average_weights;

  /**
   * @brief Column-major generalized eigenvector matrix.
   */
  std::vector<double> eigenvector_matrix;

  /**
   * @brief Structure Hamiltonian and overlap matrices.
   */
  StructureAccumulationResult structure_matrices;
};

}  // namespace xmvb::vb
