#pragma once

#include <vector>

#include "vbscf/structures/expansion/types.hpp"

namespace xmvb::vb {

/**
 * @brief Result of a single VBSCF matrix evaluation step.
 *
 * This object captures the complete state needed by downstream RDM and
 * gradient code:
 * - optional full structure Hamiltonian and overlap matrices
 * - consecutive lowest generalized eigenpairs required by the caller
 * - selected-state averaging metadata
 */
struct VbScfResult {
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
   * @brief Available consecutive lowest generalized eigenvalues.
   *
   * Davidson results retain only the requested low roots. Dense reference
   * results contain the complete spectrum.
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
   * @brief Column-major generalized eigenvectors aligned with available roots.
   */
  std::vector<double> eigenvector_matrix;

  /**
   * @brief Exact diagonal of the structure overlap matrix.
   *
   * Davidson uses this linear-storage metric data to report coefficients in
   * individually normalized structure coordinates without forming full S.
   */
  std::vector<double> structure_overlap_diagonal;

  /**
   * @brief Column-major products `S C` aligned with available Davidson roots.
   *
   * These products are sufficient for exact Coulson--Chirgwin weights. Dense
   * results may leave this empty because the explicit overlap matrix exists.
   */
  std::vector<double> overlap_eigenvector_matrix;

  /**
   * @brief Optional explicit structure Hamiltonian and overlap matrices.
   *
   * These remain empty throughout matrix-free Davidson optimization and
   * reporting. Dense reference runs materialize them explicitly.
   */
  StructureAccumulationResult structure_matrices;
};

}  // namespace xmvb::vb
