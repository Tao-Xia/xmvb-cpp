#pragma once

#include <vector>

namespace xmvb::vb {

/**
 * @brief Structure-level accumulated matrices.
 *
 * All matrices use column-major storage:
 * `matrix_data[column * n_structures + row]`.
 */
struct StructureAccumulationResult {
  /**
   * @brief Number of VB structures.
   */
  int n_structures = 0;

  /**
   * @brief Structure overlap matrix.
   */
  std::vector<double> overlap_matrix;

  /**
   * @brief Structure Hamiltonian matrix.
   */
  std::vector<double> hamiltonian_matrix;

  /**
   * @brief Structure one-electron Hamiltonian matrix.
   */
  std::vector<double> one_electron_hamiltonian_matrix;

  /**
   * @brief Determinant overlap cache used by the legacy implementation.
   *
   * For diagonal determinant pairs this stores the overlap determinant at the
   * determinant index.
   */
  std::vector<double> determinant_overlap_cache;
};

}  // namespace xmvb::vb
