#pragma once

#include <cstddef>
#include <vector>

#include "core/shared_vector.hpp"

namespace xmvb::vb {

/**
 * @brief One determinant-to-structure expansion term.
 *
 * In the legacy VBSCF code each determinant can contribute to one or more
 * structures with a sign factor. This structure makes that mapping explicit.
 */
struct StructureExpansionTerm {
  /**
   * @brief Zero-based structure index.
   */
  int structure_index = 0;

  /**
   * @brief Signed coefficient contributed by the determinant to the structure.
   *
   * The current legacy mapping uses `+1` or `-1`, but this representation
   * intentionally allows any real coefficient.
   */
  double coefficient = 0.0;
};

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
   * @brief Determinant overlap cache used by the legacy implementation.
   *
   * For diagonal determinant pairs this stores the overlap determinant at the
   * determinant index.
   */
  std::vector<double> determinant_overlap_cache;
};

struct HamiltonianOverlapMatrices {
  std::vector<double> hamiltonian_matrix;
  std::vector<double> overlap_matrix;
};

/**
 * @brief Raw VB structure definitions read directly from the input deck.
 *
 * This object stores the unspecialized structure representation before the
 * legacy `rdm_vbscf` code expands it into unique determinants.
 */
struct RawStructureData {
  /**
   * @brief Number of structures.
   */
  int n_structures = 0;

  /**
   * @brief Total number of electrons.
   */
  int n_total_electrons = 0;

  /**
   * @brief Number of active electrons.
   */
  int n_active_electrons = 0;

  /**
   * @brief Spin multiplicity.
   */
  int spin_multiplicity = 1;

  /**
   * @brief Legacy wavefunction type flag.
   */
  int wavefunction_type = 0;

  /**
   * @brief Legacy VB function type flag.
   */
  int vb_function_type = 0;

  /**
   * @brief Flat structure orbital storage in legacy one-based ordering.
   *
   * The data is packed structure-by-structure. Each structure contributes
   * `n_total_electrons` orbital labels using the legacy one-based convention.
   */
  std::vector<int> raw_structure_orbitals;

  std::size_t flat_orbital_count() const {
    return xmvb::to_size(n_structures) *
           xmvb::to_size(n_total_electrons);
  }

  const int* structure_orbitals_data(int structure_index) const {
    return raw_structure_orbitals.data() +
           xmvb::to_size(structure_index) *
               xmvb::to_size(n_total_electrons);
  }
};

/**
 * @brief Explicit full-determinant input for structure Hamiltonian assembly.
 *
 * This object contains all data needed by the pure C++ builder to evaluate
 * structure-level Hamiltonian and overlap matrices. All orbital matrices use
 * column-major storage so they can be compared directly with the legacy
 * Fortran implementation.
 */
struct FullDeterminantStructureData {
  /**
   * @brief Number of VB structures.
   */
  int n_structures = 0;

  /**
   * @brief Number of active orbitals.
   */
  int n_active_orbitals = 0;

  /**
   * @brief Zero-based alpha occupied orbitals for each full determinant.
   */
  SharedVector<std::vector<int>> alpha_det;

  /**
   * @brief Zero-based beta occupied orbitals for each full determinant.
   */
  SharedVector<std::vector<int>> beta_det;

  /**
   * @brief Determinant-to-structure expansion coefficients.
   */
  SharedVector<std::vector<StructureExpansionTerm>> determinant_to_structure_terms;

  /**
   * @brief Column-major active-space orbital overlap matrix `ovlp_act`.
   */
  SharedVector<double> ovlp_act;

  /**
   * @brief Column-major one-electron Hamiltonian matrix.
   */
  SharedVector<double> h1e_act;

  /**
   * @brief Packed two-electron integral storage using the legacy VB index map.
   */
  SharedVector<double> eri_act;
};

}  // namespace xmvb::vb
