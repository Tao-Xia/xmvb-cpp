#pragma once

#include <cstddef>
#include <vector>

#include "core/containers/shared_vector.hpp"

namespace xmvb::vb {

/**
 * @brief One determinant-to-structure expansion term.
 *
 * Each determinant can contribute to one or more structures with a signed
 * coefficient. This structure makes that mapping explicit.
 */
struct StructureExpansionTerm {
  /**
   * @brief Zero-based structure index.
   */
  int structure_index = 0;

  /**
   * @brief Signed coefficient contributed by the determinant to the structure.
   *
   * The current expansion uses `+1` or `-1`, but this representation
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
   * @brief Determinant overlap cache used during structure assembly.
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
 * This object stores the unspecialized structure representation before it is
 * expanded into unique determinants.
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
   * @brief Input wavefunction type flag.
   */
  int wavefunction_type = 0;

  /**
   * @brief Input VB function type flag.
   */
  int vb_function_type = 0;

  /**
   * @brief Flat structure orbital storage in one-based input ordering.
   *
   * The data is packed structure-by-structure. Each structure contributes
   * `n_total_electrons` one-based orbital labels.
   */
  std::vector<int> raw_structure_orbitals;

  std::size_t flat_orbital_count() const {
    return n_structures *
           n_total_electrons;
  }

  const int* structure_orbitals_data(int structure_index) const {
    return raw_structure_orbitals.data() +
           structure_index *
               n_total_electrons;
  }
};

/**
 * @brief Explicit full-determinant input for structure Hamiltonian assembly.
 *
 * This object contains all data needed by the structure builder to evaluate
 * structure-level Hamiltonian and overlap matrices. All orbital matrices use
 * column-major storage.
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
   * @brief Packed two-electron integral storage using the packed pair index map.
   */
  SharedVector<double> eri_act;
};

}  // namespace xmvb::vb
