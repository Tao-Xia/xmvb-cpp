#pragma once

#include "core/shared_vector.hpp"
#include "vb/matrices/structure_expansion_term.hpp"

namespace xmvb::vb {

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
  SharedVector<std::vector<int>> alpha_occupied_orbitals_by_determinant;

  /**
   * @brief Zero-based beta occupied orbitals for each full determinant.
   */
  SharedVector<std::vector<int>> beta_occupied_orbitals_by_determinant;

  /**
   * @brief Determinant-to-structure expansion coefficients.
   */
  SharedVector<std::vector<StructureExpansionTerm>> determinant_to_structure_terms;

  /**
   * @brief Column-major active-space orbital overlap matrix.
   */
  SharedVector<double> basis_overlap_matrix;

  /**
   * @brief Column-major one-electron Hamiltonian matrix.
   */
  SharedVector<double> one_electron_matrix;

  /**
   * @brief Packed two-electron integral storage using the legacy VB index map.
   */
  SharedVector<double> packed_two_electron_integrals;
};

}  // namespace xmvb::vb
