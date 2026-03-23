#pragma once

#include "core/shared_vector.hpp"

namespace xmvb::vb {

/**
 * @brief Input data required to rebuild auxiliary orbitals in C++.
 */
struct OrbitalPreparationInput {
  /**
   * @brief Number of basis functions.
   */
  int n_basis_functions = 0;

  /**
   * @brief Number of orbitals stored in the VB orbital parameterization.
   */
  int n_orbitals = 0;

  /**
   * @brief Number of active orbitals.
   */
  int n_active_orbitals = 0;

  /**
   * @brief Number of total electrons.
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
   * @brief Sparse orbital coefficient values, grouped by orbital.
   */
  SharedVector<double> orbital_value_table;

  /**
   * @brief One-based basis-function indices corresponding to `orbital_value_table`.
   */
  SharedVector<int> orbital_basis_index_table;

  /**
   * @brief Number of explicit coefficients stored for each orbital.
   */
  SharedVector<int> orbital_basis_counts;

  /**
   * @brief Original parameter-space coefficient counts for each orbital.
   *
   * This corresponds to legacy `ma0`, which is the count used to enumerate
   * variational parameters and to accumulate the final orbital gradient.
   */
  SharedVector<int> original_orbital_basis_counts;

  /**
   * @brief Column-major AO overlap matrix.
   */
  SharedVector<double> basis_overlap_matrix;
};

}  // namespace xmvb::vb
