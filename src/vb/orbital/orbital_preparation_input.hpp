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
   * @brief Legacy orbital-type selector `orbtyp`.
   *
   * The value follows the C runtime constants in `vb/vb.h` (`HAO_TYP`,
   * `BDO_TYP`, `OEO_TYP`, ...). The C++ loader uses this to distinguish the
   * physical orbital manifold from the support layout carried by a particular
   * guess file.
   */
  int orbital_type = 0;

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
   * @brief Optional pre-expansion sparse support counts for MO-gauge canonicalization.
   *
   * When `GUESS=MO` needs inactive-support expansion for the optimizer chart,
   * the accepted-point gauge fix still needs access to the original sparse
   * inactive supports. These counts mirror the pre-expansion orbital-support
   * layout used only as a reference gauge, not as the active optimization
   * parameterization.
   */
  SharedVector<int> mo_gauge_reference_orbital_basis_counts;

  /**
   * @brief Optional pre-expansion sparse basis-index table for MO-gauge canonicalization.
   *
   * Entries follow the same padded row-major layout as
   * `orbital_basis_index_table`, but they retain the original sparse supports
   * from before any inactive-only union-support expansion.
   */
  SharedVector<int> mo_gauge_reference_orbital_basis_index_table;

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
  SharedVector<double> active_orbital_overlap_matrix;

  /**
   * @brief Legacy HF overlap matrix used by GUESS=AUTO block diagonalization.
   *
   * This is distinct from `active_orbital_overlap_matrix` (`vb->ssf`) and is
   * the correct metric for reproducing legacy HF-driven AUTO guesses.
   */
  SharedVector<double> hf_overlap_matrix;

  /**
   * @brief Legacy-detected number of orbital blocks.
   *
   * When present, this should be preferred over C++ block inference so that
   * block-wise guesses match the legacy runtime layout.
   */
  int n_blocks = 0;

  /**
   * @brief Legacy block storage leading dimension.
   *
   * Legacy stores `blocks` as a dense `(n_blocks, block_storage_dimension)`
   * integer table.
   */
  int block_storage_dimension = 0;

  /**
   * @brief Whether legacy detected partially overlapping blocks.
   */
  int block_partial_overlap = 0;

  /**
   * @brief Legacy block membership table.
   *
   * Entries are orbital indices in the legacy `blocks` array layout.
   */
  SharedVector<int> block_members;

  /**
   * @brief Number of orbitals in each legacy block.
   */
  SharedVector<int> block_orbital_counts;

  /**
   * @brief Number of basis functions in the representative orbital of each block.
   */
  SharedVector<int> block_basis_counts;

  /**
   * @brief Legacy AO scaling factors `snorm`.
   *
   * Legacy block guesses are back-scaled with `snorm`, not with a separately
   * inferred AO normalization. Keeping the exact vector is important for parity.
   */
  SharedVector<double> ao_normalization;
};

}  // namespace xmvb::vb
