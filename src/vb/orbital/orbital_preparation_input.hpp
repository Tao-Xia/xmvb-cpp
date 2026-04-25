#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>

namespace xmvb::vb {

/**
 * @brief Input data required to rebuild auxiliary orbitals in C++.
 */
struct OrbitalPreparationInput {
  /**
   * @brief Number of basis functions.
   */
  std::size_t n_basis_functions = 0;

  /**
   * @brief Number of orbitals stored in the VB orbital parameterization.
   */
  std::size_t n_orbitals = 0;

  /**
   * @brief Number of active orbitals.
   */
  std::size_t n_active_orbitals = 0;

  /**
   * @brief Number of total electrons.
   */
  std::size_t n_total_electrons = 0;

  /**
   * @brief Number of active electrons.
   */
  std::size_t n_active_electrons = 0;

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
  std::vector<double> orbital_value_table;

  /**
   * @brief One-based basis-function indices corresponding to `orbital_value_table`.
   */
  std::vector<int> orbital_basis_index_table;

  /**
   * @brief Number of explicit coefficients stored for each orbital.
   */
  std::vector<int> orbital_basis_counts;

  /**
   * @brief Optional pre-expansion sparse support counts for MO-gauge canonicalization.
   *
   * When `GUESS=MO` needs inactive-support expansion for the optimizer chart,
   * the accepted-point gauge fix still needs access to the original sparse
   * inactive supports. These counts mirror the pre-expansion orbital-support
   * layout used only as a reference gauge, not as the active optimization
   * parameterization.
   */
  std::vector<int> mo_gauge_reference_orbital_basis_counts;

  /**
   * @brief Optional pre-expansion sparse basis-index table for MO-gauge canonicalization.
   *
   * Entries follow the same padded row-major layout as
   * `orbital_basis_index_table`, but they retain the original sparse supports
   * from before any inactive-only union-support expansion.
   */
  std::vector<int> mo_gauge_reference_orbital_basis_index_table;

  /**
   * @brief Original parameter-space coefficient counts for each orbital.
   *
   * This corresponds to legacy `ma0`, which is the count used to enumerate
   * variational parameters and to accumulate the final orbital gradient.
   */
  std::vector<int> original_orbital_basis_counts;

  /**
   * @brief Column-major AO overlap matrix.
   */
  Eigen::MatrixXd active_orbital_overlap_matrix;

  /**
   * @brief Legacy HF overlap matrix used by GUESS=AUTO block diagonalization.
   *
   * This is distinct from `active_orbital_overlap_matrix` (`vb->ssf`) and is
   * the correct metric for reproducing legacy HF-driven AUTO guesses. When
   * this matrix is empty, the C++ guess path reuses
   * `active_orbital_overlap_matrix` instead of storing a duplicate copy.
   */
  Eigen::MatrixXd hf_overlap_matrix;

  /**
   * @brief Legacy-detected number of orbital blocks.
   *
   * When present, this should be preferred over C++ block inference so that
   * block-wise guesses match the legacy runtime layout.
   */
  std::size_t n_blocks = 0;

  /**
   * @brief Legacy block storage leading dimension.
   *
   * Legacy stores `blocks` as a dense `(n_blocks, block_storage_dimension)`
   * integer table.
   */
  std::size_t block_storage_dimension = 0;

  /**
   * @brief Whether legacy detected partially overlapping blocks.
   */
  int block_partial_overlap = 0;

  /**
   * @brief Legacy block membership table.
   *
   * Entries are orbital indices in the legacy `blocks` array layout.
   */
  std::vector<int> block_members;

  /**
   * @brief Number of orbitals in each legacy block.
   */
  std::vector<int> block_orbital_counts;

  /**
   * @brief Number of basis functions in the representative orbital of each block.
   */
  std::vector<int> block_basis_counts;

  /**
   * @brief Legacy AO scaling factors `snorm`.
   *
   * Legacy block guesses are back-scaled with `snorm`, not with a separately
   * inferred AO normalization. Keeping the exact vector is important for parity.
   */
  std::vector<double> ao_normalization;
};

/**
 * @brief Returns the stored sparse support length for an orbital.
 *
 * The legacy input sometimes stores `orbital_basis_counts == 0/1` for cases
 * where the padded one-based basis-index row is the authoritative support
 * description. This helper preserves that storage convention and returns the
 * number of occupied slots in the padded row.
 */
inline int stored_sparse_orbital_coefficient_count(
    const OrbitalPreparationInput& input,
    std::size_t orbital_index) {
  const int explicit_count = input.orbital_basis_counts[orbital_index];
  if (explicit_count > 1) {
    return explicit_count;
  }

  int coefficient_count = 0;
  while (coefficient_count < static_cast<int>(input.n_basis_functions)) {
    const int basis_function_index =
        input.orbital_basis_index_table
            [orbital_index * input.n_basis_functions + coefficient_count];
    if (basis_function_index == 0) {
      break;
    }
    ++coefficient_count;
  }
  return coefficient_count;
}

/**
 * @brief Returns the number of variational sparse orbital parameters.
 *
 * This differs from the stored support length for expanded-support HAO/MO
 * cases: `original_orbital_basis_counts` mirrors the legacy `ma0` parameter
 * count and should be used when enumerating differentiable orbital variables.
 */
inline int differentiable_sparse_orbital_parameter_count(
    const OrbitalPreparationInput& input,
    std::size_t orbital_index) {
  const bool have_original_counts =
      input.original_orbital_basis_counts.size() == input.n_orbitals;
  const int explicit_count = have_original_counts
      ? input.original_orbital_basis_counts[orbital_index]
      : input.orbital_basis_counts[orbital_index];
  if (explicit_count > 1) {
    return explicit_count;
  }
  if (explicit_count == 1) {
    return 1;
  }
  return stored_sparse_orbital_coefficient_count(input, orbital_index);
}

}  // namespace xmvb::vb
