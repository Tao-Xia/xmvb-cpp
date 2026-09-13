#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>

namespace xmvb::vb {

/**
 * @brief Symmetric AO-pair operator in compressed-row form.
 *
 * Column and value arrays share the same edge ordering so repeated AO-pair
 * actions stream both operands without indirect integral-value lookups.
 */
struct AoPairGraph {
  std::vector<int> row_offsets;
  std::vector<int> columns;
  std::vector<double> values;

  bool empty() const noexcept { return row_offsets.empty(); }

  /** @brief Partitions whole rows into contiguous, edge-balanced ranges. */
  std::vector<std::size_t> balanced_row_boundaries(
      int n_partitions) const;
};

/**
 * @brief Raw AO integral data required by the VBSCF matrix builders.
 */
struct AoIntegralInput {
  /**
   * @brief Number of AO basis functions.
   */
  int n_basis_functions = 0;

  /**
   * @brief Column-major AO core Hamiltonian matrix `HHF`.
   */
  Eigen::MatrixXd ao_core_hamiltonian_matrix;

  /**
   * @brief Sparse AO two-electron integral values `ggf`.
   */
  std::vector<double> ao_two_electron_integral_values;

  /**
   * @brief Flattened AO two-electron index table `g2eidx` with 4 entries per integral.
   */
  std::vector<int> ao_two_electron_integral_indices;

  /** @brief Symmetric operator used for AO-pair contractions. */
  AoPairGraph pair_graph;

};

}  // namespace xmvb::vb
