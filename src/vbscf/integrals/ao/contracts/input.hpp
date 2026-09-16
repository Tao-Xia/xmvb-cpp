#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>

namespace xmvb::vb {

struct AoPairIntegral {
  int left_pair;
  int right_pair;
  double value;
};

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
  /** Graph rows and edge offsets in original unique-integral order. */
  std::vector<int> integral_rows;
  std::vector<int> integral_edges;
  std::vector<int> pair_first;
  std::vector<int> pair_second;

  bool empty() const noexcept { return row_offsets.empty(); }

  /** @brief Number of unique permutational AO integrals. */
  std::size_t integral_count() const noexcept {
    return integral_edges.size();
  }

  AoPairIntegral integral(std::size_t index) const noexcept {
    const int row = integral_rows[index];
    const int edge = integral_edges[index];
    return {row, columns[edge], values[edge]};
  }

  /** @brief Visits each unique AO integral once as `(value, i, j, k, l)`. */
  template <typename Function>
  void for_each_integral(Function&& function) const {
    for (std::size_t index = 0; index < integral_edges.size(); ++index) {
      const AoPairIntegral eri = integral(index);
      function(
          eri.value,
          pair_first[eri.left_pair],
          pair_second[eri.left_pair],
          pair_first[eri.right_pair],
          pair_second[eri.right_pair]);
    }
  }

  /** @brief Partitions whole rows into contiguous, edge-balanced ranges. */
  std::vector<std::size_t> balanced_row_boundaries(
      int n_partitions) const;
};

/**
 * @brief AO integral data required by the VBSCF matrix builders.
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

  /** @brief Canonical AO two-electron integral representation. */
  AoPairGraph pair_graph;

};

}  // namespace xmvb::vb
