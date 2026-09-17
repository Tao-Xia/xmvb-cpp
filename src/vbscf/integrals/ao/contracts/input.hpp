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
 * @brief Exact symmetric AO-pair operator with adaptive storage.
 *
 * Small operators use a symmetric compressed-row graph for fast repeated
 * actions. Operators whose directed edge count exceeds the 32-bit CSR address
 * space retain the canonical permutationally unique integral stream instead.
 */
struct AoPairGraph {
  std::vector<int> row_offsets;
  std::vector<int> columns;
  std::vector<double> values;
  /** CSR rows and edge offsets in original unique-integral order. */
  std::vector<int> integral_rows;
  std::vector<int> integral_edges;
  /**
   * Compact unique-integral payload used when a symmetric CSR graph would
   * exceed its 32-bit edge-address space. Pair-exchange symmetry is retained:
   * every `(row,column,value)` appears exactly once.
   */
  std::vector<int> integral_columns;
  std::vector<double> integral_values;
  std::vector<int> pair_first;
  std::vector<int> pair_second;

  /** @brief Whether this object contains no AO-pair operator. */
  bool empty() const noexcept { return pair_first.empty(); }

  /** @brief Whether the fast symmetric CSR representation is present. */
  bool has_row_graph() const noexcept { return !row_offsets.empty(); }

  /** @brief Whether the canonical unique-integral representation is present. */
  bool has_compact_integral_stream() const noexcept {
    return !integral_values.empty();
  }

  /** @brief Number of unique permutational AO integrals. */
  std::size_t integral_count() const noexcept {
    return integral_rows.size();
  }

  AoPairIntegral integral(std::size_t index) const noexcept {
    const int row = integral_rows[index];
    if (has_compact_integral_stream()) {
      return {row, integral_columns[index], integral_values[index]};
    }
    const int edge = integral_edges[index];
    return {row, columns[edge], values[edge]};
  }

  /** @brief Visits each unique AO integral once as `(value, i, j, k, l)`. */
  template <typename Function>
  void for_each_integral(Function&& function) const {
    for (std::size_t index = 0; index < integral_count(); ++index) {
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
