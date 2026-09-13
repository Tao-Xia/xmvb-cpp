#pragma once

#include <Eigen/Core>

#include <vector>

namespace xmvb::vb {

/**
 * @brief Symmetric AO-pair operator in compressed-row form.
 *
 * Each edge references one value in `AoIntegralInput::ao_two_electron_integral_values`.
 */
struct AoPairGraph {
  std::vector<int> row_offsets;
  std::vector<int> columns;
  std::vector<int> eri_indices;

  bool empty() const noexcept { return row_offsets.empty(); }
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
