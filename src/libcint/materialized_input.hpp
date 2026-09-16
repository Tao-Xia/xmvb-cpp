#pragma once

#include <Eigen/Core>

#include <vector>

#include "vbscf/integrals/ao/contracts/input.hpp"

namespace xmvb::vb {

/**
 * @brief Materialized AO integral buffers produced by some integral backend.
 *
 * The backend may be a libcint materializer or any other producer of
 * the same dense/sparse AO buffer contract. Pair indices avoid retaining or
 * staging a redundant four-index AO representation.
 */
struct MaterializedAoIntegralBuffers {
  int n_basis_functions = 0;
  Eigen::MatrixXd ao_core_hamiltonian_matrix;
  std::vector<int> left_pair_indices;
  std::vector<int> right_pair_indices;
  std::vector<double> two_electron_values;
};

AoIntegralInput build_core_hamiltonian_only_ao_integral_input(
    int n_basis_functions,
    Eigen::MatrixXd ao_core_hamiltonian_matrix);

AoIntegralInput build_materialized_ao_integral_input(
    MaterializedAoIntegralBuffers buffers);

}  // namespace xmvb::vb
