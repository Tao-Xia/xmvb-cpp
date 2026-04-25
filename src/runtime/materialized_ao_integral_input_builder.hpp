#pragma once

#include <Eigen/Core>

#include <cstddef>
#include <vector>

#include "vb/orbital/ao_integral_input.hpp"

namespace xmvb::vb {

/**
 * @brief Materialized AO integral buffers produced by some integral backend.
 *
 * The backend may be a pure C++ libcint materializer or any other producer of
 * the same dense/sparse AO buffer contract. This builder turns those raw
 * buffers into the enriched `AoIntegralInput` used by the exact VBSCF kernels.
 */
struct MaterializedAoIntegralBuffers {
  int n_basis_functions = 0;
  Eigen::MatrixXd ao_core_hamiltonian_matrix;
  std::vector<double> ao_two_electron_integral_values;
  std::vector<int> ao_two_electron_integral_indices;
};

struct MaterializedAoIntegralInputBuildOptions {
  bool build_pair_indices = true;
  bool build_pair_graph = false;
  bool build_ao_effective_one_electron_graph = true;
  std::size_t max_ao_effective_one_electron_graph_bytes =
      256ull * 1024ull * 1024ull;
};

AoIntegralInput build_core_hamiltonian_only_ao_integral_input(
    int n_basis_functions,
    Eigen::MatrixXd ao_core_hamiltonian_matrix);

AoIntegralInput build_materialized_ao_integral_input(
    MaterializedAoIntegralBuffers buffers,
    const MaterializedAoIntegralInputBuildOptions& options = {});

}  // namespace xmvb::vb
