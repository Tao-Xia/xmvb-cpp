#pragma once

#include <cstddef>
#include <vector>

#include "vb/orbital/ao_integral_input.hpp"

namespace xmvb::vb {

/**
 * @brief Materialized AO integral buffers produced by some integral backend.
 *
 * The backend may be the current legacy runtime or a future C++-native libcint
 * implementation. This builder turns those raw dense/sparse AO buffers into the
 * enriched `AoIntegralInput` used by the exact VBSCF kernels.
 */
struct MaterializedAoIntegralBuffers {
  int n_basis_functions = 0;
  std::vector<double> ao_core_hamiltonian_matrix;
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
    std::vector<double> ao_core_hamiltonian_matrix);

AoIntegralInput build_materialized_ao_integral_input(
    MaterializedAoIntegralBuffers buffers,
    const MaterializedAoIntegralInputBuildOptions& options = {});

}  // namespace xmvb::vb
