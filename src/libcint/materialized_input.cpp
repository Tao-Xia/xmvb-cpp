#include "libcint/materialized_input.hpp"

#include <stdexcept>

#include "vbscf/integrals/ao/pairs/two_electron_index.hpp"

namespace xmvb::vb {

AoIntegralInput build_core_hamiltonian_only_ao_integral_input(
    int n_basis_functions,
    Eigen::MatrixXd ao_core_hamiltonian_matrix) {
  if (n_basis_functions <= 0) {
    throw std::invalid_argument("n_basis_functions must be positive");
  }

  if (ao_core_hamiltonian_matrix.rows() != n_basis_functions ||
      ao_core_hamiltonian_matrix.cols() != n_basis_functions) {
    throw std::invalid_argument(
        "AO core Hamiltonian shape does not match n_basis_functions");
  }

  AoIntegralInput ao_integral_input;
  ao_integral_input.n_basis_functions = n_basis_functions;
  ao_integral_input.ao_core_hamiltonian_matrix =
      std::move(ao_core_hamiltonian_matrix);
  return ao_integral_input;
}

AoIntegralInput build_materialized_ao_integral_input(
    MaterializedAoIntegralBuffers buffers) {
  if (buffers.n_basis_functions <= 0) {
    throw std::invalid_argument("n_basis_functions must be positive");
  }

  if (buffers.ao_core_hamiltonian_matrix.rows() != buffers.n_basis_functions ||
      buffers.ao_core_hamiltonian_matrix.cols() != buffers.n_basis_functions) {
    throw std::invalid_argument(
        "AO core Hamiltonian shape does not match n_basis_functions");
  }
  if (buffers.ao_two_electron_integral_indices.size() !=
      buffers.ao_two_electron_integral_values.size() * 4) {
    throw std::invalid_argument("AO two-electron index/value sizes are inconsistent");
  }

  AoIntegralInput ao_integral_input;
  ao_integral_input.n_basis_functions = buffers.n_basis_functions;
  ao_integral_input.ao_core_hamiltonian_matrix =
      std::move(buffers.ao_core_hamiltonian_matrix);
  ao_integral_input.ao_two_electron_integral_values =
      std::move(buffers.ao_two_electron_integral_values);
  ao_integral_input.ao_two_electron_integral_indices =
      std::move(buffers.ao_two_electron_integral_indices);
  ao_integral_input.pair_graph = build_ao_pair_graph(
      ao_integral_input.ao_two_electron_integral_indices,
      ao_integral_input.ao_two_electron_integral_values,
      buffers.n_basis_functions);
  return ao_integral_input;
}

}  // namespace xmvb::vb
