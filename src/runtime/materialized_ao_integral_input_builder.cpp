#include "runtime/materialized_ao_integral_input_builder.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "vb/orbital/ao_effective_one_electron_graph_operator.hpp"
#include "vb/orbital/ao_two_electron_pair_index_utils.hpp"

namespace xmvb::vb {

namespace {

std::size_t estimate_ao_effective_one_electron_graph_bytes(
    std::size_t n_integrals,
    int n_basis_functions) {
  if (n_basis_functions <= 0) {
    throw std::invalid_argument("n_basis_functions must be positive");
  }

  constexpr std::size_t kEdgesPerIntegral = 6;
  constexpr std::size_t kBytesPerEdge =
      sizeof(int) + sizeof(double);
  const std::size_t matrix_size =
      n_basis_functions * n_basis_functions;
  if (n_integrals >
      std::numeric_limits<std::size_t>::max() / kEdgesPerIntegral) {
    return std::numeric_limits<std::size_t>::max();
  }
  const std::size_t n_edges = n_integrals * kEdgesPerIntegral;
  if (n_edges >
      std::numeric_limits<std::size_t>::max() / kBytesPerEdge) {
    return std::numeric_limits<std::size_t>::max();
  }
  const std::size_t row_offset_bytes =
      (matrix_size + 1) * sizeof(int);
  if (row_offset_bytes >
      std::numeric_limits<std::size_t>::max() - n_edges * kBytesPerEdge) {
    return std::numeric_limits<std::size_t>::max();
  }
  return row_offset_bytes + n_edges * kBytesPerEdge;
}

}  // namespace

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
    MaterializedAoIntegralBuffers buffers,
    const MaterializedAoIntegralInputBuildOptions& options) {
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
  const std::size_t n_integrals =
      ao_integral_input.ao_two_electron_integral_values.size();
  ao_integral_input.ao_two_electron_integral_symmetry_shifts.resize(
      n_integrals,
      0);
  ao_integral_input.ao_effective_one_electron_linear_indices.resize(
      n_integrals * 10,
      0);

#pragma omp parallel for schedule(static)
  for (std::ptrdiff_t integral_offset = 0;
       integral_offset < static_cast<std::ptrdiff_t>(n_integrals);
       ++integral_offset) {
    const std::size_t integral_index = integral_offset;
    const int i = ao_integral_input.ao_two_electron_integral_indices[integral_index * 4];
    const int j = ao_integral_input.ao_two_electron_integral_indices[integral_index * 4 + 1];
    const int k = ao_integral_input.ao_two_electron_integral_indices[integral_index * 4 + 2];
    const int l = ao_integral_input.ao_two_electron_integral_indices[integral_index * 4 + 3];

    std::uint8_t symmetry_shift = 0;
    if (i == j) {
      ++symmetry_shift;
    }
    if (k == l) {
      ++symmetry_shift;
    }
    if (i == k && j == l) {
      ++symmetry_shift;
    }
    ao_integral_input.ao_two_electron_integral_symmetry_shifts[integral_index] =
        symmetry_shift;

    const int col_i = i * buffers.n_basis_functions;
    const int col_j = j * buffers.n_basis_functions;
    const int col_k = k * buffers.n_basis_functions;
    const int col_l = l * buffers.n_basis_functions;
    int* linear_index_row =
        ao_integral_input.ao_effective_one_electron_linear_indices.data() +
        integral_index * 10;
    linear_index_row[0] = col_j + i;  // ij
    linear_index_row[1] = col_l + k;  // kl
    linear_index_row[2] = col_k + i;  // ik
    linear_index_row[3] = col_l + j;  // jl
    linear_index_row[4] = col_l + i;  // il
    linear_index_row[5] = col_k + j;  // jk
    linear_index_row[6] = col_j + l;  // lj
    linear_index_row[7] = col_i + k;  // ki
    linear_index_row[8] = col_j + k;  // kj
    linear_index_row[9] = col_i + l;  // li
  }
  if (options.build_ao_effective_one_electron_graph) {
    const std::size_t estimated_graph_bytes =
        estimate_ao_effective_one_electron_graph_bytes(
            n_integrals,
            buffers.n_basis_functions);
    if (estimated_graph_bytes <= options.max_ao_effective_one_electron_graph_bytes) {
      const auto graph = build_ao_effective_one_electron_graph(
          ao_integral_input.ao_effective_one_electron_linear_indices,
          ao_integral_input.ao_two_electron_integral_values,
          ao_integral_input.ao_two_electron_integral_symmetry_shifts,
          buffers.n_basis_functions);
      ao_integral_input.ao_effective_one_electron_graph_row_offsets =
          graph.row_offsets;
      ao_integral_input.ao_effective_one_electron_graph_source_indices =
          graph.source_indices;
      ao_integral_input.ao_effective_one_electron_graph_signed_weights =
          graph.signed_weights;
      ao_integral_input.ao_effective_one_electron_graph_transpose_source_offsets =
          graph.transpose_source_offsets;
      ao_integral_input.ao_effective_one_electron_graph_transpose_row_indices =
          graph.transpose_row_indices;
      ao_integral_input.ao_effective_one_electron_graph_transpose_signed_weights =
          graph.transpose_signed_weights;
    }
  }
  std::vector<int> ao_two_electron_pair_indices;
  if (options.build_pair_indices || options.build_pair_graph) {
    ao_two_electron_pair_indices =
        build_ao_two_electron_pair_indices(
            ao_integral_input.ao_two_electron_integral_indices,
            buffers.n_basis_functions);
  }
  if (options.build_pair_graph) {
    const auto pair_graph = build_ao_two_electron_pair_graph(
        ao_two_electron_pair_indices,
        buffers.n_basis_functions);
    ao_integral_input.ao_two_electron_pair_graph_row_offsets =
        pair_graph.row_offsets;
    ao_integral_input.ao_two_electron_pair_graph_column_indices =
        pair_graph.column_pair_indices;
    ao_integral_input.ao_two_electron_pair_graph_integral_indices =
        pair_graph.integral_indices;
  }
  if (options.build_pair_indices) {
    ao_integral_input.ao_two_electron_pair_indices =
        std::move(ao_two_electron_pair_indices);
  }
  return ao_integral_input;
}

}  // namespace xmvb::vb
