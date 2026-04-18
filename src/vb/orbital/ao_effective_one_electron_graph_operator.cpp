#include "vb/orbital/ao_effective_one_electron_graph_operator.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace xmvb::vb {

namespace {

constexpr double kIntegralSymmetryMultipliers[] = {
    1.0,
    0.5,
    0.25,
    0.125,
};

std::size_t ao_matrix_size(int n_basis_functions) {
  if (n_basis_functions <= 0) {
    throw std::invalid_argument("n_basis_functions must be positive");
  }
  return xmvb::to_size(n_basis_functions) * n_basis_functions;
}

void validate_graph_storage_shapes(
    const AoIntegralInput& ao_integral_input,
    std::size_t matrix_size) {
  const auto& row_offsets =
      ao_integral_input.ao_effective_one_electron_graph_row_offsets;
  const auto& source_indices =
      ao_integral_input.ao_effective_one_electron_graph_source_indices;
  const auto& signed_weights =
      ao_integral_input.ao_effective_one_electron_graph_signed_weights;
  if (row_offsets.size() != matrix_size + 1) {
    throw std::invalid_argument(
        "AO-H1E graph row-offset size does not match AO matrix size");
  }
  if (row_offsets.empty()) {
    throw std::invalid_argument("AO-H1E graph row offsets must not be empty");
  }
  if (row_offsets.front() != 0) {
    throw std::invalid_argument("AO-H1E graph row offsets must start at zero");
  }
  if (row_offsets.back() < 0) {
    throw std::invalid_argument("AO-H1E graph edge count must be nonnegative");
  }
  if (source_indices.size() != xmvb::to_size(row_offsets.back()) ||
      signed_weights.size() != xmvb::to_size(row_offsets.back())) {
    throw std::invalid_argument(
        "AO-H1E graph edge arrays do not match the stored edge count");
  }
}

void validate_transpose_graph_storage_shapes(
    const AoIntegralInput& ao_integral_input,
    std::size_t matrix_size) {
  const auto& source_offsets =
      ao_integral_input.ao_effective_one_electron_graph_transpose_source_offsets;
  const auto& row_indices =
      ao_integral_input.ao_effective_one_electron_graph_transpose_row_indices;
  const auto& signed_weights =
      ao_integral_input.ao_effective_one_electron_graph_transpose_signed_weights;
  if (source_offsets.empty() && row_indices.empty() && signed_weights.empty()) {
    return;
  }
  if (source_offsets.size() != matrix_size + 1) {
    throw std::invalid_argument(
        "AO-H1E transpose graph source-offset size does not match AO matrix size");
  }
  if (source_offsets.front() != 0) {
    throw std::invalid_argument(
        "AO-H1E transpose graph source offsets must start at zero");
  }
  if (source_offsets.back() < 0) {
    throw std::invalid_argument(
        "AO-H1E transpose graph edge count must be nonnegative");
  }
  if (row_indices.size() != xmvb::to_size(source_offsets.back()) ||
      signed_weights.size() != xmvb::to_size(source_offsets.back())) {
    throw std::invalid_argument(
        "AO-H1E transpose graph edge arrays do not match the stored edge count");
  }
}

bool transpose_graph_available(const AoIntegralInput& ao_integral_input) noexcept {
  return !ao_integral_input
              .ao_effective_one_electron_graph_transpose_source_offsets.empty();
}

int sanitize_graph_thread_count(int n_threads) {
  if (n_threads <= 0) {
    throw std::invalid_argument("AO-H1E graph thread count must be positive");
  }
  return n_threads;
}

}  // namespace

AoEffectiveOneElectronGraphBuffers build_ao_effective_one_electron_graph(
    const std::vector<int>& ao_effective_one_electron_linear_indices,
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<std::uint8_t>& ao_two_electron_integral_symmetry_shifts,
    int n_basis_functions) {
  const std::size_t matrix_size = ao_matrix_size(n_basis_functions);
  if (ao_effective_one_electron_linear_indices.size() % 10 != 0) {
    throw std::invalid_argument(
        "AO-H1E graph requires 10 linear indices per AO integral");
  }
  const std::size_t n_integrals =
      ao_effective_one_electron_linear_indices.size() / 10;
  if (ao_two_electron_integral_values.size() != n_integrals ||
      ao_two_electron_integral_symmetry_shifts.size() != n_integrals) {
    throw std::invalid_argument(
        "AO-H1E graph integral buffers have inconsistent sizes");
  }

  constexpr std::size_t kEdgesPerIntegral = 6;
  if (n_integrals >
      xmvb::to_size(std::numeric_limits<int>::max()) / kEdgesPerIntegral) {
    throw std::overflow_error("AO-H1E graph edge count exceeds 32-bit storage");
  }
  const std::size_t n_edges = n_integrals * kEdgesPerIntegral;

  std::vector<int> row_counts(matrix_size, 0);
  for (std::size_t integral_index = 0; integral_index < n_integrals; ++integral_index) {
    const int* linear_indices =
        ao_effective_one_electron_linear_indices.data() + integral_index * 10;
    for (int entry_index = 0; entry_index < 6; ++entry_index) {
      const int row_index = linear_indices[entry_index];
      if (row_index < 0 || xmvb::to_size(row_index) >= matrix_size) {
        throw std::invalid_argument("AO-H1E graph destination row is out of range");
      }
      ++row_counts[xmvb::to_size(row_index)];
    }
    const int source_entries[] = {
        linear_indices[1],
        linear_indices[0],
        linear_indices[6],
        linear_indices[7],
        linear_indices[8],
        linear_indices[9],
    };
    for (int source_entry : source_entries) {
      if (source_entry < 0 || xmvb::to_size(source_entry) >= matrix_size) {
        throw std::invalid_argument("AO-H1E graph source row is out of range");
      }
    }
  }

  AoEffectiveOneElectronGraphBuffers graph;
  graph.row_offsets.resize(matrix_size + 1, 0);
  for (std::size_t row_index = 0; row_index < matrix_size; ++row_index) {
    graph.row_offsets[row_index + 1] =
        graph.row_offsets[row_index] + row_counts[row_index];
  }
  graph.source_indices.resize(n_edges, 0);
  graph.signed_weights.resize(n_edges, 0.0);

  std::vector<int> next_offsets = graph.row_offsets;
  for (std::size_t integral_index = 0; integral_index < n_integrals; ++integral_index) {
    const int* linear_indices =
        ao_effective_one_electron_linear_indices.data() + integral_index * 10;
    const std::uint8_t symmetry_shift =
        ao_two_electron_integral_symmetry_shifts[integral_index];
    if (symmetry_shift >= 4) {
      throw std::invalid_argument("AO-H1E graph symmetry shift is out of range");
    }
    const double scaled_value =
        ao_two_electron_integral_values[integral_index] *
        kIntegralSymmetryMultipliers[symmetry_shift];

    const int destination_entries[] = {
        linear_indices[0],
        linear_indices[1],
        linear_indices[2],
        linear_indices[3],
        linear_indices[4],
        linear_indices[5],
    };
    const int source_entries[] = {
        linear_indices[1],
        linear_indices[0],
        linear_indices[6],
        linear_indices[7],
        linear_indices[8],
        linear_indices[9],
    };
    const double signed_edge_weights[] = {
        4.0 * scaled_value,
        4.0 * scaled_value,
        -scaled_value,
        -scaled_value,
        -scaled_value,
        -scaled_value,
    };
    for (int entry_index = 0; entry_index < 6; ++entry_index) {
      const int row_index = destination_entries[entry_index];
      const int destination_offset = next_offsets[xmvb::to_size(row_index)]++;
      graph.source_indices[xmvb::to_size(destination_offset)] =
          source_entries[entry_index];
      graph.signed_weights[xmvb::to_size(destination_offset)] =
          signed_edge_weights[entry_index];
    }
  }

  std::vector<int> source_counts(matrix_size, 0);
  for (std::size_t edge_index = 0; edge_index < n_edges; ++edge_index) {
    ++source_counts[xmvb::to_size(graph.source_indices[edge_index])];
  }
  graph.transpose_source_offsets.resize(matrix_size + 1, 0);
  for (std::size_t source_index = 0; source_index < matrix_size; ++source_index) {
    graph.transpose_source_offsets[source_index + 1] =
        graph.transpose_source_offsets[source_index] + source_counts[source_index];
  }
  graph.transpose_row_indices.resize(n_edges, 0);
  graph.transpose_signed_weights.resize(n_edges, 0.0);
  std::vector<int> next_source_offsets = graph.transpose_source_offsets;
  for (std::size_t row_index = 0; row_index < matrix_size; ++row_index) {
    for (int edge_offset = graph.row_offsets[row_index];
         edge_offset < graph.row_offsets[row_index + 1];
         ++edge_offset) {
      const std::size_t edge_index = xmvb::to_size(edge_offset);
      const int source_index = graph.source_indices[edge_index];
      const int destination_offset =
          next_source_offsets[xmvb::to_size(source_index)]++;
      graph.transpose_row_indices[xmvb::to_size(destination_offset)] =
          static_cast<int>(row_index);
      graph.transpose_signed_weights[xmvb::to_size(destination_offset)] =
          graph.signed_weights[edge_index];
    }
  }

  return graph;
}
bool ao_effective_one_electron_graph_available(
    const AoIntegralInput& ao_integral_input) noexcept {
  return !ao_integral_input.ao_effective_one_electron_graph_row_offsets.empty();
}

std::vector<double> apply_ao_effective_one_electron_graph_forward(
    const double* source_matrix_storage,
    const AoIntegralInput& ao_integral_input) {
  return apply_ao_effective_one_electron_graph_forward(
      source_matrix_storage,
      ao_integral_input,
      1);
}

std::vector<double> apply_ao_effective_one_electron_graph_forward(
    const double* source_matrix_storage,
    const AoIntegralInput& ao_integral_input,
    int n_threads) {
  if (source_matrix_storage == nullptr) {
    throw std::invalid_argument("AO-H1E graph forward source must not be null");
  }
  const std::size_t matrix_size =
      ao_matrix_size(ao_integral_input.n_basis_functions);
  validate_graph_storage_shapes(ao_integral_input, matrix_size);
  n_threads = sanitize_graph_thread_count(n_threads);

  std::vector<double> result(matrix_size, 0.0);
  const auto& row_offsets =
      ao_integral_input.ao_effective_one_electron_graph_row_offsets;
  const auto& source_indices =
      ao_integral_input.ao_effective_one_electron_graph_source_indices;
  const auto& signed_weights =
      ao_integral_input.ao_effective_one_electron_graph_signed_weights;

  if (n_threads <= 1) {
    for (std::size_t row_index = 0; row_index < matrix_size; ++row_index) {
      double row_value = 0.0;
      for (int edge_offset = row_offsets[row_index];
           edge_offset < row_offsets[row_index + 1];
           ++edge_offset) {
        row_value +=
            signed_weights[xmvb::to_size(edge_offset)] *
            source_matrix_storage[
                xmvb::to_size(source_indices[xmvb::to_size(edge_offset)])];
      }
      result[row_index] = row_value;
    }
    return result;
  }

#pragma omp parallel for schedule(static) num_threads(n_threads)
  for (std::ptrdiff_t row_offset = 0;
       row_offset < static_cast<std::ptrdiff_t>(matrix_size);
       ++row_offset) {
    const std::size_t row_index = xmvb::to_size(row_offset);
    double row_value = 0.0;
    for (int edge_offset = row_offsets[row_index];
         edge_offset < row_offsets[row_index + 1];
         ++edge_offset) {
      row_value +=
          signed_weights[xmvb::to_size(edge_offset)] *
          source_matrix_storage[
              xmvb::to_size(source_indices[xmvb::to_size(edge_offset)])];
    }
    result[row_index] = row_value;
  }
  return result;
}

std::vector<double> apply_ao_effective_one_electron_graph_transpose(
    const double* row_adjoint_storage,
    const AoIntegralInput& ao_integral_input) {
  if (row_adjoint_storage == nullptr) {
    throw std::invalid_argument("AO-H1E graph transpose source must not be null");
  }
  const std::size_t matrix_size =
      ao_matrix_size(ao_integral_input.n_basis_functions);
  validate_graph_storage_shapes(ao_integral_input, matrix_size);
  validate_transpose_graph_storage_shapes(ao_integral_input, matrix_size);

  std::vector<double> result(matrix_size, 0.0);

  if (transpose_graph_available(ao_integral_input)) {
    const auto& source_offsets =
        ao_integral_input
            .ao_effective_one_electron_graph_transpose_source_offsets;
    const auto& row_indices =
        ao_integral_input.ao_effective_one_electron_graph_transpose_row_indices;
    const auto& signed_weights =
        ao_integral_input
            .ao_effective_one_electron_graph_transpose_signed_weights;
    for (std::size_t source_index = 0; source_index < matrix_size; ++source_index) {
      double source_value = 0.0;
      for (int edge_offset = source_offsets[source_index];
           edge_offset < source_offsets[source_index + 1];
           ++edge_offset) {
        const std::size_t edge_index = xmvb::to_size(edge_offset);
        source_value +=
            signed_weights[edge_index] *
            row_adjoint_storage[xmvb::to_size(row_indices[edge_index])];
      }
      result[source_index] = source_value;
    }
    return result;
  }

  const auto& row_offsets =
      ao_integral_input.ao_effective_one_electron_graph_row_offsets;
  const auto& source_indices =
      ao_integral_input.ao_effective_one_electron_graph_source_indices;
  const auto& signed_weights =
      ao_integral_input.ao_effective_one_electron_graph_signed_weights;
  for (std::size_t row_index = 0; row_index < matrix_size; ++row_index) {
    const double row_adjoint = row_adjoint_storage[row_index];
    if (row_adjoint == 0.0) {
      continue;
    }
    for (int edge_offset = row_offsets[row_index];
         edge_offset < row_offsets[row_index + 1];
         ++edge_offset) {
      result[xmvb::to_size(source_indices[xmvb::to_size(edge_offset)])] +=
          signed_weights[xmvb::to_size(edge_offset)] * row_adjoint;
    }
  }
  return result;
}

AoEffectiveOneElectronGraphFusedResult
apply_fused_ao_effective_one_electron_graph(
    const double* source_matrix_storage,
    const double* row_adjoint_storage,
    const AoIntegralInput& ao_integral_input) {
  return apply_fused_ao_effective_one_electron_graph(
      source_matrix_storage,
      row_adjoint_storage,
      ao_integral_input,
      1);
}

AoEffectiveOneElectronGraphFusedResult
apply_fused_ao_effective_one_electron_graph(
    const double* source_matrix_storage,
    const double* row_adjoint_storage,
    const AoIntegralInput& ao_integral_input,
    int n_threads) {
  AoEffectiveOneElectronGraphFusedResult result;
  apply_fused_ao_effective_one_electron_graph(
      source_matrix_storage,
      row_adjoint_storage,
      ao_integral_input,
      n_threads,
      &result.forward_output,
      &result.transpose_output);
  return result;
}

void apply_fused_ao_effective_one_electron_graph(
    const double* source_matrix_storage,
    const double* row_adjoint_storage,
    const AoIntegralInput& ao_integral_input,
    std::vector<double>* forward_output,
    std::vector<double>* transpose_output) {
  apply_fused_ao_effective_one_electron_graph(
      source_matrix_storage,
      row_adjoint_storage,
      ao_integral_input,
      1,
      forward_output,
      transpose_output);
}

void apply_fused_ao_effective_one_electron_graph(
    const double* source_matrix_storage,
    const double* row_adjoint_storage,
    const AoIntegralInput& ao_integral_input,
    int n_threads,
    std::vector<double>* forward_output,
    std::vector<double>* transpose_output) {
  if (source_matrix_storage == nullptr || row_adjoint_storage == nullptr) {
    throw std::invalid_argument("AO-H1E graph fused inputs must not be null");
  }
  if (forward_output == nullptr || transpose_output == nullptr) {
    throw std::invalid_argument("AO-H1E graph fused outputs must not be null");
  }
  const std::size_t matrix_size =
      ao_matrix_size(ao_integral_input.n_basis_functions);
  validate_graph_storage_shapes(ao_integral_input, matrix_size);
  validate_transpose_graph_storage_shapes(ao_integral_input, matrix_size);
  n_threads = sanitize_graph_thread_count(n_threads);

  forward_output->assign(matrix_size, 0.0);
  transpose_output->assign(matrix_size, 0.0);
  const auto& row_offsets =
      ao_integral_input.ao_effective_one_electron_graph_row_offsets;
  const auto& source_indices =
      ao_integral_input.ao_effective_one_electron_graph_source_indices;
  const auto& signed_weights =
      ao_integral_input.ao_effective_one_electron_graph_signed_weights;

  // The graph rows correspond to destination AO matrix entries. A single row
  // sweep therefore computes both `K * x` and `K^T * lambda`: the row-local
  // accumulator forms the forward output, while each edge contributes its
  // transpose pullback into the source AO entry referenced by that edge.
  if (n_threads <= 1) {
    for (std::size_t row_index = 0; row_index < matrix_size; ++row_index) {
      double row_forward_value = 0.0;
      const double row_adjoint = row_adjoint_storage[row_index];
      for (int edge_offset = row_offsets[row_index];
           edge_offset < row_offsets[row_index + 1];
           ++edge_offset) {
        const std::size_t edge_index = xmvb::to_size(edge_offset);
        const std::size_t source_index =
            xmvb::to_size(source_indices[edge_index]);
        const double signed_weight = signed_weights[edge_index];
        row_forward_value += signed_weight * source_matrix_storage[source_index];
        (*transpose_output)[source_index] += signed_weight * row_adjoint;
      }
      (*forward_output)[row_index] = row_forward_value;
    }
    return;
  }
#pragma omp parallel for schedule(static) num_threads(n_threads)
  for (std::ptrdiff_t row_offset = 0;
       row_offset < static_cast<std::ptrdiff_t>(matrix_size);
       ++row_offset) {
    const std::size_t row_index = xmvb::to_size(row_offset);
    double row_forward_value = 0.0;
    for (int edge_offset = row_offsets[row_index];
         edge_offset < row_offsets[row_index + 1];
         ++edge_offset) {
      const std::size_t edge_index = xmvb::to_size(edge_offset);
      const std::size_t source_index =
          xmvb::to_size(source_indices[edge_index]);
      row_forward_value += signed_weights[edge_index] *
          source_matrix_storage[source_index];
    }
    (*forward_output)[row_index] = row_forward_value;
  }

  if (transpose_graph_available(ao_integral_input)) {
    const auto& source_offsets =
        ao_integral_input
            .ao_effective_one_electron_graph_transpose_source_offsets;
    const auto& row_indices =
        ao_integral_input.ao_effective_one_electron_graph_transpose_row_indices;
    const auto& transpose_signed_weights =
        ao_integral_input
            .ao_effective_one_electron_graph_transpose_signed_weights;

    // The source-owned transpose graph lets each thread write a disjoint block
    // of `K^T * lambda`, so the fused exact_ctx kernel no longer needs one
    // full AO-matrix-sized transpose buffer per OpenMP worker.
#pragma omp parallel for schedule(static) num_threads(n_threads)
    for (std::ptrdiff_t source_offset = 0;
         source_offset < static_cast<std::ptrdiff_t>(matrix_size);
         ++source_offset) {
      const std::size_t source_index = xmvb::to_size(source_offset);
      double source_value = 0.0;
      for (int edge_offset = source_offsets[source_index];
           edge_offset < source_offsets[source_index + 1];
           ++edge_offset) {
        const std::size_t edge_index = xmvb::to_size(edge_offset);
        source_value +=
            transpose_signed_weights[edge_index] *
            row_adjoint_storage[xmvb::to_size(row_indices[edge_index])];
      }
      (*transpose_output)[source_index] = source_value;
    }
    return;
  }

  std::vector<std::vector<double>> partial_transpose_outputs(
      xmvb::to_size(n_threads),
      std::vector<double>(matrix_size, 0.0));

#pragma omp parallel num_threads(n_threads)
  {
    int thread_index = 0;
#ifdef _OPENMP
    thread_index = omp_get_thread_num();
#endif
    auto& local_transpose_output =
        partial_transpose_outputs[xmvb::to_size(thread_index)];

#pragma omp for schedule(static)
    for (std::ptrdiff_t row_offset = 0;
         row_offset < static_cast<std::ptrdiff_t>(matrix_size);
         ++row_offset) {
      const std::size_t row_index = xmvb::to_size(row_offset);
      const double row_adjoint = row_adjoint_storage[row_index];
      for (int edge_offset = row_offsets[row_index];
           edge_offset < row_offsets[row_index + 1];
           ++edge_offset) {
        const std::size_t edge_index = xmvb::to_size(edge_offset);
        const std::size_t source_index =
            xmvb::to_size(source_indices[edge_index]);
        local_transpose_output[source_index] +=
            signed_weights[edge_index] * row_adjoint;
      }
    }
  }

  for (const auto& partial_transpose_output : partial_transpose_outputs) {
    for (std::size_t index = 0; index < matrix_size; ++index) {
      (*transpose_output)[index] += partial_transpose_output[index];
    }
  }
}

}  // namespace xmvb::vb
