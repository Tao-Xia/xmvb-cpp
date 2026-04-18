#include "runtime/cpp_closed_shell_fock_builder.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace xmvb::vb {

namespace {

inline std::size_t matrix_index(int row, int column, int n_basis_functions) {
  return xmvb::to_size(column) * n_basis_functions + row;
}

int append_unique_quartet(
    const std::array<int, 32>& candidates,
    int candidate_offset,
    int unique_count,
    std::array<int, 32>* unique_quartets) {
  if (unique_quartets == nullptr) {
    throw std::invalid_argument("unique_quartets must not be null");
  }

  const int a = candidates[xmvb::to_size(candidate_offset)];
  const int b = candidates[xmvb::to_size(candidate_offset) + 1];
  const int c = candidates[xmvb::to_size(candidate_offset) + 2];
  const int d = candidates[xmvb::to_size(candidate_offset) + 3];
  for (int quartet_index = 0; quartet_index < unique_count; ++quartet_index) {
    const int existing_offset = quartet_index * 4;
    if ((*unique_quartets)[xmvb::to_size(existing_offset)] == a &&
        (*unique_quartets)[xmvb::to_size(existing_offset) + 1] == b &&
        (*unique_quartets)[xmvb::to_size(existing_offset) + 2] == c &&
        (*unique_quartets)[xmvb::to_size(existing_offset) + 3] == d) {
      return unique_count;
    }
  }

  const int destination_offset = unique_count * 4;
  (*unique_quartets)[xmvb::to_size(destination_offset)] = a;
  (*unique_quartets)[xmvb::to_size(destination_offset) + 1] = b;
  (*unique_quartets)[xmvb::to_size(destination_offset) + 2] = c;
  (*unique_quartets)[xmvb::to_size(destination_offset) + 3] = d;
  return unique_count + 1;
}

int build_unique_ordered_quartets(
    int i,
    int j,
    int k,
    int l,
    std::array<int, 32>* unique_quartets) {
  if (unique_quartets == nullptr) {
    throw std::invalid_argument("unique_quartets must not be null");
  }

  const std::array<int, 32> candidates = {
      i, j, k, l,
      j, i, k, l,
      i, j, l, k,
      j, i, l, k,
      k, l, i, j,
      l, k, i, j,
      k, l, j, i,
      l, k, j, i,
  };

  int unique_count = 0;
  for (int candidate_index = 0; candidate_index < 8; ++candidate_index) {
    unique_count = append_unique_quartet(
        candidates,
        candidate_index * 4,
        unique_count,
        unique_quartets);
  }
  return unique_count;
}

void accumulate_integral_class(
    double integral_value,
    int i,
    int j,
    int k,
    int l,
    const double* density_projector,
    int n_basis_functions,
    double* local_fock_matrix) {
  std::array<int, 32> quartets = {};
  const int quartet_count =
      build_unique_ordered_quartets(i, j, k, l, &quartets);
  for (int quartet_index = 0; quartet_index < quartet_count; ++quartet_index) {
    const int offset = quartet_index * 4;
    const int a = quartets[xmvb::to_size(offset)];
    const int b = quartets[xmvb::to_size(offset) + 1];
    const int c = quartets[xmvb::to_size(offset) + 2];
    const int d = quartets[xmvb::to_size(offset) + 3];
    local_fock_matrix[matrix_index(a, b, n_basis_functions)] +=
        2.0 * density_projector[matrix_index(c, d, n_basis_functions)] * integral_value;
    local_fock_matrix[matrix_index(a, c, n_basis_functions)] -=
        density_projector[matrix_index(b, d, n_basis_functions)] * integral_value;
  }
}

void symmetrize_in_place(
    std::vector<double>* matrix,
    int n_basis_functions) {
  if (matrix == nullptr) {
    throw std::invalid_argument("matrix must not be null");
  }

  for (int column = 0; column < n_basis_functions; ++column) {
    for (int row = 0; row < column; ++row) {
      const std::size_t upper_index = matrix_index(row, column, n_basis_functions);
      const std::size_t lower_index = matrix_index(column, row, n_basis_functions);
      const double average = 0.5 * ((*matrix)[upper_index] + (*matrix)[lower_index]);
      (*matrix)[upper_index] = average;
      (*matrix)[lower_index] = average;
    }
  }
}

}  // namespace

std::vector<double> CppClosedShellFockBuilder::build(
    const std::vector<double>& density_projector,
    const AoIntegralInput& ao_integral_input) const {
  const int n_basis_functions = ao_integral_input.n_basis_functions;
  if (n_basis_functions <= 0) {
    throw std::invalid_argument("n_basis_functions must be positive");
  }

  const std::size_t matrix_size =
      xmvb::to_size(n_basis_functions) * n_basis_functions;
  if (density_projector.size() != matrix_size) {
    throw std::invalid_argument("density_projector size does not match n_basis_functions");
  }

  const auto& core_hamiltonian_matrix =
      ao_integral_input.ao_core_hamiltonian_matrix.vector();
  const auto& integral_values =
      ao_integral_input.ao_two_electron_integral_values.vector();
  const auto& integral_indices =
      ao_integral_input.ao_two_electron_integral_indices.vector();
  if (core_hamiltonian_matrix.size() != matrix_size) {
    throw std::invalid_argument("ao_core_hamiltonian_matrix size does not match n_basis_functions");
  }
  if (integral_indices.size() != integral_values.size() * 4) {
    throw std::invalid_argument("AO two-electron index/value sizes are inconsistent");
  }

  std::vector<double> fock_matrix = core_hamiltonian_matrix;

  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif

  if (n_threads <= 1) {
    for (std::size_t integral_index = 0; integral_index < integral_values.size(); ++integral_index) {
      const int* indices = integral_indices.data() + integral_index * 4;
      accumulate_integral_class(
          integral_values[integral_index],
          indices[0],
          indices[1],
          indices[2],
          indices[3],
          density_projector.data(),
          n_basis_functions,
          fock_matrix.data());
    }
  } else {
    std::vector<std::vector<double>> partial_fock_matrices(
        xmvb::to_size(n_threads),
        std::vector<double>(matrix_size, 0.0));

#pragma omp parallel
    {
      int thread_index = 0;
#ifdef _OPENMP
      thread_index = omp_get_thread_num();
#endif
      auto& local_fock_matrix = partial_fock_matrices[xmvb::to_size(thread_index)];

#pragma omp for schedule(static)
      for (std::ptrdiff_t integral_offset = 0;
           integral_offset < static_cast<std::ptrdiff_t>(integral_values.size());
           ++integral_offset) {
        const std::size_t integral_index = xmvb::to_size(integral_offset);
        const int* indices = integral_indices.data() + integral_index * 4;
        accumulate_integral_class(
            integral_values[integral_index],
            indices[0],
            indices[1],
            indices[2],
            indices[3],
            density_projector.data(),
            n_basis_functions,
            local_fock_matrix.data());
      }
    }

    for (const auto& local_fock_matrix : partial_fock_matrices) {
      for (std::size_t value_index = 0; value_index < matrix_size; ++value_index) {
        fock_matrix[value_index] += local_fock_matrix[value_index];
      }
    }
  }

  symmetrize_in_place(&fock_matrix, n_basis_functions);
  return fock_matrix;
}

}  // namespace xmvb::vb
