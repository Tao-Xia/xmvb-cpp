#include "runtime/cpp_closed_shell_fock_builder.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace xmvb::vb {

namespace {

inline std::size_t matrix_index(
    std::size_t row,
    std::size_t column,
    std::size_t n_basis_functions) {
  return column * n_basis_functions + row;
}

int append_unique_quartet(
    const std::array<int, 32>& candidates,
    int candidate_offset,
    int unique_count,
    std::array<int, 32>* unique_quartets) {
  if (unique_quartets == nullptr) {
    throw std::invalid_argument("unique_quartets must not be null");
  }

  const std::size_t candidate_base = candidate_offset;
  const int a = candidates[candidate_base];
  const int b = candidates[candidate_base + 1];
  const int c = candidates[candidate_base + 2];
  const int d = candidates[candidate_base + 3];
  for (int quartet_index = 0; quartet_index < unique_count; ++quartet_index) {
    const std::size_t existing_base = quartet_index * 4;
    if ((*unique_quartets)[existing_base] == a &&
        (*unique_quartets)[existing_base + 1] == b &&
        (*unique_quartets)[existing_base + 2] == c &&
        (*unique_quartets)[existing_base + 3] == d) {
      return unique_count;
    }
  }

  const std::size_t destination_base = unique_count * 4;
  (*unique_quartets)[destination_base] = a;
  (*unique_quartets)[destination_base + 1] = b;
  (*unique_quartets)[destination_base + 2] = c;
  (*unique_quartets)[destination_base + 3] = d;
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
    const std::size_t quartet_base = quartet_index * 4;
    const int a = quartets[quartet_base];
    const int b = quartets[quartet_base + 1];
    const int c = quartets[quartet_base + 2];
    const int d = quartets[quartet_base + 3];
    local_fock_matrix[matrix_index(a, b, n_basis_functions)] +=
        2.0 * density_projector[matrix_index(c, d, n_basis_functions)] *
        integral_value;
    local_fock_matrix[matrix_index(a, c, n_basis_functions)] -=
        density_projector[matrix_index(b, d, n_basis_functions)] *
        integral_value;
  }
}

void symmetrize_in_place(Eigen::MatrixXd* matrix) {
  if (matrix == nullptr) {
    throw std::invalid_argument("matrix must not be null");
  }
  const int n_basis_functions = matrix->rows();
  for (int column = 0; column < n_basis_functions; ++column) {
    for (int row = 0; row < column; ++row) {
      const double average = 0.5 * ((*matrix)(row, column) + (*matrix)(column, row));
      (*matrix)(row, column) = average;
      (*matrix)(column, row) = average;
    }
  }
}

}  // namespace

Eigen::MatrixXd CppClosedShellFockBuilder::build(
    const Eigen::Ref<const Eigen::MatrixXd>& density_projector,
    const AoIntegralInput& ao_integral_input) const {
  const int n_basis_functions = ao_integral_input.n_basis_functions;
  if (n_basis_functions <= 0) {
    throw std::invalid_argument("n_basis_functions must be positive");
  }

  const std::size_t matrix_size =
      static_cast<std::size_t>(n_basis_functions) * n_basis_functions;
  if (density_projector.rows() != n_basis_functions ||
      density_projector.cols() != n_basis_functions) {
    throw std::invalid_argument("density_projector shape does not match n_basis_functions");
  }

  const auto& core_hamiltonian_matrix =
      ao_integral_input.ao_core_hamiltonian_matrix;
  const auto& integral_values =
      ao_integral_input.ao_two_electron_integral_values;
  const auto& integral_indices =
      ao_integral_input.ao_two_electron_integral_indices;
  if (core_hamiltonian_matrix.rows() != n_basis_functions ||
      core_hamiltonian_matrix.cols() != n_basis_functions) {
    throw std::invalid_argument(
        "ao_core_hamiltonian_matrix shape does not match n_basis_functions");
  }
  if (integral_indices.size() != integral_values.size() * 4) {
    throw std::invalid_argument("AO two-electron index/value sizes are inconsistent");
  }

  Eigen::MatrixXd fock_matrix = core_hamiltonian_matrix;

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
        n_threads,
        std::vector<double>(matrix_size, 0.0));

#pragma omp parallel
    {
      int thread_index = 0;
#ifdef _OPENMP
      thread_index = omp_get_thread_num();
#endif
      auto& local_fock_matrix = partial_fock_matrices[thread_index];

#pragma omp for schedule(static)
      for (std::ptrdiff_t integral_offset = 0;
           integral_offset < static_cast<std::ptrdiff_t>(integral_values.size());
           ++integral_offset) {
        const std::size_t integral_index = integral_offset;
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
        fock_matrix.data()[value_index] += local_fock_matrix[value_index];
      }
    }
  }

  symmetrize_in_place(&fock_matrix);
  return fock_matrix;
}

}  // namespace xmvb::vb
