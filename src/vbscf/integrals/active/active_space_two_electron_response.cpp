#include "vbscf/integrals/active/active_space_two_electron_response.hpp"
#include <atomic>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "core/openmp_utils.hpp"
#include "vbscf/core/eigen_storage.hpp"
#include "vbscf/integrals/active/two_electron_indexer.hpp"

namespace xmvb::vb {

namespace {

struct ActivePair {
  int first = 0;
  int second = 0;
};

std::size_t ao_pair_index(int first, int second) {
  if (first >= second) {
    const std::size_t first_index = first;
    return first_index * (first_index + 1) / 2 + second;
  }
  const std::size_t second_index = second;
  return second_index * (second_index + 1) / 2 + first;
}

std::vector<std::size_t> build_pair_row_offsets_hvp(
    std::size_t n_ao_pairs,
    std::size_t n_active_pairs) {
  std::vector<std::size_t> row_offsets(n_ao_pairs, 0);
  for (std::size_t ao_pair_offset = 0; ao_pair_offset < n_ao_pairs; ++ao_pair_offset) {
    row_offsets[ao_pair_offset] = ao_pair_offset * n_active_pairs;
  }
  return row_offsets;
}

void build_ao_pair_component_tables(
    int n_basis_functions,
    std::vector<int>* first_indices,
    std::vector<int>* second_indices) {
  if (first_indices == nullptr || second_indices == nullptr) {
    throw std::invalid_argument("AO pair component tables must not be null");
  }
  const std::size_t basis_count = n_basis_functions;
  const std::size_t n_ao_pairs = basis_count * (basis_count + 1) / 2;
  first_indices->assign(n_ao_pairs, 0);
  second_indices->assign(n_ao_pairs, 0);
  std::size_t pair_index = 0;
  for (int first_basis_function = 0;
       first_basis_function < n_basis_functions;
       ++first_basis_function) {
    for (int second_basis_function = 0;
         second_basis_function <= first_basis_function;
         ++second_basis_function) {
      (*first_indices)[pair_index] = first_basis_function;
      (*second_indices)[pair_index] = second_basis_function;
      ++pair_index;
    }
  }
}

void resize_for_overwrite(
    std::vector<double>* values,
    std::size_t size) {
  if (values == nullptr) {
    throw std::invalid_argument("workspace buffer must not be null");
  }
  if (values->size() != size) {
    values->resize(size);
  }
}

void resize_and_zero(
    std::vector<double>* values,
    std::size_t size) {
  if (values == nullptr) {
    throw std::invalid_argument("workspace buffer must not be null");
  }
  values->assign(size, 0.0);
}

template <int NActivePairs>
inline void accumulate_scaled_active_pair_row_hvp(
    double* target_row,
    const double* source_row,
    double scale) {
#pragma omp simd
  for (int active_pair_index = 0;
       active_pair_index < NActivePairs;
       ++active_pair_index) {
    target_row[active_pair_index] += scale * source_row[active_pair_index];
  }
}

template <int NActivePairs>
std::vector<double> apply_sparse_ao_integral_matrix_single_thread_fixed_hvp(
    const double* ao_two_electron_integral_values_data,
    const int* ao_two_electron_pair_indices_data,
    const double* transformed_pair_coefficients_data,
    const std::size_t* pair_row_offsets_data,
    std::size_t n_ao_pairs,
    std::size_t n_integrals) {
  std::vector<double> pair_gradients(
      n_ao_pairs * NActivePairs,
      0.0);
  double* pair_gradients_data = pair_gradients.data();
  for (std::size_t integral_index = 0;
       integral_index < n_integrals;
       ++integral_index) {
    const double ao_integral_value =
        ao_two_electron_integral_values_data[integral_index];
    const std::size_t left_pair_index =
        ao_two_electron_pair_indices_data[integral_index * 2];
    const std::size_t right_pair_index =
        ao_two_electron_pair_indices_data[integral_index * 2 + 1];
    const double* right_row =
        transformed_pair_coefficients_data +
        pair_row_offsets_data[right_pair_index];
    double* left_gradient_row =
        pair_gradients_data + pair_row_offsets_data[left_pair_index];
    accumulate_scaled_active_pair_row_hvp<NActivePairs>(
        left_gradient_row,
        right_row,
        ao_integral_value);

    if (left_pair_index != right_pair_index) {
      const double* left_row =
          transformed_pair_coefficients_data +
          pair_row_offsets_data[left_pair_index];
      double* right_gradient_row =
          pair_gradients_data + pair_row_offsets_data[right_pair_index];
      accumulate_scaled_active_pair_row_hvp<NActivePairs>(
          right_gradient_row,
          left_row,
          ao_integral_value);
    }
  }
  return pair_gradients;
}

template <int NActivePairs>
void apply_ao_pair_graph_matrix_single_thread_fixed_hvp(
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& row_offsets,
    const std::vector<int>& column_pair_indices,
    const std::vector<int>& integral_indices,
    const double* transformed_pair_coefficients_data,
    const std::size_t* pair_row_offsets_data,
    std::size_t n_ao_pairs,
    std::vector<double>* pair_gradients) {
  resize_and_zero(
      pair_gradients,
      n_ao_pairs * NActivePairs);
  double* pair_gradients_data = pair_gradients->data();
  for (std::size_t row_index = 0; row_index < n_ao_pairs; ++row_index) {
    double* target_row =
        pair_gradients_data + pair_row_offsets_data[row_index];
    const int begin = row_offsets[row_index];
    const int end = row_offsets[row_index + 1];
    for (int entry_offset = begin; entry_offset < end; ++entry_offset) {
      const int column_pair_index =
          column_pair_indices[entry_offset];
      const int integral_index =
          integral_indices[entry_offset];
      const double ao_integral_value =
          ao_two_electron_integral_values[integral_index];
      const double* source_row =
          transformed_pair_coefficients_data +
          pair_row_offsets_data[column_pair_index];
      accumulate_scaled_active_pair_row_hvp<NActivePairs>(
          target_row,
          source_row,
          ao_integral_value);
    }
  }
}

template <int NActivePairs>
void apply_ao_pair_graph_matrix_parallel_fixed_hvp(
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& row_offsets,
    const std::vector<int>& column_pair_indices,
    const std::vector<int>& integral_indices,
    const double* transformed_pair_coefficients_data,
    const std::size_t* pair_row_offsets_data,
    std::size_t n_ao_pairs,
    int n_threads,
    std::vector<double>* pair_gradients) {
  if (pair_gradients == nullptr) {
    throw std::invalid_argument("AO pair-gradient output must not be null");
  }
  resize_and_zero(
      pair_gradients,
      n_ao_pairs * NActivePairs);

  // The materialized exact-2e kernel spends most of its time in this AO-pair
  // graph matvec. For the small active spaces used by the current TN-HVP
  // workloads, `n_active_pairs` is one of a few triangular numbers, so
  // specializing the inner row accumulation removes the tiny dynamic loop from
  // every graph edge without changing the row-parallel schedule.
#pragma omp parallel for schedule(guided, 64) num_threads(n_threads)
  for (std::ptrdiff_t row_offset = 0;
       row_offset < static_cast<std::ptrdiff_t>(n_ao_pairs);
       ++row_offset) {
    const std::size_t row_index = row_offset;
    double* target_row =
        pair_gradients->data() + pair_row_offsets_data[row_index];
    const int begin = row_offsets[row_index];
    const int end = row_offsets[row_index + 1];
    for (int entry_offset = begin; entry_offset < end; ++entry_offset) {
      const int column_pair_index =
          column_pair_indices[entry_offset];
      const int integral_index =
          integral_indices[entry_offset];
      const double ao_integral_value =
          ao_two_electron_integral_values[integral_index];
      const double* source_row =
          transformed_pair_coefficients_data +
          pair_row_offsets_data[column_pair_index];
      accumulate_scaled_active_pair_row_hvp<NActivePairs>(
          target_row,
          source_row,
          ao_integral_value);
    }
  }
}

template <int NActivePairs>
void apply_ao_pair_graph_matrix_row_major_fixed_hvp(
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& row_offsets,
    const std::vector<int>& column_pair_indices,
    const std::vector<int>& integral_indices,
    const double* transformed_pair_coefficients_data,
    std::size_t n_ao_pairs,
    int n_threads,
    double* pair_gradients_data) {
  if (transformed_pair_coefficients_data == nullptr ||
      pair_gradients_data == nullptr) {
    throw std::invalid_argument("AO pair graph row-major HVP buffers must not be null");
  }
#pragma omp parallel for schedule(guided, 64) num_threads(n_threads)
  for (std::ptrdiff_t row_offset = 0;
       row_offset < static_cast<std::ptrdiff_t>(n_ao_pairs);
       ++row_offset) {
    const std::size_t row_index = row_offset;
    double* target_row =
        pair_gradients_data + row_index * NActivePairs;
    const int begin = row_offsets[row_index];
    const int end = row_offsets[row_index + 1];
    for (int entry_offset = begin; entry_offset < end; ++entry_offset) {
      const int column_pair_index =
          column_pair_indices[entry_offset];
      const int integral_index =
          integral_indices[entry_offset];
      const double ao_integral_value =
          ao_two_electron_integral_values[integral_index];
      const double* source_row =
          transformed_pair_coefficients_data +
          static_cast<std::size_t>(column_pair_index) * NActivePairs;
      accumulate_scaled_active_pair_row_hvp<NActivePairs>(
          target_row,
          source_row,
          ao_integral_value);
    }
  }
}

std::vector<ActivePair> build_active_pair_list(int n_active_orbitals) {
  std::vector<ActivePair> active_pairs;
  active_pairs.reserve(
      n_active_orbitals * (n_active_orbitals + 1) / 2);
  for (int first = 0; first < n_active_orbitals; ++first) {
    for (int second = 0; second <= first; ++second) {
      active_pairs.push_back({first, second});
    }
  }
  return active_pairs;
}

ExactCtxPairMatrix build_active_pair_gradient_matrix(
    const std::vector<double>& packed_active_two_electron_gradient,
    const std::vector<ActivePair>& active_pairs) {
  const std::size_t n_active_pairs = active_pairs.size();
  ExactCtxPairMatrix active_pair_gradient_matrix =
      ExactCtxPairMatrix::Zero(
          static_cast<Eigen::Index>(n_active_pairs),
          static_cast<Eigen::Index>(n_active_pairs));

  for (std::size_t row_index = 0; row_index < n_active_pairs; ++row_index) {
    const auto& row_pair = active_pairs[row_index];
    for (std::size_t column_index = 0; column_index < n_active_pairs; ++column_index) {
      const auto& column_pair = active_pairs[column_index];
      const int packed_index =
          (row_index >= column_index)
              ? TwoElectronIndexer::two_electron_storage_index(
                    row_pair.first,
                    row_pair.second,
                    column_pair.first,
                    column_pair.second)
              : TwoElectronIndexer::two_electron_storage_index(
                    column_pair.first,
                    column_pair.second,
                    row_pair.first,
                    row_pair.second);
      double value =
          packed_active_two_electron_gradient[packed_index];
      if (row_index == column_index) {
        value *= 2.0;
      }
      active_pair_gradient_matrix(
          static_cast<Eigen::Index>(row_index),
          static_cast<Eigen::Index>(column_index)) = value;
    }
  }

  return active_pair_gradient_matrix;
}

void build_ao_pair_to_active_pair_coefficients(
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    int n_basis_functions,
    int n_active_orbitals,
    const std::vector<ActivePair>& active_pairs,
    ExactCtxPairMatrix* ao_pair_to_active_pair_coefficients) {
  if (ao_pair_to_active_pair_coefficients == nullptr) {
    throw std::invalid_argument("AO-pair coefficient output must not be null");
  }
  if (dense_active_coefficients.rows() != n_basis_functions ||
      dense_active_coefficients.cols() != n_active_orbitals) {
    throw std::invalid_argument("dense active coefficient matrix shape mismatch");
  }
  const std::size_t n_ao_pairs =
      n_basis_functions * (n_basis_functions + 1) / 2;
  const std::size_t n_active_pairs = active_pairs.size();
  ao_pair_to_active_pair_coefficients->resize(
      static_cast<Eigen::Index>(n_ao_pairs),
      static_cast<Eigen::Index>(n_active_pairs));

#pragma omp parallel for schedule(static)
  for (int first_basis_function = 0;
       first_basis_function < n_basis_functions;
       ++first_basis_function) {
    for (int second_basis_function = 0;
         second_basis_function <= first_basis_function;
         ++second_basis_function) {
      const Eigen::Index ao_pair_offset =
          static_cast<Eigen::Index>(
              ao_pair_index(first_basis_function, second_basis_function));
      for (std::size_t active_pair_index_offset = 0;
           active_pair_index_offset < n_active_pairs;
           ++active_pair_index_offset) {
        const auto& active_pair = active_pairs[active_pair_index_offset];
        double coefficient =
            dense_active_coefficients(first_basis_function, active_pair.first) *
            dense_active_coefficients(second_basis_function, active_pair.second);
        if (first_basis_function != second_basis_function) {
          coefficient +=
              dense_active_coefficients(second_basis_function, active_pair.first) *
              dense_active_coefficients(first_basis_function, active_pair.second);
        }
        (*ao_pair_to_active_pair_coefficients)(
            ao_pair_offset,
            static_cast<Eigen::Index>(active_pair_index_offset)) = coefficient;
      }
    }
  }
}

ExactCtxPairMatrix build_ao_pair_to_active_pair_coefficients(
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    int n_basis_functions,
    int n_active_orbitals,
    const std::vector<ActivePair>& active_pairs) {
  ExactCtxPairMatrix ao_pair_to_active_pair_coefficients;
  build_ao_pair_to_active_pair_coefficients(
      dense_active_coefficients,
      n_basis_functions,
      n_active_orbitals,
      active_pairs,
      &ao_pair_to_active_pair_coefficients);
  return ao_pair_to_active_pair_coefficients;
}

void build_mixed_ao_pair_to_active_pair_coefficients(
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    int n_basis_functions,
    int n_active_orbitals,
    const std::vector<ActivePair>& active_pairs,
    ExactCtxPairMatrix* mixed_ao_pair_to_active_pair_coefficients) {
  if (mixed_ao_pair_to_active_pair_coefficients == nullptr) {
    throw std::invalid_argument("mixed AO-pair coefficient output must not be null");
  }
  if (dense_active_coefficients.rows() != n_basis_functions ||
      dense_active_coefficients.cols() != n_active_orbitals ||
      dense_active_direction.rows() != n_basis_functions ||
      dense_active_direction.cols() != n_active_orbitals) {
    throw std::invalid_argument("mixed AO-pair coefficient matrix shape mismatch");
  }
  const std::size_t n_ao_pairs =
      n_basis_functions * (n_basis_functions + 1) / 2;
  const std::size_t n_active_pairs = active_pairs.size();
  mixed_ao_pair_to_active_pair_coefficients->resize(
      static_cast<Eigen::Index>(n_ao_pairs),
      static_cast<Eigen::Index>(n_active_pairs));

#pragma omp parallel for schedule(static)
  for (int first_basis_function = 0;
       first_basis_function < n_basis_functions;
       ++first_basis_function) {
    for (int second_basis_function = 0;
         second_basis_function <= first_basis_function;
         ++second_basis_function) {
      const Eigen::Index ao_pair_offset =
          static_cast<Eigen::Index>(
              ao_pair_index(first_basis_function, second_basis_function));
      for (std::size_t active_pair_index_offset = 0;
           active_pair_index_offset < n_active_pairs;
           ++active_pair_index_offset) {
        const auto& active_pair = active_pairs[active_pair_index_offset];
        double coefficient =
            dense_active_direction(first_basis_function, active_pair.first) *
                dense_active_coefficients(second_basis_function, active_pair.second) +
            dense_active_coefficients(first_basis_function, active_pair.first) *
                dense_active_direction(second_basis_function, active_pair.second);
        if (first_basis_function != second_basis_function) {
          coefficient +=
              dense_active_direction(second_basis_function, active_pair.first) *
                  dense_active_coefficients(first_basis_function, active_pair.second) +
              dense_active_coefficients(second_basis_function, active_pair.first) *
                  dense_active_direction(first_basis_function, active_pair.second);
        }
        (*mixed_ao_pair_to_active_pair_coefficients)(
            ao_pair_offset,
            static_cast<Eigen::Index>(active_pair_index_offset)) = coefficient;
      }
    }
  }
}

ExactCtxPairMatrix build_mixed_ao_pair_to_active_pair_coefficients(
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    int n_basis_functions,
    int n_active_orbitals,
    const std::vector<ActivePair>& active_pairs) {
  ExactCtxPairMatrix mixed_ao_pair_to_active_pair_coefficients;
  build_mixed_ao_pair_to_active_pair_coefficients(
      dense_active_coefficients,
      dense_active_direction,
      n_basis_functions,
      n_active_orbitals,
      active_pairs,
      &mixed_ao_pair_to_active_pair_coefficients);
  return mixed_ao_pair_to_active_pair_coefficients;
}

void build_mixed_ao_pair_to_active_pair_coefficients_from_cache(
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const ExactPackedActiveTwoElectronAdjointCache& cache,
    ExactCtxPairMatrix* mixed_ao_pair_to_active_pair_coefficients) {
  const int n_basis_functions = cache.n_basis_functions;
  const int n_active_orbitals = cache.n_active_orbitals;
  const std::size_t n_active_pairs = cache.active_pair_first_indices.size();
  if (cache.active_pair_second_indices.size() != n_active_pairs) {
    throw std::invalid_argument("exact 2e cache active-pair index size mismatch");
  }
  if (mixed_ao_pair_to_active_pair_coefficients == nullptr ||
      dense_active_coefficients.rows() != n_basis_functions ||
      dense_active_coefficients.cols() != n_active_orbitals ||
      dense_active_direction.rows() != n_basis_functions ||
      dense_active_direction.cols() != n_active_orbitals) {
    throw std::invalid_argument("dense active coefficient matrix shape mismatch");
  }

  const std::size_t n_ao_pairs =
      n_basis_functions * (n_basis_functions + 1) / 2;
  mixed_ao_pair_to_active_pair_coefficients->resize(
      static_cast<Eigen::Index>(n_ao_pairs),
      static_cast<Eigen::Index>(n_active_pairs));

#pragma omp parallel for schedule(static)
  for (int first_basis_function = 0;
       first_basis_function < n_basis_functions;
       ++first_basis_function) {
    for (int second_basis_function = 0;
         second_basis_function <= first_basis_function;
         ++second_basis_function) {
      const Eigen::Index ao_pair_offset =
          static_cast<Eigen::Index>(
              ao_pair_index(first_basis_function, second_basis_function));
      for (std::size_t active_pair_index = 0;
           active_pair_index < n_active_pairs;
           ++active_pair_index) {
        const int first_active =
            cache.active_pair_first_indices[active_pair_index];
        const int second_active =
            cache.active_pair_second_indices[active_pair_index];
        double coefficient =
            dense_active_direction(first_basis_function, first_active) *
                dense_active_coefficients(second_basis_function, second_active) +
            dense_active_coefficients(first_basis_function, first_active) *
                dense_active_direction(second_basis_function, second_active);
        if (first_basis_function != second_basis_function) {
          coefficient +=
              dense_active_direction(second_basis_function, first_active) *
                  dense_active_coefficients(first_basis_function, second_active) +
              dense_active_coefficients(second_basis_function, first_active) *
                  dense_active_direction(first_basis_function, second_active);
        }
        (*mixed_ao_pair_to_active_pair_coefficients)(
            ao_pair_offset,
            static_cast<Eigen::Index>(active_pair_index)) = coefficient;
      }
    }
  }
}

ExactCtxPairMatrix build_mixed_ao_pair_to_active_pair_coefficients_from_cache(
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const ExactPackedActiveTwoElectronAdjointCache& cache) {
  ExactCtxPairMatrix mixed_ao_pair_to_active_pair_coefficients;
  build_mixed_ao_pair_to_active_pair_coefficients_from_cache(
      dense_active_coefficients,
      dense_active_direction,
      cache,
      &mixed_ao_pair_to_active_pair_coefficients);
  return mixed_ao_pair_to_active_pair_coefficients;
}
void multiply_pair_coefficients_by_gradient_matrix(
    const ExactCtxPairMatrix& ao_pair_to_active_pair_coefficients,
    const ExactCtxPairMatrix& active_pair_gradient_matrix,
    ExactCtxPairMatrix* transformed_pair_coefficients) {
  if (transformed_pair_coefficients == nullptr) {
    throw std::invalid_argument("transformed pair coefficient output must not be null");
  }
  if (ao_pair_to_active_pair_coefficients.cols() != active_pair_gradient_matrix.rows() ||
      active_pair_gradient_matrix.rows() != active_pair_gradient_matrix.cols()) {
    throw std::invalid_argument("active-pair gradient matrix shape mismatch");
  }
  transformed_pair_coefficients->resize(
      ao_pair_to_active_pair_coefficients.rows(),
      active_pair_gradient_matrix.cols());
  transformed_pair_coefficients->noalias() =
      ao_pair_to_active_pair_coefficients * active_pair_gradient_matrix;
}

ExactCtxPairMatrix multiply_pair_coefficients_by_gradient_matrix(
    const ExactCtxPairMatrix& ao_pair_to_active_pair_coefficients,
    const ExactCtxPairMatrix& active_pair_gradient_matrix) {
  ExactCtxPairMatrix transformed_pair_coefficients;
  multiply_pair_coefficients_by_gradient_matrix(
      ao_pair_to_active_pair_coefficients,
      active_pair_gradient_matrix,
      &transformed_pair_coefficients);
  return transformed_pair_coefficients;
}

void apply_sparse_ao_integral_matrix_from_pair_indices(
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& ao_two_electron_pair_indices,
    const std::vector<double>& transformed_pair_coefficients,
    int n_basis_functions,
    std::size_t n_active_pairs,
    std::vector<double>* pair_gradients) {
  if (ao_two_electron_pair_indices.size() != ao_two_electron_integral_values.size() * 2) {
    throw std::invalid_argument("AO two-electron pair index/value sizes are inconsistent");
  }
  if (pair_gradients == nullptr) {
    throw std::invalid_argument("AO pair-gradient output must not be null");
  }

  const std::size_t n_ao_pairs =
      n_basis_functions * (n_basis_functions + 1) / 2;
  const std::vector<std::size_t> pair_row_offsets =
      build_pair_row_offsets_hvp(n_ao_pairs, n_active_pairs);
  const std::size_t* pair_row_offsets_data = pair_row_offsets.data();
  const double* ao_two_electron_integral_values_data =
      ao_two_electron_integral_values.data();
  const int* ao_two_electron_pair_indices_data =
      ao_two_electron_pair_indices.data();
  const double* transformed_pair_coefficients_data =
      transformed_pair_coefficients.data();
  int n_threads = 1;
#ifdef _OPENMP
  n_threads = xmvb::effective_openmp_thread_count();
#endif
  if (n_threads <= 1) {
    switch (n_active_pairs) {
      case 1:
        *pair_gradients = apply_sparse_ao_integral_matrix_single_thread_fixed_hvp<1>(
            ao_two_electron_integral_values_data,
            ao_two_electron_pair_indices_data,
            transformed_pair_coefficients_data,
            pair_row_offsets_data,
            n_ao_pairs,
            ao_two_electron_integral_values.size());
        return;
      case 3:
        *pair_gradients = apply_sparse_ao_integral_matrix_single_thread_fixed_hvp<3>(
            ao_two_electron_integral_values_data,
            ao_two_electron_pair_indices_data,
            transformed_pair_coefficients_data,
            pair_row_offsets_data,
            n_ao_pairs,
            ao_two_electron_integral_values.size());
        return;
      case 6:
        *pair_gradients = apply_sparse_ao_integral_matrix_single_thread_fixed_hvp<6>(
            ao_two_electron_integral_values_data,
            ao_two_electron_pair_indices_data,
            transformed_pair_coefficients_data,
            pair_row_offsets_data,
            n_ao_pairs,
            ao_two_electron_integral_values.size());
        return;
      case 10:
        *pair_gradients = apply_sparse_ao_integral_matrix_single_thread_fixed_hvp<10>(
            ao_two_electron_integral_values_data,
            ao_two_electron_pair_indices_data,
            transformed_pair_coefficients_data,
            pair_row_offsets_data,
            n_ao_pairs,
            ao_two_electron_integral_values.size());
        return;
      case 15:
        *pair_gradients = apply_sparse_ao_integral_matrix_single_thread_fixed_hvp<15>(
            ao_two_electron_integral_values_data,
            ao_two_electron_pair_indices_data,
            transformed_pair_coefficients_data,
            pair_row_offsets_data,
            n_ao_pairs,
            ao_two_electron_integral_values.size());
        return;
      case 21:
        *pair_gradients = apply_sparse_ao_integral_matrix_single_thread_fixed_hvp<21>(
            ao_two_electron_integral_values_data,
            ao_two_electron_pair_indices_data,
            transformed_pair_coefficients_data,
            pair_row_offsets_data,
            n_ao_pairs,
            ao_two_electron_integral_values.size());
        return;
      case 28:
        *pair_gradients = apply_sparse_ao_integral_matrix_single_thread_fixed_hvp<28>(
            ao_two_electron_integral_values_data,
            ao_two_electron_pair_indices_data,
            transformed_pair_coefficients_data,
            pair_row_offsets_data,
            n_ao_pairs,
            ao_two_electron_integral_values.size());
        return;
      case 36:
        *pair_gradients = apply_sparse_ao_integral_matrix_single_thread_fixed_hvp<36>(
            ao_two_electron_integral_values_data,
            ao_two_electron_pair_indices_data,
            transformed_pair_coefficients_data,
            pair_row_offsets_data,
            n_ao_pairs,
            ao_two_electron_integral_values.size());
        return;
      case 45:
        *pair_gradients = apply_sparse_ao_integral_matrix_single_thread_fixed_hvp<45>(
            ao_two_electron_integral_values_data,
            ao_two_electron_pair_indices_data,
            transformed_pair_coefficients_data,
            pair_row_offsets_data,
            n_ao_pairs,
            ao_two_electron_integral_values.size());
        return;
      case 55:
        *pair_gradients = apply_sparse_ao_integral_matrix_single_thread_fixed_hvp<55>(
            ao_two_electron_integral_values_data,
            ao_two_electron_pair_indices_data,
            transformed_pair_coefficients_data,
            pair_row_offsets_data,
            n_ao_pairs,
            ao_two_electron_integral_values.size());
        return;
      default:
        break;
    }

    resize_and_zero(pair_gradients, n_ao_pairs * n_active_pairs);
    double* pair_gradients_data = pair_gradients->data();
    for (std::size_t integral_index = 0;
         integral_index < ao_two_electron_integral_values.size();
         ++integral_index) {
      const double ao_integral_value =
          ao_two_electron_integral_values_data[integral_index];
      const std::size_t left_pair_index =
          ao_two_electron_pair_indices_data[integral_index * 2];
      const std::size_t right_pair_index =
          ao_two_electron_pair_indices_data[integral_index * 2 + 1];
      const double* right_row =
          transformed_pair_coefficients_data +
          pair_row_offsets_data[right_pair_index];
      double* left_gradient_row =
          pair_gradients_data + pair_row_offsets_data[left_pair_index];
#pragma omp simd
      for (std::size_t active_pair_index = 0;
           active_pair_index < n_active_pairs;
           ++active_pair_index) {
        left_gradient_row[active_pair_index] +=
            ao_integral_value * right_row[active_pair_index];
      }

      if (left_pair_index != right_pair_index) {
        const double* left_row =
            transformed_pair_coefficients_data +
            pair_row_offsets_data[left_pair_index];
        double* right_gradient_row =
            pair_gradients_data + pair_row_offsets_data[right_pair_index];
#pragma omp simd
        for (std::size_t active_pair_index = 0;
             active_pair_index < n_active_pairs;
             ++active_pair_index) {
          right_gradient_row[active_pair_index] +=
              ao_integral_value * left_row[active_pair_index];
        }
      }
    }
    return;
  }

  std::vector<std::vector<double>> partial_pair_gradients;
  std::atomic<int> invalid_integral_index(-1);
  partial_pair_gradients.assign(
      n_threads,
      std::vector<double>(n_ao_pairs * n_active_pairs, 0.0));

#pragma omp parallel num_threads(n_threads)
  {
    int thread_index = 0;
#ifdef _OPENMP
    thread_index = omp_get_thread_num();
#endif
    auto& local_pair_gradients =
        partial_pair_gradients[thread_index];

#pragma omp for schedule(guided, 256)
    for (std::ptrdiff_t integral_offset = 0;
         integral_offset <
             static_cast<std::ptrdiff_t>(ao_two_electron_integral_values.size());
         ++integral_offset) {
      const std::size_t integral_index = integral_offset;
      const double ao_integral_value = ao_two_electron_integral_values[integral_index];
      const int left_pair_index = ao_two_electron_pair_indices[integral_index * 2];
      const int right_pair_index = ao_two_electron_pair_indices[integral_index * 2 + 1];
      if (left_pair_index < 0 || right_pair_index < 0 ||
          left_pair_index >= static_cast<int>(n_ao_pairs) ||
          right_pair_index >= static_cast<int>(n_ao_pairs)) {
        int expected = -1;
        invalid_integral_index.compare_exchange_strong(
            expected,
            static_cast<int>(integral_index));
        continue;
      }

      const double* right_row =
          transformed_pair_coefficients_data +
          pair_row_offsets_data[right_pair_index];
      double* left_gradient_row =
          local_pair_gradients.data() +
          pair_row_offsets_data[left_pair_index];
      for (std::size_t active_pair_index = 0;
           active_pair_index < n_active_pairs;
           ++active_pair_index) {
        left_gradient_row[active_pair_index] +=
            ao_integral_value * right_row[active_pair_index];
      }

      if (left_pair_index != right_pair_index) {
        const double* left_row =
            transformed_pair_coefficients_data +
            pair_row_offsets_data[left_pair_index];
        double* right_gradient_row =
            local_pair_gradients.data() +
            pair_row_offsets_data[right_pair_index];
        for (std::size_t active_pair_index = 0;
             active_pair_index < n_active_pairs;
             ++active_pair_index) {
          right_gradient_row[active_pair_index] +=
              ao_integral_value * left_row[active_pair_index];
        }
      }
    }
  }

  resize_and_zero(pair_gradients, n_ao_pairs * n_active_pairs);
  for (const auto& partial_pair_gradient : partial_pair_gradients) {
    for (std::size_t index = 0; index < pair_gradients->size(); ++index) {
      (*pair_gradients)[index] += partial_pair_gradient[index];
    }
  }
  if (invalid_integral_index.load() >= 0) {
    throw std::invalid_argument("packed AO pair index out of range");
  }
}

std::vector<double> apply_sparse_ao_integral_matrix_from_pair_indices(
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& ao_two_electron_pair_indices,
    const std::vector<double>& transformed_pair_coefficients,
    int n_basis_functions,
    std::size_t n_active_pairs) {
  std::vector<double> pair_gradients;
  apply_sparse_ao_integral_matrix_from_pair_indices(
      ao_two_electron_integral_values,
      ao_two_electron_pair_indices,
      transformed_pair_coefficients,
      n_basis_functions,
      n_active_pairs,
      &pair_gradients);
  return pair_gradients;
}

void apply_sparse_ao_integral_matrix_from_four_indices(
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& ao_two_electron_integral_indices,
    const std::vector<double>& transformed_pair_coefficients,
    int n_basis_functions,
    std::size_t n_active_pairs,
    std::vector<double>* pair_gradients) {
  if (ao_two_electron_integral_indices.size() != ao_two_electron_integral_values.size() * 4) {
    throw std::invalid_argument("AO two-electron index/value sizes are inconsistent");
  }
  if (pair_gradients == nullptr) {
    throw std::invalid_argument("AO pair-gradient output must not be null");
  }

  const std::size_t n_ao_pairs =
      n_basis_functions * (n_basis_functions + 1) / 2;
  std::vector<std::vector<double>> partial_pair_gradients;
  std::atomic<int> invalid_integral_index(-1);
  int n_threads = 1;
#ifdef _OPENMP
  n_threads = xmvb::effective_openmp_thread_count();
#endif
  partial_pair_gradients.assign(
      n_threads,
      std::vector<double>(n_ao_pairs * n_active_pairs, 0.0));

#pragma omp parallel num_threads(n_threads)
  {
    int thread_index = 0;
#ifdef _OPENMP
    thread_index = omp_get_thread_num();
#endif
    auto& local_pair_gradients =
        partial_pair_gradients[thread_index];

#pragma omp for schedule(guided, 256)
    for (std::ptrdiff_t integral_offset = 0;
         integral_offset <
             static_cast<std::ptrdiff_t>(ao_two_electron_integral_values.size());
         ++integral_offset) {
      const std::size_t integral_index = integral_offset;
      const double ao_integral_value = ao_two_electron_integral_values[integral_index];
      const int i = ao_two_electron_integral_indices[integral_index * 4];
      const int j = ao_two_electron_integral_indices[integral_index * 4 + 1];
      const int k = ao_two_electron_integral_indices[integral_index * 4 + 2];
      const int l = ao_two_electron_integral_indices[integral_index * 4 + 3];

      if (i < 0 || i >= n_basis_functions ||
          j < 0 || j >= n_basis_functions ||
          k < 0 || k >= n_basis_functions ||
          l < 0 || l >= n_basis_functions) {
        int expected = -1;
        invalid_integral_index.compare_exchange_strong(
            expected,
            static_cast<int>(integral_index));
        continue;
      }

      const std::size_t left_pair_index = ao_pair_index(i, j);
      const std::size_t right_pair_index = ao_pair_index(k, l);
      const double* right_row =
          transformed_pair_coefficients.data() +
          right_pair_index * n_active_pairs;
      double* left_gradient_row =
          local_pair_gradients.data() +
          left_pair_index * n_active_pairs;
      for (std::size_t active_pair_index = 0;
           active_pair_index < n_active_pairs;
           ++active_pair_index) {
        left_gradient_row[active_pair_index] +=
            ao_integral_value * right_row[active_pair_index];
      }

      if (left_pair_index != right_pair_index) {
        const double* left_row =
            transformed_pair_coefficients.data() +
            left_pair_index * n_active_pairs;
        double* right_gradient_row =
            local_pair_gradients.data() +
            right_pair_index * n_active_pairs;
        for (std::size_t active_pair_index = 0;
             active_pair_index < n_active_pairs;
             ++active_pair_index) {
          right_gradient_row[active_pair_index] +=
              ao_integral_value * left_row[active_pair_index];
        }
      }
    }
  }

  if (invalid_integral_index.load() >= 0) {
    throw std::invalid_argument("AO two-electron index out of range");
  }

  resize_and_zero(pair_gradients, n_ao_pairs * n_active_pairs);
  for (const auto& partial_pair_gradient : partial_pair_gradients) {
    for (std::size_t index = 0; index < pair_gradients->size(); ++index) {
      (*pair_gradients)[index] += partial_pair_gradient[index];
    }
  }
}

std::vector<double> apply_sparse_ao_integral_matrix_from_four_indices(
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& ao_two_electron_integral_indices,
    const std::vector<double>& transformed_pair_coefficients,
    int n_basis_functions,
    std::size_t n_active_pairs) {
  std::vector<double> pair_gradients;
  apply_sparse_ao_integral_matrix_from_four_indices(
      ao_two_electron_integral_values,
      ao_two_electron_integral_indices,
      transformed_pair_coefficients,
      n_basis_functions,
      n_active_pairs,
      &pair_gradients);
  return pair_gradients;
}

void apply_ao_pair_graph_matrix(
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& row_offsets,
    const std::vector<int>& column_pair_indices,
    const std::vector<int>& integral_indices,
    const std::vector<double>& transformed_pair_coefficients,
    std::size_t n_ao_pairs,
    std::size_t n_active_pairs,
    std::vector<double>* pair_gradients) {
  if (row_offsets.size() != n_ao_pairs + 1) {
    throw std::invalid_argument("AO pair graph row offset size mismatch");
  }
  if (column_pair_indices.size() != integral_indices.size()) {
    throw std::invalid_argument("AO pair graph column/integral size mismatch");
  }
  if (row_offsets.back() != static_cast<int>(column_pair_indices.size())) {
    throw std::invalid_argument("AO pair graph row offsets do not cover all entries");
  }

  const std::vector<std::size_t> pair_row_offsets =
      build_pair_row_offsets_hvp(n_ao_pairs, n_active_pairs);
  const std::size_t* pair_row_offsets_data = pair_row_offsets.data();
  int n_threads = 1;
#ifdef _OPENMP
  n_threads = xmvb::effective_openmp_thread_count();
#endif
  if (n_threads <= 1) {
    switch (n_active_pairs) {
      case 1:
        apply_ao_pair_graph_matrix_single_thread_fixed_hvp<1>(
            ao_two_electron_integral_values,
            row_offsets,
            column_pair_indices,
            integral_indices,
            transformed_pair_coefficients.data(),
            pair_row_offsets_data,
            n_ao_pairs,
            pair_gradients);
        return;
      case 3:
        apply_ao_pair_graph_matrix_single_thread_fixed_hvp<3>(
            ao_two_electron_integral_values,
            row_offsets,
            column_pair_indices,
            integral_indices,
            transformed_pair_coefficients.data(),
            pair_row_offsets_data,
            n_ao_pairs,
            pair_gradients);
        return;
      case 6:
        apply_ao_pair_graph_matrix_single_thread_fixed_hvp<6>(
            ao_two_electron_integral_values,
            row_offsets,
            column_pair_indices,
            integral_indices,
            transformed_pair_coefficients.data(),
            pair_row_offsets_data,
            n_ao_pairs,
            pair_gradients);
        return;
      case 10:
        apply_ao_pair_graph_matrix_single_thread_fixed_hvp<10>(
            ao_two_electron_integral_values,
            row_offsets,
            column_pair_indices,
            integral_indices,
            transformed_pair_coefficients.data(),
            pair_row_offsets_data,
            n_ao_pairs,
            pair_gradients);
        return;
      case 15:
        apply_ao_pair_graph_matrix_single_thread_fixed_hvp<15>(
            ao_two_electron_integral_values,
            row_offsets,
            column_pair_indices,
            integral_indices,
            transformed_pair_coefficients.data(),
            pair_row_offsets_data,
            n_ao_pairs,
            pair_gradients);
        return;
      case 21:
        apply_ao_pair_graph_matrix_single_thread_fixed_hvp<21>(
            ao_two_electron_integral_values,
            row_offsets,
            column_pair_indices,
            integral_indices,
            transformed_pair_coefficients.data(),
            pair_row_offsets_data,
            n_ao_pairs,
            pair_gradients);
        return;
      case 28:
        apply_ao_pair_graph_matrix_single_thread_fixed_hvp<28>(
            ao_two_electron_integral_values,
            row_offsets,
            column_pair_indices,
            integral_indices,
            transformed_pair_coefficients.data(),
            pair_row_offsets_data,
            n_ao_pairs,
            pair_gradients);
        return;
      case 36:
        apply_ao_pair_graph_matrix_single_thread_fixed_hvp<36>(
            ao_two_electron_integral_values,
            row_offsets,
            column_pair_indices,
            integral_indices,
            transformed_pair_coefficients.data(),
            pair_row_offsets_data,
            n_ao_pairs,
            pair_gradients);
        return;
      case 45:
        apply_ao_pair_graph_matrix_single_thread_fixed_hvp<45>(
            ao_two_electron_integral_values,
            row_offsets,
            column_pair_indices,
            integral_indices,
            transformed_pair_coefficients.data(),
            pair_row_offsets_data,
            n_ao_pairs,
            pair_gradients);
        return;
      case 55:
        apply_ao_pair_graph_matrix_single_thread_fixed_hvp<55>(
            ao_two_electron_integral_values,
            row_offsets,
            column_pair_indices,
            integral_indices,
            transformed_pair_coefficients.data(),
            pair_row_offsets_data,
            n_ao_pairs,
            pair_gradients);
        return;
      default:
        break;
    }
  }

  switch (n_active_pairs) {
    case 1:
      apply_ao_pair_graph_matrix_parallel_fixed_hvp<1>(
          ao_two_electron_integral_values,
          row_offsets,
          column_pair_indices,
          integral_indices,
          transformed_pair_coefficients.data(),
          pair_row_offsets_data,
          n_ao_pairs,
          n_threads,
          pair_gradients);
      return;
    case 3:
      apply_ao_pair_graph_matrix_parallel_fixed_hvp<3>(
          ao_two_electron_integral_values,
          row_offsets,
          column_pair_indices,
          integral_indices,
          transformed_pair_coefficients.data(),
          pair_row_offsets_data,
          n_ao_pairs,
          n_threads,
          pair_gradients);
      return;
    case 6:
      apply_ao_pair_graph_matrix_parallel_fixed_hvp<6>(
          ao_two_electron_integral_values,
          row_offsets,
          column_pair_indices,
          integral_indices,
          transformed_pair_coefficients.data(),
          pair_row_offsets_data,
          n_ao_pairs,
          n_threads,
          pair_gradients);
      return;
    case 10:
      apply_ao_pair_graph_matrix_parallel_fixed_hvp<10>(
          ao_two_electron_integral_values,
          row_offsets,
          column_pair_indices,
          integral_indices,
          transformed_pair_coefficients.data(),
          pair_row_offsets_data,
          n_ao_pairs,
          n_threads,
          pair_gradients);
      return;
    case 15:
      apply_ao_pair_graph_matrix_parallel_fixed_hvp<15>(
          ao_two_electron_integral_values,
          row_offsets,
          column_pair_indices,
          integral_indices,
          transformed_pair_coefficients.data(),
          pair_row_offsets_data,
          n_ao_pairs,
          n_threads,
          pair_gradients);
      return;
    case 21:
      apply_ao_pair_graph_matrix_parallel_fixed_hvp<21>(
          ao_two_electron_integral_values,
          row_offsets,
          column_pair_indices,
          integral_indices,
          transformed_pair_coefficients.data(),
          pair_row_offsets_data,
          n_ao_pairs,
          n_threads,
          pair_gradients);
      return;
    case 28:
      apply_ao_pair_graph_matrix_parallel_fixed_hvp<28>(
          ao_two_electron_integral_values,
          row_offsets,
          column_pair_indices,
          integral_indices,
          transformed_pair_coefficients.data(),
          pair_row_offsets_data,
          n_ao_pairs,
          n_threads,
          pair_gradients);
      return;
    case 36:
      apply_ao_pair_graph_matrix_parallel_fixed_hvp<36>(
          ao_two_electron_integral_values,
          row_offsets,
          column_pair_indices,
          integral_indices,
          transformed_pair_coefficients.data(),
          pair_row_offsets_data,
          n_ao_pairs,
          n_threads,
          pair_gradients);
      return;
    case 45:
      apply_ao_pair_graph_matrix_parallel_fixed_hvp<45>(
          ao_two_electron_integral_values,
          row_offsets,
          column_pair_indices,
          integral_indices,
          transformed_pair_coefficients.data(),
          pair_row_offsets_data,
          n_ao_pairs,
          n_threads,
          pair_gradients);
      return;
    case 55:
      apply_ao_pair_graph_matrix_parallel_fixed_hvp<55>(
          ao_two_electron_integral_values,
          row_offsets,
          column_pair_indices,
          integral_indices,
          transformed_pair_coefficients.data(),
          pair_row_offsets_data,
          n_ao_pairs,
          n_threads,
          pair_gradients);
      return;
    default:
      break;
  }

  resize_and_zero(
      pair_gradients,
      n_ao_pairs * n_active_pairs);
#pragma omp parallel for schedule(guided, 64) num_threads(n_threads)
  for (std::ptrdiff_t row_offset = 0;
       row_offset < static_cast<std::ptrdiff_t>(n_ao_pairs);
       ++row_offset) {
    const std::size_t row_index = row_offset;
    double* target_row =
        pair_gradients->data() + pair_row_offsets_data[row_index];
    const int begin = row_offsets[row_index];
    const int end = row_offsets[row_index + 1];
    for (int entry_offset = begin; entry_offset < end; ++entry_offset) {
      const int column_pair_index =
          column_pair_indices[entry_offset];
      const int integral_index =
          integral_indices[entry_offset];
      const double ao_integral_value =
          ao_two_electron_integral_values[integral_index];
      const double* source_row =
          transformed_pair_coefficients.data() +
          pair_row_offsets_data[column_pair_index];
      for (std::size_t active_pair_index = 0;
           active_pair_index < n_active_pairs;
           ++active_pair_index) {
        target_row[active_pair_index] +=
            ao_integral_value * source_row[active_pair_index];
      }
    }
  }
}

std::vector<double> apply_ao_pair_graph_matrix(
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& row_offsets,
    const std::vector<int>& column_pair_indices,
    const std::vector<int>& integral_indices,
    const std::vector<double>& transformed_pair_coefficients,
    std::size_t n_ao_pairs,
    std::size_t n_active_pairs) {
  std::vector<double> pair_gradients;
  apply_ao_pair_graph_matrix(
      ao_two_electron_integral_values,
      row_offsets,
      column_pair_indices,
      integral_indices,
      transformed_pair_coefficients,
      n_ao_pairs,
      n_active_pairs,
      &pair_gradients);
  return pair_gradients;
}

void apply_exact_ao_pair_kernel(
    const AoIntegralInput& ao_integral_input,
    const std::vector<double>& transformed_pair_coefficients,
    int n_basis_functions,
    std::size_t n_active_pairs,
    std::vector<double>* pair_gradients) {
  if (pair_gradients == nullptr) {
    throw std::invalid_argument("exact AO-pair kernel output must not be null");
  }
  const std::size_t n_ao_pairs =
      n_basis_functions * (n_basis_functions + 1) / 2;
  int n_threads = 1;
#ifdef _OPENMP
  n_threads = xmvb::effective_openmp_thread_count();
#endif
  if (!ao_integral_input.ao_two_electron_pair_graph_row_offsets.empty()) {
    // The row-oriented AO-pair graph is the block-contraction form of the
    // exact kernel.  It is also the better fit for the single-thread HVP path
    // because it keeps each output AO-pair row hot while streaming its
    // contributing columns, instead of repeatedly scattering into two random
    // rows per integral.
    return apply_ao_pair_graph_matrix(
        ao_integral_input.ao_two_electron_integral_values,
        ao_integral_input.ao_two_electron_pair_graph_row_offsets,
        ao_integral_input.ao_two_electron_pair_graph_column_indices,
        ao_integral_input.ao_two_electron_pair_graph_integral_indices,
        transformed_pair_coefficients,
        n_ao_pairs,
        n_active_pairs,
        pair_gradients);
  }
  if (!ao_integral_input.ao_two_electron_pair_indices.empty()) {
    if (n_threads <= 1) {
      const auto pair_row_offsets =
          build_pair_row_offsets_hvp(n_ao_pairs, n_active_pairs);
      const double* ao_two_electron_integral_values_data =
          ao_integral_input.ao_two_electron_integral_values.data();
      const int* ao_two_electron_pair_indices_data =
          ao_integral_input.ao_two_electron_pair_indices.data();
      const double* transformed_pair_coefficients_data =
          transformed_pair_coefficients.data();
      const std::size_t* pair_row_offsets_data = pair_row_offsets.data();
      switch (n_active_pairs) {
        case 1:
          *pair_gradients = apply_sparse_ao_integral_matrix_single_thread_fixed_hvp<1>(
              ao_two_electron_integral_values_data,
              ao_two_electron_pair_indices_data,
              transformed_pair_coefficients_data,
              pair_row_offsets_data,
              n_ao_pairs,
              ao_integral_input.ao_two_electron_integral_values.size());
          return;
        case 3:
          *pair_gradients = apply_sparse_ao_integral_matrix_single_thread_fixed_hvp<3>(
              ao_two_electron_integral_values_data,
              ao_two_electron_pair_indices_data,
              transformed_pair_coefficients_data,
              pair_row_offsets_data,
              n_ao_pairs,
              ao_integral_input.ao_two_electron_integral_values.size());
          return;
        case 6:
          *pair_gradients = apply_sparse_ao_integral_matrix_single_thread_fixed_hvp<6>(
              ao_two_electron_integral_values_data,
              ao_two_electron_pair_indices_data,
              transformed_pair_coefficients_data,
              pair_row_offsets_data,
              n_ao_pairs,
              ao_integral_input.ao_two_electron_integral_values.size());
          return;
        case 10:
          *pair_gradients = apply_sparse_ao_integral_matrix_single_thread_fixed_hvp<10>(
              ao_two_electron_integral_values_data,
              ao_two_electron_pair_indices_data,
              transformed_pair_coefficients_data,
              pair_row_offsets_data,
              n_ao_pairs,
              ao_integral_input.ao_two_electron_integral_values.size());
          return;
        case 15:
          *pair_gradients = apply_sparse_ao_integral_matrix_single_thread_fixed_hvp<15>(
              ao_two_electron_integral_values_data,
              ao_two_electron_pair_indices_data,
              transformed_pair_coefficients_data,
              pair_row_offsets_data,
              n_ao_pairs,
              ao_integral_input.ao_two_electron_integral_values.size());
          return;
        case 21:
          *pair_gradients = apply_sparse_ao_integral_matrix_single_thread_fixed_hvp<21>(
              ao_two_electron_integral_values_data,
              ao_two_electron_pair_indices_data,
              transformed_pair_coefficients_data,
              pair_row_offsets_data,
              n_ao_pairs,
              ao_integral_input.ao_two_electron_integral_values.size());
          return;
        case 28:
          *pair_gradients = apply_sparse_ao_integral_matrix_single_thread_fixed_hvp<28>(
              ao_two_electron_integral_values_data,
              ao_two_electron_pair_indices_data,
              transformed_pair_coefficients_data,
              pair_row_offsets_data,
              n_ao_pairs,
              ao_integral_input.ao_two_electron_integral_values.size());
          return;
        case 36:
          *pair_gradients = apply_sparse_ao_integral_matrix_single_thread_fixed_hvp<36>(
              ao_two_electron_integral_values_data,
              ao_two_electron_pair_indices_data,
              transformed_pair_coefficients_data,
              pair_row_offsets_data,
              n_ao_pairs,
              ao_integral_input.ao_two_electron_integral_values.size());
          return;
        case 45:
          *pair_gradients = apply_sparse_ao_integral_matrix_single_thread_fixed_hvp<45>(
              ao_two_electron_integral_values_data,
              ao_two_electron_pair_indices_data,
              transformed_pair_coefficients_data,
              pair_row_offsets_data,
              n_ao_pairs,
              ao_integral_input.ao_two_electron_integral_values.size());
          return;
        case 55:
          *pair_gradients = apply_sparse_ao_integral_matrix_single_thread_fixed_hvp<55>(
              ao_two_electron_integral_values_data,
              ao_two_electron_pair_indices_data,
              transformed_pair_coefficients_data,
              pair_row_offsets_data,
              n_ao_pairs,
              ao_integral_input.ao_two_electron_integral_values.size());
          return;
        default:
          break;
      }
    }
    apply_sparse_ao_integral_matrix_from_pair_indices(
        ao_integral_input.ao_two_electron_integral_values,
        ao_integral_input.ao_two_electron_pair_indices,
        transformed_pair_coefficients,
        n_basis_functions,
        n_active_pairs,
        pair_gradients);
    return;
  }
  apply_sparse_ao_integral_matrix_from_four_indices(
      ao_integral_input.ao_two_electron_integral_values,
      ao_integral_input.ao_two_electron_integral_indices,
      transformed_pair_coefficients,
      n_basis_functions,
      n_active_pairs,
      pair_gradients);
}

std::vector<double> apply_exact_ao_pair_kernel(
    const AoIntegralInput& ao_integral_input,
    const std::vector<double>& transformed_pair_coefficients,
    int n_basis_functions,
    std::size_t n_active_pairs) {
  std::vector<double> pair_gradients;
  apply_exact_ao_pair_kernel(
      ao_integral_input,
      transformed_pair_coefficients,
      n_basis_functions,
      n_active_pairs,
      &pair_gradients);
  return pair_gradients;
}

void apply_exact_ao_pair_kernel(
    const AoIntegralInput& ao_integral_input,
    const ExactCtxPairMatrix& transformed_pair_coefficients,
    int n_basis_functions,
    std::size_t n_active_pairs,
    ExactCtxPairMatrix* pair_gradients) {
  if (pair_gradients == nullptr) {
    throw std::invalid_argument("exact AO-pair kernel output must not be null");
  }
  const std::size_t n_ao_pairs =
      n_basis_functions * (n_basis_functions + 1) / 2;
  if (transformed_pair_coefficients.rows() != static_cast<Eigen::Index>(n_ao_pairs) ||
      transformed_pair_coefficients.cols() != static_cast<Eigen::Index>(n_active_pairs)) {
    throw std::invalid_argument("transformed pair coefficient matrix shape mismatch");
  }

  pair_gradients->resize(
      static_cast<Eigen::Index>(n_ao_pairs),
      static_cast<Eigen::Index>(n_active_pairs));
  pair_gradients->setZero();

  if (!ao_integral_input.ao_two_electron_pair_graph_row_offsets.empty()) {
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = xmvb::effective_openmp_thread_count();
#endif
    const double* transformed_pair_coefficients_data =
        transformed_pair_coefficients.data();
    double* pair_gradients_data = pair_gradients->data();
    switch (n_active_pairs) {
      case 1:
        apply_ao_pair_graph_matrix_row_major_fixed_hvp<1>(
            ao_integral_input.ao_two_electron_integral_values,
            ao_integral_input.ao_two_electron_pair_graph_row_offsets,
            ao_integral_input.ao_two_electron_pair_graph_column_indices,
            ao_integral_input.ao_two_electron_pair_graph_integral_indices,
            transformed_pair_coefficients_data,
            n_ao_pairs,
            n_threads,
            pair_gradients_data);
        return;
      case 3:
        apply_ao_pair_graph_matrix_row_major_fixed_hvp<3>(
            ao_integral_input.ao_two_electron_integral_values,
            ao_integral_input.ao_two_electron_pair_graph_row_offsets,
            ao_integral_input.ao_two_electron_pair_graph_column_indices,
            ao_integral_input.ao_two_electron_pair_graph_integral_indices,
            transformed_pair_coefficients_data,
            n_ao_pairs,
            n_threads,
            pair_gradients_data);
        return;
      case 6:
        apply_ao_pair_graph_matrix_row_major_fixed_hvp<6>(
            ao_integral_input.ao_two_electron_integral_values,
            ao_integral_input.ao_two_electron_pair_graph_row_offsets,
            ao_integral_input.ao_two_electron_pair_graph_column_indices,
            ao_integral_input.ao_two_electron_pair_graph_integral_indices,
            transformed_pair_coefficients_data,
            n_ao_pairs,
            n_threads,
            pair_gradients_data);
        return;
      case 10:
        apply_ao_pair_graph_matrix_row_major_fixed_hvp<10>(
            ao_integral_input.ao_two_electron_integral_values,
            ao_integral_input.ao_two_electron_pair_graph_row_offsets,
            ao_integral_input.ao_two_electron_pair_graph_column_indices,
            ao_integral_input.ao_two_electron_pair_graph_integral_indices,
            transformed_pair_coefficients_data,
            n_ao_pairs,
            n_threads,
            pair_gradients_data);
        return;
      case 15:
        apply_ao_pair_graph_matrix_row_major_fixed_hvp<15>(
            ao_integral_input.ao_two_electron_integral_values,
            ao_integral_input.ao_two_electron_pair_graph_row_offsets,
            ao_integral_input.ao_two_electron_pair_graph_column_indices,
            ao_integral_input.ao_two_electron_pair_graph_integral_indices,
            transformed_pair_coefficients_data,
            n_ao_pairs,
            n_threads,
            pair_gradients_data);
        return;
      case 21:
        apply_ao_pair_graph_matrix_row_major_fixed_hvp<21>(
            ao_integral_input.ao_two_electron_integral_values,
            ao_integral_input.ao_two_electron_pair_graph_row_offsets,
            ao_integral_input.ao_two_electron_pair_graph_column_indices,
            ao_integral_input.ao_two_electron_pair_graph_integral_indices,
            transformed_pair_coefficients_data,
            n_ao_pairs,
            n_threads,
            pair_gradients_data);
        return;
      case 28:
        apply_ao_pair_graph_matrix_row_major_fixed_hvp<28>(
            ao_integral_input.ao_two_electron_integral_values,
            ao_integral_input.ao_two_electron_pair_graph_row_offsets,
            ao_integral_input.ao_two_electron_pair_graph_column_indices,
            ao_integral_input.ao_two_electron_pair_graph_integral_indices,
            transformed_pair_coefficients_data,
            n_ao_pairs,
            n_threads,
            pair_gradients_data);
        return;
      case 36:
        apply_ao_pair_graph_matrix_row_major_fixed_hvp<36>(
            ao_integral_input.ao_two_electron_integral_values,
            ao_integral_input.ao_two_electron_pair_graph_row_offsets,
            ao_integral_input.ao_two_electron_pair_graph_column_indices,
            ao_integral_input.ao_two_electron_pair_graph_integral_indices,
            transformed_pair_coefficients_data,
            n_ao_pairs,
            n_threads,
            pair_gradients_data);
        return;
      case 45:
        apply_ao_pair_graph_matrix_row_major_fixed_hvp<45>(
            ao_integral_input.ao_two_electron_integral_values,
            ao_integral_input.ao_two_electron_pair_graph_row_offsets,
            ao_integral_input.ao_two_electron_pair_graph_column_indices,
            ao_integral_input.ao_two_electron_pair_graph_integral_indices,
            transformed_pair_coefficients_data,
            n_ao_pairs,
            n_threads,
            pair_gradients_data);
        return;
      case 55:
        apply_ao_pair_graph_matrix_row_major_fixed_hvp<55>(
            ao_integral_input.ao_two_electron_integral_values,
            ao_integral_input.ao_two_electron_pair_graph_row_offsets,
            ao_integral_input.ao_two_electron_pair_graph_column_indices,
            ao_integral_input.ao_two_electron_pair_graph_integral_indices,
            transformed_pair_coefficients_data,
            n_ao_pairs,
            n_threads,
            pair_gradients_data);
        return;
      default:
        break;
    }
#pragma omp parallel for schedule(guided, 64) num_threads(n_threads)
    for (std::ptrdiff_t row_offset = 0;
         row_offset < static_cast<std::ptrdiff_t>(n_ao_pairs);
         ++row_offset) {
      const std::size_t row_index = row_offset;
      double* target_row =
          pair_gradients_data + row_index * n_active_pairs;
      for (int entry_offset = ao_integral_input.ao_two_electron_pair_graph_row_offsets[row_index];
           entry_offset <
               ao_integral_input.ao_two_electron_pair_graph_row_offsets[row_index + 1];
           ++entry_offset) {
        const int column_pair_index =
            ao_integral_input.ao_two_electron_pair_graph_column_indices[entry_offset];
        const int integral_index =
            ao_integral_input.ao_two_electron_pair_graph_integral_indices[entry_offset];
        const double ao_integral_value =
            ao_integral_input.ao_two_electron_integral_values[integral_index];
        const double* source_row =
            transformed_pair_coefficients_data +
            static_cast<std::size_t>(column_pair_index) * n_active_pairs;
#pragma omp simd
        for (std::size_t active_pair_index = 0;
             active_pair_index < n_active_pairs;
             ++active_pair_index) {
          target_row[active_pair_index] +=
              ao_integral_value * source_row[active_pair_index];
        }
      }
    }
    return;
  }

  if (!ao_integral_input.ao_two_electron_pair_indices.empty()) {
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = xmvb::effective_openmp_thread_count();
#endif
    if (n_threads <= 1) {
      for (std::size_t integral_index = 0;
           integral_index < ao_integral_input.ao_two_electron_integral_values.size();
           ++integral_index) {
        const double ao_integral_value =
            ao_integral_input.ao_two_electron_integral_values[integral_index];
        const int left_pair_index =
            ao_integral_input.ao_two_electron_pair_indices[integral_index * 2];
        const int right_pair_index =
            ao_integral_input.ao_two_electron_pair_indices[integral_index * 2 + 1];
        for (std::size_t active_pair_index = 0;
             active_pair_index < n_active_pairs;
             ++active_pair_index) {
          (*pair_gradients)(
              left_pair_index,
              static_cast<Eigen::Index>(active_pair_index)) +=
              ao_integral_value *
              transformed_pair_coefficients(
                  right_pair_index,
                  static_cast<Eigen::Index>(active_pair_index));
          if (left_pair_index != right_pair_index) {
            (*pair_gradients)(
                right_pair_index,
                static_cast<Eigen::Index>(active_pair_index)) +=
                ao_integral_value *
                transformed_pair_coefficients(
                    left_pair_index,
                    static_cast<Eigen::Index>(active_pair_index));
          }
        }
      }
      return;
    }

    std::vector<ExactCtxPairMatrix> partial_pair_gradients(
        std::max(1, n_threads),
        ExactCtxPairMatrix::Zero(
            static_cast<Eigen::Index>(n_ao_pairs),
            static_cast<Eigen::Index>(n_active_pairs)));
#pragma omp parallel
    {
      int thread_index = 0;
#ifdef _OPENMP
      thread_index = omp_get_thread_num();
#endif
      ExactCtxPairMatrix& local_pair_gradients =
          partial_pair_gradients[thread_index];
#pragma omp for schedule(guided, 256)
      for (std::ptrdiff_t integral_offset = 0;
           integral_offset <
               static_cast<std::ptrdiff_t>(
                   ao_integral_input.ao_two_electron_integral_values.size());
           ++integral_offset) {
        const std::size_t integral_index = integral_offset;
        const double ao_integral_value =
            ao_integral_input.ao_two_electron_integral_values[integral_index];
        const int left_pair_index =
            ao_integral_input.ao_two_electron_pair_indices[integral_index * 2];
        const int right_pair_index =
            ao_integral_input.ao_two_electron_pair_indices[integral_index * 2 + 1];
        for (std::size_t active_pair_index = 0;
             active_pair_index < n_active_pairs;
             ++active_pair_index) {
          local_pair_gradients(
              left_pair_index,
              static_cast<Eigen::Index>(active_pair_index)) +=
              ao_integral_value *
              transformed_pair_coefficients(
                  right_pair_index,
                  static_cast<Eigen::Index>(active_pair_index));
          if (left_pair_index != right_pair_index) {
            local_pair_gradients(
                right_pair_index,
                static_cast<Eigen::Index>(active_pair_index)) +=
                ao_integral_value *
                transformed_pair_coefficients(
                    left_pair_index,
                    static_cast<Eigen::Index>(active_pair_index));
          }
        }
      }
    }
    for (const ExactCtxPairMatrix& partial_pair_gradient : partial_pair_gradients) {
      pair_gradients->noalias() += partial_pair_gradient;
    }
    return;
  }

  std::vector<ExactCtxPairMatrix> partial_pair_gradients;
  int n_threads = 1;
#ifdef _OPENMP
  n_threads = xmvb::effective_openmp_thread_count();
#endif
  partial_pair_gradients.assign(
      std::max(1, n_threads),
      ExactCtxPairMatrix::Zero(
          static_cast<Eigen::Index>(n_ao_pairs),
          static_cast<Eigen::Index>(n_active_pairs)));
#pragma omp parallel
  {
    int thread_index = 0;
#ifdef _OPENMP
    thread_index = omp_get_thread_num();
#endif
    ExactCtxPairMatrix& local_pair_gradients =
        partial_pair_gradients[thread_index];
#pragma omp for schedule(guided, 256)
    for (std::ptrdiff_t integral_offset = 0;
         integral_offset <
             static_cast<std::ptrdiff_t>(
                 ao_integral_input.ao_two_electron_integral_values.size());
         ++integral_offset) {
      const std::size_t integral_index = integral_offset;
      const double ao_integral_value =
          ao_integral_input.ao_two_electron_integral_values[integral_index];
      const int i = ao_integral_input.ao_two_electron_integral_indices[integral_index * 4];
      const int j = ao_integral_input.ao_two_electron_integral_indices[integral_index * 4 + 1];
      const int k = ao_integral_input.ao_two_electron_integral_indices[integral_index * 4 + 2];
      const int l = ao_integral_input.ao_two_electron_integral_indices[integral_index * 4 + 3];
      const std::size_t left_pair_index = ao_pair_index(i, j);
      const std::size_t right_pair_index = ao_pair_index(k, l);
      for (std::size_t active_pair_index = 0;
           active_pair_index < n_active_pairs;
           ++active_pair_index) {
        local_pair_gradients(
            static_cast<Eigen::Index>(left_pair_index),
            static_cast<Eigen::Index>(active_pair_index)) +=
            ao_integral_value *
            transformed_pair_coefficients(
                static_cast<Eigen::Index>(right_pair_index),
                static_cast<Eigen::Index>(active_pair_index));
        if (left_pair_index != right_pair_index) {
          local_pair_gradients(
              static_cast<Eigen::Index>(right_pair_index),
              static_cast<Eigen::Index>(active_pair_index)) +=
              ao_integral_value *
              transformed_pair_coefficients(
                  static_cast<Eigen::Index>(left_pair_index),
                  static_cast<Eigen::Index>(active_pair_index));
        }
      }
    }
  }
  for (const ExactCtxPairMatrix& partial_pair_gradient : partial_pair_gradients) {
    pair_gradients->noalias() += partial_pair_gradient;
  }
}

void accumulate_backpropagated_pair_coefficients_to_dense_active_coefficients_from_cache(
    const ExactCtxPairMatrix& pair_gradients,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    const ExactPackedActiveTwoElectronAdjointCache& cache,
    Eigen::MatrixXd* dense_active_gradients) {
  if (dense_active_gradients == nullptr) {
    throw std::invalid_argument("dense active gradient output must not be null");
  }
  const int n_basis_functions = cache.n_basis_functions;
  const int n_active_orbitals = cache.n_active_orbitals;
  const std::size_t n_active_pairs = cache.active_pair_first_indices.size();
  const std::size_t n_ao_pairs =
      static_cast<std::size_t>(n_basis_functions) * (n_basis_functions + 1) / 2;
  if (cache.active_pair_second_indices.size() != n_active_pairs ||
      pair_gradients.rows() != static_cast<Eigen::Index>(n_ao_pairs) ||
      pair_gradients.cols() != static_cast<Eigen::Index>(n_active_pairs) ||
      dense_active_coefficients.rows() != n_basis_functions ||
      dense_active_coefficients.cols() != n_active_orbitals) {
    throw std::invalid_argument("pair-gradient / dense-active matrix shape mismatch");
  }

  // This Eigen path is the accepted-point HVP accumulator. Preserve any fixed
  // term already stored in the output and zero only when we need a fresh shape.
  if (dense_active_gradients->rows() != n_basis_functions ||
      dense_active_gradients->cols() != n_active_orbitals) {
    dense_active_gradients->resize(n_basis_functions, n_active_orbitals);
    dense_active_gradients->setZero();
  }
  int n_threads = 1;
#ifdef _OPENMP
  n_threads = xmvb::effective_openmp_thread_count();
#endif
  if (n_threads <= 1) {
    for (int first_basis_function = 0;
         first_basis_function < n_basis_functions;
         ++first_basis_function) {
      for (int second_basis_function = 0;
           second_basis_function <= first_basis_function;
           ++second_basis_function) {
        const Eigen::Index pair_row =
            static_cast<Eigen::Index>(
                ao_pair_index(first_basis_function, second_basis_function));
        for (std::size_t active_pair_index = 0;
             active_pair_index < n_active_pairs;
             ++active_pair_index) {
          const double pair_gradient =
              pair_gradients(pair_row, static_cast<Eigen::Index>(active_pair_index));
          if (pair_gradient == 0.0) {
            continue;
          }
          const int first_active =
              cache.active_pair_first_indices[active_pair_index];
          const int second_active =
              cache.active_pair_second_indices[active_pair_index];
          (*dense_active_gradients)(first_basis_function, first_active) +=
              pair_gradient *
              dense_active_coefficients(second_basis_function, second_active);
          (*dense_active_gradients)(first_basis_function, second_active) +=
              pair_gradient *
              dense_active_coefficients(second_basis_function, first_active);
          if (second_basis_function != first_basis_function) {
            (*dense_active_gradients)(second_basis_function, first_active) +=
                pair_gradient *
                dense_active_coefficients(first_basis_function, second_active);
            (*dense_active_gradients)(second_basis_function, second_active) +=
                pair_gradient *
                dense_active_coefficients(first_basis_function, first_active);
          }
        }
      }
    }
    return;
  }

#pragma omp parallel for schedule(static)
  for (int basis_function_index = 0;
       basis_function_index < n_basis_functions;
       ++basis_function_index) {
    for (int other_basis_function = 0;
         other_basis_function < n_basis_functions;
         ++other_basis_function) {
      const Eigen::Index pair_row =
          static_cast<Eigen::Index>(
              ao_pair_index(basis_function_index, other_basis_function));
      for (std::size_t active_pair_index = 0;
           active_pair_index < n_active_pairs;
           ++active_pair_index) {
        const double pair_gradient =
            pair_gradients(pair_row, static_cast<Eigen::Index>(active_pair_index));
        if (pair_gradient == 0.0) {
          continue;
        }
        const int first_active =
            cache.active_pair_first_indices[active_pair_index];
        const int second_active =
            cache.active_pair_second_indices[active_pair_index];
        (*dense_active_gradients)(basis_function_index, first_active) +=
            pair_gradient *
            dense_active_coefficients(other_basis_function, second_active);
        (*dense_active_gradients)(basis_function_index, second_active) +=
            pair_gradient *
            dense_active_coefficients(other_basis_function, first_active);
      }
    }
  }
}

}  // namespace

std::vector<double>
compute_exact_packed_active_two_electron_integral_directional_derivative(
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const AoIntegralInput& ao_integral_input,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult* accepted_active_space_two_electron_result) {
  ExactPackedActiveTwoElectronDirectionalDerivativeWorkspace workspace;
  std::vector<double> delta_packed_active_two_electron_integrals;
  compute_exact_packed_active_two_electron_integral_directional_derivative(
      dense_active_coefficients,
      dense_active_direction,
      ao_integral_input,
      n_active_orbitals,
      &workspace,
      &delta_packed_active_two_electron_integrals,
      accepted_active_space_two_electron_result);
  return delta_packed_active_two_electron_integrals;
}

void compute_exact_packed_active_two_electron_integral_directional_derivative(
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const AoIntegralInput& ao_integral_input,
    int n_active_orbitals,
    ExactPackedActiveTwoElectronDirectionalDerivativeWorkspace* workspace,
    std::vector<double>* delta_packed_active_two_electron_integrals,
    const ActiveSpaceTwoElectronResult* accepted_active_space_two_electron_result) {
  const int n_basis_functions = ao_integral_input.n_basis_functions;
  if (n_basis_functions <= 0 || n_active_orbitals <= 0) {
    throw std::invalid_argument(
        "exact packed delta GGO dimensions must be positive");
  }
  if (workspace == nullptr) {
    throw std::invalid_argument("exact packed delta GGO workspace must not be null");
  }
  if (delta_packed_active_two_electron_integrals == nullptr) {
    throw std::invalid_argument("exact packed delta GGO output must not be null");
  }

  if (dense_active_coefficients.rows() != n_basis_functions ||
      dense_active_coefficients.cols() != n_active_orbitals ||
      dense_active_direction.rows() != n_basis_functions ||
      dense_active_direction.cols() != n_active_orbitals) {
    throw std::invalid_argument(
        "dense active coefficient shape mismatch in delta GGO");
  }

  workspace->dense_active_direction = dense_active_direction;

  const auto active_pairs = build_active_pair_list(n_active_orbitals);
  const std::size_t n_active_pairs = active_pairs.size();
  const std::size_t n_ao_pairs =
      n_basis_functions * (n_basis_functions + 1) / 2;

  build_ao_pair_to_active_pair_coefficients(
      dense_active_coefficients,
      n_basis_functions,
      n_active_orbitals,
      active_pairs,
      &workspace->pair_coefficients);
  if (accepted_active_space_two_electron_result != nullptr &&
      accepted_active_space_two_electron_result->dense_ao_pair_products.size() != 0) {
    if (accepted_active_space_two_electron_result->dense_ao_pair_products.size() !=
        n_ao_pairs * n_active_pairs) {
      throw std::invalid_argument(
          "accepted dense AO pair product size mismatch in delta GGO");
    }
    workspace->base_pair_products =
        accepted_active_space_two_electron_result->dense_ao_pair_products;
  } else {
    apply_exact_ao_pair_kernel(
        ao_integral_input,
        workspace->pair_coefficients,
        n_basis_functions,
        n_active_pairs,
        &workspace->base_pair_products);
  }

  build_mixed_ao_pair_to_active_pair_coefficients(
      dense_active_coefficients,
      workspace->dense_active_direction,
      n_basis_functions,
      n_active_orbitals,
      active_pairs,
      &workspace->directional_pair_coefficients);
  apply_exact_ao_pair_kernel(
      ao_integral_input,
      workspace->directional_pair_coefficients,
      n_basis_functions,
      n_active_pairs,
      &workspace->directional_pair_products);

  workspace->delta_active_pair_matrix.resize(
      static_cast<Eigen::Index>(n_active_pairs),
      static_cast<Eigen::Index>(n_active_pairs));
  const Eigen::MatrixXd directional_active_pair_contraction =
      workspace->directional_pair_coefficients.transpose() *
      workspace->base_pair_products;
  // K is symmetric in the packed AO-pair basis, hence
  // B^T K D = (D^T K B)^T. Form the directional tensor with one GEMM.
  workspace->delta_active_pair_matrix =
      directional_active_pair_contraction +
      directional_active_pair_contraction.transpose();

  const std::size_t packed_size =
      n_active_pairs * (n_active_pairs + 1) / 2;
  resize_for_overwrite(
      delta_packed_active_two_electron_integrals,
      packed_size);
  for (std::size_t left_active_pair_index = 0;
       left_active_pair_index < n_active_pairs;
       ++left_active_pair_index) {
    for (std::size_t right_active_pair_index = 0;
         right_active_pair_index <= left_active_pair_index;
         ++right_active_pair_index) {
      const int packed_index =
          TwoElectronIndexer::packed_pair_of_pairs_index(
              static_cast<int>(left_active_pair_index),
              static_cast<int>(right_active_pair_index));
      (*delta_packed_active_two_electron_integrals)[
          packed_index] =
          workspace->delta_active_pair_matrix(
              static_cast<Eigen::Index>(left_active_pair_index),
              static_cast<Eigen::Index>(right_active_pair_index));
    }
  }
}

void compute_exact_packed_active_two_electron_integral_directional_derivative(
    const ExactPackedActiveTwoElectronAdjointCache& accepted_cache,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const AoIntegralInput& ao_integral_input,
    ExactPackedActiveTwoElectronDirectionalDerivativeWorkspace* workspace,
    std::vector<double>* delta_packed_active_two_electron_integrals) {
  if (workspace == nullptr) {
    throw std::invalid_argument("exact packed delta GGO workspace must not be null");
  }
  if (delta_packed_active_two_electron_integrals == nullptr) {
    throw std::invalid_argument("exact packed delta GGO output must not be null");
  }

  const int n_basis_functions = ao_integral_input.n_basis_functions;
  const int n_active_orbitals = accepted_cache.n_active_orbitals;
  if (n_basis_functions <= 0 || n_active_orbitals <= 0) {
    throw std::invalid_argument(
        "exact packed delta GGO cache dimensions must be positive");
  }
  if (accepted_cache.n_basis_functions != n_basis_functions) {
    throw std::invalid_argument("exact packed delta GGO cache basis mismatch");
  }
  if (dense_active_direction.rows() != n_basis_functions ||
      dense_active_direction.cols() != n_active_orbitals) {
    throw std::invalid_argument(
        "dense active direction shape mismatch in cached delta GGO");
  }

  if (accepted_cache.accepted_dense_active_coefficients.rows() !=
          n_basis_functions ||
      accepted_cache.accepted_dense_active_coefficients.cols() !=
          n_active_orbitals) {
    throw std::invalid_argument(
        "cached dense active coefficient size mismatch in delta GGO");
  }
  workspace->dense_active_direction = dense_active_direction;

  const std::size_t n_active_pairs =
      accepted_cache.active_pair_first_indices.size();
  if (accepted_cache.active_pair_second_indices.size() != n_active_pairs) {
    throw std::invalid_argument("exact packed delta GGO cache pair-index mismatch");
  }
  const std::size_t n_ao_pairs =
      n_basis_functions * (n_basis_functions + 1) / 2;
  if (accepted_cache.accepted_pair_coefficients.rows() !=
          static_cast<Eigen::Index>(n_ao_pairs) ||
      accepted_cache.accepted_pair_coefficients.cols() !=
          static_cast<Eigen::Index>(n_active_pairs) ||
      accepted_cache.accepted_base_pair_products.rows() !=
          static_cast<Eigen::Index>(n_ao_pairs) ||
      accepted_cache.accepted_base_pair_products.cols() !=
          static_cast<Eigen::Index>(n_active_pairs)) {
    throw std::invalid_argument(
        "cached accepted pair buffers size mismatch in delta GGO");
  }

  build_mixed_ao_pair_to_active_pair_coefficients_from_cache(
      accepted_cache.accepted_dense_active_coefficients,
      workspace->dense_active_direction,
      accepted_cache,
      &workspace->directional_pair_coefficients);
  apply_exact_ao_pair_kernel(
      ao_integral_input,
      workspace->directional_pair_coefficients,
      n_basis_functions,
      n_active_pairs,
      &workspace->directional_pair_products);

  workspace->delta_active_pair_matrix.resize(
      static_cast<Eigen::Index>(n_active_pairs),
      static_cast<Eigen::Index>(n_active_pairs));
  const Eigen::MatrixXd directional_active_pair_contraction =
      workspace->directional_pair_coefficients.transpose() *
      accepted_cache.accepted_base_pair_products;
  // Reuse symmetry of the accepted AO-pair kernel instead of multiplying the
  // directional product by the accepted coefficients a second time.
  workspace->delta_active_pair_matrix =
      directional_active_pair_contraction +
      directional_active_pair_contraction.transpose();

  const std::size_t packed_size =
      n_active_pairs * (n_active_pairs + 1) / 2;
  resize_for_overwrite(
      delta_packed_active_two_electron_integrals,
      packed_size);
  for (std::size_t left_active_pair_index = 0;
       left_active_pair_index < n_active_pairs;
       ++left_active_pair_index) {
    for (std::size_t right_active_pair_index = 0;
         right_active_pair_index <= left_active_pair_index;
         ++right_active_pair_index) {
      const int packed_index =
          TwoElectronIndexer::packed_pair_of_pairs_index(
              static_cast<int>(left_active_pair_index),
              static_cast<int>(right_active_pair_index));
      (*delta_packed_active_two_electron_integrals)[
          packed_index] =
          workspace->delta_active_pair_matrix(
              static_cast<Eigen::Index>(left_active_pair_index),
              static_cast<Eigen::Index>(right_active_pair_index));
    }
  }
}

Eigen::MatrixXd
compute_exact_packed_active_two_electron_integral_directional_derivative_batch(
    const ExactPackedActiveTwoElectronAdjointCache& accepted_cache,
    const std::vector<Eigen::MatrixXd>& dense_active_directions,
    const AoIntegralInput& ao_integral_input,
    std::vector<ExactCtxPairMatrix>* directional_pair_products) {
  const int n_basis_functions = accepted_cache.n_basis_functions;
  const int n_active_orbitals = accepted_cache.n_active_orbitals;
  const Eigen::Index n_directions =
      static_cast<Eigen::Index>(dense_active_directions.size());
  const Eigen::Index n_active_pairs =
      static_cast<Eigen::Index>(
          accepted_cache.active_pair_first_indices.size());
  const Eigen::Index n_ao_pairs =
      static_cast<Eigen::Index>(n_basis_functions) *
      (n_basis_functions + 1) / 2;
  const Eigen::Index packed_size =
      n_active_pairs * (n_active_pairs + 1) / 2;
  Eigen::MatrixXd packed_directions(packed_size, n_directions);
  if (n_directions == 0) {
    if (directional_pair_products != nullptr) {
      directional_pair_products->clear();
    }
    return packed_directions;
  }
  if (n_basis_functions <= 0 || n_active_orbitals <= 0 ||
      n_active_pairs <= 0 ||
      accepted_cache.active_pair_second_indices.size() !=
          static_cast<std::size_t>(n_active_pairs) ||
      accepted_cache.accepted_pair_coefficients.rows() != n_ao_pairs ||
      accepted_cache.accepted_pair_coefficients.cols() != n_active_pairs ||
      accepted_cache.accepted_base_pair_products.rows() != n_ao_pairs ||
      accepted_cache.accepted_base_pair_products.cols() != n_active_pairs) {
    throw std::invalid_argument(
        "cached exact delta GGO batch has inconsistent accepted dimensions");
  }

  ExactCtxPairMatrix combined_directional_coefficients(
      n_ao_pairs,
      n_active_pairs * n_directions);
  for (Eigen::Index direction = 0;
       direction < n_directions;
       ++direction) {
    if (dense_active_directions[direction].rows() != n_basis_functions ||
        dense_active_directions[direction].cols() != n_active_orbitals) {
      throw std::invalid_argument(
          "dense active direction shape mismatch in delta GGO batch");
    }
    ExactCtxPairMatrix directional_coefficients;
    build_mixed_ao_pair_to_active_pair_coefficients_from_cache(
        accepted_cache.accepted_dense_active_coefficients,
        dense_active_directions[direction],
        accepted_cache,
        &directional_coefficients);
    combined_directional_coefficients.middleCols(
        direction * n_active_pairs,
        n_active_pairs) = directional_coefficients;
  }

  ExactCtxPairMatrix combined_directional_products;
  apply_exact_ao_pair_kernel(
      ao_integral_input,
      combined_directional_coefficients,
      n_basis_functions,
      static_cast<std::size_t>(n_active_pairs * n_directions),
      &combined_directional_products);
  if (directional_pair_products != nullptr) {
    directional_pair_products->resize(n_directions);
  }

  for (Eigen::Index direction = 0;
       direction < n_directions;
       ++direction) {
    const auto directional_coefficients =
        combined_directional_coefficients.middleCols(
            direction * n_active_pairs,
            n_active_pairs);
    const auto directional_products =
        combined_directional_products.middleCols(
            direction * n_active_pairs,
            n_active_pairs);
    if (directional_pair_products != nullptr) {
      (*directional_pair_products)[direction] = directional_products;
    }
    const Eigen::MatrixXd directional_active_pair_contraction =
        directional_coefficients.transpose() *
        accepted_cache.accepted_base_pair_products;
    // Apply the same symmetric-kernel identity independently to every block
    // direction; the directional products remain available for direct-core HVP.
    const Eigen::MatrixXd delta_active_pair_matrix =
        directional_active_pair_contraction +
        directional_active_pair_contraction.transpose();
    for (Eigen::Index left = 0; left < n_active_pairs; ++left) {
      for (Eigen::Index right = 0; right <= left; ++right) {
        const int packed_index =
            TwoElectronIndexer::packed_pair_of_pairs_index(
                static_cast<int>(left),
                static_cast<int>(right));
        packed_directions(packed_index, direction) =
            delta_active_pair_matrix(left, right);
      }
    }
  }
  return packed_directions;
}

Eigen::MatrixXd backpropagate_exact_packed_active_two_electron_gradient(
    const std::vector<double>& packed_active_two_electron_gradient,
    const ExactPackedActiveTwoElectronAdjointCache& accepted_cache) {
  const int n_basis_functions = accepted_cache.n_basis_functions;
  const int n_active_orbitals = accepted_cache.n_active_orbitals;
  const std::size_t n_active_pairs =
      accepted_cache.active_pair_first_indices.size();
  const std::size_t expected_packed_size =
      n_active_pairs * (n_active_pairs + 1) / 2;
  const std::size_t n_ao_pairs =
      static_cast<std::size_t>(n_basis_functions) *
      (n_basis_functions + 1) / 2;
  if (n_basis_functions <= 0 || n_active_orbitals <= 0 ||
      accepted_cache.active_pair_second_indices.size() != n_active_pairs ||
      packed_active_two_electron_gradient.size() != expected_packed_size ||
      accepted_cache.accepted_base_pair_products.rows() !=
          static_cast<Eigen::Index>(n_ao_pairs) ||
      accepted_cache.accepted_base_pair_products.cols() !=
          static_cast<Eigen::Index>(n_active_pairs) ||
      accepted_cache.accepted_dense_active_coefficients.rows() !=
          n_basis_functions ||
      accepted_cache.accepted_dense_active_coefficients.cols() !=
          n_active_orbitals) {
    throw std::invalid_argument(
        "cached exact active-2e adjoint pullback dimensions are inconsistent");
  }

  const auto active_pairs = build_active_pair_list(n_active_orbitals);
  const ExactCtxPairMatrix active_pair_gradient_matrix =
      build_active_pair_gradient_matrix(
          packed_active_two_electron_gradient,
          active_pairs);
  ExactCtxPairMatrix pair_gradients(n_ao_pairs, n_active_pairs);
  pair_gradients.noalias() =
      accepted_cache.accepted_base_pair_products *
      active_pair_gradient_matrix;

  Eigen::MatrixXd dense_active_gradient = Eigen::MatrixXd::Zero(
      n_basis_functions,
      n_active_orbitals);
  accumulate_backpropagated_pair_coefficients_to_dense_active_coefficients_from_cache(
      pair_gradients,
      accepted_cache.accepted_dense_active_coefficients,
      accepted_cache,
      &dense_active_gradient);
  return dense_active_gradient;
}

ExactPackedActiveTwoElectronAdjointCache
build_exact_packed_active_two_electron_adjoint_cache(
    const std::vector<double>& packed_active_two_electron_gradient,
    const Eigen::Ref<const Eigen::MatrixXd>& accepted_dense_active_coefficients,
    const AoIntegralInput& ao_integral_input,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult* accepted_active_space_two_electron_result) {
  const int n_basis_functions = ao_integral_input.n_basis_functions;
  if (n_basis_functions <= 0 || n_active_orbitals <= 0) {
    throw std::invalid_argument("exact 2e HVP cache dimensions must be positive");
  }
  if (ao_integral_input.ao_two_electron_integral_values.empty()) {
    throw std::invalid_argument("exact 2e HVP cache requires materialized AO integrals");
  }

  if (accepted_dense_active_coefficients.rows() != n_basis_functions ||
      accepted_dense_active_coefficients.cols() != n_active_orbitals) {
    throw std::invalid_argument("accepted dense active coefficient shape mismatch");
  }

  const auto active_pairs = build_active_pair_list(n_active_orbitals);
  const std::size_t n_active_pairs = active_pairs.size();
  const std::size_t n_ao_pairs =
      n_basis_functions * (n_basis_functions + 1) / 2;
  const std::size_t expected_packed_gradient_size =
      n_active_pairs * (n_active_pairs + 1) / 2;
  if (packed_active_two_electron_gradient.size() != expected_packed_gradient_size) {
    throw std::invalid_argument("packed active two-electron gradient size mismatch");
  }

  ExactPackedActiveTwoElectronAdjointCache cache;
  cache.n_basis_functions = n_basis_functions;
  cache.n_active_orbitals = n_active_orbitals;
  cache.accepted_dense_active_coefficients = accepted_dense_active_coefficients;
  build_ao_pair_component_tables(
      n_basis_functions,
      &cache.ao_pair_first_indices,
      &cache.ao_pair_second_indices);
  cache.active_pair_first_indices.reserve(n_active_pairs);
  cache.active_pair_second_indices.reserve(n_active_pairs);
  for (const auto& active_pair : active_pairs) {
    cache.active_pair_first_indices.push_back(active_pair.first);
    cache.active_pair_second_indices.push_back(active_pair.second);
  }
  build_ao_pair_to_active_pair_coefficients(
      cache.accepted_dense_active_coefficients,
      n_basis_functions,
      n_active_orbitals,
      active_pairs,
      &cache.accepted_pair_coefficients);
  cache.active_pair_gradient_matrix =
      build_active_pair_gradient_matrix(
          packed_active_two_electron_gradient,
          active_pairs);
  if (accepted_active_space_two_electron_result != nullptr &&
      accepted_active_space_two_electron_result->dense_ao_pair_products.size() != 0) {
    if (accepted_active_space_two_electron_result->dense_ao_pair_products.size() !=
        n_ao_pairs * n_active_pairs) {
      throw std::invalid_argument("accepted dense AO pair product cache size mismatch");
    }
    cache.accepted_base_pair_products =
        accepted_active_space_two_electron_result->dense_ao_pair_products;
  } else {
    apply_exact_ao_pair_kernel(
        ao_integral_input,
        cache.accepted_pair_coefficients,
        n_basis_functions,
        n_active_pairs,
        &cache.accepted_base_pair_products);
  }

  cache.accepted_base_pair_gradients =
      multiply_pair_coefficients_by_gradient_matrix(
          cache.accepted_base_pair_products,
          cache.active_pair_gradient_matrix);
  return cache;
}

Eigen::MatrixXd apply_exact_packed_active_two_electron_adjoint_hessian_vector(
    const ExactPackedActiveTwoElectronAdjointCache& accepted_cache,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const AoIntegralInput& ao_integral_input) {
  ExactPackedActiveTwoElectronApplyWorkspace workspace;
  Eigen::MatrixXd dense_active_gradient_direction;
  apply_exact_packed_active_two_electron_adjoint_hessian_vector(
      accepted_cache,
      dense_active_direction,
      ao_integral_input,
      &workspace,
      &dense_active_gradient_direction);
  return dense_active_gradient_direction;
}

void apply_exact_packed_active_two_electron_adjoint_hessian_vector(
    const ExactPackedActiveTwoElectronAdjointCache& accepted_cache,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const AoIntegralInput& ao_integral_input,
    ExactPackedActiveTwoElectronApplyWorkspace* workspace,
    Eigen::MatrixXd* dense_active_gradient_direction) {
  if (workspace == nullptr) {
    throw std::invalid_argument("exact 2e workspace must not be null");
  }
  const int n_basis_functions = ao_integral_input.n_basis_functions;
  const int n_active_orbitals = accepted_cache.n_active_orbitals;
  if (n_basis_functions <= 0 || n_active_orbitals <= 0) {
    throw std::invalid_argument("exact two-electron HVP dimensions must be positive");
  }
  if (accepted_cache.n_basis_functions != n_basis_functions) {
    throw std::invalid_argument("exact 2e cache basis dimension mismatch");
  }
  if (ao_integral_input.ao_two_electron_integral_values.empty()) {
    throw std::invalid_argument("exact two-electron HVP requires materialized AO integrals");
  }

  if (dense_active_direction.rows() != n_basis_functions ||
      dense_active_direction.cols() != n_active_orbitals) {
    throw std::invalid_argument("dense active coefficient shape mismatch in exact two-electron HVP");
  }
  const std::size_t n_active_pairs =
      accepted_cache.active_pair_first_indices.size();
  if (accepted_cache.active_pair_second_indices.size() != n_active_pairs) {
    throw std::invalid_argument("exact 2e cache active-pair index size mismatch");
  }
  const std::size_t n_ao_pairs =
      n_basis_functions * (n_basis_functions + 1) / 2;
  workspace->dense_active_direction = dense_active_direction;

  if (accepted_cache.active_pair_gradient_matrix.rows() !=
          static_cast<Eigen::Index>(n_active_pairs) ||
      accepted_cache.active_pair_gradient_matrix.cols() !=
          static_cast<Eigen::Index>(n_active_pairs)) {
    throw std::invalid_argument("exact 2e cache pair-gradient size mismatch");
  }
  if (accepted_cache.accepted_base_pair_gradients.rows() !=
          static_cast<Eigen::Index>(n_ao_pairs) ||
      accepted_cache.accepted_base_pair_gradients.cols() !=
          static_cast<Eigen::Index>(n_active_pairs)) {
    throw std::invalid_argument("exact 2e cache base-pair-gradient size mismatch");
  }

  workspace->dense_active_gradient_direction.resize(
      n_basis_functions,
      n_active_orbitals);
  workspace->dense_active_gradient_direction.setZero();
  // Keep the fixed-backprop term on the generic packed-pair contraction until
  // the cached full-matrix shortcut is validated against finite differences on
  // the sparse mixed-chart exact_ctx path. The direct-core 241 diagnostic
  // currently shows the exact-2e mismatch lives in this stage rather than in
  // the AO-H1E or orbital-pullback chains.
  accumulate_backpropagated_pair_coefficients_to_dense_active_coefficients_from_cache(
      accepted_cache.accepted_base_pair_gradients,
      workspace->dense_active_direction,
      accepted_cache,
      &workspace->dense_active_gradient_direction);
  build_mixed_ao_pair_to_active_pair_coefficients_from_cache(
      accepted_cache.accepted_dense_active_coefficients,
      workspace->dense_active_direction,
      accepted_cache,
      &workspace->mixed_pair_coefficients);

  multiply_pair_coefficients_by_gradient_matrix(
      workspace->mixed_pair_coefficients,
      accepted_cache.active_pair_gradient_matrix,
      &workspace->transformed_pair_coefficients);
  apply_exact_ao_pair_kernel(
      ao_integral_input,
      workspace->transformed_pair_coefficients,
      n_basis_functions,
      n_active_pairs,
      &workspace->pair_gradients);
  accumulate_backpropagated_pair_coefficients_to_dense_active_coefficients_from_cache(
      workspace->pair_gradients,
      accepted_cache.accepted_dense_active_coefficients,
      accepted_cache,
      &workspace->dense_active_gradient_direction);

  if (dense_active_gradient_direction != nullptr) {
    *dense_active_gradient_direction =
        workspace->dense_active_gradient_direction;
  }
}

void apply_exact_packed_active_two_electron_adjoint_hessian_vector_fused(
    const ExactPackedActiveTwoElectronAdjointCache& accepted_cache,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const AoIntegralInput& ao_integral_input,
    const ExactCtxPairMatrix& directional_pair_products,
    ExactPackedActiveTwoElectronApplyWorkspace* workspace,
    Eigen::MatrixXd* dense_active_gradient_direction) {
  if (workspace == nullptr) {
    throw std::invalid_argument("exact 2e fused workspace must not be null");
  }
  const int n_basis_functions = ao_integral_input.n_basis_functions;
  const int n_active_orbitals = accepted_cache.n_active_orbitals;
  const std::size_t n_active_pairs =
      accepted_cache.active_pair_first_indices.size();
  const std::size_t n_ao_pairs =
      n_basis_functions * (n_basis_functions + 1) / 2;

  workspace->dense_active_direction = dense_active_direction;
  workspace->dense_active_gradient_direction.resize(
      n_basis_functions, n_active_orbitals);
  workspace->dense_active_gradient_direction.setZero();

  // Fixed backprop term (same as non-fused path).
  accumulate_backpropagated_pair_coefficients_to_dense_active_coefficients_from_cache(
      accepted_cache.accepted_base_pair_gradients,
      workspace->dense_active_direction,
      accepted_cache,
      &workspace->dense_active_gradient_direction);

  // Reuse forward K*mixed: pair_gradients = (K * mixed) * gradient_matrix.
  // Saves one full apply_exact_ao_pair_kernel call (~500M FLOPs) per HVP.
  workspace->pair_gradients.resize(n_ao_pairs, n_active_pairs);
  workspace->pair_gradients.noalias() =
      directional_pair_products * accepted_cache.active_pair_gradient_matrix;

  // Final backprop (same as non-fused path).
  accumulate_backpropagated_pair_coefficients_to_dense_active_coefficients_from_cache(
      workspace->pair_gradients,
      accepted_cache.accepted_dense_active_coefficients,
      accepted_cache,
      &workspace->dense_active_gradient_direction);

  if (dense_active_gradient_direction != nullptr) {
    *dense_active_gradient_direction =
        workspace->dense_active_gradient_direction;
  }
}

Eigen::MatrixXd apply_exact_packed_active_two_electron_adjoint_hessian_vector(
    const std::vector<double>& packed_active_two_electron_gradient,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const AoIntegralInput& ao_integral_input,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult* accepted_active_space_two_electron_result) {
  const auto accepted_cache =
      build_exact_packed_active_two_electron_adjoint_cache(
          packed_active_two_electron_gradient,
          dense_active_coefficients,
          ao_integral_input,
          n_active_orbitals,
          accepted_active_space_two_electron_result);
  return apply_exact_packed_active_two_electron_adjoint_hessian_vector(
      accepted_cache,
      dense_active_direction,
      ao_integral_input);
}

}  // namespace xmvb::vb
