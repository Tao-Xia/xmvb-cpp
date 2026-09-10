#include "vbscf/integrals/active/active_space_two_electron_response_internal.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "core/parallel/openmp.hpp"

namespace xmvb::vb::detail {

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
void apply_ao_pair_graph_matrix_row_major_fixed_hvp(
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& row_offsets,
    const std::vector<int>& column_pair_indices,
    const std::vector<int>& integral_indices,
    const double* transformed_pair_coefficients_data,
    std::size_t n_bf_pairs,
    int n_threads,
    double* pair_gradients_data) {
  if (transformed_pair_coefficients_data == nullptr ||
      pair_gradients_data == nullptr) {
    throw std::invalid_argument("AO pair graph row-major HVP buffers must not be null");
  }
#pragma omp parallel for schedule(guided, 64) num_threads(n_threads)
  for (std::ptrdiff_t row_offset = 0;
       row_offset < static_cast<std::ptrdiff_t>(n_bf_pairs);
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

void apply_exact_ao_pair_kernel(
    const AoIntegralInput& ao_integral_input,
    const ExactCtxPairMatrix& transformed_pair_coefficients,
    int n_bf,
    std::size_t n_active_pairs,
    ExactCtxPairMatrix* pair_gradients) {
  if (pair_gradients == nullptr) {
    throw std::invalid_argument("exact AO-pair kernel output must not be null");
  }
  const std::size_t n_bf_pairs =
      n_bf * (n_bf + 1) / 2;
  if (transformed_pair_coefficients.rows() != static_cast<Eigen::Index>(n_bf_pairs) ||
      transformed_pair_coefficients.cols() != static_cast<Eigen::Index>(n_active_pairs)) {
    throw std::invalid_argument("transformed pair coefficient matrix shape mismatch");
  }

  pair_gradients->resize(
      static_cast<Eigen::Index>(n_bf_pairs),
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
            n_bf_pairs,
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
            n_bf_pairs,
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
            n_bf_pairs,
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
            n_bf_pairs,
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
            n_bf_pairs,
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
            n_bf_pairs,
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
            n_bf_pairs,
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
            n_bf_pairs,
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
            n_bf_pairs,
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
            n_bf_pairs,
            n_threads,
            pair_gradients_data);
        return;
      default:
        break;
    }
#pragma omp parallel for schedule(guided, 64) num_threads(n_threads)
    for (std::ptrdiff_t row_offset = 0;
         row_offset < static_cast<std::ptrdiff_t>(n_bf_pairs);
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
            static_cast<Eigen::Index>(n_bf_pairs),
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
          static_cast<Eigen::Index>(n_bf_pairs),
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

}  // namespace xmvb::vb::detail
