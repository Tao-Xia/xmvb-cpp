#include "vb/orbital/active_space_two_electron_builder.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "core/openmp_utils.hpp"
#include "vb/matrices/eigen_matrix_storage_utils.hpp"
#include "vb/orbital/ao_two_electron_pair_index_utils.hpp"
#include "vb/matrices/two_electron_indexer.hpp"

namespace xmvb::vb {

namespace {

struct ActivePair {
  int first = 0;
  int second = 0;
};

struct SparseActivePairCoefficient {
  int packed_active_pair_index = 0;
  double value = 0.0;
};

struct SparseAoPairCoefficients {
  std::vector<int> row_offsets;
  std::vector<int> packed_active_pair_indices;
  std::vector<double> values;
};

std::size_t packed_active_pair_count(int n_active_orbitals) {
  const std::size_t n_active = static_cast<std::size_t>(n_active_orbitals);
  return n_active * (n_active + 1) / 2;
}

std::size_t ao_pair_index(int first, int second) {
  if (first >= second) {
    const std::size_t first_index = static_cast<std::size_t>(first);
    return first_index * (first_index + 1) / 2 +
        static_cast<std::size_t>(second);
  }
  const std::size_t second_index = static_cast<std::size_t>(second);
  return second_index * (second_index + 1) / 2 +
      static_cast<std::size_t>(first);
}

std::vector<std::size_t> build_pair_row_offsets(
    std::size_t n_ao_pairs,
    std::size_t n_active_pairs) {
  std::vector<std::size_t> row_offsets(n_ao_pairs, 0);
  for (std::size_t ao_pair_index = 0; ao_pair_index < n_ao_pairs; ++ao_pair_index) {
    row_offsets[ao_pair_index] = ao_pair_index * n_active_pairs;
  }
  return row_offsets;
}

template <int NActivePairs>
inline void accumulate_scaled_active_pair_row(
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
std::vector<double> apply_sparse_ao_integral_matrix_single_thread_fixed(
    const double* ao_two_electron_integral_values_data,
    const int* ao_two_electron_pair_indices_data,
    const double* pair_coefficients_data,
    const std::size_t* pair_row_offsets_data,
    std::size_t n_ao_pairs,
    std::size_t n_integrals) {
  constexpr std::size_t kActivePairCount = static_cast<std::size_t>(NActivePairs);
  std::vector<double> pair_products(
      n_ao_pairs * kActivePairCount,
      0.0);
  double* pair_products_data = pair_products.data();
  for (std::size_t integral_index = 0;
       integral_index < n_integrals;
       ++integral_index) {
    const double ao_integral_value =
        ao_two_electron_integral_values_data[integral_index];
    const std::size_t left_pair_index = static_cast<std::size_t>(
        ao_two_electron_pair_indices_data[integral_index * 2]);
    const std::size_t right_pair_index = static_cast<std::size_t>(
        ao_two_electron_pair_indices_data[integral_index * 2 + 1]);
    const double* right_row =
        pair_coefficients_data + pair_row_offsets_data[right_pair_index];
    double* left_product_row =
        pair_products_data + pair_row_offsets_data[left_pair_index];
    accumulate_scaled_active_pair_row<NActivePairs>(
        left_product_row,
        right_row,
        ao_integral_value);

    if (left_pair_index != right_pair_index) {
      const double* left_row =
          pair_coefficients_data + pair_row_offsets_data[left_pair_index];
      double* right_product_row =
          pair_products_data + pair_row_offsets_data[right_pair_index];
      accumulate_scaled_active_pair_row<NActivePairs>(
          right_product_row,
          left_row,
          ao_integral_value);
    }
  }
  return pair_products;
}

template <int NActivePairs>
std::vector<double> apply_ao_pair_graph_matrix_single_thread_fixed(
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& row_offsets,
    const std::vector<int>& column_pair_indices,
    const std::vector<int>& integral_indices,
    const double* pair_coefficients_data,
    const std::size_t* pair_row_offsets_data,
    std::size_t n_ao_pairs) {
  constexpr std::size_t kActivePairCount = static_cast<std::size_t>(NActivePairs);
  std::vector<double> pair_products(
      n_ao_pairs * kActivePairCount,
      0.0);
  double* pair_products_data = pair_products.data();
  for (std::size_t row_index = 0; row_index < n_ao_pairs; ++row_index) {
    double* target_row =
        pair_products_data + pair_row_offsets_data[row_index];
    const int begin = row_offsets[row_index];
    const int end = row_offsets[row_index + 1];
    for (int entry_offset = begin; entry_offset < end; ++entry_offset) {
      const int column_pair_index = column_pair_indices[entry_offset];
      const int integral_index = integral_indices[entry_offset];
      const double ao_integral_value =
          ao_two_electron_integral_values[static_cast<std::size_t>(integral_index)];
      const double* source_row =
          pair_coefficients_data +
          pair_row_offsets_data[static_cast<std::size_t>(column_pair_index)];
      accumulate_scaled_active_pair_row<NActivePairs>(
          target_row,
          source_row,
          ao_integral_value);
    }
  }
  return pair_products;
}

std::vector<ActivePair> build_active_pair_list(int n_active_orbitals) {
  std::vector<ActivePair> active_pairs;
  active_pairs.reserve(packed_active_pair_count(n_active_orbitals));
  for (int first = 0; first < n_active_orbitals; ++first) {
    for (int second = 0; second <= first; ++second) {
      active_pairs.push_back({first, second});
    }
  }
  return active_pairs;
}

std::vector<double> build_dense_active_coefficients(
    const Eigen::Ref<const Eigen::MatrixXd>& active_auxiliary_orbitals,
    int n_basis_functions,
    int n_active_orbitals) {
  const std::size_t n_basis = static_cast<std::size_t>(n_basis_functions);
  const std::size_t basis_stride = static_cast<std::size_t>(n_active_orbitals);
  std::vector<double> dense_active_coefficients(
      n_basis * basis_stride,
      0.0);

#pragma omp parallel for schedule(static)
  for (int basis_function_index = 0;
       basis_function_index < n_basis_functions;
       ++basis_function_index) {
    double* basis_coefficients =
        dense_active_coefficients.data() +
        static_cast<std::size_t>(basis_function_index) * basis_stride;
#pragma omp simd
    for (int active_orbital_index = 0;
         active_orbital_index < n_active_orbitals;
         ++active_orbital_index) {
      basis_coefficients[active_orbital_index] =
          active_auxiliary_orbitals(basis_function_index, active_orbital_index);
    }
  }

  return dense_active_coefficients;
}

std::vector<double> build_dense_active_coefficients(
    const OrbitalPreparationResult& orbital_preparation_result,
    int n_basis_functions,
    int n_active_orbitals) {
  const std::size_t n_basis = static_cast<std::size_t>(n_basis_functions);
  if (orbital_preparation_result.active_sparse_row_offsets.size() !=
      n_basis + 1) {
    throw std::invalid_argument("active_sparse_row_offsets size mismatch");
  }
  if (orbital_preparation_result.active_sparse_orbital_indices.size() !=
      orbital_preparation_result.active_sparse_values.size()) {
    throw std::invalid_argument("active sparse index/value sizes are inconsistent");
  }

  const std::size_t basis_stride = static_cast<std::size_t>(n_active_orbitals);
  std::vector<double> dense_active_coefficients(
      n_basis * basis_stride,
      0.0);
  for (int basis_function_index = 0;
       basis_function_index < n_basis_functions;
       ++basis_function_index) {
    const int begin =
        orbital_preparation_result.active_sparse_row_offsets[basis_function_index];
    const int end =
        orbital_preparation_result.active_sparse_row_offsets[basis_function_index + 1];
    double* basis_coefficients =
        dense_active_coefficients.data() +
        static_cast<std::size_t>(basis_function_index) * basis_stride;
    for (int offset = begin; offset < end; ++offset) {
      const int active_orbital_index =
          orbital_preparation_result.active_sparse_orbital_indices[offset];
      if (active_orbital_index < 0 || active_orbital_index >= n_active_orbitals) {
        throw std::invalid_argument("active sparse orbital index out of range");
      }
      basis_coefficients[active_orbital_index] =
          orbital_preparation_result.active_sparse_values[offset];
    }
  }

  return dense_active_coefficients;
}

std::vector<double> build_ao_pair_to_active_pair_coefficients(
    const std::vector<double>& dense_active_coefficients,
    int n_basis_functions,
    int n_active_orbitals,
    const std::vector<ActivePair>& active_pairs) {
  const std::size_t n_basis = static_cast<std::size_t>(n_basis_functions);
  const std::size_t n_ao_pairs = n_basis * (n_basis + 1) / 2;
  const std::size_t n_active_pairs = active_pairs.size();
  const std::size_t basis_stride = static_cast<std::size_t>(n_active_orbitals);
  std::vector<double> ao_pair_to_active_pair_coefficients(
      n_ao_pairs * n_active_pairs,
      0.0);

#pragma omp parallel for schedule(static)
  for (int first_basis_function = 0;
       first_basis_function < n_basis_functions;
       ++first_basis_function) {
    const double* first_coefficients =
        dense_active_coefficients.data() +
        static_cast<std::size_t>(first_basis_function) * basis_stride;
    for (int second_basis_function = 0;
         second_basis_function <= first_basis_function;
         ++second_basis_function) {
      const double* second_coefficients =
          dense_active_coefficients.data() +
          static_cast<std::size_t>(second_basis_function) * basis_stride;
      double* pair_coefficients =
          ao_pair_to_active_pair_coefficients.data() +
          ao_pair_index(first_basis_function, second_basis_function) * n_active_pairs;
#pragma omp simd
      for (std::size_t active_pair_index_offset = 0;
           active_pair_index_offset < n_active_pairs;
           ++active_pair_index_offset) {
        const auto& active_pair =
            active_pairs[active_pair_index_offset];
        double coefficient =
            first_coefficients[active_pair.first] *
            second_coefficients[active_pair.second];
        if (first_basis_function != second_basis_function) {
          coefficient +=
              second_coefficients[active_pair.first] *
              first_coefficients[active_pair.second];
        }
        pair_coefficients[active_pair_index_offset] = coefficient;
      }
    }
  }

  return ao_pair_to_active_pair_coefficients;
}

std::vector<double> apply_sparse_ao_integral_matrix(
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& ao_two_electron_pair_indices,
    const std::vector<double>& pair_coefficients,
    int n_basis_functions,
    std::size_t n_active_pairs) {
  const std::size_t n_ao_pairs =
      n_basis_functions * (n_basis_functions + 1) / 2;
  const double* ao_two_electron_integral_values_data =
      ao_two_electron_integral_values.data();
  const int* ao_two_electron_pair_indices_data =
      ao_two_electron_pair_indices.data();
  const double* pair_coefficients_data = pair_coefficients.data();
    const std::vector<std::size_t> pair_row_offsets =
      build_pair_row_offsets(n_ao_pairs, n_active_pairs);
  const std::size_t* pair_row_offsets_data = pair_row_offsets.data();

  int n_threads = 1;
#ifdef _OPENMP
  n_threads = xmvb::effective_openmp_thread_count();
#endif
  if (n_threads <= 1) {
    switch (n_active_pairs) {
      case 1:
        return apply_sparse_ao_integral_matrix_single_thread_fixed<1>(
            ao_two_electron_integral_values_data,
            ao_two_electron_pair_indices_data,
            pair_coefficients_data,
            pair_row_offsets_data,
            n_ao_pairs,
            ao_two_electron_integral_values.size());
      case 3:
        return apply_sparse_ao_integral_matrix_single_thread_fixed<3>(
            ao_two_electron_integral_values_data,
            ao_two_electron_pair_indices_data,
            pair_coefficients_data,
            pair_row_offsets_data,
            n_ao_pairs,
            ao_two_electron_integral_values.size());
      case 6:
        return apply_sparse_ao_integral_matrix_single_thread_fixed<6>(
            ao_two_electron_integral_values_data,
            ao_two_electron_pair_indices_data,
            pair_coefficients_data,
            pair_row_offsets_data,
            n_ao_pairs,
            ao_two_electron_integral_values.size());
      case 10:
        return apply_sparse_ao_integral_matrix_single_thread_fixed<10>(
            ao_two_electron_integral_values_data,
            ao_two_electron_pair_indices_data,
            pair_coefficients_data,
            pair_row_offsets_data,
            n_ao_pairs,
            ao_two_electron_integral_values.size());
      case 15:
        return apply_sparse_ao_integral_matrix_single_thread_fixed<15>(
            ao_two_electron_integral_values_data,
            ao_two_electron_pair_indices_data,
            pair_coefficients_data,
            pair_row_offsets_data,
            n_ao_pairs,
            ao_two_electron_integral_values.size());
      case 21:
        return apply_sparse_ao_integral_matrix_single_thread_fixed<21>(
            ao_two_electron_integral_values_data,
            ao_two_electron_pair_indices_data,
            pair_coefficients_data,
            pair_row_offsets_data,
            n_ao_pairs,
            ao_two_electron_integral_values.size());
      case 28:
        return apply_sparse_ao_integral_matrix_single_thread_fixed<28>(
            ao_two_electron_integral_values_data,
            ao_two_electron_pair_indices_data,
            pair_coefficients_data,
            pair_row_offsets_data,
            n_ao_pairs,
            ao_two_electron_integral_values.size());
      case 36:
        return apply_sparse_ao_integral_matrix_single_thread_fixed<36>(
            ao_two_electron_integral_values_data,
            ao_two_electron_pair_indices_data,
            pair_coefficients_data,
            pair_row_offsets_data,
            n_ao_pairs,
            ao_two_electron_integral_values.size());
      case 45:
        return apply_sparse_ao_integral_matrix_single_thread_fixed<45>(
            ao_two_electron_integral_values_data,
            ao_two_electron_pair_indices_data,
            pair_coefficients_data,
            pair_row_offsets_data,
            n_ao_pairs,
            ao_two_electron_integral_values.size());
      case 55:
        return apply_sparse_ao_integral_matrix_single_thread_fixed<55>(
            ao_two_electron_integral_values_data,
            ao_two_electron_pair_indices_data,
            pair_coefficients_data,
            pair_row_offsets_data,
            n_ao_pairs,
            ao_two_electron_integral_values.size());
      default:
        break;
    }

    std::vector<double> pair_products(n_ao_pairs * n_active_pairs, 0.0);
    double* pair_products_data = pair_products.data();
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
          pair_coefficients_data + pair_row_offsets_data[right_pair_index];
      double* left_product_row =
          pair_products_data + pair_row_offsets_data[left_pair_index];
#pragma omp simd
      for (std::size_t active_pair_index = 0;
           active_pair_index < n_active_pairs;
           ++active_pair_index) {
        left_product_row[active_pair_index] +=
            ao_integral_value * right_row[active_pair_index];
      }

      if (left_pair_index != right_pair_index) {
        const double* left_row =
            pair_coefficients_data + pair_row_offsets_data[left_pair_index];
        double* right_product_row =
            pair_products_data + pair_row_offsets_data[right_pair_index];
#pragma omp simd
        for (std::size_t active_pair_index = 0;
             active_pair_index < n_active_pairs;
             ++active_pair_index) {
          right_product_row[active_pair_index] +=
              ao_integral_value * left_row[active_pair_index];
        }
      }
    }
    return pair_products;
  }

  std::vector<std::vector<double>> partial_pair_products;
  partial_pair_products.assign(
      n_threads,
      std::vector<double>(n_ao_pairs * n_active_pairs, 0.0));

#pragma omp parallel
  {
    int thread_index = 0;
#ifdef _OPENMP
    thread_index = omp_get_thread_num();
#endif
    auto& local_pair_products =
        partial_pair_products[thread_index];

#pragma omp for schedule(guided, 256)
    for (std::ptrdiff_t integral_offset = 0;
         integral_offset < static_cast<std::ptrdiff_t>(ao_two_electron_integral_values.size());
         ++integral_offset) {
      const std::size_t integral_index = integral_offset;
      const double ao_integral_value =
          ao_two_electron_integral_values_data[integral_index];
      const std::size_t left_pair_index =
          ao_two_electron_pair_indices_data[integral_index * 2];
      const std::size_t right_pair_index =
          ao_two_electron_pair_indices_data[integral_index * 2 + 1];
      const double* right_row =
          pair_coefficients_data +
          pair_row_offsets_data[right_pair_index];
      double* left_product_row =
          local_pair_products.data() +
          pair_row_offsets_data[left_pair_index];
#pragma omp simd
      for (std::size_t active_pair_index = 0;
           active_pair_index < n_active_pairs;
           ++active_pair_index) {
        left_product_row[active_pair_index] +=
            ao_integral_value * right_row[active_pair_index];
      }

      if (left_pair_index != right_pair_index) {
        const double* left_row =
            pair_coefficients_data +
            pair_row_offsets_data[left_pair_index];
        double* right_product_row =
            local_pair_products.data() +
            pair_row_offsets_data[right_pair_index];
#pragma omp simd
        for (std::size_t active_pair_index = 0;
             active_pair_index < n_active_pairs;
             ++active_pair_index) {
          right_product_row[active_pair_index] +=
              ao_integral_value * left_row[active_pair_index];
        }
      }
    }
  }

  std::vector<double> pair_products(n_ao_pairs * n_active_pairs, 0.0);
  for (const auto& partial_pair_product : partial_pair_products) {
    for (std::size_t index = 0; index < pair_products.size(); ++index) {
      pair_products[index] += partial_pair_product[index];
    }
  }

  return pair_products;
}

std::vector<double> apply_ao_pair_graph_matrix(
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& row_offsets,
    const std::vector<int>& column_pair_indices,
    const std::vector<int>& integral_indices,
    const std::vector<double>& pair_coefficients,
    std::size_t n_ao_pairs,
    std::size_t n_active_pairs) {
  if (row_offsets.size() != n_ao_pairs + 1) {
    throw std::invalid_argument("AO pair graph row offset size mismatch");
  }
  if (column_pair_indices.size() != integral_indices.size()) {
    throw std::invalid_argument("AO pair graph column/integral size mismatch");
  }
  if (row_offsets.back() != static_cast<int>(column_pair_indices.size())) {
    throw std::invalid_argument("AO pair graph row offsets do not cover all entries");
  }

  std::vector<double> pair_products(n_ao_pairs * n_active_pairs, 0.0);
  const std::vector<std::size_t> pair_row_offsets =
      build_pair_row_offsets(n_ao_pairs, n_active_pairs);
  const std::size_t* pair_row_offsets_data = pair_row_offsets.data();
  int n_threads = 1;
#ifdef _OPENMP
  n_threads = xmvb::effective_openmp_thread_count();
#endif
  if (n_threads <= 1) {
    switch (n_active_pairs) {
      case 1:
        return apply_ao_pair_graph_matrix_single_thread_fixed<1>(
            ao_two_electron_integral_values,
            row_offsets,
            column_pair_indices,
            integral_indices,
            pair_coefficients.data(),
            pair_row_offsets_data,
            n_ao_pairs);
      case 3:
        return apply_ao_pair_graph_matrix_single_thread_fixed<3>(
            ao_two_electron_integral_values,
            row_offsets,
            column_pair_indices,
            integral_indices,
            pair_coefficients.data(),
            pair_row_offsets_data,
            n_ao_pairs);
      case 6:
        return apply_ao_pair_graph_matrix_single_thread_fixed<6>(
            ao_two_electron_integral_values,
            row_offsets,
            column_pair_indices,
            integral_indices,
            pair_coefficients.data(),
            pair_row_offsets_data,
            n_ao_pairs);
      case 10:
        return apply_ao_pair_graph_matrix_single_thread_fixed<10>(
            ao_two_electron_integral_values,
            row_offsets,
            column_pair_indices,
            integral_indices,
            pair_coefficients.data(),
            pair_row_offsets_data,
            n_ao_pairs);
      case 15:
        return apply_ao_pair_graph_matrix_single_thread_fixed<15>(
            ao_two_electron_integral_values,
            row_offsets,
            column_pair_indices,
            integral_indices,
            pair_coefficients.data(),
            pair_row_offsets_data,
            n_ao_pairs);
      case 21:
        return apply_ao_pair_graph_matrix_single_thread_fixed<21>(
            ao_two_electron_integral_values,
            row_offsets,
            column_pair_indices,
            integral_indices,
            pair_coefficients.data(),
            pair_row_offsets_data,
            n_ao_pairs);
      case 28:
        return apply_ao_pair_graph_matrix_single_thread_fixed<28>(
            ao_two_electron_integral_values,
            row_offsets,
            column_pair_indices,
            integral_indices,
            pair_coefficients.data(),
            pair_row_offsets_data,
            n_ao_pairs);
      case 36:
        return apply_ao_pair_graph_matrix_single_thread_fixed<36>(
            ao_two_electron_integral_values,
            row_offsets,
            column_pair_indices,
            integral_indices,
            pair_coefficients.data(),
            pair_row_offsets_data,
            n_ao_pairs);
      case 45:
        return apply_ao_pair_graph_matrix_single_thread_fixed<45>(
            ao_two_electron_integral_values,
            row_offsets,
            column_pair_indices,
            integral_indices,
            pair_coefficients.data(),
            pair_row_offsets_data,
            n_ao_pairs);
      case 55:
        return apply_ao_pair_graph_matrix_single_thread_fixed<55>(
            ao_two_electron_integral_values,
            row_offsets,
            column_pair_indices,
            integral_indices,
            pair_coefficients.data(),
            pair_row_offsets_data,
            n_ao_pairs);
      default:
        break;
    }

    for (std::size_t row_index = 0; row_index < n_ao_pairs; ++row_index) {
      double* target_row =
          pair_products.data() + pair_row_offsets_data[row_index];
      const int begin = row_offsets[row_index];
      const int end = row_offsets[row_index + 1];
      for (int entry_offset = begin; entry_offset < end; ++entry_offset) {
        const int column_pair_index = column_pair_indices[entry_offset];
        const int integral_index = integral_indices[entry_offset];
        const double ao_integral_value = ao_two_electron_integral_values[integral_index];
        const double* source_row =
            pair_coefficients.data() +
            pair_row_offsets_data[column_pair_index];
#pragma omp simd
        for (std::size_t active_pair_index = 0;
             active_pair_index < n_active_pairs;
             ++active_pair_index) {
          target_row[active_pair_index] +=
              ao_integral_value * source_row[active_pair_index];
        }
      }
    }
    return pair_products;
  }

#pragma omp parallel for schedule(guided, 64)
  for (std::ptrdiff_t row_offset = 0;
       row_offset < static_cast<std::ptrdiff_t>(n_ao_pairs);
       ++row_offset) {
    const std::size_t row_index = row_offset;
    double* target_row =
        pair_products.data() + pair_row_offsets_data[row_index];
    const int begin = row_offsets[row_index];
    const int end = row_offsets[row_index + 1];
    for (int entry_offset = begin; entry_offset < end; ++entry_offset) {
      const int column_pair_index = column_pair_indices[entry_offset];
      const int integral_index = integral_indices[entry_offset];
      const double ao_integral_value = ao_two_electron_integral_values[integral_index];
      const double* source_row =
          pair_coefficients.data() +
          pair_row_offsets_data[column_pair_index];
#pragma omp simd
      for (std::size_t active_pair_index = 0;
           active_pair_index < n_active_pairs;
           ++active_pair_index) {
        target_row[active_pair_index] +=
            ao_integral_value * source_row[active_pair_index];
      }
    }
  }

  return pair_products;
}

std::vector<double> contract_pair_coefficients_to_packed_active_integrals(
    const std::vector<double>& ao_pair_to_active_pair_coefficients,
    const std::vector<double>& pair_products,
    std::size_t n_ao_pairs,
    std::size_t n_active_pairs) {
  Eigen::MatrixXd active_pair_matrix =
      Eigen::MatrixXd::Zero(
          static_cast<Eigen::Index>(n_active_pairs),
          static_cast<Eigen::Index>(n_active_pairs));
  for (std::size_t ao_pair_index = 0;
       ao_pair_index < n_ao_pairs;
       ++ao_pair_index) {
    const Eigen::Map<const Eigen::VectorXd> pair_coefficient_row(
        ao_pair_to_active_pair_coefficients.data() +
            ao_pair_index * n_active_pairs,
        static_cast<Eigen::Index>(n_active_pairs));
    const Eigen::Map<const Eigen::VectorXd> pair_product_row(
        pair_products.data() + ao_pair_index * n_active_pairs,
        static_cast<Eigen::Index>(n_active_pairs));
    active_pair_matrix.noalias() +=
        pair_coefficient_row * pair_product_row.transpose();
  }
  std::vector<double> packed_active_two_electron_integrals(
      n_active_pairs * (n_active_pairs + 1) / 2,
      0.0);
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
      packed_active_two_electron_integrals[packed_index] =
          active_pair_matrix(
              static_cast<Eigen::Index>(left_active_pair_index),
              static_cast<Eigen::Index>(right_active_pair_index));
    }
  }
  return packed_active_two_electron_integrals;
}

std::vector<SparseActivePairCoefficient> build_sparse_active_pair_row(
    const std::vector<int>& active_sparse_row_offsets,
    const std::vector<int>& active_sparse_orbital_indices,
    const std::vector<double>& active_sparse_values,
    int first_basis_function,
    int second_basis_function) {
  const int first_begin =
      active_sparse_row_offsets[first_basis_function];
  const int first_end =
      active_sparse_row_offsets[first_basis_function + 1];
  const int second_begin =
      active_sparse_row_offsets[second_basis_function];
  const int second_end =
      active_sparse_row_offsets[second_basis_function + 1];
  const int first_count = first_end - first_begin;
  const int second_count = second_end - second_begin;
  if (first_count <= 0 || second_count <= 0) {
    return {};
  }

  std::vector<SparseActivePairCoefficient> pair_coefficients;
  if (first_basis_function == second_basis_function) {
    pair_coefficients.reserve(
        first_count * (first_count + 1) / 2);
    for (int first_offset = first_begin; first_offset < first_end; ++first_offset) {
      const int first_active_orbital =
          active_sparse_orbital_indices[first_offset];
      const double first_value = active_sparse_values[first_offset];
      for (int second_offset = first_begin; second_offset <= first_offset; ++second_offset) {
        const double coefficient =
            first_value * active_sparse_values[second_offset];
        if (coefficient == 0.0) {
          continue;
        }
        const int second_active_orbital =
            active_sparse_orbital_indices[second_offset];
        pair_coefficients.push_back(
            {TwoElectronIndexer::packed_pair_index(
                 first_active_orbital,
                 second_active_orbital),
             coefficient});
      }
    }
    return pair_coefficients;
  }

  pair_coefficients.reserve(
      first_count * second_count);
  for (int first_offset = first_begin; first_offset < first_end; ++first_offset) {
    const int first_active_orbital =
        active_sparse_orbital_indices[first_offset];
    const double first_value = active_sparse_values[first_offset];
    for (int second_offset = second_begin; second_offset < second_end; ++second_offset) {
      const double coefficient =
          first_value * active_sparse_values[second_offset];
      if (coefficient == 0.0) {
        continue;
      }
      const int second_active_orbital =
          active_sparse_orbital_indices[second_offset];
      pair_coefficients.push_back(
          {TwoElectronIndexer::packed_pair_index(
               first_active_orbital,
               second_active_orbital),
           coefficient});
    }
  }

  if (pair_coefficients.size() <= 1) {
    return pair_coefficients;
  }

  std::sort(
      pair_coefficients.begin(),
      pair_coefficients.end(),
      [](const SparseActivePairCoefficient& left, const SparseActivePairCoefficient& right) {
        return left.packed_active_pair_index < right.packed_active_pair_index;
      });

  std::size_t write_index = 0;
  for (std::size_t read_index = 0; read_index < pair_coefficients.size();) {
    const int packed_active_pair_index =
        pair_coefficients[read_index].packed_active_pair_index;
    double coefficient = 0.0;
    do {
      coefficient += pair_coefficients[read_index].value;
      ++read_index;
    } while (read_index < pair_coefficients.size() &&
             pair_coefficients[read_index].packed_active_pair_index ==
                 packed_active_pair_index);
    if (coefficient == 0.0) {
      continue;
    }
    pair_coefficients[write_index++] = {packed_active_pair_index, coefficient};
  }
  pair_coefficients.resize(write_index);
  return pair_coefficients;
}

SparseAoPairCoefficients build_sparse_ao_pair_coefficients(
    const std::vector<int>& active_sparse_row_offsets,
    const std::vector<int>& active_sparse_orbital_indices,
    const std::vector<double>& active_sparse_values,
    int n_basis_functions,
    int n_active_orbitals) {
  if (active_sparse_row_offsets.size() != n_basis_functions + 1) {
    throw std::invalid_argument("active_sparse_row_offsets size mismatch");
  }
  if (active_sparse_orbital_indices.size() != active_sparse_values.size()) {
    throw std::invalid_argument("active sparse index/value sizes are inconsistent");
  }
  for (std::size_t index = 0; index < active_sparse_orbital_indices.size(); ++index) {
    const int active_orbital_index = active_sparse_orbital_indices[index];
    if (active_orbital_index < 0 || active_orbital_index >= n_active_orbitals) {
      throw std::invalid_argument("active sparse orbital index out of range");
    }
  }

  const std::size_t n_ao_pairs =
      n_basis_functions * (n_basis_functions + 1) / 2;
  std::vector<std::vector<SparseActivePairCoefficient>> pair_rows(n_ao_pairs);

#pragma omp parallel for schedule(static)
  for (int first_basis_function = 0;
       first_basis_function < n_basis_functions;
       ++first_basis_function) {
    for (int second_basis_function = 0;
         second_basis_function <= first_basis_function;
         ++second_basis_function) {
      pair_rows[ao_pair_index(first_basis_function, second_basis_function)] =
          build_sparse_active_pair_row(
              active_sparse_row_offsets,
              active_sparse_orbital_indices,
              active_sparse_values,
              first_basis_function,
              second_basis_function);
    }
  }

  SparseAoPairCoefficients sparse_pair_coefficients;
  sparse_pair_coefficients.row_offsets.resize(n_ao_pairs + 1, 0);
  std::size_t nnz = 0;
  for (std::size_t ao_pair = 0; ao_pair < n_ao_pairs; ++ao_pair) {
    sparse_pair_coefficients.row_offsets[ao_pair] = static_cast<int>(nnz);
    nnz += pair_rows[ao_pair].size();
  }
  sparse_pair_coefficients.row_offsets[n_ao_pairs] = static_cast<int>(nnz);
  sparse_pair_coefficients.packed_active_pair_indices.resize(nnz);
  sparse_pair_coefficients.values.resize(nnz);

#pragma omp parallel for schedule(static)
  for (std::ptrdiff_t ao_pair_offset = 0;
       ao_pair_offset < static_cast<std::ptrdiff_t>(n_ao_pairs);
       ++ao_pair_offset) {
    const std::size_t ao_pair = ao_pair_offset;
    const int begin = sparse_pair_coefficients.row_offsets[ao_pair];
    const auto& source_row = pair_rows[ao_pair];
    for (std::size_t entry_index = 0; entry_index < source_row.size(); ++entry_index) {
      const std::size_t target_index =
          begin + entry_index;
      sparse_pair_coefficients.packed_active_pair_indices[target_index] =
          source_row[entry_index].packed_active_pair_index;
      sparse_pair_coefficients.values[target_index] =
          source_row[entry_index].value;
    }
  }

  return sparse_pair_coefficients;
}

ActiveSpaceTwoElectronResult build_packed_active_two_electron_integrals(
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& ao_two_electron_pair_indices,
    std::vector<double> dense_active_coefficients,
    int n_basis_functions,
    int n_active_orbitals) {
  const auto active_pairs =
      build_active_pair_list(n_active_orbitals);
  const std::size_t n_active_pairs = active_pairs.size();
  const std::size_t n_ao_pairs =
      n_basis_functions * (n_basis_functions + 1) / 2;
  const auto ao_pair_to_active_pair_coefficients =
      build_ao_pair_to_active_pair_coefficients(
          dense_active_coefficients,
          n_basis_functions,
          n_active_orbitals,
          active_pairs);

  auto pair_products =
      apply_sparse_ao_integral_matrix(
          ao_two_electron_integral_values,
          ao_two_electron_pair_indices,
          ao_pair_to_active_pair_coefficients,
          n_basis_functions,
          n_active_pairs);
  auto packed_active_two_electron_integrals =
      contract_pair_coefficients_to_packed_active_integrals(
          ao_pair_to_active_pair_coefficients,
          pair_products,
          n_ao_pairs,
          n_active_pairs);

  ActiveSpaceTwoElectronResult result;
  result.packed_active_two_electron_integrals =
      std::move(packed_active_two_electron_integrals);
  result.dense_active_coefficients =
      copy_legacy_row_buffer_to_matrix(
          dense_active_coefficients,
          n_basis_functions,
          n_active_orbitals);
  result.dense_ao_pair_products =
      copy_legacy_row_buffer_to_matrix(
          pair_products,
          static_cast<int>(n_ao_pairs),
          static_cast<int>(n_active_pairs));
  return result;
}

ActiveSpaceTwoElectronResult build_packed_active_two_electron_integrals_graph(
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& row_offsets,
    const std::vector<int>& column_pair_indices,
    const std::vector<int>& integral_indices,
    std::vector<double> dense_active_coefficients,
    int n_basis_functions,
    int n_active_orbitals) {
  const auto active_pairs =
      build_active_pair_list(n_active_orbitals);
  const std::size_t n_active_pairs = active_pairs.size();
  const std::size_t n_ao_pairs =
      n_basis_functions * (n_basis_functions + 1) / 2;
  const auto ao_pair_to_active_pair_coefficients =
      build_ao_pair_to_active_pair_coefficients(
          dense_active_coefficients,
          n_basis_functions,
          n_active_orbitals,
          active_pairs);

  auto pair_products =
      apply_ao_pair_graph_matrix(
          ao_two_electron_integral_values,
          row_offsets,
          column_pair_indices,
          integral_indices,
          ao_pair_to_active_pair_coefficients,
          n_ao_pairs,
          n_active_pairs);
  auto packed_active_two_electron_integrals =
      contract_pair_coefficients_to_packed_active_integrals(
          ao_pair_to_active_pair_coefficients,
          pair_products,
          n_ao_pairs,
          n_active_pairs);

  ActiveSpaceTwoElectronResult result;
  result.packed_active_two_electron_integrals =
      std::move(packed_active_two_electron_integrals);
  result.dense_active_coefficients =
      copy_legacy_row_buffer_to_matrix(
          dense_active_coefficients,
          n_basis_functions,
          n_active_orbitals);
  result.dense_ao_pair_products =
      copy_legacy_row_buffer_to_matrix(
          pair_products,
          static_cast<int>(n_ao_pairs),
          static_cast<int>(n_active_pairs));
  return result;
}

ActiveSpaceTwoElectronResult build_packed_active_two_electron_integrals_sparse(
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& ao_two_electron_pair_indices,
    const SparseAoPairCoefficients& sparse_pair_coefficients,
    int n_basis_functions,
    int n_active_orbitals) {
  const int last_active_pair_index =
      TwoElectronIndexer::packed_pair_index(
          n_active_orbitals - 1,
          n_active_orbitals - 1);
  const std::size_t packed_size =
      TwoElectronIndexer::packed_pair_of_pairs_index(
          last_active_pair_index,
          last_active_pair_index) +
      1;

  int n_threads = 1;
#ifdef _OPENMP
  n_threads = xmvb::effective_openmp_thread_count();
#endif

  std::vector<std::vector<double>> partial_packed_integrals(
      n_threads,
      std::vector<double>(packed_size, 0.0));

#pragma omp parallel
  {
    int thread_index = 0;
#ifdef _OPENMP
    thread_index = omp_get_thread_num();
#endif
    auto& local_packed_integrals =
        partial_packed_integrals[thread_index];

#pragma omp for schedule(guided, 256)
    for (std::ptrdiff_t integral_offset = 0;
         integral_offset < static_cast<std::ptrdiff_t>(ao_two_electron_integral_values.size());
         ++integral_offset) {
      const std::size_t integral_index = integral_offset;
      const double ao_integral_value = ao_two_electron_integral_values[integral_index];
      const std::size_t left_ao_pair_index =
          ao_two_electron_pair_indices[integral_index * 2];
      const std::size_t right_ao_pair_index =
          ao_two_electron_pair_indices[integral_index * 2 + 1];
      const int left_begin =
          sparse_pair_coefficients.row_offsets[left_ao_pair_index];
      const int left_end =
          sparse_pair_coefficients.row_offsets[left_ao_pair_index + 1];
      const int right_begin =
          sparse_pair_coefficients.row_offsets[right_ao_pair_index];
      const int right_end =
          sparse_pair_coefficients.row_offsets[right_ao_pair_index + 1];
      if (left_begin == left_end || right_begin == right_end) {
        continue;
      }

      if (left_ao_pair_index == right_ao_pair_index) {
        for (int left_offset = left_begin; left_offset < left_end; ++left_offset) {
          const int left_active_pair_index =
              sparse_pair_coefficients.packed_active_pair_indices[
                  left_offset];
          const double left_value =
              sparse_pair_coefficients.values[left_offset];
          for (int right_offset = left_begin; right_offset <= left_offset; ++right_offset) {
            const int right_active_pair_index =
                sparse_pair_coefficients.packed_active_pair_indices[
                    right_offset];
            const double coefficient =
                left_value *
                sparse_pair_coefficients.values[right_offset];
            if (coefficient == 0.0) {
              continue;
            }
            const int packed_index =
                TwoElectronIndexer::packed_pair_of_pairs_index(
                    left_active_pair_index,
                    right_active_pair_index);
            local_packed_integrals[packed_index] +=
                ao_integral_value * coefficient;
          }
        }
        continue;
      }

      for (int left_offset = left_begin; left_offset < left_end; ++left_offset) {
        const int left_active_pair_index =
            sparse_pair_coefficients.packed_active_pair_indices[
                left_offset];
        const double left_value =
            sparse_pair_coefficients.values[left_offset];
        for (int right_offset = right_begin; right_offset < right_end; ++right_offset) {
          const double coefficient =
              left_value *
              sparse_pair_coefficients.values[right_offset];
          if (coefficient == 0.0) {
            continue;
          }
          const int right_active_pair_index =
              sparse_pair_coefficients.packed_active_pair_indices[
                  right_offset];
          const int packed_index =
              TwoElectronIndexer::packed_pair_of_pairs_index(
                  left_active_pair_index,
                  right_active_pair_index);
          local_packed_integrals[packed_index] +=
              ao_integral_value * coefficient;
        }
      }
    }
  }

  std::vector<double> packed_active_two_electron_integrals(packed_size, 0.0);
  for (const auto& partial_integrals : partial_packed_integrals) {
    for (std::size_t integral_index = 0;
         integral_index < packed_active_two_electron_integrals.size();
         ++integral_index) {
      packed_active_two_electron_integrals[integral_index] +=
          partial_integrals[integral_index];
    }
  }

  ActiveSpaceTwoElectronResult result;
  result.packed_active_two_electron_integrals =
      std::move(packed_active_two_electron_integrals);
  return result;
}

}  // namespace

ActiveSpaceTwoElectronResult ActiveSpaceTwoElectronBuilder::build(
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& ao_two_electron_integral_indices,
    const std::vector<double>& auxiliary_orbital_matrix,
    int n_basis_functions,
    int n_inactive_doubly_occupied_orbitals,
    int n_active_orbitals,
    const std::vector<int>* ao_two_electron_pair_indices) const {
  if (n_basis_functions <= 0 || n_active_orbitals <= 0) {
    throw std::invalid_argument("active-space two-electron dimensions must be positive");
  }
  if (ao_two_electron_integral_indices.size() != ao_two_electron_integral_values.size() * 4) {
    throw std::invalid_argument("AO two-electron index/value sizes are inconsistent");
  }
  const std::size_t auxiliary_matrix_size =
      n_basis_functions * n_basis_functions;
  if (auxiliary_orbital_matrix.size() != auxiliary_matrix_size) {
    throw std::invalid_argument("auxiliary orbital matrix size mismatch");
  }
  if (n_inactive_doubly_occupied_orbitals < 0 ||
      n_inactive_doubly_occupied_orbitals + n_active_orbitals > n_basis_functions) {
    throw std::invalid_argument("invalid active-orbital column range");
  }

  const Eigen::Map<const Eigen::MatrixXd> auxiliary_matrix(
      auxiliary_orbital_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  const auto active_auxiliary_orbitals =
      auxiliary_matrix.middleCols(
          n_inactive_doubly_occupied_orbitals,
          n_active_orbitals);
  const auto dense_active_coefficients =
      build_dense_active_coefficients(
          active_auxiliary_orbitals,
          n_basis_functions,
          n_active_orbitals);
  std::vector<int> fallback_pair_indices;
  if (ao_two_electron_pair_indices == nullptr || ao_two_electron_pair_indices->empty()) {
    fallback_pair_indices =
        build_ao_two_electron_pair_indices(
            ao_two_electron_integral_indices,
            n_basis_functions);
    ao_two_electron_pair_indices = &fallback_pair_indices;
  } else if (ao_two_electron_pair_indices->size() !=
             ao_two_electron_integral_values.size() * 2) {
    throw std::invalid_argument("AO two-electron pair index/value sizes are inconsistent");
  }
  return build_packed_active_two_electron_integrals(
      ao_two_electron_integral_values,
      *ao_two_electron_pair_indices,
      std::move(dense_active_coefficients),
      n_basis_functions,
      n_active_orbitals);
}

ActiveSpaceTwoElectronResult ActiveSpaceTwoElectronBuilder::build(
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& ao_two_electron_integral_indices,
    const OrbitalPreparationResult& orbital_preparation_result,
    int n_basis_functions,
    int n_active_orbitals,
    const std::vector<int>* ao_two_electron_pair_indices) const {
  if (n_basis_functions <= 0 || n_active_orbitals <= 0) {
    throw std::invalid_argument("active-space two-electron dimensions must be positive");
  }
  if (ao_two_electron_integral_indices.size() != ao_two_electron_integral_values.size() * 4) {
    throw std::invalid_argument("AO two-electron index/value sizes are inconsistent");
  }
  std::vector<int> fallback_pair_indices;
  if (ao_two_electron_pair_indices == nullptr || ao_two_electron_pair_indices->empty()) {
    fallback_pair_indices =
        build_ao_two_electron_pair_indices(
            ao_two_electron_integral_indices,
            n_basis_functions);
    ao_two_electron_pair_indices = &fallback_pair_indices;
  } else if (ao_two_electron_pair_indices->size() !=
             ao_two_electron_integral_values.size() * 2) {
    throw std::invalid_argument("AO two-electron pair index/value sizes are inconsistent");
  }

  const std::size_t n_active_pairs =
      packed_active_pair_count(n_active_orbitals);
  if (n_active_pairs <= 64) {
    const auto dense_active_coefficients =
        build_dense_active_coefficients(
            orbital_preparation_result,
            n_basis_functions,
            n_active_orbitals);
    return build_packed_active_two_electron_integrals(
        ao_two_electron_integral_values,
        *ao_two_electron_pair_indices,
        std::move(dense_active_coefficients),
        n_basis_functions,
        n_active_orbitals);
  }

  const auto sparse_pair_coefficients =
      build_sparse_ao_pair_coefficients(
          orbital_preparation_result.active_sparse_row_offsets,
          orbital_preparation_result.active_sparse_orbital_indices,
          orbital_preparation_result.active_sparse_values,
          n_basis_functions,
          n_active_orbitals);
  return build_packed_active_two_electron_integrals_sparse(
      ao_two_electron_integral_values,
      *ao_two_electron_pair_indices,
      sparse_pair_coefficients,
      n_basis_functions,
      n_active_orbitals);
}

ActiveSpaceTwoElectronResult ActiveSpaceTwoElectronBuilder::build(
    const AoIntegralInput& ao_integral_input,
    const OrbitalPreparationResult& orbital_preparation_result,
    int n_active_orbitals) const {
  const int n_basis_functions = ao_integral_input.n_basis_functions;
  if (n_basis_functions <= 0 || n_active_orbitals <= 0) {
    throw std::invalid_argument("active-space two-electron dimensions must be positive");
  }
  if (ao_integral_input.ao_two_electron_integral_indices.size() !=
      ao_integral_input.ao_two_electron_integral_values.size() * 4) {
    throw std::invalid_argument("AO two-electron index/value sizes are inconsistent");
  }

  const std::size_t n_active_pairs =
      packed_active_pair_count(n_active_orbitals);
  int n_threads = 1;
#ifdef _OPENMP
  n_threads = xmvb::effective_openmp_thread_count();
#endif
  if (n_threads > 1 &&
      n_active_pairs <= 64 &&
      !ao_integral_input.ao_two_electron_pair_graph_row_offsets.empty() &&
      !ao_integral_input.ao_two_electron_pair_graph_column_indices.empty() &&
      !ao_integral_input.ao_two_electron_pair_graph_integral_indices.empty()) {
    const auto dense_active_coefficients =
        build_dense_active_coefficients(
            orbital_preparation_result,
            n_basis_functions,
            n_active_orbitals);
    return build_packed_active_two_electron_integrals_graph(
        ao_integral_input.ao_two_electron_integral_values,
        ao_integral_input.ao_two_electron_pair_graph_row_offsets,
        ao_integral_input.ao_two_electron_pair_graph_column_indices,
        ao_integral_input.ao_two_electron_pair_graph_integral_indices,
        std::move(dense_active_coefficients),
        n_basis_functions,
        n_active_orbitals);
  }

  const std::vector<int>* ao_two_electron_pair_indices =
      ao_integral_input.ao_two_electron_pair_indices.empty()
          ? nullptr
          : &ao_integral_input.ao_two_electron_pair_indices;
  return build(
      ao_integral_input.ao_two_electron_integral_values,
      ao_integral_input.ao_two_electron_integral_indices,
      orbital_preparation_result,
      n_basis_functions,
      n_active_orbitals,
      ao_two_electron_pair_indices);
}

}  // namespace xmvb::vb
