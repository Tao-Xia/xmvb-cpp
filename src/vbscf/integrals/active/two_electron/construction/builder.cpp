#include "vbscf/integrals/active/two_electron/construction/builder.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "core/openmp.hpp"
#include "vbscf/core/storage/eigen.hpp"
#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"
#include "vbscf/integrals/active/two_electron/transformation/ao_pair_operator.hpp"

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

/**
 * @brief Decides whether dense pair coefficients and products fit the base
 * exact-integral problem storage.
 *
 * The decision is dimension- and representation-based. It contains no
 * molecule-specific active-pair threshold: the two dense pair matrices used
 * during construction may not exceed the storage already required by the AO
 * pair graph and the packed active tensor.
 */
bool retain_dense_pair_products(
    const AoPairGraph& graph,
    std::size_t n_ao_pairs,
    std::size_t n_active_pairs) {
  const long double pair_workspace_bytes =
      2.0L * static_cast<long double>(n_ao_pairs) *
      static_cast<long double>(n_active_pairs) * sizeof(double);
  const long double graph_bytes =
      static_cast<long double>(graph.row_offsets.size()) * sizeof(int) +
      static_cast<long double>(graph.columns.size()) * sizeof(int) +
      static_cast<long double>(graph.values.size()) * sizeof(double);
  const long double packed_active_bytes =
      static_cast<long double>(n_active_pairs) *
      static_cast<long double>(n_active_pairs + 1) * 0.5L * sizeof(double);
  return pair_workspace_bytes <= graph_bytes + packed_active_bytes;
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

std::vector<double> contract_pair_coefficients_to_packed_active_integrals(
    const Eigen::Ref<const ExactCtxPairMatrix>& pair_coefficients,
    const Eigen::Ref<const ExactCtxPairMatrix>& pair_products) {
  if (pair_coefficients.rows() != pair_products.rows() ||
      pair_coefficients.cols() != pair_products.cols()) {
    throw std::invalid_argument(
        "AO-pair coefficient and product shapes differ");
  }
  const Eigen::Index n_active_pairs = pair_coefficients.cols();
  const Eigen::MatrixXd active_pair_matrix =
      pair_coefficients.transpose() * pair_products;
  std::vector<double> packed_active_two_electron_integrals(
      static_cast<std::size_t>(
          n_active_pairs * (n_active_pairs + 1) / 2),
      0.0);
  for (int left = 0; left < n_active_pairs; ++left) {
    for (int right = 0; right <= left; ++right) {
      const int packed_index =
          TwoElectronIndexer::packed_pair_of_pairs_index(
              left,
              right);
      packed_active_two_electron_integrals[packed_index] =
          active_pair_matrix(left, right);
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

ActiveSpaceTwoElectronResult build_packed_active_two_electron_integrals_graph(
    const AoIntegralInput& ao_integral_input,
    std::vector<double> dense_active_coefficients,
    int n_basis_functions,
    int n_active_orbitals) {
  const auto active_pairs =
      build_active_pair_list(n_active_orbitals);
  const std::size_t n_active_pairs = active_pairs.size();
  const std::size_t n_ao_pairs =
      n_basis_functions * (n_basis_functions + 1) / 2;
  const auto pair_coefficient_storage =
      build_ao_pair_to_active_pair_coefficients(
          dense_active_coefficients,
          n_basis_functions,
          n_active_orbitals,
          active_pairs);
  const Eigen::Map<const ExactCtxPairMatrix> pair_coefficients(
      pair_coefficient_storage.data(),
      static_cast<Eigen::Index>(n_ao_pairs),
      static_cast<Eigen::Index>(n_active_pairs));
  ExactCtxPairMatrix pair_products;
  detail::apply_exact_ao_pair_kernel(
      ao_integral_input,
      pair_coefficients,
      n_basis_functions,
      n_active_pairs,
      &pair_products);
  auto packed_active_two_electron_integrals =
      contract_pair_coefficients_to_packed_active_integrals(
          pair_coefficients,
          pair_products);

  ActiveSpaceTwoElectronResult result;
  result.packed_active_two_electron_integrals =
      std::move(packed_active_two_electron_integrals);
  result.dense_active_coefficients =
      copy_row_major_buffer_to_matrix(
          dense_active_coefficients,
          n_basis_functions,
          n_active_orbitals);
  result.dense_ao_pair_products = pair_products;
  return result;
}

ActiveSpaceTwoElectronResult build_packed_active_two_electron_integrals_sparse(
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& ao_two_electron_integral_indices,
    const SparseAoPairCoefficients& sparse_pair_coefficients,
    Eigen::MatrixXd dense_active_coefficients,
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

#pragma omp parallel num_threads(n_threads)
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
      const int* eri =
          ao_two_electron_integral_indices.data() + 4 * integral_index;
      const std::size_t left_ao_pair_index =
          ao_pair_index(eri[0], eri[1]);
      const std::size_t right_ao_pair_index =
          ao_pair_index(eri[2], eri[3]);
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
  result.dense_active_coefficients =
      std::move(dense_active_coefficients);
  return result;
}

}  // namespace

ActiveSpaceTwoElectronResult ActiveSpaceTwoElectronBuilder::build(
    const AoIntegralInput& ao_integral_input,
    const OrbitalPreparationResult& orbital_preparation_result,
    int n_active_orbitals) const {
  const int n_bf = ao_integral_input.n_basis_functions;
  if (n_bf <= 0 || n_active_orbitals <= 0) {
    throw std::invalid_argument("active-space two-electron dimensions must be positive");
  }
  if (ao_integral_input.ao_two_electron_integral_indices.size() !=
      ao_integral_input.ao_two_electron_integral_values.size() * 4) {
    throw std::invalid_argument("AO two-electron index/value sizes are inconsistent");
  }

  const std::size_t n_active_pairs =
      packed_active_pair_count(n_active_orbitals);
  const std::size_t n_ao_pairs =
      static_cast<std::size_t>(n_bf) * (n_bf + 1) / 2;
  if (retain_dense_pair_products(
          ao_integral_input.pair_graph,
          n_ao_pairs,
          n_active_pairs)) {
    const auto dense_active_coefficients =
        build_dense_active_coefficients(
            orbital_preparation_result,
            n_bf,
            n_active_orbitals);
    return build_packed_active_two_electron_integrals_graph(
        ao_integral_input,
        std::move(dense_active_coefficients),
        n_bf,
        n_active_orbitals);
  }

  const auto sparse_pair_coefficients =
      build_sparse_ao_pair_coefficients(
          orbital_preparation_result.active_sparse_row_offsets,
          orbital_preparation_result.active_sparse_orbital_indices,
          orbital_preparation_result.active_sparse_values,
          n_bf,
          n_active_orbitals);
  auto dense_active_coefficients =
      copy_row_major_buffer_to_matrix(
          build_dense_active_coefficients(
              orbital_preparation_result,
              n_bf,
              n_active_orbitals),
          n_bf,
          n_active_orbitals);
  return build_packed_active_two_electron_integrals_sparse(
      ao_integral_input.ao_two_electron_integral_values,
      ao_integral_input.ao_two_electron_integral_indices,
      sparse_pair_coefficients,
      std::move(dense_active_coefficients),
      n_bf,
      n_active_orbitals);
}

}  // namespace xmvb::vb
