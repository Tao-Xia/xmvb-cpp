#include "vb/orbital/active_space_two_electron_backpropagator.hpp"

#include <atomic>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "core/openmp_utils.hpp"
#include "vb/matrices/eigen_matrix_storage_utils.hpp"
#include "vb/matrices/cpp_vb_input_ri_cache.hpp"
#include "vb/matrices/two_electron_indexer.hpp"

namespace xmvb::vb {

namespace {

struct ActivePair {
  int first = 0;
  int second = 0;
};

std::size_t ao_pair_index(int first, int second) {
  if (first >= second) {
    return first * (first + 1) / 2 + second;
  }
  return second * (second + 1) / 2 + first;
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

std::vector<double> build_dense_active_coefficients(
    const Eigen::Ref<const Eigen::MatrixXd>& active_auxiliary_orbitals,
    int n_basis_functions,
    int n_active_orbitals) {
  std::vector<double> dense_active_coefficients(
      n_basis_functions * n_active_orbitals,
      0.0);

#pragma omp parallel for schedule(static)
  for (int basis_function_index = 0;
       basis_function_index < n_basis_functions;
       ++basis_function_index) {
    double* basis_coefficients =
        dense_active_coefficients.data() +
        basis_function_index * n_active_orbitals;
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
  if (orbital_preparation_result.active_sparse_row_offsets.size() !=
      n_basis_functions + 1) {
    throw std::invalid_argument("active_sparse_row_offsets size mismatch");
  }
  if (orbital_preparation_result.active_sparse_orbital_indices.size() !=
      orbital_preparation_result.active_sparse_values.size()) {
    throw std::invalid_argument("active sparse index/value sizes are inconsistent");
  }

  std::vector<double> dense_active_coefficients(
      n_basis_functions * n_active_orbitals,
      0.0);

#pragma omp parallel for schedule(static)
  for (int basis_function_index = 0;
       basis_function_index < n_basis_functions;
       ++basis_function_index) {
    const int begin =
        orbital_preparation_result.active_sparse_row_offsets[
            basis_function_index];
    const int end =
        orbital_preparation_result.active_sparse_row_offsets[
            basis_function_index + 1];
    double* basis_coefficients =
        dense_active_coefficients.data() +
        basis_function_index * n_active_orbitals;
    for (int offset = begin; offset < end; ++offset) {
      const int active_orbital_index =
          orbital_preparation_result.active_sparse_orbital_indices[
              offset];
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
  const std::size_t n_ao_pairs =
      n_basis_functions * (n_basis_functions + 1) / 2;
  const std::size_t n_active_pairs = active_pairs.size();
  std::vector<double> ao_pair_to_active_pair_coefficients(
      n_ao_pairs * n_active_pairs,
      0.0);

#pragma omp parallel for schedule(static)
  for (int first_basis_function = 0;
       first_basis_function < n_basis_functions;
       ++first_basis_function) {
    const double* first_coefficients =
        dense_active_coefficients.data() +
        first_basis_function * n_active_orbitals;
    for (int second_basis_function = 0;
         second_basis_function <= first_basis_function;
         ++second_basis_function) {
      const double* second_coefficients =
          dense_active_coefficients.data() +
          second_basis_function * n_active_orbitals;
      double* pair_coefficients =
          ao_pair_to_active_pair_coefficients.data() +
          ao_pair_index(first_basis_function, second_basis_function) * n_active_pairs;
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

Eigen::MatrixXd build_active_pair_gradient_matrix(
    const std::vector<double>& packed_active_two_electron_gradient,
    const std::vector<ActivePair>& active_pairs) {
  const std::size_t n_active_pairs = active_pairs.size();
  Eigen::MatrixXd active_pair_gradient_matrix =
      Eigen::MatrixXd::Zero(
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

std::vector<double> multiply_pair_coefficients_by_gradient_matrix(
    const std::vector<double>& ao_pair_to_active_pair_coefficients,
    const Eigen::Ref<const Eigen::MatrixXd>& active_pair_gradient_matrix,
    std::size_t n_ao_pairs,
    std::size_t n_active_pairs) {
  std::vector<double> transformed_pair_coefficients(
      n_ao_pairs * n_active_pairs,
      0.0);
  if (active_pair_gradient_matrix.rows() != static_cast<Eigen::Index>(n_active_pairs) ||
      active_pair_gradient_matrix.cols() != static_cast<Eigen::Index>(n_active_pairs)) {
    throw std::invalid_argument("active-pair gradient matrix shape mismatch");
  }

#pragma omp parallel for schedule(static)
  for (std::ptrdiff_t ao_pair_offset = 0;
       ao_pair_offset < static_cast<std::ptrdiff_t>(n_ao_pairs);
       ++ao_pair_offset) {
    const std::size_t ao_pair_index = ao_pair_offset;
    const Eigen::Map<const Eigen::VectorXd> pair_coefficient_row(
        ao_pair_to_active_pair_coefficients.data() +
            ao_pair_index * n_active_pairs,
        static_cast<Eigen::Index>(n_active_pairs));
    Eigen::Map<Eigen::VectorXd> transformed_pair_coefficient_row(
        transformed_pair_coefficients.data() +
            ao_pair_index * n_active_pairs,
        static_cast<Eigen::Index>(n_active_pairs));
    transformed_pair_coefficient_row.noalias() =
        active_pair_gradient_matrix.transpose() * pair_coefficient_row;
  }

  return transformed_pair_coefficients;
}

std::vector<double> apply_sparse_ao_integral_matrix(
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& ao_two_electron_integral_indices,
    const std::vector<double>& transformed_pair_coefficients,
    int n_basis_functions,
    std::size_t n_active_pairs) {
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
         integral_offset < static_cast<std::ptrdiff_t>(ao_two_electron_integral_values.size());
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

  std::vector<double> pair_gradients(n_ao_pairs * n_active_pairs, 0.0);
  for (const auto& partial_pair_gradient : partial_pair_gradients) {
    for (std::size_t index = 0; index < pair_gradients.size(); ++index) {
      pair_gradients[index] += partial_pair_gradient[index];
    }
  }

  return pair_gradients;
}

std::vector<double> backpropagate_pair_coefficients_to_dense_active_coefficients(
    const std::vector<double>& pair_gradients,
    const std::vector<double>& dense_active_coefficients,
    int n_basis_functions,
    int n_active_orbitals,
    const std::vector<ActivePair>& active_pairs) {
  std::vector<double> dense_active_gradients(
      n_basis_functions * n_active_orbitals,
      0.0);

  const std::size_t n_active_pairs = active_pairs.size();
  const std::size_t n_ao_pairs =
      n_basis_functions * (n_basis_functions + 1) / 2;
  if (pair_gradients.size() != n_ao_pairs * n_active_pairs) {
    throw std::invalid_argument("pair gradient size mismatch");
  }
  if (dense_active_coefficients.size() !=
      n_basis_functions * n_active_orbitals) {
    throw std::invalid_argument("dense active coefficient size mismatch");
  }

#pragma omp parallel for schedule(static)
  for (int basis_function_index = 0;
       basis_function_index < n_basis_functions;
       ++basis_function_index) {
    double* gradient_row =
        dense_active_gradients.data() +
        basis_function_index * n_active_orbitals;
    for (int other_basis_function = 0;
         other_basis_function < n_basis_functions;
         ++other_basis_function) {
      const double* other_coefficients =
          dense_active_coefficients.data() +
          other_basis_function * n_active_orbitals;
      const double* pair_gradient_row =
          pair_gradients.data() +
          ao_pair_index(basis_function_index, other_basis_function) *
              n_active_pairs;

      for (std::size_t active_pair_index = 0;
           active_pair_index < n_active_pairs;
           ++active_pair_index) {
        const double pair_gradient = pair_gradient_row[active_pair_index];
        if (pair_gradient == 0.0) {
          continue;
        }

        const auto& active_pair = active_pairs[active_pair_index];
        gradient_row[active_pair.first] +=
            pair_gradient * other_coefficients[active_pair.second];
        gradient_row[active_pair.second] +=
            pair_gradient * other_coefficients[active_pair.first];
      }
    }
  }

  return dense_active_gradients;
}

Eigen::MatrixXd build_active_auxiliary_gradient_matrix(
    const std::vector<double>& dense_active_gradients,
    int n_basis_functions,
    int n_active_orbitals) {
  if (dense_active_gradients.size() !=
      n_basis_functions * n_active_orbitals) {
    throw std::invalid_argument("dense active gradient size mismatch");
  }

  Eigen::MatrixXd active_auxiliary_gradient =
      Eigen::MatrixXd::Zero(n_basis_functions, n_active_orbitals);
#pragma omp parallel for schedule(static)
  for (int active_orbital_index = 0;
       active_orbital_index < n_active_orbitals;
       ++active_orbital_index) {
    for (int basis_function_index = 0;
         basis_function_index < n_basis_functions;
         ++basis_function_index) {
      active_auxiliary_gradient(
          basis_function_index,
          active_orbital_index) =
          dense_active_gradients[basis_function_index *
                                     n_active_orbitals +
                                 active_orbital_index];
    }
  }

  return active_auxiliary_gradient;
}

ActiveSpaceTwoElectronBackpropagationResult backpropagate_packed_active_two_electron_integrals(
    const std::vector<double>& packed_active_two_electron_gradient,
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& ao_two_electron_integral_indices,
    const std::vector<double>& dense_active_coefficients,
    const std::vector<double>* cached_dense_ao_pair_products,
    int n_basis_functions,
    int n_inactive_doubly_occupied_orbitals,
    int n_active_orbitals) {
  const auto active_pairs =
      build_active_pair_list(n_active_orbitals);
  const std::size_t n_active_pairs = active_pairs.size();
  const std::size_t n_ao_pairs =
      n_basis_functions * (n_basis_functions + 1) / 2;
  const auto active_pair_gradient_matrix =
      build_active_pair_gradient_matrix(
          packed_active_two_electron_gradient,
          active_pairs);
  std::vector<double> pair_gradients;
  if (cached_dense_ao_pair_products != nullptr &&
      !cached_dense_ao_pair_products->empty()) {
    const std::size_t expected_cache_size = n_ao_pairs * n_active_pairs;
    if (cached_dense_ao_pair_products->size() != expected_cache_size) {
      throw std::runtime_error("dense AO pair product cache size mismatch");
    }
    pair_gradients =
        multiply_pair_coefficients_by_gradient_matrix(
            *cached_dense_ao_pair_products,
            active_pair_gradient_matrix,
            n_ao_pairs,
            n_active_pairs);
  } else {
    const auto ao_pair_to_active_pair_coefficients =
        build_ao_pair_to_active_pair_coefficients(
            dense_active_coefficients,
            n_basis_functions,
            n_active_orbitals,
            active_pairs);
    const auto transformed_pair_coefficients =
        multiply_pair_coefficients_by_gradient_matrix(
            ao_pair_to_active_pair_coefficients,
            active_pair_gradient_matrix,
            n_ao_pairs,
            n_active_pairs);
    pair_gradients =
        apply_sparse_ao_integral_matrix(
            ao_two_electron_integral_values,
            ao_two_electron_integral_indices,
            transformed_pair_coefficients,
            n_basis_functions,
            n_active_pairs);
  }
  const auto dense_active_gradients =
      backpropagate_pair_coefficients_to_dense_active_coefficients(
          pair_gradients,
          dense_active_coefficients,
          n_basis_functions,
          n_active_orbitals,
          active_pairs);

  ActiveSpaceTwoElectronBackpropagationResult result;
  result.active_auxiliary_orbital_gradient =
      build_active_auxiliary_gradient_matrix(
          dense_active_gradients,
          n_basis_functions,
          n_active_orbitals);
  return result;
}

ActiveSpaceTwoElectronBackpropagationResult
backpropagate_ri_active_pair_factors(
    const std::vector<double>& ri_active_pair_factor_gradient,
    const LibcintRiIntegralProviderResult& ao_ri_result,
    const std::vector<double>& dense_active_coefficients,
    int n_basis_functions,
    int n_inactive_doubly_occupied_orbitals,
    int n_active_orbitals) {
  if (ao_ri_result.n_basis_functions != n_basis_functions) {
    throw std::invalid_argument("AO RI cache basis-function count mismatch");
  }

  const std::size_t n_active_pairs =
      n_active_orbitals * (n_active_orbitals + 1) / 2;
  const std::size_t n_ao_pairs =
      n_basis_functions * (n_basis_functions + 1) / 2;
  const std::size_t expected_gradient_size =
      ao_ri_result.n_auxiliary_functions * n_active_pairs;
  if (ri_active_pair_factor_gradient.size() != expected_gradient_size) {
    throw std::invalid_argument("RI active-pair-factor gradient size mismatch");
  }
  if (ao_ri_result.metric_whitened_ao_pair_factors.rows() !=
          ao_ri_result.n_auxiliary_functions ||
      ao_ri_result.metric_whitened_ao_pair_factors.cols() !=
          static_cast<Eigen::Index>(n_ao_pairs)) {
    throw std::invalid_argument("AO RI factor matrix shape mismatch");
  }

  const Eigen::Map<const Eigen::MatrixXd> active_pair_factor_gradient_matrix(
      ri_active_pair_factor_gradient.data(),
      ao_ri_result.n_auxiliary_functions,
      static_cast<Eigen::Index>(n_active_pairs));
  Eigen::MatrixXd ao_pair_transform_gradient(
      static_cast<Eigen::Index>(n_ao_pairs),
      static_cast<Eigen::Index>(n_active_pairs));
  ao_pair_transform_gradient.noalias() =
      ao_ri_result.metric_whitened_ao_pair_factors.transpose() *
      active_pair_factor_gradient_matrix;
  const std::vector<double> pair_gradients =
      copy_matrix_to_legacy_row_buffer(ao_pair_transform_gradient);

  const auto active_pairs =
      build_active_pair_list(n_active_orbitals);
  const auto dense_active_gradients =
      backpropagate_pair_coefficients_to_dense_active_coefficients(
          pair_gradients,
          dense_active_coefficients,
          n_basis_functions,
          n_active_orbitals,
          active_pairs);

  ActiveSpaceTwoElectronBackpropagationResult result;
  result.active_auxiliary_orbital_gradient =
      build_active_auxiliary_gradient_matrix(
          dense_active_gradients,
          n_basis_functions,
          n_active_orbitals);
  return result;
}

}  // namespace

ActiveSpaceTwoElectronBackpropagationResult
ActiveSpaceTwoElectronBackpropagator::backpropagate(
    const std::vector<double>& ri_active_pair_factor_gradient,
    const CppVbInput& input,
    const OrbitalPreparationResult& orbital_preparation_result,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    int n_basis_functions,
    int n_inactive_doubly_occupied_orbitals,
    int n_active_orbitals) const {
  if (n_basis_functions <= 0 || n_active_orbitals <= 0) {
    throw std::invalid_argument("active-space RI backprop dimensions must be positive");
  }
  if (active_space_two_electron_result.representation !=
      ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity) {
    throw std::invalid_argument("RI backpropagate() requires an RI active-space result");
  }
  if (n_inactive_doubly_occupied_orbitals < 0 ||
      n_inactive_doubly_occupied_orbitals + n_active_orbitals > n_basis_functions) {
    throw std::invalid_argument("invalid active-orbital column range");
  }

  const std::vector<double>* dense_active_coefficients = nullptr;
  std::vector<double> cached_dense_active_coefficients;
  std::vector<double> fallback_dense_active_coefficients;
  if (active_space_two_electron_result.dense_active_coefficients.size() == 0) {
    fallback_dense_active_coefficients =
        build_dense_active_coefficients(
            orbital_preparation_result,
            n_basis_functions,
            n_active_orbitals);
    dense_active_coefficients = &fallback_dense_active_coefficients;
  } else {
    cached_dense_active_coefficients =
        copy_matrix_to_legacy_row_buffer(
            active_space_two_electron_result.dense_active_coefficients);
    dense_active_coefficients = &cached_dense_active_coefficients;
  }

  const auto& ao_ri_result = ensure_cpp_vb_input_ri_cache(input);
  return backpropagate_ri_active_pair_factors(
      ri_active_pair_factor_gradient,
      ao_ri_result,
      *dense_active_coefficients,
      n_basis_functions,
      n_inactive_doubly_occupied_orbitals,
      n_active_orbitals);
}

ActiveSpaceTwoElectronBackpropagationResult
ActiveSpaceTwoElectronBackpropagator::backpropagate(
    const std::vector<double>& packed_active_two_electron_gradient,
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& ao_two_electron_integral_indices,
    const std::vector<double>& auxiliary_orbital_matrix,
    int n_basis_functions,
    int n_inactive_doubly_occupied_orbitals,
    int n_active_orbitals) const {
  if (n_basis_functions <= 0 || n_active_orbitals <= 0) {
    throw std::invalid_argument("active-space two-electron backprop dimensions must be positive");
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
  return backpropagate_packed_active_two_electron_integrals(
      packed_active_two_electron_gradient,
      ao_two_electron_integral_values,
      ao_two_electron_integral_indices,
      dense_active_coefficients,
      nullptr,
      n_basis_functions,
      n_inactive_doubly_occupied_orbitals,
      n_active_orbitals);
}

ActiveSpaceTwoElectronBackpropagationResult
ActiveSpaceTwoElectronBackpropagator::backpropagate(
    const std::vector<double>& packed_active_two_electron_gradient,
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& ao_two_electron_integral_indices,
    const OrbitalPreparationResult& orbital_preparation_result,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    int n_basis_functions,
    int n_inactive_doubly_occupied_orbitals,
    int n_active_orbitals) const {
  if (n_basis_functions <= 0 || n_active_orbitals <= 0) {
    throw std::invalid_argument("active-space two-electron backprop dimensions must be positive");
  }
  if (ao_two_electron_integral_indices.size() != ao_two_electron_integral_values.size() * 4) {
    throw std::invalid_argument("AO two-electron index/value sizes are inconsistent");
  }
  if (n_inactive_doubly_occupied_orbitals < 0 ||
      n_inactive_doubly_occupied_orbitals + n_active_orbitals > n_basis_functions) {
    throw std::invalid_argument("invalid active-orbital column range");
  }

  const std::vector<double>* dense_active_coefficients = nullptr;
  std::vector<double> cached_dense_active_coefficients;
  std::vector<double> fallback_dense_active_coefficients;
  if (active_space_two_electron_result.dense_active_coefficients.size() == 0) {
    fallback_dense_active_coefficients =
        build_dense_active_coefficients(
            orbital_preparation_result,
            n_basis_functions,
            n_active_orbitals);
    dense_active_coefficients = &fallback_dense_active_coefficients;
  } else {
    cached_dense_active_coefficients =
        copy_matrix_to_legacy_row_buffer(
            active_space_two_electron_result.dense_active_coefficients);
    dense_active_coefficients = &cached_dense_active_coefficients;
  }
  const std::vector<double>* cached_dense_ao_pair_products = nullptr;
  std::vector<double> cached_dense_ao_pair_products_storage;
  if (active_space_two_electron_result.dense_ao_pair_products.size() != 0) {
    cached_dense_ao_pair_products_storage =
        copy_matrix_to_legacy_row_buffer(
            active_space_two_electron_result.dense_ao_pair_products);
    cached_dense_ao_pair_products = &cached_dense_ao_pair_products_storage;
  }
  return backpropagate_packed_active_two_electron_integrals(
      packed_active_two_electron_gradient,
      ao_two_electron_integral_values,
      ao_two_electron_integral_indices,
      *dense_active_coefficients,
      cached_dense_ao_pair_products,
      n_basis_functions,
      n_inactive_doubly_occupied_orbitals,
      n_active_orbitals);
}

ActiveSpaceTwoElectronBackpropagationResult
ActiveSpaceTwoElectronBackpropagator::backpropagate(
    const std::vector<double>& packed_active_two_electron_gradient,
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& ao_two_electron_integral_indices,
    const OrbitalPreparationResult& orbital_preparation_result,
    int n_basis_functions,
    int n_inactive_doubly_occupied_orbitals,
    int n_active_orbitals) const {
  if (n_basis_functions <= 0 || n_active_orbitals <= 0) {
    throw std::invalid_argument("active-space two-electron backprop dimensions must be positive");
  }
  if (ao_two_electron_integral_indices.size() != ao_two_electron_integral_values.size() * 4) {
    throw std::invalid_argument("AO two-electron index/value sizes are inconsistent");
  }
  if (n_inactive_doubly_occupied_orbitals < 0 ||
      n_inactive_doubly_occupied_orbitals + n_active_orbitals > n_basis_functions) {
    throw std::invalid_argument("invalid active-orbital column range");
  }

  const auto dense_active_coefficients =
      build_dense_active_coefficients(
          orbital_preparation_result,
          n_basis_functions,
          n_active_orbitals);
  return backpropagate_packed_active_two_electron_integrals(
      packed_active_two_electron_gradient,
      ao_two_electron_integral_values,
      ao_two_electron_integral_indices,
      dense_active_coefficients,
      nullptr,
      n_basis_functions,
      n_inactive_doubly_occupied_orbitals,
      n_active_orbitals);
}

}  // namespace xmvb::vb
