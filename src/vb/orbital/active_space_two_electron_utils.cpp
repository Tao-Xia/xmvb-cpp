#include "vb/orbital/active_space_two_electron_utils.hpp"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <cblas.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "vb/matrices/eigen_matrix_storage_utils.hpp"
#include "vb/matrices/two_electron_indexer.hpp"

namespace xmvb::vb {

namespace {

constexpr std::size_t kMaxAcceptedBasePairGradientMatrixCacheBytes =
    64u * 1024u * 1024u;

struct ActivePair {
  int first = 0;
  int second = 0;
};

std::size_t ao_pair_index(int first, int second) {
  if (first >= second) {
    return xmvb::to_size(first) * (first + 1) / 2 + second;
  }
  return xmvb::to_size(second) * (second + 1) / 2 + first;
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
  const std::size_t n_ao_pairs =
      xmvb::to_size(n_basis_functions) * (n_basis_functions + 1) / 2;
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

void resize_and_zero(
    std::vector<double>* values,
    std::size_t size) {
  if (values == nullptr) {
    throw std::invalid_argument("workspace buffer must not be null");
  }
  values->assign(size, 0.0);
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

void copy_matrix_to_legacy_row_buffer_inplace(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix,
    std::vector<double>* values) {
  if (values == nullptr) {
    throw std::invalid_argument("legacy row-buffer output must not be null");
  }
  values->resize(
      xmvb::to_size(matrix.rows()) * xmvb::to_size(matrix.cols()));
  for (int row = 0; row < matrix.rows(); ++row) {
    double* target_row =
        values->data() + xmvb::to_size(row) * xmvb::to_size(matrix.cols());
    for (int col = 0; col < matrix.cols(); ++col) {
      target_row[col] = matrix(row, col);
    }
  }
}

void copy_legacy_row_buffer_to_matrix_inplace(
    const std::vector<double>& values,
    int n_rows,
    int n_cols,
    Eigen::MatrixXd* matrix) {
  if (matrix == nullptr) {
    throw std::invalid_argument("dense matrix output must not be null");
  }
  if (n_rows < 0 || n_cols < 0) {
    throw std::invalid_argument("matrix dimensions must be non-negative");
  }
  if (values.size() != xmvb::to_size(n_rows) * xmvb::to_size(n_cols)) {
    throw std::invalid_argument("legacy row buffer size does not match matrix dimensions");
  }
  matrix->resize(n_rows, n_cols);
  for (int row = 0; row < n_rows; ++row) {
    const double* source_row =
        values.data() + xmvb::to_size(row) * xmvb::to_size(n_cols);
    for (int col = 0; col < n_cols; ++col) {
      (*matrix)(row, col) = source_row[col];
    }
  }
}

bool parse_env_flag_with_default(
    const char* variable_name,
    bool default_value) {
  const char* value = std::getenv(variable_name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }
  return std::strcmp(value, "0") != 0 &&
      std::strcmp(value, "false") != 0 &&
      std::strcmp(value, "FALSE") != 0;
}

int parse_env_int_with_default(
    const char* variable_name,
    int default_value) {
  const char* value = std::getenv(variable_name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || (end != nullptr && *end != '\0') || parsed <= 0 ||
      parsed > std::numeric_limits<int>::max()) {
    return default_value;
  }
  return static_cast<int>(parsed);
}

bool exact_2e_fused_hvp_enabled() {
  return parse_env_flag_with_default(
      "XMVB_CPP_ENABLE_EXACT_2E_FUSED_HVP",
      true);
}

bool exact_2e_fused_hvp_supported_active_space(
    int n_active_orbitals) {
  // The fused single-thread kernels are specialized for the small active-space
  // regime that dominates the current TN-HVP workloads. Larger active spaces
  // automatically fall back to the materialized pair-gradient path.
  return n_active_orbitals > 0 && n_active_orbitals <= 10;
}

bool exact_2e_fused_hvp_supported_integral_layout(
    const AoIntegralInput& ao_integral_input) {
  return !ao_integral_input.ao_two_electron_pair_graph_row_offsets.empty() ||
      !ao_integral_input.ao_two_electron_pair_indices.empty();
}

bool exact_2e_fused_hvp_applicable(
    const AoIntegralInput& ao_integral_input,
    int n_active_orbitals) {
  if (!exact_2e_fused_hvp_enabled() ||
      !exact_2e_fused_hvp_supported_active_space(n_active_orbitals) ||
      !exact_2e_fused_hvp_supported_integral_layout(ao_integral_input)) {
    return false;
  }
  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif
  return n_threads <= 1;
}

int exact_2e_cached_row_direct_max_threads(
    int n_active_orbitals) {
  int default_value = 1;
  if (n_active_orbitals >= 9) {
    default_value = 2;
  }
  return parse_env_int_with_default(
      "XMVB_CPP_EXACT_2E_CACHED_ROW_DIRECT_MAX_THREADS",
      default_value);
}

bool should_cache_accepted_base_pair_gradient_matrices(
    std::size_t n_ao_pairs,
    int n_active_orbitals) {
  if (n_active_orbitals <= 0) {
    return false;
  }
  const std::size_t n_active_square =
      xmvb::to_size(n_active_orbitals) * n_active_orbitals;
  if (n_active_square == 0) {
    return false;
  }
  if (n_ao_pairs > std::numeric_limits<std::size_t>::max() / n_active_square) {
    return false;
  }
  const std::size_t element_count = n_ao_pairs * n_active_square;
  if (element_count >
      std::numeric_limits<std::size_t>::max() / sizeof(double)) {
    return false;
  }
  return element_count * sizeof(double) <=
      kMaxAcceptedBasePairGradientMatrixCacheBytes;
}

std::vector<double> build_full_active_pair_gradient_matrices_from_cache(
    const std::vector<double>& packed_pair_gradients,
    const ExactPackedActiveTwoElectronAdjointCache& cache) {
  const int n_basis_functions = cache.n_basis_functions;
  const int n_active_orbitals = cache.n_active_orbitals;
  const std::size_t n_active_pairs = cache.active_pair_first_indices.size();
  if (cache.active_pair_second_indices.size() != n_active_pairs) {
    throw std::invalid_argument("exact 2e cache active-pair index size mismatch");
  }
  const std::size_t n_ao_pairs =
      xmvb::to_size(n_basis_functions) * (n_basis_functions + 1) / 2;
  if (packed_pair_gradients.size() != n_ao_pairs * n_active_pairs) {
    throw std::invalid_argument("packed pair gradient size mismatch");
  }
  const std::size_t n_active_square =
      xmvb::to_size(n_active_orbitals) * n_active_orbitals;
  std::vector<double> full_pair_gradient_matrices(
      n_ao_pairs * n_active_square,
      0.0);

#pragma omp parallel for schedule(static)
  for (std::ptrdiff_t ao_pair_offset = 0;
       ao_pair_offset < static_cast<std::ptrdiff_t>(n_ao_pairs);
       ++ao_pair_offset) {
    const std::size_t pair_index = xmvb::to_size(ao_pair_offset);
    const double* packed_row =
        packed_pair_gradients.data() + pair_index * n_active_pairs;
    double* full_matrix =
        full_pair_gradient_matrices.data() + pair_index * n_active_square;
    for (std::size_t active_pair_index = 0;
         active_pair_index < n_active_pairs;
         ++active_pair_index) {
      const int first_active =
          cache.active_pair_first_indices[active_pair_index];
      const int second_active =
          cache.active_pair_second_indices[active_pair_index];
      const double packed_gradient = packed_row[active_pair_index];
      full_matrix[xmvb::to_size(first_active) * n_active_orbitals + second_active] +=
          packed_gradient;
      full_matrix[xmvb::to_size(second_active) * n_active_orbitals + first_active] +=
          packed_gradient;
    }
  }

  return full_pair_gradient_matrices;
}

template <int NActiveOrbitals>
inline void accumulate_active_orbital_matrix_vector_product(
    double* target_row,
    const double* matrix_row_major,
    const double* source_row) {
  for (int row_index = 0; row_index < NActiveOrbitals; ++row_index) {
    const double* matrix_row =
        matrix_row_major + xmvb::to_size(row_index) * NActiveOrbitals;
    double row_value = 0.0;
#pragma omp simd reduction(+ : row_value)
    for (int column_index = 0; column_index < NActiveOrbitals; ++column_index) {
      row_value += matrix_row[column_index] * source_row[column_index];
    }
    target_row[row_index] += row_value;
  }
}

template <int NActiveOrbitals>
inline void accumulate_packed_active_pair_gradient_row_to_active_gradient(
    double* target_row,
    const double* packed_pair_gradient_row,
    const double* source_row) {
  int packed_pair_index = 0;
  for (int first_active = 0;
       first_active < NActiveOrbitals;
       ++first_active) {
    for (int second_active = 0;
         second_active < first_active;
         ++second_active) {
      const double pair_gradient =
          packed_pair_gradient_row[packed_pair_index++];
      if (pair_gradient == 0.0) {
        continue;
      }
      target_row[first_active] +=
          pair_gradient * source_row[second_active];
      target_row[second_active] +=
          pair_gradient * source_row[first_active];
    }
    const double diagonal_pair_gradient =
        packed_pair_gradient_row[packed_pair_index++];
    if (diagonal_pair_gradient != 0.0) {
      target_row[first_active] +=
          2.0 * diagonal_pair_gradient * source_row[first_active];
    }
  }
}

template <int NActiveOrbitals>
inline void accumulate_scaled_packed_active_pair_gradient_row_to_active_gradient(
    double* target_row,
    const double* packed_pair_gradient_row,
    const double* source_row,
    double scale) {
  if (scale == 0.0) {
    return;
  }
  int packed_pair_index = 0;
  for (int first_active = 0;
       first_active < NActiveOrbitals;
       ++first_active) {
    for (int second_active = 0;
         second_active < first_active;
         ++second_active) {
      const double pair_gradient =
          scale * packed_pair_gradient_row[packed_pair_index++];
      if (pair_gradient == 0.0) {
        continue;
      }
      target_row[first_active] +=
          pair_gradient * source_row[second_active];
      target_row[second_active] +=
          pair_gradient * source_row[first_active];
    }
    const double diagonal_pair_gradient =
        scale * packed_pair_gradient_row[packed_pair_index++];
    if (diagonal_pair_gradient != 0.0) {
      target_row[first_active] +=
          2.0 * diagonal_pair_gradient * source_row[first_active];
    }
  }
}

template <int NActiveOrbitals>
inline void accumulate_scaled_active_gradient_row(
    double* target_row,
    const double* source_row,
    double scale) {
  if (scale == 0.0) {
    return;
  }
#pragma omp simd
  for (int active_orbital_index = 0;
       active_orbital_index < NActiveOrbitals;
       ++active_orbital_index) {
    target_row[active_orbital_index] +=
        scale * source_row[active_orbital_index];
  }
}

inline void accumulate_scaled_active_gradient_row_dynamic(
    double* target_row,
    const double* source_row,
    int n_active_orbitals,
    double scale) {
  if (scale == 0.0) {
    return;
  }
#pragma omp simd
  for (int active_orbital_index = 0;
       active_orbital_index < n_active_orbitals;
       ++active_orbital_index) {
    target_row[active_orbital_index] +=
        scale * source_row[active_orbital_index];
  }
}

inline void accumulate_active_orbital_matrix_vector_product_dynamic(
    double* target_row,
    const double* matrix_row_major,
    const double* source_row,
    int n_active_orbitals) {
  for (int row_index = 0; row_index < n_active_orbitals; ++row_index) {
    const double* matrix_row =
        matrix_row_major + xmvb::to_size(row_index) * n_active_orbitals;
    double row_value = 0.0;
#pragma omp simd reduction(+ : row_value)
    for (int column_index = 0; column_index < n_active_orbitals; ++column_index) {
      row_value += matrix_row[column_index] * source_row[column_index];
    }
    target_row[row_index] += row_value;
  }
}

template <int NActiveOrbitals>
inline void accumulate_scaled_cached_backprop_rows_to_active_gradient(
    double* target_row,
    const double* basis_backprop_rows,
    const double* pair_source_row,
    double scale) {
  if (scale == 0.0) {
    return;
  }
  constexpr std::size_t kNActivePairs =
      xmvb::to_size(NActiveOrbitals) * (NActiveOrbitals + 1) / 2;
  for (std::size_t active_pair_index = 0;
       active_pair_index < kNActivePairs;
       ++active_pair_index) {
    const double pair_scale = scale * pair_source_row[active_pair_index];
    if (pair_scale == 0.0) {
      continue;
    }
    const double* backprop_row =
        basis_backprop_rows + active_pair_index * NActiveOrbitals;
#pragma omp simd
    for (int active_orbital_index = 0;
         active_orbital_index < NActiveOrbitals;
         ++active_orbital_index) {
      target_row[active_orbital_index] +=
          pair_scale * backprop_row[active_orbital_index];
    }
  }
}

inline void accumulate_scaled_cached_backprop_rows_to_active_gradient_dynamic(
    double* target_row,
    const double* basis_backprop_rows,
    const double* pair_source_row,
    std::size_t n_active_pairs,
    int n_active_orbitals,
    double scale) {
  if (scale == 0.0) {
    return;
  }
  for (std::size_t active_pair_index = 0;
       active_pair_index < n_active_pairs;
       ++active_pair_index) {
    const double pair_scale = scale * pair_source_row[active_pair_index];
    if (pair_scale == 0.0) {
      continue;
    }
    const double* backprop_row =
        basis_backprop_rows +
        active_pair_index * xmvb::to_size(n_active_orbitals);
#pragma omp simd
    for (int active_orbital_index = 0;
         active_orbital_index < n_active_orbitals;
         ++active_orbital_index) {
      target_row[active_orbital_index] +=
          pair_scale * backprop_row[active_orbital_index];
    }
  }
}

void reduce_partial_dense_active_gradients(
    const std::vector<double>& partial_dense_active_gradients,
    int n_threads,
    std::size_t dense_size,
    std::vector<double>* dense_active_gradients) {
  if (dense_active_gradients == nullptr ||
      dense_active_gradients->size() != dense_size) {
    throw std::invalid_argument("dense active gradient reduction size mismatch");
  }
  if (n_threads <= 0) {
    throw std::invalid_argument("dense active gradient reduction thread count must be positive");
  }
  if (partial_dense_active_gradients.size() !=
      xmvb::to_size(n_threads) * dense_size) {
    throw std::invalid_argument("dense active gradient reduction buffer size mismatch");
  }

  // Parallel HVPs use one dense-active accumulator per thread and only reduce
  // `n_basis * n_active` data at the end, instead of reducing an
  // `n_ao_pairs * n_active_pairs` workspace for every thread.
  const double* partial_data = partial_dense_active_gradients.data();
  double* dense_data = dense_active_gradients->data();
  for (int thread_index = 0; thread_index < n_threads; ++thread_index) {
    const double* local_dense =
        partial_data + xmvb::to_size(thread_index) * dense_size;
    for (std::size_t dense_index = 0;
         dense_index < dense_size;
         ++dense_index) {
      dense_data[dense_index] += local_dense[dense_index];
    }
  }
}

template <typename ApplyRowFn>
void apply_cached_row_graph_to_owned_basis_rows(
    int n_basis_functions,
    int n_threads,
    ApplyRowFn&& apply_row_fn) {
  if (n_threads <= 1) {
    for (int target_basis = 0; target_basis < n_basis_functions; ++target_basis) {
      apply_row_fn(target_basis);
    }
    return;
  }

#pragma omp parallel for schedule(static) num_threads(n_threads)
  for (int target_basis = 0; target_basis < n_basis_functions; ++target_basis) {
    apply_row_fn(target_basis);
  }
}

template <int NActiveOrbitals>
void accumulate_cached_row_graph_to_basis_gradient_fixed(
    int target_basis,
    int n_basis_functions,
    const std::vector<int>& row_offsets,
    const std::vector<int>& column_pair_indices,
    const std::vector<int>& integral_indices,
    const std::vector<double>& ao_two_electron_integral_values,
    const double* pair_source_coefficients_data,
    const double* accepted_backprop_rows,
    double* target_gradient) {
  constexpr std::size_t kNActivePairs =
      xmvb::to_size(NActiveOrbitals) * (NActiveOrbitals + 1) / 2;

  // The multi-thread cached-row kernel assigns each physical AO basis row to a
  // single OpenMP worker. Every AO-pair graph row contributes to one or two
  // basis rows, so iterating by owned target basis removes the previous
  // thread-private `n_basis x n_active` dense buffer and the final full-size
  // reduction without changing the mathematical contraction.
  for (int paired_basis = 0; paired_basis <= target_basis; ++paired_basis) {
    const std::size_t row_index = ao_pair_index(target_basis, paired_basis);
    const double* paired_basis_backprop_rows =
        accepted_backprop_rows +
        xmvb::to_size(paired_basis) * kNActivePairs * NActiveOrbitals;
    const int begin = row_offsets[row_index];
    const int end = row_offsets[row_index + 1];
    for (int entry_offset = begin; entry_offset < end; ++entry_offset) {
      const std::size_t column_pair_index = xmvb::to_size(
          column_pair_indices[xmvb::to_size(entry_offset)]);
      const int integral_index = integral_indices[xmvb::to_size(entry_offset)];
      const double ao_integral_value =
          ao_two_electron_integral_values[xmvb::to_size(integral_index)];
      const double* source_row =
          pair_source_coefficients_data +
          column_pair_index * kNActivePairs;
      accumulate_scaled_cached_backprop_rows_to_active_gradient<
          NActiveOrbitals>(
          target_gradient,
          paired_basis_backprop_rows,
          source_row,
          ao_integral_value);
    }
  }

  for (int paired_basis = target_basis + 1;
       paired_basis < n_basis_functions;
       ++paired_basis) {
    const std::size_t row_index = ao_pair_index(paired_basis, target_basis);
    const double* paired_basis_backprop_rows =
        accepted_backprop_rows +
        xmvb::to_size(paired_basis) * kNActivePairs * NActiveOrbitals;
    const int begin = row_offsets[row_index];
    const int end = row_offsets[row_index + 1];
    for (int entry_offset = begin; entry_offset < end; ++entry_offset) {
      const std::size_t column_pair_index = xmvb::to_size(
          column_pair_indices[xmvb::to_size(entry_offset)]);
      const int integral_index = integral_indices[xmvb::to_size(entry_offset)];
      const double ao_integral_value =
          ao_two_electron_integral_values[xmvb::to_size(integral_index)];
      const double* source_row =
          pair_source_coefficients_data +
          column_pair_index * kNActivePairs;
      accumulate_scaled_cached_backprop_rows_to_active_gradient<
          NActiveOrbitals>(
          target_gradient,
          paired_basis_backprop_rows,
          source_row,
          ao_integral_value);
    }
  }
}

void accumulate_cached_row_graph_to_basis_gradient_dynamic(
    int target_basis,
    int n_basis_functions,
    int n_active_orbitals,
    std::size_t n_active_pairs,
    const std::vector<int>& row_offsets,
    const std::vector<int>& column_pair_indices,
    const std::vector<int>& integral_indices,
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<double>& pair_source_coefficients,
    const double* accepted_backprop_rows,
    double* target_gradient) {
  for (int paired_basis = 0; paired_basis <= target_basis; ++paired_basis) {
    const std::size_t row_index = ao_pair_index(target_basis, paired_basis);
    const double* paired_basis_backprop_rows =
        accepted_backprop_rows +
        xmvb::to_size(paired_basis) * n_active_pairs * n_active_orbitals;
    const int begin = row_offsets[row_index];
    const int end = row_offsets[row_index + 1];
    for (int entry_offset = begin; entry_offset < end; ++entry_offset) {
      const std::size_t column_pair_index = xmvb::to_size(
          column_pair_indices[xmvb::to_size(entry_offset)]);
      const int integral_index = integral_indices[xmvb::to_size(entry_offset)];
      const double ao_integral_value =
          ao_two_electron_integral_values[xmvb::to_size(integral_index)];
      const double* source_row =
          pair_source_coefficients.data() +
          column_pair_index * n_active_pairs;
      accumulate_scaled_cached_backprop_rows_to_active_gradient_dynamic(
          target_gradient,
          paired_basis_backprop_rows,
          source_row,
          n_active_pairs,
          n_active_orbitals,
          ao_integral_value);
    }
  }

  for (int paired_basis = target_basis + 1;
       paired_basis < n_basis_functions;
       ++paired_basis) {
    const std::size_t row_index = ao_pair_index(paired_basis, target_basis);
    const double* paired_basis_backprop_rows =
        accepted_backprop_rows +
        xmvb::to_size(paired_basis) * n_active_pairs * n_active_orbitals;
    const int begin = row_offsets[row_index];
    const int end = row_offsets[row_index + 1];
    for (int entry_offset = begin; entry_offset < end; ++entry_offset) {
      const std::size_t column_pair_index = xmvb::to_size(
          column_pair_indices[xmvb::to_size(entry_offset)]);
      const int integral_index = integral_indices[xmvb::to_size(entry_offset)];
      const double ao_integral_value =
          ao_two_electron_integral_values[xmvb::to_size(integral_index)];
      const double* source_row =
          pair_source_coefficients.data() +
          column_pair_index * n_active_pairs;
      accumulate_scaled_cached_backprop_rows_to_active_gradient_dynamic(
          target_gradient,
          paired_basis_backprop_rows,
          source_row,
          n_active_pairs,
          n_active_orbitals,
          ao_integral_value);
    }
  }
}

template <int NActiveOrbitals>
void build_mixed_ao_pair_to_active_pair_coefficients_from_cache_fixed(
    const std::vector<double>& dense_active_coefficients,
    const std::vector<double>& dense_active_direction,
    int n_basis_functions,
    std::vector<double>* mixed_ao_pair_to_active_pair_coefficients) {
  const std::size_t n_active_pairs =
      xmvb::to_size(NActiveOrbitals) * (NActiveOrbitals + 1) / 2;
  const std::size_t n_ao_pairs =
      xmvb::to_size(n_basis_functions) * (n_basis_functions + 1) / 2;
  resize_for_overwrite(
      mixed_ao_pair_to_active_pair_coefficients,
      n_ao_pairs * n_active_pairs);

#pragma omp parallel for schedule(static)
  for (int first_basis_function = 0;
       first_basis_function < n_basis_functions;
       ++first_basis_function) {
    const double* first_coefficients =
        dense_active_coefficients.data() +
        xmvb::to_size(first_basis_function) * NActiveOrbitals;
    const double* first_directions =
        dense_active_direction.data() +
        xmvb::to_size(first_basis_function) * NActiveOrbitals;
    for (int second_basis_function = 0;
         second_basis_function <= first_basis_function;
         ++second_basis_function) {
      const double* second_coefficients =
          dense_active_coefficients.data() +
          xmvb::to_size(second_basis_function) * NActiveOrbitals;
      const double* second_directions =
          dense_active_direction.data() +
          xmvb::to_size(second_basis_function) * NActiveOrbitals;
      double* pair_coefficients =
          mixed_ao_pair_to_active_pair_coefficients->data() +
          ao_pair_index(first_basis_function, second_basis_function) * n_active_pairs;
      int packed_pair_index = 0;
      for (int first_active = 0;
           first_active < NActiveOrbitals;
           ++first_active) {
        for (int second_active = 0;
             second_active <= first_active;
             ++second_active) {
          double coefficient =
              first_directions[first_active] * second_coefficients[second_active] +
              first_coefficients[first_active] * second_directions[second_active];
          if (first_basis_function != second_basis_function) {
            coefficient +=
                second_directions[first_active] * first_coefficients[second_active] +
                second_coefficients[first_active] * first_directions[second_active];
          }
          pair_coefficients[packed_pair_index++] = coefficient;
        }
      }
    }
  }
}

template <int NActiveOrbitals>
void build_accepted_active_pair_gradient_backprop_rows_fixed(
    const ExactPackedActiveTwoElectronAdjointCache& cache,
    std::vector<double>* accepted_backprop_rows) {
  const int n_basis_functions = cache.n_basis_functions;
  const std::size_t n_active_pairs =
      xmvb::to_size(NActiveOrbitals) * (NActiveOrbitals + 1) / 2;
  if (cache.active_pair_gradient_buffer.size() != n_active_pairs * n_active_pairs) {
    throw std::invalid_argument(
        "accepted active-pair backprop cache gradient size mismatch");
  }
  if (cache.accepted_dense_active_coefficients_buffer.size() !=
      xmvb::to_size(n_basis_functions) * NActiveOrbitals) {
    throw std::invalid_argument(
        "accepted active-pair backprop cache coefficient size mismatch");
  }

  resize_and_zero(
      accepted_backprop_rows,
      xmvb::to_size(n_basis_functions) * n_active_pairs * NActiveOrbitals);

#pragma omp parallel for schedule(static)
  for (int basis_function_index = 0;
       basis_function_index < n_basis_functions;
       ++basis_function_index) {
    const double* accepted_coefficient_row =
        cache.accepted_dense_active_coefficients_buffer.data() +
        xmvb::to_size(basis_function_index) * NActiveOrbitals;
    double* accepted_backprop_basis_rows =
        accepted_backprop_rows->data() +
        xmvb::to_size(basis_function_index) * n_active_pairs * NActiveOrbitals;
    for (std::size_t active_pair_index = 0;
         active_pair_index < n_active_pairs;
         ++active_pair_index) {
      const double* packed_pair_gradient_row =
          cache.active_pair_gradient_buffer.data() +
          active_pair_index * n_active_pairs;
      double* backpropagated_active_row =
          accepted_backprop_basis_rows +
          active_pair_index * NActiveOrbitals;
      accumulate_packed_active_pair_gradient_row_to_active_gradient<NActiveOrbitals>(
          backpropagated_active_row,
          packed_pair_gradient_row,
          accepted_coefficient_row);
    }
  }
}

void build_accepted_active_pair_gradient_backprop_rows(
    const ExactPackedActiveTwoElectronAdjointCache& cache,
    std::vector<double>* accepted_backprop_rows) {
  const int n_basis_functions = cache.n_basis_functions;
  const int n_active_orbitals = cache.n_active_orbitals;
  const std::size_t n_active_pairs = cache.active_pair_first_indices.size();
  if (cache.active_pair_second_indices.size() != n_active_pairs) {
    throw std::invalid_argument("exact 2e cache active-pair index size mismatch");
  }
  switch (n_active_orbitals) {
    case 1:
      build_accepted_active_pair_gradient_backprop_rows_fixed<1>(
          cache,
          accepted_backprop_rows);
      return;
    case 2:
      build_accepted_active_pair_gradient_backprop_rows_fixed<2>(
          cache,
          accepted_backprop_rows);
      return;
    case 3:
      build_accepted_active_pair_gradient_backprop_rows_fixed<3>(
          cache,
          accepted_backprop_rows);
      return;
    case 4:
      build_accepted_active_pair_gradient_backprop_rows_fixed<4>(
          cache,
          accepted_backprop_rows);
      return;
    case 5:
      build_accepted_active_pair_gradient_backprop_rows_fixed<5>(
          cache,
          accepted_backprop_rows);
      return;
    case 6:
      build_accepted_active_pair_gradient_backprop_rows_fixed<6>(
          cache,
          accepted_backprop_rows);
      return;
    case 7:
      build_accepted_active_pair_gradient_backprop_rows_fixed<7>(
          cache,
          accepted_backprop_rows);
      return;
    case 8:
      build_accepted_active_pair_gradient_backprop_rows_fixed<8>(
          cache,
          accepted_backprop_rows);
      return;
    case 9:
      build_accepted_active_pair_gradient_backprop_rows_fixed<9>(
          cache,
          accepted_backprop_rows);
      return;
    case 10:
      build_accepted_active_pair_gradient_backprop_rows_fixed<10>(
          cache,
          accepted_backprop_rows);
      return;
    default:
      break;
  }

  if (cache.active_pair_gradient_buffer.size() != n_active_pairs * n_active_pairs) {
    throw std::invalid_argument(
        "accepted active-pair backprop cache gradient size mismatch");
  }
  if (cache.accepted_dense_active_coefficients_buffer.size() !=
      xmvb::to_size(n_basis_functions) * n_active_orbitals) {
    throw std::invalid_argument(
        "accepted active-pair backprop cache coefficient size mismatch");
  }

  resize_and_zero(
      accepted_backprop_rows,
      xmvb::to_size(n_basis_functions) * n_active_pairs * n_active_orbitals);

#pragma omp parallel for schedule(static)
  for (int basis_function_index = 0;
       basis_function_index < n_basis_functions;
       ++basis_function_index) {
    const double* accepted_coefficient_row =
        cache.accepted_dense_active_coefficients_buffer.data() +
        xmvb::to_size(basis_function_index) * n_active_orbitals;
    double* accepted_backprop_basis_rows =
        accepted_backprop_rows->data() +
        xmvb::to_size(basis_function_index) * n_active_pairs * n_active_orbitals;
    for (std::size_t active_pair_index = 0;
         active_pair_index < n_active_pairs;
         ++active_pair_index) {
      const double* packed_pair_gradient_row =
          cache.active_pair_gradient_buffer.data() +
          active_pair_index * n_active_pairs;
      double* backpropagated_active_row =
          accepted_backprop_basis_rows +
          active_pair_index * n_active_orbitals;
      for (std::size_t gradient_pair_index = 0;
           gradient_pair_index < n_active_pairs;
           ++gradient_pair_index) {
        const double pair_gradient = packed_pair_gradient_row[gradient_pair_index];
        if (pair_gradient == 0.0) {
          continue;
        }
        const int first_active =
            cache.active_pair_first_indices[gradient_pair_index];
        const int second_active =
            cache.active_pair_second_indices[gradient_pair_index];
        backpropagated_active_row[first_active] +=
            pair_gradient * accepted_coefficient_row[second_active];
        backpropagated_active_row[second_active] +=
            pair_gradient * accepted_coefficient_row[first_active];
      }
    }
  }
}

template <int NActiveOrbitals>
void
backpropagate_fixed_pair_gradient_matrices_to_dense_active_coefficients_from_cache_fixed(
    const std::vector<double>& dense_active_direction,
    const ExactPackedActiveTwoElectronAdjointCache& cache,
    std::vector<double>* dense_active_gradients) {
  const int n_basis_functions = cache.n_basis_functions;
  const std::size_t n_ao_pairs =
      xmvb::to_size(n_basis_functions) * (n_basis_functions + 1) / 2;
  const std::size_t n_active_square =
      xmvb::to_size(NActiveOrbitals) * NActiveOrbitals;
  if (cache.accepted_base_pair_gradient_matrices_buffer.size() !=
      n_ao_pairs * n_active_square) {
    throw std::invalid_argument("fixed pair gradient matrix cache size mismatch");
  }
  resize_and_zero(
      dense_active_gradients,
      xmvb::to_size(n_basis_functions) * NActiveOrbitals);

  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif
  if (n_threads <= 1) {
    for (int first_basis_function = 0;
         first_basis_function < n_basis_functions;
         ++first_basis_function) {
      double* first_gradient_row =
          dense_active_gradients->data() +
          xmvb::to_size(first_basis_function) * NActiveOrbitals;
      const double* first_direction_row =
          dense_active_direction.data() +
          xmvb::to_size(first_basis_function) * NActiveOrbitals;
      for (int second_basis_function = 0;
           second_basis_function <= first_basis_function;
           ++second_basis_function) {
        const double* second_direction_row =
            dense_active_direction.data() +
            xmvb::to_size(second_basis_function) * NActiveOrbitals;
        const double* pair_gradient_matrix =
            cache.accepted_base_pair_gradient_matrices_buffer.data() +
            ao_pair_index(first_basis_function, second_basis_function) *
                n_active_square;
        accumulate_active_orbital_matrix_vector_product<NActiveOrbitals>(
            first_gradient_row,
            pair_gradient_matrix,
            second_direction_row);
        if (second_basis_function != first_basis_function) {
          double* second_gradient_row =
              dense_active_gradients->data() +
              xmvb::to_size(second_basis_function) * NActiveOrbitals;
          accumulate_active_orbital_matrix_vector_product<NActiveOrbitals>(
              second_gradient_row,
              pair_gradient_matrix,
              first_direction_row);
        }
      }
    }
    return;
  }

#pragma omp parallel for schedule(static)
  for (int basis_function_index = 0;
       basis_function_index < n_basis_functions;
       ++basis_function_index) {
    double* gradient_row =
        dense_active_gradients->data() +
        xmvb::to_size(basis_function_index) * NActiveOrbitals;
    for (int other_basis_function = 0;
         other_basis_function < n_basis_functions;
         ++other_basis_function) {
      const double* other_direction_row =
          dense_active_direction.data() +
          xmvb::to_size(other_basis_function) * NActiveOrbitals;
      const double* pair_gradient_matrix =
          cache.accepted_base_pair_gradient_matrices_buffer.data() +
          ao_pair_index(basis_function_index, other_basis_function) *
              n_active_square;
      accumulate_active_orbital_matrix_vector_product<NActiveOrbitals>(
          gradient_row,
          pair_gradient_matrix,
          other_direction_row);
    }
  }
}

void
backpropagate_fixed_pair_gradient_matrices_to_dense_active_coefficients_from_cache(
    const std::vector<double>& dense_active_direction,
    const ExactPackedActiveTwoElectronAdjointCache& cache,
    std::vector<double>* dense_active_gradients) {
  const int n_basis_functions = cache.n_basis_functions;
  const int n_active_orbitals = cache.n_active_orbitals;
  const std::size_t expected_dense_size =
      xmvb::to_size(n_basis_functions) * n_active_orbitals;
  if (dense_active_direction.size() != expected_dense_size) {
    throw std::invalid_argument("dense active direction size mismatch");
  }
  switch (n_active_orbitals) {
    case 1:
      backpropagate_fixed_pair_gradient_matrices_to_dense_active_coefficients_from_cache_fixed<1>(
          dense_active_direction,
          cache,
          dense_active_gradients);
      return;
    case 2:
      backpropagate_fixed_pair_gradient_matrices_to_dense_active_coefficients_from_cache_fixed<2>(
          dense_active_direction,
          cache,
          dense_active_gradients);
      return;
    case 3:
      backpropagate_fixed_pair_gradient_matrices_to_dense_active_coefficients_from_cache_fixed<3>(
          dense_active_direction,
          cache,
          dense_active_gradients);
      return;
    case 4:
      backpropagate_fixed_pair_gradient_matrices_to_dense_active_coefficients_from_cache_fixed<4>(
          dense_active_direction,
          cache,
          dense_active_gradients);
      return;
    case 5:
      backpropagate_fixed_pair_gradient_matrices_to_dense_active_coefficients_from_cache_fixed<5>(
          dense_active_direction,
          cache,
          dense_active_gradients);
      return;
    case 6:
      backpropagate_fixed_pair_gradient_matrices_to_dense_active_coefficients_from_cache_fixed<6>(
          dense_active_direction,
          cache,
          dense_active_gradients);
      return;
    case 7:
      backpropagate_fixed_pair_gradient_matrices_to_dense_active_coefficients_from_cache_fixed<7>(
          dense_active_direction,
          cache,
          dense_active_gradients);
      return;
    case 8:
      backpropagate_fixed_pair_gradient_matrices_to_dense_active_coefficients_from_cache_fixed<8>(
          dense_active_direction,
          cache,
          dense_active_gradients);
      return;
    case 9:
      backpropagate_fixed_pair_gradient_matrices_to_dense_active_coefficients_from_cache_fixed<9>(
          dense_active_direction,
          cache,
          dense_active_gradients);
      return;
    case 10:
      backpropagate_fixed_pair_gradient_matrices_to_dense_active_coefficients_from_cache_fixed<10>(
          dense_active_direction,
          cache,
          dense_active_gradients);
      return;
    default:
      break;
  }

  const std::size_t n_ao_pairs =
      xmvb::to_size(n_basis_functions) * (n_basis_functions + 1) / 2;
  const std::size_t n_active_square =
      xmvb::to_size(n_active_orbitals) * n_active_orbitals;
  if (cache.accepted_base_pair_gradient_matrices_buffer.size() !=
      n_ao_pairs * n_active_square) {
    throw std::invalid_argument("fixed pair gradient matrix cache size mismatch");
  }

  resize_and_zero(
      dense_active_gradients,
      expected_dense_size);
  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif
  if (n_threads <= 1) {
    for (int first_basis_function = 0;
         first_basis_function < n_basis_functions;
         ++first_basis_function) {
      double* first_gradient_row =
          dense_active_gradients->data() +
          xmvb::to_size(first_basis_function) * n_active_orbitals;
      const double* first_direction_row =
          dense_active_direction.data() +
          xmvb::to_size(first_basis_function) * n_active_orbitals;
      for (int second_basis_function = 0;
           second_basis_function <= first_basis_function;
           ++second_basis_function) {
        const double* second_direction_row =
            dense_active_direction.data() +
            xmvb::to_size(second_basis_function) * n_active_orbitals;
        const double* pair_gradient_matrix =
            cache.accepted_base_pair_gradient_matrices_buffer.data() +
            ao_pair_index(first_basis_function, second_basis_function) *
                n_active_square;
        accumulate_active_orbital_matrix_vector_product_dynamic(
            first_gradient_row,
            pair_gradient_matrix,
            second_direction_row,
            n_active_orbitals);
        if (second_basis_function != first_basis_function) {
          double* second_gradient_row =
              dense_active_gradients->data() +
              xmvb::to_size(second_basis_function) * n_active_orbitals;
          accumulate_active_orbital_matrix_vector_product_dynamic(
              second_gradient_row,
              pair_gradient_matrix,
              first_direction_row,
              n_active_orbitals);
        }
      }
    }
    return;
  }

#pragma omp parallel for schedule(static)
  for (int basis_function_index = 0;
       basis_function_index < n_basis_functions;
       ++basis_function_index) {
    double* gradient_row =
        dense_active_gradients->data() +
        xmvb::to_size(basis_function_index) * n_active_orbitals;
    for (int other_basis_function = 0;
         other_basis_function < n_basis_functions;
         ++other_basis_function) {
      const double* other_direction_row =
          dense_active_direction.data() +
          xmvb::to_size(other_basis_function) * n_active_orbitals;
      const double* pair_gradient_matrix =
          cache.accepted_base_pair_gradient_matrices_buffer.data() +
          ao_pair_index(basis_function_index, other_basis_function) *
              n_active_square;
      accumulate_active_orbital_matrix_vector_product_dynamic(
          gradient_row,
          pair_gradient_matrix,
          other_direction_row,
          n_active_orbitals);
    }
  }
}

std::vector<double>
backpropagate_fixed_pair_gradient_matrices_to_dense_active_coefficients_from_cache(
    const std::vector<double>& dense_active_direction,
    const ExactPackedActiveTwoElectronAdjointCache& cache) {
  std::vector<double> dense_active_gradients;
  backpropagate_fixed_pair_gradient_matrices_to_dense_active_coefficients_from_cache(
      dense_active_direction,
      cache,
      &dense_active_gradients);
  return dense_active_gradients;
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
      n_ao_pairs * xmvb::to_size(NActivePairs),
      0.0);
  double* pair_gradients_data = pair_gradients.data();
  for (std::size_t integral_index = 0;
       integral_index < n_integrals;
       ++integral_index) {
    const double ao_integral_value =
        ao_two_electron_integral_values_data[integral_index];
    const std::size_t left_pair_index = xmvb::to_size(
        ao_two_electron_pair_indices_data[integral_index * 2]);
    const std::size_t right_pair_index = xmvb::to_size(
        ao_two_electron_pair_indices_data[integral_index * 2 + 1]);
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
      n_ao_pairs * xmvb::to_size(NActivePairs));
  double* pair_gradients_data = pair_gradients->data();
  for (std::size_t row_index = 0; row_index < n_ao_pairs; ++row_index) {
    double* target_row =
        pair_gradients_data + pair_row_offsets_data[row_index];
    const int begin = row_offsets[row_index];
    const int end = row_offsets[row_index + 1];
    for (int entry_offset = begin; entry_offset < end; ++entry_offset) {
      const int column_pair_index =
          column_pair_indices[xmvb::to_size(entry_offset)];
      const int integral_index =
          integral_indices[xmvb::to_size(entry_offset)];
      const double ao_integral_value =
          ao_two_electron_integral_values[xmvb::to_size(integral_index)];
      const double* source_row =
          transformed_pair_coefficients_data +
          pair_row_offsets_data[xmvb::to_size(column_pair_index)];
      accumulate_scaled_active_pair_row_hvp<NActivePairs>(
          target_row,
          source_row,
          ao_integral_value);
    }
  }
}

template <int NActiveOrbitals>
void apply_sparse_ao_integral_matrix_and_backprop_single_thread_fixed_hvp(
    const double* ao_two_electron_integral_values_data,
    const int* ao_two_electron_pair_indices_data,
    std::size_t n_integrals,
    const double* transformed_pair_coefficients_data,
    const ExactPackedActiveTwoElectronAdjointCache& cache,
    std::vector<double>* dense_active_gradients) {
  constexpr std::size_t kNActivePairs =
      xmvb::to_size(NActiveOrbitals) * (NActiveOrbitals + 1) / 2;
  if (dense_active_gradients == nullptr) {
    throw std::invalid_argument("dense active gradient output must not be null");
  }
  if (cache.ao_pair_first_indices.size() != cache.ao_pair_second_indices.size()) {
    throw std::invalid_argument("AO pair component cache size mismatch");
  }
  if (dense_active_gradients->size() !=
      xmvb::to_size(cache.n_basis_functions) * NActiveOrbitals) {
    throw std::invalid_argument("dense active gradient output size mismatch");
  }
  const double* dense_active_coefficients_data =
      cache.accepted_dense_active_coefficients_buffer.data();

  for (std::size_t integral_index = 0;
       integral_index < n_integrals;
       ++integral_index) {
    const double ao_integral_value =
        ao_two_electron_integral_values_data[integral_index];
    const std::size_t left_pair_index = xmvb::to_size(
        ao_two_electron_pair_indices_data[integral_index * 2]);
    const std::size_t right_pair_index = xmvb::to_size(
        ao_two_electron_pair_indices_data[integral_index * 2 + 1]);

    const int left_first_basis =
        cache.ao_pair_first_indices[left_pair_index];
    const int left_second_basis =
        cache.ao_pair_second_indices[left_pair_index];
    double* left_first_gradient =
        dense_active_gradients->data() +
        xmvb::to_size(left_first_basis) * NActiveOrbitals;
    const double* left_second_coefficients =
        dense_active_coefficients_data +
        xmvb::to_size(left_second_basis) * NActiveOrbitals;
    const double* right_row =
        transformed_pair_coefficients_data +
        right_pair_index * kNActivePairs;
    accumulate_scaled_packed_active_pair_gradient_row_to_active_gradient<
        NActiveOrbitals>(
        left_first_gradient,
        right_row,
        left_second_coefficients,
        ao_integral_value);
    if (left_second_basis != left_first_basis) {
      double* left_second_gradient =
          dense_active_gradients->data() +
          xmvb::to_size(left_second_basis) * NActiveOrbitals;
      const double* left_first_coefficients =
          dense_active_coefficients_data +
          xmvb::to_size(left_first_basis) * NActiveOrbitals;
      accumulate_scaled_packed_active_pair_gradient_row_to_active_gradient<
          NActiveOrbitals>(
          left_second_gradient,
          right_row,
          left_first_coefficients,
          ao_integral_value);
    }

    if (left_pair_index == right_pair_index) {
      continue;
    }

    const int right_first_basis =
        cache.ao_pair_first_indices[right_pair_index];
    const int right_second_basis =
        cache.ao_pair_second_indices[right_pair_index];
    double* right_first_gradient =
        dense_active_gradients->data() +
        xmvb::to_size(right_first_basis) * NActiveOrbitals;
    const double* right_second_coefficients =
        dense_active_coefficients_data +
        xmvb::to_size(right_second_basis) * NActiveOrbitals;
    const double* left_row =
        transformed_pair_coefficients_data +
        left_pair_index * kNActivePairs;
    accumulate_scaled_packed_active_pair_gradient_row_to_active_gradient<
        NActiveOrbitals>(
        right_first_gradient,
        left_row,
        right_second_coefficients,
        ao_integral_value);
    if (right_second_basis != right_first_basis) {
      double* right_second_gradient =
          dense_active_gradients->data() +
          xmvb::to_size(right_second_basis) * NActiveOrbitals;
      const double* right_first_coefficients =
          dense_active_coefficients_data +
          xmvb::to_size(right_first_basis) * NActiveOrbitals;
      accumulate_scaled_packed_active_pair_gradient_row_to_active_gradient<
          NActiveOrbitals>(
          right_second_gradient,
          left_row,
          right_first_coefficients,
          ao_integral_value);
    }
  }
}

template <int NActiveOrbitals>
void apply_sparse_ao_integral_matrix_and_backprop_from_cached_rows_fixed(
    const double* ao_two_electron_integral_values_data,
    const int* ao_two_electron_pair_indices_data,
    std::size_t n_integrals,
    const double* pair_source_coefficients_data,
    const ExactPackedActiveTwoElectronAdjointCache& cache,
    std::vector<double>* dense_active_gradients) {
  constexpr std::size_t kNActivePairs =
      xmvb::to_size(NActiveOrbitals) * (NActiveOrbitals + 1) / 2;
  const int n_basis_functions = cache.n_basis_functions;
  const std::size_t n_ao_pairs = cache.ao_pair_first_indices.size();
  const std::size_t dense_size =
      xmvb::to_size(n_basis_functions) * NActiveOrbitals;
  const std::size_t expected_backprop_size =
      xmvb::to_size(n_basis_functions) * kNActivePairs * NActiveOrbitals;
  if (dense_active_gradients == nullptr ||
      cache.ao_pair_second_indices.size() != n_ao_pairs ||
      cache.accepted_active_pair_gradient_backprop_rows_buffer.size() !=
          expected_backprop_size) {
    throw std::invalid_argument(
        "cached-row sparse AO integral / cache size mismatch");
  }
  if (dense_active_gradients->size() != dense_size) {
    throw std::invalid_argument("dense active gradient output size mismatch");
  }
  const double* accepted_backprop_rows =
      cache.accepted_active_pair_gradient_backprop_rows_buffer.data();
  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif
  if (n_threads <= 1) {
    for (std::size_t integral_index = 0;
         integral_index < n_integrals;
         ++integral_index) {
      const double ao_integral_value =
          ao_two_electron_integral_values_data[integral_index];
      const std::size_t left_pair_index = xmvb::to_size(
          ao_two_electron_pair_indices_data[integral_index * 2]);
      const std::size_t right_pair_index = xmvb::to_size(
          ao_two_electron_pair_indices_data[integral_index * 2 + 1]);

      const int left_first_basis =
          cache.ao_pair_first_indices[left_pair_index];
      const int left_second_basis =
          cache.ao_pair_second_indices[left_pair_index];
      double* left_first_gradient =
          dense_active_gradients->data() +
          xmvb::to_size(left_first_basis) * NActiveOrbitals;
      const double* left_second_basis_backprop_rows =
          accepted_backprop_rows +
          xmvb::to_size(left_second_basis) * kNActivePairs * NActiveOrbitals;
      const double* right_row =
          pair_source_coefficients_data +
          right_pair_index * kNActivePairs;
      accumulate_scaled_cached_backprop_rows_to_active_gradient<
          NActiveOrbitals>(
          left_first_gradient,
          left_second_basis_backprop_rows,
          right_row,
          ao_integral_value);
      if (left_second_basis != left_first_basis) {
        double* left_second_gradient =
            dense_active_gradients->data() +
            xmvb::to_size(left_second_basis) * NActiveOrbitals;
        const double* left_first_basis_backprop_rows =
            accepted_backprop_rows +
            xmvb::to_size(left_first_basis) * kNActivePairs * NActiveOrbitals;
        accumulate_scaled_cached_backprop_rows_to_active_gradient<
            NActiveOrbitals>(
            left_second_gradient,
            left_first_basis_backprop_rows,
            right_row,
            ao_integral_value);
      }

      if (left_pair_index == right_pair_index) {
        continue;
      }

      const int right_first_basis =
          cache.ao_pair_first_indices[right_pair_index];
      const int right_second_basis =
          cache.ao_pair_second_indices[right_pair_index];
      double* right_first_gradient =
          dense_active_gradients->data() +
          xmvb::to_size(right_first_basis) * NActiveOrbitals;
      const double* right_second_basis_backprop_rows =
          accepted_backprop_rows +
          xmvb::to_size(right_second_basis) * kNActivePairs * NActiveOrbitals;
      const double* left_row =
          pair_source_coefficients_data +
          left_pair_index * kNActivePairs;
      accumulate_scaled_cached_backprop_rows_to_active_gradient<
          NActiveOrbitals>(
          right_first_gradient,
          right_second_basis_backprop_rows,
          left_row,
          ao_integral_value);
      if (right_second_basis != right_first_basis) {
        double* right_second_gradient =
            dense_active_gradients->data() +
            xmvb::to_size(right_second_basis) * NActiveOrbitals;
        const double* right_first_basis_backprop_rows =
            accepted_backprop_rows +
            xmvb::to_size(right_first_basis) * kNActivePairs * NActiveOrbitals;
        accumulate_scaled_cached_backprop_rows_to_active_gradient<
            NActiveOrbitals>(
            right_second_gradient,
            right_first_basis_backprop_rows,
            left_row,
            ao_integral_value);
      }
    }
    return;
  }

  std::vector<double> partial_dense_active_gradients(
      xmvb::to_size(n_threads) * dense_size,
      0.0);
  std::atomic<int> invalid_integral_index(-1);
#pragma omp parallel
  {
    int thread_index = 0;
#ifdef _OPENMP
    thread_index = omp_get_thread_num();
#endif
    double* local_dense_gradients =
        partial_dense_active_gradients.data() +
        xmvb::to_size(thread_index) * dense_size;

#pragma omp for schedule(guided, 256)
    for (std::ptrdiff_t integral_offset = 0;
         integral_offset < static_cast<std::ptrdiff_t>(n_integrals);
         ++integral_offset) {
      const std::size_t integral_index = xmvb::to_size(integral_offset);
      const double ao_integral_value =
          ao_two_electron_integral_values_data[integral_index];
      const int left_pair_index =
          ao_two_electron_pair_indices_data[integral_index * 2];
      const int right_pair_index =
          ao_two_electron_pair_indices_data[integral_index * 2 + 1];
      if (left_pair_index < 0 || right_pair_index < 0 ||
          left_pair_index >= static_cast<int>(n_ao_pairs) ||
          right_pair_index >= static_cast<int>(n_ao_pairs)) {
        int expected = -1;
        invalid_integral_index.compare_exchange_strong(
            expected,
            static_cast<int>(integral_index));
        continue;
      }

      const int left_first_basis =
          cache.ao_pair_first_indices[xmvb::to_size(left_pair_index)];
      const int left_second_basis =
          cache.ao_pair_second_indices[xmvb::to_size(left_pair_index)];
      double* left_first_gradient =
          local_dense_gradients +
          xmvb::to_size(left_first_basis) * NActiveOrbitals;
      const double* left_second_basis_backprop_rows =
          accepted_backprop_rows +
          xmvb::to_size(left_second_basis) * kNActivePairs * NActiveOrbitals;
      const double* right_row =
          pair_source_coefficients_data +
          xmvb::to_size(right_pair_index) * kNActivePairs;
      accumulate_scaled_cached_backprop_rows_to_active_gradient<
          NActiveOrbitals>(
          left_first_gradient,
          left_second_basis_backprop_rows,
          right_row,
          ao_integral_value);
      if (left_second_basis != left_first_basis) {
        double* left_second_gradient =
            local_dense_gradients +
            xmvb::to_size(left_second_basis) * NActiveOrbitals;
        const double* left_first_basis_backprop_rows =
            accepted_backprop_rows +
            xmvb::to_size(left_first_basis) * kNActivePairs * NActiveOrbitals;
        accumulate_scaled_cached_backprop_rows_to_active_gradient<
            NActiveOrbitals>(
            left_second_gradient,
            left_first_basis_backprop_rows,
            right_row,
            ao_integral_value);
      }

      if (left_pair_index == right_pair_index) {
        continue;
      }

      const int right_first_basis =
          cache.ao_pair_first_indices[xmvb::to_size(right_pair_index)];
      const int right_second_basis =
          cache.ao_pair_second_indices[xmvb::to_size(right_pair_index)];
      double* right_first_gradient =
          local_dense_gradients +
          xmvb::to_size(right_first_basis) * NActiveOrbitals;
      const double* right_second_basis_backprop_rows =
          accepted_backprop_rows +
          xmvb::to_size(right_second_basis) * kNActivePairs * NActiveOrbitals;
      const double* left_row =
          pair_source_coefficients_data +
          xmvb::to_size(left_pair_index) * kNActivePairs;
      accumulate_scaled_cached_backprop_rows_to_active_gradient<
          NActiveOrbitals>(
          right_first_gradient,
          right_second_basis_backprop_rows,
          left_row,
          ao_integral_value);
      if (right_second_basis != right_first_basis) {
        double* right_second_gradient =
            local_dense_gradients +
            xmvb::to_size(right_second_basis) * NActiveOrbitals;
        const double* right_first_basis_backprop_rows =
            accepted_backprop_rows +
            xmvb::to_size(right_first_basis) * kNActivePairs * NActiveOrbitals;
        accumulate_scaled_cached_backprop_rows_to_active_gradient<
            NActiveOrbitals>(
            right_second_gradient,
            right_first_basis_backprop_rows,
            left_row,
            ao_integral_value);
      }
    }
  }

  if (invalid_integral_index.load() >= 0) {
    throw std::invalid_argument("packed AO pair index out of range");
  }
  reduce_partial_dense_active_gradients(
      partial_dense_active_gradients,
      n_threads,
      dense_size,
      dense_active_gradients);
}

void apply_sparse_ao_integral_matrix_and_backprop_from_cached_rows(
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& ao_two_electron_pair_indices,
    const std::vector<double>& pair_source_coefficients,
    const ExactPackedActiveTwoElectronAdjointCache& cache,
    std::vector<double>* dense_active_gradients) {
  const int n_basis_functions = cache.n_basis_functions;
  const int n_active_orbitals = cache.n_active_orbitals;
  const std::size_t n_active_pairs = cache.active_pair_first_indices.size();
  const std::size_t n_ao_pairs =
      xmvb::to_size(n_basis_functions) * (n_basis_functions + 1) / 2;
  const std::size_t dense_size =
      xmvb::to_size(n_basis_functions) * n_active_orbitals;
  const std::size_t expected_backprop_size =
      xmvb::to_size(n_basis_functions) * n_active_pairs * n_active_orbitals;
  if (ao_two_electron_pair_indices.size() !=
          ao_two_electron_integral_values.size() * 2 ||
      pair_source_coefficients.size() != n_ao_pairs * n_active_pairs ||
      cache.ao_pair_first_indices.size() != n_ao_pairs ||
      cache.ao_pair_second_indices.size() != n_ao_pairs ||
      cache.accepted_active_pair_gradient_backprop_rows_buffer.size() !=
          expected_backprop_size ||
      dense_active_gradients == nullptr) {
    throw std::invalid_argument(
        "cached-row sparse AO integral backprop input size mismatch");
  }

  switch (n_active_orbitals) {
    case 1:
      apply_sparse_ao_integral_matrix_and_backprop_from_cached_rows_fixed<1>(
          ao_two_electron_integral_values.data(),
          ao_two_electron_pair_indices.data(),
          ao_two_electron_integral_values.size(),
          pair_source_coefficients.data(),
          cache,
          dense_active_gradients);
      return;
    case 2:
      apply_sparse_ao_integral_matrix_and_backprop_from_cached_rows_fixed<2>(
          ao_two_electron_integral_values.data(),
          ao_two_electron_pair_indices.data(),
          ao_two_electron_integral_values.size(),
          pair_source_coefficients.data(),
          cache,
          dense_active_gradients);
      return;
    case 3:
      apply_sparse_ao_integral_matrix_and_backprop_from_cached_rows_fixed<3>(
          ao_two_electron_integral_values.data(),
          ao_two_electron_pair_indices.data(),
          ao_two_electron_integral_values.size(),
          pair_source_coefficients.data(),
          cache,
          dense_active_gradients);
      return;
    case 4:
      apply_sparse_ao_integral_matrix_and_backprop_from_cached_rows_fixed<4>(
          ao_two_electron_integral_values.data(),
          ao_two_electron_pair_indices.data(),
          ao_two_electron_integral_values.size(),
          pair_source_coefficients.data(),
          cache,
          dense_active_gradients);
      return;
    case 5:
      apply_sparse_ao_integral_matrix_and_backprop_from_cached_rows_fixed<5>(
          ao_two_electron_integral_values.data(),
          ao_two_electron_pair_indices.data(),
          ao_two_electron_integral_values.size(),
          pair_source_coefficients.data(),
          cache,
          dense_active_gradients);
      return;
    case 6:
      apply_sparse_ao_integral_matrix_and_backprop_from_cached_rows_fixed<6>(
          ao_two_electron_integral_values.data(),
          ao_two_electron_pair_indices.data(),
          ao_two_electron_integral_values.size(),
          pair_source_coefficients.data(),
          cache,
          dense_active_gradients);
      return;
    case 7:
      apply_sparse_ao_integral_matrix_and_backprop_from_cached_rows_fixed<7>(
          ao_two_electron_integral_values.data(),
          ao_two_electron_pair_indices.data(),
          ao_two_electron_integral_values.size(),
          pair_source_coefficients.data(),
          cache,
          dense_active_gradients);
      return;
    case 8:
      apply_sparse_ao_integral_matrix_and_backprop_from_cached_rows_fixed<8>(
          ao_two_electron_integral_values.data(),
          ao_two_electron_pair_indices.data(),
          ao_two_electron_integral_values.size(),
          pair_source_coefficients.data(),
          cache,
          dense_active_gradients);
      return;
    case 9:
      apply_sparse_ao_integral_matrix_and_backprop_from_cached_rows_fixed<9>(
          ao_two_electron_integral_values.data(),
          ao_two_electron_pair_indices.data(),
          ao_two_electron_integral_values.size(),
          pair_source_coefficients.data(),
          cache,
          dense_active_gradients);
      return;
    case 10:
      apply_sparse_ao_integral_matrix_and_backprop_from_cached_rows_fixed<10>(
          ao_two_electron_integral_values.data(),
          ao_two_electron_pair_indices.data(),
          ao_two_electron_integral_values.size(),
          pair_source_coefficients.data(),
          cache,
          dense_active_gradients);
      return;
    default:
      break;
  }

  if (dense_active_gradients->size() != dense_size) {
    throw std::invalid_argument("dense active gradient output size mismatch");
  }
  const double* accepted_backprop_rows =
      cache.accepted_active_pair_gradient_backprop_rows_buffer.data();
  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif
  if (n_threads <= 1) {
    for (std::size_t integral_index = 0;
         integral_index < ao_two_electron_integral_values.size();
         ++integral_index) {
      const double ao_integral_value =
          ao_two_electron_integral_values[integral_index];
      const std::size_t left_pair_index = xmvb::to_size(
          ao_two_electron_pair_indices[integral_index * 2]);
      const std::size_t right_pair_index = xmvb::to_size(
          ao_two_electron_pair_indices[integral_index * 2 + 1]);

      const int left_first_basis =
          cache.ao_pair_first_indices[left_pair_index];
      const int left_second_basis =
          cache.ao_pair_second_indices[left_pair_index];
      double* left_first_gradient =
          dense_active_gradients->data() +
          xmvb::to_size(left_first_basis) * n_active_orbitals;
      const double* left_second_basis_backprop_rows =
          accepted_backprop_rows +
          xmvb::to_size(left_second_basis) * n_active_pairs * n_active_orbitals;
      const double* right_row =
          pair_source_coefficients.data() +
          right_pair_index * n_active_pairs;
      accumulate_scaled_cached_backprop_rows_to_active_gradient_dynamic(
          left_first_gradient,
          left_second_basis_backprop_rows,
          right_row,
          n_active_pairs,
          n_active_orbitals,
          ao_integral_value);
      if (left_second_basis != left_first_basis) {
        double* left_second_gradient =
            dense_active_gradients->data() +
            xmvb::to_size(left_second_basis) * n_active_orbitals;
        const double* left_first_basis_backprop_rows =
            accepted_backprop_rows +
            xmvb::to_size(left_first_basis) * n_active_pairs * n_active_orbitals;
        accumulate_scaled_cached_backprop_rows_to_active_gradient_dynamic(
            left_second_gradient,
            left_first_basis_backprop_rows,
            right_row,
            n_active_pairs,
            n_active_orbitals,
            ao_integral_value);
      }

      if (left_pair_index == right_pair_index) {
        continue;
      }

      const int right_first_basis =
          cache.ao_pair_first_indices[right_pair_index];
      const int right_second_basis =
          cache.ao_pair_second_indices[right_pair_index];
      double* right_first_gradient =
          dense_active_gradients->data() +
          xmvb::to_size(right_first_basis) * n_active_orbitals;
      const double* right_second_basis_backprop_rows =
          accepted_backprop_rows +
          xmvb::to_size(right_second_basis) * n_active_pairs * n_active_orbitals;
      const double* left_row =
          pair_source_coefficients.data() +
          left_pair_index * n_active_pairs;
      accumulate_scaled_cached_backprop_rows_to_active_gradient_dynamic(
          right_first_gradient,
          right_second_basis_backprop_rows,
          left_row,
          n_active_pairs,
          n_active_orbitals,
          ao_integral_value);
      if (right_second_basis != right_first_basis) {
        double* right_second_gradient =
            dense_active_gradients->data() +
            xmvb::to_size(right_second_basis) * n_active_orbitals;
        const double* right_first_basis_backprop_rows =
            accepted_backprop_rows +
            xmvb::to_size(right_first_basis) * n_active_pairs * n_active_orbitals;
        accumulate_scaled_cached_backprop_rows_to_active_gradient_dynamic(
            right_second_gradient,
            right_first_basis_backprop_rows,
            left_row,
            n_active_pairs,
            n_active_orbitals,
            ao_integral_value);
      }
    }
    return;
  }

  std::vector<double> partial_dense_active_gradients(
      xmvb::to_size(n_threads) * dense_size,
      0.0);
  std::atomic<int> invalid_integral_index(-1);
#pragma omp parallel
  {
    int thread_index = 0;
#ifdef _OPENMP
    thread_index = omp_get_thread_num();
#endif
    double* local_dense_gradients =
        partial_dense_active_gradients.data() +
        xmvb::to_size(thread_index) * dense_size;

#pragma omp for schedule(guided, 256)
    for (std::ptrdiff_t integral_offset = 0;
         integral_offset <
             static_cast<std::ptrdiff_t>(ao_two_electron_integral_values.size());
         ++integral_offset) {
      const std::size_t integral_index = xmvb::to_size(integral_offset);
      const double ao_integral_value =
          ao_two_electron_integral_values[integral_index];
      const int left_pair_index =
          ao_two_electron_pair_indices[integral_index * 2];
      const int right_pair_index =
          ao_two_electron_pair_indices[integral_index * 2 + 1];
      if (left_pair_index < 0 || right_pair_index < 0 ||
          left_pair_index >= static_cast<int>(n_ao_pairs) ||
          right_pair_index >= static_cast<int>(n_ao_pairs)) {
        int expected = -1;
        invalid_integral_index.compare_exchange_strong(
            expected,
            static_cast<int>(integral_index));
        continue;
      }

      const int left_first_basis =
          cache.ao_pair_first_indices[xmvb::to_size(left_pair_index)];
      const int left_second_basis =
          cache.ao_pair_second_indices[xmvb::to_size(left_pair_index)];
      double* left_first_gradient =
          local_dense_gradients +
          xmvb::to_size(left_first_basis) * n_active_orbitals;
      const double* left_second_basis_backprop_rows =
          accepted_backprop_rows +
          xmvb::to_size(left_second_basis) * n_active_pairs * n_active_orbitals;
      const double* right_row =
          pair_source_coefficients.data() +
          xmvb::to_size(right_pair_index) * n_active_pairs;
      accumulate_scaled_cached_backprop_rows_to_active_gradient_dynamic(
          left_first_gradient,
          left_second_basis_backprop_rows,
          right_row,
          n_active_pairs,
          n_active_orbitals,
          ao_integral_value);
      if (left_second_basis != left_first_basis) {
        double* left_second_gradient =
            local_dense_gradients +
            xmvb::to_size(left_second_basis) * n_active_orbitals;
        const double* left_first_basis_backprop_rows =
            accepted_backprop_rows +
            xmvb::to_size(left_first_basis) * n_active_pairs * n_active_orbitals;
        accumulate_scaled_cached_backprop_rows_to_active_gradient_dynamic(
            left_second_gradient,
            left_first_basis_backprop_rows,
            right_row,
            n_active_pairs,
            n_active_orbitals,
            ao_integral_value);
      }

      if (left_pair_index == right_pair_index) {
        continue;
      }

      const int right_first_basis =
          cache.ao_pair_first_indices[xmvb::to_size(right_pair_index)];
      const int right_second_basis =
          cache.ao_pair_second_indices[xmvb::to_size(right_pair_index)];
      double* right_first_gradient =
          local_dense_gradients +
          xmvb::to_size(right_first_basis) * n_active_orbitals;
      const double* right_second_basis_backprop_rows =
          accepted_backprop_rows +
          xmvb::to_size(right_second_basis) * n_active_pairs * n_active_orbitals;
      const double* left_row =
          pair_source_coefficients.data() +
          xmvb::to_size(left_pair_index) * n_active_pairs;
      accumulate_scaled_cached_backprop_rows_to_active_gradient_dynamic(
          right_first_gradient,
          right_second_basis_backprop_rows,
          left_row,
          n_active_pairs,
          n_active_orbitals,
          ao_integral_value);
      if (right_second_basis != right_first_basis) {
        double* right_second_gradient =
            local_dense_gradients +
            xmvb::to_size(right_second_basis) * n_active_orbitals;
        const double* right_first_basis_backprop_rows =
            accepted_backprop_rows +
            xmvb::to_size(right_first_basis) * n_active_pairs * n_active_orbitals;
        accumulate_scaled_cached_backprop_rows_to_active_gradient_dynamic(
            right_second_gradient,
            right_first_basis_backprop_rows,
            left_row,
            n_active_pairs,
            n_active_orbitals,
            ao_integral_value);
      }
    }
  }

  if (invalid_integral_index.load() >= 0) {
    throw std::invalid_argument("packed AO pair index out of range");
  }
  reduce_partial_dense_active_gradients(
      partial_dense_active_gradients,
      n_threads,
      dense_size,
      dense_active_gradients);
}

template <int NActiveOrbitals>
void apply_ao_pair_graph_matrix_and_backprop_single_thread_fixed_hvp(
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& row_offsets,
    const std::vector<int>& column_pair_indices,
    const std::vector<int>& integral_indices,
    const double* transformed_pair_coefficients_data,
    const ExactPackedActiveTwoElectronAdjointCache& cache,
    std::vector<double>* dense_active_gradients) {
  constexpr std::size_t kNActivePairs =
      xmvb::to_size(NActiveOrbitals) * (NActiveOrbitals + 1) / 2;
  if (dense_active_gradients == nullptr) {
    throw std::invalid_argument("dense active gradient output must not be null");
  }
  if (cache.ao_pair_first_indices.size() + 1 != row_offsets.size() ||
      column_pair_indices.size() != integral_indices.size()) {
    throw std::invalid_argument("AO pair graph/cache size mismatch");
  }
  if (dense_active_gradients->size() !=
      xmvb::to_size(cache.n_basis_functions) * NActiveOrbitals) {
    throw std::invalid_argument("dense active gradient output size mismatch");
  }
  const double* dense_active_coefficients_data =
      cache.accepted_dense_active_coefficients_buffer.data();

  for (std::size_t row_index = 0;
       row_index < cache.ao_pair_first_indices.size();
       ++row_index) {
    const int first_basis =
        cache.ao_pair_first_indices[row_index];
    const int second_basis =
        cache.ao_pair_second_indices[row_index];
    double* first_gradient =
        dense_active_gradients->data() +
        xmvb::to_size(first_basis) * NActiveOrbitals;
    const double* second_coefficients =
        dense_active_coefficients_data +
        xmvb::to_size(second_basis) * NActiveOrbitals;
    double* second_gradient =
        dense_active_gradients->data() +
        xmvb::to_size(second_basis) * NActiveOrbitals;
    const double* first_coefficients =
        dense_active_coefficients_data +
        xmvb::to_size(first_basis) * NActiveOrbitals;
    const int begin = row_offsets[row_index];
    const int end = row_offsets[row_index + 1];
    for (int entry_offset = begin; entry_offset < end; ++entry_offset) {
      const std::size_t column_pair_index = xmvb::to_size(
          column_pair_indices[xmvb::to_size(entry_offset)]);
      const int integral_index =
          integral_indices[xmvb::to_size(entry_offset)];
      const double ao_integral_value =
          ao_two_electron_integral_values[xmvb::to_size(integral_index)];
      const double* source_row =
          transformed_pair_coefficients_data +
          column_pair_index * kNActivePairs;
      accumulate_scaled_packed_active_pair_gradient_row_to_active_gradient<
          NActiveOrbitals>(
          first_gradient,
          source_row,
          second_coefficients,
          ao_integral_value);
      if (second_basis != first_basis) {
        accumulate_scaled_packed_active_pair_gradient_row_to_active_gradient<
            NActiveOrbitals>(
            second_gradient,
            source_row,
            first_coefficients,
            ao_integral_value);
      }
    }
  }
}

template <int NActiveOrbitals>
void apply_ao_pair_graph_matrix_and_backprop_from_cached_rows_fixed(
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& row_offsets,
    const std::vector<int>& column_pair_indices,
    const std::vector<int>& integral_indices,
    const double* pair_source_coefficients_data,
    const ExactPackedActiveTwoElectronAdjointCache& cache,
    std::vector<double>* dense_active_gradients) {
  constexpr std::size_t kNActivePairs =
      xmvb::to_size(NActiveOrbitals) * (NActiveOrbitals + 1) / 2;
  const int n_basis_functions = cache.n_basis_functions;
  const std::size_t n_ao_pairs = cache.ao_pair_first_indices.size();
  const std::size_t dense_size =
      xmvb::to_size(n_basis_functions) * NActiveOrbitals;
  const std::size_t expected_backprop_size =
      xmvb::to_size(n_basis_functions) * kNActivePairs * NActiveOrbitals;
  if (dense_active_gradients == nullptr) {
    throw std::invalid_argument("dense active gradient output must not be null");
  }
  if (row_offsets.size() != n_ao_pairs + 1 ||
      column_pair_indices.size() != integral_indices.size() ||
      cache.ao_pair_second_indices.size() != n_ao_pairs ||
      cache.accepted_active_pair_gradient_backprop_rows_buffer.size() !=
          expected_backprop_size) {
    throw std::invalid_argument(
        "cached-row AO pair graph / cache size mismatch");
  }
  if (dense_active_gradients->size() != dense_size) {
    throw std::invalid_argument("dense active gradient output size mismatch");
  }
  const double* accepted_backprop_rows =
      cache.accepted_active_pair_gradient_backprop_rows_buffer.data();
  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif
  if (n_threads <= 1) {
    for (std::size_t row_index = 0; row_index < n_ao_pairs; ++row_index) {
      const int first_basis =
          cache.ao_pair_first_indices[row_index];
      const int second_basis =
          cache.ao_pair_second_indices[row_index];
      double* first_gradient =
          dense_active_gradients->data() +
          xmvb::to_size(first_basis) * NActiveOrbitals;
      const double* second_basis_backprop_rows =
          accepted_backprop_rows +
          xmvb::to_size(second_basis) * kNActivePairs * NActiveOrbitals;
      double* second_gradient =
          dense_active_gradients->data() +
          xmvb::to_size(second_basis) * NActiveOrbitals;
      const double* first_basis_backprop_rows =
          accepted_backprop_rows +
          xmvb::to_size(first_basis) * kNActivePairs * NActiveOrbitals;
      const int begin = row_offsets[row_index];
      const int end = row_offsets[row_index + 1];
      for (int entry_offset = begin; entry_offset < end; ++entry_offset) {
        const std::size_t column_pair_index = xmvb::to_size(
            column_pair_indices[xmvb::to_size(entry_offset)]);
        const int integral_index =
            integral_indices[xmvb::to_size(entry_offset)];
        const double ao_integral_value =
            ao_two_electron_integral_values[xmvb::to_size(integral_index)];
        const double* source_row =
            pair_source_coefficients_data +
            column_pair_index * kNActivePairs;
        accumulate_scaled_cached_backprop_rows_to_active_gradient<
            NActiveOrbitals>(
            first_gradient,
            second_basis_backprop_rows,
            source_row,
            ao_integral_value);
        if (second_basis != first_basis) {
          accumulate_scaled_cached_backprop_rows_to_active_gradient<
              NActiveOrbitals>(
              second_gradient,
              first_basis_backprop_rows,
              source_row,
              ao_integral_value);
        }
      }
    }
    return;
  }

  apply_cached_row_graph_to_owned_basis_rows(
      n_basis_functions,
      n_threads,
      [&](int target_basis) {
        double* target_gradient =
            dense_active_gradients->data() +
            xmvb::to_size(target_basis) * NActiveOrbitals;
        accumulate_cached_row_graph_to_basis_gradient_fixed<NActiveOrbitals>(
            target_basis,
            n_basis_functions,
            row_offsets,
            column_pair_indices,
            integral_indices,
            ao_two_electron_integral_values,
            pair_source_coefficients_data,
            accepted_backprop_rows,
            target_gradient);
      });
}

void apply_ao_pair_graph_matrix_and_backprop_from_cached_rows(
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& row_offsets,
    const std::vector<int>& column_pair_indices,
    const std::vector<int>& integral_indices,
    const std::vector<double>& pair_source_coefficients,
    const ExactPackedActiveTwoElectronAdjointCache& cache,
    std::vector<double>* dense_active_gradients) {
  const int n_basis_functions = cache.n_basis_functions;
  const int n_active_orbitals = cache.n_active_orbitals;
  const std::size_t n_active_pairs = cache.active_pair_first_indices.size();
  const std::size_t n_ao_pairs =
      xmvb::to_size(n_basis_functions) * (n_basis_functions + 1) / 2;
  const std::size_t dense_size =
      xmvb::to_size(n_basis_functions) * n_active_orbitals;
  const std::size_t expected_backprop_size =
      xmvb::to_size(n_basis_functions) * n_active_pairs * n_active_orbitals;
  if (pair_source_coefficients.size() != n_ao_pairs * n_active_pairs ||
      row_offsets.size() != n_ao_pairs + 1 ||
      column_pair_indices.size() != integral_indices.size() ||
      cache.ao_pair_first_indices.size() != n_ao_pairs ||
      cache.ao_pair_second_indices.size() != n_ao_pairs ||
      cache.accepted_active_pair_gradient_backprop_rows_buffer.size() !=
          expected_backprop_size ||
      dense_active_gradients == nullptr) {
    throw std::invalid_argument(
        "cached-row AO pair graph backprop input size mismatch");
  }

  switch (n_active_orbitals) {
    case 1:
      apply_ao_pair_graph_matrix_and_backprop_from_cached_rows_fixed<1>(
          ao_two_electron_integral_values,
          row_offsets,
          column_pair_indices,
          integral_indices,
          pair_source_coefficients.data(),
          cache,
          dense_active_gradients);
      return;
    case 2:
      apply_ao_pair_graph_matrix_and_backprop_from_cached_rows_fixed<2>(
          ao_two_electron_integral_values,
          row_offsets,
          column_pair_indices,
          integral_indices,
          pair_source_coefficients.data(),
          cache,
          dense_active_gradients);
      return;
    case 3:
      apply_ao_pair_graph_matrix_and_backprop_from_cached_rows_fixed<3>(
          ao_two_electron_integral_values,
          row_offsets,
          column_pair_indices,
          integral_indices,
          pair_source_coefficients.data(),
          cache,
          dense_active_gradients);
      return;
    case 4:
      apply_ao_pair_graph_matrix_and_backprop_from_cached_rows_fixed<4>(
          ao_two_electron_integral_values,
          row_offsets,
          column_pair_indices,
          integral_indices,
          pair_source_coefficients.data(),
          cache,
          dense_active_gradients);
      return;
    case 5:
      apply_ao_pair_graph_matrix_and_backprop_from_cached_rows_fixed<5>(
          ao_two_electron_integral_values,
          row_offsets,
          column_pair_indices,
          integral_indices,
          pair_source_coefficients.data(),
          cache,
          dense_active_gradients);
      return;
    case 6:
      apply_ao_pair_graph_matrix_and_backprop_from_cached_rows_fixed<6>(
          ao_two_electron_integral_values,
          row_offsets,
          column_pair_indices,
          integral_indices,
          pair_source_coefficients.data(),
          cache,
          dense_active_gradients);
      return;
    case 7:
      apply_ao_pair_graph_matrix_and_backprop_from_cached_rows_fixed<7>(
          ao_two_electron_integral_values,
          row_offsets,
          column_pair_indices,
          integral_indices,
          pair_source_coefficients.data(),
          cache,
          dense_active_gradients);
      return;
    case 8:
      apply_ao_pair_graph_matrix_and_backprop_from_cached_rows_fixed<8>(
          ao_two_electron_integral_values,
          row_offsets,
          column_pair_indices,
          integral_indices,
          pair_source_coefficients.data(),
          cache,
          dense_active_gradients);
      return;
    case 9:
      apply_ao_pair_graph_matrix_and_backprop_from_cached_rows_fixed<9>(
          ao_two_electron_integral_values,
          row_offsets,
          column_pair_indices,
          integral_indices,
          pair_source_coefficients.data(),
          cache,
          dense_active_gradients);
      return;
    case 10:
      apply_ao_pair_graph_matrix_and_backprop_from_cached_rows_fixed<10>(
          ao_two_electron_integral_values,
          row_offsets,
          column_pair_indices,
          integral_indices,
          pair_source_coefficients.data(),
          cache,
          dense_active_gradients);
      return;
    default:
      break;
  }

  if (dense_active_gradients->size() != dense_size) {
    throw std::invalid_argument("dense active gradient output size mismatch");
  }
  const double* accepted_backprop_rows =
      cache.accepted_active_pair_gradient_backprop_rows_buffer.data();
  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif
  if (n_threads <= 1) {
    for (std::size_t row_index = 0; row_index < n_ao_pairs; ++row_index) {
      const int first_basis =
          cache.ao_pair_first_indices[row_index];
      const int second_basis =
          cache.ao_pair_second_indices[row_index];
      double* first_gradient =
          dense_active_gradients->data() +
          xmvb::to_size(first_basis) * n_active_orbitals;
      const double* second_basis_backprop_rows =
          accepted_backprop_rows +
          xmvb::to_size(second_basis) * n_active_pairs * n_active_orbitals;
      double* second_gradient =
          dense_active_gradients->data() +
          xmvb::to_size(second_basis) * n_active_orbitals;
      const double* first_basis_backprop_rows =
          accepted_backprop_rows +
          xmvb::to_size(first_basis) * n_active_pairs * n_active_orbitals;
      const int begin = row_offsets[row_index];
      const int end = row_offsets[row_index + 1];
      for (int entry_offset = begin; entry_offset < end; ++entry_offset) {
        const std::size_t column_pair_index = xmvb::to_size(
            column_pair_indices[xmvb::to_size(entry_offset)]);
        const int integral_index =
            integral_indices[xmvb::to_size(entry_offset)];
        const double ao_integral_value =
            ao_two_electron_integral_values[xmvb::to_size(integral_index)];
        const double* source_row =
            pair_source_coefficients.data() +
            column_pair_index * n_active_pairs;
        accumulate_scaled_cached_backprop_rows_to_active_gradient_dynamic(
            first_gradient,
            second_basis_backprop_rows,
            source_row,
            n_active_pairs,
            n_active_orbitals,
            ao_integral_value);
        if (second_basis != first_basis) {
          accumulate_scaled_cached_backprop_rows_to_active_gradient_dynamic(
              second_gradient,
              first_basis_backprop_rows,
              source_row,
              n_active_pairs,
              n_active_orbitals,
              ao_integral_value);
        }
      }
    }
    return;
  }

  apply_cached_row_graph_to_owned_basis_rows(
      n_basis_functions,
      n_threads,
      [&](int target_basis) {
        double* target_gradient =
            dense_active_gradients->data() +
            xmvb::to_size(target_basis) * n_active_orbitals;
        accumulate_cached_row_graph_to_basis_gradient_dynamic(
            target_basis,
            n_basis_functions,
            n_active_orbitals,
            n_active_pairs,
            row_offsets,
            column_pair_indices,
            integral_indices,
            ao_two_electron_integral_values,
            pair_source_coefficients,
            accepted_backprop_rows,
            target_gradient);
      });
}

std::vector<ActivePair> build_active_pair_list(int n_active_orbitals) {
  std::vector<ActivePair> active_pairs;
  active_pairs.reserve(
      xmvb::to_size(n_active_orbitals) * (n_active_orbitals + 1) / 2);
  for (int first = 0; first < n_active_orbitals; ++first) {
    for (int second = 0; second <= first; ++second) {
      active_pairs.push_back({first, second});
    }
  }
  return active_pairs;
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
          packed_active_two_electron_gradient[xmvb::to_size(packed_index)];
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
    const std::vector<double>& dense_active_coefficients,
    int n_basis_functions,
    int n_active_orbitals,
    const std::vector<ActivePair>& active_pairs,
    std::vector<double>* ao_pair_to_active_pair_coefficients) {
  const std::size_t n_ao_pairs =
      xmvb::to_size(n_basis_functions) * (n_basis_functions + 1) / 2;
  const std::size_t n_active_pairs = active_pairs.size();
  resize_for_overwrite(
      ao_pair_to_active_pair_coefficients,
      n_ao_pairs * n_active_pairs);

#pragma omp parallel for schedule(static)
  for (int first_basis_function = 0;
       first_basis_function < n_basis_functions;
       ++first_basis_function) {
    const double* first_coefficients =
        dense_active_coefficients.data() +
        xmvb::to_size(first_basis_function) * n_active_orbitals;
    for (int second_basis_function = 0;
         second_basis_function <= first_basis_function;
         ++second_basis_function) {
      const double* second_coefficients =
          dense_active_coefficients.data() +
          xmvb::to_size(second_basis_function) * n_active_orbitals;
      double* pair_coefficients =
          ao_pair_to_active_pair_coefficients->data() +
          ao_pair_index(first_basis_function, second_basis_function) * n_active_pairs;
      for (std::size_t active_pair_index_offset = 0;
           active_pair_index_offset < n_active_pairs;
           ++active_pair_index_offset) {
        const auto& active_pair = active_pairs[active_pair_index_offset];
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
}

std::vector<double> build_ao_pair_to_active_pair_coefficients(
    const std::vector<double>& dense_active_coefficients,
    int n_basis_functions,
    int n_active_orbitals,
    const std::vector<ActivePair>& active_pairs) {
  std::vector<double> ao_pair_to_active_pair_coefficients;
  build_ao_pair_to_active_pair_coefficients(
      dense_active_coefficients,
      n_basis_functions,
      n_active_orbitals,
      active_pairs,
      &ao_pair_to_active_pair_coefficients);
  return ao_pair_to_active_pair_coefficients;
}

void build_mixed_ao_pair_to_active_pair_coefficients(
    const std::vector<double>& dense_active_coefficients,
    const std::vector<double>& dense_active_direction,
    int n_basis_functions,
    int n_active_orbitals,
    const std::vector<ActivePair>& active_pairs,
    std::vector<double>* mixed_ao_pair_to_active_pair_coefficients) {
  const std::size_t n_ao_pairs =
      xmvb::to_size(n_basis_functions) * (n_basis_functions + 1) / 2;
  const std::size_t n_active_pairs = active_pairs.size();
  resize_for_overwrite(
      mixed_ao_pair_to_active_pair_coefficients,
      n_ao_pairs * n_active_pairs);

#pragma omp parallel for schedule(static)
  for (int first_basis_function = 0;
       first_basis_function < n_basis_functions;
       ++first_basis_function) {
    const double* first_coefficients =
        dense_active_coefficients.data() +
        xmvb::to_size(first_basis_function) * n_active_orbitals;
    const double* first_directions =
        dense_active_direction.data() +
        xmvb::to_size(first_basis_function) * n_active_orbitals;
    for (int second_basis_function = 0;
         second_basis_function <= first_basis_function;
         ++second_basis_function) {
      const double* second_coefficients =
          dense_active_coefficients.data() +
          xmvb::to_size(second_basis_function) * n_active_orbitals;
      const double* second_directions =
          dense_active_direction.data() +
          xmvb::to_size(second_basis_function) * n_active_orbitals;
      double* pair_coefficients =
          mixed_ao_pair_to_active_pair_coefficients->data() +
          ao_pair_index(first_basis_function, second_basis_function) * n_active_pairs;
      for (std::size_t active_pair_index_offset = 0;
           active_pair_index_offset < n_active_pairs;
           ++active_pair_index_offset) {
        const auto& active_pair = active_pairs[active_pair_index_offset];
        double coefficient =
            first_directions[active_pair.first] *
                second_coefficients[active_pair.second] +
            first_coefficients[active_pair.first] *
                second_directions[active_pair.second];
        if (first_basis_function != second_basis_function) {
          coefficient +=
              second_directions[active_pair.first] *
                  first_coefficients[active_pair.second] +
              second_coefficients[active_pair.first] *
                  first_directions[active_pair.second];
        }
        pair_coefficients[active_pair_index_offset] = coefficient;
      }
    }
  }
}

std::vector<double> build_mixed_ao_pair_to_active_pair_coefficients(
    const std::vector<double>& dense_active_coefficients,
    const std::vector<double>& dense_active_direction,
    int n_basis_functions,
    int n_active_orbitals,
    const std::vector<ActivePair>& active_pairs) {
  std::vector<double> mixed_ao_pair_to_active_pair_coefficients;
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
    const std::vector<double>& dense_active_coefficients,
    const std::vector<double>& dense_active_direction,
    const ExactPackedActiveTwoElectronAdjointCache& cache,
    std::vector<double>* mixed_ao_pair_to_active_pair_coefficients) {
  const int n_basis_functions = cache.n_basis_functions;
  const int n_active_orbitals = cache.n_active_orbitals;
  const std::size_t n_active_pairs = cache.active_pair_first_indices.size();
  if (cache.active_pair_second_indices.size() != n_active_pairs) {
    throw std::invalid_argument("exact 2e cache active-pair index size mismatch");
  }
  const std::size_t expected_dense_size =
      xmvb::to_size(n_basis_functions) * n_active_orbitals;
  if (dense_active_coefficients.size() != expected_dense_size ||
      dense_active_direction.size() != expected_dense_size) {
    throw std::invalid_argument("dense active coefficient size mismatch");
  }

  switch (n_active_orbitals) {
    case 1:
      build_mixed_ao_pair_to_active_pair_coefficients_from_cache_fixed<1>(
          dense_active_coefficients,
          dense_active_direction,
          n_basis_functions,
          mixed_ao_pair_to_active_pair_coefficients);
      return;
    case 2:
      build_mixed_ao_pair_to_active_pair_coefficients_from_cache_fixed<2>(
          dense_active_coefficients,
          dense_active_direction,
          n_basis_functions,
          mixed_ao_pair_to_active_pair_coefficients);
      return;
    case 3:
      build_mixed_ao_pair_to_active_pair_coefficients_from_cache_fixed<3>(
          dense_active_coefficients,
          dense_active_direction,
          n_basis_functions,
          mixed_ao_pair_to_active_pair_coefficients);
      return;
    case 4:
      build_mixed_ao_pair_to_active_pair_coefficients_from_cache_fixed<4>(
          dense_active_coefficients,
          dense_active_direction,
          n_basis_functions,
          mixed_ao_pair_to_active_pair_coefficients);
      return;
    case 5:
      build_mixed_ao_pair_to_active_pair_coefficients_from_cache_fixed<5>(
          dense_active_coefficients,
          dense_active_direction,
          n_basis_functions,
          mixed_ao_pair_to_active_pair_coefficients);
      return;
    case 6:
      build_mixed_ao_pair_to_active_pair_coefficients_from_cache_fixed<6>(
          dense_active_coefficients,
          dense_active_direction,
          n_basis_functions,
          mixed_ao_pair_to_active_pair_coefficients);
      return;
    case 7:
      build_mixed_ao_pair_to_active_pair_coefficients_from_cache_fixed<7>(
          dense_active_coefficients,
          dense_active_direction,
          n_basis_functions,
          mixed_ao_pair_to_active_pair_coefficients);
      return;
    case 8:
      build_mixed_ao_pair_to_active_pair_coefficients_from_cache_fixed<8>(
          dense_active_coefficients,
          dense_active_direction,
          n_basis_functions,
          mixed_ao_pair_to_active_pair_coefficients);
      return;
    case 9:
      build_mixed_ao_pair_to_active_pair_coefficients_from_cache_fixed<9>(
          dense_active_coefficients,
          dense_active_direction,
          n_basis_functions,
          mixed_ao_pair_to_active_pair_coefficients);
      return;
    case 10:
      build_mixed_ao_pair_to_active_pair_coefficients_from_cache_fixed<10>(
          dense_active_coefficients,
          dense_active_direction,
          n_basis_functions,
          mixed_ao_pair_to_active_pair_coefficients);
      return;
    default:
      break;
  }

  const std::size_t n_ao_pairs =
      xmvb::to_size(n_basis_functions) * (n_basis_functions + 1) / 2;
  resize_for_overwrite(
      mixed_ao_pair_to_active_pair_coefficients,
      n_ao_pairs * n_active_pairs);

#pragma omp parallel for schedule(static)
  for (int first_basis_function = 0;
       first_basis_function < n_basis_functions;
       ++first_basis_function) {
    const double* first_coefficients =
        dense_active_coefficients.data() +
        xmvb::to_size(first_basis_function) * n_active_orbitals;
    const double* first_directions =
        dense_active_direction.data() +
        xmvb::to_size(first_basis_function) * n_active_orbitals;
    for (int second_basis_function = 0;
         second_basis_function <= first_basis_function;
         ++second_basis_function) {
      const double* second_coefficients =
          dense_active_coefficients.data() +
          xmvb::to_size(second_basis_function) * n_active_orbitals;
      const double* second_directions =
          dense_active_direction.data() +
          xmvb::to_size(second_basis_function) * n_active_orbitals;
      double* pair_coefficients =
          mixed_ao_pair_to_active_pair_coefficients->data() +
          ao_pair_index(first_basis_function, second_basis_function) * n_active_pairs;
      for (std::size_t active_pair_index = 0;
           active_pair_index < n_active_pairs;
           ++active_pair_index) {
        const int first_active =
            cache.active_pair_first_indices[active_pair_index];
        const int second_active =
            cache.active_pair_second_indices[active_pair_index];
        double coefficient =
            first_directions[first_active] * second_coefficients[second_active] +
            first_coefficients[first_active] * second_directions[second_active];
        if (first_basis_function != second_basis_function) {
          coefficient +=
              second_directions[first_active] * first_coefficients[second_active] +
              second_coefficients[first_active] * first_directions[second_active];
        }
        pair_coefficients[active_pair_index] = coefficient;
      }
    }
  }
}

std::vector<double> build_mixed_ao_pair_to_active_pair_coefficients_from_cache(
    const std::vector<double>& dense_active_coefficients,
    const std::vector<double>& dense_active_direction,
    const ExactPackedActiveTwoElectronAdjointCache& cache) {
  std::vector<double> mixed_ao_pair_to_active_pair_coefficients;
  build_mixed_ao_pair_to_active_pair_coefficients_from_cache(
      dense_active_coefficients,
      dense_active_direction,
      cache,
      &mixed_ao_pair_to_active_pair_coefficients);
  return mixed_ao_pair_to_active_pair_coefficients;
}
void multiply_pair_coefficients_by_gradient_matrix(
    const std::vector<double>& ao_pair_to_active_pair_coefficients,
    const Eigen::Ref<const Eigen::MatrixXd>& active_pair_gradient_matrix,
    const std::vector<double>& active_pair_gradient_buffer,
    std::size_t n_ao_pairs,
    std::size_t n_active_pairs,
    std::vector<double>* transformed_pair_coefficients) {
  if (ao_pair_to_active_pair_coefficients.size() !=
      n_ao_pairs * n_active_pairs) {
    throw std::invalid_argument("AO-pair coefficient matrix shape mismatch");
  }
  if (active_pair_gradient_matrix.rows() != static_cast<Eigen::Index>(n_active_pairs) ||
      active_pair_gradient_matrix.cols() != static_cast<Eigen::Index>(n_active_pairs)) {
    throw std::invalid_argument("active-pair gradient matrix shape mismatch");
  }
  resize_for_overwrite(
      transformed_pair_coefficients,
      n_ao_pairs * n_active_pairs);

  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif
  if (n_threads <= 1) {
    if (active_pair_gradient_buffer.size() !=
        n_active_pairs * n_active_pairs) {
      throw std::invalid_argument("active-pair gradient buffer shape mismatch");
    }
    if (n_ao_pairs > xmvb::to_size(std::numeric_limits<int>::max()) ||
        n_active_pairs > xmvb::to_size(std::numeric_limits<int>::max())) {
      throw std::invalid_argument("BLAS pair-gradient multiply dimension overflow");
    }

    // In single-threaded exact-2e applications the accepted-point pair-gradient
    // contraction is dense and contiguous on both operands. Use one BLAS call
    // at the legacy row-buffer boundary instead of dispatching thousands of
    // tiny active-pair matvecs.
    const int row_count = static_cast<int>(n_ao_pairs);
    const int inner_count = static_cast<int>(n_active_pairs);
    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        row_count,
        inner_count,
        inner_count,
        1.0,
        ao_pair_to_active_pair_coefficients.data(),
        inner_count,
        active_pair_gradient_buffer.data(),
        inner_count,
        0.0,
        transformed_pair_coefficients->data(),
        inner_count);
    return;
  }

#pragma omp parallel for schedule(static)
  for (std::ptrdiff_t ao_pair_offset = 0;
       ao_pair_offset < static_cast<std::ptrdiff_t>(n_ao_pairs);
       ++ao_pair_offset) {
    const std::size_t ao_pair_index = xmvb::to_size(ao_pair_offset);
    const Eigen::Map<const Eigen::VectorXd> pair_coefficient_row(
        ao_pair_to_active_pair_coefficients.data() +
            ao_pair_index * n_active_pairs,
        static_cast<Eigen::Index>(n_active_pairs));
    Eigen::Map<Eigen::VectorXd> transformed_pair_coefficient_row(
        transformed_pair_coefficients->data() +
            ao_pair_index * n_active_pairs,
        static_cast<Eigen::Index>(n_active_pairs));
    transformed_pair_coefficient_row.noalias() =
        active_pair_gradient_matrix.transpose() * pair_coefficient_row;
  }
}

std::vector<double> multiply_pair_coefficients_by_gradient_matrix(
    const std::vector<double>& ao_pair_to_active_pair_coefficients,
    const Eigen::Ref<const Eigen::MatrixXd>& active_pair_gradient_matrix,
    const std::vector<double>& active_pair_gradient_buffer,
    std::size_t n_ao_pairs,
    std::size_t n_active_pairs) {
  std::vector<double> transformed_pair_coefficients;
  multiply_pair_coefficients_by_gradient_matrix(
      ao_pair_to_active_pair_coefficients,
      active_pair_gradient_matrix,
      active_pair_gradient_buffer,
      n_ao_pairs,
      n_active_pairs,
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
      xmvb::to_size(n_basis_functions) * (n_basis_functions + 1) / 2;
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
  n_threads = omp_get_max_threads();
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
      const std::size_t left_pair_index = xmvb::to_size(
          ao_two_electron_pair_indices_data[integral_index * 2]);
      const std::size_t right_pair_index = xmvb::to_size(
          ao_two_electron_pair_indices_data[integral_index * 2 + 1]);
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
      xmvb::to_size(n_threads),
      std::vector<double>(n_ao_pairs * n_active_pairs, 0.0));

#pragma omp parallel
  {
    int thread_index = 0;
#ifdef _OPENMP
    thread_index = omp_get_thread_num();
#endif
    auto& local_pair_gradients =
        partial_pair_gradients[xmvb::to_size(thread_index)];

#pragma omp for schedule(guided, 256)
    for (std::ptrdiff_t integral_offset = 0;
         integral_offset <
             static_cast<std::ptrdiff_t>(ao_two_electron_integral_values.size());
         ++integral_offset) {
      const std::size_t integral_index = xmvb::to_size(integral_offset);
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
          pair_row_offsets_data[xmvb::to_size(right_pair_index)];
      double* left_gradient_row =
          local_pair_gradients.data() +
          pair_row_offsets_data[xmvb::to_size(left_pair_index)];
      for (std::size_t active_pair_index = 0;
           active_pair_index < n_active_pairs;
           ++active_pair_index) {
        left_gradient_row[active_pair_index] +=
            ao_integral_value * right_row[active_pair_index];
      }

      if (left_pair_index != right_pair_index) {
        const double* left_row =
            transformed_pair_coefficients_data +
            pair_row_offsets_data[xmvb::to_size(left_pair_index)];
        double* right_gradient_row =
            local_pair_gradients.data() +
            pair_row_offsets_data[xmvb::to_size(right_pair_index)];
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
      xmvb::to_size(n_basis_functions) * (n_basis_functions + 1) / 2;
  std::vector<std::vector<double>> partial_pair_gradients;
  std::atomic<int> invalid_integral_index(-1);
  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif
  partial_pair_gradients.assign(
      xmvb::to_size(n_threads),
      std::vector<double>(n_ao_pairs * n_active_pairs, 0.0));

#pragma omp parallel
  {
    int thread_index = 0;
#ifdef _OPENMP
    thread_index = omp_get_thread_num();
#endif
    auto& local_pair_gradients =
        partial_pair_gradients[xmvb::to_size(thread_index)];

#pragma omp for schedule(guided, 256)
    for (std::ptrdiff_t integral_offset = 0;
         integral_offset <
             static_cast<std::ptrdiff_t>(ao_two_electron_integral_values.size());
         ++integral_offset) {
      const std::size_t integral_index = xmvb::to_size(integral_offset);
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
  n_threads = omp_get_max_threads();
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

  resize_and_zero(
      pair_gradients,
      n_ao_pairs * n_active_pairs);
#pragma omp parallel for schedule(guided, 64)
  for (std::ptrdiff_t row_offset = 0;
       row_offset < static_cast<std::ptrdiff_t>(n_ao_pairs);
       ++row_offset) {
    const std::size_t row_index = xmvb::to_size(row_offset);
    double* target_row =
        pair_gradients->data() + pair_row_offsets_data[row_index];
    const int begin = row_offsets[row_index];
    const int end = row_offsets[row_index + 1];
    for (int entry_offset = begin; entry_offset < end; ++entry_offset) {
      const int column_pair_index =
          column_pair_indices[xmvb::to_size(entry_offset)];
      const int integral_index =
          integral_indices[xmvb::to_size(entry_offset)];
      const double ao_integral_value =
          ao_two_electron_integral_values[xmvb::to_size(integral_index)];
      const double* source_row =
          transformed_pair_coefficients.data() +
          pair_row_offsets_data[xmvb::to_size(column_pair_index)];
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
      xmvb::to_size(n_basis_functions) * (n_basis_functions + 1) / 2;
  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif
  if (!ao_integral_input.ao_two_electron_pair_graph_row_offsets.empty()) {
    // The row-oriented AO-pair graph is the block-contraction form of the
    // exact kernel.  It is also the better fit for the single-thread HVP path
    // because it keeps each output AO-pair row hot while streaming its
    // contributing columns, instead of repeatedly scattering into two random
    // rows per integral.
    return apply_ao_pair_graph_matrix(
        ao_integral_input.ao_two_electron_integral_values.vector(),
        ao_integral_input.ao_two_electron_pair_graph_row_offsets.vector(),
        ao_integral_input.ao_two_electron_pair_graph_column_indices.vector(),
        ao_integral_input.ao_two_electron_pair_graph_integral_indices.vector(),
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
        ao_integral_input.ao_two_electron_integral_values.vector(),
        ao_integral_input.ao_two_electron_pair_indices.vector(),
        transformed_pair_coefficients,
        n_basis_functions,
        n_active_pairs,
        pair_gradients);
    return;
  }
  apply_sparse_ao_integral_matrix_from_four_indices(
      ao_integral_input.ao_two_electron_integral_values.vector(),
      ao_integral_input.ao_two_electron_integral_indices.vector(),
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

bool apply_exact_ao_pair_kernel_and_backprop_to_dense_active_coefficients_from_cache(
    const AoIntegralInput& ao_integral_input,
    const std::vector<double>& transformed_pair_coefficients,
    const ExactPackedActiveTwoElectronAdjointCache& cache,
    std::vector<double>* dense_active_gradients) {
  if (dense_active_gradients == nullptr) {
    throw std::invalid_argument("dense active gradient output must not be null");
  }
  const int n_basis_functions = ao_integral_input.n_basis_functions;
  const int n_active_orbitals = cache.n_active_orbitals;
  const std::size_t n_active_pairs =
      xmvb::to_size(n_active_orbitals) * (n_active_orbitals + 1) / 2;
  const std::size_t n_ao_pairs =
      xmvb::to_size(n_basis_functions) * (n_basis_functions + 1) / 2;
  const std::size_t expected_transformed_size =
      n_ao_pairs * n_active_pairs;
  if (cache.n_basis_functions != n_basis_functions ||
      transformed_pair_coefficients.size() != expected_transformed_size ||
      cache.accepted_dense_active_coefficients_buffer.size() !=
          xmvb::to_size(n_basis_functions) * n_active_orbitals ||
      cache.ao_pair_first_indices.size() != n_ao_pairs ||
      cache.ao_pair_second_indices.size() != n_ao_pairs ||
      dense_active_gradients->size() !=
          xmvb::to_size(n_basis_functions) * n_active_orbitals) {
    throw std::invalid_argument(
        "exact 2e fused AO-pair backprop input size mismatch");
  }

  if (!exact_2e_fused_hvp_applicable(
          ao_integral_input,
          n_active_orbitals)) {
    return false;
  }

  // The expensive exact 2e matvec previously materialized
  //
  //   pair_gradients = K * transformed_pair_coefficients
  //
  // as a full `n_ao_pairs x n_active_pairs` buffer and then performed a second
  // sweep to backpropagate that buffer into dense active-orbital rows.  In the
  // accepted-point HVP path the active coefficients are fixed, so for each
  // AO-pair target row we can apply the same packed-pair backpropagator while
  // streaming the AO kernel and avoid that intermediate entirely.
  if (!ao_integral_input.ao_two_electron_pair_graph_row_offsets.empty()) {
    switch (n_active_orbitals) {
      case 1:
        apply_ao_pair_graph_matrix_and_backprop_single_thread_fixed_hvp<1>(
            ao_integral_input.ao_two_electron_integral_values.vector(),
            ao_integral_input.ao_two_electron_pair_graph_row_offsets.vector(),
            ao_integral_input.ao_two_electron_pair_graph_column_indices.vector(),
            ao_integral_input.ao_two_electron_pair_graph_integral_indices.vector(),
            transformed_pair_coefficients.data(),
            cache,
            dense_active_gradients);
        return true;
      case 2:
        apply_ao_pair_graph_matrix_and_backprop_single_thread_fixed_hvp<2>(
            ao_integral_input.ao_two_electron_integral_values.vector(),
            ao_integral_input.ao_two_electron_pair_graph_row_offsets.vector(),
            ao_integral_input.ao_two_electron_pair_graph_column_indices.vector(),
            ao_integral_input.ao_two_electron_pair_graph_integral_indices.vector(),
            transformed_pair_coefficients.data(),
            cache,
            dense_active_gradients);
        return true;
      case 3:
        apply_ao_pair_graph_matrix_and_backprop_single_thread_fixed_hvp<3>(
            ao_integral_input.ao_two_electron_integral_values.vector(),
            ao_integral_input.ao_two_electron_pair_graph_row_offsets.vector(),
            ao_integral_input.ao_two_electron_pair_graph_column_indices.vector(),
            ao_integral_input.ao_two_electron_pair_graph_integral_indices.vector(),
            transformed_pair_coefficients.data(),
            cache,
            dense_active_gradients);
        return true;
      case 4:
        apply_ao_pair_graph_matrix_and_backprop_single_thread_fixed_hvp<4>(
            ao_integral_input.ao_two_electron_integral_values.vector(),
            ao_integral_input.ao_two_electron_pair_graph_row_offsets.vector(),
            ao_integral_input.ao_two_electron_pair_graph_column_indices.vector(),
            ao_integral_input.ao_two_electron_pair_graph_integral_indices.vector(),
            transformed_pair_coefficients.data(),
            cache,
            dense_active_gradients);
        return true;
      case 5:
        apply_ao_pair_graph_matrix_and_backprop_single_thread_fixed_hvp<5>(
            ao_integral_input.ao_two_electron_integral_values.vector(),
            ao_integral_input.ao_two_electron_pair_graph_row_offsets.vector(),
            ao_integral_input.ao_two_electron_pair_graph_column_indices.vector(),
            ao_integral_input.ao_two_electron_pair_graph_integral_indices.vector(),
            transformed_pair_coefficients.data(),
            cache,
            dense_active_gradients);
        return true;
      case 6:
        apply_ao_pair_graph_matrix_and_backprop_single_thread_fixed_hvp<6>(
            ao_integral_input.ao_two_electron_integral_values.vector(),
            ao_integral_input.ao_two_electron_pair_graph_row_offsets.vector(),
            ao_integral_input.ao_two_electron_pair_graph_column_indices.vector(),
            ao_integral_input.ao_two_electron_pair_graph_integral_indices.vector(),
            transformed_pair_coefficients.data(),
            cache,
            dense_active_gradients);
        return true;
      case 7:
        apply_ao_pair_graph_matrix_and_backprop_single_thread_fixed_hvp<7>(
            ao_integral_input.ao_two_electron_integral_values.vector(),
            ao_integral_input.ao_two_electron_pair_graph_row_offsets.vector(),
            ao_integral_input.ao_two_electron_pair_graph_column_indices.vector(),
            ao_integral_input.ao_two_electron_pair_graph_integral_indices.vector(),
            transformed_pair_coefficients.data(),
            cache,
            dense_active_gradients);
        return true;
      case 8:
        apply_ao_pair_graph_matrix_and_backprop_single_thread_fixed_hvp<8>(
            ao_integral_input.ao_two_electron_integral_values.vector(),
            ao_integral_input.ao_two_electron_pair_graph_row_offsets.vector(),
            ao_integral_input.ao_two_electron_pair_graph_column_indices.vector(),
            ao_integral_input.ao_two_electron_pair_graph_integral_indices.vector(),
            transformed_pair_coefficients.data(),
            cache,
            dense_active_gradients);
        return true;
      case 9:
        apply_ao_pair_graph_matrix_and_backprop_single_thread_fixed_hvp<9>(
            ao_integral_input.ao_two_electron_integral_values.vector(),
            ao_integral_input.ao_two_electron_pair_graph_row_offsets.vector(),
            ao_integral_input.ao_two_electron_pair_graph_column_indices.vector(),
            ao_integral_input.ao_two_electron_pair_graph_integral_indices.vector(),
            transformed_pair_coefficients.data(),
            cache,
            dense_active_gradients);
        return true;
      case 10:
        apply_ao_pair_graph_matrix_and_backprop_single_thread_fixed_hvp<10>(
            ao_integral_input.ao_two_electron_integral_values.vector(),
            ao_integral_input.ao_two_electron_pair_graph_row_offsets.vector(),
            ao_integral_input.ao_two_electron_pair_graph_column_indices.vector(),
            ao_integral_input.ao_two_electron_pair_graph_integral_indices.vector(),
            transformed_pair_coefficients.data(),
            cache,
            dense_active_gradients);
        return true;
      default:
        return false;
    }
  }
  if (!ao_integral_input.ao_two_electron_pair_indices.empty()) {
    switch (n_active_orbitals) {
      case 1:
        apply_sparse_ao_integral_matrix_and_backprop_single_thread_fixed_hvp<1>(
            ao_integral_input.ao_two_electron_integral_values.data(),
            ao_integral_input.ao_two_electron_pair_indices.data(),
            ao_integral_input.ao_two_electron_integral_values.size(),
            transformed_pair_coefficients.data(),
            cache,
            dense_active_gradients);
        return true;
      case 2:
        apply_sparse_ao_integral_matrix_and_backprop_single_thread_fixed_hvp<2>(
            ao_integral_input.ao_two_electron_integral_values.data(),
            ao_integral_input.ao_two_electron_pair_indices.data(),
            ao_integral_input.ao_two_electron_integral_values.size(),
            transformed_pair_coefficients.data(),
            cache,
            dense_active_gradients);
        return true;
      case 3:
        apply_sparse_ao_integral_matrix_and_backprop_single_thread_fixed_hvp<3>(
            ao_integral_input.ao_two_electron_integral_values.data(),
            ao_integral_input.ao_two_electron_pair_indices.data(),
            ao_integral_input.ao_two_electron_integral_values.size(),
            transformed_pair_coefficients.data(),
            cache,
            dense_active_gradients);
        return true;
      case 4:
        apply_sparse_ao_integral_matrix_and_backprop_single_thread_fixed_hvp<4>(
            ao_integral_input.ao_two_electron_integral_values.data(),
            ao_integral_input.ao_two_electron_pair_indices.data(),
            ao_integral_input.ao_two_electron_integral_values.size(),
            transformed_pair_coefficients.data(),
            cache,
            dense_active_gradients);
        return true;
      case 5:
        apply_sparse_ao_integral_matrix_and_backprop_single_thread_fixed_hvp<5>(
            ao_integral_input.ao_two_electron_integral_values.data(),
            ao_integral_input.ao_two_electron_pair_indices.data(),
            ao_integral_input.ao_two_electron_integral_values.size(),
            transformed_pair_coefficients.data(),
            cache,
            dense_active_gradients);
        return true;
      case 6:
        apply_sparse_ao_integral_matrix_and_backprop_single_thread_fixed_hvp<6>(
            ao_integral_input.ao_two_electron_integral_values.data(),
            ao_integral_input.ao_two_electron_pair_indices.data(),
            ao_integral_input.ao_two_electron_integral_values.size(),
            transformed_pair_coefficients.data(),
            cache,
            dense_active_gradients);
        return true;
      case 7:
        apply_sparse_ao_integral_matrix_and_backprop_single_thread_fixed_hvp<7>(
            ao_integral_input.ao_two_electron_integral_values.data(),
            ao_integral_input.ao_two_electron_pair_indices.data(),
            ao_integral_input.ao_two_electron_integral_values.size(),
            transformed_pair_coefficients.data(),
            cache,
            dense_active_gradients);
        return true;
      case 8:
        apply_sparse_ao_integral_matrix_and_backprop_single_thread_fixed_hvp<8>(
            ao_integral_input.ao_two_electron_integral_values.data(),
            ao_integral_input.ao_two_electron_pair_indices.data(),
            ao_integral_input.ao_two_electron_integral_values.size(),
            transformed_pair_coefficients.data(),
            cache,
            dense_active_gradients);
        return true;
      case 9:
        apply_sparse_ao_integral_matrix_and_backprop_single_thread_fixed_hvp<9>(
            ao_integral_input.ao_two_electron_integral_values.data(),
            ao_integral_input.ao_two_electron_pair_indices.data(),
            ao_integral_input.ao_two_electron_integral_values.size(),
            transformed_pair_coefficients.data(),
            cache,
            dense_active_gradients);
        return true;
      case 10:
        apply_sparse_ao_integral_matrix_and_backprop_single_thread_fixed_hvp<10>(
            ao_integral_input.ao_two_electron_integral_values.data(),
            ao_integral_input.ao_two_electron_pair_indices.data(),
            ao_integral_input.ao_two_electron_integral_values.size(),
            transformed_pair_coefficients.data(),
            cache,
            dense_active_gradients);
        return true;
      default:
        return false;
    }
  }
  return false;
}

bool apply_exact_ao_pair_kernel_and_backprop_from_cached_rows(
    const AoIntegralInput& ao_integral_input,
    const std::vector<double>& pair_source_coefficients,
    const ExactPackedActiveTwoElectronAdjointCache& cache,
    std::vector<double>* dense_active_gradients) {
  if (dense_active_gradients == nullptr) {
    throw std::invalid_argument("dense active gradient output must not be null");
  }
  const int n_basis_functions = ao_integral_input.n_basis_functions;
  const int n_active_orbitals = cache.n_active_orbitals;
  const std::size_t n_active_pairs =
      xmvb::to_size(n_active_orbitals) * (n_active_orbitals + 1) / 2;
  const std::size_t n_ao_pairs =
      xmvb::to_size(n_basis_functions) * (n_basis_functions + 1) / 2;
  const std::size_t expected_pair_source_size =
      n_ao_pairs * n_active_pairs;
  const std::size_t expected_backprop_size =
      xmvb::to_size(n_basis_functions) * n_active_pairs * n_active_orbitals;
  if (cache.n_basis_functions != n_basis_functions ||
      pair_source_coefficients.size() != expected_pair_source_size ||
      cache.ao_pair_first_indices.size() != n_ao_pairs ||
      cache.ao_pair_second_indices.size() != n_ao_pairs ||
      dense_active_gradients->size() !=
          xmvb::to_size(n_basis_functions) * n_active_orbitals) {
    throw std::invalid_argument(
        "exact 2e cached-row AO-pair backprop input size mismatch");
  }

  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif
  const int max_cached_row_threads =
      exact_2e_cached_row_direct_max_threads(n_active_orbitals);
  if (!exact_2e_fused_hvp_enabled() ||
      cache.accepted_active_pair_gradient_backprop_rows_buffer.size() !=
          expected_backprop_size ||
      n_threads > max_cached_row_threads) {
    return false;
  }

  // This path applies the AO-pair kernel to the mixed pair coefficients first
  // and folds the accepted adjoint / active-gradient backprop into the same
  // streaming contraction. It removes the `mixed * pair_gradient_matrix`
  // matrix multiply from the HVP hot path and keeps the multithreaded
  // workspace proportional to dense-active rows instead of AO-pair rows.
  //
  // However the cached-row kernel is still a low-arithmetic-intensity streaming
  // contraction over short active-pair rows. Large OpenMP teams mostly add
  // bandwidth pressure and reduction overhead, so above a small thread cap we
  // intentionally fall back to the older materialize-then-backprop path.
  if (!ao_integral_input.ao_two_electron_pair_graph_row_offsets.empty()) {
    apply_ao_pair_graph_matrix_and_backprop_from_cached_rows(
        ao_integral_input.ao_two_electron_integral_values.vector(),
        ao_integral_input.ao_two_electron_pair_graph_row_offsets.vector(),
        ao_integral_input.ao_two_electron_pair_graph_column_indices.vector(),
        ao_integral_input.ao_two_electron_pair_graph_integral_indices.vector(),
        pair_source_coefficients,
        cache,
        dense_active_gradients);
    return true;
  }
  if (!ao_integral_input.ao_two_electron_pair_indices.empty()) {
    apply_sparse_ao_integral_matrix_and_backprop_from_cached_rows(
        ao_integral_input.ao_two_electron_integral_values.vector(),
        ao_integral_input.ao_two_electron_pair_indices.vector(),
        pair_source_coefficients,
        cache,
        dense_active_gradients);
    return true;
  }
  return false;
}

std::vector<double> backpropagate_pair_coefficients_to_dense_active_coefficients(
    const std::vector<double>& pair_gradients,
    const std::vector<double>& dense_active_coefficients,
    int n_basis_functions,
    int n_active_orbitals,
    const std::vector<ActivePair>& active_pairs) {
  std::vector<double> dense_active_gradients(
      xmvb::to_size(n_basis_functions) * n_active_orbitals,
      0.0);

  const std::size_t n_active_pairs = active_pairs.size();
  const std::size_t n_ao_pairs =
      xmvb::to_size(n_basis_functions) * (n_basis_functions + 1) / 2;
  if (pair_gradients.size() != n_ao_pairs * n_active_pairs) {
    throw std::invalid_argument("pair gradient size mismatch");
  }
  if (dense_active_coefficients.size() !=
      xmvb::to_size(n_basis_functions) * n_active_orbitals) {
    throw std::invalid_argument("dense active coefficient size mismatch");
  }

#pragma omp parallel for schedule(static)
  for (int basis_function_index = 0;
       basis_function_index < n_basis_functions;
       ++basis_function_index) {
    double* gradient_row =
        dense_active_gradients.data() +
        xmvb::to_size(basis_function_index) * n_active_orbitals;
    for (int other_basis_function = 0;
         other_basis_function < n_basis_functions;
         ++other_basis_function) {
      const double* other_coefficients =
          dense_active_coefficients.data() +
          xmvb::to_size(other_basis_function) * n_active_orbitals;
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

void accumulate_backpropagated_pair_coefficients_to_dense_active_coefficients_from_cache(
    const std::vector<double>& pair_gradients,
    const std::vector<double>& dense_active_coefficients,
    const ExactPackedActiveTwoElectronAdjointCache& cache,
    std::vector<double>* dense_active_gradients) {
  if (dense_active_gradients == nullptr) {
    throw std::invalid_argument("dense active gradient output must not be null");
  }
  const int n_basis_functions = cache.n_basis_functions;
  const int n_active_orbitals = cache.n_active_orbitals;
  const std::size_t n_active_pairs = cache.active_pair_first_indices.size();
  if (cache.active_pair_second_indices.size() != n_active_pairs) {
    throw std::invalid_argument("exact 2e cache active-pair index size mismatch");
  }
  const std::size_t n_ao_pairs =
      xmvb::to_size(n_basis_functions) * (n_basis_functions + 1) / 2;
  if (pair_gradients.size() != n_ao_pairs * n_active_pairs) {
    throw std::invalid_argument("pair gradient size mismatch");
  }
  if (dense_active_coefficients.size() !=
      xmvb::to_size(n_basis_functions) * n_active_orbitals) {
    throw std::invalid_argument("dense active coefficient size mismatch");
  }
  if (dense_active_gradients->size() !=
      xmvb::to_size(n_basis_functions) * n_active_orbitals) {
    throw std::invalid_argument("dense active gradient output size mismatch");
  }

  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif
  if (n_threads <= 1) {
    for (int first_basis_function = 0;
         first_basis_function < n_basis_functions;
         ++first_basis_function) {
      double* first_gradient_row =
          dense_active_gradients->data() +
          xmvb::to_size(first_basis_function) * n_active_orbitals;
      const double* first_coefficients =
          dense_active_coefficients.data() +
          xmvb::to_size(first_basis_function) * n_active_orbitals;
      for (int second_basis_function = 0;
           second_basis_function <= first_basis_function;
           ++second_basis_function) {
        double* second_gradient_row =
            dense_active_gradients->data() +
            xmvb::to_size(second_basis_function) * n_active_orbitals;
        const double* second_coefficients =
            dense_active_coefficients.data() +
            xmvb::to_size(second_basis_function) * n_active_orbitals;
        const double* pair_gradient_row =
            pair_gradients.data() +
            ao_pair_index(first_basis_function, second_basis_function) *
                n_active_pairs;

        for (std::size_t active_pair_index = 0;
             active_pair_index < n_active_pairs;
             ++active_pair_index) {
          const double pair_gradient = pair_gradient_row[active_pair_index];
          if (pair_gradient == 0.0) {
            continue;
          }

          const int first_active =
              cache.active_pair_first_indices[active_pair_index];
          const int second_active =
              cache.active_pair_second_indices[active_pair_index];
          first_gradient_row[first_active] +=
              pair_gradient * second_coefficients[second_active];
          first_gradient_row[second_active] +=
              pair_gradient * second_coefficients[first_active];
          if (second_basis_function != first_basis_function) {
            second_gradient_row[first_active] +=
                pair_gradient * first_coefficients[second_active];
            second_gradient_row[second_active] +=
                pair_gradient * first_coefficients[first_active];
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
    double* gradient_row =
        dense_active_gradients->data() +
        xmvb::to_size(basis_function_index) * n_active_orbitals;
    for (int other_basis_function = 0;
         other_basis_function < n_basis_functions;
         ++other_basis_function) {
      const double* other_coefficients =
          dense_active_coefficients.data() +
          xmvb::to_size(other_basis_function) * n_active_orbitals;
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

        const int first_active =
            cache.active_pair_first_indices[active_pair_index];
        const int second_active =
            cache.active_pair_second_indices[active_pair_index];
        gradient_row[first_active] +=
            pair_gradient * other_coefficients[second_active];
        gradient_row[second_active] +=
            pair_gradient * other_coefficients[first_active];
      }
    }
  }
}

template <int NActiveOrbitals>
void accumulate_backpropagated_pair_coefficients_to_dense_active_coefficients_from_cache_fixed(
    const std::vector<double>& pair_gradients,
    const std::vector<double>& dense_active_coefficients,
    int n_basis_functions,
    std::vector<double>* dense_active_gradients) {
  const std::size_t n_active_pairs =
      xmvb::to_size(NActiveOrbitals) * (NActiveOrbitals + 1) / 2;
  const std::size_t n_ao_pairs =
      xmvb::to_size(n_basis_functions) * (n_basis_functions + 1) / 2;
  if (pair_gradients.size() != n_ao_pairs * n_active_pairs) {
    throw std::invalid_argument("pair gradient size mismatch");
  }
  if (dense_active_coefficients.size() !=
      xmvb::to_size(n_basis_functions) * NActiveOrbitals) {
    throw std::invalid_argument("dense active coefficient size mismatch");
  }
  if (dense_active_gradients == nullptr ||
      dense_active_gradients->size() !=
          xmvb::to_size(n_basis_functions) * NActiveOrbitals) {
    throw std::invalid_argument("dense active gradient output size mismatch");
  }

  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif
  if (n_threads <= 1) {
    for (int first_basis_function = 0;
         first_basis_function < n_basis_functions;
         ++first_basis_function) {
      double* first_gradient_row =
          dense_active_gradients->data() +
          xmvb::to_size(first_basis_function) * NActiveOrbitals;
      const double* first_coefficients =
          dense_active_coefficients.data() +
          xmvb::to_size(first_basis_function) * NActiveOrbitals;
      for (int second_basis_function = 0;
           second_basis_function <= first_basis_function;
           ++second_basis_function) {
        double* second_gradient_row =
            dense_active_gradients->data() +
            xmvb::to_size(second_basis_function) * NActiveOrbitals;
        const double* second_coefficients =
            dense_active_coefficients.data() +
            xmvb::to_size(second_basis_function) * NActiveOrbitals;
        const double* pair_gradient_row =
            pair_gradients.data() +
            ao_pair_index(first_basis_function, second_basis_function) *
                n_active_pairs;
        accumulate_packed_active_pair_gradient_row_to_active_gradient<NActiveOrbitals>(
            first_gradient_row,
            pair_gradient_row,
            second_coefficients);
        if (second_basis_function != first_basis_function) {
          accumulate_packed_active_pair_gradient_row_to_active_gradient<NActiveOrbitals>(
              second_gradient_row,
              pair_gradient_row,
              first_coefficients);
        }
      }
    }
    return;
  }

#pragma omp parallel for schedule(static)
  for (int basis_function_index = 0;
       basis_function_index < n_basis_functions;
       ++basis_function_index) {
    double* gradient_row =
        dense_active_gradients->data() +
        xmvb::to_size(basis_function_index) * NActiveOrbitals;
    for (int other_basis_function = 0;
         other_basis_function < n_basis_functions;
         ++other_basis_function) {
      const double* other_coefficients =
          dense_active_coefficients.data() +
          xmvb::to_size(other_basis_function) * NActiveOrbitals;
      const double* pair_gradient_row =
          pair_gradients.data() +
          ao_pair_index(basis_function_index, other_basis_function) *
              n_active_pairs;
      accumulate_packed_active_pair_gradient_row_to_active_gradient<NActiveOrbitals>(
          gradient_row,
          pair_gradient_row,
          other_coefficients);
    }
  }
}

std::vector<double> backpropagate_pair_coefficients_to_dense_active_coefficients_from_cache(
    const std::vector<double>& pair_gradients,
    const std::vector<double>& dense_active_coefficients,
    const ExactPackedActiveTwoElectronAdjointCache& cache) {
  std::vector<double> dense_active_gradients(
      xmvb::to_size(cache.n_basis_functions) * cache.n_active_orbitals,
      0.0);
  switch (cache.n_active_orbitals) {
    case 1:
      accumulate_backpropagated_pair_coefficients_to_dense_active_coefficients_from_cache_fixed<1>(
          pair_gradients,
          dense_active_coefficients,
          cache.n_basis_functions,
          &dense_active_gradients);
      return dense_active_gradients;
    case 2:
      accumulate_backpropagated_pair_coefficients_to_dense_active_coefficients_from_cache_fixed<2>(
          pair_gradients,
          dense_active_coefficients,
          cache.n_basis_functions,
          &dense_active_gradients);
      return dense_active_gradients;
    case 3:
      accumulate_backpropagated_pair_coefficients_to_dense_active_coefficients_from_cache_fixed<3>(
          pair_gradients,
          dense_active_coefficients,
          cache.n_basis_functions,
          &dense_active_gradients);
      return dense_active_gradients;
    case 4:
      accumulate_backpropagated_pair_coefficients_to_dense_active_coefficients_from_cache_fixed<4>(
          pair_gradients,
          dense_active_coefficients,
          cache.n_basis_functions,
          &dense_active_gradients);
      return dense_active_gradients;
    case 5:
      accumulate_backpropagated_pair_coefficients_to_dense_active_coefficients_from_cache_fixed<5>(
          pair_gradients,
          dense_active_coefficients,
          cache.n_basis_functions,
          &dense_active_gradients);
      return dense_active_gradients;
    case 6:
      accumulate_backpropagated_pair_coefficients_to_dense_active_coefficients_from_cache_fixed<6>(
          pair_gradients,
          dense_active_coefficients,
          cache.n_basis_functions,
          &dense_active_gradients);
      return dense_active_gradients;
    case 7:
      accumulate_backpropagated_pair_coefficients_to_dense_active_coefficients_from_cache_fixed<7>(
          pair_gradients,
          dense_active_coefficients,
          cache.n_basis_functions,
          &dense_active_gradients);
      return dense_active_gradients;
    case 8:
      accumulate_backpropagated_pair_coefficients_to_dense_active_coefficients_from_cache_fixed<8>(
          pair_gradients,
          dense_active_coefficients,
          cache.n_basis_functions,
          &dense_active_gradients);
      return dense_active_gradients;
    case 9:
      accumulate_backpropagated_pair_coefficients_to_dense_active_coefficients_from_cache_fixed<9>(
          pair_gradients,
          dense_active_coefficients,
          cache.n_basis_functions,
          &dense_active_gradients);
      return dense_active_gradients;
    case 10:
      accumulate_backpropagated_pair_coefficients_to_dense_active_coefficients_from_cache_fixed<10>(
          pair_gradients,
          dense_active_coefficients,
          cache.n_basis_functions,
          &dense_active_gradients);
      return dense_active_gradients;
    default:
      break;
  }
  accumulate_backpropagated_pair_coefficients_to_dense_active_coefficients_from_cache(
      pair_gradients,
      dense_active_coefficients,
      cache,
      &dense_active_gradients);
  return dense_active_gradients;
}

template <int NActiveOrbitals>
void
accumulate_pair_products_to_dense_active_coefficients_from_cached_rows_fixed(
    const std::vector<double>& pair_products,
    const ExactPackedActiveTwoElectronAdjointCache& cache,
    std::vector<double>* dense_active_gradients) {
  const int n_basis_functions = cache.n_basis_functions;
  const std::size_t n_active_pairs =
      xmvb::to_size(NActiveOrbitals) * (NActiveOrbitals + 1) / 2;
  const std::size_t n_ao_pairs =
      xmvb::to_size(n_basis_functions) * (n_basis_functions + 1) / 2;
  const std::size_t expected_backprop_size =
      xmvb::to_size(n_basis_functions) * n_active_pairs * NActiveOrbitals;
  if (pair_products.size() != n_ao_pairs * n_active_pairs) {
    throw std::invalid_argument("pair product size mismatch");
  }
  if (cache.accepted_active_pair_gradient_backprop_rows_buffer.size() !=
      expected_backprop_size) {
    throw std::invalid_argument("accepted active-pair backprop row size mismatch");
  }
  if (cache.ao_pair_first_indices.size() != n_ao_pairs ||
      cache.ao_pair_second_indices.size() != n_ao_pairs) {
    throw std::invalid_argument("AO pair component cache size mismatch");
  }
  if (dense_active_gradients == nullptr ||
      dense_active_gradients->size() !=
          xmvb::to_size(n_basis_functions) * NActiveOrbitals) {
    throw std::invalid_argument("dense active gradient output size mismatch");
  }

  const double* accepted_backprop_rows =
      cache.accepted_active_pair_gradient_backprop_rows_buffer.data();
  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif
  if (n_threads <= 1) {
    for (std::size_t ao_pair_index_offset = 0;
         ao_pair_index_offset < n_ao_pairs;
         ++ao_pair_index_offset) {
      const int first_basis_function =
          cache.ao_pair_first_indices[ao_pair_index_offset];
      const int second_basis_function =
          cache.ao_pair_second_indices[ao_pair_index_offset];
      const double* pair_product_row =
          pair_products.data() + ao_pair_index_offset * n_active_pairs;
      double* first_gradient_row =
          dense_active_gradients->data() +
          xmvb::to_size(first_basis_function) * NActiveOrbitals;
      const double* second_basis_backprop_rows =
          accepted_backprop_rows +
          xmvb::to_size(second_basis_function) * n_active_pairs * NActiveOrbitals;
      for (std::size_t active_pair_index = 0;
           active_pair_index < n_active_pairs;
           ++active_pair_index) {
        accumulate_scaled_active_gradient_row<NActiveOrbitals>(
            first_gradient_row,
            second_basis_backprop_rows +
                active_pair_index * NActiveOrbitals,
            pair_product_row[active_pair_index]);
      }
      if (second_basis_function != first_basis_function) {
        double* second_gradient_row =
            dense_active_gradients->data() +
            xmvb::to_size(second_basis_function) * NActiveOrbitals;
        const double* first_basis_backprop_rows =
            accepted_backprop_rows +
            xmvb::to_size(first_basis_function) * n_active_pairs * NActiveOrbitals;
        for (std::size_t active_pair_index = 0;
             active_pair_index < n_active_pairs;
             ++active_pair_index) {
          accumulate_scaled_active_gradient_row<NActiveOrbitals>(
              second_gradient_row,
              first_basis_backprop_rows +
                  active_pair_index * NActiveOrbitals,
              pair_product_row[active_pair_index]);
        }
      }
    }
    return;
  }

#pragma omp parallel for schedule(static)
  for (int basis_function_index = 0;
       basis_function_index < n_basis_functions;
       ++basis_function_index) {
    double* gradient_row =
        dense_active_gradients->data() +
        xmvb::to_size(basis_function_index) * NActiveOrbitals;
    for (int other_basis_function = 0;
         other_basis_function < n_basis_functions;
         ++other_basis_function) {
      const double* pair_product_row =
          pair_products.data() +
          ao_pair_index(basis_function_index, other_basis_function) *
              n_active_pairs;
      const double* other_basis_backprop_rows =
          accepted_backprop_rows +
          xmvb::to_size(other_basis_function) * n_active_pairs * NActiveOrbitals;
      for (std::size_t active_pair_index = 0;
           active_pair_index < n_active_pairs;
           ++active_pair_index) {
        accumulate_scaled_active_gradient_row<NActiveOrbitals>(
            gradient_row,
            other_basis_backprop_rows +
                active_pair_index * NActiveOrbitals,
            pair_product_row[active_pair_index]);
      }
    }
  }
}

void accumulate_pair_products_to_dense_active_coefficients_from_cached_rows(
    const std::vector<double>& pair_products,
    const ExactPackedActiveTwoElectronAdjointCache& cache,
    std::vector<double>* dense_active_gradients) {
  const int n_basis_functions = cache.n_basis_functions;
  const int n_active_orbitals = cache.n_active_orbitals;
  const std::size_t n_active_pairs = cache.active_pair_first_indices.size();
  if (cache.active_pair_second_indices.size() != n_active_pairs) {
    throw std::invalid_argument("exact 2e cache active-pair index size mismatch");
  }
  switch (n_active_orbitals) {
    case 1:
      accumulate_pair_products_to_dense_active_coefficients_from_cached_rows_fixed<1>(
          pair_products,
          cache,
          dense_active_gradients);
      return;
    case 2:
      accumulate_pair_products_to_dense_active_coefficients_from_cached_rows_fixed<2>(
          pair_products,
          cache,
          dense_active_gradients);
      return;
    case 3:
      accumulate_pair_products_to_dense_active_coefficients_from_cached_rows_fixed<3>(
          pair_products,
          cache,
          dense_active_gradients);
      return;
    case 4:
      accumulate_pair_products_to_dense_active_coefficients_from_cached_rows_fixed<4>(
          pair_products,
          cache,
          dense_active_gradients);
      return;
    case 5:
      accumulate_pair_products_to_dense_active_coefficients_from_cached_rows_fixed<5>(
          pair_products,
          cache,
          dense_active_gradients);
      return;
    case 6:
      accumulate_pair_products_to_dense_active_coefficients_from_cached_rows_fixed<6>(
          pair_products,
          cache,
          dense_active_gradients);
      return;
    case 7:
      accumulate_pair_products_to_dense_active_coefficients_from_cached_rows_fixed<7>(
          pair_products,
          cache,
          dense_active_gradients);
      return;
    case 8:
      accumulate_pair_products_to_dense_active_coefficients_from_cached_rows_fixed<8>(
          pair_products,
          cache,
          dense_active_gradients);
      return;
    case 9:
      accumulate_pair_products_to_dense_active_coefficients_from_cached_rows_fixed<9>(
          pair_products,
          cache,
          dense_active_gradients);
      return;
    case 10:
      accumulate_pair_products_to_dense_active_coefficients_from_cached_rows_fixed<10>(
          pair_products,
          cache,
          dense_active_gradients);
      return;
    default:
      break;
  }

  const std::size_t n_ao_pairs =
      xmvb::to_size(n_basis_functions) * (n_basis_functions + 1) / 2;
  const std::size_t expected_backprop_size =
      xmvb::to_size(n_basis_functions) * n_active_pairs * n_active_orbitals;
  if (pair_products.size() != n_ao_pairs * n_active_pairs) {
    throw std::invalid_argument("pair product size mismatch");
  }
  if (cache.accepted_active_pair_gradient_backprop_rows_buffer.size() !=
      expected_backprop_size) {
    throw std::invalid_argument("accepted active-pair backprop row size mismatch");
  }
  if (cache.ao_pair_first_indices.size() != n_ao_pairs ||
      cache.ao_pair_second_indices.size() != n_ao_pairs) {
    throw std::invalid_argument("AO pair component cache size mismatch");
  }
  if (dense_active_gradients == nullptr ||
      dense_active_gradients->size() !=
          xmvb::to_size(n_basis_functions) * n_active_orbitals) {
    throw std::invalid_argument("dense active gradient output size mismatch");
  }

  const double* accepted_backprop_rows =
      cache.accepted_active_pair_gradient_backprop_rows_buffer.data();
  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif
  if (n_threads <= 1) {
    for (std::size_t ao_pair_index_offset = 0;
         ao_pair_index_offset < n_ao_pairs;
         ++ao_pair_index_offset) {
      const int first_basis_function =
          cache.ao_pair_first_indices[ao_pair_index_offset];
      const int second_basis_function =
          cache.ao_pair_second_indices[ao_pair_index_offset];
      const double* pair_product_row =
          pair_products.data() + ao_pair_index_offset * n_active_pairs;
      double* first_gradient_row =
          dense_active_gradients->data() +
          xmvb::to_size(first_basis_function) * n_active_orbitals;
      const double* second_basis_backprop_rows =
          accepted_backprop_rows +
          xmvb::to_size(second_basis_function) * n_active_pairs * n_active_orbitals;
      for (std::size_t active_pair_index = 0;
           active_pair_index < n_active_pairs;
           ++active_pair_index) {
        accumulate_scaled_active_gradient_row_dynamic(
            first_gradient_row,
            second_basis_backprop_rows +
                active_pair_index * n_active_orbitals,
            n_active_orbitals,
            pair_product_row[active_pair_index]);
      }
      if (second_basis_function != first_basis_function) {
        double* second_gradient_row =
            dense_active_gradients->data() +
            xmvb::to_size(second_basis_function) * n_active_orbitals;
        const double* first_basis_backprop_rows =
            accepted_backprop_rows +
            xmvb::to_size(first_basis_function) * n_active_pairs * n_active_orbitals;
        for (std::size_t active_pair_index = 0;
             active_pair_index < n_active_pairs;
             ++active_pair_index) {
          accumulate_scaled_active_gradient_row_dynamic(
              second_gradient_row,
              first_basis_backprop_rows +
                  active_pair_index * n_active_orbitals,
              n_active_orbitals,
              pair_product_row[active_pair_index]);
        }
      }
    }
    return;
  }

#pragma omp parallel for schedule(static)
  for (int basis_function_index = 0;
       basis_function_index < n_basis_functions;
       ++basis_function_index) {
    double* gradient_row =
        dense_active_gradients->data() +
        xmvb::to_size(basis_function_index) * n_active_orbitals;
    for (int other_basis_function = 0;
         other_basis_function < n_basis_functions;
         ++other_basis_function) {
      const double* pair_product_row =
          pair_products.data() +
          ao_pair_index(basis_function_index, other_basis_function) *
              n_active_pairs;
      const double* other_basis_backprop_rows =
          accepted_backprop_rows +
          xmvb::to_size(other_basis_function) * n_active_pairs * n_active_orbitals;
      for (std::size_t active_pair_index = 0;
           active_pair_index < n_active_pairs;
           ++active_pair_index) {
        accumulate_scaled_active_gradient_row_dynamic(
            gradient_row,
            other_basis_backprop_rows +
                active_pair_index * n_active_orbitals,
            n_active_orbitals,
            pair_product_row[active_pair_index]);
      }
    }
  }
}

const std::vector<double>* find_packed_integrals(
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_active_orbitals) {
  if (two_electron_view.packed_active_two_electron_integrals == nullptr) {
    return nullptr;
  }
  const auto& packed_active_two_electron_integrals =
      *two_electron_view.packed_active_two_electron_integrals;
  const std::size_t expected_size =
      packed_active_two_electron_integral_count(n_active_orbitals);
  if (packed_active_two_electron_integrals.size() != expected_size) {
    return nullptr;
  }
  return &packed_active_two_electron_integrals;
}

const std::vector<double>& require_packed_integrals(
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_active_orbitals) {
  const std::vector<double>* packed_active_two_electron_integrals =
      find_packed_integrals(two_electron_view, n_active_orbitals);
  if (packed_active_two_electron_integrals == nullptr) {
    throw std::invalid_argument("packed active-space two-electron buffer is unavailable");
  }
  return *packed_active_two_electron_integrals;
}

const Eigen::MatrixXd& require_ri_active_pair_factors(
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_active_orbitals) {
  if (two_electron_view.ri_active_pair_factors == nullptr) {
    throw std::invalid_argument("RI active-pair-factor buffer is unavailable");
  }
  const auto& ri_active_pair_factors = *two_electron_view.ri_active_pair_factors;
  const std::size_t expected_size =
      xmvb::to_size(two_electron_view.n_auxiliary_functions) *
      xmvb::to_size(packed_active_pair_count(n_active_orbitals));
  if (two_electron_view.n_auxiliary_functions <= 0 ||
      ri_active_pair_factors.rows() != two_electron_view.n_auxiliary_functions ||
      ri_active_pair_factors.cols() != packed_active_pair_count(n_active_orbitals) ||
      xmvb::to_size(ri_active_pair_factors.size()) != expected_size) {
    throw std::invalid_argument("RI active-pair-factor buffer size mismatch");
  }
  return ri_active_pair_factors;
}

double lookup_ri_active_space_two_electron_kernel_value(
    const Eigen::MatrixXd& ri_active_pair_factors,
    int row_packed_pair_index,
    int column_packed_pair_index) {
  double kernel_value = 0.0;
  for (int auxiliary_function_index = 0;
       auxiliary_function_index < ri_active_pair_factors.rows();
       ++auxiliary_function_index) {
    kernel_value +=
        ri_active_pair_factors(auxiliary_function_index, row_packed_pair_index) *
        ri_active_pair_factors(auxiliary_function_index, column_packed_pair_index);
  }
  return kernel_value;
}

std::vector<double> apply_ri_active_space_two_electron_kernel_to_sparse_projection(
    const Eigen::MatrixXd& ri_active_pair_factors,
    const std::vector<int>& packed_pair_indices,
    const std::vector<double>& packed_pair_values) {
  const int n_auxiliary_functions = static_cast<int>(ri_active_pair_factors.rows());
  const int n_packed_active_pairs = static_cast<int>(ri_active_pair_factors.cols());
  std::vector<double> auxiliary_projection(
      xmvb::to_size(n_auxiliary_functions),
      0.0);
  for (std::size_t entry_index = 0;
       entry_index < packed_pair_indices.size();
       ++entry_index) {
    const int packed_pair_index = packed_pair_indices[entry_index];
    const double packed_pair_value = packed_pair_values[entry_index];
    if (packed_pair_index < 0 || packed_pair_index >= n_packed_active_pairs) {
      throw std::invalid_argument("packed active-pair index out of range");
    }
    for (int auxiliary_function_index = 0;
         auxiliary_function_index < n_auxiliary_functions;
         ++auxiliary_function_index) {
      auxiliary_projection[xmvb::to_size(auxiliary_function_index)] +=
          ri_active_pair_factors(auxiliary_function_index, packed_pair_index) *
          packed_pair_value;
    }
  }

  std::vector<double> projected_pair_values(
      xmvb::to_size(n_packed_active_pairs),
      0.0);
  for (int packed_pair_index = 0;
       packed_pair_index < n_packed_active_pairs;
       ++packed_pair_index) {
    double projected_value = 0.0;
    for (int auxiliary_function_index = 0;
         auxiliary_function_index < n_auxiliary_functions;
         ++auxiliary_function_index) {
      projected_value +=
          ri_active_pair_factors(auxiliary_function_index, packed_pair_index) *
          auxiliary_projection[xmvb::to_size(auxiliary_function_index)];
    }
    projected_pair_values[xmvb::to_size(packed_pair_index)] = projected_value;
  }

  return projected_pair_values;
}

}  // namespace

ActiveSpaceTwoElectronView make_active_space_two_electron_view(
    const std::vector<double>& packed_active_two_electron_integrals) {
  ActiveSpaceTwoElectronView view;
  view.representation = ActiveSpaceTwoElectronRepresentation::PackedExact;
  view.packed_active_two_electron_integrals = &packed_active_two_electron_integrals;
  return view;
}

ActiveSpaceTwoElectronView make_active_space_two_electron_view(
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result) {
  ActiveSpaceTwoElectronView view;
  view.representation = active_space_two_electron_result.representation;
  view.packed_active_two_electron_integrals =
      &active_space_two_electron_result.packed_active_two_electron_integrals;
  view.n_auxiliary_functions = active_space_two_electron_result.n_auxiliary_functions;
  view.ri_active_pair_factors = &active_space_two_electron_result.ri_active_pair_factors;
  return view;
}

int packed_active_pair_count(int n_active_orbitals) {
  if (n_active_orbitals <= 0) {
    throw std::invalid_argument("n_active_orbitals must be positive");
  }
  return n_active_orbitals * (n_active_orbitals + 1) / 2;
}

std::size_t packed_active_two_electron_integral_count(int n_active_orbitals) {
  const std::size_t n_packed_active_pairs =
      xmvb::to_size(packed_active_pair_count(n_active_orbitals));
  return n_packed_active_pairs * (n_packed_active_pairs + 1) / 2;
}

int infer_active_orbital_count_from_packed_pair_count(int n_packed_active_pairs) {
  if (n_packed_active_pairs <= 0) {
    throw std::invalid_argument("n_packed_active_pairs must be positive");
  }
  const double discriminant = 1.0 + 8.0 * static_cast<double>(n_packed_active_pairs);
  const int n_active_orbitals =
      static_cast<int>((std::sqrt(discriminant) - 1.0) / 2.0 + 0.5);
  if (n_active_orbitals * (n_active_orbitals + 1) / 2 != n_packed_active_pairs) {
    throw std::invalid_argument("packed active-pair count is not triangular");
  }
  return n_active_orbitals;
}

int infer_active_orbital_count_from_packed_integral_count(std::size_t packed_integral_count) {
  if (packed_integral_count == 0) {
    throw std::invalid_argument("packed_integral_count must be positive");
  }
  const double discriminant = 1.0 + 8.0 * static_cast<double>(packed_integral_count);
  const int n_packed_active_pairs =
      static_cast<int>((std::sqrt(discriminant) - 1.0) / 2.0 + 0.5);
  if (xmvb::to_size(n_packed_active_pairs) *
          xmvb::to_size(n_packed_active_pairs + 1) / 2 !=
      packed_integral_count) {
    throw std::invalid_argument("packed active-space two-electron count is not triangular");
  }
  return infer_active_orbital_count_from_packed_pair_count(n_packed_active_pairs);
}

double lookup_active_space_two_electron_kernel_value(
    const ActiveSpaceTwoElectronView& two_electron_view,
    int row_packed_pair_index,
    int column_packed_pair_index,
    int n_active_orbitals) {
  const int n_packed_active_pairs = packed_active_pair_count(n_active_orbitals);
  if (row_packed_pair_index < 0 || row_packed_pair_index >= n_packed_active_pairs ||
      column_packed_pair_index < 0 || column_packed_pair_index >= n_packed_active_pairs) {
    throw std::invalid_argument("packed active-pair index out of range");
  }

  if (const std::vector<double>* packed_active_two_electron_integrals =
          find_packed_integrals(two_electron_view, n_active_orbitals);
      packed_active_two_electron_integrals != nullptr) {
    // A prebuilt packed `GGO` buffer turns the determinant-pair hot path back
    // into a direct table lookup, avoiding an auxiliary-length RI dot product
    // for every requested pair-of-pairs kernel value.
    const int packed_pair_of_pairs_index =
        TwoElectronIndexer::packed_pair_of_pairs_index(
            row_packed_pair_index,
            column_packed_pair_index);
    return (*packed_active_two_electron_integrals)[xmvb::to_size(
        packed_pair_of_pairs_index)];
  }

  switch (two_electron_view.representation) {
    case ActiveSpaceTwoElectronRepresentation::PackedExact: {
      throw std::invalid_argument("packed active-space two-electron buffer is unavailable");
    }
    case ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity: {
      const auto& ri_active_pair_factors =
          require_ri_active_pair_factors(two_electron_view, n_active_orbitals);
      return lookup_ri_active_space_two_electron_kernel_value(
          ri_active_pair_factors,
          row_packed_pair_index,
          column_packed_pair_index);
    }
  }
  throw std::invalid_argument("unknown active-space two-electron representation");
}

std::vector<double> apply_active_space_two_electron_kernel_to_sparse_projection(
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_active_orbitals,
    const std::vector<int>& packed_pair_indices,
    const std::vector<double>& packed_pair_values) {
  if (packed_pair_indices.size() != packed_pair_values.size()) {
    throw std::invalid_argument("packed active-pair projection sizes are inconsistent");
  }

  const int n_packed_active_pairs = packed_active_pair_count(n_active_orbitals);
  if (const std::vector<double>* packed_active_two_electron_integrals =
          find_packed_integrals(two_electron_view, n_active_orbitals);
      packed_active_two_electron_integrals != nullptr) {
    std::vector<double> projected_pair_values(
        xmvb::to_size(n_packed_active_pairs),
        0.0);
    for (std::size_t entry_index = 0;
         entry_index < packed_pair_indices.size();
         ++entry_index) {
      const int packed_pair_index = packed_pair_indices[entry_index];
      const double packed_pair_value = packed_pair_values[entry_index];
      if (packed_pair_index < 0 || packed_pair_index >= n_packed_active_pairs) {
        throw std::invalid_argument("packed active-pair index out of range");
      }
      for (int row_packed_pair_index = 0;
           row_packed_pair_index < n_packed_active_pairs;
           ++row_packed_pair_index) {
        const int packed_pair_of_pairs_index =
            TwoElectronIndexer::packed_pair_of_pairs_index(
                row_packed_pair_index,
                packed_pair_index);
        projected_pair_values[xmvb::to_size(row_packed_pair_index)] +=
            (*packed_active_two_electron_integrals)[xmvb::to_size(
                packed_pair_of_pairs_index)] *
            packed_pair_value;
      }
    }
    return projected_pair_values;
  }

  switch (two_electron_view.representation) {
    case ActiveSpaceTwoElectronRepresentation::PackedExact: {
      throw std::invalid_argument("packed active-space two-electron buffer is unavailable");
    }
    case ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity: {
      const auto& ri_active_pair_factors =
          require_ri_active_pair_factors(two_electron_view, n_active_orbitals);
      return apply_ri_active_space_two_electron_kernel_to_sparse_projection(
          ri_active_pair_factors,
          packed_pair_indices,
          packed_pair_values);
    }
  }
  throw std::invalid_argument("unknown active-space two-electron representation");
}

Eigen::VectorXd apply_active_space_two_electron_kernel_to_sparse_projection_subset(
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_active_orbitals,
    const std::vector<int>& packed_pair_indices,
    const std::vector<double>& packed_pair_values,
    const std::vector<int>& target_packed_pair_indices) {
  if (packed_pair_indices.size() != packed_pair_values.size()) {
    throw std::invalid_argument("packed active-pair projection sizes are inconsistent");
  }

  Eigen::VectorXd projected_pair_values =
      Eigen::VectorXd::Zero(static_cast<int>(target_packed_pair_indices.size()));
  if (target_packed_pair_indices.empty()) {
    return projected_pair_values;
  }

  const int n_packed_active_pairs = packed_active_pair_count(n_active_orbitals);
  if (const std::vector<double>* packed_active_two_electron_integrals =
          find_packed_integrals(two_electron_view, n_active_orbitals);
      packed_active_two_electron_integrals != nullptr) {
    for (std::size_t target_index = 0;
         target_index < target_packed_pair_indices.size();
         ++target_index) {
      const int row_packed_pair_index =
          target_packed_pair_indices[target_index];
      if (row_packed_pair_index < 0 || row_packed_pair_index >= n_packed_active_pairs) {
        throw std::invalid_argument("packed active-pair index out of range");
      }
      double projected_value = 0.0;
      for (std::size_t entry_index = 0;
           entry_index < packed_pair_indices.size();
           ++entry_index) {
        const int packed_pair_index = packed_pair_indices[entry_index];
        if (packed_pair_index < 0 || packed_pair_index >= n_packed_active_pairs) {
          throw std::invalid_argument("packed active-pair index out of range");
        }
        const int packed_pair_of_pairs_index =
            TwoElectronIndexer::packed_pair_of_pairs_index(
                row_packed_pair_index,
                packed_pair_index);
        projected_value +=
            (*packed_active_two_electron_integrals)[xmvb::to_size(
                packed_pair_of_pairs_index)] *
            packed_pair_values[entry_index];
      }
      projected_pair_values(static_cast<int>(target_index)) = projected_value;
    }
    return projected_pair_values;
  }

  switch (two_electron_view.representation) {
    case ActiveSpaceTwoElectronRepresentation::PackedExact: {
      throw std::invalid_argument("packed active-space two-electron buffer is unavailable");
    }
    case ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity: {
      const auto& ri_active_pair_factors =
          require_ri_active_pair_factors(two_electron_view, n_active_orbitals);
      Eigen::VectorXd auxiliary_projection =
          Eigen::VectorXd::Zero(ri_active_pair_factors.rows());
      for (std::size_t entry_index = 0;
           entry_index < packed_pair_indices.size();
           ++entry_index) {
        const int packed_pair_index = packed_pair_indices[entry_index];
        if (packed_pair_index < 0 || packed_pair_index >= n_packed_active_pairs) {
          throw std::invalid_argument("packed active-pair index out of range");
        }
        auxiliary_projection.noalias() +=
            packed_pair_values[entry_index] *
            ri_active_pair_factors.col(packed_pair_index);
      }

      for (std::size_t target_index = 0;
           target_index < target_packed_pair_indices.size();
           ++target_index) {
        const int row_packed_pair_index =
            target_packed_pair_indices[target_index];
        if (row_packed_pair_index < 0 || row_packed_pair_index >= n_packed_active_pairs) {
          throw std::invalid_argument("packed active-pair index out of range");
        }
        projected_pair_values(static_cast<int>(target_index)) =
            ri_active_pair_factors.col(row_packed_pair_index).dot(auxiliary_projection);
      }
      return projected_pair_values;
    }
  }
  throw std::invalid_argument("unknown active-space two-electron representation");
}

std::vector<double> reconstruct_packed_active_two_electron_integrals(
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_active_orbitals) {
  if (two_electron_view.representation ==
      ActiveSpaceTwoElectronRepresentation::PackedExact) {
    return require_packed_integrals(two_electron_view, n_active_orbitals);
  }

  const int n_packed_active_pairs = packed_active_pair_count(n_active_orbitals);
  std::vector<double> packed_active_two_electron_integrals(
      packed_active_two_electron_integral_count(n_active_orbitals),
      0.0);
  for (int column_packed_pair_index = 0;
       column_packed_pair_index < n_packed_active_pairs;
       ++column_packed_pair_index) {
    for (int row_packed_pair_index = 0;
         row_packed_pair_index <= column_packed_pair_index;
         ++row_packed_pair_index) {
      const int packed_pair_of_pairs_index =
          TwoElectronIndexer::packed_pair_of_pairs_index(
              row_packed_pair_index,
              column_packed_pair_index);
      packed_active_two_electron_integrals[xmvb::to_size(
          packed_pair_of_pairs_index)] =
          lookup_active_space_two_electron_kernel_value(
              two_electron_view,
              row_packed_pair_index,
              column_packed_pair_index,
              n_active_orbitals);
    }
  }
  return packed_active_two_electron_integrals;
}

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

  copy_matrix_to_legacy_row_buffer_inplace(
      dense_active_coefficients,
      &workspace->dense_active_coefficients_buffer);
  copy_matrix_to_legacy_row_buffer_inplace(
      dense_active_direction,
      &workspace->dense_active_direction_buffer);

  const auto active_pairs = build_active_pair_list(n_active_orbitals);
  const std::size_t n_active_pairs = active_pairs.size();
  const std::size_t n_ao_pairs =
      xmvb::to_size(n_basis_functions) * (n_basis_functions + 1) / 2;

  build_ao_pair_to_active_pair_coefficients(
      workspace->dense_active_coefficients_buffer,
      n_basis_functions,
      n_active_orbitals,
      active_pairs,
      &workspace->pair_coefficients_buffer);
  if (accepted_active_space_two_electron_result != nullptr &&
      accepted_active_space_two_electron_result->dense_ao_pair_products.size() != 0) {
    if (xmvb::to_size(accepted_active_space_two_electron_result->dense_ao_pair_products.size()) !=
        n_ao_pairs * n_active_pairs) {
      throw std::invalid_argument(
          "accepted dense AO pair product size mismatch in delta GGO");
    }
    copy_matrix_to_legacy_row_buffer_inplace(
        accepted_active_space_two_electron_result->dense_ao_pair_products,
        &workspace->base_pair_products_buffer);
  } else {
    apply_exact_ao_pair_kernel(
        ao_integral_input,
        workspace->pair_coefficients_buffer,
        n_basis_functions,
        n_active_pairs,
        &workspace->base_pair_products_buffer);
  }

  build_mixed_ao_pair_to_active_pair_coefficients(
      workspace->dense_active_coefficients_buffer,
      workspace->dense_active_direction_buffer,
      n_basis_functions,
      n_active_orbitals,
      active_pairs,
      &workspace->directional_pair_coefficients_buffer);
  apply_exact_ao_pair_kernel(
      ao_integral_input,
      workspace->directional_pair_coefficients_buffer,
      n_basis_functions,
      n_active_pairs,
      &workspace->directional_pair_products_buffer);

  workspace->delta_active_pair_matrix.resize(
      static_cast<Eigen::Index>(n_active_pairs),
      static_cast<Eigen::Index>(n_active_pairs));
  workspace->delta_active_pair_matrix.setZero();
  for (std::size_t ao_pair_index = 0;
       ao_pair_index < n_ao_pairs;
       ++ao_pair_index) {
    const Eigen::Map<const Eigen::VectorXd> pair_coefficient_row(
        workspace->pair_coefficients_buffer.data() +
            ao_pair_index * n_active_pairs,
        static_cast<Eigen::Index>(n_active_pairs));
    const Eigen::Map<const Eigen::VectorXd> pair_product_row(
        workspace->base_pair_products_buffer.data() +
            ao_pair_index * n_active_pairs,
        static_cast<Eigen::Index>(n_active_pairs));
    const Eigen::Map<const Eigen::VectorXd> directional_pair_coefficient_row(
        workspace->directional_pair_coefficients_buffer.data() +
            ao_pair_index * n_active_pairs,
        static_cast<Eigen::Index>(n_active_pairs));
    const Eigen::Map<const Eigen::VectorXd> directional_pair_product_row(
        workspace->directional_pair_products_buffer.data() +
            ao_pair_index * n_active_pairs,
        static_cast<Eigen::Index>(n_active_pairs));
    workspace->delta_active_pair_matrix.noalias() +=
        directional_pair_coefficient_row * pair_product_row.transpose();
    workspace->delta_active_pair_matrix.noalias() +=
        pair_coefficient_row * directional_pair_product_row.transpose();
  }

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
      (*delta_packed_active_two_electron_integrals)[xmvb::to_size(
          packed_index)] =
          workspace->delta_active_pair_matrix(
              static_cast<Eigen::Index>(left_active_pair_index),
              static_cast<Eigen::Index>(right_active_pair_index));
    }
  }
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
      xmvb::to_size(n_basis_functions) * (n_basis_functions + 1) / 2;
  const std::size_t expected_packed_gradient_size =
      n_active_pairs * (n_active_pairs + 1) / 2;
  if (packed_active_two_electron_gradient.size() != expected_packed_gradient_size) {
    throw std::invalid_argument("packed active two-electron gradient size mismatch");
  }

  ExactPackedActiveTwoElectronAdjointCache cache;
  cache.n_basis_functions = n_basis_functions;
  cache.n_active_orbitals = n_active_orbitals;
  cache.accepted_dense_active_coefficients = accepted_dense_active_coefficients;
  cache.accepted_dense_active_coefficients_buffer =
      copy_matrix_to_legacy_row_buffer(accepted_dense_active_coefficients);
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
  cache.active_pair_gradient_matrix =
      build_active_pair_gradient_matrix(
          packed_active_two_electron_gradient,
          active_pairs);
  cache.active_pair_gradient_buffer =
      copy_matrix_to_legacy_row_buffer(cache.active_pair_gradient_matrix);
  if (exact_2e_fused_hvp_enabled() &&
      exact_2e_fused_hvp_supported_integral_layout(ao_integral_input)) {
    // These accepted-point backprop rows scale only with AO rows and active
    // pairs, not with determinant / spin-string counts. Keep them available
    // for the main HVP path so the AO kernel can stream directly into dense
    // active gradients without paying for `mixed * pair_gradient_matrix`
    // inside every Krylov matvec.
    build_accepted_active_pair_gradient_backprop_rows(
        cache,
        &cache.accepted_active_pair_gradient_backprop_rows_buffer);
  }

  if (accepted_active_space_two_electron_result != nullptr &&
      accepted_active_space_two_electron_result->dense_ao_pair_products.size() != 0) {
    if (xmvb::to_size(accepted_active_space_two_electron_result->dense_ao_pair_products.size()) !=
        n_ao_pairs * n_active_pairs) {
      throw std::invalid_argument("accepted dense AO pair product cache size mismatch");
    }
    cache.accepted_pair_products_buffer =
        copy_matrix_to_legacy_row_buffer(
            accepted_active_space_two_electron_result->dense_ao_pair_products);
  } else {
    const auto accepted_pair_coefficients =
        build_ao_pair_to_active_pair_coefficients(
            cache.accepted_dense_active_coefficients_buffer,
            n_basis_functions,
            n_active_orbitals,
            active_pairs);
    cache.accepted_pair_products_buffer =
        apply_exact_ao_pair_kernel(
            ao_integral_input,
            accepted_pair_coefficients,
            n_basis_functions,
            n_active_pairs);
  }

  cache.accepted_base_pair_gradients_buffer =
      multiply_pair_coefficients_by_gradient_matrix(
          cache.accepted_pair_products_buffer,
          cache.active_pair_gradient_matrix,
          cache.active_pair_gradient_buffer,
          n_ao_pairs,
          n_active_pairs);
  if (should_cache_accepted_base_pair_gradient_matrices(
          n_ao_pairs,
          n_active_orbitals)) {
    cache.accepted_base_pair_gradient_matrices_buffer =
        build_full_active_pair_gradient_matrices_from_cache(
            cache.accepted_base_pair_gradients_buffer,
            cache);
  }
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

  if (accepted_cache.accepted_dense_active_coefficients.rows() != n_basis_functions ||
      accepted_cache.accepted_dense_active_coefficients.cols() != n_active_orbitals ||
      dense_active_direction.rows() != n_basis_functions ||
      dense_active_direction.cols() != n_active_orbitals) {
    throw std::invalid_argument("dense active coefficient shape mismatch in exact two-electron HVP");
  }
  const std::size_t expected_dense_size =
      xmvb::to_size(n_basis_functions) * n_active_orbitals;
  copy_matrix_to_legacy_row_buffer_inplace(
      dense_active_direction,
      &workspace->dense_active_direction_buffer);

  const std::size_t n_active_pairs =
      accepted_cache.active_pair_first_indices.size();
  if (accepted_cache.active_pair_second_indices.size() != n_active_pairs) {
    throw std::invalid_argument("exact 2e cache active-pair index size mismatch");
  }
  const std::size_t n_ao_pairs =
      xmvb::to_size(n_basis_functions) * (n_basis_functions + 1) / 2;
  if (accepted_cache.active_pair_gradient_matrix.rows() !=
          static_cast<Eigen::Index>(n_active_pairs) ||
      accepted_cache.active_pair_gradient_matrix.cols() !=
          static_cast<Eigen::Index>(n_active_pairs) ||
      accepted_cache.active_pair_gradient_buffer.size() !=
          n_active_pairs * n_active_pairs) {
    throw std::invalid_argument("exact 2e cache pair-gradient size mismatch");
  }
  if (accepted_cache.accepted_base_pair_gradients_buffer.size() !=
      n_ao_pairs * n_active_pairs) {
    throw std::invalid_argument("exact 2e cache base-pair-gradient size mismatch");
  }

  if (!accepted_cache.accepted_base_pair_gradient_matrices_buffer.empty()) {
    backpropagate_fixed_pair_gradient_matrices_to_dense_active_coefficients_from_cache(
        workspace->dense_active_direction_buffer,
        accepted_cache,
        &workspace->dense_active_gradient_direction_buffer);
  } else {
    resize_and_zero(
        &workspace->dense_active_gradient_direction_buffer,
        expected_dense_size);
    accumulate_backpropagated_pair_coefficients_to_dense_active_coefficients_from_cache(
        accepted_cache.accepted_base_pair_gradients_buffer,
        workspace->dense_active_direction_buffer,
        accepted_cache,
        &workspace->dense_active_gradient_direction_buffer);
  }

  build_mixed_ao_pair_to_active_pair_coefficients_from_cache(
      accepted_cache.accepted_dense_active_coefficients_buffer,
      workspace->dense_active_direction_buffer,
      accepted_cache,
      &workspace->mixed_pair_coefficients_buffer);
  if (apply_exact_ao_pair_kernel_and_backprop_from_cached_rows(
          ao_integral_input,
          workspace->mixed_pair_coefficients_buffer,
          accepted_cache,
          &workspace->dense_active_gradient_direction_buffer)) {
    if (dense_active_gradient_direction != nullptr) {
      copy_legacy_row_buffer_to_matrix_inplace(
          workspace->dense_active_gradient_direction_buffer,
          n_basis_functions,
          n_active_orbitals,
          dense_active_gradient_direction);
    }
    return;
  }

  multiply_pair_coefficients_by_gradient_matrix(
      workspace->mixed_pair_coefficients_buffer,
      accepted_cache.active_pair_gradient_matrix,
      accepted_cache.active_pair_gradient_buffer,
      n_ao_pairs,
      n_active_pairs,
      &workspace->transformed_pair_coefficients_buffer);
  if (exact_2e_fused_hvp_enabled() &&
      apply_exact_ao_pair_kernel_and_backprop_to_dense_active_coefficients_from_cache(
          ao_integral_input,
          workspace->transformed_pair_coefficients_buffer,
          accepted_cache,
          &workspace->dense_active_gradient_direction_buffer)) {
    if (dense_active_gradient_direction != nullptr) {
      copy_legacy_row_buffer_to_matrix_inplace(
          workspace->dense_active_gradient_direction_buffer,
          n_basis_functions,
          n_active_orbitals,
          dense_active_gradient_direction);
    }
    return;
  }

  workspace->pair_gradients_buffer =
      apply_exact_ao_pair_kernel(
          ao_integral_input,
          workspace->transformed_pair_coefficients_buffer,
          n_basis_functions,
          n_active_pairs);
  accumulate_backpropagated_pair_coefficients_to_dense_active_coefficients_from_cache(
      workspace->pair_gradients_buffer,
      accepted_cache.accepted_dense_active_coefficients_buffer,
      accepted_cache,
      &workspace->dense_active_gradient_direction_buffer);
  if (dense_active_gradient_direction != nullptr) {
    copy_legacy_row_buffer_to_matrix_inplace(
        workspace->dense_active_gradient_direction_buffer,
        n_basis_functions,
        n_active_orbitals,
        dense_active_gradient_direction);
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
