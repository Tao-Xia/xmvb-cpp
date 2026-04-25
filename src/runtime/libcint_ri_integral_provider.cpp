#include "runtime/libcint_ri_integral_provider.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "runtime/libcint_direct_shell_evaluator.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace xmvb::vb {

namespace {

constexpr int kWhitenColumnBlockSize = 256;
constexpr std::size_t kDenseAoFactorMatrixCacheMaxBytes =
    256ull * 1024ull * 1024ull;

struct ThreeCenterShellTask {
  int primary_left_shell = 0;
  int primary_right_shell = 0;
  int auxiliary_shell = 0;
};

int ao_pair_index(int first, int second) {
  if (first >= second) {
    return first * (first + 1) / 2 + second;
  }
  return second * (second + 1) / 2 + first;
}

bool ri_dense_ao_factor_cache_enabled() {
  const char* value = std::getenv("XMVB_CPP_ENABLE_RI_AO_FACTOR_MATRIX_CACHE");
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  if (std::strcmp(value, "1") == 0 ||
      std::strcmp(value, "true") == 0 ||
      std::strcmp(value, "TRUE") == 0) {
    return true;
  }
  if (std::strcmp(value, "0") == 0 ||
      std::strcmp(value, "false") == 0 ||
      std::strcmp(value, "FALSE") == 0) {
    return false;
  }
  throw std::invalid_argument(
      "XMVB_CPP_ENABLE_RI_AO_FACTOR_MATRIX_CACHE must be a boolean");
}

bool can_build_dense_lower_ao_factor_cache(
    int n_auxiliary_functions,
    int n_basis_functions) {
  if (!ri_dense_ao_factor_cache_enabled()) {
    return false;
  }
  const std::size_t matrix_size =
      n_basis_functions *
      n_basis_functions;
  if (matrix_size == 0 ||
      matrix_size >
          std::numeric_limits<std::size_t>::max() /
              n_auxiliary_functions) {
    return false;
  }
  const std::size_t dense_value_count =
      matrix_size * n_auxiliary_functions;
  return dense_value_count <=
      kDenseAoFactorMatrixCacheMaxBytes / sizeof(double);
}

std::vector<double> build_dense_lower_ao_factor_matrices(
    const Eigen::Ref<const Eigen::MatrixXd>& packed_factor_rows,
    int n_auxiliary_functions,
    int n_basis_functions) {
  const std::size_t matrix_size =
      n_basis_functions *
      n_basis_functions;
  std::vector<double> dense_lower_factor_matrices(
      n_auxiliary_functions * matrix_size,
      0.0);

#pragma omp parallel for schedule(static)
  for (int auxiliary_index = 0;
       auxiliary_index < n_auxiliary_functions;
       ++auxiliary_index) {
    double* dense_matrix =
        dense_lower_factor_matrices.data() +
        auxiliary_index * matrix_size;

    std::size_t packed_index = 0;
    for (int column = 0; column < n_basis_functions; ++column) {
      for (int row = 0; row <= column; ++row) {
        // `metric_whitened_ao_pair_factors` follow the same packed ordering that
        // `unpack_packed_factor_row_lower_triangle()` consumes in the AO RI
        // operator: outer loop over the physical row index of the lower
        // triangle, inner loop over the physical column index.  For a
        // column-major dense matrix that means writing entry `(column, row)`,
        // not `(row, column)`.
        //
        // The source matrix is also column-major, so packed-factor rows are not
        // contiguous. Access through `(auxiliary, packed_pair)` indexing rather
        // than row-pointer arithmetic.
        dense_matrix[
            row * n_basis_functions +
            column] =
            packed_factor_rows(auxiliary_index, static_cast<Eigen::Index>(packed_index++));
      }
    }
  }
  return dense_lower_factor_matrices;
}

std::vector<ThreeCenterShellTask> build_three_center_shell_tasks(
    int n_primary_shells,
    int n_auxiliary_shells) {
  std::vector<ThreeCenterShellTask> tasks;
  tasks.reserve(
      n_auxiliary_shells *
      n_primary_shells *
      (n_primary_shells + 1) / 2);
  for (int primary_left_shell = 0; primary_left_shell < n_primary_shells; ++primary_left_shell) {
    for (int primary_right_shell = 0;
         primary_right_shell <= primary_left_shell;
         ++primary_right_shell) {
      for (int auxiliary_shell = 0; auxiliary_shell < n_auxiliary_shells; ++auxiliary_shell) {
        ThreeCenterShellTask task;
        task.primary_left_shell = primary_left_shell;
        task.primary_right_shell = primary_right_shell;
        task.auxiliary_shell = auxiliary_shell;
        tasks.push_back(task);
      }
    }
  }
  return tasks;
}

void scatter_three_center_shell_block(
    const LibcintThreeCenterShellBlock& block,
    Eigen::MatrixXd* raw_ao_pair_factors) {
  if (raw_ao_pair_factors == nullptr) {
    throw std::invalid_argument("raw_ao_pair_factors must not be null");
  }
  for (int auxiliary_local = 0;
       auxiliary_local < block.auxiliary_ao_count;
       ++auxiliary_local) {
    const int auxiliary_index = block.auxiliary_ao_offset + auxiliary_local;
    for (int right_local = 0;
         right_local < block.primary_right_ao_count;
         ++right_local) {
      const int right_index = block.primary_right_ao_offset + right_local;
      for (int left_local = 0;
           left_local < block.primary_left_ao_count;
           ++left_local) {
        const int left_index = block.primary_left_ao_offset + left_local;
        if (right_index > left_index) {
          continue;
        }
        const int packed_pair_index = ao_pair_index(left_index, right_index);
        const std::size_t local_index =
            left_local +
            right_local * block.primary_left_ao_count +
            auxiliary_local *
                block.primary_left_ao_count * block.primary_right_ao_count;
        (*raw_ao_pair_factors)(auxiliary_index, packed_pair_index) =
            block.values[local_index];
      }
    }
  }
}

}  // namespace

LibcintRiIntegralProviderResult build_libcint_ri_integral_provider_result(
    const LibcintInput& primary_input,
    const LibcintInput& auxiliary_input,
    double metric_eigenvalue_cutoff) {
  if (!(metric_eigenvalue_cutoff >= 0.0)) {
    throw std::invalid_argument("metric_eigenvalue_cutoff must be non-negative");
  }

  LibcintRiIntegralProviderResult result;
  result.auxiliary_input = auxiliary_input;
  LibcintDirectShellEvaluator evaluator(primary_input, result.auxiliary_input);

  result.n_basis_functions = evaluator.n_basis_functions();
  result.n_auxiliary_functions = evaluator.n_auxiliary_basis_functions();
  result.n_packed_ao_pairs =
      result.n_basis_functions * (result.n_basis_functions + 1) / 2;
  if (result.n_auxiliary_functions <= 0 || result.n_packed_ao_pairs <= 0) {
    throw std::runtime_error("invalid RI dimensions inferred from libcint inputs");
  }

  Eigen::MatrixXd auxiliary_metric(
      result.n_auxiliary_functions,
      result.n_auxiliary_functions);
  auxiliary_metric.setZero();
  for (int left_shell = 0; left_shell < result.auxiliary_input.n_shells; ++left_shell) {
    for (int right_shell = 0; right_shell <= left_shell; ++right_shell) {
      const auto block =
          evaluator.evaluate_auxiliary_metric_shell_pair(left_shell, right_shell);
      for (int column = 0; column < block.right_ao_count; ++column) {
        const int global_column = block.right_ao_offset + column;
        for (int row = 0; row < block.left_ao_count; ++row) {
          const int global_row = block.left_ao_offset + row;
          const std::size_t local_index =
              row +
              column * block.left_ao_count;
          const double value = block.values[local_index];
          auxiliary_metric(global_row, global_column) = value;
          auxiliary_metric(global_column, global_row) = value;
        }
      }
    }
  }
  Eigen::MatrixXd raw_ao_pair_factors(
      result.n_auxiliary_functions,
      result.n_packed_ao_pairs);
  raw_ao_pair_factors.setZero();
  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif
  if (n_threads <= 1) {
    for (int left_shell = 0; left_shell < primary_input.n_shells; ++left_shell) {
      for (int right_shell = 0; right_shell <= left_shell; ++right_shell) {
        for (int auxiliary_shell = 0;
             auxiliary_shell < result.auxiliary_input.n_shells;
             ++auxiliary_shell) {
          const auto block =
              evaluator.evaluate_three_center_shell_block(left_shell, right_shell, auxiliary_shell);
          scatter_three_center_shell_block(
              block,
              &raw_ao_pair_factors);
        }
      }
    }
  } else {
    const auto three_center_tasks =
        build_three_center_shell_tasks(
            primary_input.n_shells,
            result.auxiliary_input.n_shells);

#pragma omp parallel
    {
      LibcintDirectShellEvaluator local_evaluator(primary_input, result.auxiliary_input);
#pragma omp for schedule(dynamic)
      for (std::ptrdiff_t task_index = 0;
           task_index < static_cast<std::ptrdiff_t>(three_center_tasks.size());
           ++task_index) {
        const auto& task = three_center_tasks[task_index];
        const auto block =
            local_evaluator.evaluate_three_center_shell_block(
                task.primary_left_shell,
                task.primary_right_shell,
                task.auxiliary_shell);
        scatter_three_center_shell_block(
            block,
            &raw_ao_pair_factors);
      }
    }
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> metric_solver(auxiliary_metric);
  if (metric_solver.info() != Eigen::Success) {
    throw std::runtime_error("failed to diagonalize the RI auxiliary metric");
  }

  const auto eigenvalues = metric_solver.eigenvalues();
  Eigen::MatrixXd inverse_sqrt_metric =
      metric_solver.eigenvectors() *
      eigenvalues.unaryExpr(
          [metric_eigenvalue_cutoff](double eigenvalue) -> double {
            if (eigenvalue <= metric_eigenvalue_cutoff) {
              return 0.0;
            }
            return 1.0 / std::sqrt(eigenvalue);
          }).asDiagonal() *
      metric_solver.eigenvectors().transpose();
  Eigen::MatrixXd whitened_ao_pair_factors(
      result.n_auxiliary_functions,
      result.n_packed_ao_pairs);

#pragma omp parallel for schedule(static)
  for (int pair_column_offset = 0;
       pair_column_offset < result.n_packed_ao_pairs;
       pair_column_offset += kWhitenColumnBlockSize) {
    const int block_column_count =
        std::min(kWhitenColumnBlockSize,
                 result.n_packed_ao_pairs - pair_column_offset);
    whitened_ao_pair_factors.middleCols(pair_column_offset, block_column_count).noalias() =
        inverse_sqrt_metric *
        raw_ao_pair_factors.middleCols(pair_column_offset, block_column_count);
  }
  result.auxiliary_metric_matrix = auxiliary_metric;
  result.metric_whitened_ao_pair_factors = whitened_ao_pair_factors;
  if (can_build_dense_lower_ao_factor_cache(
          result.n_auxiliary_functions,
          result.n_basis_functions)) {
    result.metric_whitened_ao_factor_matrices_lower =
        build_dense_lower_ao_factor_matrices(
            whitened_ao_pair_factors,
            result.n_auxiliary_functions,
            result.n_basis_functions);
  }
  return result;
}

LibcintRiIntegralProviderResult LibcintRiIntegralProvider::build(
    const LibcintInput& primary_input,
    const LibcintRiIntegralProviderOptions& options) const {
  LibcintAuxiliaryBasisBuilder auxiliary_basis_builder;
  const LibcintInput auxiliary_input = auxiliary_basis_builder.build(
      primary_input,
      options.auxiliary_basis_options);
  return build_libcint_ri_integral_provider_result(
      primary_input,
      auxiliary_input,
      options.metric_eigenvalue_cutoff);
}

LibcintRiIntegralProviderResult LibcintRiIntegralProvider::build(
    const LibcintInput& primary_input,
    const LibcintInput& auxiliary_input,
    const LibcintRiIntegralProviderOptions& options) const {
  return build_libcint_ri_integral_provider_result(
      primary_input,
      auxiliary_input,
      options.metric_eigenvalue_cutoff);
}

}  // namespace xmvb::vb
