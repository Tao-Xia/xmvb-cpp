#include "runtime/libcint_materialized_integral_provider.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "runtime/libcint_direct_shell_evaluator.hpp"

namespace xmvb::vb {

namespace {

int pair_index(int first, int second) {
  if (first >= second) {
    return first * (first + 1) / 2 + second;
  }
  return second * (second + 1) / 2 + first;
}

}  // namespace

MaterializedAoIntegralBuffers LibcintMaterializedIntegralProvider::build(
    const LibcintInput& input,
    const LibcintMaterializedIntegralProviderOptions& options) const {
  if (!(options.integral_tolerance >= 0.0)) {
    throw std::invalid_argument("integral_tolerance must be non-negative");
  }

  LibcintDirectShellEvaluator evaluator(input);
  const int n_basis_functions = evaluator.n_basis_functions();
  const int n_shells = input.n_shells;
  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif

  std::vector<std::unique_ptr<LibcintDirectShellEvaluator>> thread_evaluators;
  if (n_threads > 1) {
    thread_evaluators.reserve(n_threads - 1);
    for (int thread_index = 1; thread_index < n_threads; ++thread_index) {
      thread_evaluators.push_back(std::make_unique<LibcintDirectShellEvaluator>(input));
    }
  }

  MaterializedAoIntegralBuffers buffers;
  buffers.n_basis_functions = n_basis_functions;
  buffers.ao_core_hamiltonian_matrix =
      Eigen::MatrixXd::Zero(n_basis_functions, n_basis_functions);

  if (n_threads <= 1) {
    for (int left_shell = 0; left_shell < n_shells; ++left_shell) {
      for (int right_shell = 0; right_shell <= left_shell; ++right_shell) {
        const auto core_h_block =
            evaluator.evaluate_core_hamiltonian_shell_pair(left_shell, right_shell);
        for (int column = 0; column < core_h_block.right_ao_count; ++column) {
          const int global_column = core_h_block.right_ao_offset + column;
          for (int row = 0; row < core_h_block.left_ao_count; ++row) {
            const int global_row = core_h_block.left_ao_offset + row;
            const std::size_t local_index =
                column * core_h_block.left_ao_count + row;
            const double value = core_h_block.values[local_index];
            buffers.ao_core_hamiltonian_matrix(global_row, global_column) = value;
            buffers.ao_core_hamiltonian_matrix(global_column, global_row) = value;
          }
        }
      }
    }
  } else {
#pragma omp parallel
    {
      int thread_index = 0;
#ifdef _OPENMP
      thread_index = omp_get_thread_num();
#endif
      LibcintDirectShellEvaluator* thread_evaluator = &evaluator;
      if (thread_index > 0) {
        thread_evaluator =
            thread_evaluators[thread_index - 1].get();
      }

#pragma omp for schedule(dynamic)
      for (int left_shell = 0; left_shell < n_shells; ++left_shell) {
        for (int right_shell = 0; right_shell <= left_shell; ++right_shell) {
          const auto core_h_block =
              thread_evaluator->evaluate_core_hamiltonian_shell_pair(left_shell, right_shell);
          for (int column = 0; column < core_h_block.right_ao_count; ++column) {
            const int global_column = core_h_block.right_ao_offset + column;
            for (int row = 0; row < core_h_block.left_ao_count; ++row) {
              const int global_row = core_h_block.left_ao_offset + row;
              const std::size_t local_index =
                  column * core_h_block.left_ao_count + row;
              const double value = core_h_block.values[local_index];
              buffers.ao_core_hamiltonian_matrix(global_row, global_column) = value;
              buffers.ao_core_hamiltonian_matrix(global_column, global_row) = value;
            }
          }
        }
      }
    }
  }

  std::vector<double> shell_pair_maxima;
  if (options.use_historical_shell_pair_prescreen) {
    shell_pair_maxima.assign(
        n_shells * (n_shells + 1) / 2,
        0.0);
    if (n_threads <= 1) {
      for (int shell_i = 0; shell_i < n_shells; ++shell_i) {
        for (int shell_j = 0; shell_j <= shell_i; ++shell_j) {
          shell_pair_maxima[pair_index(shell_i, shell_j)] =
              evaluator.evaluate_max_abs_raw_two_electron_shell_pair(shell_i, shell_j);
        }
      }
    } else {
#pragma omp parallel
      {
        int thread_index = 0;
#ifdef _OPENMP
        thread_index = omp_get_thread_num();
#endif
        LibcintDirectShellEvaluator* thread_evaluator = &evaluator;
        if (thread_index > 0) {
          thread_evaluator =
              thread_evaluators[thread_index - 1].get();
        }

#pragma omp for schedule(dynamic)
        for (int shell_i = 0; shell_i < n_shells; ++shell_i) {
          for (int shell_j = 0; shell_j <= shell_i; ++shell_j) {
            shell_pair_maxima[pair_index(shell_i, shell_j)] =
                thread_evaluator->evaluate_max_abs_raw_two_electron_shell_pair(
                    shell_i,
                    shell_j);
          }
        }
      }
    }
  }

  auto append_shell_i_integrals =
      [&](LibcintDirectShellEvaluator& shell_evaluator,
          int shell_i,
          std::vector<double>& values,
          std::vector<int>& indices) {
        // Keep the canonical `(i >= j, i >= k, k >= l)` AO ordering used by
        // the downstream sparse AO kernels, but write the final `(i,j,k,l)`
        // tuples directly instead of staging packed indices and decoding them
        // again after a global sort.
        for (int shell_j = 0; shell_j <= shell_i; ++shell_j) {
          for (int shell_k = 0; shell_k <= shell_i; ++shell_k) {
            for (int shell_l = 0; shell_l <= shell_k; ++shell_l) {
              if (options.use_historical_shell_pair_prescreen) {
                const double shell_pair_product =
                    shell_pair_maxima[pair_index(shell_i, shell_j)] *
                    shell_pair_maxima[pair_index(shell_k, shell_l)];
                if (std::sqrt(shell_pair_product) < options.integral_tolerance) {
                  continue;
                }
              }
              const auto quartet = shell_evaluator.evaluate_two_electron_shell_quartet(
                  shell_i,
                  shell_j,
                  shell_k,
                  shell_l);
              for (int local_i = 0; local_i < quartet.ao_count_i; ++local_i) {
                const int i = quartet.ao_offset_i + local_i;
                for (int local_j = 0; local_j < quartet.ao_count_j; ++local_j) {
                  const int j = quartet.ao_offset_j + local_j;
                  if (j > i) {
                    continue;
                  }
                  for (int local_k = 0; local_k < quartet.ao_count_k; ++local_k) {
                    const int k = quartet.ao_offset_k + local_k;
                    if (k > i) {
                      continue;
                    }
                    for (int local_l = 0; local_l < quartet.ao_count_l; ++local_l) {
                      const int l = quartet.ao_offset_l + local_l;
                      if ((i == k && l > j) || (i != k && l > k)) {
                        continue;
                      }
                      const std::size_t local_index =
                          local_i +
                          local_j * quartet.ao_count_i +
                          local_k * quartet.ao_count_i * quartet.ao_count_j +
                          local_l * quartet.ao_count_i *
                              quartet.ao_count_j * quartet.ao_count_k;
                      const double value = quartet.values[local_index];
                      if (std::abs(value) < options.integral_tolerance) {
                        continue;
                      }
                      values.push_back(value);
                      indices.push_back(i);
                      indices.push_back(j);
                      indices.push_back(k);
                      indices.push_back(l);
                    }
                  }
                }
              }
            }
          }
        }
      };

  if (n_threads <= 1) {
    for (int shell_i = 0; shell_i < n_shells; ++shell_i) {
      append_shell_i_integrals(
          evaluator,
          shell_i,
          buffers.ao_two_electron_integral_values,
          buffers.ao_two_electron_integral_indices);
    }
  } else {
    std::vector<std::vector<double>> shell_integral_values(n_shells);
    std::vector<std::vector<int>> shell_integral_indices(n_shells);

#pragma omp parallel
    {
      int thread_index = 0;
#ifdef _OPENMP
      thread_index = omp_get_thread_num();
#endif
      LibcintDirectShellEvaluator* thread_evaluator = &evaluator;
      if (thread_index > 0) {
        thread_evaluator =
            thread_evaluators[thread_index - 1].get();
      }

#pragma omp for schedule(dynamic)
      for (int shell_i = 0; shell_i < n_shells; ++shell_i) {
        append_shell_i_integrals(
            *thread_evaluator,
            shell_i,
            shell_integral_values[shell_i],
            shell_integral_indices[shell_i]);
      }
    }

    std::size_t total_integral_count = 0;
    for (const auto& values : shell_integral_values) {
      total_integral_count += values.size();
    }
    buffers.ao_two_electron_integral_values.reserve(total_integral_count);
    buffers.ao_two_electron_integral_indices.reserve(total_integral_count * 4);
    for (int shell_i = 0; shell_i < n_shells; ++shell_i) {
      auto& values = shell_integral_values[shell_i];
      auto& indices = shell_integral_indices[shell_i];
      buffers.ao_two_electron_integral_values.insert(
          buffers.ao_two_electron_integral_values.end(),
          std::make_move_iterator(values.begin()),
          std::make_move_iterator(values.end()));
      buffers.ao_two_electron_integral_indices.insert(
          buffers.ao_two_electron_integral_indices.end(),
          std::make_move_iterator(indices.begin()),
          std::make_move_iterator(indices.end()));
    }
  }

  return buffers;
}

}  // namespace xmvb::vb
