#include "libcint/materialized_provider.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "libcint/direct_shell.hpp"

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

  auto visit_shell_i_integrals =
      [&](LibcintDirectShellEvaluator& shell_evaluator,
          int shell_i,
          auto&& visit_integral) {
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
                      visit_integral(
                          value,
                          pair_index(i, j),
                          pair_index(k, l));
                    }
                  }
                }
              }
            }
          }
        }
      };

  const std::size_t n_basis = static_cast<std::size_t>(n_basis_functions);
  const std::size_t n_ao_pairs = n_basis * (n_basis + 1) / 2;
  const std::size_t max_32_bit_graph_edges =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  const bool symmetric_graph_can_exceed_32_bits =
      n_ao_pairs != 0 &&
      n_ao_pairs > max_32_bit_graph_edges / n_ao_pairs;

  if (symmetric_graph_can_exceed_32_bits) {
    // A fully connected symmetric AO-pair CSR graph has n_ao_pairs^2
    // directed edges.  Once that can exceed INT_MAX, the downstream graph
    // keeps this once-symmetry-reduced integral stream instead.  Avoid the
    // much larger materialization peak from retaining every per-shell vector
    // while also growing/copying a complete aggregate: count first, allocate
    // the exact final buffers, then let each shell write its disjoint range.
    std::vector<std::size_t> shell_integral_counts(n_shells, 0);
    if (n_threads <= 1) {
      for (int shell_i = 0; shell_i < n_shells; ++shell_i) {
        visit_shell_i_integrals(
            evaluator,
            shell_i,
            [&](double, int, int) {
              ++shell_integral_counts[shell_i];
            });
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
          thread_evaluator = thread_evaluators[thread_index - 1].get();
        }

#pragma omp for schedule(dynamic)
        for (int shell_i = 0; shell_i < n_shells; ++shell_i) {
          std::size_t count = 0;
          visit_shell_i_integrals(
              *thread_evaluator,
              shell_i,
              [&](double, int, int) { ++count; });
          shell_integral_counts[shell_i] = count;
        }
      }
    }

    std::vector<std::size_t> shell_integral_offsets(n_shells + 1, 0);
    for (int shell_i = 0; shell_i < n_shells; ++shell_i) {
      const std::size_t previous_offset = shell_integral_offsets[shell_i];
      const std::size_t shell_count = shell_integral_counts[shell_i];
      if (shell_count >
          std::numeric_limits<std::size_t>::max() - previous_offset) {
        throw std::overflow_error("materialized ERI count exceeds size_t");
      }
      shell_integral_offsets[shell_i + 1] = previous_offset + shell_count;
    }
    const std::size_t total_integral_count = shell_integral_offsets.back();
    if (total_integral_count >
        std::numeric_limits<std::size_t>::max() / 2) {
      throw std::overflow_error("directed ERI capacity exceeds size_t");
    }
    buffers.two_electron_values.reserve(2 * total_integral_count);
    buffers.two_electron_values.resize(total_integral_count);
    buffers.left_pair_indices.resize(total_integral_count);
    buffers.right_pair_indices.resize(total_integral_count);

    std::vector<std::size_t> filled_shell_counts(n_shells, 0);
    auto fill_shell_i = [&](LibcintDirectShellEvaluator& shell_evaluator,
                            int shell_i) {
      const std::size_t begin = shell_integral_offsets[shell_i];
      const std::size_t expected_count = shell_integral_counts[shell_i];
      std::size_t count = 0;
      visit_shell_i_integrals(
          shell_evaluator,
          shell_i,
          [&](double value, int left_pair, int right_pair) {
            if (count < expected_count) {
              const std::size_t offset = begin + count;
              buffers.two_electron_values[offset] = value;
              buffers.left_pair_indices[offset] = left_pair;
              buffers.right_pair_indices[offset] = right_pair;
            }
            ++count;
          });
      filled_shell_counts[shell_i] = count;
    };

    if (n_threads <= 1) {
      for (int shell_i = 0; shell_i < n_shells; ++shell_i) {
        fill_shell_i(evaluator, shell_i);
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
          thread_evaluator = thread_evaluators[thread_index - 1].get();
        }

#pragma omp for schedule(dynamic)
        for (int shell_i = 0; shell_i < n_shells; ++shell_i) {
          fill_shell_i(*thread_evaluator, shell_i);
        }
      }
    }

    for (int shell_i = 0; shell_i < n_shells; ++shell_i) {
      if (filled_shell_counts[shell_i] != shell_integral_counts[shell_i]) {
        throw std::runtime_error(
            "materialized ERI count changed between count and fill passes");
      }
    }
  } else if (n_threads <= 1) {
    for (int shell_i = 0; shell_i < n_shells; ++shell_i) {
      visit_shell_i_integrals(
          evaluator,
          shell_i,
          [&](double value, int left_pair, int right_pair) {
            buffers.two_electron_values.push_back(value);
            buffers.left_pair_indices.push_back(left_pair);
            buffers.right_pair_indices.push_back(right_pair);
          });
    }
  } else {
    std::vector<std::vector<double>> shell_integral_values(n_shells);
    std::vector<std::vector<int>> shell_left_pairs(n_shells);
    std::vector<std::vector<int>> shell_right_pairs(n_shells);

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
        visit_shell_i_integrals(
            *thread_evaluator,
            shell_i,
            [&](double value, int left_pair, int right_pair) {
              shell_integral_values[shell_i].push_back(value);
              shell_left_pairs[shell_i].push_back(left_pair);
              shell_right_pairs[shell_i].push_back(right_pair);
            });
      }
    }

    std::size_t total_integral_count = 0;
    for (const auto& values : shell_integral_values) {
      total_integral_count += values.size();
    }
    if (total_integral_count >
        std::numeric_limits<std::size_t>::max() / 2) {
      throw std::overflow_error("directed ERI capacity exceeds size_t");
    }
    // Reserve the final symmetric-CSR value capacity now. The graph builder
    // expands this same allocation in place instead of retaining the raw ERI
    // values beside a second directed value array.
    buffers.two_electron_values.reserve(2 * total_integral_count);
    buffers.left_pair_indices.reserve(total_integral_count);
    buffers.right_pair_indices.reserve(total_integral_count);
    for (int shell_i = 0; shell_i < n_shells; ++shell_i) {
      auto& values = shell_integral_values[shell_i];
      auto& left_pairs = shell_left_pairs[shell_i];
      auto& right_pairs = shell_right_pairs[shell_i];
      buffers.two_electron_values.insert(
          buffers.two_electron_values.end(),
          std::make_move_iterator(values.begin()),
          std::make_move_iterator(values.end()));
      buffers.left_pair_indices.insert(
          buffers.left_pair_indices.end(),
          std::make_move_iterator(left_pairs.begin()),
          std::make_move_iterator(left_pairs.end()));
      buffers.right_pair_indices.insert(
          buffers.right_pair_indices.end(),
          std::make_move_iterator(right_pairs.begin()),
          std::make_move_iterator(right_pairs.end()));
    }
  }

  return buffers;
}

}  // namespace xmvb::vb
