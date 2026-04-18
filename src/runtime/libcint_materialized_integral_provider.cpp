#include "runtime/libcint_materialized_integral_provider.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "runtime/libcint_direct_shell_evaluator.hpp"

namespace xmvb::vb {

namespace {

struct PackedIntegralEntry {
  long packed_index = 0;
  double value = 0.0;
};

int pair_index(int first, int second) {
  if (first >= second) {
    return first * (first + 1) / 2 + second;
  }
  return second * (second + 1) / 2 + first;
}

long packed_integral_index(int i, int j, int k, int l) {
  const long ij = static_cast<long>(pair_index(i, j));
  const long kl = static_cast<long>(pair_index(k, l));
  if (ij >= kl) {
    return ij * (ij + 1) / 2 + kl;
  }
  return kl * (kl + 1) / 2 + ij;
}

long triangular_number(long index) {
  return index * (index + 1) / 2;
}

/**
 * @brief Inverts the packed triangular index used for AO pair storage.
 *
 * The materialized libcint path stores `(ij|kl)` using the same lower-triangular
 * packing as the legacy runtime. The hot path later needs to recover `(ij, kl)`
 * and then `(i, j, k, l)` from that packed index. Using the previous linear
 * search here makes the decode cost scale with the pair index itself, which is
 * prohibitively expensive for large AO bases. This helper computes the inverse
 * in constant time up to a tiny rounding correction.
 */
long inverse_triangular_index(long packed_value) {
  if (packed_value < 0) {
    throw std::invalid_argument("packed triangular index must be non-negative");
  }

  const long double discriminant =
      8.0L * static_cast<long double>(packed_value) + 1.0L;
  long candidate = static_cast<long>(
      (std::sqrt(discriminant) - 1.0L) * 0.5L);
  while (triangular_number(candidate + 1) <= packed_value) {
    ++candidate;
  }
  while (candidate > 0 && triangular_number(candidate) > packed_value) {
    --candidate;
  }
  return candidate;
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
    thread_evaluators.reserve(xmvb::to_size(n_threads - 1));
    for (int thread_index = 1; thread_index < n_threads; ++thread_index) {
      thread_evaluators.push_back(std::make_unique<LibcintDirectShellEvaluator>(input));
    }
  }

  MaterializedAoIntegralBuffers buffers;
  buffers.n_basis_functions = n_basis_functions;
  buffers.ao_core_hamiltonian_matrix.assign(
      xmvb::to_size(n_basis_functions) * n_basis_functions,
      0.0);

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
                xmvb::to_size(column) * core_h_block.left_ao_count + row;
            const double value = core_h_block.values[local_index];
            buffers.ao_core_hamiltonian_matrix[xmvb::to_size(global_column) *
                                                   n_basis_functions +
                                               global_row] = value;
            buffers.ao_core_hamiltonian_matrix[xmvb::to_size(global_row) *
                                                   n_basis_functions +
                                               global_column] = value;
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
            thread_evaluators[xmvb::to_size(thread_index - 1)].get();
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
                  xmvb::to_size(column) * core_h_block.left_ao_count + row;
              const double value = core_h_block.values[local_index];
              buffers.ao_core_hamiltonian_matrix[xmvb::to_size(global_column) *
                                                     n_basis_functions +
                                                 global_row] = value;
              buffers.ao_core_hamiltonian_matrix[xmvb::to_size(global_row) *
                                                     n_basis_functions +
                                                 global_column] = value;
            }
          }
        }
      }
    }
  }

  std::vector<double> shell_pair_maxima;
  if (options.use_legacy_shell_pair_prescreen) {
    shell_pair_maxima.assign(
        xmvb::to_size(n_shells) * (n_shells + 1) / 2,
        0.0);
    if (n_threads <= 1) {
      for (int shell_i = 0; shell_i < n_shells; ++shell_i) {
        for (int shell_j = 0; shell_j <= shell_i; ++shell_j) {
          shell_pair_maxima[xmvb::to_size(pair_index(shell_i, shell_j))] =
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
              thread_evaluators[xmvb::to_size(thread_index - 1)].get();
        }

#pragma omp for schedule(dynamic)
        for (int shell_i = 0; shell_i < n_shells; ++shell_i) {
          for (int shell_j = 0; shell_j <= shell_i; ++shell_j) {
            shell_pair_maxima[xmvb::to_size(pair_index(shell_i, shell_j))] =
                thread_evaluator->evaluate_max_abs_raw_two_electron_shell_pair(
                    shell_i,
                    shell_j);
          }
        }
      }
    }
  }

  std::vector<PackedIntegralEntry> packed_integrals;
  if (n_threads <= 1) {
    for (int shell_i = 0; shell_i < n_shells; ++shell_i) {
      for (int shell_j = 0; shell_j <= shell_i; ++shell_j) {
        for (int shell_k = 0; shell_k <= shell_i; ++shell_k) {
          for (int shell_l = 0; shell_l <= shell_k; ++shell_l) {
            if (options.use_legacy_shell_pair_prescreen) {
              const double shell_pair_product =
                  shell_pair_maxima[xmvb::to_size(pair_index(shell_i, shell_j))] *
                  shell_pair_maxima[xmvb::to_size(pair_index(shell_k, shell_l))];
              if (std::sqrt(shell_pair_product) < options.integral_tolerance) {
                continue;
              }
            }
            const auto quartet =
                evaluator.evaluate_two_electron_shell_quartet(shell_i, shell_j, shell_k, shell_l);
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
                        xmvb::to_size(local_i) +
                        xmvb::to_size(local_j) * quartet.ao_count_i +
                        xmvb::to_size(local_k) * quartet.ao_count_i * quartet.ao_count_j +
                        xmvb::to_size(local_l) * quartet.ao_count_i *
                            quartet.ao_count_j * quartet.ao_count_k;
                    const double value = quartet.values[local_index];
                    if (std::abs(value) < options.integral_tolerance) {
                      continue;
                    }
                    packed_integrals.push_back({
                        .packed_index = packed_integral_index(i, j, k, l),
                        .value = value,
                    });
                  }
                }
              }
            }
          }
        }
      }
    }
  } else {
    std::vector<std::vector<PackedIntegralEntry>> thread_integrals(
        xmvb::to_size(n_threads));

#pragma omp parallel
    {
      int thread_index = 0;
#ifdef _OPENMP
      thread_index = omp_get_thread_num();
#endif
      LibcintDirectShellEvaluator* thread_evaluator = &evaluator;
      if (thread_index > 0) {
        thread_evaluator =
            thread_evaluators[xmvb::to_size(thread_index - 1)].get();
      }
      auto& local_integrals = thread_integrals[xmvb::to_size(thread_index)];

#pragma omp for schedule(dynamic)
      for (int shell_i = 0; shell_i < n_shells; ++shell_i) {
        for (int shell_j = 0; shell_j <= shell_i; ++shell_j) {
          for (int shell_k = 0; shell_k <= shell_i; ++shell_k) {
            for (int shell_l = 0; shell_l <= shell_k; ++shell_l) {
              if (options.use_legacy_shell_pair_prescreen) {
                const double shell_pair_product =
                    shell_pair_maxima[xmvb::to_size(pair_index(shell_i, shell_j))] *
                    shell_pair_maxima[xmvb::to_size(pair_index(shell_k, shell_l))];
                if (std::sqrt(shell_pair_product) < options.integral_tolerance) {
                  continue;
                }
              }
              const auto quartet = thread_evaluator->evaluate_two_electron_shell_quartet(
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
                          xmvb::to_size(local_i) +
                          xmvb::to_size(local_j) * quartet.ao_count_i +
                          xmvb::to_size(local_k) *
                              quartet.ao_count_i * quartet.ao_count_j +
                          xmvb::to_size(local_l) * quartet.ao_count_i *
                              quartet.ao_count_j * quartet.ao_count_k;
                      const double value = quartet.values[local_index];
                      if (std::abs(value) < options.integral_tolerance) {
                        continue;
                      }
                      local_integrals.push_back({
                          .packed_index = packed_integral_index(i, j, k, l),
                          .value = value,
                      });
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    std::size_t total_integral_count = 0;
    for (const auto& local_integrals : thread_integrals) {
      total_integral_count += local_integrals.size();
    }
    packed_integrals.reserve(total_integral_count);
    for (auto& local_integrals : thread_integrals) {
      packed_integrals.insert(
          packed_integrals.end(),
          std::make_move_iterator(local_integrals.begin()),
          std::make_move_iterator(local_integrals.end()));
    }
  }

  std::sort(
      packed_integrals.begin(),
      packed_integrals.end(),
      [](const PackedIntegralEntry& left, const PackedIntegralEntry& right) {
        return left.packed_index < right.packed_index;
      });

  buffers.ao_two_electron_integral_values.reserve(packed_integrals.size());
  buffers.ao_two_electron_integral_indices.reserve(packed_integrals.size() * 4);
  for (const PackedIntegralEntry& packed_integral : packed_integrals) {
    const long ij = inverse_triangular_index(packed_integral.packed_index);
    const long kl = packed_integral.packed_index - triangular_number(ij);
    const long i = inverse_triangular_index(ij);
    const long j = ij - triangular_number(i);
    const long k = inverse_triangular_index(kl);
    const long l = kl - triangular_number(k);
    buffers.ao_two_electron_integral_values.push_back(packed_integral.value);
    buffers.ao_two_electron_integral_indices.push_back(static_cast<int>(i));
    buffers.ao_two_electron_integral_indices.push_back(static_cast<int>(j));
    buffers.ao_two_electron_integral_indices.push_back(static_cast<int>(k));
    buffers.ao_two_electron_integral_indices.push_back(static_cast<int>(l));
  }

  return buffers;
}

}  // namespace xmvb::vb
