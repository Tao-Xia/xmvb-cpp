#include "vb/orbital/ao_two_electron_pair_index_utils.hpp"

#include <atomic>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace xmvb::vb {

namespace {

std::size_t ao_pair_index(int first, int second) {
  if (first >= second) {
    return xmvb::to_size(first) * (first + 1) / 2 + second;
  }
  return xmvb::to_size(second) * (second + 1) / 2 + first;
}

}  // namespace

std::vector<int> build_ao_two_electron_pair_indices(
    const std::vector<int>& ao_two_electron_integral_indices,
    int n_basis_functions) {
  if (n_basis_functions <= 0) {
    throw std::invalid_argument("n_basis_functions must be positive");
  }
  if (ao_two_electron_integral_indices.size() % 4 != 0) {
    throw std::invalid_argument("AO two-electron index/value sizes are inconsistent");
  }

  const std::size_t n_ao_pairs =
      xmvb::to_size(n_basis_functions) * (n_basis_functions + 1) / 2;
  if (n_ao_pairs > xmvb::to_size(std::numeric_limits<int>::max())) {
    throw std::overflow_error("AO pair index exceeds 32-bit storage");
  }

  const std::size_t n_integrals = ao_two_electron_integral_indices.size() / 4;
  std::vector<int> ao_two_electron_pair_indices(n_integrals * 2, 0);
  std::atomic<int> invalid_integral_index(-1);

#pragma omp parallel for schedule(static)
  for (std::ptrdiff_t integral_offset = 0;
       integral_offset < static_cast<std::ptrdiff_t>(n_integrals);
       ++integral_offset) {
    const std::size_t integral_index = xmvb::to_size(integral_offset);
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

    ao_two_electron_pair_indices[integral_index * 2] =
        static_cast<int>(ao_pair_index(i, j));
    ao_two_electron_pair_indices[integral_index * 2 + 1] =
        static_cast<int>(ao_pair_index(k, l));
  }

  if (invalid_integral_index.load() >= 0) {
    throw std::invalid_argument("AO two-electron index out of range");
  }

  return ao_two_electron_pair_indices;
}

AoTwoElectronPairGraph build_ao_two_electron_pair_graph(
    const std::vector<int>& ao_two_electron_pair_indices,
    int n_basis_functions) {
  if (n_basis_functions <= 0) {
    throw std::invalid_argument("n_basis_functions must be positive");
  }
  if (ao_two_electron_pair_indices.size() % 2 != 0) {
    throw std::invalid_argument("AO two-electron pair indices must contain 2 entries per integral");
  }

  const std::size_t n_ao_pairs =
      xmvb::to_size(n_basis_functions) * (n_basis_functions + 1) / 2;
  if (n_ao_pairs > xmvb::to_size(std::numeric_limits<int>::max())) {
    throw std::overflow_error("AO pair index exceeds 32-bit storage");
  }

  const std::size_t n_integrals = ao_two_electron_pair_indices.size() / 2;
  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif

  // The pair graph is reused across every exact SCF objective.  For large AO
  // ERI lists, building it serially dominates load time, so we count per-row
  // entries with thread-local histograms and later assign each thread a stable
  // write range inside every row.  This keeps the graph deterministic while
  // avoiding atomics in the hot integral loops.
  std::vector<std::vector<int>> thread_row_counts(
      xmvb::to_size(n_threads),
      std::vector<int>(n_ao_pairs, 0));
  std::atomic<int> invalid_integral_index(-1);

#pragma omp parallel
  {
    int thread_index = 0;
#ifdef _OPENMP
    thread_index = omp_get_thread_num();
#endif
    auto& local_row_counts = thread_row_counts[xmvb::to_size(thread_index)];

#pragma omp for schedule(static)
    for (std::ptrdiff_t integral_offset = 0;
         integral_offset < static_cast<std::ptrdiff_t>(n_integrals);
         ++integral_offset) {
      const std::size_t integral_index = xmvb::to_size(integral_offset);
      const int left_pair_index = ao_two_electron_pair_indices[integral_index * 2];
      const int right_pair_index = ao_two_electron_pair_indices[integral_index * 2 + 1];
      if (left_pair_index < 0 || left_pair_index >= static_cast<int>(n_ao_pairs) ||
          right_pair_index < 0 || right_pair_index >= static_cast<int>(n_ao_pairs)) {
        int expected = -1;
        invalid_integral_index.compare_exchange_strong(
            expected,
            static_cast<int>(integral_index));
        continue;
      }
      ++local_row_counts[xmvb::to_size(left_pair_index)];
      if (right_pair_index != left_pair_index) {
        ++local_row_counts[xmvb::to_size(right_pair_index)];
      }
    }
  }

  if (invalid_integral_index.load() >= 0) {
    throw std::invalid_argument("AO two-electron pair index out of range");
  }

  std::vector<int> row_counts(n_ao_pairs, 0);
  for (std::size_t row_index = 0; row_index < n_ao_pairs; ++row_index) {
    int total_count = 0;
    for (int thread_index = 0; thread_index < n_threads; ++thread_index) {
      total_count += thread_row_counts[xmvb::to_size(thread_index)][row_index];
    }
    row_counts[row_index] = total_count;
  }

  AoTwoElectronPairGraph graph;
  graph.row_offsets.resize(n_ao_pairs + 1, 0);
  for (std::size_t row_index = 0; row_index < n_ao_pairs; ++row_index) {
    graph.row_offsets[row_index + 1] = graph.row_offsets[row_index] + row_counts[row_index];
  }
  graph.column_pair_indices.resize(xmvb::to_size(graph.row_offsets.back()));
  graph.integral_indices.resize(xmvb::to_size(graph.row_offsets.back()));

  std::vector<std::vector<int>> thread_next_offsets(
      xmvb::to_size(n_threads),
      std::vector<int>(n_ao_pairs, 0));
  for (std::size_t row_index = 0; row_index < n_ao_pairs; ++row_index) {
    int next_offset = graph.row_offsets[row_index];
    for (int thread_index = 0; thread_index < n_threads; ++thread_index) {
      thread_next_offsets[xmvb::to_size(thread_index)][row_index] = next_offset;
      next_offset += thread_row_counts[xmvb::to_size(thread_index)][row_index];
    }
  }

#pragma omp parallel
  {
    int thread_index = 0;
#ifdef _OPENMP
    thread_index = omp_get_thread_num();
#endif
    auto& local_next_offsets =
        thread_next_offsets[xmvb::to_size(thread_index)];

#pragma omp for schedule(static)
    for (std::ptrdiff_t integral_offset = 0;
         integral_offset < static_cast<std::ptrdiff_t>(n_integrals);
         ++integral_offset) {
      const std::size_t integral_index = xmvb::to_size(integral_offset);
      const int left_pair_index = ao_two_electron_pair_indices[integral_index * 2];
      const int right_pair_index = ao_two_electron_pair_indices[integral_index * 2 + 1];

      const int left_offset =
          local_next_offsets[xmvb::to_size(left_pair_index)]++;
      graph.column_pair_indices[xmvb::to_size(left_offset)] = right_pair_index;
      graph.integral_indices[xmvb::to_size(left_offset)] =
          static_cast<int>(integral_index);

      if (right_pair_index != left_pair_index) {
        const int right_offset =
            local_next_offsets[xmvb::to_size(right_pair_index)]++;
        graph.column_pair_indices[xmvb::to_size(right_offset)] = left_pair_index;
        graph.integral_indices[xmvb::to_size(right_offset)] =
            static_cast<int>(integral_index);
      }
    }
  }

  return graph;
}

}  // namespace xmvb::vb
