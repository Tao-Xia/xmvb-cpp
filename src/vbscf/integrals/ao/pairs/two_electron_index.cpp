#include "vbscf/integrals/ao/pairs/two_electron_index.hpp"

#include <algorithm>
#include <atomic>
#include <limits>
#include <stdexcept>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "core/openmp.hpp"

namespace xmvb::vb {

namespace {

std::size_t ao_pair_index(int first, int second) {
  if (first >= second) {
    const std::size_t first_index = first;
    return first_index * (first_index + 1) / 2 + second;
  }
  const std::size_t second_index = second;
  return second_index * (second_index + 1) / 2 + first;
}

std::pair<int, int> eri_pair_indices(
    const std::vector<int>& eri_indices,
    std::size_t eri) {
  const int* index = eri_indices.data() + 4 * eri;
  return {
      static_cast<int>(ao_pair_index(index[0], index[1])),
      static_cast<int>(ao_pair_index(index[2], index[3]))};
}

}  // namespace

std::vector<std::size_t> AoPairGraph::balanced_row_boundaries(
    int n_partitions) const {
  if (n_partitions <= 0 || row_offsets.empty() ||
      row_offsets.front() != 0 || row_offsets.back() < 0 ||
      static_cast<std::size_t>(row_offsets.back()) != columns.size() ||
      columns.size() != eri_indices.size()) {
    throw std::invalid_argument(
        "cannot partition an invalid AO-pair graph");
  }
  const std::size_t n_rows = row_offsets.size() - 1;
  n_partitions = std::min(
      n_partitions,
      static_cast<int>(n_rows));
  std::vector<std::size_t> boundaries(n_partitions + 1, 0);
  boundaries.back() = n_rows;
  for (int partition = 1;
       partition < n_partitions;
       ++partition) {
    const std::size_t edge_target =
        columns.size() * static_cast<std::size_t>(partition) /
        static_cast<std::size_t>(n_partitions);
    boundaries[partition] = static_cast<std::size_t>(
        std::lower_bound(
            row_offsets.begin(),
            row_offsets.end(),
            static_cast<int>(edge_target)) -
        row_offsets.begin());
  }
  return boundaries;
}

AoPairGraph build_ao_pair_graph(
    const std::vector<int>& eri_indices,
    int n_bf) {
  if (n_bf <= 0) {
    throw std::invalid_argument("n_bf must be positive");
  }
  if (eri_indices.size() % 4 != 0) {
    throw std::invalid_argument("each AO ERI must have four indices");
  }

  const std::size_t n_basis = n_bf;
  const std::size_t n_ao_pairs = n_basis * (n_basis + 1) / 2;
  if (n_ao_pairs > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("AO pair index exceeds 32-bit storage");
  }

  const std::size_t n_integrals = eri_indices.size() / 4;
  int n_threads = 1;
  n_threads = xmvb::effective_openmp_thread_count();
  if (n_integrals == 0) {
    n_threads = 1;
  } else if (n_integrals < static_cast<std::size_t>(n_threads)) {
    n_threads = static_cast<int>(n_integrals);
  }
  if (n_threads < 1) {
    n_threads = 1;
  }
  const std::size_t thread_count = n_threads;

  // The pair graph is reused across every exact SCF objective.  For large AO
  // ERI lists, building it serially dominates load time, so we count per-row
  // entries with thread-local histograms and later assign each thread a stable
  // write range inside every row.  This keeps the graph deterministic while
  // avoiding atomics in the hot integral loops.
  std::vector<std::vector<int>> thread_row_counts(
      thread_count,
      std::vector<int>(n_ao_pairs, 0));
  std::atomic<int> invalid_integral_index(-1);

  // Match the OpenMP team to the allocated thread-local histograms, including
  // when graph construction is invoked from an existing parallel region.
#pragma omp parallel num_threads(n_threads)
  {
    int thread_index = 0;
#ifdef _OPENMP
    thread_index = omp_get_thread_num();
#endif
    auto& local_row_counts = thread_row_counts[thread_index];

#pragma omp for schedule(static)
    for (std::ptrdiff_t integral_offset = 0;
         integral_offset < static_cast<std::ptrdiff_t>(n_integrals);
         ++integral_offset) {
      const std::size_t integral_index = integral_offset;
      const int* index = eri_indices.data() + 4 * integral_index;
      if (index[0] < 0 || index[0] >= n_bf ||
          index[1] < 0 || index[1] >= n_bf ||
          index[2] < 0 || index[2] >= n_bf ||
          index[3] < 0 || index[3] >= n_bf) {
        int expected = -1;
        invalid_integral_index.compare_exchange_strong(
            expected,
            static_cast<int>(integral_index));
        continue;
      }
      const auto [left_pair_index, right_pair_index] =
          eri_pair_indices(eri_indices, integral_index);
      ++local_row_counts[left_pair_index];
      if (right_pair_index != left_pair_index) {
        ++local_row_counts[right_pair_index];
      }
    }
  }

  if (invalid_integral_index.load() >= 0) {
    throw std::invalid_argument("AO ERI index out of range");
  }

  std::vector<int> row_counts(n_ao_pairs, 0);
  for (std::size_t row_index = 0; row_index < n_ao_pairs; ++row_index) {
    int total_count = 0;
    for (int thread_index = 0; thread_index < n_threads; ++thread_index) {
      total_count += thread_row_counts[thread_index][row_index];
    }
    row_counts[row_index] = total_count;
  }

  AoPairGraph graph;
  graph.row_offsets.resize(n_ao_pairs + 1, 0);
  for (std::size_t row_index = 0; row_index < n_ao_pairs; ++row_index) {
    graph.row_offsets[row_index + 1] = graph.row_offsets[row_index] + row_counts[row_index];
  }
  graph.columns.resize(graph.row_offsets.back());
  graph.eri_indices.resize(graph.row_offsets.back());

  std::vector<std::vector<int>> thread_next_offsets(
      thread_count,
      std::vector<int>(n_ao_pairs, 0));
  for (std::size_t row_index = 0; row_index < n_ao_pairs; ++row_index) {
    int next_offset = graph.row_offsets[row_index];
    for (int thread_index = 0; thread_index < n_threads; ++thread_index) {
      thread_next_offsets[thread_index][row_index] = next_offset;
      next_offset += thread_row_counts[thread_index][row_index];
    }
  }

#pragma omp parallel num_threads(n_threads)
  {
    int thread_index = 0;
#ifdef _OPENMP
    thread_index = omp_get_thread_num();
#endif
    auto& local_next_offsets = thread_next_offsets[thread_index];

#pragma omp for schedule(static)
    for (std::ptrdiff_t integral_offset = 0;
         integral_offset < static_cast<std::ptrdiff_t>(n_integrals);
         ++integral_offset) {
      const std::size_t integral_index = integral_offset;
      const auto [left_pair_index, right_pair_index] =
          eri_pair_indices(eri_indices, integral_index);

      const int left_offset = local_next_offsets[left_pair_index]++;
      graph.columns[left_offset] = right_pair_index;
      graph.eri_indices[left_offset] =
          static_cast<int>(integral_index);

      if (right_pair_index != left_pair_index) {
        const int right_offset = local_next_offsets[right_pair_index]++;
        graph.columns[right_offset] = left_pair_index;
        graph.eri_indices[right_offset] =
            static_cast<int>(integral_index);
      }
    }
  }

  return graph;
}

}  // namespace xmvb::vb
