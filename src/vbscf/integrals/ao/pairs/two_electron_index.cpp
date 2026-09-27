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

std::vector<std::size_t> AoPairGraph::balanced_row_boundaries(
    int n_partitions) const {
  if (n_partitions <= 0 || row_offsets.empty() ||
      row_offsets.front() != 0 || row_offsets.back() < 0 ||
      static_cast<std::size_t>(row_offsets.back()) != columns.size() ||
      columns.size() != values.size()) {
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
    std::vector<int> left_pairs,
    std::vector<int> right_pairs,
    std::vector<double> values,
    int n_bf) {
  if (n_bf <= 0) {
    throw std::invalid_argument("n_bf must be positive");
  }
  const std::size_t n_basis = n_bf;
  const std::size_t n_ao_pairs = n_basis * (n_basis + 1) / 2;
  if (n_ao_pairs > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("AO pair index exceeds 32-bit storage");
  }

  const std::size_t n_integrals = values.size();
  if (left_pairs.size() != n_integrals ||
      right_pairs.size() != n_integrals) {
    throw std::invalid_argument("AO-pair index/value counts do not match");
  }
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
  std::vector<std::vector<int>> thread_row_offsets(
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
    auto& local_row_offsets = thread_row_offsets[thread_index];

#pragma omp for schedule(static)
    for (std::ptrdiff_t integral_offset = 0;
         integral_offset < static_cast<std::ptrdiff_t>(n_integrals);
         ++integral_offset) {
      const std::size_t integral_index = integral_offset;
      const int left_pair_index = left_pairs[integral_index];
      const int right_pair_index = right_pairs[integral_index];
      if (left_pair_index < 0 ||
          left_pair_index >= static_cast<int>(n_ao_pairs) ||
          right_pair_index < 0 ||
          right_pair_index >= static_cast<int>(n_ao_pairs)) {
        int expected = -1;
        invalid_integral_index.compare_exchange_strong(
            expected,
            static_cast<int>(integral_index));
        continue;
      }
      ++local_row_offsets[left_pair_index];
      if (right_pair_index != left_pair_index) {
        ++local_row_offsets[right_pair_index];
      }
    }
  }

  if (invalid_integral_index.load() >= 0) {
    throw std::invalid_argument("AO-pair index out of range");
  }

  std::vector<int> row_counts(n_ao_pairs, 0);
  std::size_t n_graph_edges = 0;
  for (std::size_t row_index = 0; row_index < n_ao_pairs; ++row_index) {
    int total_count = 0;
    for (int thread_index = 0; thread_index < n_threads; ++thread_index) {
      total_count += thread_row_offsets[thread_index][row_index];
    }
    row_counts[row_index] = total_count;
    n_graph_edges += static_cast<std::size_t>(total_count);
  }

  AoPairGraph graph;
  graph.pair_first.reserve(n_ao_pairs);
  graph.pair_second.reserve(n_ao_pairs);
  for (int first = 0; first < n_bf; ++first) {
    for (int second = 0; second <= first; ++second) {
      graph.pair_first.push_back(first);
      graph.pair_second.push_back(second);
    }
  }
  // The symmetric CSR representation duplicates every off-diagonal unique
  // integral.  Preserve the once-symmetry-reduced stream when those directed
  // edges no longer fit the 32-bit hot-path offsets.
  if (n_graph_edges >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    graph.integral_rows = std::move(left_pairs);
    graph.integral_columns = std::move(right_pairs);
    graph.integral_values = std::move(values);
    return graph;
  }

  graph.row_offsets.resize(n_ao_pairs + 1, 0);
  for (std::size_t row_index = 0; row_index < n_ao_pairs; ++row_index) {
    graph.row_offsets[row_index + 1] = graph.row_offsets[row_index] + row_counts[row_index];
  }
  // Convert the thread-local counts in place into disjoint write cursors.
  // Keeping a second thread-by-row table doubles the graph-construction
  // workspace and can dominate the raw integral payload for large AO bases.
  for (std::size_t row_index = 0; row_index < n_ao_pairs; ++row_index) {
    int next_offset = graph.row_offsets[row_index];
    for (int thread_index = 0; thread_index < n_threads; ++thread_index) {
      const int row_count = thread_row_offsets[thread_index][row_index];
      thread_row_offsets[thread_index][row_index] = next_offset;
      next_offset += row_count;
    }
  }

  graph.columns.resize(graph.row_offsets.back());
#pragma omp parallel num_threads(n_threads)
  {
    int thread_index = 0;
#ifdef _OPENMP
    thread_index = omp_get_thread_num();
#endif
    auto& local_next_offsets = thread_row_offsets[thread_index];

#pragma omp for schedule(static)
    for (std::ptrdiff_t integral_offset = 0;
         integral_offset < static_cast<std::ptrdiff_t>(n_integrals);
         ++integral_offset) {
      const std::size_t integral_index = integral_offset;
      const int left_pair_index = left_pairs[integral_index];
      const int right_pair_index = right_pairs[integral_index];
      const int left_offset = local_next_offsets[left_pair_index]++;
      graph.columns[left_offset] = right_pair_index;
      // Reuse the right-pair input buffer as the unique-integral-to-CSR-edge
      // map.  It becomes the graph's persistent integral-edge table below.
      right_pairs[integral_index] = left_offset;

      if (right_pair_index != left_pair_index) {
        const int right_offset = local_next_offsets[right_pair_index]++;
        graph.columns[right_offset] = left_pair_index;
      }
    }
  }

  // Expand the raw unique-integral value vector into the directed CSR value
  // buffer in place.  The primary-edge map is injective, so each relocation is
  // a chain or cycle and needs only one carried scalar.  Libcint reserves the
  // final directed capacity before handing this buffer to the graph builder;
  // no second O(N_ERI) value allocation is then required at peak memory.
  values.resize(graph.columns.size());
  std::vector<unsigned char> relocated(n_integrals, 0u);
  for (std::size_t start = 0; start < n_integrals; ++start) {
    if (relocated[start] != 0u) {
      continue;
    }
    std::size_t source = start;
    double carried = values[source];
    while (true) {
      const std::size_t target =
          static_cast<std::size_t>(right_pairs[source]);
      std::swap(carried, values[target]);
      relocated[source] = 1u;
      if (target >= n_integrals || relocated[target] != 0u) {
        break;
      }
      source = target;
    }
  }

  // Recover the same disjoint row cursors and replay the input ordering to
  // duplicate off-diagonal values into their transposed CSR edges.  The first
  // edge is already the primary location populated by the in-place move.
  for (std::size_t row_index = 0; row_index < n_ao_pairs; ++row_index) {
    int next_offset = graph.row_offsets[row_index];
    for (int thread_index = 0; thread_index < n_threads; ++thread_index) {
      const int segment_end = thread_row_offsets[thread_index][row_index];
      thread_row_offsets[thread_index][row_index] = next_offset;
      next_offset = segment_end;
    }
  }
#pragma omp parallel num_threads(n_threads)
  {
    int thread_index = 0;
#ifdef _OPENMP
    thread_index = omp_get_thread_num();
#endif
    auto& local_next_offsets = thread_row_offsets[thread_index];
#pragma omp for schedule(static)
    for (std::ptrdiff_t integral_offset = 0;
         integral_offset < static_cast<std::ptrdiff_t>(n_integrals);
         ++integral_offset) {
      const std::size_t integral_index = integral_offset;
      const int left_pair_index = left_pairs[integral_index];
      const int primary_offset = right_pairs[integral_index];
      const int right_pair_index = graph.columns[primary_offset];
      const int replayed_primary = local_next_offsets[left_pair_index]++;
      if (replayed_primary != primary_offset) {
        invalid_integral_index.store(static_cast<int>(integral_index));
        continue;
      }
      if (right_pair_index != left_pair_index) {
        const int transposed_offset = local_next_offsets[right_pair_index]++;
        values[transposed_offset] = values[primary_offset];
      }
    }
  }
  if (invalid_integral_index.load() >= 0) {
    throw std::logic_error("AO-pair graph replay order is inconsistent");
  }

  graph.values = std::move(values);
  graph.integral_rows = std::move(left_pairs);
  graph.integral_edges = std::move(right_pairs);

  return graph;
}

}  // namespace xmvb::vb
