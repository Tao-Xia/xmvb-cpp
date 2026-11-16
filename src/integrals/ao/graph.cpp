#include "integrals/ao/graph.hpp"
#include "core/index.hpp"

#include <algorithm>
#include <atomic>
#include <limits>
#include <stdexcept>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "parallel/openmp.hpp"

namespace xmvb::integrals {
std::vector<std::size_t> AoPairGraph::balanced_row_boundaries(int n_partitions) const {
    if (n_partitions <= 0 || row_offsets.empty() || row_offsets.front() != 0 ||
        row_offsets.back() < 0 || static_cast<std::size_t>(row_offsets.back()) != columns.size() ||
        columns.size() != values.size()) {
        throw std::invalid_argument("cannot partition an invalid AO-pair graph");
    }
    const std::size_t n_rows = row_offsets.size() - 1;
    n_partitions = static_cast<int>(std::min<std::size_t>(n_partitions, n_rows));
    std::vector<std::size_t> boundaries(n_partitions + 1, 0);
    boundaries.back() = n_rows;
    for (int partition = 1; partition < n_partitions; ++partition) {
        const std::size_t edge_target = columns.size() * static_cast<std::size_t>(partition) /
                                        static_cast<std::size_t>(n_partitions);
        boundaries[partition] = static_cast<std::size_t>(
            std::lower_bound(
                row_offsets.begin(), row_offsets.end(), static_cast<PairAddress>(edge_target)) -
            row_offsets.begin());
    }
    return boundaries;
}

AoPairGraph build_ao_pair_graph(std::vector<PairAddress> left_pairs,
                                std::vector<PairAddress> right_pairs,
                                std::vector<double> values,
                                int n_bf,
                                AoPairLayout layout) {
    if (n_bf <= 0) {
        throw std::invalid_argument("n_bf must be positive");
    }
    const std::size_t n_basis = n_bf;
    const std::size_t n_ao_pairs = core::triangular_count(n_basis);

    const std::size_t n_integrals = values.size();
    if (left_pairs.size() != n_integrals || right_pairs.size() != n_integrals) {
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
    AoPairGraph graph;
    graph.pair_first.reserve(n_ao_pairs);
    graph.pair_second.reserve(n_ao_pairs);
    for (int first = 0; first < n_bf; ++first)
        for (int second = 0; second <= first; ++second) {
            graph.pair_first.push_back(first);
            graph.pair_second.push_back(second);
        }
    if (layout == AoPairLayout::Unique || layout == AoPairLayout::Chunked) {
        int invalid = 0;
#pragma omp parallel for num_threads(n_threads) reduction(| : invalid) schedule(static)
        for (std::ptrdiff_t k = 0; k < static_cast<std::ptrdiff_t>(n_integrals); ++k)
            invalid |= left_pairs[k] < 0 || right_pairs[k] < 0 ||
                       static_cast<std::size_t>(left_pairs[k]) >= n_ao_pairs ||
                       static_cast<std::size_t>(right_pairs[k]) >= n_ao_pairs;
        if (invalid)
            throw std::invalid_argument("AO-pair index out of range");
        graph.integral_rows = std::move(left_pairs);
        graph.integral_columns = std::move(right_pairs);
        graph.integral_values = std::move(values);
        return graph;
    }
    const std::size_t thread_count = n_threads;

    // The pair graph is reused across every exact SCF objective.  For large AO
    // ERI lists, building it serially dominates load time, so we count per-row
    // entries with thread-local histograms and later assign each thread a stable
    // write range inside every row.  This keeps the graph deterministic while
    // avoiding atomics in the hot integral loops.
    std::vector<std::vector<PairAddress>> thread_row_offsets(
        thread_count, std::vector<PairAddress>(n_ao_pairs, 0));
    auto thread_primary_offsets = thread_row_offsets;
    std::atomic<std::ptrdiff_t> invalid_integral_index(-1);

    // Match the OpenMP team to the allocated thread-local histograms, including
    // when graph construction is invoked from an existing parallel region.
#pragma omp parallel num_threads(n_threads)
    {
        int thread_index = 0;
#ifdef _OPENMP
        thread_index = omp_get_thread_num();
#endif
        auto& local_row_offsets = thread_row_offsets[thread_index];
        auto& local_primary_offsets = thread_primary_offsets[thread_index];

#pragma omp for schedule(static)
        for (std::ptrdiff_t integral_offset = 0;
             integral_offset < static_cast<std::ptrdiff_t>(n_integrals);
             ++integral_offset) {
            const std::size_t integral_index = integral_offset;
            const auto left_pair_index = left_pairs[integral_index];
            const auto right_pair_index = right_pairs[integral_index];
            if (left_pair_index < 0 ||
                left_pair_index >= core::checked_index<PairAddress>(n_ao_pairs) ||
                right_pair_index < 0 ||
                right_pair_index >= core::checked_index<PairAddress>(n_ao_pairs)) {
                std::ptrdiff_t expected = -1;
                invalid_integral_index.compare_exchange_strong(
                    expected, static_cast<std::ptrdiff_t>(integral_index));
                continue;
            }
            ++local_row_offsets[left_pair_index];
            ++local_primary_offsets[left_pair_index];
            if (right_pair_index != left_pair_index) {
                ++local_row_offsets[right_pair_index];
            }
        }
    }

    if (invalid_integral_index.load() >= 0) {
        throw std::invalid_argument("AO-pair index out of range");
    }

    std::vector<PairAddress> row_counts(n_ao_pairs, 0);
    graph.integral_offsets.assign(n_ao_pairs + 1, 0);
    std::size_t n_graph_edges = 0;
    for (std::size_t row_index = 0; row_index < n_ao_pairs; ++row_index) {
        PairAddress total_count = 0;
        for (int thread_index = 0; thread_index < n_threads; ++thread_index) {
            total_count += thread_row_offsets[thread_index][row_index];
            graph.integral_offsets[row_index + 1] +=
                thread_primary_offsets[thread_index][row_index];
        }
        graph.integral_offsets[row_index + 1] += graph.integral_offsets[row_index];
        row_counts[row_index] = total_count;
        n_graph_edges =
            core::checked_add(n_graph_edges, core::checked_index<std::size_t>(total_count));
    }

    core::checked_index<PairAddress>(n_graph_edges);
    graph.row_offsets.resize(core::checked_add(n_ao_pairs, 1), 0);
    for (std::size_t row_index = 0; row_index < n_ao_pairs; ++row_index) {
        graph.row_offsets[row_index + 1] = graph.row_offsets[row_index] + row_counts[row_index];
    }
    // First group records into balanced contiguous row domains. This coarse
    // permutation touches only a worker-sized set of destination streams.
    // Each domain then sorts its rows independently, with no record copies.
    const auto parts = std::min<std::size_t>(n_threads, n_ao_pairs);
    std::vector<std::size_t> rows(parts + 1, 0);
    rows.back() = n_ao_pairs;
    for (std::size_t part = 1; part < parts; ++part) {
        const auto target = (n_integrals / parts) * part + std::min(n_integrals % parts, part);
        rows[part] = static_cast<std::size_t>(std::lower_bound(graph.integral_offsets.begin(),
                                                               graph.integral_offsets.end(),
                                                               static_cast<PairAddress>(target)) -
                                              graph.integral_offsets.begin());
    }
    const auto exchange = [&](PairAddress source, PairAddress target) {
        std::swap(left_pairs[source], left_pairs[target]);
        std::swap(right_pairs[source], right_pairs[target]);
        std::swap(values[source], values[target]);
    };
    if (parts > 1) {
        std::vector<std::size_t> owner(n_ao_pairs);
        std::vector<PairAddress> next(parts);
        for (std::size_t part = 0; part < parts; ++part) {
            std::fill(owner.begin() + rows[part], owner.begin() + rows[part + 1], part);
            next[part] = graph.integral_offsets[rows[part]];
        }
        for (std::size_t part = 0; part < parts; ++part)
            while (next[part] < graph.integral_offsets[rows[part + 1]]) {
                const auto source = next[part];
                const auto target_part = owner[left_pairs[source]];
                if (target_part == part)
                    ++next[part];
                else
                    exchange(source, next[target_part]++);
            }
    }
    // All global record addresses remain 64-bit. Parallel domains own disjoint
    // row cursors and records, including when OpenMP reduces its team size.
    std::vector<PairAddress> next(graph.integral_offsets.begin(), graph.integral_offsets.end() - 1);
#pragma omp parallel for num_threads(n_threads) schedule(dynamic, 1)
    for (std::ptrdiff_t part = 0; part < static_cast<std::ptrdiff_t>(parts); ++part) {
        for (auto row = rows[part]; row < rows[part + 1]; ++row) {
            while (next[row] < graph.integral_offsets[row + 1]) {
                const auto source = next[row];
                const auto target_row = left_pairs[source];
                if (target_row == static_cast<PairAddress>(row))
                    ++next[row];
                else
                    exchange(source, next[target_row]++);
            }
        }
    }
    std::vector<PairAddress>().swap(left_pairs);
    std::vector<std::vector<PairAddress>>().swap(thread_primary_offsets);
    // Directed row prefixes lie at or to the right of their unique intervals.
    // Move from the last row backward, so no unread source is overwritten.
    graph.columns = std::move(right_pairs);
    graph.columns.resize(n_graph_edges);
    values.resize(n_graph_edges);
    for (std::size_t row = n_ao_pairs; row-- > 0;) {
        const auto begin = graph.integral_offsets[row];
        const auto count = graph.integral_offsets[row + 1] - begin;
        const auto target = graph.row_offsets[row];
        if (begin == target)
            continue;
        std::move_backward(graph.columns.begin() + begin,
                           graph.columns.begin() + begin + count,
                           graph.columns.begin() + target + count);
        std::move_backward(values.begin() + begin,
                           values.begin() + begin + count,
                           values.begin() + target + count);
    }
    graph.values = std::move(values);
    for (auto& counts : thread_row_offsets)
        std::fill(counts.begin(), counts.end(), 0);
    // Recount and fill mirrors in one fixed team. This preserves deterministic
    // row ownership even if OpenMP dynamically reduces the requested team size.
#pragma omp parallel num_threads(n_threads)
    {
        int thread = 0, team = 1;
#ifdef _OPENMP
        thread = omp_get_thread_num();
        team = omp_get_num_threads();
#endif
        const auto boundary = [&](int t) {
            return (n_integrals / team) * t + std::min<std::size_t>(n_integrals % team, t);
        };
        auto cursor = AoPairGraph::Cursor(graph, boundary(thread));
        auto& offsets = thread_row_offsets[thread];
        for (auto k = boundary(thread); k < boundary(thread + 1); ++k, cursor.advance()) {
            const auto eri = cursor.value();
            if (eri.left_pair != eri.right_pair)
                ++offsets[eri.right_pair];
        }
#pragma omp barrier
#pragma omp for schedule(static)
        for (std::ptrdiff_t row = 0; row < static_cast<std::ptrdiff_t>(n_ao_pairs); ++row) {
            PairAddress next = graph.row_offsets[row] + graph.integral_offsets[row + 1] -
                               graph.integral_offsets[row];
            for (int t = 0; t < team; ++t) {
                const auto count = thread_row_offsets[t][row];
                thread_row_offsets[t][row] = next;
                next += count;
            }
            if (next != graph.row_offsets[row + 1])
                invalid_integral_index.store(row);
        }
        auto reader = AoPairGraph::Cursor(graph, boundary(thread));
        for (auto k = boundary(thread); k < boundary(thread + 1); ++k, reader.advance()) {
            const auto eri = reader.value();
            if (eri.left_pair != eri.right_pair) {
                const auto edge = offsets[eri.right_pair]++;
                graph.columns[edge] = eri.left_pair;
                graph.values[edge] = eri.value;
            }
        }
    }
    if (invalid_integral_index.load() >= 0)
        throw std::logic_error("AO-pair mirror counts do not match CSR rows");

    return graph;
}

AoPairGraph build_ao_pair_graph(std::vector<AoIntegralChunk> chunks, int n_bf) {
    auto graph = build_ao_pair_graph({}, {}, {}, n_bf, AoPairLayout::Unique);
    graph.chunk_offsets.push_back(0);
    for (const auto& chunk : chunks) {
        if (chunk.rows.size() != chunk.values.size() || chunk.columns.size() != chunk.values.size())
            throw std::invalid_argument("AO integral chunk dimensions");
        const auto previous = graph.chunk_offsets.back();
        if (chunk.values.size() > std::numeric_limits<std::size_t>::max() - previous)
            throw std::overflow_error("AO integral chunk count overflow");
        graph.chunk_offsets.push_back(previous + chunk.values.size());
    }
    int invalid = 0;
    const auto pairs = graph.pair_first.size();
#pragma omp parallel for reduction(| : invalid) schedule(dynamic, 1)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(chunks.size()); ++i)
        for (std::size_t k = 0; k < chunks[i].values.size(); ++k)
            invalid |= chunks[i].rows[k] < 0 || chunks[i].columns[k] < 0 ||
                       static_cast<std::size_t>(chunks[i].rows[k]) >= pairs ||
                       static_cast<std::size_t>(chunks[i].columns[k]) >= pairs;
    if (invalid)
        throw std::invalid_argument("AO integral chunk pair index out of range");
    graph.chunks = std::move(chunks);
    return graph;
}

} // namespace xmvb::integrals
