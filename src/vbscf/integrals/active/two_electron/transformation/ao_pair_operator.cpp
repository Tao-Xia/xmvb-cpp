#include "vbscf/integrals/active/two_electron/transformation/ao_pair_operator.hpp"

#include <cstddef>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "core/openmp.hpp"

namespace xmvb::vb::detail {

void apply_exact_ao_pair_kernel(
    const AoIntegralInput& ao,
    const Eigen::Ref<const ExactCtxPairMatrix>& coefficients,
    int n_bf,
    std::size_t n_active_pairs,
    ExactCtxPairMatrix* result) {
  if (result == nullptr) {
    throw std::invalid_argument("AO-pair result must not be null");
  }

  const std::size_t n_bf_pairs = n_bf * (n_bf + 1) / 2;
  if (coefficients.rows() != static_cast<Eigen::Index>(n_bf_pairs) ||
      coefficients.cols() != static_cast<Eigen::Index>(n_active_pairs)) {
    throw std::invalid_argument("AO-pair coefficient shape mismatch");
  }

  const AoPairGraph& graph = ao.pair_graph;
  if (graph.row_offsets.size() != n_bf_pairs + 1 ||
      graph.columns.size() != graph.eri_indices.size() ||
      graph.eri_indices.size() !=
          static_cast<std::size_t>(graph.row_offsets.back())) {
    throw std::invalid_argument("invalid AO-pair graph");
  }

  result->setZero(
      static_cast<Eigen::Index>(n_bf_pairs),
      static_cast<Eigen::Index>(n_active_pairs));

  int n_threads = 1;
#ifdef _OPENMP
  n_threads = effective_openmp_thread_count();
#endif
  const double* source = coefficients.data();
  double* target = result->data();

  n_threads = std::min(n_threads, static_cast<int>(n_bf_pairs));
  const auto row_boundaries =
      graph.balanced_row_boundaries(n_threads);

#pragma omp parallel num_threads(n_threads)
  {
    int thread = 0;
#ifdef _OPENMP
    thread = omp_get_thread_num();
#endif
    for (std::size_t row = row_boundaries[thread];
         row < row_boundaries[thread + 1];
         ++row) {
      double* target_row = target + row * n_active_pairs;
      int edge = graph.row_offsets[row];
      const int end = graph.row_offsets[row + 1];
      for (; edge + 3 < end; edge += 4) {
        const int column_0 = graph.columns[edge];
        const int column_1 = graph.columns[edge + 1];
        const int column_2 = graph.columns[edge + 2];
        const int column_3 = graph.columns[edge + 3];
        const double value_0 = ao.ao_two_electron_integral_values[
            graph.eri_indices[edge]];
        const double value_1 = ao.ao_two_electron_integral_values[
            graph.eri_indices[edge + 1]];
        const double value_2 = ao.ao_two_electron_integral_values[
            graph.eri_indices[edge + 2]];
        const double value_3 = ao.ao_two_electron_integral_values[
            graph.eri_indices[edge + 3]];
        const double* source_0 =
            source + static_cast<std::size_t>(column_0) * n_active_pairs;
        const double* source_1 =
            source + static_cast<std::size_t>(column_1) * n_active_pairs;
        const double* source_2 =
            source + static_cast<std::size_t>(column_2) * n_active_pairs;
        const double* source_3 =
            source + static_cast<std::size_t>(column_3) * n_active_pairs;
#pragma omp simd
        for (std::size_t pair = 0; pair < n_active_pairs; ++pair) {
          double accumulated = target_row[pair];
          accumulated += value_0 * source_0[pair];
          accumulated += value_1 * source_1[pair];
          accumulated += value_2 * source_2[pair];
          accumulated += value_3 * source_3[pair];
          target_row[pair] = accumulated;
        }
      }
      for (; edge < end; ++edge) {
        const int column = graph.columns[edge];
        const int eri = graph.eri_indices[edge];
        const double value = ao.ao_two_electron_integral_values[eri];
        const double* source_row =
            source + static_cast<std::size_t>(column) * n_active_pairs;
#pragma omp simd
        for (std::size_t pair = 0; pair < n_active_pairs; ++pair) {
          target_row[pair] += value * source_row[pair];
        }
      }
    }
  }
}

}  // namespace xmvb::vb::detail
