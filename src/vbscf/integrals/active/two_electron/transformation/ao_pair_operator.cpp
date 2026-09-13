#include "vbscf/integrals/active/two_electron/response/internal.hpp"

#include <cstddef>
#include <stdexcept>

#include "core/openmp.hpp"

namespace xmvb::vb::detail {

void apply_exact_ao_pair_kernel(
    const AoIntegralInput& ao,
    const ExactCtxPairMatrix& coefficients,
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

#pragma omp parallel for schedule(guided, 64) num_threads(n_threads)
  for (std::ptrdiff_t row = 0;
       row < static_cast<std::ptrdiff_t>(n_bf_pairs);
       ++row) {
    double* target_row = target + row * n_active_pairs;
    for (int edge = graph.row_offsets[row];
         edge < graph.row_offsets[row + 1];
         ++edge) {
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

}  // namespace xmvb::vb::detail
