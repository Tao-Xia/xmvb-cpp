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
      graph.columns.size() != graph.values.size() ||
      graph.values.size() !=
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
        const double value_0 = graph.values[edge];
        const double value_1 = graph.values[edge + 1];
        const double value_2 = graph.values[edge + 2];
        const double value_3 = graph.values[edge + 3];
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
        const double value = graph.values[edge];
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

void apply_generated_pair_rows(
    const AoIntegralInput& ao,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    const Eigen::MatrixXd* dense_active_direction,
    const ExactPackedActiveTwoElectronAdjointCache& cache,
    Eigen::Index row_begin,
    Eigen::Index row_count,
    ExactCtxPairMatrix* pair_products,
    ExactCtxPairMatrix* directional_pair_products) {
  if (pair_products == nullptr) {
    throw std::invalid_argument("generated AO-pair result must not be null");
  }
  const int n_bf = cache.n_basis_functions;
  const Eigen::Index n_bf_pairs =
      static_cast<Eigen::Index>(n_bf) * (n_bf + 1) / 2;
  const Eigen::Index n_active_pairs =
      static_cast<Eigen::Index>(cache.active_pair_first_indices.size());
  const bool build_direction = dense_active_direction != nullptr;
  if (n_bf != ao.n_basis_functions || n_bf <= 0 ||
      cache.n_active_orbitals <= 0 || row_begin < 0 || row_count < 0 ||
      row_begin + row_count > n_bf_pairs ||
      dense_active_coefficients.rows() != n_bf ||
      dense_active_coefficients.cols() != cache.n_active_orbitals ||
      cache.ao_pair_first_indices.size() !=
          static_cast<std::size_t>(n_bf_pairs) ||
      cache.ao_pair_second_indices.size() !=
          static_cast<std::size_t>(n_bf_pairs) ||
      cache.active_pair_second_indices.size() !=
          static_cast<std::size_t>(n_active_pairs) ||
      (build_direction &&
       (dense_active_direction->rows() != n_bf ||
        dense_active_direction->cols() != cache.n_active_orbitals)) ||
      (build_direction != (directional_pair_products != nullptr))) {
    throw std::invalid_argument("generated AO-pair row dimensions are inconsistent");
  }
  const AoPairGraph& graph = ao.pair_graph;
  if (graph.row_offsets.size() != static_cast<std::size_t>(n_bf_pairs + 1) ||
      graph.columns.size() != graph.values.size() ||
      graph.values.size() !=
          static_cast<std::size_t>(graph.row_offsets.back())) {
    throw std::invalid_argument("invalid AO-pair graph");
  }

  pair_products->setZero(row_count, n_active_pairs);
  if (directional_pair_products != nullptr) {
    directional_pair_products->setZero(row_count, n_active_pairs);
  }
#pragma omp parallel for schedule(static)
  for (Eigen::Index local_row = 0; local_row < row_count; ++local_row) {
    const Eigen::Index target_row = row_begin + local_row;
    const int edge_begin = graph.row_offsets[target_row];
    const int edge_end = graph.row_offsets[target_row + 1];
    for (int edge = edge_begin; edge < edge_end; ++edge) {
      const Eigen::Index source_row = graph.columns[edge];
      const int first_bf = cache.ao_pair_first_indices[source_row];
      const int second_bf = cache.ao_pair_second_indices[source_row];
      const double integral = graph.values[edge];
      for (Eigen::Index active_pair = 0;
           active_pair < n_active_pairs;
           ++active_pair) {
        const int first_active = cache.active_pair_first_indices[active_pair];
        const int second_active = cache.active_pair_second_indices[active_pair];
        double coefficient =
            dense_active_coefficients(first_bf, first_active) *
            dense_active_coefficients(second_bf, second_active);
        if (first_bf != second_bf) {
          coefficient +=
              dense_active_coefficients(second_bf, first_active) *
              dense_active_coefficients(first_bf, second_active);
        }
        (*pair_products)(local_row, active_pair) += integral * coefficient;

        if (build_direction) {
          double mixed =
              (*dense_active_direction)(first_bf, first_active) *
                  dense_active_coefficients(second_bf, second_active) +
              dense_active_coefficients(first_bf, first_active) *
                  (*dense_active_direction)(second_bf, second_active);
          if (first_bf != second_bf) {
            mixed +=
                (*dense_active_direction)(second_bf, first_active) *
                    dense_active_coefficients(first_bf, second_active) +
                dense_active_coefficients(second_bf, first_active) *
                    (*dense_active_direction)(first_bf, second_active);
          }
          (*directional_pair_products)(local_row, active_pair) +=
              integral * mixed;
        }
      }
    }
  }
}

}  // namespace xmvb::vb::detail
