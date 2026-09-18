#include "vbscf/integrals/active/two_electron/response/internal.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "core/openmp.hpp"
#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"
#include "vbscf/integrals/active/two_electron/transformation/packed_pair_map.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace xmvb::vb::detail {

void build_ao_pair_component_tables(
    int n_bf,
    std::vector<int>* first_indices,
    std::vector<int>* second_indices) {
  if (first_indices == nullptr || second_indices == nullptr) {
    throw std::invalid_argument("AO pair component tables must not be null");
  }
  const std::size_t basis_count = n_bf;
  const std::size_t n_bf_pairs = basis_count * (basis_count + 1) / 2;
  first_indices->assign(n_bf_pairs, 0);
  second_indices->assign(n_bf_pairs, 0);
  std::size_t pair_index = 0;
  for (int first_basis_function = 0;
       first_basis_function < n_bf;
       ++first_basis_function) {
    for (int second_basis_function = 0;
         second_basis_function <= first_basis_function;
         ++second_basis_function) {
      (*first_indices)[pair_index] = first_basis_function;
      (*second_indices)[pair_index] = second_basis_function;
      ++pair_index;
    }
  }
}

void resize_for_overwrite(
    std::vector<double>* values,
    std::size_t size) {
  if (values == nullptr) {
    throw std::invalid_argument("workspace buffer must not be null");
  }
  if (values->size() != size) {
    values->resize(size);
  }
}


std::vector<ActivePair> build_active_pair_list(int n_ao) {
  std::vector<ActivePair> active_pairs;
  active_pairs.reserve(
      n_ao * (n_ao + 1) / 2);
  for (int first = 0; first < n_ao; ++first) {
    for (int second = 0; second <= first; ++second) {
      active_pairs.push_back({first, second});
    }
  }
  return active_pairs;
}

ExactCtxPairMatrix build_active_pair_gradient_matrix(
    const std::vector<double>& packed_active_two_electron_gradient,
    const std::vector<ActivePair>& active_pairs) {
  const std::size_t n_active_pairs = active_pairs.size();
  ExactCtxPairMatrix active_pair_gradient_matrix =
      ExactCtxPairMatrix::Zero(
          static_cast<Eigen::Index>(n_active_pairs),
          static_cast<Eigen::Index>(n_active_pairs));

  for (std::size_t row_index = 0; row_index < n_active_pairs; ++row_index) {
    const auto& row_pair = active_pairs[row_index];
    for (std::size_t column_index = 0; column_index < n_active_pairs; ++column_index) {
      const auto& column_pair = active_pairs[column_index];
      const int packed_index =
          (row_index >= column_index)
              ? TwoElectronIndexer::two_electron_storage_index(
                    row_pair.first,
                    row_pair.second,
                    column_pair.first,
                    column_pair.second)
              : TwoElectronIndexer::two_electron_storage_index(
                    column_pair.first,
                    column_pair.second,
                    row_pair.first,
                    row_pair.second);
      double value =
          packed_active_two_electron_gradient[packed_index];
      if (row_index == column_index) {
        value *= 2.0;
      }
      active_pair_gradient_matrix(
          static_cast<Eigen::Index>(row_index),
          static_cast<Eigen::Index>(column_index)) = value;
    }
  }

  return active_pair_gradient_matrix;
}

void build_ao_pair_to_active_pair_coefficients(
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    int n_bf,
    int n_ao,
    const std::vector<ActivePair>& active_pairs,
    ExactCtxPairMatrix* ao_pair_to_active_pair_coefficients) {
  if (ao_pair_to_active_pair_coefficients == nullptr) {
    throw std::invalid_argument("AO-pair coefficient output must not be null");
  }
  if (dense_active_coefficients.rows() != n_bf ||
      dense_active_coefficients.cols() != n_ao) {
    throw std::invalid_argument("dense active coefficient matrix shape mismatch");
  }
  const std::size_t expected_active_pairs =
      static_cast<std::size_t>(n_ao) * (n_ao + 1) / 2;
  if (active_pairs.size() != expected_active_pairs) {
    throw std::invalid_argument("active-pair list size mismatch");
  }
  build_packed_orbital_pair_map(
      dense_active_coefficients,
      ao_pair_to_active_pair_coefficients);
}

namespace {

void validate_pair_row_range(
    const ExactPackedActiveTwoElectronAdjointCache& cache,
    Eigen::Index row_begin,
    Eigen::Index row_count) {
  const Eigen::Index n_bf_pairs =
      static_cast<Eigen::Index>(cache.n_basis_functions) *
      (cache.n_basis_functions + 1) / 2;
  if (cache.n_basis_functions <= 0 || cache.n_active_orbitals <= 0 ||
      row_begin < 0 || row_count < 0 ||
      row_begin + row_count > n_bf_pairs ||
      cache.ao_pair_first_indices.size() !=
          static_cast<std::size_t>(n_bf_pairs) ||
      cache.ao_pair_second_indices.size() !=
          static_cast<std::size_t>(n_bf_pairs) ||
      cache.active_pair_first_indices.size() !=
          cache.active_pair_second_indices.size()) {
    throw std::invalid_argument("invalid AO-pair row range");
  }
}

}  // namespace

void build_mixed_pair_rows(
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const ExactPackedActiveTwoElectronAdjointCache& cache,
    Eigen::Index row_begin,
    Eigen::Index row_count,
    ExactCtxPairMatrix* mixed_pair_coefficients) {
  validate_pair_row_range(cache, row_begin, row_count);
  if (mixed_pair_coefficients == nullptr ||
      dense_active_coefficients.rows() != cache.n_basis_functions ||
      dense_active_coefficients.cols() != cache.n_active_orbitals ||
      dense_active_direction.rows() != cache.n_basis_functions ||
      dense_active_direction.cols() != cache.n_active_orbitals) {
    throw std::invalid_argument("mixed pair-row dimensions are inconsistent");
  }
  build_packed_orbital_pair_map_directional_derivative_rows(
      dense_active_coefficients,
      dense_active_direction,
      row_begin,
      row_count,
      mixed_pair_coefficients);
}

void build_mixed_ao_pair_to_active_pair_coefficients_from_cache(
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const ExactPackedActiveTwoElectronAdjointCache& cache,
    ExactCtxPairMatrix* mixed_ao_pair_to_active_pair_coefficients) {
  const int n_bf = cache.n_basis_functions;
  const int n_ao = cache.n_active_orbitals;
  const std::size_t n_active_pairs = cache.active_pair_first_indices.size();
  if (cache.active_pair_second_indices.size() != n_active_pairs) {
    throw std::invalid_argument("exact 2e cache active-pair index size mismatch");
  }
  if (mixed_ao_pair_to_active_pair_coefficients == nullptr ||
      dense_active_coefficients.rows() != n_bf ||
      dense_active_coefficients.cols() != n_ao ||
      dense_active_direction.rows() != n_bf ||
      dense_active_direction.cols() != n_ao) {
    throw std::invalid_argument("dense active coefficient matrix shape mismatch");
  }

  build_packed_orbital_pair_map_directional_derivative(
      dense_active_coefficients,
      dense_active_direction,
      mixed_ao_pair_to_active_pair_coefficients);
}

ExactCtxPairMatrix build_mixed_ao_pair_to_active_pair_coefficients_from_cache(
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const ExactPackedActiveTwoElectronAdjointCache& cache) {
  ExactCtxPairMatrix mixed_ao_pair_to_active_pair_coefficients;
  build_mixed_ao_pair_to_active_pair_coefficients_from_cache(
      dense_active_coefficients,
      dense_active_direction,
      cache,
      &mixed_ao_pair_to_active_pair_coefficients);
  return mixed_ao_pair_to_active_pair_coefficients;
}
void multiply_pair_coefficients_by_gradient_matrix(
    const ExactCtxPairMatrix& ao_pair_to_active_pair_coefficients,
    const ExactCtxPairMatrix& active_pair_gradient_matrix,
    ExactCtxPairMatrix* transformed_pair_coefficients) {
  if (transformed_pair_coefficients == nullptr) {
    throw std::invalid_argument("transformed pair coefficient output must not be null");
  }
  if (ao_pair_to_active_pair_coefficients.cols() != active_pair_gradient_matrix.rows() ||
      active_pair_gradient_matrix.rows() != active_pair_gradient_matrix.cols()) {
    throw std::invalid_argument("active-pair gradient matrix shape mismatch");
  }
  transformed_pair_coefficients->resize(
      ao_pair_to_active_pair_coefficients.rows(),
      active_pair_gradient_matrix.cols());
  transformed_pair_coefficients->noalias() =
      ao_pair_to_active_pair_coefficients * active_pair_gradient_matrix;
}

ExactCtxPairMatrix multiply_pair_coefficients_by_gradient_matrix(
    const ExactCtxPairMatrix& ao_pair_to_active_pair_coefficients,
    const ExactCtxPairMatrix& active_pair_gradient_matrix) {
  ExactCtxPairMatrix transformed_pair_coefficients;
  multiply_pair_coefficients_by_gradient_matrix(
      ao_pair_to_active_pair_coefficients,
      active_pair_gradient_matrix,
      &transformed_pair_coefficients);
  return transformed_pair_coefficients;
}

void accumulate_backpropagated_pair_coefficients_to_dense_active_coefficients_from_cache(
    const ExactCtxPairMatrix& pair_gradients,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    const ExactPackedActiveTwoElectronAdjointCache& cache,
    Eigen::MatrixXd* dense_active_gradients) {
  if (dense_active_gradients == nullptr) {
    throw std::invalid_argument("dense active gradient output must not be null");
  }
  const int n_bf = cache.n_basis_functions;
  const int n_ao = cache.n_active_orbitals;
  const std::size_t n_active_pairs = cache.active_pair_first_indices.size();
  const std::size_t n_bf_pairs =
      static_cast<std::size_t>(n_bf) * (n_bf + 1) / 2;
  if (cache.active_pair_second_indices.size() != n_active_pairs ||
      pair_gradients.rows() != static_cast<Eigen::Index>(n_bf_pairs) ||
      pair_gradients.cols() != static_cast<Eigen::Index>(n_active_pairs) ||
      dense_active_coefficients.rows() != n_bf ||
      dense_active_coefficients.cols() != n_ao) {
    throw std::invalid_argument("pair-gradient / dense-active matrix shape mismatch");
  }

  accumulate_packed_orbital_pair_map_adjoint(
      pair_gradients,
      dense_active_coefficients,
      dense_active_gradients);
}

void accumulate_pair_gradient_rows(
    const ExactCtxPairMatrix& pair_gradients,
    Eigen::Index row_begin,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    const ExactPackedActiveTwoElectronAdjointCache& cache,
    Eigen::MatrixXd* dense_active_gradients) {
  const Eigen::Index row_count = pair_gradients.rows();
  validate_pair_row_range(cache, row_begin, row_count);
  const Eigen::Index n_active_pairs =
      static_cast<Eigen::Index>(cache.active_pair_first_indices.size());
  if (dense_active_gradients == nullptr ||
      pair_gradients.cols() != n_active_pairs ||
      dense_active_coefficients.rows() != cache.n_basis_functions ||
      dense_active_coefficients.cols() != cache.n_active_orbitals) {
    throw std::invalid_argument("pair-gradient row dimensions are inconsistent");
  }
  accumulate_packed_orbital_pair_map_adjoint_rows(
      pair_gradients,
      row_begin,
      dense_active_coefficients,
      dense_active_gradients);
}

void accumulate_pair_product_adjoint(
    const ExactCtxPairMatrix& pair_products,
    const ExactCtxPairMatrix& active_pair_gradient,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    const ExactPackedActiveTwoElectronAdjointCache& cache,
    Eigen::MatrixXd* dense_active_gradients) {
  if (dense_active_gradients == nullptr) {
    throw std::invalid_argument("dense active gradient output must not be null");
  }
  const int n_bf = cache.n_basis_functions;
  const int n_ao = cache.n_active_orbitals;
  const Eigen::Index n_active_pairs =
      static_cast<Eigen::Index>(cache.active_pair_first_indices.size());
  const Eigen::Index n_bf_pairs =
      static_cast<Eigen::Index>(n_bf) * (n_bf + 1) / 2;
  if (n_bf <= 0 || n_ao <= 0 ||
      cache.active_pair_second_indices.size() !=
          static_cast<std::size_t>(n_active_pairs) ||
      cache.ao_pair_first_indices.size() !=
          static_cast<std::size_t>(n_bf_pairs) ||
      cache.ao_pair_second_indices.size() !=
          static_cast<std::size_t>(n_bf_pairs) ||
      pair_products.rows() != n_bf_pairs ||
      pair_products.cols() != n_active_pairs ||
      active_pair_gradient.rows() != n_active_pairs ||
      active_pair_gradient.cols() != n_active_pairs ||
      dense_active_coefficients.rows() != n_bf ||
      dense_active_coefficients.cols() != n_ao) {
    throw std::invalid_argument("pair-product adjoint dimensions are inconsistent");
  }
  if (dense_active_gradients->rows() != n_bf ||
      dense_active_gradients->cols() != n_ao) {
    dense_active_gradients->setZero(n_bf, n_ao);
  }

  int n_threads = 1;
#ifdef _OPENMP
  n_threads = std::min(
      xmvb::effective_openmp_thread_count(),
      std::max(1, n_bf));
#endif
  std::vector<Eigen::MatrixXd> thread_gradients;
  thread_gradients.reserve(n_threads);
  for (int thread = 0; thread < n_threads; ++thread) {
    thread_gradients.emplace_back(Eigen::MatrixXd::Zero(n_bf, n_ao));
  }

  // One basis-size row tile keeps the temporary at O(n_bf * A_pair), while
  // still presenting a matrix multiplication large enough for Eigen kernels.
  const Eigen::Index tile_rows = n_bf;
  const Eigen::Index n_tiles =
      (n_bf_pairs + tile_rows - 1) / tile_rows;
#pragma omp parallel num_threads(n_threads)
  {
    int thread = 0;
#ifdef _OPENMP
    thread = omp_get_thread_num();
#endif
    Eigen::MatrixXd& local_gradient = thread_gradients[thread];
    ExactCtxPairMatrix pair_gradient_tile;
#pragma omp for schedule(static)
    for (Eigen::Index tile = 0; tile < n_tiles; ++tile) {
      const Eigen::Index row_begin = tile * tile_rows;
      const Eigen::Index row_count =
          std::min(tile_rows, n_bf_pairs - row_begin);
      pair_gradient_tile.resize(row_count, n_active_pairs);
      pair_gradient_tile.noalias() =
          pair_products.middleRows(row_begin, row_count) * active_pair_gradient;

      accumulate_packed_orbital_pair_map_adjoint_rows(
          pair_gradient_tile,
          row_begin,
          dense_active_coefficients,
          &local_gradient);
    }
  }

  for (const Eigen::MatrixXd& thread_gradient : thread_gradients) {
    *dense_active_gradients += thread_gradient;
  }
}

}  // namespace xmvb::vb::detail
