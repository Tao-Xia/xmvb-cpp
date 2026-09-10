#include "vbscf/integrals/active/two_electron/response/internal.hpp"

#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "core/parallel/openmp.hpp"
#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"

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
  const std::size_t n_bf_pairs =
      n_bf * (n_bf + 1) / 2;
  const std::size_t n_active_pairs = active_pairs.size();
  ao_pair_to_active_pair_coefficients->resize(
      static_cast<Eigen::Index>(n_bf_pairs),
      static_cast<Eigen::Index>(n_active_pairs));

#pragma omp parallel for schedule(static)
  for (int first_basis_function = 0;
       first_basis_function < n_bf;
       ++first_basis_function) {
    for (int second_basis_function = 0;
         second_basis_function <= first_basis_function;
         ++second_basis_function) {
      const Eigen::Index ao_pair_offset =
          static_cast<Eigen::Index>(
              ao_pair_index(first_basis_function, second_basis_function));
      for (std::size_t active_pair_index_offset = 0;
           active_pair_index_offset < n_active_pairs;
           ++active_pair_index_offset) {
        const auto& active_pair = active_pairs[active_pair_index_offset];
        double coefficient =
            dense_active_coefficients(first_basis_function, active_pair.first) *
            dense_active_coefficients(second_basis_function, active_pair.second);
        if (first_basis_function != second_basis_function) {
          coefficient +=
              dense_active_coefficients(second_basis_function, active_pair.first) *
              dense_active_coefficients(first_basis_function, active_pair.second);
        }
        (*ao_pair_to_active_pair_coefficients)(
            ao_pair_offset,
            static_cast<Eigen::Index>(active_pair_index_offset)) = coefficient;
      }
    }
  }
}

ExactCtxPairMatrix build_ao_pair_to_active_pair_coefficients(
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    int n_bf,
    int n_ao,
    const std::vector<ActivePair>& active_pairs) {
  ExactCtxPairMatrix ao_pair_to_active_pair_coefficients;
  build_ao_pair_to_active_pair_coefficients(
      dense_active_coefficients,
      n_bf,
      n_ao,
      active_pairs,
      &ao_pair_to_active_pair_coefficients);
  return ao_pair_to_active_pair_coefficients;
}

void build_mixed_ao_pair_to_active_pair_coefficients(
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    int n_bf,
    int n_ao,
    const std::vector<ActivePair>& active_pairs,
    ExactCtxPairMatrix* mixed_ao_pair_to_active_pair_coefficients) {
  if (mixed_ao_pair_to_active_pair_coefficients == nullptr) {
    throw std::invalid_argument("mixed AO-pair coefficient output must not be null");
  }
  if (dense_active_coefficients.rows() != n_bf ||
      dense_active_coefficients.cols() != n_ao ||
      dense_active_direction.rows() != n_bf ||
      dense_active_direction.cols() != n_ao) {
    throw std::invalid_argument("mixed AO-pair coefficient matrix shape mismatch");
  }
  const std::size_t n_bf_pairs =
      n_bf * (n_bf + 1) / 2;
  const std::size_t n_active_pairs = active_pairs.size();
  mixed_ao_pair_to_active_pair_coefficients->resize(
      static_cast<Eigen::Index>(n_bf_pairs),
      static_cast<Eigen::Index>(n_active_pairs));

#pragma omp parallel for schedule(static)
  for (int first_basis_function = 0;
       first_basis_function < n_bf;
       ++first_basis_function) {
    for (int second_basis_function = 0;
         second_basis_function <= first_basis_function;
         ++second_basis_function) {
      const Eigen::Index ao_pair_offset =
          static_cast<Eigen::Index>(
              ao_pair_index(first_basis_function, second_basis_function));
      for (std::size_t active_pair_index_offset = 0;
           active_pair_index_offset < n_active_pairs;
           ++active_pair_index_offset) {
        const auto& active_pair = active_pairs[active_pair_index_offset];
        double coefficient =
            dense_active_direction(first_basis_function, active_pair.first) *
                dense_active_coefficients(second_basis_function, active_pair.second) +
            dense_active_coefficients(first_basis_function, active_pair.first) *
                dense_active_direction(second_basis_function, active_pair.second);
        if (first_basis_function != second_basis_function) {
          coefficient +=
              dense_active_direction(second_basis_function, active_pair.first) *
                  dense_active_coefficients(first_basis_function, active_pair.second) +
              dense_active_coefficients(second_basis_function, active_pair.first) *
                  dense_active_direction(first_basis_function, active_pair.second);
        }
        (*mixed_ao_pair_to_active_pair_coefficients)(
            ao_pair_offset,
            static_cast<Eigen::Index>(active_pair_index_offset)) = coefficient;
      }
    }
  }
}

ExactCtxPairMatrix build_mixed_ao_pair_to_active_pair_coefficients(
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    int n_bf,
    int n_ao,
    const std::vector<ActivePair>& active_pairs) {
  ExactCtxPairMatrix mixed_ao_pair_to_active_pair_coefficients;
  build_mixed_ao_pair_to_active_pair_coefficients(
      dense_active_coefficients,
      dense_active_direction,
      n_bf,
      n_ao,
      active_pairs,
      &mixed_ao_pair_to_active_pair_coefficients);
  return mixed_ao_pair_to_active_pair_coefficients;
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

  const std::size_t n_bf_pairs =
      n_bf * (n_bf + 1) / 2;
  mixed_ao_pair_to_active_pair_coefficients->resize(
      static_cast<Eigen::Index>(n_bf_pairs),
      static_cast<Eigen::Index>(n_active_pairs));

#pragma omp parallel for schedule(static)
  for (int first_basis_function = 0;
       first_basis_function < n_bf;
       ++first_basis_function) {
    for (int second_basis_function = 0;
         second_basis_function <= first_basis_function;
         ++second_basis_function) {
      const Eigen::Index ao_pair_offset =
          static_cast<Eigen::Index>(
              ao_pair_index(first_basis_function, second_basis_function));
      for (std::size_t active_pair_index = 0;
           active_pair_index < n_active_pairs;
           ++active_pair_index) {
        const int first_active =
            cache.active_pair_first_indices[active_pair_index];
        const int second_active =
            cache.active_pair_second_indices[active_pair_index];
        double coefficient =
            dense_active_direction(first_basis_function, first_active) *
                dense_active_coefficients(second_basis_function, second_active) +
            dense_active_coefficients(first_basis_function, first_active) *
                dense_active_direction(second_basis_function, second_active);
        if (first_basis_function != second_basis_function) {
          coefficient +=
              dense_active_direction(second_basis_function, first_active) *
                  dense_active_coefficients(first_basis_function, second_active) +
              dense_active_coefficients(second_basis_function, first_active) *
                  dense_active_direction(first_basis_function, second_active);
        }
        (*mixed_ao_pair_to_active_pair_coefficients)(
            ao_pair_offset,
            static_cast<Eigen::Index>(active_pair_index)) = coefficient;
      }
    }
  }
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

  // This Eigen path is the accepted-point HVP accumulator. Preserve any fixed
  // term already stored in the output and zero only when we need a fresh shape.
  if (dense_active_gradients->rows() != n_bf ||
      dense_active_gradients->cols() != n_ao) {
    dense_active_gradients->resize(n_bf, n_ao);
    dense_active_gradients->setZero();
  }
  int n_threads = 1;
#ifdef _OPENMP
  n_threads = xmvb::effective_openmp_thread_count();
#endif
  if (n_threads <= 1) {
    for (int first_basis_function = 0;
         first_basis_function < n_bf;
         ++first_basis_function) {
      for (int second_basis_function = 0;
           second_basis_function <= first_basis_function;
           ++second_basis_function) {
        const Eigen::Index pair_row =
            static_cast<Eigen::Index>(
                ao_pair_index(first_basis_function, second_basis_function));
        for (std::size_t active_pair_index = 0;
             active_pair_index < n_active_pairs;
             ++active_pair_index) {
          const double pair_gradient =
              pair_gradients(pair_row, static_cast<Eigen::Index>(active_pair_index));
          if (pair_gradient == 0.0) {
            continue;
          }
          const int first_active =
              cache.active_pair_first_indices[active_pair_index];
          const int second_active =
              cache.active_pair_second_indices[active_pair_index];
          (*dense_active_gradients)(first_basis_function, first_active) +=
              pair_gradient *
              dense_active_coefficients(second_basis_function, second_active);
          (*dense_active_gradients)(first_basis_function, second_active) +=
              pair_gradient *
              dense_active_coefficients(second_basis_function, first_active);
          if (second_basis_function != first_basis_function) {
            (*dense_active_gradients)(second_basis_function, first_active) +=
                pair_gradient *
                dense_active_coefficients(first_basis_function, second_active);
            (*dense_active_gradients)(second_basis_function, second_active) +=
                pair_gradient *
                dense_active_coefficients(first_basis_function, first_active);
          }
        }
      }
    }
    return;
  }

#pragma omp parallel for schedule(static)
  for (int basis_function_index = 0;
       basis_function_index < n_bf;
       ++basis_function_index) {
    for (int other_basis_function = 0;
         other_basis_function < n_bf;
         ++other_basis_function) {
      const Eigen::Index pair_row =
          static_cast<Eigen::Index>(
              ao_pair_index(basis_function_index, other_basis_function));
      for (std::size_t active_pair_index = 0;
           active_pair_index < n_active_pairs;
           ++active_pair_index) {
        const double pair_gradient =
            pair_gradients(pair_row, static_cast<Eigen::Index>(active_pair_index));
        if (pair_gradient == 0.0) {
          continue;
        }
        const int first_active =
            cache.active_pair_first_indices[active_pair_index];
        const int second_active =
            cache.active_pair_second_indices[active_pair_index];
        (*dense_active_gradients)(basis_function_index, first_active) +=
            pair_gradient *
            dense_active_coefficients(other_basis_function, second_active);
        (*dense_active_gradients)(basis_function_index, second_active) +=
            pair_gradient *
            dense_active_coefficients(other_basis_function, first_active);
      }
    }
  }
}

}  // namespace xmvb::vb::detail
