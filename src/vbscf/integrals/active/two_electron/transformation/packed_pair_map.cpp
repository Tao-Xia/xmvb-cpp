#include "vbscf/integrals/active/two_electron/transformation/packed_pair_map.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "core/openmp.hpp"

namespace xmvb::vb {
namespace {

Eigen::Index packed_pair_count(Eigen::Index dimension) {
  return dimension * (dimension + 1) / 2;
}

void validate_coefficients(
    const Eigen::Ref<const Eigen::MatrixXd>& coefficients) {
  if (coefficients.rows() <= 0 || coefficients.cols() <= 0) {
    throw std::invalid_argument(
        "packed orbital pair map requires positive coefficient dimensions");
  }
}

void unpack_pair_index(
    Eigen::Index pair_index,
    int* first,
    int* second) {
  Eigen::Index first_index = static_cast<Eigen::Index>(
      (std::sqrt(8.0 * static_cast<double>(pair_index) + 1.0) - 1.0) /
      2.0);
  while (packed_pair_count(first_index) > pair_index) {
    --first_index;
  }
  while (packed_pair_count(first_index + 1) <= pair_index) {
    ++first_index;
  }
  *first = static_cast<int>(first_index);
  *second = static_cast<int>(pair_index - packed_pair_count(first_index));
}

void validate_row_range(
    Eigen::Index n_basis_functions,
    Eigen::Index row_begin,
    Eigen::Index row_count) {
  const Eigen::Index n_ao_pairs = packed_pair_count(n_basis_functions);
  if (row_begin < 0 || row_count < 0 ||
      row_begin + row_count > n_ao_pairs) {
    throw std::invalid_argument("packed orbital pair-map row range is invalid");
  }
}

void ensure_gradient_shape(
    Eigen::Index n_basis_functions,
    Eigen::Index n_active_orbitals,
    Eigen::MatrixXd* gradient) {
  if (gradient == nullptr) {
    throw std::invalid_argument(
        "packed orbital pair-map gradient must not be null");
  }
  if (gradient->rows() != n_basis_functions ||
      gradient->cols() != n_active_orbitals) {
    gradient->setZero(n_basis_functions, n_active_orbitals);
  }
}

}  // namespace

void build_packed_orbital_pair_map(
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    PackedOrbitalPairMapMatrix* pair_map) {
  validate_coefficients(dense_active_coefficients);
  if (pair_map == nullptr) {
    throw std::invalid_argument("packed orbital pair-map output must not be null");
  }
  const Eigen::Index n_bf = dense_active_coefficients.rows();
  const Eigen::Index n_active = dense_active_coefficients.cols();
  pair_map->resize(
      packed_pair_count(n_bf),
      packed_pair_count(n_active));

#pragma omp parallel for schedule(static)
  for (Eigen::Index first_basis = 0;
       first_basis < n_bf;
       ++first_basis) {
    for (Eigen::Index second_basis = 0;
         second_basis <= first_basis;
         ++second_basis) {
      const Eigen::Index ao_pair =
          packed_pair_count(first_basis) + second_basis;
      Eigen::Index active_pair = 0;
      for (Eigen::Index first_active = 0;
           first_active < n_active;
           ++first_active) {
        for (Eigen::Index second_active = 0;
             second_active <= first_active;
             ++second_active, ++active_pair) {
          double value =
              dense_active_coefficients(first_basis, first_active) *
              dense_active_coefficients(second_basis, second_active);
          if (first_basis != second_basis) {
            value +=
                dense_active_coefficients(second_basis, first_active) *
                dense_active_coefficients(first_basis, second_active);
          }
          (*pair_map)(ao_pair, active_pair) = value;
        }
      }
    }
  }
}

void build_packed_orbital_pair_map_directional_derivative(
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    PackedOrbitalPairMapMatrix* directional_pair_map) {
  build_packed_orbital_pair_map_directional_derivative_rows(
      dense_active_coefficients,
      dense_active_direction,
      0,
      packed_pair_count(dense_active_coefficients.rows()),
      directional_pair_map);
}

void build_packed_orbital_pair_map_directional_derivative_rows(
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    Eigen::Index row_begin,
    Eigen::Index row_count,
    PackedOrbitalPairMapMatrix* directional_pair_map) {
  validate_coefficients(dense_active_coefficients);
  const Eigen::Index n_bf = dense_active_coefficients.rows();
  const Eigen::Index n_active = dense_active_coefficients.cols();
  validate_row_range(n_bf, row_begin, row_count);
  if (directional_pair_map == nullptr ||
      dense_active_direction.rows() != n_bf ||
      dense_active_direction.cols() != n_active) {
    throw std::invalid_argument(
        "packed orbital pair-map direction dimensions are inconsistent");
  }
  directional_pair_map->resize(row_count, packed_pair_count(n_active));

#pragma omp parallel for schedule(static)
  for (Eigen::Index local_row = 0; local_row < row_count; ++local_row) {
    int first_basis = 0;
    int second_basis = 0;
    unpack_pair_index(
        row_begin + local_row,
        &first_basis,
        &second_basis);
    Eigen::Index active_pair = 0;
    for (Eigen::Index first_active = 0;
         first_active < n_active;
         ++first_active) {
      for (Eigen::Index second_active = 0;
           second_active <= first_active;
           ++second_active, ++active_pair) {
        double value =
            dense_active_direction(first_basis, first_active) *
                dense_active_coefficients(second_basis, second_active) +
            dense_active_coefficients(first_basis, first_active) *
                dense_active_direction(second_basis, second_active);
        if (first_basis != second_basis) {
          value +=
              dense_active_direction(second_basis, first_active) *
                  dense_active_coefficients(first_basis, second_active) +
              dense_active_coefficients(second_basis, first_active) *
                  dense_active_direction(first_basis, second_active);
        }
        (*directional_pair_map)(local_row, active_pair) = value;
      }
    }
  }
}

void accumulate_packed_orbital_pair_map_adjoint(
    const Eigen::Ref<const PackedOrbitalPairMapMatrix>& pair_adjoint,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    Eigen::MatrixXd* dense_active_gradient) {
  validate_coefficients(dense_active_coefficients);
  const Eigen::Index n_bf = dense_active_coefficients.rows();
  const Eigen::Index n_active = dense_active_coefficients.cols();
  if (pair_adjoint.rows() != packed_pair_count(n_bf) ||
      pair_adjoint.cols() != packed_pair_count(n_active)) {
    throw std::invalid_argument(
        "packed orbital pair-map adjoint dimensions are inconsistent");
  }
  ensure_gradient_shape(n_bf, n_active, dense_active_gradient);

  int n_threads = 1;
#ifdef _OPENMP
  n_threads = std::min(
      xmvb::effective_openmp_thread_count(),
      std::max(1, static_cast<int>(n_bf)));
#endif
  if (n_threads <= 1) {
    accumulate_packed_orbital_pair_map_adjoint_rows(
        pair_adjoint,
        0,
        dense_active_coefficients,
        dense_active_gradient);
    return;
  }
  std::vector<Eigen::MatrixXd> thread_gradients;
  thread_gradients.reserve(n_threads);
  for (int thread = 0; thread < n_threads; ++thread) {
    thread_gradients.emplace_back(Eigen::MatrixXd::Zero(n_bf, n_active));
  }

  const Eigen::Index tile_rows = n_bf;
  const Eigen::Index n_tiles =
      (pair_adjoint.rows() + tile_rows - 1) / tile_rows;
#pragma omp parallel num_threads(n_threads)
  {
    int thread = 0;
#ifdef _OPENMP
    thread = omp_get_thread_num();
#endif
#pragma omp for schedule(static)
    for (Eigen::Index tile = 0; tile < n_tiles; ++tile) {
      const Eigen::Index row_begin = tile * tile_rows;
      const Eigen::Index row_count =
          std::min(tile_rows, pair_adjoint.rows() - row_begin);
      accumulate_packed_orbital_pair_map_adjoint_rows(
          pair_adjoint.middleRows(row_begin, row_count),
          row_begin,
          dense_active_coefficients,
          &thread_gradients[thread]);
    }
  }
  for (const Eigen::MatrixXd& thread_gradient : thread_gradients) {
    *dense_active_gradient += thread_gradient;
  }
}

void accumulate_packed_orbital_pair_map_adjoint_rows(
    const Eigen::Ref<const PackedOrbitalPairMapMatrix>& pair_adjoint,
    Eigen::Index row_begin,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    Eigen::MatrixXd* dense_active_gradient) {
  validate_coefficients(dense_active_coefficients);
  const Eigen::Index n_bf = dense_active_coefficients.rows();
  const Eigen::Index n_active = dense_active_coefficients.cols();
  validate_row_range(n_bf, row_begin, pair_adjoint.rows());
  if (pair_adjoint.cols() != packed_pair_count(n_active)) {
    throw std::invalid_argument(
        "packed orbital pair-map row adjoint dimensions are inconsistent");
  }
  ensure_gradient_shape(n_bf, n_active, dense_active_gradient);

  for (Eigen::Index local_row = 0;
       local_row < pair_adjoint.rows();
       ++local_row) {
    int first_basis = 0;
    int second_basis = 0;
    unpack_pair_index(
        row_begin + local_row,
        &first_basis,
        &second_basis);
    Eigen::Index active_pair = 0;
    for (Eigen::Index first_active = 0;
         first_active < n_active;
         ++first_active) {
      for (Eigen::Index second_active = 0;
           second_active <= first_active;
           ++second_active, ++active_pair) {
        const double pair_gradient = pair_adjoint(local_row, active_pair);
        if (pair_gradient == 0.0) {
          continue;
        }
        (*dense_active_gradient)(first_basis, first_active) +=
            pair_gradient *
            dense_active_coefficients(second_basis, second_active);
        (*dense_active_gradient)(first_basis, second_active) +=
            pair_gradient *
            dense_active_coefficients(second_basis, first_active);
        if (first_basis != second_basis) {
          (*dense_active_gradient)(second_basis, first_active) +=
              pair_gradient *
              dense_active_coefficients(first_basis, second_active);
          (*dense_active_gradient)(second_basis, second_active) +=
              pair_gradient *
              dense_active_coefficients(first_basis, first_active);
        }
      }
    }
  }
}

}  // namespace xmvb::vb
