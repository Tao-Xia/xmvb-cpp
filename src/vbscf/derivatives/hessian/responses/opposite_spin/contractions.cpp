#include "vbscf/derivatives/hessian/responses/opposite_spin/backward.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "vbscf/determinants/pairs/contractions.hpp"
#include "vbscf/determinants/algebra/cofactor_differential.hpp"
#include "vbscf/structures/assembly/local_contractions.hpp"
#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"
#include "vbscf/integrals/active/two_electron/construction/kernel.hpp"
#include "vbscf/derivatives/hessian/responses/opposite_spin/pair_response_internal.hpp"
#include "vbscf/derivatives/hessian/responses/opposite_spin/overlap_contractions_internal.hpp"
#include "vbscf/derivatives/hessian/responses/opposite_spin/packed_contractions_internal.hpp"
#include "vbscf/derivatives/hessian/responses/opposite_spin/tile_kernels_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/backward.hpp"

namespace xmvb::vb {

using detail::DirectionalOppositeSpinPairData;
using detail::kOppositeSpinUniqueTileSize;

namespace {


constexpr double kContributionTolerance = 1.0e-15;
constexpr int kOppositeSpinBackwardOverlapBlockSize = 32;

int opposite_spin_backward_overlap_block_size() {
  return kOppositeSpinBackwardOverlapBlockSize;
}

bool selected_state_has_local_support(
    const SelectedStateDeterminantCoefficients& state_coefficients) {
  return !state_coefficients.alpha_support.empty() &&
      !state_coefficients.beta_support.empty() &&
      state_coefficients.local_coefficient_matrix.size() != 0;
}

void validate_local_state_coefficient_matrix(
    const SelectedStateDeterminantCoefficients& state_coefficients) {
  if (state_coefficients.local_coefficient_matrix.rows() !=
          static_cast<int>(state_coefficients.alpha_support.size()) ||
      state_coefficients.local_coefficient_matrix.cols() !=
          static_cast<int>(state_coefficients.beta_support.size())) {
    throw std::invalid_argument(
        "selected-state local_coefficient_matrix shape does not match support dimensions");
  }
}

bool dense_matrix_is_effectively_zero(const Eigen::MatrixXd& matrix) {
  return matrix.size() == 0 || matrix.cwiseAbs().maxCoeff() <= kContributionTolerance;
}

std::vector<Eigen::SparseMatrix<double, Eigen::ColMajor, int>> build_first_order_sparse_matrix_block(
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    int n_unique_determinants,
    int packed_pair_begin,
    int packed_pair_end) {
  if (packed_pair_begin < 0 ||
      packed_pair_end < packed_pair_begin) {
    throw std::invalid_argument(
        "packed-pair block range must satisfy 0 <= begin <= end");
  }
  const int block_size = packed_pair_end - packed_pair_begin;
  std::vector<Eigen::SparseMatrix<double, Eigen::ColMajor, int>> sparse_matrices;
  sparse_matrices.resize(block_size);
  if (block_size == 0) {
    return sparse_matrices;
  }

  std::vector<std::vector<Eigen::Triplet<double, int>>> triplets_by_local_index(
      block_size);
  for (int local_index = 0; local_index < block_size; ++local_index) {
    sparse_matrices[local_index].resize(
        n_unique_determinants,
        n_unique_determinants);
  }

  for (int right_unique_index = 0;
       right_unique_index < n_unique_determinants;
       ++right_unique_index) {
    for (int left_unique_index = 0;
         left_unique_index < n_unique_determinants;
         ++left_unique_index) {
      const auto& pair_evaluation =
          ordered_pair_cache[ordered_spin_pair_storage_index(
              left_unique_index,
              right_unique_index,
              n_unique_determinants)];
      const auto& projection =
          pair_evaluation.opposite_spin_pair_cache.first_order_cofactor_projection;
      for (std::size_t entry_index = 0;
           entry_index < projection.packed_pair_indices.size();
           ++entry_index) {
        const int packed_pair_index = projection.packed_pair_indices[entry_index];
        if (packed_pair_index < packed_pair_begin ||
            packed_pair_index >= packed_pair_end) {
          continue;
        }

        const double packed_pair_value =
            projection.packed_pair_values[entry_index];
        if (std::abs(packed_pair_value) <= kContributionTolerance) {
          continue;
        }
        triplets_by_local_index[
            packed_pair_index - packed_pair_begin].emplace_back(
                left_unique_index,
                right_unique_index,
                packed_pair_value);
      }
    }
  }

  for (int local_index = 0; local_index < block_size; ++local_index) {
    auto& sparse_matrix = sparse_matrices[local_index];
    const auto& triplets =
        triplets_by_local_index[local_index];
    if (!triplets.empty()) {
      sparse_matrix.setFromTriplets(triplets.begin(), triplets.end());
    }
  }
  return sparse_matrices;
}

std::vector<Eigen::SparseMatrix<double, Eigen::ColMajor, int>> build_directional_first_order_sparse_matrix_block(
    const std::vector<DirectionalOppositeSpinPairData>& directional_pair_data,
    int n_unique_determinants,
    int packed_pair_begin,
    int packed_pair_end) {
  if (packed_pair_begin < 0 ||
      packed_pair_end < packed_pair_begin) {
    throw std::invalid_argument(
        "packed-pair block range must satisfy 0 <= begin <= end");
  }
  const int block_size = packed_pair_end - packed_pair_begin;
  std::vector<Eigen::SparseMatrix<double, Eigen::ColMajor, int>> sparse_matrices;
  sparse_matrices.resize(block_size);
  if (block_size == 0) {
    return sparse_matrices;
  }

  std::vector<std::vector<Eigen::Triplet<double, int>>> triplets_by_local_index(
      block_size);
  for (int local_index = 0; local_index < block_size; ++local_index) {
    sparse_matrices[local_index].resize(
        n_unique_determinants,
        n_unique_determinants);
  }

  for (int right_unique_index = 0;
       right_unique_index < n_unique_determinants;
       ++right_unique_index) {
    for (int left_unique_index = 0;
         left_unique_index < n_unique_determinants;
         ++left_unique_index) {
      const auto& directional_entry =
          directional_pair_data[ordered_spin_pair_storage_index(
              left_unique_index,
              right_unique_index,
              n_unique_determinants)];
      const auto& projection =
          directional_entry.delta_first_order_cofactor_projection;
      for (std::size_t entry_index = 0;
           entry_index < projection.packed_pair_indices.size();
           ++entry_index) {
        const int packed_pair_index = projection.packed_pair_indices[entry_index];
        if (packed_pair_index < packed_pair_begin ||
            packed_pair_index >= packed_pair_end) {
          continue;
        }

        const double packed_pair_value =
            projection.packed_pair_values[entry_index];
        if (std::abs(packed_pair_value) <= kContributionTolerance) {
          continue;
        }
        triplets_by_local_index[
            packed_pair_index - packed_pair_begin].emplace_back(
                left_unique_index,
                right_unique_index,
                packed_pair_value);
      }
    }
  }

  for (int local_index = 0; local_index < block_size; ++local_index) {
    auto& sparse_matrix = sparse_matrices[local_index];
    const auto& triplets =
        triplets_by_local_index[local_index];
    if (!triplets.empty()) {
      sparse_matrix.setFromTriplets(triplets.begin(), triplets.end());
    }
  }
  return sparse_matrices;
}

std::vector<Eigen::SparseMatrix<double, Eigen::ColMajor, int>>
build_weighted_cofactor_projected_sparse_matrix_block(
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    int n_unique_determinants,
    int packed_pair_begin,
    int packed_pair_end) {
  if (packed_pair_begin < 0 ||
      packed_pair_end < packed_pair_begin) {
    throw std::invalid_argument(
        "packed-pair sparse-image block range must satisfy 0 <= begin <= end");
  }
  const int block_size = packed_pair_end - packed_pair_begin;
  std::vector<Eigen::SparseMatrix<double, Eigen::ColMajor, int>> weighted_images(
      block_size);
  for (auto& weighted_image : weighted_images) {
    weighted_image.resize(n_unique_determinants, n_unique_determinants);
  }
  if (block_size == 0) {
    return weighted_images;
  }

  std::vector<std::vector<Eigen::Triplet<double, int>>> triplets_by_local_index(
      block_size);
  for (int right_unique_index = 0;
       right_unique_index < n_unique_determinants;
       ++right_unique_index) {
    for (int left_unique_index = 0;
         left_unique_index < n_unique_determinants;
         ++left_unique_index) {
      const auto& pair_evaluation =
          ordered_pair_cache[ordered_spin_pair_storage_index(
              left_unique_index,
              right_unique_index,
              n_unique_determinants)];
      const auto& projection =
          pair_evaluation.opposite_spin_pair_cache.first_order_cofactor_projection;
      if (projection.projected_pair_values.empty()) {
        continue;
      }

      const int local_end = std::min(
          block_size,
          static_cast<int>(projection.projected_pair_values.size()) -
              packed_pair_begin);
      for (int local_index = 0; local_index < local_end; ++local_index) {
        const double value =
            projection.projected_pair_values[packed_pair_begin + local_index];
        if (std::abs(value) <= kContributionTolerance) {
          continue;
        }
        triplets_by_local_index[local_index].emplace_back(
            left_unique_index,
            right_unique_index,
            value);
      }
    }
  }

  for (int local_index = 0; local_index < block_size; ++local_index) {
    auto& weighted_image = weighted_images[local_index];
    const auto& triplets = triplets_by_local_index[local_index];
    if (!triplets.empty()) {
      weighted_image.setFromTriplets(triplets.begin(), triplets.end());
    }
  }
  return weighted_images;
}

std::vector<Eigen::SparseMatrix<double, Eigen::ColMajor, int>>
build_directional_weighted_cofactor_projected_sparse_matrix_block(
    const std::vector<DirectionalOppositeSpinPairData>& directional_pair_data,
    int n_unique_determinants,
    int packed_pair_begin,
    int packed_pair_end) {
  if (packed_pair_begin < 0 ||
      packed_pair_end < packed_pair_begin) {
    throw std::invalid_argument(
        "directional packed-pair sparse-image block range must satisfy 0 <= begin <= end");
  }
  const int block_size = packed_pair_end - packed_pair_begin;
  std::vector<Eigen::SparseMatrix<double, Eigen::ColMajor, int>> weighted_images(
      block_size);
  for (auto& weighted_image : weighted_images) {
    weighted_image.resize(n_unique_determinants, n_unique_determinants);
  }
  if (block_size == 0) {
    return weighted_images;
  }

  std::vector<std::vector<Eigen::Triplet<double, int>>> triplets_by_local_index(
      block_size);
  for (int right_unique_index = 0;
       right_unique_index < n_unique_determinants;
       ++right_unique_index) {
    for (int left_unique_index = 0;
         left_unique_index < n_unique_determinants;
         ++left_unique_index) {
      const std::size_t ordered_pair_index = ordered_spin_pair_storage_index(
          left_unique_index,
          right_unique_index,
          n_unique_determinants);
      const auto& directional_projection =
          directional_pair_data[ordered_pair_index]
              .delta_first_order_cofactor_projection;
      if (directional_projection.projected_pair_values.empty()) {
        continue;
      }

      const int local_end = std::min(
          block_size,
          static_cast<int>(directional_projection.projected_pair_values.size()) -
              packed_pair_begin);
      for (int local_index = 0; local_index < local_end; ++local_index) {
        const double value =
            directional_projection.projected_pair_values[
                packed_pair_begin + local_index];
        if (std::abs(value) <= kContributionTolerance) {
          continue;
        }
        triplets_by_local_index[local_index].emplace_back(
            left_unique_index,
            right_unique_index,
            value);
      }
    }
  }

  for (int local_index = 0; local_index < block_size; ++local_index) {
    auto& weighted_image = weighted_images[local_index];
    const auto& triplets = triplets_by_local_index[local_index];
    if (!triplets.empty()) {
      weighted_image.setFromTriplets(triplets.begin(), triplets.end());
    }
  }
  return weighted_images;
}

double contract_sparse_matrix_tile_with_dense_tile_matrix(
    const Eigen::SparseMatrix<double, Eigen::ColMajor, int>& sparse_matrix,
    const Eigen::MatrixXd& dense_tile_matrix,
    int dense_tile_column,
    int tile_left_size,
    int row_begin,
    int column_begin) {
  const int row_end = row_begin + tile_left_size;
  const int column_end =
      column_begin + dense_tile_matrix.rows() / tile_left_size;
  double contraction = 0.0;
  for (int column = column_begin; column < column_end; ++column) {
    for (Eigen::SparseMatrix<double, Eigen::ColMajor, int>::InnerIterator iterator(
             sparse_matrix,
             column);
         iterator;
         ++iterator) {
      const int row = iterator.row();
      if (row < row_begin || row >= row_end) {
        continue;
      }
      const int tile_index =
          (row - row_begin) + tile_left_size * (column - column_begin);
      contraction +=
          iterator.value() *
          dense_tile_matrix(tile_index, dense_tile_column);
    }
  }
  return contraction;
}

void accumulate_alpha_pair_matrix_tile(
    const SelectedStateDeterminantMatrices& selected_states,
    const Eigen::SparseMatrix<double, Eigen::ColMajor, int>& beta_pair_matrix,
    int alpha_left_begin,
    int alpha_left_end,
    int alpha_right_begin,
    int alpha_right_end,
    Eigen::MatrixXd* alpha_pair_tile) {
  alpha_pair_tile->setZero(
      alpha_left_end - alpha_left_begin,
      alpha_right_end - alpha_right_begin);

  const int beta_tile_size = kOppositeSpinUniqueTileSize;
  for (const auto& state_coefficients : selected_states.states) {
    if (!selected_state_has_local_support(state_coefficients)) {
      continue;
    }
    validate_local_state_coefficient_matrix(state_coefficients);

    auto build_partner_tiles = [&](
        const std::vector<int>& beta_support,
        const SupportWindow& beta_left_window,
        const SupportWindow& beta_right_window,
        Eigen::MatrixXd* beta_tile,
        Eigen::MatrixXd* unused_tile) {
      gather_scalar_block_from_support(
          beta_support,
          beta_left_window,
          beta_support,
          beta_right_window,
          [&](int row, int column) {
            return beta_pair_matrix.coeff(row, column);
          },
          beta_tile);
      unused_tile->setZero(beta_tile->rows(), beta_tile->cols());
    };
    auto consume_images = [&](
        const SupportWindow& alpha_left_window,
        const SupportWindow& alpha_right_window,
        const Eigen::MatrixXd& alpha_image,
        const Eigen::MatrixXd& unused_image) {
      (void)unused_image;
      scatter_add_dense_submatrix_to_tile(
          alpha_image,
          state_coefficients.alpha_support,
          alpha_left_window,
          alpha_left_begin,
          state_coefficients.alpha_support,
          alpha_right_window,
          alpha_right_begin,
          state_coefficients.normalized_state_weight,
          alpha_pair_tile);
    };

    for_each_alpha_oriented_support_local_image_pair(
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        alpha_left_begin,
        alpha_left_end,
        alpha_right_begin,
        alpha_right_end,
        beta_tile_size,
        build_partner_tiles,
        consume_images);
  }
}

void accumulate_directional_alpha_pair_matrix_tile(
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    const Eigen::SparseMatrix<double, Eigen::ColMajor, int>& beta_pair_matrix,
    int alpha_left_begin,
    int alpha_left_end,
    int alpha_right_begin,
    int alpha_right_end,
    Eigen::MatrixXd* alpha_pair_tile) {
  alpha_pair_tile->setZero(
      alpha_left_end - alpha_left_begin,
      alpha_right_end - alpha_right_begin);

  const int beta_tile_size = kOppositeSpinUniqueTileSize;
  Eigen::MatrixXd beta_tile;
  Eigen::MatrixXd beta_push;
  Eigen::MatrixXd alpha_image;
  for (std::size_t state_offset = 0;
       state_offset < selected_states.states.size();
       ++state_offset) {
    const auto& state_coefficients = selected_states.states[state_offset];
    const auto& directional_state_coefficients =
        directional_selected_states.states[state_offset];
    if (!selected_state_has_local_support(state_coefficients) ||
        !selected_state_has_local_support(directional_state_coefficients)) {
      continue;
    }
    validate_local_state_coefficient_matrix(state_coefficients);
    validate_local_state_coefficient_matrix(directional_state_coefficients);

    // Directional selected-state packed-gradient weight:
    //   d(C U_beta C^T) = dC U_beta C^T + C U_beta dC^T.
    // The accepted and directional coefficient supports may differ, so both
    // product-rule terms are streamed with their own left/right local supports
    // instead of materializing a union-support dense coefficient block.
    auto accumulate_mixed_image = [&](
        const Eigen::MatrixXd& left_coefficients,
        const std::vector<int>& left_alpha_support,
        const std::vector<int>& left_beta_support,
        const Eigen::MatrixXd& right_coefficients,
        const std::vector<int>& right_alpha_support,
        const std::vector<int>& right_beta_support) {
      const SupportWindow alpha_left_window =
          find_support_window(
              left_alpha_support,
              alpha_left_begin,
              alpha_left_end);
      const SupportWindow alpha_right_window =
          find_support_window(
              right_alpha_support,
              alpha_right_begin,
              alpha_right_end);
      if (alpha_left_window.empty() ||
          alpha_right_window.empty() ||
          left_beta_support.empty() ||
          right_beta_support.empty()) {
        return;
      }

      const int n_left_beta_support =
          static_cast<int>(left_beta_support.size());
      const int n_right_beta_support =
          static_cast<int>(right_beta_support.size());
      for (int beta_left_begin_local = 0;
           beta_left_begin_local < n_left_beta_support;
           beta_left_begin_local += beta_tile_size) {
        const SupportWindow beta_left_window{
            beta_left_begin_local,
            std::min(n_left_beta_support, beta_left_begin_local + beta_tile_size),
        };
        const auto left_block =
            left_coefficients.block(
                alpha_left_window.begin,
                beta_left_window.begin,
                alpha_left_window.size(),
                beta_left_window.size());

        for (int beta_right_begin_local = 0;
             beta_right_begin_local < n_right_beta_support;
             beta_right_begin_local += beta_tile_size) {
          const SupportWindow beta_right_window{
              beta_right_begin_local,
              std::min(
                  n_right_beta_support,
                  beta_right_begin_local + beta_tile_size),
          };
          const auto right_block =
              right_coefficients.block(
                  alpha_right_window.begin,
                  beta_right_window.begin,
                  alpha_right_window.size(),
                  beta_right_window.size());
          gather_scalar_block_from_support(
              left_beta_support,
              beta_left_window,
              right_beta_support,
              beta_right_window,
              [&](int row, int column) {
                return beta_pair_matrix.coeff(row, column);
              },
              &beta_tile);
          beta_push.noalias() = left_block * beta_tile;
          alpha_image.noalias() = beta_push * right_block.transpose();
          scatter_add_dense_submatrix_to_tile(
              alpha_image,
              left_alpha_support,
              alpha_left_window,
              alpha_left_begin,
              right_alpha_support,
              alpha_right_window,
              alpha_right_begin,
              state_coefficients.normalized_state_weight,
              alpha_pair_tile);
        }
      }
    };

    accumulate_mixed_image(
        directional_state_coefficients.local_coefficient_matrix,
        directional_state_coefficients.alpha_support,
        directional_state_coefficients.beta_support,
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support);
    accumulate_mixed_image(
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        directional_state_coefficients.local_coefficient_matrix,
        directional_state_coefficients.alpha_support,
        directional_state_coefficients.beta_support);
  }
}

void accumulate_beta_pair_matrix_tile(
    const SelectedStateDeterminantMatrices& selected_states,
    const Eigen::SparseMatrix<double, Eigen::ColMajor, int>& alpha_pair_matrix,
    int beta_left_begin,
    int beta_left_end,
    int beta_right_begin,
    int beta_right_end,
    Eigen::MatrixXd* beta_pair_tile) {
  beta_pair_tile->setZero(
      beta_left_end - beta_left_begin,
      beta_right_end - beta_right_begin);

  const int alpha_tile_size = kOppositeSpinUniqueTileSize;
  for (const auto& state_coefficients : selected_states.states) {
    if (!selected_state_has_local_support(state_coefficients)) {
      continue;
    }
    validate_local_state_coefficient_matrix(state_coefficients);

    auto build_partner_tiles = [&](
        const std::vector<int>& alpha_support,
        const SupportWindow& alpha_left_window,
        const SupportWindow& alpha_right_window,
        Eigen::MatrixXd* alpha_tile,
        Eigen::MatrixXd* unused_tile) {
      gather_scalar_block_from_support(
          alpha_support,
          alpha_left_window,
          alpha_support,
          alpha_right_window,
          [&](int row, int column) {
            return alpha_pair_matrix.coeff(row, column);
          },
          alpha_tile);
      unused_tile->setZero(alpha_tile->rows(), alpha_tile->cols());
    };
    auto consume_images = [&](
        const SupportWindow& beta_left_window,
        const SupportWindow& beta_right_window,
        const Eigen::MatrixXd& beta_image,
        const Eigen::MatrixXd& unused_image) {
      (void)unused_image;
      scatter_add_dense_submatrix_to_tile(
          beta_image,
          state_coefficients.beta_support,
          beta_left_window,
          beta_left_begin,
          state_coefficients.beta_support,
          beta_right_window,
          beta_right_begin,
          state_coefficients.normalized_state_weight,
          beta_pair_tile);
    };

    for_each_beta_oriented_support_local_image_pair(
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        beta_left_begin,
        beta_left_end,
        beta_right_begin,
        beta_right_end,
        alpha_tile_size,
        build_partner_tiles,
        consume_images);
  }
}

void accumulate_directional_beta_pair_matrix_tile(
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    const Eigen::SparseMatrix<double, Eigen::ColMajor, int>& alpha_pair_matrix,
    int beta_left_begin,
    int beta_left_end,
    int beta_right_begin,
    int beta_right_end,
    Eigen::MatrixXd* beta_pair_tile) {
  beta_pair_tile->setZero(
      beta_left_end - beta_left_begin,
      beta_right_end - beta_right_begin);

  const int alpha_tile_size = kOppositeSpinUniqueTileSize;
  Eigen::MatrixXd alpha_tile;
  Eigen::MatrixXd alpha_push;
  Eigen::MatrixXd beta_image;
  for (std::size_t state_offset = 0;
       state_offset < selected_states.states.size();
       ++state_offset) {
    const auto& state_coefficients = selected_states.states[state_offset];
    const auto& directional_state_coefficients =
        directional_selected_states.states[state_offset];
    if (!selected_state_has_local_support(state_coefficients) ||
        !selected_state_has_local_support(directional_state_coefficients)) {
      continue;
    }
    validate_local_state_coefficient_matrix(state_coefficients);
    validate_local_state_coefficient_matrix(directional_state_coefficients);

    // Beta-side product rule:
    //   d(C^T U_alpha C) = dC^T U_alpha C + C^T U_alpha dC.
    auto accumulate_mixed_image = [&](
        const Eigen::MatrixXd& left_coefficients,
        const std::vector<int>& left_alpha_support,
        const std::vector<int>& left_beta_support,
        const Eigen::MatrixXd& right_coefficients,
        const std::vector<int>& right_alpha_support,
        const std::vector<int>& right_beta_support) {
      const SupportWindow beta_left_window =
          find_support_window(
              left_beta_support,
              beta_left_begin,
              beta_left_end);
      const SupportWindow beta_right_window =
          find_support_window(
              right_beta_support,
              beta_right_begin,
              beta_right_end);
      if (beta_left_window.empty() ||
          beta_right_window.empty() ||
          left_alpha_support.empty() ||
          right_alpha_support.empty()) {
        return;
      }

      const int n_left_alpha_support =
          static_cast<int>(left_alpha_support.size());
      const int n_right_alpha_support =
          static_cast<int>(right_alpha_support.size());
      for (int alpha_left_begin_local = 0;
           alpha_left_begin_local < n_left_alpha_support;
           alpha_left_begin_local += alpha_tile_size) {
        const SupportWindow alpha_left_window{
            alpha_left_begin_local,
            std::min(
                n_left_alpha_support,
                alpha_left_begin_local + alpha_tile_size),
        };
        const auto left_block =
            left_coefficients.block(
                alpha_left_window.begin,
                beta_left_window.begin,
                alpha_left_window.size(),
                beta_left_window.size());

        for (int alpha_right_begin_local = 0;
             alpha_right_begin_local < n_right_alpha_support;
             alpha_right_begin_local += alpha_tile_size) {
          const SupportWindow alpha_right_window{
              alpha_right_begin_local,
              std::min(
                  n_right_alpha_support,
                  alpha_right_begin_local + alpha_tile_size),
          };
          const auto right_block =
              right_coefficients.block(
                  alpha_right_window.begin,
                  beta_right_window.begin,
                  alpha_right_window.size(),
                  beta_right_window.size());
          gather_scalar_block_from_support(
              left_alpha_support,
              alpha_left_window,
              right_alpha_support,
              alpha_right_window,
              [&](int row, int column) {
                return alpha_pair_matrix.coeff(row, column);
              },
              &alpha_tile);
          alpha_push.noalias() = left_block.transpose() * alpha_tile;
          beta_image.noalias() = alpha_push * right_block;
          scatter_add_dense_submatrix_to_tile(
              beta_image,
              left_beta_support,
              beta_left_window,
              beta_left_begin,
              right_beta_support,
              beta_right_window,
              beta_right_begin,
              state_coefficients.normalized_state_weight,
              beta_pair_tile);
        }
      }
    };

    accumulate_mixed_image(
        directional_state_coefficients.local_coefficient_matrix,
        directional_state_coefficients.alpha_support,
        directional_state_coefficients.beta_support,
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support);
    accumulate_mixed_image(
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        directional_state_coefficients.local_coefficient_matrix,
        directional_state_coefficients.alpha_support,
        directional_state_coefficients.beta_support);
  }
}

}  // namespace

namespace detail {

void accumulate_spin_overlap_gradient_direction_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const CofactorDifferential& cofactor,
    const Eigen::MatrixXd& delta_overlap_submatrix,
    const Eigen::MatrixXd& cofactor_weight,
    const Eigen::MatrixXd& delta_cofactor_weight,
    int n_active_orbitals,
    std::vector<double>* active_orbital_overlap_gradient) {
  // E = <W, C(S)>: d(grad_S E) = C''(S)[dS,W] + C'(S)[dW].
  // Tile weights use left-by-right storage, hence the transposes here.
  const Eigen::MatrixXd gradient_direction =
      cofactor.mixed(delta_overlap_submatrix, cofactor_weight.transpose()) +
      cofactor.first(delta_cofactor_weight.transpose());
  for (int left = 0; left < static_cast<int>(occ_L.size()); ++left)
    for (int right = 0; right < static_cast<int>(occ_R.size()); ++right)
      (*active_orbital_overlap_gradient)[occ_L[left] * n_active_orbitals + occ_R[right]] +=
          gradient_direction(right, left);
}

void accumulate_opposite_spin_packed_gradient_by_tiles(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_packed_active_pairs,
    int sparse_block_size,
    int dense_batch_size,
    std::vector<double>* packed_active_two_electron_gradient) {
  const int unique_tile_size = kOppositeSpinUniqueTileSize;
  Eigen::MatrixXd alpha_pair_weight_tile;
  for (int beta_batch_begin = 0;
       beta_batch_begin < n_packed_active_pairs;
       beta_batch_begin += dense_batch_size) {
    const int beta_batch_end =
        std::min(n_packed_active_pairs, beta_batch_begin + dense_batch_size);
    const auto beta_sparse_batch = build_first_order_sparse_matrix_block(
        same_spin_pair_cache.beta_pair_cache_ref(),
        selected_states.n_unique_beta,
        beta_batch_begin,
        beta_batch_end);

    for (int alpha_left_begin = 0;
         alpha_left_begin < selected_states.n_unique_alpha;
         alpha_left_begin += unique_tile_size) {
      const int alpha_left_end =
          std::min(
              selected_states.n_unique_alpha,
              alpha_left_begin + unique_tile_size);
      for (int alpha_right_begin = 0;
           alpha_right_begin < selected_states.n_unique_alpha;
           alpha_right_begin += unique_tile_size) {
        const int alpha_right_end =
            std::min(
                selected_states.n_unique_alpha,
                alpha_right_begin + unique_tile_size);

        std::vector<int> active_beta_packed_pair_indices;
        const int alpha_tile_left_size = alpha_left_end - alpha_left_begin;
        const int alpha_tile_area =
            alpha_tile_left_size * (alpha_right_end - alpha_right_begin);
        Eigen::MatrixXd alpha_pair_weight_tiles(
            alpha_tile_area,
            beta_batch_end - beta_batch_begin);
        active_beta_packed_pair_indices.reserve(beta_sparse_batch.size());
        int active_beta_count = 0;
        for (int beta_local_index = 0;
             beta_local_index < beta_batch_end - beta_batch_begin;
             ++beta_local_index) {
          accumulate_alpha_pair_matrix_tile(
              selected_states,
              beta_sparse_batch[beta_local_index],
              alpha_left_begin,
              alpha_left_end,
              alpha_right_begin,
              alpha_right_end,
              &alpha_pair_weight_tile);
          if (dense_matrix_is_effectively_zero(alpha_pair_weight_tile)) {
            continue;
          }
          active_beta_packed_pair_indices.push_back(
              beta_batch_begin + beta_local_index);
          alpha_pair_weight_tiles.col(active_beta_count) =
              Eigen::Map<const Eigen::VectorXd>(
                  alpha_pair_weight_tile.data(),
                  alpha_pair_weight_tile.size());
          ++active_beta_count;
        }
        if (active_beta_packed_pair_indices.empty()) {
          continue;
        }

        for (int alpha_block_begin = 0;
             alpha_block_begin < n_packed_active_pairs;
             alpha_block_begin += sparse_block_size) {
          const int alpha_block_end =
              std::min(
                  n_packed_active_pairs,
                  alpha_block_begin + sparse_block_size);
          const auto alpha_sparse_block = build_first_order_sparse_matrix_block(
              same_spin_pair_cache.alpha_pair_cache_ref(),
              selected_states.n_unique_alpha,
              alpha_block_begin,
              alpha_block_end);

          for (std::size_t beta_active_index = 0;
               beta_active_index < active_beta_packed_pair_indices.size();
               ++beta_active_index) {
            const int beta_packed_pair_index =
                active_beta_packed_pair_indices[beta_active_index];
            for (int alpha_local_index = 0;
                 alpha_local_index < alpha_block_end - alpha_block_begin;
                 ++alpha_local_index) {
              const double packed_gradient_value =
                  contract_sparse_matrix_tile_with_dense_tile_matrix(
                      alpha_sparse_block[alpha_local_index],
                      alpha_pair_weight_tiles,
                      static_cast<int>(beta_active_index),
                      alpha_tile_left_size,
                      alpha_left_begin,
                      alpha_right_begin);
              if (std::abs(packed_gradient_value) <= kContributionTolerance) {
                continue;
              }

              const int alpha_packed_pair_index =
                  alpha_block_begin + alpha_local_index;
              const int packed_pair_of_pairs_index =
                  TwoElectronIndexer::packed_pair_of_pairs_index(
                      beta_packed_pair_index,
                      alpha_packed_pair_index);
              (*packed_active_two_electron_gradient)[
                  packed_pair_of_pairs_index] += packed_gradient_value;
            }
          }
        }
      }
    }
  }
}

void accumulate_directional_opposite_spin_packed_gradient_by_tiles(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    int n_packed_active_pairs,
    int sparse_block_size,
    int dense_batch_size,
    std::vector<double>* packed_active_two_electron_gradient) {
  const int unique_tile_size = kOppositeSpinUniqueTileSize;
  Eigen::MatrixXd alpha_pair_weight_tile;
  for (int beta_batch_begin = 0;
       beta_batch_begin < n_packed_active_pairs;
       beta_batch_begin += dense_batch_size) {
    const int beta_batch_end =
        std::min(n_packed_active_pairs, beta_batch_begin + dense_batch_size);
    const auto beta_sparse_batch = build_first_order_sparse_matrix_block(
        same_spin_pair_cache.beta_pair_cache_ref(),
        selected_states.n_unique_beta,
        beta_batch_begin,
        beta_batch_end);

    for (int alpha_left_begin = 0;
         alpha_left_begin < selected_states.n_unique_alpha;
         alpha_left_begin += unique_tile_size) {
      const int alpha_left_end =
          std::min(
              selected_states.n_unique_alpha,
              alpha_left_begin + unique_tile_size);
      for (int alpha_right_begin = 0;
           alpha_right_begin < selected_states.n_unique_alpha;
           alpha_right_begin += unique_tile_size) {
        const int alpha_right_end =
            std::min(
                selected_states.n_unique_alpha,
                alpha_right_begin + unique_tile_size);

        std::vector<int> active_beta_packed_pair_indices;
        const int alpha_tile_left_size = alpha_left_end - alpha_left_begin;
        const int alpha_tile_area =
            alpha_tile_left_size * (alpha_right_end - alpha_right_begin);
        Eigen::MatrixXd alpha_pair_weight_tiles(
            alpha_tile_area,
            beta_batch_end - beta_batch_begin);
        active_beta_packed_pair_indices.reserve(beta_sparse_batch.size());
        int active_beta_count = 0;
        for (int beta_local_index = 0;
             beta_local_index < beta_batch_end - beta_batch_begin;
             ++beta_local_index) {
          accumulate_directional_alpha_pair_matrix_tile(
              selected_states,
              directional_selected_states,
              beta_sparse_batch[beta_local_index],
              alpha_left_begin,
              alpha_left_end,
              alpha_right_begin,
              alpha_right_end,
              &alpha_pair_weight_tile);
          if (dense_matrix_is_effectively_zero(alpha_pair_weight_tile)) {
            continue;
          }
          active_beta_packed_pair_indices.push_back(
              beta_batch_begin + beta_local_index);
          alpha_pair_weight_tiles.col(active_beta_count) =
              Eigen::Map<const Eigen::VectorXd>(
                  alpha_pair_weight_tile.data(),
                  alpha_pair_weight_tile.size());
          ++active_beta_count;
        }
        if (active_beta_packed_pair_indices.empty()) {
          continue;
        }

        for (int alpha_block_begin = 0;
             alpha_block_begin < n_packed_active_pairs;
             alpha_block_begin += sparse_block_size) {
          const int alpha_block_end =
              std::min(
                  n_packed_active_pairs,
                  alpha_block_begin + sparse_block_size);
          const auto alpha_sparse_block = build_first_order_sparse_matrix_block(
              same_spin_pair_cache.alpha_pair_cache_ref(),
              selected_states.n_unique_alpha,
              alpha_block_begin,
              alpha_block_end);

          for (std::size_t beta_active_index = 0;
               beta_active_index < active_beta_packed_pair_indices.size();
               ++beta_active_index) {
            const int beta_packed_pair_index =
                active_beta_packed_pair_indices[beta_active_index];
            for (int alpha_local_index = 0;
                 alpha_local_index < alpha_block_end - alpha_block_begin;
                 ++alpha_local_index) {
              const double packed_gradient_value =
                  contract_sparse_matrix_tile_with_dense_tile_matrix(
                      alpha_sparse_block[alpha_local_index],
                      alpha_pair_weight_tiles,
                      static_cast<int>(beta_active_index),
                      alpha_tile_left_size,
                      alpha_left_begin,
                      alpha_right_begin);
              if (std::abs(packed_gradient_value) <= kContributionTolerance) {
                continue;
              }

              const int alpha_packed_pair_index =
                  alpha_block_begin + alpha_local_index;
              const int packed_pair_of_pairs_index =
                  TwoElectronIndexer::packed_pair_of_pairs_index(
                      beta_packed_pair_index,
                      alpha_packed_pair_index);
              (*packed_active_two_electron_gradient)[
                  packed_pair_of_pairs_index] += packed_gradient_value;
            }
          }
        }
      }
    }
  }
}

void accumulate_local_opposite_spin_packed_gradient_by_tiles(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const std::vector<DirectionalOppositeSpinPairData>& alpha_directional_pair_data,
    const std::vector<DirectionalOppositeSpinPairData>& beta_directional_pair_data,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_packed_active_pairs,
    int sparse_block_size,
    int dense_batch_size,
    std::vector<double>* packed_active_two_electron_gradient) {
  const int unique_tile_size = kOppositeSpinUniqueTileSize;
  Eigen::MatrixXd alpha_pair_weight_tile;
  Eigen::MatrixXd alpha_directional_pair_weight_tile;
  for (int beta_batch_begin = 0;
       beta_batch_begin < n_packed_active_pairs;
       beta_batch_begin += dense_batch_size) {
    const int beta_batch_end =
        std::min(n_packed_active_pairs, beta_batch_begin + dense_batch_size);
    const auto beta_sparse_batch = build_first_order_sparse_matrix_block(
        same_spin_pair_cache.beta_pair_cache_ref(),
        selected_states.n_unique_beta,
        beta_batch_begin,
        beta_batch_end);
    const auto beta_directional_sparse_batch =
        build_directional_first_order_sparse_matrix_block(
            beta_directional_pair_data,
            selected_states.n_unique_beta,
            beta_batch_begin,
            beta_batch_end);

    for (int alpha_left_begin = 0;
         alpha_left_begin < selected_states.n_unique_alpha;
         alpha_left_begin += unique_tile_size) {
      const int alpha_left_end =
          std::min(
              selected_states.n_unique_alpha,
              alpha_left_begin + unique_tile_size);
      for (int alpha_right_begin = 0;
           alpha_right_begin < selected_states.n_unique_alpha;
           alpha_right_begin += unique_tile_size) {
        const int alpha_right_end =
            std::min(
                selected_states.n_unique_alpha,
                alpha_right_begin + unique_tile_size);

        std::vector<int> active_beta_packed_pair_indices;
        const int alpha_tile_left_size = alpha_left_end - alpha_left_begin;
        const int alpha_tile_area =
            alpha_tile_left_size * (alpha_right_end - alpha_right_begin);
        Eigen::MatrixXd alpha_pair_weight_tiles(
            alpha_tile_area,
            beta_batch_end - beta_batch_begin);
        Eigen::MatrixXd alpha_directional_pair_weight_tiles(
            alpha_tile_area,
            beta_batch_end - beta_batch_begin);
        std::vector<unsigned char> accepted_tile_nonzero;
        std::vector<unsigned char> directional_tile_nonzero;
        active_beta_packed_pair_indices.reserve(beta_sparse_batch.size());
        accepted_tile_nonzero.reserve(beta_sparse_batch.size());
        directional_tile_nonzero.reserve(beta_sparse_batch.size());
        int active_beta_count = 0;
        for (int beta_local_index = 0;
             beta_local_index < beta_batch_end - beta_batch_begin;
             ++beta_local_index) {
          accumulate_alpha_pair_matrix_tile(
              selected_states,
              beta_sparse_batch[beta_local_index],
              alpha_left_begin,
              alpha_left_end,
              alpha_right_begin,
              alpha_right_end,
              &alpha_pair_weight_tile);
          accumulate_alpha_pair_matrix_tile(
              selected_states,
              beta_directional_sparse_batch[beta_local_index],
              alpha_left_begin,
              alpha_left_end,
              alpha_right_begin,
              alpha_right_end,
              &alpha_directional_pair_weight_tile);
          const bool accepted_nonzero =
              !dense_matrix_is_effectively_zero(alpha_pair_weight_tile);
          const bool directional_nonzero =
              !dense_matrix_is_effectively_zero(
                  alpha_directional_pair_weight_tile);
          if (!accepted_nonzero && !directional_nonzero) {
            continue;
          }
          active_beta_packed_pair_indices.push_back(
              beta_batch_begin + beta_local_index);
          alpha_pair_weight_tiles.col(active_beta_count) =
              Eigen::Map<const Eigen::VectorXd>(
                  alpha_pair_weight_tile.data(),
                  alpha_pair_weight_tile.size());
          alpha_directional_pair_weight_tiles.col(active_beta_count) =
              Eigen::Map<const Eigen::VectorXd>(
                  alpha_directional_pair_weight_tile.data(),
                  alpha_directional_pair_weight_tile.size());
          accepted_tile_nonzero.push_back(accepted_nonzero ? 1u : 0u);
          directional_tile_nonzero.push_back(directional_nonzero ? 1u : 0u);
          ++active_beta_count;
        }
        if (active_beta_packed_pair_indices.empty()) {
          continue;
        }

        for (int alpha_block_begin = 0;
             alpha_block_begin < n_packed_active_pairs;
             alpha_block_begin += sparse_block_size) {
          const int alpha_block_end =
              std::min(
                  n_packed_active_pairs,
                  alpha_block_begin + sparse_block_size);
          const auto alpha_sparse_block = build_first_order_sparse_matrix_block(
              same_spin_pair_cache.alpha_pair_cache_ref(),
              selected_states.n_unique_alpha,
              alpha_block_begin,
              alpha_block_end);
          const auto alpha_directional_sparse_block =
              build_directional_first_order_sparse_matrix_block(
                  alpha_directional_pair_data,
                  selected_states.n_unique_alpha,
                  alpha_block_begin,
                  alpha_block_end);

          for (std::size_t beta_active_index = 0;
               beta_active_index < active_beta_packed_pair_indices.size();
               ++beta_active_index) {
            const int beta_packed_pair_index =
                active_beta_packed_pair_indices[beta_active_index];
            for (int alpha_local_index = 0;
                 alpha_local_index < alpha_block_end - alpha_block_begin;
                 ++alpha_local_index) {
              double packed_gradient_value = 0.0;
              if (accepted_tile_nonzero[beta_active_index] != 0u) {
                packed_gradient_value +=
                    contract_sparse_matrix_tile_with_dense_tile_matrix(
                        alpha_directional_sparse_block[alpha_local_index],
                        alpha_pair_weight_tiles,
                        static_cast<int>(beta_active_index),
                        alpha_tile_left_size,
                        alpha_left_begin,
                        alpha_right_begin);
              }
              if (directional_tile_nonzero[beta_active_index] != 0u) {
                packed_gradient_value +=
                    contract_sparse_matrix_tile_with_dense_tile_matrix(
                        alpha_sparse_block[alpha_local_index],
                        alpha_directional_pair_weight_tiles,
                        static_cast<int>(beta_active_index),
                        alpha_tile_left_size,
                        alpha_left_begin,
                        alpha_right_begin);
              }
              if (std::abs(packed_gradient_value) <= kContributionTolerance) {
                continue;
              }

              const int alpha_packed_pair_index =
                  alpha_block_begin + alpha_local_index;
              const int packed_pair_of_pairs_index =
                  TwoElectronIndexer::packed_pair_of_pairs_index(
                      beta_packed_pair_index,
                      alpha_packed_pair_index);
              (*packed_active_two_electron_gradient)[
                  packed_pair_of_pairs_index] += packed_gradient_value;
            }
          }
        }
      }
    }
  }
}

// Store the overlap-adjoint tile bundle as one column-major matrix. Column P is
// W_P(left_local, right_local) with left_local as the fastest index, which
// preserves the mathematical tile layout while avoiding one allocation per
// packed active pair.
Eigen::MatrixXd build_alpha_overlap_weight_tile_matrix(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_packed_active_pairs,
    int alpha_left_begin,
    int alpha_left_end,
    int alpha_right_begin,
    int alpha_right_end) {
  const int n_left = alpha_left_end - alpha_left_begin;
  const int n_right = alpha_right_end - alpha_right_begin;
  Eigen::MatrixXd tile_matrix =
      Eigen::MatrixXd::Zero(n_left * n_right, n_packed_active_pairs);

  Eigen::MatrixXd tile;
  const int block_size =
      std::min(n_packed_active_pairs, opposite_spin_backward_overlap_block_size());
  for (int packed_pair_begin = 0;
       packed_pair_begin < n_packed_active_pairs;
       packed_pair_begin += block_size) {
    const int packed_pair_end =
        std::min(n_packed_active_pairs, packed_pair_begin + block_size);
    const auto beta_weighted_sparse_block =
        build_weighted_cofactor_projected_sparse_matrix_block(
            same_spin_pair_cache.beta_pair_cache_ref(),
            selected_states.n_unique_beta,
            packed_pair_begin,
            packed_pair_end);
    for (int local_index = 0;
         local_index < packed_pair_end - packed_pair_begin;
         ++local_index) {
      accumulate_alpha_pair_matrix_tile(
          selected_states,
          beta_weighted_sparse_block[local_index],
          alpha_left_begin,
          alpha_left_end,
          alpha_right_begin,
          alpha_right_end,
          &tile);
      if (!dense_matrix_is_effectively_zero(tile)) {
        tile_matrix.col(packed_pair_begin + local_index) =
            Eigen::Map<const Eigen::VectorXd>(tile.data(), tile.size());
      }
    }
  }
  return tile_matrix;
}

Eigen::MatrixXd build_beta_overlap_weight_tile_matrix(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_packed_active_pairs,
    int beta_left_begin,
    int beta_left_end,
    int beta_right_begin,
    int beta_right_end) {
  const int n_left = beta_left_end - beta_left_begin;
  const int n_right = beta_right_end - beta_right_begin;
  Eigen::MatrixXd tile_matrix =
      Eigen::MatrixXd::Zero(n_left * n_right, n_packed_active_pairs);

  Eigen::MatrixXd tile;
  const int block_size =
      std::min(n_packed_active_pairs, opposite_spin_backward_overlap_block_size());
  for (int packed_pair_begin = 0;
       packed_pair_begin < n_packed_active_pairs;
       packed_pair_begin += block_size) {
    const int packed_pair_end =
        std::min(n_packed_active_pairs, packed_pair_begin + block_size);
    const auto alpha_weighted_sparse_block =
        build_weighted_cofactor_projected_sparse_matrix_block(
            same_spin_pair_cache.alpha_pair_cache_ref(),
            selected_states.n_unique_alpha,
            packed_pair_begin,
            packed_pair_end);
    for (int local_index = 0;
         local_index < packed_pair_end - packed_pair_begin;
         ++local_index) {
      accumulate_beta_pair_matrix_tile(
          selected_states,
          alpha_weighted_sparse_block[local_index],
          beta_left_begin,
          beta_left_end,
          beta_right_begin,
          beta_right_end,
          &tile);
      if (!dense_matrix_is_effectively_zero(tile)) {
        tile_matrix.col(packed_pair_begin + local_index) =
            Eigen::Map<const Eigen::VectorXd>(tile.data(), tile.size());
      }
    }
  }
  return tile_matrix;
}

Eigen::MatrixXd build_directional_alpha_overlap_weight_tile_matrix(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    int n_packed_active_pairs,
    int alpha_left_begin,
    int alpha_left_end,
    int alpha_right_begin,
    int alpha_right_end) {
  const int n_left = alpha_left_end - alpha_left_begin;
  const int n_right = alpha_right_end - alpha_right_begin;
  Eigen::MatrixXd tile_matrix =
      Eigen::MatrixXd::Zero(n_left * n_right, n_packed_active_pairs);

  Eigen::MatrixXd tile;
  const int block_size =
      std::min(n_packed_active_pairs, opposite_spin_backward_overlap_block_size());
  for (int packed_pair_begin = 0;
       packed_pair_begin < n_packed_active_pairs;
       packed_pair_begin += block_size) {
    const int packed_pair_end =
        std::min(n_packed_active_pairs, packed_pair_begin + block_size);
    const auto beta_weighted_sparse_block =
        build_weighted_cofactor_projected_sparse_matrix_block(
            same_spin_pair_cache.beta_pair_cache_ref(),
            selected_states.n_unique_beta,
            packed_pair_begin,
            packed_pair_end);
    for (int local_index = 0;
         local_index < packed_pair_end - packed_pair_begin;
         ++local_index) {
      accumulate_directional_alpha_pair_matrix_tile(
          selected_states,
          directional_selected_states,
          beta_weighted_sparse_block[local_index],
          alpha_left_begin,
          alpha_left_end,
          alpha_right_begin,
          alpha_right_end,
          &tile);
      if (!dense_matrix_is_effectively_zero(tile)) {
        tile_matrix.col(packed_pair_begin + local_index) =
            Eigen::Map<const Eigen::VectorXd>(tile.data(), tile.size());
      }
    }
  }
  return tile_matrix;
}

Eigen::MatrixXd build_directional_beta_overlap_weight_tile_matrix(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    int n_packed_active_pairs,
    int beta_left_begin,
    int beta_left_end,
    int beta_right_begin,
    int beta_right_end) {
  const int n_left = beta_left_end - beta_left_begin;
  const int n_right = beta_right_end - beta_right_begin;
  Eigen::MatrixXd tile_matrix =
      Eigen::MatrixXd::Zero(n_left * n_right, n_packed_active_pairs);

  Eigen::MatrixXd tile;
  const int block_size =
      std::min(n_packed_active_pairs, opposite_spin_backward_overlap_block_size());
  for (int packed_pair_begin = 0;
       packed_pair_begin < n_packed_active_pairs;
       packed_pair_begin += block_size) {
    const int packed_pair_end =
        std::min(n_packed_active_pairs, packed_pair_begin + block_size);
    const auto alpha_weighted_sparse_block =
        build_weighted_cofactor_projected_sparse_matrix_block(
            same_spin_pair_cache.alpha_pair_cache_ref(),
            selected_states.n_unique_alpha,
            packed_pair_begin,
            packed_pair_end);
    for (int local_index = 0;
         local_index < packed_pair_end - packed_pair_begin;
         ++local_index) {
      accumulate_directional_beta_pair_matrix_tile(
          selected_states,
          directional_selected_states,
          alpha_weighted_sparse_block[local_index],
          beta_left_begin,
          beta_left_end,
          beta_right_begin,
          beta_right_end,
          &tile);
      if (!dense_matrix_is_effectively_zero(tile)) {
        tile_matrix.col(packed_pair_begin + local_index) =
            Eigen::Map<const Eigen::VectorXd>(tile.data(), tile.size());
      }
    }
  }
  return tile_matrix;
}

Eigen::MatrixXd build_local_directional_alpha_overlap_weight_tile_matrix(
    const std::vector<DirectionalOppositeSpinPairData>& beta_directional_pair_data,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_packed_active_pairs,
    int alpha_left_begin,
    int alpha_left_end,
    int alpha_right_begin,
    int alpha_right_end) {
  const int n_left = alpha_left_end - alpha_left_begin;
  const int n_right = alpha_right_end - alpha_right_begin;
  Eigen::MatrixXd tile_matrix =
      Eigen::MatrixXd::Zero(n_left * n_right, n_packed_active_pairs);

  Eigen::MatrixXd tile;
  const int block_size =
      std::min(n_packed_active_pairs, opposite_spin_backward_overlap_block_size());
  for (int packed_pair_begin = 0;
       packed_pair_begin < n_packed_active_pairs;
       packed_pair_begin += block_size) {
    const int packed_pair_end =
        std::min(n_packed_active_pairs, packed_pair_begin + block_size);
    const auto beta_directional_weighted_sparse_block =
        build_directional_weighted_cofactor_projected_sparse_matrix_block(
            beta_directional_pair_data,
            selected_states.n_unique_beta,
            packed_pair_begin,
            packed_pair_end);
    for (int local_index = 0;
         local_index < packed_pair_end - packed_pair_begin;
         ++local_index) {
      accumulate_alpha_pair_matrix_tile(
          selected_states,
          beta_directional_weighted_sparse_block[local_index],
          alpha_left_begin,
          alpha_left_end,
          alpha_right_begin,
          alpha_right_end,
          &tile);
      if (!dense_matrix_is_effectively_zero(tile)) {
        tile_matrix.col(packed_pair_begin + local_index) =
            Eigen::Map<const Eigen::VectorXd>(tile.data(), tile.size());
      }
    }
  }
  return tile_matrix;
}

Eigen::MatrixXd build_local_directional_beta_overlap_weight_tile_matrix(
    const std::vector<DirectionalOppositeSpinPairData>& alpha_directional_pair_data,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_packed_active_pairs,
    int beta_left_begin,
    int beta_left_end,
    int beta_right_begin,
    int beta_right_end) {
  const int n_left = beta_left_end - beta_left_begin;
  const int n_right = beta_right_end - beta_right_begin;
  Eigen::MatrixXd tile_matrix =
      Eigen::MatrixXd::Zero(n_left * n_right, n_packed_active_pairs);

  Eigen::MatrixXd tile;
  const int block_size =
      std::min(n_packed_active_pairs, opposite_spin_backward_overlap_block_size());
  for (int packed_pair_begin = 0;
       packed_pair_begin < n_packed_active_pairs;
       packed_pair_begin += block_size) {
    const int packed_pair_end =
        std::min(n_packed_active_pairs, packed_pair_begin + block_size);
    const auto alpha_directional_weighted_sparse_block =
        build_directional_weighted_cofactor_projected_sparse_matrix_block(
            alpha_directional_pair_data,
            selected_states.n_unique_alpha,
            packed_pair_begin,
            packed_pair_end);
    for (int local_index = 0;
         local_index < packed_pair_end - packed_pair_begin;
         ++local_index) {
      accumulate_beta_pair_matrix_tile(
          selected_states,
          alpha_directional_weighted_sparse_block[local_index],
          beta_left_begin,
          beta_left_end,
          beta_right_begin,
          beta_right_end,
          &tile);
      if (!dense_matrix_is_effectively_zero(tile)) {
        tile_matrix.col(packed_pair_begin + local_index) =
            Eigen::Map<const Eigen::VectorXd>(tile.data(), tile.size());
      }
    }
  }
  return tile_matrix;
}

void build_local_packed_pair_dense_image_matrix_from_tile_matrix(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& packed_pair_tile_matrix,
    int tile_index,
    Eigen::MatrixXd* pair_dense_image) {
  pair_dense_image->resize(
      static_cast<int>(occ_R.size()),
      static_cast<int>(occ_L.size()));
  for (int left_column = 0;
       left_column < static_cast<int>(occ_L.size());
       ++left_column) {
    const int orbital_index_left = occ_L[left_column];
    for (int right_row = 0;
         right_row < static_cast<int>(occ_R.size());
         ++right_row) {
      const int orbital_index_right = occ_R[right_row];
      const int packed_pair_index =
          TwoElectronIndexer::packed_pair_index(
              orbital_index_right,
              orbital_index_left);
      (*pair_dense_image)(right_row, left_column) =
          packed_pair_tile_matrix(tile_index, packed_pair_index);
    }
  }
}

void build_inverse_overlap_gradient_from_tile_matrix(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& packed_pair_tile_matrix,
    int tile_index,
    Eigen::MatrixXd* inverse_overlap_gradient) {
  const int n_electrons = static_cast<int>(occ_L.size());
  inverse_overlap_gradient->setZero(n_electrons, n_electrons);
  for (int left_column = 0; left_column < n_electrons; ++left_column) {
    const int orbital_index_left = occ_L[left_column];
    for (int right_row = 0; right_row < n_electrons; ++right_row) {
      const int orbital_index_right = occ_R[right_row];
      const int packed_pair_index =
          TwoElectronIndexer::packed_pair_index(
              orbital_index_right,
              orbital_index_left);
      (*inverse_overlap_gradient)(left_column, right_row) =
          packed_pair_tile_matrix(tile_index, packed_pair_index);
    }
  }
}


}  // namespace detail

}  // namespace xmvb::vb
