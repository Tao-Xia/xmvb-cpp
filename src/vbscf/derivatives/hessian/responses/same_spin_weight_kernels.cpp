#include "vbscf/derivatives/hessian/responses/same_spin_weight_builder_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin_tile_weights_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin_weight_kernels_internal.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <cblas.h>

#include "vbscf/derivatives/hessian/responses/same_spin_pair_response_internal.hpp"
#include "vbscf/determinants/spin_pair_contractions.hpp"
#include "vbscf/structures/support_local_contractions.hpp"
#include "vbscf/integrals/active/active_space_two_electron_kernel.hpp"

namespace xmvb::vb::detail {

namespace {

constexpr double kContributionTolerance = 1.0e-15;
constexpr int kSameSpinBackwardPairTileSize = 64;

std::size_t square_storage_size(int dimension) {
  return static_cast<std::size_t>(dimension) *
      static_cast<std::size_t>(dimension);
}

int same_spin_backward_pair_tile_size() {
  return kSameSpinBackwardPairTileSize;
}

}  // namespace

double max_abs_dense_matrix(const Eigen::MatrixXd& matrix) {
  if (matrix.size() == 0) {
    return 0.0;
  }
  return matrix.cwiseAbs().maxCoeff();
}

void set_symmetric_matrix_entry(
    Eigen::MatrixXd* matrix,
    int row,
    int column,
    double value) {
  (*matrix)(row, column) = value;
  (*matrix)(column, row) = value;
}

void multiply_left_symmetric(
    const Eigen::MatrixXd& symmetric_left,
    const Eigen::MatrixXd& right,
    Eigen::MatrixXd* output) {
  if (output == nullptr) {
    throw std::invalid_argument("output must not be null");
  }
  if (symmetric_left.rows() != symmetric_left.cols() ||
      symmetric_left.cols() != right.rows()) {
    throw std::invalid_argument("left symmetric multiply shape mismatch");
  }
  output->resize(symmetric_left.rows(), right.cols());
  if (output->size() == 0) {
    output->setZero();
    return;
  }
  cblas_dsymm(
      CblasColMajor,
      CblasLeft,
      CblasUpper,
      symmetric_left.rows(),
      right.cols(),
      1.0,
      symmetric_left.data(),
      symmetric_left.outerStride(),
      right.data(),
      right.outerStride(),
      0.0,
      output->data(),
      output->outerStride());
}

void multiply_right_symmetric(
    const Eigen::MatrixXd& left,
    const Eigen::MatrixXd& symmetric_right,
    Eigen::MatrixXd* output) {
  if (output == nullptr) {
    throw std::invalid_argument("output must not be null");
  }
  if (symmetric_right.rows() != symmetric_right.cols() ||
      left.cols() != symmetric_right.rows()) {
    throw std::invalid_argument("right symmetric multiply shape mismatch");
  }
  output->resize(left.rows(), symmetric_right.cols());
  if (output->size() == 0) {
    output->setZero();
    return;
  }
  cblas_dsymm(
      CblasColMajor,
      CblasRight,
      CblasUpper,
      left.rows(),
      symmetric_right.cols(),
      1.0,
      symmetric_right.data(),
      symmetric_right.outerStride(),
      left.data(),
      left.outerStride(),
      0.0,
      output->data(),
      output->outerStride());
}

bool selected_state_has_local_support(
    const SelectedStateDeterminantCoefficients& state_coefficients) {
  return !state_coefficients.alpha_support.empty() &&
      !state_coefficients.beta_support.empty() &&
      state_coefficients.local_coefficient_matrix.size() != 0;
}

bool selected_state_has_close_shell_diagonal(
    const SelectedStateDeterminantCoefficients& state_coefficients,
    int n_unique_alpha,
    int n_unique_beta) {
  return state_coefficients.close_shell_diagonal &&
      n_unique_alpha == n_unique_beta &&
      state_coefficients.diagonal_coefficients.size() ==
          static_cast<std::size_t>(n_unique_alpha);
}

bool selected_state_has_local_close_shell_diagonal(
    const SelectedStateDeterminantCoefficients& state_coefficients) {
  return state_coefficients.close_shell_diagonal &&
      state_coefficients.alpha_support == state_coefficients.beta_support &&
      state_coefficients.local_diagonal_coefficients.size() ==
          state_coefficients.alpha_support.size();
}

void accumulate_diagonal_kernel_image_global(
    const std::vector<double>& diagonal_coefficients,
    const Eigen::MatrixXd& partner_kernel_matrix,
    double scale,
    Eigen::MatrixXd* global_weight_matrix) {
  if (global_weight_matrix == nullptr) {
    throw std::invalid_argument("global_weight_matrix must not be null");
  }
  if (std::abs(scale) <= kContributionTolerance ||
      diagonal_coefficients.empty()) {
    return;
  }
  if (partner_kernel_matrix.rows() != static_cast<int>(diagonal_coefficients.size()) ||
      partner_kernel_matrix.cols() != static_cast<int>(diagonal_coefficients.size())) {
    throw std::invalid_argument(
        "partner kernel shape does not match close-shell diagonal coefficients");
  }

  for (int column = 0;
       column < static_cast<int>(diagonal_coefficients.size());
       ++column) {
    const double right_coefficient = diagonal_coefficients[column];
    if (std::abs(right_coefficient) <= kContributionTolerance) {
      continue;
    }
    for (int row = 0;
         row < static_cast<int>(diagonal_coefficients.size());
         ++row) {
      const double left_coefficient = diagonal_coefficients[row];
      if (std::abs(left_coefficient) <= kContributionTolerance) {
        continue;
      }
      (*global_weight_matrix)(row, column) +=
          scale *
          left_coefficient *
          partner_kernel_matrix(row, column) *
          right_coefficient;
    }
  }
}

void accumulate_directional_diagonal_kernel_image_global(
    const std::vector<double>& diagonal_coefficients,
    const std::vector<double>& directional_diagonal_coefficients,
    const Eigen::MatrixXd& partner_kernel_matrix,
    double scale,
    Eigen::MatrixXd* global_weight_matrix) {
  if (global_weight_matrix == nullptr) {
    throw std::invalid_argument("global_weight_matrix must not be null");
  }
  if (std::abs(scale) <= kContributionTolerance ||
      diagonal_coefficients.empty()) {
    return;
  }
  if (directional_diagonal_coefficients.size() != diagonal_coefficients.size() ||
      partner_kernel_matrix.rows() != static_cast<int>(diagonal_coefficients.size()) ||
      partner_kernel_matrix.cols() != static_cast<int>(diagonal_coefficients.size())) {
    throw std::invalid_argument(
        "directional close-shell diagonal contraction dimensions do not match");
  }

  for (int column = 0;
       column < static_cast<int>(diagonal_coefficients.size());
       ++column) {
    const double right_coefficient = diagonal_coefficients[column];
    const double directional_right =
        directional_diagonal_coefficients[column];
    if (std::abs(right_coefficient) <= kContributionTolerance &&
        std::abs(directional_right) <= kContributionTolerance) {
      continue;
    }
    for (int row = 0;
         row < static_cast<int>(diagonal_coefficients.size());
         ++row) {
      const double left_coefficient = diagonal_coefficients[row];
      const double directional_left =
          directional_diagonal_coefficients[row];
      const double directional_value =
          directional_left * right_coefficient +
          left_coefficient * directional_right;
      if (std::abs(directional_value) <= kContributionTolerance) {
        continue;
      }
      (*global_weight_matrix)(row, column) +=
          scale *
          partner_kernel_matrix(row, column) *
          directional_value;
    }
  }
}

void validate_state_coefficient_matrix(
    const SelectedStateDeterminantCoefficients& state_coefficients,
    int n_unique_alpha,
    int n_unique_beta) {
  if (state_coefficients.coefficient_matrix.rows() != n_unique_alpha ||
      state_coefficients.coefficient_matrix.cols() != n_unique_beta) {
    throw std::invalid_argument(
        "selected-state coefficient_matrix shape does not match unique-spin dimensions");
  }
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

void scatter_add_dense_submatrix(
    const Eigen::MatrixXd& local_matrix,
    const std::vector<int>& row_indices,
    const std::vector<int>& column_indices,
    double scale,
    Eigen::MatrixXd* global_matrix) {
  if (global_matrix == nullptr) {
    throw std::invalid_argument("global_matrix must not be null");
  }
  if (local_matrix.rows() != static_cast<int>(row_indices.size()) ||
      local_matrix.cols() != static_cast<int>(column_indices.size())) {
    throw std::invalid_argument(
        "local_matrix shape does not match scatter support dimensions");
  }
  if (std::abs(scale) <= kContributionTolerance) {
    return;
  }
  for (int column_local = 0;
       column_local < static_cast<int>(column_indices.size());
       ++column_local) {
    const int column_global = column_indices[column_local];
    for (int row_local = 0;
         row_local < static_cast<int>(row_indices.size());
         ++row_local) {
      const int row_global = row_indices[row_local];
      (*global_matrix)(row_global, column_global) +=
          scale * local_matrix(row_local, column_local);
    }
  }
}

std::vector<int> build_merged_support_indices(
    const std::vector<int>& first,
    const std::vector<int>& second) {
  std::vector<int> merged;
  merged.reserve(first.size() + second.size());
  std::set_union(
      first.begin(),
      first.end(),
      second.begin(),
      second.end(),
      std::back_inserter(merged));
  return merged;
}

void accumulate_selected_state_alpha_image(
    const SelectedStateDeterminantCoefficients& state_coefficients,
    const Eigen::MatrixXd& beta_kernel_subblock,
    double scale,
    Eigen::MatrixXd* global_alpha_weight_matrix,
    Eigen::MatrixXd* beta_push,
    Eigen::MatrixXd* alpha_image) {
  if (global_alpha_weight_matrix == nullptr ||
      beta_push == nullptr ||
      alpha_image == nullptr) {
    throw std::invalid_argument("alpha image outputs must not be null");
  }
  if (std::abs(scale) <= kContributionTolerance ||
      !selected_state_has_local_support(state_coefficients)) {
    return;
  }
  if (selected_state_has_local_close_shell_diagonal(state_coefficients)) {
    const auto& local_diagonal_coefficients =
        state_coefficients.local_diagonal_coefficients;
    if (beta_kernel_subblock.rows() !=
            static_cast<int>(local_diagonal_coefficients.size()) ||
        beta_kernel_subblock.cols() !=
            static_cast<int>(local_diagonal_coefficients.size())) {
      throw std::invalid_argument(
          "beta support kernel shape does not match close-shell local diagonal coefficients");
    }
    alpha_image->setZero(
        static_cast<int>(local_diagonal_coefficients.size()),
        static_cast<int>(local_diagonal_coefficients.size()));
    for (int column = 0;
         column < static_cast<int>(local_diagonal_coefficients.size());
         ++column) {
      const double right_coefficient =
          local_diagonal_coefficients[column];
      if (std::abs(right_coefficient) <= kContributionTolerance) {
        continue;
      }
      for (int row = 0;
           row < static_cast<int>(local_diagonal_coefficients.size());
           ++row) {
        const double left_coefficient =
            local_diagonal_coefficients[row];
        if (std::abs(left_coefficient) <= kContributionTolerance) {
          continue;
        }
        (*alpha_image)(row, column) =
            left_coefficient *
            beta_kernel_subblock(row, column) *
            right_coefficient;
      }
    }
    scatter_add_dense_submatrix(
        *alpha_image,
        state_coefficients.alpha_support,
        state_coefficients.alpha_support,
        scale,
        global_alpha_weight_matrix);
    return;
  }
  validate_local_state_coefficient_matrix(state_coefficients);
  const auto& local_coefficients = state_coefficients.local_coefficient_matrix;
  if (beta_kernel_subblock.rows() != local_coefficients.cols() ||
      beta_kernel_subblock.cols() != local_coefficients.cols()) {
    throw std::invalid_argument(
        "beta support kernel shape does not match selected-state local coefficients");
  }

  multiply_right_symmetric(
      local_coefficients,
      beta_kernel_subblock,
      beta_push);
  alpha_image->noalias() =
      (*beta_push) * local_coefficients.transpose();
  scatter_add_dense_submatrix(
      *alpha_image,
      state_coefficients.alpha_support,
      state_coefficients.alpha_support,
      scale,
      global_alpha_weight_matrix);
}

void accumulate_selected_state_beta_image(
    const SelectedStateDeterminantCoefficients& state_coefficients,
    const Eigen::MatrixXd& alpha_kernel_subblock,
    double scale,
    Eigen::MatrixXd* global_beta_weight_matrix,
    Eigen::MatrixXd* alpha_push,
    Eigen::MatrixXd* beta_image) {
  if (global_beta_weight_matrix == nullptr ||
      alpha_push == nullptr ||
      beta_image == nullptr) {
    throw std::invalid_argument("beta image outputs must not be null");
  }
  if (std::abs(scale) <= kContributionTolerance ||
      !selected_state_has_local_support(state_coefficients)) {
    return;
  }
  if (selected_state_has_local_close_shell_diagonal(state_coefficients)) {
    const auto& local_diagonal_coefficients =
        state_coefficients.local_diagonal_coefficients;
    if (alpha_kernel_subblock.rows() !=
            static_cast<int>(local_diagonal_coefficients.size()) ||
        alpha_kernel_subblock.cols() !=
            static_cast<int>(local_diagonal_coefficients.size())) {
      throw std::invalid_argument(
          "alpha support kernel shape does not match close-shell local diagonal coefficients");
    }
    beta_image->setZero(
        static_cast<int>(local_diagonal_coefficients.size()),
        static_cast<int>(local_diagonal_coefficients.size()));
    for (int column = 0;
         column < static_cast<int>(local_diagonal_coefficients.size());
         ++column) {
      const double right_coefficient =
          local_diagonal_coefficients[column];
      if (std::abs(right_coefficient) <= kContributionTolerance) {
        continue;
      }
      for (int row = 0;
           row < static_cast<int>(local_diagonal_coefficients.size());
           ++row) {
        const double left_coefficient =
            local_diagonal_coefficients[row];
        if (std::abs(left_coefficient) <= kContributionTolerance) {
          continue;
        }
        (*beta_image)(row, column) =
            left_coefficient *
            alpha_kernel_subblock(row, column) *
            right_coefficient;
      }
    }
    scatter_add_dense_submatrix(
        *beta_image,
        state_coefficients.beta_support,
        state_coefficients.beta_support,
        scale,
        global_beta_weight_matrix);
    return;
  }
  validate_local_state_coefficient_matrix(state_coefficients);
  const auto& local_coefficients = state_coefficients.local_coefficient_matrix;
  if (alpha_kernel_subblock.rows() != local_coefficients.rows() ||
      alpha_kernel_subblock.cols() != local_coefficients.rows()) {
    throw std::invalid_argument(
        "alpha support kernel shape does not match selected-state local coefficients");
  }

  multiply_left_symmetric(
      alpha_kernel_subblock,
      local_coefficients,
      alpha_push);
  beta_image->noalias() =
      local_coefficients.transpose() * (*alpha_push);
  scatter_add_dense_submatrix(
      *beta_image,
      state_coefficients.beta_support,
      state_coefficients.beta_support,
      scale,
      global_beta_weight_matrix);
}

double same_spin_pair_overlap_scalar(
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    int n_unique_determinants,
    int row,
    int column) {
  const int left_id = std::min(row, column);
  const int right_id = std::max(row, column);
  const auto& pair_evaluation =
      ordered_pair_cache[ordered_spin_pair_storage_index(
          left_id,
          right_id,
          n_unique_determinants)];
  return pair_evaluation.overlap_result.overlap_determinant;
}

double same_spin_pair_total_scalar(
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    int n_unique_determinants,
    int row,
    int column) {
  const int left_id = std::min(row, column);
  const int right_id = std::max(row, column);
  const auto& pair_evaluation =
      ordered_pair_cache[ordered_spin_pair_storage_index(
          left_id,
          right_id,
          n_unique_determinants)];
  return pair_evaluation.total_hamiltonian;
}

template <typename PartnerTileBuilder>
void accumulate_alpha_single_kernel_image_tile(
    const Eigen::MatrixXd& coefficient_matrix,
    const std::vector<int>& alpha_support,
    const std::vector<int>& beta_support,
    int alpha_left_begin,
    int alpha_left_end,
    int alpha_right_begin,
    int alpha_right_end,
    int beta_tile_size,
    PartnerTileBuilder&& build_partner_tile,
    double scale,
    Eigen::MatrixXd* tile_matrix) {
  if (std::abs(scale) <= kContributionTolerance) {
    return;
  }

  const SupportWindow alpha_left_window =
      find_support_window(alpha_support, alpha_left_begin, alpha_left_end);
  const SupportWindow alpha_right_window =
      find_support_window(alpha_support, alpha_right_begin, alpha_right_end);
  if (alpha_left_window.empty() ||
      alpha_right_window.empty() ||
      beta_support.empty()) {
    return;
  }

  // This forms one alpha-side tile of C K_beta C^T. Only the requested
  // alpha rows/columns are materialized; the partner beta support is streamed
  // in bounded tiles so directional overlap-energy terms do not allocate a
  // full unique-spin weight matrix.
  Eigen::MatrixXd partner_tile;
  Eigen::MatrixXd partner_push;
  Eigen::MatrixXd image;
  const int n_beta_support = static_cast<int>(beta_support.size());
  for (int beta_left_begin_local = 0;
       beta_left_begin_local < n_beta_support;
       beta_left_begin_local += beta_tile_size) {
    const SupportWindow beta_left_window{
        beta_left_begin_local,
        std::min(n_beta_support, beta_left_begin_local + beta_tile_size),
    };
    const auto left_block =
        coefficient_matrix.block(
            alpha_left_window.begin,
            beta_left_window.begin,
            alpha_left_window.size(),
            beta_left_window.size());

    for (int beta_right_begin_local = 0;
         beta_right_begin_local < n_beta_support;
         beta_right_begin_local += beta_tile_size) {
      const SupportWindow beta_right_window{
          beta_right_begin_local,
          std::min(n_beta_support, beta_right_begin_local + beta_tile_size),
      };
      const auto right_block =
          coefficient_matrix.block(
              alpha_right_window.begin,
              beta_right_window.begin,
              alpha_right_window.size(),
              beta_right_window.size());

      build_partner_tile(
          beta_support,
          beta_left_window,
          beta_support,
          beta_right_window,
          &partner_tile);
      partner_push.noalias() = left_block * partner_tile;
      image.noalias() = partner_push * right_block.transpose();
      scatter_add_dense_submatrix_to_tile(
          image,
          alpha_support,
          alpha_left_window,
          alpha_left_begin,
          alpha_support,
          alpha_right_window,
          alpha_right_begin,
          scale,
          tile_matrix);
    }
  }
}

template <typename PartnerTileBuilder>
void accumulate_beta_single_kernel_image_tile(
    const Eigen::MatrixXd& coefficient_matrix,
    const std::vector<int>& alpha_support,
    const std::vector<int>& beta_support,
    int beta_left_begin,
    int beta_left_end,
    int beta_right_begin,
    int beta_right_end,
    int alpha_tile_size,
    PartnerTileBuilder&& build_partner_tile,
    double scale,
    Eigen::MatrixXd* tile_matrix) {
  if (std::abs(scale) <= kContributionTolerance) {
    return;
  }

  const SupportWindow beta_left_window =
      find_support_window(beta_support, beta_left_begin, beta_left_end);
  const SupportWindow beta_right_window =
      find_support_window(beta_support, beta_right_begin, beta_right_end);
  if (beta_left_window.empty() ||
      beta_right_window.empty() ||
      alpha_support.empty()) {
    return;
  }

  // Beta-side analogue of the single-kernel alpha tile above. It forms
  // C^T K_alpha C only on the requested beta tile and streams the alpha
  // partner support in bounded chunks.
  Eigen::MatrixXd partner_tile;
  Eigen::MatrixXd partner_push;
  Eigen::MatrixXd image;
  const int n_alpha_support = static_cast<int>(alpha_support.size());
  for (int alpha_left_begin_local = 0;
       alpha_left_begin_local < n_alpha_support;
       alpha_left_begin_local += alpha_tile_size) {
    const SupportWindow alpha_left_window{
        alpha_left_begin_local,
        std::min(n_alpha_support, alpha_left_begin_local + alpha_tile_size),
    };
    const auto left_block =
        coefficient_matrix.block(
            alpha_left_window.begin,
            beta_left_window.begin,
            alpha_left_window.size(),
            beta_left_window.size());

    for (int alpha_right_begin_local = 0;
         alpha_right_begin_local < n_alpha_support;
         alpha_right_begin_local += alpha_tile_size) {
      const SupportWindow alpha_right_window{
          alpha_right_begin_local,
          std::min(n_alpha_support, alpha_right_begin_local + alpha_tile_size),
      };
      const auto right_block =
          coefficient_matrix.block(
              alpha_right_window.begin,
              beta_right_window.begin,
              alpha_right_window.size(),
              beta_right_window.size());

      build_partner_tile(
          alpha_support,
          alpha_left_window,
          alpha_support,
          alpha_right_window,
          &partner_tile);
      partner_push.noalias() = left_block.transpose() * partner_tile;
      image.noalias() = partner_push * right_block;
      scatter_add_dense_submatrix_to_tile(
          image,
          beta_support,
          beta_left_window,
          beta_left_begin,
          beta_support,
          beta_right_window,
          beta_right_begin,
          scale,
          tile_matrix);
    }
  }
}

void accumulate_alpha_accepted_tile_weights(
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies,
    const std::vector<SpinDeterminantPairEvaluation>& partner_pair_cache,
    int n_unique_partner,
    int left_begin,
    int left_end,
    int right_begin,
    int right_end,
    SameSpinAcceptedTileWeights* weights) {
  weights->reset(left_end - left_begin, right_end - right_begin);

  const int partner_tile_size = same_spin_backward_pair_tile_size();
  for (std::size_t state_offset = 0;
       state_offset < selected_states.states.size();
       ++state_offset) {
    const auto& state_coefficients = selected_states.states[state_offset];
    if (!selected_state_has_local_support(state_coefficients)) {
      continue;
    }
    validate_local_state_coefficient_matrix(state_coefficients);

    const double state_weight =
        state_coefficients.normalized_state_weight;
    const double overlap_weight =
        -selected_state_energies[state_offset] * state_weight;
    auto build_partner_tiles = [&](
        const std::vector<int>& support,
        const SupportWindow& left_window,
        const SupportWindow& right_window,
        Eigen::MatrixXd* overlap_tile,
        Eigen::MatrixXd* total_tile) {
      gather_scalar_block_from_support(
          support,
          left_window,
          support,
          right_window,
          [&](int row, int column) {
            return same_spin_pair_overlap_scalar(
                partner_pair_cache,
                n_unique_partner,
                row,
                column);
          },
          overlap_tile);
      gather_scalar_block_from_support(
          support,
          left_window,
          support,
          right_window,
          [&](int row, int column) {
            return same_spin_pair_total_scalar(
                partner_pair_cache,
                n_unique_partner,
                row,
                column);
          },
          total_tile);
    };
    auto consume_images = [&](
        const SupportWindow& left_window,
        const SupportWindow& right_window,
        const Eigen::MatrixXd& overlap_image,
        const Eigen::MatrixXd& total_image) {
      scatter_add_dense_submatrix_to_tile(
          overlap_image,
          state_coefficients.alpha_support,
          left_window,
          left_begin,
          state_coefficients.alpha_support,
          right_window,
          right_begin,
          state_weight,
          &weights->hamiltonian);
      scatter_add_dense_submatrix_to_tile(
          overlap_image,
          state_coefficients.alpha_support,
          left_window,
          left_begin,
          state_coefficients.alpha_support,
          right_window,
          right_begin,
          overlap_weight,
          &weights->overlap);
      scatter_add_dense_submatrix_to_tile(
          total_image,
          state_coefficients.alpha_support,
          left_window,
          left_begin,
          state_coefficients.alpha_support,
          right_window,
          right_begin,
          state_weight,
          &weights->partner_total);
    };

    for_each_alpha_oriented_support_local_image_pair(
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        left_begin,
        left_end,
        right_begin,
        right_end,
        partner_tile_size,
        build_partner_tiles,
        consume_images);
  }
}

void accumulate_beta_accepted_tile_weights(
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies,
    const std::vector<SpinDeterminantPairEvaluation>& partner_pair_cache,
    int n_unique_partner,
    int left_begin,
    int left_end,
    int right_begin,
    int right_end,
    SameSpinAcceptedTileWeights* weights) {
  weights->reset(left_end - left_begin, right_end - right_begin);

  const int partner_tile_size = same_spin_backward_pair_tile_size();
  for (std::size_t state_offset = 0;
       state_offset < selected_states.states.size();
       ++state_offset) {
    const auto& state_coefficients = selected_states.states[state_offset];
    if (!selected_state_has_local_support(state_coefficients)) {
      continue;
    }
    validate_local_state_coefficient_matrix(state_coefficients);

    const double state_weight =
        state_coefficients.normalized_state_weight;
    const double overlap_weight =
        -selected_state_energies[state_offset] * state_weight;
    auto build_partner_tiles = [&](
        const std::vector<int>& support,
        const SupportWindow& left_window,
        const SupportWindow& right_window,
        Eigen::MatrixXd* overlap_tile,
        Eigen::MatrixXd* total_tile) {
      gather_scalar_block_from_support(
          support,
          left_window,
          support,
          right_window,
          [&](int row, int column) {
            return same_spin_pair_overlap_scalar(
                partner_pair_cache,
                n_unique_partner,
                row,
                column);
          },
          overlap_tile);
      gather_scalar_block_from_support(
          support,
          left_window,
          support,
          right_window,
          [&](int row, int column) {
            return same_spin_pair_total_scalar(
                partner_pair_cache,
                n_unique_partner,
                row,
                column);
          },
          total_tile);
    };
    auto consume_images = [&](
        const SupportWindow& left_window,
        const SupportWindow& right_window,
        const Eigen::MatrixXd& overlap_image,
        const Eigen::MatrixXd& total_image) {
      scatter_add_dense_submatrix_to_tile(
          overlap_image,
          state_coefficients.beta_support,
          left_window,
          left_begin,
          state_coefficients.beta_support,
          right_window,
          right_begin,
          state_weight,
          &weights->hamiltonian);
      scatter_add_dense_submatrix_to_tile(
          overlap_image,
          state_coefficients.beta_support,
          left_window,
          left_begin,
          state_coefficients.beta_support,
          right_window,
          right_begin,
          overlap_weight,
          &weights->overlap);
      scatter_add_dense_submatrix_to_tile(
          total_image,
          state_coefficients.beta_support,
          left_window,
          left_begin,
          state_coefficients.beta_support,
          right_window,
          right_begin,
          state_weight,
          &weights->partner_total);
    };

    for_each_beta_oriented_support_local_image_pair(
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        left_begin,
        left_end,
        right_begin,
        right_end,
        partner_tile_size,
        build_partner_tiles,
        consume_images);
  }
}

void accumulate_alpha_local_tile_weights(
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies,
    const std::vector<SpinDeterminantPairEvaluation>& partner_pair_cache,
    const SameSpinDirectionalScalarMatrices& partner_directional_scalars,
    int n_unique_partner,
    int left_begin,
    int left_end,
    int right_begin,
    int right_end,
    SameSpinLocalTileWeights* weights) {
  weights->reset(left_end - left_begin, right_end - right_begin);

  const int partner_tile_size = same_spin_backward_pair_tile_size();
  for (std::size_t state_offset = 0;
       state_offset < selected_states.states.size();
       ++state_offset) {
    const auto& state_coefficients = selected_states.states[state_offset];
    if (!selected_state_has_local_support(state_coefficients)) {
      continue;
    }
    validate_local_state_coefficient_matrix(state_coefficients);

    const double state_weight =
        state_coefficients.normalized_state_weight;
    const double overlap_weight =
        -selected_state_energies[state_offset] * state_weight;

    // First contraction: accepted and directional overlap-determinant partner
    // kernels. They produce W_H/W_S and dW_H/dW_S on the current alpha tile.
    auto build_overlap_tiles = [&](
        const std::vector<int>& support,
        const SupportWindow& left_window,
        const SupportWindow& right_window,
        Eigen::MatrixXd* overlap_tile,
        Eigen::MatrixXd* delta_overlap_tile) {
      gather_scalar_block_from_support(
          support,
          left_window,
          support,
          right_window,
          [&](int row, int column) {
            return same_spin_pair_overlap_scalar(
                partner_pair_cache,
                n_unique_partner,
                row,
                column);
          },
          overlap_tile);
      gather_scalar_block_from_support(
          support,
          left_window,
          support,
          right_window,
          [&](int row, int column) {
            return partner_directional_scalars
                .delta_overlap_determinant_matrix(row, column);
          },
          delta_overlap_tile);
    };
    auto consume_overlap_images = [&](
        const SupportWindow& left_window,
        const SupportWindow& right_window,
        const Eigen::MatrixXd& overlap_image,
        const Eigen::MatrixXd& delta_overlap_image) {
      scatter_add_dense_submatrix_to_tile(
          overlap_image,
          state_coefficients.alpha_support,
          left_window,
          left_begin,
          state_coefficients.alpha_support,
          right_window,
          right_begin,
          state_weight,
          &weights->hamiltonian);
      scatter_add_dense_submatrix_to_tile(
          overlap_image,
          state_coefficients.alpha_support,
          left_window,
          left_begin,
          state_coefficients.alpha_support,
          right_window,
          right_begin,
          overlap_weight,
          &weights->overlap);
      scatter_add_dense_submatrix_to_tile(
          delta_overlap_image,
          state_coefficients.alpha_support,
          left_window,
          left_begin,
          state_coefficients.alpha_support,
          right_window,
          right_begin,
          state_weight,
          &weights->delta_hamiltonian);
      scatter_add_dense_submatrix_to_tile(
          delta_overlap_image,
          state_coefficients.alpha_support,
          left_window,
          left_begin,
          state_coefficients.alpha_support,
          right_window,
          right_begin,
          overlap_weight,
          &weights->delta_overlap);
    };
    for_each_alpha_oriented_support_local_image_pair(
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        left_begin,
        left_end,
        right_begin,
        right_end,
        partner_tile_size,
        build_overlap_tiles,
        consume_overlap_images);

    // Second contraction: accepted and directional same-spin total partner
    // kernels. The directional total is read as regular + singular without
    // materializing an additional dense sum matrix.
    auto build_total_tiles = [&](
        const std::vector<int>& support,
        const SupportWindow& left_window,
        const SupportWindow& right_window,
        Eigen::MatrixXd* total_tile,
        Eigen::MatrixXd* delta_total_tile) {
      gather_scalar_block_from_support(
          support,
          left_window,
          support,
          right_window,
          [&](int row, int column) {
            return same_spin_pair_total_scalar(
                partner_pair_cache,
                n_unique_partner,
                row,
                column);
          },
          total_tile);
      gather_scalar_block_from_support(
          support,
          left_window,
          support,
          right_window,
          [&](int row, int column) {
            return partner_directional_scalars
                       .delta_regular_total_hamiltonian_matrix(row, column) +
                partner_directional_scalars
                    .delta_singular_total_hamiltonian_matrix(row, column);
          },
          delta_total_tile);
    };
    auto consume_total_images = [&](
        const SupportWindow& left_window,
        const SupportWindow& right_window,
        const Eigen::MatrixXd& total_image,
        const Eigen::MatrixXd& delta_total_image) {
      scatter_add_dense_submatrix_to_tile(
          total_image,
          state_coefficients.alpha_support,
          left_window,
          left_begin,
          state_coefficients.alpha_support,
          right_window,
          right_begin,
          state_weight,
          &weights->partner_total);
      scatter_add_dense_submatrix_to_tile(
          delta_total_image,
          state_coefficients.alpha_support,
          left_window,
          left_begin,
          state_coefficients.alpha_support,
          right_window,
          right_begin,
          state_weight,
          &weights->delta_partner_total);
    };
    for_each_alpha_oriented_support_local_image_pair(
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        left_begin,
        left_end,
        right_begin,
        right_end,
        partner_tile_size,
        build_total_tiles,
        consume_total_images);
  }
}

void accumulate_beta_local_tile_weights(
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies,
    const std::vector<SpinDeterminantPairEvaluation>& partner_pair_cache,
    const SameSpinDirectionalScalarMatrices& partner_directional_scalars,
    int n_unique_partner,
    int left_begin,
    int left_end,
    int right_begin,
    int right_end,
    SameSpinLocalTileWeights* weights) {
  weights->reset(left_end - left_begin, right_end - right_begin);

  const int partner_tile_size = same_spin_backward_pair_tile_size();
  for (std::size_t state_offset = 0;
       state_offset < selected_states.states.size();
       ++state_offset) {
    const auto& state_coefficients = selected_states.states[state_offset];
    if (!selected_state_has_local_support(state_coefficients)) {
      continue;
    }
    validate_local_state_coefficient_matrix(state_coefficients);

    const double state_weight =
        state_coefficients.normalized_state_weight;
    const double overlap_weight =
        -selected_state_energies[state_offset] * state_weight;

    auto build_overlap_tiles = [&](
        const std::vector<int>& support,
        const SupportWindow& left_window,
        const SupportWindow& right_window,
        Eigen::MatrixXd* overlap_tile,
        Eigen::MatrixXd* delta_overlap_tile) {
      gather_scalar_block_from_support(
          support,
          left_window,
          support,
          right_window,
          [&](int row, int column) {
            return same_spin_pair_overlap_scalar(
                partner_pair_cache,
                n_unique_partner,
                row,
                column);
          },
          overlap_tile);
      gather_scalar_block_from_support(
          support,
          left_window,
          support,
          right_window,
          [&](int row, int column) {
            return partner_directional_scalars
                .delta_overlap_determinant_matrix(row, column);
          },
          delta_overlap_tile);
    };
    auto consume_overlap_images = [&](
        const SupportWindow& left_window,
        const SupportWindow& right_window,
        const Eigen::MatrixXd& overlap_image,
        const Eigen::MatrixXd& delta_overlap_image) {
      scatter_add_dense_submatrix_to_tile(
          overlap_image,
          state_coefficients.beta_support,
          left_window,
          left_begin,
          state_coefficients.beta_support,
          right_window,
          right_begin,
          state_weight,
          &weights->hamiltonian);
      scatter_add_dense_submatrix_to_tile(
          overlap_image,
          state_coefficients.beta_support,
          left_window,
          left_begin,
          state_coefficients.beta_support,
          right_window,
          right_begin,
          overlap_weight,
          &weights->overlap);
      scatter_add_dense_submatrix_to_tile(
          delta_overlap_image,
          state_coefficients.beta_support,
          left_window,
          left_begin,
          state_coefficients.beta_support,
          right_window,
          right_begin,
          state_weight,
          &weights->delta_hamiltonian);
      scatter_add_dense_submatrix_to_tile(
          delta_overlap_image,
          state_coefficients.beta_support,
          left_window,
          left_begin,
          state_coefficients.beta_support,
          right_window,
          right_begin,
          overlap_weight,
          &weights->delta_overlap);
    };
    for_each_beta_oriented_support_local_image_pair(
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        left_begin,
        left_end,
        right_begin,
        right_end,
        partner_tile_size,
        build_overlap_tiles,
        consume_overlap_images);

    auto build_total_tiles = [&](
        const std::vector<int>& support,
        const SupportWindow& left_window,
        const SupportWindow& right_window,
        Eigen::MatrixXd* total_tile,
        Eigen::MatrixXd* delta_total_tile) {
      gather_scalar_block_from_support(
          support,
          left_window,
          support,
          right_window,
          [&](int row, int column) {
            return same_spin_pair_total_scalar(
                partner_pair_cache,
                n_unique_partner,
                row,
                column);
          },
          total_tile);
      gather_scalar_block_from_support(
          support,
          left_window,
          support,
          right_window,
          [&](int row, int column) {
            return partner_directional_scalars
                       .delta_regular_total_hamiltonian_matrix(row, column) +
                partner_directional_scalars
                    .delta_singular_total_hamiltonian_matrix(row, column);
          },
          delta_total_tile);
    };
    auto consume_total_images = [&](
        const SupportWindow& left_window,
        const SupportWindow& right_window,
        const Eigen::MatrixXd& total_image,
        const Eigen::MatrixXd& delta_total_image) {
      scatter_add_dense_submatrix_to_tile(
          total_image,
          state_coefficients.beta_support,
          left_window,
          left_begin,
          state_coefficients.beta_support,
          right_window,
          right_begin,
          state_weight,
          &weights->partner_total);
      scatter_add_dense_submatrix_to_tile(
          delta_total_image,
          state_coefficients.beta_support,
          left_window,
          left_begin,
          state_coefficients.beta_support,
          right_window,
          right_begin,
          state_weight,
          &weights->delta_partner_total);
    };
    for_each_beta_oriented_support_local_image_pair(
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        left_begin,
        left_end,
        right_begin,
        right_end,
        partner_tile_size,
        build_total_tiles,
        consume_total_images);
  }
}

void accumulate_alpha_directional_tile_weights(
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    const std::vector<double>& selected_state_energies,
    const std::vector<double>& directional_selected_state_energies,
    const std::vector<SpinDeterminantPairEvaluation>& partner_pair_cache,
    int n_unique_partner,
    int left_begin,
    int left_end,
    int right_begin,
    int right_end,
    SameSpinAcceptedTileWeights* weights) {
  weights->reset(left_end - left_begin, right_end - right_begin);

  const int partner_tile_size = same_spin_backward_pair_tile_size();
  for (std::size_t state_offset = 0;
       state_offset < selected_states.states.size();
       ++state_offset) {
    const auto& state_coefficients = selected_states.states[state_offset];
    const auto& directional_state_coefficients =
        directional_selected_states.states[state_offset];
    if (!selected_state_has_local_support(state_coefficients)) {
      continue;
    }
    validate_local_state_coefficient_matrix(state_coefficients);

    const double state_weight =
        state_coefficients.normalized_state_weight;
    const double state_energy = selected_state_energies[state_offset];
    const double directional_state_energy =
        directional_selected_state_energies[state_offset];

    auto build_overlap_tile = [&](
        const std::vector<int>& left_support,
        const SupportWindow& left_window,
        const std::vector<int>& right_support,
        const SupportWindow& right_window,
        Eigen::MatrixXd* overlap_tile) {
      gather_scalar_block_from_support(
          left_support,
          left_window,
          right_support,
          right_window,
          [&](int row, int column) {
            return same_spin_pair_overlap_scalar(
                partner_pair_cache,
                n_unique_partner,
                row,
                column);
          },
          overlap_tile);
    };
    accumulate_alpha_single_kernel_image_tile(
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        left_begin,
        left_end,
        right_begin,
        right_end,
        partner_tile_size,
        build_overlap_tile,
        -state_weight * directional_state_energy,
        &weights->overlap);

    if (!selected_state_has_local_support(directional_state_coefficients)) {
      continue;
    }
    validate_local_state_coefficient_matrix(directional_state_coefficients);

    auto build_partner_tiles = [&](
        const std::vector<int>& left_support,
        const SupportWindow& left_window,
        const std::vector<int>& right_support,
        const SupportWindow& right_window,
        Eigen::MatrixXd* overlap_tile,
        Eigen::MatrixXd* total_tile) {
      gather_scalar_block_from_support(
          left_support,
          left_window,
          right_support,
          right_window,
          [&](int row, int column) {
            return same_spin_pair_overlap_scalar(
                partner_pair_cache,
                n_unique_partner,
                row,
                column);
          },
          overlap_tile);
      gather_scalar_block_from_support(
          left_support,
          left_window,
          right_support,
          right_window,
          [&](int row, int column) {
            return same_spin_pair_total_scalar(
                partner_pair_cache,
                n_unique_partner,
                row,
                column);
          },
          total_tile);
    };

    // Directional selected-state response at fixed partner kernels:
    // d(C K C^T) = dC K C^T + C K dC^T.
    auto consume_left_directional_images = [&](
        const SupportWindow& left_window,
        const SupportWindow& right_window,
        const Eigen::MatrixXd& overlap_image,
        const Eigen::MatrixXd& total_image) {
      scatter_add_dense_submatrix_to_tile(
          overlap_image,
          directional_state_coefficients.alpha_support,
          left_window,
          left_begin,
          state_coefficients.alpha_support,
          right_window,
          right_begin,
          state_weight,
          &weights->hamiltonian);
      scatter_add_dense_submatrix_to_tile(
          overlap_image,
          directional_state_coefficients.alpha_support,
          left_window,
          left_begin,
          state_coefficients.alpha_support,
          right_window,
          right_begin,
          -state_weight * state_energy,
          &weights->overlap);
      scatter_add_dense_submatrix_to_tile(
          total_image,
          directional_state_coefficients.alpha_support,
          left_window,
          left_begin,
          state_coefficients.alpha_support,
          right_window,
          right_begin,
          state_weight,
          &weights->partner_total);
    };
    for_each_alpha_oriented_support_local_mixed_image_pair(
        directional_state_coefficients.local_coefficient_matrix,
        directional_state_coefficients.alpha_support,
        directional_state_coefficients.beta_support,
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        left_begin,
        left_end,
        right_begin,
        right_end,
        partner_tile_size,
        build_partner_tiles,
        consume_left_directional_images);

    auto consume_right_directional_images = [&](
        const SupportWindow& left_window,
        const SupportWindow& right_window,
        const Eigen::MatrixXd& overlap_image,
        const Eigen::MatrixXd& total_image) {
      scatter_add_dense_submatrix_to_tile(
          overlap_image,
          state_coefficients.alpha_support,
          left_window,
          left_begin,
          directional_state_coefficients.alpha_support,
          right_window,
          right_begin,
          state_weight,
          &weights->hamiltonian);
      scatter_add_dense_submatrix_to_tile(
          overlap_image,
          state_coefficients.alpha_support,
          left_window,
          left_begin,
          directional_state_coefficients.alpha_support,
          right_window,
          right_begin,
          -state_weight * state_energy,
          &weights->overlap);
      scatter_add_dense_submatrix_to_tile(
          total_image,
          state_coefficients.alpha_support,
          left_window,
          left_begin,
          directional_state_coefficients.alpha_support,
          right_window,
          right_begin,
          state_weight,
          &weights->partner_total);
    };
    for_each_alpha_oriented_support_local_mixed_image_pair(
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        directional_state_coefficients.local_coefficient_matrix,
        directional_state_coefficients.alpha_support,
        directional_state_coefficients.beta_support,
        left_begin,
        left_end,
        right_begin,
        right_end,
        partner_tile_size,
        build_partner_tiles,
        consume_right_directional_images);
  }
}

void accumulate_beta_directional_tile_weights(
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    const std::vector<double>& selected_state_energies,
    const std::vector<double>& directional_selected_state_energies,
    const std::vector<SpinDeterminantPairEvaluation>& partner_pair_cache,
    int n_unique_partner,
    int left_begin,
    int left_end,
    int right_begin,
    int right_end,
    SameSpinAcceptedTileWeights* weights) {
  weights->reset(left_end - left_begin, right_end - right_begin);

  const int partner_tile_size = same_spin_backward_pair_tile_size();
  for (std::size_t state_offset = 0;
       state_offset < selected_states.states.size();
       ++state_offset) {
    const auto& state_coefficients = selected_states.states[state_offset];
    const auto& directional_state_coefficients =
        directional_selected_states.states[state_offset];
    if (!selected_state_has_local_support(state_coefficients)) {
      continue;
    }
    validate_local_state_coefficient_matrix(state_coefficients);

    const double state_weight =
        state_coefficients.normalized_state_weight;
    const double state_energy = selected_state_energies[state_offset];
    const double directional_state_energy =
        directional_selected_state_energies[state_offset];

    auto build_overlap_tile = [&](
        const std::vector<int>& left_support,
        const SupportWindow& left_window,
        const std::vector<int>& right_support,
        const SupportWindow& right_window,
        Eigen::MatrixXd* overlap_tile) {
      gather_scalar_block_from_support(
          left_support,
          left_window,
          right_support,
          right_window,
          [&](int row, int column) {
            return same_spin_pair_overlap_scalar(
                partner_pair_cache,
                n_unique_partner,
                row,
                column);
          },
          overlap_tile);
    };
    accumulate_beta_single_kernel_image_tile(
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        left_begin,
        left_end,
        right_begin,
        right_end,
        partner_tile_size,
        build_overlap_tile,
        -state_weight * directional_state_energy,
        &weights->overlap);

    if (!selected_state_has_local_support(directional_state_coefficients)) {
      continue;
    }
    validate_local_state_coefficient_matrix(directional_state_coefficients);

    auto build_partner_tiles = [&](
        const std::vector<int>& left_support,
        const SupportWindow& left_window,
        const std::vector<int>& right_support,
        const SupportWindow& right_window,
        Eigen::MatrixXd* overlap_tile,
        Eigen::MatrixXd* total_tile) {
      gather_scalar_block_from_support(
          left_support,
          left_window,
          right_support,
          right_window,
          [&](int row, int column) {
            return same_spin_pair_overlap_scalar(
                partner_pair_cache,
                n_unique_partner,
                row,
                column);
          },
          overlap_tile);
      gather_scalar_block_from_support(
          left_support,
          left_window,
          right_support,
          right_window,
          [&](int row, int column) {
            return same_spin_pair_total_scalar(
                partner_pair_cache,
                n_unique_partner,
                row,
                column);
          },
          total_tile);
    };

    auto consume_left_directional_images = [&](
        const SupportWindow& left_window,
        const SupportWindow& right_window,
        const Eigen::MatrixXd& overlap_image,
        const Eigen::MatrixXd& total_image) {
      scatter_add_dense_submatrix_to_tile(
          overlap_image,
          directional_state_coefficients.beta_support,
          left_window,
          left_begin,
          state_coefficients.beta_support,
          right_window,
          right_begin,
          state_weight,
          &weights->hamiltonian);
      scatter_add_dense_submatrix_to_tile(
          overlap_image,
          directional_state_coefficients.beta_support,
          left_window,
          left_begin,
          state_coefficients.beta_support,
          right_window,
          right_begin,
          -state_weight * state_energy,
          &weights->overlap);
      scatter_add_dense_submatrix_to_tile(
          total_image,
          directional_state_coefficients.beta_support,
          left_window,
          left_begin,
          state_coefficients.beta_support,
          right_window,
          right_begin,
          state_weight,
          &weights->partner_total);
    };
    for_each_beta_oriented_support_local_mixed_image_pair(
        directional_state_coefficients.local_coefficient_matrix,
        directional_state_coefficients.alpha_support,
        directional_state_coefficients.beta_support,
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        left_begin,
        left_end,
        right_begin,
        right_end,
        partner_tile_size,
        build_partner_tiles,
        consume_left_directional_images);

    auto consume_right_directional_images = [&](
        const SupportWindow& left_window,
        const SupportWindow& right_window,
        const Eigen::MatrixXd& overlap_image,
        const Eigen::MatrixXd& total_image) {
      scatter_add_dense_submatrix_to_tile(
          overlap_image,
          state_coefficients.beta_support,
          left_window,
          left_begin,
          directional_state_coefficients.beta_support,
          right_window,
          right_begin,
          state_weight,
          &weights->hamiltonian);
      scatter_add_dense_submatrix_to_tile(
          overlap_image,
          state_coefficients.beta_support,
          left_window,
          left_begin,
          directional_state_coefficients.beta_support,
          right_window,
          right_begin,
          -state_weight * state_energy,
          &weights->overlap);
      scatter_add_dense_submatrix_to_tile(
          total_image,
          state_coefficients.beta_support,
          left_window,
          left_begin,
          directional_state_coefficients.beta_support,
          right_window,
          right_begin,
          state_weight,
          &weights->partner_total);
    };
    for_each_beta_oriented_support_local_mixed_image_pair(
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        directional_state_coefficients.local_coefficient_matrix,
        directional_state_coefficients.alpha_support,
        directional_state_coefficients.beta_support,
        left_begin,
        left_end,
        right_begin,
        right_end,
        partner_tile_size,
        build_partner_tiles,
        consume_right_directional_images);
  }
}

SameSpinPairScalarMatrices build_pair_scalar_matrices(
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    int n_unique_determinants) {
  const std::size_t expected_size = square_storage_size(n_unique_determinants);
  if (ordered_pair_cache.size() != expected_size) {
    throw std::invalid_argument(
        "ordered same-spin pair cache size does not match unique-spin dimensions");
  }

  SameSpinPairScalarMatrices matrices;
  matrices.overlap_determinant_matrix =
      Eigen::MatrixXd::Zero(n_unique_determinants, n_unique_determinants);
  matrices.regular_total_hamiltonian_matrix =
      Eigen::MatrixXd::Zero(n_unique_determinants, n_unique_determinants);
  matrices.singular_total_hamiltonian_matrix =
      Eigen::MatrixXd::Zero(n_unique_determinants, n_unique_determinants);

  for (int left_id = 0; left_id < n_unique_determinants; ++left_id) {
    for (int right_id = left_id; right_id < n_unique_determinants; ++right_id) {
      const auto& pair_evaluation =
          ordered_pair_cache[ordered_spin_pair_storage_index(
              left_id,
              right_id,
              n_unique_determinants)];
      set_symmetric_matrix_entry(
          &matrices.overlap_determinant_matrix,
          left_id,
          right_id,
          pair_evaluation.overlap_result.overlap_determinant);
      if (pair_evaluation.overlap_result.nullity == 0) {
        set_symmetric_matrix_entry(
            &matrices.regular_total_hamiltonian_matrix,
            left_id,
            right_id,
            pair_evaluation.total_hamiltonian);
      } else {
        set_symmetric_matrix_entry(
            &matrices.singular_total_hamiltonian_matrix,
            left_id,
            right_id,
            pair_evaluation.total_hamiltonian);
      }
    }
  }

  return matrices;
}

SingleChannelSameSpinWeightMatrices
build_dense_same_spin_weight_matrices_from_partner_kernels(
    const SelectedStateDeterminantMatrices& selected_states,
    const Eigen::MatrixXd& alpha_partner_kernel_matrix,
    const Eigen::MatrixXd& beta_partner_kernel_matrix,
    const std::vector<double>& per_state_scales,
    bool close_shell_same_spin) {
  if (per_state_scales.size() != selected_states.states.size()) {
    throw std::invalid_argument(
        "per_state_scales must align with selected_states.states");
  }

  SingleChannelSameSpinWeightMatrices weight_matrices;
  weight_matrices.alpha_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.beta_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);

  Eigen::MatrixXd beta_push(
      selected_states.n_unique_alpha,
      selected_states.n_unique_beta);
  Eigen::MatrixXd alpha_push(
      selected_states.n_unique_alpha,
      selected_states.n_unique_beta);
  Eigen::MatrixXd alpha_image(
      selected_states.n_unique_alpha,
      selected_states.n_unique_alpha);
  Eigen::MatrixXd beta_image(
      selected_states.n_unique_beta,
      selected_states.n_unique_beta);
  Eigen::MatrixXd coefficient_matrix_dense(
      selected_states.n_unique_alpha,
      selected_states.n_unique_beta);
  for (std::size_t state_offset = 0;
       state_offset < selected_states.states.size();
       ++state_offset) {
    const double scale = per_state_scales[state_offset];
    if (std::abs(scale) <= kContributionTolerance) {
      continue;
    }

    validate_state_coefficient_matrix(
        selected_states.states[state_offset],
        selected_states.n_unique_alpha,
        selected_states.n_unique_beta);
    const auto& state_coefficients = selected_states.states[state_offset];
    if (selected_state_has_close_shell_diagonal(
            state_coefficients,
            selected_states.n_unique_alpha,
            selected_states.n_unique_beta)) {
      accumulate_diagonal_kernel_image_global(
          state_coefficients.diagonal_coefficients,
          beta_partner_kernel_matrix,
          scale,
          &weight_matrices.alpha_weight_matrix);
      if (!close_shell_same_spin) {
        accumulate_diagonal_kernel_image_global(
            state_coefficients.diagonal_coefficients,
            alpha_partner_kernel_matrix,
            scale,
            &weight_matrices.beta_weight_matrix);
      }
      continue;
    }
    coefficient_matrix_dense = state_coefficients.coefficient_matrix;

    multiply_right_symmetric(
        coefficient_matrix_dense,
        beta_partner_kernel_matrix,
        &beta_push);
    alpha_image.noalias() = beta_push * coefficient_matrix_dense.transpose();
    weight_matrices.alpha_weight_matrix.noalias() += scale * alpha_image;

    if (close_shell_same_spin) {
      continue;
    }

    multiply_left_symmetric(
        alpha_partner_kernel_matrix,
        coefficient_matrix_dense,
        &alpha_push);
    beta_image.noalias() = coefficient_matrix_dense.transpose() * alpha_push;
    weight_matrices.beta_weight_matrix.noalias() += scale * beta_image;
  }

  if (close_shell_same_spin) {
    weight_matrices.beta_weight_matrix =
        weight_matrices.alpha_weight_matrix;
  }
  return weight_matrices;
}

SingleChannelSameSpinWeightMatrices
build_support_sparse_same_spin_weight_matrices_from_partner_kernels(
    const SelectedStateDeterminantMatrices& selected_states,
    const Eigen::MatrixXd& alpha_partner_kernel_matrix,
    const Eigen::MatrixXd& beta_partner_kernel_matrix,
    const std::vector<double>& per_state_scales,
    bool close_shell_same_spin) {
  if (per_state_scales.size() != selected_states.states.size()) {
    throw std::invalid_argument(
        "per_state_scales must align with selected_states.states");
  }

  SingleChannelSameSpinWeightMatrices weight_matrices;
  weight_matrices.alpha_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.beta_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);

  Eigen::MatrixXd alpha_partner_subblock;
  Eigen::MatrixXd beta_partner_subblock;
  Eigen::MatrixXd beta_push;
  Eigen::MatrixXd alpha_push;
  Eigen::MatrixXd alpha_image;
  Eigen::MatrixXd beta_image;
  for (std::size_t state_offset = 0;
       state_offset < selected_states.states.size();
       ++state_offset) {
    const auto& state_coefficients = selected_states.states[state_offset];
    const double scale = per_state_scales[state_offset];
    if (std::abs(scale) <= kContributionTolerance ||
        !selected_state_has_local_support(state_coefficients)) {
      continue;
    }

    gather_dense_submatrix(
        alpha_partner_kernel_matrix,
        state_coefficients.alpha_support,
        state_coefficients.alpha_support,
        &alpha_partner_subblock);
    gather_dense_submatrix(
        beta_partner_kernel_matrix,
        state_coefficients.beta_support,
        state_coefficients.beta_support,
        &beta_partner_subblock);
    accumulate_selected_state_alpha_image(
        state_coefficients,
        beta_partner_subblock,
        scale,
        &weight_matrices.alpha_weight_matrix,
        &beta_push,
        &alpha_image);
    if (close_shell_same_spin) {
      continue;
    }
    accumulate_selected_state_beta_image(
        state_coefficients,
        alpha_partner_subblock,
        scale,
        &weight_matrices.beta_weight_matrix,
        &alpha_push,
        &beta_image);
  }

  if (close_shell_same_spin) {
    weight_matrices.beta_weight_matrix =
        weight_matrices.alpha_weight_matrix;
  }
  return weight_matrices;
}

SingleChannelSameSpinWeightMatrices
build_same_spin_weight_matrices_from_partner_kernels(
    const SelectedStateDeterminantMatrices& selected_states,
    const Eigen::MatrixXd& alpha_partner_kernel_matrix,
    const Eigen::MatrixXd& beta_partner_kernel_matrix,
    const std::vector<double>& per_state_scales,
    bool close_shell_same_spin) {
  if (should_use_support_sparse_selected_state_contractions(selected_states)) {
    return build_support_sparse_same_spin_weight_matrices_from_partner_kernels(
        selected_states,
        alpha_partner_kernel_matrix,
        beta_partner_kernel_matrix,
        per_state_scales,
        close_shell_same_spin);
  }
  return build_dense_same_spin_weight_matrices_from_partner_kernels(
      selected_states,
      alpha_partner_kernel_matrix,
      beta_partner_kernel_matrix,
      per_state_scales,
      close_shell_same_spin);
}



}  // namespace xmvb::vb::detail
