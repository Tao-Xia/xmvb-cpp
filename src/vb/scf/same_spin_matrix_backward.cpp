#include "vb/scf/same_spin_matrix_backward.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <cblas.h>

#include "vb/matrices/spin_pair_utils.hpp"
#include "vb/matrices/cofactor_differential.hpp"
#include "vb/matrices/support_local_contraction_kernels.hpp"
#include "vb/matrices/two_electron_indexer.hpp"
#include "vb/orbital/active_space_two_electron_utils.hpp"

namespace xmvb::vb {

namespace {

constexpr double kContributionTolerance = 1.0e-15;
constexpr int kSameSpinBackwardPairTileSize = 64;
constexpr std::size_t kSameSpinAcceptedTileMinDenseBytes =
    256ull * 1024ull * 1024ull;

std::size_t square_storage_size(int dimension) {
  return (dimension) * (dimension);
}

int same_spin_backward_pair_tile_size() {
  return kSameSpinBackwardPairTileSize;
}

std::size_t same_spin_accepted_tile_min_dense_bytes() {
  return kSameSpinAcceptedTileMinDenseBytes;
}

bool should_use_same_spin_accepted_tile_backward(
    const SelectedStateDeterminantMatrices& selected_states) {
  const std::size_t alpha_size =
      static_cast<std::size_t>(selected_states.n_unique_alpha);
  const std::size_t beta_size =
      static_cast<std::size_t>(selected_states.n_unique_beta);
  const std::size_t dense_weight_bytes =
      4ull * sizeof(double) *
      (alpha_size * alpha_size + beta_size * beta_size);
  return dense_weight_bytes >= same_spin_accepted_tile_min_dense_bytes();
}

bool should_use_same_spin_local_tile_backward(
    const SelectedStateDeterminantMatrices& selected_states) {
  return should_use_same_spin_accepted_tile_backward(selected_states);
}

bool should_use_same_spin_directional_tile_backward(
    const SelectedStateDeterminantMatrices& selected_states) {
  return should_use_same_spin_accepted_tile_backward(selected_states);
}

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

struct SameSpinExactWeightMatrices {
  Eigen::MatrixXd alpha_hamiltonian_weight_matrix;
  Eigen::MatrixXd alpha_overlap_weight_matrix;
  Eigen::MatrixXd alpha_partner_total_transfer_matrix;
  Eigen::MatrixXd alpha_singular_partner_transfer_matrix;
  Eigen::MatrixXd beta_hamiltonian_weight_matrix;
  Eigen::MatrixXd beta_overlap_weight_matrix;
  Eigen::MatrixXd beta_partner_total_transfer_matrix;
  Eigen::MatrixXd beta_singular_partner_transfer_matrix;
};

struct SameSpinPairScalarMatrices {
  Eigen::MatrixXd overlap_determinant_matrix;
  Eigen::MatrixXd regular_total_hamiltonian_matrix;
  Eigen::MatrixXd singular_total_hamiltonian_matrix;
};

struct SingleChannelSameSpinWeightMatrices {
  Eigen::MatrixXd alpha_weight_matrix;
  Eigen::MatrixXd beta_weight_matrix;
};

struct SameSpinLocalResponseWeightMatrices {
  Eigen::MatrixXd alpha_delta_hamiltonian_weight_matrix;
  Eigen::MatrixXd alpha_delta_overlap_weight_matrix;
  Eigen::MatrixXd alpha_delta_partner_total_transfer_matrix;
  Eigen::MatrixXd beta_delta_hamiltonian_weight_matrix;
  Eigen::MatrixXd beta_delta_overlap_weight_matrix;
  Eigen::MatrixXd beta_delta_partner_total_transfer_matrix;
};

struct SameSpinAcceptedTileWeights {
  Eigen::MatrixXd hamiltonian;
  Eigen::MatrixXd overlap;
  Eigen::MatrixXd partner_total;

  void reset(int n_rows, int n_cols) {
    hamiltonian.setZero(n_rows, n_cols);
    overlap.setZero(n_rows, n_cols);
    partner_total.setZero(n_rows, n_cols);
  }
};

struct SameSpinLocalTileWeights {
  Eigen::MatrixXd hamiltonian;
  Eigen::MatrixXd overlap;
  Eigen::MatrixXd partner_total;
  Eigen::MatrixXd delta_hamiltonian;
  Eigen::MatrixXd delta_overlap;
  Eigen::MatrixXd delta_partner_total;

  void reset(int n_rows, int n_cols) {
    hamiltonian.setZero(n_rows, n_cols);
    overlap.setZero(n_rows, n_cols);
    partner_total.setZero(n_rows, n_cols);
    delta_hamiltonian.setZero(n_rows, n_cols);
    delta_overlap.setZero(n_rows, n_cols);
    delta_partner_total.setZero(n_rows, n_cols);
  }
};

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

void accumulate_one_electron_gradient_contribution_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& cofactor_1st,
    double weight,
    Eigen::MatrixXd* active_one_electron_gradient) {
  if (active_one_electron_gradient == nullptr) {
    throw std::invalid_argument("active_one_electron_gradient must not be null");
  }
  if (std::abs(weight) <= kContributionTolerance) {
    return;
  }

  for (int left_column = 0; left_column < static_cast<int>(occ_L.size()); ++left_column) {
    const int orbital_index_left = occ_L[left_column];
    for (int right_row = 0; right_row < static_cast<int>(occ_R.size()); ++right_row) {
      const int orbital_index_right = occ_R[right_row];
      (*active_one_electron_gradient)(orbital_index_right, orbital_index_left) +=
          weight * cofactor_1st(right_row, left_column);
    }
  }
}

void accumulate_overlap_block_gradient_contribution_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& overlap_block_gradient,
    double weight,
    int n_active_orbitals,
    std::vector<double>* active_orbital_overlap_gradient) {
  if (active_orbital_overlap_gradient == nullptr) {
    throw std::invalid_argument("active_orbital_overlap_gradient must not be null");
  }
  if (std::abs(weight) <= kContributionTolerance) {
    return;
  }
  if (overlap_block_gradient.rows() != static_cast<int>(occ_R.size()) ||
      overlap_block_gradient.cols() != static_cast<int>(occ_L.size())) {
    throw std::invalid_argument(
        "overlap_block_gradient dimensions do not match occupied lists");
  }

  for (int left_column = 0; left_column < static_cast<int>(occ_L.size()); ++left_column) {
    const int orbital_index_left = occ_L[left_column];
    for (int right_row = 0; right_row < static_cast<int>(occ_R.size()); ++right_row) {
      const int orbital_index_right = occ_R[right_row];
      (*active_orbital_overlap_gradient)[orbital_index_left *
                                             n_active_orbitals +
                                         orbital_index_right] +=
          weight * overlap_block_gradient(right_row, left_column);
    }
  }
}

void accumulate_deleted_minor_same_spin_two_electron_gradient_contribution_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const CofactorDifferential& cofactor,
    double weight,
    std::vector<double>* packed_active_two_electron_gradient) {
  if (packed_active_two_electron_gradient == nullptr) {
    throw std::invalid_argument("packed_active_two_electron_gradient must not be null");
  }
  if (std::abs(weight) <= kContributionTolerance) {
    return;
  }

  const int n_electrons = static_cast<int>(occ_L.size());
  if (n_electrons < 2) {
    return;
  }

  const Eigen::MatrixXd second = cofactor.second();
  for (int left_first = 0; left_first < n_electrons - 1; ++left_first) {
    const int orbital_index_left_first = occ_L[left_first];
    for (int right_first = 0; right_first < n_electrons - 1; ++right_first) {
      const int orbital_index_right_first = occ_R[right_first];
      for (int left_second = left_first + 1; left_second < n_electrons; ++left_second) {
        const int orbital_index_left_second = occ_L[left_second];
        for (int right_second = right_first + 1; right_second < n_electrons; ++right_second) {
          const int orbital_index_right_second = occ_R[right_second];
          const double second_order_cofactor =
              weight * second(right_second*(right_second-1)/2+right_first,
                     left_second*(left_second-1)/2+left_first);
          if (std::abs(second_order_cofactor) <= kContributionTolerance) {
            continue;
          }

          const int direct_index = TwoElectronIndexer::two_electron_storage_index(
              orbital_index_right_first,
              orbital_index_left_first,
              orbital_index_right_second,
              orbital_index_left_second);
          const int exchange_index = TwoElectronIndexer::two_electron_storage_index(
              orbital_index_right_first,
              orbital_index_left_second,
              orbital_index_right_second,
              orbital_index_left_first);
          (*packed_active_two_electron_gradient)[direct_index] +=
              second_order_cofactor;
          (*packed_active_two_electron_gradient)[exchange_index] -=
              second_order_cofactor;
        }
      }
    }
  }
}

Eigen::MatrixXd build_spin_one_electron_block_matrix_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act) {
  const int n_electrons = static_cast<int>(occ_L.size());
  Eigen::MatrixXd one_electron_block(n_electrons, n_electrons);
  for (int left_column = 0; left_column < n_electrons; ++left_column) {
    const int orbital_index_left = occ_L[left_column];
    for (int right_row = 0; right_row < n_electrons; ++right_row) {
      const int orbital_index_right = occ_R[right_row];
      one_electron_block(right_row, left_column) =
          h1e_act(orbital_index_right, orbital_index_left);
    }
  }
  return one_electron_block;
}

Eigen::MatrixXd build_local_overlap_direction_matrix(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& delta_ao_overlap_matrix,
    int n_active_orbitals) {
  return build_overlap_submatrix(
      occ_L,
      occ_R,
      delta_ao_overlap_matrix,
      n_active_orbitals);
}

SameSpinPolynomialDirectionalPairData build_polynomial_spin_directional_data(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::Ref<const Eigen::MatrixXd>& active_one_electron_matrix,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const SpinDeterminantPairEvaluation& pair_evaluation,
    int n_active_orbitals,
    const std::vector<double>& delta_ao_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals,
    bool need_overlap_gradient = true) {
  SameSpinPolynomialDirectionalPairData result;
  const CofactorDifferential& cofactor =
      cached_cofactor_differential(pair_evaluation);
  const Eigen::MatrixXd ds = build_local_overlap_direction_matrix(
      occ_L, occ_R, delta_ao_overlap_matrix, n_active_orbitals);
  const Eigen::MatrixXd dh = build_spin_one_electron_block_matrix_local(
      occ_L, occ_R, Eigen::Map<const Eigen::MatrixXd>(
          delta_active_one_electron_matrix.data(), n_active_orbitals, n_active_orbitals));
  const Eigen::MatrixXd dg = build_spin_antisymmetrized_interaction_direction(
      occ_L, occ_R, delta_packed_active_two_electron_integrals);
  result.cofactor_1st = cofactor.value();
  result.delta_cofactor_1st = cofactor.first(ds);
  result.delta_overlap_determinant = (result.cofactor_1st.cwiseProduct(ds)).sum();
  result.delta_total_hamiltonian =
      (dh.cwiseProduct(result.cofactor_1st)).sum() +
      (dg.cwiseProduct(cofactor.second())).sum() +
      (pair_evaluation.same_spin_overlap_hamiltonian_gradient.cwiseProduct(ds)).sum();
  if (need_overlap_gradient)
    result.delta_same_spin_overlap_hamiltonian_gradient =
        cofactor.mixed(ds, pair_evaluation.same_spin_one_electron_block) +
        cofactor.first(dh) +
        cofactor.second_contraction_gradient_direction(
            ds, pair_evaluation.same_spin_antisymmetrized_interaction, dg);
  return result;
}

void accumulate_directional_one_electron_gradient_contribution_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& cofactor_1st,
    const Eigen::MatrixXd& delta_cofactor_1st,
    double weight,
    double delta_weight,
    Eigen::MatrixXd* active_one_electron_gradient) {
  if (active_one_electron_gradient == nullptr) {
    throw std::invalid_argument("active_one_electron_gradient must not be null");
  }
  if (std::abs(weight) <= kContributionTolerance &&
      std::abs(delta_weight) <= kContributionTolerance) {
    return;
  }

  for (int left_column = 0; left_column < static_cast<int>(occ_L.size()); ++left_column) {
    const int orbital_index_left = occ_L[left_column];
    for (int right_row = 0; right_row < static_cast<int>(occ_R.size()); ++right_row) {
      const int orbital_index_right = occ_R[right_row];
      (*active_one_electron_gradient)(orbital_index_right, orbital_index_left) +=
          delta_weight * cofactor_1st(right_row, left_column) +
          weight * delta_cofactor_1st(right_row, left_column);
    }
  }
}

void accumulate_directional_deleted_minor_same_spin_two_electron_gradient_contribution_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const CofactorDifferential& cofactor,
    const std::vector<double>& delta_ao_overlap_matrix,
    int n_active_orbitals,
    double weight,
    double delta_weight,
    std::vector<double>* packed_active_two_electron_gradient) {
  if (packed_active_two_electron_gradient == nullptr) {
    throw std::invalid_argument("packed_active_two_electron_gradient must not be null");
  }
  if (std::abs(weight) <= kContributionTolerance &&
      std::abs(delta_weight) <= kContributionTolerance) {
    return;
  }

  const int n_electrons = static_cast<int>(occ_L.size());
  if (n_electrons < 2) {
    return;
  }

  const Eigen::MatrixXd delta_overlap_submatrix =
      build_local_overlap_direction_matrix(
          occ_L,
          occ_R,
          delta_ao_overlap_matrix,
          n_active_orbitals);
  const Eigen::MatrixXd second = cofactor.second();
  const Eigen::MatrixXd delta_second = cofactor.second_first(delta_overlap_submatrix);
  for (int left_first = 0; left_first < n_electrons - 1; ++left_first) {
    const int orbital_index_left_first = occ_L[left_first];
    for (int right_first = 0; right_first < n_electrons - 1; ++right_first) {
      const int orbital_index_right_first = occ_R[right_first];
      for (int left_second = left_first + 1; left_second < n_electrons; ++left_second) {
        const int orbital_index_left_second = occ_L[left_second];
        for (int right_second = right_first + 1; right_second < n_electrons; ++right_second) {
          const int orbital_index_right_second = occ_R[right_second];
          const double second_order_cofactor =
              second(right_second*(right_second-1)/2+right_first,
                     left_second*(left_second-1)/2+left_first);
          const double directional_second_order_cofactor =
              delta_second(right_second*(right_second-1)/2+right_first,
                           left_second*(left_second-1)/2+left_first);
          const double total_directional_second_order_cofactor =
              delta_weight * second_order_cofactor +
              weight * directional_second_order_cofactor;
          if (std::abs(total_directional_second_order_cofactor) <=
              kContributionTolerance) {
            continue;
          }

          const int direct_index = TwoElectronIndexer::two_electron_storage_index(
              orbital_index_right_first,
              orbital_index_left_first,
              orbital_index_right_second,
              orbital_index_left_second);
          const int exchange_index = TwoElectronIndexer::two_electron_storage_index(
              orbital_index_right_first,
              orbital_index_left_second,
              orbital_index_right_second,
              orbital_index_left_first);
          (*packed_active_two_electron_gradient)[direct_index] +=
              total_directional_second_order_cofactor;
          (*packed_active_two_electron_gradient)[exchange_index] -=
              total_directional_second_order_cofactor;
        }
      }
    }
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

SameSpinExactWeightMatrices build_dense_exact_same_spin_weight_matrices(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies) {
  if (selected_states.states.size() != selected_state_energies.size()) {
    throw std::invalid_argument(
        "selected_state_energies must align with selected_states.states");
  }

  SameSpinExactWeightMatrices weight_matrices;
  const bool close_shell_same_spin =
      same_spin_pair_cache.close_shell_reuses_same_spin_pair_cache();
  weight_matrices.alpha_hamiltonian_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.alpha_overlap_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.alpha_partner_total_transfer_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.alpha_singular_partner_transfer_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.beta_hamiltonian_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);
  weight_matrices.beta_overlap_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);
  weight_matrices.beta_partner_total_transfer_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);
  weight_matrices.beta_singular_partner_transfer_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);

  const SameSpinPairScalarMatrices alpha_scalar_matrices =
      build_pair_scalar_matrices(
          same_spin_pair_cache.alpha_pair_cache_ref(),
          selected_states.n_unique_alpha);
  SameSpinPairScalarMatrices beta_scalar_matrices;
  if (!close_shell_same_spin) {
    beta_scalar_matrices =
        build_pair_scalar_matrices(
            same_spin_pair_cache.beta_pair_cache_ref(),
            selected_states.n_unique_beta);
  }

  // Each selected state already stores the determinant expansion in unique
  // alpha/beta matrix form `C^(n)`. The separable same-spin channels therefore
  // compress exactly to BLAS-3 style products
  //
  //   C^(n) B_beta [C^(n)]^T
  //   [C^(n)]^T A_alpha C^(n)
  //
  // rather than rebuilding full determinant-pair adjoints and scattering them
  // back to the same unique-spin spaces.
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
    const auto& state_coefficients = selected_states.states[state_offset];
    const double state_weight = state_coefficients.normalized_state_weight;
    const double overlap_weight = -selected_state_energies[state_offset] * state_weight;

    validate_state_coefficient_matrix(
        state_coefficients,
        selected_states.n_unique_alpha,
        selected_states.n_unique_beta);
    if (selected_state_has_close_shell_diagonal(
            state_coefficients,
            selected_states.n_unique_alpha,
            selected_states.n_unique_beta)) {
      accumulate_diagonal_kernel_image_global(
          state_coefficients.diagonal_coefficients,
          close_shell_same_spin
              ? alpha_scalar_matrices.overlap_determinant_matrix
              : beta_scalar_matrices.overlap_determinant_matrix,
          state_weight,
          &weight_matrices.alpha_hamiltonian_weight_matrix);
      accumulate_diagonal_kernel_image_global(
          state_coefficients.diagonal_coefficients,
          close_shell_same_spin
              ? alpha_scalar_matrices.overlap_determinant_matrix
              : beta_scalar_matrices.overlap_determinant_matrix,
          overlap_weight,
          &weight_matrices.alpha_overlap_weight_matrix);
      accumulate_diagonal_kernel_image_global(
          state_coefficients.diagonal_coefficients,
          close_shell_same_spin
              ? alpha_scalar_matrices.regular_total_hamiltonian_matrix
              : beta_scalar_matrices.regular_total_hamiltonian_matrix,
          state_weight,
          &weight_matrices.alpha_partner_total_transfer_matrix);
      accumulate_diagonal_kernel_image_global(
          state_coefficients.diagonal_coefficients,
          close_shell_same_spin
              ? alpha_scalar_matrices.singular_total_hamiltonian_matrix
              : beta_scalar_matrices.singular_total_hamiltonian_matrix,
          state_weight,
          &weight_matrices.alpha_singular_partner_transfer_matrix);
      if (!close_shell_same_spin) {
        accumulate_diagonal_kernel_image_global(
            state_coefficients.diagonal_coefficients,
            alpha_scalar_matrices.overlap_determinant_matrix,
            state_weight,
            &weight_matrices.beta_hamiltonian_weight_matrix);
        accumulate_diagonal_kernel_image_global(
            state_coefficients.diagonal_coefficients,
            alpha_scalar_matrices.overlap_determinant_matrix,
            overlap_weight,
            &weight_matrices.beta_overlap_weight_matrix);
        accumulate_diagonal_kernel_image_global(
            state_coefficients.diagonal_coefficients,
            alpha_scalar_matrices.regular_total_hamiltonian_matrix,
            state_weight,
            &weight_matrices.beta_partner_total_transfer_matrix);
        accumulate_diagonal_kernel_image_global(
            state_coefficients.diagonal_coefficients,
            alpha_scalar_matrices.singular_total_hamiltonian_matrix,
            state_weight,
            &weight_matrices.beta_singular_partner_transfer_matrix);
      }
      continue;
    }
    coefficient_matrix_dense = state_coefficients.coefficient_matrix;

    // Reuse the same dense scratch buffers for every selected state so the hot
    // BLAS-3 contractions avoid repeated allocation churn.
    multiply_right_symmetric(
        coefficient_matrix_dense,
        close_shell_same_spin
            ? alpha_scalar_matrices.overlap_determinant_matrix
            : beta_scalar_matrices.overlap_determinant_matrix,
        &beta_push);
    alpha_image.noalias() = beta_push * coefficient_matrix_dense.transpose();

    weight_matrices.alpha_hamiltonian_weight_matrix.noalias() +=
        state_weight * alpha_image;
    weight_matrices.alpha_overlap_weight_matrix.noalias() +=
        overlap_weight * alpha_image;
    if (!close_shell_same_spin) {
      multiply_left_symmetric(
          alpha_scalar_matrices.overlap_determinant_matrix,
          coefficient_matrix_dense,
          &alpha_push);
      beta_image.noalias() = coefficient_matrix_dense.transpose() * alpha_push;
      weight_matrices.beta_hamiltonian_weight_matrix.noalias() +=
          state_weight * beta_image;
      weight_matrices.beta_overlap_weight_matrix.noalias() +=
          overlap_weight * beta_image;
    }

    multiply_right_symmetric(
        coefficient_matrix_dense,
        close_shell_same_spin
            ? alpha_scalar_matrices.regular_total_hamiltonian_matrix
            : beta_scalar_matrices.regular_total_hamiltonian_matrix,
        &beta_push);
    alpha_image.noalias() = beta_push * coefficient_matrix_dense.transpose();
    weight_matrices.alpha_partner_total_transfer_matrix.noalias() +=
        state_weight * alpha_image;

    multiply_right_symmetric(
        coefficient_matrix_dense,
        close_shell_same_spin
            ? alpha_scalar_matrices.singular_total_hamiltonian_matrix
            : beta_scalar_matrices.singular_total_hamiltonian_matrix,
        &beta_push);
    alpha_image.noalias() = beta_push * coefficient_matrix_dense.transpose();
    weight_matrices.alpha_singular_partner_transfer_matrix.noalias() +=
        state_weight * alpha_image;

    if (!close_shell_same_spin) {
      multiply_left_symmetric(
          alpha_scalar_matrices.regular_total_hamiltonian_matrix,
          coefficient_matrix_dense,
          &alpha_push);
      beta_image.noalias() = coefficient_matrix_dense.transpose() * alpha_push;
      weight_matrices.beta_partner_total_transfer_matrix.noalias() +=
          state_weight * beta_image;

      multiply_left_symmetric(
          alpha_scalar_matrices.singular_total_hamiltonian_matrix,
          coefficient_matrix_dense,
          &alpha_push);
      beta_image.noalias() = coefficient_matrix_dense.transpose() * alpha_push;
      weight_matrices.beta_singular_partner_transfer_matrix.noalias() +=
          state_weight * beta_image;
    }
  }

  if (close_shell_same_spin) {
    weight_matrices.beta_hamiltonian_weight_matrix =
        weight_matrices.alpha_hamiltonian_weight_matrix;
    weight_matrices.beta_overlap_weight_matrix =
        weight_matrices.alpha_overlap_weight_matrix;
    weight_matrices.beta_partner_total_transfer_matrix =
        weight_matrices.alpha_partner_total_transfer_matrix;
    weight_matrices.beta_singular_partner_transfer_matrix =
        weight_matrices.alpha_singular_partner_transfer_matrix;
  }

  return weight_matrices;
}

SameSpinExactWeightMatrices build_support_sparse_exact_same_spin_weight_matrices(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies) {
  if (selected_states.states.size() != selected_state_energies.size()) {
    throw std::invalid_argument(
        "selected_state_energies must align with selected_states.states");
  }

  SameSpinExactWeightMatrices weight_matrices;
  const bool close_shell_same_spin =
      same_spin_pair_cache.close_shell_reuses_same_spin_pair_cache();
  weight_matrices.alpha_hamiltonian_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.alpha_overlap_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.alpha_partner_total_transfer_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.alpha_singular_partner_transfer_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.beta_hamiltonian_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);
  weight_matrices.beta_overlap_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);
  weight_matrices.beta_partner_total_transfer_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);
  weight_matrices.beta_singular_partner_transfer_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);

  const SameSpinPairScalarMatrices alpha_scalar_matrices =
      build_pair_scalar_matrices(
          same_spin_pair_cache.alpha_pair_cache_ref(),
          selected_states.n_unique_alpha);
  SameSpinPairScalarMatrices beta_scalar_matrices;
  if (!close_shell_same_spin) {
    beta_scalar_matrices =
        build_pair_scalar_matrices(
            same_spin_pair_cache.beta_pair_cache_ref(),
            selected_states.n_unique_beta);
  }

  // Support-aware exact backward contraction:
  // each selected state contributes only on its trimmed unique-spin block
  // `C^(n)[A_n, B_n]`, so all same-spin adjoint images can be formed from the
  // corresponding local alpha/beta submatrices and scattered back afterwards.
  Eigen::MatrixXd alpha_overlap_subblock;
  Eigen::MatrixXd alpha_regular_total_subblock;
  Eigen::MatrixXd alpha_singular_total_subblock;
  Eigen::MatrixXd beta_overlap_subblock;
  Eigen::MatrixXd beta_regular_total_subblock;
  Eigen::MatrixXd beta_singular_total_subblock;
  Eigen::MatrixXd beta_push;
  Eigen::MatrixXd alpha_push;
  Eigen::MatrixXd alpha_image;
  Eigen::MatrixXd beta_image;
  for (std::size_t state_offset = 0;
       state_offset < selected_states.states.size();
       ++state_offset) {
    const auto& state_coefficients = selected_states.states[state_offset];
    if (!selected_state_has_local_support(state_coefficients)) {
      continue;
    }

    const double state_weight = state_coefficients.normalized_state_weight;
    const double overlap_weight =
        -selected_state_energies[state_offset] * state_weight;

    gather_dense_submatrix(
        alpha_scalar_matrices.overlap_determinant_matrix,
        state_coefficients.alpha_support,
        state_coefficients.alpha_support,
        &alpha_overlap_subblock);
    gather_dense_submatrix(
        alpha_scalar_matrices.regular_total_hamiltonian_matrix,
        state_coefficients.alpha_support,
        state_coefficients.alpha_support,
        &alpha_regular_total_subblock);
    gather_dense_submatrix(
        alpha_scalar_matrices.singular_total_hamiltonian_matrix,
        state_coefficients.alpha_support,
        state_coefficients.alpha_support,
        &alpha_singular_total_subblock);
    gather_dense_submatrix(
        close_shell_same_spin
            ? alpha_scalar_matrices.overlap_determinant_matrix
            : beta_scalar_matrices.overlap_determinant_matrix,
        state_coefficients.beta_support,
        state_coefficients.beta_support,
        &beta_overlap_subblock);
    gather_dense_submatrix(
        close_shell_same_spin
            ? alpha_scalar_matrices.regular_total_hamiltonian_matrix
            : beta_scalar_matrices.regular_total_hamiltonian_matrix,
        state_coefficients.beta_support,
        state_coefficients.beta_support,
        &beta_regular_total_subblock);
    gather_dense_submatrix(
        close_shell_same_spin
            ? alpha_scalar_matrices.singular_total_hamiltonian_matrix
            : beta_scalar_matrices.singular_total_hamiltonian_matrix,
        state_coefficients.beta_support,
        state_coefficients.beta_support,
        &beta_singular_total_subblock);

    accumulate_selected_state_alpha_image(
        state_coefficients,
        beta_overlap_subblock,
        state_weight,
        &weight_matrices.alpha_hamiltonian_weight_matrix,
        &beta_push,
        &alpha_image);
    accumulate_selected_state_alpha_image(
        state_coefficients,
        beta_overlap_subblock,
        overlap_weight,
        &weight_matrices.alpha_overlap_weight_matrix,
        &beta_push,
        &alpha_image);
    if (!close_shell_same_spin) {
      accumulate_selected_state_beta_image(
          state_coefficients,
          alpha_overlap_subblock,
          state_weight,
          &weight_matrices.beta_hamiltonian_weight_matrix,
          &alpha_push,
          &beta_image);
      accumulate_selected_state_beta_image(
          state_coefficients,
          alpha_overlap_subblock,
          overlap_weight,
          &weight_matrices.beta_overlap_weight_matrix,
          &alpha_push,
          &beta_image);
    }
    accumulate_selected_state_alpha_image(
        state_coefficients,
        beta_regular_total_subblock,
        state_weight,
        &weight_matrices.alpha_partner_total_transfer_matrix,
        &beta_push,
        &alpha_image);
    accumulate_selected_state_alpha_image(
        state_coefficients,
        beta_singular_total_subblock,
        state_weight,
        &weight_matrices.alpha_singular_partner_transfer_matrix,
        &beta_push,
        &alpha_image);
    if (!close_shell_same_spin) {
      accumulate_selected_state_beta_image(
          state_coefficients,
          alpha_regular_total_subblock,
          state_weight,
          &weight_matrices.beta_partner_total_transfer_matrix,
          &alpha_push,
          &beta_image);
      accumulate_selected_state_beta_image(
          state_coefficients,
          alpha_singular_total_subblock,
          state_weight,
          &weight_matrices.beta_singular_partner_transfer_matrix,
          &alpha_push,
          &beta_image);
    }
  }

  if (close_shell_same_spin) {
    weight_matrices.beta_hamiltonian_weight_matrix =
        weight_matrices.alpha_hamiltonian_weight_matrix;
    weight_matrices.beta_overlap_weight_matrix =
        weight_matrices.alpha_overlap_weight_matrix;
    weight_matrices.beta_partner_total_transfer_matrix =
        weight_matrices.alpha_partner_total_transfer_matrix;
    weight_matrices.beta_singular_partner_transfer_matrix =
        weight_matrices.alpha_singular_partner_transfer_matrix;
  }

  return weight_matrices;
}

SameSpinExactWeightMatrices build_exact_same_spin_weight_matrices(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies) {
  if (should_use_support_sparse_selected_state_contractions(selected_states)) {
    return build_support_sparse_exact_same_spin_weight_matrices(
        same_spin_pair_cache,
        selected_states,
        selected_state_energies);
  }
  return build_dense_exact_same_spin_weight_matrices(
      same_spin_pair_cache,
      selected_states,
      selected_state_energies);
}

void validate_directional_selected_state_inputs(
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    const std::vector<double>& selected_state_energies,
    const std::vector<double>& directional_selected_state_energies) {
  if (selected_states.n_unique_alpha != directional_selected_states.n_unique_alpha ||
      selected_states.n_unique_beta != directional_selected_states.n_unique_beta ||
      selected_states.n_determinants != directional_selected_states.n_determinants ||
      selected_states.selected_state_indices !=
          directional_selected_states.selected_state_indices ||
      selected_states.states.size() != directional_selected_states.states.size()) {
    throw std::invalid_argument(
        "directional selected-state matrices do not match accepted-point dimensions");
  }
  if (selected_state_energies.size() != selected_states.states.size() ||
      directional_selected_state_energies.size() != selected_states.states.size()) {
    throw std::invalid_argument(
        "selected-state energy directions must align with selected_states.states");
  }
}

SameSpinExactWeightMatrices build_dense_directional_exact_same_spin_weight_matrices(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    const std::vector<double>& selected_state_energies,
    const std::vector<double>& directional_selected_state_energies) {
  validate_directional_selected_state_inputs(
      selected_states,
      directional_selected_states,
      selected_state_energies,
      directional_selected_state_energies);

  SameSpinExactWeightMatrices weight_matrices;
  const bool close_shell_same_spin =
      same_spin_pair_cache.close_shell_reuses_same_spin_pair_cache();
  weight_matrices.alpha_hamiltonian_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.alpha_overlap_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.alpha_partner_total_transfer_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.alpha_singular_partner_transfer_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.beta_hamiltonian_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);
  weight_matrices.beta_overlap_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);
  weight_matrices.beta_partner_total_transfer_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);
  weight_matrices.beta_singular_partner_transfer_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);

  const SameSpinPairScalarMatrices alpha_scalar_matrices =
      build_pair_scalar_matrices(
          same_spin_pair_cache.alpha_pair_cache_ref(),
          selected_states.n_unique_alpha);
  SameSpinPairScalarMatrices beta_scalar_matrices;
  if (!close_shell_same_spin) {
    beta_scalar_matrices =
        build_pair_scalar_matrices(
            same_spin_pair_cache.beta_pair_cache_ref(),
            selected_states.n_unique_beta);
  }

  Eigen::MatrixXd coefficient_matrix_dense(
      selected_states.n_unique_alpha,
      selected_states.n_unique_beta);
  Eigen::MatrixXd directional_coefficient_matrix_dense(
      selected_states.n_unique_alpha,
      selected_states.n_unique_beta);
  Eigen::MatrixXd base_push(
      selected_states.n_unique_alpha,
      selected_states.n_unique_beta);
  Eigen::MatrixXd directional_push(
      selected_states.n_unique_alpha,
      selected_states.n_unique_beta);
  Eigen::MatrixXd base_image_overlap(
      selected_states.n_unique_alpha,
      selected_states.n_unique_alpha);
  Eigen::MatrixXd directional_image_overlap(
      selected_states.n_unique_alpha,
      selected_states.n_unique_alpha);
  Eigen::MatrixXd base_image_transfer(
      selected_states.n_unique_alpha,
      selected_states.n_unique_alpha);
  Eigen::MatrixXd directional_image_transfer(
      selected_states.n_unique_alpha,
      selected_states.n_unique_alpha);
  Eigen::MatrixXd base_beta_image_overlap(
      selected_states.n_unique_beta,
      selected_states.n_unique_beta);
  Eigen::MatrixXd directional_beta_image_overlap(
      selected_states.n_unique_beta,
      selected_states.n_unique_beta);
  Eigen::MatrixXd base_beta_image_transfer(
      selected_states.n_unique_beta,
      selected_states.n_unique_beta);
  Eigen::MatrixXd directional_beta_image_transfer(
      selected_states.n_unique_beta,
      selected_states.n_unique_beta);

  for (std::size_t state_offset = 0;
       state_offset < selected_states.states.size();
       ++state_offset) {
    const auto& state_coefficients = selected_states.states[state_offset];
    const auto& directional_state_coefficients =
        directional_selected_states.states[state_offset];
    const double state_weight = state_coefficients.normalized_state_weight;
    const double state_energy = selected_state_energies[state_offset];
    const double directional_state_energy =
        directional_selected_state_energies[state_offset];

    validate_state_coefficient_matrix(
        state_coefficients,
        selected_states.n_unique_alpha,
        selected_states.n_unique_beta);
    validate_state_coefficient_matrix(
        directional_state_coefficients,
        directional_selected_states.n_unique_alpha,
        directional_selected_states.n_unique_beta);
    if (selected_state_has_close_shell_diagonal(
            state_coefficients,
            selected_states.n_unique_alpha,
            selected_states.n_unique_beta) &&
        selected_state_has_close_shell_diagonal(
            directional_state_coefficients,
            directional_selected_states.n_unique_alpha,
            directional_selected_states.n_unique_beta)) {
      accumulate_directional_diagonal_kernel_image_global(
          state_coefficients.diagonal_coefficients,
          directional_state_coefficients.diagonal_coefficients,
          close_shell_same_spin
              ? alpha_scalar_matrices.overlap_determinant_matrix
              : beta_scalar_matrices.overlap_determinant_matrix,
          state_weight,
          &weight_matrices.alpha_hamiltonian_weight_matrix);
      accumulate_diagonal_kernel_image_global(
          state_coefficients.diagonal_coefficients,
          close_shell_same_spin
              ? alpha_scalar_matrices.overlap_determinant_matrix
              : beta_scalar_matrices.overlap_determinant_matrix,
          -state_weight * directional_state_energy,
          &weight_matrices.alpha_overlap_weight_matrix);
      accumulate_directional_diagonal_kernel_image_global(
          state_coefficients.diagonal_coefficients,
          directional_state_coefficients.diagonal_coefficients,
          close_shell_same_spin
              ? alpha_scalar_matrices.overlap_determinant_matrix
              : beta_scalar_matrices.overlap_determinant_matrix,
          -state_weight * state_energy,
          &weight_matrices.alpha_overlap_weight_matrix);
      accumulate_directional_diagonal_kernel_image_global(
          state_coefficients.diagonal_coefficients,
          directional_state_coefficients.diagonal_coefficients,
          close_shell_same_spin
              ? alpha_scalar_matrices.regular_total_hamiltonian_matrix
              : beta_scalar_matrices.regular_total_hamiltonian_matrix,
          state_weight,
          &weight_matrices.alpha_partner_total_transfer_matrix);
      accumulate_directional_diagonal_kernel_image_global(
          state_coefficients.diagonal_coefficients,
          directional_state_coefficients.diagonal_coefficients,
          close_shell_same_spin
              ? alpha_scalar_matrices.singular_total_hamiltonian_matrix
              : beta_scalar_matrices.singular_total_hamiltonian_matrix,
          state_weight,
          &weight_matrices.alpha_singular_partner_transfer_matrix);
      if (!close_shell_same_spin) {
        accumulate_directional_diagonal_kernel_image_global(
            state_coefficients.diagonal_coefficients,
            directional_state_coefficients.diagonal_coefficients,
            alpha_scalar_matrices.overlap_determinant_matrix,
            state_weight,
            &weight_matrices.beta_hamiltonian_weight_matrix);
        accumulate_diagonal_kernel_image_global(
            state_coefficients.diagonal_coefficients,
            alpha_scalar_matrices.overlap_determinant_matrix,
            -state_weight * directional_state_energy,
            &weight_matrices.beta_overlap_weight_matrix);
        accumulate_directional_diagonal_kernel_image_global(
            state_coefficients.diagonal_coefficients,
            directional_state_coefficients.diagonal_coefficients,
            alpha_scalar_matrices.overlap_determinant_matrix,
            -state_weight * state_energy,
            &weight_matrices.beta_overlap_weight_matrix);
        accumulate_directional_diagonal_kernel_image_global(
            state_coefficients.diagonal_coefficients,
            directional_state_coefficients.diagonal_coefficients,
            alpha_scalar_matrices.regular_total_hamiltonian_matrix,
            state_weight,
            &weight_matrices.beta_partner_total_transfer_matrix);
        accumulate_directional_diagonal_kernel_image_global(
            state_coefficients.diagonal_coefficients,
            directional_state_coefficients.diagonal_coefficients,
            alpha_scalar_matrices.singular_total_hamiltonian_matrix,
            state_weight,
            &weight_matrices.beta_singular_partner_transfer_matrix);
      }
      continue;
    }
    coefficient_matrix_dense = state_coefficients.coefficient_matrix;
    directional_coefficient_matrix_dense =
        directional_state_coefficients.coefficient_matrix;
    // The directional response is linear in both `delta C` and `delta E`.
    // Rescaling them here keeps the BLAS products inside a safe dynamic range
    // without changing the exact directional result.
    const double directional_scale =
        std::max(
            1.0,
            std::max(
                max_abs_dense_matrix(directional_coefficient_matrix_dense),
                std::abs(directional_state_energy)));
    if (directional_scale != 1.0) {
      directional_coefficient_matrix_dense /= directional_scale;
    }
    const double scaled_directional_state_energy =
        directional_state_energy / directional_scale;

    multiply_right_symmetric(
        coefficient_matrix_dense,
        close_shell_same_spin
            ? alpha_scalar_matrices.overlap_determinant_matrix
            : beta_scalar_matrices.overlap_determinant_matrix,
        &base_push);
    base_image_overlap.noalias() =
        base_push * coefficient_matrix_dense.transpose();
    multiply_right_symmetric(
        directional_coefficient_matrix_dense,
        close_shell_same_spin
            ? alpha_scalar_matrices.overlap_determinant_matrix
            : beta_scalar_matrices.overlap_determinant_matrix,
        &directional_push);
    directional_image_overlap.noalias() =
        directional_push * coefficient_matrix_dense.transpose();
    directional_image_overlap.noalias() +=
        base_push * directional_coefficient_matrix_dense.transpose();

    weight_matrices.alpha_hamiltonian_weight_matrix.noalias() +=
        state_weight * directional_scale * directional_image_overlap;
    weight_matrices.alpha_overlap_weight_matrix.noalias() -=
        state_weight * directional_scale *
        (scaled_directional_state_energy * base_image_overlap +
         state_energy * directional_image_overlap);

    multiply_right_symmetric(
        coefficient_matrix_dense,
        close_shell_same_spin
            ? alpha_scalar_matrices.regular_total_hamiltonian_matrix
            : beta_scalar_matrices.regular_total_hamiltonian_matrix,
        &base_push);
    base_image_transfer.noalias() =
        base_push * coefficient_matrix_dense.transpose();
    multiply_right_symmetric(
        directional_coefficient_matrix_dense,
        close_shell_same_spin
            ? alpha_scalar_matrices.regular_total_hamiltonian_matrix
            : beta_scalar_matrices.regular_total_hamiltonian_matrix,
        &directional_push);
    directional_image_transfer.noalias() =
        directional_push * coefficient_matrix_dense.transpose();
    directional_image_transfer.noalias() +=
        base_push * directional_coefficient_matrix_dense.transpose();
    weight_matrices.alpha_partner_total_transfer_matrix.noalias() +=
        state_weight * directional_scale * directional_image_transfer;

    multiply_right_symmetric(
        coefficient_matrix_dense,
        close_shell_same_spin
            ? alpha_scalar_matrices.singular_total_hamiltonian_matrix
            : beta_scalar_matrices.singular_total_hamiltonian_matrix,
        &base_push);
    base_image_transfer.noalias() =
        base_push * coefficient_matrix_dense.transpose();
    multiply_right_symmetric(
        directional_coefficient_matrix_dense,
        close_shell_same_spin
            ? alpha_scalar_matrices.singular_total_hamiltonian_matrix
            : beta_scalar_matrices.singular_total_hamiltonian_matrix,
        &directional_push);
    directional_image_transfer.noalias() =
        directional_push * coefficient_matrix_dense.transpose();
    directional_image_transfer.noalias() +=
        base_push * directional_coefficient_matrix_dense.transpose();
    weight_matrices.alpha_singular_partner_transfer_matrix.noalias() +=
        state_weight * directional_scale * directional_image_transfer;

    if (!close_shell_same_spin) {
      multiply_left_symmetric(
          alpha_scalar_matrices.overlap_determinant_matrix,
          coefficient_matrix_dense,
          &base_push);
      base_beta_image_overlap.noalias() =
          coefficient_matrix_dense.transpose() * base_push;
      multiply_left_symmetric(
          alpha_scalar_matrices.overlap_determinant_matrix,
          directional_coefficient_matrix_dense,
          &directional_push);
      directional_beta_image_overlap.noalias() =
          directional_coefficient_matrix_dense.transpose() * base_push;
      directional_beta_image_overlap.noalias() +=
          coefficient_matrix_dense.transpose() * directional_push;

      weight_matrices.beta_hamiltonian_weight_matrix.noalias() +=
          state_weight * directional_scale * directional_beta_image_overlap;
      weight_matrices.beta_overlap_weight_matrix.noalias() -=
          state_weight * directional_scale *
          (scaled_directional_state_energy * base_beta_image_overlap +
           state_energy * directional_beta_image_overlap);

      multiply_left_symmetric(
          alpha_scalar_matrices.regular_total_hamiltonian_matrix,
          coefficient_matrix_dense,
          &base_push);
      base_beta_image_transfer.noalias() =
          coefficient_matrix_dense.transpose() * base_push;
      multiply_left_symmetric(
          alpha_scalar_matrices.regular_total_hamiltonian_matrix,
          directional_coefficient_matrix_dense,
          &directional_push);
      directional_beta_image_transfer.noalias() =
          directional_coefficient_matrix_dense.transpose() * base_push;
      directional_beta_image_transfer.noalias() +=
          coefficient_matrix_dense.transpose() * directional_push;
      weight_matrices.beta_partner_total_transfer_matrix.noalias() +=
          state_weight * directional_scale * directional_beta_image_transfer;

      multiply_left_symmetric(
          alpha_scalar_matrices.singular_total_hamiltonian_matrix,
          coefficient_matrix_dense,
          &base_push);
      base_beta_image_transfer.noalias() =
          coefficient_matrix_dense.transpose() * base_push;
      multiply_left_symmetric(
          alpha_scalar_matrices.singular_total_hamiltonian_matrix,
          directional_coefficient_matrix_dense,
          &directional_push);
      directional_beta_image_transfer.noalias() =
          directional_coefficient_matrix_dense.transpose() * base_push;
      directional_beta_image_transfer.noalias() +=
          coefficient_matrix_dense.transpose() * directional_push;
      weight_matrices.beta_singular_partner_transfer_matrix.noalias() +=
          state_weight * directional_scale * directional_beta_image_transfer;
    }
  }

  if (close_shell_same_spin) {
    weight_matrices.beta_hamiltonian_weight_matrix =
        weight_matrices.alpha_hamiltonian_weight_matrix;
    weight_matrices.beta_overlap_weight_matrix =
        weight_matrices.alpha_overlap_weight_matrix;
    weight_matrices.beta_partner_total_transfer_matrix =
        weight_matrices.alpha_partner_total_transfer_matrix;
    weight_matrices.beta_singular_partner_transfer_matrix =
        weight_matrices.alpha_singular_partner_transfer_matrix;
  }

  return weight_matrices;
}

SameSpinExactWeightMatrices build_support_sparse_directional_exact_same_spin_weight_matrices(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    const std::vector<double>& selected_state_energies,
    const std::vector<double>& directional_selected_state_energies) {
  validate_directional_selected_state_inputs(
      selected_states,
      directional_selected_states,
      selected_state_energies,
      directional_selected_state_energies);

  SameSpinExactWeightMatrices weight_matrices;
  const bool close_shell_same_spin =
      same_spin_pair_cache.close_shell_reuses_same_spin_pair_cache();
  weight_matrices.alpha_hamiltonian_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.alpha_overlap_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.alpha_partner_total_transfer_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.alpha_singular_partner_transfer_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.beta_hamiltonian_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);
  weight_matrices.beta_overlap_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);
  weight_matrices.beta_partner_total_transfer_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);
  weight_matrices.beta_singular_partner_transfer_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);

  const SameSpinPairScalarMatrices alpha_scalar_matrices =
      build_pair_scalar_matrices(
          same_spin_pair_cache.alpha_pair_cache_ref(),
          selected_states.n_unique_alpha);
  SameSpinPairScalarMatrices beta_scalar_matrices;
  if (!close_shell_same_spin) {
    beta_scalar_matrices =
        build_pair_scalar_matrices(
            same_spin_pair_cache.beta_pair_cache_ref(),
            selected_states.n_unique_beta);
  }

  Eigen::MatrixXd alpha_overlap_subblock;
  Eigen::MatrixXd alpha_regular_total_subblock;
  Eigen::MatrixXd alpha_singular_total_subblock;
  Eigen::MatrixXd beta_overlap_subblock;
  Eigen::MatrixXd beta_regular_total_subblock;
  Eigen::MatrixXd beta_singular_total_subblock;
  Eigen::MatrixXd coefficient_matrix_union;
  Eigen::MatrixXd directional_coefficient_matrix_union;
  Eigen::MatrixXd base_push;
  Eigen::MatrixXd directional_push;
  Eigen::MatrixXd alpha_base_image;
  Eigen::MatrixXd alpha_directional_image;
  Eigen::MatrixXd beta_base_image;
  Eigen::MatrixXd beta_directional_image;

  for (std::size_t state_offset = 0;
       state_offset < selected_states.states.size();
       ++state_offset) {
    const auto& state_coefficients = selected_states.states[state_offset];
    const auto& directional_state_coefficients =
        directional_selected_states.states[state_offset];
    const double state_weight = state_coefficients.normalized_state_weight;
    const double state_energy = selected_state_energies[state_offset];
    const double directional_state_energy =
        directional_selected_state_energies[state_offset];

    const std::vector<int> alpha_union_support =
        build_merged_support_indices(
            state_coefficients.alpha_support,
            directional_state_coefficients.alpha_support);
    const std::vector<int> beta_union_support =
        build_merged_support_indices(
            state_coefficients.beta_support,
            directional_state_coefficients.beta_support);
    if (alpha_union_support.empty() || beta_union_support.empty()) {
      continue;
    }

    // The directional selected-state response can activate rows/columns that
    // are zero at the accepted point, so the sparse contraction must run on the
    // union support rather than only on the accepted-state block.
    gather_dense_submatrix(
        state_coefficients.coefficient_matrix,
        alpha_union_support,
        beta_union_support,
        &coefficient_matrix_union);
    gather_dense_submatrix(
        directional_state_coefficients.coefficient_matrix,
        alpha_union_support,
        beta_union_support,
        &directional_coefficient_matrix_union);

    const double directional_scale =
        std::max(
            1.0,
            std::max(
                max_abs_dense_matrix(directional_coefficient_matrix_union),
                std::abs(directional_state_energy)));
    if (directional_scale != 1.0) {
      directional_coefficient_matrix_union /= directional_scale;
    }
    const double scaled_directional_state_energy =
        directional_state_energy / directional_scale;

    gather_dense_submatrix(
        alpha_scalar_matrices.overlap_determinant_matrix,
        alpha_union_support,
        alpha_union_support,
        &alpha_overlap_subblock);
    gather_dense_submatrix(
        alpha_scalar_matrices.regular_total_hamiltonian_matrix,
        alpha_union_support,
        alpha_union_support,
        &alpha_regular_total_subblock);
    gather_dense_submatrix(
        alpha_scalar_matrices.singular_total_hamiltonian_matrix,
        alpha_union_support,
        alpha_union_support,
        &alpha_singular_total_subblock);
    gather_dense_submatrix(
        close_shell_same_spin
            ? alpha_scalar_matrices.overlap_determinant_matrix
            : beta_scalar_matrices.overlap_determinant_matrix,
        beta_union_support,
        beta_union_support,
        &beta_overlap_subblock);
    gather_dense_submatrix(
        close_shell_same_spin
            ? alpha_scalar_matrices.regular_total_hamiltonian_matrix
            : beta_scalar_matrices.regular_total_hamiltonian_matrix,
        beta_union_support,
        beta_union_support,
        &beta_regular_total_subblock);
    gather_dense_submatrix(
        close_shell_same_spin
            ? alpha_scalar_matrices.singular_total_hamiltonian_matrix
            : beta_scalar_matrices.singular_total_hamiltonian_matrix,
        beta_union_support,
        beta_union_support,
        &beta_singular_total_subblock);

    multiply_right_symmetric(
        coefficient_matrix_union,
        beta_overlap_subblock,
        &base_push);
    alpha_base_image.noalias() =
        base_push * coefficient_matrix_union.transpose();
    multiply_right_symmetric(
        directional_coefficient_matrix_union,
        beta_overlap_subblock,
        &directional_push);
    alpha_directional_image.noalias() =
        directional_push * coefficient_matrix_union.transpose();
    alpha_directional_image.noalias() +=
        base_push * directional_coefficient_matrix_union.transpose();

    scatter_add_dense_submatrix(
        alpha_directional_image,
        alpha_union_support,
        alpha_union_support,
        state_weight * directional_scale,
        &weight_matrices.alpha_hamiltonian_weight_matrix);
    scatter_add_dense_submatrix(
        alpha_base_image,
        alpha_union_support,
        alpha_union_support,
        -state_weight * directional_scale * scaled_directional_state_energy,
        &weight_matrices.alpha_overlap_weight_matrix);
    scatter_add_dense_submatrix(
        alpha_directional_image,
        alpha_union_support,
        alpha_union_support,
        -state_weight * directional_scale * state_energy,
        &weight_matrices.alpha_overlap_weight_matrix);

    multiply_right_symmetric(
        coefficient_matrix_union,
        beta_regular_total_subblock,
        &base_push);
    multiply_right_symmetric(
        directional_coefficient_matrix_union,
        beta_regular_total_subblock,
        &directional_push);
    alpha_directional_image.noalias() =
        directional_push * coefficient_matrix_union.transpose();
    alpha_directional_image.noalias() +=
        base_push * directional_coefficient_matrix_union.transpose();
    scatter_add_dense_submatrix(
        alpha_directional_image,
        alpha_union_support,
        alpha_union_support,
        state_weight * directional_scale,
        &weight_matrices.alpha_partner_total_transfer_matrix);

    multiply_right_symmetric(
        coefficient_matrix_union,
        beta_singular_total_subblock,
        &base_push);
    multiply_right_symmetric(
        directional_coefficient_matrix_union,
        beta_singular_total_subblock,
        &directional_push);
    alpha_directional_image.noalias() =
        directional_push * coefficient_matrix_union.transpose();
    alpha_directional_image.noalias() +=
        base_push * directional_coefficient_matrix_union.transpose();
    scatter_add_dense_submatrix(
        alpha_directional_image,
        alpha_union_support,
        alpha_union_support,
        state_weight * directional_scale,
        &weight_matrices.alpha_singular_partner_transfer_matrix);

    if (!close_shell_same_spin) {
      multiply_left_symmetric(
          alpha_overlap_subblock,
          coefficient_matrix_union,
          &base_push);
      beta_base_image.noalias() =
          coefficient_matrix_union.transpose() * base_push;
      multiply_left_symmetric(
          alpha_overlap_subblock,
          directional_coefficient_matrix_union,
          &directional_push);
      beta_directional_image.noalias() =
          directional_coefficient_matrix_union.transpose() * base_push;
      beta_directional_image.noalias() +=
          coefficient_matrix_union.transpose() * directional_push;

      scatter_add_dense_submatrix(
          beta_directional_image,
          beta_union_support,
          beta_union_support,
          state_weight * directional_scale,
          &weight_matrices.beta_hamiltonian_weight_matrix);
      scatter_add_dense_submatrix(
          beta_base_image,
          beta_union_support,
          beta_union_support,
          -state_weight * directional_scale * scaled_directional_state_energy,
          &weight_matrices.beta_overlap_weight_matrix);
      scatter_add_dense_submatrix(
          beta_directional_image,
          beta_union_support,
          beta_union_support,
          -state_weight * directional_scale * state_energy,
          &weight_matrices.beta_overlap_weight_matrix);

      multiply_left_symmetric(
          alpha_regular_total_subblock,
          coefficient_matrix_union,
          &base_push);
      multiply_left_symmetric(
          alpha_regular_total_subblock,
          directional_coefficient_matrix_union,
          &directional_push);
      beta_directional_image.noalias() =
          directional_coefficient_matrix_union.transpose() * base_push;
      beta_directional_image.noalias() +=
          coefficient_matrix_union.transpose() * directional_push;
      scatter_add_dense_submatrix(
          beta_directional_image,
          beta_union_support,
          beta_union_support,
          state_weight * directional_scale,
          &weight_matrices.beta_partner_total_transfer_matrix);

      multiply_left_symmetric(
          alpha_singular_total_subblock,
          coefficient_matrix_union,
          &base_push);
      multiply_left_symmetric(
          alpha_singular_total_subblock,
          directional_coefficient_matrix_union,
          &directional_push);
      beta_directional_image.noalias() =
          directional_coefficient_matrix_union.transpose() * base_push;
      beta_directional_image.noalias() +=
          coefficient_matrix_union.transpose() * directional_push;
      scatter_add_dense_submatrix(
          beta_directional_image,
          beta_union_support,
          beta_union_support,
          state_weight * directional_scale,
          &weight_matrices.beta_singular_partner_transfer_matrix);
    }
  }

  if (close_shell_same_spin) {
    weight_matrices.beta_hamiltonian_weight_matrix =
        weight_matrices.alpha_hamiltonian_weight_matrix;
    weight_matrices.beta_overlap_weight_matrix =
        weight_matrices.alpha_overlap_weight_matrix;
    weight_matrices.beta_partner_total_transfer_matrix =
        weight_matrices.alpha_partner_total_transfer_matrix;
    weight_matrices.beta_singular_partner_transfer_matrix =
        weight_matrices.alpha_singular_partner_transfer_matrix;
  }

  return weight_matrices;
}

void validate_full_matrix_same_spin_inputs(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies) {
  if (!same_spin_pair_cache.enabled()) {
    throw std::invalid_argument(
        "same-spin matrix backward requires an enabled same-spin cache");
  }
  if (selected_states.states.size() != selected_state_energies.size()) {
    throw std::invalid_argument(
        "selected_state_energies must align with selected_states.states");
  }
  if (static_cast<int>(same_spin_pair_cache.alpha_reuse_table.unique_determinants.size()) !=
          selected_states.n_unique_alpha ||
      static_cast<int>(same_spin_pair_cache.beta_reuse_table.unique_determinants.size()) !=
          selected_states.n_unique_beta) {
    throw std::invalid_argument(
        "same-spin cache dimensions do not match selected-state matrices");
  }
}

SameSpinDirectionalScalarMatrices build_directional_pair_scalar_matrices(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    int n_unique_determinants,
    int n_active_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& active_one_electron_matrix,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const std::vector<double>& delta_ao_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals) {
  // For the local-response we only need the partner-sector directional scalars
  // `δdet` and `δH_same`. These remain symmetric on the ordered unique-spin
  // space, so the same BLAS-3/sparse contraction used by the accepted-point
  // same-spin weights can compress them back to alpha/beta pair weights.
  const std::size_t expected_size = square_storage_size(n_unique_determinants);
  if (ordered_pair_cache.size() != expected_size) {
    throw std::invalid_argument(
        "ordered same-spin pair cache size does not match unique-spin dimensions");
  }

  SameSpinDirectionalScalarMatrices scalar_matrices;
  scalar_matrices.delta_overlap_determinant_matrix =
      Eigen::MatrixXd::Zero(n_unique_determinants, n_unique_determinants);
  scalar_matrices.delta_regular_total_hamiltonian_matrix =
      Eigen::MatrixXd::Zero(n_unique_determinants, n_unique_determinants);
  scalar_matrices.delta_singular_total_hamiltonian_matrix =
      Eigen::MatrixXd::Zero(n_unique_determinants, n_unique_determinants);
  scalar_matrices.ordered_pair_data.resize(expected_size);

  for (int left_id = 0; left_id < n_unique_determinants; ++left_id) {
    for (int right_id = left_id; right_id < n_unique_determinants; ++right_id) {
      const auto& pair_evaluation =
          ordered_pair_cache[ordered_spin_pair_storage_index(
              left_id,
              right_id,
              n_unique_determinants)];
      const auto& occ_L = unique_determinants[left_id];
      const auto& occ_R = unique_determinants[right_id];
      const auto& overlap_result = pair_evaluation.overlap_result;

      SameSpinPolynomialDirectionalPairData directional_data =
          build_polynomial_spin_directional_data(
              occ_L,
              occ_R,
              active_one_electron_matrix,
              active_space_two_electron_result,
              pair_evaluation,
              n_active_orbitals,
              delta_ao_overlap_matrix,
              delta_active_one_electron_matrix,
              delta_packed_active_two_electron_integrals);
      const std::size_t forward_index = ordered_spin_pair_storage_index(
          left_id, right_id, n_unique_determinants);
      scalar_matrices.ordered_pair_data[forward_index] = directional_data;
      if (left_id != right_id) {
        SameSpinPolynomialDirectionalPairData transposed = directional_data;
        transposed.cofactor_1st.transposeInPlace();
        transposed.delta_cofactor_1st.transposeInPlace();
        transposed.delta_same_spin_overlap_hamiltonian_gradient.transposeInPlace();
        scalar_matrices.ordered_pair_data[ordered_spin_pair_storage_index(
            right_id, left_id, n_unique_determinants)] = std::move(transposed);
      }
      set_symmetric_matrix_entry(
          &scalar_matrices.delta_overlap_determinant_matrix,
          left_id,
          right_id,
          directional_data.delta_overlap_determinant);
      set_symmetric_matrix_entry(
          (overlap_result.nullity == 0 && overlap_result.overlap_determinant != 0.0
               ? &scalar_matrices.delta_regular_total_hamiltonian_matrix
               : &scalar_matrices.delta_singular_total_hamiltonian_matrix),
          left_id,
          right_id,
          directional_data.delta_total_hamiltonian);
    }
  }

  return scalar_matrices;
}

SameSpinLocalResponseWeightMatrices build_local_same_spin_response_weight_matrices(
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies,
    const SameSpinDirectionalScalarMatrices& alpha_directional_scalars,
    const SameSpinDirectionalScalarMatrices& beta_directional_scalars,
    bool close_shell_same_spin) {
  // The accepted selected-state coefficients stay fixed. Only the partner
  // kernels change, so the same unique-spin coefficient matrices `C^(n)` map
  // `δdet_partner` and `δH_same,partner` back to the active same-spin weights.
  if (selected_state_energies.size() != selected_states.states.size()) {
    throw std::invalid_argument(
        "selected_state_energies must align with selected_states.states");
  }

  std::vector<double> hamiltonian_scales;
  std::vector<double> overlap_scales;
  hamiltonian_scales.reserve(selected_states.states.size());
  overlap_scales.reserve(selected_states.states.size());
  for (std::size_t state_offset = 0;
       state_offset < selected_states.states.size();
       ++state_offset) {
    const double state_weight =
        selected_states.states[state_offset].normalized_state_weight;
    hamiltonian_scales.push_back(state_weight);
    overlap_scales.push_back(-selected_state_energies[state_offset] * state_weight);
  }

  SameSpinLocalResponseWeightMatrices weight_matrices;
  const SingleChannelSameSpinWeightMatrices delta_hamiltonian_weights =
      build_same_spin_weight_matrices_from_partner_kernels(
          selected_states,
          alpha_directional_scalars.delta_overlap_determinant_matrix,
          beta_directional_scalars.delta_overlap_determinant_matrix,
          hamiltonian_scales,
          close_shell_same_spin);
  weight_matrices.alpha_delta_hamiltonian_weight_matrix =
      std::move(delta_hamiltonian_weights.alpha_weight_matrix);
  weight_matrices.beta_delta_hamiltonian_weight_matrix =
      std::move(delta_hamiltonian_weights.beta_weight_matrix);

  const SingleChannelSameSpinWeightMatrices delta_overlap_weights =
      build_same_spin_weight_matrices_from_partner_kernels(
          selected_states,
          alpha_directional_scalars.delta_overlap_determinant_matrix,
          beta_directional_scalars.delta_overlap_determinant_matrix,
          overlap_scales,
          close_shell_same_spin);
  weight_matrices.alpha_delta_overlap_weight_matrix =
      std::move(delta_overlap_weights.alpha_weight_matrix);
  weight_matrices.beta_delta_overlap_weight_matrix =
      std::move(delta_overlap_weights.beta_weight_matrix);

  const SingleChannelSameSpinWeightMatrices delta_partner_weights =
      build_same_spin_weight_matrices_from_partner_kernels(
          selected_states,
          alpha_directional_scalars.delta_regular_total_hamiltonian_matrix +
              alpha_directional_scalars.delta_singular_total_hamiltonian_matrix,
          beta_directional_scalars.delta_regular_total_hamiltonian_matrix +
              beta_directional_scalars.delta_singular_total_hamiltonian_matrix,
          hamiltonian_scales,
          close_shell_same_spin);
  weight_matrices.alpha_delta_partner_total_transfer_matrix =
      std::move(delta_partner_weights.alpha_weight_matrix);
  weight_matrices.beta_delta_partner_total_transfer_matrix =
      std::move(delta_partner_weights.beta_weight_matrix);
  return weight_matrices;
}

bool same_spin_local_tile_has_any_weight(
    const Eigen::MatrixXd& hamiltonian_weight_matrix,
    const Eigen::MatrixXd& overlap_weight_matrix,
    const Eigen::MatrixXd& partner_total_transfer_matrix,
    const Eigen::MatrixXd& delta_hamiltonian_weight_matrix,
    const Eigen::MatrixXd& delta_overlap_weight_matrix,
    const Eigen::MatrixXd& delta_partner_total_transfer_matrix,
    int left_begin,
    int left_end,
    int right_begin,
    int right_end) {
  for (int left_id = left_begin; left_id < left_end; ++left_id) {
    for (int right_id = right_begin; right_id < right_end; ++right_id) {
      if (std::abs(hamiltonian_weight_matrix(left_id, right_id)) > kContributionTolerance ||
          std::abs(overlap_weight_matrix(left_id, right_id)) > kContributionTolerance ||
          std::abs(partner_total_transfer_matrix(left_id, right_id)) > kContributionTolerance ||
          std::abs(delta_hamiltonian_weight_matrix(left_id, right_id)) > kContributionTolerance ||
          std::abs(delta_overlap_weight_matrix(left_id, right_id)) > kContributionTolerance ||
          std::abs(delta_partner_total_transfer_matrix(left_id, right_id)) >
              kContributionTolerance) {
        return true;
      }
    }
  }
  return false;
}

template <typename HWeight, typename SWeight, typename TWeight>
bool same_spin_weight_tile_has_any_weight(
    const HWeight& hamiltonian_weight_tile,
    const SWeight& overlap_weight_tile,
    const TWeight& partner_total_transfer_tile) {
  for (int column = 0; column < hamiltonian_weight_tile.cols(); ++column) {
    for (int row = 0; row < hamiltonian_weight_tile.rows(); ++row) {
      if (std::abs(hamiltonian_weight_tile(row, column)) > kContributionTolerance ||
          std::abs(overlap_weight_tile(row, column)) > kContributionTolerance ||
          std::abs(partner_total_transfer_tile(row, column)) >
              kContributionTolerance) {
        return true;
      }
    }
  }
  return false;
}

template <
    typename HWeight,
    typename SWeight,
    typename TWeight,
    typename DHWeight,
    typename DSWeight,
    typename DTWeight>
bool same_spin_local_weight_tile_has_any_weight(
    const HWeight& hamiltonian_weight_tile,
    const SWeight& overlap_weight_tile,
    const TWeight& partner_total_transfer_tile,
    const DHWeight& delta_hamiltonian_weight_tile,
    const DSWeight& delta_overlap_weight_tile,
    const DTWeight& delta_partner_total_transfer_tile) {
  for (int column = 0; column < hamiltonian_weight_tile.cols(); ++column) {
    for (int row = 0; row < hamiltonian_weight_tile.rows(); ++row) {
      if (std::abs(hamiltonian_weight_tile(row, column)) >
              kContributionTolerance ||
          std::abs(overlap_weight_tile(row, column)) >
              kContributionTolerance ||
          std::abs(partner_total_transfer_tile(row, column)) >
              kContributionTolerance ||
          std::abs(delta_hamiltonian_weight_tile(row, column)) >
              kContributionTolerance ||
          std::abs(delta_overlap_weight_tile(row, column)) >
              kContributionTolerance ||
          std::abs(delta_partner_total_transfer_tile(row, column)) >
              kContributionTolerance) {
        return true;
      }
    }
  }
  return false;
}

template <typename HWeight, typename SWeight, typename TWeight>
void accumulate_spin_matrix_backward_tile(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    const HWeight& hamiltonian_weight_tile,
    const SWeight& overlap_weight_tile,
    const TWeight& partner_total_transfer_tile,
    int left_begin,
    int right_begin,
    int n_unique_determinants,
    int n_active_orbitals,
    Eigen::MatrixXd* active_one_electron_gradient,
    std::vector<double>* active_orbital_overlap_gradient,
    std::vector<double>* packed_active_two_electron_gradient) {
  for (int left_local = 0;
       left_local < hamiltonian_weight_tile.rows();
       ++left_local) {
    const int left_id = left_begin + left_local;
    for (int right_local = 0;
         right_local < hamiltonian_weight_tile.cols();
         ++right_local) {
      const int right_id = right_begin + right_local;
      const double hamiltonian_weight =
          hamiltonian_weight_tile(left_local, right_local);
      const double overlap_weight =
          overlap_weight_tile(left_local, right_local);
      const double partner_total =
          partner_total_transfer_tile(left_local, right_local);
      if (std::abs(hamiltonian_weight) <= kContributionTolerance &&
          std::abs(overlap_weight) <= kContributionTolerance &&
          std::abs(partner_total) <= kContributionTolerance) {
        continue;
      }

      const auto& pair_evaluation =
          ordered_pair_cache[ordered_spin_pair_storage_index(
              left_id,
              right_id,
              n_unique_determinants)];
      const auto& occ_L = unique_determinants[left_id];
      const auto& occ_R = unique_determinants[right_id];
      const auto& overlap_result = pair_evaluation.overlap_result;
      const bool has_hamiltonian_weight =
          std::abs(hamiltonian_weight) > kContributionTolerance;

      const double determinant_overlap_weight =
          overlap_weight + partner_total;
      const bool need_first_order_cofactor =
          has_hamiltonian_weight ||
          std::abs(determinant_overlap_weight) > kContributionTolerance;
      Eigen::MatrixXd cofactor_1st;
      if (need_first_order_cofactor) {
        cofactor_1st = calc_cofactor_1st(overlap_result);
      }

      if (has_hamiltonian_weight) {
        if (cofactor_1st.size() != 0) {
          accumulate_one_electron_gradient_contribution_local(
              occ_L,
              occ_R,
              cofactor_1st,
              hamiltonian_weight,
              active_one_electron_gradient);
        }
        accumulate_deleted_minor_same_spin_two_electron_gradient_contribution_local(
            occ_L,
            occ_R,
            cached_cofactor_differential(pair_evaluation),
            hamiltonian_weight,
            packed_active_two_electron_gradient);
        if (pair_evaluation.same_spin_overlap_hamiltonian_gradient.rows() !=
                static_cast<int>(occ_R.size()) ||
            pair_evaluation.same_spin_overlap_hamiltonian_gradient.cols() !=
                static_cast<int>(occ_L.size())) {
          throw std::runtime_error(
              "matrix-form same-spin backward requires cached overlap Hamiltonian gradients");
        }
        accumulate_overlap_block_gradient_contribution_local(
            occ_L,
            occ_R,
            pair_evaluation.same_spin_overlap_hamiltonian_gradient,
            hamiltonian_weight,
            n_active_orbitals,
            active_orbital_overlap_gradient);
      }

      if (std::abs(determinant_overlap_weight) <= kContributionTolerance ||
          cofactor_1st.size() == 0) {
        continue;
      }
      accumulate_overlap_block_gradient_contribution_local(
          occ_L,
          occ_R,
          cofactor_1st,
          determinant_overlap_weight,
          n_active_orbitals,
          active_orbital_overlap_gradient);
    }
  }
}

void accumulate_spin_matrix_backward(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    const Eigen::MatrixXd& hamiltonian_weight_matrix,
    const Eigen::MatrixXd& overlap_weight_matrix,
    const Eigen::MatrixXd& partner_total_transfer_matrix,
    int n_unique_determinants,
    int n_active_orbitals,
    Eigen::MatrixXd* active_one_electron_gradient,
    std::vector<double>* active_orbital_overlap_gradient,
    std::vector<double>* packed_active_two_electron_gradient) {
  if (active_one_electron_gradient == nullptr ||
      active_orbital_overlap_gradient == nullptr ||
      packed_active_two_electron_gradient == nullptr) {
    throw std::invalid_argument("same-spin matrix backward output buffers must not be null");
  }

  const int pair_tile_size = std::min(
      n_unique_determinants,
      same_spin_backward_pair_tile_size());
  for (int left_begin = 0; left_begin < n_unique_determinants; left_begin += pair_tile_size) {
    const int left_end =
        std::min(n_unique_determinants, left_begin + pair_tile_size);
    for (int right_begin = 0;
         right_begin < n_unique_determinants;
         right_begin += pair_tile_size) {
      const int right_end =
          std::min(n_unique_determinants, right_begin + pair_tile_size);
      // Tile the ordered unique-pair sweep so sparse/support-aware same-spin
      // weights can skip large zero regions without touching every cached pair.
      const auto hamiltonian_weight_tile =
          hamiltonian_weight_matrix.block(
              left_begin,
              right_begin,
              left_end - left_begin,
              right_end - right_begin);
      const auto overlap_weight_tile =
          overlap_weight_matrix.block(
              left_begin,
              right_begin,
              left_end - left_begin,
              right_end - right_begin);
      const auto partner_total_transfer_tile =
          partner_total_transfer_matrix.block(
              left_begin,
              right_begin,
              left_end - left_begin,
              right_end - right_begin);
      if (!same_spin_weight_tile_has_any_weight(
              hamiltonian_weight_tile,
              overlap_weight_tile,
              partner_total_transfer_tile)) {
        continue;
      }
      accumulate_spin_matrix_backward_tile(
          unique_determinants,
          ordered_pair_cache,
          hamiltonian_weight_tile,
          overlap_weight_tile,
          partner_total_transfer_tile,
          left_begin,
          right_begin,
          n_unique_determinants,
          n_active_orbitals,
          active_one_electron_gradient,
          active_orbital_overlap_gradient,
          packed_active_two_electron_gradient);
    }
  }
}

template <
    typename HWeight,
    typename SWeight,
    typename TWeight,
    typename DHWeight,
    typename DSWeight,
    typename DTWeight>
void accumulate_spin_local_matrix_backward_tile(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    const std::vector<SameSpinPolynomialDirectionalPairData>& ordered_directional_data,
    const HWeight& hamiltonian_weight_tile,
    const SWeight& overlap_weight_tile,
    const TWeight& partner_total_transfer_tile,
    const DHWeight& delta_hamiltonian_weight_tile,
    const DSWeight& delta_overlap_weight_tile,
    const DTWeight& delta_partner_total_transfer_tile,
    int left_begin,
    int right_begin,
    int n_unique_determinants,
    int n_active_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& active_one_electron_matrix,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const std::vector<double>& delta_ao_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals,
    Eigen::MatrixXd* active_one_electron_gradient,
    std::vector<double>* active_orbital_overlap_gradient,
    std::vector<double>* packed_active_two_electron_gradient) {
  // Pair-local directional data still depends on the target ordered pair, but
  // all six scalar adjoints are tile-local. Reuse matrix workspaces across the
  // ordered-pair sweep to avoid allocating temporary inverse-gradient matrices
  // inside the hot loop.
  for (int left_local = 0;
       left_local < hamiltonian_weight_tile.rows();
       ++left_local) {
    const int left_id = left_begin + left_local;
    for (int right_local = 0;
         right_local < hamiltonian_weight_tile.cols();
         ++right_local) {
      const int right_id = right_begin + right_local;
      const double hamiltonian_weight =
          hamiltonian_weight_tile(left_local, right_local);
      const double overlap_weight =
          overlap_weight_tile(left_local, right_local);
      const double partner_total =
          partner_total_transfer_tile(left_local, right_local);
      const double delta_hamiltonian_weight =
          delta_hamiltonian_weight_tile(left_local, right_local);
      const double delta_overlap_weight =
          delta_overlap_weight_tile(left_local, right_local);
      const double delta_partner_total =
          delta_partner_total_transfer_tile(left_local, right_local);
      if (std::abs(hamiltonian_weight) <= kContributionTolerance &&
          std::abs(overlap_weight) <= kContributionTolerance &&
          std::abs(partner_total) <= kContributionTolerance &&
          std::abs(delta_hamiltonian_weight) <= kContributionTolerance &&
          std::abs(delta_overlap_weight) <= kContributionTolerance &&
          std::abs(delta_partner_total) <= kContributionTolerance) {
        continue;
      }

      const auto& pair_evaluation =
          ordered_pair_cache[ordered_spin_pair_storage_index(
              left_id,
              right_id,
              n_unique_determinants)];
      const auto& occ_L = unique_determinants[left_id];
      const auto& occ_R = unique_determinants[right_id];
      const auto& overlap_result = pair_evaluation.overlap_result;

      const SameSpinPolynomialDirectionalPairData& directional_data =
          ordered_directional_data[ordered_spin_pair_storage_index(
              left_id, right_id, n_unique_determinants)];

      accumulate_directional_one_electron_gradient_contribution_local(
          occ_L,
          occ_R,
          directional_data.cofactor_1st,
          directional_data.delta_cofactor_1st,
          hamiltonian_weight,
          delta_hamiltonian_weight,
          active_one_electron_gradient);
      accumulate_directional_deleted_minor_same_spin_two_electron_gradient_contribution_local(
          occ_L,
          occ_R,
          cached_cofactor_differential(pair_evaluation),
          delta_ao_overlap_matrix,
          n_active_orbitals,
          hamiltonian_weight,
          delta_hamiltonian_weight,
          packed_active_two_electron_gradient);

      if (pair_evaluation.same_spin_overlap_hamiltonian_gradient.rows() !=
              static_cast<int>(occ_R.size()) ||
          pair_evaluation.same_spin_overlap_hamiltonian_gradient.cols() !=
              static_cast<int>(occ_L.size())) {
        throw std::runtime_error(
            "same-spin local-response requires cached overlap Hamiltonian gradients");
      }
      accumulate_overlap_block_gradient_contribution_local(
          occ_L,
          occ_R,
          pair_evaluation.same_spin_overlap_hamiltonian_gradient,
          delta_hamiltonian_weight,
          n_active_orbitals,
          active_orbital_overlap_gradient);
      accumulate_overlap_block_gradient_contribution_local(
          occ_L,
          occ_R,
          directional_data.delta_same_spin_overlap_hamiltonian_gradient,
          hamiltonian_weight,
          n_active_orbitals,
          active_orbital_overlap_gradient);

      const double determinant_overlap_weight =
          overlap_weight + partner_total;
      const double delta_determinant_overlap_weight =
          delta_overlap_weight + delta_partner_total;
      if (directional_data.cofactor_1st.size() != 0) {
        accumulate_overlap_block_gradient_contribution_local(
            occ_L,
            occ_R,
            directional_data.cofactor_1st,
            delta_determinant_overlap_weight,
            n_active_orbitals,
            active_orbital_overlap_gradient);
      }
      if (directional_data.delta_cofactor_1st.size() != 0 &&
          std::abs(determinant_overlap_weight) > kContributionTolerance) {
        accumulate_overlap_block_gradient_contribution_local(
            occ_L,
            occ_R,
            directional_data.delta_cofactor_1st,
            determinant_overlap_weight,
            n_active_orbitals,
            active_orbital_overlap_gradient);
      }
    }
  }
}

void accumulate_spin_local_matrix_backward(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    const std::vector<SameSpinPolynomialDirectionalPairData>& ordered_directional_data,
    const Eigen::MatrixXd& hamiltonian_weight_matrix,
    const Eigen::MatrixXd& overlap_weight_matrix,
    const Eigen::MatrixXd& partner_total_transfer_matrix,
    const Eigen::MatrixXd& delta_hamiltonian_weight_matrix,
    const Eigen::MatrixXd& delta_overlap_weight_matrix,
    const Eigen::MatrixXd& delta_partner_total_transfer_matrix,
    int n_unique_determinants,
    int n_active_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& active_one_electron_matrix,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const std::vector<double>& delta_ao_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals,
    Eigen::MatrixXd* active_one_electron_gradient,
    std::vector<double>* active_orbital_overlap_gradient,
    std::vector<double>* packed_active_two_electron_gradient) {
  // The compressed same-spin local weights are symmetric on the unique-spin
  // spaces, but the local-response kernels still depend on the ordered
  // left/right role of each unique-determinant pair through the directional
  // cofactors and overlap-gradient adjoints. Reusing the accepted-state
  // half-pair shortcut here corrupts the SSO/HHO local response. Sweep the
  // full ordered pair grid just like the validated accepted/directional
  // same-spin backward path.
  if (active_one_electron_gradient == nullptr ||
      active_orbital_overlap_gradient == nullptr ||
      packed_active_two_electron_gradient == nullptr) {
    throw std::invalid_argument(
        "same-spin local matrix backward output buffers must not be null");
  }

  const int pair_tile_size = std::min(
      n_unique_determinants,
      same_spin_backward_pair_tile_size());
  for (int left_begin = 0; left_begin < n_unique_determinants; left_begin += pair_tile_size) {
    const int left_end =
        std::min(n_unique_determinants, left_begin + pair_tile_size);
    for (int right_begin = 0;
         right_begin < n_unique_determinants;
         right_begin += pair_tile_size) {
      const int right_end =
          std::min(n_unique_determinants, right_begin + pair_tile_size);
      if (!same_spin_local_tile_has_any_weight(
              hamiltonian_weight_matrix,
              overlap_weight_matrix,
              partner_total_transfer_matrix,
              delta_hamiltonian_weight_matrix,
              delta_overlap_weight_matrix,
              delta_partner_total_transfer_matrix,
              left_begin,
              left_end,
              right_begin,
              right_end)) {
        continue;
      }

      for (int left_id = left_begin; left_id < left_end; ++left_id) {
        for (int right_id = right_begin; right_id < right_end; ++right_id) {
          const double hamiltonian_weight =
              hamiltonian_weight_matrix(left_id, right_id);
          const double overlap_weight =
              overlap_weight_matrix(left_id, right_id);
          const double partner_total =
              partner_total_transfer_matrix(left_id, right_id);
          const double delta_hamiltonian_weight =
              delta_hamiltonian_weight_matrix(left_id, right_id);
          const double delta_overlap_weight =
              delta_overlap_weight_matrix(left_id, right_id);
          const double delta_partner_total =
              delta_partner_total_transfer_matrix(left_id, right_id);
          if (std::abs(hamiltonian_weight) <= kContributionTolerance &&
              std::abs(overlap_weight) <= kContributionTolerance &&
              std::abs(partner_total) <= kContributionTolerance &&
              std::abs(delta_hamiltonian_weight) <= kContributionTolerance &&
              std::abs(delta_overlap_weight) <= kContributionTolerance &&
              std::abs(delta_partner_total) <= kContributionTolerance) {
            continue;
          }

          const auto& pair_evaluation =
              ordered_pair_cache[ordered_spin_pair_storage_index(
                  left_id,
                  right_id,
                  n_unique_determinants)];
          const auto& occ_L = unique_determinants[left_id];
          const auto& occ_R = unique_determinants[right_id];
          const auto& overlap_result = pair_evaluation.overlap_result;

          const SameSpinPolynomialDirectionalPairData& directional_data =
              ordered_directional_data[ordered_spin_pair_storage_index(
                  left_id, right_id, n_unique_determinants)];

          accumulate_directional_one_electron_gradient_contribution_local(
              occ_L,
              occ_R,
              directional_data.cofactor_1st,
              directional_data.delta_cofactor_1st,
              hamiltonian_weight,
              delta_hamiltonian_weight,
              active_one_electron_gradient);
          accumulate_directional_deleted_minor_same_spin_two_electron_gradient_contribution_local(
              occ_L,
              occ_R,
              cached_cofactor_differential(pair_evaluation),
              delta_ao_overlap_matrix,
              n_active_orbitals,
              hamiltonian_weight,
              delta_hamiltonian_weight,
              packed_active_two_electron_gradient);

          if (pair_evaluation.same_spin_overlap_hamiltonian_gradient.rows() !=
                  static_cast<int>(occ_R.size()) ||
              pair_evaluation.same_spin_overlap_hamiltonian_gradient.cols() !=
                  static_cast<int>(occ_L.size())) {
            throw std::runtime_error(
                "same-spin local-response requires cached overlap Hamiltonian gradients");
          }
          accumulate_overlap_block_gradient_contribution_local(
              occ_L,
              occ_R,
              pair_evaluation.same_spin_overlap_hamiltonian_gradient,
              delta_hamiltonian_weight,
              n_active_orbitals,
              active_orbital_overlap_gradient);
          accumulate_overlap_block_gradient_contribution_local(
              occ_L,
              occ_R,
              directional_data.delta_same_spin_overlap_hamiltonian_gradient,
              hamiltonian_weight,
              n_active_orbitals,
              active_orbital_overlap_gradient);

          const double determinant_overlap_weight =
              overlap_weight + partner_total;
          const double delta_determinant_overlap_weight =
              delta_overlap_weight + delta_partner_total;
          if (directional_data.cofactor_1st.size() != 0) {
            accumulate_overlap_block_gradient_contribution_local(
                occ_L,
                occ_R,
                directional_data.cofactor_1st,
                delta_determinant_overlap_weight,
                n_active_orbitals,
                active_orbital_overlap_gradient);
          }
          if (directional_data.delta_cofactor_1st.size() != 0 &&
              std::abs(determinant_overlap_weight) > kContributionTolerance) {
            accumulate_overlap_block_gradient_contribution_local(
                occ_L,
                occ_R,
                directional_data.delta_cofactor_1st,
                determinant_overlap_weight,
                n_active_orbitals,
                active_orbital_overlap_gradient);
          }
        }
      }
    }
  }
}

SameSpinMatrixBackwardContribution
build_support_sparse_same_spin_backward_contribution_by_tiles(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies,
    int n_active_orbitals) {
  // Support-sparse accepted backward in true tile form:
  //   build W_H/W_S/W_partner only for the current unique-spin tile,
  //   consume that tile immediately in pair-cache backward,
  //   then discard the tile workspace.
  //
  // This removes the persistent O(N_unique^2) same-spin weight matrices from
  // the accepted-point support-sparse path while keeping the exact pair-local
  // gradient formulas unchanged.
  SameSpinMatrixBackwardContribution result;
  result.active_orbital_overlap_gradient.assign(
      square_storage_size(n_active_orbitals),
      0.0);
  result.active_one_electron_gradient.assign(
      square_storage_size(n_active_orbitals),
      0.0);
  result.packed_active_two_electron_gradient.assign(
      packed_active_two_electron_integral_count(n_active_orbitals),
      0.0);

  Eigen::MatrixXd active_one_electron_gradient =
      Eigen::MatrixXd::Zero(n_active_orbitals, n_active_orbitals);
  const bool close_shell_same_spin =
      same_spin_pair_cache.close_shell_reuses_same_spin_pair_cache();
  const int tile_size = same_spin_backward_pair_tile_size();
  SameSpinAcceptedTileWeights tile_weights;

  const int alpha_tile_size =
      std::min(selected_states.n_unique_alpha, tile_size);
  for (int left_begin = 0;
       left_begin < selected_states.n_unique_alpha;
       left_begin += alpha_tile_size) {
    const int left_end =
        std::min(selected_states.n_unique_alpha, left_begin + alpha_tile_size);
    for (int right_begin = 0;
         right_begin < selected_states.n_unique_alpha;
         right_begin += alpha_tile_size) {
      const int right_end =
          std::min(selected_states.n_unique_alpha, right_begin + alpha_tile_size);
      accumulate_alpha_accepted_tile_weights(
          selected_states,
          selected_state_energies,
          close_shell_same_spin
              ? same_spin_pair_cache.alpha_pair_cache_ref()
              : same_spin_pair_cache.beta_pair_cache_ref(),
          close_shell_same_spin
              ? selected_states.n_unique_alpha
              : selected_states.n_unique_beta,
          left_begin,
          left_end,
          right_begin,
          right_end,
          &tile_weights);
      if (!same_spin_weight_tile_has_any_weight(
              tile_weights.hamiltonian,
              tile_weights.overlap,
              tile_weights.partner_total)) {
        continue;
      }
      accumulate_spin_matrix_backward_tile(
          same_spin_pair_cache.alpha_reuse_table.unique_determinants,
          same_spin_pair_cache.alpha_pair_cache_ref(),
          tile_weights.hamiltonian,
          tile_weights.overlap,
          tile_weights.partner_total,
          left_begin,
          right_begin,
          selected_states.n_unique_alpha,
          n_active_orbitals,
          &active_one_electron_gradient,
          &result.active_orbital_overlap_gradient,
          &result.packed_active_two_electron_gradient);
    }
  }

  if (close_shell_same_spin) {
    active_one_electron_gradient *= 2.0;
    for (double& value : result.active_orbital_overlap_gradient) {
      value *= 2.0;
    }
    for (double& value : result.packed_active_two_electron_gradient) {
      value *= 2.0;
    }
  } else {
    const int beta_tile_size =
        std::min(selected_states.n_unique_beta, tile_size);
    for (int left_begin = 0;
         left_begin < selected_states.n_unique_beta;
         left_begin += beta_tile_size) {
      const int left_end =
          std::min(selected_states.n_unique_beta, left_begin + beta_tile_size);
      for (int right_begin = 0;
           right_begin < selected_states.n_unique_beta;
           right_begin += beta_tile_size) {
        const int right_end =
            std::min(selected_states.n_unique_beta, right_begin + beta_tile_size);
        accumulate_beta_accepted_tile_weights(
            selected_states,
            selected_state_energies,
            same_spin_pair_cache.alpha_pair_cache_ref(),
            selected_states.n_unique_alpha,
            left_begin,
            left_end,
            right_begin,
            right_end,
            &tile_weights);
        if (!same_spin_weight_tile_has_any_weight(
                tile_weights.hamiltonian,
                tile_weights.overlap,
                tile_weights.partner_total)) {
          continue;
        }
        accumulate_spin_matrix_backward_tile(
            same_spin_pair_cache.beta_reuse_table.unique_determinants,
            same_spin_pair_cache.beta_pair_cache_ref(),
            tile_weights.hamiltonian,
            tile_weights.overlap,
            tile_weights.partner_total,
            left_begin,
            right_begin,
            selected_states.n_unique_beta,
            n_active_orbitals,
            &active_one_electron_gradient,
            &result.active_orbital_overlap_gradient,
            &result.packed_active_two_electron_gradient);
      }
    }
  }

  result.active_one_electron_gradient.assign(
      active_one_electron_gradient.data(),
      active_one_electron_gradient.data() + active_one_electron_gradient.size());
  return result;
}

SameSpinMatrixBackwardContribution
build_support_sparse_directional_same_spin_backward_contribution_by_tiles(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    const std::vector<double>& selected_state_energies,
    const std::vector<double>& directional_selected_state_energies,
    int n_active_orbitals) {
  // Directional selected-state HVP in true tile form:
  //   build only dW_H/dW_S/dW_partner for the current unique-spin tile,
  //   consume that tile immediately in the same-spin pair-cache backward,
  //   and discard it. The product rule terms use support-local mixed
  //   contractions, so no union-support global weight matrix is allocated.
  SameSpinMatrixBackwardContribution result;
  result.active_orbital_overlap_gradient.assign(
      square_storage_size(n_active_orbitals),
      0.0);
  result.active_one_electron_gradient.assign(
      square_storage_size(n_active_orbitals),
      0.0);
  result.packed_active_two_electron_gradient.assign(
      packed_active_two_electron_integral_count(n_active_orbitals),
      0.0);

  Eigen::MatrixXd active_one_electron_gradient =
      Eigen::MatrixXd::Zero(n_active_orbitals, n_active_orbitals);
  const bool close_shell_same_spin =
      same_spin_pair_cache.close_shell_reuses_same_spin_pair_cache();
  const int tile_size = same_spin_backward_pair_tile_size();
  SameSpinAcceptedTileWeights tile_weights;

  const int alpha_tile_size =
      std::min(selected_states.n_unique_alpha, tile_size);
  for (int left_begin = 0;
       left_begin < selected_states.n_unique_alpha;
       left_begin += alpha_tile_size) {
    const int left_end =
        std::min(selected_states.n_unique_alpha, left_begin + alpha_tile_size);
    for (int right_begin = 0;
         right_begin < selected_states.n_unique_alpha;
         right_begin += alpha_tile_size) {
      const int right_end =
          std::min(selected_states.n_unique_alpha, right_begin + alpha_tile_size);
      accumulate_alpha_directional_tile_weights(
          selected_states,
          directional_selected_states,
          selected_state_energies,
          directional_selected_state_energies,
          close_shell_same_spin
              ? same_spin_pair_cache.alpha_pair_cache_ref()
              : same_spin_pair_cache.beta_pair_cache_ref(),
          close_shell_same_spin
              ? selected_states.n_unique_alpha
              : selected_states.n_unique_beta,
          left_begin,
          left_end,
          right_begin,
          right_end,
          &tile_weights);
      if (!same_spin_weight_tile_has_any_weight(
              tile_weights.hamiltonian,
              tile_weights.overlap,
              tile_weights.partner_total)) {
        continue;
      }
      accumulate_spin_matrix_backward_tile(
          same_spin_pair_cache.alpha_reuse_table.unique_determinants,
          same_spin_pair_cache.alpha_pair_cache_ref(),
          tile_weights.hamiltonian,
          tile_weights.overlap,
          tile_weights.partner_total,
          left_begin,
          right_begin,
          selected_states.n_unique_alpha,
          n_active_orbitals,
          &active_one_electron_gradient,
          &result.active_orbital_overlap_gradient,
          &result.packed_active_two_electron_gradient);
    }
  }

  if (close_shell_same_spin) {
    active_one_electron_gradient *= 2.0;
    for (double& value : result.active_orbital_overlap_gradient) {
      value *= 2.0;
    }
    for (double& value : result.packed_active_two_electron_gradient) {
      value *= 2.0;
    }
  } else {
    const int beta_tile_size =
        std::min(selected_states.n_unique_beta, tile_size);
    for (int left_begin = 0;
         left_begin < selected_states.n_unique_beta;
         left_begin += beta_tile_size) {
      const int left_end =
          std::min(selected_states.n_unique_beta, left_begin + beta_tile_size);
      for (int right_begin = 0;
           right_begin < selected_states.n_unique_beta;
           right_begin += beta_tile_size) {
        const int right_end =
            std::min(selected_states.n_unique_beta, right_begin + beta_tile_size);
        accumulate_beta_directional_tile_weights(
            selected_states,
            directional_selected_states,
            selected_state_energies,
            directional_selected_state_energies,
            same_spin_pair_cache.alpha_pair_cache_ref(),
            selected_states.n_unique_alpha,
            left_begin,
            left_end,
            right_begin,
            right_end,
            &tile_weights);
        if (!same_spin_weight_tile_has_any_weight(
                tile_weights.hamiltonian,
                tile_weights.overlap,
                tile_weights.partner_total)) {
          continue;
        }
        accumulate_spin_matrix_backward_tile(
            same_spin_pair_cache.beta_reuse_table.unique_determinants,
            same_spin_pair_cache.beta_pair_cache_ref(),
            tile_weights.hamiltonian,
            tile_weights.overlap,
            tile_weights.partner_total,
            left_begin,
            right_begin,
            selected_states.n_unique_beta,
            n_active_orbitals,
            &active_one_electron_gradient,
            &result.active_orbital_overlap_gradient,
            &result.packed_active_two_electron_gradient);
      }
    }
  }

  result.active_one_electron_gradient.assign(
      active_one_electron_gradient.data(),
      active_one_electron_gradient.data() + active_one_electron_gradient.size());
  return result;
}

SameSpinMatrixBackwardContribution
build_support_sparse_local_same_spin_backward_contribution_by_tiles(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies,
    int n_active_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& active_one_electron_matrix,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const std::vector<double>& delta_ao_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals,
    const SameSpinDirectionalPairCache* precomputed_directional_pair_cache) {
  // Local HVP tile path:
  //   keep only partner directional scalar matrices,
  //   build accepted/directional W tiles on demand from support-local C blocks,
  //   immediately consume those tiles in pair-local directional backward.
  //
  // This removes the global accepted W and local-response dW matrices from the
  // support-sparse local HVP path. Tile workspaces are declared once and reset
  // per tile rather than reallocated inside the pair loops.
  SameSpinMatrixBackwardContribution result;
  result.active_orbital_overlap_gradient.assign(
      square_storage_size(n_active_orbitals),
      0.0);
  result.active_one_electron_gradient.assign(
      square_storage_size(n_active_orbitals),
      0.0);
  result.packed_active_two_electron_gradient.assign(
      packed_active_two_electron_integral_count(n_active_orbitals),
      0.0);

  Eigen::MatrixXd active_one_electron_gradient =
      Eigen::MatrixXd::Zero(n_active_orbitals, n_active_orbitals);
  const bool close_shell_same_spin =
      same_spin_pair_cache.close_shell_reuses_same_spin_pair_cache();
  SameSpinDirectionalPairCache owned_directional_pair_cache;
  if (precomputed_directional_pair_cache == nullptr) {
    owned_directional_pair_cache = build_same_spin_directional_pair_cache(
        same_spin_pair_cache,
        n_active_orbitals,
        active_one_electron_matrix,
        active_space_two_electron_result,
        delta_ao_overlap_matrix,
        delta_active_one_electron_matrix,
        delta_packed_active_two_electron_integrals);
    precomputed_directional_pair_cache = &owned_directional_pair_cache;
  }
  if (precomputed_directional_pair_cache->close_shell_same_spin !=
      close_shell_same_spin) {
    throw std::invalid_argument(
        "precomputed same-spin directional cache has inconsistent spin sharing");
  }
  const auto& alpha_directional_scalars =
      precomputed_directional_pair_cache->alpha;
  const auto& beta_directional_scalars = close_shell_same_spin
      ? precomputed_directional_pair_cache->alpha
      : precomputed_directional_pair_cache->beta;

  const int tile_size = same_spin_backward_pair_tile_size();
  SameSpinLocalTileWeights tile_weights;
  const int alpha_tile_size =
      std::min(selected_states.n_unique_alpha, tile_size);
  for (int left_begin = 0;
       left_begin < selected_states.n_unique_alpha;
       left_begin += alpha_tile_size) {
    const int left_end =
        std::min(selected_states.n_unique_alpha, left_begin + alpha_tile_size);
    for (int right_begin = 0;
         right_begin < selected_states.n_unique_alpha;
         right_begin += alpha_tile_size) {
      const int right_end =
          std::min(selected_states.n_unique_alpha, right_begin + alpha_tile_size);
      accumulate_alpha_local_tile_weights(
          selected_states,
          selected_state_energies,
          close_shell_same_spin
              ? same_spin_pair_cache.alpha_pair_cache_ref()
              : same_spin_pair_cache.beta_pair_cache_ref(),
          close_shell_same_spin
              ? alpha_directional_scalars
              : beta_directional_scalars,
          close_shell_same_spin
              ? selected_states.n_unique_alpha
              : selected_states.n_unique_beta,
          left_begin,
          left_end,
          right_begin,
          right_end,
          &tile_weights);
      if (!same_spin_local_weight_tile_has_any_weight(
              tile_weights.hamiltonian,
              tile_weights.overlap,
              tile_weights.partner_total,
              tile_weights.delta_hamiltonian,
              tile_weights.delta_overlap,
              tile_weights.delta_partner_total)) {
        continue;
      }
      accumulate_spin_local_matrix_backward_tile(
          same_spin_pair_cache.alpha_reuse_table.unique_determinants,
          same_spin_pair_cache.alpha_pair_cache_ref(),
          alpha_directional_scalars.ordered_pair_data,
          tile_weights.hamiltonian,
          tile_weights.overlap,
          tile_weights.partner_total,
          tile_weights.delta_hamiltonian,
          tile_weights.delta_overlap,
          tile_weights.delta_partner_total,
          left_begin,
          right_begin,
          selected_states.n_unique_alpha,
          n_active_orbitals,
          active_one_electron_matrix,
          active_space_two_electron_result,
          delta_ao_overlap_matrix,
          delta_active_one_electron_matrix,
          delta_packed_active_two_electron_integrals,
          &active_one_electron_gradient,
          &result.active_orbital_overlap_gradient,
          &result.packed_active_two_electron_gradient);
    }
  }

  if (close_shell_same_spin) {
    active_one_electron_gradient *= 2.0;
    for (double& value : result.active_orbital_overlap_gradient) {
      value *= 2.0;
    }
    for (double& value : result.packed_active_two_electron_gradient) {
      value *= 2.0;
    }
  } else {
    const int beta_tile_size =
        std::min(selected_states.n_unique_beta, tile_size);
    for (int left_begin = 0;
         left_begin < selected_states.n_unique_beta;
         left_begin += beta_tile_size) {
      const int left_end =
          std::min(selected_states.n_unique_beta, left_begin + beta_tile_size);
      for (int right_begin = 0;
           right_begin < selected_states.n_unique_beta;
           right_begin += beta_tile_size) {
        const int right_end =
            std::min(selected_states.n_unique_beta, right_begin + beta_tile_size);
        accumulate_beta_local_tile_weights(
            selected_states,
            selected_state_energies,
            same_spin_pair_cache.alpha_pair_cache_ref(),
            alpha_directional_scalars,
            selected_states.n_unique_alpha,
            left_begin,
            left_end,
            right_begin,
            right_end,
            &tile_weights);
        if (!same_spin_local_weight_tile_has_any_weight(
                tile_weights.hamiltonian,
                tile_weights.overlap,
                tile_weights.partner_total,
                tile_weights.delta_hamiltonian,
                tile_weights.delta_overlap,
                tile_weights.delta_partner_total)) {
          continue;
        }
        accumulate_spin_local_matrix_backward_tile(
            same_spin_pair_cache.beta_reuse_table.unique_determinants,
            same_spin_pair_cache.beta_pair_cache_ref(),
            beta_directional_scalars.ordered_pair_data,
            tile_weights.hamiltonian,
            tile_weights.overlap,
            tile_weights.partner_total,
            tile_weights.delta_hamiltonian,
            tile_weights.delta_overlap,
            tile_weights.delta_partner_total,
            left_begin,
            right_begin,
            selected_states.n_unique_beta,
            n_active_orbitals,
            active_one_electron_matrix,
            active_space_two_electron_result,
            delta_ao_overlap_matrix,
            delta_active_one_electron_matrix,
            delta_packed_active_two_electron_integrals,
            &active_one_electron_gradient,
            &result.active_orbital_overlap_gradient,
            &result.packed_active_two_electron_gradient);
      }
    }
  }

  result.active_one_electron_gradient.assign(
      active_one_electron_gradient.data(),
      active_one_electron_gradient.data() + active_one_electron_gradient.size());
  return result;
}

}  // namespace

SameSpinDirectionalPairCache build_same_spin_directional_pair_cache(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    int n_active_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& active_one_electron_matrix,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const std::vector<double>& delta_ao_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals) {
  SameSpinDirectionalPairCache result;
  result.close_shell_same_spin =
      same_spin_pair_cache.close_shell_reuses_same_spin_pair_cache();
  result.alpha = build_directional_pair_scalar_matrices(
      same_spin_pair_cache.alpha_reuse_table.unique_determinants,
      same_spin_pair_cache.alpha_pair_cache_ref(),
      static_cast<int>(
          same_spin_pair_cache.alpha_reuse_table.unique_determinants.size()),
      n_active_orbitals,
      active_one_electron_matrix,
      active_space_two_electron_result,
      delta_ao_overlap_matrix,
      delta_active_one_electron_matrix,
      delta_packed_active_two_electron_integrals);
  if (!result.close_shell_same_spin) {
    result.beta = build_directional_pair_scalar_matrices(
        same_spin_pair_cache.beta_reuse_table.unique_determinants,
        same_spin_pair_cache.beta_pair_cache_ref(),
        static_cast<int>(
            same_spin_pair_cache.beta_reuse_table.unique_determinants.size()),
        n_active_orbitals,
        active_one_electron_matrix,
        active_space_two_electron_result,
        delta_ao_overlap_matrix,
        delta_active_one_electron_matrix,
        delta_packed_active_two_electron_integrals);
  }
  return result;
}

SameSpinMatrixBackwardContribution build_same_spin_matrix_backward_contribution(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies,
    int n_active_orbitals) {
  validate_full_matrix_same_spin_inputs(
      same_spin_pair_cache,
      selected_states,
      selected_state_energies);

  if (should_use_support_sparse_selected_state_contractions(selected_states) &&
      should_use_same_spin_accepted_tile_backward(selected_states)) {
    return build_support_sparse_same_spin_backward_contribution_by_tiles(
        same_spin_pair_cache,
        selected_states,
        selected_state_energies,
        n_active_orbitals);
  }

  // Compress the selected-state determinant coefficients directly onto the
  // unique alpha/beta same-spin channels. This keeps the backward algebra exact
  // while replacing the previous full determinant-pair scatter with dense
  // matrix products on the unique-spin spaces.
  const SameSpinExactWeightMatrices exact_weight_matrices =
      build_exact_same_spin_weight_matrices(
          same_spin_pair_cache,
          selected_states,
          selected_state_energies);
  const bool close_shell_same_spin =
      same_spin_pair_cache.close_shell_reuses_same_spin_pair_cache();
  const Eigen::MatrixXd alpha_total_partner_transfer_matrix =
      exact_weight_matrices.alpha_partner_total_transfer_matrix +
      exact_weight_matrices.alpha_singular_partner_transfer_matrix;

  SameSpinMatrixBackwardContribution result;
  result.active_orbital_overlap_gradient.assign(
      square_storage_size(n_active_orbitals),
      0.0);
  result.active_one_electron_gradient.assign(
      square_storage_size(n_active_orbitals),
      0.0);
  result.packed_active_two_electron_gradient.assign(
      packed_active_two_electron_integral_count(n_active_orbitals),
      0.0);

  Eigen::MatrixXd active_one_electron_gradient =
      Eigen::MatrixXd::Zero(n_active_orbitals, n_active_orbitals);
  accumulate_spin_matrix_backward(
      same_spin_pair_cache.alpha_reuse_table.unique_determinants,
      same_spin_pair_cache.alpha_pair_cache_ref(),
      exact_weight_matrices.alpha_hamiltonian_weight_matrix,
      exact_weight_matrices.alpha_overlap_weight_matrix,
      alpha_total_partner_transfer_matrix,
      selected_states.n_unique_alpha,
      n_active_orbitals,
      &active_one_electron_gradient,
      &result.active_orbital_overlap_gradient,
      &result.packed_active_two_electron_gradient);
  if (close_shell_same_spin) {
    active_one_electron_gradient *= 2.0;
    for (double& value : result.active_orbital_overlap_gradient) {
      value *= 2.0;
    }
    for (double& value : result.packed_active_two_electron_gradient) {
      value *= 2.0;
    }
  } else {
    const Eigen::MatrixXd beta_total_partner_transfer_matrix =
        exact_weight_matrices.beta_partner_total_transfer_matrix +
        exact_weight_matrices.beta_singular_partner_transfer_matrix;
    accumulate_spin_matrix_backward(
        same_spin_pair_cache.beta_reuse_table.unique_determinants,
        same_spin_pair_cache.beta_pair_cache_ref(),
        exact_weight_matrices.beta_hamiltonian_weight_matrix,
        exact_weight_matrices.beta_overlap_weight_matrix,
        beta_total_partner_transfer_matrix,
        selected_states.n_unique_beta,
        n_active_orbitals,
        &active_one_electron_gradient,
        &result.active_orbital_overlap_gradient,
        &result.packed_active_two_electron_gradient);
  }

  result.active_one_electron_gradient.assign(
      active_one_electron_gradient.data(),
      active_one_electron_gradient.data() + active_one_electron_gradient.size());
  return result;
}

SameSpinMatrixBackwardContribution
build_directional_same_spin_matrix_backward_contribution(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    const std::vector<double>& selected_state_energies,
    const std::vector<double>& directional_selected_state_energies,
    int n_active_orbitals) {
  validate_full_matrix_same_spin_inputs(
      same_spin_pair_cache,
      selected_states,
      selected_state_energies);
  validate_directional_selected_state_inputs(
      selected_states,
      directional_selected_states,
      selected_state_energies,
      directional_selected_state_energies);

  if (should_use_support_sparse_selected_state_contractions(selected_states) &&
      should_use_same_spin_directional_tile_backward(selected_states)) {
    return build_support_sparse_directional_same_spin_backward_contribution_by_tiles(
        same_spin_pair_cache,
        selected_states,
        directional_selected_states,
        selected_state_energies,
        directional_selected_state_energies,
        n_active_orbitals);
  }

  const SameSpinExactWeightMatrices directional_weight_matrices =
      should_use_support_sparse_selected_state_contractions(selected_states)
          ? build_support_sparse_directional_exact_same_spin_weight_matrices(
                same_spin_pair_cache,
                selected_states,
                directional_selected_states,
                selected_state_energies,
                directional_selected_state_energies)
          : build_dense_directional_exact_same_spin_weight_matrices(
                same_spin_pair_cache,
                selected_states,
                directional_selected_states,
                selected_state_energies,
                directional_selected_state_energies);
  const bool close_shell_same_spin =
      same_spin_pair_cache.close_shell_reuses_same_spin_pair_cache();
  const Eigen::MatrixXd alpha_total_partner_transfer_matrix =
      directional_weight_matrices.alpha_partner_total_transfer_matrix +
      directional_weight_matrices.alpha_singular_partner_transfer_matrix;

  SameSpinMatrixBackwardContribution result;
  result.active_orbital_overlap_gradient.assign(
      square_storage_size(n_active_orbitals),
      0.0);
  result.active_one_electron_gradient.assign(
      square_storage_size(n_active_orbitals),
      0.0);
  result.packed_active_two_electron_gradient.assign(
      packed_active_two_electron_integral_count(n_active_orbitals),
      0.0);

  Eigen::MatrixXd active_one_electron_gradient =
      Eigen::MatrixXd::Zero(n_active_orbitals, n_active_orbitals);
  accumulate_spin_matrix_backward(
      same_spin_pair_cache.alpha_reuse_table.unique_determinants,
      same_spin_pair_cache.alpha_pair_cache_ref(),
      directional_weight_matrices.alpha_hamiltonian_weight_matrix,
      directional_weight_matrices.alpha_overlap_weight_matrix,
      alpha_total_partner_transfer_matrix,
      selected_states.n_unique_alpha,
      n_active_orbitals,
      &active_one_electron_gradient,
      &result.active_orbital_overlap_gradient,
      &result.packed_active_two_electron_gradient);
  if (close_shell_same_spin) {
    active_one_electron_gradient *= 2.0;
    for (double& value : result.active_orbital_overlap_gradient) {
      value *= 2.0;
    }
    for (double& value : result.packed_active_two_electron_gradient) {
      value *= 2.0;
    }
  } else {
    const Eigen::MatrixXd beta_total_partner_transfer_matrix =
        directional_weight_matrices.beta_partner_total_transfer_matrix +
        directional_weight_matrices.beta_singular_partner_transfer_matrix;
    accumulate_spin_matrix_backward(
        same_spin_pair_cache.beta_reuse_table.unique_determinants,
        same_spin_pair_cache.beta_pair_cache_ref(),
        directional_weight_matrices.beta_hamiltonian_weight_matrix,
        directional_weight_matrices.beta_overlap_weight_matrix,
        beta_total_partner_transfer_matrix,
        selected_states.n_unique_beta,
        n_active_orbitals,
        &active_one_electron_gradient,
        &result.active_orbital_overlap_gradient,
        &result.packed_active_two_electron_gradient);
  }

  result.active_one_electron_gradient.assign(
      active_one_electron_gradient.data(),
      active_one_electron_gradient.data() + active_one_electron_gradient.size());
  return result;
}

SameSpinMatrixBackwardContribution
build_local_same_spin_matrix_backward_contribution(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies,
    int n_active_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& active_one_electron_matrix,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const std::vector<double>& delta_ao_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals,
    const SameSpinDirectionalPairCache* precomputed_directional_pair_cache) {
  // Matrix-form local same-spin HVP:
  // 1. compress accepted determinant/structure adjoints onto unique-spin pair weights,
  // 2. compress partner directional scalars onto `δW`,
  // 3. sweep only the ordered unique-spin pairs to accumulate the exact local
  //    response on `(δS_act, δh_act, δg_act)`.
  validate_full_matrix_same_spin_inputs(
      same_spin_pair_cache,
      selected_states,
      selected_state_energies);

  if (should_use_support_sparse_selected_state_contractions(selected_states) &&
      should_use_same_spin_local_tile_backward(selected_states)) {
    return build_support_sparse_local_same_spin_backward_contribution_by_tiles(
        same_spin_pair_cache,
        selected_states,
        selected_state_energies,
        n_active_orbitals,
        active_one_electron_matrix,
        active_space_two_electron_result,
        delta_ao_overlap_matrix,
        delta_active_one_electron_matrix,
        delta_packed_active_two_electron_integrals,
        precomputed_directional_pair_cache);
  }

  const SameSpinExactWeightMatrices exact_weight_matrices =
      build_exact_same_spin_weight_matrices(
          same_spin_pair_cache,
          selected_states,
          selected_state_energies);
  const bool close_shell_same_spin =
      same_spin_pair_cache.close_shell_reuses_same_spin_pair_cache();
  const Eigen::MatrixXd alpha_total_partner_transfer_matrix =
      exact_weight_matrices.alpha_partner_total_transfer_matrix +
      exact_weight_matrices.alpha_singular_partner_transfer_matrix;

  SameSpinDirectionalPairCache owned_directional_pair_cache;
  if (precomputed_directional_pair_cache == nullptr) {
    owned_directional_pair_cache = build_same_spin_directional_pair_cache(
        same_spin_pair_cache,
        n_active_orbitals,
        active_one_electron_matrix,
        active_space_two_electron_result,
        delta_ao_overlap_matrix,
        delta_active_one_electron_matrix,
        delta_packed_active_two_electron_integrals);
    precomputed_directional_pair_cache = &owned_directional_pair_cache;
  }
  if (precomputed_directional_pair_cache->close_shell_same_spin !=
      close_shell_same_spin) {
    throw std::invalid_argument(
        "precomputed same-spin directional cache has inconsistent spin sharing");
  }
  const auto& alpha_directional_scalars =
      precomputed_directional_pair_cache->alpha;
  const auto& beta_directional_scalars = close_shell_same_spin
      ? precomputed_directional_pair_cache->alpha
      : precomputed_directional_pair_cache->beta;
  const SameSpinLocalResponseWeightMatrices local_weight_matrices =
      build_local_same_spin_response_weight_matrices(
          selected_states,
          selected_state_energies,
          alpha_directional_scalars,
          close_shell_same_spin
              ? alpha_directional_scalars
              : beta_directional_scalars,
          close_shell_same_spin);

  SameSpinMatrixBackwardContribution result;
  result.active_orbital_overlap_gradient.assign(
      square_storage_size(n_active_orbitals),
      0.0);
  result.active_one_electron_gradient.assign(
      square_storage_size(n_active_orbitals),
      0.0);
  result.packed_active_two_electron_gradient.assign(
      packed_active_two_electron_integral_count(n_active_orbitals),
      0.0);

  Eigen::MatrixXd active_one_electron_gradient =
      Eigen::MatrixXd::Zero(n_active_orbitals, n_active_orbitals);
  accumulate_spin_local_matrix_backward(
      same_spin_pair_cache.alpha_reuse_table.unique_determinants,
      same_spin_pair_cache.alpha_pair_cache_ref(),
      alpha_directional_scalars.ordered_pair_data,
      exact_weight_matrices.alpha_hamiltonian_weight_matrix,
      exact_weight_matrices.alpha_overlap_weight_matrix,
      alpha_total_partner_transfer_matrix,
      local_weight_matrices.alpha_delta_hamiltonian_weight_matrix,
      local_weight_matrices.alpha_delta_overlap_weight_matrix,
      local_weight_matrices.alpha_delta_partner_total_transfer_matrix,
      selected_states.n_unique_alpha,
      n_active_orbitals,
      active_one_electron_matrix,
      active_space_two_electron_result,
      delta_ao_overlap_matrix,
      delta_active_one_electron_matrix,
      delta_packed_active_two_electron_integrals,
      &active_one_electron_gradient,
      &result.active_orbital_overlap_gradient,
      &result.packed_active_two_electron_gradient);
  if (close_shell_same_spin) {
    active_one_electron_gradient *= 2.0;
    for (double& value : result.active_orbital_overlap_gradient) {
      value *= 2.0;
    }
    for (double& value : result.packed_active_two_electron_gradient) {
      value *= 2.0;
    }
  } else {
    const Eigen::MatrixXd beta_total_partner_transfer_matrix =
        exact_weight_matrices.beta_partner_total_transfer_matrix +
        exact_weight_matrices.beta_singular_partner_transfer_matrix;
    accumulate_spin_local_matrix_backward(
        same_spin_pair_cache.beta_reuse_table.unique_determinants,
        same_spin_pair_cache.beta_pair_cache_ref(),
        beta_directional_scalars.ordered_pair_data,
        exact_weight_matrices.beta_hamiltonian_weight_matrix,
        exact_weight_matrices.beta_overlap_weight_matrix,
        beta_total_partner_transfer_matrix,
        local_weight_matrices.beta_delta_hamiltonian_weight_matrix,
        local_weight_matrices.beta_delta_overlap_weight_matrix,
        local_weight_matrices.beta_delta_partner_total_transfer_matrix,
        selected_states.n_unique_beta,
        n_active_orbitals,
        active_one_electron_matrix,
        active_space_two_electron_result,
        delta_ao_overlap_matrix,
        delta_active_one_electron_matrix,
        delta_packed_active_two_electron_integrals,
        &active_one_electron_gradient,
        &result.active_orbital_overlap_gradient,
        &result.packed_active_two_electron_gradient);
  }

  result.active_one_electron_gradient.assign(
      active_one_electron_gradient.data(),
      active_one_electron_gradient.data() + active_one_electron_gradient.size());
  return result;
}

}  // namespace xmvb::vb
