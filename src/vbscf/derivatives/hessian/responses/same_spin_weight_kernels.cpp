#include "vbscf/derivatives/hessian/responses/same_spin_matrix_weights_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin_tile_weights_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin_tile_policy_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin_weight_kernels_internal.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <cblas.h>

#include "vbscf/derivatives/hessian/responses/same_spin_pair_response_internal.hpp"
#include "vbscf/determinants/spin_pair_contractions.hpp"
#include "vbscf/structures/assembly/local_contractions.hpp"
#include "vbscf/integrals/active/active_space_two_electron_kernel.hpp"

namespace xmvb::vb::detail {

namespace {

constexpr double kContributionTolerance = 1.0e-15;

std::size_t square_storage_size(int dimension) {
  return static_cast<std::size_t>(dimension) *
      static_cast<std::size_t>(dimension);
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



}  // namespace xmvb::vb::detail
