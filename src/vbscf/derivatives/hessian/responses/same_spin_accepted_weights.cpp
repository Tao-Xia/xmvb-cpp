#include "vbscf/derivatives/hessian/responses/same_spin_matrix_weights_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin_weight_kernels_internal.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "vbscf/determinants/spin_pair_contractions.hpp"

namespace xmvb::vb::detail {

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

}  // namespace xmvb::vb::detail
