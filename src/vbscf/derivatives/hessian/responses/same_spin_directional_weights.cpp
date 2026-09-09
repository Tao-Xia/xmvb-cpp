#include "vbscf/derivatives/hessian/responses/same_spin_matrix_weights_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin_tile_weights_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin_weight_kernels_internal.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "vbscf/determinants/spin_pair_contractions.hpp"
#include "vbscf/structures/support_local_contractions.hpp"

namespace xmvb::vb::detail {

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

}  // namespace xmvb::vb::detail
