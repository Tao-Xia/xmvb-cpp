#include "vbscf/derivatives/hessian/responses/same_spin_backward.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/responses/same_spin_backward_kernels_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin_matrix_weights_internal.hpp"

namespace xmvb::vb {

using detail::account_for_close_shell_spin_reuse;
using detail::accumulate_spin_local_matrix_backward;
using detail::accumulate_spin_matrix_backward;
using detail::build_dense_directional_exact_same_spin_weight_matrices;
using detail::build_exact_same_spin_weight_matrices;
using detail::build_local_same_spin_response_weight_matrices;
using detail::build_support_sparse_directional_same_spin_backward_contribution_by_tiles;
using detail::build_support_sparse_local_same_spin_backward_contribution_by_tiles;
using detail::build_support_sparse_same_spin_backward_contribution_by_tiles;
using detail::finalize_backward_contribution;
using detail::make_zero_backward_contribution;
using detail::SameSpinExactWeightMatrices;
using detail::SameSpinLocalResponseWeightMatrices;
using detail::validate_directional_selected_state_inputs;
using detail::validate_full_matrix_same_spin_inputs;

SameSpinMatrixBackwardContribution build_same_spin_matrix_backward_contribution(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies,
    int n_active_orbitals) {
  validate_full_matrix_same_spin_inputs(
      same_spin_pair_cache,
      selected_states,
      selected_state_energies);

  if (should_use_support_sparse_selected_state_contractions(selected_states)) {
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

  SameSpinMatrixBackwardContribution result =
      make_zero_backward_contribution(n_active_orbitals);

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
    account_for_close_shell_spin_reuse(
        &active_one_electron_gradient,
        &result);
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

  return finalize_backward_contribution(
      std::move(result),
      active_one_electron_gradient);
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

  if (should_use_support_sparse_selected_state_contractions(selected_states)) {
    return build_support_sparse_directional_same_spin_backward_contribution_by_tiles(
        same_spin_pair_cache,
        selected_states,
        directional_selected_states,
        selected_state_energies,
        directional_selected_state_energies,
        n_active_orbitals);
  }

  const SameSpinExactWeightMatrices directional_weight_matrices =
      build_dense_directional_exact_same_spin_weight_matrices(
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

  SameSpinMatrixBackwardContribution result =
      make_zero_backward_contribution(n_active_orbitals);

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
    account_for_close_shell_spin_reuse(
        &active_one_electron_gradient,
        &result);
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

  return finalize_backward_contribution(
      std::move(result),
      active_one_electron_gradient);
}

SameSpinMatrixBackwardContribution
build_local_same_spin_matrix_backward_contribution(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies,
    int n_active_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& active_one_electron_matrix,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const ActiveSpaceIntegralDirectionView& direction,
    const SameSpinDirectionalPairCache& directional_pair_cache) {
  // Matrix-form local same-spin HVP:
  // 1. compress accepted determinant/structure adjoints onto unique-spin pair weights,
  // 2. compress partner directional scalars onto `δW`,
  // 3. sweep only the ordered unique-spin pairs to accumulate the exact local
  //    response on `(δS_act, δh_act, δg_act)`.
  validate_full_matrix_same_spin_inputs(
      same_spin_pair_cache,
      selected_states,
      selected_state_energies);

  if (should_use_support_sparse_selected_state_contractions(selected_states)) {
    return build_support_sparse_local_same_spin_backward_contribution_by_tiles(
        same_spin_pair_cache,
        selected_states,
        selected_state_energies,
        n_active_orbitals,
        active_one_electron_matrix,
        active_space_two_electron_result,
        direction,
        directional_pair_cache);
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

  if (directional_pair_cache.close_shell_same_spin !=
      close_shell_same_spin) {
    throw std::invalid_argument(
        "precomputed same-spin directional cache has inconsistent spin sharing");
  }
  const auto& alpha_directional_scalars =
      directional_pair_cache.alpha;
  const auto& beta_directional_scalars = close_shell_same_spin
      ? directional_pair_cache.alpha
      : directional_pair_cache.beta;
  const SameSpinLocalResponseWeightMatrices local_weight_matrices =
      build_local_same_spin_response_weight_matrices(
          selected_states,
          selected_state_energies,
          alpha_directional_scalars,
          close_shell_same_spin
              ? alpha_directional_scalars
              : beta_directional_scalars,
          close_shell_same_spin);

  SameSpinMatrixBackwardContribution result =
      make_zero_backward_contribution(n_active_orbitals);

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
      direction.overlap,
      &active_one_electron_gradient,
      &result.active_orbital_overlap_gradient,
      &result.packed_active_two_electron_gradient);
  if (close_shell_same_spin) {
    account_for_close_shell_spin_reuse(
        &active_one_electron_gradient,
        &result);
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
        direction.overlap,
        &active_one_electron_gradient,
        &result.active_orbital_overlap_gradient,
        &result.packed_active_two_electron_gradient);
  }

  return finalize_backward_contribution(
      std::move(result),
      active_one_electron_gradient);
}
}  // namespace xmvb::vb
