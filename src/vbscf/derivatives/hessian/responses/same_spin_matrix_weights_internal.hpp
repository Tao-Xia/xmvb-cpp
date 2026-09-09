#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/responses/same_spin_backward.hpp"

namespace xmvb::vb::detail {

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

struct SameSpinLocalResponseWeightMatrices {
  Eigen::MatrixXd alpha_delta_hamiltonian_weight_matrix;
  Eigen::MatrixXd alpha_delta_overlap_weight_matrix;
  Eigen::MatrixXd alpha_delta_partner_total_transfer_matrix;
  Eigen::MatrixXd beta_delta_hamiltonian_weight_matrix;
  Eigen::MatrixXd beta_delta_overlap_weight_matrix;
  Eigen::MatrixXd beta_delta_partner_total_transfer_matrix;
};

SameSpinExactWeightMatrices build_exact_same_spin_weight_matrices(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies);

void validate_directional_selected_state_inputs(
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    const std::vector<double>& selected_state_energies,
    const std::vector<double>& directional_selected_state_energies);

SameSpinExactWeightMatrices
build_dense_directional_exact_same_spin_weight_matrices(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    const std::vector<double>& selected_state_energies,
    const std::vector<double>& directional_selected_state_energies);

SameSpinExactWeightMatrices
build_support_sparse_directional_exact_same_spin_weight_matrices(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    const std::vector<double>& selected_state_energies,
    const std::vector<double>& directional_selected_state_energies);

void validate_full_matrix_same_spin_inputs(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies);

SameSpinLocalResponseWeightMatrices build_local_same_spin_response_weight_matrices(
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies,
    const SameSpinDirectionalScalarMatrices& alpha_directional_scalars,
    const SameSpinDirectionalScalarMatrices& beta_directional_scalars,
    bool close_shell_same_spin);

}  // namespace xmvb::vb::detail
