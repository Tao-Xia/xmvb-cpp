#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/responses/same_spin_response.hpp"

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

void accumulate_alpha_accepted_tile_weights(
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies,
    const std::vector<SpinDeterminantPairEvaluation>& partner_pair_cache,
    int n_unique_partner,
    int left_begin,
    int left_end,
    int right_begin,
    int right_end,
    SameSpinAcceptedTileWeights* weights);

void accumulate_beta_accepted_tile_weights(
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies,
    const std::vector<SpinDeterminantPairEvaluation>& partner_pair_cache,
    int n_unique_partner,
    int left_begin,
    int left_end,
    int right_begin,
    int right_end,
    SameSpinAcceptedTileWeights* weights);

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
    SameSpinLocalTileWeights* weights);

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
    SameSpinLocalTileWeights* weights);

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
    SameSpinAcceptedTileWeights* weights);

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
    SameSpinAcceptedTileWeights* weights);

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

SameSpinDirectionalScalarMatrices build_directional_pair_scalar_matrices(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    int n_unique_determinants,
    int n_active_orbitals,
    const ActiveSpaceIntegralDirectionView& direction);

SameSpinLocalResponseWeightMatrices build_local_same_spin_response_weight_matrices(
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies,
    const SameSpinDirectionalScalarMatrices& alpha_directional_scalars,
    const SameSpinDirectionalScalarMatrices& beta_directional_scalars,
    bool close_shell_same_spin);

}  // namespace xmvb::vb::detail
