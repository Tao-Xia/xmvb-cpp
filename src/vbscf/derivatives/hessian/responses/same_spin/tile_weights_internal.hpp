#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/responses/same_spin/backward.hpp"

namespace xmvb::vb::detail {

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

}  // namespace xmvb::vb::detail
