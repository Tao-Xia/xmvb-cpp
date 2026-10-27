#pragma once

#include <cstdint>

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

void accumulate_alpha_accepted_tile_weights(
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies,
    const std::vector<SpinDeterminantPairEvaluation>& partner_pair_cache,
    std::int64_t n_unique_partner,
    std::int64_t left_begin,
    std::int64_t left_end,
    std::int64_t right_begin,
    std::int64_t right_end,
    SameSpinAcceptedTileWeights* weights);

void accumulate_beta_accepted_tile_weights(
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies,
    const std::vector<SpinDeterminantPairEvaluation>& partner_pair_cache,
    std::int64_t n_unique_partner,
    std::int64_t left_begin,
    std::int64_t left_end,
    std::int64_t right_begin,
    std::int64_t right_end,
    SameSpinAcceptedTileWeights* weights);

void accumulate_alpha_directional_tile_weights(
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    const std::vector<double>& selected_state_energies,
    const std::vector<double>& directional_selected_state_energies,
    const std::vector<SpinDeterminantPairEvaluation>& partner_pair_cache,
    std::int64_t n_unique_partner,
    std::int64_t left_begin,
    std::int64_t left_end,
    std::int64_t right_begin,
    std::int64_t right_end,
    SameSpinAcceptedTileWeights* weights);

void accumulate_beta_directional_tile_weights(
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    const std::vector<double>& selected_state_energies,
    const std::vector<double>& directional_selected_state_energies,
    const std::vector<SpinDeterminantPairEvaluation>& partner_pair_cache,
    std::int64_t n_unique_partner,
    std::int64_t left_begin,
    std::int64_t left_end,
    std::int64_t right_begin,
    std::int64_t right_end,
    SameSpinAcceptedTileWeights* weights);

} // namespace xmvb::vb::detail
