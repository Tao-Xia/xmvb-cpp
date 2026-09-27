#include "vbscf/derivatives/hessian/responses/same_spin/tile_weights_internal.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/responses/same_spin/tile_policy_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/weight_kernels_internal.hpp"
#include "vbscf/determinants/pairs/contractions.hpp"
#include "vbscf/structures/assembly/local_contractions.hpp"

namespace xmvb::vb::detail {

namespace {

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

}  // namespace

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
    for_each_sparse_coefficient_pair_in_tile(
        state_coefficients.local_sparse_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        state_coefficients.local_sparse_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        left_begin,
        left_end,
        right_begin,
        right_end,
        [&](int row, int column, int partner_left, int partner_right,
            double coefficient_product) {
          const double overlap = same_spin_pair_overlap_scalar(
              partner_pair_cache,
              n_unique_partner,
              partner_left,
              partner_right);
          const double total = same_spin_pair_total_scalar(
              partner_pair_cache,
              n_unique_partner,
              partner_left,
              partner_right);
          weights->hamiltonian(row, column) +=
              state_weight * coefficient_product * overlap;
          weights->overlap(row, column) +=
              overlap_weight * coefficient_product * overlap;
          weights->partner_total(row, column) +=
              state_weight * coefficient_product * total;
        });
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
    for_each_sparse_coefficient_pair_in_tile(
        state_coefficients.local_sparse_coefficient_transpose,
        state_coefficients.beta_support,
        state_coefficients.alpha_support,
        state_coefficients.local_sparse_coefficient_transpose,
        state_coefficients.beta_support,
        state_coefficients.alpha_support,
        left_begin,
        left_end,
        right_begin,
        right_end,
        [&](int row, int column, int partner_left, int partner_right,
            double coefficient_product) {
          const double overlap = same_spin_pair_overlap_scalar(
              partner_pair_cache,
              n_unique_partner,
              partner_left,
              partner_right);
          const double total = same_spin_pair_total_scalar(
              partner_pair_cache,
              n_unique_partner,
              partner_left,
              partner_right);
          weights->hamiltonian(row, column) +=
              state_weight * coefficient_product * overlap;
          weights->overlap(row, column) +=
              overlap_weight * coefficient_product * overlap;
          weights->partner_total(row, column) +=
              state_weight * coefficient_product * total;
        });
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
    for_each_sparse_coefficient_pair_in_tile(
        state_coefficients.local_sparse_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        state_coefficients.local_sparse_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        left_begin,
        left_end,
        right_begin,
        right_end,
        [&](int row, int column, int partner_left, int partner_right,
            double coefficient_product) {
          weights->overlap(row, column) -=
              state_weight * directional_state_energy *
              coefficient_product * same_spin_pair_overlap_scalar(
                  partner_pair_cache,
                  n_unique_partner,
                  partner_left,
                  partner_right);
        });

    if (!selected_state_has_local_support(directional_state_coefficients)) {
      continue;
    }
    validate_local_state_coefficient_matrix(directional_state_coefficients);
    const auto accumulate_mixed = [&](
        const SparseLocalCoefficientMatrix& left_coefficients,
        const std::vector<int>& left_alpha_support,
        const std::vector<int>& left_beta_support,
        const SparseLocalCoefficientMatrix& right_coefficients,
        const std::vector<int>& right_alpha_support,
        const std::vector<int>& right_beta_support) {
      for_each_sparse_coefficient_pair_in_tile(
          left_coefficients,
          left_alpha_support,
          left_beta_support,
          right_coefficients,
          right_alpha_support,
          right_beta_support,
          left_begin,
          left_end,
          right_begin,
          right_end,
          [&](int row, int column, int partner_left, int partner_right,
              double coefficient_product) {
            const double overlap = same_spin_pair_overlap_scalar(
                partner_pair_cache,
                n_unique_partner,
                partner_left,
                partner_right);
            const double total = same_spin_pair_total_scalar(
                partner_pair_cache,
                n_unique_partner,
                partner_left,
                partner_right);
            weights->hamiltonian(row, column) +=
                state_weight * coefficient_product * overlap;
            weights->overlap(row, column) -=
                state_weight * state_energy * coefficient_product * overlap;
            weights->partner_total(row, column) +=
                state_weight * coefficient_product * total;
          });
    };
    accumulate_mixed(
        directional_state_coefficients.local_sparse_coefficient_matrix,
        directional_state_coefficients.alpha_support,
        directional_state_coefficients.beta_support,
        state_coefficients.local_sparse_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support);
    accumulate_mixed(
        state_coefficients.local_sparse_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        directional_state_coefficients.local_sparse_coefficient_matrix,
        directional_state_coefficients.alpha_support,
        directional_state_coefficients.beta_support);
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
    for_each_sparse_coefficient_pair_in_tile(
        state_coefficients.local_sparse_coefficient_transpose,
        state_coefficients.beta_support,
        state_coefficients.alpha_support,
        state_coefficients.local_sparse_coefficient_transpose,
        state_coefficients.beta_support,
        state_coefficients.alpha_support,
        left_begin,
        left_end,
        right_begin,
        right_end,
        [&](int row, int column, int partner_left, int partner_right,
            double coefficient_product) {
          weights->overlap(row, column) -=
              state_weight * directional_state_energy *
              coefficient_product * same_spin_pair_overlap_scalar(
                  partner_pair_cache,
                  n_unique_partner,
                  partner_left,
                  partner_right);
        });

    if (!selected_state_has_local_support(directional_state_coefficients)) {
      continue;
    }
    validate_local_state_coefficient_matrix(directional_state_coefficients);
    const auto accumulate_mixed = [&](
        const SparseLocalCoefficientMatrix& left_coefficients,
        const std::vector<int>& left_beta_support,
        const std::vector<int>& left_alpha_support,
        const SparseLocalCoefficientMatrix& right_coefficients,
        const std::vector<int>& right_beta_support,
        const std::vector<int>& right_alpha_support) {
      for_each_sparse_coefficient_pair_in_tile(
          left_coefficients,
          left_beta_support,
          left_alpha_support,
          right_coefficients,
          right_beta_support,
          right_alpha_support,
          left_begin,
          left_end,
          right_begin,
          right_end,
          [&](int row, int column, int partner_left, int partner_right,
              double coefficient_product) {
            const double overlap = same_spin_pair_overlap_scalar(
                partner_pair_cache,
                n_unique_partner,
                partner_left,
                partner_right);
            const double total = same_spin_pair_total_scalar(
                partner_pair_cache,
                n_unique_partner,
                partner_left,
                partner_right);
            weights->hamiltonian(row, column) +=
                state_weight * coefficient_product * overlap;
            weights->overlap(row, column) -=
                state_weight * state_energy * coefficient_product * overlap;
            weights->partner_total(row, column) +=
                state_weight * coefficient_product * total;
          });
    };
    accumulate_mixed(
        directional_state_coefficients.local_sparse_coefficient_transpose,
        directional_state_coefficients.beta_support,
        directional_state_coefficients.alpha_support,
        state_coefficients.local_sparse_coefficient_transpose,
        state_coefficients.beta_support,
        state_coefficients.alpha_support);
    accumulate_mixed(
        state_coefficients.local_sparse_coefficient_transpose,
        state_coefficients.beta_support,
        state_coefficients.alpha_support,
        directional_state_coefficients.local_sparse_coefficient_transpose,
        directional_state_coefficients.beta_support,
        directional_state_coefficients.alpha_support);
  }
}

}  // namespace xmvb::vb::detail
