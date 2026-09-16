#include "vbscf/derivatives/hessian/responses/opposite_spin/directional_pair_graph_internal.hpp"

#include <cstddef>

namespace xmvb::vb::detail {

DirectionalPairGraph::DirectionalPairGraph(
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    PrimarySpin primary_spin) {
  const bool alpha_primary = primary_spin == PrimarySpin::Alpha;
  const int n_primary = alpha_primary
      ? selected_states.n_unique_alpha
      : selected_states.n_unique_beta;
  states_.resize(selected_states.states.size());
  for (std::size_t state_index = 0;
       state_index < selected_states.states.size();
       ++state_index) {
    const auto& accepted = selected_states.states[state_index];
    const auto& directional = directional_selected_states.states[state_index];
    auto& state = states_[state_index];
    state.accepted_coefficients = alpha_primary
        ? &accepted.local_sparse_coefficient_matrix
        : &accepted.local_sparse_coefficient_transpose;
    state.directional_coefficients = alpha_primary
        ? &directional.local_sparse_coefficient_matrix
        : &directional.local_sparse_coefficient_transpose;
    const auto& accepted_primary_support = alpha_primary
        ? accepted.alpha_support
        : accepted.beta_support;
    const auto& directional_primary_support = alpha_primary
        ? directional.alpha_support
        : directional.beta_support;
    state.accepted_partner_support = alpha_primary
        ? &accepted.beta_support
        : &accepted.alpha_support;
    state.directional_partner_support = alpha_primary
        ? &directional.beta_support
        : &directional.alpha_support;
    state.accepted_primary_to_local.assign(n_primary, -1);
    state.directional_primary_to_local.assign(n_primary, -1);
    for (int local = 0;
         local < static_cast<int>(accepted_primary_support.size());
         ++local) {
      state.accepted_primary_to_local[accepted_primary_support[local]] = local;
    }
    for (int local = 0;
         local < static_cast<int>(directional_primary_support.size());
         ++local) {
      state.directional_primary_to_local[
          directional_primary_support[local]] = local;
    }
    state.weight = accepted.normalized_state_weight;
  }
}

void DirectionalPairGraph::accumulate_partner_projection(
    int primary_left,
    int primary_right,
    const std::vector<SpinDeterminantPairEvaluation>& partner_pair_cache,
    int n_unique_partner,
    std::vector<double>* partner_image,
    std::vector<unsigned char>* touched_flags,
    std::vector<int>* touched_channels) const {
  const auto accumulate_row_product =
      [&](const SparseLocalCoefficientMatrix& left_coefficients,
          int left_local,
          const std::vector<int>& left_partner_support,
          const SparseLocalCoefficientMatrix& right_coefficients,
          int right_local,
          const std::vector<int>& right_partner_support,
          double scale) {
        if (left_local < 0 || right_local < 0 || scale == 0.0) {
          return;
        }
        for (SparseLocalCoefficientMatrix::InnerIterator left_entry(
                 left_coefficients,
                 left_local);
             left_entry;
             ++left_entry) {
          const int partner_left = left_partner_support[left_entry.col()];
          for (SparseLocalCoefficientMatrix::InnerIterator right_entry(
                   right_coefficients,
                   right_local);
               right_entry;
               ++right_entry) {
            const int partner_right = right_partner_support[right_entry.col()];
            const double coefficient =
                scale * left_entry.value() * right_entry.value();
            const auto& projection =
                partner_pair_cache[ordered_spin_pair_storage_index(
                    partner_left,
                    partner_right,
                    n_unique_partner)]
                    .opposite_spin_pair_cache.first_order_cofactor_projection;
            for (std::size_t entry = 0;
                 entry < projection.packed_pair_indices.size();
                 ++entry) {
              const int channel = projection.packed_pair_indices[entry];
              if ((*touched_flags)[channel] == 0u) {
                (*touched_flags)[channel] = 1u;
                touched_channels->push_back(channel);
              }
              (*partner_image)[channel] +=
                  coefficient * projection.packed_pair_values[entry];
            }
          }
        }
      };

  for (const auto& state : states_) {
    accumulate_row_product(
        *state.directional_coefficients,
        state.directional_primary_to_local[primary_left],
        *state.directional_partner_support,
        *state.accepted_coefficients,
        state.accepted_primary_to_local[primary_right],
        *state.accepted_partner_support,
        state.weight);
    accumulate_row_product(
        *state.accepted_coefficients,
        state.accepted_primary_to_local[primary_left],
        *state.accepted_partner_support,
        *state.directional_coefficients,
        state.directional_primary_to_local[primary_right],
        *state.directional_partner_support,
        state.weight);
  }
}

}  // namespace xmvb::vb::detail
