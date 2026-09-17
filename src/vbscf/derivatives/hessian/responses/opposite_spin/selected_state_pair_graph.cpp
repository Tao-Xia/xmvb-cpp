#include "vbscf/derivatives/hessian/responses/opposite_spin/selected_state_pair_graph_internal.hpp"

#include <cstddef>

namespace xmvb::vb::detail {

SelectedStatePairGraph::SelectedStatePairGraph(
    const SelectedStateDeterminantMatrices& selected_states,
    PrimarySpin primary_spin) {
  const int n_primary = primary_spin == PrimarySpin::Alpha
      ? selected_states.n_unique_alpha
      : selected_states.n_unique_beta;
  terms_.reserve(selected_states.states.size());
  for (const auto& state : selected_states.states) {
    add_term(
        state,
        state,
        state.normalized_state_weight,
        primary_spin,
        n_primary);
  }
}

SelectedStatePairGraph::SelectedStatePairGraph(
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    PrimarySpin primary_spin) {
  const int n_primary = primary_spin == PrimarySpin::Alpha
      ? selected_states.n_unique_alpha
      : selected_states.n_unique_beta;
  terms_.reserve(2 * selected_states.states.size());
  for (std::size_t state_index = 0;
       state_index < selected_states.states.size();
       ++state_index) {
    const auto& accepted = selected_states.states[state_index];
    const auto& directional = directional_selected_states.states[state_index];
    add_term(
        directional,
        accepted,
        accepted.normalized_state_weight,
        primary_spin,
        n_primary);
    add_term(
        accepted,
        directional,
        accepted.normalized_state_weight,
        primary_spin,
        n_primary);
  }
}

void SelectedStatePairGraph::add_term(
    const SelectedStateDeterminantCoefficients& left,
    const SelectedStateDeterminantCoefficients& right,
    double weight,
    PrimarySpin primary_spin,
    int n_primary) {
  const bool alpha_primary = primary_spin == PrimarySpin::Alpha;
  terms_.emplace_back();
  auto& term = terms_.back();
  term.left_coefficients = alpha_primary
      ? &left.local_sparse_coefficient_matrix
      : &left.local_sparse_coefficient_transpose;
  term.right_coefficients = alpha_primary
      ? &right.local_sparse_coefficient_matrix
      : &right.local_sparse_coefficient_transpose;
  const auto& left_primary_support = alpha_primary
      ? left.alpha_support
      : left.beta_support;
  const auto& right_primary_support = alpha_primary
      ? right.alpha_support
      : right.beta_support;
  term.left_partner_support = alpha_primary
      ? &left.beta_support
      : &left.alpha_support;
  term.right_partner_support = alpha_primary
      ? &right.beta_support
      : &right.alpha_support;
  term.left_primary_to_local.assign(n_primary, -1);
  term.right_primary_to_local.assign(n_primary, -1);
  for (int local = 0;
       local < static_cast<int>(left_primary_support.size());
       ++local) {
    term.left_primary_to_local[left_primary_support[local]] = local;
  }
  for (int local = 0;
       local < static_cast<int>(right_primary_support.size());
       ++local) {
    term.right_primary_to_local[right_primary_support[local]] = local;
  }
  term.weight = weight;
}

void SelectedStatePairGraph::accumulate_partner_projection(
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

  for (const auto& term : terms_) {
    accumulate_row_product(
        *term.left_coefficients,
        term.left_primary_to_local[primary_left],
        *term.left_partner_support,
        *term.right_coefficients,
        term.right_primary_to_local[primary_right],
        *term.right_partner_support,
        term.weight);
  }
}

}  // namespace xmvb::vb::detail
