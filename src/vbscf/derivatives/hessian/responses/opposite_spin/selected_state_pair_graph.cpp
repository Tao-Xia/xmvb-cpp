#include "vbscf/derivatives/hessian/responses/opposite_spin/selected_state_pair_graph_internal.hpp"

#include <cstddef>
#include <stdexcept>

#include "vbscf/derivatives/hessian/responses/opposite_spin/pair_response_internal.hpp"

namespace xmvb::vb::detail {

namespace {

void accumulate_projected_channels(
    const OppositeSpinPackedPairProjection& projection,
    double coefficient,
    const std::vector<int>& target_channels,
    std::vector<double>* target_values) {
  if (projection.packed_pair_indices.empty()) {
    if (!projection.packed_pair_values.empty() ||
        !projection.projected_pair_values.empty()) {
      throw std::logic_error(
          "empty opposite-spin projection has inconsistent cached values");
    }
    return;
  }
  if (projection.packed_pair_indices.size() !=
      projection.packed_pair_values.size()) {
    throw std::logic_error(
        "opposite-spin sparse projection has inconsistent dimensions");
  }
  if (target_values == nullptr ||
      target_values->size() != target_channels.size()) {
    throw std::invalid_argument(
        "opposite-spin projected-channel target has inconsistent dimensions");
  }
  for (std::size_t target = 0; target < target_channels.size(); ++target) {
    const int channel = target_channels[target];
    if (channel < 0 ||
        channel >= static_cast<int>(projection.projected_pair_values.size())) {
      throw std::logic_error(
          "nonzero opposite-spin projection is missing its dense kernel image");
    }
    (*target_values)[target] +=
        coefficient * projection.projected_pair_values[channel];
  }
}

}  // namespace

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

template <typename AccumulatePair>
void SelectedStatePairGraph::for_each_partner_pair(
    int primary_left,
    int primary_right,
    AccumulatePair&& accumulate_pair) const {
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
            accumulate_pair(partner_left, partner_right, coefficient);
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

template <typename ProjectionAt>
void SelectedStatePairGraph::accumulate_partner_projection_impl(
    int primary_left,
    int primary_right,
    ProjectionAt&& projection_at,
    std::vector<double>* partner_image,
    std::vector<unsigned char>* touched_flags,
    std::vector<int>* touched_channels) const {
  for_each_partner_pair(
      primary_left,
      primary_right,
      [&](int partner_left, int partner_right, double coefficient) {
        const auto& projection = projection_at(partner_left, partner_right);
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
      });
}

void SelectedStatePairGraph::accumulate_partner_projection(
    int primary_left,
    int primary_right,
    const std::vector<SpinDeterminantPairEvaluation>& partner_pair_cache,
    int n_unique_partner,
    std::vector<double>* partner_image,
    std::vector<unsigned char>* touched_flags,
    std::vector<int>* touched_channels) const {
  accumulate_partner_projection_impl(
      primary_left,
      primary_right,
      [&](int partner_left, int partner_right) -> const auto& {
        return partner_pair_cache[ordered_spin_pair_storage_index(
            partner_left,
            partner_right,
            n_unique_partner)]
            .opposite_spin_pair_cache.first_order_cofactor_projection;
      },
      partner_image,
      touched_flags,
      touched_channels);
}

void SelectedStatePairGraph::accumulate_partner_projected_values(
    int primary_left,
    int primary_right,
    const std::vector<SpinDeterminantPairEvaluation>& partner_pair_cache,
    int n_unique_partner,
    const std::vector<int>& target_channels,
    std::vector<double>* target_values) const {
  for_each_partner_pair(
      primary_left,
      primary_right,
      [&](int partner_left, int partner_right, double coefficient) {
        const auto& projection =
            partner_pair_cache[ordered_spin_pair_storage_index(
                partner_left,
                partner_right,
                n_unique_partner)]
                .opposite_spin_pair_cache.first_order_cofactor_projection;
        accumulate_projected_channels(
            projection,
            coefficient,
            target_channels,
            target_values);
      });
}

void SelectedStatePairGraph::accumulate_partner_projected_values(
    int primary_left,
    int primary_right,
    const std::vector<DirectionalOppositeSpinPairData>& partner_pair_data,
    int n_unique_partner,
    const std::vector<int>& target_channels,
    std::vector<double>* target_values) const {
  for_each_partner_pair(
      primary_left,
      primary_right,
      [&](int partner_left, int partner_right, double coefficient) {
        const auto& projection =
            partner_pair_data[ordered_spin_pair_storage_index(
                partner_left,
                partner_right,
                n_unique_partner)]
                .delta_first_order_cofactor_projection;
        accumulate_projected_channels(
            projection,
            coefficient,
            target_channels,
            target_values);
      });
}

void SelectedStatePairGraph::accumulate_partner_projection(
    int primary_left,
    int primary_right,
    const std::vector<DirectionalOppositeSpinPairData>& partner_pair_data,
    int n_unique_partner,
    std::vector<double>* partner_image,
    std::vector<unsigned char>* touched_flags,
    std::vector<int>* touched_channels) const {
  accumulate_partner_projection_impl(
      primary_left,
      primary_right,
      [&](int partner_left, int partner_right) -> const auto& {
        return partner_pair_data[ordered_spin_pair_storage_index(
            partner_left,
            partner_right,
            n_unique_partner)]
            .delta_first_order_cofactor_projection;
      },
      partner_image,
      touched_flags,
      touched_channels);
}

}  // namespace xmvb::vb::detail
