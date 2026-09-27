#include "vbscf/derivatives/hessian/responses/opposite_spin/backward.hpp"

#include <chrono>
#include <stdexcept>

#include "vbscf/derivatives/hessian/responses/opposite_spin/pair_response_internal.hpp"
#include "vbscf/derivatives/hessian/responses/opposite_spin/overlap_contractions_internal.hpp"
#include "vbscf/derivatives/hessian/responses/opposite_spin/packed_contractions_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/backward.hpp"
#include "vbscf/determinants/pairs/contractions.hpp"
#include "vbscf/integrals/active/two_electron/construction/kernel.hpp"
#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"

namespace xmvb::vb {
namespace {

int require_packed_pair_count(
    const SameSpinPairCacheContext& same_spin_pair_cache) {
  const int alpha_count = infer_n_packed_active_pairs(
      same_spin_pair_cache.alpha_pair_cache_ref(),
      "alpha");
  const int beta_count = infer_n_packed_active_pairs(
      same_spin_pair_cache.beta_pair_cache_ref(),
      "beta");
  if (alpha_count != beta_count) {
    throw std::invalid_argument(
        "alpha/beta same-spin caches disagree on n_packed_active_pairs");
  }
  return alpha_count;
}

void validate_backward_inputs(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states) {
  if (!same_spin_pair_cache.enabled()) {
    throw std::invalid_argument(
        "pair-graph opposite-spin backward requires an enabled same-spin cache");
  }
  if (selected_states.n_unique_alpha !=
          static_cast<int>(same_spin_pair_cache.alpha_reuse_table.unique_determinants.size()) ||
      selected_states.n_unique_beta !=
          static_cast<int>(same_spin_pair_cache.beta_reuse_table.unique_determinants.size())) {
    throw std::invalid_argument(
        "selected-state dimensions do not match same-spin cache reuse tables");
  }
}

void validate_directional_states(
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states) {
  if (selected_states.n_unique_alpha != directional_selected_states.n_unique_alpha ||
      selected_states.n_unique_beta != directional_selected_states.n_unique_beta ||
      selected_states.n_determinants != directional_selected_states.n_determinants ||
      selected_states.selected_state_indices !=
          directional_selected_states.selected_state_indices ||
      selected_states.states.size() != directional_selected_states.states.size()) {
    throw std::invalid_argument(
        "directional selected-state matrices do not match accepted-point dimensions");
  }
}

OppositeSpinBackwardContribution make_zero_contribution(
    int n_active_orbitals) {
  OppositeSpinBackwardContribution result;
  result.active_orbital_overlap_gradient.assign(
      static_cast<std::size_t>(n_active_orbitals) *
          static_cast<std::size_t>(n_active_orbitals),
      0.0);
  result.packed_active_two_electron_gradient.assign(
      packed_active_two_electron_integral_count(n_active_orbitals),
      0.0);
  return result;
}

void combine_overlap_channels(
    const std::vector<double>& alpha,
    const std::vector<double>& beta,
    std::vector<double>* result) {
  for (std::size_t index = 0; index < result->size(); ++index) {
    (*result)[index] = alpha[index] + beta[index];
  }
}

}  // namespace

OppositeSpinBackwardContribution
build_opposite_spin_backward_contribution(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result) {
  validate_backward_inputs(same_spin_pair_cache, selected_states);
  OppositeSpinBackwardContribution result =
      make_zero_contribution(n_active_orbitals);
  const int n_packed_pairs = require_packed_pair_count(same_spin_pair_cache);
  if (n_packed_pairs == 0) {
    return result;
  }
  std::vector<double> alpha_overlap(
      result.active_orbital_overlap_gradient.size(), 0.0);
  std::vector<double> beta_overlap(
      result.active_orbital_overlap_gradient.size(), 0.0);
  detail::accumulate_opposite_spin_packed_gradient_by_pair_graph(
      same_spin_pair_cache,
      selected_states,
      n_packed_pairs,
      &result.packed_active_two_electron_gradient);
  detail::accumulate_alpha_overlap_gradient_by_pair_graph(
      same_spin_pair_cache,
      selected_states,
      n_active_orbitals,
      active_space_two_electron_result,
      &alpha_overlap);
  detail::accumulate_beta_overlap_gradient_by_pair_graph(
      same_spin_pair_cache,
      selected_states,
      n_active_orbitals,
      active_space_two_electron_result,
      &beta_overlap);
  combine_overlap_channels(
      alpha_overlap,
      beta_overlap,
      &result.active_orbital_overlap_gradient);
  return result;
}

OppositeSpinBackwardContribution
build_directional_opposite_spin_backward_contribution(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result) {
  validate_backward_inputs(same_spin_pair_cache, selected_states);
  validate_directional_states(selected_states, directional_selected_states);
  OppositeSpinBackwardContribution result =
      make_zero_contribution(n_active_orbitals);
  const int n_packed_pairs = require_packed_pair_count(same_spin_pair_cache);
  if (n_packed_pairs == 0) {
    return result;
  }
  std::vector<double> alpha_overlap(
      result.active_orbital_overlap_gradient.size(), 0.0);
  std::vector<double> beta_overlap(
      result.active_orbital_overlap_gradient.size(), 0.0);
  auto start = std::chrono::steady_clock::now();
  detail::accumulate_directional_opposite_spin_packed_gradient_by_pair_graph(
      same_spin_pair_cache,
      selected_states,
      directional_selected_states,
      n_packed_pairs,
      &result.packed_active_two_electron_gradient);
  result.timing.packed_gradient_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start).count();
  start = std::chrono::steady_clock::now();
  detail::accumulate_directional_alpha_overlap_gradient(
      same_spin_pair_cache,
      selected_states,
      directional_selected_states,
      n_active_orbitals,
      active_space_two_electron_result,
      &alpha_overlap);
  result.timing.alpha_overlap_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start).count();
  start = std::chrono::steady_clock::now();
  detail::accumulate_directional_beta_overlap_gradient(
      same_spin_pair_cache,
      selected_states,
      directional_selected_states,
      n_active_orbitals,
      active_space_two_electron_result,
      &beta_overlap);
  result.timing.beta_overlap_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start).count();
  combine_overlap_channels(
      alpha_overlap,
      beta_overlap,
      &result.active_orbital_overlap_gradient);
  return result;
}

}  // namespace xmvb::vb
