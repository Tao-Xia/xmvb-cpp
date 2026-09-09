#include "vbscf/derivatives/hessian/responses/opposite_spin_backward.hpp"

#include <algorithm>
#include <stdexcept>

#include "vbscf/derivatives/hessian/responses/opposite_spin_pair_response_internal.hpp"
#include "vbscf/derivatives/hessian/responses/opposite_spin_contractions_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin_response.hpp"
#include "vbscf/determinants/spin_pair_contractions.hpp"
#include "vbscf/integrals/active/active_space_two_electron_kernel.hpp"
#include "vbscf/integrals/active/two_electron_indexer.hpp"

namespace xmvb::vb {
namespace {

struct PackedGradientBlocking {
  int sparse_block_size = 0;
  int dense_batch_size = 0;
};

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

OppositeSpinMatrixBackwardContribution make_zero_contribution(
    int n_active_orbitals) {
  OppositeSpinMatrixBackwardContribution result;
  result.active_orbital_overlap_gradient.assign(
      static_cast<std::size_t>(n_active_orbitals) *
          static_cast<std::size_t>(n_active_orbitals),
      0.0);
  result.packed_active_two_electron_gradient.assign(
      packed_active_two_electron_integral_count(n_active_orbitals),
      0.0);
  return result;
}

PackedGradientBlocking choose_packed_gradient_blocking(int n_packed_pairs) {
  return {
      std::min(n_packed_pairs, 32),
      std::min(n_packed_pairs, 8)};
}

}  // namespace

OppositeSpinMatrixBackwardContribution
build_opposite_spin_matrix_backward_contribution(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_active_orbitals) {
  detail::validate_opposite_spin_backward_inputs(
      same_spin_pair_cache,
      selected_states);
  OppositeSpinMatrixBackwardContribution result =
      make_zero_contribution(n_active_orbitals);
  const int n_packed_pairs = require_packed_pair_count(same_spin_pair_cache);
  if (n_packed_pairs == 0) {
    return result;
  }
  const PackedGradientBlocking blocking =
      choose_packed_gradient_blocking(n_packed_pairs);

  detail::accumulate_opposite_spin_packed_gradient_by_tiles(
      same_spin_pair_cache,
      selected_states,
      n_packed_pairs,
      blocking.sparse_block_size,
      blocking.dense_batch_size,
      &result.packed_active_two_electron_gradient);
  detail::accumulate_alpha_overlap_gradient(
      same_spin_pair_cache,
      selected_states,
      n_active_orbitals,
      &result.active_orbital_overlap_gradient);
  detail::accumulate_beta_overlap_gradient(
      same_spin_pair_cache,
      selected_states,
      n_active_orbitals,
      &result.active_orbital_overlap_gradient);
  return result;
}

OppositeSpinMatrixBackwardContribution
build_directional_opposite_spin_matrix_backward_contribution(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    int n_active_orbitals) {
  detail::validate_opposite_spin_backward_inputs(
      same_spin_pair_cache,
      selected_states);
  detail::validate_directional_selected_state_inputs(
      selected_states,
      directional_selected_states);
  OppositeSpinMatrixBackwardContribution result =
      make_zero_contribution(n_active_orbitals);
  const int n_packed_pairs = require_packed_pair_count(same_spin_pair_cache);
  if (n_packed_pairs == 0) {
    return result;
  }
  const PackedGradientBlocking blocking =
      choose_packed_gradient_blocking(n_packed_pairs);

  detail::accumulate_directional_opposite_spin_packed_gradient_by_tiles(
      same_spin_pair_cache,
      selected_states,
      directional_selected_states,
      n_packed_pairs,
      blocking.sparse_block_size,
      blocking.dense_batch_size,
      &result.packed_active_two_electron_gradient);
  detail::accumulate_directional_alpha_overlap_gradient(
      same_spin_pair_cache,
      selected_states,
      directional_selected_states,
      n_active_orbitals,
      &result.active_orbital_overlap_gradient);
  detail::accumulate_directional_beta_overlap_gradient(
      same_spin_pair_cache,
      selected_states,
      directional_selected_states,
      n_active_orbitals,
      &result.active_orbital_overlap_gradient);
  return result;
}

OppositeSpinMatrixBackwardContribution
build_local_opposite_spin_matrix_backward_contribution(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const ActiveSpaceIntegralDirectionView& direction,
    const SameSpinDirectionalPairCache& directional_pair_cache) {
  detail::validate_opposite_spin_backward_inputs(
      same_spin_pair_cache,
      selected_states);
  OppositeSpinMatrixBackwardContribution result =
      make_zero_contribution(n_active_orbitals);
  const int n_packed_pairs = require_packed_pair_count(same_spin_pair_cache);
  if (n_packed_pairs == 0) {
    return result;
  }

  const auto alpha_directional_pairs =
      detail::build_directional_opposite_spin_pair_data(
          same_spin_pair_cache.alpha_reuse_table.unique_determinants,
          same_spin_pair_cache.alpha_pair_cache_ref(),
          selected_states.n_unique_alpha,
          n_active_orbitals,
          active_space_two_electron_result,
          direction,
          directional_pair_cache.alpha.ordered_pair_data);
  const auto& beta_same_spin_pairs =
      directional_pair_cache.close_shell_same_spin
          ? directional_pair_cache.alpha.ordered_pair_data
          : directional_pair_cache.beta.ordered_pair_data;
  const auto beta_directional_pairs =
      detail::build_directional_opposite_spin_pair_data(
          same_spin_pair_cache.beta_reuse_table.unique_determinants,
          same_spin_pair_cache.beta_pair_cache_ref(),
          selected_states.n_unique_beta,
          n_active_orbitals,
          active_space_two_electron_result,
          direction,
          beta_same_spin_pairs);
  const PackedGradientBlocking blocking =
      choose_packed_gradient_blocking(n_packed_pairs);

  detail::accumulate_local_opposite_spin_packed_gradient_by_tiles(
      same_spin_pair_cache,
      alpha_directional_pairs,
      beta_directional_pairs,
      selected_states,
      n_packed_pairs,
      blocking.sparse_block_size,
      blocking.dense_batch_size,
      &result.packed_active_two_electron_gradient);
  detail::accumulate_local_alpha_overlap_gradient(
      same_spin_pair_cache,
      alpha_directional_pairs,
      beta_directional_pairs,
      selected_states,
      n_active_orbitals,
      &result.active_orbital_overlap_gradient);
  detail::accumulate_local_beta_overlap_gradient(
      same_spin_pair_cache,
      alpha_directional_pairs,
      beta_directional_pairs,
      selected_states,
      n_active_orbitals,
      &result.active_orbital_overlap_gradient);
  return result;
}

}  // namespace xmvb::vb
