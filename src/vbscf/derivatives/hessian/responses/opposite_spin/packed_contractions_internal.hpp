#pragma once

#include <vector>

#include "vbscf/derivatives/hessian/responses/opposite_spin/pair_response_internal.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "vbscf/structures/assembly/selected_coefficients.hpp"

namespace xmvb::vb::detail {

void accumulate_opposite_spin_packed_gradient_by_tiles(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_packed_active_pairs,
    int sparse_block_size,
    int dense_batch_size,
    std::vector<double>* packed_active_two_electron_gradient);

void accumulate_directional_opposite_spin_packed_gradient_by_tiles(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    int n_packed_active_pairs,
    int sparse_block_size,
    int dense_batch_size,
    std::vector<double>* packed_active_two_electron_gradient);

void accumulate_local_opposite_spin_packed_gradient_by_tiles(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const std::vector<DirectionalOppositeSpinPairData>&
        alpha_directional_pair_data,
    const std::vector<DirectionalOppositeSpinPairData>&
        beta_directional_pair_data,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_packed_active_pairs,
    int sparse_block_size,
    int dense_batch_size,
    std::vector<double>* packed_active_two_electron_gradient);

}  // namespace xmvb::vb::detail
