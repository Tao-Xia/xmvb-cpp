#pragma once

#include <vector>

#include "vbscf/derivatives/hessian/responses/opposite_spin_pair_response_internal.hpp"
#include "vbscf/determinants/same_spin_pair_cache.hpp"
#include "vbscf/structures/selected_state_coefficients.hpp"

namespace xmvb::vb::detail {

void validate_opposite_spin_backward_inputs(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states);

void validate_directional_selected_state_inputs(
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states);

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

void accumulate_alpha_overlap_gradient(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_active_orbitals,
    std::vector<double>* active_orbital_overlap_gradient);

void accumulate_beta_overlap_gradient(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_active_orbitals,
    std::vector<double>* active_orbital_overlap_gradient);

void accumulate_directional_alpha_overlap_gradient(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    int n_active_orbitals,
    std::vector<double>* active_orbital_overlap_gradient);

void accumulate_directional_beta_overlap_gradient(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    int n_active_orbitals,
    std::vector<double>* active_orbital_overlap_gradient);

void accumulate_local_alpha_overlap_gradient(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const std::vector<DirectionalOppositeSpinPairData>&
        alpha_directional_pair_data,
    const std::vector<DirectionalOppositeSpinPairData>&
        beta_directional_pair_data,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_active_orbitals,
    std::vector<double>* active_orbital_overlap_gradient);

void accumulate_local_beta_overlap_gradient(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const std::vector<DirectionalOppositeSpinPairData>&
        alpha_directional_pair_data,
    const std::vector<DirectionalOppositeSpinPairData>&
        beta_directional_pair_data,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_active_orbitals,
    std::vector<double>* active_orbital_overlap_gradient);

}  // namespace xmvb::vb::detail
