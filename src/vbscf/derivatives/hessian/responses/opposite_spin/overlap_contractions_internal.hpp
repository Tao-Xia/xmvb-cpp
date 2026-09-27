#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/responses/opposite_spin/pair_response_internal.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "vbscf/structures/assembly/selected_coefficients.hpp"

namespace xmvb::vb::detail {

/** Accumulates one ordered primary-pair term of the local overlap adjoint. */
void accumulate_pair_overlap_gradient_direction(
    const std::vector<int>& occupied_left,
    const std::vector<int>& occupied_right,
    const CofactorDifferential& accepted_cofactor,
    const Eigen::MatrixXd& directional_overlap_submatrix,
    const Eigen::MatrixXd& accepted_cofactor_weight,
    const Eigen::MatrixXd& directional_cofactor_weight,
    int n_active_orbitals,
    std::vector<double>* active_orbital_overlap_gradient);

void accumulate_alpha_overlap_gradient_by_pair_graph(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    std::vector<double>* active_orbital_overlap_gradient);

void accumulate_beta_overlap_gradient_by_pair_graph(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    std::vector<double>* active_orbital_overlap_gradient);

void accumulate_directional_alpha_overlap_gradient(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    std::vector<double>* active_orbital_overlap_gradient);

void accumulate_directional_beta_overlap_gradient(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
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
