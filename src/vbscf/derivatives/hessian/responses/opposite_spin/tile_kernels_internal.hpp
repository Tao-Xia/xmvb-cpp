#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/responses/opposite_spin/pair_response_internal.hpp"
#include "vbscf/determinants/algebra/cofactor_differential.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "vbscf/structures/assembly/selected_coefficients.hpp"

namespace xmvb::vb::detail {

inline constexpr int kOppositeSpinUniqueTileSize = 64;

void accumulate_spin_overlap_gradient_direction_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const CofactorDifferential& cofactor,
    const Eigen::MatrixXd& delta_overlap_submatrix,
    const Eigen::MatrixXd& cofactor_weight,
    const Eigen::MatrixXd& delta_cofactor_weight,
    int n_active_orbitals,
    std::vector<double>* active_orbital_overlap_gradient);

Eigen::MatrixXd build_alpha_overlap_weight_tile_matrix(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_packed_active_pairs,
    int alpha_left_begin,
    int alpha_left_end,
    int alpha_right_begin,
    int alpha_right_end);

Eigen::MatrixXd build_beta_overlap_weight_tile_matrix(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_packed_active_pairs,
    int beta_left_begin,
    int beta_left_end,
    int beta_right_begin,
    int beta_right_end);

Eigen::MatrixXd build_directional_alpha_overlap_weight_tile_matrix(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    int n_packed_active_pairs,
    int alpha_left_begin,
    int alpha_left_end,
    int alpha_right_begin,
    int alpha_right_end);

Eigen::MatrixXd build_directional_beta_overlap_weight_tile_matrix(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    int n_packed_active_pairs,
    int beta_left_begin,
    int beta_left_end,
    int beta_right_begin,
    int beta_right_end);

Eigen::MatrixXd build_local_directional_alpha_overlap_weight_tile_matrix(
    const std::vector<DirectionalOppositeSpinPairData>&
        beta_directional_pair_data,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_packed_active_pairs,
    int alpha_left_begin,
    int alpha_left_end,
    int alpha_right_begin,
    int alpha_right_end);

Eigen::MatrixXd build_local_directional_beta_overlap_weight_tile_matrix(
    const std::vector<DirectionalOppositeSpinPairData>&
        alpha_directional_pair_data,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_packed_active_pairs,
    int beta_left_begin,
    int beta_left_end,
    int beta_right_begin,
    int beta_right_end);

void build_local_packed_pair_dense_image_matrix_from_tile_matrix(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& packed_pair_tile_matrix,
    int tile_index,
    Eigen::MatrixXd* pair_dense_image);

void build_inverse_overlap_gradient_from_tile_matrix(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& packed_pair_tile_matrix,
    int tile_index,
    Eigen::MatrixXd* inverse_overlap_gradient);

}  // namespace xmvb::vb::detail
