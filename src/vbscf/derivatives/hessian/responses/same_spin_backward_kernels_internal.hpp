#pragma once

#include <utility>
#include <vector>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/responses/same_spin_backward.hpp"

namespace xmvb::vb::detail {

SameSpinMatrixBackwardContribution make_zero_backward_contribution(
    int n_active_orbitals);

void account_for_close_shell_spin_reuse(
    Eigen::MatrixXd* active_one_electron_gradient,
    SameSpinMatrixBackwardContribution* result);

SameSpinMatrixBackwardContribution finalize_backward_contribution(
    SameSpinMatrixBackwardContribution result,
    const Eigen::MatrixXd& active_one_electron_gradient);

void accumulate_spin_matrix_backward(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    const Eigen::MatrixXd& hamiltonian_weight_matrix,
    const Eigen::MatrixXd& overlap_weight_matrix,
    const Eigen::MatrixXd& partner_total_transfer_matrix,
    int n_unique_determinants,
    int n_active_orbitals,
    Eigen::MatrixXd* active_one_electron_gradient,
    std::vector<double>* active_orbital_overlap_gradient,
    std::vector<double>* packed_active_two_electron_gradient);

void accumulate_spin_local_matrix_backward(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    const std::vector<SameSpinPolynomialDirectionalPairData>&
        ordered_directional_data,
    const Eigen::MatrixXd& hamiltonian_weight_matrix,
    const Eigen::MatrixXd& overlap_weight_matrix,
    const Eigen::MatrixXd& partner_total_transfer_matrix,
    const Eigen::MatrixXd& delta_hamiltonian_weight_matrix,
    const Eigen::MatrixXd& delta_overlap_weight_matrix,
    const Eigen::MatrixXd& delta_partner_total_transfer_matrix,
    int n_unique_determinants,
    int n_active_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& active_one_electron_matrix,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const std::vector<double>& delta_ao_overlap_matrix,
    Eigen::MatrixXd* active_one_electron_gradient,
    std::vector<double>* active_orbital_overlap_gradient,
    std::vector<double>* packed_active_two_electron_gradient);

SameSpinMatrixBackwardContribution
build_support_sparse_same_spin_backward_contribution_by_tiles(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies,
    int n_active_orbitals);

SameSpinMatrixBackwardContribution
build_support_sparse_directional_same_spin_backward_contribution_by_tiles(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    const std::vector<double>& selected_state_energies,
    const std::vector<double>& directional_selected_state_energies,
    int n_active_orbitals);

SameSpinMatrixBackwardContribution
build_support_sparse_local_same_spin_backward_contribution_by_tiles(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies,
    int n_active_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& active_one_electron_matrix,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const ActiveSpaceIntegralDirectionView& direction,
    const SameSpinDirectionalPairCache& directional_pair_cache);

}  // namespace xmvb::vb::detail
