#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/responses/same_spin/backward.hpp"
#include "vbscf/determinants/algebra/cofactor_differential.hpp"

namespace xmvb::vb::detail {

SameSpinDirectionalScalarMatrices build_directional_pair_scalar_matrices(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    int n_unique_determinants,
    int n_active_orbitals,
    const ActiveSpaceIntegralDirectionView& direction);

void accumulate_one_electron_gradient_contribution_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& cofactor_1st,
    double weight,
    Eigen::MatrixXd* active_one_electron_gradient);

void accumulate_overlap_block_gradient_contribution_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& overlap_block_gradient,
    double weight,
    int n_active_orbitals,
    std::vector<double>* active_orbital_overlap_gradient);

void accumulate_deleted_minor_same_spin_two_electron_gradient_contribution_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const CofactorDifferential& cofactor,
    double weight,
    std::vector<double>* packed_active_two_electron_gradient);

SameSpinPolynomialDirectionalPairData build_polynomial_spin_directional_data(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const SpinDeterminantPairEvaluation& pair_evaluation,
    int n_active_orbitals,
    const ActiveSpaceIntegralDirectionView& direction,
    bool need_overlap_gradient = true);

void accumulate_directional_one_electron_gradient_contribution_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& cofactor_1st,
    const Eigen::MatrixXd& delta_cofactor_1st,
    double weight,
    double delta_weight,
    Eigen::MatrixXd* active_one_electron_gradient);

void accumulate_directional_deleted_minor_same_spin_two_electron_gradient_contribution_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const CofactorDifferential& cofactor,
    const std::vector<double>& delta_ao_overlap_matrix,
    int n_active_orbitals,
    double weight,
    double delta_weight,
    std::vector<double>* packed_active_two_electron_gradient);

}  // namespace xmvb::vb::detail
