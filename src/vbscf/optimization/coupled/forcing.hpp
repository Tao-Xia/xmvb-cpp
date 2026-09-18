#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/exact/operator.hpp"
#include "vbscf/optimization/coupled/operator.hpp"

namespace xmvb::vb {

/**
 * @brief Converts physical selected-state H/S images into symmetric @f$Bp@f$.
 *
 * Selected columns and layout clusters share one contiguous order. For each
 * equal-weight cluster this forms
 *
 * @f[
 * \sqrt{2w}
 * \begin{bmatrix}
 * (\delta H)C-(\delta S)C\Lambda\\
 * \tfrac12 C^T(\delta S)C
 * \end{bmatrix}.
 * @f]
 *
 * The lower block is symmetrized to remove floating-point skew from the
 * derivative of the symmetric overlap metric.
 */
Eigen::VectorXd pack_selected_subspace_forcing(
    const SelectedSubspaceResponseLayout& layout,
    const Eigen::Ref<const Eigen::VectorXd>& selected_energies,
    const Eigen::Ref<const Eigen::MatrixXd>& selected_eigenvectors,
    const SelectedStructureDirection& direction);

/** @brief Packs several direction-major selected-state image blocks. */
Eigen::MatrixXd pack_selected_subspace_forcing_block(
    const SelectedSubspaceResponseLayout& layout,
    const Eigen::Ref<const Eigen::VectorXd>& selected_energies,
    const Eigen::Ref<const Eigen::MatrixXd>& selected_eigenvectors,
    const std::vector<SelectedStructureDirection>& directions);

/**
 * @brief Production matrix-free @f$Bp@f$ action at one accepted orbital point.
 *
 * This is the direct bridge between ExactHvpOperator's directional H/S
 * construction and CoupledNewtonOperator's weight-scaled response coordinates.
 * It performs no eigensystem-response solve and materializes no Hessian.
 */
Eigen::MatrixXd apply_exact_orbital_to_selected_subspace(
    const ExactHvpOperator& exact_operator,
    const SelectedSubspaceResponseLayout& layout,
    const Eigen::Ref<const Eigen::VectorXd>& selected_energies,
    const Eigen::Ref<const Eigen::MatrixXd>& selected_eigenvectors,
    const Eigen::Ref<const Eigen::MatrixXd>& orbital_directions);

/**
 * @brief Production matrix-free @f$B^T z@f$ action.
 *
 * Packed response columns are unpacked with their @f$\sqrt{2w}@f$ coordinate
 * scaling, assembled into one coefficient block and one block-diagonal full
 * multiplier matrix, and pulled back by the accepted exact orbital operator.
 */
Eigen::MatrixXd apply_exact_selected_subspace_to_orbital(
    const ExactHvpOperator& exact_operator,
    const SelectedSubspaceResponseLayout& layout,
    const Eigen::Ref<const Eigen::MatrixXd>& response_directions);

/**
 * @brief Production matrix-free selected-subspace response Hessian @f$Cq@f$.
 *
 * The layout must be the maximal contiguous clustering of the accepted states
 * by normalized weight. The physical response matrices are sent to the exact
 * accepted-point H/S action as one block and then returned in the same
 * @f$\sqrt{2w}@f$ coordinates as the input.
 */
Eigen::MatrixXd apply_exact_selected_subspace_hessian(
    const ExactHvpOperator& exact_operator,
    const SelectedSubspaceResponseLayout& layout,
    const Eigen::Ref<const Eigen::MatrixXd>& response_directions);

}  // namespace xmvb::vb
