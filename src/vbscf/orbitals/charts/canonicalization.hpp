#pragma once

#include <Eigen/Core>

#include "vbscf/orbitals/preparation/input.hpp"

namespace xmvb::vb {

struct OrbitalPreparationResult;

void overwrite_sparse_orbitals_from_dense_physical_frame(
    const Eigen::MatrixXd& dense_orbitals,
    OrbitalPreparationInput* orbital_preparation_input);

Eigen::MatrixXd build_self_adjoint_matrix_power(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix,
    double exponent,
    const char* label);

Eigen::MatrixXd build_inactive_metric_inverse(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_physical_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& basis_overlap_matrix);

Eigen::MatrixXd build_metric_preserving_inactive_repaired_active_physical_orbitals(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_physical_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& current_active_auxiliary,
    const Eigen::Ref<const Eigen::MatrixXd>& current_active_physical_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& reference_active_physical_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& basis_overlap_matrix);

Eigen::MatrixXd build_metric_preserving_oeo_repaired_normalized_orbital_matrix(
    const OrbitalPreparationInput& orbital_preparation_input,
    const OrbitalPreparationResult& orbital_result,
    const Eigen::Ref<const Eigen::MatrixXd>& reference_normalized_orbital_matrix);

}  // namespace xmvb::vb
