#pragma once

#include <Eigen/Core>

namespace xmvb::vb::biorthogonal_vbscf {

/**
 * @brief Right-orbital / left-dual-orbital frame for one biorthogonal setup.
 *
 * `right_orbitals` and `left_dual_orbitals` both have dimensions
 * `(n_ao, n_orbitals)`. `right_orbital_overlap` is the right-orbital overlap
 * matrix
 *
 * `X = C^T S C`
 *
 * and `right_orbital_overlap_inverse` stores `X^{-1}`. The residual is the
 * Frobenius norm of
 *
 * `left_dual_orbitals^T * ao_overlap * right_orbitals - I`.
 */
struct BiorthogonalOrbitalFrame {
  Eigen::MatrixXd right_orbitals;
  Eigen::MatrixXd right_orbital_overlap;
  Eigen::MatrixXd right_orbital_overlap_inverse;
  Eigen::MatrixXd left_dual_orbitals;
  double biorthogonality_residual_frobenius_norm = 0.0;
};

/**
 * @brief Builds the induced left-dual orbital frame from one right-orbital set.
 *
 * This routine constructs the biorthogonal left orbitals
 *
 * `\widetilde C = C (C^T S C)^{-1}`
 *
 * and records the matrices that will later be reused by determinant matrix
 * elements and one-sided orbital gradient pullbacks.
 */
BiorthogonalOrbitalFrame build_biorthogonal_orbital_frame(
    const Eigen::Ref<const Eigen::MatrixXd>& ao_overlap_matrix,
    const Eigen::Ref<const Eigen::MatrixXd>& right_orbitals);

/**
 * @brief Validates the dimensions, finiteness, and biorthogonality residual.
 *
 * `residual_tolerance` is compared against the stored Frobenius residual
 * `||\widetilde C^T S C - I||_F`.
 */
void validate_biorthogonal_orbital_frame(
    const BiorthogonalOrbitalFrame& orbital_frame,
    double residual_tolerance);

}  // namespace xmvb::vb::biorthogonal_vbscf
