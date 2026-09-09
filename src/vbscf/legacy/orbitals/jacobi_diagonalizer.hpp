#pragma once

#include <Eigen/Core>

namespace xmvb::vb {

struct LegacyJacobiDiagonalizationResult {
  Eigen::MatrixXd eigenvectors;
  Eigen::VectorXd eigenvalues;
};

/**
 * @brief Diagonalizes a real symmetric matrix in the legacy XMVB Jacobi order.
 *
 * XMVB's `hes_Diag` / `Diag` routines do not sort eigenpairs after the Jacobi
 * sweeps.  The final column order therefore depends on the sweep order itself
 * and becomes part of the accepted-point orbital gauge for projector-built
 * virtual spaces.  Reproducing that unsorted Jacobi order is important for
 * small open-shell OEO cases where a merely equivalent virtual subspace can
 * still steer the optimizer toward a different occupied-orbital representative.
 *
 * @param matrix Input self-adjoint matrix in the current chart.
 * @param tolerance Maximum allowed squared off-diagonal element at convergence.
 * @param max_sweeps Safety cap on Jacobi sweeps.
 * @return Legacy-order eigenvectors in columns and the matching diagonal values.
 */
LegacyJacobiDiagonalizationResult diagonalize_self_adjoint_legacy_jacobi(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix,
    double tolerance = 1.0e-30,
    int max_sweeps = 0);

}  // namespace xmvb::vb
