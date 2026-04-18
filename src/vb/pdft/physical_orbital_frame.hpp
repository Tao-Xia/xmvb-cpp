#pragma once

#include <Eigen/Core>

namespace xmvb::vb {

/**
 * @brief Localized physical representative reconstructed from the internal frame.
 *
 * The accepted-point inactive-orthogonal formulation distinguishes the
 * internal working frame `(Q_i, T_a)` from the localized physical occupied
 * representative `(C_i, C_a)`. The two are related by the small matrices
 *
 * `C_i = Q_i U_i`
 * and
 * `C_a = T_a + Q_i K_a`.
 *
 * Caching those matrices at the accepted point makes later chart transport and
 * representative diagnostics explicit instead of rediscovering the same small
 * linear algebra ad hoc inside the optimizer.
 */
struct LocalizedRepresentativeSelector {
  /**
   * @brief Column-major inactive representative matrix `U_i`.
   *
   * Dimensions:
   * `n_inactive_doubly_occupied_orbitals x n_inactive_doubly_occupied_orbitals`.
   */
  Eigen::MatrixXd inactive_right_transform;

  /**
   * @brief Column-major inverse-transpose inactive representative matrix `U_i^{-T}`.
   *
   * Dimensions:
   * `n_inactive_doubly_occupied_orbitals x n_inactive_doubly_occupied_orbitals`.
   */
  Eigen::MatrixXd inactive_inverse_transpose_right_transform;

  /**
   * @brief Column-major inactive-null active coefficients `K_a`.
   *
   * This stores the inactive-frame coefficients of the localized physical
   * active orbitals:
   * `C_a = T_a + Q_i K_a`.
   *
   * Dimensions:
   * `n_inactive_doubly_occupied_orbitals x n_active_orbitals`.
   */
  Eigen::MatrixXd active_inactive_coefficients;
};

/**
 * @brief Normalized physical orbital coefficients reconstructed from sparse input.
 *
 * The managed C++ VBSCF matrix path works with projected auxiliary orbitals
 * after the inactive-space elimination step. Real-space post-VBSCF methods
 * such as a future `VB-PDFT` implementation should instead start from the
 * physical normalized orbital coefficients before that projection. This
 * lightweight container makes those physical coefficients available alongside
 * the existing auxiliary-orbital intermediates.
 */
struct PhysicalOrbitalFrame {
  /**
   * @brief Column-major normalized physical orbital matrix `C`.
   *
   * Dimensions: `n_basis_functions x n_orbitals`.
   */
  Eigen::MatrixXd normalized_orbital_matrix;

  /**
   * @brief Column-major inactive physical orbital block `C_inactive`.
   *
   * Dimensions: `n_basis_functions x n_inactive_doubly_occupied_orbitals`.
   */
  Eigen::MatrixXd inactive_physical_orbital_matrix;

  /**
   * @brief Column-major inactive orthonormal frame `Q_inactive`.
   *
   * This stores the accepted-point inactive block after the gauge transform
   * `Q_i = C_i (C_i^T S C_i)^{-1/2}`. The frame is explicitly
   * `S`-orthonormal and is the natural state variable for the proposed
   * orthonormal-inactive nonredundant chart, while the original physical
   * inactive block is kept above for legacy compatibility and diagnostics.
   *
   * Dimensions: `n_basis_functions x n_inactive_doubly_occupied_orbitals`.
   */
  Eigen::MatrixXd inactive_orthonormal_orbital_matrix;

  /**
   * @brief Column-major active physical orbital block `C_active`.
   *
   * Dimensions: `n_basis_functions x n_active_orbitals`.
   */
  Eigen::MatrixXd active_physical_orbital_matrix;

  /**
   * @brief Column-major gauge transform `R_i = (C_i^T S C_i)^{-1/2}`.
   *
   * This is the small inactive-space matrix used to transform the physical
   * inactive block into the orthonormal frame. Keeping it in the accepted-point
   * state makes it possible to compare the current physical gauge against the
   * proposed orthonormal gauge without recomputing the small matrix from
   * scratch in every diagnostic or migration step.
   *
   * Dimensions:
   * `n_inactive_doubly_occupied_orbitals x n_inactive_doubly_occupied_orbitals`.
   */
  Eigen::MatrixXd inactive_orthonormal_gauge_transform;

  /**
   * @brief Accepted-point selector mapping the internal frame to the localized representative.
   *
   * This records the small accepted-point matrices that recover the current
   * localized physical occupied orbitals from the internal inactive-orthogonal
   * frame and the projected active block.
   */
  LocalizedRepresentativeSelector localized_representative_selector;
};

}  // namespace xmvb::vb
