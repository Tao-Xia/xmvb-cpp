#pragma once

#include <Eigen/Core>

#include <vector>

#include "vb/pdft/physical_orbital_frame.hpp"

namespace xmvb::vb {

/**
 * @brief Result of the C++ auxiliary-orbital preparation step.
 */
struct OrbitalPreparationResult {
  /**
   * @brief CSR-style offsets for nonzero active auxiliary coefficients grouped by AO basis row.
   */
  std::vector<int> active_sparse_row_offsets;

  /**
   * @brief Active-orbital indices for nonzero active auxiliary coefficients.
   */
  std::vector<int> active_sparse_orbital_indices;

  /**
   * @brief Values for nonzero active auxiliary coefficients.
   */
  std::vector<double> active_sparse_values;

  /**
   * @brief Column-major auxiliary orbital coefficient matrix.
   */
  Eigen::MatrixXd auxiliary_orbital_matrix;

  /**
   * @brief Column-major active-orbital overlap matrix.
   */
  std::vector<double> active_orbital_overlap_matrix;

  /**
   * @brief Column-major inactive density matrix.
   */
  Eigen::MatrixXd inactive_density_matrix;

  /**
   * @brief Column-major inactive orthonormal projector `Q_i Q_i^T`.
   *
   * This is numerically equal to the current inactive density matrix
   * `C_i (C_i^T S C_i)^{-1} C_i^T`, but it is stored separately so the
   * orthonormal-inactive gauge migration can validate the new representation
   * directly at the accepted point before changing downstream formulas.
   */
  Eigen::MatrixXd inactive_orthonormal_projector_matrix;

  /**
   * @brief Column-major inactive-density factors `F` with `P11 = F F^T`.
   *
   * The matrix uses the project-standard Eigen storage with dimensions
   * `n_basis_functions x n_inactive_doubly_occupied_orbitals`. RI AO-H1E
   * kernels reuse these explicit factors so they do not need to rediscover the
   * low-rank inactive density by diagonalizing the full AO matrix every call.
   */
  Eigen::MatrixXd inactive_density_low_rank_factors;

  /**
   * @brief Column-major occupied-space projector `A1 = I - P11 * S`.
   */
  std::vector<double> occupied_space_projector;

  /**
   * @brief Column-major inactive auxiliary transform `A3 = T_inactive * V11^{-1}`.
   */
  std::vector<double> inactive_auxiliary_transform;

  /**
   * @brief Column-major inactive-active overlap block `A2 = A3^T * (S * T_active)`.
   */
  std::vector<double> inactive_active_overlap_matrix;

  /**
   * @brief Column-major projected active overlap block `A4 = A1^T * (S * T_active)`.
   */
  std::vector<double> projected_active_overlap_matrix;

  /**
   * @brief Column-major inverse of the full auxiliary orbital matrix `A5 = Taux^{-1}`.
   */
  std::vector<double> auxiliary_orbital_inverse_matrix;

  /**
   * @brief Normalized physical orbital coefficients before inactive-space projection.
   *
   * The current matrix-form VBSCF path primarily consumes the auxiliary
   * orbital matrices above. Post-VBSCF real-space methods such as a future
   * `VB-PDFT` implementation should instead start from this physical orbital
   * frame.
   */
  PhysicalOrbitalFrame physical_orbital_frame;
};

}  // namespace xmvb::vb
