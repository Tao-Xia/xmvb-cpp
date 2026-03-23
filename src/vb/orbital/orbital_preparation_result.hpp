#pragma once

#include <vector>

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
  std::vector<double> auxiliary_orbital_matrix;

  /**
   * @brief Column-major active-orbital overlap matrix.
   */
  std::vector<double> active_orbital_overlap_matrix;

  /**
   * @brief Column-major inactive density matrix.
   */
  std::vector<double> inactive_density_matrix;

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
};

}  // namespace xmvb::vb
