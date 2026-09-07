#pragma once

#include <Eigen/Core>

#include "vb/orbital/orbital_preparation_input.hpp"
#include "vb/orbital/sparse_orbital_parameter_view.hpp"

namespace xmvb::vb {

/**
 * @brief Independent audit of the strict-sparse VBSCF orbital quotient.
 *
 * The algebraic gauge is constructed from support-admissible transformations
 *   dC_I = C_I K,
 *   dC_A = C_I L + C_A diag(alpha).
 * It is then compared with the null space of the independently differentiated
 * physical map (P_I, [O_I c_A]).  This routine is diagnostic only and does not
 * alter the optimizer coordinate space.
 */
struct SparseOrbitalGaugeAudit {
  int packed_dimension = 0;
  int gauge_parameter_dimension = 0;
  int support_constraint_rank = 0;
  int admissible_gauge_parameter_dimension = 0;
  int gauge_rank = 0;
  int quotient_dimension = 0;
  int physical_jacobian_rank = 0;
  int physical_jacobian_nullity = 0;
  int unmapped_parameter_count = 0;

  double relative_gauge_annihilation_residual = 0.0;
  double maximum_gauge_kernel_principal_angle_sine = 0.0;

  int current_reduced_dimension = 0;
  int current_physical_image_rank = 0;
  int current_retained_gauge_dimension = 0;
  int current_missing_physical_dimension = 0;
};

SparseOrbitalGaugeAudit audit_sparse_orbital_gauge(
    const OrbitalPreparationInput& input,
    const SparseOrbitalParameterView& parameter_view,
    const Eigen::MatrixXd* current_packed_reduced_basis = nullptr);

}  // namespace xmvb::vb
