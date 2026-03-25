#pragma once

#include <vector>

#include "vb/scf/cpp_vb_scf_result.hpp"

namespace xmvb::vb {

/**
 * @brief Result of the C++ orbital-gradient evaluation.
 *
 * The gradient is defined with respect to the explicit sparse orbital
 * coefficient table stored in `OrbitalPreparationInput::orbital_value_table`.
 * Entries corresponding to unused sparse slots are set to zero.
 */
struct CppOrbitalGradientResult {
  /**
   * @brief Wall time spent in the active-space gradient evaluator.
   */
  double active_space_gradient_wall_time_seconds = 0.0;

  /**
   * @brief Wall time spent in orbital preparation inside the active-space stage.
   */
  double orbital_preparation_wall_time_seconds = 0.0;

  /**
   * @brief Wall time spent building the AO effective one-electron matrix inside the active-space stage.
   */
  double ao_effective_one_electron_wall_time_seconds = 0.0;

  /**
   * @brief Wall time spent building the active-space one-electron matrix inside the active-space stage.
   */
  double active_one_electron_wall_time_seconds = 0.0;

  /**
   * @brief Wall time spent building packed active-space two-electron integrals inside the active-space stage.
   */
  double active_two_electron_wall_time_seconds = 0.0;

  /**
   * @brief Wall time spent assembling the structure matrices inside the active-space stage.
   */
  double structure_matrix_wall_time_seconds = 0.0;

  /**
   * @brief Wall time spent in the generalized eigensolver inside the active-space stage.
   */
  double eigensolver_wall_time_seconds = 0.0;

  /**
   * @brief Wall time spent in the active-space adjoint sweep.
   */
  double active_space_adjoint_wall_time_seconds = 0.0;

  /**
   * @brief Wall time spent backpropagating through the active-space matrix layer.
   */
  double active_space_matrix_backpropagation_wall_time_seconds = 0.0;

  /**
   * @brief Wall time spent backpropagating packed active-space two-electron integrals.
   */
  double active_space_two_electron_backpropagation_wall_time_seconds = 0.0;

  /**
   * @brief Wall time spent backpropagating through the AO effective one-electron layer.
   */
  double ao_effective_one_electron_backpropagation_wall_time_seconds = 0.0;

  /**
   * @brief Wall time spent projecting the gradient back to sparse orbital coefficients.
   */
  double orbital_backpropagation_wall_time_seconds = 0.0;

  /**
   * @brief Total wall time for the orbital gradient evaluation.
   */
  double total_wall_time_seconds = 0.0;

  /**
   * @brief Baseline single-step VBSCF evaluation.
   */
  CppVbScfResult scf_result;

  /**
   * @brief Column-major active-orbital overlap matrix `SSO`.
   */
  std::vector<double> active_orbital_overlap_matrix;

  /**
   * @brief Column-major active-space effective one-electron matrix `HHO`.
   */
  std::vector<double> active_one_electron_integrals;

  /**
   * @brief Packed active-space two-electron integrals used to derive `J` and `K`.
   */
  std::vector<double> packed_active_two_electron_integrals;

  /**
   * @brief Historical configuration field retained for evaluator compatibility.
   */
  double finite_difference_step = 0.0;

  /**
   * @brief Flat indices of the explicit sparse orbital coefficients.
   */
  std::vector<int> differentiable_parameter_indices;

  /**
   * @brief Energy gradient with respect to `orbital_value_table`.
   */
  std::vector<double> sparse_orbital_energy_gradient;

  /**
   * @brief Exact `E11` gradient with respect to `orbital_value_table`.
   */
  std::vector<double> sparse_orbital_reference_energy_gradient;
};

}  // namespace xmvb::vb
