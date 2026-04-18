#pragma once

#include <memory>
#include <vector>

#include "vb/orbital/active_space_one_electron_result.hpp"
#include "vb/orbital/active_space_two_electron_result.hpp"
#include "vb/orbital/ao_effective_one_electron_result.hpp"
#include "vb/orbital/orbital_preparation_result.hpp"
#include "vb/scf/cpp_vb_scf_result.hpp"

namespace xmvb::vb {

struct CppActiveSpaceSecondOrderContext;

/**
 * @brief Analytic gradient result for the active-space integral layer.
 *
 * This result exposes the derivative of the selected-state VBSCF energy with
 * respect to the active-space effective one-electron matrix `HHO` and the
 * packed active-space two-electron integrals `GGO`.
 */
struct CppActiveSpaceGradientResult {
  /**
   * @brief Wall time spent in orbital preparation.
   */
  double orbital_preparation_wall_time_seconds = 0.0;

  /**
   * @brief Wall time spent building the AO effective one-electron matrix.
   */
  double ao_effective_one_electron_wall_time_seconds = 0.0;

  /**
   * @brief Wall time spent building the active-space one-electron matrix.
   */
  double active_one_electron_wall_time_seconds = 0.0;

  /**
   * @brief Wall time spent building the active-space two-electron representation.
   */
  double active_two_electron_wall_time_seconds = 0.0;

  /**
   * @brief Wall time spent assembling the structure Hamiltonian/overlap matrices.
   */
  double structure_matrix_wall_time_seconds = 0.0;

  /**
   * @brief Wall time spent solving the generalized eigenproblem.
   */
  double eigensolver_wall_time_seconds = 0.0;

  /**
   * @brief Wall time spent in the active-space adjoint sweep.
   */
  double adjoint_wall_time_seconds = 0.0;

  /**
   * @brief Total wall time for the active-space gradient evaluation.
   */
  double total_wall_time_seconds = 0.0;

  /**
   * @brief Baseline orbital-preparation intermediates.
   */
  OrbitalPreparationResult orbital_preparation_result;

  /**
   * @brief Baseline AO effective one-electron input.
   */
  AoEffectiveOneElectronResult ao_effective_one_electron_result;

  /**
   * @brief Column-major active-orbital overlap matrix `SSO`.
   */
  std::vector<double> active_orbital_overlap_matrix;

  /**
   * @brief Baseline single-step VBSCF result.
   */
  CppVbScfResult scf_result;

  /**
   * @brief Baseline active-space one-electron input.
   */
  ActiveSpaceOneElectronResult active_space_one_electron_result;

  /**
   * @brief Baseline active-space two-electron input.
   */
  ActiveSpaceTwoElectronResult active_space_two_electron_result;

  /**
   * @brief Experimental column-major gradient with respect to `SSO`.
   *
   * This field is populated for ongoing analytic-overlap development, but the
   * validated active-space regression currently covers only `HHO` and `GGO`.
   */
  std::vector<double> active_orbital_overlap_gradient;

  /**
   * @brief Column-major gradient with respect to `HHO`.
   */
  std::vector<double> active_one_electron_gradient;

  /**
   * @brief Gradient with respect to packed `GGO`.
   */
  std::vector<double> packed_active_two_electron_gradient;

  /**
   * @brief Accepted-point forward context for future exact matrix-free HVPs.
   *
   * This optional cache persists the heavy same-spin reuse data, structure
   * matrices, eigensystem, and selected-state matrix-form bundles created by a
   * relaxed active-space evaluation.
   */
  std::shared_ptr<CppActiveSpaceSecondOrderContext> second_order_context;
};

}  // namespace xmvb::vb
