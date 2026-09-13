#pragma once

#include <optional>
#include <vector>

#include "core/eigensolver.hpp"
#include "vbscf/integrals/active/preparation/space.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "vbscf/structures/expansion/types.hpp"
#include "vbscf/structures/assembly/selected_coefficients.hpp"
#include "vbscf/structures/assembly/action.hpp"

namespace xmvb::vb {

/**
 * @brief Accepted-point active-space context for exact matrix-free second-order work.
 *
 * This object persists the heavy forward intermediates generated at one relaxed
 * VBSCF evaluation so later orbital second-order operators can reuse them
 * without rebuilding the same-spin cache, structure matrices, or selected-state
 * determinant coefficient bundles.
 */
struct AcceptedPointContext {
  /**
   * @brief Prepared active-space tensors and one-electron reference data.
   */
  PreparedActiveSpaceContext prepared_active_space;

  /**
   * @brief Unique-spin reuse tables and cached ordered same-spin determinant pairs.
   */
  SameSpinPairCacheContext same_spin_pair_cache;

  /**
   * @brief Self-contained matrix-free accepted structure problem.
   *
   * The action supplies selected-root response solves without accessing or
   * reconstructing the dense reference matrices retained during this
   * migration stage.
   */
  std::optional<StructureAction> structure_action;

  /**
   * @brief Accepted-point structure Hamiltonian/overlap matrices.
   */
  StructureAccumulationResult structure_matrices;

  /**
   * @brief Accepted-point active-space adjoint with respect to `SSO`.
   *
   * This is the exact accepted-point reverse-mode weight used by the orbital
   * backpropagation layer. Matrix-free second-order probes can reuse it as a
   * frozen active-space adjoint without rebuilding the structure/eigen response.
   */
  std::vector<double> active_orbital_overlap_gradient;

  /**
   * @brief Accepted-point active-space adjoint with respect to `HHO`.
   */
  std::vector<double> active_one_electron_gradient;

  /**
   * @brief Accepted-point active-space adjoint with respect to packed `GGO`.
   */
  std::vector<double> packed_active_two_electron_gradient;

  /**
   * @brief Accepted-point generalized eigensystem on the structure basis.
   */
  xmvb::core::GeneralizedEigenResult eigen_result;

  /**
   * @brief Selected states carried by this context.
   */
  std::vector<int> selected_state_indices;

  /**
   * @brief Normalized state-average weights aligned with `selected_state_indices`.
   */
  std::vector<double> normalized_state_weights;

  /**
   * @brief Energies of the selected states extracted from `eigen_result`.
   */
  std::vector<double> selected_state_energies;

  /**
   * @brief State-dependent determinant coefficient matrices on unique-spin space.
   */
  SelectedStateDeterminantMatrices selected_state_matrices;

  /**
   * @brief Number of active orbitals used to index `HHO`, `SSO`, and packed `GGO`.
   */
  int n_active_orbitals = 0;

  /**
   * @brief Whether the accepted point supports matrix-form same-spin adjoints.
   */
  bool use_full_matrix_form_adjoint = false;

  /**
   * @brief Whether the accepted point supports matrix-form opposite-spin channels.
   */
  bool use_matrix_form_opposite_spin = false;
};

}  // namespace xmvb::vb
