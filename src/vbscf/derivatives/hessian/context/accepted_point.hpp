#pragma once

#include <optional>
#include <vector>

#include "vbscf/integrals/active/preparation/space.hpp"
#include "vbscf/core/contracts/eigensolver.hpp"
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
 * without rebuilding the same-spin cache, structure action, or selected-state
 * determinant coefficient bundles.
 */
struct AcceptedPointContext {
  /** @brief Outer accuracy contract inherited by structure response solves. */
  StructureSolveAccuracy structure_solve_accuracy;

  /**
   * @brief Prepared active-space tensors and one-electron reference data.
   */
  PreparedActiveSpaceContext prepared_active_space;

  /**
   * @brief Unique-spin reuse tables and cached ordered same-spin determinant pairs.
   */
  SameSpinPairCacheContext same_spin_pair_cache;

  /**
   * @brief On-demand matrix-free accepted structure problem.
   *
   * A forward Davidson solve may transfer its action here.  Dense and
   * matrix-backed solves leave it empty; the exact HVP operator constructs it
   * only if an outer structure response is actually admitted.
   */
  mutable std::optional<StructureAction> structure_action;

  /**
   * @brief Accepted Hamiltonian and overlap images of the selected states.
   *
   * The generalized-eigen response and coupled structure operator share these
   * images. They are populated together with the canonical structure action so
   * neither consumer repeats the same selected-state block action.
   */
  mutable std::optional<StructureActionResult> selected_state_structure_images;

  /** @brief Direction-independent direct-CI data reused by every exact HVP. */
  mutable std::optional<StructureAdjointState> structure_adjoint_state;

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
   * @brief Accepted total adjoint of the active auxiliary orbitals.
   *
   * The complete orbital-gradient sweep has already accumulated the active
   * overlap, one-electron, and two-electron contributions into this matrix.
   * Exact HVP construction reuses it for the fixed-upstream pullback instead
   * of repeating those accepted-point reverse sweeps.
   */
  Eigen::MatrixXd total_active_auxiliary_gradient;

  /**
   * @brief Accepted total adjoint of the inactive density matrix.
   *
   * Stored in column-major AO-matrix order. This includes the reference,
   * active-space, and AO effective-one-electron contributions accumulated by
   * the complete orbital-gradient evaluation.
   */
  std::vector<double> total_inactive_density_gradient;

  /**
   * @brief Selected states carried by this context.
   */
  std::vector<int> selected_state_indices;

  /**
   * @brief Normalized state-average weights aligned with `selected_state_indices`.
   */
  std::vector<double> normalized_state_weights;

  /**
   * @brief Accepted energies aligned with `selected_state_indices`.
   */
  std::vector<double> selected_state_energies;

  /**
   * @brief Selected accepted eigenvectors, one state per column.
   */
  Eigen::MatrixXd selected_state_eigenvectors;

  /** @brief Available roots retained for recycling at the next objective call. */
  Eigen::MatrixXd root_eigenvectors;

  /** @brief Complete structure eigenvalues retained only by the dense solver. */
  std::vector<double> full_structure_eigenvalues;

  /**
   * @brief State-dependent determinant coefficient matrices on unique-spin space.
   */
  SelectedStateDeterminantMatrices selected_state_matrices;

  /**
   * @brief Number of active orbitals used to index `HHO`, `SSO`, and packed `GGO`.
   */
  int n_active_orbitals = 0;

  /** @brief Dimension of the structure space acted on by `structure_action`. */
  int n_structures = 0;

  /**
   * @brief Whether the accepted point supports matrix-form same-spin adjoints.
   */
  bool use_full_matrix_form_adjoint = false;

  /**
   * @brief Whether the accepted point supports pair-graph opposite-spin adjoints.
   */
  bool use_pair_graph_opposite_spin_adjoint = false;
};

}  // namespace xmvb::vb
