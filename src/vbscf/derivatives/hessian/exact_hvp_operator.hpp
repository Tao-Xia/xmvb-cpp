#pragma once

#include <memory>
#include <vector>

#include <Eigen/Core>

#include "vbscf/core/vbscf_input.hpp"
#include "vbscf/structures/coefficient_blocks.hpp"
#include "vbscf/integrals/active/active_space_two_electron_response_types.hpp"
#include "vbscf/orbitals/charts/orbital_chart.hpp"
#include "vbscf/orbitals/charts/sparse_parameter_layout.hpp"
#include "vbscf/derivatives/hessian/accepted_point_context.hpp"
#include "vbscf/derivatives/hessian/responses/active_space_integral_direction.hpp"
#include "vbscf/derivatives/hessian/structure_response_internal.hpp"
#include "vbscf/structures/selected_state_coefficients.hpp"

namespace xmvb::vb {

struct AcceptedOrbitalPreparationCache;

/**
 * @brief Selects which HVP sub-components to include.
 *
 * The orbital Hessian-vector product decomposes into three additive
 * contributions (direct core, fixed-upstream pullback, outer response) that
 * can be independently toggled.
 */
struct HvpComponents {
  bool direct_core_response = true;
  bool fixed_upstream_pullback = true;
  bool outer_response = true;
};

/**
 * @brief Accepted-point matrix-free orbital second-order operator.
 *
 * This module is the home for exact direct-action orbital Hessian contributions
 * evaluated at one accepted VBSCF point. The accepted-point context caches the
 * relaxed active-space intermediates, structure matrices, eigensystem, and
 * selected-state adjoints so `H v` applications can stay entirely analytic:
 *
 * - the local orbital / integral-transformation chain is differentiated
 *   directly at fixed accepted active-space adjoints;
 * - the outer structure/eigen response is differentiated analytically through
 *   `\delta SSO`, `\delta HHO`, and `\delta GGO`, then pulled back to the
 *   orbital space as `J^T \delta \lambda`.
 *
 */
class ExactHvpOperator {
public:
  struct Diagnostics {
    bool supports_analytic_core_model = false;
    bool outer_response_enabled = false;
    bool used_reduced_curvature_diagonal = false;
    bool has_same_spin_matrix_form = false;
    bool has_opposite_spin_matrix_form = false;
    bool outer_response_local_only_approximation = false;
    bool outer_response_energy_only_approximation = false;
    int n_selected_states = 0;
    int n_active_orbitals = 0;
    int n_blocks = 0;
    std::size_t apply_count = 0;
    std::size_t batch_apply_count = 0;
    double total_apply_wall_time_seconds = 0.0;
    double core_setup_wall_time_seconds = 0.0;
    double ao_effective_one_electron_build_wall_time_seconds = 0.0;
    double ao_effective_one_electron_fused_wall_time_seconds = 0.0;
    double active_two_electron_wall_time_seconds = 0.0;
    double ao_effective_one_electron_backprop_wall_time_seconds = 0.0;
    double orbital_backprop_wall_time_seconds = 0.0;
    double fixed_upstream_pullback_wall_time_seconds = 0.0;
    double outer_response_wall_time_seconds = 0.0;
    // Outer response sub-stage timings
    double outer_response_active_space_integrals_wall_time_seconds = 0.0;
    double outer_response_structure_matrices_wall_time_seconds = 0.0;
    double outer_response_eigensystem_wall_time_seconds = 0.0;
    double outer_response_pair_weights_wall_time_seconds = 0.0;
    double outer_response_active_gradient_wall_time_seconds = 0.0;
    double outer_response_orbital_pullback_wall_time_seconds = 0.0;
  };

  ExactHvpOperator(
      std::shared_ptr<const AcceptedPointContext> accepted_point_context,
      const VbScfInput* current_input,
      SparseParameterLayout parameter_view,
      const OrbitalChart* nonredundant_space);

  ~ExactHvpOperator();

  /**
   * @brief Applies the accepted-point reduced-space second-order model.
   *
   * The returned vector lives in the same reduced coordinates as the input.
   * HvpComponents selects which sub-components (direct core, fixed-upstream
   * pullback, outer response) are included.
   */
  Eigen::VectorXd apply_reduced(
      const Eigen::VectorXd& reduced_direction,
      HvpComponents components = {}) const;

  /**
   * @brief Applies the matrix-free operator to a block of directions.
   *
   * Columns are independent reduced-space directions.  This interface is the
   * semantic foundation for fused block contractions; it never materializes
   * the reduced Hessian.
   */
  Eigen::MatrixXd apply_reduced_batch(
      const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions,
      HvpComponents components = {}) const;

  bool supports_analytic_core_model() const noexcept;

  Diagnostics diagnostics() const;

private:
  struct PrecomputedDirection;

  Eigen::VectorXd apply_reduced_impl(
      const Eigen::VectorXd& reduced_direction,
      HvpComponents components,
      const Eigen::VectorXd* precomputed_delta_ao_effective_h1e,
      const Eigen::VectorXd* precomputed_inactive_density_gradient,
      const Eigen::VectorXd* precomputed_delta_packed_active_two_electron,
      const ExactCtxPairMatrix* precomputed_directional_pair_products,
      const PrecomputedDirection* precomputed_direction) const;

  struct ApplyTimingTotals {
    std::size_t apply_count = 0;
    std::size_t batch_apply_count = 0;
    double total_apply_wall_time_seconds = 0.0;
    double core_setup_wall_time_seconds = 0.0;
    double ao_effective_one_electron_build_wall_time_seconds = 0.0;
    double ao_effective_one_electron_fused_wall_time_seconds = 0.0;
    double active_two_electron_wall_time_seconds = 0.0;
    double ao_effective_one_electron_backprop_wall_time_seconds = 0.0;
    double orbital_backprop_wall_time_seconds = 0.0;
    double fixed_upstream_pullback_wall_time_seconds = 0.0;
    double outer_response_wall_time_seconds = 0.0;
    double outer_response_active_space_integrals_wall_time_seconds = 0.0;
    double outer_response_structure_matrices_wall_time_seconds = 0.0;
    double outer_response_eigensystem_wall_time_seconds = 0.0;
    double outer_response_pair_weights_wall_time_seconds = 0.0;
    double outer_response_active_gradient_wall_time_seconds = 0.0;
    double outer_response_orbital_pullback_wall_time_seconds = 0.0;
  };

  std::shared_ptr<const AcceptedPointContext> accepted_point_context_;
  const VbScfInput* current_input_ = nullptr;
  SparseParameterLayout parameter_view_;
  const OrbitalChart* nonredundant_space_ = nullptr;
  Eigen::MatrixXd accepted_active_auxiliary_orbitals_;
  // Accepted-point active-auxiliary contractions stay constant across all HVP
  // applications at one optimizer iterate, so caching them once avoids
  // repeating the expensive `S * A` / `F * A` multiplies inside every Krylov
  // matvec.
  Eigen::MatrixXd accepted_basis_overlap_times_active_auxiliary_orbitals_;
  Eigen::MatrixXd accepted_ao_effective_one_electron_times_active_auxiliary_orbitals_;
  Eigen::MatrixXd
      accepted_ao_effective_one_electron_transpose_times_active_auxiliary_orbitals_;
  Eigen::MatrixXd accepted_dense_active_coefficients_;
  Eigen::MatrixXd accepted_sso_gradient_symmetric_;
  Eigen::MatrixXd accepted_hho_gradient_symmetric_;
  Eigen::MatrixXd accepted_active_auxiliary_orbitals_times_hho_gradient_symmetric_;
  mutable Eigen::MatrixXd ao_h1e_symmetrized_gradient_workspace_;
  Eigen::MatrixXd accepted_total_active_auxiliary_gradient_;
  std::vector<double> accepted_total_inactive_density_gradient_;
  Eigen::MatrixXd zero_core_hamiltonian_;
  mutable std::vector<double> ao_h1e_delta_h1e_workspace_;
  mutable std::vector<double> ao_h1e_inactive_density_gradient_workspace_;
  mutable std::vector<Eigen::MatrixXd> ao_h1e_partial_delta_h1e_workspaces_;
  mutable std::vector<Eigen::MatrixXd>
      ao_h1e_partial_inactive_density_gradient_workspaces_;
  ExactPackedActiveTwoElectronAdjointCache accepted_exact_two_electron_cache_;
  mutable ExactPackedActiveTwoElectronApplyWorkspace
      accepted_exact_two_electron_apply_workspace_;
  mutable ActiveSpaceIntegralDirectionWorkspace
      outer_response_integral_direction_workspace_;
  mutable std::vector<double>
      outer_response_symmetric_active_overlap_gradient_workspace_;
  mutable std::vector<double>
      outer_response_symmetric_active_one_electron_gradient_workspace_;
  std::unique_ptr<AcceptedOrbitalPreparationCache> accepted_orbital_preparation_cache_;
  std::vector<StructureCoefficientBlock> structure_coefficient_blocks_;
  // Frozen accepted-point selected-state response metadata.  Each HVP still
  // builds fresh directional structure columns, but gap/gauge data and selected
  // eigenvector bookkeeping are reused across Krylov matvecs.
  AcceptedOuterResponseContext accepted_outer_response_context_;
  mutable ApplyTimingTotals apply_timing_totals_;
};

}  // namespace xmvb::vb
