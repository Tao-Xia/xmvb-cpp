#pragma once

#include <memory>
#include <vector>

#include <Eigen/Core>

#include "vb/matrices/cpp_vb_input.hpp"
#include "vb/matrices/structure_coefficient_blocks.hpp"
#include "vb/orbital/active_space_two_electron_utils.hpp"
#include "vb/orbital/nonredundant_orbital_space.hpp"
#include "vb/orbital/sparse_orbital_parameter_view.hpp"
#include "vb/scf/cpp_active_space_second_order_context.hpp"
#include "vb/scf/exact_orbital_second_order_operator_outer_response_internal.hpp"
#include "vb/scf/opposite_spin_matrix_backward.hpp"
#include "vb/scf/same_spin_matrix_backward.hpp"
#include "vb/scf/selected_state_determinant_matrices.hpp"

namespace xmvb::vb {

struct AcceptedOrbitalPreparationCache;

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
class ExactOrbitalSecondOrderOperator {
public:
  struct Diagnostics {
    bool supports_analytic_core_model = false;
    bool outer_response_enabled = false;
    bool internal_inactive_chart_runtime_enabled = false;
    bool uses_internal_inactive_chart = false;
    bool used_reduced_curvature_diagonal = false;
    bool has_same_spin_matrix_form = false;
    bool has_opposite_spin_matrix_form = false;
    int n_selected_states = 0;
    int n_active_orbitals = 0;
    int n_blocks = 0;
    std::size_t apply_count = 0;
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

  struct DirectionalStructureDiagnostics {
    std::vector<double> active_orbital_overlap_matrix;
    std::vector<double> active_one_electron_matrix;
    std::vector<double> packed_active_two_electron_integrals;
    std::vector<double> active_orbital_overlap_gradient;
    std::vector<double> active_one_electron_gradient;
    std::vector<double> packed_active_two_electron_gradient;
    std::vector<double> overlap_matrix;
    std::vector<double> hamiltonian_matrix;
  };

  struct DirectCoreDiagnostics {
    Eigen::MatrixXd delta_matrix_active_auxiliary_gradient;
    Eigen::MatrixXd delta_two_electron_active_auxiliary_gradient;
    Eigen::MatrixXd delta_total_active_auxiliary_gradient;
    Eigen::MatrixXd delta_ao_effective_one_electron_matrix;
    Eigen::MatrixXd delta_ao_backpropagated_inactive_density_gradient;
    Eigen::MatrixXd delta_total_inactive_density_gradient;
    Eigen::VectorXd reduced_response;
  };

  struct FixedUpstreamDiagnostics {
    Eigen::MatrixXd original_orbital_gradient;
    std::vector<double> orbital_value_gradient;
    Eigen::VectorXd packed_response;
    Eigen::VectorXd reduced_response;
  };

  ExactOrbitalSecondOrderOperator(
      std::shared_ptr<const CppActiveSpaceSecondOrderContext> accepted_point_context,
      const CppVbInput* current_input,
      SparseOrbitalParameterView parameter_view,
      const NonredundantOrbitalSpace* nonredundant_space);

  ~ExactOrbitalSecondOrderOperator();

  /**
   * @brief Applies the accepted-point reduced-space second-order model.
   *
   * The returned vector lives in the same reduced coordinates as the input and
   * includes both the accepted-point local core differentiation and the fully
   * analytic relaxed structure/eigen response.
   */
  Eigen::VectorXd apply_reduced(
      const Eigen::VectorXd& reduced_direction) const;

  /**
   * @brief Applies the full accepted-point reduced HVP without cache shortcuts.
   *
   * This diagnostics hook keeps the historical rebuild-everything path
   * available after the production entry point switches to the accepted-point
   * orbital-preparation cache.
   */
  Eigen::VectorXd apply_reduced_uncached(
      const Eigen::VectorXd& reduced_direction) const;

  /**
   * @brief Applies the full accepted-point reduced HVP with cache shortcuts.
   *
   * This diagnostics hook forces the accepted-point orbital-preparation cache
   * on so cached and uncached full-model implementations can be compared
   * explicitly before enabling the cached path in the optimizer.
   */
  Eigen::VectorXd apply_reduced_cached(
      const Eigen::VectorXd& reduced_direction) const;

  /**
   * @brief Applies only the accepted-point local/core part of the reduced HVP.
   *
   * This skips the relaxed outer structure/eigen response while preserving the
   * exact accepted-point orbital / active-space chain rule. It is used by the
   * optimizer as a cheaper inexact Newton matvec when the full outer-response
   * operator is too expensive to apply on every inner CG iteration.
   */
  Eigen::VectorXd apply_reduced_without_outer_response(
      const Eigen::VectorXd& reduced_direction) const;

  /**
   * @brief Applies the local/core reduced HVP without cache shortcuts.
   *
   * This diagnostics hook bypasses the accepted-point orbital-preparation
   * cache so the cached cheap/core optimizer path can be checked explicitly.
   */
  Eigen::VectorXd apply_reduced_without_outer_response_uncached(
      const Eigen::VectorXd& reduced_direction) const;

  /**
   * @brief Applies the accepted-point local/core reduced HVP with cache shortcuts.
   *
   * This diagnostics hook forces the accepted-point orbital-preparation cache
   * on so cached and uncached cheap/core implementations can be compared
   * explicitly before enabling the cached path in the optimizer.
   */
  Eigen::VectorXd apply_reduced_without_outer_response_cached(
      const Eigen::VectorXd& reduced_direction) const;

  /**
   * @brief Applies only the direct local/core upstream-delta contribution.
   *
   * This diagnostics hook keeps the accepted-point orbital backpropagator
   * linearized at the accepted input, but excludes the separate fixed-upstream
   * pullback term.
   */
  Eigen::VectorXd apply_reduced_core_direct_only(
      const Eigen::VectorXd& reduced_direction) const;

  /**
   * @brief Applies only the direct local/core upstream-delta contribution with cache shortcuts.
   *
   * This diagnostics hook forces the accepted-point orbital-preparation cache
   * on so cached and uncached implementations can be compared explicitly.
   */
  Eigen::VectorXd apply_reduced_core_direct_only_cached(
      const Eigen::VectorXd& reduced_direction) const;

  /**
   * @brief Applies only the fixed-upstream orbital pullback contribution.
   *
   * This diagnostics hook isolates the derivative of the orbital backprop
   * stage with respect to the orbital-preparation input while holding the
   * accepted-point upstream adjoints fixed.
   */
  Eigen::VectorXd apply_reduced_fixed_upstream_only(
      const Eigen::VectorXd& reduced_direction) const;

  /**
   * @brief Applies only the fixed-upstream orbital pullback contribution with cache shortcuts.
   *
   * This diagnostics hook forces the accepted-point orbital-preparation cache
   * on so cached and uncached implementations can be compared explicitly.
   */
  Eigen::VectorXd apply_reduced_fixed_upstream_only_cached(
      const Eigen::VectorXd& reduced_direction) const;

  /**
   * @brief Applies only the direct local/core upstream-delta contribution without cache shortcuts.
   *
   * This diagnostics hook bypasses the accepted-point orbital-preparation cache
   * so the direct-core path can be compared against the cached implementation.
   */
  Eigen::VectorXd apply_reduced_core_direct_only_uncached(
      const Eigen::VectorXd& reduced_direction) const;

  /**
   * @brief Applies only the fixed-upstream pullback contribution without cache shortcuts.
   *
   * This diagnostics hook bypasses the accepted-point orbital-preparation cache
   * so the fixed-upstream pullback can be compared against the cached
   * implementation.
   */
  Eigen::VectorXd apply_reduced_fixed_upstream_only_uncached(
      const Eigen::VectorXd& reduced_direction) const;

  /**
   * @brief Applies only the relaxed outer-response part of the reduced HVP.
   *
   * The local/core accepted-point orbital chain is skipped, so callers can add
   * this directional correction onto an already available cheap/core HVP result
   * instead of recomputing the full exact_ctx matvec from scratch.
   */
  Eigen::VectorXd apply_reduced_outer_response_only(
      const Eigen::VectorXd& reduced_direction) const;

  /**
   * @brief Applies only the relaxed outer-response HVP without cache shortcuts.
   *
   * This diagnostics hook preserves a direct uncached baseline for the
   * outer-response orbital pullback after the production path enables cache
   * reuse by default.
   */
  Eigen::VectorXd apply_reduced_outer_response_only_uncached(
      const Eigen::VectorXd& reduced_direction) const;

  /**
   * @brief Applies only the relaxed outer-response part of the reduced HVP with cache shortcuts.
   *
   * This diagnostics hook forces the accepted-point orbital-preparation cache
   * on so cached and uncached outer-response pullbacks can be compared
   * explicitly before enabling the cached path in the optimizer.
   */
  Eigen::VectorXd apply_reduced_outer_response_only_cached(
      const Eigen::VectorXd& reduced_direction) const;

  /**
   * @brief Returns the analytic directional structure matrices for one reduced direction.
   *
   * This is a diagnostics hook used to compare the accepted-point analytic
   * `\delta H_str` / `\delta S_str` against finite differences of the full
   * structure build.
   */
  DirectionalStructureDiagnostics compute_directional_structure_diagnostics(
      const Eigen::VectorXd& reduced_direction) const;

  /**
   * @brief Exposes the accepted-point direct-core intermediates for debugging.
   *
   * This diagnostics hook follows the same uncached analytic path as
   * `apply_reduced_core_direct_only_uncached()` and materializes the dense
   * AO/orbital pullback blocks before the final reduced-space projection.
   */
  DirectCoreDiagnostics compute_direct_core_diagnostics(
      const Eigen::VectorXd& reduced_direction) const;

  /**
   * @brief Exposes the raw fixed-upstream orbital pullback before reduction.
   *
   * This diagnostics hook isolates the directional derivative of the orbital
   * normalization/scatter stage with accepted upstream adjoints held fixed, so
   * raw sparse-slot, packed, and reduced responses can be compared separately.
   */
  FixedUpstreamDiagnostics compute_fixed_upstream_diagnostics(
      const Eigen::VectorXd& reduced_direction) const;

  /**
   * @brief Expands one reduced direction into explicit block-local rotations.
   *
   * This is primarily used while porting old orbital-Hessian formulas into a
   * direct-action form that works on accepted-point block rotations rather than
   * on packed sparse coefficients.
   */
  std::vector<NonredundantOrbitalSpace::BlockRotationDirection>
  expand_block_rotation_directions(
      const Eigen::VectorXd& reduced_direction) const;

  bool supports_analytic_core_model() const noexcept;

  Diagnostics diagnostics() const;

private:
  struct ApplyTimingTotals {
    std::size_t apply_count = 0;
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

  Eigen::VectorXd apply_reduced_impl(
      const Eigen::VectorXd& reduced_direction,
      bool include_direct_core_response,
      bool include_fixed_upstream_pullback,
      bool include_outer_response,
      bool use_cached_orbital_preparation_cache) const;

  std::shared_ptr<const CppActiveSpaceSecondOrderContext> accepted_point_context_;
  const CppVbInput* current_input_ = nullptr;
  SparseOrbitalParameterView parameter_view_;
  const NonredundantOrbitalSpace* nonredundant_space_ = nullptr;
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
  bool has_accepted_exact_two_electron_cache_ = false;
  mutable ExactPackedActiveTwoElectronApplyWorkspace
      accepted_exact_two_electron_apply_workspace_;
  mutable std::vector<double>
      outer_response_delta_active_orbital_overlap_matrix_workspace_;
  mutable std::vector<double>
      outer_response_delta_active_one_electron_matrix_workspace_;
  mutable std::vector<double>
      outer_response_symmetric_active_overlap_gradient_workspace_;
  mutable std::vector<double>
      outer_response_symmetric_active_one_electron_gradient_workspace_;
  mutable std::vector<double>
      outer_response_delta_packed_active_two_electron_workspace_;
  mutable ExactPackedActiveTwoElectronDirectionalDerivativeWorkspace
      outer_response_exact_two_electron_directional_workspace_;
  std::unique_ptr<AcceptedOrbitalPreparationCache> accepted_orbital_preparation_cache_;
  std::vector<StructureCoefficientBlock> structure_coefficient_blocks_;
  // Frozen accepted-point selected-state response metadata.  Each HVP still
  // builds fresh directional structure columns, but gap/gauge data and selected
  // eigenvector bookkeeping are reused across Krylov matvecs.
  AcceptedOuterResponseLinearResponseCache accepted_outer_response_cache_;
  mutable ApplyTimingTotals apply_timing_totals_;
};

/**
 * @brief Pairwise analytic reference for the local opposite-spin outer response.
 *
 * This diagnostics helper exposes the currently trusted pairwise fallback so
 * matrix-form implementations can be compared against the exact per-pair
 * local-response algebra without reconstructing the whole HVP.
 */
OppositeSpinMatrixBackwardContribution
build_pairwise_local_opposite_spin_matrix_backward_reference(
    const CppVbInput& input,
    const CppActiveSpaceSecondOrderContext& accepted_point_context,
    const DeterminantPairWeightTablesFromCoefficients& determinant_pair_weights,
    const std::vector<double>& delta_active_orbital_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals);

/**
 * @brief Pairwise analytic reference for the local same-spin outer response.
 *
 * This diagnostics helper exposes the trusted pairwise same-spin local term,
 * but returns the active overlap / one-electron channels with true matrix
 * semantics. The raw determinant-level canonical sweep stores off-diagonal
 * HHO/SSO asymmetrically; matrix-form same-spin compression loses that
 * bookkeeping detail and only preserves the physically relevant symmetric
 * matrix.
 */
SameSpinMatrixBackwardContribution
build_pairwise_local_same_spin_matrix_backward_reference(
    const CppVbInput& input,
    const CppActiveSpaceSecondOrderContext& accepted_point_context,
    const DeterminantPairWeightTablesFromCoefficients& determinant_pair_weights,
    const std::vector<double>& delta_active_orbital_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals);

}  // namespace xmvb::vb
