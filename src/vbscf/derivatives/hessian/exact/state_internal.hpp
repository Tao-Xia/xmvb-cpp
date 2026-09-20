#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include "vbscf/core/contracts/input.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/derivatives/hessian/exact/operator.hpp"
#include "vbscf/derivatives/hessian/responses/active_space/integral_direction.hpp"
#include "vbscf/derivatives/hessian/context/response_internal.hpp"
#include "vbscf/derivatives/hessian/responses/orbital/preparation.hpp"
#include "vbscf/integrals/active/two_electron/response/types.hpp"
#include "vbscf/integrals/active/two_electron/response/ri.hpp"
#include "vbscf/integrals/ao/one_electron/direct_operator.hpp"
#include "vbscf/integrals/ao/one_electron/ri_operator.hpp"
#include "vbscf/orbitals/charts/chart.hpp"
#include "vbscf/orbitals/charts/layout.hpp"
#include "vbscf/structures/assembly/coefficient_blocks.hpp"

namespace xmvb::vb {

struct AcceptedOrbitalPreparationCache;

struct ExactHvpOperator::State {
  struct PrecomputedDirection;

  State(
      std::shared_ptr<const AcceptedPointContext> accepted_point_context,
      const VbScfInput* current_input,
      SparseParameterLayout parameter_view,
      const OrbitalChart* nonredundant_space);

  Eigen::VectorXd apply_reduced(
      const Eigen::VectorXd& reduced_direction,
      HvpComponents components) const;

  Eigen::MatrixXd apply_reduced_batch(
      const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions,
      HvpComponents components) const;

  Eigen::VectorXd apply_structure_response_adjoint(
      const Eigen::Ref<const Eigen::MatrixXd>& coefficient_response,
      const Eigen::Ref<const Eigen::MatrixXd>& state_multipliers) const;

  SelectedStructureDirection apply_selected_structure_direction(
      const Eigen::VectorXd& reduced_direction) const;

  std::vector<SelectedStructureResponse>
  apply_selected_structure_response_batch(
      const std::vector<SelectedStructureResponse>& responses) const;

  const std::vector<double>& selected_state_weights() const noexcept;
  int n_structures() const noexcept;
  bool represents_accepted_point(
      const AcceptedPointContext& accepted_point) const noexcept;

  bool supports_analytic_core_model() const noexcept;
  Diagnostics diagnostics() const;

private:
  const AcceptedOuterResponseContext& outer_response_context() const;

  PrecomputedDirection prepare_direction(
      Eigen::VectorXd packed_direction) const;

  ActiveSpaceIntegralDirectionView build_active_integral_direction(
      const OrbitalPreparationDirectionalResult& orbital_direction,
      const Eigen::Ref<const Eigen::MatrixXd>& delta_h1e_times_active,
      const Eigen::VectorXd* precomputed_delta_packed_two_electron,
      Eigen::VectorXd* ri_delta_packed_two_electron,
      ActiveSpaceIntegralDirectionWorkspace* workspace) const;

  Eigen::VectorXd apply_reduced_impl(
      const Eigen::VectorXd& reduced_direction,
      HvpComponents components,
      const Eigen::VectorXd* precomputed_delta_ao_effective_h1e,
      const Eigen::VectorXd* precomputed_inactive_density_gradient,
      const Eigen::VectorXd* precomputed_delta_packed_active_two_electron,
      const Eigen::MatrixXd* precomputed_ri_active_pair_factor_direction,
      const ExactCtxPairMatrix* precomputed_directional_pair_products,
      const Eigen::MatrixXd* precomputed_two_electron_fixed_adjoint,
      const PrecomputedDirection* precomputed_direction) const;

  struct ApplyTimingTotals {
    std::size_t apply_count = 0;
    std::size_t batch_apply_count = 0;
    std::size_t structure_response_block_actions = 0;
    int max_structure_response_iterations = 0;
    double max_structure_response_relative_residual = 0.0;
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
    double outer_response_local_active_gradient_wall_time_seconds = 0.0;
    double outer_response_structure_active_gradient_wall_time_seconds = 0.0;
    double outer_response_selected_state_rebuild_wall_time_seconds = 0.0;
    double outer_response_same_spin_backward_wall_time_seconds = 0.0;
    double outer_response_opposite_spin_backward_wall_time_seconds = 0.0;
    double outer_response_opposite_spin_packed_gradient_wall_time_seconds = 0.0;
    double outer_response_opposite_spin_alpha_overlap_wall_time_seconds = 0.0;
    double outer_response_opposite_spin_beta_overlap_wall_time_seconds = 0.0;
    double outer_response_orbital_pullback_wall_time_seconds = 0.0;
  };

  std::shared_ptr<const AcceptedPointContext> accepted_point_context_;
  const VbScfInput* current_input_ = nullptr;
  SparseParameterLayout parameter_view_;
  const OrbitalChart* nonredundant_space_ = nullptr;
  Eigen::MatrixXd accepted_active_auxiliary_orbitals_;
  Eigen::MatrixXd accepted_basis_overlap_times_active_auxiliary_orbitals_;
  Eigen::MatrixXd accepted_ao_effective_one_electron_times_active_auxiliary_orbitals_;
  Eigen::MatrixXd
      accepted_ao_effective_one_electron_transpose_times_active_auxiliary_orbitals_;
  Eigen::MatrixXd accepted_sso_gradient_symmetric_;
  Eigen::MatrixXd accepted_hho_gradient_symmetric_;
  Eigen::MatrixXd accepted_active_auxiliary_orbitals_times_hho_gradient_symmetric_;
  mutable Eigen::MatrixXd ao_h1e_symmetrized_gradient_workspace_;
  mutable AoH1eFusedWorkspace ao_h1e_fused_workspace_;
  mutable AoEffectiveOneElectronRiFusedWorkspace ri_ao_h1e_fused_workspace_;
  mutable Eigen::MatrixXd ri_ao_h1e_forward_workspace_;
  mutable Eigen::MatrixXd ri_ao_h1e_adjoint_workspace_;
  mutable AoEffectiveOneElectronRiStrategy ri_ao_h1e_strategy_ =
      AoEffectiveOneElectronRiStrategy::Uncalibrated;
  Eigen::MatrixXd accepted_total_active_auxiliary_gradient_;
  std::vector<double> accepted_total_inactive_density_gradient_;
  Eigen::MatrixXd zero_core_hamiltonian_;
  mutable std::vector<double> ao_h1e_delta_h1e_workspace_;
  mutable std::vector<double> ao_h1e_inactive_density_gradient_workspace_;
  ExactPackedActiveTwoElectronAdjointCache accepted_exact_two_electron_cache_;
  std::optional<RiActiveTwoElectronResponseCache>
      accepted_ri_two_electron_cache_;
  const RiAoFactorization* accepted_ri_factorization_ = nullptr;
  mutable ExactPackedActiveTwoElectronApplyWorkspace
      accepted_exact_two_electron_apply_workspace_;
  mutable ActiveSpaceIntegralDirectionWorkspace
      outer_response_integral_direction_workspace_;
  mutable std::vector<double>
      outer_response_symmetric_active_overlap_gradient_workspace_;
  mutable std::vector<double>
      outer_response_symmetric_active_one_electron_gradient_workspace_;
  std::unique_ptr<AcceptedOrbitalPreparationCache> accepted_orbital_preparation_cache_;
  mutable std::unique_ptr<AcceptedOuterResponseContext>
      accepted_outer_response_context_;
  mutable ApplyTimingTotals apply_timing_totals_;
};

}  // namespace xmvb::vb
