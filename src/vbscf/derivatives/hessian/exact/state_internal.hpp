#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <Eigen/Core>

#include "vbscf/core/contracts/input.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/derivatives/hessian/exact/operator.hpp"
#include "vbscf/derivatives/hessian/responses/active_space/integral_direction.hpp"
#include "vbscf/derivatives/hessian/context/response_internal.hpp"
#include "vbscf/integrals/active/two_electron/response/types.hpp"
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

  bool supports_analytic_core_model() const noexcept;
  Diagnostics diagnostics() const;

private:
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
  AcceptedOuterResponseContext accepted_outer_response_context_;
  mutable ApplyTimingTotals apply_timing_totals_;
};

}  // namespace xmvb::vb
