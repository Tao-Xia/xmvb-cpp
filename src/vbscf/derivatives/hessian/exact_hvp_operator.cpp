#include "vbscf/derivatives/hessian/exact_hvp_operator.hpp"

#include <stdexcept>
#include <utility>

#include <Eigen/Core>

#include "vbscf/structures/coefficient_blocks.hpp"
#include "vbscf/integrals/active/active_space_two_electron_adjoint.hpp"
#include "vbscf/derivatives/hessian/responses/active_space_outer_response.hpp"
#include "vbscf/derivatives/hessian/responses/orbital_preparation_response.hpp"

namespace xmvb::vb {

namespace {

bool exact_ctx_stage1_analytic_core_enabled() {
  return true;
}

}  // namespace

ExactHvpOperator::ExactHvpOperator(
    std::shared_ptr<const AcceptedPointContext> accepted_point_context,
    const VbScfInput* current_input,
    SparseParameterLayout parameter_view,
    const OrbitalChart* nonredundant_space)
    : accepted_point_context_(std::move(accepted_point_context)),
      current_input_(current_input),
      parameter_view_(std::move(parameter_view)),
      nonredundant_space_(nonredundant_space) {
  if (accepted_point_context_ == nullptr) {
    throw std::invalid_argument(
        "accepted-point second-order context must not be null");
  }
  if (current_input_ == nullptr) {
    throw std::invalid_argument("current_input must not be null");
  }
  if (nonredundant_space_ == nullptr) {
    throw std::invalid_argument("nonredundant_space must not be null");
  }
  if (supports_analytic_core_model()) {
    const int n_basis_functions =
        current_input_->orbital_preparation_input.n_basis_functions;
    const int n_active_orbitals =
        current_input_->orbital_preparation_input.n_active_orbitals;
    const int n_inactive_doubly_occupied_orbitals =
        (current_input_->orbital_preparation_input.n_total_electrons -
         current_input_->orbital_preparation_input.n_active_electrons) /
        2;
    const std::size_t ao_matrix_size =
        n_basis_functions * n_basis_functions;
    const std::size_t active_matrix_size =
        n_active_orbitals * n_active_orbitals;
    if (accepted_point_context_->prepared_active_space.orbital_result
                .auxiliary_orbital_matrix.size() == ao_matrix_size &&
        accepted_point_context_->active_orbital_overlap_gradient.size() ==
            active_matrix_size &&
        accepted_point_context_->active_one_electron_gradient.size() ==
            active_matrix_size) {
      const Eigen::Map<const Eigen::MatrixXd> basis_overlap(
          current_input_->orbital_preparation_input.ao_overlap_matrix.data(),
          n_basis_functions,
          n_basis_functions);
      const Eigen::Map<const Eigen::MatrixXd> accepted_auxiliary_matrix(
          accepted_point_context_->prepared_active_space.orbital_result
              .auxiliary_orbital_matrix.data(),
          n_basis_functions,
          n_basis_functions);
      const Eigen::Map<const Eigen::MatrixXd> accepted_ao_effective_h1e(
          accepted_point_context_->prepared_active_space.ao_effective_one_electron_result
              .ao_effective_h1e.data(),
          n_basis_functions,
          n_basis_functions);
      accepted_active_auxiliary_orbitals_ =
          accepted_auxiliary_matrix.middleCols(
              n_inactive_doubly_occupied_orbitals,
              n_active_orbitals);
      accepted_basis_overlap_times_active_auxiliary_orbitals_ =
          basis_overlap * accepted_active_auxiliary_orbitals_;
      accepted_ao_effective_one_electron_times_active_auxiliary_orbitals_ =
          accepted_ao_effective_h1e * accepted_active_auxiliary_orbitals_;
      accepted_ao_effective_one_electron_transpose_times_active_auxiliary_orbitals_ =
          accepted_ao_effective_h1e.transpose() *
          accepted_active_auxiliary_orbitals_;
      const Eigen::Map<const Eigen::MatrixXd> accepted_sso_gradient(
          accepted_point_context_->active_orbital_overlap_gradient.data(),
          n_active_orbitals,
          n_active_orbitals);
      accepted_sso_gradient_symmetric_ =
          accepted_sso_gradient + accepted_sso_gradient.transpose();
      const Eigen::Map<const Eigen::MatrixXd> accepted_hho_gradient(
          accepted_point_context_->active_one_electron_gradient.data(),
          n_active_orbitals,
          n_active_orbitals);
      accepted_hho_gradient_symmetric_ =
          accepted_hho_gradient + accepted_hho_gradient.transpose();
      accepted_active_auxiliary_orbitals_times_hho_gradient_symmetric_ =
          accepted_active_auxiliary_orbitals_ *
          accepted_hho_gradient_symmetric_;
      const auto& accepted_active_space_two_electron_result =
          accepted_point_context_->prepared_active_space.active_space_two_electron_result;
      if (accepted_active_space_two_electron_result.dense_active_coefficients.size() != 0) {
        accepted_dense_active_coefficients_ =
            accepted_active_space_two_electron_result.dense_active_coefficients;
      } else {
        accepted_dense_active_coefficients_ = accepted_active_auxiliary_orbitals_;
      }
      zero_core_hamiltonian_ =
          Eigen::MatrixXd::Zero(n_basis_functions, n_basis_functions);
      accepted_exact_two_electron_cache_ =
          build_exact_packed_active_two_electron_adjoint_cache(
              accepted_point_context_->packed_active_two_electron_gradient,
              accepted_dense_active_coefficients_,
              current_input_->ao_integral_input,
              n_active_orbitals,
              &accepted_point_context_->prepared_active_space
                   .active_space_two_electron_result);
    }

    const auto accepted_orbital_backprop_inputs =
        build_accepted_orbital_backprop_inputs(
            *accepted_point_context_,
            *current_input_);
    accepted_total_active_auxiliary_gradient_ =
        accepted_orbital_backprop_inputs.total_active_auxiliary_gradient;
    accepted_total_inactive_density_gradient_ =
        accepted_orbital_backprop_inputs.total_inactive_density_gradient;

    accepted_orbital_preparation_cache_ =
        std::make_unique<AcceptedOrbitalPreparationCache>(
            build_accepted_orbital_preparation_cache(
                current_input_->orbital_preparation_input,
                accepted_total_active_auxiliary_gradient_,
                accepted_total_inactive_density_gradient_));

    if (accepted_point_context_->same_spin_pair_cache.enabled()) {
      structure_coefficient_blocks_ =
          build_structure_coefficient_blocks(
              current_input_->structure_data.determinant_to_structure_terms,
              current_input_->structure_data.n_structures,
              accepted_point_context_->same_spin_pair_cache.alpha_reuse_table,
              accepted_point_context_->same_spin_pair_cache.beta_reuse_table,
              true);
      accepted_outer_response_context_ =
          build_accepted_outer_response_context(
              current_input_,
              accepted_point_context_.get(),
              &structure_coefficient_blocks_);
    }
  }
}

ExactHvpOperator::~ExactHvpOperator() = default;

bool ExactHvpOperator::supports_analytic_core_model() const noexcept {
  if (accepted_point_context_ == nullptr ||
      current_input_ == nullptr ||
      nonredundant_space_ == nullptr) {
    return false;
  }
  if (!exact_ctx_stage1_analytic_core_enabled()) {
    return false;
  }
  if (current_input_->standard_two_electron_mode ==
      StandardTwoElectronMode::ResolutionOfIdentity) {
    return false;
  }
  if (current_input_->ao_integral_input.ao_two_electron_integral_values.empty() ||
      current_input_->ao_integral_input.ao_two_electron_integral_indices.empty()) {
    return false;
  }
  const int n_active_orbitals =
      current_input_->orbital_preparation_input.n_active_orbitals;
  return n_active_orbitals > 0 &&
      has_active_matrix_gradient(*accepted_point_context_, n_active_orbitals);
}

ExactHvpOperator::Diagnostics
ExactHvpOperator::diagnostics() const {
  Diagnostics info;
  if (accepted_point_context_ == nullptr || nonredundant_space_ == nullptr) {
    return info;
  }
  info.supports_analytic_core_model = supports_analytic_core_model();
  info.outer_response_enabled = true;
  info.outer_response_local_only_approximation = false;
  info.outer_response_energy_only_approximation = false;
  info.used_reduced_curvature_diagonal =
      nonredundant_space_->has_reduced_curvature_diagonal();
  info.has_same_spin_matrix_form =
      accepted_point_context_->use_full_matrix_form_adjoint;
  info.has_opposite_spin_matrix_form =
      accepted_point_context_->use_matrix_form_opposite_spin;
  info.n_selected_states =
      static_cast<int>(accepted_point_context_->selected_state_indices.size());
  info.n_active_orbitals = accepted_point_context_->n_active_orbitals;
  info.n_blocks = nonredundant_space_->n_blocks();
  info.apply_count = apply_timing_totals_.apply_count;
  info.batch_apply_count = apply_timing_totals_.batch_apply_count;
  info.total_apply_wall_time_seconds =
      apply_timing_totals_.total_apply_wall_time_seconds;
  info.core_setup_wall_time_seconds =
      apply_timing_totals_.core_setup_wall_time_seconds;
  info.ao_effective_one_electron_build_wall_time_seconds =
      apply_timing_totals_.ao_effective_one_electron_build_wall_time_seconds;
  info.ao_effective_one_electron_fused_wall_time_seconds =
      apply_timing_totals_.ao_effective_one_electron_fused_wall_time_seconds;
  info.active_two_electron_wall_time_seconds =
      apply_timing_totals_.active_two_electron_wall_time_seconds;
  info.ao_effective_one_electron_backprop_wall_time_seconds =
      apply_timing_totals_.ao_effective_one_electron_backprop_wall_time_seconds;
  info.orbital_backprop_wall_time_seconds =
      apply_timing_totals_.orbital_backprop_wall_time_seconds;
  info.fixed_upstream_pullback_wall_time_seconds =
      apply_timing_totals_.fixed_upstream_pullback_wall_time_seconds;
  info.outer_response_wall_time_seconds =
      apply_timing_totals_.outer_response_wall_time_seconds;
  info.outer_response_active_space_integrals_wall_time_seconds =
      apply_timing_totals_.outer_response_active_space_integrals_wall_time_seconds;
  info.outer_response_structure_matrices_wall_time_seconds =
      apply_timing_totals_.outer_response_structure_matrices_wall_time_seconds;
  info.outer_response_eigensystem_wall_time_seconds =
      apply_timing_totals_.outer_response_eigensystem_wall_time_seconds;
  info.outer_response_pair_weights_wall_time_seconds =
      apply_timing_totals_.outer_response_pair_weights_wall_time_seconds;
  info.outer_response_active_gradient_wall_time_seconds =
      apply_timing_totals_.outer_response_active_gradient_wall_time_seconds;
  info.outer_response_orbital_pullback_wall_time_seconds =
      apply_timing_totals_.outer_response_orbital_pullback_wall_time_seconds;
  return info;
}

}  // namespace xmvb::vb
