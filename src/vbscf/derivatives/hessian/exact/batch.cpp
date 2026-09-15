#include "vbscf/derivatives/hessian/exact/operator.hpp"

#include "vbscf/derivatives/hessian/exact/ao_one_electron_internal.hpp"
#include "vbscf/derivatives/hessian/exact/apply_internal.hpp"

#include <chrono>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/responses/orbital/preparation.hpp"
#include "vbscf/derivatives/hessian/responses/structure/directional.hpp"
#include "vbscf/integrals/active/two_electron/response/directional.hpp"
#include "vbscf/integrals/ao/one_electron/direct_operator.hpp"

namespace xmvb::vb {

Eigen::MatrixXd ExactHvpOperator::apply_reduced_batch(
    const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions,
    HvpComponents components) const {
  return state_->apply_reduced_batch(reduced_directions, components);
}

Eigen::MatrixXd ExactHvpOperator::State::apply_reduced_batch(
    const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions,
    HvpComponents components) const {
  ++apply_timing_totals_.batch_apply_count;
  Eigen::MatrixXd responses(
      reduced_directions.rows(),
      reduced_directions.cols());
  if (reduced_directions.cols() <= 1) {
    for (Eigen::Index column = 0;
         column < reduced_directions.cols();
         ++column) {
      responses.col(column) =
          apply_reduced(reduced_directions.col(column), components);
    }
    return responses;
  }
  if (!supports_analytic_core_model()) {
    throw std::runtime_error(
        "exact_ctx analytic core HVP is unavailable for the current accepted point");
  }

  const int n_basis_functions =
      current_input_->orbital_preparation_input.n_basis_functions;
  const int n_active_orbitals =
      current_input_->orbital_preparation_input.n_active_orbitals;
  const int n_inactive_doubly_occupied_orbitals =
      (current_input_->orbital_preparation_input.n_total_electrons -
       current_input_->orbital_preparation_input.n_active_electrons) /
      2;
  const Eigen::Index ao_matrix_size =
      static_cast<Eigen::Index>(n_basis_functions) * n_basis_functions;
  const Eigen::Index n_directions = reduced_directions.cols();
  Eigen::MatrixXd inactive_density_columns(ao_matrix_size, n_directions);
  Eigen::MatrixXd symmetrized_pullback_columns(ao_matrix_size, n_directions);
  std::vector<PrecomputedDirection> precomputed_directions;
  precomputed_directions.reserve(n_directions);
  std::vector<Eigen::MatrixXd> dense_active_directions;
  dense_active_directions.reserve(n_directions);
  for (Eigen::Index column = 0; column < n_directions; ++column) {
    PrecomputedDirection precomputed;
    precomputed.packed_direction =
        nonredundant_space_->expand_step(reduced_directions.col(column));
    precomputed.dense_orbital_tangent_context =
        build_dense_orbital_tangent_context(
            current_input_->orbital_preparation_input,
            parameter_view_,
            precomputed.packed_direction,
            *accepted_orbital_preparation_cache_);
    precomputed.orbital_preparation_directional_result =
        build_orbital_preparation_directional_result(
            current_input_->orbital_preparation_input,
            precomputed.dense_orbital_tangent_context,
            n_inactive_doubly_occupied_orbitals,
            n_active_orbitals,
            *accepted_orbital_preparation_cache_);
    const auto& directional_result =
        precomputed.orbital_preparation_directional_result;
    dense_active_directions.push_back(
        directional_result.delta_active_auxiliary_orbitals);
    Eigen::MatrixXd pullback_source = directional_result.delta_inactive_density;
    pullback_source.noalias() +=
        (directional_result.delta_active_auxiliary_orbitals *
         accepted_hho_gradient_symmetric_) *
        accepted_active_auxiliary_orbitals_.transpose();
    inactive_density_columns.col(column) = Eigen::Map<const Eigen::VectorXd>(
        directional_result.delta_inactive_density.data(), ao_matrix_size);
    const Eigen::MatrixXd symmetrized_pullback =
        pullback_source + pullback_source.transpose();
    symmetrized_pullback_columns.col(column) =
        Eigen::Map<const Eigen::VectorXd>(
            symmetrized_pullback.data(), ao_matrix_size);
    precomputed_directions.push_back(std::move(precomputed));
  }

  const auto batch_h1e_start_time = std::chrono::steady_clock::now();
  Eigen::MatrixXd delta_h1e_columns;
  Eigen::MatrixXd inactive_density_gradient_columns;
  apply_ao_h1e_fused_batch(
      inactive_density_columns,
      symmetrized_pullback_columns,
      current_input_->ao_integral_input,
      detail::choose_exact_ao_h1e_thread_count(
          current_input_->orbital_preparation_input),
      &delta_h1e_columns,
      &inactive_density_gradient_columns);
  for (Eigen::Index column = 0; column < n_directions; ++column) {
    Eigen::Map<Eigen::MatrixXd> delta_h1e(
        delta_h1e_columns.col(column).data(),
        n_basis_functions,
        n_basis_functions);
    for (int row = 0; row < n_basis_functions; ++row) {
      for (int matrix_column = 0; matrix_column <= row; ++matrix_column) {
        delta_h1e(row, matrix_column) += delta_h1e(matrix_column, row);
        delta_h1e(matrix_column, row) = delta_h1e(row, matrix_column);
      }
    }
  }
  const double batch_h1e_seconds =
      detail::exact_hvp_elapsed_seconds(batch_h1e_start_time);
  apply_timing_totals_.ao_effective_one_electron_fused_wall_time_seconds +=
      batch_h1e_seconds;
  apply_timing_totals_.total_apply_wall_time_seconds += batch_h1e_seconds;

  Eigen::MatrixXd delta_packed_active_two_electron_columns;
  std::vector<ExactCtxPairMatrix> directional_pair_products;
  std::vector<Eigen::MatrixXd> two_electron_fixed_adjoint_directions;
  if (components.outer_response) {
    const auto batch_active_two_electron_start_time =
        std::chrono::steady_clock::now();
    delta_packed_active_two_electron_columns =
        compute_exact_packed_active_two_electron_integral_directional_derivative_batch(
            accepted_exact_two_electron_cache_,
            dense_active_directions,
            current_input_->ao_integral_input,
            &directional_pair_products,
            &two_electron_fixed_adjoint_directions);
    const double batch_active_two_electron_seconds =
        detail::exact_hvp_elapsed_seconds(batch_active_two_electron_start_time);
    apply_timing_totals_
        .outer_response_active_space_integrals_wall_time_seconds +=
        batch_active_two_electron_seconds;
    apply_timing_totals_.outer_response_wall_time_seconds +=
        batch_active_two_electron_seconds;
    apply_timing_totals_.total_apply_wall_time_seconds +=
        batch_active_two_electron_seconds;
  }

  if (components.outer_response) {
    const auto outer_batch_start = std::chrono::steady_clock::now();
    const int n_selected_states = static_cast<int>(
        accepted_point_context_->selected_state_indices.size());
    const int n_structures = accepted_point_context_->n_structures;
    Eigen::MatrixXd delta_hamiltonian_selected(
        n_structures, n_directions * n_selected_states);
    Eigen::MatrixXd delta_overlap_selected(
        n_structures, n_directions * n_selected_states);
    const ActiveSpaceIntegralDirectionContext integral_context{
        *current_input_,
        accepted_exact_two_electron_cache_,
        accepted_active_auxiliary_orbitals_,
        accepted_basis_overlap_times_active_auxiliary_orbitals_,
        accepted_ao_effective_one_electron_times_active_auxiliary_orbitals_,
        accepted_ao_effective_one_electron_transpose_times_active_auxiliary_orbitals_,
        *accepted_exact_two_electron_cache_.accepted_active_coefficients};

    double integral_seconds = 0.0;
    double structure_seconds = 0.0;
    for (Eigen::Index column = 0; column < n_directions; ++column) {
      PrecomputedOuterResponse& outer =
          precomputed_directions[column].outer_response.emplace();
      const Eigen::Map<const Eigen::MatrixXd> delta_h1e(
          delta_h1e_columns.col(column).data(),
          n_basis_functions,
          n_basis_functions);
      const Eigen::MatrixXd delta_h1e_times_active =
          delta_h1e * accepted_active_auxiliary_orbitals_;

      const auto integral_start = std::chrono::steady_clock::now();
      const Eigen::VectorXd delta_packed_two_electron =
          delta_packed_active_two_electron_columns.col(column);
      const ActiveSpaceIntegralDirectionView integral_direction =
          build_active_space_integral_direction(
              integral_context,
              ActiveSpaceIntegralTangent{
                  precomputed_directions[column]
                      .orbital_preparation_directional_result
                      .delta_active_auxiliary_orbitals,
                  precomputed_directions[column]
                      .orbital_preparation_directional_result
                      .delta_active_auxiliary_orbitals,
                  delta_h1e_times_active,
                  &delta_packed_two_electron},
              &outer.integral_direction);
      integral_seconds += detail::exact_hvp_elapsed_seconds(integral_start);

      const auto structure_start = std::chrono::steady_clock::now();
      outer.pair_cache = build_same_spin_directional_pair_cache(
          accepted_point_context_->same_spin_pair_cache,
          n_active_orbitals,
          integral_direction);
      const SelectedStateDirectionalStructureImages images =
          build_selected_structure_direction(
              accepted_outer_response_context_,
              integral_direction,
              outer.pair_cache);
      const int first = static_cast<int>(column) * n_selected_states;
      delta_hamiltonian_selected.middleCols(first, n_selected_states) =
          images.delta_hamiltonian_selected;
      delta_overlap_selected.middleCols(first, n_selected_states) =
          images.delta_overlap_selected;
      outer.integral_direction.packed_two_electron.clear();
      structure_seconds += detail::exact_hvp_elapsed_seconds(structure_start);
    }
    apply_timing_totals_
        .outer_response_active_space_integrals_wall_time_seconds +=
        integral_seconds;
    apply_timing_totals_
        .outer_response_structure_matrices_wall_time_seconds +=
        structure_seconds;

    const auto eigensystem_start = std::chrono::steady_clock::now();
    const SelectedStateGeneralizedEigenDirectionalResponse block_response =
        accepted_outer_response_context_
            .selected_state_eigen_response_operator.apply_direction_block(
                delta_hamiltonian_selected,
                delta_overlap_selected);
    apply_timing_totals_.outer_response_eigensystem_wall_time_seconds +=
        detail::exact_hvp_elapsed_seconds(eigensystem_start);

    for (Eigen::Index column = 0; column < n_directions; ++column) {
      const int first = static_cast<int>(column) * n_selected_states;
      auto& response = precomputed_directions[column]
                           .outer_response
                           ->selected_state_response;
      response.delta_selected_eigenvector_matrix =
          block_response.delta_selected_eigenvector_matrix.middleCols(
              first, n_selected_states);
      response.delta_selected_eigenvalues.assign(
          block_response.delta_selected_eigenvalues.begin() + first,
          block_response.delta_selected_eigenvalues.begin() +
              first + n_selected_states);
      response.linear_iterations.assign(
          block_response.linear_iterations.begin() + first,
          block_response.linear_iterations.begin() +
              first + n_selected_states);
      if (column == 0) {
        response.block_actions = block_response.block_actions;
        response.max_relative_residual =
            block_response.max_relative_residual;
      }
    }
    const double outer_batch_seconds =
        detail::exact_hvp_elapsed_seconds(outer_batch_start);
    apply_timing_totals_.outer_response_wall_time_seconds +=
        outer_batch_seconds;
    apply_timing_totals_.total_apply_wall_time_seconds +=
        outer_batch_seconds;
  }

  for (Eigen::Index column = 0;
       column < reduced_directions.cols();
       ++column) {
    const Eigen::VectorXd delta_h1e = delta_h1e_columns.col(column);
    const Eigen::VectorXd inactive_density_gradient =
        inactive_density_gradient_columns.col(column);
    const Eigen::VectorXd delta_packed_active_two_electron =
        components.outer_response
            ? delta_packed_active_two_electron_columns.col(column)
            : Eigen::VectorXd();
    if (components.outer_response) {
      auto& packed = precomputed_directions[column]
                         .outer_response
                         ->integral_direction
                         .packed_two_electron;
      packed.assign(
          delta_packed_active_two_electron.data(),
          delta_packed_active_two_electron.data() +
              delta_packed_active_two_electron.size());
    }
    const Eigen::VectorXd response = apply_reduced_impl(
        reduced_directions.col(column),
        components,
        &delta_h1e,
        &inactive_density_gradient,
        components.outer_response
            ? &delta_packed_active_two_electron
            : nullptr,
        components.outer_response
            ? &directional_pair_products[column]
            : nullptr,
        components.outer_response &&
                two_electron_fixed_adjoint_directions[column].size() != 0
            ? &two_electron_fixed_adjoint_directions[column]
            : nullptr,
        &precomputed_directions[column]);
    responses.col(column) = response;
    if (components.outer_response) {
      precomputed_directions[column]
          .outer_response
          ->integral_direction
          .packed_two_electron.clear();
    }
  }
  return responses;
}

}  // namespace xmvb::vb
