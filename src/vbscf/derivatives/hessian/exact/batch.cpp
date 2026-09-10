#include "vbscf/derivatives/hessian/exact/operator.hpp"

#include "vbscf/derivatives/hessian/exact/ao_one_electron_internal.hpp"
#include "vbscf/derivatives/hessian/exact/apply_internal.hpp"

#include <chrono>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/responses/orbital/preparation.hpp"
#include "vbscf/integrals/active/two_electron/response/directional.hpp"
#include "vbscf/integrals/ao/one_electron/graph_operator.hpp"

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
  if (reduced_directions.cols() <= 1 ||
      !ao_effective_one_electron_graph_available(
          current_input_->ao_integral_input)) {
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
  apply_fused_ao_effective_one_electron_graph_batch(
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
  if (components.outer_response) {
    const auto batch_active_two_electron_start_time =
        std::chrono::steady_clock::now();
    delta_packed_active_two_electron_columns =
        compute_exact_packed_active_two_electron_integral_directional_derivative_batch(
            accepted_exact_two_electron_cache_,
            dense_active_directions,
            current_input_->ao_integral_input,
            &directional_pair_products);
    const double batch_active_two_electron_seconds =
        detail::exact_hvp_elapsed_seconds(batch_active_two_electron_start_time);
    apply_timing_totals_
        .outer_response_active_space_integrals_wall_time_seconds +=
        batch_active_two_electron_seconds;
    apply_timing_totals_.total_apply_wall_time_seconds +=
        batch_active_two_electron_seconds;
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
    responses.col(column) = apply_reduced_impl(
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
        &precomputed_directions[column]);
  }
  return responses;
}

}  // namespace xmvb::vb
