#include "vbscf/derivatives/hessian/exact/operator.hpp"

#include "vbscf/derivatives/hessian/exact/ao_one_electron_internal.hpp"
#include "vbscf/derivatives/hessian/exact/apply_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "core/eigen_response.hpp"
#include "vbscf/derivatives/hessian/coupled/coupling.hpp"
#include "vbscf/derivatives/hessian/responses/orbital/preparation.hpp"
#include "vbscf/derivatives/hessian/responses/structure/directional.hpp"
#include "vbscf/integrals/active/two_electron/response/directional.hpp"
#include "vbscf/integrals/active/two_electron/response/ri.hpp"
#include "vbscf/integrals/ao/one_electron/direct_operator.hpp"
#include "vbscf/integrals/ao/one_electron/ri_operator.hpp"

namespace xmvb::vb {
namespace {

constexpr std::size_t kOrbitalCouplingBatchWorkspaceBytes =
    256ULL * 1024ULL * 1024ULL;
constexpr Eigen::Index kMaximumOrbitalCouplingBatchColumns = 8;

Eigen::Index bounded_orbital_coupling_width(
    Eigen::Index requested,
    int n_basis_functions,
    int n_active_orbitals,
    int n_structures,
    int n_states) {
  const std::size_t n_bf = static_cast<std::size_t>(n_basis_functions);
  const std::size_t n_active = static_cast<std::size_t>(n_active_orbitals);
  const std::size_t ao_matrix = n_bf * n_bf;
  const std::size_t ao_pairs = n_bf * (n_bf + 1ULL) / 2ULL;
  const std::size_t active_pairs =
      n_active * (n_active + 1ULL) / 2ULL;
  const std::size_t packed_active_two_electron =
      active_pairs * (active_pairs + 1ULL) / 2ULL;
  // Four AO matrices cover the two batch inputs and two batch outputs. The
  // remaining terms bound the exact-2e directional pair products and their
  // fixed-adjoint image, dense active direction, packed active 2e direction,
  // and selected-state forcing retained for one column. RI allocates less,
  // so this bound remains conservative for both integral representations.
  const std::size_t elements_per_column =
      4ULL * ao_matrix + 2ULL * ao_pairs * active_pairs +
      n_bf * n_active +
      packed_active_two_electron +
      static_cast<std::size_t>(n_structures) *
          static_cast<std::size_t>(n_states);
  const std::size_t bytes_per_column =
      std::max<std::size_t>(sizeof(double),
                            elements_per_column * sizeof(double));
  const Eigen::Index budget_width = static_cast<Eigen::Index>(
      std::max<std::size_t>(
          1ULL, kOrbitalCouplingBatchWorkspaceBytes / bytes_per_column));
  return std::max<Eigen::Index>(
      1,
      std::min({requested,
                kMaximumOrbitalCouplingBatchColumns,
                budget_width}));
}

}  // namespace

Eigen::MatrixXd ExactHvpOperator::apply_reduced_batch(
    const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions,
    HvpComponents components) const {
  return state_->apply_reduced_batch(reduced_directions, components);
}

OrbitalCouplingBlockAction ExactHvpOperator::apply_orbital_coupling_batch(
    const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions) const {
  return state_->apply_orbital_coupling_batch(reduced_directions);
}

Eigen::MatrixXd ExactHvpOperator::State::apply_reduced_batch(
    const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions,
    HvpComponents components) const {
  return apply_reduced_batch_impl(reduced_directions, components, nullptr);
}

OrbitalCouplingBlockAction
ExactHvpOperator::State::apply_orbital_coupling_batch(
    const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions) const {
  if (reduced_directions.rows() != nonredundant_space_->reduced_size() ||
      reduced_directions.cols() <= 0 || !reduced_directions.allFinite()) {
    throw std::invalid_argument(
        "orbital coupling block has incompatible dimensions or values");
  }
  ++apply_timing_totals_.orbital_coupling_batch_count;
  const Eigen::Index n_states = static_cast<Eigen::Index>(
      accepted_point_context_->selected_state_indices.size());
  const Eigen::Index chunk_width = bounded_orbital_coupling_width(
      reduced_directions.cols(),
      current_input_->orbital_preparation_input.n_basis_functions,
      current_input_->orbital_preparation_input.n_active_orbitals,
      accepted_point_context_->n_structures,
      static_cast<int>(n_states));

  OrbitalCouplingBlockAction result;
  result.orbital_hessian.resize(
      reduced_directions.rows(), reduced_directions.cols());
  result.scaled_structure_forcing.resize(
      static_cast<std::size_t>(reduced_directions.cols()));
  for (Eigen::Index first = 0; first < reduced_directions.cols();
       first += chunk_width) {
    const Eigen::Index width = std::min(
        chunk_width, reduced_directions.cols() - first);
    OrbitalCouplingBlockAction chunk;
    apply_reduced_batch_impl(
        reduced_directions.middleCols(first, width),
        {.direct_core_response = true,
         .fixed_upstream_pullback = true,
         .local_active_response = true,
         .structure_response = false},
        &chunk);
    result.orbital_hessian.middleCols(first, width) =
        chunk.orbital_hessian;
    for (Eigen::Index column = 0; column < width; ++column) {
      result.scaled_structure_forcing[static_cast<std::size_t>(first + column)] =
          std::move(chunk.scaled_structure_forcing[
              static_cast<std::size_t>(column)]);
    }
    ++apply_timing_totals_.orbital_coupling_batch_chunk_count;
    apply_timing_totals_.max_orbital_coupling_batch_width = std::max(
        apply_timing_totals_.max_orbital_coupling_batch_width,
        static_cast<std::size_t>(width));
  }
  return result;
}

Eigen::MatrixXd ExactHvpOperator::State::apply_reduced_batch_impl(
    const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions,
    HvpComponents components,
    OrbitalCouplingBlockAction* coupling_output) const {
  ++apply_timing_totals_.batch_apply_count;
  Eigen::MatrixXd responses(
      reduced_directions.rows(),
      reduced_directions.cols());
  if (coupling_output != nullptr) {
    coupling_output->orbital_hessian.resize(
        reduced_directions.rows(), reduced_directions.cols());
    coupling_output->scaled_structure_forcing.resize(
        static_cast<std::size_t>(reduced_directions.cols()));
  }
  if (reduced_directions.cols() <= 1) {
    for (Eigen::Index column = 0;
         column < reduced_directions.cols();
         ++column) {
      if (coupling_output == nullptr) {
        responses.col(column) =
            apply_reduced(reduced_directions.col(column), components);
      } else {
        OrbitalCouplingAction action =
            apply_orbital_coupling(reduced_directions.col(column));
        responses.col(column) = action.orbital_hessian;
        coupling_output->orbital_hessian.col(column) =
            action.orbital_hessian;
        coupling_output->scaled_structure_forcing[
            static_cast<std::size_t>(column)] =
                std::move(action.scaled_structure_forcing);
      }
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
    PrecomputedDirection precomputed = prepare_direction(
        nonredundant_space_->expand_step(reduced_directions.col(column)));
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
  if (accepted_ri_factorization_ != nullptr) {
    delta_h1e_columns.resize(ao_matrix_size, n_directions);
    inactive_density_gradient_columns.resize(
        ao_matrix_size, n_directions);
    for (Eigen::Index column = 0; column < n_directions; ++column) {
      const Eigen::Map<const Eigen::MatrixXd> source(
          inactive_density_columns.col(column).data(),
          n_basis_functions, n_basis_functions);
      const Eigen::Map<const Eigen::MatrixXd> adjoint(
          symmetrized_pullback_columns.col(column).data(),
          n_basis_functions, n_basis_functions);
      apply_ao_effective_one_electron_ri_operator_adaptive(
          source, adjoint, *accepted_ri_factorization_,
          &ri_ao_h1e_fused_workspace_, &ri_ao_h1e_strategy_,
          &ri_ao_h1e_forward_workspace_, &ri_ao_h1e_adjoint_workspace_);
      delta_h1e_columns.col(column) = Eigen::Map<const Eigen::VectorXd>(
          ri_ao_h1e_forward_workspace_.data(), ao_matrix_size);
      std::vector<double> encoded_adjoint;
      detail::encode_symmetric_ao_gradient(
          ri_ao_h1e_adjoint_workspace_, &encoded_adjoint);
      inactive_density_gradient_columns.col(column) =
          Eigen::Map<const Eigen::VectorXd>(
              encoded_adjoint.data(), ao_matrix_size);
    }
  } else {
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
      detail::symmetrize_exact_ao_h1e_forward(delta_h1e);
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
  std::vector<Eigen::MatrixXd> ri_active_pair_factor_directions;
  const bool build_orbital_coupling = coupling_output != nullptr;
  const bool compute_outer_response =
      components.local_active_response || components.structure_response ||
      build_orbital_coupling;
  if (accepted_ri_two_electron_cache_.has_value() &&
      (compute_outer_response || components.direct_core_response)) {
    const auto batch_active_two_electron_start_time =
        std::chrono::steady_clock::now();
    ri_active_pair_factor_directions =
        compute_ri_active_pair_factor_directional_derivative_batch(
            *accepted_ri_two_electron_cache_, dense_active_directions);
    const double batch_active_two_electron_seconds =
        detail::exact_hvp_elapsed_seconds(batch_active_two_electron_start_time);
    apply_timing_totals_.active_two_electron_wall_time_seconds +=
        batch_active_two_electron_seconds;
    apply_timing_totals_.total_apply_wall_time_seconds +=
        batch_active_two_electron_seconds;
  } else if (compute_outer_response) {
    const auto batch_active_two_electron_start_time =
        std::chrono::steady_clock::now();
    delta_packed_active_two_electron_columns =
        compute_exact_packed_active_two_electron_integral_directional_derivative_batch(
            accepted_exact_two_electron_cache_,
            dense_active_directions,
            current_input_->ao_integral_input,
            components.direct_core_response
                ? &directional_pair_products
                : nullptr,
            components.direct_core_response
                ? &two_electron_fixed_adjoint_directions
                : nullptr);
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

  if (compute_outer_response) {
    const auto outer_batch_start = std::chrono::steady_clock::now();
    const int n_selected_states = static_cast<int>(
        accepted_point_context_->selected_state_indices.size());
    const int n_structures = accepted_point_context_->n_structures;
    Eigen::MatrixXd delta_hamiltonian_selected;
    Eigen::MatrixXd delta_overlap_selected;
    if (components.structure_response) {
      delta_hamiltonian_selected.resize(
          n_structures, n_directions * n_selected_states);
      delta_overlap_selected.resize(
          n_structures, n_directions * n_selected_states);
    }
    double integral_seconds = 0.0;
    double structure_seconds = 0.0;
    const bool direct_active_gradient =
        outer_response_context()
            .selected_state_eigen_response_operator.structure_action
            ->supports_integral_direction();
    const bool pair_cache_required =
        !direct_active_gradient &&
        (components.local_active_response || components.structure_response ||
         build_orbital_coupling);
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
      Eigen::VectorXd delta_packed_two_electron;
      if (accepted_ri_two_electron_cache_.has_value()) {
        const std::vector<double> packed =
            compute_ri_packed_active_two_electron_integral_directional_derivative(
                *accepted_ri_two_electron_cache_,
                ri_active_pair_factor_directions[
                    static_cast<std::size_t>(column)]);
        delta_packed_two_electron = Eigen::Map<const Eigen::VectorXd>(
            packed.data(), static_cast<Eigen::Index>(packed.size()));
      } else {
        delta_packed_two_electron =
            delta_packed_active_two_electron_columns.col(column);
      }
      const ActiveSpaceIntegralDirectionView integral_direction =
          build_active_integral_direction(
              precomputed_directions[column]
                  .orbital_preparation_directional_result,
              delta_h1e_times_active,
              &delta_packed_two_electron,
              nullptr,
              &outer.integral_direction);
      integral_seconds += detail::exact_hvp_elapsed_seconds(integral_start);

      const auto structure_start = std::chrono::steady_clock::now();
      if (pair_cache_required) {
        outer.pair_cache = build_same_spin_directional_pair_cache(
            accepted_point_context_->same_spin_pair_cache,
            n_active_orbitals,
            integral_direction,
            &accepted_point_context_->prepared_active_space
                 .active_space_one_electron_result.h1e_act,
            accepted_ri_two_electron_cache_.has_value()
                ? accepted_ri_two_electron_cache_->accepted_active_pair_factors
                : nullptr,
            accepted_ri_two_electron_cache_.has_value()
                ? &ri_active_pair_factor_directions[
                      static_cast<std::size_t>(column)]
                : nullptr);
      }
      if (components.structure_response || build_orbital_coupling) {
        SelectedStateDirectionalStructureImages images =
            build_selected_structure_direction(
                outer_response_context(),
                integral_direction,
                outer.pair_cache);
        outer.direct_ci_direction = std::move(images.direct_ci_direction);
        if (build_orbital_coupling) {
          const auto& eigen = outer_response_context()
              .selected_state_eigen_response_operator;
          const xmvb::core::EqualWeightEigenCoupling coupling =
              xmvb::core::prepare_equal_weight_generalized_eigen_coupling(
                  eigen.selected_eigenvalues,
                  eigen.selected_eigenvectors,
                  eigen.overlap_selected,
                  eigen.selected_residuals,
                  images.delta_hamiltonian_selected,
                  images.delta_overlap_selected);
          outer.gauge_coupling = scale_equal_weight_structure_coupling(
              coupling, accepted_point_context_->normalized_state_weights);
          coupling_output->scaled_structure_forcing[
              static_cast<std::size_t>(column)] =
                  outer.gauge_coupling->horizontal_forcing.scaled_coefficients;
        } else {
          const int first = static_cast<int>(column) * n_selected_states;
          delta_hamiltonian_selected.middleCols(first, n_selected_states) =
              images.delta_hamiltonian_selected;
          delta_overlap_selected.middleCols(first, n_selected_states) =
              images.delta_overlap_selected;
        }
      }
      outer.integral_direction.packed_two_electron.clear();
      structure_seconds += detail::exact_hvp_elapsed_seconds(structure_start);
    }
    apply_timing_totals_
        .outer_response_active_space_integrals_wall_time_seconds +=
        integral_seconds;
    apply_timing_totals_
        .outer_response_structure_matrices_wall_time_seconds +=
        structure_seconds;

    if (components.structure_response) {
      const auto eigensystem_start = std::chrono::steady_clock::now();
      const SelectedStateGeneralizedEigenDirectionalResponse block_response =
          outer_response_context()
              .selected_state_eigen_response_operator.apply_direction_block(
                  delta_hamiltonian_selected,
                  delta_overlap_selected,
                  components.response_relative_residual_tolerance,
                  components.freeze_structure_response);
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
        response.selected_matrix_responses = {
            block_response.selected_matrix_responses[column]};
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
    Eigen::VectorXd delta_packed_active_two_electron;
    if (compute_outer_response) {
      if (accepted_ri_two_electron_cache_.has_value()) {
        const std::vector<double> packed =
            compute_ri_packed_active_two_electron_integral_directional_derivative(
                *accepted_ri_two_electron_cache_,
                ri_active_pair_factor_directions[
                    static_cast<std::size_t>(column)]);
        delta_packed_active_two_electron = Eigen::Map<const Eigen::VectorXd>(
            packed.data(), static_cast<Eigen::Index>(packed.size()));
      } else {
        delta_packed_active_two_electron =
            delta_packed_active_two_electron_columns.col(column);
      }
    }
    if (compute_outer_response) {
      auto& packed = precomputed_directions[column]
                         .outer_response
                         ->integral_direction
                         .packed_two_electron;
      packed.assign(
          delta_packed_active_two_electron.data(),
          delta_packed_active_two_electron.data() +
              delta_packed_active_two_electron.size());
    }
    OrbitalCouplingAction column_coupling;
    const Eigen::VectorXd response = apply_reduced_impl(
        reduced_directions.col(column),
        components,
        &delta_h1e,
        &inactive_density_gradient,
        (components.local_active_response || components.structure_response)
            ? &delta_packed_active_two_electron
            : nullptr,
        accepted_ri_two_electron_cache_.has_value() &&
                (compute_outer_response || components.direct_core_response)
            ? &ri_active_pair_factor_directions[
                  static_cast<std::size_t>(column)]
            : nullptr,
        components.direct_core_response &&
                compute_outer_response &&
                !accepted_ri_two_electron_cache_.has_value()
            ? &directional_pair_products[column]
            : nullptr,
        components.direct_core_response &&
                compute_outer_response &&
                !accepted_ri_two_electron_cache_.has_value() &&
                two_electron_fixed_adjoint_directions[column].size() != 0
            ? &two_electron_fixed_adjoint_directions[column]
            : nullptr,
        &precomputed_directions[column],
        build_orbital_coupling ? &column_coupling : nullptr);
    responses.col(column) = response;
    if (build_orbital_coupling) {
      coupling_output->orbital_hessian.col(column) = response;
      coupling_output->scaled_structure_forcing[
          static_cast<std::size_t>(column)] =
              std::move(column_coupling.scaled_structure_forcing);
    }
    if (compute_outer_response) {
      precomputed_directions[column]
          .outer_response
          ->integral_direction
          .packed_two_electron.clear();
    }
  }
  return responses;
}

}  // namespace xmvb::vb
