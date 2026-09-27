#include "vbscf/derivatives/hessian/exact/operator.hpp"

#include "vbscf/derivatives/hessian/exact/apply_internal.hpp"

#include "vbscf/derivatives/hessian/exact/ao_one_electron_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

#include "core/eigen_response.hpp"
#include "vbscf/derivatives/hessian/coupled/coupling.hpp"
#include "vbscf/integrals/active/two_electron/response/adjoint.hpp"
#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"
#include "vbscf/derivatives/hessian/responses/active_space/integral_direction.hpp"
#include "vbscf/derivatives/hessian/responses/active_space/local_tile_accumulator_internal.hpp"
#include "vbscf/derivatives/hessian/responses/active_space/outer_response.hpp"
#include "vbscf/derivatives/hessian/responses/orbital/preparation.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/backward.hpp"
#include "vbscf/derivatives/hessian/responses/structure/directional.hpp"
#include "vbscf/structures/assembly/selected_coefficients.hpp"

namespace xmvb::vb {

namespace {

void add_orbital_value_gradient_in_place(
    std::vector<double>* target,
    const std::vector<double>& contribution,
    const char* /*label*/) {
  if (target->empty()) {
    *target = contribution;
    return;
  }
  for (std::size_t index = 0; index < target->size(); ++index) {
    (*target)[index] += contribution[index];
  }
}

void throw_if_nonfinite(
    const std::vector<double>& values,
    const char* label) {
  const auto nonfinite_it =
      std::find_if(
          values.begin(),
          values.end(),
          [](double value) { return !std::isfinite(value); });
  if (nonfinite_it == values.end()) {
    return;
  }
  throw std::runtime_error(
      std::string(label) + " contains non-finite values");
}

void throw_if_nonfinite(
    const Eigen::MatrixXd& values,
    const char* label) {
  if (values.allFinite()) {
    return;
  }
  throw std::runtime_error(
      std::string(label) + " contains non-finite values");
}

void throw_if_nonfinite(
    const Eigen::VectorXd& values,
    const char* label) {
  if (values.allFinite()) {
    return;
  }
  throw std::runtime_error(
      std::string(label) + " contains non-finite values");
}

ActiveSpaceGradientDirection make_active_gradient_direction(
    const StructureActiveIntegralAdjoint& adjoint,
    int n_active_orbitals) {
  const int n_pairs = packed_active_pair_count(n_active_orbitals);
  if (adjoint.overlap.rows() != n_active_orbitals ||
      adjoint.overlap.cols() != n_active_orbitals ||
      adjoint.one_electron.rows() != n_active_orbitals ||
      adjoint.one_electron.cols() != n_active_orbitals ||
      adjoint.pair_kernel.rows() != n_pairs ||
      adjoint.pair_kernel.cols() != n_pairs) {
    throw std::runtime_error(
        "direct-CI active-gradient direction has inconsistent dimensions");
  }
  ActiveSpaceGradientDirection result =
      make_zero_active_space_gradient_direction(n_active_orbitals);
  result.active_orbital_overlap_gradient.assign(
      adjoint.overlap.data(),
      adjoint.overlap.data() + adjoint.overlap.size());
  result.active_one_electron_gradient.assign(
      adjoint.one_electron.data(),
      adjoint.one_electron.data() + adjoint.one_electron.size());
  for (int column = 0; column < n_pairs; ++column) {
    for (int row = 0; row <= column; ++row) {
      result.packed_active_two_electron_gradient[
          TwoElectronIndexer::packed_pair_of_pairs_index(row, column)] =
          row == column
          ? adjoint.pair_kernel(row, column)
          : adjoint.pair_kernel(row, column) +
                adjoint.pair_kernel(column, row);
    }
  }
  return result;
}

}  // namespace

Eigen::VectorXd ExactHvpOperator::apply_reduced(
    const Eigen::VectorXd& reduced_direction,
    HvpComponents components) const {
  return state_->apply_reduced(reduced_direction, components);
}

Eigen::VectorXd ExactHvpOperator::State::apply_reduced(
    const Eigen::VectorXd& reduced_direction,
    HvpComponents components) const {
  return apply_reduced_impl(
      reduced_direction,
      components,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr);
}

OrbitalCouplingAction ExactHvpOperator::apply_orbital_coupling(
    const Eigen::VectorXd& reduced_direction) const {
  return state_->apply_orbital_coupling(reduced_direction);
}

OrbitalCouplingAction ExactHvpOperator::State::apply_orbital_coupling(
    const Eigen::VectorXd& reduced_direction) const {
  OrbitalCouplingAction result;
  result.orbital_hessian = apply_reduced_impl(
      reduced_direction,
      {.direct_core_response = true,
       .fixed_upstream_pullback = true,
       .local_active_response = true,
       .structure_response = false},
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      &result);
  return result;
}

StructureResponseBlock ExactHvpOperator::solve_structure_response_block(
    const std::vector<Eigen::MatrixXd>& scaled_structure_forcing,
    double relative_residual_tolerance) const {
  return state_->solve_structure_response_block(
      scaled_structure_forcing, relative_residual_tolerance);
}

StructureResponseBlock ExactHvpOperator::State::solve_structure_response_block(
    const std::vector<Eigen::MatrixXd>& scaled_structure_forcing,
    double relative_residual_tolerance) const {
  if (scaled_structure_forcing.empty() ||
      !(relative_residual_tolerance > 0.0) ||
      relative_residual_tolerance >= 1.0 ||
      !std::isfinite(relative_residual_tolerance)) {
    throw std::invalid_argument("invalid structure-response forcing block");
  }
  const int n_structures = accepted_point_context_->n_structures;
  const int n_states = static_cast<int>(
      accepted_point_context_->selected_state_indices.size());
  const int n_directions =
      static_cast<int>(scaled_structure_forcing.size());
  Eigen::MatrixXd raw_forcing(n_structures, n_states * n_directions);
  for (int direction = 0; direction < n_directions; ++direction) {
    const Eigen::MatrixXd& forcing =
        scaled_structure_forcing[static_cast<std::size_t>(direction)];
    if (forcing.rows() != n_structures || forcing.cols() != n_states ||
        !forcing.allFinite()) {
      throw std::invalid_argument(
          "structure-response forcing has inconsistent dimensions");
    }
    for (int state = 0; state < n_states; ++state) {
      const double weight = accepted_point_context_->normalized_state_weights[
          static_cast<std::size_t>(state)];
      if (!(weight > 0.0) || !std::isfinite(weight)) {
        throw std::runtime_error(
            "structure-response coordinate weight is invalid");
      }
      raw_forcing.col(direction * n_states + state) =
          forcing.col(state) / std::sqrt(2.0 * weight);
    }
  }

  const auto& response_operator =
      outer_response_context().selected_state_eigen_response_operator;
  const SelectedStateGeneralizedEigenDirectionalResponse response =
      response_operator.apply_direction_block(
          raw_forcing,
          Eigen::MatrixXd::Zero(n_structures, n_states * n_directions),
          relative_residual_tolerance,
          false);
  apply_timing_totals_.structure_response_block_actions +=
      response.block_actions;
  if (!response.linear_iterations.empty()) {
    apply_timing_totals_.max_structure_response_iterations = std::max(
        apply_timing_totals_.max_structure_response_iterations,
        *std::max_element(
            response.linear_iterations.begin(),
            response.linear_iterations.end()));
  }
  apply_timing_totals_.max_structure_response_relative_residual = std::max(
      apply_timing_totals_.max_structure_response_relative_residual,
      response.max_relative_residual);

  StructureResponseBlock result;
  if (response.equation_residuals.size() !=
      scaled_structure_forcing.size()) {
    throw std::runtime_error(
        "structure response lacks bordered residual certificates: got " +
        std::to_string(response.equation_residuals.size()) + ", expected " +
        std::to_string(scaled_structure_forcing.size()));
  }
  result.scaled_coefficients.reserve(scaled_structure_forcing.size());
  result.coefficient_responses.reserve(scaled_structure_forcing.size());
  result.adjoint_multipliers.reserve(scaled_structure_forcing.size());
  result.scaled_equation_residuals.reserve(
      scaled_structure_forcing.size());
  const Eigen::MatrixXd selected_metric =
      response_operator.selected_eigenvectors.transpose() *
      response_operator.overlap_selected;
  for (int direction = 0; direction < n_directions; ++direction) {
    Eigen::MatrixXd raw =
        response.delta_selected_eigenvector_matrix.middleCols(
            direction * n_states, n_states);
    Eigen::MatrixXd scaled = raw;
    Eigen::MatrixXd scaled_residual = response.equation_residuals[
        static_cast<std::size_t>(direction)];
    for (int state = 0; state < n_states; ++state) {
      const double weight = accepted_point_context_->normalized_state_weights[
          static_cast<std::size_t>(state)];
      scaled.col(state) *= std::sqrt(2.0 * weight);
      scaled_residual.col(state) *= std::sqrt(2.0 * weight);
    }
    result.scaled_coefficients.push_back(std::move(scaled));
    result.coefficient_responses.push_back(std::move(raw));
    result.adjoint_multipliers.push_back(
        -response.selected_matrix_responses[
            static_cast<std::size_t>(direction)] +
        selected_metric.partialPivLu().solve(
            response_operator.selected_eigenvectors.transpose() *
            raw_forcing.middleCols(direction * n_states, n_states)));
    result.scaled_equation_residuals.push_back(
        std::move(scaled_residual));
  }
  result.revision = response_operator.revision();
  result.block_actions = response.block_actions;
  result.max_relative_residual = response.max_relative_residual;
  return result;
}

Eigen::VectorXd ExactHvpOperator::apply_structure_coupling_adjoint(
    const Eigen::Ref<const Eigen::MatrixXd>& coefficient_response,
    const Eigen::Ref<const Eigen::MatrixXd>& adjoint_multipliers) const {
  return state_->apply_structure_coupling_adjoint(
      coefficient_response, adjoint_multipliers);
}

Eigen::VectorXd ExactHvpOperator::State::apply_structure_coupling_adjoint(
    const Eigen::Ref<const Eigen::MatrixXd>& coefficient_response,
    const Eigen::Ref<const Eigen::MatrixXd>& adjoint_multipliers) const {
  return apply_structure_response_adjoint(
      coefficient_response, adjoint_multipliers);
}

Eigen::VectorXd ExactHvpOperator::State::apply_reduced_impl(
    const Eigen::VectorXd& reduced_direction,
    HvpComponents components,
    const Eigen::VectorXd* precomputed_delta_ao_effective_h1e,
    const Eigen::VectorXd* precomputed_inactive_density_gradient,
    const Eigen::VectorXd* precomputed_delta_packed_active_two_electron,
    const Eigen::MatrixXd* precomputed_ri_active_pair_factor_direction,
    const ExactCtxPairMatrix* precomputed_directional_pair_products,
    const Eigen::MatrixXd* precomputed_two_electron_fixed_adjoint,
    const PrecomputedDirection* precomputed_direction,
    OrbitalCouplingAction* coupling_output) const {
  const auto apply_start_time = std::chrono::steady_clock::now();
  auto record_apply_wall_time = [&]() {
    ++apply_timing_totals_.apply_count;
    apply_timing_totals_.total_apply_wall_time_seconds +=
        detail::exact_hvp_elapsed_seconds(apply_start_time);
  };
  Eigen::VectorXd response =
      Eigen::VectorXd::Zero(reduced_direction.size());

  if (!supports_analytic_core_model()) {
    throw std::runtime_error(
        "exact_ctx analytic core HVP is unavailable for the current accepted point");
  }

  const int n_basis_functions =
      current_input_->orbital_preparation_input.n_basis_functions;
  const int n_active_orbitals =
      current_input_->orbital_preparation_input.n_active_orbitals;
  const std::size_t ao_matrix_size =
      n_basis_functions * n_basis_functions;
  if (coupling_output != nullptr && components.structure_response) {
    throw std::invalid_argument(
        "coupled orbital action cannot solve the structure response");
  }
  if (coupling_output != nullptr) {
    coupling_output->scaled_structure_forcing = Eigen::MatrixXd::Zero(
        accepted_point_context_->n_structures,
        static_cast<int>(accepted_point_context_->selected_state_indices.size()));
  }

  const auto core_setup_start_time = std::chrono::steady_clock::now();
  const Eigen::VectorXd local_packed_direction =
      precomputed_direction == nullptr
          ? nonredundant_space_->expand_step(reduced_direction)
          : Eigen::VectorXd();
  const Eigen::VectorXd& packed_direction =
      precomputed_direction != nullptr
          ? precomputed_direction->packed_direction
          : local_packed_direction;
  if (packed_direction.norm() == 0.0) {
    record_apply_wall_time();
    return Eigen::VectorXd::Zero(reduced_direction.size());
  }

  const AcceptedOrbitalPreparationCache* orbital_preparation_cache =
      accepted_orbital_preparation_cache_.get();
  std::optional<PrecomputedDirection> local_precomputed_direction;
  if (precomputed_direction == nullptr) {
    local_precomputed_direction.emplace(
        prepare_direction(local_packed_direction));
  }
  const PrecomputedDirection& prepared_direction =
      precomputed_direction != nullptr
          ? *precomputed_direction
          : *local_precomputed_direction;
  const DenseOrbitalTangentContext& dense_orbital_tangent_context =
      prepared_direction.dense_orbital_tangent_context;
  const Eigen::VectorXd input_retract_tangent =
      components.fixed_upstream_pullback
          ? nonredundant_space_->expand_retract_input_tangent(
                current_input_->orbital_preparation_input,
                reduced_direction)
          : Eigen::VectorXd();
  const OrbitalPreparationDirectionalResult&
      orbital_preparation_directional_result =
          prepared_direction.orbital_preparation_directional_result;

  const Eigen::Map<const Eigen::MatrixXd> basis_overlap(
      current_input_->orbital_preparation_input.ao_overlap_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  const Eigen::Map<const Eigen::MatrixXd> ao_effective_h1e(
      accepted_point_context_->prepared_active_space.ao_effective_one_electron_result
          .ao_effective_h1e.data(),
      n_basis_functions,
      n_basis_functions);
  const auto& delta_dense_active_coefficients =
      orbital_preparation_directional_result.delta_active_auxiliary_orbitals;
  apply_timing_totals_.core_setup_wall_time_seconds +=
      detail::exact_hvp_elapsed_seconds(core_setup_start_time);
  // `F11` and `delta F11` are explicitly symmetrized in the AO-H1E builder, so
  // the directional active-space matrix gradient only needs the two distinct
  // left contractions `delta F11 * T_active` and `F11 * delta T_active`.
  const Eigen::MatrixXd delta_active_times_hho_symmetric =
      orbital_preparation_directional_result.delta_active_auxiliary_orbitals *
      accepted_hho_gradient_symmetric_;
  Eigen::MatrixXd total_ao_effective_one_electron_direction =
      orbital_preparation_directional_result.delta_inactive_density;
  total_ao_effective_one_electron_direction.noalias() +=
      delta_active_times_hho_symmetric *
      accepted_active_auxiliary_orbitals_.transpose();
  const double* delta_ao_effective_h1e_data = nullptr;
  const double* inactive_density_gradient_data = nullptr;
  if (precomputed_delta_ao_effective_h1e != nullptr ||
      precomputed_inactive_density_gradient != nullptr) {
    if (precomputed_delta_ao_effective_h1e == nullptr ||
        precomputed_inactive_density_gradient == nullptr ||
        precomputed_delta_ao_effective_h1e->size() !=
            static_cast<Eigen::Index>(ao_matrix_size) ||
        precomputed_inactive_density_gradient->size() !=
            static_cast<Eigen::Index>(ao_matrix_size)) {
      throw std::invalid_argument(
          "precomputed block AO-H1E response has inconsistent dimensions");
    }
    delta_ao_effective_h1e_data =
        precomputed_delta_ao_effective_h1e->data();
    inactive_density_gradient_data =
        precomputed_inactive_density_gradient->data();
  } else {
    const auto ao_effective_one_electron_fused_start_time =
        std::chrono::steady_clock::now();
    if (accepted_ri_factorization_ != nullptr) {
      ao_h1e_symmetrized_gradient_workspace_ =
          total_ao_effective_one_electron_direction +
          total_ao_effective_one_electron_direction.transpose();
      const auto& delta_inactive_density =
          orbital_preparation_directional_result.delta_inactive_density;
      apply_ao_effective_one_electron_ri_operator_adaptive(
          delta_inactive_density,
          ao_h1e_symmetrized_gradient_workspace_,
          *accepted_ri_factorization_,
          &ri_ao_h1e_fused_workspace_,
          &ri_ao_h1e_strategy_,
          &ri_ao_h1e_forward_workspace_,
          &ri_ao_h1e_adjoint_workspace_);
      ao_h1e_delta_h1e_workspace_.assign(
          ri_ao_h1e_forward_workspace_.data(),
          ri_ao_h1e_forward_workspace_.data() +
              ri_ao_h1e_forward_workspace_.size());
      detail::encode_symmetric_ao_gradient(
          ri_ao_h1e_adjoint_workspace_,
          &ao_h1e_inactive_density_gradient_workspace_);
    } else {
      detail::apply_fused_exact_ao_one_electron_response(
          orbital_preparation_directional_result.delta_inactive_density,
          total_ao_effective_one_electron_direction,
          current_input_->ao_integral_input,
          current_input_->orbital_preparation_input,
          &ao_h1e_fused_workspace_,
          &ao_h1e_symmetrized_gradient_workspace_,
          &ao_h1e_delta_h1e_workspace_,
          &ao_h1e_inactive_density_gradient_workspace_);
    }
    delta_ao_effective_h1e_data = ao_h1e_delta_h1e_workspace_.data();
    inactive_density_gradient_data =
        ao_h1e_inactive_density_gradient_workspace_.data();
    apply_timing_totals_.ao_effective_one_electron_fused_wall_time_seconds +=
        detail::exact_hvp_elapsed_seconds(ao_effective_one_electron_fused_start_time);
  }
  const Eigen::Map<const Eigen::MatrixXd> delta_ao_effective_h1e(
      delta_ao_effective_h1e_data,
      n_basis_functions,
      n_basis_functions);

  const bool build_orbital_coupling = coupling_output != nullptr;
  const bool compute_outer_response = components.local_active_response ||
      components.structure_response || build_orbital_coupling;
  const bool direct_active_gradient = compute_outer_response &&
      outer_response_context()
          .selected_state_eigen_response_operator.structure_action
          ->supports_integral_direction();
  const bool packed_outer_two_electron_required =
      !accepted_ri_two_electron_cache_.has_value() ||
      direct_active_gradient ||
      has_polynomial_same_spin_response_pairs(
          accepted_point_context_->same_spin_pair_cache);
  std::optional<Eigen::MatrixXd> ri_directional_active_pair_factors;
  Eigen::VectorXd ri_delta_packed_active_two_electron;
  if (accepted_ri_two_electron_cache_.has_value() &&
      (compute_outer_response || components.direct_core_response)) {
    if (precomputed_ri_active_pair_factor_direction != nullptr) {
      if (precomputed_ri_active_pair_factor_direction->rows() !=
              accepted_ri_two_electron_cache_->n_auxiliary_functions ||
          precomputed_ri_active_pair_factor_direction->cols() !=
              static_cast<Eigen::Index>(
                  accepted_ri_two_electron_cache_
                      ->active_pair_first_indices.size())) {
        throw std::invalid_argument(
            "precomputed RI active-pair factor direction has inconsistent dimensions");
      }
      ri_directional_active_pair_factors =
          *precomputed_ri_active_pair_factor_direction;
    } else {
      ri_directional_active_pair_factors.emplace(
          compute_ri_active_pair_factor_directional_derivative(
              *accepted_ri_two_electron_cache_,
              delta_dense_active_coefficients));
    }
    if (compute_outer_response && packed_outer_two_electron_required) {
      const std::vector<double> packed_direction =
          compute_ri_packed_active_two_electron_integral_directional_derivative(
              *accepted_ri_two_electron_cache_,
              *ri_directional_active_pair_factors);
      ri_delta_packed_active_two_electron = Eigen::Map<const Eigen::VectorXd>(
          packed_direction.data(),
          static_cast<Eigen::Index>(packed_direction.size()));
    }
  }
  std::vector<double> combined_core_orbital_value_gradient;
  Eigen::MatrixXd delta_ao_effective_h1e_times_active_auxiliary_orbitals;
  if (compute_outer_response) {
    // Full HVPs need `delta F11 * A_active` both for the direct-core
    // `delta F11 * A_active * G_hho` pullback and for outer-response
    // `delta HHO = A_active^T * delta F11 * A_active`.  Materializing it once
    // avoids repeating the same AO-by-active contraction in the two stages.
    delta_ao_effective_h1e_times_active_auxiliary_orbitals.noalias() =
        delta_ao_effective_h1e * accepted_active_auxiliary_orbitals_;
  }

  if (compute_outer_response) {
    const auto outer_response_start_time =
        std::chrono::steady_clock::now();
    const PrecomputedOuterResponse* precomputed_outer_response =
        precomputed_direction != nullptr &&
            precomputed_direction->outer_response.has_value()
        ? &precomputed_direction->outer_response.value()
        : nullptr;

    const auto active_space_integrals_start_time =
        std::chrono::steady_clock::now();
    const ActiveSpaceIntegralDirectionView active_space_integral_direction =
        precomputed_outer_response != nullptr
        ? ActiveSpaceIntegralDirectionView{
              precomputed_outer_response->integral_direction.overlap,
              precomputed_outer_response->integral_direction.one_electron,
              precomputed_outer_response->integral_direction
                  .packed_two_electron}
        : build_active_integral_direction(
              orbital_preparation_directional_result,
              delta_ao_effective_h1e_times_active_auxiliary_orbitals,
              packed_outer_two_electron_required
                  ? (precomputed_delta_packed_active_two_electron != nullptr
                         ? precomputed_delta_packed_active_two_electron
                         : (accepted_ri_two_electron_cache_.has_value()
                                ? &ri_delta_packed_active_two_electron
                                : nullptr))
                  : nullptr,
              &ri_delta_packed_active_two_electron,
              &outer_response_integral_direction_workspace_,
              packed_outer_two_electron_required);
    if (precomputed_outer_response == nullptr) {
      apply_timing_totals_
          .outer_response_active_space_integrals_wall_time_seconds +=
          detail::exact_hvp_elapsed_seconds(active_space_integrals_start_time);
    }

    const auto structure_matrices_start_time =
        std::chrono::steady_clock::now();
    const StructureAction* accepted_structure_action =
        outer_response_context()
            .selected_state_eigen_response_operator.structure_action;
    const bool use_directional_pair_tiles =
        precomputed_outer_response == nullptr && !direct_active_gradient &&
        (components.local_active_response || components.structure_response ||
         build_orbital_coupling);
    std::unique_ptr<detail::LocalActiveSpaceTileAccumulator>
        local_active_tile_accumulator;
    if (use_directional_pair_tiles && components.local_active_response) {
      local_active_tile_accumulator =
          std::make_unique<detail::LocalActiveSpaceTileAccumulator>(
              *current_input_,
              *accepted_point_context_,
              active_space_integral_direction);
    }
    const auto consume_local_active_tile =
        [&](bool alpha_channel,
            bool beta_channel,
            const AcceptedSpinPairTile& accepted,
            const detail::SameSpinDirectionalPairTileView& same_spin,
            const detail::DirectionalOppositeSpinPairTileView* opposite_spin) {
          if (local_active_tile_accumulator) {
            local_active_tile_accumulator->consume(
                alpha_channel,
                beta_channel,
                accepted,
                same_spin,
                opposite_spin);
          }
        };
    bool directional_pair_tiles_consumed = false;
    const SelectedStateGeneralizedEigenDirectionalResponse*
        directional_selected_state_response = nullptr;
    SelectedStateGeneralizedEigenDirectionalResponse
        local_directional_selected_state_response;
    std::optional<StructureIntegralDirection> local_direct_ci_direction;
    std::optional<ScaledStructureCoupling> local_gauge_coupling;
    const ScaledStructureCoupling* gauge_coupling = nullptr;
    if (components.structure_response || build_orbital_coupling) {
      if (precomputed_outer_response != nullptr) {
        if (build_orbital_coupling) {
          if (!precomputed_outer_response->gauge_coupling.has_value()) {
            throw std::logic_error(
                "precomputed coupled orbital action is missing its gauge coupling");
          }
          gauge_coupling = &*precomputed_outer_response->gauge_coupling;
          coupling_output->scaled_structure_forcing =
              gauge_coupling->horizontal_forcing.scaled_coefficients;
        } else {
          directional_selected_state_response =
              &precomputed_outer_response->selected_state_response;
        }
      } else {
        SelectedStateDirectionalStructureImages images =
            use_directional_pair_tiles
            ? build_selected_structure_direction_from_pair_tiles(
                  outer_response_context(),
                  active_space_integral_direction,
                  accepted_ri_two_electron_cache_.has_value()
                      ? accepted_ri_two_electron_cache_
                            ->accepted_active_pair_factors
                      : nullptr,
                  ri_directional_active_pair_factors.has_value()
                      ? &*ri_directional_active_pair_factors
                      : nullptr,
                  consume_local_active_tile)
            : build_selected_structure_direction(
                  outer_response_context(),
                  active_space_integral_direction,
                  accepted_ri_two_electron_cache_.has_value()
                      ? accepted_ri_two_electron_cache_
                            ->accepted_active_pair_factors
                      : nullptr,
                  ri_directional_active_pair_factors.has_value()
                      ? &*ri_directional_active_pair_factors
                      : nullptr);
        directional_pair_tiles_consumed = use_directional_pair_tiles;
        local_direct_ci_direction = std::move(images.direct_ci_direction);
        apply_timing_totals_
            .outer_response_structure_matrices_wall_time_seconds +=
            detail::exact_hvp_elapsed_seconds(structure_matrices_start_time);

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
          local_gauge_coupling = scale_equal_weight_structure_coupling(
              coupling, accepted_point_context_->normalized_state_weights);
          gauge_coupling = &*local_gauge_coupling;
          coupling_output->scaled_structure_forcing =
              gauge_coupling->horizontal_forcing.scaled_coefficients;
        } else {
          const auto eigensystem_start_time =
              std::chrono::steady_clock::now();
          local_directional_selected_state_response =
              outer_response_context()
                  .selected_state_eigen_response_operator.apply(
                      images,
                      components.response_relative_residual_tolerance,
                      components.freeze_structure_response);
          apply_timing_totals_.outer_response_eigensystem_wall_time_seconds +=
              detail::exact_hvp_elapsed_seconds(eigensystem_start_time);
          directional_selected_state_response =
              &local_directional_selected_state_response;
        }
      }
      if (directional_selected_state_response != nullptr) {
        apply_timing_totals_.structure_response_block_actions +=
            directional_selected_state_response->block_actions;
        if (!directional_selected_state_response->linear_iterations.empty()) {
          apply_timing_totals_.max_structure_response_iterations = std::max(
              apply_timing_totals_.max_structure_response_iterations,
              *std::max_element(
                  directional_selected_state_response->linear_iterations.begin(),
                  directional_selected_state_response->linear_iterations.end()));
        }
        apply_timing_totals_.max_structure_response_relative_residual = std::max(
            apply_timing_totals_.max_structure_response_relative_residual,
            directional_selected_state_response->max_relative_residual);
        if (directional_selected_state_response
                ->selected_matrix_responses.size() != 1) {
          throw std::logic_error(
              "one HVP direction must produce one selected-space response matrix");
        }
      }
    }

    Eigen::MatrixXd state_multipliers;
    if (directional_selected_state_response != nullptr) {
      state_multipliers =
          -directional_selected_state_response
               ->selected_matrix_responses.front();
    } else if (gauge_coupling != nullptr) {
      state_multipliers = gauge_coupling->gauge_adjoint_multipliers;
    }

    std::optional<SelectedStateDeterminantMatrices>
        directional_selected_states;
    if (directional_selected_state_response != nullptr) {
      const auto selected_state_rebuild_start_time =
          std::chrono::steady_clock::now();
      directional_selected_states.emplace(
          build_selected_state_determinant_matrices_from_selected_columns(
              current_input_->structure_data,
              directional_selected_state_response
                  ->delta_selected_eigenvector_matrix,
              accepted_point_context_->selected_state_indices,
              accepted_point_context_->normalized_state_weights,
              accepted_point_context_->same_spin_pair_cache));
      apply_timing_totals_
          .outer_response_selected_state_rebuild_wall_time_seconds +=
          detail::exact_hvp_elapsed_seconds(selected_state_rebuild_start_time);
    } else if (gauge_coupling != nullptr) {
      const auto selected_state_rebuild_start_time =
          std::chrono::steady_clock::now();
      directional_selected_states.emplace(
          build_selected_state_determinant_matrices_from_selected_columns(
              current_input_->structure_data,
              gauge_coupling->gauge_coefficient_response,
              accepted_point_context_->selected_state_indices,
              accepted_point_context_->normalized_state_weights,
              accepted_point_context_->same_spin_pair_cache));
      apply_timing_totals_
          .outer_response_selected_state_rebuild_wall_time_seconds +=
          detail::exact_hvp_elapsed_seconds(selected_state_rebuild_start_time);
    }

    if (use_directional_pair_tiles && local_active_tile_accumulator &&
        !directional_pair_tiles_consumed) {
      detail::stream_directional_pair_tiles(
          accepted_point_context_->same_spin_pair_cache,
          current_input_->orbital_preparation_input.n_active_orbitals,
          accepted_point_context_->prepared_active_space.orbital_result
              .active_orbital_overlap_matrix,
          accepted_point_context_->prepared_active_space
              .active_space_one_electron_result.h1e_act,
          accepted_point_context_->prepared_active_space
              .active_space_two_electron_result,
          active_space_integral_direction,
          accepted_ri_two_electron_cache_.has_value()
              ? accepted_ri_two_electron_cache_->accepted_active_pair_factors
              : nullptr,
          ri_directional_active_pair_factors.has_value()
              ? &*ri_directional_active_pair_factors
              : nullptr,
          true,
          consume_local_active_tile);
      directional_pair_tiles_consumed = true;
    }

    const auto active_gradient_start_time = std::chrono::steady_clock::now();
    ActiveSpaceGradientDirection directional_active_space_gradient;
    if (direct_active_gradient) {
      if (!accepted_point_context_->structure_adjoint_state.has_value()) {
        accepted_point_context_->structure_adjoint_state =
            accepted_structure_action->prepare_active_adjoint(
                accepted_point_context_->selected_state_matrices,
                accepted_point_context_->selected_state_energies);
      }
      const StructureIntegralDirection* direct_ci_direction = nullptr;
      if (precomputed_outer_response != nullptr &&
          precomputed_outer_response->direct_ci_direction.has_value()) {
        direct_ci_direction =
            &*precomputed_outer_response->direct_ci_direction;
      } else if (local_direct_ci_direction.has_value()) {
        direct_ci_direction = &*local_direct_ci_direction;
      }
      directional_active_space_gradient = make_active_gradient_direction(
          accepted_structure_action->active_integral_adjoint_direction(
              *accepted_point_context_->structure_adjoint_state,
              directional_selected_states
                  ? &directional_selected_states.value()
                  : nullptr,
              directional_selected_states ? &state_multipliers : nullptr,
              active_space_integral_direction.overlap,
              active_space_integral_direction.one_electron,
              active_space_integral_direction.packed_two_electron,
              direct_ci_direction,
              components.local_active_response),
          n_active_orbitals);
      const double direct_seconds = detail::exact_hvp_elapsed_seconds(
          active_gradient_start_time);
      if (components.local_active_response) {
        apply_timing_totals_
            .outer_response_local_active_gradient_wall_time_seconds +=
            direct_seconds;
      } else {
        apply_timing_totals_
            .outer_response_structure_active_gradient_wall_time_seconds +=
            direct_seconds;
      }
    } else {
      const auto local_active_gradient_start_time =
          std::chrono::steady_clock::now();
      if (components.local_active_response && precomputed_outer_response != nullptr &&
          precomputed_outer_response->local_active_gradient.has_value()) {
        directional_active_space_gradient =
            *precomputed_outer_response->local_active_gradient;
      } else if (components.local_active_response &&
                 local_active_tile_accumulator) {
        directional_active_space_gradient =
            local_active_tile_accumulator->finish();
      } else if (components.local_active_response) {
        throw std::logic_error(
            "local active response is missing its pair-tile accumulation");
      } else {
        directional_active_space_gradient =
            make_zero_active_space_gradient_direction(n_active_orbitals);
      }
      apply_timing_totals_
          .outer_response_local_active_gradient_wall_time_seconds +=
          detail::exact_hvp_elapsed_seconds(
              local_active_gradient_start_time);

      const auto structure_active_gradient_start_time =
          std::chrono::steady_clock::now();
      if (directional_selected_states) {
        const SelectedStateResponseTiming response_timing =
            add_selected_subspace_response_to_active_space_gradient(
                *current_input_,
                *accepted_point_context_,
                directional_selected_states.value(),
                state_multipliers,
                &directional_active_space_gradient);
        apply_timing_totals_
            .outer_response_same_spin_backward_wall_time_seconds +=
            response_timing.same_spin_seconds;
        apply_timing_totals_
            .outer_response_opposite_spin_backward_wall_time_seconds +=
            response_timing.opposite_spin_seconds;
        apply_timing_totals_
            .outer_response_opposite_spin_packed_gradient_wall_time_seconds +=
            response_timing.opposite_spin_packed_gradient_seconds;
        apply_timing_totals_
            .outer_response_opposite_spin_alpha_overlap_wall_time_seconds +=
            response_timing.opposite_spin_alpha_overlap_seconds;
        apply_timing_totals_
            .outer_response_opposite_spin_beta_overlap_wall_time_seconds +=
            response_timing.opposite_spin_beta_overlap_seconds;
      }
      apply_timing_totals_
          .outer_response_structure_active_gradient_wall_time_seconds +=
          detail::exact_hvp_elapsed_seconds(
              structure_active_gradient_start_time);
    }
    apply_timing_totals_.outer_response_active_gradient_wall_time_seconds +=
        detail::exact_hvp_elapsed_seconds(active_gradient_start_time);
    validate_outer_response_active_gradient(
        directional_active_space_gradient);

    const auto orbital_pullback_start_time =
        std::chrono::steady_clock::now();
    const std::vector<double> outer_response_orbital_value_gradient =
        build_orbital_value_gradient_from_active_space_gradient_direction(
            *current_input_,
            *accepted_point_context_,
            directional_active_space_gradient,
            *orbital_preparation_cache,
            accepted_ri_two_electron_cache_.has_value()
                ? nullptr
                : &accepted_exact_two_electron_cache_,
            accepted_ri_two_electron_cache_.has_value()
                ? &*accepted_ri_two_electron_cache_
                : nullptr,
            accepted_ri_factorization_,
            &outer_response_symmetric_active_overlap_gradient_workspace_,
            &outer_response_symmetric_active_one_electron_gradient_workspace_);
    add_orbital_value_gradient_in_place(
        &combined_core_orbital_value_gradient,
        outer_response_orbital_value_gradient,
        "outer-response");
    apply_timing_totals_.outer_response_orbital_pullback_wall_time_seconds +=
        detail::exact_hvp_elapsed_seconds(orbital_pullback_start_time);

    apply_timing_totals_.outer_response_wall_time_seconds +=
        detail::exact_hvp_elapsed_seconds(outer_response_start_time);
  }
  if (components.direct_core_response) {
    Eigen::MatrixXd delta_auxiliary_active_gradient =
        basis_overlap *
        orbital_preparation_directional_result.delta_active_auxiliary_orbitals *
        accepted_sso_gradient_symmetric_;
    if (compute_outer_response) {
      delta_auxiliary_active_gradient.noalias() +=
          delta_ao_effective_h1e_times_active_auxiliary_orbitals *
          accepted_hho_gradient_symmetric_;
    } else {
      delta_auxiliary_active_gradient.noalias() +=
          delta_ao_effective_h1e *
          accepted_active_auxiliary_orbitals_times_hho_gradient_symmetric_;
    }
    delta_auxiliary_active_gradient.noalias() +=
        ao_effective_h1e *
        delta_active_times_hho_symmetric;
    const auto active_two_electron_start_time =
        std::chrono::steady_clock::now();
    if (accepted_ri_two_electron_cache_.has_value()) {
      accepted_exact_two_electron_apply_workspace_
          .dense_active_gradient_direction =
          apply_ri_packed_active_two_electron_adjoint_hessian_vector(
              accepted_point_context_->packed_active_two_electron_gradient,
              *accepted_ri_two_electron_cache_,
              delta_dense_active_coefficients,
              ri_directional_active_pair_factors.has_value()
                  ? &*ri_directional_active_pair_factors
                  : nullptr);
    } else if (precomputed_two_electron_fixed_adjoint != nullptr &&
        precomputed_two_electron_fixed_adjoint->rows() == n_basis_functions &&
        precomputed_two_electron_fixed_adjoint->cols() == n_active_orbitals) {
      accepted_exact_two_electron_apply_workspace_
          .dense_active_gradient_direction =
              *precomputed_two_electron_fixed_adjoint;
    } else if (compute_outer_response &&
        accepted_exact_two_electron_cache_.accepted_pair_products == nullptr &&
        outer_response_integral_direction_workspace_.two_electron
                .dense_fixed_adjoint_direction.rows() == n_basis_functions &&
        outer_response_integral_direction_workspace_.two_electron
                .dense_fixed_adjoint_direction.cols() == n_active_orbitals) {
      accepted_exact_two_electron_apply_workspace_
          .dense_active_gradient_direction =
              outer_response_integral_direction_workspace_.two_electron
                  .dense_fixed_adjoint_direction;
    } else if (compute_outer_response &&
        (precomputed_directional_pair_products != nullptr
             ? precomputed_directional_pair_products
             : &outer_response_integral_direction_workspace_.two_electron
                    .directional_pair_products)->size() > 0) {
      apply_exact_packed_active_two_electron_adjoint_hessian_vector_fused(
          accepted_exact_two_electron_cache_,
          delta_dense_active_coefficients,
          current_input_->ao_integral_input,
          *(precomputed_directional_pair_products != nullptr
                ? precomputed_directional_pair_products
                : &outer_response_integral_direction_workspace_.two_electron
                       .directional_pair_products),
          &accepted_exact_two_electron_apply_workspace_,
          &accepted_exact_two_electron_apply_workspace_
               .dense_active_gradient_direction);
    } else {
      apply_exact_packed_active_two_electron_adjoint_hessian_vector(
          accepted_exact_two_electron_cache_,
          delta_dense_active_coefficients,
          current_input_->ao_integral_input,
          &accepted_exact_two_electron_apply_workspace_,
          &accepted_exact_two_electron_apply_workspace_
               .dense_active_gradient_direction);
    }
    apply_timing_totals_.active_two_electron_wall_time_seconds +=
        detail::exact_hvp_elapsed_seconds(active_two_electron_start_time);
    delta_auxiliary_active_gradient.noalias() +=
        accepted_exact_two_electron_apply_workspace_
            .dense_active_gradient_direction;

    Eigen::MatrixXd total_inactive_density_direction =
        delta_ao_effective_h1e;
    const Eigen::Map<const Eigen::MatrixXd> ao_backpropagated_inactive_density(
        inactive_density_gradient_data,
        n_basis_functions,
        n_basis_functions);
    total_inactive_density_direction.noalias() +=
        ao_backpropagated_inactive_density;

    const auto orbital_backprop_start_time =
        std::chrono::steady_clock::now();
    const std::vector<double> orbital_value_gradient =
        backpropagate_active_space_orbital_gradient(
            delta_auxiliary_active_gradient,
            total_inactive_density_direction,
            current_input_->orbital_preparation_input,
            accepted_point_context_->prepared_active_space.orbital_result,
            *orbital_preparation_cache);
    add_orbital_value_gradient_in_place(
        &combined_core_orbital_value_gradient,
        orbital_value_gradient,
        "direct-core");
    apply_timing_totals_.orbital_backprop_wall_time_seconds +=
        detail::exact_hvp_elapsed_seconds(orbital_backprop_start_time);

  }

  if (components.fixed_upstream_pullback &&
      accepted_point_context_->total_active_auxiliary_gradient.rows() ==
          n_basis_functions &&
      accepted_point_context_->total_active_auxiliary_gradient.cols() ==
          n_active_orbitals &&
      accepted_point_context_->total_inactive_density_gradient.size() ==
          ao_matrix_size) {
    const auto fixed_upstream_pullback_start_time =
        std::chrono::steady_clock::now();
    const std::vector<double> fixed_upstream_orbital_value_gradient =
        apply_fixed_upstream_orbital_pullback_direction(
            current_input_->orbital_preparation_input,
            dense_orbital_tangent_context,
            accepted_point_context_->total_active_auxiliary_gradient,
            orbital_preparation_directional_result
                .basis_overlap_times_delta_active_orbitals,
            accepted_point_context_->total_inactive_density_gradient,
            input_retract_tangent,
            *orbital_preparation_cache);
    add_orbital_value_gradient_in_place(
        &combined_core_orbital_value_gradient,
        fixed_upstream_orbital_value_gradient,
        "fixed-upstream");
    apply_timing_totals_.fixed_upstream_pullback_wall_time_seconds +=
        detail::exact_hvp_elapsed_seconds(fixed_upstream_pullback_start_time);
  }

  if (!combined_core_orbital_value_gradient.empty()) {
    // Full exact_ctx matvecs used to project the direct-core/fixed-upstream
    // pullback and the outer-response pullback separately.  Both contributions
    // live in the same raw sparse-orbital coefficient chart, so combining them
    // before `gather_from_full + project_reduced_gradient` removes one full
    // packed/reduced projection from every HVP apply without changing the
    // accepted-point nonredundant semantics.
    const Eigen::VectorXd packed_response =
        parameter_view_.gather_from_full(combined_core_orbital_value_gradient);
    response += nonredundant_space_->project_reduced_gradient(packed_response);

    // The accepted-point lift x(d) = x_0 + U_p d is linear in the raw sparse
    // coefficients, so it contributes no separate chart-curvature term.
    // Orbital normalization and inactive projection are nonlinear downstream
    // maps; their second-order pullback is included in the fixed-upstream term
    // accumulated above.
  }

  record_apply_wall_time();
  return response;
}

Eigen::VectorXd ExactHvpOperator::State::apply_structure_response_adjoint(
    const Eigen::Ref<const Eigen::MatrixXd>& coefficient_response,
    const Eigen::Ref<const Eigen::MatrixXd>& state_multipliers) const {
  const int n_structures = accepted_point_context_->n_structures;
  const int n_states = static_cast<int>(
      accepted_point_context_->selected_state_indices.size());
  if (coefficient_response.rows() != n_structures ||
      coefficient_response.cols() != n_states ||
      state_multipliers.rows() != n_states ||
      state_multipliers.cols() != n_states ||
      !coefficient_response.allFinite() ||
      !state_multipliers.allFinite()) {
    throw std::invalid_argument(
        "structure-response adjoint has incompatible dimensions or values");
  }
  if (accepted_point_context_->normalized_state_weights.size() !=
      static_cast<std::size_t>(n_states)) {
    throw std::logic_error(
        "structure-response adjoint is missing selected-state weights");
  }
  if (!supports_analytic_core_model() ||
      accepted_orbital_preparation_cache_ == nullptr) {
    throw std::logic_error(
        "structure-response adjoint requires the analytic orbital pullback");
  }

  const int n_active =
      current_input_->orbital_preparation_input.n_active_orbitals;
  const auto selected_state_start = std::chrono::steady_clock::now();
  const SelectedStateDeterminantMatrices directional_selected_states =
      build_selected_state_determinant_matrices_from_selected_columns(
          current_input_->structure_data,
          coefficient_response,
          accepted_point_context_->selected_state_indices,
          accepted_point_context_->normalized_state_weights,
          accepted_point_context_->same_spin_pair_cache);
  apply_timing_totals_
      .outer_response_selected_state_rebuild_wall_time_seconds +=
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - selected_state_start).count();

  const StructureAction* structure_action =
      outer_response_context()
          .selected_state_eigen_response_operator.structure_action;
  if (structure_action == nullptr) {
    throw std::logic_error(
        "structure-response adjoint requires a structure action");
  }

  const auto active_gradient_start = std::chrono::steady_clock::now();
  ActiveSpaceGradientDirection active_gradient;
  if (structure_action->supports_integral_direction()) {
    if (!accepted_point_context_->structure_adjoint_state.has_value()) {
      accepted_point_context_->structure_adjoint_state =
          structure_action->prepare_active_adjoint(
              accepted_point_context_->selected_state_matrices,
              accepted_point_context_->selected_state_energies);
    }
    active_gradient = make_active_gradient_direction(
        structure_action->active_integral_response_adjoint(
            *accepted_point_context_->structure_adjoint_state,
            directional_selected_states,
            state_multipliers),
        n_active);
  } else {
    active_gradient = make_zero_active_space_gradient_direction(n_active);
    add_selected_subspace_response_to_active_space_gradient(
        *current_input_,
        *accepted_point_context_,
        directional_selected_states,
        state_multipliers,
        &active_gradient);
  }
  validate_outer_response_active_gradient(active_gradient);
  const double active_gradient_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - active_gradient_start).count();
  apply_timing_totals_.outer_response_active_gradient_wall_time_seconds +=
      active_gradient_seconds;
  apply_timing_totals_
      .outer_response_structure_active_gradient_wall_time_seconds +=
      active_gradient_seconds;

  const auto orbital_pullback_start = std::chrono::steady_clock::now();
  const std::vector<double> orbital_value_gradient =
      build_orbital_value_gradient_from_active_space_gradient_direction(
          *current_input_,
          *accepted_point_context_,
          active_gradient,
          *accepted_orbital_preparation_cache_,
          accepted_ri_two_electron_cache_.has_value()
              ? nullptr
              : &accepted_exact_two_electron_cache_,
          accepted_ri_two_electron_cache_.has_value()
              ? &*accepted_ri_two_electron_cache_
              : nullptr,
          accepted_ri_factorization_,
          &outer_response_symmetric_active_overlap_gradient_workspace_,
          &outer_response_symmetric_active_one_electron_gradient_workspace_);
  apply_timing_totals_.outer_response_orbital_pullback_wall_time_seconds +=
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - orbital_pullback_start).count();
  const Eigen::VectorXd packed =
      parameter_view_.gather_from_full(orbital_value_gradient);
  return nonredundant_space_->project_reduced_gradient(packed);
}

}  // namespace xmvb::vb
