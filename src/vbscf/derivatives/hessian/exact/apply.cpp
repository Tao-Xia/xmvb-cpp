#include "vbscf/derivatives/hessian/exact/operator.hpp"
#include "core/timing.hpp"

#include "vbscf/derivatives/hessian/exact/apply_internal.hpp"
#include "integrals/active/frame.hpp"
#include "integrals/response/orbital_pair.hpp"

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

#include "linalg/eigen_response.hpp"
#include "vbscf/derivatives/hessian/coupled/coupling.hpp"
#include "integrals/response/adjoint.hpp"
#include "core/pair_index.hpp"
#include "vbscf/derivatives/hessian/responses/active_space/integral_direction.hpp"
#include "vbscf/derivatives/hessian/responses/active_space/local_tile_accumulator_internal.hpp"
#include "vbscf/derivatives/hessian/responses/active_space/outer_response.hpp"
#include "vbscf/derivatives/hessian/responses/orbital/preparation.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/backward.hpp"
#include "vbscf/derivatives/hessian/responses/structure/directional.hpp"
#include "vbscf/structures/assembly/selected_coefficients.hpp"

namespace xmvb::vb {

namespace {

void add_orbital_value_gradient_in_place(std::vector<double>* target,
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

void throw_if_nonfinite(const std::vector<double>& values, const char* label) {
    const auto nonfinite_it = std::find_if(
        values.begin(), values.end(), [](double value) { return !std::isfinite(value); });
    if (nonfinite_it == values.end()) {
        return;
    }
    throw std::runtime_error(std::string(label) + " contains non-finite values");
}

void throw_if_nonfinite(const Eigen::MatrixXd& values, const char* label) {
    if (values.allFinite()) {
        return;
    }
    throw std::runtime_error(std::string(label) + " contains non-finite values");
}

void throw_if_nonfinite(const Eigen::VectorXd& values, const char* label) {
    if (values.allFinite()) {
        return;
    }
    throw std::runtime_error(std::string(label) + " contains non-finite values");
}

using detail::make_active_gradient_direction;
} // namespace

Eigen::VectorXd ExactHvpOperator::apply_reduced(const Eigen::VectorXd& reduced_direction,
                                                HvpComponents components) const {
    return state_->apply_reduced(reduced_direction, components);
}

Eigen::VectorXd ExactHvpOperator::State::apply_reduced(const Eigen::VectorXd& reduced_direction,
                                                       HvpComponents components) const {
    return apply_reduced_impl(reduced_direction,
                              components,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr);
}

OrbitalCouplingAction
ExactHvpOperator::apply_orbital_coupling(const Eigen::VectorXd& reduced_direction,
                                         StructureResponseRequest* ci_response) const {
    return state_->apply_orbital_coupling(reduced_direction, ci_response);
}

Eigen::VectorXd ExactHvpOperator::apply_orbital(const Eigen::VectorXd& direction) const {
    return state_->apply_orbital(direction);
}

Eigen::VectorXd ExactHvpOperator::State::apply_orbital(const Eigen::VectorXd& direction) const {
    OrbitalCouplingAction gauge;
    return apply_reduced_impl(direction,
                              {.direct_core_response = true,
                               .fixed_upstream_pullback = true,
                               .local_active_response = true,
                               .structure_response = false},
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              &gauge,
                              false);
}

OrbitalCouplingAction
ExactHvpOperator::State::apply_orbital_coupling(const Eigen::VectorXd& reduced_direction,
                                                StructureResponseRequest* ci_response) const {
    OrbitalCouplingAction result;
    result.orbital_hessian = apply_reduced_impl(reduced_direction,
                                                {.direct_core_response = true,
                                                 .fixed_upstream_pullback = true,
                                                 .local_active_response = true,
                                                 .structure_response = false},
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                &result,
                                                true,
                                                ci_response);
    return result;
}

StructureResponseBlock ExactHvpOperator::solve_structure_response_block(
    const std::vector<Eigen::MatrixXd>& scaled_structure_forcing,
    double relative_residual_tolerance) const {
    return state_->solve_structure_response_block(scaled_structure_forcing,
                                                  relative_residual_tolerance);
}

StructureResponseBlock ExactHvpOperator::State::solve_structure_response_block(
    const std::vector<Eigen::MatrixXd>& scaled_structure_forcing,
    double relative_residual_tolerance) const {
    xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::StructureSolve);
    if (scaled_structure_forcing.empty() || !(relative_residual_tolerance > 0.0) ||
        relative_residual_tolerance >= 1.0 || !std::isfinite(relative_residual_tolerance)) {
        throw std::invalid_argument("invalid structure-response forcing block");
    }
    const int n_structures = accepted_point_context_->n_structures;
    const int n_states = static_cast<int>(accepted_point_context_->selected_state_indices.size());
    const int n_directions = static_cast<int>(scaled_structure_forcing.size());
    Eigen::MatrixXd raw_forcing(n_structures, n_states * n_directions);
    for (int direction = 0; direction < n_directions; ++direction) {
        const Eigen::MatrixXd& forcing =
            scaled_structure_forcing[static_cast<std::size_t>(direction)];
        if (forcing.rows() != n_structures || forcing.cols() != n_states || !forcing.allFinite()) {
            throw std::invalid_argument("structure-response forcing has inconsistent dimensions");
        }
        for (int state = 0; state < n_states; ++state) {
            const double weight =
                accepted_point_context_->normalized_state_weights[static_cast<std::size_t>(state)];
            if (!(weight > 0.0) || !std::isfinite(weight)) {
                throw std::runtime_error("structure-response coordinate weight is invalid");
            }
            raw_forcing.col(direction * n_states + state) =
                forcing.col(state) / std::sqrt(2.0 * weight);
        }
    }

    const auto& response_operator = outer_response_context().selected_state_eigen_response_operator;
    const SelectedStateGeneralizedEigenDirectionalResponse response =
        response_operator.apply_direction_block(
            raw_forcing,
            Eigen::MatrixXd::Zero(n_structures, n_states * n_directions),
            relative_residual_tolerance,
            false);
    apply_timing_totals_.structure_response_block_actions += response.block_actions;
    if (!response.linear_iterations.empty()) {
        apply_timing_totals_.max_structure_response_iterations =
            std::max(apply_timing_totals_.max_structure_response_iterations,
                     *std::max_element(response.linear_iterations.begin(),
                                       response.linear_iterations.end()));
    }
    apply_timing_totals_.max_structure_response_relative_residual =
        std::max(apply_timing_totals_.max_structure_response_relative_residual,
                 response.max_relative_residual);

    StructureResponseBlock result;
    if (response.equation_residuals.size() != scaled_structure_forcing.size()) {
        throw std::runtime_error("structure response lacks bordered residual certificates: got " +
                                 std::to_string(response.equation_residuals.size()) +
                                 ", expected " + std::to_string(scaled_structure_forcing.size()));
    }
    result.scaled_coefficients.reserve(scaled_structure_forcing.size());
    result.coefficient_responses.reserve(scaled_structure_forcing.size());
    result.adjoint_multipliers.reserve(scaled_structure_forcing.size());
    result.scaled_equation_residuals.reserve(scaled_structure_forcing.size());
    const Eigen::MatrixXd selected_metric =
        response_operator.selected_eigenvectors.transpose() * response_operator.overlap_selected;
    for (int direction = 0; direction < n_directions; ++direction) {
        Eigen::MatrixXd raw =
            response.delta_selected_eigenvector_matrix.middleCols(direction * n_states, n_states);
        Eigen::MatrixXd scaled = raw;
        Eigen::MatrixXd scaled_residual =
            response.equation_residuals[static_cast<std::size_t>(direction)];
        for (int state = 0; state < n_states; ++state) {
            const double weight =
                accepted_point_context_->normalized_state_weights[static_cast<std::size_t>(state)];
            scaled.col(state) *= std::sqrt(2.0 * weight);
            scaled_residual.col(state) *= std::sqrt(2.0 * weight);
        }
        result.scaled_coefficients.push_back(std::move(scaled));
        result.coefficient_responses.push_back(std::move(raw));
        result.adjoint_multipliers.push_back(
            -response.selected_matrix_responses[static_cast<std::size_t>(direction)] +
            selected_metric.partialPivLu().solve(
                response_operator.selected_eigenvectors.transpose() *
                raw_forcing.middleCols(direction * n_states, n_states)));
        result.scaled_equation_residuals.push_back(std::move(scaled_residual));
    }
    result.revision = response_operator.revision();
    result.block_actions = response.block_actions;
    result.max_relative_residual = response.max_relative_residual;
    return result;
}

Eigen::VectorXd ExactHvpOperator::apply_structure_coupling_adjoint(
    const Eigen::Ref<const Eigen::MatrixXd>& coefficient_response,
    const Eigen::Ref<const Eigen::MatrixXd>& adjoint_multipliers) const {
    return state_->apply_structure_coupling_adjoint(coefficient_response, adjoint_multipliers);
}

Eigen::VectorXd ExactHvpOperator::State::apply_structure_coupling_adjoint(
    const Eigen::Ref<const Eigen::MatrixXd>& coefficient_response,
    const Eigen::Ref<const Eigen::MatrixXd>& adjoint_multipliers) const {
    return apply_structure_response_adjoint(coefficient_response, adjoint_multipliers);
}

Eigen::VectorXd ExactHvpOperator::apply_structure_coupling_adjoint(
    const StructureActiveIntegralAdjoint& adjoint) const {
    return state_->apply_structure_coupling_adjoint(adjoint);
}

Eigen::VectorXd ExactHvpOperator::State::apply_structure_coupling_adjoint(
    const StructureActiveIntegralAdjoint& adjoint) const {
    xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::StructureAdjoint);
    return pullback_structure_response(make_active_gradient_direction(
        adjoint, current_input_->orbital_preparation_input.n_active_orbitals));
}

Eigen::VectorXd ExactHvpOperator::State::apply_reduced_impl(
    const Eigen::VectorXd& reduced_direction,
    HvpComponents components,
    const Eigen::VectorXd* precomputed_ao_response,
    const Eigen::MatrixXd* precomputed_active_projection,
    const Eigen::VectorXd* precomputed_delta_packed_active_two_electron,
    const Eigen::MatrixXd* precomputed_ri_active_pair_factor_direction,
    const xmvb::integrals::ExactCtxPairMatrix* precomputed_directional_pair_products,
    const PrecomputedDirection* precomputed_direction,
    OrbitalCouplingAction* coupling_output,
    bool horizontal_forcing,
    StructureResponseRequest* ci_response) const {
    xmvb::profiling::ScopedPhase apply_phase(xmvb::profiling::Phase::HvpApply);
    const auto apply_start_time = std::chrono::steady_clock::now();
    auto record_apply_wall_time = [&]() {
        ++apply_timing_totals_.apply_count;
        apply_timing_totals_.total_apply_wall_time_seconds +=
            detail::exact_hvp_elapsed_seconds(apply_start_time);
    };
    Eigen::VectorXd response = Eigen::VectorXd::Zero(reduced_direction.size());

    if (!supports_analytic_core_model()) {
        throw std::runtime_error(
            "exact_ctx analytic core HVP is unavailable for the current accepted point");
    }

    const int n_basis_functions = current_input_->orbital_preparation_input.n_basis_functions;
    const int n_active_orbitals = current_input_->orbital_preparation_input.n_active_orbitals;
    const std::size_t ao_matrix_size = n_basis_functions * n_basis_functions;
    if (coupling_output != nullptr && components.structure_response) {
        throw std::invalid_argument("coupled orbital action cannot solve the structure response");
    }
    if (coupling_output != nullptr && horizontal_forcing) {
        const auto n_states = accepted_point_context_->selected_state_eigenvectors.cols();
        coupling_output->gauge_frame = Eigen::MatrixXd::Zero(n_states, n_states);
        coupling_output->scaled_structure_forcing = Eigen::MatrixXd::Zero(
            accepted_point_context_->n_structures,
            static_cast<int>(accepted_point_context_->selected_state_indices.size()));
    }

    const auto core_setup_start_time = std::chrono::steady_clock::now();
    xmvb::profiling::ScopedPhase setup_phase(xmvb::profiling::Phase::HvpSetup);
    const Eigen::VectorXd local_packed_direction =
        precomputed_direction == nullptr ? nonredundant_space_->expand_step(reduced_direction)
                                         : Eigen::VectorXd();
    const Eigen::VectorXd& packed_direction = precomputed_direction != nullptr
                                                  ? precomputed_direction->packed_direction
                                                  : local_packed_direction;
    if (packed_direction.norm() == 0.0) {
        // A pure CI direction needs no orbital integral transform or mixed sweep.
        if (ci_response) {
            auto& accepted = accepted_point_context_->structure_adjoint_state;
            const auto& action = structure_action();
            if (!accepted)
                accepted = action.prepare_active_adjoint(
                    accepted_point_context_->selected_state_eigenvectors,
                    accepted_point_context_->normalized_state_weights,
                    accepted_point_context_->selected_state_energies);
            ci_response->result = action.apply_response(*accepted, ci_response->coefficients);
        }
        record_apply_wall_time();
        return Eigen::VectorXd::Zero(reduced_direction.size());
    }

    const AcceptedOrbitalPreparationCache* orbital_preparation_cache =
        accepted_orbital_preparation_cache_.get();
    std::optional<PrecomputedDirection> local_precomputed_direction;
    if (precomputed_direction == nullptr) {
        local_precomputed_direction.emplace(prepare_direction(local_packed_direction));
    }
    const PrecomputedDirection& prepared_direction =
        precomputed_direction != nullptr ? *precomputed_direction : *local_precomputed_direction;
    const DenseOrbitalTangentContext& dense_orbital_tangent_context =
        prepared_direction.dense_orbital_tangent_context;
    const Eigen::VectorXd input_retract_tangent =
        components.fixed_upstream_pullback
            ? nonredundant_space_->expand_retract_input_tangent(
                  current_input_->orbital_preparation_input, reduced_direction)
            : Eigen::VectorXd();
    const OrbitalPreparationDirectionalResult& orbital_preparation_directional_result =
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
    const auto& delta_dense_active_coefficients = prepared_direction.integral_tangent;
    apply_timing_totals_.core_setup_wall_time_seconds +=
        detail::exact_hvp_elapsed_seconds(core_setup_start_time);
    setup_phase.stop();
    // `F11` and `delta F11` are explicitly symmetrized in the AO-H1E builder, so
    // the directional active-space matrix gradient only needs the two distinct
    // left contractions `delta F11 * T_active` and `F11 * delta T_active`.
    const Eigen::MatrixXd delta_active_times_hho_symmetric =
        orbital_preparation_directional_result.delta_active_auxiliary_orbitals *
        accepted_hho_gradient_symmetric_;
    Eigen::MatrixXd total_inactive_density_direction;
    Eigen::MatrixXd inactive_density_projection;
    Eigen::MatrixXd delta_ao_effective_h1e_times_active_auxiliary_orbitals;
    Eigen::MatrixXd ri_factor_direction, ri_fixed_response;
    const Eigen::MatrixXd* supplied_ri_factor_direction =
        precomputed_ri_active_pair_factor_direction;
    const Eigen::MatrixXd* fixed_ri_two_electron = prepared_direction.ri_fixed_two_electron.size()
                                                       ? &prepared_direction.ri_fixed_two_electron
                                                       : nullptr;
    const int n_inactive = (current_input_->orbital_preparation_input.n_total_electrons -
                            current_input_->orbital_preparation_input.n_active_electrons) /
                           2;
    const Eigen::Index response_columns =
        accepted_ri_factorization_ != nullptr ? n_inactive : n_basis_functions;
    if (precomputed_ao_response != nullptr || precomputed_active_projection != nullptr) {
        if (precomputed_ao_response == nullptr || precomputed_active_projection == nullptr ||
            precomputed_ao_response->size() != n_basis_functions * response_columns ||
            precomputed_active_projection->rows() != n_basis_functions ||
            precomputed_active_projection->cols() != accepted_active_auxiliary_orbitals_.cols()) {
            throw std::invalid_argument(
                "precomputed block AO-H1E response has inconsistent dimensions");
        }
        const Eigen::Map<const Eigen::MatrixXd> response(
            precomputed_ao_response->data(), n_basis_functions, response_columns);
        if (accepted_ri_factorization_ != nullptr)
            inactive_density_projection = response;
        else
            total_inactive_density_direction = response;
        delta_ao_effective_h1e_times_active_auxiliary_orbitals = *precomputed_active_projection;
    } else {
        const auto ao_effective_one_electron_fused_start_time = std::chrono::steady_clock::now();
        xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::AoResponse);
        if (accepted_ri_factorization_ != nullptr) {
            const Eigen::Map<const Eigen::MatrixXd> weights(delta_active_times_hho_symmetric.data(),
                                                            delta_active_times_hho_symmetric.size(),
                                                            1);
            auto block =
                ri_ao_responses({&prepared_direction, 1}, weights, components.direct_core_response);
            auto& ri = block.front();
            inactive_density_projection = std::move(ri.inactive);
            delta_ao_effective_h1e_times_active_auxiliary_orbitals = std::move(ri.active);
            if (ri.fixed_two_electron.size()) {
                ri_factor_direction = std::move(ri.factor_direction);
                ri_fixed_response = std::move(ri.fixed_two_electron);
                supplied_ri_factor_direction = &ri_factor_direction;
                fixed_ri_two_electron = &ri_fixed_response;
            }
        } else {
            Eigen::MatrixXd total_ao_effective_one_electron_direction =
                orbital_preparation_directional_result.delta_inactive_density;
            total_ao_effective_one_electron_direction.noalias() +=
                delta_active_times_hho_symmetric * accepted_active_auxiliary_orbitals_.transpose();
            ao_h1e_symmetrized_gradient_workspace_ =
                total_ao_effective_one_electron_direction +
                total_ao_effective_one_electron_direction.transpose();
            const Eigen::Map<const Eigen::MatrixXd> density(
                orbital_preparation_directional_result.delta_inactive_density.data(),
                ao_matrix_size,
                1);
            const Eigen::Map<const Eigen::MatrixXd> pullback(
                ao_h1e_symmetrized_gradient_workspace_.data(), ao_matrix_size, 1);
            const auto& frame = accepted_point_context_->prepared_active_space
                                    .active_space_two_electron_result.orbital_transform;
            const Eigen::MatrixXd integral_weights =
                frame.size() ? (delta_active_times_hho_symmetric * frame.transpose()).eval()
                             : delta_active_times_hho_symmetric;
            const Eigen::Map<const Eigen::MatrixXd> weights(
                integral_weights.data(), integral_weights.size(), 1);
            Eigen::MatrixXd response, projection;
            detail::apply_exact_ao_response(density,
                                            pullback,
                                            weights,
                                            accepted_exact_two_electron_cache_,
                                            current_input_->ao_integral_input,
                                            &ao_h1e_batch_workspace_,
                                            &response,
                                            &projection);
            total_inactive_density_direction = Eigen::Map<const Eigen::MatrixXd>(
                response.data(), n_basis_functions, n_basis_functions);
            delta_ao_effective_h1e_times_active_auxiliary_orbitals =
                Eigen::Map<const Eigen::MatrixXd>(projection.data(),
                                                  n_basis_functions,
                                                  accepted_active_auxiliary_orbitals_.cols());
        }
        apply_timing_totals_.ao_effective_one_electron_fused_wall_time_seconds +=
            detail::exact_hvp_elapsed_seconds(ao_effective_one_electron_fused_start_time);
    }

    const bool build_orbital_coupling = coupling_output != nullptr;
    const bool compute_outer_response =
        components.local_active_response || components.structure_response || build_orbital_coupling;
    const bool direct_active_gradient =
        compute_outer_response &&
        outer_response_context()
            .selected_state_eigen_response_operator.structure_action->supports_integral_direction();
    // CI needs delta G, not delta B. A resident three-active image removes its
    // auxiliary contraction even when the fixed RI jet separately needs delta B.
    const bool ri_orbital_pair_direction =
        direct_active_gradient && accepted_ri_two_electron_cache_ &&
        integrals::ri_uses_orbital_pairs(*accepted_ri_two_electron_cache_);
    const bool packed_outer_two_electron_required =
        !accepted_ri_two_electron_cache_ || ri_orbital_pair_direction;
    Eigen::VectorXd local_packed_two_electron;
    const Eigen::VectorXd* packed_two_electron_direction =
        precomputed_delta_packed_active_two_electron;
    if (ri_orbital_pair_direction && packed_two_electron_direction == nullptr) {
        profiling::ScopedPhase phase(profiling::Phase::ActiveDirection);
        const auto start = std::chrono::steady_clock::now();
        const auto& cache = *accepted_ri_two_electron_cache_;
        local_packed_two_electron =
            integrals::packed_orbital_pair_direction(*cache.orbital_pair_products,
                                                     delta_dense_active_coefficients,
                                                     cache.active_pair_first_indices,
                                                     cache.active_pair_second_indices,
                                                     cache.n_active_orbitals);
        packed_two_electron_direction = &local_packed_two_electron;
        apply_timing_totals_.outer_response_active_space_integrals_wall_time_seconds +=
            detail::exact_hvp_elapsed_seconds(start);
    }
    std::optional<Eigen::MatrixXd> local_ri_factor_direction;
    const Eigen::MatrixXd* ri_directional_active_pair_factors = nullptr;
    if (accepted_ri_two_electron_cache_.has_value() &&
        (components.direct_core_response ||
         (compute_outer_response && !ri_orbital_pair_direction))) {
        if (supplied_ri_factor_direction != nullptr) {
            if (supplied_ri_factor_direction->rows() !=
                    accepted_ri_two_electron_cache_->n_auxiliary_functions ||
                supplied_ri_factor_direction->cols() !=
                    static_cast<Eigen::Index>(
                        accepted_ri_two_electron_cache_->active_pair_first_indices.size())) {
                throw std::invalid_argument(
                    "precomputed RI active-pair factor direction has inconsistent dimensions");
            }
            ri_directional_active_pair_factors = supplied_ri_factor_direction;
        } else {
            local_ri_factor_direction.emplace(
                xmvb::integrals::compute_ri_active_pair_factor_directional_derivative(
                    *accepted_ri_two_electron_cache_, delta_dense_active_coefficients));
            ri_directional_active_pair_factors = &*local_ri_factor_direction;
        }
    }
    const Eigen::MatrixXd* selected_ri_factor_direction =
        ri_orbital_pair_direction ? nullptr : ri_directional_active_pair_factors;
    std::vector<double> combined_core_orbital_value_gradient;
    bool two_electron_jet_consumed = false;

    if (compute_outer_response) {
        const auto outer_response_start_time = std::chrono::steady_clock::now();
        const PrecomputedOuterResponse* precomputed_outer_response =
            precomputed_direction != nullptr && precomputed_direction->outer_response.has_value()
                ? &precomputed_direction->outer_response.value()
                : nullptr;

        const auto active_space_integrals_start_time = std::chrono::steady_clock::now();
        xmvb::profiling::ScopedPhase active_direction_phase(
            xmvb::profiling::Phase::ActiveDirection);
        const ActiveSpaceIntegralDirectionView active_space_integral_direction =
            precomputed_outer_response != nullptr
                ? ActiveSpaceIntegralDirectionView{precomputed_outer_response->integral_direction
                                                       .overlap,
                                                   precomputed_outer_response->integral_direction
                                                       .one_electron,
                                                   precomputed_outer_response->integral_direction
                                                       .packed_two_electron}
                : build_active_integral_direction(
                      prepared_direction,
                      delta_ao_effective_h1e_times_active_auxiliary_orbitals,
                      packed_outer_two_electron_required ? packed_two_electron_direction : nullptr,
                      &outer_response_integral_direction_workspace_,
                      packed_outer_two_electron_required);
        if (precomputed_outer_response == nullptr) {
            apply_timing_totals_.outer_response_active_space_integrals_wall_time_seconds +=
                detail::exact_hvp_elapsed_seconds(active_space_integrals_start_time);
        }

        const auto structure_matrices_start_time = std::chrono::steady_clock::now();
        active_direction_phase.stop();
        xmvb::profiling::ScopedPhase selected_direction_phase(
            xmvb::profiling::Phase::SelectedDirection);
        const bool use_directional_pair_tiles =
            precomputed_outer_response == nullptr && !direct_active_gradient &&
            (components.local_active_response || components.structure_response ||
             build_orbital_coupling);
        std::unique_ptr<detail::LocalActiveSpaceTileAccumulator> local_active_tile_accumulator;
        if (use_directional_pair_tiles && components.local_active_response) {
            local_active_tile_accumulator =
                std::make_unique<detail::LocalActiveSpaceTileAccumulator>(
                    *current_input_, *accepted_point_context_, active_space_integral_direction);
        }
        const auto consume_local_active_tile =
            [&](bool alpha_channel,
                bool beta_channel,
                const AcceptedSpinPairTile& accepted,
                const detail::SameSpinDirectionalPairTileView& same_spin,
                const detail::DirectionalOppositeSpinPairTileView* opposite_spin) {
                if (local_active_tile_accumulator) {
                    local_active_tile_accumulator->consume(
                        alpha_channel, beta_channel, accepted, same_spin, opposite_spin);
                }
            };
        bool directional_pair_tiles_consumed = false;
        const SelectedStateGeneralizedEigenDirectionalResponse*
            directional_selected_state_response = nullptr;
        SelectedStateGeneralizedEigenDirectionalResponse local_directional_selected_state_response;
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
                    coupling_output->gauge_frame = gauge_coupling->gauge_frame;
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
                                  ? accepted_ri_two_electron_cache_->accepted_active_pair_factors
                                  : nullptr,
                              ri_directional_active_pair_factors,
                              consume_local_active_tile,
                              ci_response)
                        : build_selected_structure_direction(
                              outer_response_context(),
                              active_space_integral_direction,
                              accepted_ri_two_electron_cache_.has_value() &&
                                      !ri_orbital_pair_direction
                                  ? accepted_ri_two_electron_cache_->accepted_active_pair_factors
                                  : nullptr,
                              selected_ri_factor_direction,
                              ci_response);
                directional_pair_tiles_consumed = use_directional_pair_tiles;
                local_direct_ci_direction = std::move(images.direct_ci_direction);
                apply_timing_totals_.outer_response_structure_matrices_wall_time_seconds +=
                    detail::exact_hvp_elapsed_seconds(structure_matrices_start_time);

                if (build_orbital_coupling) {
                    const auto& eigen =
                        outer_response_context().selected_state_eigen_response_operator;
                    const xmvb::linalg::EqualWeightEigenCoupling coupling =
                        xmvb::linalg::prepare_equal_weight_generalized_eigen_coupling(
                            eigen.selected_eigenvalues,
                            eigen.selected_eigenvectors,
                            eigen.overlap_selected,
                            eigen.selected_residuals,
                            images.delta_hamiltonian_selected,
                            images.delta_overlap_selected,
                            horizontal_forcing);
                    local_gauge_coupling = scale_equal_weight_structure_coupling(
                        coupling, accepted_point_context_->normalized_state_weights);
                    gauge_coupling = &*local_gauge_coupling;
                    coupling_output->scaled_structure_forcing =
                        gauge_coupling->horizontal_forcing.scaled_coefficients;
                    coupling_output->gauge_frame = gauge_coupling->gauge_frame;
                    if (local_direct_ci_direction.has_value() &&
                        local_direct_ci_direction->connection.has_value()) {
                        auto diagnostic = std::move(*local_direct_ci_direction->connection);
                        // Include normalization/gauge terms even for a nonzero Ritz residual.
                        diagnostic.representation_forcing +=
                            coupling.horizontal_forcing - coupling.forcing;
                        for (Eigen::Index state = 0; state < diagnostic.overlap.cols(); ++state) {
                            const double scale = std::sqrt(
                                2.0 * accepted_point_context_->normalized_state_weights[state]);
                            diagnostic.representation_forcing.col(state) *= scale;
                            diagnostic.integral_forcing.col(state) *= scale;
                            diagnostic.overlap.col(state) *= scale;
                            diagnostic.squared_norm[state] *= scale * scale;
                        }
                        coupling_output->connection = std::move(diagnostic);
                    }
                } else {
                    const auto eigensystem_start_time = std::chrono::steady_clock::now();
                    local_directional_selected_state_response =
                        outer_response_context().selected_state_eigen_response_operator.apply(
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
                    apply_timing_totals_.max_structure_response_iterations =
                        std::max(apply_timing_totals_.max_structure_response_iterations,
                                 *std::max_element(
                                     directional_selected_state_response->linear_iterations.begin(),
                                     directional_selected_state_response->linear_iterations.end()));
                }
                apply_timing_totals_.max_structure_response_relative_residual =
                    std::max(apply_timing_totals_.max_structure_response_relative_residual,
                             directional_selected_state_response->max_relative_residual);
                if (directional_selected_state_response->selected_matrix_responses.size() != 1) {
                    throw std::logic_error(
                        "one HVP direction must produce one selected-space response matrix");
                }
            }
        }

        if (use_directional_pair_tiles && local_active_tile_accumulator &&
            !directional_pair_tiles_consumed) {
            detail::stream_directional_pair_tiles(
                accepted_point_context_->same_spin_pair_cache,
                current_input_->orbital_preparation_input.n_active_orbitals,
                accepted_point_context_->prepared_active_space.orbital_result
                    .active_orbital_overlap_matrix,
                accepted_point_context_->prepared_active_space.active_space_one_electron_result
                    .h1e_act,
                accepted_point_context_->prepared_active_space.active_space_two_electron_result,
                active_space_integral_direction,
                accepted_ri_two_electron_cache_.has_value()
                    ? accepted_ri_two_electron_cache_->accepted_active_pair_factors
                    : nullptr,
                ri_directional_active_pair_factors,
                true,
                consume_local_active_tile,
                static_cast<int>(accepted_point_context_->selected_state_matrices.states.size()));
        }

        selected_direction_phase.stop();
        xmvb::profiling::ScopedPhase backward_phase(xmvb::profiling::Phase::ActiveBackward);
        std::optional<ActiveSpaceGradientDirection> local_active_gradient;
        const ActiveSpaceGradientDirection* active_gradient =
            precomputed_outer_response != nullptr && precomputed_outer_response->active_gradient
                ? &*precomputed_outer_response->active_gradient
                : nullptr;
        if (active_gradient == nullptr) {
            const auto* direct_ci =
                precomputed_outer_response != nullptr &&
                        precomputed_outer_response->direct_ci_direction
                    ? &*precomputed_outer_response->direct_ci_direction
                    : (local_direct_ci_direction ? &*local_direct_ci_direction : nullptr);
            const auto* local = precomputed_outer_response != nullptr &&
                                        precomputed_outer_response->local_active_gradient
                                    ? &*precomputed_outer_response->local_active_gradient
                                    : nullptr;
            local_active_gradient = active_response_gradient({active_space_integral_direction,
                                                              directional_selected_state_response,
                                                              gauge_coupling,
                                                              direct_ci,
                                                              selected_ri_factor_direction,
                                                              local,
                                                              local_active_tile_accumulator.get()},
                                                             components);
            active_gradient = &*local_active_gradient;
        }
        const auto& directional_active_space_gradient = *active_gradient;

        Eigen::MatrixXd two_electron_jet;
        const Eigen::MatrixXd* two_electron_orbital_gradient = nullptr;
        if (precomputed_outer_response != nullptr &&
            precomputed_outer_response->changing_two_electron.size()) {
            two_electron_jet = precomputed_outer_response->changing_two_electron;
            if (components.direct_core_response) {
                if (fixed_ri_two_electron)
                    two_electron_jet += *fixed_ri_two_electron;
                else
                    two_electron_jet +=
                        integrals::apply_ri_packed_active_two_electron_adjoint_hessian_vector(
                            accepted_point_context_->packed_active_two_electron_gradient,
                            *accepted_ri_two_electron_cache_,
                            delta_dense_active_coefficients,
                            ri_directional_active_pair_factors);
                two_electron_jet_consumed = true;
            }
            two_electron_jet = integrals::active_frame_pullback(
                two_electron_jet,
                accepted_point_context_->prepared_active_space.active_space_two_electron_result
                    .orbital_transform);
            two_electron_orbital_gradient = &two_electron_jet;
        } else if (components.direct_core_response) {
            xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::TwoElectronResponse);
            const auto start = std::chrono::steady_clock::now();
            if (accepted_ri_two_electron_cache_) {
                if (fixed_ri_two_electron) {
                    // The fixed term already consumed the same L*D as the AO response.
                    // Only the changing cotangent remains after the selected CI sweep.
                    two_electron_jet =
                        *fixed_ri_two_electron +
                        integrals::backpropagate_ri_packed_active_two_electron_gradient(
                            directional_active_space_gradient.packed_active_two_electron_gradient,
                            *accepted_ri_two_electron_cache_);
                } else {
                    two_electron_jet = integrals::apply_ri_active_two_electron_jet(
                        accepted_point_context_->packed_active_two_electron_gradient,
                        *accepted_ri_two_electron_cache_,
                        delta_dense_active_coefficients,
                        directional_active_space_gradient.packed_active_two_electron_gradient,
                        ri_directional_active_pair_factors);
                }
            } else {
                const auto* products = precomputed_directional_pair_products
                                           ? precomputed_directional_pair_products
                                           : &outer_response_integral_direction_workspace_
                                                  .two_electron.directional_pair_products;
                integrals::apply_exact_active_two_electron_jet(
                    accepted_exact_two_electron_cache_,
                    delta_dense_active_coefficients,
                    directional_active_space_gradient.packed_active_two_electron_gradient,
                    current_input_->ao_integral_input,
                    &accepted_exact_two_electron_apply_workspace_,
                    &two_electron_jet,
                    products->size() ? products : nullptr);
            }
            two_electron_jet = integrals::active_frame_pullback(
                two_electron_jet,
                accepted_point_context_->prepared_active_space.active_space_two_electron_result
                    .orbital_transform);
            two_electron_orbital_gradient = &two_electron_jet;
            two_electron_jet_consumed = true;
            apply_timing_totals_.active_two_electron_wall_time_seconds +=
                detail::exact_hvp_elapsed_seconds(start);
        }

        const auto orbital_pullback_start_time = std::chrono::steady_clock::now();
        backward_phase.stop();
        xmvb::profiling::ScopedPhase pullback_phase(xmvb::profiling::Phase::OrbitalPullback);
        const std::vector<double> outer_response_orbital_value_gradient =
            build_orbital_value_gradient_from_active_space_gradient_direction(
                *current_input_,
                *accepted_point_context_,
                directional_active_space_gradient,
                *orbital_preparation_cache,
                accepted_ri_two_electron_cache_.has_value() ? nullptr
                                                            : &accepted_exact_two_electron_cache_,
                accepted_ri_two_electron_cache_.has_value() ? &*accepted_ri_two_electron_cache_
                                                            : nullptr,
                accepted_ri_factorization_,
                &outer_response_symmetric_active_overlap_gradient_workspace_,
                &outer_response_symmetric_active_one_electron_gradient_workspace_,
                two_electron_orbital_gradient);
        add_orbital_value_gradient_in_place(&combined_core_orbital_value_gradient,
                                            outer_response_orbital_value_gradient,
                                            "outer-response");
        apply_timing_totals_.outer_response_orbital_pullback_wall_time_seconds +=
            detail::exact_hvp_elapsed_seconds(orbital_pullback_start_time);

        apply_timing_totals_.outer_response_wall_time_seconds +=
            detail::exact_hvp_elapsed_seconds(outer_response_start_time);
    }
    if (components.direct_core_response) {
        Eigen::MatrixXd delta_auxiliary_active_gradient =
            basis_overlap * orbital_preparation_directional_result.delta_active_auxiliary_orbitals *
            accepted_sso_gradient_symmetric_;
        delta_auxiliary_active_gradient.noalias() +=
            delta_ao_effective_h1e_times_active_auxiliary_orbitals *
            accepted_hho_projection_weight_;
        delta_auxiliary_active_gradient.noalias() +=
            ao_effective_h1e * delta_active_times_hho_symmetric;
        if (!two_electron_jet_consumed) {
            const auto active_two_electron_start_time = std::chrono::steady_clock::now();
            xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::TwoElectronResponse);
            if (accepted_ri_two_electron_cache_.has_value()) {
                if (fixed_ri_two_electron) {
                    accepted_exact_two_electron_apply_workspace_.dense_active_gradient_direction =
                        *fixed_ri_two_electron;
                } else {
                    accepted_exact_two_electron_apply_workspace_.dense_active_gradient_direction =
                        integrals::apply_ri_packed_active_two_electron_adjoint_hessian_vector(
                            accepted_point_context_->packed_active_two_electron_gradient,
                            *accepted_ri_two_electron_cache_,
                            delta_dense_active_coefficients,
                            ri_directional_active_pair_factors);
                }
            } else if (compute_outer_response &&
                       (precomputed_directional_pair_products != nullptr
                            ? precomputed_directional_pair_products
                            : &outer_response_integral_direction_workspace_.two_electron
                                   .directional_pair_products)
                               ->size() > 0) {
                xmvb::integrals::
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
                xmvb::integrals::apply_exact_packed_active_two_electron_adjoint_hessian_vector(
                    accepted_exact_two_electron_cache_,
                    delta_dense_active_coefficients,
                    current_input_->ao_integral_input,
                    &accepted_exact_two_electron_apply_workspace_,
                    &accepted_exact_two_electron_apply_workspace_.dense_active_gradient_direction);
            }
            apply_timing_totals_.active_two_electron_wall_time_seconds +=
                detail::exact_hvp_elapsed_seconds(active_two_electron_start_time);
            delta_auxiliary_active_gradient.noalias() += integrals::active_frame_pullback(
                accepted_exact_two_electron_apply_workspace_.dense_active_gradient_direction,
                accepted_point_context_->prepared_active_space.active_space_two_electron_result
                    .orbital_transform);
        }

        const auto orbital_backprop_start_time = std::chrono::steady_clock::now();
        const std::vector<double> orbital_value_gradient =
            accepted_ri_factorization_ != nullptr
                ? backpropagate_active_space_orbital_gradient(
                      delta_auxiliary_active_gradient,
                      InactiveDensityProjection{inactive_density_projection},
                      current_input_->orbital_preparation_input,
                      accepted_point_context_->prepared_active_space.orbital_result,
                      *orbital_preparation_cache)
                : backpropagate_active_space_orbital_gradient(
                      delta_auxiliary_active_gradient,
                      total_inactive_density_direction,
                      current_input_->orbital_preparation_input,
                      accepted_point_context_->prepared_active_space.orbital_result,
                      *orbital_preparation_cache);
        add_orbital_value_gradient_in_place(
            &combined_core_orbital_value_gradient, orbital_value_gradient, "direct-core");
        apply_timing_totals_.orbital_backprop_wall_time_seconds +=
            detail::exact_hvp_elapsed_seconds(orbital_backprop_start_time);
    }

    if (components.fixed_upstream_pullback &&
        accepted_point_context_->total_active_auxiliary_gradient.rows() == n_basis_functions &&
        accepted_point_context_->total_active_auxiliary_gradient.cols() == n_active_orbitals &&
        accepted_point_context_->total_inactive_density_gradient.size() == ao_matrix_size) {
        const auto fixed_upstream_pullback_start_time = std::chrono::steady_clock::now();
        xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::FixedPullback);
        const std::vector<double> fixed_upstream_orbital_value_gradient =
            apply_fixed_upstream_orbital_pullback_direction(
                current_input_->orbital_preparation_input,
                dense_orbital_tangent_context,
                accepted_point_context_->total_active_auxiliary_gradient,
                orbital_preparation_directional_result.basis_overlap_times_delta_active_orbitals,
                accepted_point_context_->total_inactive_density_gradient,
                input_retract_tangent,
                *orbital_preparation_cache);
        add_orbital_value_gradient_in_place(&combined_core_orbital_value_gradient,
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
    xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::StructureAdjoint);
    const int n_structures = accepted_point_context_->n_structures;
    const int n_states = static_cast<int>(accepted_point_context_->selected_state_indices.size());
    if (coefficient_response.rows() != n_structures || coefficient_response.cols() != n_states ||
        state_multipliers.rows() != n_states || state_multipliers.cols() != n_states ||
        !coefficient_response.allFinite() || !state_multipliers.allFinite()) {
        throw std::invalid_argument(
            "structure-response adjoint has incompatible dimensions or values");
    }
    if (accepted_point_context_->normalized_state_weights.size() !=
        static_cast<std::size_t>(n_states)) {
        throw std::logic_error("structure-response adjoint is missing selected-state weights");
    }
    if (!supports_analytic_core_model() || accepted_orbital_preparation_cache_ == nullptr) {
        throw std::logic_error("structure-response adjoint requires the analytic orbital pullback");
    }

    const int n_active = current_input_->orbital_preparation_input.n_active_orbitals;
    const StructureAction* structure_action =
        outer_response_context().selected_state_eigen_response_operator.structure_action;
    if (structure_action == nullptr) {
        throw std::logic_error("structure-response adjoint requires a structure action");
    }

    const auto active_gradient_start = std::chrono::steady_clock::now();
    ActiveSpaceGradientDirection active_gradient;
    if (structure_action->supports_integral_direction()) {
        if (!accepted_point_context_->structure_adjoint_state.has_value()) {
            accepted_point_context_->structure_adjoint_state =
                structure_action->prepare_active_adjoint(
                    accepted_point_context_->selected_state_eigenvectors,
                    accepted_point_context_->normalized_state_weights,
                    accepted_point_context_->selected_state_energies);
        }
        active_gradient =
            make_active_gradient_direction(structure_action->active_integral_response_adjoint(
                                               *accepted_point_context_->structure_adjoint_state,
                                               coefficient_response,
                                               state_multipliers),
                                           n_active);
    } else {
        const auto selected_state_start = std::chrono::steady_clock::now();
        const auto directional_selected_states =
            build_selected_state_determinant_matrices_from_selected_columns(
                current_input_->structure_data,
                coefficient_response,
                accepted_point_context_->selected_state_indices,
                accepted_point_context_->normalized_state_weights,
                accepted_point_context_->same_spin_pair_cache);
        apply_timing_totals_.outer_response_selected_state_rebuild_wall_time_seconds +=
            detail::exact_hvp_elapsed_seconds(selected_state_start);
        active_gradient = make_zero_active_space_gradient_direction(n_active);
        add_selected_subspace_response_to_active_space_gradient(*current_input_,
                                                                *accepted_point_context_,
                                                                directional_selected_states,
                                                                state_multipliers,
                                                                &active_gradient);
    }
    validate_outer_response_active_gradient(active_gradient);
    const double active_gradient_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - active_gradient_start)
            .count();
    apply_timing_totals_.outer_response_active_gradient_wall_time_seconds +=
        active_gradient_seconds;
    apply_timing_totals_.outer_response_structure_active_gradient_wall_time_seconds +=
        active_gradient_seconds;

    return pullback_structure_response(active_gradient);
}

Eigen::VectorXd ExactHvpOperator::State::pullback_structure_response(
    const ActiveSpaceGradientDirection& active_gradient) const {
    const auto orbital_pullback_start = std::chrono::steady_clock::now();
    const std::vector<double> orbital_value_gradient =
        build_orbital_value_gradient_from_active_space_gradient_direction(
            *current_input_,
            *accepted_point_context_,
            active_gradient,
            *accepted_orbital_preparation_cache_,
            accepted_ri_two_electron_cache_.has_value() ? nullptr
                                                        : &accepted_exact_two_electron_cache_,
            accepted_ri_two_electron_cache_.has_value() ? &*accepted_ri_two_electron_cache_
                                                        : nullptr,
            accepted_ri_factorization_,
            &outer_response_symmetric_active_overlap_gradient_workspace_,
            &outer_response_symmetric_active_one_electron_gradient_workspace_);
    apply_timing_totals_.outer_response_orbital_pullback_wall_time_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - orbital_pullback_start)
            .count();
    const Eigen::VectorXd packed = parameter_view_.gather_from_full(orbital_value_gradient);
    return nonredundant_space_->project_reduced_gradient(packed);
}

} // namespace xmvb::vb
