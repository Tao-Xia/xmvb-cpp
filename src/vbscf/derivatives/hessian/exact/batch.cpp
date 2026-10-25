#include "vbscf/derivatives/hessian/exact/operator.hpp"
#include "core/timing.hpp"

#include "vbscf/derivatives/hessian/exact/ao_one_electron_internal.hpp"
#include "vbscf/derivatives/hessian/exact/apply_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "linalg/eigen_response.hpp"
#include "parallel/openmp.hpp"
#include "vbscf/derivatives/hessian/coupled/coupling.hpp"
#include "vbscf/derivatives/hessian/responses/active_space/local_tile_accumulator_internal.hpp"
#include "vbscf/derivatives/hessian/responses/orbital/preparation.hpp"
#include "vbscf/derivatives/hessian/responses/structure/directional.hpp"
#include "integrals/response/directional.hpp"
#include "integrals/response/ri.hpp"
#include "integrals/response/orbital_pair.hpp"
#include "integrals/ao/exact.hpp"
#include "integrals/ao/ri_operator.hpp"

namespace xmvb::vb {
namespace {

constexpr std::size_t kOrbitalCouplingBatchWorkspaceBytes = 256ULL * 1024ULL * 1024ULL;
constexpr Eigen::Index kMaximumOrbitalCouplingBatchColumns = 8;

Eigen::Index bounded_orbital_coupling_width(Eigen::Index requested,
                                            const OrbitalPreparationInput& input,
                                            int n_structures,
                                            int n_states) {
    const std::size_t n_bf = input.n_basis_functions;
    const std::size_t n_active = input.n_active_orbitals;
    const std::size_t ao_matrix = n_bf * n_bf;
    const std::size_t ao_pairs = n_bf * (n_bf + 1ULL) / 2ULL;
    const std::size_t active_pairs = n_active * (n_active + 1ULL) / 2ULL;
    const std::size_t packed_active_two_electron = active_pairs * (active_pairs + 1ULL) / 2ULL;
    const std::size_t workers = effective_openmp_thread_count();
    // Per column: two inputs, two outputs, two shared interleaved channels,
    // one retained density direction, and two AO reductions per worker.
    // Include copied tangent contexts, active directions, exact pair images,
    // and selected-state forcing. RI uses less. This limits the column-dependent
    // workspace, not total process RSS or resident accepted-point caches.
    // A single column is the irreducible working-set floor when it exceeds 256 MiB.
    const std::size_t elements_per_column =
        (7ULL + 2ULL * workers) * ao_matrix + 3ULL * n_bf * input.n_orbitals +
        6ULL * n_bf * n_active + 2ULL * ao_pairs * active_pairs + 2ULL * workers +
        4ULL * packed_active_two_electron + 2ULL * n_active * n_active +
        static_cast<std::size_t>(n_structures) * static_cast<std::size_t>(n_states);
    const std::size_t bytes_per_column =
        std::max<std::size_t>(sizeof(double), elements_per_column * sizeof(double));
    const Eigen::Index budget_width = static_cast<Eigen::Index>(
        std::max<std::size_t>(1ULL, kOrbitalCouplingBatchWorkspaceBytes / bytes_per_column));
    return std::max<Eigen::Index>(
        1, std::min({requested, kMaximumOrbitalCouplingBatchColumns, budget_width}));
}

} // namespace

Eigen::MatrixXd
ExactHvpOperator::apply_reduced_batch(const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions,
                                      HvpComponents components) const {
    return state_->apply_reduced_batch(reduced_directions, components);
}

OrbitalCouplingBlockAction ExactHvpOperator::apply_orbital_coupling_batch(
    const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions) const {
    return state_->apply_orbital_coupling_batch(reduced_directions);
}

Eigen::MatrixXd ExactHvpOperator::State::apply_reduced_batch(
    const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions, HvpComponents components) const {
    if (reduced_directions.rows() != nonredundant_space_->reduced_size() ||
        !reduced_directions.allFinite()) {
        throw std::invalid_argument("HVP block has incompatible dimensions or values");
    }
    const Eigen::Index chunk_width = bounded_orbital_coupling_width(
        reduced_directions.cols(),
        current_input_->orbital_preparation_input,
        accepted_point_context_->n_structures,
        static_cast<int>(accepted_point_context_->selected_state_indices.size()));
    Eigen::MatrixXd result(reduced_directions.rows(), reduced_directions.cols());
    for (Eigen::Index first = 0; first < reduced_directions.cols(); first += chunk_width) {
        const Eigen::Index width = std::min(chunk_width, reduced_directions.cols() - first);
        result.middleCols(first, width) = apply_reduced_batch_impl(
            reduced_directions.middleCols(first, width), components, nullptr);
    }
    return result;
}

OrbitalCouplingBlockAction ExactHvpOperator::State::apply_orbital_coupling_batch(
    const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions) const {
    if (reduced_directions.rows() != nonredundant_space_->reduced_size() ||
        reduced_directions.cols() <= 0 || !reduced_directions.allFinite()) {
        throw std::invalid_argument("orbital coupling block has incompatible dimensions or values");
    }
    ++apply_timing_totals_.orbital_coupling_batch_count;
    const Eigen::Index n_states =
        static_cast<Eigen::Index>(accepted_point_context_->selected_state_indices.size());
    const Eigen::Index chunk_width =
        bounded_orbital_coupling_width(reduced_directions.cols(),
                                       current_input_->orbital_preparation_input,
                                       accepted_point_context_->n_structures,
                                       static_cast<int>(n_states));

    OrbitalCouplingBlockAction result;
    result.orbital_hessian.resize(reduced_directions.rows(), reduced_directions.cols());
    result.scaled_structure_forcing.resize(static_cast<std::size_t>(reduced_directions.cols()));
    for (Eigen::Index first = 0; first < reduced_directions.cols(); first += chunk_width) {
        const Eigen::Index width = std::min(chunk_width, reduced_directions.cols() - first);
        OrbitalCouplingBlockAction chunk;
        apply_reduced_batch_impl(reduced_directions.middleCols(first, width),
                                 {.direct_core_response = true,
                                  .fixed_upstream_pullback = true,
                                  .local_active_response = true,
                                  .structure_response = false},
                                 &chunk);
        result.orbital_hessian.middleCols(first, width) = chunk.orbital_hessian;
        for (Eigen::Index column = 0; column < width; ++column) {
            result.scaled_structure_forcing[static_cast<std::size_t>(first + column)] =
                std::move(chunk.scaled_structure_forcing[static_cast<std::size_t>(column)]);
        }
        ++apply_timing_totals_.orbital_coupling_batch_chunk_count;
        apply_timing_totals_.max_orbital_coupling_batch_width = std::max(
            apply_timing_totals_.max_orbital_coupling_batch_width, static_cast<std::size_t>(width));
    }
    return result;
}

Eigen::MatrixXd ExactHvpOperator::State::apply_reduced_batch_impl(
    const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions,
    HvpComponents components,
    OrbitalCouplingBlockAction* coupling_output) const {
    ++apply_timing_totals_.batch_apply_count;
    Eigen::MatrixXd responses(reduced_directions.rows(), reduced_directions.cols());
    if (coupling_output != nullptr) {
        coupling_output->orbital_hessian.resize(reduced_directions.rows(),
                                                reduced_directions.cols());
        coupling_output->scaled_structure_forcing.resize(
            static_cast<std::size_t>(reduced_directions.cols()));
    }
    if (reduced_directions.cols() <= 1) {
        for (Eigen::Index column = 0; column < reduced_directions.cols(); ++column) {
            if (coupling_output == nullptr) {
                responses.col(column) = apply_reduced(reduced_directions.col(column), components);
            } else {
                OrbitalCouplingAction action =
                    apply_orbital_coupling(reduced_directions.col(column), nullptr);
                responses.col(column) = action.orbital_hessian;
                coupling_output->orbital_hessian.col(column) = action.orbital_hessian;
                coupling_output->scaled_structure_forcing[static_cast<std::size_t>(column)] =
                    std::move(action.scaled_structure_forcing);
            }
        }
        return responses;
    }
    if (!supports_analytic_core_model()) {
        throw std::runtime_error(
            "exact_ctx analytic core HVP is unavailable for the current accepted point");
    }
    const int n_basis_functions = current_input_->orbital_preparation_input.n_basis_functions;
    const int n_active_orbitals = current_input_->orbital_preparation_input.n_active_orbitals;
    const Eigen::Index ao_matrix_size =
        static_cast<Eigen::Index>(n_basis_functions) * n_basis_functions;
    const Eigen::Index n_directions = reduced_directions.cols();
    Eigen::MatrixXd inactive_density_columns, symmetrized_pullback_columns;
    if (!accepted_ri_factorization_) {
        inactive_density_columns.resize(ao_matrix_size, n_directions);
        symmetrized_pullback_columns.resize(ao_matrix_size, n_directions);
    }
    Eigen::MatrixXd weighted_active_columns(n_basis_functions * n_active_orbitals, n_directions);
    std::vector<PrecomputedDirection> precomputed_directions;
    precomputed_directions.reserve(n_directions);
    std::vector<Eigen::MatrixXd> dense_active_directions;
    dense_active_directions.reserve(n_directions);
    for (Eigen::Index column = 0; column < n_directions; ++column) {
        PrecomputedDirection precomputed =
            prepare_direction(nonredundant_space_->expand_step(reduced_directions.col(column)));
        const auto& directional_result = precomputed.orbital_preparation_directional_result;
        dense_active_directions.push_back(precomputed.integral_tangent);
        const Eigen::MatrixXd weighted_active =
            directional_result.delta_active_auxiliary_orbitals * accepted_hho_gradient_symmetric_;
        weighted_active_columns.col(column) =
            Eigen::Map<const Eigen::VectorXd>(weighted_active.data(), weighted_active.size());
        if (!accepted_ri_factorization_) {
            Eigen::MatrixXd pullback_source = directional_result.delta_inactive_density;
            pullback_source.noalias() +=
                weighted_active * accepted_active_auxiliary_orbitals_.transpose();
            inactive_density_columns.col(column) = Eigen::Map<const Eigen::VectorXd>(
                directional_result.delta_inactive_density.data(), ao_matrix_size);
            const Eigen::MatrixXd symmetric = pullback_source + pullback_source.transpose();
            symmetrized_pullback_columns.col(column) =
                Eigen::Map<const Eigen::VectorXd>(symmetric.data(), ao_matrix_size);
        }
        precomputed_directions.push_back(std::move(precomputed));
    }

    const auto batch_h1e_start_time = std::chrono::steady_clock::now();
    Eigen::MatrixXd combined_ao_columns;
    Eigen::MatrixXd active_projection_columns;
    std::vector<Eigen::MatrixXd> ri_active_pair_factor_directions;
    if (accepted_ri_factorization_ != nullptr) {
        const int n_inactive = (current_input_->orbital_preparation_input.n_total_electrons -
                                current_input_->orbital_preparation_input.n_active_electrons) /
                               2;
        const Eigen::Index projection_size = n_basis_functions * n_inactive;
        combined_ao_columns.resize(projection_size, n_directions);
        active_projection_columns.resize(n_basis_functions * n_active_orbitals, n_directions);
        auto ri_responses = ri_ao_responses(
            precomputed_directions, weighted_active_columns, components.direct_core_response);
        for (Eigen::Index column = 0; column < n_directions; ++column) {
            auto& ri = ri_responses[column];
            combined_ao_columns.col(column) =
                Eigen::Map<const Eigen::VectorXd>(ri.inactive.data(), projection_size);
            active_projection_columns.col(column) = Eigen::Map<const Eigen::VectorXd>(
                ri.active.data(), n_basis_functions * n_active_orbitals);
            if (ri.fixed_two_electron.size()) {
                precomputed_directions[column].ri_fixed_two_electron =
                    std::move(ri.fixed_two_electron);
                ri_active_pair_factor_directions.push_back(std::move(ri.factor_direction));
            }
        }
    } else {
        const auto& frame = accepted_point_context_->prepared_active_space
                                .active_space_two_electron_result.orbital_transform;
        if (frame.size()) {
            for (Eigen::Index column = 0; column < n_directions; ++column) {
                Eigen::Map<Eigen::MatrixXd> weights(weighted_active_columns.col(column).data(),
                                                    n_basis_functions,
                                                    n_active_orbitals);
                weights = (weights * frame.transpose()).eval();
            }
        }
        detail::apply_exact_ao_response(inactive_density_columns,
                                        symmetrized_pullback_columns,
                                        weighted_active_columns,
                                        accepted_exact_two_electron_cache_,
                                        current_input_->ao_integral_input,
                                        &ao_h1e_batch_workspace_,
                                        &combined_ao_columns,
                                        &active_projection_columns);
    }
    const double batch_h1e_seconds = detail::exact_hvp_elapsed_seconds(batch_h1e_start_time);
    apply_timing_totals_.ao_effective_one_electron_fused_wall_time_seconds += batch_h1e_seconds;
    apply_timing_totals_.total_apply_wall_time_seconds += batch_h1e_seconds;

    Eigen::MatrixXd delta_packed_active_two_electron_columns;
    std::vector<xmvb::integrals::ExactCtxPairMatrix> directional_pair_products;
    const bool build_orbital_coupling = coupling_output != nullptr;
    const bool compute_outer_response =
        components.local_active_response || components.structure_response || build_orbital_coupling;
    const bool direct_active_gradient =
        compute_outer_response &&
        outer_response_context()
            .selected_state_eigen_response_operator.structure_action->supports_integral_direction();
    // The fixed RI jet keeps delta B; CI can form delta G from the smaller
    // resident three-active image without transforming auxiliary factors again.
    const bool ri_orbital_pair_direction =
        direct_active_gradient && accepted_ri_two_electron_cache_ &&
        integrals::ri_uses_orbital_pairs(*accepted_ri_two_electron_cache_);
    const bool packed_outer_two_electron_required =
        !accepted_ri_two_electron_cache_ || ri_orbital_pair_direction;
    if (accepted_ri_two_electron_cache_.has_value()) {
        const auto batch_active_two_electron_start_time = std::chrono::steady_clock::now();
        if (ri_active_pair_factor_directions.empty() &&
            (components.direct_core_response ||
             (compute_outer_response && !ri_orbital_pair_direction))) {
            ri_active_pair_factor_directions =
                integrals::compute_ri_active_pair_factor_directional_derivative_batch(
                    *accepted_ri_two_electron_cache_, dense_active_directions);
        }
        if (ri_orbital_pair_direction) {
            profiling::ScopedPhase phase(profiling::Phase::ActiveDirection);
            const auto& cache = *accepted_ri_two_electron_cache_;
            const Eigen::Index pairs = cache.active_pair_first_indices.size();
            delta_packed_active_two_electron_columns.resize(pairs * (pairs + 1) / 2, n_directions);
            for (Eigen::Index column = 0; column < n_directions; ++column)
                delta_packed_active_two_electron_columns.col(column) =
                    integrals::packed_orbital_pair_direction(*cache.orbital_pair_products,
                                                             dense_active_directions[column],
                                                             cache.active_pair_first_indices,
                                                             cache.active_pair_second_indices,
                                                             cache.n_active_orbitals);
        }
        const double batch_active_two_electron_seconds =
            detail::exact_hvp_elapsed_seconds(batch_active_two_electron_start_time);
        apply_timing_totals_.active_two_electron_wall_time_seconds +=
            batch_active_two_electron_seconds;
        apply_timing_totals_.total_apply_wall_time_seconds += batch_active_two_electron_seconds;
    } else if (compute_outer_response) {
        const auto batch_active_two_electron_start_time = std::chrono::steady_clock::now();
        delta_packed_active_two_electron_columns = xmvb::integrals::
            compute_exact_packed_active_two_electron_integral_directional_derivative_batch(
                accepted_exact_two_electron_cache_,
                dense_active_directions,
                current_input_->ao_integral_input,
                components.direct_core_response ? &directional_pair_products : nullptr,
                nullptr);
        const double batch_active_two_electron_seconds =
            detail::exact_hvp_elapsed_seconds(batch_active_two_electron_start_time);
        apply_timing_totals_.outer_response_active_space_integrals_wall_time_seconds +=
            batch_active_two_electron_seconds;
        apply_timing_totals_.outer_response_wall_time_seconds += batch_active_two_electron_seconds;
        apply_timing_totals_.total_apply_wall_time_seconds += batch_active_two_electron_seconds;
    }

    if (compute_outer_response) {
        const auto outer_batch_start = std::chrono::steady_clock::now();
        const int n_selected_states =
            static_cast<int>(accepted_point_context_->selected_state_indices.size());
        const int n_structures = accepted_point_context_->n_structures;
        Eigen::MatrixXd delta_hamiltonian_selected;
        Eigen::MatrixXd delta_overlap_selected;
        if (components.structure_response) {
            delta_hamiltonian_selected.resize(n_structures, n_directions * n_selected_states);
            delta_overlap_selected.resize(n_structures, n_directions * n_selected_states);
        }
        double integral_seconds = 0.0;
        double structure_seconds = 0.0;
        std::vector<ActiveSpaceIntegralDirectionView> integral_directions;
        integral_directions.reserve(static_cast<std::size_t>(n_directions));
        for (Eigen::Index column = 0; column < n_directions; ++column) {
            PrecomputedOuterResponse& outer =
                precomputed_directions[column].outer_response.emplace();
            const Eigen::Map<const Eigen::MatrixXd> delta_h1e_times_active(
                active_projection_columns.col(column).data(), n_basis_functions, n_active_orbitals);

            const auto integral_start = std::chrono::steady_clock::now();
            Eigen::VectorXd delta_packed_two_electron;
            if (packed_outer_two_electron_required) {
                delta_packed_two_electron = delta_packed_active_two_electron_columns.col(column);
            }
            integral_directions.emplace_back(build_active_integral_direction(
                precomputed_directions[column],
                delta_h1e_times_active,
                packed_outer_two_electron_required ? &delta_packed_two_electron : nullptr,
                &outer.integral_direction,
                packed_outer_two_electron_required));
            integral_seconds += detail::exact_hvp_elapsed_seconds(integral_start);
        }

        const auto structure_start = std::chrono::steady_clock::now();
        for (Eigen::Index column = 0; column < n_directions; ++column) {
            PrecomputedOuterResponse& outer = *precomputed_directions[column].outer_response;
            std::unique_ptr<detail::LocalActiveSpaceTileAccumulator> local_active_accumulator;
            if (!direct_active_gradient && components.local_active_response) {
                local_active_accumulator =
                    std::make_unique<detail::LocalActiveSpaceTileAccumulator>(
                        *current_input_,
                        *accepted_point_context_,
                        integral_directions[static_cast<std::size_t>(column)]);
            }
            const auto consume_local_active_tile =
                [&](bool alpha_channel,
                    bool beta_channel,
                    const AcceptedSpinPairTile& accepted,
                    const detail::SameSpinDirectionalPairTileView& same_spin,
                    const detail::DirectionalOppositeSpinPairTileView* opposite_spin) {
                    if (local_active_accumulator) {
                        local_active_accumulator->consume(
                            alpha_channel, beta_channel, accepted, same_spin, opposite_spin);
                    }
                };
            bool pair_tiles_consumed = false;
            if (components.structure_response || build_orbital_coupling) {
                SelectedStateDirectionalStructureImages images =
                    direct_active_gradient
                        ? build_selected_structure_direction(
                              outer_response_context(),
                              integral_directions[static_cast<std::size_t>(column)],
                              accepted_ri_two_electron_cache_.has_value() &&
                                      !ri_orbital_pair_direction
                                  ? accepted_ri_two_electron_cache_->accepted_active_pair_factors
                                  : nullptr,
                              accepted_ri_two_electron_cache_.has_value() &&
                                      !ri_orbital_pair_direction
                                  ? &ri_active_pair_factor_directions[static_cast<std::size_t>(
                                        column)]
                                  : nullptr)
                        : build_selected_structure_direction_from_pair_tiles(
                              outer_response_context(),
                              integral_directions[static_cast<std::size_t>(column)],
                              accepted_ri_two_electron_cache_.has_value()
                                  ? accepted_ri_two_electron_cache_->accepted_active_pair_factors
                                  : nullptr,
                              accepted_ri_two_electron_cache_.has_value()
                                  ? &ri_active_pair_factor_directions[static_cast<std::size_t>(
                                        column)]
                                  : nullptr,
                              consume_local_active_tile);
                pair_tiles_consumed = !direct_active_gradient;
                outer.direct_ci_direction = std::move(images.direct_ci_direction);
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
                            images.delta_overlap_selected);
                    outer.gauge_coupling = scale_equal_weight_structure_coupling(
                        coupling, accepted_point_context_->normalized_state_weights);
                    coupling_output->scaled_structure_forcing[static_cast<std::size_t>(column)] =
                        outer.gauge_coupling->horizontal_forcing.scaled_coefficients;
                } else {
                    const int first = static_cast<int>(column) * n_selected_states;
                    delta_hamiltonian_selected.middleCols(first, n_selected_states) =
                        images.delta_hamiltonian_selected;
                    delta_overlap_selected.middleCols(first, n_selected_states) =
                        images.delta_overlap_selected;
                }
            }
            if (local_active_accumulator && !pair_tiles_consumed) {
                detail::stream_directional_pair_tiles(
                    accepted_point_context_->same_spin_pair_cache,
                    n_active_orbitals,
                    accepted_point_context_->prepared_active_space.orbital_result
                        .active_orbital_overlap_matrix,
                    accepted_point_context_->prepared_active_space.active_space_one_electron_result
                        .h1e_act,
                    accepted_point_context_->prepared_active_space.active_space_two_electron_result,
                    integral_directions[static_cast<std::size_t>(column)],
                    accepted_ri_two_electron_cache_.has_value()
                        ? accepted_ri_two_electron_cache_->accepted_active_pair_factors
                        : nullptr,
                    accepted_ri_two_electron_cache_.has_value()
                        ? &ri_active_pair_factor_directions[static_cast<std::size_t>(column)]
                        : nullptr,
                    true,
                    consume_local_active_tile,
                    n_selected_states);
            }
            if (local_active_accumulator) {
                outer.local_active_gradient = local_active_accumulator->finish();
            }
            // Retain exact tangents until their local adjoint consumer finishes.
            // RI directions never materialize this buffer.
        }
        structure_seconds += detail::exact_hvp_elapsed_seconds(structure_start);
        apply_timing_totals_.outer_response_active_space_integrals_wall_time_seconds +=
            integral_seconds;
        apply_timing_totals_.outer_response_structure_matrices_wall_time_seconds +=
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
                auto& response =
                    precomputed_directions[column].outer_response->selected_state_response;
                response.delta_selected_eigenvector_matrix =
                    block_response.delta_selected_eigenvector_matrix.middleCols(first,
                                                                                n_selected_states);
                response.selected_matrix_responses = {
                    block_response.selected_matrix_responses[column]};
                response.linear_iterations.assign(block_response.linear_iterations.begin() + first,
                                                  block_response.linear_iterations.begin() + first +
                                                      n_selected_states);
                if (column == 0) {
                    response.block_actions = block_response.block_actions;
                    response.max_relative_residual = block_response.max_relative_residual;
                }
            }
        }
        const double outer_batch_seconds = detail::exact_hvp_elapsed_seconds(outer_batch_start);
        apply_timing_totals_.outer_response_wall_time_seconds += outer_batch_seconds;
        apply_timing_totals_.total_apply_wall_time_seconds += outer_batch_seconds;
    }

    if (accepted_ri_two_electron_cache_ && compute_outer_response) {
        const auto start = std::chrono::steady_clock::now();
        std::vector<const std::vector<double>*> cotangents;
        cotangents.reserve(n_directions);
        for (Eigen::Index column = 0; column < n_directions; ++column) {
            auto& outer = *precomputed_directions[column].outer_response;
            const auto* states =
                components.structure_response ? &outer.selected_state_response : nullptr;
            outer.active_gradient = active_response_gradient(
                {{outer.integral_direction.overlap,
                  outer.integral_direction.one_electron,
                  outer.integral_direction.packed_two_electron},
                 states,
                 outer.gauge_coupling ? &*outer.gauge_coupling : nullptr,
                 outer.direct_ci_direction ? &*outer.direct_ci_direction : nullptr,
                 !ri_orbital_pair_direction && !ri_active_pair_factor_directions.empty()
                     ? &ri_active_pair_factor_directions[column]
                     : nullptr,
                 outer.local_active_gradient ? &*outer.local_active_gradient : nullptr},
                components);
            cotangents.push_back(&outer.active_gradient->packed_active_two_electron_gradient);
        }
        {
            profiling::ScopedPhase phase(profiling::Phase::TwoElectronResponse);
            const auto pullback_start = std::chrono::steady_clock::now();
            auto images = integrals::backpropagate_ri_packed_active_two_electron_gradient_batch(
                cotangents, *accepted_ri_two_electron_cache_);
            for (Eigen::Index column = 0; column < n_directions; ++column)
                precomputed_directions[column].outer_response->changing_two_electron =
                    std::move(images[column]);
            apply_timing_totals_.active_two_electron_wall_time_seconds +=
                detail::exact_hvp_elapsed_seconds(pullback_start);
        }
        const double seconds = detail::exact_hvp_elapsed_seconds(start);
        apply_timing_totals_.outer_response_wall_time_seconds += seconds;
        apply_timing_totals_.total_apply_wall_time_seconds += seconds;
    }

    for (Eigen::Index column = 0; column < reduced_directions.cols(); ++column) {
        const Eigen::VectorXd combined_ao = combined_ao_columns.col(column);
        const Eigen::MatrixXd active_projection = Eigen::Map<const Eigen::MatrixXd>(
            active_projection_columns.col(column).data(), n_basis_functions, n_active_orbitals);
        Eigen::VectorXd delta_packed_active_two_electron;
        if (compute_outer_response && packed_outer_two_electron_required)
            delta_packed_active_two_electron = delta_packed_active_two_electron_columns.col(column);
        OrbitalCouplingAction column_coupling;
        const Eigen::VectorXd response = apply_reduced_impl(
            reduced_directions.col(column),
            components,
            &combined_ao,
            &active_projection,
            compute_outer_response && packed_outer_two_electron_required
                ? &delta_packed_active_two_electron
                : nullptr,
            !ri_active_pair_factor_directions.empty()
                ? &ri_active_pair_factor_directions[static_cast<std::size_t>(column)]
                : nullptr,
            components.direct_core_response && compute_outer_response &&
                    !accepted_ri_two_electron_cache_.has_value()
                ? &directional_pair_products[column]
                : nullptr,
            &precomputed_directions[column],
            build_orbital_coupling ? &column_coupling : nullptr);
        responses.col(column) = response;
        if (build_orbital_coupling) {
            coupling_output->orbital_hessian.col(column) = response;
            coupling_output->scaled_structure_forcing[static_cast<std::size_t>(column)] =
                std::move(column_coupling.scaled_structure_forcing);
        }
        if (compute_outer_response) {
            precomputed_directions[column]
                .outer_response->integral_direction.packed_two_electron.clear();
        }
    }
    return responses;
}

} // namespace xmvb::vb
