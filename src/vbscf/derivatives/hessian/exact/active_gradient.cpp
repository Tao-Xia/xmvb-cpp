#include "vbscf/derivatives/hessian/exact/apply_internal.hpp"
#include "vbscf/derivatives/hessian/responses/active_space/local_tile_accumulator_internal.hpp"
#include "vbscf/structures/assembly/selected_coefficients.hpp"
#include "integrals/active/raw.hpp"
#include "core/pair_index.hpp"
#include "core/timing.hpp"

#include <stdexcept>

namespace xmvb::vb {
namespace detail {
ActiveSpaceGradientDirection
make_active_gradient_direction(const StructureActiveIntegralAdjoint& adjoint,
                               int n_active_orbitals) {
    const int n_pairs = xmvb::integrals::packed_active_pair_count(n_active_orbitals);
    if (adjoint.overlap.rows() != n_active_orbitals ||
        adjoint.overlap.cols() != n_active_orbitals ||
        adjoint.one_electron.rows() != n_active_orbitals ||
        adjoint.one_electron.cols() != n_active_orbitals || adjoint.pair_kernel.rows() != n_pairs ||
        adjoint.pair_kernel.cols() != n_pairs) {
        throw std::runtime_error("direct-CI active-gradient direction has inconsistent dimensions");
    }
    ActiveSpaceGradientDirection result =
        make_zero_active_space_gradient_direction(n_active_orbitals);
    result.active_orbital_overlap_gradient.assign(adjoint.overlap.data(),
                                                  adjoint.overlap.data() + adjoint.overlap.size());
    result.active_one_electron_gradient.assign(
        adjoint.one_electron.data(), adjoint.one_electron.data() + adjoint.one_electron.size());
    for (int column = 0; column < n_pairs; ++column) {
        for (int row = 0; row <= column; ++row) {
            result.packed_active_two_electron_gradient
                [xmvb::core::PairIndex::packed_pair_of_pairs_index(row, column)] =
                row == column ? adjoint.pair_kernel(row, column)
                              : adjoint.pair_kernel(row, column) + adjoint.pair_kernel(column, row);
        }
    }
    return result;
}

} // namespace detail
namespace {
using detail::make_active_gradient_direction;

/** @brief Add a known quadratic coefficient scaling without another pullback. */
void add_scaled(std::vector<double>& target, const std::vector<double>& source, double scale) {
    for (std::size_t i = 0; i < target.size(); ++i)
        target[i] += scale * source[i];
}

} // namespace

ActiveSpaceGradientDirection
ExactHvpOperator::State::active_response_gradient(const ActiveResponseView& view,
                                                  HvpComponents components) const {
    const int n_active_orbitals = current_input_->orbital_preparation_input.n_active_orbitals;
    const auto* accepted_structure_action =
        outer_response_context().selected_state_eigen_response_operator.structure_action;
    const bool direct_active_gradient = accepted_structure_action->supports_integral_direction();
    const auto* directional_selected_state_response = view.states;
    const auto* gauge_coupling = view.gauge;
    const auto& active_space_integral_direction = view.integrals;
    const auto* selected_ri_factor_direction = view.ri_direction;
    auto* local_active_tile_accumulator = view.local_tiles;
    Eigen::MatrixXd state_multipliers;
    if (directional_selected_state_response != nullptr) {
        state_multipliers = -directional_selected_state_response->selected_matrix_responses.front();
    } else if (gauge_coupling != nullptr) {
        state_multipliers = gauge_coupling->gauge_adjoint_multipliers;
    }

    const bool reuse_normalization = direct_active_gradient && gauge_coupling != nullptr &&
                                     gauge_coupling->gauge_frame.rows() == 1;
    Eigen::MatrixXd frame_response;
    const Eigen::MatrixXd* coefficient_response = nullptr;
    if (directional_selected_state_response != nullptr) {
        coefficient_response =
            &directional_selected_state_response->delta_selected_eigenvector_matrix;
    } else if (gauge_coupling != nullptr && !reuse_normalization) {
        frame_response.noalias() =
            accepted_point_context_->selected_state_eigenvectors * gauge_coupling->gauge_frame;
        coefficient_response = &frame_response;
    }

    std::optional<SelectedStateDeterminantMatrices> directional_selected_states;
    if (coefficient_response != nullptr && !direct_active_gradient) {
        const auto selected_state_rebuild_start_time = std::chrono::steady_clock::now();
        directional_selected_states.emplace(
            build_selected_state_determinant_matrices_from_selected_columns(
                current_input_->structure_data,
                *coefficient_response,
                accepted_point_context_->selected_state_indices,
                accepted_point_context_->normalized_state_weights,
                accepted_point_context_->same_spin_pair_cache));
        apply_timing_totals_.outer_response_selected_state_rebuild_wall_time_seconds +=
            detail::exact_hvp_elapsed_seconds(selected_state_rebuild_start_time);
    }

    const auto active_gradient_start_time = std::chrono::steady_clock::now();
    ActiveSpaceGradientDirection directional_active_space_gradient;
    if (direct_active_gradient) {
        if (!accepted_point_context_->structure_adjoint_state.has_value()) {
            accepted_point_context_->structure_adjoint_state =
                accepted_structure_action->prepare_active_adjoint(
                    accepted_point_context_->selected_state_eigenvectors,
                    accepted_point_context_->normalized_state_weights,
                    accepted_point_context_->selected_state_energies);
        }
        const StructureIntegralDirection* direct_ci_direction = view.direct_ci;
        directional_active_space_gradient = make_active_gradient_direction(
            accepted_structure_action->active_integral_adjoint_direction(
                *accepted_point_context_->structure_adjoint_state,
                coefficient_response,
                state_multipliers.size() ? &state_multipliers : nullptr,
                active_space_integral_direction.overlap,
                active_space_integral_direction.one_electron,
                active_space_integral_direction.packed_two_electron,
                direct_ci_direction,
                components.local_active_response,
                selected_ri_factor_direction),
            n_active_orbitals);
        if (reuse_normalization) {
            // Q_g=aC changes every fixed-energy active adjoint by exactly 2a.
            // The multiplier and the orbital-map derivative were kept above.
            const double scale = 2 * gauge_coupling->gauge_frame(0, 0);
            add_scaled(directional_active_space_gradient.active_orbital_overlap_gradient,
                       accepted_point_context_->active_orbital_overlap_gradient,
                       scale);
            add_scaled(directional_active_space_gradient.active_one_electron_gradient,
                       accepted_point_context_->active_one_electron_gradient,
                       scale);
            add_scaled(directional_active_space_gradient.packed_active_two_electron_gradient,
                       accepted_point_context_->packed_active_two_electron_gradient,
                       scale);
            ++apply_timing_totals_.normalization_adjoint_reuses;
        }
        const double direct_seconds = detail::exact_hvp_elapsed_seconds(active_gradient_start_time);
        if (components.local_active_response) {
            apply_timing_totals_.outer_response_local_active_gradient_wall_time_seconds +=
                direct_seconds;
        } else {
            apply_timing_totals_.outer_response_structure_active_gradient_wall_time_seconds +=
                direct_seconds;
        }
    } else {
        const auto local_active_gradient_start_time = std::chrono::steady_clock::now();
        if (components.local_active_response && view.local_gradient != nullptr) {
            directional_active_space_gradient = *view.local_gradient;
        } else if (components.local_active_response && local_active_tile_accumulator) {
            directional_active_space_gradient = local_active_tile_accumulator->finish();
        } else if (components.local_active_response) {
            throw std::logic_error("local active response is missing its pair-tile accumulation");
        } else {
            directional_active_space_gradient =
                make_zero_active_space_gradient_direction(n_active_orbitals);
        }
        apply_timing_totals_.outer_response_local_active_gradient_wall_time_seconds +=
            detail::exact_hvp_elapsed_seconds(local_active_gradient_start_time);

        if (components.local_active_response) {
            directional_active_space_gradient.packed_active_two_electron_gradient =
                xmvb::integrals::materialize_framed_pair_kernel_adjoint(
                    directional_active_space_gradient.packed_active_two_electron_gradient,
                    accepted_point_context_->prepared_active_space.active_space_two_electron_result
                        .orbital_transform);
        }

        const auto structure_active_gradient_start_time = std::chrono::steady_clock::now();
        if (directional_selected_states) {
            const SelectedStateResponseTiming response_timing =
                add_selected_subspace_response_to_active_space_gradient(
                    *current_input_,
                    *accepted_point_context_,
                    directional_selected_states.value(),
                    state_multipliers,
                    &directional_active_space_gradient);
            apply_timing_totals_.outer_response_same_spin_backward_wall_time_seconds +=
                response_timing.same_spin_seconds;
            apply_timing_totals_.outer_response_opposite_spin_backward_wall_time_seconds +=
                response_timing.opposite_spin_seconds;
            apply_timing_totals_.outer_response_opposite_spin_packed_gradient_wall_time_seconds +=
                response_timing.opposite_spin_packed_gradient_seconds;
            apply_timing_totals_.outer_response_opposite_spin_alpha_overlap_wall_time_seconds +=
                response_timing.opposite_spin_alpha_overlap_seconds;
            apply_timing_totals_.outer_response_opposite_spin_beta_overlap_wall_time_seconds +=
                response_timing.opposite_spin_beta_overlap_seconds;
        }
        apply_timing_totals_.outer_response_structure_active_gradient_wall_time_seconds +=
            detail::exact_hvp_elapsed_seconds(structure_active_gradient_start_time);
    }
    apply_timing_totals_.outer_response_active_gradient_wall_time_seconds +=
        detail::exact_hvp_elapsed_seconds(active_gradient_start_time);
    validate_outer_response_active_gradient(directional_active_space_gradient);
    return directional_active_space_gradient;
}

} // namespace xmvb::vb
