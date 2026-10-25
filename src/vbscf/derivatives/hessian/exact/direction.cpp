#include "vbscf/derivatives/hessian/exact/apply_internal.hpp"
#include "vbscf/derivatives/hessian/exact/state_internal.hpp"

#include <stdexcept>
#include <algorithm>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/responses/active_space/integral_direction.hpp"
#include "vbscf/derivatives/hessian/responses/orbital/preparation.hpp"
#include "integrals/response/ri.hpp"
#include "integrals/active/frame.hpp"

namespace xmvb::vb {

ExactHvpOperator::State::PrecomputedDirection
ExactHvpOperator::State::prepare_direction(Eigen::VectorXd packed_direction) const {
    const auto& orbital_input = current_input_->orbital_preparation_input;
    const int n_inactive = (orbital_input.n_total_electrons - orbital_input.n_active_electrons) / 2;
    PrecomputedDirection result;
    result.packed_direction = std::move(packed_direction);
    result.dense_orbital_tangent_context =
        build_dense_orbital_tangent_context(orbital_input,
                                            parameter_view_,
                                            result.packed_direction,
                                            *accepted_orbital_preparation_cache_);
    result.orbital_preparation_directional_result =
        build_orbital_preparation_directional_result(orbital_input,
                                                     result.dense_orbital_tangent_context,
                                                     n_inactive,
                                                     orbital_input.n_active_orbitals,
                                                     *accepted_orbital_preparation_cache_,
                                                     accepted_ri_factorization_ == nullptr);
    result.integral_tangent = integrals::active_frame_tangent(
        result.orbital_preparation_directional_result.delta_active_auxiliary_orbitals,
        accepted_point_context_->prepared_active_space.active_space_two_electron_result
            .orbital_transform);
    return result;
}

std::vector<integrals::RiAoProjectedResponse> ExactHvpOperator::State::ri_ao_responses(
    std::span<const PrecomputedDirection> directions,
    const Eigen::Ref<const Eigen::MatrixXd>& weighted_active_columns,
    bool fixed_two_electron) const {
    if (directions.empty())
        return {};
    const auto& geometry = *accepted_orbital_preparation_cache_;
    const auto& input = current_input_->orbital_preparation_input;
    const int n = input.n_basis_functions, na = input.n_active_orbitals;
    const int ni = (input.n_total_electrons - input.n_active_electrons) / 2;
    if (!ni) {
        std::vector<integrals::RiAoProjectedResponse> result(directions.size());
        for (auto& value : result) {
            value.inactive.resize(n, 0);
            value.active.setZero(n, na);
        }
        return result;
    }

    // The block shares one accepted inactive support. Absent columns are exact
    // zeros, not a numerical truncation of any direction.
    std::vector<int> moving;
    for (int i = 0; i < ni; ++i) {
        const bool touched =
            std::any_of(directions.begin(), directions.end(), [&](const auto& direction) {
                return !direction.dense_orbital_tangent_context.delta_normalized_orbitals.col(i)
                            .isZero(0);
            });
        if (touched)
            moving.push_back(i);
    }
    Eigen::MatrixXd duals(n, moving.size());
    for (int i = 0; i < static_cast<int>(moving.size()); ++i)
        duals.col(i) = geometry.inactive_auxiliary.col(moving[i]);
    const auto sci = geometry.basis_overlap_times_normalized.leftCols(ni);
    const auto& frame = accepted_point_context_->prepared_active_space
                            .active_space_two_electron_result.orbital_transform;
    const Eigen::MatrixXd active_weight =
        fixed_two_electron
            ? (frame.size()
                   ? Eigen::MatrixXd(frame * accepted_hho_gradient_symmetric_ * frame.transpose())
                   : accepted_hho_gradient_symmetric_)
            : Eigen::MatrixXd{};
    const integrals::RiActiveHvpInput jet{
        active_weight, accepted_point_context_->packed_active_two_electron_gradient};

    std::vector<Eigen::MatrixXd> horizontal, weights;
    std::vector<integrals::RiAoResponseDirection> requests;
    horizontal.reserve(directions.size());
    weights.reserve(directions.size());
    requests.reserve(directions.size());
    for (std::size_t column = 0; column < directions.size(); ++column) {
        const auto& direction = directions[column];
        Eigen::MatrixXd delta(n, moving.size());
        for (int i = 0; i < static_cast<int>(moving.size()); ++i)
            delta.col(i) =
                direction.dense_orbital_tangent_context.delta_normalized_orbitals.col(moving[i]);
        horizontal.emplace_back(delta - geometry.inactive_auxiliary * (sci.transpose() * delta));
        const Eigen::Map<const Eigen::MatrixXd> weighted(
            weighted_active_columns.col(column).data(), n, na);
        weights.emplace_back(frame.size() ? Eigen::MatrixXd(weighted * frame.transpose())
                                          : Eigen::MatrixXd(weighted));
        requests.push_back({horizontal.back(),
                            weights.back(),
                            fixed_two_electron ? &direction.integral_tangent : nullptr});
    }
    auto response = integrals::apply_ri_projected_ao_response_batch(
        *accepted_ri_two_electron_cache_,
        duals,
        requests,
        geometry.inactive_auxiliary,
        ri_ao_response_workspace_,
        workspace::limit(),
        {accepted_ri_inactive_one_index_factors_.get(), moving},
        fixed_two_electron ? &jet : nullptr);
    return response;
}

ActiveSpaceIntegralDirectionView ExactHvpOperator::State::build_active_integral_direction(
    const PrecomputedDirection& direction,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_h1e_times_active,
    const Eigen::VectorXd* precomputed_delta_packed_two_electron,
    ActiveSpaceIntegralDirectionWorkspace* workspace,
    bool packed_two_electron_required) const {
    const auto& accepted_two_electron =
        accepted_point_context_->prepared_active_space.active_space_two_electron_result;
    const ActiveSpaceIntegralDirectionContext integral_context{
        *current_input_,
        accepted_ri_two_electron_cache_.has_value() ? nullptr : &accepted_exact_two_electron_cache_,
        accepted_basis_overlap_times_active_auxiliary_orbitals_,
        accepted_point_context_->prepared_active_space.active_space_one_electron_result.ao_image,
        accepted_two_electron.dense_active_coefficients};
    return xmvb::vb::build_active_space_integral_direction(
        integral_context,
        ActiveSpaceIntegralTangent{
            direction.orbital_preparation_directional_result.delta_active_auxiliary_orbitals,
            direction.integral_tangent,
            delta_h1e_times_active,
            precomputed_delta_packed_two_electron,
            packed_two_electron_required},
        workspace);
}

} // namespace xmvb::vb
