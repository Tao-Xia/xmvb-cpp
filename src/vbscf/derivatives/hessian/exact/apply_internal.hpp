#pragma once

#include <chrono>
#include <optional>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/exact/state_internal.hpp"
#include "vbscf/derivatives/hessian/context/response_internal.hpp"
#include "vbscf/derivatives/hessian/coupled/coupling.hpp"
#include "vbscf/derivatives/hessian/responses/active_space/integral_direction.hpp"
#include "vbscf/derivatives/hessian/responses/active_space/outer_response.hpp"
#include "vbscf/derivatives/hessian/responses/orbital/preparation.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/backward.hpp"

namespace xmvb::vb {

namespace detail {
class LocalActiveSpaceTileAccumulator;
}

/** @brief Borrowed inputs to the common selected-state adjoint differential. */
struct ActiveResponseView {
    ActiveSpaceIntegralDirectionView integrals;
    const SelectedStateGeneralizedEigenDirectionalResponse* states = nullptr;
    const ScaledStructureCoupling* gauge = nullptr;
    const StructureIntegralDirection* direct_ci = nullptr;
    const Eigen::MatrixXd* ri_direction = nullptr;
    const ActiveSpaceGradientDirection* local_gradient = nullptr;
    detail::LocalActiveSpaceTileAccumulator* local_tiles = nullptr;
};

/** @brief Directional outer-response data shared by the block and scalar stages. */
struct PrecomputedOuterResponse {
    ActiveSpaceIntegralDirectionWorkspace integral_direction;
    std::optional<ActiveSpaceGradientDirection> local_active_gradient;
    std::optional<StructureIntegralDirection> direct_ci_direction;
    SelectedStateGeneralizedEigenDirectionalResponse selected_state_response;
    std::optional<ScaledStructureCoupling> gauge_coupling;
    /** Complete active cotangent, shared by the block integral pullback. */
    std::optional<ActiveSpaceGradientDirection> active_gradient;
    /** Integral-frame changing 2e image, not yet pulled back to raw orbitals. */
    Eigen::MatrixXd changing_two_electron;
};

struct ExactHvpOperator::State::PrecomputedDirection {
    Eigen::VectorXd packed_direction;
    DenseOrbitalTangentContext dense_orbital_tangent_context;
    OrbitalPreparationDirectionalResult orbital_preparation_directional_result;
    /** Fixed accepted integral frame, dC R^-1; shared by forward and reverse. */
    Eigen::MatrixXd integral_tangent;
    /** Fixed 2e response consumed during the shared RI AO direction sweep. */
    Eigen::MatrixXd ri_fixed_two_electron;
    std::optional<PrecomputedOuterResponse> outer_response;
};

namespace detail {

ActiveSpaceGradientDirection
make_active_gradient_direction(const StructureActiveIntegralAdjoint& adjoint,
                               int n_active_orbitals);

inline double exact_hvp_elapsed_seconds(const std::chrono::steady_clock::time_point& start_time) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
}

} // namespace detail
} // namespace xmvb::vb
