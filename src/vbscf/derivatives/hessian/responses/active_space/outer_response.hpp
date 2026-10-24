#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/core/contracts/input.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/derivatives/hessian/responses/active_space/integral_direction.hpp"
#include "vbscf/derivatives/hessian/responses/orbital/preparation.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/backward.hpp"
#include "integrals/response/types.hpp"
#include "integrals/response/ri.hpp"
#include "integrals/ao/ri.hpp"

namespace xmvb::vb {

struct ActiveSpaceGradientDirection {
    std::vector<double> active_orbital_overlap_gradient;
    std::vector<double> active_one_electron_gradient;
    std::vector<double> packed_active_two_electron_gradient;
};

/** @brief Wall-clock breakdown of the selected-state adjoint contraction. */
struct SelectedStateResponseTiming {
    double same_spin_seconds = 0.0;
    double opposite_spin_seconds = 0.0;
    double opposite_spin_packed_gradient_seconds = 0.0;
    double opposite_spin_alpha_overlap_seconds = 0.0;
    double opposite_spin_beta_overlap_seconds = 0.0;
};

/** @brief Allocates a zero active-space gradient in canonical storage. */
ActiveSpaceGradientDirection make_zero_active_space_gradient_direction(int n_active_orbitals);

void validate_outer_response_active_gradient(
    const ActiveSpaceGradientDirection& active_space_gradient);

/** @brief Adds the selected-state coefficient/energy response contribution. */
SelectedStateResponseTiming add_selected_state_response_to_active_space_gradient(
    const VbScfInput& input,
    const AcceptedPointContext& accepted_point_context,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    const std::vector<double>& directional_selected_state_energies,
    ActiveSpaceGradientDirection* active_space_gradient);

/**
 * @brief Adds the exact selected-subspace coupling adjoint.
 *
 * The coefficient block is the physical (unscaled) @f$Z@f$ response.  The
 * multiplier is the full selected-state matrix @f$M@f$; its off-diagonal
 * entries are required for an equal-weight multistate horizontal gauge.
 */
SelectedStateResponseTiming add_selected_subspace_response_to_active_space_gradient(
    const VbScfInput& input,
    const AcceptedPointContext& accepted_point_context,
    const SelectedStateDeterminantMatrices& coefficient_response,
    const Eigen::Ref<const Eigen::MatrixXd>& state_multipliers,
    ActiveSpaceGradientDirection* active_space_gradient);

std::vector<double> build_orbital_value_gradient_from_active_space_gradient_direction(
    const VbScfInput& input,
    const AcceptedPointContext& accepted_point_context,
    const ActiveSpaceGradientDirection& active_space_gradient_direction,
    const AcceptedOrbitalPreparationCache& orbital_preparation_cache,
    const xmvb::integrals::ExactPackedActiveTwoElectronAdjointCache* exact_two_electron_cache,
    const xmvb::integrals::RiActiveTwoElectronResponseCache* ri_two_electron_cache,
    const xmvb::integrals::RiAoFactorization* ri_factorization,
    std::vector<double>* symmetric_active_overlap_gradient_workspace,
    std::vector<double>* symmetric_active_one_electron_gradient_workspace,
    const Eigen::MatrixXd* two_electron_orbital_gradient = nullptr);

bool has_active_matrix_gradient(const AcceptedPointContext& context, int n_active_orbitals);

} // namespace xmvb::vb
