#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/core/vbscf_input.hpp"
#include "vbscf/derivatives/hessian/accepted_point_context.hpp"
#include "vbscf/derivatives/hessian/responses/orbital_preparation_response.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin_response.hpp"
#include "vbscf/integrals/active/active_space_two_electron_response.hpp"

namespace xmvb::vb {

struct ActiveSpaceGradientDirection {
  std::vector<double> active_orbital_overlap_gradient;
  std::vector<double> active_one_electron_gradient;
  std::vector<double> packed_active_two_electron_gradient;
};

void validate_outer_response_active_gradient(
    const ActiveSpaceGradientDirection& active_space_gradient);

ActiveSpaceGradientDirection build_active_space_gradient_direction_from_outer_response(
    const VbScfInput& input,
    const AcceptedPointContext& accepted_point_context,
    const std::vector<double>& delta_ao_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    const std::vector<double>& directional_selected_state_energies,
    const SameSpinDirectionalPairCache* directional_pair_cache = nullptr);

std::vector<double> build_orbital_value_gradient_from_active_space_gradient_direction(
    const VbScfInput& input,
    const AcceptedPointContext& accepted_point_context,
    const ActiveSpaceGradientDirection& active_space_gradient_direction,
    const AcceptedOrbitalPreparationCache& orbital_preparation_cache,
    const ExactPackedActiveTwoElectronAdjointCache* exact_two_electron_cache,
    std::vector<double>* symmetric_active_overlap_gradient_workspace,
    std::vector<double>* symmetric_active_one_electron_gradient_workspace);

bool has_active_matrix_gradient(
    const AcceptedPointContext& context,
    int n_active_orbitals);

struct AcceptedOrbitalBackpropInputs {
  Eigen::MatrixXd total_active_auxiliary_gradient;
  std::vector<double> total_inactive_density_gradient;
};

AcceptedOrbitalBackpropInputs build_accepted_orbital_backprop_inputs(
    const AcceptedPointContext& context,
    const VbScfInput& input);

}  // namespace xmvb::vb
