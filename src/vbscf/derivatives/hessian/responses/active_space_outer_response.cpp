#include "vbscf/derivatives/hessian/responses/active_space_outer_response.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/responses/opposite_spin_response.hpp"
#include "vbscf/integrals/active/active_space_matrix_backpropagator.hpp"
#include "vbscf/integrals/active/active_space_two_electron_backpropagator.hpp"
#include "vbscf/integrals/active/two_electron_indexer.hpp"
#include "vbscf/integrals/ao/ao_effective_one_electron_backpropagator.hpp"

namespace xmvb::vb {
namespace {

void throw_if_nonfinite(
    const std::vector<double>& values,
    const char* label) {
  const auto nonfinite_it = std::find_if(
      values.begin(),
      values.end(),
      [](double value) { return !std::isfinite(value); });
  if (nonfinite_it == values.end()) {
    return;
  }
  throw std::runtime_error(std::string(label) + " contains non-finite values");
}

void throw_if_nonfinite(
    const Eigen::MatrixXd& values,
    const char* label) {
  if (!values.allFinite()) {
    throw std::runtime_error(std::string(label) + " contains non-finite values");
  }
}

void resize_for_overwrite(
    std::vector<double>* values,
    std::size_t size) {
  if (values->size() != size) {
    values->resize(size);
  }
}

}  // namespace

void validate_outer_response_active_gradient(
    const ActiveSpaceGradientDirection& active_space_gradient) {
  throw_if_nonfinite(
      active_space_gradient.active_orbital_overlap_gradient,
      "exact outer-response active overlap gradient");
  throw_if_nonfinite(
      active_space_gradient.active_one_electron_gradient,
      "exact outer-response active one-electron gradient");
  throw_if_nonfinite(
      active_space_gradient.packed_active_two_electron_gradient,
      "exact outer-response active two-electron gradient");
}

static void validate_selected_state_determinant_matrices(
    const SelectedStateDeterminantMatrices& selected_state_matrices,
    const char* label) {
  for (const auto& state : selected_state_matrices.states) {
    throw_if_nonfinite(
        state.determinant_coefficients,
        label);
    throw_if_nonfinite(
        state.coefficient_matrix,
        label);
    throw_if_nonfinite(
        state.local_coefficient_matrix,
        label);
  }
}

static void validate_same_spin_matrix_backward_contribution(
    const SameSpinMatrixBackwardContribution& contribution,
    const char* label) {
  throw_if_nonfinite(
      contribution.active_orbital_overlap_gradient,
      label);
  throw_if_nonfinite(
      contribution.active_one_electron_gradient,
      label);
  throw_if_nonfinite(
      contribution.packed_active_two_electron_gradient,
      label);
}

static void validate_opposite_spin_matrix_backward_contribution(
    const OppositeSpinMatrixBackwardContribution& contribution,
    const char* label) {
  throw_if_nonfinite(
      contribution.active_orbital_overlap_gradient,
      label);
  throw_if_nonfinite(
      contribution.packed_active_two_electron_gradient,
      label);
}

static void accumulate_scaled_vector(
    const std::vector<double>& source,
    double scale,
    std::vector<double>* target) {
  if (scale == 0.0) {
    return;
  }
  for (std::size_t index = 0; index < source.size(); ++index) {
    (*target)[index] += scale * source[index];
  }
}

static void accumulate_scaled_same_spin_contribution(
    const SameSpinMatrixBackwardContribution& contribution,
    double scale,
    ActiveSpaceGradientDirection* target) {
  accumulate_scaled_vector(
      contribution.active_orbital_overlap_gradient,
      scale,
      &target->active_orbital_overlap_gradient);
  accumulate_scaled_vector(
      contribution.active_one_electron_gradient,
      scale,
      &target->active_one_electron_gradient);
  accumulate_scaled_vector(
      contribution.packed_active_two_electron_gradient,
      scale,
      &target->packed_active_two_electron_gradient);
}

static ActiveSpaceGradientDirection
build_local_active_space_gradient_direction_from_outer_response(
    const VbScfInput& input,
    const AcceptedPointContext& accepted_point_context,
    const std::vector<double>& delta_ao_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals,
    const SameSpinDirectionalPairCache* directional_pair_cache = nullptr) {
  if (!accepted_point_context.use_matrix_form_opposite_spin) {
    throw std::runtime_error(
        "outer-response active-gradient direction requires selected-state matrices");
  }
  ActiveSpaceGradientDirection direction =
      {};
  direction.active_orbital_overlap_gradient.assign(
      input.orbital_preparation_input.n_active_orbitals *
          input.orbital_preparation_input.n_active_orbitals,
      0.0);
  direction.active_one_electron_gradient.assign(
      input.orbital_preparation_input.n_active_orbitals *
          input.orbital_preparation_input.n_active_orbitals,
      0.0);
  direction.packed_active_two_electron_gradient.assign(
      packed_active_two_electron_integral_count(
          input.orbital_preparation_input.n_active_orbitals),
      0.0);
  // The local outer-response is fully matrix-form again: same-spin uses the
  // repaired canonical half-pair contraction and opposite-spin stays on the
  // already-validated matrix-form block contraction. HHO/SSO are symmetrized
  // later before the orbital pullback, so matching the pairwise canonical
  // storage convention here removes the previous diagnostic mismatch without
  // changing the physical HVP.
  const SameSpinMatrixBackwardContribution matrix_form_local_same_spin_response =
      build_local_same_spin_matrix_backward_contribution(
          accepted_point_context.same_spin_pair_cache,
          accepted_point_context.selected_state_matrices,
          accepted_point_context.selected_state_energies,
          input.orbital_preparation_input.n_active_orbitals,
          accepted_point_context.prepared_active_space
              .active_space_one_electron_result.h1e_act,
          accepted_point_context.prepared_active_space.active_space_two_electron_result,
          delta_ao_overlap_matrix,
          delta_active_one_electron_matrix,
          delta_packed_active_two_electron_integrals,
          directional_pair_cache);
  validate_same_spin_matrix_backward_contribution(
      matrix_form_local_same_spin_response,
      "exact outer-response local same-spin backward contribution");
  const OppositeSpinMatrixBackwardContribution matrix_form_local_opposite_spin_response =
      build_local_opposite_spin_matrix_backward_contribution(
          accepted_point_context.same_spin_pair_cache,
          accepted_point_context.selected_state_matrices,
          input.orbital_preparation_input.n_active_orbitals,
          accepted_point_context.prepared_active_space.active_space_two_electron_result,
          delta_ao_overlap_matrix,
          delta_packed_active_two_electron_integrals,
          directional_pair_cache);
  validate_opposite_spin_matrix_backward_contribution(
      matrix_form_local_opposite_spin_response,
      "exact outer-response local opposite-spin backward contribution");
  accumulate_scaled_same_spin_contribution(
      matrix_form_local_same_spin_response,
      1.0,
      &direction);
  accumulate_scaled_vector(
      matrix_form_local_opposite_spin_response.active_orbital_overlap_gradient,
      1.0,
      &direction.active_orbital_overlap_gradient);
  accumulate_scaled_vector(
      matrix_form_local_opposite_spin_response.packed_active_two_electron_gradient,
      1.0,
      &direction.packed_active_two_electron_gradient);
  return direction;
}

ActiveSpaceGradientDirection build_active_space_gradient_direction_from_outer_response(
    const VbScfInput& input,
    const AcceptedPointContext& accepted_point_context,
    const std::vector<double>& delta_ao_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    const std::vector<double>& directional_selected_state_energies,
    const SameSpinDirectionalPairCache* directional_pair_cache) {
  validate_selected_state_determinant_matrices(
      directional_selected_states,
      "exact outer-response directional selected-state coefficients");
  ActiveSpaceGradientDirection direction =
      build_local_active_space_gradient_direction_from_outer_response(
          input,
          accepted_point_context,
          delta_ao_overlap_matrix,
          delta_active_one_electron_matrix,
          delta_packed_active_two_electron_integrals,
          directional_pair_cache);
  if (!accepted_point_context.use_full_matrix_form_adjoint) {
    throw std::runtime_error(
        "exact outer-response active-gradient direction requires the "
        "same-spin matrix-form adjoint path");
  }

  const SameSpinMatrixBackwardContribution matrix_form_same_spin_direction =
      build_directional_same_spin_matrix_backward_contribution(
          accepted_point_context.same_spin_pair_cache,
          accepted_point_context.selected_state_matrices,
          directional_selected_states,
          accepted_point_context.selected_state_energies,
          directional_selected_state_energies,
          input.orbital_preparation_input.n_active_orbitals);
  validate_same_spin_matrix_backward_contribution(
      matrix_form_same_spin_direction,
      "exact outer-response directional same-spin backward contribution");
  const OppositeSpinMatrixBackwardContribution matrix_form_opposite_spin_direction =
      build_directional_opposite_spin_matrix_backward_contribution(
          accepted_point_context.same_spin_pair_cache,
          accepted_point_context.selected_state_matrices,
          directional_selected_states,
          input.orbital_preparation_input.n_active_orbitals);
  validate_opposite_spin_matrix_backward_contribution(
      matrix_form_opposite_spin_direction,
      "exact outer-response directional opposite-spin backward contribution");
  accumulate_scaled_same_spin_contribution(
      matrix_form_same_spin_direction,
      1.0,
      &direction);
  accumulate_scaled_vector(
      matrix_form_opposite_spin_direction.active_orbital_overlap_gradient,
      1.0,
      &direction.active_orbital_overlap_gradient);
  accumulate_scaled_vector(
      matrix_form_opposite_spin_direction.packed_active_two_electron_gradient,
      1.0,
      &direction.packed_active_two_electron_gradient);
  return direction;
}

static void write_symmetric_active_matrix_average_local(
    const std::vector<double>& matrix_storage,
    int dimension,
    std::vector<double>* symmetric_storage) {
  const std::size_t expected_size = dimension * dimension;
  resize_for_overwrite(symmetric_storage, expected_size);
  for (int column = 0; column < dimension; ++column) {
    for (int row = 0; row < dimension; ++row) {
      (*symmetric_storage)[(column) * (dimension) + (row)] =
          0.5 *
          (matrix_storage[(column) * (dimension) + (row)] +
           matrix_storage[(row) * (dimension) + (column)]);
    }
  }
}

std::vector<double> build_orbital_value_gradient_from_active_space_gradient_direction(
    const VbScfInput& input,
    const AcceptedPointContext& accepted_point_context,
    const ActiveSpaceGradientDirection& active_space_gradient_direction,
    const AcceptedOrbitalPreparationCache& orbital_preparation_cache,
    const ExactPackedActiveTwoElectronAdjointCache* exact_two_electron_cache,
    std::vector<double>* symmetric_active_overlap_gradient_workspace,
    std::vector<double>* symmetric_active_one_electron_gradient_workspace) {
  const int n_inactive_doubly_occupied_orbitals =
      (input.orbital_preparation_input.n_total_electrons -
       input.orbital_preparation_input.n_active_electrons) / 2;
  const int n_active_orbitals =
      input.orbital_preparation_input.n_active_orbitals;
  const auto& prepared_active_space =
      accepted_point_context.prepared_active_space;
  const auto& orbital_result = prepared_active_space.orbital_result;
  const auto& ao_effective_one_electron_result =
      prepared_active_space.ao_effective_one_electron_result;
  const auto& active_space_two_electron_result =
      prepared_active_space.active_space_two_electron_result;
  // The outer-response adjoint is assembled in canonical pairwise storage, so
  // the orbital pullback must consume the symmetric active-space gradients
  // through the same vector-storage overload used by the fixed-adjoint path.
  // The matrix overload applies a different SSO normalization and therefore
  // does not represent the same physical pullback.
  std::vector<double> local_symmetric_active_overlap_gradient;
  std::vector<double> local_symmetric_active_one_electron_gradient;
  std::vector<double>& symmetric_active_overlap_gradient =
      symmetric_active_overlap_gradient_workspace != nullptr
          ? *symmetric_active_overlap_gradient_workspace
          : local_symmetric_active_overlap_gradient;
  std::vector<double>& symmetric_active_one_electron_gradient =
      symmetric_active_one_electron_gradient_workspace != nullptr
          ? *symmetric_active_one_electron_gradient_workspace
          : local_symmetric_active_one_electron_gradient;
  write_symmetric_active_matrix_average_local(
      active_space_gradient_direction.active_orbital_overlap_gradient,
      n_active_orbitals,
      &symmetric_active_overlap_gradient);
  write_symmetric_active_matrix_average_local(
      active_space_gradient_direction.active_one_electron_gradient,
      n_active_orbitals,
      &symmetric_active_one_electron_gradient);

  ActiveSpaceMatrixBackpropagator matrix_backpropagator;
  const auto active_space_matrix_backpropagation_result =
      matrix_backpropagator.backpropagate(
          symmetric_active_overlap_gradient,
          symmetric_active_one_electron_gradient,
          input.orbital_preparation_input.ao_overlap_matrix,
          ao_effective_one_electron_result.ao_effective_h1e,
          orbital_result.auxiliary_orbital_matrix,
          input.orbital_preparation_input.n_basis_functions,
          n_inactive_doubly_occupied_orbitals,
          n_active_orbitals);

  Eigen::MatrixXd active_two_electron_auxiliary_gradient;
  if (exact_two_electron_cache != nullptr &&
      exact_two_electron_cache->n_basis_functions > 0) {
    active_two_electron_auxiliary_gradient =
        backpropagate_exact_packed_active_two_electron_gradient(
            active_space_gradient_direction
                .packed_active_two_electron_gradient,
            *exact_two_electron_cache);
  } else {
    ActiveSpaceTwoElectronBackpropagator
        active_space_two_electron_backpropagator;
    active_two_electron_auxiliary_gradient =
        active_space_two_electron_backpropagator
            .backpropagate(
                active_space_gradient_direction
                    .packed_active_two_electron_gradient,
                input.ao_integral_input.ao_two_electron_integral_values,
                input.ao_integral_input.ao_two_electron_integral_indices,
                orbital_result,
                active_space_two_electron_result,
                input.orbital_preparation_input.n_basis_functions,
                n_inactive_doubly_occupied_orbitals,
                n_active_orbitals)
            .active_auxiliary_orbital_gradient;
  }

  AoEffectiveOneElectronBackpropagator ao_effective_one_electron_backpropagator;
  const auto ao_effective_one_electron_backpropagation_result =
      ao_effective_one_electron_backpropagator.backpropagate(
          active_space_matrix_backpropagation_result.ao_effective_one_electron_gradient,
          input.ao_integral_input);

  std::vector<double> total_inactive_density_gradient =
      ao_effective_one_electron_backpropagation_result.inactive_density_gradient;
  Eigen::MatrixXd total_active_auxiliary_gradient =
      active_space_matrix_backpropagation_result.active_auxiliary_orbital_gradient;
  total_active_auxiliary_gradient.noalias() +=
      active_two_electron_auxiliary_gradient;
  const Eigen::Map<const Eigen::MatrixXd> total_inactive_density_gradient_matrix(
      total_inactive_density_gradient.data(),
      input.orbital_preparation_input.n_basis_functions,
      input.orbital_preparation_input.n_basis_functions);
  const std::vector<double> orbital_value_gradient =
      backpropagate_active_space_orbital_gradient(
          total_active_auxiliary_gradient,
          total_inactive_density_gradient_matrix,
          input.orbital_preparation_input,
          orbital_result,
          orbital_preparation_cache);
  throw_if_nonfinite(
      orbital_value_gradient,
      "exact outer-response orbital-value gradient");
  return orbital_value_gradient;
}

bool has_active_matrix_gradient(
    const AcceptedPointContext& context,
    int n_active_orbitals) {
  const std::size_t active_matrix_size =
      n_active_orbitals * n_active_orbitals;
  return context.active_orbital_overlap_gradient.size() == active_matrix_size &&
      context.active_one_electron_gradient.size() == active_matrix_size &&
      context.packed_active_two_electron_gradient.size() ==
      packed_active_two_electron_integral_count(n_active_orbitals);
}

AcceptedOrbitalBackpropInputs build_accepted_orbital_backprop_inputs(
    const AcceptedPointContext& accepted_point_context,
    const VbScfInput& input) {
  const int n_basis_functions = input.orbital_preparation_input.n_basis_functions;
  const int n_active_orbitals = input.orbital_preparation_input.n_active_orbitals;
  const int n_inactive_doubly_occupied_orbitals =
      (input.orbital_preparation_input.n_total_electrons -
       input.orbital_preparation_input.n_active_electrons) /
      2;
  const std::size_t ao_matrix_size =
      n_basis_functions * n_basis_functions;

  const auto& orbital_result =
      accepted_point_context.prepared_active_space.orbital_result;
  const auto& ao_effective_one_electron_result =
      accepted_point_context.prepared_active_space.ao_effective_one_electron_result;
  if (orbital_result.auxiliary_orbital_matrix.size() != ao_matrix_size ||
      orbital_result.inactive_density_matrix.size() != ao_matrix_size ||
      ao_effective_one_electron_result.ao_effective_h1e.size() != ao_matrix_size ||
      input.ao_integral_input.ao_core_hamiltonian_matrix.size() != ao_matrix_size) {
    throw std::invalid_argument(
        "accepted-point orbital backprop inputs have inconsistent AO matrix sizes");
  }

  ActiveSpaceMatrixBackpropagator matrix_backpropagator;
  const auto matrix_backpropagation_result =
      matrix_backpropagator.backpropagate(
          accepted_point_context.active_orbital_overlap_gradient,
          accepted_point_context.active_one_electron_gradient,
          input.orbital_preparation_input.ao_overlap_matrix,
          ao_effective_one_electron_result.ao_effective_h1e,
          orbital_result.auxiliary_orbital_matrix,
          n_basis_functions,
          n_inactive_doubly_occupied_orbitals,
          n_active_orbitals);

  ActiveSpaceTwoElectronBackpropagator active_space_two_electron_backpropagator;
  const auto active_space_two_electron_backpropagation_result =
      active_space_two_electron_backpropagator.backpropagate(
          accepted_point_context.packed_active_two_electron_gradient,
          input.ao_integral_input.ao_two_electron_integral_values,
          input.ao_integral_input.ao_two_electron_integral_indices,
          orbital_result,
          accepted_point_context.prepared_active_space.active_space_two_electron_result,
          n_basis_functions,
          n_inactive_doubly_occupied_orbitals,
          n_active_orbitals);

  std::vector<double> total_inactive_density_gradient =
      std::vector<double>(
          ao_effective_one_electron_result.ao_effective_h1e.data(),
          ao_effective_one_electron_result.ao_effective_h1e.data() +
              ao_effective_one_electron_result.ao_effective_h1e.size());
  const double* ao_core_hamiltonian_data =
      input.ao_integral_input.ao_core_hamiltonian_matrix.data();
  for (std::size_t index = 0;
       index < total_inactive_density_gradient.size();
       ++index) {
    total_inactive_density_gradient[index] +=
        ao_core_hamiltonian_data[index];
  }

  std::vector<double> total_ao_effective_one_electron_gradient =
      matrix_backpropagation_result.ao_effective_one_electron_gradient;
  for (std::size_t index = 0;
       index < total_ao_effective_one_electron_gradient.size();
       ++index) {
    total_ao_effective_one_electron_gradient[index] +=
        orbital_result.inactive_density_matrix.data()[index];
  }

  AoEffectiveOneElectronBackpropagator ao_backpropagator;
  const auto ao_effective_one_electron_backpropagation_result =
      ao_backpropagator.backpropagate(
          total_ao_effective_one_electron_gradient,
          input.ao_integral_input);
  for (std::size_t index = 0;
       index < total_inactive_density_gradient.size();
       ++index) {
    total_inactive_density_gradient[index] +=
        ao_effective_one_electron_backpropagation_result
            .inactive_density_gradient[index];
  }

  Eigen::MatrixXd total_active_auxiliary_gradient =
      matrix_backpropagation_result.active_auxiliary_orbital_gradient;
  total_active_auxiliary_gradient.noalias() +=
      active_space_two_electron_backpropagation_result
          .active_auxiliary_orbital_gradient;

  AcceptedOrbitalBackpropInputs result;
  result.total_active_auxiliary_gradient =
      std::move(total_active_auxiliary_gradient);
  result.total_inactive_density_gradient = std::move(total_inactive_density_gradient);
  return result;
}



}  // namespace xmvb::vb
