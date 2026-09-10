#include "vbscf/derivatives/hessian/responses/same_spin/matrix_weights_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/weight_kernels_internal.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

namespace xmvb::vb::detail {

SameSpinLocalResponseWeightMatrices build_local_same_spin_response_weight_matrices(
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies,
    const SameSpinDirectionalScalarMatrices& alpha_directional_scalars,
    const SameSpinDirectionalScalarMatrices& beta_directional_scalars,
    bool close_shell_same_spin) {
  // The accepted selected-state coefficients stay fixed. Only the partner
  // kernels change, so the same unique-spin coefficient matrices `C^(n)` map
  // `δdet_partner` and `δH_same,partner` back to the active same-spin weights.
  if (selected_state_energies.size() != selected_states.states.size()) {
    throw std::invalid_argument(
        "selected_state_energies must align with selected_states.states");
  }

  std::vector<double> hamiltonian_scales;
  std::vector<double> overlap_scales;
  hamiltonian_scales.reserve(selected_states.states.size());
  overlap_scales.reserve(selected_states.states.size());
  for (std::size_t state_offset = 0;
       state_offset < selected_states.states.size();
       ++state_offset) {
    const double state_weight =
        selected_states.states[state_offset].normalized_state_weight;
    hamiltonian_scales.push_back(state_weight);
    overlap_scales.push_back(-selected_state_energies[state_offset] * state_weight);
  }

  SameSpinLocalResponseWeightMatrices weight_matrices;
  const SingleChannelSameSpinWeightMatrices delta_hamiltonian_weights =
      build_dense_same_spin_weight_matrices_from_partner_kernels(
          selected_states,
          alpha_directional_scalars.delta_overlap_determinant_matrix,
          beta_directional_scalars.delta_overlap_determinant_matrix,
          hamiltonian_scales,
          close_shell_same_spin);
  weight_matrices.alpha_delta_hamiltonian_weight_matrix =
      std::move(delta_hamiltonian_weights.alpha_weight_matrix);
  weight_matrices.beta_delta_hamiltonian_weight_matrix =
      std::move(delta_hamiltonian_weights.beta_weight_matrix);

  const SingleChannelSameSpinWeightMatrices delta_overlap_weights =
      build_dense_same_spin_weight_matrices_from_partner_kernels(
          selected_states,
          alpha_directional_scalars.delta_overlap_determinant_matrix,
          beta_directional_scalars.delta_overlap_determinant_matrix,
          overlap_scales,
          close_shell_same_spin);
  weight_matrices.alpha_delta_overlap_weight_matrix =
      std::move(delta_overlap_weights.alpha_weight_matrix);
  weight_matrices.beta_delta_overlap_weight_matrix =
      std::move(delta_overlap_weights.beta_weight_matrix);

  const SingleChannelSameSpinWeightMatrices delta_partner_weights =
      build_dense_same_spin_weight_matrices_from_partner_kernels(
          selected_states,
          alpha_directional_scalars.delta_regular_total_hamiltonian_matrix +
              alpha_directional_scalars.delta_singular_total_hamiltonian_matrix,
          beta_directional_scalars.delta_regular_total_hamiltonian_matrix +
              beta_directional_scalars.delta_singular_total_hamiltonian_matrix,
          hamiltonian_scales,
          close_shell_same_spin);
  weight_matrices.alpha_delta_partner_total_transfer_matrix =
      std::move(delta_partner_weights.alpha_weight_matrix);
  weight_matrices.beta_delta_partner_total_transfer_matrix =
      std::move(delta_partner_weights.beta_weight_matrix);
  return weight_matrices;
}
}  // namespace xmvb::vb::detail
