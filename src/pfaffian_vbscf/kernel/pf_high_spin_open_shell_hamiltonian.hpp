#pragma once

#include <vector>

#include "pfaffian_vbscf/data/pf_state.hpp"
#include "pfaffian_vbscf/types/eigen_types.hpp"

namespace xmvb::pfaffian_vbscf {

/**
 * @brief Full high-spin fixed-`M_s` open-shell pair result.
 *
 * `overlap_spatial_gradient` is `dS_IJ / dS_uv`.
 * `one_electron_matrix_gradient` is `dH_IJ / dh_uv` for the one-electron block.
 * `total_hamiltonian_spatial_gradient` is the full `dH_IJ / dS_uv`.
 */
struct PfHighSpinOpenShellHamiltonianResult {
  double overlap = 0.0;
  double one_electron_hamiltonian = 0.0;
  double total_hamiltonian = 0.0;
  int interpolation_degree = 0;
  Matrix overlap_spatial_gradient;
  Matrix total_hamiltonian_spatial_gradient;
  Matrix one_electron_matrix_gradient;
  ScalarBuffer packed_two_electron_gradient;
  bool gradients_computed = false;
};

/**
 * @brief Evaluates one high-spin blocked-alpha open-shell pair matrix element.
 *
 * The bra-side singlet sector is encoded by `left_pair_ba` and the ket-side
 * singlet sector by `right_pair_ab`. Only blocked-alpha fixed-`M_s`
 * open-shell states are currently supported.
 */
PfHighSpinOpenShellHamiltonianResult evaluate_high_spin_open_shell_pair_hamiltonian(
    const ConstMatrixRef& left_pair_ba,
    const ConstMatrixRef& right_pair_ab,
    const std::vector<int>& left_blocked_alpha_orbitals,
    const std::vector<int>& right_blocked_alpha_orbitals,
    const ConstMatrixRef& spatial_overlap_matrix,
    const ConstMatrixRef& one_electron_matrix,
    const ScalarBuffer& packed_two_electron_integrals,
    int n_singlet_pairs,
    bool include_gradients = false);

/**
 * @brief Convenience wrapper around
 * `evaluate_high_spin_open_shell_pair_hamiltonian(...)` for `PfState` inputs.
 */
PfHighSpinOpenShellHamiltonianResult evaluate_high_spin_open_shell_pf_state_pair_hamiltonian(
    const PfState& left_state,
    const PfState& right_state,
    const ConstMatrixRef& spatial_overlap_matrix,
    const ConstMatrixRef& one_electron_matrix,
    const ScalarBuffer& packed_two_electron_integrals,
    bool include_gradients = false);

}  // namespace xmvb::pfaffian_vbscf
