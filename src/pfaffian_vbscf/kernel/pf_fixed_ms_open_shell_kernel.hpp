#pragma once

#include <vector>

#include "pfaffian_vbscf/data/pf_state.hpp"
#include "pfaffian_vbscf/types/eigen_types.hpp"

namespace xmvb::pfaffian_vbscf {

/**
 * @brief Exact fixed-`M_s` open-shell pairwise overlap and one-electron result.
 *
 * `spatial_overlap_gradient` is the analytic derivative of the pairwise
 * overlap with respect to the active-space spatial overlap matrix. As in the
 * closed-shell production path, this is the unnormalized spatial transition
 * one-particle density used by the one-electron contraction.
 */
struct PfFixedMsOpenShellResult {
  double overlap = 0.0;
  Matrix spatial_overlap_gradient;
  double one_electron_hamiltonian = 0.0;
  int interpolation_degree = 0;
};

/**
 * @brief Full fixed-`M_s` open-shell pair Hamiltonian result.
 *
 * This forward evaluator currently returns the exact overlap, one-electron,
 * two-electron, and total Hamiltonian matrix elements. Gradient support remains
 * on the overlap/one-electron helper above and can be extended onto this
 * result later.
 */
struct PfFixedMsOpenShellHamiltonianResult {
  double overlap = 0.0;
  double one_electron_hamiltonian = 0.0;
  double two_electron_hamiltonian = 0.0;
  double total_hamiltonian = 0.0;
  int interpolation_degree = 0;
  Matrix overlap_spatial_gradient;
  Matrix total_hamiltonian_spatial_gradient;
  Matrix one_electron_matrix_gradient;
  ScalarBuffer packed_two_electron_gradient;
  bool gradients_computed = false;
};

/**
 * @brief Evaluates one fixed-`M_s` open-shell pairwise overlap and one-electron
 * matrix element from pair blocks plus blocked alpha/beta orbitals.
 *
 * The singlet-pair sector is encoded by the bra-side `beta/alpha` block and
 * ket-side `alpha/beta` block of the Pfaffian state.
 */
PfFixedMsOpenShellResult evaluate_fixed_ms_open_shell_overlap_and_one_electron(
    const ConstMatrixRef& left_pair_ba,
    const ConstMatrixRef& right_pair_ab,
    const std::vector<int>& left_blocked_alpha_orbitals,
    const std::vector<int>& left_blocked_beta_orbitals,
    const std::vector<int>& right_blocked_alpha_orbitals,
    const std::vector<int>& right_blocked_beta_orbitals,
    const ConstMatrixRef& spatial_overlap_matrix,
    const ConstMatrixRef& one_electron_matrix,
    int n_singlet_pairs);

/**
 * @brief Convenience wrapper around
 * `evaluate_fixed_ms_open_shell_overlap_and_one_electron(...)` for two
 * structure-built `PfState` objects.
 */
PfFixedMsOpenShellResult evaluate_fixed_ms_open_shell_pf_state_pair(
    const PfState& left_state,
    const PfState& right_state,
    const ConstMatrixRef& spatial_overlap_matrix,
    const ConstMatrixRef& one_electron_matrix);

/**
 * @brief Evaluates one exact fixed-`M_s` open-shell pair Hamiltonian matrix
 * element.
 *
 * The overlap interpolation still runs over the singlet-pair degree only, but
 * each sample now contracts the full one- and two-electron Hamiltonian through
 * analytic sample-space alpha/beta trace factors.
 */
PfFixedMsOpenShellHamiltonianResult evaluate_fixed_ms_open_shell_pair_hamiltonian(
    const ConstMatrixRef& left_pair_ba,
    const ConstMatrixRef& right_pair_ab,
    const std::vector<int>& left_blocked_alpha_orbitals,
    const std::vector<int>& left_blocked_beta_orbitals,
    const std::vector<int>& right_blocked_alpha_orbitals,
    const std::vector<int>& right_blocked_beta_orbitals,
    const ConstMatrixRef& spatial_overlap_matrix,
    const ConstMatrixRef& one_electron_matrix,
    const ScalarBuffer& packed_two_electron_integrals,
    int n_singlet_pairs,
    bool include_gradients = false);

/**
 * @brief Convenience wrapper around
 * `evaluate_fixed_ms_open_shell_pair_hamiltonian(...)` for `PfState` inputs.
 */
PfFixedMsOpenShellHamiltonianResult evaluate_fixed_ms_open_shell_pf_state_pair_hamiltonian(
    const PfState& left_state,
    const PfState& right_state,
    const ConstMatrixRef& spatial_overlap_matrix,
    const ConstMatrixRef& one_electron_matrix,
    const ScalarBuffer& packed_two_electron_integrals,
    bool include_gradients = false);

}  // namespace xmvb::pfaffian_vbscf
