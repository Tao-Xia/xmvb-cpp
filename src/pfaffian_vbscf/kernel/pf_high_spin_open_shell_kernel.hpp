#pragma once

#include <vector>

#include "pfaffian_vbscf/data/pf_state.hpp"
#include "pfaffian_vbscf/types/eigen_types.hpp"

namespace xmvb::pfaffian_vbscf {

/**
 * @brief Exact high-spin fixed-`M_s` open-shell forward result through the
 * one-electron layer.
 *
 * `spatial_overlap_gradient` is the analytic derivative of the pairwise
 * overlap with respect to the active-space spatial overlap matrix. As in the
 * closed-shell production path, this is the unnormalized spatial transition
 * one-particle density used by the one-electron contraction.
 */
struct PfHighSpinOpenShellResult {
  double overlap = 0.0;
  Matrix spatial_overlap_gradient;
  double one_electron_hamiltonian = 0.0;
  int interpolation_degree = 0;
};

/**
 * @brief Evaluates one high-spin fixed-`M_s` open-shell pairwise overlap and
 * one-electron matrix element from pair blocks plus blocked alpha orbitals.
 *
 * This forward path is currently intended for states with blocked alpha
 * orbitals only. The singlet-pair sector is encoded by the bra-side
 * `beta/alpha` block and ket-side `alpha/beta` block of the Pfaffian state.
 */
PfHighSpinOpenShellResult evaluate_high_spin_open_shell_overlap_and_one_electron(
    const ConstMatrixRef& left_pair_ba,
    const ConstMatrixRef& right_pair_ab,
    const std::vector<int>& left_blocked_alpha_orbitals,
    const std::vector<int>& right_blocked_alpha_orbitals,
    const ConstMatrixRef& spatial_overlap_matrix,
    const ConstMatrixRef& one_electron_matrix,
    int n_singlet_pairs);

/**
 * @brief Convenience wrapper around
 * `evaluate_high_spin_open_shell_overlap_and_one_electron(...)` for two
 * structure-built `PfState` objects.
 */
PfHighSpinOpenShellResult evaluate_high_spin_open_shell_pf_state_pair(
    const PfState& left_state,
    const PfState& right_state,
    const ConstMatrixRef& spatial_overlap_matrix,
    const ConstMatrixRef& one_electron_matrix);

}  // namespace xmvb::pfaffian_vbscf
