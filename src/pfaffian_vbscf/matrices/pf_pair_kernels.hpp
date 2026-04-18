#pragma once

#include "pfaffian_vbscf/types/eigen_types.hpp"

namespace xmvb::pfaffian_vbscf {

/**
 * @brief One Pfaffian basis-pair matrix element.
 */
struct PfPairKernelResult {
  double overlap = 0.0;
  double one_electron_hamiltonian = 0.0;
  double total_hamiltonian = 0.0;
};

/**
 * @brief Evaluates one Pfaffian basis-pair overlap and Hamiltonian element.
 *
 * The implementation follows the validated source-derivative Hamiltonian
 * algebra used in the prototype validator, specialized to the current 1D
 * trace-projection model adopted in `trace_projector`.
 */
PfPairKernelResult evaluate_pf_pair_kernel(
    const ConstMatrixRef& left_pairing_matrix,
    const ConstMatrixRef& right_pairing_matrix,
    const ConstMatrixRef& spatial_overlap_matrix,
    const ConstMatrixRef& one_electron_matrix,
    const ScalarBuffer& packed_active_two_electron_integrals,
    int n_alpha_electrons,
    int n_beta_electrons,
    int n_pairs);

}  // namespace xmvb::pfaffian_vbscf
