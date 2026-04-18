#pragma once

#include "pfaffian_vbscf/kernel/pf_kernel_cache.hpp"

namespace xmvb::pfaffian_vbscf {

/**
 * @brief Evaluates the closed-shell two-electron contribution directly from
 * metric-whitened RI active-pair factors.
 */
double evaluate_closed_shell_two_electron_spatial_ri(
    const PfKernelCache& cache,
    int n_auxiliary_functions,
    const ScalarBuffer& ri_active_pair_factors);

/**
 * @brief Backpropagates a packed active-space two-electron gradient through
 * the RI Gram map `g = L^T L`.
 *
 * `packed_two_electron_grad` is the lower-triangular packed gradient with
 * respect to the active-pair Gram matrix. The output gradient is stored in the
 * same row-major `[auxiliary_function][packed_active_pair]` layout as
 * `ri_active_pair_factors`.
 */
void backpropagate_closed_shell_ri_factor_gram(
    int n_active_orbitals,
    int n_auxiliary_functions,
    const ScalarBuffer& ri_active_pair_factors,
    const ScalarBuffer& packed_two_electron_grad,
    ScalarBuffer* ri_active_pair_factor_grad);

}  // namespace xmvb::pfaffian_vbscf
