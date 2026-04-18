#pragma once

#include "pfaffian_vbscf/kernel/pf_kernel_cache.hpp"

namespace xmvb::pfaffian_vbscf {

/**
 * @brief Evaluates the exact closed-shell two-electron contribution through a
 * spatial-kernel staging path.
 *
 * This helper is restricted to the current singlet `AB/BA` Pf basis and keeps
 * the numerically validated closed-shell formula isolated from the generic
 * tensor-term path.
 */
double evaluate_closed_shell_two_electron_spatial_exact(
    const PfKernelCache& cache,
    const ScalarBuffer& ggo);

}  // namespace xmvb::pfaffian_vbscf
