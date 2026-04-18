#pragma once

#include <vector>

#include "pfaffian_vbscf/kernel/pf_kernel_cache.hpp"

namespace xmvb::pfaffian_vbscf {

/**
 * @brief Builds the two-electron Hamiltonian tensor terms for one Pfaffian
 * forward cache.
 *
 * The returned list includes the exact projected cross-trace separable
 * contribution and the cache-weighted cumulant terms driven by the trace
 * projection.
 */
std::vector<PfTensorTerm> build_two_electron_hamiltonian_tensor_terms(
    const PfKernelCache& cache);

}  // namespace xmvb::pfaffian_vbscf
