#pragma once

#include <cstddef>

#include <Eigen/Core>

#include "vbscf/integrals/active/two_electron/response/types.hpp"
#include "vbscf/integrals/ao/contracts/input.hpp"

namespace xmvb::vb::detail {

/** @brief Applies the symmetric sparse AO-pair ERI operator to a vector block. */
void apply_exact_ao_pair_kernel(
    const AoIntegralInput& ao_integral_input,
    const Eigen::Ref<const ExactCtxPairMatrix>& pair_coefficients,
    int n_bf,
    std::size_t n_active_pairs,
    ExactCtxPairMatrix* pair_products);

}  // namespace xmvb::vb::detail
