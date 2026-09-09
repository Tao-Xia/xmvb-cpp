#pragma once

#include "vbscf/core/vbscf_input.hpp"
#include "vbscf/integrals/ao/ri_factorization.hpp"

namespace xmvb::vb {

/**
 * @brief Returns a molecule-static AO-side RI cache, building it on demand.
 */
const RiAoFactorization& ensure_vbscf_input_ri_cache(
    const VbScfInput& input);

}  // namespace xmvb::vb
