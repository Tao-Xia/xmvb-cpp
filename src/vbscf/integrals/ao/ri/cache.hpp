#pragma once

#include "vbscf/core/contracts/input.hpp"
#include "integrals/ao/ri.hpp"

namespace xmvb::vb {

/**
 * @brief Returns a molecule-static AO-side RI cache, building it on demand.
 */
const xmvb::integrals::RiAoFactorization& ensure_vbscf_input_ri_cache(const VbScfInput& input);

} // namespace xmvb::vb
