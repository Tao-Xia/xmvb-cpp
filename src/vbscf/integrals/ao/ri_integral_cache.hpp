#pragma once

#include "runtime/libcint_ri_integral_provider.hpp"
#include "vbscf/core/vbscf_input.hpp"

namespace xmvb::vb {

/**
 * @brief Returns a molecule-static AO-side RI cache, building it on demand.
 */
const LibcintRiIntegralProviderResult& ensure_vbscf_input_ri_cache(
    const VbScfInput& input,
    const LibcintRiIntegralProviderOptions& options = {});

}  // namespace xmvb::vb
