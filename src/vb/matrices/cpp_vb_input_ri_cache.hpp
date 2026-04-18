#pragma once

#include "runtime/libcint_ri_integral_provider.hpp"
#include "vb/matrices/cpp_vb_input.hpp"

namespace xmvb::vb {

/**
 * @brief Returns a molecule-static AO-side RI cache, building it on demand.
 */
const LibcintRiIntegralProviderResult& ensure_cpp_vb_input_ri_cache(
    const CppVbInput& input,
    const LibcintRiIntegralProviderOptions& options = {});

}  // namespace xmvb::vb
