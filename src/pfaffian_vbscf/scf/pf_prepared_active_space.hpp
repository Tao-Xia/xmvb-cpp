#pragma once

#include "vb/matrices/prepared_active_space_context.hpp"

namespace xmvb::pfaffian_vbscf {

/**
 * @brief Builds the Pfaffian active-space context, optionally using the Pf-RI
 * two-electron preparation path.
 */
xmvb::vb::TimedPreparedActiveSpaceContext prepare_pf_timed_active_space_context(
    const xmvb::vb::CppVbInput& input);

}  // namespace xmvb::pfaffian_vbscf
