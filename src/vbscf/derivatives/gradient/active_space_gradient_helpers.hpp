#pragma once

#include "vbscf/core/vbscf_input.hpp"
#include "vbscf/integrals/active/preparation/space.hpp"
#include "vbscf/derivatives/gradient/active_space_gradient_result.hpp"

namespace xmvb::vb {

/**
 * @brief Initializes one active-space gradient result from a prepared probe context.
 *
 * This helper fills the reusable orbital-preparation and active-space input
 * tensors inside `ActiveSpaceGradientResult` while zero-initializing the
 * active-space adjoint buffers. The production gradient and exact_ctx HVP
 * paths share this result-layout contract.
 */
void initialize_active_space_gradient_probe_result(
    const VbScfInput& input,
    const TimedPreparedActiveSpaceContext& timed_active_space_context,
    ActiveSpaceGradientResult* result);

/**
 * @brief Move-based probe-result initializer for one prepared active-space context.
 *
 * This overload transfers the heavy prepared tensors into `result` once the
 * caller no longer needs to retain the temporary `TimedPreparedActiveSpaceContext`.
 */
void initialize_active_space_gradient_probe_result(
    const VbScfInput& input,
    TimedPreparedActiveSpaceContext&& timed_active_space_context,
    ActiveSpaceGradientResult* result);

}  // namespace xmvb::vb
