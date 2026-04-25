#pragma once

#include "vb/matrices/cpp_vb_input.hpp"
#include "vb/matrices/prepared_active_space_context.hpp"
#include "vb/scf/cpp_active_space_gradient_result.hpp"

namespace xmvb::vb {

/**
 * @brief Initializes one active-space gradient result from a prepared probe context.
 *
 * This helper fills the reusable orbital-preparation and active-space input
 * tensors inside `CppActiveSpaceGradientResult` while zero-initializing the
 * active-space adjoint buffers. The production gradient and exact_ctx HVP
 * paths share this result-layout contract.
 */
void initialize_active_space_gradient_probe_result(
    const CppVbInput& input,
    const TimedPreparedActiveSpaceContext& timed_active_space_context,
    CppActiveSpaceGradientResult* result);

/**
 * @brief Move-based probe-result initializer for one prepared active-space context.
 *
 * This overload transfers the heavy prepared tensors into `result` once the
 * caller no longer needs to retain the temporary `TimedPreparedActiveSpaceContext`.
 */
void initialize_active_space_gradient_probe_result(
    const CppVbInput& input,
    TimedPreparedActiveSpaceContext&& timed_active_space_context,
    CppActiveSpaceGradientResult* result);

}  // namespace xmvb::vb
