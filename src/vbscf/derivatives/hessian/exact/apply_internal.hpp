#pragma once

#include <chrono>
#include <optional>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/exact/state_internal.hpp"
#include "vbscf/derivatives/hessian/context/response_internal.hpp"
#include "vbscf/derivatives/hessian/responses/active_space/integral_direction.hpp"
#include "vbscf/derivatives/hessian/responses/orbital/preparation.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/backward.hpp"

namespace xmvb::vb {

/** @brief Directional outer-response data shared by the block and scalar stages. */
struct PrecomputedOuterResponse {
  ActiveSpaceIntegralDirectionWorkspace integral_direction;
  SameSpinDirectionalPairCache pair_cache;
  SelectedStateGeneralizedEigenDirectionalResponse selected_state_response;
};

struct ExactHvpOperator::State::PrecomputedDirection {
  Eigen::VectorXd packed_direction;
  DenseOrbitalTangentContext dense_orbital_tangent_context;
  OrbitalPreparationDirectionalResult orbital_preparation_directional_result;
  std::optional<PrecomputedOuterResponse> outer_response;
};

namespace detail {

inline double exact_hvp_elapsed_seconds(
    const std::chrono::steady_clock::time_point& start_time) {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now() - start_time)
      .count();
}

}  // namespace detail
}  // namespace xmvb::vb
