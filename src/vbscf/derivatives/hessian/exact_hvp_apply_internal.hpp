#pragma once

#include <chrono>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/exact_hvp_operator.hpp"
#include "vbscf/derivatives/hessian/responses/orbital_preparation_response.hpp"

namespace xmvb::vb {

struct ExactHvpOperator::PrecomputedDirection {
  Eigen::VectorXd packed_direction;
  DenseOrbitalTangentContext dense_orbital_tangent_context;
  OrbitalPreparationDirectionalResult orbital_preparation_directional_result;
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
