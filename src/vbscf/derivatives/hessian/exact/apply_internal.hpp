#pragma once

#include <chrono>
#include <optional>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/exact/state_internal.hpp"
#include "vbscf/derivatives/hessian/context/response_internal.hpp"
#include "vbscf/derivatives/hessian/coupled/coupling.hpp"
#include "vbscf/derivatives/hessian/responses/active_space/integral_direction.hpp"
#include "vbscf/derivatives/hessian/responses/orbital/preparation.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/backward.hpp"

namespace xmvb::vb {

/** @brief Directional outer-response data shared by the block and scalar stages. */
struct PrecomputedOuterResponse {
  ActiveSpaceIntegralDirectionWorkspace integral_direction;
  SameSpinDirectionalPairCache pair_cache;
  std::optional<StructureIntegralDirection> direct_ci_direction;
  SelectedStateGeneralizedEigenDirectionalResponse selected_state_response;
  std::optional<ScaledStructureCoupling> gauge_coupling;
};

struct ExactHvpOperator::State::PrecomputedDirection {
  Eigen::VectorXd packed_direction;
  DenseOrbitalTangentContext dense_orbital_tangent_context;
  OrbitalPreparationDirectionalResult orbital_preparation_directional_result;
  std::optional<PrecomputedOuterResponse> outer_response;
};

namespace detail {

inline void encode_symmetric_ao_gradient(
    const Eigen::Ref<const Eigen::MatrixXd>& symmetric_gradient,
    std::vector<double>* encoded_gradient) {
  if (encoded_gradient == nullptr ||
      symmetric_gradient.rows() != symmetric_gradient.cols()) {
    throw std::invalid_argument("invalid symmetric AO-gradient output");
  }
  const int n_bf = static_cast<int>(symmetric_gradient.rows());
  encoded_gradient->assign(
      static_cast<std::size_t>(n_bf) * n_bf, 0.0);
  Eigen::Map<Eigen::MatrixXd> encoded(
      encoded_gradient->data(), n_bf, n_bf);
  encoded.triangularView<Eigen::Lower>() =
      symmetric_gradient.triangularView<Eigen::Lower>();
  encoded.diagonal() *= 0.5;
}

inline double exact_hvp_elapsed_seconds(
    const std::chrono::steady_clock::time_point& start_time) {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now() - start_time)
      .count();
}

}  // namespace detail
}  // namespace xmvb::vb
