#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/coupled/structure.hpp"

namespace xmvb::core {
struct EqualWeightEigenCoupling;
}

namespace xmvb::vb {

/**
 * @brief VBSCF-scaled structure coupling for one orbital direction.
 *
 * `horizontal_forcing` is @f$Bp@f$ in coordinates
 * @f$z_s=\sqrt{2w_s}q_s@f$. The gauge coefficient response is unscaled because
 * it is consumed by the existing selected-state adjoint. The accompanying
 * multiplier already has the adjoint sign convention, namely @f$-M_g@f$.
 */
struct ScaledStructureCoupling {
  StructureTangent horizontal_forcing;
  Eigen::MatrixXd gauge_coefficient_response;
  Eigen::MatrixXd gauge_adjoint_multipliers;
};

/**
 * @brief Applies VBSCF state weights to solve-free equal-weight coupling data.
 *
 * A single state or an equally weighted positive multistate objective is
 * required. No response equation is solved and no accepted-point cache is
 * modified.
 */
ScaledStructureCoupling scale_equal_weight_structure_coupling(
    const xmvb::core::EqualWeightEigenCoupling& coupling,
    const std::vector<double>& normalized_state_weights);

}  // namespace xmvb::vb
