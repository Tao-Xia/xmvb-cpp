#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/coupled/structure.hpp"

namespace xmvb::linalg {
struct EqualWeightEigenCoupling;
}

namespace xmvb::vb {

/**
 * @brief VBSCF-scaled structure coupling for one orbital direction.
 *
 * `horizontal_forcing` is @f$Bp@f$ in coordinates
 * @f$z_s=\sqrt{2w_s}q_s@f$. The gauge frame is unscaled and retained without
 * expanding C A_g into structure coefficients. The accompanying
 * multiplier already has the adjoint sign convention, namely @f$-M_g@f$.
 */
struct ScaledStructureCoupling {
    StructureTangent horizontal_forcing;
    /** Known internal frame A_g in Q_g=C A_g; never inferred from Q_g. */
    Eigen::MatrixXd gauge_frame;
    Eigen::MatrixXd gauge_adjoint_multipliers;
};

/**
 * @brief Applies VBSCF state weights to solve-free equal-weight coupling data.
 *
 * A single state or an equally weighted positive multistate objective is
 * required. No response equation is solved and no accepted-point cache is
 * modified.
 * Gauge-only input has an empty horizontal forcing; it remains empty.
 */
ScaledStructureCoupling
scale_equal_weight_structure_coupling(const xmvb::linalg::EqualWeightEigenCoupling& coupling,
                                      const std::vector<double>& normalized_state_weights);

} // namespace xmvb::vb
