#pragma once

#include <functional>
#include <Eigen/Core>

namespace xmvb::vb::diagnostics {

struct CiahKeyframeProbe {
    Eigen::VectorXd gradient;
    double energy = 0.0;
};

/** @brief Opt-in derivative audit, never an optimizer acceptance rule. */
bool ciah_keyframe_audit_enabled();

/** @brief Opt-in gradient/HVP check at the first keyframe of each macro. */
void audit_ciah_keyframe(const Eigen::VectorXd& displacement,
                         const Eigen::VectorXd& hessian_displacement,
                         const Eigen::VectorXd& actual_gradient,
                         int orbital_size,
                         const std::function<CiahKeyframeProbe(const Eigen::VectorXd&)>& evaluate);

} // namespace xmvb::vb::diagnostics
