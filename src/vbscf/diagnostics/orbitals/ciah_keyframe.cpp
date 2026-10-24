#include "vbscf/diagnostics/orbitals/ciah_keyframe.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include "mcscf/optimization/trace.hpp"

namespace xmvb::vb::diagnostics {

bool ciah_keyframe_audit_enabled() {
    const char* flag = std::getenv("XMVB_CIAH_KEYFRAME_AUDIT");
    return mcscf::diagnostics::ciah_event_trace_enabled() && flag &&
           (std::strcmp(flag, "transport") == 0 || std::strcmp(flag, "1") == 0);
}

void audit_ciah_keyframe(const Eigen::VectorXd& displacement,
                         const Eigen::VectorXd& hessian_displacement,
                         const Eigen::VectorXd& actual_gradient,
                         int orbital_size,
                         const std::function<CiahKeyframeProbe(const Eigen::VectorXd&)>& evaluate) {
    for (const double epsilon : {1.0e-3, 3.0e-4}) {
        const auto plus = evaluate(epsilon * displacement);
        const auto minus = evaluate(-epsilon * displacement);
        const Eigen::VectorXd error =
            (plus.gradient - minus.gradient) / (2.0 * epsilon) - hessian_displacement;
        mcscf::diagnostics::ciah_trace("ciah_transported_hvp_audit",
                                       "epsilon=%.17g,full_error=%.17g,orbital_error=%.17g,"
                                       "structure_error=%.17g,image_norm=%.17g",
                                       epsilon,
                                       error.norm(),
                                       error.head(orbital_size).norm(),
                                       error.tail(error.size() - orbital_size).norm(),
                                       hessian_displacement.norm());
        const auto finite_plus = evaluate((1.0 + epsilon) * displacement);
        const auto finite_minus = evaluate((1.0 - epsilon) * displacement);
        const double derivative = actual_gradient.dot(displacement);
        const double finite = (finite_plus.energy - finite_minus.energy) / (2.0 * epsilon);
        mcscf::diagnostics::ciah_trace("ciah_transported_gradient_audit",
                                       "epsilon=%.17g,analytic=%.17g,finite=%.17g,error=%.17g",
                                       epsilon,
                                       derivative,
                                       finite,
                                       std::abs(derivative - finite));
    }
}

} // namespace xmvb::vb::diagnostics
