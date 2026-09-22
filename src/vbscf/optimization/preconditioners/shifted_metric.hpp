#pragma once

#include <functional>

#include <Eigen/Core>

namespace xmvb::vb {

using PreconditionerAction =
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;

/**
 * @brief Applies the inverse of a shifted local model without assembling it.
 *
 * Solves @f$(D+\lambda M)x=b@f$ by preconditioned conjugate gradients.  The
 * supplied inverse is an inexpensive approximation to the same shifted
 * operator; it affects convergence only, while the model and metric actions
 * define the solved equation.  A zero shift returns the exact local-model
 * inverse directly.
 */
Eigen::VectorXd apply_inverse_shifted_metric_model(
    const Eigen::VectorXd& covector,
    double shift,
    const PreconditionerAction& apply_model,
    const PreconditionerAction& apply_metric,
    const PreconditionerAction& apply_approximate_inverse);

}  // namespace xmvb::vb
