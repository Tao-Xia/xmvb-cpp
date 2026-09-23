#pragma once

#include <functional>

#include <Eigen/Core>

namespace xmvb::vb {

using PreconditionerAction =
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;

/**
 * @brief Approximately inverts the shifted local orbital model for NEO.
 *
 * The caller supplies the current coupled KKT residual target. The inner
 * solve stops once its own true residual reaches that target; the outer NEO
 * solver still checks the complete coupled residual and therefore remains
 * responsible for step accuracy. No response or metric term is dropped.
 */
Eigen::VectorXd apply_shifted_metric_preconditioner(
    const Eigen::VectorXd& covector,
    double shift,
    double residual_target,
    const PreconditionerAction& apply_model,
    const PreconditionerAction& apply_metric,
    const PreconditionerAction& apply_block_inverse);

}  // namespace xmvb::vb
