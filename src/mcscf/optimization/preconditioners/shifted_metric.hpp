#pragma once

#include <functional>

#include <Eigen/Core>
#include "mcscf/optimization/actions.hpp"

namespace xmvb::mcscf {

/**
 * @brief Approximately inverts a shifted local orbital model.
 *
 * The caller supplies a dimensionless residual fraction relative to the
 * current orbital covector. The inner solve measures both sides in the dual
 * norm of the positive block inverse. The calling solver checks its complete
 * physical coupled KKT residual and remains responsible for step accuracy.
 */
Eigen::VectorXd
apply_shifted_metric_preconditioner(const Eigen::VectorXd& covector,
                                    double shift,
                                    double relative_residual_target,
                                    const linalg::VectorAction& apply_model,
                                    const linalg::VectorAction& apply_metric,
                                    const linalg::VectorAction& apply_block_inverse);

} // namespace xmvb::mcscf
