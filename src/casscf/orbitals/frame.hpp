#pragma once

#include <Eigen/Core>

namespace xmvb::casscf {

/**
 * Complete an AO-metric orthonormal full MO frame from core/active seeds.
 * Columns retain the core span, then the active span projected off the core;
 * only the virtual complement is newly chosen.
 */
Eigen::MatrixXd complete_metric_orthonormal_frame(const Eigen::MatrixXd& core_active_seed,
                                                  const Eigen::MatrixXd& overlap,
                                                  int ncore,
                                                  int nactive);

} // namespace xmvb::casscf
