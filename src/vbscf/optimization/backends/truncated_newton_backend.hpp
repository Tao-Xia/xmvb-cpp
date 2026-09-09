#pragma once

#include <Eigen/Core>

#include "vbscf/optimization/backends/backend_run_result.hpp"

namespace xmvb::vb {

class SparseParameterLayout;
class VbScfObjective;
struct VbScfOptimizerOptions;
struct VbScfOptimizerResult;

namespace optimizer_detail {

BackendRunResult run_truncated_newton_backend(
    VbScfObjective* objective,
    const SparseParameterLayout& parameter_view,
    const VbScfOptimizerOptions& options,
    const Eigen::VectorXd& initial_parameters,
    const Eigen::VectorXd& initial_gradient,
    double initial_energy,
    VbScfOptimizerResult* result);

}  // namespace optimizer_detail

}  // namespace xmvb::vb
