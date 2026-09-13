#pragma once

#include <Eigen/Core>

#include "vbscf/optimization/backends/result.hpp"

namespace xmvb::vb {

class VbScfObjective;
class SparseParameterLayout;
struct VbScfOptimizerOptions;
struct VbScfOptimizerResult;

namespace optimizer_detail {

BackendRunResult run_full_space_lbfgs_backend(
    VbScfObjective* objective,
    const VbScfOptimizerOptions& options,
    const Eigen::VectorXd& initial_parameters,
    const Eigen::VectorXd& initial_gradient,
    double initial_energy,
    VbScfOptimizerResult* result);

BackendRunResult run_nonredundant_lbfgs_backend(
    VbScfObjective* objective,
    const SparseParameterLayout& parameter_view,
    const VbScfOptimizerOptions& options,
    const Eigen::VectorXd& initial_parameters,
    const Eigen::VectorXd& initial_gradient,
    double initial_energy,
    VbScfOptimizerResult* result);

}  // namespace optimizer_detail

}  // namespace xmvb::vb
