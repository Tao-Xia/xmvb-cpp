#pragma once

#include <Eigen/Core>

#include "vbscf/optimization/backends/result.hpp"

namespace xmvb::vb {

class SparseParameterLayout;
class VbScfObjective;
struct VbScfOptimizerOptions;
struct VbScfOptimizerResult;

namespace optimizer_detail {

/** @brief Runs CIAH with nonorthogonal orbitals and projected structure response. */
BackendRunResult run_ciah_backend(VbScfObjective* objective,
                                  const SparseParameterLayout& parameter_view,
                                  const VbScfOptimizerOptions& options,
                                  const Eigen::VectorXd& initial_parameters,
                                  const Eigen::VectorXd& initial_gradient,
                                  double initial_energy,
                                  VbScfOptimizerResult* result);

} // namespace optimizer_detail

} // namespace xmvb::vb
