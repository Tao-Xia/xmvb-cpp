#pragma once

#include <stdexcept>
#include <Eigen/Core>

namespace xmvb::diagnostics {

struct CurvatureDecomposition {
  Eigen::VectorXd full, core, model, diagonal_core;
  Eigen::VectorXd local_error, coupling, outer;
};

// Orthogonal target-orbital projectors T_p resolve the reduced identity.
// Evaluate D v = sum_p T_p H_core T_p v without forming any Hessian blocks.
template <typename Full, typename Core, typename Model, typename Project>
CurvatureDecomposition decompose_curvature(
    const Eigen::VectorXd& v, int blocks, Full&& full, Core&& core,
    Model&& model, Project&& project) {
  CurvatureDecomposition result;
  result.full = full(v);
  result.core = core(v);
  result.model = model(v);
  result.diagonal_core = Eigen::VectorXd::Zero(v.size());
  Eigen::VectorXd reconstructed = Eigen::VectorXd::Zero(v.size());
  for (int p = 0; p < blocks; ++p) {
    const Eigen::VectorXd local = project(v, p);
    reconstructed += local;
    if (local.stableNorm() > 0.0)
      result.diagonal_core += project(core(local), p);
  }
  if ((reconstructed - v).stableNorm() > 1e-10 * v.stableNorm())
    throw std::runtime_error("orbital projectors do not resolve the audit vector");
  result.local_error = result.diagonal_core - result.model;
  result.coupling = result.core - result.diagonal_core;
  result.outer = result.full - result.core;
  return result;
}

}  // namespace xmvb::diagnostics
