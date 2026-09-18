#pragma once

#include <limits>

#include <Eigen/Core>

#include "vbscf/optimization/coupled/projected_trust.hpp"

namespace xmvb::vb {

/** @brief Outcome of response elimination in a supplied two-space projection. */
enum class CoupledProjectedModelStatus {
  Converged,
  ResponseProjectionSingular,
  OrbitalTrustRegionFailure,
  NumericalFailure,
};

/** @brief Exact solution and diagnostics in supplied orbital/response spaces. */
struct CoupledProjectedModelResult {
  CoupledProjectedModelStatus status =
      CoupledProjectedModelStatus::NumericalFailure;
  ProjectedTrustScope scope =
      ProjectedTrustScope::SuppliedProjectedSubspace;
  ProjectedTrustResult orbital_solution;
  Eigen::VectorXd response_baseline;
  Eigen::MatrixXd response_lift;
  Eigen::VectorXd reduced_gradient;
  Eigen::MatrixXd reduced_hessian;
  Eigen::VectorXd orbital_coordinates;
  Eigen::VectorXd response_coordinates;
  Eigen::VectorXd projected_orbital_kkt_residual;
  Eigen::VectorXd projected_response_kkt_residual;
  double projected_orbital_kkt_residual_norm =
      std::numeric_limits<double>::infinity();
  double projected_response_kkt_residual_norm =
      std::numeric_limits<double>::infinity();
  double response_baseline_model_change =
      std::numeric_limits<double>::quiet_NaN();
  double reduced_incremental_model_change =
      std::numeric_limits<double>::quiet_NaN();
  double total_model_change = std::numeric_limits<double>::quiet_NaN();
  double total_predicted_decrease =
      -std::numeric_limits<double>::infinity();

  bool converged() const noexcept {
    return status == CoupledProjectedModelStatus::Converged;
  }
};

/**
 * @brief Exactly eliminates response coordinates in two supplied subspaces.
 *
 * For the projected quadratic model
 *
 * @f[
 * m(y,u)=g_v^Ty+e_w^Tu+
 * \frac12 y^TAy+y^TD^Tu+\frac12u^TCu,
 * @f]
 *
 * this routine solves @f$Cu_0=-e_w@f$ and @f$CU_y=-D@f$, then solves
 * the orbital trust-region problem with
 *
 * @f[
 * \bar g=g_v+D^Tu_0,\qquad \bar H=A+D^TU_y,
 * \qquad y^TMy\leq\Delta^2.
 * @f]
 *
 * The response is reconstructed as @f$u=u_0+U_yy@f$. No response
 * regularization or fallback is applied: a singular projected @f$C@f$ is
 * reported as `ResponseProjectionSingular`. All optimality claims remain
 * restricted to the supplied orbital and response subspaces.
 * An empty supplied response space is valid and gives the orbital-only
 * projected problem; its omitted response equations must still be checked by
 * the full-space coordinator.
 */
CoupledProjectedModelResult solve_coupled_projected_model(
    const Eigen::Ref<const Eigen::MatrixXd>& orbital_hessian,
    const Eigen::Ref<const Eigen::MatrixXd>& response_hessian,
    const Eigen::Ref<const Eigen::MatrixXd>& coupling,
    const Eigen::Ref<const Eigen::MatrixXd>& orbital_metric,
    const Eigen::Ref<const Eigen::VectorXd>& orbital_gradient,
    const Eigen::Ref<const Eigen::VectorXd>& response_gradient,
    double trust_radius);

}  // namespace xmvb::vb
