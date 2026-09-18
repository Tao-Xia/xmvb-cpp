#pragma once

#include <limits>

#include <Eigen/Core>

namespace xmvb::vb {

/** @brief Scope of every global-optimality claim made by this solver. */
enum class ProjectedTrustScope {
  /** Global only in the supplied finite-dimensional projected subspace. */
  SuppliedProjectedSubspace,
};

/** @brief Outcome of the exact projected generalized trust-region solve. */
enum class ProjectedTrustStatus {
  InteriorGlobal,
  BoundaryGlobal,
  HardCaseGlobal,
  InvalidMetric,
  NumericalFailure,
};

/**
 * @brief Certified solution of a small projected trust-region problem.
 *
 * `Global` in the status names always means global in the supplied projected
 * subspace. It makes no statement about curvature omitted from that subspace.
 */
struct ProjectedTrustResult {
  ProjectedTrustScope scope =
      ProjectedTrustScope::SuppliedProjectedSubspace;
  ProjectedTrustStatus status =
      ProjectedTrustStatus::NumericalFailure;
  Eigen::VectorXd coordinates;
  double shift = std::numeric_limits<double>::quiet_NaN();
  double metric_norm = std::numeric_limits<double>::quiet_NaN();
  double predicted_decrease = -std::numeric_limits<double>::infinity();
  double stationarity_residual = std::numeric_limits<double>::infinity();
  double complementarity_residual = std::numeric_limits<double>::infinity();
  double boundary_residual = std::numeric_limits<double>::infinity();

  bool converged() const noexcept {
    return status == ProjectedTrustStatus::InteriorGlobal ||
        status == ProjectedTrustStatus::BoundaryGlobal ||
        status == ProjectedTrustStatus::HardCaseGlobal;
  }
};

/**
 * @brief Solves a dense problem only in the supplied projected coordinates.
 *
 * This minimizes
 *
 * @f[
 * g_k^Ts+\tfrac12s^TH_ks,
 * \qquad s^TG_ks\leq\Delta^2,
 * @f]
 *
 * where @f$H_k@f$ is symmetric and @f$G_k@f$ is SPD. The metric is whitened
 * by Cholesky factorization, the whitened Hessian is diagonalized, and the
 * secular equation is solved to floating-point resolution. At the classical
 * hard case the endpoint pseudoinverse solution is augmented along a
 * minimum-curvature eigenvector to reach the boundary exactly; no conservative
 * over-shift replaces that solution.
 *
 * No full orbital Hessian is accepted or formed. Consequently every returned
 * global-optimality status is restricted to this projected problem.
 */
ProjectedTrustResult solve_projected_generalized_trust_region(
    const Eigen::Ref<const Eigen::MatrixXd>& projected_hessian,
    const Eigen::Ref<const Eigen::MatrixXd>& projected_metric,
    const Eigen::Ref<const Eigen::VectorXd>& projected_gradient,
    double trust_radius);

}  // namespace xmvb::vb
