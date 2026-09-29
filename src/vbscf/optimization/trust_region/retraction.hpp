#pragma once

#include <Eigen/Core>

#include "vbscf/orbitals/charts/chart.hpp"
#include "vbscf/orbitals/charts/layout.hpp"
#include "vbscf/orbitals/charts/physical_metric.hpp"
#include "vbscf/orbitals/preparation/input.hpp"

namespace xmvb::vb {

Eigen::VectorXd gather_nonredundant_retract_tangent(
    const OrbitalPreparationInput& orbital_preparation_input,
    const OrbitalChart& space,
    const SparseParameterLayout& parameter_view,
    const Eigen::VectorXd& reduced_step);

/**
 * @brief Accepted-point pullback metric of the additive sparse retraction.
 *
 * The algebraic quotient chart supplies the horizontal raw-coefficient lift.
 * This object measures that same lift after the inactive-subspace and active-
 * ray physical map; it never stores a full reduced metric matrix. The chart is
 * prewhitened only in independent local orbital-normalization metrics, so the
 * complete coupled reduced metric is generally not the identity. Callers must
 * use `apply` for the metric covector and `norm` for the physical tangent norm;
 * only a small projected metric may be assembled and factorized.
 */
class NonredundantRetractionMetric {
public:
  NonredundantRetractionMetric(
      const OrbitalChart& space,
      const SparseParameterLayout& parameter_view,
      const OrbitalPreparationInput& input);

  Eigen::VectorXd tangent(const Eigen::VectorXd& reduced_step) const;
  Eigen::VectorXd apply(const Eigen::VectorXd& reduced_step) const;

  /**
   * @brief Applies the inverse Riesz map without assembling the metric.
   *
   * Solves @f$M y=b@f$ by conjugate gradients using matrix-free metric
   * actions. True residuals certify convergence.  When finite-precision loss
   * of conjugacy prevents the exact-arithmetic n-step property, the solve
   * restarts after a complete Krylov cycle and continues while the certified
   * residual still decreases above roundoff.
   */
  Eigen::VectorXd solve(const Eigen::VectorXd& covector) const;

  /**
   * @brief Metric-invariant norm of a reduced gradient covector.
   *
   * For the pullback metric @f$G@f$, this returns
   * @f$\sqrt{g^T G^{-1}g}@f$.  Unlike a coordinate infinity norm, this is
   * invariant under a nonsingular change of reduced coordinates and is the
   * physical stationarity measure dual to `norm`.
   */
  double dual_norm(const Eigen::VectorXd& covector) const;

  double norm(const Eigen::VectorXd& reduced_step) const;
  Eigen::VectorXd clip_to_radius(
      const Eigen::VectorXd& reduced_step,
      double trust_radius) const;
private:
  const OrbitalChart& space_;
  const SparseParameterLayout& parameter_view_;
  OrbitalPhysicalMetric physical_metric_;
};

Eigen::VectorXd shrink_reduced_step_inside_trust_radius(
    const Eigen::VectorXd& reduced_step,
    double trust_radius,
    const NonredundantRetractionMetric& metric);

}  // namespace xmvb::vb
