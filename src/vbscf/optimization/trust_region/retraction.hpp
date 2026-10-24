#pragma once

#include <functional>
#include <mutex>
#include <Eigen/Core>

#include "vbscf/orbitals/charts/chart.hpp"
#include "vbscf/orbitals/charts/layout.hpp"
#include "vbscf/orbitals/charts/physical_metric.hpp"
#include "vbscf/orbitals/preparation/input.hpp"

namespace xmvb::vb {

Eigen::VectorXd
gather_nonredundant_retract_tangent(const OrbitalPreparationInput& orbital_preparation_input,
                                    const OrbitalChart& space,
                                    const SparseParameterLayout& parameter_view,
                                    const Eigen::VectorXd& reduced_step);

/**
 * @brief Accepted-point pullback metric of the additive sparse retraction.
 *
 * The algebraic quotient chart supplies the horizontal raw-coefficient lift.
 * This object measures that same lift after the inactive-subspace and active
 * physical map. Complete OEO uses the active span; a structure subset uses
 * individual rays. It never stores a full reduced metric matrix. The chart is
 * prewhitened only in independent local orbital-normalization metrics, so the
 * complete coupled reduced metric is generally not the identity. Callers must
 * use `apply` for the metric covector and `norm` for the physical tangent norm;
 * only a small projected metric may be assembled and factorized.
 */
class NonredundantRetractionMetric {
  public:
    NonredundantRetractionMetric(const OrbitalChart& space,
                                 const SparseParameterLayout& parameter_view,
                                 const OrbitalPreparationInput& input);

    Eigen::VectorXd tangent(const Eigen::VectorXd& reduced_step) const;
    Eigen::VectorXd apply(const Eigen::VectorXd& reduced_step) const;

    /**
   * @brief Applies the inverse Riesz map without assembling the metric.
   *
   * Solves @f$G y=b@f$ by PCG with exact orbital-diagonal metric blocks,
   * matrix-free coupled actions and recursive residuals. True residuals certify convergence;
   * a residual gap or a complete Krylov cycle triggers a restart, which
   * continues only while the certified residual decreases above roundoff.
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
    Eigen::VectorXd clip_to_radius(const Eigen::VectorXd& reduced_step, double trust_radius) const;

  private:
    const OrbitalChart& space_;
    const SparseParameterLayout& parameter_view_;
    OrbitalPhysicalMetric physical_metric_;
    /** @brief Shared lazy factorization: analytic dual norms need no PCG blocks. */
    mutable std::once_flag block_inverse_once_;
    mutable std::function<Eigen::VectorXd(const Eigen::VectorXd&)> block_inverse_;
};

Eigen::VectorXd shrink_reduced_step_inside_trust_radius(const Eigen::VectorXd& reduced_step,
                                                        double trust_radius,
                                                        const NonredundantRetractionMetric& metric);

} // namespace xmvb::vb
