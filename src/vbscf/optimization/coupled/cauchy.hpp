#pragma once

#include <limits>

#include <Eigen/Core>

#include "vbscf/optimization/coupled/operator.hpp"
#include "vbscf/optimization/krylov/minres.hpp"

namespace xmvb::vb {

/** @brief Certification state of a relaxed orbital Cauchy incumbent. */
enum class RelaxedCauchyStopReason {
  CauchyDecreaseCertified,
  InvalidInverseMetric,
  NonDescentDirection,
  InvalidOrbitalMetric,
  ResponseLiftFailure,
  NonPositivePredictedDecrease,
};

/** @brief Matrix-free relaxed Cauchy step and its response certificate. */
struct RelaxedCauchyResult {
  Eigen::VectorXd orbital_direction;
  Eigen::VectorXd response_direction;
  Eigen::VectorXd orbital_step;
  Eigen::VectorXd response_step;
  Eigen::VectorXd orbital_metric_image;
  /** @brief Explicit remaining response equation residual @f$Bd+Cy@f$. */
  Eigen::VectorXd remaining_response_residual;
  MinresResult response_linear_result;
  RelaxedCauchyStopReason stop_reason =
      RelaxedCauchyStopReason::ResponseLiftFailure;
  double directional_derivative = 0.0;
  /** @brief Stationary-limit curvature @f$d^T(Ad+B^Ty)@f$. */
  double stationary_limit_curvature = 0.0;
  /** @brief Exact returned-ray curvature @f$[d;y]^TK[d;y]@f$. */
  double coupled_ray_curvature = 0.0;
  double orbital_direction_norm = 0.0;
  double maximum_step_length = 0.0;
  double step_length = 0.0;
  double predicted_decrease = -std::numeric_limits<double>::infinity();

  /** @brief True only for an explicitly certified positive Cauchy decrease. */
  bool certified() const noexcept {
    return stop_reason ==
        RelaxedCauchyStopReason::CauchyDecreaseCertified;
  }
};

/**
 * @brief Builds a sufficient-decrease incumbent for the relaxed orbital model.
 *
 * With the defect-corrected orbital gradient @f$\bar g@f$, this routine forms
 * the inverse-metric direction @f$d=-G^{-1}\bar g@f$ and obtains its induced
 * structure response from the explicitly certified matrix-free solve
 *
 * @f[
 * Cy=-Bd.
 * @f]
 *
 * The stationary-limit directional curvature and trust-boundary length are
 *
 * @f[
 * \kappa_\star=d^T(Ad+B^Ty),\qquad
 * \alpha_{\max}=\Delta/\lVert d\rVert_G.
 * @f]
 *
 * A finite response solve leaves @f$e=Bd+Cy@f$. The exact curvature of the
 * actually returned coupled ray is therefore
 *
 * @f[
 * \kappa=[d;y]^TK[d;y]=\kappa_\star+y^Te,
 * @f]
 *
 * and this exact value, not @f$\kappa_\star@f$, defines the predicted decrease.
 * The returned length minimizes the quadratic model on this ray:
 * @f$\alpha=\alpha_{\max}@f$ for @f$\kappa\leq0@f$, otherwise
 * @f$\alpha=\min(\alpha_{\max},-\bar g^Td/\kappa)@f$.  Certification asserts
 * only positive Cauchy decrease. It makes no claim that the relaxed Hessian is
 * positive semidefinite or that the step globally solves the trust-region
 * subproblem.
 *
 * @param coupled_operator Accepted coupled Newton block operator.
 * @param corrected_orbital_gradient Defect-corrected gradient @f$\bar g@f$.
 * @param trust_radius Positive orbital trust radius @f$\Delta@f$.
 * @param response_options Explicit residual target for the @f$C@f$ solve.
 * @param apply_inverse_orbital_metric SPD action @f$G^{-1}v@f$.
 * @param apply_response_inverse_preconditioner Optional SPD MINRES inverse.
 */
RelaxedCauchyResult build_relaxed_cauchy_incumbent(
    const CoupledNewtonOperator& coupled_operator,
    const Eigen::Ref<const Eigen::VectorXd>& corrected_orbital_gradient,
    double trust_radius,
    const MinresOptions& response_options,
    const SymmetricOperatorAction& apply_inverse_orbital_metric,
    const SymmetricOperatorAction&
        apply_response_inverse_preconditioner = {});

}  // namespace xmvb::vb
