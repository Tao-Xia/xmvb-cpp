#pragma once

#include <Eigen/Core>

#include "vbscf/optimization/coupled/operator.hpp"
#include "vbscf/optimization/krylov/minres.hpp"

namespace xmvb::vb {

/** @brief Matrix-free correction for a nonstationary accepted structure state. */
struct StructureDefectCorrection {
  /** @brief Response displacement; stationary only when `converged()` is true. */
  Eigen::VectorXd response_correction;
  /**
   * @brief Orbital gradient @f$g+B^Tq_0@f$ at the corrected structure point.
   *
   * This vector is populated only when `converged()` is true.  A failed or
   * explicitly work-limited structure solve leaves it empty, so an
   * uncertified response cannot silently enter the orbital Newton equation.
   */
  Eigen::VectorXd corrected_orbital_gradient;
  /** @brief Remaining defect @f$e_0=r_s+Cq_0@f$. */
  Eigen::VectorXd remaining_structure_residual;
  /** @brief Explicitly certified matrix-free solve and work counters. */
  MinresResult linear_result;
  /** @brief Exact finite-solve model change @f$r_s^Tq_0+\tfrac12q_0^TCq_0@f$. */
  double exact_model_change = 0.0;
  /** @brief Stationary-limit value @f$\tfrac12r_s^Tq_0@f$. */
  double stationary_limit_model_change = 0.0;

  bool converged() const noexcept { return linear_result.converged(); }
};

/**
 * @brief Removes the accepted-point structure KKT defect before a Newton step.
 *
 * A finite inner eigensolve may leave a nonzero selected-subspace residual
 * @f$r_s@f$.  Expanding the coupled quadratic model about that point and
 * eliminating its structure-only stationary displacement gives
 *
 * @f[
 * Cq_0=-r_s,\qquad \widetilde g=g+B^Tq_0,
 * @f]
 *
 * For any finite-solve displacement, the exact structure-model change is
 *
 * @f[
 * \Delta m_s=r_s^Tq_0+\frac12q_0^TCq_0
 * =\frac12r_s^Tq_0+\frac12q_0^Te_0,
 * \qquad e_0=r_s+Cq_0.
 * @f]
 *
 * Only a certified solve may use @f$g+B^Tq_0@f$; otherwise the corrected
 * gradient is left empty rather than replaced by an arbitrary fallback.  At
 * convergence @f$e_0@f$ satisfies the requested residual certificate and the
 * model change approaches its stationary limit
 * @f$\tfrac12r_s^Tq_0@f$.  A zero structure residual returns exact zero
 * correction and the original orbital gradient without evaluating @f$C@f$ or
 * @f$B^T@f$.
 * With the default `MinresOptions::maximum_iterations == 0`, the algebraic
 * response dimension is the completeness limit; there is no empirical fixed
 * iteration budget.
 *
 * @param coupled_operator Accepted-point coupled Newton operator.
 * @param orbital_gradient Orbital gradient @f$g@f$ at the accepted point.
 * @param structure_residual Packed selected-subspace KKT residual @f$r_s@f$.
 * @param options MINRES residual certificate and optional explicit work cap.
 * @param apply_inverse_preconditioner Optional SPD response inverse action.
 */
StructureDefectCorrection correct_structure_defect(
    const CoupledNewtonOperator& coupled_operator,
    const Eigen::Ref<const Eigen::VectorXd>& orbital_gradient,
    const Eigen::Ref<const Eigen::VectorXd>& structure_residual,
    const MinresOptions& options,
    const SymmetricOperatorAction& apply_inverse_preconditioner = {});

}  // namespace xmvb::vb
