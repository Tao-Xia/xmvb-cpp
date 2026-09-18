#pragma once

#include <Eigen/Core>

#include "vbscf/optimization/coupled/operator.hpp"

namespace xmvb::vb {

/** @brief Exact bookkeeping for one coupled quadratic-model trial. */
struct CoupledModelReduction {
  /** @brief Defect-corrected orbital gradient @f$\bar g=g+B^Tq_0@f$. */
  Eigen::VectorXd corrected_orbital_gradient;
  /** @brief @f$r_s^Tq_0+\tfrac12q_0^TCq_0@f$. */
  double baseline_defect_model_change = 0.0;
  /** @brief Model change from @f$(p,z)@f$ about the corrected baseline. */
  double incremental_model_change = 0.0;
  /** @brief Sum of baseline and incremental model changes. */
  double total_model_change = 0.0;
  /** @brief Negative baseline model change. */
  double baseline_defect_predicted_decrease = 0.0;
  /** @brief Negative incremental model change. */
  double incremental_predicted_decrease = 0.0;
  /** @brief Negative total model change, relative to the accepted point. */
  double total_predicted_decrease = 0.0;
  /** @brief Finite-defect term @f$e_0^Tz@f$. */
  double remaining_structure_linear_change = 0.0;
  /** @brief Incremental quadratic term @f$\tfrac12[p;z]^TK[p;z]@f$. */
  double incremental_quadratic_change = 0.0;
};

/**
 * @brief Evaluates the exact coupled quadratic prediction with structure defect.
 *
 * The accepted model is
 *
 * @f[
 * m(p,q)=g^Tp+r_s^Tq+
 * \frac12[p;q]^TK[p;q].
 * @f]
 *
 * For a structure correction @f$q_0@f$, remaining residual
 * @f$e_0=r_s+Cq_0@f$, and incremental response @f$z@f$, the exact split is
 *
 * @f[
 * \delta_0=r_s^Tq_0+\frac12q_0^TCq_0,
 * @f]
 * @f[
 * \delta_{\rm inc}=(g+B^Tq_0)^Tp+e_0^Tz+
 * \frac12[p;z]^TK[p;z],
 * @f]
 * @f[
 * \operatorname{pred}_{\rm total}=-(\delta_0+\delta_{\rm inc}).
 * @f]
 *
 * `incremental_operator_image` must be the unshifted action
 * @f$K[p;z]@f$. The response Hessian image of @f$q_0@f$ is reconstructed
 * exactly as @f$Cq_0=e_0-r_s@f$, while @f$B^Tq_0@f$ is evaluated through the
 * matrix-free coupled operator.
 */
CoupledModelReduction evaluate_coupled_model_reduction(
    const CoupledNewtonOperator& coupled_operator,
    const Eigen::Ref<const Eigen::VectorXd>& accepted_orbital_gradient,
    const Eigen::Ref<const Eigen::VectorXd>& structure_residual,
    const Eigen::Ref<const Eigen::VectorXd>& response_correction,
    const Eigen::Ref<const Eigen::VectorXd>& remaining_structure_residual,
    const Eigen::Ref<const Eigen::VectorXd>& orbital_step,
    const Eigen::Ref<const Eigen::VectorXd>& response_step,
    const Eigen::Ref<const Eigen::VectorXd>& incremental_operator_image);

}  // namespace xmvb::vb
