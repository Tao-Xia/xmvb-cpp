#pragma once

#include <functional>
#include <limits>

#include <Eigen/Core>

#include "vbscf/optimization/coupled/operator.hpp"
#include "vbscf/optimization/krylov/minres.hpp"

namespace xmvb::vb {

/** @brief Builds an SPD inverse preconditioner for one orbital shift. */
using ShiftedInversePreconditioner =
    std::function<SymmetricOperatorAction(double)>;

/** @brief Termination state of a coupled orbital trust-region solve. */
enum class CoupledSubproblemStopReason {
  InteriorConverged,
  BoundaryConverged,
  LinearSolveFailure,
  InvalidOrbitalMetric,
  HardCaseUnresolved,
  ShiftBracketFailure,
  BoundaryResolutionFailure,
};

/** @brief Accuracy and spectral contract for a coupled Newton subproblem. */
struct CoupledSubproblemOptions {
  MinresOptions minres;
  /** Relative tolerance for `| ||p||_G - radius |`. */
  double boundary_relative_tolerance = 1.0e-8;
  /**
   * Certified lower endpoint for the secular shift.
   *
   * The caller asserts that the relaxed orbital Hessian plus this multiple of
   * the metric is positive semidefinite. Zero is appropriate when the relaxed
   * Hessian is already known to be positive semidefinite. A positive endpoint
   * allows regular negative-curvature boundary solutions; an unresolved hard
   * case is reported rather than hidden behind an arbitrary fallback.
   */
  double convexifying_shift_lower_bound = 0.0;
};

/** @brief Certified stationary point of the coupled trust-region equations. */
struct CoupledSubproblemResult {
  Eigen::VectorXd step;
  Eigen::VectorXd orbital_step;
  Eigen::VectorXd response_step;
  Eigen::VectorXd orbital_metric_image;
  Eigen::VectorXd unshifted_operator_image;
  MinresResult linear_result;
  CoupledSubproblemStopReason stop_reason =
      CoupledSubproblemStopReason::LinearSolveFailure;
  double orbital_shift = 0.0;
  double orbital_norm = 0.0;
  double boundary_error = std::numeric_limits<double>::infinity();
  double complementarity_residual = std::numeric_limits<double>::infinity();
  double predicted_decrease = -std::numeric_limits<double>::infinity();
  int shifted_linear_solves = 0;
  int total_minres_iterations = 0;
  int total_operator_actions = 0;
  int total_preconditioner_actions = 0;
  int total_residual_checks = 0;

  bool converged() const noexcept {
    return stop_reason == CoupledSubproblemStopReason::InteriorConverged ||
        stop_reason == CoupledSubproblemStopReason::BoundaryConverged;
  }

  bool reached_boundary() const noexcept {
    return stop_reason == CoupledSubproblemStopReason::BoundaryConverged;
  }
};

/**
 * @brief Solves the coupled Newton equations with an orbital-only trust norm.
 *
 * For each secular shift this routine solves
 *
 * @f[
 * \begin{pmatrix}A+\lambda G&B^T\\B&C\end{pmatrix}
 * \begin{pmatrix}p\\z\end{pmatrix}
 * =-\begin{pmatrix}g\\0\end{pmatrix}
 * @f]
 *
 * by matrix-free MINRES. The response coordinates are not included in the
 * trust norm. A regular boundary root is bracketed and refined from certified
 * shifted linear solves until @f$\|p\|_G=\Delta@f$. There is no fixed Krylov
 * budget in this layer: the default MINRES completeness limit is the coupled
 * algebraic dimension, while a caller may explicitly supply a work budget.
 *
 * `convexifying_shift_lower_bound` is the second-order spectral certificate.
 * The solver does not claim a global trust-region solution without it. If the
 * lower endpoint exposes the classical hard case, the routine reports
 * `HardCaseUnresolved`; constructing the missing minimum-curvature vector is a
 * separate spectral operation.
 */
CoupledSubproblemResult solve_coupled_newton_subproblem(
    const CoupledNewtonOperator& coupled_operator,
    const Eigen::Ref<const Eigen::VectorXd>& orbital_gradient,
    double trust_radius,
    const CoupledSubproblemOptions& options,
    const ShiftedInversePreconditioner& make_inverse_preconditioner = {});

}  // namespace xmvb::vb
