#pragma once

#include <Eigen/Core>

#include "vbscf/optimization/coupled/accepted_point.hpp"
#include "vbscf/optimization/coupled/solver.hpp"

namespace xmvb::vb {

/** @brief Caller-imposed work limits for one accepted-point coupled solve. */
struct CoupledWorkspaceLimits {
  /** @brief Zero selects the complete orbital coordinate dimension. */
  int maximum_orbital_dimension = 0;
  /** @brief Zero selects the complete response coordinate dimension. */
  int maximum_response_dimension = 0;
};

/** @brief Terminal state of an accepted-point coupled solve. */
enum class CoupledWorkspaceStatus {
  Converged,
  WorkLimit,
  SubproblemFailure,
  NumericalFailure,
};

/** @brief Coupled step, certificate, and exact matrix-free work counters. */
struct CoupledWorkspaceResult {
  CoupledWorkspaceStatus status = CoupledWorkspaceStatus::NumericalFailure;
  CoupledSubspaceStep step;
  CoupledActionCounts action_counts;
  int orbital_dimension = 0;
  int response_dimension = 0;
  int expansions = 0;

  bool converged() const noexcept {
    return status == CoupledWorkspaceStatus::Converged;
  }
};

/**
 * @brief Accepted-point workspace for a matrix-free coupled Newton solve.
 *
 * The workspace owns the accepted model before constructing the subspace
 * solver that refers to it. Copy and move operations are disabled so the
 * operator address retained by the projection cache cannot be invalidated.
 * The accepted model supplies the physical orbital metric @f$G@f$; the two
 * explicit inverse actions supplied here are used only to generate trial
 * subspace directions.
 *
 * Initial orbital directions contain @f$-P_p^{-1}g@f$ and, when available,
 * preconditioned @f$B^TW@f$. Initial response directions contain an optional
 * response step recycled from the preceding accepted point,
 * @f$-P_q^{-1}r_s@f$, and preconditioned cached @f$BV@f$. Subsequent
 * directions are generated only from the unresolved full coupled KKT
 * residual. There is no fixed Krylov iteration count: without caller limits,
 * finite-dimensional algebraic rank is the only work bound.
 *
 * Re-solving at a different trust radius uses the retained projected blocks.
 * A call that requires no residual expansion evaluates no matrix-free action.
 */
class AcceptedPointCoupledWorkspace {
public:
  AcceptedPointCoupledWorkspace(
      AcceptedPointCoupledModel accepted_model,
      Eigen::VectorXd orbital_gradient,
      SymmetricOperatorAction apply_inverse_orbital_preconditioner,
      Eigen::MatrixXd recycled_response_directions = Eigen::MatrixXd());

  AcceptedPointCoupledWorkspace(
      const AcceptedPointCoupledWorkspace&) = delete;
  AcceptedPointCoupledWorkspace& operator=(
      const AcceptedPointCoupledWorkspace&) = delete;
  AcceptedPointCoupledWorkspace(AcceptedPointCoupledWorkspace&&) = delete;
  AcceptedPointCoupledWorkspace& operator=(
      AcceptedPointCoupledWorkspace&&) = delete;

  const AcceptedPointCoupledModel& accepted_model() const noexcept;
  const CoupledSubspaceSolver& subspace_solver() const noexcept;

  /**
   * @brief Solves and residual-expands the cached two-space projection.
   *
   * A positive explicit limit below algebraic completion is a work policy,
   * never a convergence condition. If it prevents a requested expansion, the
   * result is `WorkLimit` and retains the unresolved KKT residual.
   */
  CoupledWorkspaceResult solve(
      double trust_radius,
      const CoupledKktTolerances& tolerances,
      const CoupledWorkspaceLimits& limits = {});

private:
  int orbital_limit(const CoupledWorkspaceLimits& limits) const;
  int response_limit(const CoupledWorkspaceLimits& limits) const;
  bool initialize(const CoupledWorkspaceLimits& limits);
  CoupledWorkspaceResult result(
      CoupledWorkspaceStatus status,
      CoupledSubspaceStep step,
      int expansions) const;

  AcceptedPointCoupledModel accepted_model_;
  Eigen::VectorXd orbital_gradient_;
  SymmetricOperatorAction apply_inverse_orbital_preconditioner_;
  SymmetricOperatorAction apply_inverse_response_preconditioner_;
  Eigen::MatrixXd recycled_response_directions_;
  CoupledSubspaceSolver subspace_solver_;
  bool initialized_ = false;
};

}  // namespace xmvb::vb
