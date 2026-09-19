#pragma once

#include <limits>

#include <Eigen/Core>

#include "vbscf/optimization/coupled/projected_model.hpp"
#include "vbscf/optimization/coupled/projection_cache.hpp"
#include "vbscf/optimization/krylov/minres.hpp"

namespace xmvb::vb {

/** @brief Full-space stopping state of a two-space coupled Newton solve. */
enum class CoupledSubspaceStatus {
  Converged,
  ExpandOrbitalSpace,
  ExpandResponseSpace,
  ExpandBothSpaces,
  EmptyOrbitalSpace,
  ResponseOperatorSingular,
  ProjectedTrustRegionFailure,
  InsufficientPredictedDecrease,
  NumericalFailure,
};

/** @brief Explicit full-space KKT tolerances for a coupled Newton step. */
struct CoupledKktTolerances {
  double orbital = 0.0;
  double response = 0.0;
};

/** @brief One radius-dependent solution reconstructed from cached actions. */
struct CoupledSubspaceStep {
  CoupledSubspaceStatus status = CoupledSubspaceStatus::NumericalFailure;
  CoupledProjectedModelResult projected;
  Eigen::VectorXd orbital_step;
  Eigen::VectorXd response_step;
  Eigen::VectorXd orbital_kkt_residual;
  Eigen::VectorXd response_kkt_residual;
  /** @brief Inf-sup repair direction when a projected response block is singular. */
  Eigen::VectorXd response_expansion_candidate;
  double orbital_backward_error = std::numeric_limits<double>::infinity();
  double response_backward_error = std::numeric_limits<double>::infinity();
  double model_change = std::numeric_limits<double>::quiet_NaN();
  double predicted_decrease = -std::numeric_limits<double>::infinity();

  bool converged() const noexcept {
    return status == CoupledSubspaceStatus::Converged;
  }
};

/** @brief Numerical ranks added by one residual-driven expansion. */
struct CoupledSubspaceExpansion {
  int orbital_rank = 0;
  int response_rank = 0;

  bool progressed() const noexcept {
    return orbital_rank != 0 || response_rank != 0;
  }
};

/**
 * @brief Matrix-free two-space coordinator for a coupled Newton subproblem.
 *
 * The solver owns a @f$G@f$-orthonormal orbital space @f$V@f$ and a Euclidean
 * response space @f$W@f$. Expanding @f$V@f$ evaluates one block each of
 * @f$G@f$, @f$A@f$, and @f$B@f$; expanding @f$W@f$ evaluates one block each
 * of @f$B^T@f$ and @f$C@f$. Calling `solve` at any number of trust radii uses
 * only these cached images and performs no matrix-free action.
 *
 * Completion is decided from the full-space shifted KKT equations
 *
 * @f[
 * r_p=g+Ap+B^Tq+\lambda Gp,
 * \qquad r_q=r_s+Bp+Cq.
 * @f]
 *
 * The reported backward errors scale each residual by the sum of the norms
 * of the terms in its own equation. Consequently no iteration count, system
 * identity, or empirical work budget enters the convergence decision. A
 * nonconverged result identifies which space must be expanded; the residuals
 * themselves are exposed so the caller can apply its chosen inverse
 * preconditioner before appending the next block.
 *
 * The referenced operator must outlive this solver.
 */
class CoupledSubspaceSolver {
public:
  CoupledSubspaceSolver(
      const CoupledNewtonOperator& coupled_operator,
      Eigen::VectorXd orbital_gradient,
      Eigen::VectorXd response_residual);

  const CoupledProjectionCache& cache() const noexcept;

  int append_orbital_block(
      const Eigen::Ref<const Eigen::MatrixXd>& candidates);
  int append_response_block(
      const Eigen::Ref<const Eigen::MatrixXd>& candidates);

  /**
   * @brief Solves the cached projection and certifies its full-space residual.
   *
   * Repeated calls, including calls with a changed trust radius, perform no
   * matrix-free operator action. `incumbent_predicted_decrease` is the exact
   * decrease of a certified feasible step retained in the supplied spaces; a
   * returned curvature-enhanced step may not be worse than that value.
   */
  CoupledSubspaceStep solve(
      double trust_radius,
      const CoupledKktTolerances& tolerances,
      double incumbent_predicted_decrease = 0.0) const;

  /**
   * @brief Expands the spaces selected by the preceding full KKT audit.
   *
   * The two inverse actions are explicit algorithm inputs; there is no hidden
   * identity fallback. A singular projected response block instead uses its
   * cached inf-sup repair direction. If a requested expansion has zero
   * numerical rank, `progressed()` is false and the caller must report a
   * numerical breakdown rather than convergence.
   */
  CoupledSubspaceExpansion expand(
      const CoupledSubspaceStep& step,
      const SymmetricOperatorAction& apply_inverse_orbital_preconditioner,
      const SymmetricOperatorAction& apply_inverse_response_preconditioner);

private:
  const CoupledNewtonOperator* coupled_operator_ = nullptr;
  Eigen::VectorXd orbital_gradient_;
  Eigen::VectorXd response_residual_;
  CoupledProjectionCache cache_;
};

}  // namespace xmvb::vb
