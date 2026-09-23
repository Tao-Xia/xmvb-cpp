#pragma once

#include <functional>
#include <Eigen/Core>

#include "vbscf/optimization/neo/solver.hpp"

namespace xmvb::vb {

/** @brief Vector in separated orbital and horizontal-structure coordinates. */
struct ResponseNeoDirection {
  Eigen::VectorXd orbital;
  Eigen::VectorXd structure;
};

/** @brief Coupled Hessian images for a block of directions. */
struct ResponseNeoDirectionBlock {
  Eigen::MatrixXd orbital;
  Eigen::MatrixXd structure;
};

/** @brief One block column of the coupled orbital--structure Hessian. */
using ResponseNeoBlockAction =
    std::function<ResponseNeoDirection(const Eigen::VectorXd&)>;
using ResponseNeoMatrixAction =
    std::function<ResponseNeoDirectionBlock(const Eigen::MatrixXd&)>;
using ResponseNeoPreconditioner =
    std::function<Eigen::VectorXd(const Eigen::VectorXd&, double)>;

/**
 * @brief Quadratic model for orbital-trust NEO with projected response.
 *
 * The coupled action represents
 * @f$[\bar A p+B^Tz,\;Bp+Cz]@f$.  Only the orbital metric defines the
 * trust region.  The separate core action applies only @f$\bar A@f$, so
 * @f$Bp@f$ can be delayed until an actual orbital candidate exists.  The
 * optional preconditioner acts only on orbital covectors.
 */
class ResponseNeoProblem {
public:
  ResponseNeoProblem(
      Eigen::VectorXd orbital_gradient,
      Eigen::Index structure_size,
      NeoAction apply_orbital_core,
      ResponseNeoBlockAction apply_orbital_coupling,
      ResponseNeoBlockAction apply_structure_coupling,
      NeoAction apply_orbital_metric,
      ResponseNeoPreconditioner apply_orbital_preconditioner = {},
      Eigen::VectorXd initial_orbital_guess = {},
      double operator_relative_accuracy = 0.0,
      ResponseNeoMatrixAction apply_structure_coupling_block = {});

  Eigen::Index orbital_size() const noexcept {
    return orbital_gradient_.size();
  }
  Eigen::Index structure_size() const noexcept { return structure_size_; }
  const Eigen::VectorXd& orbital_gradient() const noexcept {
    return orbital_gradient_;
  }

  Eigen::VectorXd apply_orbital_core(
      const Eigen::VectorXd& direction) const;
  ResponseNeoDirection apply_orbital_coupling(
      const Eigen::VectorXd& direction) const;
  ResponseNeoDirection apply_structure_coupling(
      const Eigen::VectorXd& direction) const;
  ResponseNeoDirectionBlock apply_structure_coupling_block(
      const Eigen::MatrixXd& directions) const;
  Eigen::VectorXd apply_orbital_metric(
      const Eigen::VectorXd& direction) const;
  bool has_orbital_preconditioner() const noexcept {
    return static_cast<bool>(apply_orbital_preconditioner_);
  }
  Eigen::VectorXd apply_orbital_preconditioner(
      const Eigen::VectorXd& covector,
      double shift) const;
  /** @brief Initial Davidson direction recycled within one nonlinear macro. */
  const Eigen::VectorXd& initial_orbital_guess() const noexcept {
    return initial_orbital_guess_;
  }
  double operator_relative_accuracy() const noexcept {
    return operator_relative_accuracy_;
  }

private:
  Eigen::VectorXd apply_orbital_checked(
      const NeoAction& action,
      const Eigen::VectorXd& vector,
      const char* message) const;

  const Eigen::VectorXd orbital_gradient_;
  const Eigen::Index structure_size_;
  ResponseNeoDirection apply_block_checked(
      const ResponseNeoBlockAction& action,
      const Eigen::VectorXd& vector,
      Eigen::Index expected_size) const;

  const NeoAction apply_orbital_core_;
  const ResponseNeoBlockAction apply_orbital_coupling_;
  const ResponseNeoBlockAction apply_structure_coupling_;
  const ResponseNeoMatrixAction apply_structure_coupling_block_;
  const NeoAction apply_orbital_metric_;
  const ResponseNeoPreconditioner apply_orbital_preconditioner_;
  const Eigen::VectorXd initial_orbital_guess_;
  const double operator_relative_accuracy_;
};

/** @brief Step and explicit residuals of projected-response NEO. */
struct ResponseNeoResult {
  ResponseNeoDirection step;
  ResponseNeoDirection hessian_step;
  Eigen::VectorXd orbital_metric_step;
  ResponseNeoDirection kkt_residual;
  Eigen::VectorXd minimum_curvature_orbital;
  double shift = 0.0;
  double predicted_reduction = 0.0;
  double step_norm = 0.0;
  double residual_norm = 0.0;
  double residual_target = 0.0;
  double curvature_residual_norm = 0.0;
  double curvature_residual_target = 0.0;
  int iterations = 0;
  int coupled_actions = 0;
  int orbital_actions = 0;
  /** Orbital actions that also formed the expensive structure forcing Bp. */
  int structure_forcing_actions = 0;
  int structure_actions = 0;
  bool boundary = false;
  bool hard_case = false;
  bool global_curvature_certified = false;
  NeoStopReason stop_reason = NeoStopReason::SubspaceLimit;

  bool converged() const noexcept {
    return stop_reason == NeoStopReason::Converged;
  }
};

/**
 * @brief Reusable projected-response NEO subspace at one accepted point.
 *
 * The workspace owns the independent orbital and structure bases together
 * with every metric and Hessian image used to form the projected model.
 * Changing only the trust radius therefore requires no repeated large-space
 * action.  A solve may enrich the bases when its residual certificates need
 * more information, but it never removes an accepted image.
 */
class ResponseNeoWorkspace {
public:
  explicit ResponseNeoWorkspace(const ResponseNeoProblem& problem)
      : problem_(problem),
        orbital_basis_(problem.orbital_size(), 0),
        orbital_metric_images_(problem.orbital_size(), 0),
        orbital_orbital_images_(problem.orbital_size(), 0),
        orbital_structure_images_(problem.structure_size(), 0),
        structure_basis_(problem.structure_size(), 0),
        structure_orbital_images_(problem.orbital_size(), 0),
        structure_structure_images_(problem.structure_size(), 0) {}
  /** @brief Solves one trust-region subproblem using the cached subspace. */
  ResponseNeoResult solve(const NeoOptions& options);

  Eigen::Index orbital_basis_size() const noexcept {
    return orbital_basis_.cols();
  }
  Eigen::Index structure_basis_size() const noexcept {
    return structure_basis_.cols();
  }

private:
  bool append_orbital(
      Eigen::VectorXd direction,
      int* actions,
      int* orbital_actions,
      int* structure_forcing_actions);
  bool append_structure(
      Eigen::VectorXd direction,
      int* actions,
      int* structure_actions);
  bool append_structure_block(
      Eigen::MatrixXd directions,
      int* actions,
      int* structure_actions);
  void store_structure(
      const Eigen::VectorXd& direction,
      const ResponseNeoDirection& image);
  void activate_coupling(
      int* actions,
      int* orbital_actions,
      int* structure_forcing_actions);
  bool certify_core_candidate(
      ResponseNeoResult* result,
      double residual_target,
      int maximum_dimension,
      bool require_curvature_certificate);
  void retain_single_direction_candidate(
      const ResponseNeoDirection& candidate,
      const Eigen::VectorXd& step);
  const ResponseNeoProblem& problem_;
  Eigen::MatrixXd orbital_basis_;
  Eigen::MatrixXd orbital_metric_images_;
  Eigen::MatrixXd orbital_orbital_images_;
  Eigen::MatrixXd orbital_structure_images_;
  Eigen::MatrixXd structure_basis_;
  Eigen::MatrixXd structure_orbital_images_;
  Eigen::MatrixXd structure_structure_images_;
  Eigen::MatrixXd projected_orbital_;
  Eigen::MatrixXd projected_metric_;
  Eigen::MatrixXd projected_coupling_;
  Eigen::MatrixXd projected_coupling_adjoint_;
  Eigen::MatrixXd projected_structure_;
  bool coupling_active_ = false;
  Eigen::Index orbital_canonical_ = 0;
  Eigen::Index structure_canonical_ = 0;
};

/**
 * @brief Solves the coupled model after projected structure-response removal.
 *
 * The initial orbital basis projects only @f$\bar A@f$.  Once its projected
 * solve is stationary, the workspace forms @f$Bp@f$ for that candidate,
 * solves @f$Cq=-Bp@f$ direction-locally, and checks both coupled KKT
 * residuals.  Only a failed certificate promotes the retained orbital basis
 * to the full @f$\bar A,B,C@f$ workspace.  Lowest-curvature certification is
 * optional except in a hard case.
 */
ResponseNeoResult solve_response_neo(
    const ResponseNeoProblem& problem,
    const NeoOptions& options);

}  // namespace xmvb::vb
