#pragma once

#include <Eigen/Core>

#include "vbscf/optimization/neo/problem.hpp"

namespace xmvb::vb {

/** @brief Termination state of a matrix-free NEO microiteration. */
enum class NeoStopReason {
  Converged,
  SubspaceLimit,
  NumericalFailure,
};

/** @brief Accuracy and trust-region controls for a NEO microproblem. */
struct NeoOptions {
  /** Radius measured in the physical coordinate metric. */
  double trust_radius = 1.0;
  /** Relative tolerance for the explicit KKT residual. */
  double relative_residual_tolerance = 1.0e-8;
  /** Absolute tolerance for the explicit KKT residual. */
  double absolute_residual_tolerance = 0.0;
  /**
   * Maximum physical subspace dimension. Zero selects the full algebraic
   * dimension; a positive smaller value is an explicit caller work budget.
   */
  int maximum_subspace_dimension = 0;
};

/** @brief Step and explicit optimality certificate returned by NEO. */
struct NeoResult {
  Eigen::VectorXd step;
  Eigen::VectorXd hessian_step;
  Eigen::VectorXd metric_step;
  Eigen::VectorXd kkt_residual;
  /** `[beta, y]` in the scaled augmented-Hessian equation. */
  Eigen::VectorXd augmented_eigenvector;
  double shift = 0.0;
  double predicted_reduction = 0.0;
  double step_norm = 0.0;
  double residual_norm = 0.0;
  double residual_target = 0.0;
  double augmented_eigenvalue = 0.0;
  double gradient_scale = 0.0;
  int iterations = 0;
  int hessian_actions = 0;
  bool boundary = false;
  bool hard_case = false;
  /** Whether the augmented vector represents the complete NEO step. */
  bool augmented_certificate_valid = false;
  NeoStopReason stop_reason = NeoStopReason::SubspaceLimit;

  /** @brief Whether the explicitly evaluated KKT residual is certified. */
  bool converged() const noexcept {
    return stop_reason == NeoStopReason::Converged;
  }
};

/**
 * @brief Solves a generalized-metric NEO trust-region microproblem.
 *
 * The method builds an M-orthonormal matrix-free subspace, solves the exact
 * projected norm-constrained quadratic problem, and expands with the mapped
 * KKT or lowest-curvature Ritz residual. A successful return is certified by
 * an explicit full-space KKT residual in the inverse-metric norm.
 */
NeoResult solve_neo(const NeoProblem& problem, const NeoOptions& options);

}  // namespace xmvb::vb
