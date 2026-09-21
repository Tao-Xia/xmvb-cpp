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
  /** Relative tolerance for the scale-aware Euclidean KKT residual. */
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
  /** Euclidean norm of `gradient + H * step + shift * M * step`. */
  double residual_norm = 0.0;
  /** Scale-aware Euclidean accuracy requested for `residual_norm`. */
  double residual_target = 0.0;
  /** Full-space residual of the lowest generalized-Hessian Ritz pair. */
  double curvature_residual_norm = 0.0;
  /** Requested accuracy for `curvature_residual_norm`. */
  double curvature_residual_target = 0.0;
  double augmented_eigenvalue = 0.0;
  double gradient_scale = 0.0;
  int iterations = 0;
  int hessian_actions = 0;
  bool boundary = false;
  bool hard_case = false;
  /**
   * Whether shifted-Hessian positivity has a rigorous global certificate.
   * Otherwise it uses the standard iterative-eigensolver assumption that the
   * generic starting probe overlaps the lowest eigenspace.
   */
  bool global_curvature_certified = false;
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
 * The method builds a Euclidean-orthonormal action subspace, whitens its small
 * projected physical metric, solves the projected norm-constrained quadratic
 * problem, and expands with the KKT or lowest-curvature Ritz residual. A
 * successful return requires an explicit full-space KKT residual and a
 * converged full-space residual of the lowest generalized-Hessian Ritz pair.
 * As in matrix-free Davidson/Lanczos, lowest-root identification assumes that
 * the generic starting probe overlaps the lowest eigenspace. A complete basis
 * or a caller-supplied spectral lower bound supplies the stronger rigorous
 * global certificate reported in `NeoResult`. The
 * physical metric enters only through the trust-region norm and projected
 * generalized eigenproblem; an optional preconditioner only proposes new
 * subspace directions.
 */
NeoResult solve_neo(const NeoProblem& problem, const NeoOptions& options);

}  // namespace xmvb::vb
