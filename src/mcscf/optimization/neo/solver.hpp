#pragma once

#include <Eigen/Core>

#include "mcscf/optimization/problem.hpp"

namespace xmvb::mcscf {

/** @brief Termination state of a matrix-free NEO microiteration. */
enum class NeoStopReason {
    Converged,
    /** Approximate candidate requiring nonlinear keyframe validation. */
    TrialReady,
    SubspaceLimit,
    NumericalFailure,
    NegativeCurvature,
};

/** @brief Accuracy and trust-region controls for a NEO microproblem. */
struct NeoOptions {
    /** Radius measured in the physical coordinate metric. */
    double trust_radius = 1.0;
    /** Relative tolerance for the physical dual-norm KKT residual. */
    double relative_residual_tolerance = 1.0e-8;
    /** Absolute tolerance for the explicit KKT residual. */
    double absolute_residual_tolerance = 0.0;
    /**
   * Maximum physical subspace dimension. Zero selects the full algebraic
   * dimension; a positive smaller value is an explicit caller work budget.
   */
    int maximum_subspace_dimension = 0;
    /**
   * Require convergence of the lowest-curvature Ritz residual in addition
   * to the Newton KKT residual.  This is useful for spectral diagnostics and
   * hard-case certification.  Keyframe CIAH steps may disable it: their
   * nonlinear gradient refresh is the acceptance certificate, so resolving
   * an unrelated extremal eigenvector after the Newton equation has already
   * converged only oversolves a stale quadratic model.
   */
    bool require_curvature_certificate = true;
    /**
   * Stop with a negative-curvature witness for an exactly zero gradient,
   * once its directly evaluated Rayleigh quotient clears the numerical error
   * margin. No extremal-eigenvector or nonlinear-step convergence is implied.
   */
    bool stop_on_negative_curvature = false;
};

/** @brief Step, spectral witness, and optimality diagnostics returned by NEO. */
struct NeoResult {
    Eigen::VectorXd step;
    Eigen::VectorXd hessian_step;
    Eigen::VectorXd metric_step;
    Eigen::VectorXd kkt_residual;
    /** Lowest Ritz direction, normalized in the problem's supplied metric. */
    Eigen::VectorXd minimum_curvature_direction;
    /** Matrix-free Hessian action on `minimum_curvature_direction`. */
    Eigen::VectorXd minimum_curvature_hessian_image;
    /** Matrix-free metric action on `minimum_curvature_direction`. */
    Eigen::VectorXd minimum_curvature_metric_image;
    /** Direct Rayleigh quotient of `minimum_curvature_direction`. */
    double minimum_curvature_value = 0.0;
    /** `[beta, y]` in the scaled augmented-Hessian equation. */
    Eigen::VectorXd augmented_eigenvector;
    double shift = 0.0;
    double predicted_reduction = 0.0;
    double step_norm = 0.0;
    /** Physical dual norm of `gradient + H * step + shift * M * step`. */
    double residual_norm = 0.0;
    /** Absolute plus gradient-relative dual-norm accuracy for `residual_norm`. */
    double residual_target = 0.0;
    /** Euclidean spectral residual of the lowest generalized-Hessian Ritz pair. */
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
 * `Converged` always requires an explicit full-space KKT residual.
 * Lowest-curvature Ritz convergence is additionally required when requested
 * by the caller and for a hard-case step. A regular boundary candidate is
 * certified by its exact nonlinear keyframe instead of an unrelated extremal
 * Ritz vector. As in matrix-free Davidson/Lanczos, lowest-root identification
 * assumes that the generic starting probe overlaps the lowest eigenspace. A
 * complete basis or a caller-supplied spectral lower bound supplies the
 * stronger rigorous global certificate reported in `NeoResult`. The
 * physical metric enters the trust-region norm, projected generalized
 * eigenproblem, and dual KKT certificate; an optional preconditioner only
 * proposes new subspace directions.
 * `stop_on_negative_curvature` is a zero-gradient spectral probe: it may
 * return a verified negative Rayleigh direction before KKT convergence.
 * A nonnegative Ritz value from an incomplete subspace is not a global
 * positive-semidefiniteness certificate.
 * With `allow_early_trial`, a coordinate-residual heuristic may return
 * TrialReady, never Converged. The caller must validate an actual nonlinear
 * keyframe and disable this heuristic near final stationarity.
 */
NeoResult solve_neo(const QuadraticProblem& problem,
                    const NeoOptions& options,
                    bool allow_early_trial = false);

} // namespace xmvb::mcscf
