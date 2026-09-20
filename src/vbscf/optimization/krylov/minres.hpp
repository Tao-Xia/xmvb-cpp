#pragma once

#include <functional>

#include <Eigen/Core>

namespace xmvb::vb {

/** @brief Matrix-free action of a square linear operator. */
using SymmetricOperatorAction =
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;

/** @brief Termination state of a symmetric MINRES solve. */
enum class MinresStopReason {
  Converged,
  IterationLimit,
  LanczosBreakdown,
  NonPositivePreconditioner,
};

/** @brief Accuracy and algebraic work limit for matrix-free MINRES. */
struct MinresOptions {
  /** Relative Euclidean residual tolerance, measured against `||rhs||_2`. */
  double relative_residual_tolerance = 1.0e-8;
  /** Absolute Euclidean residual tolerance. */
  double absolute_residual_tolerance = 0.0;
  /**
   * Maximum Lanczos iterations. Zero selects the algebraic dimension, which
   * is the exact-arithmetic completeness limit rather than an empirical cap.
   */
  int maximum_iterations = 0;
};

/** @brief Certified result of a matrix-free symmetric-indefinite solve. */
struct MinresResult {
  Eigen::VectorXd solution;
  /** Explicit final operator image `A * solution`. */
  Eigen::VectorXd operator_image;
  /** Explicit final residual `rhs - operator_image`. */
  Eigen::VectorXd residual;
  MinresStopReason stop_reason = MinresStopReason::IterationLimit;
  int iterations = 0;
  int operator_actions = 0;
  int preconditioner_actions = 0;
  int residual_checks = 0;
  int reliable_restarts = 0;
  double rhs_norm = 0.0;
  double residual_norm = 0.0;
  double residual_target = 0.0;
  double estimated_preconditioned_residual_norm = 0.0;

  /** @brief Whether the explicitly evaluated Euclidean residual is certified. */
  bool converged() const noexcept {
    return stop_reason == MinresStopReason::Converged;
  }
};

/**
 * @brief Solves a symmetric, possibly indefinite linear system with MINRES.
 *
 * `apply_inverse_preconditioner` must be symmetric positive definite. Passing
 * an empty action selects the identity. Convergence is never inferred from the
 * Lanczos/Givens estimate alone: every successful result is certified with an
 * explicit Euclidean residual `rhs - A * solution`. If a finite-precision
 * recurrence underestimates that residual, MINRES restarts from the explicitly
 * evaluated residual while preserving the accumulated solution.
 *
 * A symmetric block operator may shift only its leading block,
 * `A_lambda(p,z) = ((A + lambda M)p + B^T z, Bp + Cz)`. MINRES remains
 * independent of the block interpretation and leaves `lambda` to the caller.
 */
MinresResult solve_symmetric_minres(
    const SymmetricOperatorAction& apply_operator,
    const Eigen::Ref<const Eigen::VectorXd>& rhs,
    const MinresOptions& options,
    const SymmetricOperatorAction& apply_inverse_preconditioner = {},
    const Eigen::VectorXd& initial_guess = Eigen::VectorXd());

}  // namespace xmvb::vb
