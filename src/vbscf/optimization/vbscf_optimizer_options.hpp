#pragma once

#include <functional>

#include "vbscf/optimization/optimizer_backend.hpp"
#include "vbscf/optimization/vbscf_optimizer_result.hpp"

namespace xmvb::vb {

/**
 * @brief Options controlling the C++ VBSCF orbital optimization loop.
 */
struct VbScfOptimizerOptions {
  /**
   * @brief Optimization backend implementation.
   */
  VbScfOptimizerBackend backend =
      VbScfOptimizerBackend::NonredundantLbfgspp;

  /**
   * @brief Maximum number of accepted optimization iterations.
   *
   * Standalone `.xmi` runs override this from the input `itmax` keyword when
   * present; otherwise the project default is 2000.
   */
  int max_iterations = 2000;

  /**
   * @brief Convergence threshold on the gradient infinity norm.
   *
   * The plain full-space `lbfgspp` backend uses
   * the full gradient Euclidean norm, matching the original `gxn`
   * convergence check. Nonredundant backends apply the same threshold to the
   * projected / reduced gradient infinity norm.
   */
  double gradient_tolerance = 1.0e-3;

  /**
   * @brief Convergence threshold on the absolute energy change.
   *
   * This is the stopping criterion for the reported projected gradient.
   */
  double energy_tolerance = 1.0e-5;

  /**
   * @brief Maximum trial step size allowed by the line search.
   */
  double initial_step_size = 1.0;

  /**
   * @brief Minimum allowed step size before terminating the line search.
   */
  double minimum_step_size = 1.0e-7;

  /**
   * @brief Armijo sufficient-decrease constant.
   */
  double armijo_constant = 1.0e-4;

  /**
   * @brief Number of L-BFGS correction vectors.
   */
  int history_size = 100;

  /**
   * @brief Maximum Krylov iterations for the nonredundant truncated-Newton backend.
   *
   * Each inner iteration requests one matrix-free reduced Hessian-vector product.
   * A value of `0` uses the dimension-bounded safety limit of 32 iterations.
   */
  int nonredundant_truncated_newton_max_cg_iterations = 0;

  /**
   * @brief Relative finite-difference step used by the reduced-space HVP probes.
   *
   * This is used only by the diagnostic `full_fd` HVP. The analytic
   * `exact_ctx` operator has no finite-difference step parameter.
   */
  double nonredundant_truncated_newton_hvp_step_size = 1.0e-3;

  /**
   * @brief HVP model used by the nonredundant truncated-Newton backend.
   *
   * `exact_ctx` is the intended production HVP. It uses the accepted-point
   * second-order context as a landing zone for a fully analytic direct-action
   * orbital HVP, differentiating both the accepted-point local orbital /
   * integral chain and the relaxed structure/eigen response without forming the
   * dense orbital Hessian explicitly.
   *
   * `full_fd` finite-differences the fully relaxed orbital gradient. This is a
   * high-cost reference implementation kept for validation and regression
   * checks against `exact_ctx`.
   */
  NonredundantTruncatedNewtonHvpMode nonredundant_truncated_newton_hvp_mode =
      NonredundantTruncatedNewtonHvpMode::ExactContextDirectAction;

  /**
   * @brief Number of transported secant pairs used to enrich the TN preconditioner.
   *
   * This is a maximum history length. The solver may disable the transported
   * preconditioner automatically for cheap objectives where the extra algebra
   * is unlikely to repay itself. A value of `0` always falls back to the
   * reduced diagonal alone.
   */
  int nonredundant_truncated_newton_transport_history_size = 8;

  /**
   * @brief Whether to print per-iteration optimizer diagnostics.
   */
  bool verbose = true;

  /**
   * @brief Whether to retain accepted-iterate trace snapshots in memory.
   */
  bool retain_accepted_iteration_trace = true;

  /**
   * @brief Whether accepted-iteration callbacks require the `E11` orbital gradient.
   *
   * The default terminal logger does not consume this field, so leaving the
   * flag disabled avoids an extra expensive reference-energy pullback on every
   * accepted step.
   */
  bool accepted_iteration_callback_requires_reference_gradient = false;

  /**
   * @brief Whether accepted-iteration callbacks require the full snapshot payload.
   *
   * Library callbacks keep the historical full-payload behavior by default, but
   * lightweight consumers such as the terminal logger can disable this to avoid
   * copying structure matrices, orbital tables, and active-space buffers on
   * every accepted step.
   */
  bool accepted_iteration_callback_requires_full_snapshot = true;

  /**
   * @brief Optional callback invoked for the initial point and each accepted outer step.
   */
  std::function<void(const VbScfAcceptedIterationSnapshot&)>
      accepted_iteration_callback;
};

}  // namespace xmvb::vb
