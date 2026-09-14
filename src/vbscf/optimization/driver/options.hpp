#pragma once

#include <functional>

#include "vbscf/optimization/driver/backend.hpp"
#include "vbscf/optimization/driver/result.hpp"
#include "vbscf/core/contracts/eigensolver.hpp"

namespace xmvb::vb {

/**
 * @brief Options controlling the VBSCF orbital optimization loop.
 */
struct VbScfOptimizerOptions {
  /**
   * @brief Optimization backend implementation.
   */
  VbScfOptimizerBackend backend =
      VbScfOptimizerBackend::NonredundantLbfgspp;

  /** @brief Structure-space eigensolver used by objective evaluations. */
  StructureEigensolver structure_eigensolver = StructureEigensolver::Davidson;

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
  double energy_tolerance = 1.0e-7;

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
  int tnhvp_max_subspace_dimension = 0;

  /**
   * @brief Number of transported secant pairs used to enrich the TN preconditioner.
   *
   * A value of `0` disables transported secants and uses only the reduced
   * curvature diagonal.
   */
  int nonredundant_truncated_newton_transport_history_size = 8;

  /**
   * @brief Whether to print per-iteration optimizer diagnostics.
   */
  bool verbose = true;

  /**
   * @brief Whether to retain accepted-iterate trace snapshots in memory.
   */
  bool retain_accepted_iteration_trace = false;

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
   * Enable this only for callbacks that consume structure matrices, orbital
   * tables, or active-space buffers. The default avoids copying those objects
   * on every accepted step.
   */
  bool accepted_iteration_callback_requires_full_snapshot = false;

  /**
   * @brief Optional callback invoked for the initial point and each accepted outer step.
   */
  std::function<void(const VbScfAcceptedIterationSnapshot&)>
      accepted_iteration_callback;
};

}  // namespace xmvb::vb
