#pragma once

#include <functional>

#include "vbscf/optimization/driver/backend.hpp"
#include "vbscf/optimization/driver/result.hpp"
#include "vbscf/core/contracts/eigensolver.hpp"

namespace xmvb::vb {

/** @brief Initial inverse Hessian used by transported L-BFGS recursion. */
enum class LbfgsInitialInverse {
  /** Conventional secant scaling, \f$H_0=\gamma I\f$. */
  ScaledIdentity,
  /** Positive orbital-block inverse used by TNHVP and its L-BFGS ablation. */
  OrbitalBlock,
};

/**
 * @brief Options controlling the VBSCF orbital optimization loop.
 */
struct VbScfOptimizerOptions {
  /**
   * @brief Optimization backend implementation.
   */
  VbScfOptimizerBackend backend =
      VbScfOptimizerBackend::Lbfgs;

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
   * All production backends apply this threshold to the projected reduced
   * gradient infinity norm.
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
   * @brief Initial inverse Hessian used by the standalone L-BFGS backend.
   *
   * The conventional scaled identity remains the production default. The
   * orbital-block mode is an explicit ablation that shares TNHVP's local
   * inverse action while retaining the L-BFGS line search and secant update.
   */
  LbfgsInitialInverse lbfgs_initial_inverse =
      LbfgsInitialInverse::ScaledIdentity;

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
