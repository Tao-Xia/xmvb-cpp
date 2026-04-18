#pragma once

#include <functional>
#include <vector>

#include "vb/matrices/cpp_vb_input.hpp"
#include "vb/scf/cpp_orbital_gradient_evaluator.hpp"
#include "vb/scf/cpp_vb_scf_optimizer_result.hpp"
#include "vb/scf/cpp_vb_scf_evaluator.hpp"
#include "vb/vbscf_algorithm.hpp"

namespace xmvb::vb {

enum class CppVbScfOptimizerBackend {
  LegacyFortran,
  Lbfgspp,
  NonredundantProjectedGradient,
  NonredundantLbfgspp,
  NonredundantTruncatedNewton,
  DeepVBHOnnx,
  DeepVBHOnnxDirectFinal,
};

enum class NonredundantTruncatedNewtonHvpMode {
  FullFiniteDifference,
  ExactContextDirectAction,
};

inline const char* nonredundant_truncated_newton_hvp_mode_name(
    NonredundantTruncatedNewtonHvpMode mode) {
  switch (mode) {
    case NonredundantTruncatedNewtonHvpMode::FullFiniteDifference:
      return "full_fd";
    case NonredundantTruncatedNewtonHvpMode::ExactContextDirectAction:
      return "exact_ctx";
  }
  return "unknown";
}

inline bool cpp_vb_scf_optimizer_backend_supported(
    CppVbScfOptimizerBackend backend) {
  switch (backend) {
    case CppVbScfOptimizerBackend::LegacyFortran:
#ifdef XMVB_CPP_ENABLE_LEGACY_FORTRAN_BACKEND
      return true;
#else
      return false;
#endif
    case CppVbScfOptimizerBackend::Lbfgspp:
    case CppVbScfOptimizerBackend::NonredundantProjectedGradient:
    case CppVbScfOptimizerBackend::NonredundantLbfgspp:
    case CppVbScfOptimizerBackend::NonredundantTruncatedNewton:
      return true;
    case CppVbScfOptimizerBackend::DeepVBHOnnx:
#ifdef XMVB_CPP_ENABLE_ONNX_RUNTIME
      return true;
#else
      return false;
#endif
    case CppVbScfOptimizerBackend::DeepVBHOnnxDirectFinal:
#ifdef XMVB_CPP_ENABLE_ONNX_RUNTIME
      return true;
#else
      return false;
#endif
  }
  return false;
}

inline const char* cpp_vb_scf_optimizer_backend_name(
    CppVbScfOptimizerBackend backend) {
  switch (backend) {
    case CppVbScfOptimizerBackend::LegacyFortran:
      return "legacy_fortran";
    case CppVbScfOptimizerBackend::Lbfgspp:
      return "lbfgspp";
    case CppVbScfOptimizerBackend::NonredundantProjectedGradient:
      return "nonredundant_projected_gradient";
    case CppVbScfOptimizerBackend::NonredundantLbfgspp:
      return "nonredundant_lbfgspp";
    case CppVbScfOptimizerBackend::NonredundantTruncatedNewton:
      return "nonredundant_truncated_newton";
    case CppVbScfOptimizerBackend::DeepVBHOnnx:
      return "deepvbh_onnx";
    case CppVbScfOptimizerBackend::DeepVBHOnnxDirectFinal:
      return "deepvbh_onnx_direct_final";
  }
  return "unknown";
}

/**
 * @brief Options controlling the C++ VBSCF orbital optimization loop.
 */
struct CppVbScfOptimizerOptions {
  /**
   * @brief Optimization backend implementation.
   */
  CppVbScfOptimizerBackend backend =
      CppVbScfOptimizerBackend::NonredundantLbfgspp;

  /**
   * @brief Determinant evaluation algorithm used in the C++ VBSCF path.
   */
  VBSCFAlgorithm algorithm = VBSCFAlgorithm::Original;

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
   * The legacy Fortran backend and the plain full-space `lbfgspp` backend use
   * the full gradient Euclidean norm, matching the original `gxn`
   * convergence check. Nonredundant backends apply the same threshold to the
   * projected / reduced gradient infinity norm.
   */
  double gradient_tolerance = 1.0e-3;

  /**
   * @brief Convergence threshold on the absolute energy change.
   *
   * This mirrors the legacy `vb_str->epg` stopping criterion.
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
   * @brief Maximum accepted full-space polish iterations after nonredundant convergence.
   *
   * This short refinement stage only applies to the
   * `nonredundant_lbfgspp` backend after the projected-gradient criterion has
   * been met.
   */
  int nonredundant_polish_max_iterations = 0;

  /**
   * @brief Relative tightening applied to the full-space gradient tolerance in polish.
   *
   * The polish stage targets
   * `gradient_tolerance * nonredundant_polish_gradient_scale`.
   */
  double nonredundant_polish_gradient_scale = 0.25;

  /**
   * @brief Maximum Krylov iterations for the nonredundant truncated-Newton backend.
   *
   * Each inner iteration requests one matrix-free reduced Hessian-vector product.
   * A value of `0` enables the built-in auto budget, which spends more inner
   * iterations only when a single relaxed gradient evaluation is cheap.
   */
  int nonredundant_truncated_newton_max_cg_iterations = 0;

  /**
   * @brief Relative finite-difference step used by the reduced-space HVP probes.
   *
   * The current truncated-Newton backend scales its reduced-space probe
   * displacement so the packed orbital step has an approximate Euclidean norm
   * of this value. `exact_ctx` uses it as the direct-action probe scale, while
   * `full_fd` uses it as the finite-difference displacement magnitude.
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
  std::function<void(const CppVbScfAcceptedIterationSnapshot&)>
      accepted_iteration_callback;
};

/**
 * @brief Orbital optimizer for the C++ VBSCF path.
 *
 * The optimizer uses the C++ energy/gradient evaluators together
 * with a selectable L-BFGS backend.
 */
class CppVbScfOptimizer {
public:
  /**
   * @brief Creates an optimizer with default helper components.
   */
  explicit CppVbScfOptimizer(
      CppVbScfOptimizerOptions options = {});

  /**
   * @brief Creates an optimizer with explicit helper components.
   */
  CppVbScfOptimizer(
      CppOrbitalGradientEvaluator orbital_gradient_evaluator,
      CppVbScfEvaluator scf_evaluator,
      CppVbScfOptimizerOptions options);

  /**
   * @brief Optimizes the ground-state orbital parameters.
   */
  CppVbScfOptimizerResult optimize(
      const CppVbInput& input,
      double nuclear_repulsion_energy = 0.0) const;

  /**
   * @brief Optimizes a selected-state or state-averaged objective.
   */
  CppVbScfOptimizerResult optimize(
      const CppVbInput& input,
      const std::vector<int>& selected_state_indices,
      const std::vector<double>& state_average_weights,
      double nuclear_repulsion_energy) const;

private:
  CppOrbitalGradientEvaluator orbital_gradient_evaluator_;
  CppVbScfEvaluator scf_evaluator_;
  CppVbScfOptimizerOptions options_;
};

}  // namespace xmvb::vb
