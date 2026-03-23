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
  DeepVBHOnnx,
  DeepVBHOnnxDirectFinal,
};

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
      CppVbScfOptimizerBackend::Lbfgspp;

  /**
   * @brief Determinant evaluation algorithm used in the C++ VBSCF path.
   */
  VbScfAlgorithm algorithm = VbScfAlgorithm::Original;

  /**
   * @brief Maximum number of accepted optimization iterations.
   */
  int max_iterations = 25;

  /**
   * @brief Convergence threshold on the gradient infinity norm.
   */
  double gradient_tolerance = 2.0e-3;

  /**
   * @brief Convergence threshold on the absolute energy change.
   *
   * This mirrors the legacy `vb_str->epg` stopping criterion.
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
   * @brief Whether to print per-iteration optimizer diagnostics.
   */
  bool verbose = true;

  /**
   * @brief Whether to retain accepted-iterate trace snapshots in memory.
   */
  bool retain_accepted_iteration_trace = true;

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
