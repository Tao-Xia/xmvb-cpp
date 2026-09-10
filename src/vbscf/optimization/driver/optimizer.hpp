#pragma once

#include <vector>

#include "vbscf/core/vbscf_input.hpp"
#include "vbscf/derivatives/gradient/orbital_gradient_evaluator.hpp"
#include "vbscf/optimization/driver/options.hpp"
#include "vbscf/workflow/vbscf_evaluator.hpp"

namespace xmvb::vb {

/**
 * @brief VBSCF orbital optimizer.
 *
 * The optimizer uses the C++ energy/gradient evaluators together
 * with a selectable L-BFGS backend.
 */
class VbScfOptimizer {
public:
  /**
   * @brief Creates an optimizer with default helper components.
   */
  explicit VbScfOptimizer(
      VbScfOptimizerOptions options = {});

  /**
   * @brief Creates an optimizer with explicit helper components.
   */
  VbScfOptimizer(
      OrbitalGradientEvaluator orbital_gradient_evaluator,
      VbScfEvaluator scf_evaluator,
      VbScfOptimizerOptions options);

  /**
   * @brief Optimizes the ground-state orbital parameters.
   */
  VbScfOptimizerResult optimize(
      const VbScfInput& input,
      double nuclear_repulsion_energy = 0.0) const;

  /**
   * @brief Optimizes a selected-state or state-averaged objective.
   */
  VbScfOptimizerResult optimize(
      const VbScfInput& input,
      const std::vector<int>& selected_state_indices,
      const std::vector<double>& state_average_weights,
      double nuclear_repulsion_energy) const;

private:
  OrbitalGradientEvaluator orbital_gradient_evaluator_;
  VbScfEvaluator scf_evaluator_;
  VbScfOptimizerOptions options_;
};

}  // namespace xmvb::vb
