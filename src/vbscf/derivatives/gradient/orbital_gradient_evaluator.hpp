#pragma once

#include <chrono>
#include <vector>

#include "vbscf/core/vbscf_input.hpp"
#include "vbscf/integrals/active/active_space_matrix_backpropagator.hpp"
#include "vbscf/orbitals/orbital_pullback.hpp"
#include "vbscf/orbitals/orbital_preparer.hpp"
#include "vbscf/integrals/active/active_space_two_electron_backpropagator.hpp"
#include "vbscf/integrals/ao/ao_effective_one_electron_backpropagator.hpp"
#include "vbscf/integrals/ao/ao_effective_one_electron_builder.hpp"
#include "vbscf/derivatives/gradient/active_space_gradient_evaluator.hpp"
#include "vbscf/derivatives/gradient/orbital_gradient_result.hpp"
#include "vbscf/core/algorithm.hpp"

namespace xmvb::vb {

struct AcceptedPointContext;

/**
 * @brief Orbital-gradient evaluator on top of the C++ VBSCF path.
 *
 * This evaluator uses the analytic active-space gradient kernels and
 * backpropagates them to the sparse orbital coefficient table. The overlap,
 * one-electron, and two-electron active-space branches are all propagated
 * analytically.
 */
class OrbitalGradientEvaluator {
public:
  /**
   * @brief Creates an evaluator with default helper components.
   */
  explicit OrbitalGradientEvaluator(
      VbScfAlgorithm algorithm = VbScfAlgorithm::Original,
      double finite_difference_step = 1.0e-5);

  /**
   * @brief Creates an evaluator with explicit helper components.
   */
  OrbitalGradientEvaluator(
      ActiveSpaceGradientEvaluator active_space_gradient_evaluator,
      ActiveSpaceOrbitalPreparer orbital_preparer,
      AoEffectiveOneElectronBuilder ao_effective_one_electron_builder,
      AoEffectiveOneElectronBackpropagator ao_effective_one_electron_backpropagator,
      ActiveSpaceMatrixBackpropagator active_space_matrix_backpropagator,
      ActiveSpaceTwoElectronBackpropagator active_space_two_electron_backpropagator,
      ActiveSpaceOrbitalBackpropagator active_space_orbital_backpropagator,
      double finite_difference_step);

  /**
   * @brief Evaluates the ground-state orbital gradient.
   *
   * @param input C++ VB input bundle.
   * @param nuclear_repulsion_energy Nuclear repulsion energy.
   * @return OrbitalGradientResult Baseline result and orbital gradient.
   */
  OrbitalGradientResult evaluate(
      const VbScfInput& input,
      double nuclear_repulsion_energy = 0.0) const;

  /**
   * @brief Evaluates the ground-state orbital gradient without the exact `E11` orbital gradient.
   *
   * This is intended for optimization inner loops that only need the objective
   * gradient. Accepted-iterate trace generation can request the exact `E11`
   * gradient later from the cached intermediates in the returned result.
   */
  OrbitalGradientResult evaluate_without_reference_energy_gradient(
      const VbScfInput& input,
      double nuclear_repulsion_energy = 0.0) const;

  /**
   * @brief Evaluates a state-averaged orbital gradient.
   *
   * @param input C++ VB input bundle.
   * @param selected_state_indices Zero-based state indices.
   * @param state_average_weights Non-negative state-averaging weights.
   * @param nuclear_repulsion_energy Nuclear repulsion energy.
   * @return OrbitalGradientResult Baseline result and orbital gradient.
   */
  OrbitalGradientResult evaluate(
      const VbScfInput& input,
      const std::vector<int>& selected_state_indices,
      const std::vector<double>& state_average_weights,
      double nuclear_repulsion_energy) const;

  /**
   * @brief Evaluates the orbital gradient from a precomputed active-space result.
   *
   * This reuses an already differentiated active-space layer and performs only
   * the lower orbital/AO pullback plus the exact `E11` orbital gradient.
   */
  OrbitalGradientResult evaluate(
      const VbScfInput& input,
      ActiveSpaceGradientResult active_space_gradient_result) const;

  /**
   * @brief Evaluates a state-averaged orbital gradient without the exact `E11` orbital gradient.
   */
  OrbitalGradientResult evaluate_without_reference_energy_gradient(
      const VbScfInput& input,
      const std::vector<int>& selected_state_indices,
      const std::vector<double>& state_average_weights,
      double nuclear_repulsion_energy) const;

  /**
   * @brief Evaluates the orbital gradient pullback from a precomputed active-space result.
   */
  OrbitalGradientResult evaluate_without_reference_energy_gradient(
      const VbScfInput& input,
      ActiveSpaceGradientResult active_space_gradient_result) const;

  /**
   * @brief Evaluates a cheap accepted-point core gradient under frozen active-space adjoints.
   *
   * This keeps the accepted-point active-space adjoint buffers fixed and only
   * rebuilds the orbital-preparation and integral-transformation layers at the
   * new orbital point. It is intended as an intermediate HVP oracle between the
   * cheap reduced-curvature diagonal and the future full exact direct-action operator.
   */
  OrbitalGradientResult evaluate_without_reference_energy_gradient_with_fixed_active_space_adjoint(
      const VbScfInput& input,
      const AcceptedPointContext& accepted_point_context,
      double nuclear_repulsion_energy) const;

  /**
   * @brief Evaluates only the sparse orbital-energy gradient for a fixed active-space adjoint probe.
   *
   * Matrix-free exact-context HVP probes need only the orbital gradient vector
   * in the packed sparse-coefficient basis. This lightweight path skips
   * materializing the full `OrbitalGradientResult`, avoiding repeated
   * integral and bookkeeping copies on every `H v` application.
   */
  std::vector<double> evaluate_sparse_orbital_gradient_with_fixed_active_space_adjoint(
      const VbScfInput& input,
      const AcceptedPointContext& accepted_point_context,
      double nuclear_repulsion_energy) const;

  /**
   * @brief Populates the exact `E11` orbital gradient if it is not already present.
   */
  void populate_reference_energy_gradient(
      const VbScfInput& input,
      OrbitalGradientResult* result) const;

private:
  ActiveSpaceGradientEvaluator active_space_gradient_evaluator_;
  ActiveSpaceOrbitalPreparer orbital_preparer_;
  AoEffectiveOneElectronBuilder ao_effective_one_electron_builder_;
  AoEffectiveOneElectronBackpropagator ao_effective_one_electron_backpropagator_;
  ActiveSpaceMatrixBackpropagator active_space_matrix_backpropagator_;
  ActiveSpaceTwoElectronBackpropagator active_space_two_electron_backpropagator_;
  ActiveSpaceOrbitalBackpropagator active_space_orbital_backpropagator_;
  double finite_difference_step_ = 1.0e-5;

  OrbitalGradientResult evaluate_from_active_space_gradient_result(
      const VbScfInput& input,
      ActiveSpaceGradientResult active_space_gradient_result,
      const std::chrono::steady_clock::time_point& total_start_time) const;
};

}  // namespace xmvb::vb
