#pragma once

#include <chrono>
#include <vector>

#include "vb/matrices/cpp_vb_input.hpp"
#include "vb/orbital/active_space_matrix_backpropagator.hpp"
#include "vb/orbital/active_space_orbital_backpropagator.hpp"
#include "vb/orbital/active_space_orbital_preparer.hpp"
#include "vb/orbital/active_space_two_electron_backpropagator.hpp"
#include "vb/orbital/ao_effective_one_electron_backpropagator.hpp"
#include "vb/orbital/ao_effective_one_electron_builder.hpp"
#include "vb/scf/cpp_active_space_gradient_evaluator.hpp"
#include "vb/scf/cpp_orbital_gradient_result.hpp"
#include "vb/vbscf_algorithm.hpp"

namespace xmvb::vb {

struct CppActiveSpaceSecondOrderContext;

/**
 * @brief Orbital-gradient evaluator on top of the C++ VBSCF path.
 *
 * This evaluator uses the analytic active-space gradient kernels and
 * backpropagates them to the sparse orbital coefficient table. The overlap,
 * one-electron, and two-electron active-space branches are all propagated
 * analytically.
 */
class CppOrbitalGradientEvaluator {
public:
  /**
   * @brief Creates an evaluator with default helper components.
   */
  explicit CppOrbitalGradientEvaluator(
      VBSCFAlgorithm algorithm = VBSCFAlgorithm::Original,
      double finite_difference_step = 1.0e-5);

  /**
   * @brief Creates an evaluator with explicit helper components.
   */
  CppOrbitalGradientEvaluator(
      CppActiveSpaceGradientEvaluator active_space_gradient_evaluator,
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
   * @return CppOrbitalGradientResult Baseline result and orbital gradient.
   */
  CppOrbitalGradientResult evaluate(
      const CppVbInput& input,
      double nuclear_repulsion_energy = 0.0) const;

  /**
   * @brief Evaluates the ground-state orbital gradient without the exact `E11` orbital gradient.
   *
   * This is intended for optimization inner loops that only need the objective
   * gradient. Accepted-iterate trace generation can request the exact `E11`
   * gradient later from the cached intermediates in the returned result.
   */
  CppOrbitalGradientResult evaluate_without_reference_energy_gradient(
      const CppVbInput& input,
      double nuclear_repulsion_energy = 0.0) const;

  /**
   * @brief Evaluates a state-averaged orbital gradient.
   *
   * @param input C++ VB input bundle.
   * @param selected_state_indices Zero-based state indices.
   * @param state_average_weights Non-negative state-averaging weights.
   * @param nuclear_repulsion_energy Nuclear repulsion energy.
   * @return CppOrbitalGradientResult Baseline result and orbital gradient.
   */
  CppOrbitalGradientResult evaluate(
      const CppVbInput& input,
      const std::vector<int>& selected_state_indices,
      const std::vector<double>& state_average_weights,
      double nuclear_repulsion_energy) const;

  /**
   * @brief Evaluates the orbital gradient from a precomputed active-space result.
   *
   * This reuses an already differentiated active-space layer and performs only
   * the lower orbital/AO pullback plus the exact `E11` orbital gradient.
   */
  CppOrbitalGradientResult evaluate(
      const CppVbInput& input,
      CppActiveSpaceGradientResult active_space_gradient_result) const;

  /**
   * @brief Evaluates a state-averaged orbital gradient without the exact `E11` orbital gradient.
   */
  CppOrbitalGradientResult evaluate_without_reference_energy_gradient(
      const CppVbInput& input,
      const std::vector<int>& selected_state_indices,
      const std::vector<double>& state_average_weights,
      double nuclear_repulsion_energy) const;

  /**
   * @brief Evaluates the orbital gradient pullback from a precomputed active-space result.
   */
  CppOrbitalGradientResult evaluate_without_reference_energy_gradient(
      const CppVbInput& input,
      CppActiveSpaceGradientResult active_space_gradient_result) const;

  /**
   * @brief Evaluates a cheap accepted-point core gradient under frozen active-space adjoints.
   *
   * This keeps the accepted-point active-space adjoint buffers fixed and only
   * rebuilds the orbital-preparation and integral-transformation layers at the
   * new orbital point. It is intended as an intermediate HVP oracle between the
   * cheap reduced-curvature diagonal and the future full exact direct-action operator.
   */
  CppOrbitalGradientResult evaluate_without_reference_energy_gradient_with_fixed_active_space_adjoint(
      const CppVbInput& input,
      const CppActiveSpaceSecondOrderContext& accepted_point_context,
      double nuclear_repulsion_energy) const;

  /**
   * @brief Evaluates only the sparse orbital-energy gradient for a fixed active-space adjoint probe.
   *
   * Matrix-free exact-context HVP probes need only the orbital gradient vector
   * in the packed sparse-coefficient basis. This lightweight path skips
   * materializing the full `CppOrbitalGradientResult`, avoiding repeated
   * integral and bookkeeping copies on every `H v` application.
   */
  std::vector<double> evaluate_sparse_orbital_gradient_with_fixed_active_space_adjoint(
      const CppVbInput& input,
      const CppActiveSpaceSecondOrderContext& accepted_point_context,
      double nuclear_repulsion_energy) const;

  /**
   * @brief Populates the exact `E11` orbital gradient if it is not already present.
   */
  void populate_reference_energy_gradient(
      const CppVbInput& input,
      CppOrbitalGradientResult* result) const;

private:
  CppActiveSpaceGradientEvaluator active_space_gradient_evaluator_;
  ActiveSpaceOrbitalPreparer orbital_preparer_;
  AoEffectiveOneElectronBuilder ao_effective_one_electron_builder_;
  AoEffectiveOneElectronBackpropagator ao_effective_one_electron_backpropagator_;
  ActiveSpaceMatrixBackpropagator active_space_matrix_backpropagator_;
  ActiveSpaceTwoElectronBackpropagator active_space_two_electron_backpropagator_;
  ActiveSpaceOrbitalBackpropagator active_space_orbital_backpropagator_;
  double finite_difference_step_ = 1.0e-5;

  CppOrbitalGradientResult evaluate_from_active_space_gradient_result(
      const CppVbInput& input,
      CppActiveSpaceGradientResult active_space_gradient_result,
      const std::chrono::steady_clock::time_point& total_start_time) const;
};

}  // namespace xmvb::vb
