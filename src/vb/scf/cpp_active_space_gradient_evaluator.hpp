#pragma once

#include <vector>

#include "core/linear_algebra/generalized_eigensolver.hpp"
#include "vb/matrices/cpp_vb_input.hpp"
#include "vb/matrices/full_structure_builder.hpp"
#include "vb/orbital/active_space_one_electron_builder.hpp"
#include "vb/orbital/active_space_orbital_preparer.hpp"
#include "vb/orbital/active_space_two_electron_builder.hpp"
#include "vb/orbital/ao_effective_one_electron_builder.hpp"
#include "vb/scf/cpp_active_space_gradient_result.hpp"
#include "vb/vbscf_algorithm.hpp"


namespace xmvb::vb {

/**
 * @brief Analytic gradient evaluator for the active-space integral layer.
 *
 * This evaluator differentiates the C++ VBSCF energy with respect
 * to:
 * - the active-space one-electron matrix `HHO`
 * - the packed active-space two-electron integrals `GGO`
 *
 * The generalized-eigen response and determinant/structure assembly are both
 * handled analytically. This provides the core gradient needed by a future
 * fully analytic orbital-response implementation.
 */
class CppActiveSpaceGradientEvaluator {
public:
  /**
   * @brief Creates an evaluator with default helper components.
   */
  explicit CppActiveSpaceGradientEvaluator(
      VBSCFAlgorithm algorithm = VBSCFAlgorithm::Original);

  /**
   * @brief Creates an evaluator with explicit helper components.
   */
  CppActiveSpaceGradientEvaluator(
      ActiveSpaceOrbitalPreparer orbital_preparer,
      AoEffectiveOneElectronBuilder ao_effective_one_electron_builder,
      ActiveSpaceOneElectronBuilder active_space_one_electron_builder,
      ActiveSpaceTwoElectronBuilder active_space_two_electron_builder,
      FullDeterminantStructureHamiltonianOverlapBuilder structure_builder,
      xmvb::core::GeneralizedEigensolver generalized_eigensolver,
      VBSCFAlgorithm algorithm = VBSCFAlgorithm::Original);

  /**
   * @brief Evaluates the ground-state active-space analytic gradient.
   */
  CppActiveSpaceGradientResult evaluate(
      const CppVbInput& input,
      double nuclear_repulsion_energy = 0.0) const;

  /**
   * @brief Evaluates a state-averaged active-space analytic gradient.
   */
  CppActiveSpaceGradientResult evaluate(
      const CppVbInput& input,
      const std::vector<int>& selected_state_indices,
      const std::vector<double>& state_average_weights,
      double nuclear_repulsion_energy) const;

private:
  ActiveSpaceOrbitalPreparer orbital_preparer_;
  AoEffectiveOneElectronBuilder ao_effective_one_electron_builder_;
  ActiveSpaceOneElectronBuilder active_space_one_electron_builder_;
  ActiveSpaceTwoElectronBuilder active_space_two_electron_builder_;
  FullDeterminantStructureHamiltonianOverlapBuilder structure_builder_;
  xmvb::core::GeneralizedEigensolver generalized_eigensolver_;
  VBSCFAlgorithm algorithm_ = VBSCFAlgorithm::Original;
};

}  // namespace xmvb::vb
