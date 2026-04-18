#pragma once

#include <vector>

#include "core/linear_algebra/generalized_eigensolver.hpp"
#include "vb/matrices/cpp_vb_input.hpp"
#include "vb/matrices/full_structure_builder.hpp"
#include "vb/matrices/prepared_active_space_context.hpp"
#include "vb/orbital/active_space_one_electron_builder.hpp"
#include "vb/orbital/active_space_orbital_preparer.hpp"
#include "vb/orbital/active_space_two_electron_builder.hpp"
#include "vb/orbital/ao_effective_one_electron_builder.hpp"
#include "vb/scf/cpp_active_space_gradient_result.hpp"
#include "vb/vbscf_algorithm.hpp"


namespace xmvb::vb {

struct CppActiveSpaceSecondOrderContext;

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
      xmvb::core::GeneralizedEigensolver generalized_eigensolver);

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

  /**
   * @brief Evaluates the ground-state gradient reusing a prebuilt active-space context.
   *
   * This overload skips orbital preparation and active-space integral
   * construction, reusing the supplied timed context and differentiating only
   * the determinant/structure/eigensolver layers for the provided structure
   * topology.
   */
  CppActiveSpaceGradientResult evaluate(
      const CppVbInput& input,
      TimedPreparedActiveSpaceContext timed_prepared_active_space_context,
      double nuclear_repulsion_energy = 0.0) const;

  /**
   * @brief Evaluates a state-averaged gradient reusing a prebuilt active-space context.
   */
  CppActiveSpaceGradientResult evaluate(
      const CppVbInput& input,
      TimedPreparedActiveSpaceContext timed_prepared_active_space_context,
      const std::vector<int>& selected_state_indices,
      const std::vector<double>& state_average_weights,
      double nuclear_repulsion_energy) const;

  /**
   * @brief Rebuilds only the orbital-dependent integral layer under a frozen active-space adjoint.
   *
   * The returned result contains freshly rebuilt orbital preparation, AO-H1E,
   * active-space HHO/GGO inputs at `input`, while the active-space adjoint
   * buffers are copied from the accepted-point context. This provides a cheap
   * accepted-point core-curvature probe that avoids rebuilding the determinant
   * and generalized-eigen response on every HVP application.
   */
  CppActiveSpaceGradientResult evaluate_with_fixed_active_space_adjoint(
      const CppVbInput& input,
      const CppActiveSpaceSecondOrderContext& accepted_point_context,
      double nuclear_repulsion_energy = 0.0) const;

  /**
   * @brief Rebuilds only the orbital-dependent prepared active-space layer.
   *
   * Matrix-free probe paths can combine this fresh orbital-dependent context
   * with accepted-point active-space adjoints without materializing a full
   * `CppActiveSpaceGradientResult`.
   */
  TimedPreparedActiveSpaceContext prepare_timed_active_space_context_for_probe(
      const CppVbInput& input) const;

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
