#pragma once

#include <vector>

#include "core/eigensolver.hpp"
#include "vbscf/core/contracts/input.hpp"
#include "vbscf/structures/evaluation/evaluator.hpp"
#include "vbscf/core/contracts/result.hpp"

namespace xmvb::vb {

/**
 * @brief Single-step evaluator for the VBSCF matrix path.
 *
 * This class turns the C++ matrix builders into a complete VBSCF
 * step evaluation:
 * 1. construct structure Hamiltonian and overlap matrices
 * 2. solve the generalized eigenvalue problem
 * 3. form state-averaged energies and expose the eigenvectors
 */
class VbScfEvaluator {
public:
  /**
   * @brief Creates an evaluator with default helper components.
   */
  VbScfEvaluator();

  /**
   * @brief Creates an evaluator with explicit helper components.
   */
  VbScfEvaluator(
      StructureMatrixEvaluator matrix_evaluator,
      xmvb::core::GeneralizedEigensolver generalized_eigensolver);

  /**
   * @brief Evaluates the ground-state VBSCF step.
   *
   * @param input VBSCF input bundle.
   * @param nuclear_repulsion_energy Nuclear repulsion energy.
   * @return VbScfResult Structure matrices plus generalized-eigen data.
   */
  VbScfResult evaluate(
      const VbScfInput& input,
      double nuclear_repulsion_energy = 0.0) const;

  /**
   * @brief Evaluates an explicitly selected state-averaged VBSCF step.
   *
   * @param input VBSCF input bundle.
   * @param selected_state_indices Zero-based state indices.
   * @param state_average_weights Non-negative state-averaging weights.
   * @param nuclear_repulsion_energy Nuclear repulsion energy.
   * @return VbScfResult Structure matrices plus generalized-eigen data.
   */
  VbScfResult evaluate(
      const VbScfInput& input,
      const std::vector<int>& selected_state_indices,
      const std::vector<double>& state_average_weights,
      double nuclear_repulsion_energy) const;

  /**
   * @brief Evaluates only the relaxed state-averaged total energy.
   *
   * This skips eigenvector materialization and should be used for trial-point
   * screening paths that do not need gradients, selected-state coefficients,
   * or the accepted-point second-order context.
   */
  double evaluate_energy_only(
      const VbScfInput& input,
      const std::vector<int>& selected_state_indices,
      const std::vector<double>& state_average_weights,
      double nuclear_repulsion_energy) const;

private:
  StructureMatrixEvaluator matrix_evaluator_;
  xmvb::core::GeneralizedEigensolver generalized_eigensolver_;
};

}  // namespace xmvb::vb
