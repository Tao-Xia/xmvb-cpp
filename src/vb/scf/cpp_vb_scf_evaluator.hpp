#pragma once

#include <vector>

#include "core/linear_algebra/generalized_eigensolver.hpp"
#include "vb/matrices/cpp_vb_input.hpp"
#include "vb/matrices/structure_matrix_evaluator.hpp"
#include "vb/matrices/structure_subspace_builder.hpp"
#include "vb/scf/cpp_vb_scf_result.hpp"
#include "vb/vbscf_algorithm.hpp"

namespace xmvb::vb {

/**
 * @brief Single-step evaluator for the C++ VBSCF matrix path.
 *
 * This class turns the C++ matrix builders into a complete VBSCF
 * step evaluation:
 * 1. construct structure Hamiltonian and overlap matrices
 * 2. solve the generalized eigenvalue problem
 * 3. form state-averaged energies and expose the eigenvectors
 */
class CppVbScfEvaluator {
public:
  /**
   * @brief Creates an evaluator with default helper components.
   */
  explicit CppVbScfEvaluator(
      VbScfAlgorithm algorithm = VbScfAlgorithm::Original);

  /**
   * @brief Creates an evaluator with explicit helper components.
   */
  CppVbScfEvaluator(
      StructureMatrixEvaluator matrix_evaluator,
      xmvb::core::GeneralizedEigensolver generalized_eigensolver);

  /**
   * @brief Evaluates the ground-state VBSCF step.
   *
   * @param input C++ VB input bundle.
   * @param nuclear_repulsion_energy Nuclear repulsion energy.
   * @return CppVbScfResult Structure matrices plus generalized-eigen data.
   */
  CppVbScfResult evaluate(
      const CppVbInput& input,
      double nuclear_repulsion_energy = 0.0) const;

  /**
   * @brief Evaluates an explicitly selected state-averaged VBSCF step.
   *
   * @param input C++ VB input bundle.
   * @param selected_state_indices Zero-based state indices.
   * @param state_average_weights Non-negative state-averaging weights.
   * @param nuclear_repulsion_energy Nuclear repulsion energy.
   * @return CppVbScfResult Structure matrices plus generalized-eigen data.
   */
  CppVbScfResult evaluate(
      const CppVbInput& input,
      const std::vector<int>& selected_state_indices,
      const std::vector<double>& state_average_weights,
      double nuclear_repulsion_energy) const;

  /**
   * @brief Evaluates a compact selected-structure subspace on the current orbitals.
   *
   * @param input Full C++ VB input bundle.
   * @param selected_structure_indices Zero-based structure indices kept in the subspace.
   * @param nuclear_repulsion_energy Nuclear repulsion energy.
   * @return CppVbScfResult Structure matrices plus generalized-eigen data.
   */
  CppVbScfResult evaluate_subspace(
      const CppVbInput& input,
      const std::vector<int>& selected_structure_indices,
      double nuclear_repulsion_energy = 0.0) const;

  /**
   * @brief Evaluates a selected-structure subspace for chosen states.
   *
   * @param input Full C++ VB input bundle.
   * @param selected_structure_indices Zero-based structure indices kept in the subspace.
   * @param selected_state_indices Zero-based state indices in the selected subspace.
   * @param state_average_weights Non-negative state-averaging weights.
   * @param nuclear_repulsion_energy Nuclear repulsion energy.
   * @return CppVbScfResult Structure matrices plus generalized-eigen data.
   */
  CppVbScfResult evaluate_subspace(
      const CppVbInput& input,
      const std::vector<int>& selected_structure_indices,
      const std::vector<int>& selected_state_indices,
      const std::vector<double>& state_average_weights,
      double nuclear_repulsion_energy) const;

private:
  StructureMatrixEvaluator matrix_evaluator_;
  xmvb::core::GeneralizedEigensolver generalized_eigensolver_;
  StructureSubspaceBuilder subspace_builder_;
};

}  // namespace xmvb::vb
