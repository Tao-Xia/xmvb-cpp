#pragma once

#include "vb/matrices/cpp_vb_input.hpp"
#include "vb/matrices/structure_accumulation_result.hpp"
#include "vb/matrices/full_determinant_structure_hamiltonian_overlap_builder.hpp"
#include "vb/orbital/active_space_one_electron_builder.hpp"
#include "vb/orbital/active_space_orbital_preparer.hpp"
#include "vb/orbital/active_space_two_electron_builder.hpp"
#include "vb/orbital/ao_effective_one_electron_builder.hpp"
#include "vb/vbscf_algorithm.hpp"

namespace xmvb::vb {

/**
 * @brief End-to-end evaluator for C++ VB structure Hamiltonian/overlap matrices.
 *
 * This class stitches together the full C++ matrix path:
 * 1. Rebuild auxiliary orbitals and inactive density from sparse orbital input.
 * 2. Build AO `G11/F11` from AO integrals and inactive density.
 * 3. Project AO effective one-electron integrals into active-space `HHO`.
 * 4. Transform AO two-electron integrals into active-space `GGO`.
 * 5. Assemble full determinant and structure Hamiltonian/overlap matrices.
 */
class StructureMatrixEvaluator {
public:
  /**
   * @brief Creates an evaluator with default helper components.
   */
  explicit StructureMatrixEvaluator(
      VbScfAlgorithm algorithm = VbScfAlgorithm::Original);

  /**
   * @brief Creates an evaluator with explicit helper components.
   */
  StructureMatrixEvaluator(
      ActiveSpaceOrbitalPreparer orbital_preparer,
      AoEffectiveOneElectronBuilder ao_effective_one_electron_builder,
      ActiveSpaceOneElectronBuilder active_space_one_electron_builder,
      ActiveSpaceTwoElectronBuilder active_space_two_electron_builder,
      FullDeterminantStructureHamiltonianOverlapBuilder structure_builder);

  /**
   * @brief Evaluates full structure Hamiltonian and overlap matrices.
   *
   * @param input C++ VB input bundle.
   * @return StructureAccumulationResult Structure Hamiltonian and overlap matrices.
   */
  StructureAccumulationResult evaluate(
      const CppVbInput& input) const;

private:
  ActiveSpaceOrbitalPreparer orbital_preparer_;
  AoEffectiveOneElectronBuilder ao_effective_one_electron_builder_;
  ActiveSpaceOneElectronBuilder active_space_one_electron_builder_;
  ActiveSpaceTwoElectronBuilder active_space_two_electron_builder_;
  FullDeterminantStructureHamiltonianOverlapBuilder structure_builder_;
};

}  // namespace xmvb::vb
