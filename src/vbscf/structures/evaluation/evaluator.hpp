#pragma once

#include "vbscf/core/vbscf_input.hpp"
#include "vbscf/integrals/active/prepared_active_space.hpp"
#include "vbscf/structures/assembly/hamiltonian_overlap.hpp"
#include "vbscf/structures/expansion/types.hpp"
#include "vbscf/integrals/active/active_space_one_electron_builder.hpp"
#include "vbscf/orbitals/preparation/preparer.hpp"
#include "vbscf/integrals/active/active_space_two_electron_builder.hpp"
#include "vbscf/integrals/ao/ao_effective_one_electron_builder.hpp"

namespace xmvb::vb {

/**
 * @brief End-to-end evaluator for VB structure Hamiltonian/overlap matrices.
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
  StructureMatrixEvaluator();

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
   * @brief Prepares the shared active-space intermediates used by matrix/SCF code.
   */
  PreparedActiveSpaceContext prepare_active_space(
      const VbScfInput& input) const;

  /**
   * @brief Evaluates structure matrices from prebuilt active-space intermediates.
   */
  StructureAccumulationResult evaluate(
      const VbScfInput& input,
      const PreparedActiveSpaceContext& prepared_active_space) const;

private:
  ActiveSpaceOrbitalPreparer orbital_preparer_;
  AoEffectiveOneElectronBuilder ao_effective_one_electron_builder_;
  ActiveSpaceOneElectronBuilder active_space_one_electron_builder_;
  ActiveSpaceTwoElectronBuilder active_space_two_electron_builder_;
  FullDeterminantStructureHamiltonianOverlapBuilder structure_builder_;
};

}  // namespace xmvb::vb
