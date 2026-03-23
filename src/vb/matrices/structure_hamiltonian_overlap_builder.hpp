#pragma once

#include <vector>

#include "vb/matrices/determinant_hamiltonian_resolver.hpp"
#include "vb/matrices/determinant_pair_input.hpp"
#include "vb/matrices/structure_accumulation_result.hpp"
#include "vb/matrices/structure_expansion_term.hpp"
#include "vb/matrices/structure_pair_accumulator.hpp"

namespace xmvb::vb {

/**
 * @brief Builds structure-level Hamiltonian and overlap matrices from determinant data.
 *
 * This class is the first standalone C++ composition point that mirrors the
 * legacy `Hamhd` + `Hovstr1` workflow:
 * 1. Resolve determinant-level overlap and Hamiltonian for each determinant pair.
 * 2. Accumulate those contributions into structure-level matrices.
 */
class StructureHamiltonianOverlapBuilder {
public:
  /**
   * @brief Creates a builder with default determinant and structure helpers.
   */
  StructureHamiltonianOverlapBuilder();

  /**
   * @brief Creates a builder with explicit helper components.
   *
   * @param determinant_hamiltonian_resolver Determinant-level numerical kernel.
   * @param structure_pair_accumulator Structure-level accumulation kernel.
   */
  StructureHamiltonianOverlapBuilder(
      DeterminantHamiltonianResolver determinant_hamiltonian_resolver,
      StructurePairAccumulator structure_pair_accumulator);

  /**
   * @brief Builds structure-level matrices from determinant-pair data.
   *
   * @param determinant_pair_inputs Determinant pairs to evaluate.
   * @param determinant_to_structure_terms Expansion terms for each determinant.
   * @param one_electron_matrix Column-major one-electron integral matrix.
   * @param n_orbitals Total number of orbitals.
   * @param packed_two_electron_integrals Packed two-electron integral storage.
   * @param n_structures Number of structures.
   * @param n_determinants Number of determinants.
   * @return StructureAccumulationResult Structure-level matrices and overlap cache.
   */
  StructureAccumulationResult build(
      const std::vector<DeterminantPairInput>& determinant_pair_inputs,
      const std::vector<std::vector<StructureExpansionTerm>>& determinant_to_structure_terms,
      const std::vector<double>& one_electron_matrix,
      int n_orbitals,
      const std::vector<double>& packed_two_electron_integrals,
      int n_structures,
      int n_determinants) const;

private:
  DeterminantHamiltonianResolver determinant_hamiltonian_resolver_;
  StructurePairAccumulator structure_pair_accumulator_;
};

}  // namespace xmvb::vb
