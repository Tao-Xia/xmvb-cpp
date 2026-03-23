#pragma once

#include <vector>

#include "vb/matrices/determinant_hamiltonian_resolver.hpp"
#include "vb/matrices/full_determinant_structure_data.hpp"
#include "vb/matrices/determinant_overlap_resolver.hpp"
#include "vb/matrices/structure_accumulation_result.hpp"
#include "vb/matrices/structure_expansion_term.hpp"
#include "vb/matrices/structure_pair_accumulator.hpp"
#include "vb/vbscf_algorithm.hpp"

namespace xmvb::vb {

struct StructureCandidateCouplingResult {
  int candidate_structure_index = -1;
  std::vector<double> h_column;
  std::vector<double> s_column;
  double diagonal_h = 0.0;
  double diagonal_s = 0.0;
};

/**
 * @brief Builds full structure Hamiltonian and overlap matrices from full determinants.
 *
 * This builder works at the full-determinant level rather than the legacy
 * half-determinant enumeration used by `Hov1`. For each full determinant pair
 * it independently evaluates:
 * - alpha-spin determinant overlap and Hamiltonian
 * - beta-spin determinant overlap and Hamiltonian
 * - opposite-spin Coulomb coupling
 *
 * The resulting full-determinant matrix element is then accumulated into the
 * structure-level matrices through the determinant-to-structure expansion map.
 */
class FullDeterminantStructureHamiltonianOverlapBuilder {
public:
  /**
   * @brief Creates a builder with default helper components.
   */
  explicit FullDeterminantStructureHamiltonianOverlapBuilder(
      VbScfAlgorithm algorithm = VbScfAlgorithm::Original);

  /**
   * @brief Creates a builder with explicit helper components.
   *
   * @param determinant_overlap_resolver Overlap/cofactor resolver.
   * @param determinant_hamiltonian_resolver Determinant scalar Hamiltonian kernel.
   * @param structure_pair_accumulator Structure-level accumulator.
   */
  FullDeterminantStructureHamiltonianOverlapBuilder(
      DeterminantOverlapResolver determinant_overlap_resolver,
      DeterminantHamiltonianResolver determinant_hamiltonian_resolver,
      StructurePairAccumulator structure_pair_accumulator,
      VbScfAlgorithm algorithm = VbScfAlgorithm::Original);

  /**
   * @brief Builds structure-level matrices from explicit full-determinant data.
   *
   * @param alpha_occupied_orbitals_by_determinant Zero-based alpha occupations for each determinant.
   * @param beta_occupied_orbitals_by_determinant Zero-based beta occupations for each determinant.
   * @param determinant_to_structure_terms Determinant-to-structure signed expansion terms.
   * @param basis_overlap_matrix Column-major orbital overlap matrix.
   * @param one_electron_matrix Column-major one-electron matrix.
   * @param n_orbitals Total number of active orbitals.
   * @param packed_two_electron_integrals Packed two-electron storage using legacy indexing.
   * @param n_structures Number of VB structures.
   * @return StructureAccumulationResult Structure Hamiltonian and overlap matrices.
   */
  StructureAccumulationResult build(
      const std::vector<std::vector<int>>& alpha_occupied_orbitals_by_determinant,
      const std::vector<std::vector<int>>& beta_occupied_orbitals_by_determinant,
      const std::vector<std::vector<StructureExpansionTerm>>& determinant_to_structure_terms,
      const std::vector<double>& basis_overlap_matrix,
      const std::vector<double>& one_electron_matrix,
      int n_orbitals,
      const std::vector<double>& packed_two_electron_integrals,
      int n_structures) const;

  /**
   * @brief Builds structure-level matrices from an explicit input object.
   *
   * @param input Explicit full-determinant data model.
   * @return StructureAccumulationResult Structure Hamiltonian and overlap matrices.
   */
  StructureAccumulationResult build(const FullDeterminantStructureData& input) const;

  /**
   * @brief Builds exact couplings between a selected subspace and one external candidate.
   *
   * This path avoids rebuilding the already-known `selected-selected` block for
   * every candidate. It evaluates only the `selected-to-candidate` column and
   * the candidate diagonal by visiting determinant pairs that actually
   * contribute to those terms.
   *
   * @param input Explicit full-determinant data model.
   * @param selected_structure_indices Zero-based selected-structure indices in local order.
   * @param candidate_structure_index Zero-based external candidate structure index.
   * @return StructureCandidateCouplingResult Exact structure couplings for the candidate.
   */
  StructureCandidateCouplingResult build_candidate_coupling(
      const FullDeterminantStructureData& input,
      const std::vector<int>& selected_structure_indices,
      int candidate_structure_index) const;

private:
  DeterminantOverlapResolver determinant_overlap_resolver_;
  DeterminantHamiltonianResolver determinant_hamiltonian_resolver_;
  StructurePairAccumulator structure_pair_accumulator_;
  VbScfAlgorithm algorithm_ = VbScfAlgorithm::Original;
};

}  // namespace xmvb::vb
