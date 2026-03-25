#pragma once

#include <vector>

#include "vb/matrices/determinant_hamiltonian_resolver.hpp"
#include "vb/matrices/full_determinant_pair_evaluator.hpp"
#include "vb/matrices/determinant_overlap_resolver.hpp"
#include "vb/matrices/structure_pair_accumulator.hpp"
#include "vb/matrices/structure_types.hpp"
#include "vb/vbscf_algorithm.hpp"

namespace xmvb::vb {

struct FullDeterminantStructureBuildResult {
  int n_determinants = 0;
  StructureAccumulationResult structure_matrices;
  std::vector<FullDeterminantPairEvaluation> pair_evaluations;

  const FullDeterminantPairEvaluation& pair_evaluation(
      int determinant_index_left,
      int determinant_index_right) const;
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
      VBSCFAlgorithm algorithm = VBSCFAlgorithm::Original);

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
      VBSCFAlgorithm algorithm = VBSCFAlgorithm::Original);

  /**
   * @brief Builds structure-level matrices from explicit full-determinant data.
   *
   * @param alpha_det Zero-based alpha occupations for each determinant.
   * @param beta_det Zero-based beta occupations for each determinant.
   * @param determinant_to_structure_terms Determinant-to-structure signed expansion terms.
   * @param ovlp_act Column-major active-space orbital overlap matrix.
   * @param h1e_act Column-major one-electron matrix.
   * @param n_orbitals Total number of active orbitals.
   * @param eri_act Packed two-electron storage using legacy indexing.
   * @param n_structures Number of VB structures.
   * @return StructureAccumulationResult Structure Hamiltonian and overlap matrices.
   */
  StructureAccumulationResult build(
      const std::vector<std::vector<int>>& alpha_det,
      const std::vector<std::vector<int>>& beta_det,
      const std::vector<std::vector<StructureExpansionTerm>>& determinant_to_structure_terms,
      const std::vector<double>& ovlp_act,
      const std::vector<double>& h1e_act,
      int n_orbitals,
      const std::vector<double>& eri_act,
      int n_structures) const;

  FullDeterminantStructureBuildResult build_with_pair_evaluations(
      const std::vector<std::vector<int>>& alpha_det,
      const std::vector<std::vector<int>>& beta_det,
      const std::vector<std::vector<StructureExpansionTerm>>& determinant_to_structure_terms,
      const std::vector<double>& ovlp_act,
      const std::vector<double>& h1e_act,
      int n_orbitals,
      const std::vector<double>& eri_act,
      int n_structures) const;

  /**
   * @brief Builds structure-level matrices from an explicit input object.
   *
   * @param input Explicit full-determinant data model.
   * @return StructureAccumulationResult Structure Hamiltonian and overlap matrices.
   */
  StructureAccumulationResult build(const FullDeterminantStructureData& input) const;

  FullDeterminantStructureBuildResult build_with_pair_evaluations(
      const FullDeterminantStructureData& input) const;

private:
  DeterminantOverlapResolver determinant_overlap_resolver_;
  DeterminantHamiltonianResolver determinant_hamiltonian_resolver_;
  StructurePairAccumulator structure_pair_accumulator_;
  VBSCFAlgorithm algorithm_ = VBSCFAlgorithm::Original;
};

}  // namespace xmvb::vb
