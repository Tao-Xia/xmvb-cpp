#pragma once

#include <Eigen/Core>

#include <vector>

#include "vbscf/determinants/algebra/hamiltonian.hpp"
#include "vbscf/determinants/pairs/evaluator.hpp"
#include "vbscf/determinants/algebra/overlap.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "vbscf/structures/expansion/types.hpp"

namespace xmvb::vb {

struct FullDeterminantStructureBuildResult {
  int n_determinants = 0;
  StructureAccumulationResult structure_matrices;
  std::vector<DeterminantPairEvaluation> pair_evaluations;

  const DeterminantPairEvaluation& pair_evaluation(
      int determinant_index_left,
      int determinant_index_right) const;
};
/**
 * @brief Builds full structure Hamiltonian and overlap matrices from full determinants.
 *
 * This builder works directly at the full-determinant level. When many full determinants
 * share the same alpha or beta occupied string, the builder can first cache the
 * reusable unique alpha-alpha and beta-beta determinant kernels, then combine
 * those cached same-spin results with the remaining opposite-spin Coulomb term.
 *
 * In the production `build()` path, the builder always uses the tiled/block
 * unique-spin contraction. The same-spin cache still provides reusable
 * determinant-pair payloads and reuse tables for later backward passes, but
 * the forward structure assembly uses only the tiled/block unique-spin
 * contraction.
 *
 * The resulting full-determinant matrix element is then accumulated into the
 * structure-level matrices through the determinant-to-structure expansion map.
 */
class FullDeterminantStructureHamiltonianOverlapBuilder {
public:
  /**
   * @brief Creates a builder with default helper components.
   */
  FullDeterminantStructureHamiltonianOverlapBuilder();

  /**
   * @brief Creates a builder with explicit helper components.
   *
   * @param determinant_overlap_resolver Overlap/cofactor resolver.
   * @param determinant_hamiltonian_resolver Determinant scalar Hamiltonian kernel.
   */
  FullDeterminantStructureHamiltonianOverlapBuilder(
      DeterminantOverlapResolver determinant_overlap_resolver,
      DeterminantHamiltonianResolver determinant_hamiltonian_resolver);

  /**
   * @brief Builds structure-level matrices from explicit full-determinant data.
   *
   * @param alpha_det Zero-based alpha occupations for each determinant.
   * @param beta_det Zero-based beta occupations for each determinant.
   * @param determinant_to_structure_terms Determinant-to-structure signed expansion terms.
   * @param ovlp_act Column-major active-space orbital overlap matrix.
   * @param h1e_act Column-major one-electron matrix.
   * @param n_orbitals Total number of active orbitals.
   * @param eri_act Packed two-electron storage using packed indexing.
   * @param n_structures Number of VB structures.
   * @return StructureAccumulationResult Structure Hamiltonian and overlap matrices.
   */
  StructureAccumulationResult build(
      const std::vector<std::vector<int>>& alpha_det,
      const std::vector<std::vector<int>>& beta_det,
      const std::vector<std::vector<StructureExpansionTerm>>& determinant_to_structure_terms,
      const std::vector<double>& ovlp_act,
      const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
      int n_orbitals,
      const std::vector<double>& eri_act,
      int n_structures) const;

  /**
   * @brief Builds structure matrices from packed or RI active-space ERIs.
   */
  StructureAccumulationResult build(
      const std::vector<std::vector<int>>& alpha_det,
      const std::vector<std::vector<int>>& beta_det,
      const std::vector<std::vector<StructureExpansionTerm>>& determinant_to_structure_terms,
      const std::vector<double>& ovlp_act,
      const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
      int n_orbitals,
      const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
      int n_structures) const;

  /**
   * @brief Builds structure-level matrices while reusing a prebuilt same-spin cache.
   *
   * This is used by the active-space objective/gradient path to build the
   * structure matrices once, then hand the same ordered alpha/beta cache to the
   * backward pass without rebuilding it.
   */
  StructureAccumulationResult build(
      const std::vector<std::vector<int>>& alpha_det,
      const std::vector<std::vector<int>>& beta_det,
      const std::vector<std::vector<StructureExpansionTerm>>& determinant_to_structure_terms,
      const std::vector<double>& ovlp_act,
      const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
      int n_orbitals,
      const std::vector<double>& eri_act,
      int n_structures,
      const SameSpinPairCacheContext& same_spin_pair_cache) const;

  /**
   * @brief Builds structure matrices while reusing a same-spin cache and RI data.
   */
  StructureAccumulationResult build(
      const std::vector<std::vector<int>>& alpha_det,
      const std::vector<std::vector<int>>& beta_det,
      const std::vector<std::vector<StructureExpansionTerm>>& determinant_to_structure_terms,
      const std::vector<double>& ovlp_act,
      const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
      int n_orbitals,
      const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
      int n_structures,
      const SameSpinPairCacheContext& same_spin_pair_cache) const;

  /**
   * @brief Creates a configured determinant-pair evaluator for on-demand reuse.
   *
   * Large determinant spaces cannot retain every pair evaluation in memory.
   * Callers that only need the structure matrices should use `build()`, then
   * reuse the returned evaluator to recompute determinant-pair details lazily
   * during later passes such as analytic gradient accumulation.
   */
  DeterminantPairEvaluator make_pair_evaluator() const;

  FullDeterminantStructureBuildResult build_with_pair_evaluations(
      const std::vector<std::vector<int>>& alpha_det,
      const std::vector<std::vector<int>>& beta_det,
      const std::vector<std::vector<StructureExpansionTerm>>& determinant_to_structure_terms,
      const std::vector<double>& ovlp_act,
      const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
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
  FullDeterminantStructureBuildResult build_impl(
      const std::vector<std::vector<int>>& alpha_det,
      const std::vector<std::vector<int>>& beta_det,
      const std::vector<std::vector<StructureExpansionTerm>>& determinant_to_structure_terms,
      const std::vector<double>& ovlp_act,
      const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
      int n_orbitals,
      const std::vector<double>& eri_act,
      int n_structures,
      const SameSpinPairCacheContext* same_spin_pair_cache,
      bool store_pair_evaluations) const;

  FullDeterminantStructureBuildResult build_impl(
      const std::vector<std::vector<int>>& alpha_det,
      const std::vector<std::vector<int>>& beta_det,
      const std::vector<std::vector<StructureExpansionTerm>>& determinant_to_structure_terms,
      const std::vector<double>& ovlp_act,
      const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
      int n_orbitals,
      const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
      int n_structures,
      const SameSpinPairCacheContext* same_spin_pair_cache,
      bool store_pair_evaluations) const;

  DeterminantOverlapResolver determinant_overlap_resolver_;
  DeterminantHamiltonianResolver determinant_hamiltonian_resolver_;
};

}  // namespace xmvb::vb
