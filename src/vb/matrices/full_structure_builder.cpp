#include "vb/matrices/full_structure_builder.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace xmvb::vb {

namespace {

void validate_full_determinant_input(
    const std::vector<std::vector<int>>& alpha_det,
    const std::vector<std::vector<int>>& beta_det,
    const std::vector<std::vector<StructureExpansionTerm>>& determinant_to_structure_terms,
    int n_orbitals,
    int n_structures) {
  const int n_determinants = static_cast<int>(alpha_det.size());
  if (n_determinants <= 0) {
    throw std::invalid_argument("at least one determinant is required");
  }
  if (static_cast<int>(beta_det.size()) != n_determinants ||
      static_cast<int>(determinant_to_structure_terms.size()) != n_determinants) {
    throw std::invalid_argument("all determinant-indexed inputs must have the same size");
  }
  if (n_orbitals <= 0 || n_structures <= 0) {
    throw std::invalid_argument("n_orbitals and n_structures must be positive");
  }
}

void symmetrize_structure_matrices(
    StructureAccumulationResult& accumulation_result) {
  const int n_structures = accumulation_result.n_structures;
  for (int row = 0; row < n_structures; ++row) {
    for (int column = 0; column < row; ++column) {
      const std::size_t upper_index =
          static_cast<std::size_t>(row) * n_structures + column;
      const std::size_t lower_index =
          static_cast<std::size_t>(column) * n_structures + row;
      accumulation_result.overlap_matrix[lower_index] =
          accumulation_result.overlap_matrix[upper_index];
      accumulation_result.hamiltonian_matrix[lower_index] =
          accumulation_result.hamiltonian_matrix[upper_index];
      accumulation_result.one_electron_hamiltonian_matrix[lower_index] =
          accumulation_result.one_electron_hamiltonian_matrix[upper_index];
    }
  }
}

}  // namespace

const FullDeterminantPairEvaluation& FullDeterminantStructureBuildResult::pair_evaluation(
    int determinant_index_left,
    int determinant_index_right) const {
  if (determinant_index_left < 0 || determinant_index_right < 0 ||
      determinant_index_left >= n_determinants || determinant_index_right >= n_determinants) {
    throw std::out_of_range("determinant pair index out of range");
  }
  return pair_evaluations[static_cast<std::size_t>(determinant_index_left) *
                              static_cast<std::size_t>(n_determinants) +
                          static_cast<std::size_t>(determinant_index_right)];
}

FullDeterminantStructureHamiltonianOverlapBuilder::FullDeterminantStructureHamiltonianOverlapBuilder(
    VBSCFAlgorithm algorithm)
    : determinant_overlap_resolver_(),
      determinant_hamiltonian_resolver_(algorithm),
      structure_pair_accumulator_(),
      algorithm_(algorithm) {}

FullDeterminantStructureHamiltonianOverlapBuilder::FullDeterminantStructureHamiltonianOverlapBuilder(
    DeterminantOverlapResolver determinant_overlap_resolver,
    DeterminantHamiltonianResolver determinant_hamiltonian_resolver,
    StructurePairAccumulator structure_pair_accumulator,
    VBSCFAlgorithm algorithm)
    : determinant_overlap_resolver_(std::move(determinant_overlap_resolver)),
      determinant_hamiltonian_resolver_(std::move(determinant_hamiltonian_resolver)),
      structure_pair_accumulator_(std::move(structure_pair_accumulator)),
      algorithm_(algorithm) {}


StructureAccumulationResult FullDeterminantStructureHamiltonianOverlapBuilder::build(
    const std::vector<std::vector<int>>& alpha_det,
    const std::vector<std::vector<int>>& beta_det,
    const std::vector<std::vector<StructureExpansionTerm>>& determinant_to_structure_terms,
    const std::vector<double>& ovlp_act,
    const std::vector<double>& h1e_act,
    int n_orbitals,
    const std::vector<double>& eri_act,
    int n_structures) const {
  return build_with_pair_evaluations(
      alpha_det,
      beta_det,
      determinant_to_structure_terms,
      ovlp_act,
      h1e_act,
      n_orbitals,
      eri_act,
      n_structures)
      .structure_matrices;
}

FullDeterminantStructureBuildResult
FullDeterminantStructureHamiltonianOverlapBuilder::build_with_pair_evaluations(
    const std::vector<std::vector<int>>& alpha_det,
    const std::vector<std::vector<int>>& beta_det,
    const std::vector<std::vector<StructureExpansionTerm>>& determinant_to_structure_terms,
    const std::vector<double>& ovlp_act,
    const std::vector<double>& h1e_act,
    int n_orbitals,
    const std::vector<double>& eri_act,
    int n_structures) const {
  validate_full_determinant_input(
      alpha_det,
      beta_det,
      determinant_to_structure_terms,
      n_orbitals,
      n_structures);

  const int n_determinants = static_cast<int>(alpha_det.size());
  FullDeterminantStructureBuildResult build_result;
  build_result.n_determinants = n_determinants;
  build_result.structure_matrices =
      structure_pair_accumulator_.create_result(n_structures, n_determinants);
  build_result.pair_evaluations.reserve(
      static_cast<std::size_t>(n_determinants) * static_cast<std::size_t>(n_determinants));
  const FullDeterminantPairEvaluator pair_evaluator(
      determinant_overlap_resolver_,
      determinant_hamiltonian_resolver_);

  for (int left_det_idx = 0; left_det_idx < n_determinants; ++left_det_idx) {
    for (int right_det_idx = 0; right_det_idx < n_determinants; ++right_det_idx) {
      auto determinant_pair_result = pair_evaluator.evaluate(
          alpha_det[static_cast<std::size_t>(left_det_idx)],
          alpha_det[static_cast<std::size_t>(right_det_idx)],
          beta_det[static_cast<std::size_t>(left_det_idx)],
          beta_det[static_cast<std::size_t>(right_det_idx)],
          ovlp_act,
          h1e_act,
          n_orbitals,
          eri_act);
      structure_pair_accumulator_.accumulate(
          left_det_idx,
          right_det_idx,
          determinant_to_structure_terms[static_cast<std::size_t>(left_det_idx)],
          determinant_to_structure_terms[static_cast<std::size_t>(right_det_idx)],
          determinant_pair_result.overlap_determinant,
          determinant_pair_result.total_hamiltonian,
          determinant_pair_result.one_electron_hamiltonian,
          build_result.structure_matrices);
      build_result.pair_evaluations.push_back(std::move(determinant_pair_result));
    }
  }

  symmetrize_structure_matrices(build_result.structure_matrices);
  return build_result;
}


StructureAccumulationResult FullDeterminantStructureHamiltonianOverlapBuilder::build(
    const FullDeterminantStructureData& input) const 
{
  return build(
      input.alpha_det,
      input.beta_det,
      input.determinant_to_structure_terms,
      input.ovlp_act,
      input.h1e_act,
      input.n_active_orbitals,
      input.eri_act,
      input.n_structures);
}

FullDeterminantStructureBuildResult
FullDeterminantStructureHamiltonianOverlapBuilder::build_with_pair_evaluations(
    const FullDeterminantStructureData& input) const {
  return build_with_pair_evaluations(
      input.alpha_det,
      input.beta_det,
      input.determinant_to_structure_terms,
      input.ovlp_act,
      input.h1e_act,
      input.n_active_orbitals,
      input.eri_act,
      input.n_structures);
}


}  // namespace xmvb::vb
