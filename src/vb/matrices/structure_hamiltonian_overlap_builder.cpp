#include "vb/matrices/structure_hamiltonian_overlap_builder.hpp"

#include <stdexcept>

namespace xmvb::vb {

StructureHamiltonianOverlapBuilder::StructureHamiltonianOverlapBuilder()
    : determinant_hamiltonian_resolver_(),
      structure_pair_accumulator_() {}

StructureHamiltonianOverlapBuilder::StructureHamiltonianOverlapBuilder(
    DeterminantHamiltonianResolver determinant_hamiltonian_resolver,
    StructurePairAccumulator structure_pair_accumulator)
    : determinant_hamiltonian_resolver_(std::move(determinant_hamiltonian_resolver)),
      structure_pair_accumulator_(std::move(structure_pair_accumulator)) {}

StructureAccumulationResult StructureHamiltonianOverlapBuilder::build(

    const std::vector<DeterminantPairInput>& determinant_pair_inputs,
    const std::vector<std::vector<StructureExpansionTerm>>& determinant_to_structure_terms,
    const std::vector<double>& h1e_act,
    int n_orbitals,
    const std::vector<double>& eri_act,
    int n_structures,
    int n_determinants) const {
  if (n_structures <= 0) {
    throw std::invalid_argument("n_structures must be positive");
  }
  if (n_determinants <= 0) {
    throw std::invalid_argument("n_determinants must be positive");
  }
  if (static_cast<int>(determinant_to_structure_terms.size()) != n_determinants) {
    throw std::invalid_argument(
        "determinant_to_structure_terms size must match n_determinants");
  }

  StructureAccumulationResult accumulation_result =
      structure_pair_accumulator_.create_result(n_structures, n_determinants);

  for (const auto& determinant_pair_input : determinant_pair_inputs) {
    if (determinant_pair_input.determinant_index_left < 0 ||
        determinant_pair_input.determinant_index_left >= n_determinants ||
        determinant_pair_input.determinant_index_right < 0 ||
        determinant_pair_input.determinant_index_right >= n_determinants) {
      throw std::invalid_argument("determinant pair index out of range");
    }

    const auto determinant_hamiltonian_result = determinant_hamiltonian_resolver_.resolve(
        determinant_pair_input.occ_L,
        determinant_pair_input.occ_R,
        determinant_pair_input.det_ovlp_mat,
        h1e_act,
        n_orbitals,
        eri_act);

    structure_pair_accumulator_.accumulate(
        determinant_pair_input.determinant_index_left,
        determinant_pair_input.determinant_index_right,
        determinant_to_structure_terms[static_cast<std::size_t>(
            determinant_pair_input.determinant_index_left)],
        determinant_to_structure_terms[static_cast<std::size_t>(
            determinant_pair_input.determinant_index_right)],
        determinant_hamiltonian_result.overlap_determinant,
        determinant_hamiltonian_result.total_hamiltonian,
        determinant_hamiltonian_result.one_electron_hamiltonian,
        accumulation_result);
  }

  return accumulation_result;
}

}  // namespace xmvb::vb
