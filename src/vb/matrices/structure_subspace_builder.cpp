#include "vb/matrices/structure_subspace_builder.hpp"

#include <stdexcept>
#include <vector>

namespace xmvb::vb {

namespace {

std::vector<int> build_structure_index_remap(
    int n_structures,
    const std::vector<int>& selected_structure_indices) {
  if (n_structures <= 0) {
    throw std::invalid_argument("n_structures must be positive");
  }
  if (selected_structure_indices.empty()) {
    throw std::invalid_argument("selected_structure_indices must not be empty");
  }

  std::vector<int> structure_index_remap(
      static_cast<std::size_t>(n_structures),
      -1);
  for (std::size_t selected_offset = 0;
       selected_offset < selected_structure_indices.size();
       ++selected_offset) {
    const int structure_index = selected_structure_indices[selected_offset];
    if (structure_index < 0 || structure_index >= n_structures) {
      throw std::out_of_range("selected structure index is out of range");
    }
    if (structure_index_remap[static_cast<std::size_t>(structure_index)] >= 0) {
      throw std::invalid_argument("selected_structure_indices must be unique");
    }
    structure_index_remap[static_cast<std::size_t>(structure_index)] =
        static_cast<int>(selected_offset);
  }
  return structure_index_remap;
}

}  // namespace

FullDeterminantStructureData StructureSubspaceBuilder::build(
    const FullDeterminantStructureData& full_data,
    const std::vector<int>& selected_structure_indices) const {
  if (static_cast<int>(full_data.alpha_occupied_orbitals_by_determinant.size()) !=
          static_cast<int>(full_data.beta_occupied_orbitals_by_determinant.size()) ||
      static_cast<int>(full_data.alpha_occupied_orbitals_by_determinant.size()) !=
          static_cast<int>(full_data.determinant_to_structure_terms.size())) {
    throw std::invalid_argument("full determinant inputs are inconsistent");
  }

  const auto structure_index_remap = build_structure_index_remap(
      full_data.n_structures,
      selected_structure_indices);

  FullDeterminantStructureData result;
  result.n_structures = static_cast<int>(selected_structure_indices.size());
  result.n_active_orbitals = full_data.n_active_orbitals;
  result.basis_overlap_matrix = full_data.basis_overlap_matrix;
  result.one_electron_matrix = full_data.one_electron_matrix;
  result.packed_two_electron_integrals = full_data.packed_two_electron_integrals;

  for (std::size_t determinant_index = 0;
       determinant_index < full_data.determinant_to_structure_terms.size();
       ++determinant_index) {
    std::vector<StructureExpansionTerm> filtered_terms;
    for (const auto& term :
         full_data.determinant_to_structure_terms[determinant_index]) {
      if (term.structure_index < 0 || term.structure_index >= full_data.n_structures) {
        throw std::invalid_argument("structure expansion term index out of range");
      }
      const int remapped_index =
          structure_index_remap[static_cast<std::size_t>(term.structure_index)];
      if (remapped_index < 0) {
        continue;
      }
      filtered_terms.push_back(StructureExpansionTerm{remapped_index, term.coefficient});
    }

    if (filtered_terms.empty()) {
      continue;
    }

    result.alpha_occupied_orbitals_by_determinant.push_back(
        full_data.alpha_occupied_orbitals_by_determinant[determinant_index]);
    result.beta_occupied_orbitals_by_determinant.push_back(
        full_data.beta_occupied_orbitals_by_determinant[determinant_index]);
    result.determinant_to_structure_terms.push_back(std::move(filtered_terms));
  }

  return result;
}

CppVbInput StructureSubspaceBuilder::build(
    const CppVbInput& input,
    const std::vector<int>& selected_structure_indices) const {
  CppVbInput result;
  result.orbital_preparation_input = input.orbital_preparation_input;
  result.ao_integral_input = input.ao_integral_input;
  result.structure_data = build(input.structure_data, selected_structure_indices);
  return result;
}

}  // namespace xmvb::vb
