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
      xmvb::to_size(n_structures),
      -1);
  for (std::size_t selected_offset = 0;
       selected_offset < selected_structure_indices.size();
       ++selected_offset) {
    const int structure_index = selected_structure_indices[selected_offset];
    if (structure_index < 0 || structure_index >= n_structures) {
      throw std::out_of_range("selected structure index is out of range");
    }
    if (structure_index_remap[xmvb::to_size(structure_index)] >= 0) {
      throw std::invalid_argument("selected_structure_indices must be unique");
    }
    structure_index_remap[xmvb::to_size(structure_index)] =
        static_cast<int>(selected_offset);
  }
  return structure_index_remap;
}

}  // namespace

FullDeterminantStructureData StructureSubspaceBuilder::build(
    const FullDeterminantStructureData& full_data,
    const std::vector<int>& selected_structure_indices) const {
  if (static_cast<int>(full_data.alpha_det.size()) !=
          static_cast<int>(full_data.beta_det.size()) ||
      static_cast<int>(full_data.alpha_det.size()) !=
          static_cast<int>(full_data.determinant_to_structure_terms.size())) {
    throw std::invalid_argument("full determinant inputs are inconsistent");
  }

  const auto structure_index_remap = build_structure_index_remap(
      full_data.n_structures,
      selected_structure_indices);

  FullDeterminantStructureData result;
  result.n_structures = static_cast<int>(selected_structure_indices.size());
  result.n_active_orbitals = full_data.n_active_orbitals;
  result.ovlp_act = full_data.ovlp_act;
  result.h1e_act = full_data.h1e_act;
  result.eri_act = full_data.eri_act;

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
          structure_index_remap[xmvb::to_size(term.structure_index)];
      if (remapped_index < 0) {
        continue;
      }
      filtered_terms.push_back(StructureExpansionTerm{remapped_index, term.coefficient});
    }

    if (filtered_terms.empty()) {
      continue;
    }

    result.alpha_det.push_back(
        full_data.alpha_det[determinant_index]);
    result.beta_det.push_back(
        full_data.beta_det[determinant_index]);
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
  result.libcint_input = input.libcint_input;
  result.auxiliary_libcint_input = input.auxiliary_libcint_input;
  result.standard_two_electron_mode = input.standard_two_electron_mode;
  result.pf_two_electron_mode = input.pf_two_electron_mode;
  result.ri_integral_provider_result = input.ri_integral_provider_result;
  result.structure_data = build(input.structure_data, selected_structure_indices);
  return result;
}

}  // namespace xmvb::vb
