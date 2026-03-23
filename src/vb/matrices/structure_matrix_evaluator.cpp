#include "vb/matrices/structure_matrix_evaluator.hpp"

namespace xmvb::vb {

StructureMatrixEvaluator::
    StructureMatrixEvaluator(VbScfAlgorithm algorithm)
    : orbital_preparer_(),
      ao_effective_one_electron_builder_(),
      active_space_one_electron_builder_(),
      active_space_two_electron_builder_(),
      structure_builder_(algorithm) {}

StructureMatrixEvaluator::
    StructureMatrixEvaluator(
        ActiveSpaceOrbitalPreparer orbital_preparer,
        AoEffectiveOneElectronBuilder ao_effective_one_electron_builder,
        ActiveSpaceOneElectronBuilder active_space_one_electron_builder,
        ActiveSpaceTwoElectronBuilder active_space_two_electron_builder,
        FullDeterminantStructureHamiltonianOverlapBuilder structure_builder)
    : orbital_preparer_(std::move(orbital_preparer)),
      ao_effective_one_electron_builder_(std::move(ao_effective_one_electron_builder)),
      active_space_one_electron_builder_(std::move(active_space_one_electron_builder)),
      active_space_two_electron_builder_(std::move(active_space_two_electron_builder)),
      structure_builder_(std::move(structure_builder)) {}

StructureAccumulationResult StructureMatrixEvaluator::evaluate(
    const CppVbInput& input) const {
  const int n_inactive_doubly_occupied_orbitals =
      (input.orbital_preparation_input.n_total_electrons -
       input.orbital_preparation_input.n_active_electrons) /
      2;

  const auto orbital_result =
      orbital_preparer_.prepare(input.orbital_preparation_input);
  const auto ao_effective_one_electron_result =
      ao_effective_one_electron_builder_.build(
          orbital_result.inactive_density_matrix,
          input.ao_integral_input.ao_core_hamiltonian_matrix,
          input.ao_integral_input.ao_two_electron_integral_values,
          input.ao_integral_input.ao_two_electron_integral_indices,
          input.ao_integral_input.n_basis_functions);
  const auto active_space_one_electron_result =
      active_space_one_electron_builder_.build(
          ao_effective_one_electron_result.ao_effective_one_electron_matrix,
          orbital_result.auxiliary_orbital_matrix,
          input.ao_integral_input.n_basis_functions,
          n_inactive_doubly_occupied_orbitals,
          input.orbital_preparation_input.n_active_orbitals);
  const auto active_space_two_electron_result =
      active_space_two_electron_builder_.build(
          input.ao_integral_input.ao_two_electron_integral_values,
          input.ao_integral_input.ao_two_electron_integral_indices,
          orbital_result,
          input.ao_integral_input.n_basis_functions,
          input.orbital_preparation_input.n_active_orbitals);

  return structure_builder_.build(
      input.structure_data.alpha_occupied_orbitals_by_determinant,
      input.structure_data.beta_occupied_orbitals_by_determinant,
      input.structure_data.determinant_to_structure_terms,
      orbital_result.active_orbital_overlap_matrix,
      active_space_one_electron_result.active_one_electron_matrix,
      input.orbital_preparation_input.n_active_orbitals,
      active_space_two_electron_result.packed_active_two_electron_integrals,
      input.structure_data.n_structures);
}

}  // namespace xmvb::vb
