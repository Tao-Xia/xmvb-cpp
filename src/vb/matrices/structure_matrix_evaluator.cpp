#include "vb/matrices/structure_matrix_evaluator.hpp"

namespace xmvb::vb {

StructureMatrixEvaluator::
    StructureMatrixEvaluator(VBSCFAlgorithm algorithm)
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
  const auto prepared_active_space = prepare_active_space(input);
  return evaluate(input, prepared_active_space);
}

PreparedActiveSpaceContext StructureMatrixEvaluator::prepare_active_space(
    const CppVbInput& input) const {
  return prepare_active_space_context(
      input,
      orbital_preparer_,
      ao_effective_one_electron_builder_,
      active_space_one_electron_builder_,
      active_space_two_electron_builder_);
}

StructureAccumulationResult StructureMatrixEvaluator::evaluate(
    const CppVbInput& input,
    const PreparedActiveSpaceContext& prepared_active_space) const {
  return structure_builder_.build(
      input.structure_data.alpha_det,
      input.structure_data.beta_det,
      input.structure_data.determinant_to_structure_terms,
      prepared_active_space.orbital_result.active_orbital_overlap_matrix,
      prepared_active_space.active_space_one_electron_result.h1e_act,
      input.orbital_preparation_input.n_active_orbitals,
      prepared_active_space.active_space_two_electron_result,
      input.structure_data.n_structures);
}

}  // namespace xmvb::vb
