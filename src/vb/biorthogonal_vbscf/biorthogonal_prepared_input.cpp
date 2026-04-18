#include "vb/biorthogonal_vbscf/biorthogonal_prepared_input.hpp"

#include <stdexcept>

#include "vb/matrices/structure_matrix_evaluator.hpp"

namespace xmvb::vb::biorthogonal_vbscf {

xmvb::vb::FullDeterminantStructureData build_biorthogonal_full_structure_data(
    const xmvb::vb::CppVbInput& input,
    const xmvb::vb::PreparedActiveSpaceContext& prepared_active_space) {
  const int n_active_orbitals = input.orbital_preparation_input.n_active_orbitals;
  if (n_active_orbitals <= 0) {
    throw std::invalid_argument(
        "CppVbInput must carry a positive active-orbital count");
  }

  // Biorthogonal determinant kernels consume one explicit object containing
  // both the full determinant topology and the prepared active-space tensors.
  xmvb::vb::FullDeterminantStructureData structure_data = input.structure_data;
  structure_data.n_active_orbitals = n_active_orbitals;
  structure_data.ovlp_act =
      prepared_active_space.orbital_result.active_orbital_overlap_matrix;
  structure_data.h1e_act =
      prepared_active_space.active_space_one_electron_result.h1e_act;
  structure_data.eri_act =
      prepared_active_space.active_space_two_electron_result
          .packed_active_two_electron_integrals;
  return structure_data;
}

PreparedBiorthogonalInput prepare_biorthogonal_input(
    const xmvb::vb::CppVbInput& input) {
  StructureMatrixEvaluator structure_matrix_evaluator;

  PreparedBiorthogonalInput prepared_input;
  prepared_input.prepared_active_space =
      structure_matrix_evaluator.prepare_active_space(input);
  prepared_input.structure_data =
      build_biorthogonal_full_structure_data(
          input,
          prepared_input.prepared_active_space);
  return prepared_input;
}

}  // namespace xmvb::vb::biorthogonal_vbscf
