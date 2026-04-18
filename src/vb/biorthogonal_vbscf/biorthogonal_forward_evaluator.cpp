#include "vb/biorthogonal_vbscf/biorthogonal_forward_evaluator.hpp"

#include <stdexcept>
#include <string>

#include "vb/biorthogonal_vbscf/biorthogonal_prepared_input.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_structure_local_utils.hpp"
#include "vb/orbital/active_space_two_electron_utils.hpp"

namespace xmvb::vb::biorthogonal_vbscf {

namespace {

template <typename TwoElectronKernel>
BiorthogonalForwardEvaluationResult evaluate_biorthogonal_forward_from_active_space(
    const xmvb::vb::FullDeterminantStructureData& structure_data,
    const std::vector<double>& right_orbital_overlap,
    const std::vector<double>& right_right_one_electron,
    const TwoElectronKernel& right_right_two_electron_kernel,
    double imaginary_tolerance) {
  if (imaginary_tolerance < 0.0) {
    throw std::invalid_argument("imaginary_tolerance must be non-negative");
  }

  BiorthogonalForwardEvaluationResult result;
  result.n_active_orbitals = structure_data.n_active_orbitals;
  result.n_structures = structure_data.n_structures;
  result.structure_expansion =
      build_biorthogonal_structure_expansion(structure_data);
  result.n_determinants =
      static_cast<int>(result.structure_expansion.determinants.size());

  // The one-sided dual frame is completely determined by the prepared
  // active-space overlap `X = C^T S C` and the right/right one-electron
  // Hamiltonian `HHO`. The selected-space Hamiltonian is then assembled by the
  // unique-spin/block-contracted kernel from the same determinant topology and
  // the prepared active-space 2e representation.
  result.orbital_integrals = build_biorthogonal_orbital_integrals(
      structure_data.n_active_orbitals,
      right_orbital_overlap,
      right_right_one_electron);
  result.determinant_hamiltonian.resize(0, 0);
  result.structure_hamiltonian_build_result =
      build_biorthogonal_structure_hamiltonian(
          structure_data,
          result.orbital_integrals,
          right_right_two_electron_kernel);
  result.structure_space = build_biorthogonal_selected_structure_space(
      result.structure_expansion.structure_to_determinant);
  result.projected_problem =
      build_biorthogonal_projected_structure_problem_from_selected_hamiltonian(
      result.structure_hamiltonian_build_result.selected_structure_hamiltonian,
      result.structure_space);
  result.solve_result = solve_biorthogonal_projected_structure_problem(
      result.projected_problem,
      result.structure_space,
      imaginary_tolerance);
  result.right_determinant_coefficient_matrix =
      result.structure_expansion.structure_to_determinant *
      result.solve_result.right_structure_coefficient_matrix;
  result.left_determinant_coefficient_matrix =
      result.structure_expansion.structure_to_determinant *
      result.solve_result.left_structure_coefficient_matrix;

  validate_biorthogonal_forward_evaluation_result(
      result,
      1.0e-12,
      1.0e-10,
      1.0e-8,
      1.0e-8);
  return result;
}

}  // namespace

BiorthogonalForwardEvaluationResult evaluate_biorthogonal_forward(
    const xmvb::vb::FullDeterminantStructureData& structure_data,
    double imaginary_tolerance) {
  return evaluate_biorthogonal_forward_from_active_space(
      structure_data,
      structure_data.ovlp_act,
      structure_data.h1e_act,
      make_active_space_two_electron_view(structure_data.eri_act),
      imaginary_tolerance);
}

BiorthogonalForwardEvaluationResult evaluate_biorthogonal_forward(
    const xmvb::vb::CppVbInput& input,
    double imaginary_tolerance) {
  const PreparedBiorthogonalInput prepared_input =
      prepare_biorthogonal_input(input);

  return evaluate_biorthogonal_forward_from_active_space(
      prepared_input.structure_data,
      prepared_input.prepared_active_space
          .orbital_result.active_orbital_overlap_matrix,
      prepared_input.prepared_active_space
          .active_space_one_electron_result.h1e_act,
      prepared_input.prepared_active_space
          .active_space_two_electron_result,
      imaginary_tolerance);
}

void validate_biorthogonal_forward_evaluation_result(
    const BiorthogonalForwardEvaluationResult& evaluation_result,
    double metric_min_diagonal_tolerance,
    double orthogonalization_reconstruction_tolerance,
    double residual_tolerance,
    double biorthogonality_tolerance) {
  if (evaluation_result.n_active_orbitals <= 0) {
    throw std::invalid_argument("n_active_orbitals must be positive");
  }
  if (evaluation_result.n_structures <= 0) {
    throw std::invalid_argument("n_structures must be positive");
  }
  if (evaluation_result.n_determinants <= 0) {
    throw std::invalid_argument("n_determinants must be positive");
  }

  validate_biorthogonal_structure_expansion(
      evaluation_result.structure_expansion,
      evaluation_result.n_structures,
      evaluation_result.n_active_orbitals);
  validate_biorthogonal_orbital_integrals(
      evaluation_result.orbital_integrals);
  if (evaluation_result.determinant_hamiltonian.size() != 0) {
    if (evaluation_result.determinant_hamiltonian.rows() !=
            evaluation_result.n_determinants ||
        evaluation_result.determinant_hamiltonian.cols() !=
            evaluation_result.n_determinants) {
      throw std::invalid_argument(
          "determinant_hamiltonian dimensions are inconsistent");
    }
    throw_if_nonfinite(
        evaluation_result.determinant_hamiltonian,
        "determinant_hamiltonian");
  }
  validate_biorthogonal_structure_hamiltonian_build_result(
      evaluation_result.structure_hamiltonian_build_result,
      evaluation_result.n_structures);
  validate_biorthogonal_selected_structure_space(
      evaluation_result.structure_space,
      metric_min_diagonal_tolerance);
  validate_biorthogonal_projected_structure_problem(
      evaluation_result.projected_problem,
      evaluation_result.n_structures,
      orthogonalization_reconstruction_tolerance);
  validate_biorthogonal_projected_structure_solve_result(
      evaluation_result.solve_result,
      evaluation_result.n_structures,
      residual_tolerance,
      biorthogonality_tolerance);
  if (evaluation_result.right_determinant_coefficient_matrix.rows() !=
          evaluation_result.n_determinants ||
      evaluation_result.right_determinant_coefficient_matrix.cols() !=
          evaluation_result.n_structures) {
    throw std::invalid_argument(
        "right_determinant_coefficient_matrix dimensions are inconsistent");
  }
  if (evaluation_result.left_determinant_coefficient_matrix.rows() !=
          evaluation_result.n_determinants ||
      evaluation_result.left_determinant_coefficient_matrix.cols() !=
          evaluation_result.n_structures) {
    throw std::invalid_argument(
        "left_determinant_coefficient_matrix dimensions are inconsistent");
  }
  throw_if_nonfinite(
      evaluation_result.right_determinant_coefficient_matrix,
      "right_determinant_coefficient_matrix");
  throw_if_nonfinite(
      evaluation_result.left_determinant_coefficient_matrix,
      "left_determinant_coefficient_matrix");
}

}  // namespace xmvb::vb::biorthogonal_vbscf
