#include "vb/biorthogonal_vbscf/biorthogonal_exact_selected_structure_scf.hpp"

#include <stdexcept>
#include <vector>

#include "vb/scf/selected_state_determinant_matrices.hpp"

namespace xmvb::vb::biorthogonal_vbscf {

namespace {

std::vector<double> dense_matrix_to_vector(
    const Eigen::MatrixXd& matrix) {
  return std::vector<double>(
      matrix.data(),
      matrix.data() + matrix.size());
}

double compute_average_structure_overlap(
    const Eigen::MatrixXd& overlap_matrix) {
  if (overlap_matrix.rows() != overlap_matrix.cols()) {
    throw std::invalid_argument("overlap_matrix must be square");
  }
  if (overlap_matrix.rows() == 0) {
    throw std::invalid_argument("overlap_matrix must be non-empty");
  }

  double diagonal_sum = 0.0;
  for (int structure_index = 0;
       structure_index < overlap_matrix.rows();
       ++structure_index) {
    diagonal_sum += overlap_matrix(structure_index, structure_index);
  }
  return diagonal_sum / static_cast<double>(overlap_matrix.rows());
}

void validate_selected_state_indices(
    const std::vector<int>& selected_state_indices,
    int n_states) {
  if (selected_state_indices.empty()) {
    throw std::invalid_argument("selected_state_indices must not be empty");
  }
  for (const int state_index : selected_state_indices) {
    if (state_index < 0 || state_index >= n_states) {
      throw std::out_of_range("selected state index is out of range");
    }
  }
}

BiorthogonalExactSelectedStructureScfResult build_scf_result(
    const PreparedBiorthogonalInput& prepared_input,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& normalized_state_weights,
    double nuclear_repulsion_energy,
    BiorthogonalExactSelectedStructureEvaluationResult exact_evaluation) {
  if (selected_state_indices.size() != normalized_state_weights.size()) {
    throw std::invalid_argument(
        "selected_state_indices and normalized_state_weights must be aligned");
  }

  BiorthogonalExactSelectedStructureScfResult result;
  result.nuclear_repulsion_energy = nuclear_repulsion_energy;
  result.one_electron_reference_energy =
      prepared_input.prepared_active_space.one_electron_reference_energy;
  result.selected_state_indices = selected_state_indices;
  result.state_average_weights = normalized_state_weights;
  result.exact_evaluation = std::move(exact_evaluation);

  // The exact selected-space overlap and Hamiltonian are the physical
  // nonorthogonal structure matrices for the chosen structure subset. Exporting
  // them here lets higher-level callers reuse the same SCF-style matrix view
  // without re-deriving anything from the low-level exact result.
  result.structure_matrices.n_structures =
      result.exact_evaluation.n_selected_structures;
  result.structure_matrices.overlap_matrix =
      dense_matrix_to_vector(result.exact_evaluation.physical_structure_overlap);
  result.structure_matrices.hamiltonian_matrix =
      dense_matrix_to_vector(result.exact_evaluation.physical_structure_hamiltonian);
  result.average_structure_overlap =
      compute_average_structure_overlap(
          result.exact_evaluation.physical_structure_overlap);

  result.selected_state_total_energies.resize(selected_state_indices.size(), 0.0);
  for (std::size_t selection_index = 0;
       selection_index < selected_state_indices.size();
       ++selection_index) {
    const int state_index = selected_state_indices[selection_index];
    const double state_energy =
        result.exact_evaluation.eigenvalues[xmvb::to_size(state_index)];
    result.electronic_energy +=
        normalized_state_weights[selection_index] * state_energy;
    result.selected_state_total_energies[selection_index] =
        state_energy + nuclear_repulsion_energy;
  }
  result.total_energy =
      result.one_electron_reference_energy +
      result.electronic_energy +
      nuclear_repulsion_energy;
  return result;
}

PreparedBiorthogonalInput build_prepared_input(
    const xmvb::vb::CppVbInput& input,
    const xmvb::vb::PreparedActiveSpaceContext& prepared_active_space) {
  PreparedBiorthogonalInput prepared_input;
  prepared_input.prepared_active_space = prepared_active_space;
  prepared_input.structure_data =
      build_biorthogonal_full_structure_data(
          input,
          prepared_active_space);
  return prepared_input;
}

}  // namespace

xmvb::vb::CppVbScfResult
build_cpp_vb_scf_result_from_exact_selected_structure_result(
    const BiorthogonalExactSelectedStructureScfResult& exact_result) {
  xmvb::vb::CppVbScfResult result;
  result.n_structures = exact_result.exact_evaluation.n_selected_structures;
  result.nuclear_repulsion_energy = exact_result.nuclear_repulsion_energy;
  result.one_electron_reference_energy = exact_result.one_electron_reference_energy;
  result.electronic_energy = exact_result.electronic_energy;
  result.total_energy = exact_result.total_energy;
  result.average_structure_overlap = exact_result.average_structure_overlap;
  result.electronic_state_energies = exact_result.exact_evaluation.eigenvalues;
  result.selected_state_total_energies = exact_result.selected_state_total_energies;
  result.selected_state_indices = exact_result.selected_state_indices;
  result.state_average_weights = exact_result.state_average_weights;
  result.eigenvector_matrix =
      dense_matrix_to_vector(exact_result.exact_evaluation.structure_coefficient_matrix);
  result.structure_matrices = exact_result.structure_matrices;
  return result;
}

BiorthogonalExactSelectedStructureScfResult
evaluate_biorthogonal_exact_selected_structure_scf(
    const PreparedBiorthogonalInput& prepared_input,
    const std::vector<int>& selected_structure_indices,
    double nuclear_repulsion_energy,
    double symmetry_tolerance) {
  return evaluate_biorthogonal_exact_selected_structure_scf(
      prepared_input,
      selected_structure_indices,
      {0},
      {1.0},
      nuclear_repulsion_energy,
      symmetry_tolerance);
}

BiorthogonalExactSelectedStructureScfResult
evaluate_biorthogonal_exact_selected_structure_scf(
    const PreparedBiorthogonalInput& prepared_input,
    const std::vector<int>& selected_structure_indices,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    double nuclear_repulsion_energy,
    double symmetry_tolerance) {
  const auto exact_evaluation =
      evaluate_biorthogonal_exact_selected_structure_subspace(
          prepared_input.structure_data,
          selected_structure_indices,
          symmetry_tolerance);
  validate_selected_state_indices(
      selected_state_indices,
      exact_evaluation.n_selected_structures);
  const std::vector<double> normalized_state_weights =
      xmvb::vb::normalize_state_average_weights(state_average_weights);
  return build_scf_result(
      prepared_input,
      selected_state_indices,
      normalized_state_weights,
      nuclear_repulsion_energy,
      exact_evaluation);
}

BiorthogonalExactSelectedStructureScfResult
evaluate_biorthogonal_exact_selected_structure_scf(
    const xmvb::vb::CppVbInput& input,
    const xmvb::vb::PreparedActiveSpaceContext& prepared_active_space,
    const std::vector<int>& selected_structure_indices,
    double nuclear_repulsion_energy,
    double symmetry_tolerance) {
  return evaluate_biorthogonal_exact_selected_structure_scf(
      input,
      prepared_active_space,
      selected_structure_indices,
      {0},
      {1.0},
      nuclear_repulsion_energy,
      symmetry_tolerance);
}

BiorthogonalExactSelectedStructureScfResult
evaluate_biorthogonal_exact_selected_structure_scf(
    const xmvb::vb::CppVbInput& input,
    const xmvb::vb::PreparedActiveSpaceContext& prepared_active_space,
    const std::vector<int>& selected_structure_indices,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    double nuclear_repulsion_energy,
    double symmetry_tolerance) {
  return evaluate_biorthogonal_exact_selected_structure_scf(
      build_prepared_input(input, prepared_active_space),
      selected_structure_indices,
      selected_state_indices,
      state_average_weights,
      nuclear_repulsion_energy,
      symmetry_tolerance);
}

BiorthogonalExactSelectedStructureScfResult
evaluate_biorthogonal_exact_selected_structure_scf(
    const xmvb::vb::CppVbInput& input,
    const std::vector<int>& selected_structure_indices,
    double nuclear_repulsion_energy,
    double symmetry_tolerance) {
  return evaluate_biorthogonal_exact_selected_structure_scf(
      prepare_biorthogonal_input(input),
      selected_structure_indices,
      nuclear_repulsion_energy,
      symmetry_tolerance);
}

BiorthogonalExactSelectedStructureScfResult
evaluate_biorthogonal_exact_selected_structure_scf(
    const xmvb::vb::CppVbInput& input,
    const std::vector<int>& selected_structure_indices,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    double nuclear_repulsion_energy,
    double symmetry_tolerance) {
  return evaluate_biorthogonal_exact_selected_structure_scf(
      prepare_biorthogonal_input(input),
      selected_structure_indices,
      selected_state_indices,
      state_average_weights,
      nuclear_repulsion_energy,
      symmetry_tolerance);
}

}  // namespace xmvb::vb::biorthogonal_vbscf
