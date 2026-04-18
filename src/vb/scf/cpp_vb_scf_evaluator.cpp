#include "vb/scf/cpp_vb_scf_evaluator.hpp"

#include <stdexcept>
#include <utility>

#include <Eigen/Core>

#include "vb/biorthogonal_vbscf/biorthogonal_exact_selected_structure.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_exact_selected_structure_scf.hpp"

namespace xmvb::vb {

namespace {

double compute_average_structure_overlap(
    const std::vector<double>& overlap_matrix,
    int n_structures) {
  double diagonal_sum = 0.0;
  for (int structure_index = 0; structure_index < n_structures; ++structure_index) {
    const std::size_t diagonal_index =
        xmvb::to_size(structure_index) * n_structures + structure_index;
    diagonal_sum += overlap_matrix[diagonal_index];
  }
  return diagonal_sum / static_cast<double>(n_structures);
}

double compute_average_structure_overlap(
    const Eigen::MatrixXd& overlap_matrix) {
  if (overlap_matrix.rows() != overlap_matrix.cols()) {
    throw std::invalid_argument("overlap_matrix must be square");
  }
  if (overlap_matrix.rows() <= 0) {
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

std::vector<double> dense_matrix_to_vector(
    const Eigen::MatrixXd& matrix) {
  return std::vector<double>(
      matrix.data(),
      matrix.data() + matrix.size());
}

double compute_selected_state_average_energy(
    const std::vector<double>& eigenvalues,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& normalized_weights) {
  double energy = 0.0;
  for (std::size_t selection_index = 0;
       selection_index < selected_state_indices.size();
       ++selection_index) {
    energy +=
        normalized_weights[selection_index] *
        eigenvalues[xmvb::to_size(selected_state_indices[selection_index])];
  }
  return energy;
}

CppVbScfResult build_cpp_vb_scf_result_from_exact_selected_matrix_build(
    const biorthogonal_vbscf::BiorthogonalExactSelectedStructureMatrixBuildResult&
        matrix_result,
    const xmvb::core::GeneralizedEigensolver& generalized_eigensolver,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& normalized_weights,
    double one_electron_reference_energy,
    double nuclear_repulsion_energy) {
  CppVbScfResult result;
  result.n_structures = matrix_result.n_selected_structures;
  result.nuclear_repulsion_energy = nuclear_repulsion_energy;
  result.one_electron_reference_energy = one_electron_reference_energy;
  result.selected_state_indices = selected_state_indices;
  result.state_average_weights = normalized_weights;
  result.structure_matrices.n_structures = matrix_result.n_selected_structures;
  result.structure_matrices.overlap_matrix =
      dense_matrix_to_vector(matrix_result.physical_structure_overlap);
  result.structure_matrices.hamiltonian_matrix =
      dense_matrix_to_vector(matrix_result.physical_structure_hamiltonian);
  result.average_structure_overlap =
      compute_average_structure_overlap(
          matrix_result.physical_structure_overlap);

  const auto eigen_result = generalized_eigensolver.solve(
      result.structure_matrices.hamiltonian_matrix,
      result.structure_matrices.overlap_matrix,
      result.n_structures);
  result.electronic_state_energies = eigen_result.eigenvalues;
  result.eigenvector_matrix = eigen_result.eigenvector_matrix;

  result.electronic_energy = compute_selected_state_average_energy(
      result.electronic_state_energies,
      selected_state_indices,
      normalized_weights);
  result.selected_state_total_energies.resize(selected_state_indices.size(), 0.0);
  for (std::size_t selection_index = 0;
       selection_index < selected_state_indices.size();
       ++selection_index) {
    const int state_index = selected_state_indices[selection_index];
    const double electronic_state_energy =
        result.electronic_state_energies[xmvb::to_size(state_index)];
    result.selected_state_total_energies[selection_index] =
        electronic_state_energy + nuclear_repulsion_energy;
  }
  result.total_energy =
      result.one_electron_reference_energy +
      result.electronic_energy +
      nuclear_repulsion_energy;
  return result;
}

std::vector<double> normalize_state_average_weights(
    const std::vector<double>& state_average_weights) {
  double weight_sum = 0.0;
  for (const double state_weight : state_average_weights) {
    if (state_weight < 0.0) {
      throw std::invalid_argument("state_average_weights must be non-negative");
    }
    weight_sum += state_weight;
  }
  if (weight_sum <= 0.0) {
    throw std::invalid_argument("state_average_weights must sum to a positive value");
  }

  std::vector<double> normalized_weights = state_average_weights;
  for (double& state_weight : normalized_weights) {
    state_weight /= weight_sum;
  }
  return normalized_weights;
}

std::vector<int> build_full_structure_index_range(int n_structures) {
  if (n_structures < 0) {
    throw std::invalid_argument("n_structures must be non-negative");
  }
  std::vector<int> indices(xmvb::to_size(n_structures));
  for (int structure_index = 0; structure_index < n_structures; ++structure_index) {
    indices[xmvb::to_size(structure_index)] = structure_index;
  }
  return indices;
}

void validate_state_selection(
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    int n_structures) {
  if (selected_state_indices.empty()) {
    throw std::invalid_argument("selected_state_indices must not be empty");
  }
  if (selected_state_indices.size() != state_average_weights.size()) {
    throw std::invalid_argument(
        "selected_state_indices and state_average_weights must have the same length");
  }
  for (const int state_index : selected_state_indices) {
    if (state_index < 0 || state_index >= n_structures) {
      throw std::out_of_range("selected state index is out of range");
    }
  }
}

}  // namespace

CppVbScfEvaluator::CppVbScfEvaluator(VBSCFAlgorithm algorithm)
    : matrix_evaluator_(algorithm),
      generalized_eigensolver_(),
      subspace_builder_(),
      algorithm_(algorithm) {}

CppVbScfEvaluator::CppVbScfEvaluator(
    StructureMatrixEvaluator matrix_evaluator,
    xmvb::core::GeneralizedEigensolver generalized_eigensolver)
    : matrix_evaluator_(std::move(matrix_evaluator)),
      generalized_eigensolver_(std::move(generalized_eigensolver)),
      subspace_builder_(),
      algorithm_(VBSCFAlgorithm::Original) {}

CppVbScfResult CppVbScfEvaluator::evaluate(
    const CppVbInput& input,
    double nuclear_repulsion_energy) const {
  return evaluate(input, {0}, {1.0}, nuclear_repulsion_energy);
}

CppVbScfResult CppVbScfEvaluator::evaluate(
    const CppVbInput& input,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    double nuclear_repulsion_energy) const {
  if (input.structure_data.n_structures <= 0) {
    throw std::invalid_argument("input.structure_data.n_structures must be positive");
  }

  validate_state_selection(
      selected_state_indices,
      state_average_weights,
      input.structure_data.n_structures);
  const std::vector<double> normalized_weights =
      normalize_state_average_weights(state_average_weights);

  if (algorithm_ == VBSCFAlgorithm::BiorthogonalExactSelected) {
    const auto prepared_active_space = matrix_evaluator_.prepare_active_space(input);
    const auto matrix_result =
        biorthogonal_vbscf::build_biorthogonal_exact_selected_structure_matrices(
            input,
            prepared_active_space,
            build_full_structure_index_range(input.structure_data.n_structures),
            1.0e-8);
    return build_cpp_vb_scf_result_from_exact_selected_matrix_build(
        matrix_result,
        generalized_eigensolver_,
        selected_state_indices,
        normalized_weights,
        prepared_active_space.one_electron_reference_energy,
        nuclear_repulsion_energy);
  }

  CppVbScfResult result;
  result.n_structures = input.structure_data.n_structures;
  result.nuclear_repulsion_energy = nuclear_repulsion_energy;
  result.selected_state_indices = selected_state_indices;
  result.state_average_weights = normalized_weights;
  const auto prepared_active_space = matrix_evaluator_.prepare_active_space(input);
  result.structure_matrices = matrix_evaluator_.evaluate(
      input,
      prepared_active_space);
  result.one_electron_reference_energy =
      prepared_active_space.one_electron_reference_energy;
  result.average_structure_overlap = compute_average_structure_overlap(
      result.structure_matrices.overlap_matrix,
      result.n_structures);

  const auto eigen_result = generalized_eigensolver_.solve(
      result.structure_matrices.hamiltonian_matrix,
      result.structure_matrices.overlap_matrix,
      result.n_structures);
  result.electronic_state_energies = eigen_result.eigenvalues;
  result.eigenvector_matrix = eigen_result.eigenvector_matrix;

  result.electronic_energy = compute_selected_state_average_energy(
      result.electronic_state_energies,
      selected_state_indices,
      normalized_weights);
  result.selected_state_total_energies.resize(selected_state_indices.size(), 0.0);
  for (std::size_t selection_index = 0;
       selection_index < selected_state_indices.size();
       ++selection_index) {
    const int state_index = selected_state_indices[selection_index];
    const double electronic_state_energy =
        result.electronic_state_energies[xmvb::to_size(state_index)];
    result.selected_state_total_energies[selection_index] =
        electronic_state_energy + nuclear_repulsion_energy;
  }
  result.total_energy =
      result.one_electron_reference_energy +
      result.electronic_energy +
      nuclear_repulsion_energy;

  return result;
}

double CppVbScfEvaluator::evaluate_energy_only(
    const CppVbInput& input,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    double nuclear_repulsion_energy) const {
  if (input.structure_data.n_structures <= 0) {
    throw std::invalid_argument("input.structure_data.n_structures must be positive");
  }

  validate_state_selection(
      selected_state_indices,
      state_average_weights,
      input.structure_data.n_structures);
  const std::vector<double> normalized_weights =
      normalize_state_average_weights(state_average_weights);

  if (algorithm_ == VBSCFAlgorithm::BiorthogonalExactSelected) {
    const auto prepared_active_space = matrix_evaluator_.prepare_active_space(input);
    const auto matrix_result =
        biorthogonal_vbscf::build_biorthogonal_exact_selected_structure_matrices(
            input,
            prepared_active_space,
            build_full_structure_index_range(input.structure_data.n_structures),
            1.0e-8);
    const std::vector<double> eigenvalues =
        generalized_eigensolver_.solve_eigenvalues_only(
            dense_matrix_to_vector(matrix_result.physical_structure_hamiltonian),
            dense_matrix_to_vector(matrix_result.physical_structure_overlap),
            matrix_result.n_selected_structures);
    return
        prepared_active_space.one_electron_reference_energy +
        compute_selected_state_average_energy(
            eigenvalues,
            selected_state_indices,
            normalized_weights) +
        nuclear_repulsion_energy;
  }

  const auto prepared_active_space = matrix_evaluator_.prepare_active_space(input);
  const auto structure_matrices = matrix_evaluator_.evaluate(
      input,
      prepared_active_space);
  const std::vector<double> eigenvalues =
      generalized_eigensolver_.solve_eigenvalues_only(
          structure_matrices.hamiltonian_matrix,
          structure_matrices.overlap_matrix,
          input.structure_data.n_structures);
  return
      prepared_active_space.one_electron_reference_energy +
      compute_selected_state_average_energy(
          eigenvalues,
          selected_state_indices,
          normalized_weights) +
      nuclear_repulsion_energy;
}

CppVbScfResult CppVbScfEvaluator::evaluate_subspace(
    const CppVbInput& input,
    const std::vector<int>& selected_structure_indices,
    double nuclear_repulsion_energy) const {
  return evaluate_subspace(
      input,
      selected_structure_indices,
      {0},
      {1.0},
      nuclear_repulsion_energy);
}

CppVbScfResult CppVbScfEvaluator::evaluate_subspace(
    const CppVbInput& input,
    const std::vector<int>& selected_structure_indices,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    double nuclear_repulsion_energy) const {
  if (algorithm_ == VBSCFAlgorithm::BiorthogonalExactSelected) {
    const auto prepared_active_space = matrix_evaluator_.prepare_active_space(input);
    validate_state_selection(
        selected_state_indices,
        state_average_weights,
        static_cast<int>(selected_structure_indices.size()));
    const std::vector<double> normalized_weights =
        normalize_state_average_weights(state_average_weights);
    const auto matrix_result =
        biorthogonal_vbscf::build_biorthogonal_exact_selected_structure_matrices(
            input,
            prepared_active_space,
            selected_structure_indices,
            1.0e-8);
    return build_cpp_vb_scf_result_from_exact_selected_matrix_build(
        matrix_result,
        generalized_eigensolver_,
        selected_state_indices,
        normalized_weights,
        prepared_active_space.one_electron_reference_energy,
        nuclear_repulsion_energy);
  }

  const CppVbInput subspace_input =
      subspace_builder_.build(input, selected_structure_indices);
  return evaluate(
      subspace_input,
      selected_state_indices,
      state_average_weights,
      nuclear_repulsion_energy);
}

}  // namespace xmvb::vb
