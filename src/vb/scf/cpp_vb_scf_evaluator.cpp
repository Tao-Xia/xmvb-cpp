#include "vb/scf/cpp_vb_scf_evaluator.hpp"

#include <stdexcept>
#include <utility>

#include <Eigen/Core>

namespace xmvb::vb {

namespace {

double compute_average_structure_overlap(
    const std::vector<double>& overlap_matrix,
    int n_structures) {
  double diagonal_sum = 0.0;
  for (int structure_index = 0; structure_index < n_structures; ++structure_index) {
    const std::size_t diagonal_index =
        structure_index * n_structures + structure_index;
    diagonal_sum += overlap_matrix[diagonal_index];
  }
  return diagonal_sum / static_cast<double>(n_structures);
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
        eigenvalues[selected_state_indices[selection_index]];
  }
  return energy;
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
      generalized_eigensolver_() {}

CppVbScfEvaluator::CppVbScfEvaluator(
    StructureMatrixEvaluator matrix_evaluator,
    xmvb::core::GeneralizedEigensolver generalized_eigensolver)
    : matrix_evaluator_(std::move(matrix_evaluator)),
      generalized_eigensolver_(std::move(generalized_eigensolver)) {}

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
        result.electronic_state_energies[state_index];
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

}  // namespace xmvb::vb
