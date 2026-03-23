#include "vb/scf/cpp_orbital_gradient_evaluator.hpp"

#include <chrono>
#include <stdexcept>
#include <utility>
#include <vector>

namespace xmvb::vb {

namespace {

int get_sparse_coefficient_count(
    const OrbitalPreparationInput& orbital_preparation_input,
    int orbital_index) {
  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  const int explicit_count =
      orbital_preparation_input.orbital_basis_counts[static_cast<std::size_t>(orbital_index)];
  if (explicit_count > 1) {
    return explicit_count;
  }

  int coefficient_count = 0;
  while (coefficient_count < n_basis_functions) {
    const int basis_function_index =
        orbital_preparation_input.orbital_basis_index_table
            [static_cast<std::size_t>(orbital_index) * n_basis_functions + coefficient_count];
    if (basis_function_index == 0) {
      break;
    }
    ++coefficient_count;
  }
  return coefficient_count;
}

std::vector<int> collect_differentiable_parameter_indices(
    const OrbitalPreparationInput& orbital_preparation_input) {
  std::vector<int> differentiable_parameter_indices;
  for (int orbital_index = 0; orbital_index < orbital_preparation_input.n_orbitals; ++orbital_index) {
    const int coefficient_count =
        get_sparse_coefficient_count(orbital_preparation_input, orbital_index);
    for (int coefficient_index = 0; coefficient_index < coefficient_count; ++coefficient_index) {
      differentiable_parameter_indices.push_back(
          orbital_index * orbital_preparation_input.n_basis_functions + coefficient_index);
    }
  }
  return differentiable_parameter_indices;
}

std::vector<double> build_reference_energy_inactive_density_gradient(
    const CppVbInput& input,
    const OrbitalPreparationResult& orbital_result,
    const AoEffectiveOneElectronResult& ao_effective_one_electron_result,
    const AoEffectiveOneElectronBackpropagator& ao_effective_one_electron_backpropagator) {
  std::vector<double> inactive_density_gradient(
      orbital_result.inactive_density_matrix.size(),
      0.0);
  if (inactive_density_gradient.size() !=
          ao_effective_one_electron_result.ao_effective_h1e.size() ||
      inactive_density_gradient.size() !=
          input.ao_integral_input.ao_core_hamiltonian_matrix.size()) {
    throw std::runtime_error("one-electron reference energy gradient size mismatch");
  }

  for (std::size_t index = 0; index < inactive_density_gradient.size(); ++index) {
    inactive_density_gradient[index] =
        ao_effective_one_electron_result.ao_effective_h1e[index] +
        input.ao_integral_input.ao_core_hamiltonian_matrix[index];
  }

  const auto ao_effective_one_electron_backpropagation_result =
      ao_effective_one_electron_backpropagator.backpropagate(
          orbital_result.inactive_density_matrix,
          input.ao_integral_input.ao_two_electron_integral_values,
          input.ao_integral_input.ao_two_electron_integral_indices,
          input.orbital_preparation_input.n_basis_functions);
  for (std::size_t index = 0; index < inactive_density_gradient.size(); ++index) {
    inactive_density_gradient[index] +=
        ao_effective_one_electron_backpropagation_result.inactive_density_gradient[index];
  }
  return inactive_density_gradient;
}

std::vector<double> build_reference_energy_orbital_gradient(
    const CppVbInput& input,
    const OrbitalPreparationResult& orbital_result,
    const AoEffectiveOneElectronResult& ao_effective_one_electron_result,
    const AoEffectiveOneElectronBackpropagator& ao_effective_one_electron_backpropagator,
    const ActiveSpaceOrbitalBackpropagator& active_space_orbital_backpropagator) {
  const std::vector<double> zero_auxiliary_gradient(
      orbital_result.auxiliary_orbital_matrix.size(),
      0.0);
  const std::vector<double> zero_active_orbital_overlap_gradient(
      static_cast<std::size_t>(input.orbital_preparation_input.n_active_orbitals) *
          input.orbital_preparation_input.n_active_orbitals,
      0.0);
  const std::vector<double> reference_energy_inactive_density_gradient =
      build_reference_energy_inactive_density_gradient(
          input,
          orbital_result,
          ao_effective_one_electron_result,
          ao_effective_one_electron_backpropagator);
  const auto reference_energy_orbital_backpropagation_result =
      active_space_orbital_backpropagator.backpropagate(
          zero_auxiliary_gradient,
          zero_active_orbital_overlap_gradient,
          reference_energy_inactive_density_gradient,
          input.orbital_preparation_input);
  return reference_energy_orbital_backpropagation_result.orbital_value_gradient;
}

}  // namespace

CppOrbitalGradientEvaluator::CppOrbitalGradientEvaluator(
    VBSCFAlgorithm algorithm,
    double finite_difference_step)
    : active_space_gradient_evaluator_(algorithm),
      orbital_preparer_(),
      ao_effective_one_electron_builder_(),
      ao_effective_one_electron_backpropagator_(),
      active_space_matrix_backpropagator_(),
      active_space_two_electron_backpropagator_(),
      active_space_orbital_backpropagator_(),
      finite_difference_step_(finite_difference_step) {
  if (finite_difference_step_ <= 0.0) {
    throw std::invalid_argument("finite_difference_step must be positive");
  }
}

CppOrbitalGradientEvaluator::CppOrbitalGradientEvaluator(
    CppActiveSpaceGradientEvaluator active_space_gradient_evaluator,
    ActiveSpaceOrbitalPreparer orbital_preparer,
    AoEffectiveOneElectronBuilder ao_effective_one_electron_builder,
    AoEffectiveOneElectronBackpropagator ao_effective_one_electron_backpropagator,
    ActiveSpaceMatrixBackpropagator active_space_matrix_backpropagator,
    ActiveSpaceTwoElectronBackpropagator active_space_two_electron_backpropagator,
    ActiveSpaceOrbitalBackpropagator active_space_orbital_backpropagator,
    double finite_difference_step)
    : active_space_gradient_evaluator_(std::move(active_space_gradient_evaluator)),
      orbital_preparer_(std::move(orbital_preparer)),
      ao_effective_one_electron_builder_(std::move(ao_effective_one_electron_builder)),
      ao_effective_one_electron_backpropagator_(std::move(ao_effective_one_electron_backpropagator)),
      active_space_matrix_backpropagator_(std::move(active_space_matrix_backpropagator)),
      active_space_two_electron_backpropagator_(std::move(active_space_two_electron_backpropagator)),
      active_space_orbital_backpropagator_(std::move(active_space_orbital_backpropagator)),
      finite_difference_step_(finite_difference_step) {
  if (finite_difference_step_ <= 0.0) {
    throw std::invalid_argument("finite_difference_step must be positive");
  }
}

CppOrbitalGradientResult CppOrbitalGradientEvaluator::evaluate(
    const CppVbInput& input,
    double nuclear_repulsion_energy) const {
  return evaluate(input, {0}, {1.0}, nuclear_repulsion_energy);
}

CppOrbitalGradientResult CppOrbitalGradientEvaluator::evaluate(
    const CppVbInput& input,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    double nuclear_repulsion_energy) const {
  const auto total_start_time = std::chrono::steady_clock::now();
  if (input.orbital_preparation_input.orbital_value_table.empty()) {
    throw std::invalid_argument("orbital_value_table must not be empty");
  }

  const int n_inactive_doubly_occupied_orbitals =
      (input.orbital_preparation_input.n_total_electrons -
       input.orbital_preparation_input.n_active_electrons) / 2;
  auto stage_start_time = std::chrono::steady_clock::now();
  const auto active_space_gradient_result = active_space_gradient_evaluator_.evaluate(
      input,
      selected_state_indices,
      state_average_weights,
      nuclear_repulsion_energy);
  CppOrbitalGradientResult result;
  result.active_space_gradient_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();
  result.orbital_preparation_wall_time_seconds =
      active_space_gradient_result.orbital_preparation_wall_time_seconds;
  result.ao_effective_one_electron_wall_time_seconds =
      active_space_gradient_result.ao_effective_one_electron_wall_time_seconds;
  result.active_one_electron_wall_time_seconds =
      active_space_gradient_result.active_one_electron_wall_time_seconds;
  result.active_two_electron_wall_time_seconds =
      active_space_gradient_result.active_two_electron_wall_time_seconds;
  result.structure_matrix_wall_time_seconds =
      active_space_gradient_result.structure_matrix_wall_time_seconds;
  result.eigensolver_wall_time_seconds =
      active_space_gradient_result.eigensolver_wall_time_seconds;
  result.active_space_adjoint_wall_time_seconds =
      active_space_gradient_result.adjoint_wall_time_seconds;
  const auto& orbital_result = active_space_gradient_result.orbital_preparation_result;
  const auto& ao_effective_one_electron_result =
      active_space_gradient_result.ao_effective_one_electron_result;
  std::vector<double> total_inactive_density_gradient(
      orbital_result.inactive_density_matrix.size(),
      0.0);
  if (total_inactive_density_gradient.size() !=
          ao_effective_one_electron_result.ao_effective_h1e.size() ||
      total_inactive_density_gradient.size() !=
          input.ao_integral_input.ao_core_hamiltonian_matrix.size()) {
    throw std::runtime_error("one-electron reference energy gradient size mismatch");
  }
  for (std::size_t index = 0; index < total_inactive_density_gradient.size(); ++index) {
    total_inactive_density_gradient[index] =
        ao_effective_one_electron_result.ao_effective_h1e[index] +
        input.ao_integral_input.ao_core_hamiltonian_matrix[index];
  }
  const std::vector<double> zero_active_orbital_overlap_gradient(
      active_space_gradient_result.active_orbital_overlap_gradient.size(),
      0.0);

  stage_start_time = std::chrono::steady_clock::now();
  const auto active_space_matrix_backpropagation_result =
      active_space_matrix_backpropagator_.backpropagate(
          active_space_gradient_result.active_orbital_overlap_gradient,
          active_space_gradient_result.active_one_electron_gradient,
          input.orbital_preparation_input.active_orbital_overlap_matrix,
          ao_effective_one_electron_result.ao_effective_h1e,
          orbital_result.auxiliary_orbital_matrix,
          input.orbital_preparation_input.n_basis_functions,
          n_inactive_doubly_occupied_orbitals,
          input.orbital_preparation_input.n_active_orbitals);
  result.active_space_matrix_backpropagation_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();
  std::vector<double> total_ao_effective_one_electron_gradient =
      active_space_matrix_backpropagation_result.ao_effective_one_electron_gradient;
  if (total_ao_effective_one_electron_gradient.size() !=
      orbital_result.inactive_density_matrix.size()) {
    throw std::runtime_error("ao effective one-electron gradient size mismatch");
  }
  for (std::size_t index = 0; index < total_ao_effective_one_electron_gradient.size(); ++index) {
    total_ao_effective_one_electron_gradient[index] +=
        orbital_result.inactive_density_matrix[index];
  }
  stage_start_time = std::chrono::steady_clock::now();
  const auto active_space_two_electron_backpropagation_result =
      active_space_two_electron_backpropagator_.backpropagate(
          active_space_gradient_result.packed_active_two_electron_gradient,
          input.ao_integral_input.ao_two_electron_integral_values,
          input.ao_integral_input.ao_two_electron_integral_indices,
          orbital_result,
          input.orbital_preparation_input.n_basis_functions,
          n_inactive_doubly_occupied_orbitals,
          input.orbital_preparation_input.n_active_orbitals);
  result.active_space_two_electron_backpropagation_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();
  stage_start_time = std::chrono::steady_clock::now();
  const auto ao_effective_one_electron_backpropagation_result =
      ao_effective_one_electron_backpropagator_.backpropagate(
          total_ao_effective_one_electron_gradient,
          input.ao_integral_input.ao_two_electron_integral_values,
          input.ao_integral_input.ao_two_electron_integral_indices,
          input.orbital_preparation_input.n_basis_functions);
  result.ao_effective_one_electron_backpropagation_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();
  for (std::size_t index = 0; index < total_inactive_density_gradient.size(); ++index) {
    total_inactive_density_gradient[index] +=
        ao_effective_one_electron_backpropagation_result.inactive_density_gradient[index];
  }

  std::vector<double> total_auxiliary_orbital_gradient =
      active_space_matrix_backpropagation_result.auxiliary_orbital_gradient;
  if (total_auxiliary_orbital_gradient.size() !=
      active_space_two_electron_backpropagation_result.auxiliary_orbital_gradient.size()) {
    throw std::runtime_error("auxiliary orbital backpropagation result size mismatch");
  }
  for (std::size_t index = 0; index < total_auxiliary_orbital_gradient.size(); ++index) {
    total_auxiliary_orbital_gradient[index] +=
        active_space_two_electron_backpropagation_result.auxiliary_orbital_gradient[index];
  }

  stage_start_time = std::chrono::steady_clock::now();
  const auto orbital_backpropagation_result =
      active_space_orbital_backpropagator_.backpropagate(
          total_auxiliary_orbital_gradient,
          zero_active_orbital_overlap_gradient,
          total_inactive_density_gradient,
          input.orbital_preparation_input);
  result.orbital_backpropagation_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();
  result.finite_difference_step = finite_difference_step_;
  result.scf_result = active_space_gradient_result.scf_result;
  result.sparse_orbital_energy_gradient =
      orbital_backpropagation_result.orbital_value_gradient;
  result.sparse_orbital_reference_energy_gradient =
      build_reference_energy_orbital_gradient(
          input,
          orbital_result,
          ao_effective_one_electron_result,
          ao_effective_one_electron_backpropagator_,
          active_space_orbital_backpropagator_);
  result.differentiable_parameter_indices = collect_differentiable_parameter_indices(
      input.orbital_preparation_input);
  result.total_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start_time).count();
  return result;
}

}  // namespace xmvb::vb
