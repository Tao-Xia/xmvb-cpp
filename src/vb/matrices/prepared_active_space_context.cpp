#include "vb/matrices/prepared_active_space_context.hpp"

#include <chrono>
#include <stdexcept>

#include <Eigen/Core>

namespace xmvb::vb {

namespace {

using Matrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using ConstMatrixMap = Eigen::Map<const Matrix>;

}  // namespace

double compute_one_electron_reference_energy(
    const std::vector<double>& inactive_density_matrix,
    const std::vector<double>& ao_effective_h1e,
    const std::vector<double>& ao_core_hamiltonian_matrix,
    int n_basis_functions) {
  if (n_basis_functions <= 0) {
    throw std::invalid_argument("n_basis_functions must be positive");
  }

  const std::size_t matrix_size =
      static_cast<std::size_t>(n_basis_functions) * static_cast<std::size_t>(n_basis_functions);
  if (inactive_density_matrix.size() != matrix_size ||
      ao_effective_h1e.size() != matrix_size ||
      ao_core_hamiltonian_matrix.size() != matrix_size) {
    throw std::invalid_argument("one-electron reference energy input size mismatch");
  }

  const ConstMatrixMap inactive_density(
      inactive_density_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  const ConstMatrixMap effective_h1e(
      ao_effective_h1e.data(),
      n_basis_functions,
      n_basis_functions);
  const ConstMatrixMap core_hamiltonian(
      ao_core_hamiltonian_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  return (inactive_density.array() *
          (effective_h1e.array() + core_hamiltonian.array())).sum();
}

TimedPreparedActiveSpaceContext prepare_timed_active_space_context(
    const CppVbInput& input,
    const ActiveSpaceOrbitalPreparer& orbital_preparer,
    const AoEffectiveOneElectronBuilder& ao_effective_one_electron_builder,
    const ActiveSpaceOneElectronBuilder& active_space_one_electron_builder,
    const ActiveSpaceTwoElectronBuilder& active_space_two_electron_builder) {
  TimedPreparedActiveSpaceContext timed_context;
  auto& context = timed_context.prepared_active_space;

  auto stage_start_time = std::chrono::steady_clock::now();
  context.n_inactive_doubly_occupied_orbitals =
      (input.orbital_preparation_input.n_total_electrons -
       input.orbital_preparation_input.n_active_electrons) /
      2;
  context.orbital_result =
      orbital_preparer.prepare(input.orbital_preparation_input);
  timed_context.timings.orbital_preparation_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();

  stage_start_time = std::chrono::steady_clock::now();
  context.ao_effective_one_electron_result =
      ao_effective_one_electron_builder.build(
          context.orbital_result.inactive_density_matrix,
          input.ao_integral_input.ao_core_hamiltonian_matrix,
          input.ao_integral_input.ao_two_electron_integral_values,
          input.ao_integral_input.ao_two_electron_integral_indices,
          input.ao_integral_input.n_basis_functions);
  timed_context.timings.ao_effective_one_electron_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();

  stage_start_time = std::chrono::steady_clock::now();
  context.active_space_one_electron_result =
      active_space_one_electron_builder.build(
          context.ao_effective_one_electron_result.ao_effective_h1e,
          context.orbital_result.auxiliary_orbital_matrix,
          input.ao_integral_input.n_basis_functions,
          context.n_inactive_doubly_occupied_orbitals,
          input.orbital_preparation_input.n_active_orbitals);
  timed_context.timings.active_one_electron_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();

  stage_start_time = std::chrono::steady_clock::now();
  context.active_space_two_electron_result =
      active_space_two_electron_builder.build(
          input.ao_integral_input.ao_two_electron_integral_values,
          input.ao_integral_input.ao_two_electron_integral_indices,
          context.orbital_result,
          input.ao_integral_input.n_basis_functions,
          input.orbital_preparation_input.n_active_orbitals);
  timed_context.timings.active_two_electron_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();
  context.one_electron_reference_energy =
      compute_one_electron_reference_energy(
          context.orbital_result.inactive_density_matrix,
          context.ao_effective_one_electron_result.ao_effective_h1e,
          input.ao_integral_input.ao_core_hamiltonian_matrix,
          input.ao_integral_input.n_basis_functions);
  return timed_context;
}

PreparedActiveSpaceContext prepare_active_space_context(
    const CppVbInput& input,
    const ActiveSpaceOrbitalPreparer& orbital_preparer,
    const AoEffectiveOneElectronBuilder& ao_effective_one_electron_builder,
    const ActiveSpaceOneElectronBuilder& active_space_one_electron_builder,
    const ActiveSpaceTwoElectronBuilder& active_space_two_electron_builder) {
  return prepare_timed_active_space_context(
             input,
             orbital_preparer,
             ao_effective_one_electron_builder,
             active_space_one_electron_builder,
             active_space_two_electron_builder)
      .prepared_active_space;
}

}  // namespace xmvb::vb
