#include "pfaffian_vbscf/scf/pf_orbital_grad_eval.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>

#include "vb/matrices/cpp_vb_input_ri_cache.hpp"

namespace xmvb::pfaffian_vbscf {

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(const Clock::time_point& start_time) {
  return std::chrono::duration<double>(Clock::now() - start_time).count();
}

int get_sparse_coefficient_count(
    const xmvb::vb::OrbitalPreparationInput& input,
    int orbital_index) {
  const int n_basis_functions = input.n_basis_functions;
  const int explicit_count =
      input.orbital_basis_counts[orbital_index];
  if (explicit_count > 1) {
    return explicit_count;
  }

  int coefficient_count = 0;
  while (coefficient_count < n_basis_functions) {
    const int basis_index =
        input.orbital_basis_index_table[
            orbital_index * n_basis_functions +
            coefficient_count];
    if (basis_index == 0) {
      break;
    }
    ++coefficient_count;
  }
  return coefficient_count;
}

std::vector<int> collect_differentiable_parameter_indices(
    const xmvb::vb::OrbitalPreparationInput& input) {
  std::vector<int> parameter_indices;
  for (int orbital_index = 0; orbital_index < input.n_orbitals; ++orbital_index) {
    const int coefficient_count =
        get_sparse_coefficient_count(input, orbital_index);
    for (int coefficient_index = 0;
         coefficient_index < coefficient_count;
         ++coefficient_index) {
      parameter_indices.push_back(
          orbital_index * input.n_basis_functions + coefficient_index);
    }
  }
  return parameter_indices;
}

PfOrbitalGradResult finish_orbital_gradient_backpropagation(
    const xmvb::vb::CppVbInput& input,
    const PfActiveGradResult& active_gradient_result,
    const Clock::time_point& total_start_time,
    const xmvb::vb::AoEffectiveOneElectronBackpropagator&
        ao_effective_one_electron_backpropagator,
    const xmvb::vb::ActiveSpaceMatrixBackpropagator&
        active_space_matrix_backpropagator,
    const xmvb::vb::ActiveSpaceTwoElectronBackpropagator&
        active_space_two_electron_backpropagator,
    const xmvb::vb::ActiveSpaceOrbitalBackpropagator&
        active_space_orbital_backpropagator) {
  const int n_inactive_doubly_occupied_orbitals =
      (input.orbital_preparation_input.n_total_electrons -
       input.orbital_preparation_input.n_active_electrons) / 2;
  PfOrbitalGradResult result;
  result.orbital_prepare_dt = active_gradient_result.orbital_prepare_dt;
  result.ao_h1e_build_dt = active_gradient_result.ao_h1e_build_dt;
  result.active_h1e_build_dt = active_gradient_result.active_h1e_build_dt;
  result.active_2e_build_dt = active_gradient_result.active_2e_build_dt;
  result.pf_matrix_forward_dt = active_gradient_result.matrix_dt;
  result.pf_adjoint_dt = active_gradient_result.adj_dt;
  result.forward_wall_time_seconds = active_gradient_result.total_dt;
  result.active_space_grad_dt = result.forward_wall_time_seconds;
  result.scf_result = active_gradient_result.scf_result;

  std::vector<double> total_inactive_density_gradient(
      active_gradient_result.orbital_preparation_result.inactive_density_matrix.size(),
      0.0);
  if (total_inactive_density_gradient.size() !=
          active_gradient_result.ao_effective_one_electron_result.ao_effective_h1e.size() ||
      total_inactive_density_gradient.size() !=
          input.ao_integral_input.ao_core_hamiltonian_matrix.size()) {
    throw std::runtime_error("one-electron reference gradient size mismatch");
  }
  const double* ao_core_hamiltonian_data =
      input.ao_integral_input.ao_core_hamiltonian_matrix.data();
  const double* ao_effective_h1e_data =
      active_gradient_result.ao_effective_one_electron_result.ao_effective_h1e.data();
  for (std::size_t index = 0; index < total_inactive_density_gradient.size(); ++index) {
    total_inactive_density_gradient[index] =
        ao_effective_h1e_data[index] +
        ao_core_hamiltonian_data[index];
  }

  const Clock::time_point backprop_start_time = Clock::now();
  const Clock::time_point matrix_backprop_start_time = Clock::now();
  const auto active_space_matrix_backpropagation_result =
      active_space_matrix_backpropagator.backpropagate(
          active_gradient_result.sso_grad,
          active_gradient_result.hho_grad,
          input.orbital_preparation_input.ao_overlap_matrix,
          active_gradient_result.ao_effective_one_electron_result.ao_effective_h1e,
          active_gradient_result.orbital_preparation_result.auxiliary_orbital_matrix,
          input.orbital_preparation_input.n_basis_functions,
          n_inactive_doubly_occupied_orbitals,
          input.orbital_preparation_input.n_active_orbitals);
  result.matrix_backprop_dt = seconds_since(matrix_backprop_start_time);
  std::vector<double> total_ao_effective_one_electron_gradient =
      active_space_matrix_backpropagation_result.ao_effective_one_electron_gradient;
  if (total_ao_effective_one_electron_gradient.size() !=
      active_gradient_result.orbital_preparation_result.inactive_density_matrix.size()) {
    throw std::runtime_error("ao effective one-electron gradient size mismatch");
  }
  for (std::size_t index = 0; index < total_ao_effective_one_electron_gradient.size(); ++index) {
    total_ao_effective_one_electron_gradient[index] +=
        active_gradient_result.orbital_preparation_result
            .inactive_density_matrix.data()[index];
  }

  const Clock::time_point two_electron_backprop_start_time = Clock::now();
  const auto active_space_two_electron_backpropagation_result =
      active_gradient_result.act_eri_result.representation ==
              xmvb::vb::ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity
          ? active_space_two_electron_backpropagator.backpropagate(
                active_gradient_result.ri_active_pair_factor_grad,
                input,
                active_gradient_result.orbital_preparation_result,
                active_gradient_result.act_eri_result,
                input.orbital_preparation_input.n_basis_functions,
                n_inactive_doubly_occupied_orbitals,
                input.orbital_preparation_input.n_active_orbitals)
          : active_space_two_electron_backpropagator.backpropagate(
                active_gradient_result.ggo_grad,
                input.ao_integral_input.ao_two_electron_integral_values,
                input.ao_integral_input.ao_two_electron_integral_indices,
                active_gradient_result.orbital_preparation_result,
                active_gradient_result.act_eri_result,
                input.orbital_preparation_input.n_basis_functions,
                n_inactive_doubly_occupied_orbitals,
                input.orbital_preparation_input.n_active_orbitals);
  result.two_electron_backprop_dt = seconds_since(two_electron_backprop_start_time);
  const Clock::time_point ao_h1e_backprop_start_time = Clock::now();
  const auto ao_effective_one_electron_backpropagation_result =
      active_gradient_result.act_eri_result.representation ==
              xmvb::vb::ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity
          ? ao_effective_one_electron_backpropagator.backpropagate(
                total_ao_effective_one_electron_gradient,
                xmvb::vb::ensure_cpp_vb_input_ri_cache(input),
                input.ao_integral_input.n_basis_functions)
          : ao_effective_one_electron_backpropagator.backpropagate(
                total_ao_effective_one_electron_gradient,
                input.ao_integral_input);
  result.ao_h1e_backprop_dt = seconds_since(ao_h1e_backprop_start_time);
  for (std::size_t index = 0; index < total_inactive_density_gradient.size(); ++index) {
    total_inactive_density_gradient[index] +=
        ao_effective_one_electron_backpropagation_result
            .inactive_density_gradient[index];
  }

  if (active_space_matrix_backpropagation_result
              .active_auxiliary_orbital_gradient.rows() !=
          input.orbital_preparation_input.n_basis_functions ||
      active_space_matrix_backpropagation_result
              .active_auxiliary_orbital_gradient.cols() !=
          input.orbital_preparation_input.n_active_orbitals ||
      active_space_two_electron_backpropagation_result
              .active_auxiliary_orbital_gradient.rows() !=
          input.orbital_preparation_input.n_basis_functions ||
      active_space_two_electron_backpropagation_result
              .active_auxiliary_orbital_gradient.cols() !=
          input.orbital_preparation_input.n_active_orbitals) {
    throw std::runtime_error(
        "auxiliary orbital backpropagation result size mismatch");
  }
  Eigen::MatrixXd total_active_auxiliary_gradient =
      active_space_matrix_backpropagation_result.active_auxiliary_orbital_gradient;
  total_active_auxiliary_gradient.noalias() +=
      active_space_two_electron_backpropagation_result.active_auxiliary_orbital_gradient;
  const Eigen::Map<const Eigen::MatrixXd> total_inactive_density_gradient_matrix(
      total_inactive_density_gradient.data(),
      input.orbital_preparation_input.n_basis_functions,
      input.orbital_preparation_input.n_basis_functions);
  const Clock::time_point orbital_backprop_start_time = Clock::now();
  const auto orbital_backpropagation_result =
      active_space_orbital_backpropagator.backpropagate(
          total_active_auxiliary_gradient,
          total_inactive_density_gradient_matrix,
          input.orbital_preparation_input,
          active_gradient_result.orbital_preparation_result);
  result.orbital_backprop_dt = seconds_since(orbital_backprop_start_time);
  result.backprop_wall_time_seconds = seconds_since(backprop_start_time);

  result.sparse_orbital_energy_gradient =
      orbital_backpropagation_result.orbital_value_gradient;
  result.differentiable_parameter_indices =
      collect_differentiable_parameter_indices(input.orbital_preparation_input);
  result.total_dt = seconds_since(total_start_time);
  result.total_wall_time_seconds = seconds_since(total_start_time);
  return result;
}

}  // namespace

PfOrbitalGradEval::PfOrbitalGradEval()
    : active_gradient_evaluator_(),
      spin_adapted_active_gradient_evaluator_(),
      ao_effective_one_electron_backpropagator_(),
      active_space_matrix_backpropagator_(),
      active_space_two_electron_backpropagator_(),
      active_space_orbital_backpropagator_() {}

PfOrbitalGradEval::PfOrbitalGradEval(
    PfActiveGradEval active_gradient_evaluator,
    xmvb::vb::AoEffectiveOneElectronBackpropagator
        ao_effective_one_electron_backpropagator,
    xmvb::vb::ActiveSpaceMatrixBackpropagator
        active_space_matrix_backpropagator,
    xmvb::vb::ActiveSpaceTwoElectronBackpropagator
        active_space_two_electron_backpropagator,
    xmvb::vb::ActiveSpaceOrbitalBackpropagator
        active_space_orbital_backpropagator)
    : active_gradient_evaluator_(std::move(active_gradient_evaluator)),
      spin_adapted_active_gradient_evaluator_(),
      ao_effective_one_electron_backpropagator_(
          std::move(ao_effective_one_electron_backpropagator)),
      active_space_matrix_backpropagator_(
          std::move(active_space_matrix_backpropagator)),
      active_space_two_electron_backpropagator_(
          std::move(active_space_two_electron_backpropagator)),
      active_space_orbital_backpropagator_(
          std::move(active_space_orbital_backpropagator)) {}

PfOrbitalGradResult PfOrbitalGradEval::eval(
    const xmvb::vb::CppVbInput& input,
    const PfBasisData& basis,
    double nuclear_repulsion_energy) const {
  const Clock::time_point total_start_time = Clock::now();
  const Clock::time_point forward_start_time = Clock::now();
  const auto active_gradient_result =
      active_gradient_evaluator_.eval(input, basis, nuclear_repulsion_energy);
  PfOrbitalGradResult result =
      finish_orbital_gradient_backpropagation(
          input,
          active_gradient_result,
          total_start_time,
          ao_effective_one_electron_backpropagator_,
          active_space_matrix_backpropagator_,
          active_space_two_electron_backpropagator_,
          active_space_orbital_backpropagator_);
  result.forward_wall_time_seconds = seconds_since(forward_start_time);
  result.active_space_grad_dt = result.forward_wall_time_seconds;
  return result;
}

PfOrbitalGradResult PfOrbitalGradEval::eval(
    const xmvb::vb::CppVbInput& input,
    const PfSpinAdaptedBasisData& basis,
    double nuclear_repulsion_energy) const {
  const Clock::time_point total_start_time = Clock::now();
  const Clock::time_point forward_start_time = Clock::now();
  const auto active_gradient_result =
      spin_adapted_active_gradient_evaluator_.eval(
          input,
          basis,
          nuclear_repulsion_energy);
  PfOrbitalGradResult result =
      finish_orbital_gradient_backpropagation(
          input,
          active_gradient_result,
          total_start_time,
          ao_effective_one_electron_backpropagator_,
          active_space_matrix_backpropagator_,
          active_space_two_electron_backpropagator_,
          active_space_orbital_backpropagator_);
  result.forward_wall_time_seconds = seconds_since(forward_start_time);
  result.active_space_grad_dt = result.forward_wall_time_seconds;
  return result;
}

}  // namespace xmvb::pfaffian_vbscf
