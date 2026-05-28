#include "vb/scf/cpp_orbital_gradient_evaluator.hpp"

#include <chrono>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "vb/matrices/cpp_vb_input_ri_cache.hpp"
#include "vb/matrices/eigen_matrix_storage_utils.hpp"
#include "vb/matrices/two_electron_indexer.hpp"
#include "vb/orbital/active_space_two_electron_utils.hpp"
#include "vb/scf/cpp_active_space_second_order_context.hpp"

namespace xmvb::vb {

namespace {

bool use_standard_ri_ao_effective_one_electron_path(const CppVbInput& input) {
  return input.standard_two_electron_mode ==
      StandardTwoElectronMode::ResolutionOfIdentity;
}

std::vector<int> collect_differentiable_parameter_indices(
    const OrbitalPreparationInput& orbital_preparation_input) {
  std::vector<int> differentiable_parameter_indices;
  for (int orbital_index = 0; orbital_index < orbital_preparation_input.n_orbitals; ++orbital_index) {
    const int coefficient_count =
        differentiable_sparse_orbital_parameter_count(
            orbital_preparation_input,
            orbital_index);
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
  const int n_inactive_doubly_occupied_orbitals =
      (input.orbital_preparation_input.n_total_electrons -
       input.orbital_preparation_input.n_active_electrons) / 2;
  std::vector<double> inactive_density_gradient(
      orbital_result.inactive_density_matrix.size(),
      0.0);
  if (inactive_density_gradient.size() !=
          ao_effective_one_electron_result.ao_effective_h1e.size() ||
      inactive_density_gradient.size() !=
          input.ao_integral_input.ao_core_hamiltonian_matrix.size()) {
    throw std::runtime_error("one-electron reference energy gradient size mismatch");
  }
  const double* ao_core_hamiltonian_data =
      input.ao_integral_input.ao_core_hamiltonian_matrix.data();
  const double* ao_effective_h1e_data =
      ao_effective_one_electron_result.ao_effective_h1e.data();

  for (std::size_t index = 0; index < inactive_density_gradient.size(); ++index) {
    inactive_density_gradient[index] =
        ao_effective_h1e_data[index] +
        ao_core_hamiltonian_data[index];
  }

  const auto ao_effective_one_electron_backpropagation_result =
      use_standard_ri_ao_effective_one_electron_path(input)
          ? ao_effective_one_electron_backpropagator.backpropagate(
                std::vector<double>(
                    input.orbital_preparation_input.n_active_orbitals *
                        input.orbital_preparation_input.n_active_orbitals,
                    0.0),
                orbital_result,
                ensure_cpp_vb_input_ri_cache(input),
                input.ao_integral_input.n_basis_functions,
                n_inactive_doubly_occupied_orbitals,
                input.orbital_preparation_input.n_active_orbitals)
          : ao_effective_one_electron_backpropagator.backpropagate(
                orbital_result.inactive_density_matrix,
                input.ao_integral_input);
  for (std::size_t index = 0; index < inactive_density_gradient.size(); ++index) {
    inactive_density_gradient[index] +=
        ao_effective_one_electron_backpropagation_result.inactive_density_gradient[index];
  }
  return inactive_density_gradient;
}

Eigen::MatrixXd build_active_pair_gradient_matrix(
    const std::vector<double>& packed_active_two_electron_gradient,
    int n_active_orbitals) {
  const std::size_t n_active_pairs =
      n_active_orbitals * (n_active_orbitals + 1) / 2;
  Eigen::MatrixXd active_pair_gradient_matrix =
      Eigen::MatrixXd::Zero(
          static_cast<Eigen::Index>(n_active_pairs),
          static_cast<Eigen::Index>(n_active_pairs));

  for (int row_first = 0; row_first < n_active_orbitals; ++row_first) {
    for (int row_second = 0; row_second <= row_first; ++row_second) {
      const std::size_t row_pair_index =
          row_first * (row_first + 1) / 2 + row_second;
      for (int column_first = 0; column_first < n_active_orbitals; ++column_first) {
        for (int column_second = 0; column_second <= column_first; ++column_second) {
          const std::size_t column_pair_index =
              column_first * (column_first + 1) / 2 + column_second;
          const int packed_index =
              (row_pair_index >= column_pair_index)
                  ? TwoElectronIndexer::two_electron_storage_index(
                        row_first,
                        row_second,
                        column_first,
                        column_second)
                  : TwoElectronIndexer::two_electron_storage_index(
                        column_first,
                        column_second,
                        row_first,
                        row_second);
          double value =
              packed_active_two_electron_gradient[packed_index];
          if (row_pair_index == column_pair_index) {
            value *= 2.0;
          }
          active_pair_gradient_matrix(
              static_cast<Eigen::Index>(row_pair_index),
              static_cast<Eigen::Index>(column_pair_index)) = value;
        }
      }
    }
  }

  return active_pair_gradient_matrix;
}

std::vector<double> build_ri_active_pair_factor_gradient(
    const std::vector<double>& packed_active_two_electron_gradient,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    int n_active_orbitals) {
  if (active_space_two_electron_result.representation !=
      ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity) {
    throw std::invalid_argument("RI active-pair-factor gradient requires an RI forward result");
  }
  const std::size_t n_active_pairs =
      n_active_orbitals * (n_active_orbitals + 1) / 2;
  const std::size_t expected_factor_size =
      active_space_two_electron_result.n_auxiliary_functions *
      n_active_pairs;
  if (active_space_two_electron_result.ri_active_pair_factors.size() !=
          expected_factor_size ||
      active_space_two_electron_result.ri_active_pair_factors.rows() !=
          active_space_two_electron_result.n_auxiliary_functions ||
      active_space_two_electron_result.ri_active_pair_factors.cols() !=
          static_cast<Eigen::Index>(n_active_pairs)) {
    throw std::invalid_argument("RI active-pair-factor buffer size mismatch");
  }

  const auto active_pair_gradient_matrix =
      build_active_pair_gradient_matrix(
          packed_active_two_electron_gradient,
          n_active_orbitals);
  const Eigen::MatrixXd ri_active_pair_factor_gradient =
      active_space_two_electron_result.ri_active_pair_factors *
      active_pair_gradient_matrix;
  return flatten_matrix_column_major(ri_active_pair_factor_gradient);
}

std::vector<double> build_reference_energy_orbital_gradient(
    const CppVbInput& input,
    const OrbitalPreparationResult& orbital_result,
    const AoEffectiveOneElectronResult& ao_effective_one_electron_result,
    const AoEffectiveOneElectronBackpropagator& ao_effective_one_electron_backpropagator,
    const ActiveSpaceOrbitalBackpropagator& active_space_orbital_backpropagator) {
  const std::vector<double> zero_active_auxiliary_gradient(
      input.orbital_preparation_input.n_basis_functions *
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
          zero_active_auxiliary_gradient,
          reference_energy_inactive_density_gradient,
          input.orbital_preparation_input,
          orbital_result);
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
    CppActiveSpaceGradientResult active_space_gradient_result) const {
  auto result = evaluate_without_reference_energy_gradient(
      input,
      std::move(active_space_gradient_result));
  const auto reference_start_time = std::chrono::steady_clock::now();
  populate_reference_energy_gradient(input, &result);
  result.total_wall_time_seconds +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - reference_start_time)
          .count();
  return result;
}

CppOrbitalGradientResult CppOrbitalGradientEvaluator::evaluate_without_reference_energy_gradient(
    const CppVbInput& input,
    double nuclear_repulsion_energy) const {
  return evaluate_without_reference_energy_gradient(
      input,
      {0},
      {1.0},
      nuclear_repulsion_energy);
}

CppOrbitalGradientResult
CppOrbitalGradientEvaluator::evaluate_without_reference_energy_gradient(
    const CppVbInput& input,
    CppActiveSpaceGradientResult active_space_gradient_result) const {
  const auto total_start_time = std::chrono::steady_clock::now();
  if (input.orbital_preparation_input.orbital_value_table.empty()) {
    throw std::invalid_argument("orbital_value_table must not be empty");
  }
  return evaluate_from_active_space_gradient_result(
      input,
      std::move(active_space_gradient_result),
      total_start_time);
}

CppOrbitalGradientResult CppOrbitalGradientEvaluator::evaluate(
    const CppVbInput& input,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    double nuclear_repulsion_energy) const {
  auto result = evaluate_without_reference_energy_gradient(
      input,
      selected_state_indices,
      state_average_weights,
      nuclear_repulsion_energy);
  const auto reference_start_time = std::chrono::steady_clock::now();
  populate_reference_energy_gradient(input, &result);
  result.total_wall_time_seconds +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - reference_start_time).count();
  return result;
}

CppOrbitalGradientResult CppOrbitalGradientEvaluator::evaluate_without_reference_energy_gradient(
    const CppVbInput& input,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    double nuclear_repulsion_energy) const {
  const auto total_start_time = std::chrono::steady_clock::now();
  if (input.orbital_preparation_input.orbital_value_table.empty()) {
    throw std::invalid_argument("orbital_value_table must not be empty");
  }

  const auto active_space_gradient_result = active_space_gradient_evaluator_.evaluate(
      input,
      selected_state_indices,
      state_average_weights,
      nuclear_repulsion_energy);
  return evaluate_from_active_space_gradient_result(
      input,
      std::move(active_space_gradient_result),
      total_start_time);
}

CppOrbitalGradientResult CppOrbitalGradientEvaluator::evaluate_from_active_space_gradient_result(
    const CppVbInput& input,
    CppActiveSpaceGradientResult active_space_gradient_result,
    const std::chrono::steady_clock::time_point& total_start_time) const {
  const int n_inactive_doubly_occupied_orbitals =
      (input.orbital_preparation_input.n_total_electrons -
       input.orbital_preparation_input.n_active_electrons) / 2;
  auto stage_start_time = std::chrono::steady_clock::now();
  CppOrbitalGradientResult result;
  result.active_space_gradient_wall_time_seconds =
      active_space_gradient_result.total_wall_time_seconds;
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
  const double* ao_core_hamiltonian_data =
      input.ao_integral_input.ao_core_hamiltonian_matrix.data();
  const double* ao_effective_h1e_data =
      ao_effective_one_electron_result.ao_effective_h1e.data();
  for (std::size_t index = 0; index < total_inactive_density_gradient.size(); ++index) {
    total_inactive_density_gradient[index] =
        ao_effective_h1e_data[index] +
        ao_core_hamiltonian_data[index];
  }

  stage_start_time = std::chrono::steady_clock::now();
  const auto active_space_matrix_backpropagation_result =
      active_space_matrix_backpropagator_.backpropagate(
          active_space_gradient_result.active_orbital_overlap_gradient,
          active_space_gradient_result.active_one_electron_gradient,
          input.orbital_preparation_input.ao_overlap_matrix,
          ao_effective_one_electron_result.ao_effective_h1e,
          orbital_result.auxiliary_orbital_matrix,
          input.orbital_preparation_input.n_basis_functions,
          n_inactive_doubly_occupied_orbitals,
          input.orbital_preparation_input.n_active_orbitals);
  result.active_space_matrix_backpropagation_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();
  stage_start_time = std::chrono::steady_clock::now();
  const auto active_space_two_electron_backpropagation_result =
      active_space_gradient_result.active_space_two_electron_result.representation ==
              ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity
          ? active_space_two_electron_backpropagator_.backpropagate(
                build_ri_active_pair_factor_gradient(
                    active_space_gradient_result.packed_active_two_electron_gradient,
                    active_space_gradient_result.active_space_two_electron_result,
                    input.orbital_preparation_input.n_active_orbitals),
                input,
                orbital_result,
                active_space_gradient_result.active_space_two_electron_result,
                input.orbital_preparation_input.n_basis_functions,
                n_inactive_doubly_occupied_orbitals,
                input.orbital_preparation_input.n_active_orbitals)
          : active_space_two_electron_backpropagator_.backpropagate(
                active_space_gradient_result.packed_active_two_electron_gradient,
                input.ao_integral_input.ao_two_electron_integral_values,
                input.ao_integral_input.ao_two_electron_integral_indices,
                orbital_result,
                active_space_gradient_result.active_space_two_electron_result,
                input.orbital_preparation_input.n_basis_functions,
                n_inactive_doubly_occupied_orbitals,
                input.orbital_preparation_input.n_active_orbitals);
  result.active_space_two_electron_backpropagation_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();
  stage_start_time = std::chrono::steady_clock::now();
  const auto ao_effective_one_electron_backpropagation_result =
      use_standard_ri_ao_effective_one_electron_path(input)
          ? ao_effective_one_electron_backpropagator_.backpropagate(
                active_space_gradient_result.active_one_electron_gradient,
                orbital_result,
                ensure_cpp_vb_input_ri_cache(input),
                input.ao_integral_input.n_basis_functions,
                n_inactive_doubly_occupied_orbitals,
                input.orbital_preparation_input.n_active_orbitals)
          : [&]() {
              std::vector<double> total_ao_effective_one_electron_gradient =
                  active_space_matrix_backpropagation_result.ao_effective_one_electron_gradient;
              if (total_ao_effective_one_electron_gradient.size() !=
                  orbital_result.inactive_density_matrix.size()) {
                throw std::runtime_error("ao effective one-electron gradient size mismatch");
              }
              for (std::size_t index = 0;
                   index < total_ao_effective_one_electron_gradient.size();
                   ++index) {
                total_ao_effective_one_electron_gradient[index] +=
                    orbital_result.inactive_density_matrix.data()[index];
              }
              return ao_effective_one_electron_backpropagator_.backpropagate(
                  total_ao_effective_one_electron_gradient,
                  input.ao_integral_input);
            }();
  result.ao_effective_one_electron_backpropagation_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();
  for (std::size_t index = 0; index < total_inactive_density_gradient.size(); ++index) {
    total_inactive_density_gradient[index] +=
        ao_effective_one_electron_backpropagation_result.inactive_density_gradient[index];
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
    throw std::runtime_error("auxiliary orbital backpropagation result size mismatch");
  }
  Eigen::MatrixXd total_active_auxiliary_gradient =
      active_space_matrix_backpropagation_result.active_auxiliary_orbital_gradient;
  total_active_auxiliary_gradient.noalias() +=
      active_space_two_electron_backpropagation_result.active_auxiliary_orbital_gradient;
  const Eigen::Map<const Eigen::MatrixXd> total_inactive_density_gradient_matrix(
      total_inactive_density_gradient.data(),
      input.orbital_preparation_input.n_basis_functions,
      input.orbital_preparation_input.n_basis_functions);

  stage_start_time = std::chrono::steady_clock::now();
  const auto orbital_backpropagation_result =
      active_space_orbital_backpropagator_.backpropagate(
          total_active_auxiliary_gradient,
          total_inactive_density_gradient_matrix,
          input.orbital_preparation_input,
          orbital_result);
  result.orbital_backpropagation_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();
  result.finite_difference_step = finite_difference_step_;
  result.orbital_preparation_result =
      std::move(active_space_gradient_result.orbital_preparation_result);
  result.ao_effective_one_electron_result =
      std::move(active_space_gradient_result.ao_effective_one_electron_result);
  result.scf_result = std::move(active_space_gradient_result.scf_result);
  result.active_orbital_overlap_matrix =
      std::move(active_space_gradient_result.active_orbital_overlap_matrix);
  result.active_one_electron_integrals =
      std::move(active_space_gradient_result.active_space_one_electron_result.h1e_act);
  result.packed_active_two_electron_integrals =
      std::move(active_space_gradient_result.active_space_two_electron_result
                    .packed_active_two_electron_integrals);
  result.sparse_orbital_energy_gradient =
      orbital_backpropagation_result.orbital_value_gradient;
  result.differentiable_parameter_indices = collect_differentiable_parameter_indices(
      input.orbital_preparation_input);
  result.second_order_context = std::move(active_space_gradient_result.second_order_context);
  result.total_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start_time).count();
  return result;
}

CppOrbitalGradientResult
CppOrbitalGradientEvaluator::evaluate_without_reference_energy_gradient_with_fixed_active_space_adjoint(
    const CppVbInput& input,
    const CppActiveSpaceSecondOrderContext& accepted_point_context,
    double nuclear_repulsion_energy) const {
  const auto total_start_time = std::chrono::steady_clock::now();
  if (input.orbital_preparation_input.orbital_value_table.empty()) {
    throw std::invalid_argument("orbital_value_table must not be empty");
  }

  const auto active_space_gradient_result =
      active_space_gradient_evaluator_.evaluate_with_fixed_active_space_adjoint(
          input,
          accepted_point_context,
          nuclear_repulsion_energy);
  return evaluate_from_active_space_gradient_result(
      input,
      std::move(active_space_gradient_result),
      total_start_time);
}

std::vector<double>
CppOrbitalGradientEvaluator::evaluate_sparse_orbital_gradient_with_fixed_active_space_adjoint(
    const CppVbInput& input,
    const CppActiveSpaceSecondOrderContext& accepted_point_context,
    double nuclear_repulsion_energy) const {
  if (input.orbital_preparation_input.orbital_value_table.empty()) {
    throw std::invalid_argument("orbital_value_table must not be empty");
  }

  const int n_inactive_doubly_occupied_orbitals =
      (input.orbital_preparation_input.n_total_electrons -
       input.orbital_preparation_input.n_active_electrons) / 2;
  // Matrix-free exact-context probes need only the sparse orbital gradient
  // vector. Avoid materializing the full orbital-gradient result and its
  // cached copies on every `H v` application.
  const auto timed_active_space_context =
      active_space_gradient_evaluator_.prepare_timed_active_space_context_for_probe(
          input);
  const auto& prepared_active_space =
      timed_active_space_context.prepared_active_space;
  const auto& orbital_result = prepared_active_space.orbital_result;
  const auto& ao_effective_one_electron_result =
      prepared_active_space.ao_effective_one_electron_result;
  const auto& active_space_two_electron_result =
      prepared_active_space.active_space_two_electron_result;
  if (accepted_point_context.active_orbital_overlap_gradient.size() !=
          orbital_result.active_orbital_overlap_matrix.size() ||
      accepted_point_context.active_one_electron_gradient.size() !=
          prepared_active_space.active_space_one_electron_result.h1e_act.size() ||
      accepted_point_context.packed_active_two_electron_gradient.size() !=
          packed_active_two_electron_integral_count(
              input.orbital_preparation_input.n_active_orbitals)) {
    throw std::invalid_argument(
        "accepted-point active-space adjoint dimensions do not match the trial orbital space");
  }

  std::vector<double> total_inactive_density_gradient(
      orbital_result.inactive_density_matrix.size(),
      0.0);
  if (total_inactive_density_gradient.size() !=
          ao_effective_one_electron_result.ao_effective_h1e.size() ||
      total_inactive_density_gradient.size() !=
          input.ao_integral_input.ao_core_hamiltonian_matrix.size()) {
    throw std::runtime_error("one-electron reference energy gradient size mismatch");
  }
  const double* ao_core_hamiltonian_data =
      input.ao_integral_input.ao_core_hamiltonian_matrix.data();
  const double* ao_effective_h1e_data =
      ao_effective_one_electron_result.ao_effective_h1e.data();
  for (std::size_t index = 0; index < total_inactive_density_gradient.size(); ++index) {
    total_inactive_density_gradient[index] =
        ao_effective_h1e_data[index] +
        ao_core_hamiltonian_data[index];
  }

  const auto active_space_matrix_backpropagation_result =
      active_space_matrix_backpropagator_.backpropagate(
          accepted_point_context.active_orbital_overlap_gradient,
          accepted_point_context.active_one_electron_gradient,
          input.orbital_preparation_input.ao_overlap_matrix,
          ao_effective_one_electron_result.ao_effective_h1e,
          orbital_result.auxiliary_orbital_matrix,
          input.orbital_preparation_input.n_basis_functions,
          n_inactive_doubly_occupied_orbitals,
          input.orbital_preparation_input.n_active_orbitals);
  const auto active_space_two_electron_backpropagation_result =
      active_space_two_electron_result.representation ==
              ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity
          ? active_space_two_electron_backpropagator_.backpropagate(
                build_ri_active_pair_factor_gradient(
                    accepted_point_context.packed_active_two_electron_gradient,
                    active_space_two_electron_result,
                    input.orbital_preparation_input.n_active_orbitals),
                input,
                orbital_result,
                active_space_two_electron_result,
                input.orbital_preparation_input.n_basis_functions,
                n_inactive_doubly_occupied_orbitals,
                input.orbital_preparation_input.n_active_orbitals)
          : active_space_two_electron_backpropagator_.backpropagate(
                accepted_point_context.packed_active_two_electron_gradient,
                input.ao_integral_input.ao_two_electron_integral_values,
                input.ao_integral_input.ao_two_electron_integral_indices,
                orbital_result,
                active_space_two_electron_result,
                input.orbital_preparation_input.n_basis_functions,
                n_inactive_doubly_occupied_orbitals,
                input.orbital_preparation_input.n_active_orbitals);
  const auto ao_effective_one_electron_backpropagation_result =
      use_standard_ri_ao_effective_one_electron_path(input)
          ? ao_effective_one_electron_backpropagator_.backpropagate(
                accepted_point_context.active_one_electron_gradient,
                orbital_result,
                ensure_cpp_vb_input_ri_cache(input),
                input.ao_integral_input.n_basis_functions,
                n_inactive_doubly_occupied_orbitals,
                input.orbital_preparation_input.n_active_orbitals)
          : [&]() {
              std::vector<double> total_ao_effective_one_electron_gradient =
                  active_space_matrix_backpropagation_result.ao_effective_one_electron_gradient;
              if (total_ao_effective_one_electron_gradient.size() !=
                  orbital_result.inactive_density_matrix.size()) {
                throw std::runtime_error("ao effective one-electron gradient size mismatch");
              }
              for (std::size_t index = 0;
                   index < total_ao_effective_one_electron_gradient.size();
                   ++index) {
                total_ao_effective_one_electron_gradient[index] +=
                    orbital_result.inactive_density_matrix.data()[index];
              }
              return ao_effective_one_electron_backpropagator_.backpropagate(
                  total_ao_effective_one_electron_gradient,
                  input.ao_integral_input);
            }();
  for (std::size_t index = 0; index < total_inactive_density_gradient.size(); ++index) {
    total_inactive_density_gradient[index] +=
        ao_effective_one_electron_backpropagation_result.inactive_density_gradient[index];
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
    throw std::runtime_error("auxiliary orbital backpropagation result size mismatch");
  }
  Eigen::MatrixXd total_active_auxiliary_gradient =
      active_space_matrix_backpropagation_result.active_auxiliary_orbital_gradient;
  total_active_auxiliary_gradient.noalias() +=
      active_space_two_electron_backpropagation_result.active_auxiliary_orbital_gradient;
  const Eigen::Map<const Eigen::MatrixXd> total_inactive_density_gradient_matrix(
      total_inactive_density_gradient.data(),
      input.orbital_preparation_input.n_basis_functions,
      input.orbital_preparation_input.n_basis_functions);

  const auto orbital_backpropagation_result =
      active_space_orbital_backpropagator_.backpropagate(
          total_active_auxiliary_gradient,
          total_inactive_density_gradient_matrix,
          input.orbital_preparation_input,
          orbital_result);
  (void)nuclear_repulsion_energy;
  return orbital_backpropagation_result.orbital_value_gradient;
}

void CppOrbitalGradientEvaluator::populate_reference_energy_gradient(
    const CppVbInput& input,
    CppOrbitalGradientResult* result) const {
  if (result == nullptr) {
    throw std::invalid_argument("result must not be null");
  }
  if (!result->sparse_orbital_reference_energy_gradient.empty()) {
    return;
  }
  result->sparse_orbital_reference_energy_gradient =
      build_reference_energy_orbital_gradient(
          input,
          result->orbital_preparation_result,
          result->ao_effective_one_electron_result,
          ao_effective_one_electron_backpropagator_,
          active_space_orbital_backpropagator_);
}

}  // namespace xmvb::vb
