#include "vbscf/derivatives/gradient/orbital/evaluator.hpp"

#include <chrono>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "vbscf/integrals/ao/ri/cache.hpp"
#include "linalg/storage.hpp"
#include "core/pair_index.hpp"
#include "integrals/active/two_electron.hpp"
#include "integrals/active/frame.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"

namespace xmvb::vb {

namespace {

bool use_standard_ri_ao_effective_one_electron_path(const VbScfInput& input) {
    return input.standard_two_electron_mode == StandardTwoElectronMode::ResolutionOfIdentity;
}

std::vector<int>
collect_differentiable_parameter_indices(const OrbitalPreparationInput& orbital_preparation_input) {
    std::vector<int> differentiable_parameter_indices;
    for (int orbital_index = 0; orbital_index < orbital_preparation_input.n_orbitals;
         ++orbital_index) {
        const int coefficient_count =
            differentiable_sparse_orbital_parameter_count(orbital_preparation_input, orbital_index);
        for (int coefficient_index = 0; coefficient_index < coefficient_count;
             ++coefficient_index) {
            differentiable_parameter_indices.push_back(
                orbital_index * orbital_preparation_input.n_basis_functions + coefficient_index);
        }
    }
    return differentiable_parameter_indices;
}

std::vector<double> build_reference_energy_inactive_density_gradient(
    const VbScfInput& input,
    const OrbitalPreparationResult& orbital_result,
    const AoEffectiveOneElectronResult& ao_effective_one_electron_result,
    const AoEffectiveOneElectronBackpropagator& ao_effective_one_electron_backpropagator) {
    std::vector<double> inactive_density_gradient(orbital_result.inactive_density_matrix.size(),
                                                  0.0);
    if (inactive_density_gradient.size() !=
            ao_effective_one_electron_result.ao_effective_h1e.size() ||
        inactive_density_gradient.size() !=
            input.ao_integral_input.ao_core_hamiltonian_matrix.size()) {
        throw std::runtime_error("one-electron reference energy gradient size mismatch");
    }
    const double* ao_core_hamiltonian_data =
        input.ao_integral_input.ao_core_hamiltonian_matrix.data();
    const double* ao_effective_h1e_data = ao_effective_one_electron_result.ao_effective_h1e.data();

    for (std::size_t index = 0; index < inactive_density_gradient.size(); ++index) {
        inactive_density_gradient[index] =
            ao_effective_h1e_data[index] + ao_core_hamiltonian_data[index];
    }

    if (use_standard_ri_ao_effective_one_electron_path(input)) {
        // G is linear: the reference contribution G(2 P_i) is already 2 G(P_i).
        const int n_bf = input.ao_integral_input.n_basis_functions;
        Eigen::MatrixXd encoded = Eigen::MatrixXd::Zero(n_bf, n_bf);
        encoded.triangularView<Eigen::Lower>() =
            2 * ao_effective_one_electron_result.ao_coulomb_exchange_matrix;
        encoded.diagonal() *= 0.5;
        for (std::size_t i = 0; i < inactive_density_gradient.size(); ++i)
            inactive_density_gradient[i] += encoded.data()[i];
        return inactive_density_gradient;
    }
    const auto ao_effective_one_electron_backpropagation_result =
        ao_effective_one_electron_backpropagator.backpropagate(
            orbital_result.inactive_density_matrix, input.ao_integral_input);
    for (std::size_t index = 0; index < inactive_density_gradient.size(); ++index) {
        inactive_density_gradient[index] +=
            ao_effective_one_electron_backpropagation_result.inactive_density_gradient[index];
    }
    return inactive_density_gradient;
}

std::vector<double> build_reference_energy_orbital_gradient(
    const VbScfInput& input,
    const OrbitalPreparationResult& orbital_result,
    const AoEffectiveOneElectronResult& ao_effective_one_electron_result,
    const AoEffectiveOneElectronBackpropagator& ao_effective_one_electron_backpropagator,
    const ActiveSpaceOrbitalBackpropagator& active_space_orbital_backpropagator) {
    const std::vector<double> zero_active_auxiliary_gradient(
        input.orbital_preparation_input.n_basis_functions *
            input.orbital_preparation_input.n_active_orbitals,
        0.0);
    const std::vector<double> reference_energy_inactive_density_gradient =
        build_reference_energy_inactive_density_gradient(input,
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

} // namespace

OrbitalGradientResult OrbitalGradientEvaluator::evaluate(const VbScfInput& input,
                                                         double nuclear_repulsion_energy) const {
    return evaluate(input, {0}, {1.0}, nuclear_repulsion_energy);
}

OrbitalGradientResult
OrbitalGradientEvaluator::evaluate(const VbScfInput& input,
                                   ActiveSpaceGradientResult active_space_gradient_result) const {
    auto result =
        evaluate_without_reference_energy_gradient(input, std::move(active_space_gradient_result));
    const auto reference_start_time = std::chrono::steady_clock::now();
    populate_reference_energy_gradient(input, &result);
    result.total_wall_time_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - reference_start_time)
            .count();
    return result;
}

OrbitalGradientResult OrbitalGradientEvaluator::evaluate_without_reference_energy_gradient(
    const VbScfInput& input, double nuclear_repulsion_energy) const {
    return evaluate_without_reference_energy_gradient(input, {0}, {1.0}, nuclear_repulsion_energy);
}

OrbitalGradientResult OrbitalGradientEvaluator::evaluate_without_reference_energy_gradient(
    const VbScfInput& input, ActiveSpaceGradientResult active_space_gradient_result) const {
    const auto total_start_time = std::chrono::steady_clock::now();
    if (input.orbital_preparation_input.orbital_value_table.empty()) {
        throw std::invalid_argument("orbital_value_table must not be empty");
    }
    return evaluate_from_active_space_gradient_result(
        input, std::move(active_space_gradient_result), total_start_time);
}

OrbitalGradientResult
OrbitalGradientEvaluator::evaluate(const VbScfInput& input,
                                   const std::vector<int>& selected_state_indices,
                                   const std::vector<double>& state_average_weights,
                                   double nuclear_repulsion_energy) const {
    auto result = evaluate_without_reference_energy_gradient(
        input, selected_state_indices, state_average_weights, nuclear_repulsion_energy);
    const auto reference_start_time = std::chrono::steady_clock::now();
    populate_reference_energy_gradient(input, &result);
    result.total_wall_time_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - reference_start_time)
            .count();
    return result;
}

OrbitalGradientResult OrbitalGradientEvaluator::evaluate_without_reference_energy_gradient(
    const VbScfInput& input,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    double nuclear_repulsion_energy) const {
    const Eigen::MatrixXd no_initial_eigenvectors;
    return evaluate_without_reference_energy_gradient(input,
                                                      selected_state_indices,
                                                      state_average_weights,
                                                      nuclear_repulsion_energy,
                                                      StructureEigensolver::Dense,
                                                      StructureSolveAccuracy{},
                                                      no_initial_eigenvectors);
}

OrbitalGradientResult OrbitalGradientEvaluator::evaluate_without_reference_energy_gradient(
    const VbScfInput& input,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    double nuclear_repulsion_energy,
    StructureEigensolver structure_eigensolver,
    StructureSolveAccuracy structure_solve_accuracy,
    const Eigen::Ref<const Eigen::MatrixXd>& initial_eigenvectors) const {
    const auto total_start_time = std::chrono::steady_clock::now();
    if (input.orbital_preparation_input.orbital_value_table.empty()) {
        throw std::invalid_argument("orbital_value_table must not be empty");
    }

    const auto active_space_gradient_result =
        active_space_gradient_evaluator_.evaluate(input,
                                                  selected_state_indices,
                                                  state_average_weights,
                                                  nuclear_repulsion_energy,
                                                  structure_eigensolver,
                                                  structure_solve_accuracy,
                                                  initial_eigenvectors);
    return evaluate_from_active_space_gradient_result(
        input, std::move(active_space_gradient_result), total_start_time);
}

ActiveSpaceForwardEvaluation OrbitalGradientEvaluator::evaluate_forward(
    const VbScfInput& input,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    double nuclear_repulsion_energy,
    StructureEigensolver structure_eigensolver,
    StructureSolveAccuracy structure_solve_accuracy,
    const Eigen::Ref<const Eigen::MatrixXd>& initial_eigenvectors) const {
    if (input.orbital_preparation_input.orbital_value_table.empty()) {
        throw std::invalid_argument("orbital_value_table must not be empty");
    }
    return active_space_gradient_evaluator_.evaluate_forward(input,
                                                             selected_state_indices,
                                                             state_average_weights,
                                                             nuclear_repulsion_energy,
                                                             structure_eigensolver,
                                                             structure_solve_accuracy,
                                                             initial_eigenvectors);
}

OrbitalGradientResult
OrbitalGradientEvaluator::complete_gradient(const VbScfInput& input,
                                            ActiveSpaceForwardEvaluation forward_evaluation) const {
    const double forward_wall_time_seconds = forward_evaluation.wall_time_seconds();
    const auto completion_start_time = std::chrono::steady_clock::now();
    auto active_space_gradient_result =
        active_space_gradient_evaluator_.complete_gradient(input, std::move(forward_evaluation));
    auto result = evaluate_from_active_space_gradient_result(
        input, std::move(active_space_gradient_result), completion_start_time);
    result.total_wall_time_seconds += forward_wall_time_seconds;
    return result;
}

OrbitalFixedStructureGradient OrbitalGradientEvaluator::evaluate_fixed_structure(
    const VbScfInput& input,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    double nuclear_repulsion_energy,
    const Eigen::Ref<const Eigen::MatrixXd>& structure_coefficients,
    const SelectedMetric* metric,
    bool retain_context) const {
    const auto total_start_time = std::chrono::steady_clock::now();
    ActiveSpaceFixedStructureGradient active =
        active_space_gradient_evaluator_.evaluate_fixed_structure(input,
                                                                  selected_state_indices,
                                                                  state_average_weights,
                                                                  nuclear_repulsion_energy,
                                                                  structure_coefficients,
                                                                  metric,
                                                                  retain_context);
    OrbitalFixedStructureGradient result;
    result.normalized_coefficients = std::move(active.normalized_coefficients);
    result.structure_residuals = std::move(active.structure_residuals);
    result.coefficient_gradient = std::move(active.coefficient_gradient);
    result.gradient = evaluate_from_active_space_gradient_result(
        input, std::move(active.gradient), total_start_time);
    return result;
}

OrbitalGradientResult OrbitalGradientEvaluator::evaluate_from_active_space_gradient_result(
    const VbScfInput& input,
    ActiveSpaceGradientResult active_space_gradient_result,
    const std::chrono::steady_clock::time_point& total_start_time) const {
    const int n_inactive_doubly_occupied_orbitals =
        (input.orbital_preparation_input.n_total_electrons -
         input.orbital_preparation_input.n_active_electrons) /
        2;
    auto stage_start_time = std::chrono::steady_clock::now();
    OrbitalGradientResult result;
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
        orbital_result.inactive_density_matrix.size(), 0.0);
    if (total_inactive_density_gradient.size() !=
            ao_effective_one_electron_result.ao_effective_h1e.size() ||
        total_inactive_density_gradient.size() !=
            input.ao_integral_input.ao_core_hamiltonian_matrix.size()) {
        throw std::runtime_error("one-electron reference energy gradient size mismatch");
    }
    const double* ao_core_hamiltonian_data =
        input.ao_integral_input.ao_core_hamiltonian_matrix.data();
    const double* ao_effective_h1e_data = ao_effective_one_electron_result.ao_effective_h1e.data();
    for (std::size_t index = 0; index < total_inactive_density_gradient.size(); ++index) {
        total_inactive_density_gradient[index] =
            ao_effective_h1e_data[index] + ao_core_hamiltonian_data[index];
    }

    stage_start_time = std::chrono::steady_clock::now();
    const auto active_space_matrix_backpropagation_result = integrals::pullback_active_matrices(
        active_space_gradient_result.active_orbital_overlap_gradient,
        active_space_gradient_result.active_one_electron_gradient,
        input.orbital_preparation_input.ao_overlap_matrix,
        orbital_result.auxiliary_orbital_matrix.middleCols(
            n_inactive_doubly_occupied_orbitals, input.orbital_preparation_input.n_active_orbitals),
        active_space_gradient_result.active_space_two_electron_result.dense_active_coefficients,
        active_space_gradient_result.active_space_one_electron_result,
        active_space_gradient_result.active_space_two_electron_result.orbital_transform,
        !use_standard_ri_ao_effective_one_electron_path(input));
    result.active_space_matrix_backpropagation_wall_time_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();
    stage_start_time = std::chrono::steady_clock::now();
    const Eigen::MatrixXd active_two_electron_gradient =
        pullback_active_integrals(input,
                                  active_space_gradient_result.active_space_two_electron_result,
                                  active_space_gradient_result.packed_active_two_electron_gradient);
    result.active_space_two_electron_backpropagation_wall_time_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();
    stage_start_time = std::chrono::steady_clock::now();
    const auto ao_effective_one_electron_backpropagation_result =
        use_standard_ri_ao_effective_one_electron_path(input)
            ? ao_effective_one_electron_backpropagator_.backpropagate(
                  integrals::active_frame_one_electron_weight(
                      active_space_gradient_result.active_one_electron_gradient,
                      active_space_gradient_result.active_space_two_electron_result
                          .orbital_transform),
                  integrals::build_ri_active_two_electron_response_cache(
                      ensure_vbscf_input_ri_cache(input),
                      active_space_gradient_result.active_space_two_electron_result,
                      input.orbital_preparation_input.n_active_orbitals),
                  &ao_effective_one_electron_result)
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
                      total_ao_effective_one_electron_gradient, input.ao_integral_input);
              }();
    result.ao_effective_one_electron_backpropagation_wall_time_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();
    for (std::size_t index = 0; index < total_inactive_density_gradient.size(); ++index) {
        total_inactive_density_gradient[index] +=
            ao_effective_one_electron_backpropagation_result.inactive_density_gradient[index];
    }

    if (active_space_matrix_backpropagation_result.active_auxiliary_orbital_gradient.rows() !=
            input.orbital_preparation_input.n_basis_functions ||
        active_space_matrix_backpropagation_result.active_auxiliary_orbital_gradient.cols() !=
            input.orbital_preparation_input.n_active_orbitals ||
        active_two_electron_gradient.rows() != input.orbital_preparation_input.n_basis_functions ||
        active_two_electron_gradient.cols() != input.orbital_preparation_input.n_active_orbitals) {
        throw std::runtime_error("auxiliary orbital backpropagation result size mismatch");
    }
    Eigen::MatrixXd total_active_auxiliary_gradient =
        active_space_matrix_backpropagation_result.active_auxiliary_orbital_gradient;
    total_active_auxiliary_gradient.noalias() += active_two_electron_gradient;
    const Eigen::Map<const Eigen::MatrixXd> total_inactive_density_gradient_matrix(
        total_inactive_density_gradient.data(),
        input.orbital_preparation_input.n_basis_functions,
        input.orbital_preparation_input.n_basis_functions);

    stage_start_time = std::chrono::steady_clock::now();
    const auto orbital_backpropagation_result =
        active_space_orbital_backpropagator_.backpropagate(total_active_auxiliary_gradient,
                                                           total_inactive_density_gradient_matrix,
                                                           input.orbital_preparation_input,
                                                           orbital_result);
    result.orbital_backpropagation_wall_time_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();
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
    result.two_electron_frame =
        std::move(active_space_gradient_result.active_space_two_electron_result.orbital_transform);
    result.sparse_orbital_energy_gradient = orbital_backpropagation_result.orbital_value_gradient;
    result.differentiable_parameter_indices =
        collect_differentiable_parameter_indices(input.orbital_preparation_input);
    if (active_space_gradient_result.second_order_context != nullptr) {
        auto& accepted = *active_space_gradient_result.second_order_context;
        accepted.total_active_auxiliary_gradient = std::move(total_active_auxiliary_gradient);
        accepted.total_inactive_density_gradient = std::move(total_inactive_density_gradient);
    }
    result.second_order_context = std::move(active_space_gradient_result.second_order_context);
    result.total_wall_time_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start_time).count();
    return result;
}

OrbitalGradientResult OrbitalGradientEvaluator::
    evaluate_without_reference_energy_gradient_with_fixed_active_space_adjoint(
        const VbScfInput& input,
        const AcceptedPointContext& accepted_point_context,
        double nuclear_repulsion_energy) const {
    const auto total_start_time = std::chrono::steady_clock::now();
    if (input.orbital_preparation_input.orbital_value_table.empty()) {
        throw std::invalid_argument("orbital_value_table must not be empty");
    }

    const auto active_space_gradient_result =
        active_space_gradient_evaluator_.evaluate_with_fixed_active_space_adjoint(
            input, accepted_point_context, nuclear_repulsion_energy);
    return evaluate_from_active_space_gradient_result(
        input, std::move(active_space_gradient_result), total_start_time);
}

std::vector<double>
OrbitalGradientEvaluator::evaluate_sparse_orbital_gradient_with_fixed_active_space_adjoint(
    const VbScfInput& input,
    const AcceptedPointContext& accepted_point_context,
    double nuclear_repulsion_energy) const {
    if (input.orbital_preparation_input.orbital_value_table.empty()) {
        throw std::invalid_argument("orbital_value_table must not be empty");
    }

    const int n_inactive_doubly_occupied_orbitals =
        (input.orbital_preparation_input.n_total_electrons -
         input.orbital_preparation_input.n_active_electrons) /
        2;
    // Matrix-free exact-context probes need only the sparse orbital gradient
    // vector. Avoid materializing the full orbital-gradient result and its
    // cached copies on every `H v` application.
    const auto timed_active_space_context = prepare_timed_active_space_context(
        input,
        &accepted_point_context.prepared_active_space.active_space_two_electron_result
             .orbital_transform);
    const auto& prepared_active_space = timed_active_space_context.prepared_active_space;
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
            xmvb::integrals::packed_active_two_electron_integral_count(
                input.orbital_preparation_input.n_active_orbitals)) {
        throw std::invalid_argument(
            "accepted-point active-space adjoint dimensions do not match the trial orbital space");
    }

    std::vector<double> total_inactive_density_gradient(
        orbital_result.inactive_density_matrix.size(), 0.0);
    if (total_inactive_density_gradient.size() !=
            ao_effective_one_electron_result.ao_effective_h1e.size() ||
        total_inactive_density_gradient.size() !=
            input.ao_integral_input.ao_core_hamiltonian_matrix.size()) {
        throw std::runtime_error("one-electron reference energy gradient size mismatch");
    }
    const double* ao_core_hamiltonian_data =
        input.ao_integral_input.ao_core_hamiltonian_matrix.data();
    const double* ao_effective_h1e_data = ao_effective_one_electron_result.ao_effective_h1e.data();
    for (std::size_t index = 0; index < total_inactive_density_gradient.size(); ++index) {
        total_inactive_density_gradient[index] =
            ao_effective_h1e_data[index] + ao_core_hamiltonian_data[index];
    }

    const auto active_space_matrix_backpropagation_result = integrals::pullback_active_matrices(
        accepted_point_context.active_orbital_overlap_gradient,
        accepted_point_context.active_one_electron_gradient,
        input.orbital_preparation_input.ao_overlap_matrix,
        orbital_result.auxiliary_orbital_matrix.middleCols(
            n_inactive_doubly_occupied_orbitals, input.orbital_preparation_input.n_active_orbitals),
        active_space_two_electron_result.dense_active_coefficients,
        prepared_active_space.active_space_one_electron_result,
        active_space_two_electron_result.orbital_transform,
        !use_standard_ri_ao_effective_one_electron_path(input));
    const Eigen::MatrixXd active_two_electron_gradient =
        pullback_active_integrals(input,
                                  active_space_two_electron_result,
                                  accepted_point_context.packed_active_two_electron_gradient);
    const auto ao_effective_one_electron_backpropagation_result =
        use_standard_ri_ao_effective_one_electron_path(input)
            ? ao_effective_one_electron_backpropagator_.backpropagate(
                  integrals::active_frame_one_electron_weight(
                      accepted_point_context.active_one_electron_gradient,
                      active_space_two_electron_result.orbital_transform),
                  integrals::build_ri_active_two_electron_response_cache(
                      ensure_vbscf_input_ri_cache(input),
                      active_space_two_electron_result,
                      input.orbital_preparation_input.n_active_orbitals),
                  &ao_effective_one_electron_result)
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
                      total_ao_effective_one_electron_gradient, input.ao_integral_input);
              }();
    for (std::size_t index = 0; index < total_inactive_density_gradient.size(); ++index) {
        total_inactive_density_gradient[index] +=
            ao_effective_one_electron_backpropagation_result.inactive_density_gradient[index];
    }

    if (active_space_matrix_backpropagation_result.active_auxiliary_orbital_gradient.rows() !=
            input.orbital_preparation_input.n_basis_functions ||
        active_space_matrix_backpropagation_result.active_auxiliary_orbital_gradient.cols() !=
            input.orbital_preparation_input.n_active_orbitals ||
        active_two_electron_gradient.rows() != input.orbital_preparation_input.n_basis_functions ||
        active_two_electron_gradient.cols() != input.orbital_preparation_input.n_active_orbitals) {
        throw std::runtime_error("auxiliary orbital backpropagation result size mismatch");
    }
    Eigen::MatrixXd total_active_auxiliary_gradient =
        active_space_matrix_backpropagation_result.active_auxiliary_orbital_gradient;
    total_active_auxiliary_gradient.noalias() += active_two_electron_gradient;
    const Eigen::Map<const Eigen::MatrixXd> total_inactive_density_gradient_matrix(
        total_inactive_density_gradient.data(),
        input.orbital_preparation_input.n_basis_functions,
        input.orbital_preparation_input.n_basis_functions);

    const auto orbital_backpropagation_result =
        active_space_orbital_backpropagator_.backpropagate(total_active_auxiliary_gradient,
                                                           total_inactive_density_gradient_matrix,
                                                           input.orbital_preparation_input,
                                                           orbital_result);
    (void)nuclear_repulsion_energy;
    return orbital_backpropagation_result.orbital_value_gradient;
}

void OrbitalGradientEvaluator::populate_reference_energy_gradient(
    const VbScfInput& input, OrbitalGradientResult* result) const {
    if (result == nullptr) {
        throw std::invalid_argument("result must not be null");
    }
    if (!result->sparse_orbital_reference_energy_gradient.empty()) {
        return;
    }
    result->sparse_orbital_reference_energy_gradient =
        build_reference_energy_orbital_gradient(input,
                                                result->orbital_preparation_result,
                                                result->ao_effective_one_electron_result,
                                                ao_effective_one_electron_backpropagator_,
                                                active_space_orbital_backpropagator_);
}

} // namespace xmvb::vb
