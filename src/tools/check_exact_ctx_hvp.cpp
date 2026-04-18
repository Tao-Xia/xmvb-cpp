#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/orbital/active_space_matrix_backpropagator.hpp"
#include "vb/orbital/active_space_orbital_backpropagator.hpp"
#include "vb/orbital/active_space_two_electron_backpropagator.hpp"
#include "vb/orbital/ao_effective_one_electron_backpropagator.hpp"
#include "vb/orbital/nonredundant_optimizer_input_adapter.hpp"
#include "vb/orbital/nonredundant_orbital_space.hpp"
#include "vb/orbital/sparse_orbital_parameter_view.hpp"
#include "vb/scf/cpp_active_space_gradient_evaluator.hpp"
#include "vb/scf/cpp_orbital_gradient_evaluator.hpp"
#include "vb/scf/exact_orbital_second_order_operator.hpp"
#include "vb/scf/opposite_spin_matrix_backward.hpp"
#include "vb/scf/same_spin_matrix_backward.hpp"
#include "vb/vbscf_algorithm.hpp"

namespace {

using Matrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

struct Options {
  std::string input_path;
  double step = 1.0e-3;
  std::string probe = "full";
  bool nonredundant_adapt = false;
  xmvb::vb::AoIntegralSource ao_integral_source =
      xmvb::vb::AoIntegralSource::Auto;
};

void print_usage() {
  std::cerr
      << "usage: check_exact_ctx_hvp <input.xmi> [--step h] [--probe full|fixed] "
      << "[--ao-integral-source auto|legacy|libcint_cpp|runtime_hcore] "
      << "[--nonredundant-adapt true|false]\n";
}

bool parse_bool_argument(const std::string& value) {
  if (value == "true" || value == "1") {
    return true;
  }
  if (value == "false" || value == "0") {
    return false;
  }
  throw std::invalid_argument("invalid boolean value: " + value);
}

xmvb::vb::AoIntegralSource parse_ao_integral_source(const std::string& value) {
  if (value == "auto") {
    return xmvb::vb::AoIntegralSource::Auto;
  }
  if (value == "legacy") {
    return xmvb::vb::AoIntegralSource::LegacyRuntime;
  }
  if (value == "libcint_cpp") {
    return xmvb::vb::AoIntegralSource::LibcintMaterializedCpp;
  }
  if (value == "runtime_hcore") {
    return xmvb::vb::AoIntegralSource::RuntimeCoreHamiltonianOnly;
  }
  throw std::invalid_argument("invalid AO integral source: " + value);
}

Options parse_arguments(int argc, char** argv) {
  if (argc < 2 || ((argc - 2) % 2 != 0)) {
    print_usage();
    throw std::invalid_argument("invalid argument count");
  }

  Options options;
  options.input_path = argv[1];
  for (int argument_index = 2; argument_index < argc; argument_index += 2) {
    const std::string name = argv[argument_index];
    const std::string value = argv[argument_index + 1];
    if (name == "--step") {
      options.step = std::stod(value);
      continue;
    }
    if (name == "--probe") {
      options.probe = value;
      continue;
    }
    if (name == "--ao-integral-source") {
      options.ao_integral_source = parse_ao_integral_source(value);
      continue;
    }
    if (name == "--nonredundant-adapt") {
      options.nonredundant_adapt = parse_bool_argument(value);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + name);
  }
  if (!(options.step > 0.0)) {
    throw std::invalid_argument("--step must be positive");
  }
  if (options.probe != "full" && options.probe != "fixed") {
    throw std::invalid_argument("--probe must be either 'full' or 'fixed'");
  }
  return options;
}

double max_abs_difference(
    const Eigen::VectorXd& left,
    const Eigen::VectorXd& right) {
  if (left.size() != right.size()) {
    throw std::invalid_argument("vector size mismatch");
  }
  double max_abs_diff = 0.0;
  for (Eigen::Index index = 0; index < left.size(); ++index) {
    max_abs_diff = std::max(max_abs_diff, std::abs(left[index] - right[index]));
  }
  return max_abs_diff;
}

std::vector<double> build_full_auxiliary_gradient_from_active_block(
    const xmvb::vb::OrbitalPreparationInput& input,
    const Eigen::Ref<const Eigen::MatrixXd>& active_auxiliary_gradient) {
  const int n_basis_functions = input.n_basis_functions;
  const int n_inactive_doubly_occupied_orbitals =
      (input.n_total_electrons - input.n_active_electrons) / 2;
  const int n_active_orbitals = input.n_active_orbitals;
  if (active_auxiliary_gradient.rows() != n_basis_functions ||
      active_auxiliary_gradient.cols() != n_active_orbitals) {
    throw std::invalid_argument(
        "active auxiliary gradient shape does not match the orbital preparation input");
  }

  std::vector<double> full_auxiliary_gradient(
      xmvb::to_size(n_basis_functions) * n_basis_functions,
      0.0);
  Eigen::Map<Matrix> full_auxiliary_gradient_matrix(
      full_auxiliary_gradient.data(),
      n_basis_functions,
      n_basis_functions);
  full_auxiliary_gradient_matrix.middleCols(
      n_inactive_doubly_occupied_orbitals,
      n_active_orbitals) = active_auxiliary_gradient;
  return full_auxiliary_gradient;
}

Eigen::MatrixXd extract_active_auxiliary_gradient_block(
    const xmvb::vb::OrbitalPreparationInput& input,
    const std::vector<double>& full_auxiliary_gradient) {
  const int n_basis_functions = input.n_basis_functions;
  const int n_inactive_doubly_occupied_orbitals =
      (input.n_total_electrons - input.n_active_electrons) / 2;
  const int n_active_orbitals = input.n_active_orbitals;
  if (full_auxiliary_gradient.size() !=
      xmvb::to_size(n_basis_functions) * n_basis_functions) {
    throw std::invalid_argument(
        "full auxiliary gradient size does not match the orbital preparation input");
  }

  const Eigen::Map<const Matrix> full_auxiliary_gradient_matrix(
      full_auxiliary_gradient.data(),
      n_basis_functions,
      n_basis_functions);
  return full_auxiliary_gradient_matrix.middleCols(
      n_inactive_doubly_occupied_orbitals,
      n_active_orbitals);
}

Eigen::VectorXd apply_active_space_gradient_direction_to_orbital_response(
    const xmvb::vb::CppVbInput& input,
    const xmvb::vb::CppActiveSpaceSecondOrderContext& accepted_point_context,
    const xmvb::vb::SparseOrbitalParameterView& parameter_view,
    const xmvb::vb::NonredundantOrbitalSpace& nonredundant_space,
    const std::vector<double>& active_orbital_overlap_gradient,
    const std::vector<double>& active_one_electron_gradient,
    const std::vector<double>& packed_active_two_electron_gradient,
    bool symmetrize_matrix_gradients) {
  const int n_inactive_doubly_occupied_orbitals =
      (input.orbital_preparation_input.n_total_electrons -
       input.orbital_preparation_input.n_active_electrons) / 2;
  const int n_active_orbitals =
      input.orbital_preparation_input.n_active_orbitals;
  const auto& prepared_active_space =
      accepted_point_context.prepared_active_space;
  const auto& orbital_result = prepared_active_space.orbital_result;
  const auto& ao_effective_one_electron_result =
      prepared_active_space.ao_effective_one_electron_result;
  const auto& active_space_two_electron_result =
      prepared_active_space.active_space_two_electron_result;

  std::vector<double> matrix_overlap_gradient = active_orbital_overlap_gradient;
  std::vector<double> matrix_one_electron_gradient = active_one_electron_gradient;
  if (symmetrize_matrix_gradients) {
    const Eigen::Map<const Matrix> overlap_gradient_matrix(
        active_orbital_overlap_gradient.data(),
        n_active_orbitals,
        n_active_orbitals);
    const Eigen::Map<const Matrix> one_electron_gradient_matrix(
        active_one_electron_gradient.data(),
        n_active_orbitals,
        n_active_orbitals);
    const Matrix overlap_gradient_symmetric =
        0.5 * (overlap_gradient_matrix + overlap_gradient_matrix.transpose());
    const Matrix one_electron_gradient_symmetric =
        0.5 * (one_electron_gradient_matrix + one_electron_gradient_matrix.transpose());
    matrix_overlap_gradient.assign(
        overlap_gradient_symmetric.data(),
        overlap_gradient_symmetric.data() + overlap_gradient_symmetric.size());
    matrix_one_electron_gradient.assign(
        one_electron_gradient_symmetric.data(),
        one_electron_gradient_symmetric.data() + one_electron_gradient_symmetric.size());
  }

  xmvb::vb::ActiveSpaceMatrixBackpropagator matrix_backpropagator;
  const auto active_space_matrix_backpropagation_result =
      matrix_backpropagator.backpropagate(
          matrix_overlap_gradient,
          matrix_one_electron_gradient,
          input.orbital_preparation_input.active_orbital_overlap_matrix,
          ao_effective_one_electron_result.ao_effective_h1e,
          orbital_result.auxiliary_orbital_matrix,
          input.orbital_preparation_input.n_basis_functions,
          n_inactive_doubly_occupied_orbitals,
          n_active_orbitals);

  xmvb::vb::ActiveSpaceTwoElectronBackpropagator active_space_two_electron_backpropagator;
  const auto active_space_two_electron_backpropagation_result =
      active_space_two_electron_backpropagator.backpropagate(
          packed_active_two_electron_gradient,
          input.ao_integral_input.ao_two_electron_integral_values,
          input.ao_integral_input.ao_two_electron_integral_indices,
          orbital_result,
          active_space_two_electron_result,
          input.orbital_preparation_input.n_basis_functions,
          n_inactive_doubly_occupied_orbitals,
          n_active_orbitals);

  xmvb::vb::AoEffectiveOneElectronBackpropagator ao_effective_one_electron_backpropagator;
  const auto ao_effective_one_electron_backpropagation_result =
      ao_effective_one_electron_backpropagator.backpropagate(
          active_space_matrix_backpropagation_result.ao_effective_one_electron_gradient,
          input.ao_integral_input);

  std::vector<double> total_inactive_density_gradient =
      ao_effective_one_electron_backpropagation_result.inactive_density_gradient;
  Eigen::MatrixXd total_active_auxiliary_gradient =
      active_space_matrix_backpropagation_result
          .active_auxiliary_orbital_gradient;
  if (total_active_auxiliary_gradient.rows() !=
          active_space_two_electron_backpropagation_result
              .active_auxiliary_orbital_gradient.rows() ||
      total_active_auxiliary_gradient.cols() !=
          active_space_two_electron_backpropagation_result
              .active_auxiliary_orbital_gradient.cols()) {
    throw std::runtime_error(
        "fd outer-response active auxiliary gradient shape mismatch");
  }
  total_active_auxiliary_gradient.noalias() +=
      active_space_two_electron_backpropagation_result
          .active_auxiliary_orbital_gradient;

  xmvb::vb::ActiveSpaceOrbitalBackpropagator orbital_backpropagator;
  const Eigen::Map<const Matrix> total_inactive_density_gradient_matrix(
      total_inactive_density_gradient.data(),
      input.orbital_preparation_input.n_basis_functions,
      input.orbital_preparation_input.n_basis_functions);
  const auto orbital_backpropagation_result =
      orbital_backpropagator.backpropagate(
          total_active_auxiliary_gradient,
          total_inactive_density_gradient_matrix,
          input.orbital_preparation_input,
          orbital_result);
  const Eigen::VectorXd packed_response =
      parameter_view.gather_from_full(
          orbital_backpropagation_result.orbital_value_gradient);
  return nonredundant_space.project_reduced_gradient(packed_response);
}

double max_abs_value(const Eigen::VectorXd& values) {
  double max_abs = 0.0;
  for (Eigen::Index index = 0; index < values.size(); ++index) {
    max_abs = std::max(max_abs, std::abs(values[index]));
  }
  return max_abs;
}

struct OrbitalBackpropInputs {
  Eigen::MatrixXd total_active_auxiliary_gradient;
  std::vector<double> total_auxiliary_gradient;
  std::vector<double> total_inactive_density_gradient;
};

OrbitalBackpropInputs build_orbital_backprop_inputs(
    const xmvb::vb::CppVbInput& input,
    const xmvb::vb::CppActiveSpaceGradientResult& active_space_gradient_result) {
  if (input.standard_two_electron_mode ==
      xmvb::vb::StandardTwoElectronMode::ResolutionOfIdentity) {
    throw std::invalid_argument(
        "RI mode is not supported by check_exact_ctx_hvp backprop-input diagnostics");
  }

  const int n_inactive_doubly_occupied_orbitals =
      (input.orbital_preparation_input.n_total_electrons -
       input.orbital_preparation_input.n_active_electrons) / 2;
  const auto& orbital_result = active_space_gradient_result.orbital_preparation_result;
  const auto& ao_effective_one_electron_result =
      active_space_gradient_result.ao_effective_one_electron_result;
  const auto& active_space_two_electron_result =
      active_space_gradient_result.active_space_two_electron_result;

  xmvb::vb::ActiveSpaceMatrixBackpropagator matrix_backpropagator;
  const auto active_space_matrix_backpropagation_result =
      matrix_backpropagator.backpropagate(
          active_space_gradient_result.active_orbital_overlap_gradient,
          active_space_gradient_result.active_one_electron_gradient,
          input.orbital_preparation_input.active_orbital_overlap_matrix,
          ao_effective_one_electron_result.ao_effective_h1e,
          orbital_result.auxiliary_orbital_matrix,
          input.orbital_preparation_input.n_basis_functions,
          n_inactive_doubly_occupied_orbitals,
          input.orbital_preparation_input.n_active_orbitals);

  xmvb::vb::ActiveSpaceTwoElectronBackpropagator active_space_two_electron_backpropagator;
  const auto active_space_two_electron_backpropagation_result =
      active_space_two_electron_backpropagator.backpropagate(
          active_space_gradient_result.packed_active_two_electron_gradient,
          input.ao_integral_input.ao_two_electron_integral_values,
          input.ao_integral_input.ao_two_electron_integral_indices,
          orbital_result,
          active_space_two_electron_result,
          input.orbital_preparation_input.n_basis_functions,
          n_inactive_doubly_occupied_orbitals,
          input.orbital_preparation_input.n_active_orbitals);

  std::vector<double> total_inactive_density_gradient =
      ao_effective_one_electron_result.ao_effective_h1e;
  if (total_inactive_density_gradient.size() !=
      input.ao_integral_input.ao_core_hamiltonian_matrix.size()) {
    throw std::runtime_error(
        "core-Hamiltonian size mismatch while building diagnostic backprop inputs");
  }
  for (std::size_t index = 0;
       index < total_inactive_density_gradient.size();
       ++index) {
    total_inactive_density_gradient[index] +=
        input.ao_integral_input.ao_core_hamiltonian_matrix[index];
  }

  std::vector<double> total_ao_effective_one_electron_gradient =
      active_space_matrix_backpropagation_result.ao_effective_one_electron_gradient;
  if (total_ao_effective_one_electron_gradient.size() !=
      orbital_result.inactive_density_matrix.size()) {
    throw std::runtime_error(
        "inactive-density size mismatch while building diagnostic backprop inputs");
  }
  for (std::size_t index = 0;
       index < total_ao_effective_one_electron_gradient.size();
       ++index) {
    total_ao_effective_one_electron_gradient[index] +=
        orbital_result.inactive_density_matrix.data()[index];
  }

  xmvb::vb::AoEffectiveOneElectronBackpropagator ao_effective_one_electron_backpropagator;
  const auto ao_effective_one_electron_backpropagation_result =
      ao_effective_one_electron_backpropagator.backpropagate(
          total_ao_effective_one_electron_gradient,
          input.ao_integral_input);
  if (ao_effective_one_electron_backpropagation_result.inactive_density_gradient.size() !=
      total_inactive_density_gradient.size()) {
    throw std::runtime_error(
        "AO-H1E inactive-density size mismatch while building diagnostic backprop inputs");
  }
  for (std::size_t index = 0;
       index < total_inactive_density_gradient.size();
       ++index) {
    total_inactive_density_gradient[index] +=
        ao_effective_one_electron_backpropagation_result
            .inactive_density_gradient[index];
  }

  Eigen::MatrixXd total_active_auxiliary_gradient =
      active_space_matrix_backpropagation_result
          .active_auxiliary_orbital_gradient;
  if (total_active_auxiliary_gradient.rows() !=
          active_space_two_electron_backpropagation_result
              .active_auxiliary_orbital_gradient.rows() ||
      total_active_auxiliary_gradient.cols() !=
          active_space_two_electron_backpropagation_result
              .active_auxiliary_orbital_gradient.cols()) {
    throw std::runtime_error(
        "active auxiliary-gradient shape mismatch while building diagnostic backprop inputs");
  }
  total_active_auxiliary_gradient.noalias() +=
      active_space_two_electron_backpropagation_result
          .active_auxiliary_orbital_gradient;

  OrbitalBackpropInputs result;
  result.total_active_auxiliary_gradient = total_active_auxiliary_gradient;
  result.total_auxiliary_gradient =
      build_full_auxiliary_gradient_from_active_block(
          input.orbital_preparation_input,
          total_active_auxiliary_gradient);
  result.total_inactive_density_gradient =
      std::move(total_inactive_density_gradient);
  return result;
}

std::vector<double> finite_difference_storage(
    const std::vector<double>& plus_values,
    const std::vector<double>& minus_values,
    double epsilon) {
  if (plus_values.size() != minus_values.size()) {
    throw std::invalid_argument(
        "finite-difference storage buffers must have matching sizes");
  }
  std::vector<double> difference(plus_values.size(), 0.0);
  for (std::size_t index = 0; index < difference.size(); ++index) {
    difference[index] =
        (plus_values[index] - minus_values[index]) / (2.0 * epsilon);
  }
  return difference;
}

Eigen::VectorXd project_full_orbital_gradient_to_reduced(
    const xmvb::vb::SparseOrbitalParameterView& parameter_view,
    const xmvb::vb::NonredundantOrbitalSpace& nonredundant_space,
    const std::vector<double>& sparse_orbital_energy_gradient) {
  const Eigen::VectorXd packed_gradient =
      parameter_view.gather_from_full(sparse_orbital_energy_gradient);
  return nonredundant_space.project_reduced_gradient(packed_gradient);
}

int get_sparse_coefficient_count(
    const xmvb::vb::OrbitalPreparationInput& input,
    int orbital_index) {
  const int explicit_count =
      input.orbital_basis_counts[xmvb::to_size(orbital_index)];
  if (explicit_count > 1) {
    return explicit_count;
  }

  int coefficient_count = 0;
  while (coefficient_count < input.n_basis_functions) {
    const int basis_function_index =
        input.orbital_basis_index_table[xmvb::to_size(orbital_index) *
                                            input.n_basis_functions +
                                        coefficient_count];
    if (basis_function_index == 0) {
      break;
    }
    ++coefficient_count;
  }
  return coefficient_count;
}

std::vector<double> normalize_sparse_orbitals_for_debug(
    const xmvb::vb::OrbitalPreparationInput& input,
    const Eigen::Map<const Eigen::MatrixXd>& basis_overlap,
    std::vector<double>* squared_norms_out) {
  std::vector<double> normalized_values = input.orbital_value_table;
  squared_norms_out->assign(xmvb::to_size(input.n_orbitals), 0.0);

  for (int orbital_index = 0; orbital_index < input.n_orbitals; ++orbital_index) {
    const int coefficient_count =
        get_sparse_coefficient_count(input, orbital_index);
    double squared_norm = 0.0;
    for (int left_index = 0; left_index < coefficient_count; ++left_index) {
      const int left_basis_function =
          input.orbital_basis_index_table[xmvb::to_size(orbital_index) *
                                              input.n_basis_functions +
                                          left_index] -
          1;
      const double left_value =
          normalized_values[xmvb::to_size(orbital_index) * input.n_basis_functions +
                            left_index];
      for (int right_index = 0; right_index < coefficient_count; ++right_index) {
        const int right_basis_function =
            input.orbital_basis_index_table[xmvb::to_size(orbital_index) *
                                                input.n_basis_functions +
                                            right_index] -
            1;
        const double right_value =
            normalized_values[xmvb::to_size(orbital_index) * input.n_basis_functions +
                              right_index];
        squared_norm += left_value * right_value *
            basis_overlap(left_basis_function, right_basis_function);
      }
    }

    if (!(squared_norm > 0.0) || !std::isfinite(squared_norm)) {
      throw std::runtime_error(
          "orbital normalization failed while building fixed-upstream debug context");
    }

    (*squared_norms_out)[xmvb::to_size(orbital_index)] = squared_norm;
    const double normalization_factor = std::sqrt(1.0 / squared_norm);
    for (int coefficient_index = 0;
         coefficient_index < coefficient_count;
         ++coefficient_index) {
      normalized_values[xmvb::to_size(orbital_index) * input.n_basis_functions +
                        coefficient_index] *= normalization_factor;
    }
  }

  return normalized_values;
}

Eigen::MatrixXd expand_sparse_orbitals_for_debug(
    const xmvb::vb::OrbitalPreparationInput& input,
    const std::vector<double>& normalized_orbital_values) {
  Eigen::MatrixXd orbital_matrix =
      Eigen::MatrixXd::Zero(input.n_basis_functions, input.n_orbitals);
  for (int orbital_index = 0; orbital_index < input.n_orbitals; ++orbital_index) {
    const int coefficient_count =
        get_sparse_coefficient_count(input, orbital_index);
    for (int coefficient_index = 0;
         coefficient_index < coefficient_count;
         ++coefficient_index) {
      const int basis_function_index =
          input.orbital_basis_index_table[xmvb::to_size(orbital_index) *
                                              input.n_basis_functions +
                                          coefficient_index] -
          1;
      orbital_matrix(basis_function_index, orbital_index) =
          normalized_orbital_values[xmvb::to_size(orbital_index) *
                                        input.n_basis_functions +
                                    coefficient_index];
    }
  }
  return orbital_matrix;
}

struct FixedUpstreamForwardDebugContext {
  Eigen::MatrixXd normalized_orbitals;
  Eigen::MatrixXd original_orbital_gradient;
};

FixedUpstreamForwardDebugContext build_fixed_upstream_forward_debug_context(
    const xmvb::vb::OrbitalPreparationInput& input,
    const std::vector<double>& total_auxiliary_gradient,
    const std::vector<double>& total_inactive_density_gradient) {
  const int n_inactive_doubly_occupied_orbitals =
      (input.n_total_electrons - input.n_active_electrons) / 2;
  const Eigen::Map<const Eigen::MatrixXd> basis_overlap(
      input.active_orbital_overlap_matrix.data(),
      input.n_basis_functions,
      input.n_basis_functions);
  const Eigen::Map<const Eigen::MatrixXd> auxiliary_gradient_matrix(
      total_auxiliary_gradient.data(),
      input.n_basis_functions,
      input.n_basis_functions);
  const Eigen::Map<const Eigen::MatrixXd> inactive_density_gradient_matrix(
      total_inactive_density_gradient.data(),
      input.n_basis_functions,
      input.n_basis_functions);

  std::vector<double> squared_norms;
  const std::vector<double> normalized_orbital_values =
      normalize_sparse_orbitals_for_debug(
          input,
          basis_overlap,
          &squared_norms);
  const Eigen::MatrixXd normalized_orbitals =
      expand_sparse_orbitals_for_debug(
          input,
          normalized_orbital_values);
  const auto inactive_orbitals =
      normalized_orbitals.leftCols(n_inactive_doubly_occupied_orbitals);
  const auto active_orbitals =
      normalized_orbitals.middleCols(
          n_inactive_doubly_occupied_orbitals,
          input.n_active_orbitals);

  Eigen::MatrixXd inactive_density_matrix =
      Eigen::MatrixXd::Zero(input.n_basis_functions, input.n_basis_functions);
  if (n_inactive_doubly_occupied_orbitals > 0) {
    const Eigen::MatrixXd inactive_overlap =
        inactive_orbitals.transpose() * basis_overlap * inactive_orbitals;
    const Eigen::MatrixXd inactive_overlap_inverse = inactive_overlap.inverse();
    inactive_density_matrix =
        inactive_orbitals *
        inactive_overlap_inverse *
        inactive_orbitals.transpose();
  }

  const Eigen::MatrixXd active_auxiliary_gradient =
      auxiliary_gradient_matrix.middleCols(
          n_inactive_doubly_occupied_orbitals,
          input.n_active_orbitals);
  Eigen::MatrixXd original_orbital_gradient =
      Eigen::MatrixXd::Zero(input.n_basis_functions, input.n_orbitals);
  if (n_inactive_doubly_occupied_orbitals == 0) {
    original_orbital_gradient.middleCols(
        0,
        input.n_active_orbitals) = active_auxiliary_gradient;
  } else {
    const Eigen::MatrixXd bs_active = basis_overlap * active_orbitals;
    const Eigen::MatrixXd inactive_overlap =
        inactive_orbitals.transpose() * basis_overlap * inactive_orbitals;
    const Eigen::MatrixXd inactive_overlap_inverse = inactive_overlap.inverse();
    const Eigen::MatrixXd original_active_gradient =
        active_auxiliary_gradient -
        basis_overlap * inactive_density_matrix * active_auxiliary_gradient;
    const Eigen::MatrixXd total_inactive_gradient =
        inactive_density_gradient_matrix -
        active_auxiliary_gradient * bs_active.transpose();
    const Eigen::MatrixXd inactive_density_gradient_symmetric =
        total_inactive_gradient + total_inactive_gradient.transpose();
    const Eigen::MatrixXd inactive_overlap_inverse_gradient =
        inactive_orbitals.transpose() *
        total_inactive_gradient *
        inactive_orbitals;
    const Eigen::MatrixXd inactive_overlap_gradient =
        -inactive_overlap_inverse *
        inactive_overlap_inverse_gradient *
        inactive_overlap_inverse;
    const Eigen::MatrixXd original_inactive_gradient =
        inactive_density_gradient_symmetric *
            inactive_orbitals *
            inactive_overlap_inverse +
        basis_overlap *
            inactive_orbitals *
            (inactive_overlap_gradient + inactive_overlap_gradient.transpose());
    original_orbital_gradient.leftCols(n_inactive_doubly_occupied_orbitals) =
        original_inactive_gradient;
    original_orbital_gradient.middleCols(
        n_inactive_doubly_occupied_orbitals,
        input.n_active_orbitals) = original_active_gradient;
  }

  FixedUpstreamForwardDebugContext result;
  result.normalized_orbitals = normalized_orbitals;
  result.original_orbital_gradient = original_orbital_gradient;
  return result;
}

struct FixedUpstreamTangentDebugContext {
  Eigen::MatrixXd normalized_orbitals;
  Eigen::MatrixXd delta_normalized_orbitals;
};

FixedUpstreamTangentDebugContext build_fixed_upstream_tangent_debug_context(
    const xmvb::vb::OrbitalPreparationInput& input,
    const xmvb::vb::SparseOrbitalParameterView& parameter_view,
    const Eigen::VectorXd& packed_direction) {
  if (packed_direction.size() != parameter_view.size()) {
    throw std::invalid_argument(
        "packed direction size mismatch in fixed-upstream tangent debug context");
  }

  const Eigen::Map<const Eigen::MatrixXd> basis_overlap(
      input.active_orbital_overlap_matrix.data(),
      input.n_basis_functions,
      input.n_basis_functions);
  std::vector<double> full_direction(
      xmvb::to_size(input.n_orbitals) * input.n_basis_functions,
      0.0);
  const auto& differentiable_indices =
      parameter_view.differentiable_parameter_indices();
  for (Eigen::Index packed_index = 0;
       packed_index < packed_direction.size();
       ++packed_index) {
    full_direction[xmvb::to_size(
        differentiable_indices[xmvb::to_size(packed_index)])] =
        packed_direction[packed_index];
  }

  FixedUpstreamTangentDebugContext result;
  result.normalized_orbitals =
      Eigen::MatrixXd::Zero(input.n_basis_functions, input.n_orbitals);
  result.delta_normalized_orbitals =
      Eigen::MatrixXd::Zero(input.n_basis_functions, input.n_orbitals);

  for (int orbital_index = 0; orbital_index < input.n_orbitals; ++orbital_index) {
    Eigen::VectorXd dense_raw =
        Eigen::VectorXd::Zero(input.n_basis_functions);
    Eigen::VectorXd dense_direction =
        Eigen::VectorXd::Zero(input.n_basis_functions);
    const int coefficient_count =
        get_sparse_coefficient_count(input, orbital_index);
    for (int coefficient_index = 0;
         coefficient_index < coefficient_count;
         ++coefficient_index) {
      const int basis_function_index =
          input.orbital_basis_index_table[xmvb::to_size(orbital_index) *
                                              input.n_basis_functions +
                                          coefficient_index] -
          1;
      const int flat_index =
          orbital_index * input.n_basis_functions + coefficient_index;
      dense_raw[basis_function_index] =
          input.orbital_value_table[xmvb::to_size(flat_index)];
      dense_direction[basis_function_index] =
          full_direction[xmvb::to_size(flat_index)];
    }

    const double squared_norm = dense_raw.dot(basis_overlap * dense_raw);
    if (!(squared_norm > 0.0) || !std::isfinite(squared_norm)) {
      throw std::runtime_error(
          "orbital normalization failed in fixed-upstream tangent debug context");
    }
    const double norm = std::sqrt(squared_norm);
    const Eigen::VectorXd normalized = dense_raw / norm;
    const double direction_projection =
        dense_raw.dot(basis_overlap * dense_direction) / squared_norm;
    const Eigen::VectorXd delta_normalized =
        dense_direction / norm - normalized * direction_projection;

    result.normalized_orbitals.col(orbital_index) = normalized;
    result.delta_normalized_orbitals.col(orbital_index) = delta_normalized;
  }

  return result;
}

Eigen::MatrixXd build_fixed_upstream_delta_original_orbital_gradient(
    const xmvb::vb::OrbitalPreparationInput& input,
    const xmvb::vb::SparseOrbitalParameterView& parameter_view,
    const Eigen::VectorXd& packed_direction,
    const std::vector<double>& total_auxiliary_gradient,
    const std::vector<double>& total_inactive_density_gradient) {
  const int n_inactive_doubly_occupied_orbitals =
      (input.n_total_electrons - input.n_active_electrons) / 2;
  const auto tangent_context =
      build_fixed_upstream_tangent_debug_context(
          input,
          parameter_view,
          packed_direction);
  const Eigen::Map<const Eigen::MatrixXd> basis_overlap(
      input.active_orbital_overlap_matrix.data(),
      input.n_basis_functions,
      input.n_basis_functions);
  const Eigen::Map<const Eigen::MatrixXd> auxiliary_gradient_matrix(
      total_auxiliary_gradient.data(),
      input.n_basis_functions,
      input.n_basis_functions);
  const Eigen::Map<const Eigen::MatrixXd> inactive_density_gradient_matrix(
      total_inactive_density_gradient.data(),
      input.n_basis_functions,
      input.n_basis_functions);
  const Eigen::MatrixXd active_auxiliary_gradient =
      auxiliary_gradient_matrix.middleCols(
          n_inactive_doubly_occupied_orbitals,
          input.n_active_orbitals);

  Eigen::MatrixXd delta_original_orbital_gradient =
      Eigen::MatrixXd::Zero(input.n_basis_functions, input.n_orbitals);
  if (n_inactive_doubly_occupied_orbitals == 0) {
    return delta_original_orbital_gradient;
  }

  const auto inactive_orbitals =
      tangent_context.normalized_orbitals.leftCols(
          n_inactive_doubly_occupied_orbitals);
  const auto delta_inactive_orbitals =
      tangent_context.delta_normalized_orbitals.leftCols(
          n_inactive_doubly_occupied_orbitals);
  const auto active_orbitals =
      tangent_context.normalized_orbitals.middleCols(
          n_inactive_doubly_occupied_orbitals,
          input.n_active_orbitals);
  const auto delta_active_orbitals =
      tangent_context.delta_normalized_orbitals.middleCols(
          n_inactive_doubly_occupied_orbitals,
          input.n_active_orbitals);

  const Eigen::MatrixXd inactive_overlap =
      inactive_orbitals.transpose() * basis_overlap * inactive_orbitals;
  const Eigen::MatrixXd inactive_overlap_inverse = inactive_overlap.inverse();
  const Eigen::MatrixXd delta_inactive_overlap =
      delta_inactive_orbitals.transpose() * basis_overlap * inactive_orbitals +
      inactive_orbitals.transpose() * basis_overlap * delta_inactive_orbitals;
  const Eigen::MatrixXd delta_inactive_overlap_inverse =
      -inactive_overlap_inverse *
      delta_inactive_overlap *
      inactive_overlap_inverse;
  const Eigen::MatrixXd delta_inactive_density =
      delta_inactive_orbitals *
          inactive_overlap_inverse *
          inactive_orbitals.transpose() +
      inactive_orbitals *
          delta_inactive_overlap_inverse *
          inactive_orbitals.transpose() +
      inactive_orbitals *
          inactive_overlap_inverse *
          delta_inactive_orbitals.transpose();
  const Eigen::MatrixXd delta_original_active_gradient =
      -basis_overlap * delta_inactive_density * active_auxiliary_gradient;
  const Eigen::MatrixXd bs_active = basis_overlap * active_orbitals;
  const Eigen::MatrixXd delta_bs_active = basis_overlap * delta_active_orbitals;
  const Eigen::MatrixXd total_inactive_gradient =
      inactive_density_gradient_matrix -
      active_auxiliary_gradient * bs_active.transpose();
  const Eigen::MatrixXd delta_total_inactive_gradient =
      -active_auxiliary_gradient * delta_bs_active.transpose();
  const Eigen::MatrixXd inactive_density_gradient_symmetric =
      total_inactive_gradient + total_inactive_gradient.transpose();
  const Eigen::MatrixXd delta_inactive_density_gradient_symmetric =
      delta_total_inactive_gradient + delta_total_inactive_gradient.transpose();
  const Eigen::MatrixXd inactive_overlap_inverse_gradient =
      inactive_orbitals.transpose() * total_inactive_gradient * inactive_orbitals;
  const Eigen::MatrixXd delta_inactive_overlap_inverse_gradient =
      delta_inactive_orbitals.transpose() *
          total_inactive_gradient *
          inactive_orbitals +
      inactive_orbitals.transpose() *
          delta_total_inactive_gradient *
          inactive_orbitals +
      inactive_orbitals.transpose() *
          total_inactive_gradient *
          delta_inactive_orbitals;
  const Eigen::MatrixXd inactive_overlap_gradient =
      -inactive_overlap_inverse *
      inactive_overlap_inverse_gradient *
      inactive_overlap_inverse;
  const Eigen::MatrixXd delta_inactive_overlap_gradient =
      -delta_inactive_overlap_inverse *
          inactive_overlap_inverse_gradient *
          inactive_overlap_inverse -
      inactive_overlap_inverse *
          delta_inactive_overlap_inverse_gradient *
          inactive_overlap_inverse -
      inactive_overlap_inverse *
          inactive_overlap_inverse_gradient *
          delta_inactive_overlap_inverse;
  const Eigen::MatrixXd delta_original_inactive_gradient =
      delta_inactive_density_gradient_symmetric *
          inactive_orbitals *
          inactive_overlap_inverse +
      inactive_density_gradient_symmetric *
          delta_inactive_orbitals *
          inactive_overlap_inverse +
      inactive_density_gradient_symmetric *
          inactive_orbitals *
          delta_inactive_overlap_inverse +
      basis_overlap *
          delta_inactive_orbitals *
          (inactive_overlap_gradient + inactive_overlap_gradient.transpose()) +
      basis_overlap *
          inactive_orbitals *
          (delta_inactive_overlap_gradient +
           delta_inactive_overlap_gradient.transpose());

  delta_original_orbital_gradient.leftCols(
      n_inactive_doubly_occupied_orbitals) = delta_original_inactive_gradient;
  delta_original_orbital_gradient.middleCols(
      n_inactive_doubly_occupied_orbitals,
      input.n_active_orbitals) = delta_original_active_gradient;
  return delta_original_orbital_gradient;
}

std::pair<Eigen::Index, double> max_abs_difference_indexed(
    const Eigen::VectorXd& left,
    const Eigen::VectorXd& right) {
  if (left.size() != right.size()) {
    throw std::invalid_argument("vector size mismatch");
  }
  Eigen::Index max_index = 0;
  double max_abs_diff = 0.0;
  for (Eigen::Index index = 0; index < left.size(); ++index) {
    const double abs_diff = std::abs(left[index] - right[index]);
    if (abs_diff > max_abs_diff) {
      max_abs_diff = abs_diff;
      max_index = index;
    }
  }
  return {max_index, max_abs_diff};
}

std::size_t structure_upper_storage_index(
    int structure_row,
    int structure_column) {
  if (structure_row < 0 || structure_column < 0 ||
      structure_row > structure_column) {
    throw std::invalid_argument("structure upper-triangular index is out of range");
  }
  return xmvb::to_size(structure_column) * (structure_column + 1) / 2 +
      xmvb::to_size(structure_row);
}

Matrix unpack_symmetric_structure_matrix(
    const std::vector<double>& matrix_storage,
    int dimension) {
  const Eigen::Map<const Matrix> stored_matrix(
      matrix_storage.data(),
      dimension,
      dimension);
  Matrix symmetric_matrix = Matrix::Zero(dimension, dimension);
  for (int column = 0; column < dimension; ++column) {
    for (int row = 0; row <= column; ++row) {
      const double value = stored_matrix(row, column);
      symmetric_matrix(row, column) = value;
      symmetric_matrix(column, row) = value;
    }
  }
  return symmetric_matrix;
}

std::vector<double> pack_symmetric_structure_weight_matrix(
    const Matrix& weight_matrix) {
  const int dimension = static_cast<int>(weight_matrix.rows());
  std::vector<double> upper_weights(
      xmvb::to_size(dimension) * (dimension + 1) / 2,
      0.0);
  for (int structure_column = 0;
       structure_column < dimension;
       ++structure_column) {
    for (int structure_row = 0;
         structure_row <= structure_column;
         ++structure_row) {
      const double symmetry =
          structure_row == structure_column ? 1.0 : 2.0;
      upper_weights[structure_upper_storage_index(
          structure_row,
          structure_column)] =
          symmetry * weight_matrix(structure_row, structure_column);
    }
  }
  return upper_weights;
}

struct GeneralizedEigenDirectionalResponse {
  Matrix delta_eigenvector_matrix;
  std::vector<double> delta_eigenvalues;
};

GeneralizedEigenDirectionalResponse build_generalized_eigen_directional_response(
    const xmvb::vb::CppActiveSpaceSecondOrderContext& accepted_point_context,
    const Matrix& delta_hamiltonian,
    const Matrix& delta_overlap) {
  const int n_structures = accepted_point_context.structure_matrices.n_structures;
  const Matrix eigenvector_matrix =
      Eigen::Map<const Matrix>(
          accepted_point_context.eigen_result.eigenvector_matrix.data(),
          n_structures,
          n_structures);
  const Matrix transformed_delta_hamiltonian =
      eigenvector_matrix.transpose() * delta_hamiltonian * eigenvector_matrix;
  const Matrix transformed_delta_overlap =
      eigenvector_matrix.transpose() * delta_overlap * eigenvector_matrix;

  std::vector<double> selected_state_weights(
      xmvb::to_size(n_structures),
      0.0);
  for (std::size_t selected_state_offset = 0;
       selected_state_offset < accepted_point_context.selected_state_indices.size();
       ++selected_state_offset) {
    const int state_index =
        accepted_point_context.selected_state_indices[selected_state_offset];
    selected_state_weights[xmvb::to_size(state_index)] =
        accepted_point_context.normalized_state_weights[selected_state_offset];
  }

  Matrix eigenvector_rotation = Matrix::Zero(n_structures, n_structures);
  std::vector<double> delta_eigenvalues(xmvb::to_size(n_structures), 0.0);
  for (int state_index = 0; state_index < n_structures; ++state_index) {
    const double state_energy =
        accepted_point_context.eigen_result.eigenvalues[xmvb::to_size(state_index)];
    const double transformed_overlap_diagonal =
        transformed_delta_overlap(state_index, state_index);
    delta_eigenvalues[xmvb::to_size(state_index)] =
        transformed_delta_hamiltonian(state_index, state_index) -
        state_energy * transformed_overlap_diagonal;
    eigenvector_rotation(state_index, state_index) =
        -0.5 * transformed_overlap_diagonal;
  }

  constexpr double kDegeneracyToleranceScale = 1.0e3;
  for (int column_state = 0; column_state < n_structures; ++column_state) {
    const double column_energy =
        accepted_point_context.eigen_result.eigenvalues[xmvb::to_size(column_state)];
    for (int row_state = 0; row_state < n_structures; ++row_state) {
      if (row_state == column_state) {
        continue;
      }
      const double row_energy =
          accepted_point_context.eigen_result.eigenvalues[xmvb::to_size(row_state)];
      const double gap = column_energy - row_energy;
      const double gap_tolerance =
          kDegeneracyToleranceScale *
          std::numeric_limits<double>::epsilon() *
          std::max({1.0, std::abs(column_energy), std::abs(row_energy)});
      if (std::abs(gap) <= gap_tolerance) {
        const double row_weight =
            selected_state_weights[xmvb::to_size(row_state)];
        const double column_weight =
            selected_state_weights[xmvb::to_size(column_state)];
        if (std::abs(row_weight - column_weight) > 1.0e-12) {
          throw std::runtime_error(
              "near-degenerate selected-state block with unequal weights");
        }
        eigenvector_rotation(row_state, column_state) =
            -0.5 * transformed_delta_overlap(row_state, column_state);
        continue;
      }
      eigenvector_rotation(row_state, column_state) =
          (transformed_delta_hamiltonian(row_state, column_state) -
           accepted_point_context.eigen_result.eigenvalues[xmvb::to_size(column_state)] *
               transformed_delta_overlap(row_state, column_state)) /
          gap;
    }
  }

  GeneralizedEigenDirectionalResponse result;
  result.delta_eigenvector_matrix =
      eigenvector_matrix * eigenvector_rotation;
  result.delta_eigenvalues = std::move(delta_eigenvalues);
  return result;
}

struct StructurePairWeightTables {
  std::vector<double> hamiltonian_upper_weights;
  std::vector<double> overlap_upper_weights;
};

StructurePairWeightTables build_structure_pair_weight_tables(
    const xmvb::core::GeneralizedEigenResult& eigen_result,
    int n_structures,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights) {
  StructurePairWeightTables weights;
  const std::size_t n_upper_entries =
      xmvb::to_size(n_structures) * (n_structures + 1) / 2;
  weights.hamiltonian_upper_weights.assign(n_upper_entries, 0.0);
  weights.overlap_upper_weights.assign(n_upper_entries, 0.0);

  for (std::size_t selected_state_offset = 0;
       selected_state_offset < selected_state_indices.size();
       ++selected_state_offset) {
    const int state_index = selected_state_indices[selected_state_offset];
    const double state_weight = state_average_weights[selected_state_offset];
    const double state_energy = eigen_result.eigenvalues[xmvb::to_size(state_index)];
    const double* eigenvector_column =
        eigen_result.eigenvector_matrix.data() +
        xmvb::to_size(state_index) * n_structures;
    for (int structure_column = 0;
         structure_column < n_structures;
         ++structure_column) {
      const double coefficient_column = eigenvector_column[structure_column];
      const std::size_t column_offset =
          xmvb::to_size(structure_column) * (structure_column + 1) / 2;
      for (int structure_row = 0;
           structure_row <= structure_column;
           ++structure_row) {
        const double symmetry =
            structure_row == structure_column ? 1.0 : 2.0;
        const double weighted_product =
            symmetry * state_weight *
            eigenvector_column[structure_row] * coefficient_column;
        const std::size_t storage_index =
            column_offset + xmvb::to_size(structure_row);
        weights.hamiltonian_upper_weights[storage_index] += weighted_product;
        weights.overlap_upper_weights[storage_index] -=
            state_energy * weighted_product;
      }
    }
  }
  return weights;
}

StructurePairWeightTables build_directional_structure_pair_weight_tables(
    const xmvb::vb::CppActiveSpaceSecondOrderContext& accepted_point_context,
    const GeneralizedEigenDirectionalResponse& directional_eigensystem) {
  const int n_structures = accepted_point_context.structure_matrices.n_structures;
  const Matrix eigenvector_matrix =
      Eigen::Map<const Matrix>(
          accepted_point_context.eigen_result.eigenvector_matrix.data(),
          n_structures,
          n_structures);
  Matrix hamiltonian_weight_matrix =
      Matrix::Zero(n_structures, n_structures);
  Matrix overlap_weight_matrix =
      Matrix::Zero(n_structures, n_structures);
  for (std::size_t selected_state_offset = 0;
       selected_state_offset < accepted_point_context.selected_state_indices.size();
       ++selected_state_offset) {
    const int state_index =
        accepted_point_context.selected_state_indices[selected_state_offset];
    const double state_weight =
        accepted_point_context.normalized_state_weights[selected_state_offset];
    const double state_energy =
        accepted_point_context.eigen_result.eigenvalues[xmvb::to_size(state_index)];
    const double delta_state_energy =
        directional_eigensystem.delta_eigenvalues[xmvb::to_size(state_index)];
    const Eigen::VectorXd eigenvector =
        eigenvector_matrix.col(state_index);
    const Eigen::VectorXd delta_eigenvector =
        directional_eigensystem.delta_eigenvector_matrix.col(state_index);
    const Matrix directional_projector =
        delta_eigenvector * eigenvector.transpose() +
        eigenvector * delta_eigenvector.transpose();
    hamiltonian_weight_matrix.noalias() +=
        state_weight * directional_projector;
    overlap_weight_matrix.noalias() -=
        state_weight *
        (delta_state_energy * (eigenvector * eigenvector.transpose()) +
         state_energy * directional_projector);
  }

  StructurePairWeightTables result;
  result.hamiltonian_upper_weights =
      pack_symmetric_structure_weight_matrix(
          hamiltonian_weight_matrix);
  result.overlap_upper_weights =
      pack_symmetric_structure_weight_matrix(
          overlap_weight_matrix);
  return result;
}

double max_abs_difference(
    const std::vector<double>& left,
    const std::vector<double>& right) {
  if (left.size() != right.size()) {
    throw std::invalid_argument("vector size mismatch");
  }
  double max_abs_diff = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    max_abs_diff = std::max(max_abs_diff, std::abs(left[index] - right[index]));
  }
  return max_abs_diff;
}

std::vector<double> symmetric_average_storage(
    const std::vector<double>& matrix_storage) {
  const std::size_t flat_dimension =
      static_cast<std::size_t>(std::sqrt(static_cast<double>(matrix_storage.size())));
  if (flat_dimension * flat_dimension != matrix_storage.size()) {
    throw std::invalid_argument("matrix storage must be square");
  }
  const int dimension = static_cast<int>(flat_dimension);
  const Eigen::Map<const Matrix> matrix(
      matrix_storage.data(),
      dimension,
      dimension);
  const Matrix symmetric_average =
      0.5 * (matrix + matrix.transpose());
  return std::vector<double>(
      symmetric_average.data(),
      symmetric_average.data() + symmetric_average.size());
}

Eigen::VectorXd backpropagate_active_gradient_direction_to_orbital_response(
    const xmvb::vb::CppVbInput& input,
    const xmvb::vb::CppActiveSpaceSecondOrderContext& accepted_point_context,
    const xmvb::vb::SparseOrbitalParameterView& parameter_view,
    const xmvb::vb::NonredundantOrbitalSpace& nonredundant_space,
    const std::vector<double>& active_orbital_overlap_gradient,
    const std::vector<double>& active_one_electron_gradient,
    const std::vector<double>& packed_active_two_electron_gradient) {
  const int n_inactive_doubly_occupied_orbitals =
      (input.orbital_preparation_input.n_total_electrons -
       input.orbital_preparation_input.n_active_electrons) / 2;
  const int n_active_orbitals =
      input.orbital_preparation_input.n_active_orbitals;
  const auto& prepared_active_space =
      accepted_point_context.prepared_active_space;
  const auto& orbital_result = prepared_active_space.orbital_result;
  const auto& ao_effective_one_electron_result =
      prepared_active_space.ao_effective_one_electron_result;
  const auto& active_space_two_electron_result =
      prepared_active_space.active_space_two_electron_result;

  xmvb::vb::ActiveSpaceMatrixBackpropagator matrix_backpropagator;
  const auto active_space_matrix_backpropagation_result =
      matrix_backpropagator.backpropagate(
          active_orbital_overlap_gradient,
          active_one_electron_gradient,
          input.orbital_preparation_input.active_orbital_overlap_matrix,
          ao_effective_one_electron_result.ao_effective_h1e,
          orbital_result.auxiliary_orbital_matrix,
          input.orbital_preparation_input.n_basis_functions,
          n_inactive_doubly_occupied_orbitals,
          n_active_orbitals);

  xmvb::vb::ActiveSpaceTwoElectronBackpropagator active_space_two_electron_backpropagator;
  const auto active_space_two_electron_backpropagation_result =
      active_space_two_electron_backpropagator.backpropagate(
          packed_active_two_electron_gradient,
          input.ao_integral_input.ao_two_electron_integral_values,
          input.ao_integral_input.ao_two_electron_integral_indices,
          orbital_result,
          active_space_two_electron_result,
          input.orbital_preparation_input.n_basis_functions,
          n_inactive_doubly_occupied_orbitals,
          n_active_orbitals);

  xmvb::vb::AoEffectiveOneElectronBackpropagator ao_effective_one_electron_backpropagator;
  const auto ao_effective_one_electron_backpropagation_result =
      ao_effective_one_electron_backpropagator.backpropagate(
          active_space_matrix_backpropagation_result.ao_effective_one_electron_gradient,
          input.ao_integral_input);

  std::vector<double> total_inactive_density_gradient =
      ao_effective_one_electron_backpropagation_result.inactive_density_gradient;
  Eigen::MatrixXd total_active_auxiliary_gradient =
      active_space_matrix_backpropagation_result
          .active_auxiliary_orbital_gradient;
  if (total_active_auxiliary_gradient.rows() !=
          active_space_two_electron_backpropagation_result
              .active_auxiliary_orbital_gradient.rows() ||
      total_active_auxiliary_gradient.cols() !=
          active_space_two_electron_backpropagation_result
              .active_auxiliary_orbital_gradient.cols()) {
    throw std::runtime_error(
        "fd outer-response active auxiliary gradient shape mismatch");
  }
  total_active_auxiliary_gradient.noalias() +=
      active_space_two_electron_backpropagation_result
          .active_auxiliary_orbital_gradient;

  xmvb::vb::ActiveSpaceOrbitalBackpropagator orbital_backpropagator;
  const Eigen::Map<const Matrix> total_inactive_density_gradient_matrix(
      total_inactive_density_gradient.data(),
      input.orbital_preparation_input.n_basis_functions,
      input.orbital_preparation_input.n_basis_functions);
  const auto orbital_backpropagation_result =
      orbital_backpropagator.backpropagate(
          total_active_auxiliary_gradient,
          total_inactive_density_gradient_matrix,
          input.orbital_preparation_input,
          orbital_result);
  const Eigen::VectorXd packed_response =
      parameter_view.gather_from_full(
          orbital_backpropagation_result.orbital_value_gradient);
  return nonredundant_space.project_reduced_gradient(packed_response);
}

std::vector<double> symmetrize_square_storage_average(
    const std::vector<double>& matrix_storage,
    int dimension) {
  if (matrix_storage.size() != xmvb::to_size(dimension) * dimension) {
    throw std::invalid_argument("matrix storage size does not match dimension");
  }
  std::vector<double> symmetrized(matrix_storage.size(), 0.0);
  for (int column = 0; column < dimension; ++column) {
    for (int row = 0; row < dimension; ++row) {
      symmetrized[xmvb::to_size(column) * dimension + row] =
          0.5 *
          (matrix_storage[xmvb::to_size(column) * dimension + row] +
           matrix_storage[xmvb::to_size(row) * dimension + column]);
    }
  }
  return symmetrized;
}

std::string format_square_matrix(
    const std::vector<double>& matrix_storage,
    int dimension) {
  if (matrix_storage.size() != xmvb::to_size(dimension) * dimension) {
    throw std::invalid_argument("matrix storage size does not match dimension");
  }
  std::ostringstream stream;
  stream << std::setprecision(12);
  for (int row = 0; row < dimension; ++row) {
    for (int column = 0; column < dimension; ++column) {
      if (column != 0) {
        stream << ' ';
      }
      stream << matrix_storage[xmvb::to_size(column) * dimension + row];
    }
    if (row + 1 < dimension) {
      stream << '\n';
    }
  }
  return stream.str();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);

    xmvb::vb::CppVbInputLoadOptions load_options;
    load_options.ao_integral_source = options.ao_integral_source;
    load_options.standard_two_electron_mode =
        xmvb::vb::StandardTwoElectronMode::Exact;
    const auto load_result =
        xmvb::vb::load_cpp_vb_input_with_timings(options.input_path, load_options);
    const xmvb::vb::CppVbInput input =
        options.nonredundant_adapt
            ? xmvb::vb::build_nonredundant_optimizer_input(load_result.input)
            : load_result.input;

    xmvb::vb::CppOrbitalGradientEvaluator evaluator(
        xmvb::vb::VBSCFAlgorithm::Original);
    xmvb::vb::CppActiveSpaceGradientEvaluator active_space_evaluator(
        xmvb::vb::VBSCFAlgorithm::Original);
    const auto gradient_result =
        evaluator.evaluate_without_reference_energy_gradient(
            input,
            {0},
            {1.0},
            load_result.nuclear_repulsion_energy);
    if (gradient_result.second_order_context == nullptr) {
      throw std::runtime_error("accepted-point second-order context is unavailable");
    }

    xmvb::vb::SparseOrbitalParameterView parameter_view(
        input.orbital_preparation_input);
    const Eigen::VectorXd packed_parameters =
        parameter_view.pack(input.orbital_preparation_input);
    const Eigen::VectorXd packed_gradient =
        parameter_view.gather_from_full(
            gradient_result.sparse_orbital_energy_gradient);
    const int n_inactive_doubly_occupied_orbitals =
        (input.orbital_preparation_input.n_total_electrons -
         input.orbital_preparation_input.n_active_electrons) / 2;
    const int n_occupied_orbitals =
        n_inactive_doubly_occupied_orbitals +
        input.orbital_preparation_input.n_active_orbitals;
    const auto& normalized_orbital_matrix =
        gradient_result.orbital_preparation_result
            .physical_orbital_frame
            .normalized_orbital_matrix;
    if (normalized_orbital_matrix.size() == 0) {
      throw std::runtime_error(
          "nonredundant-space diagnostic requires the cached physical orbital frame");
    }
    xmvb::vb::NonredundantOrbitalSpace nonredundant_space(
        input.orbital_preparation_input,
        parameter_view,
        gradient_result.orbital_preparation_result
            .auxiliary_orbital_matrix
            .leftCols(n_occupied_orbitals),
        normalized_orbital_matrix,
        &gradient_result.ao_effective_one_electron_result.ao_effective_h1e);
    const Eigen::VectorXd reduced_gradient_projection =
        nonredundant_space.project_reduced_gradient(packed_gradient);
    if (reduced_gradient_projection.size() == 0) {
      throw std::runtime_error("nonredundant space is empty");
    }

    Eigen::VectorXd reduced_direction = reduced_gradient_projection;
    if (!(reduced_direction.norm() > 0.0)) {
      reduced_direction = Eigen::VectorXd::Zero(reduced_gradient_projection.size());
      reduced_direction[0] = 1.0;
    } else {
      reduced_direction /= reduced_direction.norm();
    }

    xmvb::vb::ExactOrbitalSecondOrderOperator exact_operator(
        gradient_result.second_order_context,
        &input,
        parameter_view,
        &nonredundant_space);
    if (!exact_operator.supports_analytic_core_model()) {
      throw std::runtime_error("exact_ctx analytic core model is unavailable");
    }
    setenv("XMVB_CPP_DISABLE_EXACT_CTX_OUTER_RESPONSE", "1", 1);
    const Eigen::VectorXd analytic_fixed_response =
        exact_operator.apply_reduced_uncached(reduced_direction);
    const Eigen::VectorXd analytic_fixed_response_cached =
        exact_operator.apply_reduced_without_outer_response_cached(
            reduced_direction);
    unsetenv("XMVB_CPP_DISABLE_EXACT_CTX_OUTER_RESPONSE");
    const Eigen::VectorXd analytic_direct_core_response_cached =
        exact_operator.apply_reduced_core_direct_only_cached(reduced_direction);
    const Eigen::VectorXd analytic_fixed_upstream_only_response_cached =
        exact_operator.apply_reduced_fixed_upstream_only_cached(reduced_direction);
    const Eigen::VectorXd analytic_direct_core_response =
        exact_operator.apply_reduced_core_direct_only(reduced_direction);
    const Eigen::VectorXd analytic_fixed_upstream_only_response =
        exact_operator.apply_reduced_fixed_upstream_only(reduced_direction);
    const Eigen::VectorXd analytic_direct_core_response_uncached =
        exact_operator.apply_reduced_core_direct_only_uncached(reduced_direction);
    const Eigen::VectorXd analytic_fixed_upstream_only_response_uncached =
        exact_operator.apply_reduced_fixed_upstream_only_uncached(reduced_direction);
    const Eigen::VectorXd analytic_full_response =
        exact_operator.apply_reduced_uncached(reduced_direction);
    const Eigen::VectorXd analytic_full_response_cached =
        exact_operator.apply_reduced_cached(reduced_direction);
    const Eigen::VectorXd analytic_outer_only_response =
        exact_operator.apply_reduced_outer_response_only_uncached(reduced_direction);
    const Eigen::VectorXd analytic_outer_only_response_cached =
        exact_operator.apply_reduced_outer_response_only_cached(
            reduced_direction);
    const auto analytic_directional_structure =
        exact_operator.compute_directional_structure_diagnostics(reduced_direction);

    const Eigen::VectorXd packed_direction =
        nonredundant_space.expand_step(reduced_direction);
    const double epsilon =
        options.step / std::max(1.0, packed_direction.norm());
    xmvb::vb::CppVbInput plus_input = input;
    xmvb::vb::CppVbInput minus_input = input;
    parameter_view.unpack(
        packed_parameters + epsilon * packed_direction,
        &plus_input.orbital_preparation_input);
    parameter_view.unpack(
        packed_parameters - epsilon * packed_direction,
        &minus_input.orbital_preparation_input);
    const auto plus_fixed_gradient_result =
        evaluator.evaluate_without_reference_energy_gradient_with_fixed_active_space_adjoint(
            plus_input,
            *gradient_result.second_order_context,
            load_result.nuclear_repulsion_energy);
    const auto minus_fixed_gradient_result =
        evaluator.evaluate_without_reference_energy_gradient_with_fixed_active_space_adjoint(
            minus_input,
            *gradient_result.second_order_context,
            load_result.nuclear_repulsion_energy);
    const auto plus_fixed_active_gradient_result =
        active_space_evaluator.evaluate_with_fixed_active_space_adjoint(
            plus_input,
            *gradient_result.second_order_context,
            load_result.nuclear_repulsion_energy);
    const auto minus_fixed_active_gradient_result =
        active_space_evaluator.evaluate_with_fixed_active_space_adjoint(
            minus_input,
            *gradient_result.second_order_context,
            load_result.nuclear_repulsion_energy);
    const auto accepted_fixed_active_gradient_result =
        active_space_evaluator.evaluate_with_fixed_active_space_adjoint(
            input,
            *gradient_result.second_order_context,
            load_result.nuclear_repulsion_energy);
    const auto plus_full_gradient_result =
        evaluator.evaluate_without_reference_energy_gradient(
            plus_input,
            gradient_result.second_order_context->selected_state_indices,
            gradient_result.second_order_context->normalized_state_weights,
            load_result.nuclear_repulsion_energy);
    const auto minus_full_gradient_result =
        evaluator.evaluate_without_reference_energy_gradient(
            minus_input,
            gradient_result.second_order_context->selected_state_indices,
            gradient_result.second_order_context->normalized_state_weights,
            load_result.nuclear_repulsion_energy);

    const Eigen::VectorXd plus_fixed_packed_gradient =
        parameter_view.gather_from_full(
            plus_fixed_gradient_result.sparse_orbital_energy_gradient);
    const Eigen::VectorXd minus_fixed_packed_gradient =
        parameter_view.gather_from_full(
            minus_fixed_gradient_result.sparse_orbital_energy_gradient);
    const Eigen::VectorXd fixed_fd_response =
        (nonredundant_space.project_reduced_gradient(plus_fixed_packed_gradient) -
         nonredundant_space.project_reduced_gradient(minus_fixed_packed_gradient)) /
        (2.0 * epsilon);

    const Eigen::VectorXd plus_full_packed_gradient =
        parameter_view.gather_from_full(
            plus_full_gradient_result.sparse_orbital_energy_gradient);
    const Eigen::VectorXd minus_full_packed_gradient =
        parameter_view.gather_from_full(
            minus_full_gradient_result.sparse_orbital_energy_gradient);
    const Eigen::VectorXd full_fd_response =
        (nonredundant_space.project_reduced_gradient(plus_full_packed_gradient) -
         nonredundant_space.project_reduced_gradient(minus_full_packed_gradient)) /
        (2.0 * epsilon);

    const Eigen::VectorXd analytic_outer_response =
        analytic_full_response - analytic_fixed_response;
    const Eigen::VectorXd fd_outer_response =
        full_fd_response - fixed_fd_response;
    const auto accepted_orbital_backprop_inputs =
        build_orbital_backprop_inputs(
            input,
            accepted_fixed_active_gradient_result);
    const auto plus_orbital_backprop_inputs =
        build_orbital_backprop_inputs(
            plus_input,
            plus_fixed_active_gradient_result);
    const auto minus_orbital_backprop_inputs =
        build_orbital_backprop_inputs(
            minus_input,
            minus_fixed_active_gradient_result);
    const std::vector<double> fd_total_auxiliary_gradient =
        finite_difference_storage(
            plus_orbital_backprop_inputs.total_auxiliary_gradient,
            minus_orbital_backprop_inputs.total_auxiliary_gradient,
            epsilon);
    const std::vector<double> fd_total_inactive_density_gradient =
        finite_difference_storage(
            plus_orbital_backprop_inputs.total_inactive_density_gradient,
            minus_orbital_backprop_inputs.total_inactive_density_gradient,
            epsilon);
    xmvb::vb::ActiveSpaceOrbitalBackpropagator orbital_backpropagator;
    const Eigen::MatrixXd fd_total_active_auxiliary_gradient =
        extract_active_auxiliary_gradient_block(
            input.orbital_preparation_input,
            fd_total_auxiliary_gradient);
    const Eigen::Map<const Matrix> fd_total_inactive_density_gradient_matrix(
        fd_total_inactive_density_gradient.data(),
        input.orbital_preparation_input.n_basis_functions,
        input.orbital_preparation_input.n_basis_functions);
    const Eigen::VectorXd fd_direct_upstream_response =
        project_full_orbital_gradient_to_reduced(
            parameter_view,
            nonredundant_space,
            orbital_backpropagator.backpropagate(
                fd_total_active_auxiliary_gradient,
                fd_total_inactive_density_gradient_matrix,
                input.orbital_preparation_input,
                gradient_result.orbital_preparation_result)
                .orbital_value_gradient);
    const Eigen::Map<const Matrix> accepted_total_inactive_density_gradient_matrix(
        accepted_orbital_backprop_inputs.total_inactive_density_gradient.data(),
        input.orbital_preparation_input.n_basis_functions,
        input.orbital_preparation_input.n_basis_functions);
    const Eigen::VectorXd plus_fixed_upstream_only_response =
        project_full_orbital_gradient_to_reduced(
            parameter_view,
            nonredundant_space,
            orbital_backpropagator.backpropagate(
                accepted_orbital_backprop_inputs.total_active_auxiliary_gradient,
                accepted_total_inactive_density_gradient_matrix,
                plus_input.orbital_preparation_input,
                plus_fixed_gradient_result.orbital_preparation_result)
                .orbital_value_gradient);
    const Eigen::VectorXd minus_fixed_upstream_only_response =
        project_full_orbital_gradient_to_reduced(
            parameter_view,
            nonredundant_space,
            orbital_backpropagator.backpropagate(
                accepted_orbital_backprop_inputs.total_active_auxiliary_gradient,
                accepted_total_inactive_density_gradient_matrix,
                minus_input.orbital_preparation_input,
                minus_fixed_gradient_result.orbital_preparation_result)
                .orbital_value_gradient);
    const Eigen::VectorXd fd_fixed_upstream_only_response =
        (plus_fixed_upstream_only_response -
         minus_fixed_upstream_only_response) /
        (2.0 * epsilon);
    const Eigen::VectorXd fd_fixed_response_from_split =
        fd_direct_upstream_response + fd_fixed_upstream_only_response;
    const auto plus_fixed_upstream_forward_debug_context =
        build_fixed_upstream_forward_debug_context(
            plus_input.orbital_preparation_input,
            accepted_orbital_backprop_inputs.total_auxiliary_gradient,
            accepted_orbital_backprop_inputs.total_inactive_density_gradient);
    const auto minus_fixed_upstream_forward_debug_context =
        build_fixed_upstream_forward_debug_context(
            minus_input.orbital_preparation_input,
            accepted_orbital_backprop_inputs.total_auxiliary_gradient,
            accepted_orbital_backprop_inputs.total_inactive_density_gradient);
    const Eigen::MatrixXd fd_delta_original_orbital_gradient =
        (plus_fixed_upstream_forward_debug_context.original_orbital_gradient -
         minus_fixed_upstream_forward_debug_context.original_orbital_gradient) /
        (2.0 * epsilon);
    const Eigen::MatrixXd analytic_delta_original_orbital_gradient =
        build_fixed_upstream_delta_original_orbital_gradient(
            input.orbital_preparation_input,
            parameter_view,
            packed_direction,
            accepted_orbital_backprop_inputs.total_auxiliary_gradient,
            accepted_orbital_backprop_inputs.total_inactive_density_gradient);

    const int n_structures =
        gradient_result.second_order_context->structure_matrices.n_structures;
    const Matrix plus_hamiltonian =
        unpack_symmetric_structure_matrix(
            plus_full_gradient_result.second_order_context->structure_matrices
                .hamiltonian_matrix,
            n_structures);
    const Matrix minus_hamiltonian =
        unpack_symmetric_structure_matrix(
            minus_full_gradient_result.second_order_context->structure_matrices
                .hamiltonian_matrix,
            n_structures);
    const Matrix plus_overlap =
        unpack_symmetric_structure_matrix(
            plus_full_gradient_result.second_order_context->structure_matrices
                .overlap_matrix,
            n_structures);
    const Matrix minus_overlap =
        unpack_symmetric_structure_matrix(
            minus_full_gradient_result.second_order_context->structure_matrices
                .overlap_matrix,
            n_structures);
    const Matrix fd_delta_hamiltonian =
        (plus_hamiltonian - minus_hamiltonian) / (2.0 * epsilon);
    const Matrix fd_delta_overlap =
        (plus_overlap - minus_overlap) / (2.0 * epsilon);
    const auto fd_directional_eigensystem =
        build_generalized_eigen_directional_response(
            *gradient_result.second_order_context,
            fd_delta_hamiltonian,
            fd_delta_overlap);
    const auto analytic_weight_response_from_fd_structure =
        build_directional_structure_pair_weight_tables(
            *gradient_result.second_order_context,
            fd_directional_eigensystem);
    const auto plus_structure_weights =
        build_structure_pair_weight_tables(
            plus_full_gradient_result.second_order_context->eigen_result,
            n_structures,
            plus_full_gradient_result.second_order_context->selected_state_indices,
            plus_full_gradient_result.second_order_context->normalized_state_weights);
    const auto minus_structure_weights =
        build_structure_pair_weight_tables(
            minus_full_gradient_result.second_order_context->eigen_result,
            n_structures,
            minus_full_gradient_result.second_order_context->selected_state_indices,
            minus_full_gradient_result.second_order_context->normalized_state_weights);
    std::vector<double> fd_hamiltonian_weight_direction(
        plus_structure_weights.hamiltonian_upper_weights.size(),
        0.0);
    std::vector<double> fd_overlap_weight_direction(
        plus_structure_weights.overlap_upper_weights.size(),
        0.0);
    for (std::size_t index = 0;
         index < fd_hamiltonian_weight_direction.size();
         ++index) {
      fd_hamiltonian_weight_direction[index] =
          (plus_structure_weights.hamiltonian_upper_weights[index] -
           minus_structure_weights.hamiltonian_upper_weights[index]) /
          (2.0 * epsilon);
      fd_overlap_weight_direction[index] =
          (plus_structure_weights.overlap_upper_weights[index] -
           minus_structure_weights.overlap_upper_weights[index]) /
          (2.0 * epsilon);
    }
    const double h_weight_max_abs_diff =
        max_abs_difference(
            analytic_weight_response_from_fd_structure.hamiltonian_upper_weights,
            fd_hamiltonian_weight_direction);
    const double s_weight_max_abs_diff =
        max_abs_difference(
            analytic_weight_response_from_fd_structure.overlap_upper_weights,
            fd_overlap_weight_direction);
    const std::vector<double> directional_selected_state_energies =
        xmvb::vb::gather_selected_state_energies(
            fd_directional_eigensystem.delta_eigenvalues,
            gradient_result.second_order_context->selected_state_indices);
    const std::vector<double> directional_eigenvector_storage(
        fd_directional_eigensystem.delta_eigenvector_matrix.data(),
        fd_directional_eigensystem.delta_eigenvector_matrix.data() +
            fd_directional_eigensystem.delta_eigenvector_matrix.size());
    const auto directional_selected_states =
        build_selected_state_determinant_matrices_from_normalized_weights(
            input.structure_data,
            directional_eigenvector_storage,
            gradient_result.second_order_context->selected_state_indices,
            gradient_result.second_order_context->normalized_state_weights,
            gradient_result.second_order_context->same_spin_pair_cache);
    const auto analytic_directional_pair_weights =
        build_directional_determinant_pair_weight_tables_from_coefficients(
            gradient_result.second_order_context->selected_state_matrices,
            directional_selected_states,
            gradient_result.second_order_context->selected_state_energies,
            directional_selected_state_energies);
    const auto plus_exact_pair_weights =
        build_exact_determinant_pair_weight_tables_from_eigenvalues(
            plus_full_gradient_result.second_order_context->selected_state_matrices,
            plus_full_gradient_result.second_order_context->eigen_result.eigenvalues);
    const auto minus_exact_pair_weights =
        build_exact_determinant_pair_weight_tables_from_eigenvalues(
            minus_full_gradient_result.second_order_context->selected_state_matrices,
            minus_full_gradient_result.second_order_context->eigen_result.eigenvalues);
    std::vector<double> fd_directional_pair_h(
        plus_exact_pair_weights.unordered_combined_hamiltonian_weights.size(),
        0.0);
    std::vector<double> fd_directional_pair_s(
        plus_exact_pair_weights.unordered_combined_overlap_weights.size(),
        0.0);
    for (std::size_t index = 0; index < fd_directional_pair_h.size(); ++index) {
      fd_directional_pair_h[index] =
          (plus_exact_pair_weights.unordered_combined_hamiltonian_weights[index] -
           minus_exact_pair_weights.unordered_combined_hamiltonian_weights[index]) /
          (2.0 * epsilon);
      fd_directional_pair_s[index] =
          (plus_exact_pair_weights.unordered_combined_overlap_weights[index] -
           minus_exact_pair_weights.unordered_combined_overlap_weights[index]) /
          (2.0 * epsilon);
    }
    const double pair_h_max_abs_diff =
        max_abs_difference(
            analytic_directional_pair_weights.unordered_combined_hamiltonian_weights,
            fd_directional_pair_h);
    const double pair_s_max_abs_diff =
        max_abs_difference(
            analytic_directional_pair_weights.unordered_combined_overlap_weights,
            fd_directional_pair_s);
    const double k_nan = std::numeric_limits<double>::quiet_NaN();
    bool have_directional_same_spin = false;
    bool have_directional_opposite_spin = false;
    xmvb::vb::SameSpinMatrixBackwardContribution analytic_same_spin_direction;
    xmvb::vb::OppositeSpinMatrixBackwardContribution analytic_opposite_spin_direction;
    double same_spin_fixed_hho_max_abs_diff = k_nan;
    double same_spin_fixed_sso_max_abs_diff = k_nan;
    double opposite_spin_fixed_sso_max_abs_diff = k_nan;
    double opposite_spin_fixed_ggo_max_abs_diff = k_nan;
    try {
      analytic_same_spin_direction =
          xmvb::vb::build_directional_same_spin_matrix_backward_contribution(
              gradient_result.second_order_context->same_spin_pair_cache,
              gradient_result.second_order_context->selected_state_matrices,
              directional_selected_states,
              gradient_result.second_order_context->selected_state_energies,
              directional_selected_state_energies,
              gradient_result.second_order_context->n_active_orbitals);
      const auto plus_same_spin_fixed_cache =
          xmvb::vb::build_same_spin_matrix_backward_contribution(
              gradient_result.second_order_context->same_spin_pair_cache,
              plus_full_gradient_result.second_order_context->selected_state_matrices,
              plus_full_gradient_result.second_order_context->selected_state_energies,
              gradient_result.second_order_context->n_active_orbitals);
      const auto minus_same_spin_fixed_cache =
          xmvb::vb::build_same_spin_matrix_backward_contribution(
              gradient_result.second_order_context->same_spin_pair_cache,
              minus_full_gradient_result.second_order_context->selected_state_matrices,
              minus_full_gradient_result.second_order_context->selected_state_energies,
              gradient_result.second_order_context->n_active_orbitals);
      std::vector<double> fd_same_spin_fixed_hho(
          plus_same_spin_fixed_cache.active_one_electron_gradient.size(),
          0.0);
      std::vector<double> fd_same_spin_fixed_sso(
          plus_same_spin_fixed_cache.active_orbital_overlap_gradient.size(),
          0.0);
      for (std::size_t index = 0; index < fd_same_spin_fixed_hho.size(); ++index) {
        fd_same_spin_fixed_hho[index] =
            (plus_same_spin_fixed_cache.active_one_electron_gradient[index] -
             minus_same_spin_fixed_cache.active_one_electron_gradient[index]) /
            (2.0 * epsilon);
      }
      for (std::size_t index = 0; index < fd_same_spin_fixed_sso.size(); ++index) {
        fd_same_spin_fixed_sso[index] =
            (plus_same_spin_fixed_cache.active_orbital_overlap_gradient[index] -
             minus_same_spin_fixed_cache.active_orbital_overlap_gradient[index]) /
            (2.0 * epsilon);
      }
      same_spin_fixed_hho_max_abs_diff =
          max_abs_difference(
              analytic_same_spin_direction.active_one_electron_gradient,
              fd_same_spin_fixed_hho);
      same_spin_fixed_sso_max_abs_diff =
          max_abs_difference(
              analytic_same_spin_direction.active_orbital_overlap_gradient,
              fd_same_spin_fixed_sso);
      have_directional_same_spin = true;
    } catch (const std::exception& error) {
      std::cerr << "warning: skipped directional same-spin validation: "
                << error.what() << '\n';
    }
    try {
      analytic_opposite_spin_direction =
          xmvb::vb::build_directional_opposite_spin_matrix_backward_contribution(
              gradient_result.second_order_context->same_spin_pair_cache,
              gradient_result.second_order_context->selected_state_matrices,
              directional_selected_states,
              gradient_result.second_order_context->n_active_orbitals);
      const auto plus_opposite_spin_fixed_cache =
          xmvb::vb::build_opposite_spin_matrix_backward_contribution(
              gradient_result.second_order_context->same_spin_pair_cache,
              plus_full_gradient_result.second_order_context->selected_state_matrices,
              gradient_result.second_order_context->n_active_orbitals);
      const auto minus_opposite_spin_fixed_cache =
          xmvb::vb::build_opposite_spin_matrix_backward_contribution(
              gradient_result.second_order_context->same_spin_pair_cache,
              minus_full_gradient_result.second_order_context->selected_state_matrices,
              gradient_result.second_order_context->n_active_orbitals);
      std::vector<double> fd_opposite_spin_fixed_sso(
          plus_opposite_spin_fixed_cache.active_orbital_overlap_gradient.size(),
          0.0);
      std::vector<double> fd_opposite_spin_fixed_ggo(
          plus_opposite_spin_fixed_cache.packed_active_two_electron_gradient.size(),
          0.0);
      for (std::size_t index = 0;
           index < fd_opposite_spin_fixed_sso.size();
           ++index) {
        fd_opposite_spin_fixed_sso[index] =
            (plus_opposite_spin_fixed_cache.active_orbital_overlap_gradient[index] -
             minus_opposite_spin_fixed_cache.active_orbital_overlap_gradient[index]) /
            (2.0 * epsilon);
      }
      for (std::size_t index = 0;
           index < fd_opposite_spin_fixed_ggo.size();
           ++index) {
        fd_opposite_spin_fixed_ggo[index] =
            (plus_opposite_spin_fixed_cache.packed_active_two_electron_gradient[index] -
             minus_opposite_spin_fixed_cache.packed_active_two_electron_gradient[index]) /
            (2.0 * epsilon);
      }
      opposite_spin_fixed_sso_max_abs_diff =
          max_abs_difference(
              analytic_opposite_spin_direction.active_orbital_overlap_gradient,
              fd_opposite_spin_fixed_sso);
      opposite_spin_fixed_ggo_max_abs_diff =
          max_abs_difference(
              analytic_opposite_spin_direction.packed_active_two_electron_gradient,
              fd_opposite_spin_fixed_ggo);
      have_directional_opposite_spin = true;
    } catch (const std::exception& error) {
      std::cerr << "warning: skipped directional opposite-spin validation: "
                << error.what() << '\n';
    }
    const Matrix analytic_delta_hamiltonian =
        unpack_symmetric_structure_matrix(
            analytic_directional_structure.hamiltonian_matrix,
            n_structures);
    const Matrix analytic_delta_overlap =
        unpack_symmetric_structure_matrix(
            analytic_directional_structure.overlap_matrix,
            n_structures);
    const auto analytic_directional_eigensystem =
        build_generalized_eigen_directional_response(
            *gradient_result.second_order_context,
            analytic_delta_hamiltonian,
            analytic_delta_overlap);
    const std::vector<double> analytic_directional_selected_state_energies =
        xmvb::vb::gather_selected_state_energies(
            analytic_directional_eigensystem.delta_eigenvalues,
            gradient_result.second_order_context->selected_state_indices);
    const std::vector<double> analytic_directional_eigenvector_storage(
        analytic_directional_eigensystem.delta_eigenvector_matrix.data(),
        analytic_directional_eigensystem.delta_eigenvector_matrix.data() +
            analytic_directional_eigensystem.delta_eigenvector_matrix.size());
    const auto analytic_directional_selected_states =
        build_selected_state_determinant_matrices_from_normalized_weights(
            input.structure_data,
            analytic_directional_eigenvector_storage,
            gradient_result.second_order_context->selected_state_indices,
            gradient_result.second_order_context->normalized_state_weights,
            gradient_result.second_order_context->same_spin_pair_cache);
    const auto analytic_directional_pair_weights_from_analytic_structure =
        build_directional_determinant_pair_weight_tables_from_coefficients(
            gradient_result.second_order_context->selected_state_matrices,
            analytic_directional_selected_states,
            gradient_result.second_order_context->selected_state_energies,
            analytic_directional_selected_state_energies);
    xmvb::vb::SameSpinMatrixBackwardContribution
        analytic_same_spin_direction_from_analytic_structure;
    xmvb::vb::OppositeSpinMatrixBackwardContribution
        analytic_opposite_spin_direction_from_analytic_structure;
    if (have_directional_same_spin) {
      analytic_same_spin_direction_from_analytic_structure =
          xmvb::vb::build_directional_same_spin_matrix_backward_contribution(
              gradient_result.second_order_context->same_spin_pair_cache,
              gradient_result.second_order_context->selected_state_matrices,
              analytic_directional_selected_states,
              gradient_result.second_order_context->selected_state_energies,
              analytic_directional_selected_state_energies,
              gradient_result.second_order_context->n_active_orbitals);
    }
    if (have_directional_opposite_spin) {
      analytic_opposite_spin_direction_from_analytic_structure =
          xmvb::vb::build_directional_opposite_spin_matrix_backward_contribution(
              gradient_result.second_order_context->same_spin_pair_cache,
              gradient_result.second_order_context->selected_state_matrices,
              analytic_directional_selected_states,
              gradient_result.second_order_context->n_active_orbitals);
    }
    const double structure_h_max_abs_diff =
        max_abs_difference(
            std::vector<double>(
                analytic_delta_hamiltonian.data(),
                analytic_delta_hamiltonian.data() + analytic_delta_hamiltonian.size()),
            std::vector<double>(
                fd_delta_hamiltonian.data(),
                fd_delta_hamiltonian.data() + fd_delta_hamiltonian.size()));
    const double structure_s_max_abs_diff =
        max_abs_difference(
            std::vector<double>(
                analytic_delta_overlap.data(),
                analytic_delta_overlap.data() + analytic_delta_overlap.size()),
            std::vector<double>(
                fd_delta_overlap.data(),
                fd_delta_overlap.data() + fd_delta_overlap.size()));
    const double analytic_eigensystem_pair_h_max_abs_diff =
        max_abs_difference(
            analytic_directional_pair_weights_from_analytic_structure
                .unordered_combined_hamiltonian_weights,
            fd_directional_pair_h);
    const double analytic_eigensystem_pair_s_max_abs_diff =
        max_abs_difference(
            analytic_directional_pair_weights_from_analytic_structure
                .unordered_combined_overlap_weights,
            fd_directional_pair_s);
    const auto& plus_prepared_active_space =
        plus_full_gradient_result.second_order_context->prepared_active_space;
    const auto& minus_prepared_active_space =
        minus_full_gradient_result.second_order_context->prepared_active_space;
    const Matrix plus_active_overlap =
        Eigen::Map<const Matrix>(
            plus_prepared_active_space.orbital_result.active_orbital_overlap_matrix.data(),
            gradient_result.second_order_context->n_active_orbitals,
            gradient_result.second_order_context->n_active_orbitals);
    const Matrix minus_active_overlap =
        Eigen::Map<const Matrix>(
            minus_prepared_active_space.orbital_result.active_orbital_overlap_matrix.data(),
            gradient_result.second_order_context->n_active_orbitals,
            gradient_result.second_order_context->n_active_orbitals);
    const Matrix plus_active_one_electron =
        Eigen::Map<const Matrix>(
            plus_prepared_active_space.active_space_one_electron_result.h1e_act.data(),
            gradient_result.second_order_context->n_active_orbitals,
            gradient_result.second_order_context->n_active_orbitals);
    const Matrix minus_active_one_electron =
        Eigen::Map<const Matrix>(
            minus_prepared_active_space.active_space_one_electron_result.h1e_act.data(),
            gradient_result.second_order_context->n_active_orbitals,
            gradient_result.second_order_context->n_active_orbitals);
    const Matrix fd_delta_active_overlap =
        (plus_active_overlap - minus_active_overlap) / (2.0 * epsilon);
    const Matrix fd_delta_active_one_electron =
        (plus_active_one_electron - minus_active_one_electron) / (2.0 * epsilon);
    std::vector<double> fd_delta_active_two_electron(
        plus_prepared_active_space.active_space_two_electron_result
            .packed_active_two_electron_integrals.size(),
        0.0);
    for (std::size_t index = 0;
         index < fd_delta_active_two_electron.size();
         ++index) {
      fd_delta_active_two_electron[index] =
          (plus_prepared_active_space.active_space_two_electron_result
               .packed_active_two_electron_integrals[index] -
           minus_prepared_active_space.active_space_two_electron_result
               .packed_active_two_electron_integrals[index]) /
          (2.0 * epsilon);
    }
    const double active_sso_max_abs_diff =
        max_abs_difference(
            analytic_directional_structure.active_orbital_overlap_matrix,
            std::vector<double>(
                fd_delta_active_overlap.data(),
                fd_delta_active_overlap.data() + fd_delta_active_overlap.size()));
    const double active_hho_max_abs_diff =
        max_abs_difference(
            analytic_directional_structure.active_one_electron_matrix,
            std::vector<double>(
                fd_delta_active_one_electron.data(),
                fd_delta_active_one_electron.data() +
                    fd_delta_active_one_electron.size()));
    const double active_ggo_max_abs_diff =
        max_abs_difference(
            analytic_directional_structure.packed_active_two_electron_integrals,
            fd_delta_active_two_electron);
    const auto accepted_exact_pair_weights =
        build_exact_determinant_pair_weight_tables_from_eigenvalues(
            gradient_result.second_order_context->selected_state_matrices,
            gradient_result.second_order_context->eigen_result.eigenvalues);
    const auto analytic_local_same_spin_direction =
        xmvb::vb::build_local_same_spin_matrix_backward_contribution(
            gradient_result.second_order_context->same_spin_pair_cache,
            gradient_result.second_order_context->selected_state_matrices,
            gradient_result.second_order_context->selected_state_energies,
            gradient_result.second_order_context->n_active_orbitals,
            gradient_result.second_order_context->prepared_active_space
                .active_space_one_electron_result.h1e_act,
            gradient_result.second_order_context->prepared_active_space
                .active_space_two_electron_result,
            analytic_directional_structure.active_orbital_overlap_matrix,
            analytic_directional_structure.active_one_electron_matrix,
            analytic_directional_structure.packed_active_two_electron_integrals);
    const auto pairwise_local_same_spin_direction =
        xmvb::vb::build_pairwise_local_same_spin_matrix_backward_reference(
            input,
            *gradient_result.second_order_context,
            accepted_exact_pair_weights,
            analytic_directional_structure.active_orbital_overlap_matrix,
            analytic_directional_structure.active_one_electron_matrix,
            analytic_directional_structure.packed_active_two_electron_integrals);
    const auto analytic_local_same_spin_direction_repeat =
        xmvb::vb::build_local_same_spin_matrix_backward_contribution(
            gradient_result.second_order_context->same_spin_pair_cache,
            gradient_result.second_order_context->selected_state_matrices,
            gradient_result.second_order_context->selected_state_energies,
            gradient_result.second_order_context->n_active_orbitals,
            gradient_result.second_order_context->prepared_active_space
                .active_space_one_electron_result.h1e_act,
            gradient_result.second_order_context->prepared_active_space
                .active_space_two_electron_result,
            analytic_directional_structure.active_orbital_overlap_matrix,
            analytic_directional_structure.active_one_electron_matrix,
            analytic_directional_structure.packed_active_two_electron_integrals);
    const auto analytic_local_opposite_spin_direction =
        xmvb::vb::build_local_opposite_spin_matrix_backward_contribution(
            gradient_result.second_order_context->same_spin_pair_cache,
            gradient_result.second_order_context->selected_state_matrices,
            gradient_result.second_order_context->n_active_orbitals,
            gradient_result.second_order_context->prepared_active_space
                .active_space_two_electron_result,
            analytic_directional_structure.active_orbital_overlap_matrix,
            analytic_directional_structure.packed_active_two_electron_integrals);
    const auto pairwise_local_opposite_spin_direction =
        xmvb::vb::build_pairwise_local_opposite_spin_matrix_backward_reference(
            input,
            *gradient_result.second_order_context,
            accepted_exact_pair_weights,
            analytic_directional_structure.active_orbital_overlap_matrix,
            analytic_directional_structure.active_one_electron_matrix,
            analytic_directional_structure.packed_active_two_electron_integrals);
    const auto plus_local_opposite_spin_fixed_states =
        xmvb::vb::build_opposite_spin_matrix_backward_contribution(
            plus_full_gradient_result.second_order_context->same_spin_pair_cache,
            gradient_result.second_order_context->selected_state_matrices,
            gradient_result.second_order_context->n_active_orbitals);
    const auto minus_local_opposite_spin_fixed_states =
        xmvb::vb::build_opposite_spin_matrix_backward_contribution(
            minus_full_gradient_result.second_order_context->same_spin_pair_cache,
            gradient_result.second_order_context->selected_state_matrices,
            gradient_result.second_order_context->n_active_orbitals);
    std::vector<double> fd_local_opposite_spin_sso(
        plus_local_opposite_spin_fixed_states.active_orbital_overlap_gradient.size(),
        0.0);
    std::vector<double> fd_local_opposite_spin_ggo(
        plus_local_opposite_spin_fixed_states.packed_active_two_electron_gradient.size(),
        0.0);
    for (std::size_t index = 0;
         index < fd_local_opposite_spin_sso.size();
         ++index) {
      fd_local_opposite_spin_sso[index] =
          (plus_local_opposite_spin_fixed_states
               .active_orbital_overlap_gradient[index] -
           minus_local_opposite_spin_fixed_states
               .active_orbital_overlap_gradient[index]) /
          (2.0 * epsilon);
    }
    for (std::size_t index = 0;
         index < fd_local_opposite_spin_ggo.size();
         ++index) {
      fd_local_opposite_spin_ggo[index] =
          (plus_local_opposite_spin_fixed_states
               .packed_active_two_electron_gradient[index] -
           minus_local_opposite_spin_fixed_states
               .packed_active_two_electron_gradient[index]) /
          (2.0 * epsilon);
    }
    const double local_opposite_spin_sso_max_abs_diff =
        max_abs_difference(
            analytic_local_opposite_spin_direction.active_orbital_overlap_gradient,
            fd_local_opposite_spin_sso);
    const double local_opposite_spin_ggo_max_abs_diff =
        max_abs_difference(
            analytic_local_opposite_spin_direction.packed_active_two_electron_gradient,
            fd_local_opposite_spin_ggo);
    const double pairwise_local_opposite_spin_sso_max_abs_diff =
        max_abs_difference(
            pairwise_local_opposite_spin_direction.active_orbital_overlap_gradient,
            fd_local_opposite_spin_sso);
    const double pairwise_local_opposite_spin_ggo_max_abs_diff =
        max_abs_difference(
            pairwise_local_opposite_spin_direction.packed_active_two_electron_gradient,
            fd_local_opposite_spin_ggo);
    const double matrix_vs_pairwise_local_opposite_spin_sso_max_abs_diff =
        max_abs_difference(
            analytic_local_opposite_spin_direction.active_orbital_overlap_gradient,
            pairwise_local_opposite_spin_direction.active_orbital_overlap_gradient);
    const double matrix_vs_pairwise_local_opposite_spin_ggo_max_abs_diff =
        max_abs_difference(
            analytic_local_opposite_spin_direction.packed_active_two_electron_gradient,
            pairwise_local_opposite_spin_direction.packed_active_two_electron_gradient);
    const double local_same_spin_repeat_sso_max_abs_diff =
        max_abs_difference(
            analytic_local_same_spin_direction.active_orbital_overlap_gradient,
            analytic_local_same_spin_direction_repeat.active_orbital_overlap_gradient);
    double analytic_state_same_spin_sso_max_abs_diff = k_nan;
    if (have_directional_same_spin) {
      analytic_state_same_spin_sso_max_abs_diff =
          max_abs_difference(
              analytic_same_spin_direction_from_analytic_structure
                  .active_orbital_overlap_gradient,
              analytic_same_spin_direction.active_orbital_overlap_gradient);
    }
    double analytic_state_opposite_spin_sso_max_abs_diff = k_nan;
    if (have_directional_opposite_spin) {
      analytic_state_opposite_spin_sso_max_abs_diff =
          max_abs_difference(
              analytic_opposite_spin_direction_from_analytic_structure
                  .active_orbital_overlap_gradient,
              analytic_opposite_spin_direction.active_orbital_overlap_gradient);
    }
    const bool have_full_matrix_form_sum =
        have_directional_same_spin && have_directional_opposite_spin;
    std::vector<double> matrix_form_sum_sso;
    std::vector<double> matrix_form_sum_hho;
    std::vector<double> matrix_form_sum_ggo;
    double matrix_form_sum_sso_max_abs_diff = k_nan;
    std::size_t matrix_form_sum_sso_max_index = 0;
    double matrix_form_sum_sso_max_abs_diff_at_index = k_nan;
    double matrix_form_sum_hho_max_abs_diff = k_nan;
    double matrix_form_sum_ggo_max_abs_diff = k_nan;
    if (have_full_matrix_form_sum) {
      matrix_form_sum_sso =
          analytic_local_same_spin_direction.active_orbital_overlap_gradient;
      matrix_form_sum_hho =
          analytic_local_same_spin_direction.active_one_electron_gradient;
      matrix_form_sum_ggo =
          analytic_local_same_spin_direction.packed_active_two_electron_gradient;
      for (std::size_t index = 0; index < matrix_form_sum_sso.size(); ++index) {
        matrix_form_sum_sso[index] +=
            analytic_local_opposite_spin_direction.active_orbital_overlap_gradient[index] +
            analytic_same_spin_direction_from_analytic_structure
                .active_orbital_overlap_gradient[index] +
            analytic_opposite_spin_direction_from_analytic_structure
                .active_orbital_overlap_gradient[index];
      }
      for (std::size_t index = 0; index < matrix_form_sum_hho.size(); ++index) {
        matrix_form_sum_hho[index] +=
            analytic_same_spin_direction_from_analytic_structure
                .active_one_electron_gradient[index];
      }
      for (std::size_t index = 0; index < matrix_form_sum_ggo.size(); ++index) {
        matrix_form_sum_ggo[index] +=
            analytic_local_opposite_spin_direction.packed_active_two_electron_gradient[index] +
            analytic_same_spin_direction_from_analytic_structure
                .packed_active_two_electron_gradient[index] +
            analytic_opposite_spin_direction_from_analytic_structure
                .packed_active_two_electron_gradient[index];
      }
      matrix_form_sum_sso_max_abs_diff =
          max_abs_difference(
              matrix_form_sum_sso,
              analytic_directional_structure.active_orbital_overlap_gradient);
      matrix_form_sum_sso_max_abs_diff_at_index = 0.0;
      for (std::size_t index = 0; index < matrix_form_sum_sso.size(); ++index) {
        const double abs_diff = std::abs(
            matrix_form_sum_sso[index] -
            analytic_directional_structure.active_orbital_overlap_gradient[index]);
        if (abs_diff > matrix_form_sum_sso_max_abs_diff_at_index) {
          matrix_form_sum_sso_max_abs_diff_at_index = abs_diff;
          matrix_form_sum_sso_max_index = index;
        }
      }
      matrix_form_sum_hho_max_abs_diff =
          max_abs_difference(
              matrix_form_sum_hho,
              analytic_directional_structure.active_one_electron_gradient);
      matrix_form_sum_ggo_max_abs_diff =
          max_abs_difference(
              matrix_form_sum_ggo,
              analytic_directional_structure.packed_active_two_electron_gradient);
    }
    std::vector<double> fd_delta_active_overlap_gradient(
        plus_full_gradient_result.second_order_context->active_orbital_overlap_gradient.size(),
        0.0);
    std::vector<double> fd_delta_active_one_electron_gradient(
        plus_full_gradient_result.second_order_context->active_one_electron_gradient.size(),
        0.0);
    std::vector<double> fd_delta_active_two_electron_gradient(
        plus_full_gradient_result.second_order_context->packed_active_two_electron_gradient.size(),
        0.0);
    std::vector<double> fd_delta_fixed_active_overlap_gradient(
        plus_fixed_active_gradient_result.active_orbital_overlap_gradient.size(),
        0.0);
    std::vector<double> fd_delta_fixed_active_one_electron_gradient(
        plus_fixed_active_gradient_result.active_one_electron_gradient.size(),
        0.0);
    std::vector<double> fd_delta_fixed_active_two_electron_gradient(
        plus_fixed_active_gradient_result.packed_active_two_electron_gradient.size(),
        0.0);
    for (std::size_t index = 0;
         index < fd_delta_active_overlap_gradient.size();
         ++index) {
      fd_delta_active_overlap_gradient[index] =
          (plus_full_gradient_result.second_order_context
               ->active_orbital_overlap_gradient[index] -
           minus_full_gradient_result.second_order_context
               ->active_orbital_overlap_gradient[index]) /
          (2.0 * epsilon);
    }
    for (std::size_t index = 0;
         index < fd_delta_active_one_electron_gradient.size();
         ++index) {
      fd_delta_active_one_electron_gradient[index] =
          (plus_full_gradient_result.second_order_context
               ->active_one_electron_gradient[index] -
           minus_full_gradient_result.second_order_context
               ->active_one_electron_gradient[index]) /
          (2.0 * epsilon);
    }
    for (std::size_t index = 0;
         index < fd_delta_active_two_electron_gradient.size();
         ++index) {
      fd_delta_active_two_electron_gradient[index] =
          (plus_full_gradient_result.second_order_context
               ->packed_active_two_electron_gradient[index] -
           minus_full_gradient_result.second_order_context
               ->packed_active_two_electron_gradient[index]) /
          (2.0 * epsilon);
    }
    for (std::size_t index = 0;
         index < fd_delta_fixed_active_overlap_gradient.size();
         ++index) {
      fd_delta_fixed_active_overlap_gradient[index] =
          (plus_fixed_active_gradient_result
               .active_orbital_overlap_gradient[index] -
           minus_fixed_active_gradient_result
               .active_orbital_overlap_gradient[index]) /
          (2.0 * epsilon);
    }
    for (std::size_t index = 0;
         index < fd_delta_fixed_active_one_electron_gradient.size();
         ++index) {
      fd_delta_fixed_active_one_electron_gradient[index] =
          (plus_fixed_active_gradient_result
               .active_one_electron_gradient[index] -
           minus_fixed_active_gradient_result
               .active_one_electron_gradient[index]) /
          (2.0 * epsilon);
    }
    for (std::size_t index = 0;
         index < fd_delta_fixed_active_two_electron_gradient.size();
         ++index) {
      fd_delta_fixed_active_two_electron_gradient[index] =
          (plus_fixed_active_gradient_result
               .packed_active_two_electron_gradient[index] -
           minus_fixed_active_gradient_result
               .packed_active_two_electron_gradient[index]) /
          (2.0 * epsilon);
    }
    const double matrix_vs_pairwise_local_same_spin_sso_max_abs_diff =
        max_abs_difference(
            analytic_local_same_spin_direction.active_orbital_overlap_gradient,
            pairwise_local_same_spin_direction.active_orbital_overlap_gradient);
    const double matrix_vs_pairwise_local_same_spin_hho_max_abs_diff =
        max_abs_difference(
            analytic_local_same_spin_direction.active_one_electron_gradient,
            pairwise_local_same_spin_direction.active_one_electron_gradient);
    const double matrix_vs_pairwise_local_same_spin_ggo_max_abs_diff =
        max_abs_difference(
            analytic_local_same_spin_direction.packed_active_two_electron_gradient,
            pairwise_local_same_spin_direction.packed_active_two_electron_gradient);
    std::vector<double> fd_outer_active_overlap_gradient(
        fd_delta_active_overlap_gradient.size(),
        0.0);
    std::vector<double> fd_outer_active_one_electron_gradient(
        fd_delta_active_one_electron_gradient.size(),
        0.0);
    std::vector<double> fd_outer_active_two_electron_gradient(
        fd_delta_active_two_electron_gradient.size(),
        0.0);
    for (std::size_t index = 0;
         index < fd_outer_active_overlap_gradient.size();
         ++index) {
      fd_outer_active_overlap_gradient[index] =
          fd_delta_active_overlap_gradient[index] -
          fd_delta_fixed_active_overlap_gradient[index];
    }
    for (std::size_t index = 0;
         index < fd_outer_active_one_electron_gradient.size();
         ++index) {
      fd_outer_active_one_electron_gradient[index] =
          fd_delta_active_one_electron_gradient[index] -
          fd_delta_fixed_active_one_electron_gradient[index];
    }
    for (std::size_t index = 0;
         index < fd_outer_active_two_electron_gradient.size();
         ++index) {
      fd_outer_active_two_electron_gradient[index] =
          fd_delta_active_two_electron_gradient[index] -
          fd_delta_fixed_active_two_electron_gradient[index];
    }
    const double grad_sso_max_abs_diff =
        max_abs_difference(
            analytic_directional_structure.active_orbital_overlap_gradient,
            fd_outer_active_overlap_gradient);
    const double grad_sso_symmetric_max_abs_diff =
        max_abs_difference(
            symmetric_average_storage(
                analytic_directional_structure.active_orbital_overlap_gradient),
            symmetric_average_storage(
                fd_outer_active_overlap_gradient));
    const double grad_hho_max_abs_diff =
        max_abs_difference(
            analytic_directional_structure.active_one_electron_gradient,
            fd_outer_active_one_electron_gradient);
    const double grad_hho_symmetric_max_abs_diff =
        max_abs_difference(
            symmetric_average_storage(
                analytic_directional_structure.active_one_electron_gradient),
            symmetric_average_storage(
                fd_outer_active_one_electron_gradient));
    const double grad_ggo_max_abs_diff =
        max_abs_difference(
            analytic_directional_structure.packed_active_two_electron_gradient,
            fd_outer_active_two_electron_gradient);
    const Eigen::VectorXd fd_outer_response_from_fd_active_gradient =
        backpropagate_active_gradient_direction_to_orbital_response(
            input,
            *gradient_result.second_order_context,
            parameter_view,
            nonredundant_space,
            fd_outer_active_overlap_gradient,
            fd_outer_active_one_electron_gradient,
            fd_outer_active_two_electron_gradient);
    const Eigen::VectorXd fd_fixed_response_from_fixed_active_gradient =
        backpropagate_active_gradient_direction_to_orbital_response(
            input,
            *gradient_result.second_order_context,
            parameter_view,
            nonredundant_space,
            fd_delta_fixed_active_overlap_gradient,
            fd_delta_fixed_active_one_electron_gradient,
            fd_delta_fixed_active_two_electron_gradient);
    const Eigen::VectorXd fd_full_response_from_full_active_gradient =
        backpropagate_active_gradient_direction_to_orbital_response(
            input,
            *gradient_result.second_order_context,
            parameter_view,
            nonredundant_space,
            fd_delta_active_overlap_gradient,
            fd_delta_active_one_electron_gradient,
            fd_delta_active_two_electron_gradient);
    const double fd_outer_pullback_max_abs_diff =
        max_abs_difference(
            fd_outer_response_from_fd_active_gradient,
            fd_outer_response);
    const double fd_fixed_pullback_max_abs_diff =
        max_abs_difference(
            fd_fixed_response_from_fixed_active_gradient,
            fixed_fd_response);
    const double fd_full_pullback_max_abs_diff =
        max_abs_difference(
            fd_full_response_from_full_active_gradient,
            full_fd_response);
    const double fd_direct_upstream_split_max_abs_diff =
        max_abs_difference(
            fd_direct_upstream_response,
            fixed_fd_response - fd_fixed_upstream_only_response);
    const double fd_fixed_upstream_only_split_max_abs_diff =
        max_abs_difference(
            fd_fixed_upstream_only_response,
            fixed_fd_response - fd_direct_upstream_response);
    const double fd_fixed_split_reconstruction_max_abs_diff =
        max_abs_difference(
            fd_fixed_response_from_split,
            fixed_fd_response);
    const double analytic_direct_upstream_max_abs_diff =
        max_abs_difference(
            analytic_direct_core_response,
            fd_direct_upstream_response);
    const double analytic_fixed_upstream_only_max_abs_diff =
        max_abs_difference(
            analytic_fixed_upstream_only_response,
            fd_fixed_upstream_only_response);
    const double analytic_direct_upstream_cached_max_abs_diff =
        max_abs_difference(
            analytic_direct_core_response_cached,
            fd_direct_upstream_response);
    const double analytic_fixed_upstream_only_cached_max_abs_diff =
        max_abs_difference(
            analytic_fixed_upstream_only_response_cached,
            fd_fixed_upstream_only_response);
    const double analytic_direct_upstream_uncached_max_abs_diff =
        max_abs_difference(
            analytic_direct_core_response_uncached,
            fd_direct_upstream_response);
    const double analytic_fixed_upstream_only_uncached_max_abs_diff =
        max_abs_difference(
            analytic_fixed_upstream_only_response_uncached,
            fd_fixed_upstream_only_response);
    const double analytic_direct_cached_vs_uncached_max_abs_diff =
        max_abs_difference(
            analytic_direct_core_response_cached,
            analytic_direct_core_response_uncached);
    const double analytic_fixed_cached_vs_uncached_max_abs_diff =
        max_abs_difference(
            analytic_fixed_upstream_only_response_cached,
            analytic_fixed_upstream_only_response_uncached);
    const double analytic_fixed_stage_a_max_abs_diff =
        (analytic_delta_original_orbital_gradient -
         fd_delta_original_orbital_gradient)
            .cwiseAbs()
            .maxCoeff();
    const double analytic_fixed_split_reconstruction_max_abs_diff =
        max_abs_difference(
            analytic_direct_core_response + analytic_fixed_upstream_only_response,
            analytic_fixed_response);
    const double analytic_fixed_minus_fd_direct_max_abs_diff =
        max_abs_difference(
            analytic_fixed_response - fd_direct_upstream_response,
            fd_fixed_upstream_only_response);
    const double analytic_fixed_minus_fd_fixed_upstream_only_max_abs_diff =
        max_abs_difference(
            analytic_fixed_response - fd_fixed_upstream_only_response,
            fd_direct_upstream_response);
    const int n_active_orbitals =
        gradient_result.second_order_context->n_active_orbitals;
    const std::vector<double> analytic_sym_active_overlap_gradient =
        symmetrize_square_storage_average(
            analytic_directional_structure.active_orbital_overlap_gradient,
            n_active_orbitals);
    const std::vector<double> analytic_sym_active_one_electron_gradient =
        symmetrize_square_storage_average(
            analytic_directional_structure.active_one_electron_gradient,
            n_active_orbitals);
    const std::vector<double> fd_sym_active_overlap_gradient =
        symmetrize_square_storage_average(
            fd_delta_active_overlap_gradient,
            n_active_orbitals);
    const std::vector<double> fd_sym_active_one_electron_gradient =
        symmetrize_square_storage_average(
            fd_delta_active_one_electron_gradient,
            n_active_orbitals);
    const double sym_grad_sso_max_abs_diff =
        max_abs_difference(
            analytic_sym_active_overlap_gradient,
            fd_sym_active_overlap_gradient);
    const double sym_grad_hho_max_abs_diff =
        max_abs_difference(
            analytic_sym_active_one_electron_gradient,
            fd_sym_active_one_electron_gradient);
    const Eigen::VectorXd analytic_outer_response_from_sym_active_gradient =
        backpropagate_active_gradient_direction_to_orbital_response(
            input,
            *gradient_result.second_order_context,
            parameter_view,
            nonredundant_space,
            analytic_sym_active_overlap_gradient,
            analytic_sym_active_one_electron_gradient,
            analytic_directional_structure.packed_active_two_electron_gradient);
    const Eigen::VectorXd fd_outer_response_from_sym_fd_active_gradient =
        backpropagate_active_gradient_direction_to_orbital_response(
            input,
            *gradient_result.second_order_context,
            parameter_view,
            nonredundant_space,
            fd_sym_active_overlap_gradient,
            fd_sym_active_one_electron_gradient,
            fd_delta_active_two_electron_gradient);
    const double analytic_outer_sym_pullback_max_abs_diff =
        max_abs_difference(
            analytic_outer_response_from_sym_active_gradient,
            fd_outer_response);
    const double fd_outer_sym_pullback_max_abs_diff =
        max_abs_difference(
            fd_outer_response_from_sym_fd_active_gradient,
            fd_outer_response);
    const double analytic_outer_only_vs_split_max_abs_diff =
        max_abs_difference(
            analytic_outer_only_response,
            analytic_outer_response);
    const double analytic_outer_only_cached_max_abs_diff =
        max_abs_difference(
            analytic_outer_only_response_cached,
            fd_outer_response);
    const double analytic_outer_cached_vs_uncached_max_abs_diff =
        max_abs_difference(
            analytic_outer_only_response_cached,
            analytic_outer_only_response);
    const double analytic_outer_only_vs_diag_pullback_max_abs_diff =
        max_abs_difference(
            analytic_outer_only_response,
            analytic_outer_response_from_sym_active_gradient);
    const double analytic_outer_split_vs_diag_pullback_max_abs_diff =
        max_abs_difference(
            analytic_outer_response,
            analytic_outer_response_from_sym_active_gradient);

    const Eigen::VectorXd& analytic_response =
        options.probe == "fixed" ? analytic_fixed_response : analytic_full_response;
    const Eigen::VectorXd& finite_difference_response =
        options.probe == "fixed" ? fixed_fd_response : full_fd_response;
    const double max_abs_diff =
        max_abs_difference(
            analytic_response,
            finite_difference_response);
    const double max_abs_fd =
        max_abs_value(finite_difference_response);
    const double fixed_max_abs_diff =
        max_abs_difference(
            analytic_fixed_response,
            fixed_fd_response);
    const double fixed_cached_max_abs_diff =
        max_abs_difference(
            analytic_fixed_response_cached,
            fixed_fd_response);
    const double full_max_abs_diff =
        max_abs_difference(
            analytic_full_response,
            full_fd_response);
    const double full_cached_max_abs_diff =
        max_abs_difference(
            analytic_full_response_cached,
            full_fd_response);
    const double analytic_full_cached_vs_uncached_max_abs_diff =
        max_abs_difference(
            analytic_full_response_cached,
            analytic_full_response);
    const double analytic_fixed_cached_vs_uncached_full_max_abs_diff =
        max_abs_difference(
            analytic_fixed_response_cached,
            analytic_fixed_response);
    const auto [direct_upstream_max_index, direct_upstream_max_abs_diff] =
        max_abs_difference_indexed(
            analytic_direct_core_response,
            fd_direct_upstream_response);
    const auto [fixed_upstream_only_max_index, fixed_upstream_only_max_abs_diff] =
        max_abs_difference_indexed(
            analytic_fixed_upstream_only_response,
            fd_fixed_upstream_only_response);
    Eigen::Index fixed_stage_a_max_row = 0;
    Eigen::Index fixed_stage_a_max_column = 0;
    const double analytic_fixed_stage_a_max_abs_diff_indexed =
        (analytic_delta_original_orbital_gradient -
         fd_delta_original_orbital_gradient)
            .cwiseAbs()
            .maxCoeff(
                &fixed_stage_a_max_row,
                &fixed_stage_a_max_column);
    const auto [outer_max_index, outer_max_abs_diff] =
        max_abs_difference_indexed(
            analytic_outer_response,
            fd_outer_response);

    std::cout << std::setprecision(12);
    std::cout << "input = " << options.input_path << '\n';
    std::cout << "ao_integral_source = "
              << xmvb::vb::ao_integral_source_name(load_result.ao_integral_source) << '\n';
    std::cout << "reduced_dimension = " << reduced_direction.size() << '\n';
    std::cout << "probe_mode = " << options.probe << '\n';
    std::cout << "nonredundant_adapt = "
              << (options.nonredundant_adapt ? "true" : "false") << '\n';
    std::cout << "finite_difference_step = " << options.step << '\n';
    std::cout << "use_full_matrix_form_adjoint = "
              << (gradient_result.second_order_context->use_full_matrix_form_adjoint ? 1 : 0)
              << '\n';
    std::cout << "use_matrix_form_opposite_spin = "
              << (gradient_result.second_order_context->use_matrix_form_opposite_spin ? 1 : 0)
              << '\n';
    std::cout << "analytic_inf_norm = " << max_abs_value(analytic_response) << '\n';
    std::cout << "fd_inf_norm = " << max_abs_fd << '\n';
    std::cout << "max_abs_diff = " << max_abs_diff << '\n';
    std::cout << "max_rel_diff = "
              << max_abs_diff / std::max(1.0, max_abs_fd) << '\n';
    std::cout << "analytic_fixed_inf_norm = "
              << max_abs_value(analytic_fixed_response) << '\n';
    std::cout << "fixed_fd_inf_norm = "
              << max_abs_value(fixed_fd_response) << '\n';
    std::cout << "fixed_max_abs_diff = " << fixed_max_abs_diff << '\n';
    std::cout << "fixed_cached_max_abs_diff = "
              << fixed_cached_max_abs_diff << '\n';
    std::cout << "analytic_full_inf_norm = "
              << max_abs_value(analytic_full_response) << '\n';
    std::cout << "full_fd_inf_norm = "
              << max_abs_value(full_fd_response) << '\n';
    std::cout << "full_max_abs_diff = " << full_max_abs_diff << '\n';
    std::cout << "full_cached_max_abs_diff = "
              << full_cached_max_abs_diff << '\n';
    std::cout << "analytic_outer_inf_norm = "
              << max_abs_value(analytic_outer_response) << '\n';
    std::cout << "analytic_outer_only_inf_norm = "
              << max_abs_value(analytic_outer_only_response) << '\n';
    std::cout << "fd_outer_inf_norm = "
              << max_abs_value(fd_outer_response) << '\n';
    std::cout << "outer_max_abs_diff = " << outer_max_abs_diff << '\n';
    std::cout << "analytic_outer_only_vs_split_max_abs_diff = "
              << analytic_outer_only_vs_split_max_abs_diff << '\n';
    std::cout << "analytic_outer_only_cached_max_abs_diff = "
              << analytic_outer_only_cached_max_abs_diff << '\n';
    std::cout << "analytic_outer_cached_vs_uncached_max_abs_diff = "
              << analytic_outer_cached_vs_uncached_max_abs_diff << '\n';
    std::cout << "analytic_full_cached_vs_uncached_max_abs_diff = "
              << analytic_full_cached_vs_uncached_max_abs_diff << '\n';
    std::cout << "analytic_fixed_cached_vs_uncached_full_max_abs_diff = "
              << analytic_fixed_cached_vs_uncached_full_max_abs_diff << '\n';
    std::cout << "analytic_outer_only_vs_diag_pullback_max_abs_diff = "
              << analytic_outer_only_vs_diag_pullback_max_abs_diff << '\n';
    std::cout << "analytic_outer_split_vs_diag_pullback_max_abs_diff = "
              << analytic_outer_split_vs_diag_pullback_max_abs_diff << '\n';
    std::cout << "outer_max_index = " << outer_max_index << '\n';
    std::cout << "analytic_outer_at_max = "
              << analytic_outer_response[outer_max_index] << '\n';
    std::cout << "fd_outer_at_max = "
              << fd_outer_response[outer_max_index] << '\n';
    std::cout << "weight_h_max_abs_diff_from_fd_structure = "
              << h_weight_max_abs_diff << '\n';
    std::cout << "weight_s_max_abs_diff_from_fd_structure = "
              << s_weight_max_abs_diff << '\n';
    std::cout << "pair_h_max_abs_diff = "
              << pair_h_max_abs_diff << '\n';
    std::cout << "pair_s_max_abs_diff = "
              << pair_s_max_abs_diff << '\n';
    std::cout << "same_spin_fixed_hho_max_abs_diff = "
              << same_spin_fixed_hho_max_abs_diff << '\n';
    std::cout << "same_spin_fixed_sso_max_abs_diff = "
              << same_spin_fixed_sso_max_abs_diff << '\n';
    std::cout << "opposite_spin_fixed_sso_max_abs_diff = "
              << opposite_spin_fixed_sso_max_abs_diff << '\n';
    std::cout << "opposite_spin_fixed_ggo_max_abs_diff = "
              << opposite_spin_fixed_ggo_max_abs_diff << '\n';
    std::cout << "structure_h_max_abs_diff = "
              << structure_h_max_abs_diff << '\n';
    std::cout << "structure_s_max_abs_diff = "
              << structure_s_max_abs_diff << '\n';
    std::cout << "analytic_eigensystem_pair_h_max_abs_diff = "
              << analytic_eigensystem_pair_h_max_abs_diff << '\n';
    std::cout << "analytic_eigensystem_pair_s_max_abs_diff = "
              << analytic_eigensystem_pair_s_max_abs_diff << '\n';
    std::cout << "active_sso_max_abs_diff = "
              << active_sso_max_abs_diff << '\n';
    std::cout << "active_hho_max_abs_diff = "
              << active_hho_max_abs_diff << '\n';
    std::cout << "active_ggo_max_abs_diff = "
              << active_ggo_max_abs_diff << '\n';
    std::cout << "local_opposite_spin_sso_max_abs_diff = "
              << local_opposite_spin_sso_max_abs_diff << '\n';
    std::cout << "local_opposite_spin_ggo_max_abs_diff = "
              << local_opposite_spin_ggo_max_abs_diff << '\n';
    std::cout << "pairwise_local_opposite_spin_sso_max_abs_diff = "
              << pairwise_local_opposite_spin_sso_max_abs_diff << '\n';
    std::cout << "pairwise_local_opposite_spin_ggo_max_abs_diff = "
              << pairwise_local_opposite_spin_ggo_max_abs_diff << '\n';
    std::cout << "matrix_vs_pairwise_local_opposite_spin_sso_max_abs_diff = "
              << matrix_vs_pairwise_local_opposite_spin_sso_max_abs_diff << '\n';
    std::cout << "matrix_vs_pairwise_local_opposite_spin_ggo_max_abs_diff = "
              << matrix_vs_pairwise_local_opposite_spin_ggo_max_abs_diff << '\n';
    std::cout << "local_same_spin_repeat_sso_max_abs_diff = "
              << local_same_spin_repeat_sso_max_abs_diff << '\n';
    std::cout << "matrix_vs_pairwise_local_same_spin_sso_max_abs_diff = "
              << matrix_vs_pairwise_local_same_spin_sso_max_abs_diff << '\n';
    std::cout << "matrix_vs_pairwise_local_same_spin_hho_max_abs_diff = "
              << matrix_vs_pairwise_local_same_spin_hho_max_abs_diff << '\n';
    std::cout << "matrix_vs_pairwise_local_same_spin_ggo_max_abs_diff = "
              << matrix_vs_pairwise_local_same_spin_ggo_max_abs_diff << '\n';
    std::cout << "analytic_state_same_spin_sso_max_abs_diff = "
              << analytic_state_same_spin_sso_max_abs_diff << '\n';
    std::cout << "analytic_state_opposite_spin_sso_max_abs_diff = "
              << analytic_state_opposite_spin_sso_max_abs_diff << '\n';
    std::cout << "matrix_form_sum_sso_max_abs_diff = "
              << matrix_form_sum_sso_max_abs_diff << '\n';
    if (have_full_matrix_form_sum) {
      std::cout << "matrix_form_sum_sso_max_index = "
                << matrix_form_sum_sso_max_index << '\n';
      std::cout << "matrix_form_sum_sso_max_abs_diff_indexed = "
                << matrix_form_sum_sso_max_abs_diff_at_index << '\n';
      std::cout << "matrix_form_sum_sso_at_max = "
                << matrix_form_sum_sso[matrix_form_sum_sso_max_index] << '\n';
      std::cout << "analytic_grad_sso_at_max = "
                << analytic_directional_structure
                       .active_orbital_overlap_gradient[matrix_form_sum_sso_max_index]
                << '\n';
      std::cout << "local_same_spin_sso_at_max = "
                << analytic_local_same_spin_direction
                       .active_orbital_overlap_gradient[matrix_form_sum_sso_max_index]
                << '\n';
      std::cout << "local_opposite_spin_sso_at_max = "
                << analytic_local_opposite_spin_direction
                       .active_orbital_overlap_gradient[matrix_form_sum_sso_max_index]
                << '\n';
      std::cout << "directional_same_spin_sso_at_max = "
                << analytic_same_spin_direction_from_analytic_structure
                       .active_orbital_overlap_gradient[matrix_form_sum_sso_max_index]
                << '\n';
      std::cout << "directional_opposite_spin_sso_at_max = "
                << analytic_opposite_spin_direction_from_analytic_structure
                       .active_orbital_overlap_gradient[matrix_form_sum_sso_max_index]
                << '\n';
    }
    std::cout << "matrix_form_sum_hho_max_abs_diff = "
              << matrix_form_sum_hho_max_abs_diff << '\n';
    std::cout << "matrix_form_sum_ggo_max_abs_diff = "
              << matrix_form_sum_ggo_max_abs_diff << '\n';
    std::cout << "grad_sso_max_abs_diff = "
              << grad_sso_max_abs_diff << '\n';
    std::cout << "grad_sso_symmetric_max_abs_diff = "
              << grad_sso_symmetric_max_abs_diff << '\n';
    std::cout << "grad_hho_max_abs_diff = "
              << grad_hho_max_abs_diff << '\n';
    std::cout << "grad_hho_symmetric_max_abs_diff = "
              << grad_hho_symmetric_max_abs_diff << '\n';
    std::cout << "grad_ggo_max_abs_diff = "
              << grad_ggo_max_abs_diff << '\n';
    std::cout << "sym_grad_sso_max_abs_diff = "
              << sym_grad_sso_max_abs_diff << '\n';
    std::cout << "sym_grad_hho_max_abs_diff = "
              << sym_grad_hho_max_abs_diff << '\n';
    std::cout << "fd_outer_pullback_max_abs_diff = "
              << fd_outer_pullback_max_abs_diff << '\n';
    std::cout << "fd_fixed_pullback_max_abs_diff = "
              << fd_fixed_pullback_max_abs_diff << '\n';
    std::cout << "fd_full_pullback_max_abs_diff = "
              << fd_full_pullback_max_abs_diff << '\n';
    std::cout << "fd_direct_upstream_inf_norm = "
              << max_abs_value(fd_direct_upstream_response) << '\n';
    std::cout << "fd_fixed_upstream_only_inf_norm = "
              << max_abs_value(fd_fixed_upstream_only_response) << '\n';
    std::cout << "fd_direct_upstream_split_max_abs_diff = "
              << fd_direct_upstream_split_max_abs_diff << '\n';
    std::cout << "fd_fixed_upstream_only_split_max_abs_diff = "
              << fd_fixed_upstream_only_split_max_abs_diff << '\n';
    std::cout << "fd_fixed_split_reconstruction_max_abs_diff = "
              << fd_fixed_split_reconstruction_max_abs_diff << '\n';
    std::cout << "analytic_direct_upstream_max_abs_diff = "
              << analytic_direct_upstream_max_abs_diff << '\n';
    std::cout << "direct_upstream_max_index = "
              << direct_upstream_max_index << '\n';
    std::cout << "analytic_direct_upstream_at_max = "
              << analytic_direct_core_response[direct_upstream_max_index] << '\n';
    std::cout << "fd_direct_upstream_at_max = "
              << fd_direct_upstream_response[direct_upstream_max_index] << '\n';
    std::cout << "analytic_fixed_upstream_only_max_abs_diff = "
              << analytic_fixed_upstream_only_max_abs_diff << '\n';
    std::cout << "fixed_upstream_only_max_index = "
              << fixed_upstream_only_max_index << '\n';
    std::cout << "analytic_fixed_upstream_only_at_max = "
              << analytic_fixed_upstream_only_response[fixed_upstream_only_max_index] << '\n';
    std::cout << "fd_fixed_upstream_only_at_max = "
              << fd_fixed_upstream_only_response[fixed_upstream_only_max_index] << '\n';
    std::cout << "analytic_direct_upstream_cached_max_abs_diff = "
              << analytic_direct_upstream_cached_max_abs_diff << '\n';
    std::cout << "analytic_fixed_upstream_only_cached_max_abs_diff = "
              << analytic_fixed_upstream_only_cached_max_abs_diff << '\n';
    std::cout << "analytic_direct_upstream_uncached_max_abs_diff = "
              << analytic_direct_upstream_uncached_max_abs_diff << '\n';
    std::cout << "analytic_fixed_upstream_only_uncached_max_abs_diff = "
              << analytic_fixed_upstream_only_uncached_max_abs_diff << '\n';
    std::cout << "analytic_direct_cached_vs_uncached_max_abs_diff = "
              << analytic_direct_cached_vs_uncached_max_abs_diff << '\n';
    std::cout << "analytic_fixed_cached_vs_uncached_max_abs_diff = "
              << analytic_fixed_cached_vs_uncached_max_abs_diff << '\n';
    std::cout << "analytic_fixed_stage_a_max_abs_diff = "
              << analytic_fixed_stage_a_max_abs_diff << '\n';
    std::cout << "analytic_fixed_stage_a_max_abs_diff_indexed = "
              << analytic_fixed_stage_a_max_abs_diff_indexed << '\n';
    std::cout << "fixed_stage_a_max_row = "
              << fixed_stage_a_max_row << '\n';
    std::cout << "fixed_stage_a_max_column = "
              << fixed_stage_a_max_column << '\n';
    std::cout << "analytic_fixed_stage_a_at_max = "
              << analytic_delta_original_orbital_gradient(
                     fixed_stage_a_max_row,
                     fixed_stage_a_max_column)
              << '\n';
    std::cout << "fd_fixed_stage_a_at_max = "
              << fd_delta_original_orbital_gradient(
                     fixed_stage_a_max_row,
                     fixed_stage_a_max_column)
              << '\n';
    std::cout << "analytic_fixed_split_reconstruction_max_abs_diff = "
              << analytic_fixed_split_reconstruction_max_abs_diff << '\n';
    std::cout << "analytic_fixed_minus_fd_direct_max_abs_diff = "
              << analytic_fixed_minus_fd_direct_max_abs_diff << '\n';
    std::cout << "analytic_fixed_minus_fd_fixed_upstream_only_max_abs_diff = "
              << analytic_fixed_minus_fd_fixed_upstream_only_max_abs_diff << '\n';
    std::cout << "analytic_outer_sym_pullback_max_abs_diff = "
              << analytic_outer_sym_pullback_max_abs_diff << '\n';
    std::cout << "fd_outer_sym_pullback_max_abs_diff = "
              << fd_outer_sym_pullback_max_abs_diff << '\n';
    if (n_active_orbitals <= 4) {
      std::cout << "analytic_hho_gradient_matrix =\n"
                << format_square_matrix(
                       analytic_directional_structure.active_one_electron_gradient,
                       n_active_orbitals)
                << '\n';
      std::cout << "fd_hho_gradient_matrix =\n"
                << format_square_matrix(
                       fd_delta_active_one_electron_gradient,
                       n_active_orbitals)
                << '\n';
      std::cout << "analytic_sso_gradient_matrix =\n"
                << format_square_matrix(
                       analytic_directional_structure.active_orbital_overlap_gradient,
                       n_active_orbitals)
                << '\n';
      std::cout << "fd_sso_gradient_matrix =\n"
                << format_square_matrix(
                       fd_delta_active_overlap_gradient,
                       n_active_orbitals)
                << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
