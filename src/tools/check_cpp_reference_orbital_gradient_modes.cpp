#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/matrices/cpp_vb_input_ri_cache.hpp"
#include "vb/orbital/active_space_orbital_backpropagator.hpp"
#include "vb/orbital/ao_effective_one_electron_backpropagator.hpp"
#include "vb/scf/cpp_orbital_gradient_evaluator.hpp"
#include "vb/scf/cpp_vb_scf_evaluator.hpp"
#include "vb/vbscf_algorithm.hpp"

namespace {

using Matrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

struct Options {
  std::string input_path;
  xmvb::vb::VBSCFAlgorithm algorithm = xmvb::vb::VBSCFAlgorithm::Original;
  int count = 4;
  double step = 1.0e-6;
};

int get_sparse_coefficient_count(
    const xmvb::vb::OrbitalPreparationInput& orbital_preparation_input,
    int orbital_index) {
  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  const int explicit_count =
      orbital_preparation_input.orbital_basis_counts[xmvb::to_size(orbital_index)];
  if (explicit_count > 1) {
    return explicit_count;
  }

  int coefficient_count = 0;
  while (coefficient_count < n_basis_functions) {
    const int basis_function_index =
        orbital_preparation_input.orbital_basis_index_table
            [xmvb::to_size(orbital_index) * n_basis_functions + coefficient_count];
    if (basis_function_index == 0) {
      break;
    }
    ++coefficient_count;
  }
  return coefficient_count;
}

std::vector<int> collect_differentiable_parameter_indices(
    const xmvb::vb::OrbitalPreparationInput& orbital_preparation_input) {
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

Options parse_arguments(int argc, char** argv) {
  if (argc < 2 || ((argc - 2) % 2 != 0)) {
    throw std::invalid_argument(
        "usage: check_cpp_reference_orbital_gradient_modes <input.xmi> "
        "[--algorithm original] [--count N] [--step h]");
  }
  Options options;
  options.input_path = argv[1];
  for (int argument_index = 2; argument_index < argc; argument_index += 2) {
    const std::string name = argv[argument_index];
    const std::string value = argv[argument_index + 1];
    if (name == "--algorithm") {
      if (value != "original") {
        throw std::invalid_argument("invalid algorithm: " + value);
      }
      continue;
    }
    if (name == "--count") {
      options.count = std::stoi(value);
      continue;
    }
    if (name == "--step") {
      options.step = std::stod(value);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + name);
  }
  return options;
}

double evaluate_reference_energy(
    const xmvb::vb::CppVbInput& input,
    xmvb::vb::VBSCFAlgorithm algorithm,
    double nuclear_repulsion_energy) {
  xmvb::vb::CppVbScfEvaluator evaluator(algorithm);
  return evaluator.evaluate(input, nuclear_repulsion_energy).one_electron_reference_energy;
}

std::vector<double> lower_triangle_half_diagonal(
    const std::vector<double>& symmetric_matrix,
    int n_basis_functions) {
  const Eigen::Map<const Matrix> input(
      symmetric_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  Matrix output = Matrix::Zero(n_basis_functions, n_basis_functions);
  for (int column = 0; column < n_basis_functions; ++column) {
    for (int row = column; row < n_basis_functions; ++row) {
      output(row, column) = input(row, column);
    }
    output(column, column) *= 0.5;
  }
  return std::vector<double>(output.data(), output.data() + output.size());
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    const auto load_result = xmvb::vb::load_cpp_vb_input_with_timings(options.input_path);
    const auto& input = load_result.input;
    const bool use_ri = input.ao_integral_input.ao_two_electron_integral_values.empty();

    xmvb::vb::CppOrbitalGradientEvaluator gradient_evaluator(options.algorithm);
    auto gradient_result = gradient_evaluator.evaluate_without_reference_energy_gradient(
        input,
        load_result.nuclear_repulsion_energy);
    gradient_evaluator.populate_reference_energy_gradient(input, &gradient_result);

    const std::vector<int> differentiable_parameter_indices =
        collect_differentiable_parameter_indices(input.orbital_preparation_input);
    std::vector<std::pair<double, int>> ranked_parameters;
    ranked_parameters.reserve(differentiable_parameter_indices.size());
    for (const int parameter_index : differentiable_parameter_indices) {
      ranked_parameters.emplace_back(
          std::abs(gradient_result.sparse_orbital_reference_energy_gradient[xmvb::to_size(
              parameter_index)]),
          parameter_index);
    }
    std::sort(
        ranked_parameters.begin(),
        ranked_parameters.end(),
        [](const auto& left, const auto& right) {
          if (left.first != right.first) {
            return left.first > right.first;
          }
          return left.second < right.second;
        });

    xmvb::vb::AoEffectiveOneElectronBackpropagator ao_backpropagator;
    xmvb::vb::ActiveSpaceOrbitalBackpropagator orbital_backpropagator;
    std::vector<double> base_inactive_density_gradient(
        gradient_result.orbital_preparation_result.inactive_density_matrix.size(),
        0.0);
    for (std::size_t index = 0; index < base_inactive_density_gradient.size(); ++index) {
      base_inactive_density_gradient[index] =
          gradient_result.ao_effective_one_electron_result.ao_effective_h1e[index] +
          input.ao_integral_input.ao_core_hamiltonian_matrix[index];
    }

    std::vector<double> current_backprop_matrix;
    std::vector<double> custom_backprop_matrix;
    if (use_ri) {
      current_backprop_matrix =
          ao_backpropagator.backpropagate(
              gradient_result.orbital_preparation_result.inactive_density_matrix,
              xmvb::vb::ensure_cpp_vb_input_ri_cache(input),
              input.ao_integral_input.n_basis_functions)
              .inactive_density_gradient;
      custom_backprop_matrix = lower_triangle_half_diagonal(
          current_backprop_matrix,
          input.ao_integral_input.n_basis_functions);
    }

    std::vector<double> current_total_gradient = base_inactive_density_gradient;
    std::vector<double> custom_total_gradient = base_inactive_density_gradient;
    if (use_ri) {
      for (std::size_t index = 0; index < current_total_gradient.size(); ++index) {
        current_total_gradient[index] += current_backprop_matrix[index];
        custom_total_gradient[index] += custom_backprop_matrix[index];
      }
    }

    const std::vector<double> zero_auxiliary_gradient(
        gradient_result.orbital_preparation_result.auxiliary_orbital_matrix.size(),
        0.0);
    const std::vector<double> zero_active_overlap_gradient(
        xmvb::to_size(input.orbital_preparation_input.n_active_orbitals) *
            input.orbital_preparation_input.n_active_orbitals,
        0.0);

    const std::vector<double> custom_reference_gradient =
        use_ri
            ? orbital_backpropagator.backpropagate(
                  zero_auxiliary_gradient,
                  zero_active_overlap_gradient,
                  custom_total_gradient,
                  input.orbital_preparation_input)
                  .orbital_value_gradient
            : gradient_result.sparse_orbital_reference_energy_gradient;

    const int n_to_report =
        std::min(options.count, static_cast<int>(ranked_parameters.size()));
    std::cout << std::setprecision(12);
    std::cout << "use_ri = " << (use_ri ? 1 : 0) << '\n';
    std::cout << "initial_reference_energy = "
              << gradient_result.scf_result.one_electron_reference_energy << '\n';
    std::cout << "reported_parameters = " << n_to_report << '\n';

    for (int report_index = 0; report_index < n_to_report; ++report_index) {
      const int parameter_index = ranked_parameters[xmvb::to_size(report_index)].second;
      xmvb::vb::CppVbInput plus_input = input;
      xmvb::vb::CppVbInput minus_input = input;
      plus_input.orbital_preparation_input.orbital_value_table[xmvb::to_size(
          parameter_index)] += options.step;
      minus_input.orbital_preparation_input.orbital_value_table[xmvb::to_size(
          parameter_index)] -= options.step;
      const double plus_energy =
          evaluate_reference_energy(
              plus_input,
              options.algorithm,
              load_result.nuclear_repulsion_energy);
      const double minus_energy =
          evaluate_reference_energy(
              minus_input,
              options.algorithm,
              load_result.nuclear_repulsion_energy);
      const double finite_difference = (plus_energy - minus_energy) / (2.0 * options.step);

      std::cout << "parameter[" << report_index << "]"
                << " index=" << parameter_index
                << " current=" << gradient_result.sparse_orbital_reference_energy_gradient[
                       xmvb::to_size(parameter_index)]
                << " custom=" << custom_reference_gradient[
                       xmvb::to_size(parameter_index)]
                << " fd=" << finite_difference
                << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
