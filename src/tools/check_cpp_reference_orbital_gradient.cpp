#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/scf/cpp_orbital_gradient_evaluator.hpp"
#include "vb/scf/cpp_vb_scf_evaluator.hpp"
#include "vb/vbscf_algorithm.hpp"

namespace {

struct Options {
  std::string input_path;
  xmvb::vb::VBSCFAlgorithm algorithm = xmvb::vb::VBSCFAlgorithm::Original;
  int count = 8;
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
  for (int orbital_index = 0;
       orbital_index < orbital_preparation_input.n_orbitals;
       ++orbital_index) {
    const int coefficient_count =
        get_sparse_coefficient_count(orbital_preparation_input, orbital_index);
    for (int coefficient_index = 0;
         coefficient_index < coefficient_count;
         ++coefficient_index) {
      differentiable_parameter_indices.push_back(
          orbital_index * orbital_preparation_input.n_basis_functions + coefficient_index);
    }
  }
  return differentiable_parameter_indices;
}

void print_usage() {
  std::cerr << "usage: check_cpp_reference_orbital_gradient <input.xmi> "
               "[--algorithm original] "
               "[--count N] [--step h]\n";
}

Options parse_arguments(int argc, char** argv) {
  if (argc < 2 || ((argc - 2) % 2 != 0)) {
    print_usage();
    throw std::invalid_argument("invalid argument count");
  }

  Options options;
  options.input_path = argv[1];
  for (int argument_index = 2; argument_index < argc; argument_index += 2) {
    const std::string argument_name = argv[argument_index];
    const std::string argument_value = argv[argument_index + 1];
    if (argument_name == "--algorithm") {
      if (argument_value == "original") {
        options.algorithm = xmvb::vb::VBSCFAlgorithm::Original;
      } else {
        throw std::invalid_argument("invalid algorithm: " + argument_value);
      }
      continue;
    }
    if (argument_name == "--count") {
      options.count = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--step") {
      options.step = std::stod(argument_value);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + argument_name);
  }

  if (options.count <= 0) {
    throw std::invalid_argument("--count must be positive");
  }
  if (options.step <= 0.0) {
    throw std::invalid_argument("--step must be positive");
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

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    xmvb::vb::CppVbInputLoadOptions load_options;
    // Match the production legacy guess path so open-shell `$GUS` test cases
    // reach the reference-gradient check instead of failing in the partial C++
    // read-guess parser.
    load_options.orbital_guess_source = xmvb::vb::OrbitalGuessSource::LegacyRuntime;
    const auto load_result =
        xmvb::vb::load_cpp_vb_input_with_timings(options.input_path, load_options);

    xmvb::vb::CppOrbitalGradientEvaluator gradient_evaluator(options.algorithm);
    auto gradient_result = gradient_evaluator.evaluate_without_reference_energy_gradient(
        load_result.input,
        load_result.nuclear_repulsion_energy);
    gradient_evaluator.populate_reference_energy_gradient(
        load_result.input,
        &gradient_result);

    const std::vector<int> differentiable_parameter_indices =
        collect_differentiable_parameter_indices(load_result.input.orbital_preparation_input);
    if (gradient_result.sparse_orbital_reference_energy_gradient.size() !=
        load_result.input.orbital_preparation_input.orbital_value_table.size()) {
      throw std::runtime_error("unexpected reference orbital gradient size");
    }

    std::vector<std::pair<double, int>> ranked_parameters;
    ranked_parameters.reserve(differentiable_parameter_indices.size());
    for (const int parameter_index : differentiable_parameter_indices) {
      ranked_parameters.emplace_back(
          std::abs(
              gradient_result.sparse_orbital_reference_energy_gradient[xmvb::to_size(
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

    const int n_to_report =
        std::min(options.count, static_cast<int>(ranked_parameters.size()));
    std::cout << std::setprecision(12);
    std::cout << "algorithm = " << xmvb::vb::vb_scf_algorithm_name(options.algorithm) << '\n';
    std::cout << "initial_reference_energy = "
              << gradient_result.scf_result.one_electron_reference_energy << '\n';
    std::cout << "analytic_reference_gradient_inf_norm = ";
    double gradient_inf_norm = 0.0;
    for (const int parameter_index : differentiable_parameter_indices) {
      gradient_inf_norm = std::max(
          gradient_inf_norm,
          std::abs(gradient_result.sparse_orbital_reference_energy_gradient[xmvb::to_size(
              parameter_index)]));
    }
    std::cout << gradient_inf_norm << '\n';
    std::cout << "finite_difference_step = " << options.step << '\n';
    std::cout << "reported_parameters = " << n_to_report << '\n';

    for (int report_index = 0; report_index < n_to_report; ++report_index) {
      const int parameter_index = ranked_parameters[xmvb::to_size(report_index)].second;
      xmvb::vb::CppVbInput plus_input = load_result.input;
      xmvb::vb::CppVbInput minus_input = load_result.input;
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
      const double analytic =
          gradient_result.sparse_orbital_reference_energy_gradient[xmvb::to_size(
              parameter_index)];
      const double absolute_error = std::abs(analytic - finite_difference);
      const double relative_error =
          absolute_error / std::max(1.0, std::abs(finite_difference));

      std::cout << "parameter[" << report_index << "]"
                << " index=" << parameter_index
                << " analytic=" << analytic
                << " fd=" << finite_difference
                << " abs_error=" << absolute_error
                << " rel_error=" << relative_error
                << '\n';
    }

    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
