#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_exact_selected_structure_gradient.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_exact_selected_structure_scf.hpp"
#include "vb/vbscf_algorithm.hpp"

namespace {

struct Options {
  std::string input_path;
  int subspace_size = 0;
  int count = 8;
  double step = 1.0e-5;
  double tolerance = 1.0e-4;
  xmvb::vb::VBSCFAlgorithm algorithm = xmvb::vb::VBSCFAlgorithm::Original;
};

int get_sparse_coefficient_count(
    const xmvb::vb::OrbitalPreparationInput& orbital_preparation_input,
    int orbital_index) {
  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  const bool have_original_counts =
      orbital_preparation_input.original_orbital_basis_counts.size() ==
      xmvb::to_size(orbital_preparation_input.n_orbitals);
  const int explicit_count = have_original_counts
      ? orbital_preparation_input.original_orbital_basis_counts[xmvb::to_size(orbital_index)]
      : orbital_preparation_input.orbital_basis_counts[xmvb::to_size(orbital_index)];
  if (explicit_count > 1) {
    return explicit_count;
  }
  if (explicit_count == 1) {
    return 1;
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
  std::cerr
      << "usage: check_biorthogonal_exact_selected_orbital_gradient <input.xmi> "
         "[--subspace-size N] [--count N] [--step h] [--tolerance t]\n";
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
    if (argument_name == "--subspace-size") {
      options.subspace_size = std::stoi(argument_value);
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
    if (argument_name == "--tolerance") {
      options.tolerance = std::stod(argument_value);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + argument_name);
  }

  if (options.subspace_size < 0) {
    throw std::invalid_argument("--subspace-size must be non-negative");
  }
  if (options.count <= 0) {
    throw std::invalid_argument("--count must be positive");
  }
  if (options.step <= 0.0) {
    throw std::invalid_argument("--step must be positive");
  }
  if (options.tolerance < 0.0) {
    throw std::invalid_argument("--tolerance must be non-negative");
  }
  return options;
}

std::vector<int> build_selected_structure_indices(
    int n_structures,
    int subspace_size) {
  if (n_structures <= 0) {
    throw std::invalid_argument("n_structures must be positive");
  }
  const int effective_size =
      subspace_size == 0 ? n_structures : std::min(subspace_size, n_structures);
  std::vector<int> selected_structure_indices;
  selected_structure_indices.reserve(xmvb::to_size(effective_size));
  for (int structure_index = 0; structure_index < effective_size; ++structure_index) {
    selected_structure_indices.push_back(structure_index);
  }
  return selected_structure_indices;
}

std::string format_indices(const std::vector<int>& indices) {
  std::string result = "{";
  for (std::size_t index = 0; index < indices.size(); ++index) {
    if (index > 0) {
      result += ",";
    }
    result += std::to_string(indices[index]);
  }
  result += "}";
  return result;
}

double evaluate_exact_selected_total_energy(
    const xmvb::vb::CppVbInput& input,
    const std::vector<int>& selected_structure_indices,
    double nuclear_repulsion_energy) {
  return xmvb::vb::biorthogonal_vbscf::
      evaluate_biorthogonal_exact_selected_structure_scf(
          input,
          selected_structure_indices,
          nuclear_repulsion_energy)
          .total_energy;
}

struct OrbitalGradientErrorSummary {
  double total_worst_abs_error = 0.0;
  double electronic_worst_abs_error = 0.0;
  double reference_worst_abs_error = 0.0;
};

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    xmvb::vb::CppVbInputLoadOptions load_options;
    load_options.ao_integral_source = xmvb::vb::AoIntegralSource::LibcintMaterializedCpp;
    load_options.orbital_guess_source = xmvb::vb::OrbitalGuessSource::Cpp;
    load_options.standard_two_electron_mode =
        xmvb::vb::StandardTwoElectronMode::Exact;
    const auto load_result =
        xmvb::vb::load_cpp_vb_input_with_timings(options.input_path, load_options);
    const std::vector<int> selected_structure_indices =
        build_selected_structure_indices(
            load_result.input.structure_data.n_structures,
            options.subspace_size);
    const auto gradient_result =
        xmvb::vb::biorthogonal_vbscf::
            evaluate_biorthogonal_exact_selected_structure_orbital_gradient(
                load_result.input,
                selected_structure_indices,
                options.algorithm,
                load_result.nuclear_repulsion_energy);

    if (gradient_result.sparse_orbital_energy_gradient.size() !=
        load_result.input.orbital_preparation_input.orbital_value_table.size()) {
      throw std::runtime_error("unexpected sparse orbital gradient size");
    }
    const std::vector<int> differentiable_parameter_indices =
        collect_differentiable_parameter_indices(
            load_result.input.orbital_preparation_input);
    std::vector<std::pair<double, int>> ranked_parameters;
    ranked_parameters.reserve(differentiable_parameter_indices.size());
    double analytic_gradient_inf_norm = 0.0;
    for (const int parameter_index : differentiable_parameter_indices) {
      const double gradient_value =
          gradient_result.sparse_orbital_energy_gradient[xmvb::to_size(parameter_index)];
      analytic_gradient_inf_norm =
          std::max(analytic_gradient_inf_norm, std::abs(gradient_value));
      ranked_parameters.emplace_back(std::abs(gradient_value), parameter_index);
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
    OrbitalGradientErrorSummary error_summary;
    std::cout << std::setprecision(12);
    std::cout << "algorithm = " << xmvb::vb::vb_scf_algorithm_name(options.algorithm) << '\n';
    std::cout << "selected_structure_indices = "
              << format_indices(selected_structure_indices) << '\n';
    std::cout << "subspace_size = " << selected_structure_indices.size() << '\n';
    std::cout << "initial_total_energy = " << gradient_result.scf_result.total_energy << '\n';
    std::cout << "analytic_gradient_inf_norm = " << analytic_gradient_inf_norm << '\n';
    std::cout << "finite_difference_step = " << options.step << '\n';
    std::cout << "reported_parameters = " << n_to_report << '\n';

    for (int report_index = 0; report_index < n_to_report; ++report_index) {
      const int parameter_index = ranked_parameters[xmvb::to_size(report_index)].second;
      xmvb::vb::CppVbInput plus_input = load_result.input;
      xmvb::vb::CppVbInput minus_input = load_result.input;
      plus_input.orbital_preparation_input.orbital_value_table[xmvb::to_size(parameter_index)] +=
          options.step;
      minus_input.orbital_preparation_input.orbital_value_table[xmvb::to_size(parameter_index)] -=
          options.step;

      const double plus_energy = evaluate_exact_selected_total_energy(
          plus_input,
          selected_structure_indices,
          load_result.nuclear_repulsion_energy);
      const double minus_energy = evaluate_exact_selected_total_energy(
          minus_input,
          selected_structure_indices,
          load_result.nuclear_repulsion_energy);
      const auto plus_scf_result =
          xmvb::vb::biorthogonal_vbscf::
              evaluate_biorthogonal_exact_selected_structure_scf(
                  plus_input,
                  selected_structure_indices,
                  load_result.nuclear_repulsion_energy);
      const auto minus_scf_result =
          xmvb::vb::biorthogonal_vbscf::
              evaluate_biorthogonal_exact_selected_structure_scf(
                  minus_input,
                  selected_structure_indices,
                  load_result.nuclear_repulsion_energy);
      const double total_finite_difference =
          (plus_energy - minus_energy) / (2.0 * options.step);
      const double electronic_finite_difference =
          (plus_scf_result.electronic_energy - minus_scf_result.electronic_energy) /
          (2.0 * options.step);
      const double reference_finite_difference =
          (plus_scf_result.one_electron_reference_energy -
           minus_scf_result.one_electron_reference_energy) /
          (2.0 * options.step);
      const double total_analytic =
          gradient_result.sparse_orbital_energy_gradient[xmvb::to_size(parameter_index)];
      const double reference_analytic =
          gradient_result.sparse_orbital_reference_energy_gradient[xmvb::to_size(
              parameter_index)];
      const double electronic_analytic =
          total_analytic - reference_analytic;
      const double total_absolute_error =
          std::abs(total_analytic - total_finite_difference);
      const double electronic_absolute_error =
          std::abs(electronic_analytic - electronic_finite_difference);
      const double reference_absolute_error =
          std::abs(reference_analytic - reference_finite_difference);
      const double total_relative_error =
          total_absolute_error / std::max(1.0, std::abs(total_finite_difference));
      error_summary.total_worst_abs_error =
          std::max(error_summary.total_worst_abs_error, total_absolute_error);
      error_summary.electronic_worst_abs_error =
          std::max(error_summary.electronic_worst_abs_error, electronic_absolute_error);
      error_summary.reference_worst_abs_error =
          std::max(error_summary.reference_worst_abs_error, reference_absolute_error);

      std::cout << "parameter[" << report_index << "]"
                << " index=" << parameter_index
                << " analytic=" << total_analytic
                << " fd=" << total_finite_difference
                << " abs_error=" << total_absolute_error
                << " rel_error=" << total_relative_error
                << " electronic_analytic=" << electronic_analytic
                << " electronic_fd=" << electronic_finite_difference
                << " electronic_abs_error=" << electronic_absolute_error
                << " reference_analytic=" << reference_analytic
                << " reference_fd=" << reference_finite_difference
                << " reference_abs_error=" << reference_absolute_error
                << '\n';
    }

    std::cout << "total_worst_abs_error = " << error_summary.total_worst_abs_error << '\n';
    std::cout << "electronic_worst_abs_error = "
              << error_summary.electronic_worst_abs_error << '\n';
    std::cout << "reference_worst_abs_error = "
              << error_summary.reference_worst_abs_error << '\n';
    std::cout << "worst_abs_error = " << error_summary.total_worst_abs_error << '\n';
    if (error_summary.total_worst_abs_error > options.tolerance) {
      std::cerr << "worst_abs_error exceeds tolerance\n";
      return 1;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
