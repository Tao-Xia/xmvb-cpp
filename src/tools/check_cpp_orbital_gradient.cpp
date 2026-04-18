#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/orbital/nonredundant_optimizer_input_adapter.hpp"
#include "vb/orbital/nonredundant_orbital_space.hpp"
#include "vb/orbital/sparse_orbital_parameter_view.hpp"
#include "vb/scf/cpp_orbital_gradient_evaluator.hpp"
#include "vb/scf/cpp_vb_scf_evaluator.hpp"
#include "vb/vbscf_algorithm.hpp"

namespace {

enum class SelectionMode {
  Top,
  AddedSupport,
};

enum class GradientSpace {
  Sparse,
  Reduced,
};

enum class EnergyComponent {
  Total,
  Reference,
  Nonreference,
};

struct Options {
  std::string input_path;
  std::string orbital_value_table_bin_path;
  xmvb::vb::VBSCFAlgorithm algorithm = xmvb::vb::VBSCFAlgorithm::Original;
  xmvb::vb::AoIntegralSource ao_integral_source =
      xmvb::vb::AoIntegralSource::Auto;
  xmvb::vb::StandardTwoElectronMode standard_two_electron_mode =
      xmvb::vb::StandardTwoElectronMode::Auto;
  EnergyComponent component = EnergyComponent::Total;
  int count = 8;
  double step = 1.0e-6;
  bool nonredundant_adapt = false;
  bool parameter_roundtrip = false;
  SelectionMode selection_mode = SelectionMode::Top;
  GradientSpace gradient_space = GradientSpace::Sparse;
  int orbital_index_begin = 1;
  int orbital_index_end = std::numeric_limits<int>::max();
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

std::vector<int> collect_differentiable_parameter_indices_in_orbital_range(
    const xmvb::vb::OrbitalPreparationInput& orbital_preparation_input,
    int orbital_index_begin,
    int orbital_index_end) {
  if (orbital_index_begin <= 0 || orbital_index_end <= 0) {
    throw std::invalid_argument("--orbital-range indices must be positive");
  }
  if (orbital_index_begin > orbital_index_end) {
    throw std::invalid_argument("--orbital-range begin must not exceed end");
  }

  const int clipped_begin = std::max(1, orbital_index_begin);
  const int clipped_end =
      std::min(orbital_preparation_input.n_orbitals, orbital_index_end);
  std::vector<int> differentiable_parameter_indices;
  if (clipped_begin > clipped_end) {
    return differentiable_parameter_indices;
  }

  for (int orbital_index = clipped_begin - 1;
       orbital_index <= clipped_end - 1;
       ++orbital_index) {
    const int coefficient_count =
        get_sparse_coefficient_count(orbital_preparation_input, orbital_index);
    for (int coefficient_index = 0;
         coefficient_index < coefficient_count;
         ++coefficient_index) {
      differentiable_parameter_indices.push_back(
          orbital_index * orbital_preparation_input.n_basis_functions +
          coefficient_index);
    }
  }
  return differentiable_parameter_indices;
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

SelectionMode parse_selection_mode(const std::string& value) {
  if (value == "top") {
    return SelectionMode::Top;
  }
  if (value == "added") {
    return SelectionMode::AddedSupport;
  }
  throw std::invalid_argument("invalid selection mode: " + value);
}

GradientSpace parse_gradient_space(const std::string& value) {
  if (value == "sparse") {
    return GradientSpace::Sparse;
  }
  if (value == "reduced") {
    return GradientSpace::Reduced;
  }
  throw std::invalid_argument("invalid gradient space: " + value);
}

EnergyComponent parse_energy_component(const std::string& value) {
  if (value == "total") {
    return EnergyComponent::Total;
  }
  if (value == "reference") {
    return EnergyComponent::Reference;
  }
  if (value == "nonreference") {
    return EnergyComponent::Nonreference;
  }
  throw std::invalid_argument("invalid component: " + value);
}

const char* energy_component_name(EnergyComponent component) {
  switch (component) {
    case EnergyComponent::Total:
      return "total";
    case EnergyComponent::Reference:
      return "reference";
    case EnergyComponent::Nonreference:
      return "nonreference";
  }
  return "unknown";
}

std::pair<int, int> parse_orbital_range_argument(const std::string& value) {
  const std::size_t separator = value.find(':');
  if (separator == std::string::npos) {
    throw std::invalid_argument(
        "invalid --orbital-range value, expected begin:end");
  }
  const std::string begin_text = value.substr(0, separator);
  const std::string end_text = value.substr(separator + 1);
  if (begin_text.empty() || end_text.empty()) {
    throw std::invalid_argument(
        "invalid --orbital-range value, expected begin:end");
  }
  return {std::stoi(begin_text), std::stoi(end_text)};
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

std::vector<std::unordered_set<int>> collect_orbital_support_sets(
    const xmvb::vb::OrbitalPreparationInput& orbital_preparation_input) {
  std::vector<std::unordered_set<int>> support_sets(
      xmvb::to_size(orbital_preparation_input.n_orbitals));
  for (int orbital_index = 0;
       orbital_index < orbital_preparation_input.n_orbitals;
       ++orbital_index) {
    const int coefficient_count =
        get_sparse_coefficient_count(orbital_preparation_input, orbital_index);
    auto& support_set = support_sets[xmvb::to_size(orbital_index)];
    support_set.reserve(xmvb::to_size(coefficient_count));
    for (int coefficient_index = 0;
         coefficient_index < coefficient_count;
         ++coefficient_index) {
      const int basis_function_index =
          orbital_preparation_input.orbital_basis_index_table
              [xmvb::to_size(orbital_index) *
                   orbital_preparation_input.n_basis_functions +
               coefficient_index] -
          1;
      if (basis_function_index < 0 ||
          basis_function_index >= orbital_preparation_input.n_basis_functions) {
        throw std::runtime_error("invalid sparse orbital basis index while collecting supports");
      }
      support_set.insert(basis_function_index);
    }
  }
  return support_sets;
}

std::vector<int> collect_added_support_parameter_indices(
    const xmvb::vb::OrbitalPreparationInput& original_input,
    const xmvb::vb::OrbitalPreparationInput& adapted_input) {
  if (original_input.n_orbitals != adapted_input.n_orbitals ||
      original_input.n_basis_functions != adapted_input.n_basis_functions) {
    throw std::invalid_argument(
        "original and adapted orbital inputs must have matching dimensions");
  }

  const auto original_support_sets =
      collect_orbital_support_sets(original_input);
  std::vector<int> parameter_indices;
  for (int orbital_index = 0;
       orbital_index < adapted_input.n_orbitals;
       ++orbital_index) {
    const int coefficient_count =
        get_sparse_coefficient_count(adapted_input, orbital_index);
    const auto& original_support =
        original_support_sets[xmvb::to_size(orbital_index)];
    for (int coefficient_index = 0;
         coefficient_index < coefficient_count;
         ++coefficient_index) {
      const int basis_function_index =
          adapted_input.orbital_basis_index_table
              [xmvb::to_size(orbital_index) *
                   adapted_input.n_basis_functions +
               coefficient_index] -
          1;
      if (basis_function_index < 0 ||
          basis_function_index >= adapted_input.n_basis_functions) {
        throw std::runtime_error(
            "invalid sparse orbital basis index while collecting added supports");
      }
      if (original_support.find(basis_function_index) != original_support.end()) {
        continue;
      }
      parameter_indices.push_back(
          orbital_index * adapted_input.n_basis_functions + coefficient_index);
    }
  }
  return parameter_indices;
}

std::vector<double> read_f64_binary_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open binary input file: " + path);
  }
  input.seekg(0, std::ios::end);
  const std::streamoff byte_size = input.tellg();
  if (byte_size < 0 || (byte_size % static_cast<std::streamoff>(sizeof(double))) != 0) {
    throw std::runtime_error("invalid f64 binary file size: " + path);
  }
  input.seekg(0, std::ios::beg);
  std::vector<double> values(
      static_cast<std::size_t>(byte_size / static_cast<std::streamoff>(sizeof(double))),
      0.0);
  if (!values.empty()) {
    input.read(reinterpret_cast<char*>(values.data()), byte_size);
  }
  if (!input) {
    throw std::runtime_error("failed to read binary input file: " + path);
  }
  return values;
}

void print_usage() {
  std::cerr << "usage: check_cpp_orbital_gradient <input.xmi> "
               "[--orbital-value-table-bin <path>] "
               "[--algorithm original] "
               "[--ao-integral-source auto|legacy|libcint_cpp|runtime_hcore] "
               "[--standard-two-electron-mode auto|exact|ri] "
               "[--component total|reference|nonreference] "
               "[--count N] [--step h] "
               "[--orbital-range begin:end] "
               "[--gradient-space sparse|reduced] "
               "[--nonredundant-adapt true|false] "
               "[--parameter-roundtrip true|false] "
               "[--selection top|added]\n";
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
    if (argument_name == "--orbital-value-table-bin") {
      options.orbital_value_table_bin_path = argument_value;
      continue;
    }
    if (argument_name == "--ao-integral-source") {
      options.ao_integral_source = parse_ao_integral_source(argument_value);
      continue;
    }
    if (argument_name == "--standard-two-electron-mode") {
      if (argument_value == "auto") {
        options.standard_two_electron_mode = xmvb::vb::StandardTwoElectronMode::Auto;
      } else if (argument_value == "exact") {
        options.standard_two_electron_mode = xmvb::vb::StandardTwoElectronMode::Exact;
      } else if (argument_value == "ri") {
        options.standard_two_electron_mode =
            xmvb::vb::StandardTwoElectronMode::ResolutionOfIdentity;
      } else {
        throw std::invalid_argument("invalid standard two-electron mode: " + argument_value);
      }
      continue;
    }
    if (argument_name == "--component") {
      options.component = parse_energy_component(argument_value);
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
    if (argument_name == "--orbital-range") {
      const auto orbital_range =
          parse_orbital_range_argument(argument_value);
      options.orbital_index_begin = orbital_range.first;
      options.orbital_index_end = orbital_range.second;
      continue;
    }
    if (argument_name == "--gradient-space") {
      options.gradient_space = parse_gradient_space(argument_value);
      continue;
    }
    if (argument_name == "--nonredundant-adapt") {
      options.nonredundant_adapt = parse_bool_argument(argument_value);
      continue;
    }
    if (argument_name == "--parameter-roundtrip") {
      options.parameter_roundtrip = parse_bool_argument(argument_value);
      continue;
    }
    if (argument_name == "--selection") {
      options.selection_mode = parse_selection_mode(argument_value);
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
  if (options.selection_mode == SelectionMode::AddedSupport &&
      !options.nonredundant_adapt) {
    throw std::invalid_argument("--selection added requires --nonredundant-adapt true");
  }
  if (options.selection_mode == SelectionMode::AddedSupport &&
      options.gradient_space != GradientSpace::Sparse) {
    throw std::invalid_argument("--selection added is only supported for --gradient-space sparse");
  }
  return options;
}

double evaluate_energy_component(
    const xmvb::vb::CppVbInput& input,
    xmvb::vb::VBSCFAlgorithm algorithm,
    double nuclear_repulsion_energy,
    EnergyComponent component) {
  xmvb::vb::CppVbScfEvaluator evaluator(algorithm);
  const auto result = evaluator.evaluate(input, nuclear_repulsion_energy);
  switch (component) {
    case EnergyComponent::Total:
      return result.total_energy;
    case EnergyComponent::Reference:
      return result.one_electron_reference_energy;
    case EnergyComponent::Nonreference:
      return result.total_energy - result.one_electron_reference_energy;
  }
  throw std::invalid_argument("unsupported energy component");
}

std::vector<double> build_selected_sparse_gradient(
    const xmvb::vb::CppOrbitalGradientResult& gradient_result,
    EnergyComponent component) {
  switch (component) {
    case EnergyComponent::Total:
      return gradient_result.sparse_orbital_energy_gradient;
    case EnergyComponent::Reference:
      return gradient_result.sparse_orbital_reference_energy_gradient;
    case EnergyComponent::Nonreference: {
      if (gradient_result.sparse_orbital_energy_gradient.size() !=
          gradient_result.sparse_orbital_reference_energy_gradient.size()) {
        throw std::runtime_error(
            "total and reference orbital gradient sizes differ while building nonreference component");
      }
      std::vector<double> nonreference_gradient =
          gradient_result.sparse_orbital_energy_gradient;
      for (std::size_t index = 0; index < nonreference_gradient.size(); ++index) {
        nonreference_gradient[index] -=
            gradient_result.sparse_orbital_reference_energy_gradient[index];
      }
      return nonreference_gradient;
    }
  }
  throw std::invalid_argument("unsupported gradient component");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    xmvb::vb::CppVbInputLoadOptions load_options;
    load_options.ao_integral_source = options.ao_integral_source;
    load_options.standard_two_electron_mode = options.standard_two_electron_mode;
    const auto load_result =
        xmvb::vb::load_cpp_vb_input_with_timings(options.input_path, load_options);
    const xmvb::vb::CppVbInput diagnostic_input_template =
        options.nonredundant_adapt
            ? xmvb::vb::build_nonredundant_optimizer_input(load_result.input)
            : load_result.input;
    xmvb::vb::CppVbInput diagnostic_input = diagnostic_input_template;
    if (!options.orbital_value_table_bin_path.empty()) {
      diagnostic_input.orbital_preparation_input.orbital_value_table =
          read_f64_binary_file(options.orbital_value_table_bin_path);
      if (diagnostic_input.orbital_preparation_input.orbital_value_table.size() !=
          diagnostic_input_template.orbital_preparation_input.orbital_value_table.size()) {
        throw std::runtime_error(
            "orbital_value_table override size does not match diagnostic input");
      }
    }
    if (options.parameter_roundtrip) {
      xmvb::vb::SparseOrbitalParameterView parameter_view(
          diagnostic_input.orbital_preparation_input);
      const Eigen::VectorXd packed_parameters =
          parameter_view.pack(diagnostic_input.orbital_preparation_input);
      parameter_view.unpack(
          packed_parameters,
          &diagnostic_input.orbital_preparation_input);
    }

    xmvb::vb::CppOrbitalGradientEvaluator gradient_evaluator(options.algorithm);
    auto gradient_result =
        gradient_evaluator.evaluate(diagnostic_input, load_result.nuclear_repulsion_energy);
    if (options.component != EnergyComponent::Total) {
      gradient_evaluator.populate_reference_energy_gradient(
          diagnostic_input,
          &gradient_result);
    }
    const std::vector<double> selected_sparse_gradient =
        build_selected_sparse_gradient(gradient_result, options.component);
    const double initial_component_energy = evaluate_energy_component(
        diagnostic_input,
        options.algorithm,
        load_result.nuclear_repulsion_energy,
        options.component);
    std::cout << std::setprecision(12);
    std::cout << "algorithm = " << xmvb::vb::vb_scf_algorithm_name(options.algorithm) << '\n';
    std::cout << "ao_integral_source = "
              << xmvb::vb::ao_integral_source_name(load_result.ao_integral_source) << '\n';
    std::cout << "orbital_value_table_override = "
              << (options.orbital_value_table_bin_path.empty()
                      ? "none"
                      : options.orbital_value_table_bin_path)
              << '\n';
    std::cout << "standard_two_electron_mode = "
              << xmvb::vb::standard_two_electron_mode_name(
                     load_result.standard_two_electron_mode)
              << '\n';
    std::cout << "gradient_space = "
              << (options.gradient_space == GradientSpace::Reduced ? "reduced" : "sparse")
              << '\n';
    std::cout << "nonredundant_adapt = "
              << (options.nonredundant_adapt ? "true" : "false") << '\n';
    std::cout << "parameter_roundtrip = "
              << (options.parameter_roundtrip ? "true" : "false") << '\n';
    std::cout << "selection_mode = "
              << (options.selection_mode == SelectionMode::AddedSupport
                      ? "added"
                      : "top")
              << '\n';
    std::cout << "component = " << energy_component_name(options.component) << '\n';
    std::cout << "orbital_range = "
              << options.orbital_index_begin << ':'
              << options.orbital_index_end << '\n';
    std::cout << "initial_total_energy = " << gradient_result.scf_result.total_energy << '\n';
    std::cout << "initial_component_energy = " << initial_component_energy << '\n';
    std::cout << "finite_difference_step = " << options.step << '\n';
    if (options.gradient_space == GradientSpace::Sparse) {
      std::vector<int> differentiable_parameter_indices =
          options.selection_mode == SelectionMode::AddedSupport
              ? collect_added_support_parameter_indices(
                    load_result.input.orbital_preparation_input,
                    diagnostic_input.orbital_preparation_input)
              : collect_differentiable_parameter_indices_in_orbital_range(
                    diagnostic_input.orbital_preparation_input,
                    options.orbital_index_begin,
                    options.orbital_index_end);
      if (selected_sparse_gradient.size() !=
          diagnostic_input.orbital_preparation_input.orbital_value_table.size()) {
        throw std::runtime_error("unexpected sparse orbital gradient size");
      }
      if (differentiable_parameter_indices.empty()) {
        throw std::runtime_error(
            "no differentiable parameters found in the requested orbital range");
      }

      std::vector<std::pair<double, int>> ranked_parameters;
      ranked_parameters.reserve(differentiable_parameter_indices.size());
      for (const int parameter_index : differentiable_parameter_indices) {
        ranked_parameters.emplace_back(
            std::abs(selected_sparse_gradient[xmvb::to_size(parameter_index)]),
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
      double gradient_inf_norm = 0.0;
      for (const int parameter_index : differentiable_parameter_indices) {
        gradient_inf_norm = std::max(
            gradient_inf_norm,
            std::abs(selected_sparse_gradient[xmvb::to_size(parameter_index)]));
      }
      std::cout << "analytic_gradient_inf_norm = " << gradient_inf_norm << '\n';
      std::cout << "reported_parameters = " << n_to_report << '\n';

      for (int report_index = 0; report_index < n_to_report; ++report_index) {
        const int parameter_index = ranked_parameters[xmvb::to_size(report_index)].second;
        xmvb::vb::CppVbInput plus_input = diagnostic_input;
        xmvb::vb::CppVbInput minus_input = diagnostic_input;
        plus_input.orbital_preparation_input.orbital_value_table[xmvb::to_size(parameter_index)] +=
            options.step;
        minus_input.orbital_preparation_input.orbital_value_table[xmvb::to_size(parameter_index)] -=
            options.step;

        const double plus_energy = evaluate_energy_component(
            plus_input,
            options.algorithm,
            load_result.nuclear_repulsion_energy,
            options.component);
        const double minus_energy = evaluate_energy_component(
            minus_input,
            options.algorithm,
            load_result.nuclear_repulsion_energy,
            options.component);
        const double finite_difference = (plus_energy - minus_energy) / (2.0 * options.step);
        const double analytic = selected_sparse_gradient[xmvb::to_size(parameter_index)];
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
    } else {
      xmvb::vb::SparseOrbitalParameterView parameter_view(
          diagnostic_input.orbital_preparation_input);
      const Eigen::VectorXd packed_parameters =
          parameter_view.pack(diagnostic_input.orbital_preparation_input);
      const Eigen::VectorXd packed_gradient =
          parameter_view.gather_from_full(selected_sparse_gradient);
      const int n_inactive =
          (diagnostic_input.orbital_preparation_input.n_total_electrons -
           diagnostic_input.orbital_preparation_input.n_active_electrons) / 2;
      const int n_occupied =
          n_inactive +
          diagnostic_input.orbital_preparation_input.n_active_orbitals;
      xmvb::vb::NonredundantOrbitalSpace nonredundant_space(
          diagnostic_input.orbital_preparation_input,
          parameter_view,
          gradient_result.orbital_preparation_result.auxiliary_orbital_matrix.leftCols(
              std::max(0, n_occupied)),
          gradient_result.orbital_preparation_result
              .physical_orbital_frame
              .normalized_orbital_matrix,
          &gradient_result.ao_effective_one_electron_result.ao_effective_h1e);
      const auto projection = nonredundant_space.project_gradient(packed_gradient);
      if (projection.reduced_gradient.size() == 0) {
        throw std::runtime_error("reduced nonredundant space is empty");
      }

      std::vector<std::pair<double, int>> ranked_directions;
      ranked_directions.reserve(xmvb::to_size(projection.reduced_gradient.size()));
      for (Eigen::Index reduced_index = 0;
           reduced_index < projection.reduced_gradient.size();
           ++reduced_index) {
        ranked_directions.emplace_back(
            std::abs(projection.reduced_gradient[reduced_index]),
            static_cast<int>(reduced_index));
      }
      std::sort(
          ranked_directions.begin(),
          ranked_directions.end(),
          [](const auto& left, const auto& right) {
            if (left.first != right.first) {
              return left.first > right.first;
            }
            return left.second < right.second;
          });

      const int n_to_report =
          std::min(options.count, static_cast<int>(ranked_directions.size()));
      double gradient_inf_norm = 0.0;
      for (Eigen::Index reduced_index = 0;
           reduced_index < projection.reduced_gradient.size();
           ++reduced_index) {
        gradient_inf_norm = std::max(
            gradient_inf_norm,
            std::abs(projection.reduced_gradient[reduced_index]));
      }
      std::cout << "reduced_dimension = " << projection.reduced_gradient.size() << '\n';
      std::cout << "analytic_gradient_inf_norm = " << gradient_inf_norm << '\n';
      std::cout << "reported_parameters = " << n_to_report << '\n';

      for (int report_index = 0; report_index < n_to_report; ++report_index) {
        const int reduced_index = ranked_directions[xmvb::to_size(report_index)].second;
        Eigen::VectorXd reduced_direction =
            Eigen::VectorXd::Zero(projection.reduced_gradient.size());
        reduced_direction[reduced_index] = 1.0;
        const Eigen::VectorXd packed_direction =
            nonredundant_space.expand_step(reduced_direction);
        // `projection.reduced_gradient[reduced_index]` is the derivative with
        // respect to this reduced coordinate itself, so probe the finite
        // difference directly in the reduced chart instead of re-normalizing
        // the direction by some packed-space norm.
        const double effective_step = options.step;

        xmvb::vb::CppVbInput plus_input = diagnostic_input;
        xmvb::vb::CppVbInput minus_input = diagnostic_input;
        // Reduced nonredundant coordinates represent finite occupied/virtual
        // rotations inside each support block. Mirror the optimizer manifold by
        // applying the same Cayley-style block retraction here instead of a
        // linear packed-parameter perturbation.
        plus_input.orbital_preparation_input =
            nonredundant_space.retract_step(
                diagnostic_input.orbital_preparation_input,
                reduced_direction,
                effective_step);
        minus_input.orbital_preparation_input =
            nonredundant_space.retract_step(
                diagnostic_input.orbital_preparation_input,
                reduced_direction,
                -effective_step);

        const double plus_energy = evaluate_energy_component(
            plus_input,
            options.algorithm,
            load_result.nuclear_repulsion_energy,
            options.component);
        const double minus_energy = evaluate_energy_component(
            minus_input,
            options.algorithm,
            load_result.nuclear_repulsion_energy,
            options.component);
        const double finite_difference = (plus_energy - minus_energy) / (2.0 * effective_step);
        const double analytic = projection.reduced_gradient[reduced_index];
        const double absolute_error = std::abs(analytic - finite_difference);
        const double relative_error =
            absolute_error / std::max(1.0, std::abs(finite_difference));

        std::cout << "parameter[" << report_index << "]"
                  << " index=" << reduced_index
                  << " packed_direction_norm=" << packed_direction.norm()
                  << " packed_direction_max_abs=" << packed_direction.cwiseAbs().maxCoeff()
                  << " analytic=" << analytic
                  << " fd=" << finite_difference
                  << " abs_error=" << absolute_error
                  << " rel_error=" << relative_error
                  << '\n';
      }
    }

    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
