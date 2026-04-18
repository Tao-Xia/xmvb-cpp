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
#include "vb/matrices/prepared_active_space_context.hpp"
#include "vb/orbital/active_space_two_electron_utils.hpp"
#include "vb/scf/cpp_active_space_gradient_result.hpp"
#include "vb/vbscf_algorithm.hpp"

namespace {

enum class Component {
  Overlap,
  OneElectron,
  TwoElectron,
};

struct Options {
  std::string input_path;
  int subspace_size = 0;
  int count = 8;
  double step = 1.0e-6;
  double tolerance = 1.0e-4;
  Component component = Component::Overlap;
  xmvb::vb::VBSCFAlgorithm algorithm = xmvb::vb::VBSCFAlgorithm::Original;
};

struct ProbeEntry {
  double analytic = 0.0;
  int primary_index = 0;
  int mirror_index = 0;
};

void print_usage() {
  std::cerr
      << "usage: check_biorthogonal_exact_selected_active_space_gradient <input.xmi> "
         "[--subspace-size N] [--component overlap|one_electron|two_electron] "
         "[--count N] [--step h] [--tolerance t]\n";
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
    if (argument_name == "--component") {
      if (argument_value == "overlap") {
        options.component = Component::Overlap;
      } else if (argument_value == "one_electron") {
        options.component = Component::OneElectron;
      } else if (argument_value == "two_electron") {
        options.component = Component::TwoElectron;
      } else {
        throw std::invalid_argument("invalid component: " + argument_value);
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

const std::vector<double>& component_gradient(
    const xmvb::vb::CppActiveSpaceGradientResult& result,
    Component component) {
  switch (component) {
    case Component::Overlap:
      return result.active_orbital_overlap_gradient;
    case Component::OneElectron:
      return result.active_one_electron_gradient;
    case Component::TwoElectron:
      return result.packed_active_two_electron_gradient;
  }
  throw std::invalid_argument("unknown component");
}

const char* component_name(Component component) {
  switch (component) {
    case Component::Overlap:
      return "overlap";
    case Component::OneElectron:
      return "one_electron";
    case Component::TwoElectron:
      return "two_electron";
  }
  return "unknown";
}

std::vector<ProbeEntry> build_probe_entries(
    const std::vector<double>& gradient,
    Component component,
    int n_active_orbitals) {
  std::vector<ProbeEntry> probe_entries;
  if (component == Component::TwoElectron) {
    probe_entries.reserve(gradient.size());
    for (std::size_t index = 0; index < gradient.size(); ++index) {
      probe_entries.push_back({
          .analytic = gradient[index],
          .primary_index = static_cast<int>(index),
          .mirror_index = static_cast<int>(index),
      });
    }
    return probe_entries;
  }

  const int matrix_size = n_active_orbitals * n_active_orbitals;
  if (static_cast<int>(gradient.size()) != matrix_size) {
    throw std::invalid_argument("matrix gradient size does not match n_active_orbitals");
  }

  probe_entries.reserve(xmvb::to_size(n_active_orbitals) * (n_active_orbitals + 1) / 2);
  for (int column = 0; column < n_active_orbitals; ++column) {
    for (int row = 0; row <= column; ++row) {
      const int primary_index = column * n_active_orbitals + row;
      const int mirror_index = row * n_active_orbitals + column;
      const double analytic =
          primary_index == mirror_index
              ? gradient[xmvb::to_size(primary_index)]
              : gradient[xmvb::to_size(primary_index)] +
                    gradient[xmvb::to_size(mirror_index)];
      probe_entries.push_back({
          .analytic = analytic,
          .primary_index = primary_index,
          .mirror_index = mirror_index,
      });
    }
  }
  return probe_entries;
}

xmvb::vb::PreparedActiveSpaceContext build_prepared_active_space_context(
    const xmvb::vb::CppVbInput& input,
    const xmvb::vb::CppActiveSpaceGradientResult& gradient_result,
    const std::vector<double>& active_orbital_overlap_matrix,
    const std::vector<double>& active_one_electron_integrals,
    const std::vector<double>& packed_active_two_electron_integrals) {
  xmvb::vb::PreparedActiveSpaceContext prepared_active_space;
  prepared_active_space.n_inactive_doubly_occupied_orbitals =
      (input.orbital_preparation_input.n_total_electrons -
       input.orbital_preparation_input.n_active_electrons) /
      2;
  prepared_active_space.orbital_result = gradient_result.orbital_preparation_result;
  prepared_active_space.orbital_result.active_orbital_overlap_matrix =
      active_orbital_overlap_matrix;
  prepared_active_space.ao_effective_one_electron_result =
      gradient_result.ao_effective_one_electron_result;
  prepared_active_space.active_space_one_electron_result.h1e_act =
      active_one_electron_integrals;
  prepared_active_space.active_space_two_electron_result =
      gradient_result.active_space_two_electron_result;
  prepared_active_space.active_space_two_electron_result
      .packed_active_two_electron_integrals =
          packed_active_two_electron_integrals;
  prepared_active_space.one_electron_reference_energy =
      gradient_result.scf_result.one_electron_reference_energy;
  return prepared_active_space;
}

double evaluate_exact_selected_total_energy_from_active_space(
    const xmvb::vb::CppVbInput& input,
    const xmvb::vb::CppActiveSpaceGradientResult& gradient_result,
    const std::vector<int>& selected_structure_indices,
    const std::vector<double>& active_orbital_overlap_matrix,
    const std::vector<double>& active_one_electron_integrals,
    const std::vector<double>& packed_active_two_electron_integrals,
    double nuclear_repulsion_energy) {
  // Hold all orbital/AO-level intermediates fixed and perturb only the active
  // tensors `SSO/HHO/GGO`, so the finite-difference probe matches the exact
  // selected-space active-space objective differentiated by the wrapper.
  const xmvb::vb::PreparedActiveSpaceContext prepared_active_space =
      build_prepared_active_space_context(
          input,
          gradient_result,
          active_orbital_overlap_matrix,
          active_one_electron_integrals,
          packed_active_two_electron_integrals);
  return xmvb::vb::biorthogonal_vbscf::
      evaluate_biorthogonal_exact_selected_structure_scf(
          input,
          prepared_active_space,
          selected_structure_indices,
          nuclear_repulsion_energy)
          .total_energy;
}

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
            evaluate_biorthogonal_exact_selected_structure_active_space_gradient(
                load_result.input,
                selected_structure_indices,
                options.algorithm,
                load_result.nuclear_repulsion_energy);
    const std::vector<double>& gradient =
        component_gradient(gradient_result, options.component);
    const std::vector<ProbeEntry> probe_entries =
        build_probe_entries(
            gradient,
            options.component,
            load_result.input.orbital_preparation_input.n_active_orbitals);

    const std::vector<double> baseline_packed_two =
        gradient_result.active_space_two_electron_result.packed_active_two_electron_integrals.empty()
            ? xmvb::vb::reconstruct_packed_active_two_electron_integrals(
                  xmvb::vb::make_active_space_two_electron_view(
                      gradient_result.active_space_two_electron_result),
                  load_result.input.orbital_preparation_input.n_active_orbitals)
            : gradient_result.active_space_two_electron_result
                  .packed_active_two_electron_integrals;

    std::vector<std::pair<double, int>> ranked_entries;
    ranked_entries.reserve(probe_entries.size());
    double analytic_gradient_inf_norm = 0.0;
    for (std::size_t index = 0; index < probe_entries.size(); ++index) {
      analytic_gradient_inf_norm =
          std::max(analytic_gradient_inf_norm, std::abs(probe_entries[index].analytic));
      ranked_entries.emplace_back(std::abs(probe_entries[index].analytic), static_cast<int>(index));
    }
    std::sort(
        ranked_entries.begin(),
        ranked_entries.end(),
        [](const auto& left, const auto& right) {
          if (left.first != right.first) {
            return left.first > right.first;
          }
          return left.second < right.second;
        });

    const int n_to_report =
        std::min(options.count, static_cast<int>(ranked_entries.size()));
    double worst_abs_error = 0.0;
    std::cout << std::setprecision(12);
    std::cout << "algorithm = " << xmvb::vb::vb_scf_algorithm_name(options.algorithm) << '\n';
    std::cout << "component = " << component_name(options.component) << '\n';
    std::cout << "selected_structure_indices = "
              << format_indices(selected_structure_indices) << '\n';
    std::cout << "subspace_size = " << selected_structure_indices.size() << '\n';
    std::cout << "initial_total_energy = " << gradient_result.scf_result.total_energy << '\n';
    std::cout << "analytic_gradient_inf_norm = " << analytic_gradient_inf_norm << '\n';
    std::cout << "finite_difference_step = " << options.step << '\n';
    std::cout << "reported_entries = " << n_to_report << '\n';

    for (int report_index = 0; report_index < n_to_report; ++report_index) {
      const ProbeEntry& probe_entry =
          probe_entries[xmvb::to_size(ranked_entries[xmvb::to_size(report_index)].second)];
      std::vector<double> plus_overlap = gradient_result.active_orbital_overlap_matrix;
      std::vector<double> minus_overlap = gradient_result.active_orbital_overlap_matrix;
      std::vector<double> plus_one = gradient_result.active_space_one_electron_result.h1e_act;
      std::vector<double> minus_one = gradient_result.active_space_one_electron_result.h1e_act;
      std::vector<double> plus_two = baseline_packed_two;
      std::vector<double> minus_two = baseline_packed_two;

      switch (options.component) {
        case Component::Overlap:
          plus_overlap[xmvb::to_size(probe_entry.primary_index)] += options.step;
          minus_overlap[xmvb::to_size(probe_entry.primary_index)] -= options.step;
          if (probe_entry.mirror_index != probe_entry.primary_index) {
            plus_overlap[xmvb::to_size(probe_entry.mirror_index)] += options.step;
            minus_overlap[xmvb::to_size(probe_entry.mirror_index)] -= options.step;
          }
          break;
        case Component::OneElectron:
          plus_one[xmvb::to_size(probe_entry.primary_index)] += options.step;
          minus_one[xmvb::to_size(probe_entry.primary_index)] -= options.step;
          if (probe_entry.mirror_index != probe_entry.primary_index) {
            plus_one[xmvb::to_size(probe_entry.mirror_index)] += options.step;
            minus_one[xmvb::to_size(probe_entry.mirror_index)] -= options.step;
          }
          break;
        case Component::TwoElectron:
          plus_two[xmvb::to_size(probe_entry.primary_index)] += options.step;
          minus_two[xmvb::to_size(probe_entry.primary_index)] -= options.step;
          break;
      }

      const double plus_energy = evaluate_exact_selected_total_energy_from_active_space(
          load_result.input,
          gradient_result,
          selected_structure_indices,
          plus_overlap,
          plus_one,
          plus_two,
          load_result.nuclear_repulsion_energy);
      const double minus_energy = evaluate_exact_selected_total_energy_from_active_space(
          load_result.input,
          gradient_result,
          selected_structure_indices,
          minus_overlap,
          minus_one,
          minus_two,
          load_result.nuclear_repulsion_energy);
      const double finite_difference = (plus_energy - minus_energy) / (2.0 * options.step);
      const double analytic = probe_entry.analytic;
      const double absolute_error = std::abs(analytic - finite_difference);
      const double relative_error =
          absolute_error / std::max(1.0, std::abs(finite_difference));
      worst_abs_error = std::max(worst_abs_error, absolute_error);

      std::cout << "entry[" << report_index << "]";
      if (probe_entry.primary_index == probe_entry.mirror_index) {
        std::cout << " index=" << probe_entry.primary_index;
      } else {
        std::cout << " indices=" << probe_entry.primary_index
                  << "," << probe_entry.mirror_index;
      }
      std::cout
                << " analytic=" << analytic
                << " fd=" << finite_difference
                << " abs_error=" << absolute_error
                << " rel_error=" << relative_error
                << '\n';
    }

    std::cout << "worst_abs_error = " << worst_abs_error << '\n';
    if (worst_abs_error > options.tolerance) {
      std::cerr << "worst_abs_error exceeds tolerance\n";
      return 1;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
