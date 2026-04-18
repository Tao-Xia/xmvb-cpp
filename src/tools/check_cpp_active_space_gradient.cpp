#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "core/linear_algebra/generalized_eigensolver.hpp"
#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/matrices/full_structure_builder.hpp"
#include "vb/orbital/active_space_two_electron_utils.hpp"
#include "vb/scf/cpp_active_space_gradient_evaluator.hpp"
#include "vb/vbscf_algorithm.hpp"

namespace {

enum class Component {
  Overlap,
  OneElectron,
  TwoElectron,
};

struct Options {
  std::string input_path;
  xmvb::vb::VBSCFAlgorithm algorithm = xmvb::vb::VBSCFAlgorithm::Original;
  xmvb::vb::StandardTwoElectronMode standard_two_electron_mode =
      xmvb::vb::StandardTwoElectronMode::Auto;
  Component component = Component::Overlap;
  int count = 8;
  double step = 1.0e-6;
};

void print_usage() {
  std::cerr << "usage: check_cpp_active_space_gradient <input.xmi> "
               "[--algorithm original] "
               "[--standard-two-electron-mode auto|exact|ri] "
               "[--component overlap|one_electron|two_electron] "
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

double compute_one_electron_reference_energy(
    const Eigen::MatrixXd& inactive_density_matrix,
    const std::vector<double>& ao_effective_h1e,
    const std::vector<double>& ao_core_hamiltonian_matrix,
    int n_basis_functions) {
  if (inactive_density_matrix.rows() != n_basis_functions ||
      inactive_density_matrix.cols() != n_basis_functions) {
    throw std::invalid_argument(
        "inactive density matrix shape does not match n_basis_functions");
  }
  double one_electron_reference_energy = 0.0;
  for (int column = 0; column < n_basis_functions; ++column) {
    for (int row = 0; row < n_basis_functions; ++row) {
      const std::size_t index =
          xmvb::to_size(column) * n_basis_functions + row;
      one_electron_reference_energy +=
          inactive_density_matrix.data()[index] *
          (ao_effective_h1e[index] + ao_core_hamiltonian_matrix[index]);
    }
  }
  return one_electron_reference_energy;
}

double evaluate_total_energy_from_active_space(
    const xmvb::vb::CppVbInput& input,
    const xmvb::vb::CppActiveSpaceGradientResult& baseline,
    const std::vector<double>& active_orbital_overlap_matrix,
    const std::vector<double>& h1e_act,
    const std::vector<double>& packed_active_two_electron_integrals,
    double nuclear_repulsion_energy,
    xmvb::vb::VBSCFAlgorithm algorithm) {
  xmvb::vb::FullDeterminantStructureHamiltonianOverlapBuilder structure_builder(algorithm);
  const auto structure_matrices = structure_builder.build(
      input.structure_data.alpha_det,
      input.structure_data.beta_det,
      input.structure_data.determinant_to_structure_terms,
      active_orbital_overlap_matrix,
      h1e_act,
      input.orbital_preparation_input.n_active_orbitals,
      packed_active_two_electron_integrals,
      input.structure_data.n_structures);
  xmvb::core::GeneralizedEigensolver generalized_eigensolver;
  const auto eigen_result = generalized_eigensolver.solve(
      structure_matrices.hamiltonian_matrix,
      structure_matrices.overlap_matrix,
      input.structure_data.n_structures);
  const double one_electron_reference_energy = compute_one_electron_reference_energy(
      baseline.orbital_preparation_result.inactive_density_matrix,
      baseline.ao_effective_one_electron_result.ao_effective_h1e,
      input.ao_integral_input.ao_core_hamiltonian_matrix,
      input.ao_integral_input.n_basis_functions);
  return one_electron_reference_energy + eigen_result.eigenvalues.front() + nuclear_repulsion_energy;
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

int count_nonzero_off_diagonal_entries(
    const std::vector<double>& matrix,
    int dimension,
    double tolerance) {
  if (static_cast<int>(matrix.size()) != dimension * dimension) {
    throw std::invalid_argument("matrix size does not match dimension");
  }
  int count = 0;
  for (int column = 0; column < dimension; ++column) {
    for (int row = 0; row < dimension; ++row) {
      if (row == column) {
        continue;
      }
      const double value =
          matrix[xmvb::to_size(column) * dimension + row];
      if (std::abs(value) > tolerance) {
        ++count;
      }
    }
  }
  return count;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    xmvb::vb::CppVbInputLoadOptions load_options;
    load_options.ao_integral_source = xmvb::vb::AoIntegralSource::LegacyRuntime;
    load_options.orbital_guess_source =
        xmvb::vb::OrbitalGuessSource::LegacyRuntime;
    load_options.standard_two_electron_mode = options.standard_two_electron_mode;
    const auto load_result =
        xmvb::vb::load_cpp_vb_input_with_timings(options.input_path, load_options);
    xmvb::vb::CppActiveSpaceGradientEvaluator evaluator(options.algorithm);
    const auto result = evaluator.evaluate(load_result.input, load_result.nuclear_repulsion_energy);
    const auto& gradient = component_gradient(result, options.component);
    const std::vector<double> baseline_packed_two =
        result.active_space_two_electron_result.packed_active_two_electron_integrals.empty()
            ? xmvb::vb::reconstruct_packed_active_two_electron_integrals(
                  xmvb::vb::make_active_space_two_electron_view(
                      result.active_space_two_electron_result),
                  load_result.input.orbital_preparation_input.n_active_orbitals)
            : result.active_space_two_electron_result.packed_active_two_electron_integrals;

    std::vector<std::pair<double, int>> ranked_entries;
    ranked_entries.reserve(gradient.size());
    for (std::size_t index = 0; index < gradient.size(); ++index) {
      ranked_entries.emplace_back(std::abs(gradient[index]), static_cast<int>(index));
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
    std::cout << std::setprecision(12);
    std::cout << "algorithm = " << xmvb::vb::vb_scf_algorithm_name(options.algorithm) << '\n';
    std::cout << "standard_two_electron_mode = "
              << xmvb::vb::standard_two_electron_mode_name(
                     load_result.standard_two_electron_mode)
              << '\n';
    std::cout << "component = " << component_name(options.component) << '\n';
    std::cout << "initial_total_energy = " << result.scf_result.total_energy << '\n';
    std::cout << "structure_overlap_offdiag_nnz = "
              << count_nonzero_off_diagonal_entries(
                     result.scf_result.structure_matrices.overlap_matrix,
                     result.scf_result.n_structures,
                     1.0e-12)
              << '\n';
    std::cout << "structure_hamiltonian_offdiag_nnz = "
              << count_nonzero_off_diagonal_entries(
                     result.scf_result.structure_matrices.hamiltonian_matrix,
                     result.scf_result.n_structures,
                     1.0e-12)
              << '\n';
    std::cout << "finite_difference_step = " << options.step << '\n';
    std::cout << "reported_entries = " << n_to_report << '\n';

    for (int report_index = 0; report_index < n_to_report; ++report_index) {
      const int entry_index = ranked_entries[xmvb::to_size(report_index)].second;
      std::vector<double> plus_overlap = result.active_orbital_overlap_matrix;
      std::vector<double> minus_overlap = result.active_orbital_overlap_matrix;
      std::vector<double> plus_one = result.active_space_one_electron_result.h1e_act;
      std::vector<double> minus_one = result.active_space_one_electron_result.h1e_act;
      std::vector<double> plus_two = baseline_packed_two;
      std::vector<double> minus_two = baseline_packed_two;

      switch (options.component) {
        case Component::Overlap:
          plus_overlap[xmvb::to_size(entry_index)] += options.step;
          minus_overlap[xmvb::to_size(entry_index)] -= options.step;
          break;
        case Component::OneElectron:
          plus_one[xmvb::to_size(entry_index)] += options.step;
          minus_one[xmvb::to_size(entry_index)] -= options.step;
          break;
        case Component::TwoElectron:
          plus_two[xmvb::to_size(entry_index)] += options.step;
          minus_two[xmvb::to_size(entry_index)] -= options.step;
          break;
      }

      const double plus_energy = evaluate_total_energy_from_active_space(
          load_result.input,
          result,
          plus_overlap,
          plus_one,
          plus_two,
          load_result.nuclear_repulsion_energy,
          options.algorithm);
      const double minus_energy = evaluate_total_energy_from_active_space(
          load_result.input,
          result,
          minus_overlap,
          minus_one,
          minus_two,
          load_result.nuclear_repulsion_energy,
          options.algorithm);
      const double finite_difference = (plus_energy - minus_energy) / (2.0 * options.step);
      const double analytic = gradient[xmvb::to_size(entry_index)];
      const double absolute_error = std::abs(analytic - finite_difference);
      const double relative_error =
          absolute_error / std::max(1.0, std::abs(finite_difference));

      std::cout << "entry[" << report_index << "]"
                << " index=" << entry_index
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
