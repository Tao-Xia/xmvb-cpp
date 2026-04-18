#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/pdft/selected_state_matrix_form_one_rdm_builder.hpp"
#include "vb/scf/cpp_active_space_gradient_evaluator.hpp"

namespace {

struct Options {
  std::string input_path;
  int state_index = 0;
  xmvb::vb::VBSCFAlgorithm algorithm = xmvb::vb::VBSCFAlgorithm::Original;
  xmvb::vb::StandardTwoElectronMode standard_two_electron_mode =
      xmvb::vb::StandardTwoElectronMode::Auto;
};

void print_usage() {
  std::cerr << "usage: check_selected_state_matrix_form_one_rdm <input.xmi>"
               " [--state-index N]"
               " [--algorithm original]"
               " [--standard-two-electron-mode auto|exact|ri]\n";
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
    if (name == "--state-index") {
      options.state_index = std::stoi(value);
      continue;
    }
    if (name == "--algorithm") {
      if (value == "original") {
        options.algorithm = xmvb::vb::VBSCFAlgorithm::Original;
      } else {
        throw std::invalid_argument("invalid algorithm: " + value);
      }
      continue;
    }
    if (name == "--standard-two-electron-mode") {
      if (value == "auto") {
        options.standard_two_electron_mode =
            xmvb::vb::StandardTwoElectronMode::Auto;
      } else if (value == "exact") {
        options.standard_two_electron_mode =
            xmvb::vb::StandardTwoElectronMode::Exact;
      } else if (value == "ri") {
        options.standard_two_electron_mode =
            xmvb::vb::StandardTwoElectronMode::ResolutionOfIdentity;
      } else {
        throw std::invalid_argument(
            "invalid standard two-electron mode: " + value);
      }
      continue;
    }
    throw std::invalid_argument("unknown argument: " + name);
  }

  if (options.state_index < 0) {
    throw std::invalid_argument("--state-index must be non-negative");
  }
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    xmvb::vb::CppVbInputLoadOptions load_options;
    load_options.standard_two_electron_mode = options.standard_two_electron_mode;
    const auto load_result =
        xmvb::vb::load_cpp_vb_input_with_timings(options.input_path, load_options);
    xmvb::vb::CppActiveSpaceGradientEvaluator gradient_evaluator(options.algorithm);
    const auto gradient_result =
        gradient_evaluator.evaluate(
            load_result.input,
            std::vector<int>{options.state_index},
            std::vector<double>{1.0},
            load_result.nuclear_repulsion_energy);
    xmvb::vb::SelectedStateMatrixFormOneRdmBuilder one_rdm_builder(
        options.algorithm);
    const auto one_rdm_result =
        one_rdm_builder.build_from_state_specific_gradient(
            load_result.input,
            gradient_result);

    const int n_basis_functions =
        load_result.input.orbital_preparation_input.n_basis_functions;
    const int n_active_orbitals =
        load_result.input.orbital_preparation_input.n_active_orbitals;
    const int n_inactive_doubly_occupied_orbitals =
        (load_result.input.orbital_preparation_input.n_total_electrons -
         load_result.input.orbital_preparation_input.n_active_electrons) / 2;
    const auto& physical_frame =
        gradient_result.orbital_preparation_result.physical_orbital_frame;
    const int physical_active_columns =
        n_basis_functions > 0
            ? static_cast<int>(
                  physical_frame.active_physical_orbital_matrix.size() /
                  xmvb::to_size(n_basis_functions))
            : 0;
    const int physical_inactive_columns =
        n_basis_functions > 0
            ? static_cast<int>(
                  physical_frame.inactive_physical_orbital_matrix.size() /
                  xmvb::to_size(n_basis_functions))
            : 0;

    std::cout << std::setprecision(15)
              << "state_index = " << one_rdm_result.state_index << '\n'
              << "n_basis_functions = " << n_basis_functions << '\n'
              << "n_active_orbitals = " << n_active_orbitals << '\n'
              << "n_inactive_doubly_occupied_orbitals = "
              << n_inactive_doubly_occupied_orbitals << '\n'
              << "expected_active_electrons = "
              << load_result.input.orbital_preparation_input.n_active_electrons << '\n'
              << "expected_total_electrons = "
              << load_result.input.orbital_preparation_input.n_total_electrons << '\n'
              << "active_one_rdm_asymmetry_max_abs = "
              << one_rdm_result.active_one_rdm_asymmetry_max_abs << '\n'
              << "active_metric_trace = "
              << one_rdm_result.active_metric_trace << '\n'
              << "ao_active_metric_trace = "
              << one_rdm_result.ao_active_metric_trace << '\n'
              << "ao_total_metric_trace = "
              << one_rdm_result.ao_total_metric_trace << '\n'
              << "physical_active_columns = " << physical_active_columns << '\n'
              << "physical_inactive_columns = " << physical_inactive_columns << '\n';
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "error: " << exception.what() << '\n';
    return 1;
  }
}
