#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/pdft/selected_state_matrix_form_two_rdm_builder.hpp"
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
  std::cerr << "usage: check_selected_state_matrix_form_two_rdm <input.xmi>"
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
    xmvb::vb::SelectedStateMatrixFormTwoRdmBuilder two_rdm_builder(
        options.algorithm);
    const auto two_rdm_result =
        two_rdm_builder.build_from_state_specific_gradient(
            load_result.input,
            gradient_result);

    const int n_active_orbitals =
        load_result.input.orbital_preparation_input.n_active_orbitals;
    const double unpacked_energy_abs_error =
        std::abs(two_rdm_result.packed_active_two_electron_energy -
                 two_rdm_result.unpacked_active_pair_energy);

    std::cout << std::setprecision(15)
              << "state_index = " << two_rdm_result.state_index << '\n'
              << "n_active_orbitals = " << n_active_orbitals << '\n'
              << "n_active_pairs = " << two_rdm_result.n_active_pairs << '\n'
              << "packed_two_rdm_size = "
              << two_rdm_result.packed_matrix_form_active_two_rdm.size() << '\n'
              << "active_pair_density_symmetry_max_abs = "
              << two_rdm_result.active_pair_density_symmetry_max_abs << '\n'
              << "packed_active_two_electron_energy = "
              << two_rdm_result.packed_active_two_electron_energy << '\n'
              << "unpacked_active_pair_energy = "
              << two_rdm_result.unpacked_active_pair_energy << '\n'
              << "unpacked_energy_abs_error = "
              << unpacked_energy_abs_error << '\n';
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "error: " << exception.what() << '\n';
    return 1;
  }
}
