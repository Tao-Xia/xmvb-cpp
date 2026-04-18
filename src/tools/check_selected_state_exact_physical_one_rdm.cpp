#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/pdft/selected_state_exact_physical_one_rdm_builder.hpp"
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
  std::cerr << "usage: check_selected_state_exact_physical_one_rdm <input.xmi>"
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

std::vector<Eigen::VectorXd> build_probe_vectors(int n_basis_functions) {
  std::vector<Eigen::VectorXd> probes;
  if (n_basis_functions <= 0) {
    return probes;
  }

  Eigen::VectorXd unit_probe = Eigen::VectorXd::Zero(n_basis_functions);
  unit_probe(0) = 1.0;
  probes.push_back(unit_probe);

  Eigen::VectorXd signed_probe = Eigen::VectorXd::Zero(n_basis_functions);
  const int signed_probe_count = std::min(4, n_basis_functions);
  for (int index = 0; index < signed_probe_count; ++index) {
    const double sign = (index % 2 == 0) ? 1.0 : -1.0;
    signed_probe(index) = sign / static_cast<double>(index + 1);
  }
  probes.push_back(signed_probe);

  Eigen::VectorXd ramp_probe = Eigen::VectorXd::Zero(n_basis_functions);
  const int ramp_probe_count = std::min(6, n_basis_functions);
  for (int index = 0; index < ramp_probe_count; ++index) {
    ramp_probe(index) =
        static_cast<double>(index + 1) / static_cast<double>(ramp_probe_count);
  }
  probes.push_back(ramp_probe);

  return probes;
}

Eigen::MatrixXd build_probe_matrix(
    const std::vector<Eigen::VectorXd>& probes,
    int n_basis_functions) {
  Eigen::MatrixXd probe_matrix =
      Eigen::MatrixXd::Zero(n_basis_functions, static_cast<int>(probes.size()));
  for (std::size_t probe_index = 0; probe_index < probes.size(); ++probe_index) {
    if (probes[probe_index].size() != n_basis_functions) {
      throw std::invalid_argument("probe vector size mismatch");
    }
    probe_matrix.col(static_cast<int>(probe_index)) = probes[probe_index];
  }
  return probe_matrix;
}

double evaluate_selected_state_exact_physical_density_direct(
    const xmvb::vb::SelectedStateExactPhysicalOneRdmResult& density_result,
    const Eigen::Ref<const Eigen::VectorXd>& ao_values) {
  return ao_values.dot(
      density_result.physical_ao_total_density_matrix * ao_values);
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
    xmvb::vb::SelectedStateExactPhysicalOneRdmBuilder builder(options.algorithm);
    const auto density_result =
        builder.build_from_state_specific_gradient(
            load_result.input,
            gradient_result);
    const auto probes = build_probe_vectors(density_result.n_basis_functions);
    const Eigen::MatrixXd probe_matrix =
        build_probe_matrix(probes, density_result.n_basis_functions);
    const Eigen::VectorXd batch_density_values =
        xmvb::vb::evaluate_selected_state_exact_physical_density_batch(
            density_result,
            probe_matrix);
    double max_probe_abs_error = 0.0;

    std::cout << std::setprecision(15)
              << "state_index = " << density_result.state_index << '\n'
              << "n_basis_functions = " << density_result.n_basis_functions << '\n'
              << "expected_total_electrons = "
              << load_result.input.orbital_preparation_input.n_total_electrons << '\n'
              << "selected_state_overlap_normalization = "
              << density_result.selected_state_overlap_normalization << '\n'
              << "alpha_beta_normalization_abs_error = "
              << density_result.alpha_beta_normalization_abs_error << '\n'
              << "ao_density_asymmetry_max_abs = "
              << density_result.ao_density_asymmetry_max_abs << '\n'
              << "ao_total_metric_trace = "
              << density_result.ao_total_metric_trace << '\n';
    for (std::size_t probe_index = 0; probe_index < probes.size(); ++probe_index) {
      const double scalar_density =
          xmvb::vb::evaluate_selected_state_exact_physical_density(
              density_result,
              probes[probe_index]);
      const double direct_density =
          evaluate_selected_state_exact_physical_density_direct(
              density_result,
              probes[probe_index]);
      const double batch_density =
          batch_density_values(static_cast<int>(probe_index));
      const double abs_error =
          std::max(
              std::abs(scalar_density - direct_density),
              std::abs(batch_density - scalar_density));
      max_probe_abs_error = std::max(max_probe_abs_error, abs_error);
      std::cout << "probe_" << probe_index << "_scalar_density = "
                << scalar_density << '\n'
                << "probe_" << probe_index << "_batch_density = "
                << batch_density << '\n'
                << "probe_" << probe_index << "_direct_density = "
                << direct_density << '\n'
                << "probe_" << probe_index << "_abs_error = "
                << abs_error << '\n';
    }
    std::cout << "max_probe_abs_error = " << max_probe_abs_error << '\n';
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "error: " << exception.what() << '\n';
    return 1;
  }
}
