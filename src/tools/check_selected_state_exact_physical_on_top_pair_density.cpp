#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/pdft/selected_state_exact_physical_on_top_pair_density_builder.hpp"
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
  std::cerr << "usage: check_selected_state_exact_physical_on_top_pair_density <input.xmi>"
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

std::size_t ordered_pair_storage_index(
    int left_index,
    int right_index,
    int n_support_determinants) {
  return xmvb::to_size(left_index) * n_support_determinants + right_index;
}

Eigen::MatrixXd build_transition_density_scalar_matrix(
    const std::vector<Eigen::MatrixXd>& support_occupied_physical_orbitals,
    const std::vector<Eigen::MatrixXd>& ordered_pair_first_order_cofactors,
    const Eigen::Ref<const Eigen::VectorXd>& ao_values) {
  const int n_support_determinants =
      static_cast<int>(support_occupied_physical_orbitals.size());
  const int n_occupied_orbitals =
      support_occupied_physical_orbitals.front().cols();

  Eigen::MatrixXd occupied_value_projections =
      Eigen::MatrixXd::Zero(n_occupied_orbitals, n_support_determinants);
  for (int support_index = 0;
       support_index < n_support_determinants;
       ++support_index) {
    occupied_value_projections.col(support_index).noalias() =
        support_occupied_physical_orbitals[xmvb::to_size(support_index)].transpose() *
        ao_values;
  }

  Eigen::MatrixXd transition_density_scalars =
      Eigen::MatrixXd::Zero(n_support_determinants, n_support_determinants);
  for (int left_index = 0;
       left_index < n_support_determinants;
       ++left_index) {
    for (int right_index = 0;
         right_index < n_support_determinants;
         ++right_index) {
      transition_density_scalars(right_index, left_index) =
          occupied_value_projections.col(right_index).dot(
              ordered_pair_first_order_cofactors[ordered_pair_storage_index(
                  left_index,
                  right_index,
                  n_support_determinants)] *
              occupied_value_projections.col(left_index));
    }
  }
  return transition_density_scalars;
}

double evaluate_selected_state_exact_physical_on_top_pair_density_direct(
    const xmvb::vb::SelectedStateExactPhysicalOnTopPairDensityContext& context,
    const Eigen::Ref<const Eigen::VectorXd>& ao_values) {
  const Eigen::MatrixXd alpha_transition_density_scalars =
      build_transition_density_scalar_matrix(
          context.alpha_support_occupied_physical_orbitals,
          context.alpha_pair_first_order_cofactors,
          ao_values);
  const Eigen::MatrixXd beta_transition_density_scalars =
      build_transition_density_scalar_matrix(
          context.beta_support_occupied_physical_orbitals,
          context.beta_pair_first_order_cofactors,
          ao_values);

  double on_top_pair_density = 0.0;
  for (int alpha_left = 0;
       alpha_left < context.n_alpha_support_determinants;
       ++alpha_left) {
    for (int alpha_right = 0;
         alpha_right < context.n_alpha_support_determinants;
         ++alpha_right) {
      const double alpha_scalar =
          alpha_transition_density_scalars(alpha_right, alpha_left);
      for (int beta_left = 0;
           beta_left < context.n_beta_support_determinants;
           ++beta_left) {
        const double coefficient_left =
            context.local_coefficient_matrix(alpha_left, beta_left);
        for (int beta_right = 0;
             beta_right < context.n_beta_support_determinants;
             ++beta_right) {
          on_top_pair_density +=
              coefficient_left *
              context.local_coefficient_matrix(alpha_right, beta_right) *
              alpha_scalar *
              beta_transition_density_scalars(beta_right, beta_left);
        }
      }
    }
  }
  return on_top_pair_density /
      context.selected_state_overlap_normalization;
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
    xmvb::vb::SelectedStateExactPhysicalOnTopPairDensityBuilder builder(
        options.algorithm);
    const auto context =
        builder.build_from_state_specific_gradient(
            load_result.input,
            gradient_result);

    const auto probes = build_probe_vectors(context.n_basis_functions);
    const Eigen::MatrixXd probe_matrix =
        build_probe_matrix(probes, context.n_basis_functions);
    const Eigen::VectorXd batch_values =
        xmvb::vb::evaluate_selected_state_exact_physical_on_top_pair_density_batch(
            context,
            probe_matrix);
    double max_probe_abs_error = 0.0;
    std::cout << std::setprecision(15)
              << "state_index = " << context.state_index << '\n'
              << "n_basis_functions = " << context.n_basis_functions << '\n'
              << "n_alpha_support_determinants = "
              << context.n_alpha_support_determinants << '\n'
              << "n_beta_support_determinants = "
              << context.n_beta_support_determinants << '\n'
              << "selected_state_overlap_normalization = "
              << context.selected_state_overlap_normalization << '\n'
              << "alpha_beta_normalization_abs_error = "
              << context.alpha_beta_normalization_abs_error << '\n';

    for (std::size_t probe_index = 0; probe_index < probes.size(); ++probe_index) {
      const double matrix_value =
          xmvb::vb::evaluate_selected_state_exact_physical_on_top_pair_density(
              context,
              probes[probe_index]);
      const double direct_value =
          evaluate_selected_state_exact_physical_on_top_pair_density_direct(
              context,
              probes[probe_index]);
      const double batch_value = batch_values(static_cast<int>(probe_index));
      const double abs_error =
          std::max(
              std::abs(matrix_value - direct_value),
              std::abs(batch_value - matrix_value));
      max_probe_abs_error = std::max(max_probe_abs_error, abs_error);
      std::cout << "probe_" << probe_index << "_matrix_value = "
                << matrix_value << '\n'
                << "probe_" << probe_index << "_batch_value = "
                << batch_value << '\n'
                << "probe_" << probe_index << "_direct_value = "
                << direct_value << '\n'
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
