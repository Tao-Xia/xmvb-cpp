#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/scf/cpp_active_space_gradient_evaluator.hpp"
#include "vb/orbital/active_space_two_electron_backpropagator.hpp"
#include "vb/orbital/active_space_two_electron_utils.hpp"

namespace {

struct Options {
  std::string input_path;
  double step = 1.0e-6;
};

void print_usage() {
  std::cerr
      << "usage: check_exact_two_electron_hvp <input.xmi> [--step h]\n";
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
    if (name == "--step") {
      options.step = std::stod(value);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + name);
  }
  if (!(options.step > 0.0)) {
    throw std::invalid_argument("--step must be positive");
  }
  return options;
}

Eigen::MatrixXd build_dense_active_direction(
    int n_basis_functions,
    int n_active_orbitals) {
  Eigen::MatrixXd direction =
      Eigen::MatrixXd::Zero(n_basis_functions, n_active_orbitals);
  double norm_sq = 0.0;
  for (int active_orbital_index = 0;
       active_orbital_index < n_active_orbitals;
       ++active_orbital_index) {
    for (int basis_function_index = 0;
         basis_function_index < n_basis_functions;
         ++basis_function_index) {
      const std::size_t linear_index =
          xmvb::to_size(active_orbital_index) * n_basis_functions +
          basis_function_index;
      const double value =
          std::sin(0.37 * static_cast<double>(linear_index + 1)) +
          0.5 * std::cos(0.13 * static_cast<double>(linear_index + 3));
      direction(basis_function_index, active_orbital_index) = value;
      norm_sq += value * value;
    }
  }
  const double norm = std::sqrt(norm_sq);
  direction /= norm;
  return direction;
}

Eigen::MatrixXd extract_dense_active_block(
    const std::vector<double>& full_auxiliary_gradient,
    int n_basis_functions,
    int n_inactive_doubly_occupied_orbitals,
    int n_active_orbitals) {
  const std::size_t expected_size =
      xmvb::to_size(n_basis_functions) * n_basis_functions;
  if (full_auxiliary_gradient.size() != expected_size) {
    throw std::invalid_argument("full auxiliary gradient size mismatch");
  }

  Eigen::MatrixXd dense_active_gradient =
      Eigen::MatrixXd::Zero(n_basis_functions, n_active_orbitals);
  for (int basis_function_index = 0;
       basis_function_index < n_basis_functions;
       ++basis_function_index) {
    for (int active_orbital_index = 0;
         active_orbital_index < n_active_orbitals;
         ++active_orbital_index) {
      const int column_index =
          n_inactive_doubly_occupied_orbitals + active_orbital_index;
      dense_active_gradient(basis_function_index, active_orbital_index) =
          full_auxiliary_gradient[xmvb::to_size(column_index) * n_basis_functions +
                                  basis_function_index];
    }
  }
  return dense_active_gradient;
}

double max_abs_difference(
    const Eigen::Ref<const Eigen::MatrixXd>& left,
    const Eigen::Ref<const Eigen::MatrixXd>& right) {
  if (left.rows() != right.rows() || left.cols() != right.cols()) {
    throw std::invalid_argument("matrix shape mismatch");
  }
  double max_abs_diff = 0.0;
  for (int col = 0; col < left.cols(); ++col) {
    for (int row = 0; row < left.rows(); ++row) {
      max_abs_diff = std::max(
          max_abs_diff,
          std::abs(left(row, col) - right(row, col)));
    }
  }
  return max_abs_diff;
}

double max_abs_value(const Eigen::Ref<const Eigen::MatrixXd>& values) {
  double max_abs = 0.0;
  for (int col = 0; col < values.cols(); ++col) {
    for (int row = 0; row < values.rows(); ++row) {
      max_abs = std::max(max_abs, std::abs(values(row, col)));
    }
  }
  return max_abs;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);

    xmvb::vb::CppVbInputLoadOptions load_options;
    load_options.standard_two_electron_mode =
        xmvb::vb::StandardTwoElectronMode::Exact;
    const auto load_result =
        xmvb::vb::load_cpp_vb_input_with_timings(options.input_path, load_options);
    const auto& input = load_result.input;

    xmvb::vb::CppActiveSpaceGradientEvaluator evaluator(
        xmvb::vb::VBSCFAlgorithm::Original);
    const auto gradient_result =
        evaluator.evaluate(
            input,
            {0},
            {1.0},
            load_result.nuclear_repulsion_energy);

    const int n_basis_functions = input.orbital_preparation_input.n_basis_functions;
    const int n_active_orbitals = input.orbital_preparation_input.n_active_orbitals;
    const int n_inactive_doubly_occupied_orbitals =
        (input.orbital_preparation_input.n_total_electrons -
         input.orbital_preparation_input.n_active_electrons) /
        2;
    if (n_basis_functions <= 0 || n_active_orbitals <= 0) {
      throw std::runtime_error("active-space dimensions must be positive");
    }

    const auto dense_active_direction =
        build_dense_active_direction(
            n_basis_functions,
            n_active_orbitals);
    const auto analytic_dense_active_gradient_direction =
        xmvb::vb::apply_exact_packed_active_two_electron_adjoint_hessian_vector(
            gradient_result.packed_active_two_electron_gradient,
            gradient_result.active_space_two_electron_result
                .dense_active_coefficients,
            dense_active_direction,
            input.ao_integral_input,
            n_active_orbitals,
            &gradient_result.active_space_two_electron_result);

    std::vector<double> plus_auxiliary_matrix =
        gradient_result.orbital_preparation_result.auxiliary_orbital_matrix;
    std::vector<double> minus_auxiliary_matrix =
        gradient_result.orbital_preparation_result.auxiliary_orbital_matrix;
    for (int basis_function_index = 0;
         basis_function_index < n_basis_functions;
         ++basis_function_index) {
      for (int active_orbital_index = 0;
           active_orbital_index < n_active_orbitals;
           ++active_orbital_index) {
        const int column_index =
            n_inactive_doubly_occupied_orbitals + active_orbital_index;
        const std::size_t flat_index =
            xmvb::to_size(column_index) * n_basis_functions + basis_function_index;
        const double delta =
            options.step *
            dense_active_direction(basis_function_index, active_orbital_index);
        plus_auxiliary_matrix[flat_index] += delta;
        minus_auxiliary_matrix[flat_index] -= delta;
      }
    }

    xmvb::vb::ActiveSpaceTwoElectronBackpropagator backpropagator;
    const auto plus_backpropagation_result =
        backpropagator.backpropagate(
            gradient_result.packed_active_two_electron_gradient,
            input.ao_integral_input.ao_two_electron_integral_values.vector(),
            input.ao_integral_input.ao_two_electron_integral_indices.vector(),
            plus_auxiliary_matrix,
            n_basis_functions,
            n_inactive_doubly_occupied_orbitals,
            n_active_orbitals);
    const auto minus_backpropagation_result =
        backpropagator.backpropagate(
            gradient_result.packed_active_two_electron_gradient,
            input.ao_integral_input.ao_two_electron_integral_values.vector(),
            input.ao_integral_input.ao_two_electron_integral_indices.vector(),
            minus_auxiliary_matrix,
            n_basis_functions,
            n_inactive_doubly_occupied_orbitals,
            n_active_orbitals);

    const auto plus_dense_active_gradient =
        extract_dense_active_block(
            plus_backpropagation_result.auxiliary_orbital_gradient,
            n_basis_functions,
            n_inactive_doubly_occupied_orbitals,
            n_active_orbitals);
    const auto minus_dense_active_gradient =
        extract_dense_active_block(
            minus_backpropagation_result.auxiliary_orbital_gradient,
            n_basis_functions,
            n_inactive_doubly_occupied_orbitals,
            n_active_orbitals);
    const Eigen::MatrixXd finite_difference_dense_active_gradient_direction =
        (plus_dense_active_gradient - minus_dense_active_gradient) /
        (2.0 * options.step);

    const double max_abs_diff =
        max_abs_difference(
            analytic_dense_active_gradient_direction,
            finite_difference_dense_active_gradient_direction);
    const double max_abs_fd =
        max_abs_value(finite_difference_dense_active_gradient_direction);
    std::cout << std::setprecision(12);
    std::cout << "input = " << options.input_path << '\n';
    std::cout << "n_basis_functions = " << n_basis_functions << '\n';
    std::cout << "n_active_orbitals = " << n_active_orbitals << '\n';
    std::cout << "finite_difference_step = " << options.step << '\n';
    std::cout << "analytic_inf_norm = "
              << max_abs_value(analytic_dense_active_gradient_direction) << '\n';
    std::cout << "fd_inf_norm = " << max_abs_fd << '\n';
    std::cout << "max_abs_diff = " << max_abs_diff << '\n';
    std::cout << "max_rel_diff = "
              << max_abs_diff / std::max(1.0, max_abs_fd) << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
