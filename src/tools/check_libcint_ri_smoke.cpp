#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <Eigen/Eigenvalues>

#include "input/loading/loader.hpp"
#include "input/deck/model.hpp"
#include "input/deck/primary_basis.hpp"
#include "libcint/direct_shell.hpp"
#include "libcint/ri_provider.hpp"

namespace {

struct Options {
  std::string input_path;
  std::string auxiliary_basis_name;
  int primary_left_shell = 0;
  int primary_right_shell = 0;
  int auxiliary_left_shell = 0;
  int auxiliary_right_shell = 0;
  int three_center_primary_left_shell = 0;
  int three_center_primary_right_shell = 0;
  int three_center_auxiliary_shell = 0;
};

void print_usage() {
  std::cerr << "usage: check_libcint_ri_smoke <input.xmi> "
               "[--aux-basis name] [--primary-shell-pair i j] "
               "[--aux-shell-pair a b] [--three-center i j a]\n";
}

Options parse_arguments(int argc, char** argv) {
  if (argc < 2) {
    print_usage();
    throw std::invalid_argument("missing input path");
  }

  Options options;
  options.input_path = argv[1];
  for (int argument_index = 2; argument_index < argc;) {
    const std::string argument_name = argv[argument_index++];
    if (argument_name == "--aux-basis") {
      if (argument_index >= argc) {
        throw std::invalid_argument("--aux-basis expects a basis name");
      }
      options.auxiliary_basis_name = argv[argument_index++];
      continue;
    }
    if (argument_name == "--primary-shell-pair") {
      if (argument_index + 1 >= argc) {
        throw std::invalid_argument("--primary-shell-pair expects 2 integers");
      }
      options.primary_left_shell = std::stoi(argv[argument_index++]);
      options.primary_right_shell = std::stoi(argv[argument_index++]);
      options.three_center_primary_left_shell = options.primary_left_shell;
      options.three_center_primary_right_shell = options.primary_right_shell;
      continue;
    }
    if (argument_name == "--aux-shell-pair") {
      if (argument_index + 1 >= argc) {
        throw std::invalid_argument("--aux-shell-pair expects 2 integers");
      }
      options.auxiliary_left_shell = std::stoi(argv[argument_index++]);
      options.auxiliary_right_shell = std::stoi(argv[argument_index++]);
      options.three_center_auxiliary_shell = options.auxiliary_left_shell;
      continue;
    }
    if (argument_name == "--three-center") {
      if (argument_index + 2 >= argc) {
        throw std::invalid_argument("--three-center expects 3 integers");
      }
      options.three_center_primary_left_shell = std::stoi(argv[argument_index++]);
      options.three_center_primary_right_shell = std::stoi(argv[argument_index++]);
      options.three_center_auxiliary_shell = std::stoi(argv[argument_index++]);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + argument_name);
  }
  return options;
}

double shell_block_max_abs_value(const xmvb::vb::LibcintShellBlock& block) {
  double max_abs_value = 0.0;
  for (double value : block.values) {
    max_abs_value = std::max(max_abs_value, std::abs(value));
  }
  return max_abs_value;
}

double three_center_max_abs_value(const xmvb::vb::LibcintThreeCenterShellBlock& block) {
  double max_abs_value = 0.0;
  for (double value : block.values) {
    max_abs_value = std::max(max_abs_value, std::abs(value));
  }
  return max_abs_value;
}

double metric_transpose_max_abs_diff(
    const xmvb::vb::LibcintShellBlock& left_right,
    const xmvb::vb::LibcintShellBlock& right_left) {
  if (left_right.left_ao_count != right_left.right_ao_count ||
      left_right.right_ao_count != right_left.left_ao_count) {
    throw std::invalid_argument("metric transpose block dimensions do not match");
  }

  double max_abs_diff = 0.0;
  for (int column = 0; column < left_right.right_ao_count; ++column) {
    for (int row = 0; row < left_right.left_ao_count; ++row) {
      const std::size_t left_right_index =
          row +
          column * left_right.left_ao_count;
      const std::size_t right_left_index =
          column +
          row * right_left.left_ao_count;
      max_abs_diff = std::max(
          max_abs_diff,
          std::abs(left_right.values[left_right_index] - right_left.values[right_left_index]));
    }
  }
  return max_abs_diff;
}

double three_center_transpose_max_abs_diff(
    const xmvb::vb::LibcintThreeCenterShellBlock& left_right_auxiliary,
    const xmvb::vb::LibcintThreeCenterShellBlock& right_left_auxiliary) {
  if (left_right_auxiliary.primary_left_ao_count !=
          right_left_auxiliary.primary_right_ao_count ||
      left_right_auxiliary.primary_right_ao_count !=
          right_left_auxiliary.primary_left_ao_count ||
      left_right_auxiliary.auxiliary_ao_count != right_left_auxiliary.auxiliary_ao_count) {
    throw std::invalid_argument("three-center transpose block dimensions do not match");
  }

  double max_abs_diff = 0.0;
  for (int auxiliary_local = 0;
       auxiliary_local < left_right_auxiliary.auxiliary_ao_count;
       ++auxiliary_local) {
    for (int right_local = 0;
         right_local < left_right_auxiliary.primary_right_ao_count;
         ++right_local) {
      for (int left_local = 0;
           left_local < left_right_auxiliary.primary_left_ao_count;
           ++left_local) {
        const std::size_t left_right_index =
            left_local +
            right_local *
                left_right_auxiliary.primary_left_ao_count +
            auxiliary_local *
                left_right_auxiliary.primary_left_ao_count *
                left_right_auxiliary.primary_right_ao_count;
        const std::size_t right_left_index =
            right_local +
            left_local *
                right_left_auxiliary.primary_left_ao_count +
            auxiliary_local *
                right_left_auxiliary.primary_left_ao_count *
                right_left_auxiliary.primary_right_ao_count;
        max_abs_diff = std::max(
            max_abs_diff,
            std::abs(
                left_right_auxiliary.values[left_right_index] -
                right_left_auxiliary.values[right_left_index]));
      }
    }
  }
  return max_abs_diff;
}

int packed_pair_index(int first, int second) {
  if (first < second) {
    std::swap(first, second);
  }
  return first * (first + 1) / 2 + second;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    const auto load_result = xmvb::vb::load_vbscf_input_with_timings(options.input_path);
    const auto& input = load_result.input;

    const xmvb::vb::InputDeck auxiliary_deck =
        xmvb::vb::parse_input_deck_model(options.input_path);
    std::string auxiliary_basis_name = options.auxiliary_basis_name;
    if (auxiliary_basis_name.empty()) {
      auxiliary_basis_name = auxiliary_deck.metadata.auxiliary_basis_name;
    }
    if (auxiliary_basis_name.empty()) {
      auxiliary_basis_name = auxiliary_deck.metadata.basis_name;
      if (auxiliary_basis_name.size() >= 4 &&
          auxiliary_basis_name.substr(auxiliary_basis_name.size() - 4) ==
              ".gbs") {
        auxiliary_basis_name.resize(auxiliary_basis_name.size() - 4);
      }
      auxiliary_basis_name += "-jkfit";
    }
    const auto auxiliary_input = xmvb::vb::build_input_deck_basis(
        auxiliary_deck,
        auxiliary_basis_name).libcint_input;
    xmvb::vb::LibcintDirectShellEvaluator evaluator(
        input.libcint_input,
        auxiliary_input);

    const auto metric_block = evaluator.evaluate_auxiliary_metric_shell_pair(
        options.auxiliary_left_shell,
        options.auxiliary_right_shell);
    const auto metric_block_transposed = evaluator.evaluate_auxiliary_metric_shell_pair(
        options.auxiliary_right_shell,
        options.auxiliary_left_shell);
    const auto three_center_block = evaluator.evaluate_three_center_shell_block(
        options.three_center_primary_left_shell,
        options.three_center_primary_right_shell,
        options.three_center_auxiliary_shell);
    const auto three_center_block_transposed = evaluator.evaluate_three_center_shell_block(
        options.three_center_primary_right_shell,
        options.three_center_primary_left_shell,
        options.three_center_auxiliary_shell);
    xmvb::vb::LibcintRiIntegralProvider ri_provider;
    const auto ri = ri_provider.build(
        input.libcint_input,
        auxiliary_input,
        xmvb::vb::LibcintRiIntegralProviderOptions{});
    const Eigen::MatrixXd ri_pair_kernel =
        ri.metric_whitened_ao_pair_factors.transpose() *
        ri.metric_whitened_ao_pair_factors;
    Eigen::MatrixXd exact_pair_kernel = Eigen::MatrixXd::Zero(
        ri.n_packed_ao_pairs,
        ri.n_packed_ao_pairs);
    input.ao_integral_input.pair_graph.for_each_integral(
        [&](double value, int i, int j, int k, int l) {
          const int left = packed_pair_index(i, j);
          const int right = packed_pair_index(k, l);
          exact_pair_kernel(left, right) = value;
          exact_pair_kernel(right, left) = value;
        });
    const Eigen::MatrixXd pair_error = ri_pair_kernel - exact_pair_kernel;
    const double exact_pair_norm = exact_pair_kernel.norm();

    std::cout << std::setprecision(12);
    std::cout << "primary_n_shells = " << input.libcint_input.n_shells << '\n';
    std::cout << "primary_n_basis_functions = " << evaluator.n_basis_functions() << '\n';
    std::cout << "auxiliary_basis = " << auxiliary_basis_name << '\n';
    std::cout << "auxiliary_n_shells = " << auxiliary_input.n_shells << '\n';
    std::cout << "auxiliary_n_basis_functions = "
              << evaluator.n_auxiliary_basis_functions() << '\n';
    std::cout << "metric_shell_pair = "
              << options.auxiliary_left_shell << ' '
              << options.auxiliary_right_shell << '\n';
    std::cout << "metric_ao_counts = "
              << metric_block.left_ao_count << ' '
              << metric_block.right_ao_count << '\n';
    std::cout << "metric_max_abs_value = "
              << shell_block_max_abs_value(metric_block) << '\n';
    std::cout << "metric_transpose_max_abs_diff = "
              << metric_transpose_max_abs_diff(metric_block, metric_block_transposed) << '\n';
    std::cout << "three_center_shells = "
              << options.three_center_primary_left_shell << ' '
              << options.three_center_primary_right_shell << ' '
              << options.three_center_auxiliary_shell << '\n';
    std::cout << "three_center_ao_counts = "
              << three_center_block.primary_left_ao_count << ' '
              << three_center_block.primary_right_ao_count << ' '
              << three_center_block.auxiliary_ao_count << '\n';
    std::cout << "three_center_max_abs_value = "
              << three_center_max_abs_value(three_center_block) << '\n';
    std::cout << "three_center_transpose_max_abs_diff = "
              << three_center_transpose_max_abs_diff(
                     three_center_block,
                     three_center_block_transposed)
              << '\n';
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> metric_solver(
        ri.auxiliary_metric_matrix);
    std::cout << "ri_metric_min_eigenvalue = "
              << metric_solver.eigenvalues().minCoeff() << '\n';
    std::cout << "ri_pair_kernel_max_abs_error = "
              << pair_error.cwiseAbs().maxCoeff() << '\n';
    std::cout << "ri_pair_kernel_relative_frobenius_error = "
              << pair_error.norm() / std::max(1.0, exact_pair_norm) << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
