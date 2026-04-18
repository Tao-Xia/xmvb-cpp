#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "runtime/cpp_vb_input_loader.hpp"
#include "runtime/libcint_auxiliary_basis_builder.hpp"
#include "runtime/libcint_direct_shell_evaluator.hpp"

namespace {

struct Options {
  std::string input_path;
  int level = 2;
  bool use_star = true;
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
               "[--level n] [--no-star] [--primary-shell-pair i j] "
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
    if (argument_name == "--level") {
      if (argument_index >= argc) {
        throw std::invalid_argument("--level expects 1 integer");
      }
      options.level = std::stoi(argv[argument_index++]);
      continue;
    }
    if (argument_name == "--star") {
      options.use_star = true;
      continue;
    }
    if (argument_name == "--no-star") {
      options.use_star = false;
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
          xmvb::to_size(row) +
          xmvb::to_size(column) * left_right.left_ao_count;
      const std::size_t right_left_index =
          xmvb::to_size(column) +
          xmvb::to_size(row) * right_left.left_ao_count;
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
            xmvb::to_size(left_local) +
            xmvb::to_size(right_local) *
                left_right_auxiliary.primary_left_ao_count +
            xmvb::to_size(auxiliary_local) *
                left_right_auxiliary.primary_left_ao_count *
                left_right_auxiliary.primary_right_ao_count;
        const std::size_t right_left_index =
            xmvb::to_size(right_local) +
            xmvb::to_size(left_local) *
                right_left_auxiliary.primary_left_ao_count +
            xmvb::to_size(auxiliary_local) *
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

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    const auto load_result = xmvb::vb::load_cpp_vb_input_with_timings(options.input_path);
    const auto& input = load_result.input;

    xmvb::vb::LibcintAuxiliaryBasisBuilder builder;
    xmvb::vb::LibcintAuxiliaryBasisBuilderOptions builder_options;
    builder_options.level = options.level;
    builder_options.use_star = options.use_star;
    const auto auxiliary_input = builder.build(input.libcint_input, builder_options);
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

    std::cout << std::setprecision(12);
    std::cout << "primary_n_shells = " << input.libcint_input.n_shells << '\n';
    std::cout << "primary_n_basis_functions = " << evaluator.n_basis_functions() << '\n';
    std::cout << "auxiliary_level = " << options.level << '\n';
    std::cout << "auxiliary_star = " << (options.use_star ? 1 : 0) << '\n';
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
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
