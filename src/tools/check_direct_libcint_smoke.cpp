#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "runtime/cpp_vb_input_loader.hpp"
#include "runtime/libcint_direct_shell_evaluator.hpp"

namespace {

struct Options {
  std::string input_path;
  int one_shell_left = 0;
  int one_shell_right = 0;
  int two_shell_i = 0;
  int two_shell_j = 0;
  int two_shell_k = 0;
  int two_shell_l = 0;
};

void print_usage() {
  std::cerr << "usage: check_direct_libcint_smoke <input.xmi> "
               "[--one-shell-pair i j] [--two-shell-quartet i j k l]\n";
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
    if (argument_name == "--one-shell-pair") {
      if (argument_index + 1 >= argc) {
        throw std::invalid_argument("--one-shell-pair expects 2 integers");
      }
      options.one_shell_left = std::stoi(argv[argument_index++]);
      options.one_shell_right = std::stoi(argv[argument_index++]);
      continue;
    }
    if (argument_name == "--two-shell-quartet") {
      if (argument_index + 3 >= argc) {
        throw std::invalid_argument("--two-shell-quartet expects 4 integers");
      }
      options.two_shell_i = std::stoi(argv[argument_index++]);
      options.two_shell_j = std::stoi(argv[argument_index++]);
      options.two_shell_k = std::stoi(argv[argument_index++]);
      options.two_shell_l = std::stoi(argv[argument_index++]);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + argument_name);
  }
  return options;
}

std::uint64_t encode_integral_key(int i, int j, int k, int l) {
  return (static_cast<std::uint64_t>(static_cast<std::uint16_t>(i)) << 48) |
         (static_cast<std::uint64_t>(static_cast<std::uint16_t>(j)) << 32) |
         (static_cast<std::uint64_t>(static_cast<std::uint16_t>(k)) << 16) |
         static_cast<std::uint64_t>(static_cast<std::uint16_t>(l));
}

int pair_index(int first, int second) {
  if (first >= second) {
    return first * (first + 1) / 2 + second;
  }
  return second * (second + 1) / 2 + first;
}

struct CanonicalIntegralIndex {
  int i = 0;
  int j = 0;
  int k = 0;
  int l = 0;
};

CanonicalIntegralIndex canonicalize_integral_index(int i, int j, int k, int l) {
  if (i < j) {
    std::swap(i, j);
  }
  if (k < l) {
    std::swap(k, l);
  }
  if (pair_index(i, j) < pair_index(k, l)) {
    std::swap(i, k);
    std::swap(j, l);
  }
  return {i, j, k, l};
}

std::unordered_map<std::uint64_t, double> build_two_electron_lookup(
    const xmvb::vb::AoIntegralInput& ao_integral_input) {
  std::unordered_map<std::uint64_t, double> lookup;
  lookup.reserve(ao_integral_input.ao_two_electron_integral_values.size());
  for (std::size_t integral_index = 0;
       integral_index < ao_integral_input.ao_two_electron_integral_values.size();
       ++integral_index) {
    const int i = ao_integral_input.ao_two_electron_integral_indices[integral_index * 4];
    const int j = ao_integral_input.ao_two_electron_integral_indices[integral_index * 4 + 1];
    const int k = ao_integral_input.ao_two_electron_integral_indices[integral_index * 4 + 2];
    const int l = ao_integral_input.ao_two_electron_integral_indices[integral_index * 4 + 3];
    lookup.emplace(
        encode_integral_key(i, j, k, l),
        ao_integral_input.ao_two_electron_integral_values[integral_index]);
  }
  return lookup;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    const auto load_result = xmvb::vb::load_cpp_vb_input_with_timings(options.input_path);
    const auto& input = load_result.input;
    xmvb::vb::LibcintDirectShellEvaluator evaluator(input.libcint_input);

    const auto core_h_block = evaluator.evaluate_core_hamiltonian_shell_pair(
        options.one_shell_left,
        options.one_shell_right);
    const int n_basis_functions = input.ao_integral_input.n_basis_functions;
    double max_core_h_abs_diff = 0.0;
    double max_core_h_abs_value = 0.0;
    for (int column = 0; column < core_h_block.right_ao_count; ++column) {
      for (int row = 0; row < core_h_block.left_ao_count; ++row) {
        const std::size_t local_index =
            column * core_h_block.left_ao_count + row;
        const std::size_t global_index =
            core_h_block.right_ao_offset + column * n_basis_functions +
            core_h_block.left_ao_offset + row;
        const double direct_value = core_h_block.values[local_index];
        const double stored_value =
            input.ao_integral_input.ao_core_hamiltonian_matrix[global_index];
        max_core_h_abs_diff =
            std::max(max_core_h_abs_diff, std::abs(direct_value - stored_value));
        max_core_h_abs_value =
            std::max(max_core_h_abs_value, std::max(std::abs(direct_value), std::abs(stored_value)));
      }
    }

    const auto two_electron_lookup =
        build_two_electron_lookup(input.ao_integral_input);
    const auto quartet = evaluator.evaluate_two_electron_shell_quartet(
        options.two_shell_i,
        options.two_shell_j,
        options.two_shell_k,
        options.two_shell_l);

    double max_eri_abs_diff = 0.0;
    double max_eri_abs_value = 0.0;
    int significant_eri_entries = 0;
    for (int s = 0; s < quartet.ao_count_l; ++s) {
      for (int r = 0; r < quartet.ao_count_k; ++r) {
        for (int q = 0; q < quartet.ao_count_j; ++q) {
          for (int p = 0; p < quartet.ao_count_i; ++p) {
            const std::size_t local_index =
                p +
                q * quartet.ao_count_i +
                r * quartet.ao_count_i * quartet.ao_count_j +
                s * quartet.ao_count_i * quartet.ao_count_j *
                    quartet.ao_count_k;
            const int i = quartet.ao_offset_i + p;
            const int j = quartet.ao_offset_j + q;
            const int k = quartet.ao_offset_k + r;
            const int l = quartet.ao_offset_l + s;
            const auto canonical = canonicalize_integral_index(i, j, k, l);
            const auto lookup_iterator = two_electron_lookup.find(
                encode_integral_key(canonical.i, canonical.j, canonical.k, canonical.l));
            const double direct_value = quartet.values[local_index];
            const double stored_value =
                lookup_iterator == two_electron_lookup.end() ? 0.0 : lookup_iterator->second;
            const double abs_diff = std::abs(direct_value - stored_value);
            max_eri_abs_diff = std::max(max_eri_abs_diff, abs_diff);
            max_eri_abs_value =
                std::max(max_eri_abs_value, std::max(std::abs(direct_value), std::abs(stored_value)));
            if (std::abs(direct_value) >= 1.0e-10 || std::abs(stored_value) >= 1.0e-10) {
              ++significant_eri_entries;
            }
          }
        }
      }
    }

    std::cout << std::setprecision(12);
    std::cout << "n_atoms = " << input.libcint_input.n_atoms << '\n';
    std::cout << "n_shells = " << input.libcint_input.n_shells << '\n';
    std::cout << "n_basis_functions = " << evaluator.n_basis_functions() << '\n';
    std::cout << "one_shell_pair = "
              << options.one_shell_left << ' ' << options.one_shell_right << '\n';
    std::cout << "one_shell_pair_ao_offsets = "
              << core_h_block.left_ao_offset << ' ' << core_h_block.right_ao_offset << '\n';
    std::cout << "one_shell_pair_ao_counts = "
              << core_h_block.left_ao_count << ' ' << core_h_block.right_ao_count << '\n';
    std::cout << "core_h_max_abs_diff = " << max_core_h_abs_diff << '\n';
    std::cout << "core_h_max_abs_value = " << max_core_h_abs_value << '\n';
    std::cout << "two_shell_quartet = "
              << options.two_shell_i << ' '
              << options.two_shell_j << ' '
              << options.two_shell_k << ' '
              << options.two_shell_l << '\n';
    std::cout << "two_shell_quartet_ao_offsets = "
              << quartet.ao_offset_i << ' '
              << quartet.ao_offset_j << ' '
              << quartet.ao_offset_k << ' '
              << quartet.ao_offset_l << '\n';
    std::cout << "two_shell_quartet_ao_counts = "
              << quartet.ao_count_i << ' '
              << quartet.ao_count_j << ' '
              << quartet.ao_count_k << ' '
              << quartet.ao_count_l << '\n';
    std::cout << "two_electron_significant_entries = " << significant_eri_entries << '\n';
    std::cout << "eri_max_abs_diff = " << max_eri_abs_diff << '\n';
    std::cout << "eri_max_abs_value = " << max_eri_abs_value << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
