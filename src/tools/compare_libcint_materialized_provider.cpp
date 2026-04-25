#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "runtime/cpp_vb_input_loader.hpp"
#include "runtime/libcint_materialized_integral_provider.hpp"

namespace {

struct Options {
  std::string input_path;
  double integral_tolerance = 1.0e-10;
};

void print_usage() {
  std::cerr << "usage: compare_libcint_materialized_provider <input.xmi> "
               "[--integral-tolerance tol]\n";
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
    if (name == "--integral-tolerance") {
      options.integral_tolerance = std::stod(value);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + name);
  }
  if (!(options.integral_tolerance >= 0.0)) {
    throw std::invalid_argument("--integral-tolerance must be non-negative");
  }
  return options;
}

std::uint64_t encode_integral_key(int i, int j, int k, int l) {
  return (static_cast<std::uint64_t>(static_cast<std::uint16_t>(i)) << 48) |
         (static_cast<std::uint64_t>(static_cast<std::uint16_t>(j)) << 32) |
         (static_cast<std::uint64_t>(static_cast<std::uint16_t>(k)) << 16) |
         static_cast<std::uint64_t>(static_cast<std::uint16_t>(l));
}

template <typename ValueBuffer>
std::unordered_map<std::uint64_t, double> build_integral_lookup(
    const ValueBuffer& values,
    const std::vector<int>& indices) {
  std::unordered_map<std::uint64_t, double> lookup;
  lookup.reserve(values.size());
  for (std::size_t integral_index = 0; integral_index < values.size(); ++integral_index) {
    lookup.emplace(
        encode_integral_key(
            indices[integral_index * 4],
            indices[integral_index * 4 + 1],
            indices[integral_index * 4 + 2],
            indices[integral_index * 4 + 3]),
        values[integral_index]);
  }
  return lookup;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    const auto load_result = xmvb::vb::load_cpp_vb_input_with_timings(options.input_path);

    xmvb::vb::LibcintMaterializedIntegralProvider provider;
    const auto provider_buffers = provider.build(
        load_result.input.libcint_input,
        {.integral_tolerance = options.integral_tolerance});

    const auto& legacy = load_result.input.ao_integral_input;
    double max_core_h_abs_diff = 0.0;
    double max_core_h_abs_value = 0.0;
    for (std::size_t index = 0; index < provider_buffers.ao_core_hamiltonian_matrix.size(); ++index) {
      const double provider_value = provider_buffers.ao_core_hamiltonian_matrix[index];
      const double legacy_value = legacy.ao_core_hamiltonian_matrix[index];
      max_core_h_abs_diff =
          std::max(max_core_h_abs_diff, std::abs(provider_value - legacy_value));
      max_core_h_abs_value =
          std::max(max_core_h_abs_value, std::max(std::abs(provider_value), std::abs(legacy_value)));
    }

    const auto provider_lookup = build_integral_lookup(
        provider_buffers.ao_two_electron_integral_values,
        provider_buffers.ao_two_electron_integral_indices);
    const auto legacy_lookup = build_integral_lookup(
        legacy.ao_two_electron_integral_values,
        legacy.ao_two_electron_integral_indices);

    double max_eri_abs_diff = 0.0;
    double max_eri_abs_value = 0.0;
    std::size_t missing_from_provider = 0;
    std::size_t missing_from_legacy = 0;
    for (const auto& [key, legacy_value] : legacy_lookup) {
      const auto provider_it = provider_lookup.find(key);
      const double provider_value =
          provider_it == provider_lookup.end() ? 0.0 : provider_it->second;
      if (provider_it == provider_lookup.end()) {
        ++missing_from_provider;
      }
      max_eri_abs_diff = std::max(max_eri_abs_diff, std::abs(provider_value - legacy_value));
      max_eri_abs_value =
          std::max(max_eri_abs_value, std::max(std::abs(provider_value), std::abs(legacy_value)));
    }
    for (const auto& [key, provider_value] : provider_lookup) {
      const auto legacy_it = legacy_lookup.find(key);
      if (legacy_it == legacy_lookup.end()) {
        ++missing_from_legacy;
        max_eri_abs_diff = std::max(max_eri_abs_diff, std::abs(provider_value));
        max_eri_abs_value = std::max(max_eri_abs_value, std::abs(provider_value));
      }
    }

    std::cout << std::setprecision(12);
    std::cout << "n_basis_functions = " << provider_buffers.n_basis_functions << '\n';
    std::cout << "integral_tolerance = " << options.integral_tolerance << '\n';
    std::cout << "legacy_2e_count = " << legacy.ao_two_electron_integral_values.size() << '\n';
    std::cout << "provider_2e_count = " << provider_buffers.ao_two_electron_integral_values.size()
              << '\n';
    std::cout << "core_h_max_abs_diff = " << max_core_h_abs_diff << '\n';
    std::cout << "core_h_max_abs_value = " << max_core_h_abs_value << '\n';
    std::cout << "eri_max_abs_diff = " << max_eri_abs_diff << '\n';
    std::cout << "eri_max_abs_value = " << max_eri_abs_value << '\n';
    std::cout << "missing_from_provider = " << missing_from_provider << '\n';
    std::cout << "missing_from_legacy = " << missing_from_legacy << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
