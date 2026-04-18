#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "runtime/cpp_vb_input_loader.hpp"
#include "runtime/libcint_ri_integral_provider.hpp"
#include "vb/orbital/active_space_orbital_preparer.hpp"
#include "vb/orbital/active_space_two_electron_builder.hpp"
#include "vb/orbital/ri_active_space_two_electron_builder.hpp"

namespace {

struct Options {
  std::string input_path;
  int level = 2;
  bool use_star = true;
  double metric_eigenvalue_cutoff = 1.0e-10;
};

void print_usage() {
  std::cerr << "usage: compare_ri_active_space_builder <input.xmi> "
               "[--level n] [--no-star] [--metric-cutoff value]\n";
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
    if (argument_name == "--metric-cutoff") {
      if (argument_index >= argc) {
        throw std::invalid_argument("--metric-cutoff expects 1 float");
      }
      options.metric_eigenvalue_cutoff = std::stod(argv[argument_index++]);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + argument_name);
  }
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    const auto load_result = xmvb::vb::load_cpp_vb_input_with_timings(options.input_path);
    const auto& input = load_result.input;

    xmvb::vb::ActiveSpaceOrbitalPreparer orbital_preparer;
    const auto orbital_result =
        orbital_preparer.prepare(input.orbital_preparation_input);

    xmvb::vb::ActiveSpaceTwoElectronBuilder exact_builder;
    const auto exact_result = exact_builder.build(
        input.ao_integral_input,
        orbital_result,
        input.orbital_preparation_input.n_active_orbitals);

    xmvb::vb::LibcintRiIntegralProvider provider;
    xmvb::vb::LibcintRiIntegralProviderOptions provider_options;
    provider_options.auxiliary_basis_options.level = options.level;
    provider_options.auxiliary_basis_options.use_star = options.use_star;
    provider_options.metric_eigenvalue_cutoff = options.metric_eigenvalue_cutoff;
    const auto ao_ri_result =
        input.auxiliary_libcint_input.n_shells > 0
            ? provider.build(
                  input.libcint_input,
                  input.auxiliary_libcint_input,
                  provider_options)
            : provider.build(input.libcint_input, provider_options);

    xmvb::vb::RiActiveSpaceTwoElectronBuilder ri_builder;
    xmvb::vb::RiActiveSpaceTwoElectronBuilderOptions ri_builder_options;
    ri_builder_options.reconstruct_packed_integrals = true;
    const auto ri_result = ri_builder.build(
        ao_ri_result,
        orbital_result,
        input.orbital_preparation_input.n_basis_functions,
        input.orbital_preparation_input.n_active_orbitals,
        ri_builder_options);

    if (exact_result.packed_active_two_electron_integrals.size() !=
        ri_result.packed_active_two_electron_integrals.size()) {
      throw std::runtime_error("exact and RI active integral sizes differ");
    }

    double max_abs_diff = 0.0;
    double rms_diff = 0.0;
    double max_abs_reference = 0.0;
    for (std::size_t index = 0;
         index < exact_result.packed_active_two_electron_integrals.size();
         ++index) {
      const double reference =
          exact_result.packed_active_two_electron_integrals[index];
      const double candidate =
          ri_result.packed_active_two_electron_integrals[index];
      const double abs_diff = std::abs(reference - candidate);
      max_abs_diff = std::max(max_abs_diff, abs_diff);
      rms_diff += abs_diff * abs_diff;
      max_abs_reference = std::max(max_abs_reference, std::abs(reference));
    }
    if (!exact_result.packed_active_two_electron_integrals.empty()) {
      rms_diff = std::sqrt(
          rms_diff /
          static_cast<double>(exact_result.packed_active_two_electron_integrals.size()));
    }

    std::cout << std::setprecision(12);
    std::cout << "n_basis_functions = "
              << input.orbital_preparation_input.n_basis_functions << '\n';
    std::cout << "n_active_orbitals = "
              << input.orbital_preparation_input.n_active_orbitals << '\n';
    std::cout << "auxiliary_level = " << options.level << '\n';
    std::cout << "auxiliary_star = " << (options.use_star ? 1 : 0) << '\n';
    std::cout << "n_auxiliary_functions = " << ao_ri_result.n_auxiliary_functions << '\n';
    std::cout << "n_packed_ao_pairs = " << ao_ri_result.n_packed_ao_pairs << '\n';
    std::cout << "n_active_pair_factors = "
              << ri_result.ri_active_pair_factors.size() << '\n';
    std::cout << "max_abs_reference = " << max_abs_reference << '\n';
    std::cout << "ri_exact_max_abs_diff = " << max_abs_diff << '\n';
    std::cout << "ri_exact_rms_diff = " << rms_diff << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
