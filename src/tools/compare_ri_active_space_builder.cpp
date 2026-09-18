#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "input/loading/loader.hpp"
#include "input/deck/model.hpp"
#include "input/deck/primary_basis.hpp"
#include "libcint/ri_provider.hpp"
#include "vbscf/orbitals/preparation/preparer.hpp"
#include "vbscf/integrals/active/two_electron/construction/builder.hpp"
#include "vbscf/integrals/active/two_electron/construction/ri_builder.hpp"

namespace {

struct Options {
  std::string input_path;
  std::string auxiliary_basis_name;
  double metric_eigenvalue_cutoff = 1.0e-10;
};

void print_usage() {
  std::cerr << "usage: compare_ri_active_space_builder <input.xmi> "
               "[--aux-basis name] [--metric-cutoff value]\n";
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
    const auto load_result = xmvb::vb::load_vbscf_input_with_timings(options.input_path);
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
    provider_options.metric_eigenvalue_cutoff = options.metric_eigenvalue_cutoff;
    const xmvb::vb::InputDeck input_deck =
        xmvb::vb::parse_input_deck_model(options.input_path);
    std::string auxiliary_basis_name = options.auxiliary_basis_name;
    if (auxiliary_basis_name.empty()) {
      auxiliary_basis_name = input_deck.metadata.auxiliary_basis_name;
    }
    if (auxiliary_basis_name.empty()) {
      auxiliary_basis_name = input_deck.metadata.basis_name;
      if (auxiliary_basis_name.size() >= 4 &&
          auxiliary_basis_name.substr(auxiliary_basis_name.size() - 4) ==
              ".gbs") {
        auxiliary_basis_name.resize(auxiliary_basis_name.size() - 4);
      }
      auxiliary_basis_name += "-jkfit";
    }
    const auto auxiliary_input = xmvb::vb::build_input_deck_basis(
        input_deck,
        auxiliary_basis_name).libcint_input;
    const auto ao_ri_result = provider.build(
        input.libcint_input,
        auxiliary_input,
        provider_options);

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
    std::cout << "auxiliary_basis = " << auxiliary_basis_name << '\n';
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
