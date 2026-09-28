#include <iostream>
#include <stdexcept>
#include <string>

#include "input/loading/loader.hpp"
#include "tools/pair_conditioning_census.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "vbscf/orbitals/preparation/preparer.hpp"

namespace {

struct Options {
  std::string input_path;
  long long max_pairs = 1000000;
  int worst_pairs = 8;
};

void print_usage() {
  std::cerr
      << "usage: diagnose_pair_conditioning <input.xmi>"
      << " [--max-pairs N] [--worst-pairs N]\n"
      << "       N=0 scans the complete ordered pair population\n";
}

long long parse_nonnegative(const char* text, const char* option) {
  const long long value = std::stoll(text);
  if (value < 0) {
    throw std::invalid_argument(std::string(option) + " must be non-negative");
  }
  return value;
}

Options parse_options(int argc, char** argv) {
  if (argc < 2) {
    print_usage();
    throw std::invalid_argument("missing input path");
  }
  Options options;
  options.input_path = argv[1];
  for (int index = 2; index < argc; ++index) {
    const std::string name = argv[index];
    if (name == "--max-pairs" || name == "--worst-pairs") {
      if (index + 1 >= argc) {
        throw std::invalid_argument(name + " requires a value");
      }
      const long long value = parse_nonnegative(argv[++index], name.c_str());
      if (name == "--max-pairs") {
        options.max_pairs = value;
      } else {
        options.worst_pairs = static_cast<int>(value);
      }
      continue;
    }
    throw std::invalid_argument("unknown argument: " + name);
  }
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const auto loaded = xmvb::vb::load_vbscf_input_with_timings(
        options.input_path);
    const auto orbitals = xmvb::vb::ActiveSpaceOrbitalPreparer().prepare(
        loaded.input.orbital_preparation_input);
    const int n_active_orbitals =
        loaded.input.orbital_preparation_input.n_active_orbitals;
    const auto alpha = xmvb::vb::build_spin_determinant_reuse_table(
        loaded.input.structure_data.alpha_det).unique_determinants;
    const auto beta = xmvb::vb::build_spin_determinant_reuse_table(
        loaded.input.structure_data.beta_det).unique_determinants;

    const auto alpha_census = xmvb::tools::run_pair_conditioning_census(
        alpha,
        orbitals.active_orbital_overlap_matrix,
        n_active_orbitals,
        options.max_pairs,
        options.worst_pairs);
    const bool shared_strings = alpha == beta;
    const auto beta_census = shared_strings
        ? alpha_census
        : xmvb::tools::run_pair_conditioning_census(
              beta,
              orbitals.active_orbital_overlap_matrix,
              n_active_orbitals,
              options.max_pairs,
              options.worst_pairs);

    std::cout << "input_path = " << options.input_path << '\n';
    std::cout << "n_active_orbitals = " << n_active_orbitals << '\n';
    std::cout << "alpha_beta_share_strings = " << (shared_strings ? 1 : 0)
              << '\n';
    xmvb::tools::print_pair_conditioning_census(
        "alpha", alpha, alpha_census, std::cout);
    xmvb::tools::print_pair_conditioning_census(
        "beta", beta, beta_census, std::cout);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "diagnose_pair_conditioning: " << error.what() << '\n';
    return 1;
  }
}
