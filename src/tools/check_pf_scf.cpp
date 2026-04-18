#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "pfaffian_vbscf/scf/pf_basis_factory.hpp"
#include "pfaffian_vbscf/scf/pf_spin_adapted_basis_factory.hpp"
#include "pfaffian_vbscf/scf/pf_spin_adapted_scf_eval.hpp"
#include "pfaffian_vbscf/scf/pf_scf_eval.hpp"
#include "pfaffian_vbscf/types/pf_pair_profile.hpp"
#include "runtime/cpp_vb_input_loader.hpp"

namespace {

struct Options {
  std::string input_path;
  int k = 0;
  int seed = 20260328;
  double pairing_noise = 0.0;
  bool profile = false;
  bool spin_adapted = false;
  int spin_multiplicity = 0;
  int ms_twice = std::numeric_limits<int>::min();
  xmvb::vb::PfTwoElectronMode two_electron_mode =
      xmvb::vb::PfTwoElectronMode::Auto;
};

xmvb::vb::PfTwoElectronMode parse_two_electron_mode(
    const std::string& value) {
  if (value == "auto") {
    return xmvb::vb::PfTwoElectronMode::Auto;
  }
  if (value == "exact") {
    return xmvb::vb::PfTwoElectronMode::Exact;
  }
  if (value == "ri") {
    return xmvb::vb::PfTwoElectronMode::ResolutionOfIdentity;
  }
  throw std::invalid_argument("invalid two-electron mode: " + value);
}

const char* two_electron_mode_name(
    xmvb::vb::PfTwoElectronMode mode) {
  switch (mode) {
    case xmvb::vb::PfTwoElectronMode::Auto:
      return "auto";
    case xmvb::vb::PfTwoElectronMode::Exact:
      return "exact";
    case xmvb::vb::PfTwoElectronMode::ResolutionOfIdentity:
      return "ri";
  }
  return "unknown";
}

/**
 * @brief Prints tool usage.
 */
void print_usage() {
  std::cerr << "usage: check_pf_scf <input.xmi> [--k K] [--seed S]"
               " [--pairing-noise x] [--spin-adapted]"
               " [--two-electron-mode auto|exact|ri]"
               " [--spin-multiplicity mult] [--ms-twice 2Ms] [--profile]\n"
               "  --k 0 uses all selected raw VB structures.\n";
}

/**
 * @brief Parses command-line options.
 */
Options parse_args(int argc, char** argv) {
  if (argc < 2) {
    print_usage();
    throw std::invalid_argument("invalid argument count");
  }

  Options opt;
  opt.input_path = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string name = argv[i];
    if (name == "--profile") {
      opt.profile = true;
      continue;
    }
    if (name == "--spin-adapted") {
      opt.spin_adapted = true;
      continue;
    }
    if (i + 1 >= argc) {
      print_usage();
      throw std::invalid_argument("missing value for argument: " + name);
    }
    const std::string value = argv[++i];
    if (name == "--k") {
      opt.k = std::stoi(value);
      continue;
    }
    if (name == "--seed") {
      opt.seed = std::stoi(value);
      continue;
    }
    if (name == "--pairing-noise") {
      opt.pairing_noise = std::stod(value);
      continue;
    }
    if (name == "--two-electron-mode") {
      opt.two_electron_mode = parse_two_electron_mode(value);
      continue;
    }
    if (name == "--spin-multiplicity") {
      opt.spin_multiplicity = std::stoi(value);
      continue;
    }
    if (name == "--ms-twice") {
      opt.ms_twice = std::stoi(value);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + name);
  }

  if (opt.k < 0) {
    throw std::invalid_argument("--k must be non-negative");
  }
  if (opt.pairing_noise < 0.0) {
    throw std::invalid_argument("--pairing-noise must be non-negative");
  }
  return opt;
}

void print_profile(
    const xmvb::pfaffian_vbscf::PfPairProfile& profile,
    const std::string& prefix) {
  const double other_dt = std::max(
      0.0,
      profile.forward_pair_total_dt -
          profile.forward_source_dt -
          profile.forward_same_spin_dt -
          profile.forward_mix_dt);
  const double samples_per_pair =
      (profile.forward_pair_calls > 0)
          ? static_cast<double>(profile.forward_sample_calls) /
                static_cast<double>(profile.forward_pair_calls)
          : 0.0;
  const double mix_frac =
      (profile.forward_pair_total_dt > 0.0)
          ? profile.forward_mix_dt / profile.forward_pair_total_dt
          : 0.0;
  const double non_mix_frac =
      (profile.forward_pair_total_dt > 0.0)
          ? (profile.forward_pair_total_dt - profile.forward_mix_dt) /
                profile.forward_pair_total_dt
          : 0.0;
  const double mix_only_speedup_ceiling =
      (non_mix_frac > 0.0) ? 1.0 / non_mix_frac : 0.0;

  std::cout << prefix << ".forward_pair_calls = " << profile.forward_pair_calls
            << '\n';
  std::cout << prefix << ".forward_sample_calls = "
            << profile.forward_sample_calls << '\n';
  std::cout << prefix << ".forward_avg_samples_per_pair = " << samples_per_pair
            << '\n';
  std::cout << prefix << ".forward_pair_total_dt = "
            << profile.forward_pair_total_dt << '\n';
  std::cout << prefix << ".forward_source_dt = " << profile.forward_source_dt
            << '\n';
  std::cout << prefix << ".forward_same_spin_dt = "
            << profile.forward_same_spin_dt << '\n';
  std::cout << prefix << ".forward_mix_dt = " << profile.forward_mix_dt << '\n';
  std::cout << prefix << ".forward_other_dt = " << other_dt << '\n';
  std::cout << prefix << ".forward_mix_fraction = " << mix_frac << '\n';
  std::cout << prefix << ".forward_mix_only_speedup_ceiling = "
            << mix_only_speedup_ceiling << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opt = parse_args(argc, argv);
    auto load = xmvb::vb::load_cpp_vb_input_with_timings(opt.input_path);
    load.input.pf_two_electron_mode = opt.two_electron_mode;
    const int n_act = load.input.orbital_preparation_input.n_active_orbitals;
    if (load.input.structure_data.alpha_det.empty() ||
        load.input.structure_data.beta_det.empty()) {
      throw std::runtime_error("determinant list is empty");
    }

    const int n_alpha =
        static_cast<int>(load.input.structure_data.alpha_det.front().size());
    const int n_beta =
        static_cast<int>(load.input.structure_data.beta_det.front().size());
    xmvb::pfaffian_vbscf::PfBasisFactoryOptions basis_options;
    basis_options.n_states = opt.k;
    basis_options.seed = opt.seed;
    basis_options.pairing_noise = opt.pairing_noise;
    xmvb::pfaffian_vbscf::PfScfResult result;
    int reported_k = 0;
    int reported_n_alpha = 0;
    int reported_n_beta = 0;
    int primitive_k = 0;
    if (opt.spin_adapted) {
      xmvb::pfaffian_vbscf::PfSpinAdaptedBasisFactoryOptions spin_options;
      spin_options.n_structures = opt.k;
      spin_options.seed = opt.seed;
      spin_options.pairing_noise = opt.pairing_noise;
      spin_options.target_spin_multiplicity = opt.spin_multiplicity;
      spin_options.ms_twice = opt.ms_twice;
      const auto basis =
          xmvb::pfaffian_vbscf::build_structure_pf_spin_adapted_basis(
              load.input,
              load.raw_structure_data,
              spin_options);

      xmvb::pfaffian_vbscf::PfSpinAdaptedScfEval eval;
      result = eval.eval(load.input, basis, load.nuclear_repulsion_energy);
      reported_k = basis.n_states;
      primitive_k = basis.primitive_basis.n_states;
      reported_n_alpha = basis.primitive_basis.n_alpha;
      reported_n_beta = basis.primitive_basis.n_beta;
      std::cout << "spin_adapted = true\n";
      std::cout << "spin_multiplicity = " << basis.spin_multiplicity << '\n';
      std::cout << "ms_twice = " << basis.ms_twice << '\n';
      std::cout << "primitive_k = " << primitive_k << '\n';
    } else {
      const auto basis =
          xmvb::pfaffian_vbscf::build_structure_pf_basis(
              load.input,
              load.raw_structure_data,
              basis_options);

      xmvb::pfaffian_vbscf::PfScfEval eval;
      result = eval.eval(load.input, basis, load.nuclear_repulsion_energy);
      reported_k = basis.n_states;
      reported_n_alpha = basis.n_alpha;
      reported_n_beta = basis.n_beta;
      std::cout << "spin_adapted = false\n";
    }

    std::cout << std::setprecision(12);
    std::cout << "requested_k = ";
    if (opt.k == 0) {
      std::cout << "auto\n";
    } else {
      std::cout << opt.k << '\n';
    }
    std::cout << "k = " << reported_k << '\n';
    std::cout << "seed = " << opt.seed << '\n';
    std::cout << "pairing_noise = " << opt.pairing_noise << '\n';
    std::cout << "two_electron_mode = "
              << two_electron_mode_name(opt.two_electron_mode) << '\n';
    std::cout << "source_raw_structure_count = "
              << load.source_raw_structure_count << '\n';
    std::cout << "selected_raw_structure_count = "
              << load.raw_structure_data.n_structures << '\n';
    std::cout << "expanded_determinant_count = "
              << load.input.structure_data.alpha_det.size() << '\n';
    std::cout << "n_active_orbitals = " << n_act << '\n';
    std::cout << "determinant_n_alpha = " << n_alpha << '\n';
    std::cout << "determinant_n_beta = " << n_beta << '\n';
    std::cout << "n_alpha = " << reported_n_alpha << '\n';
    std::cout << "n_beta = " << reported_n_beta << '\n';
    std::cout << "e_ref = " << result.e_ref << '\n';
    std::cout << "e_ele = " << result.e_ele << '\n';
    std::cout << "e_tot = " << result.e_tot << '\n';
    std::cout << "avg_diag_s = " << result.avg_diag_s << '\n';
    if (!result.evals.empty()) {
      std::cout << "lowest_eval = " << result.evals.front() << '\n';
    }
    if (opt.profile) {
      std::cout << "total_dt = " << result.total_dt << '\n';
      std::cout << "matrix_dt = " << result.matrix_dt << '\n';
      print_profile(result.mats.pair_profile, "pair_profile");
    }
    return 0;
  } catch (const std::exception& err) {
    std::cerr << err.what() << '\n';
    return 1;
  }
}
