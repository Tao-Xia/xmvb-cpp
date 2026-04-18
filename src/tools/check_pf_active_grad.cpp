#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/linear_algebra/generalized_eigensolver.hpp"
#include "pfaffian_vbscf/matrices/pf_matrix_builder.hpp"
#include "pfaffian_vbscf/scf/pf_active_grad_eval.hpp"
#include "pfaffian_vbscf/scf/pf_basis_factory.hpp"
#include "pfaffian_vbscf/scf/pf_spin_adapted_active_grad_eval.hpp"
#include "pfaffian_vbscf/scf/pf_spin_adapted_basis_factory.hpp"
#include "pfaffian_vbscf/scf/pf_spin_adapted_scf_eval.hpp"
#include "pfaffian_vbscf/types/pf_active_space_data.hpp"
#include "pfaffian_vbscf/types/pf_pair_profile.hpp"
#include "runtime/cpp_vb_input_loader.hpp"

namespace {

enum class Component {
  Overlap,
  OneElectron,
  TwoElectron,
};

struct Options {
  std::string input_path;
  Component component = Component::OneElectron;
  int k = 3;
  int seed = 20260328;
  int count = 8;
  double step = 1.0e-6;
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
  std::cerr << "usage: check_pf_active_grad <input.xmi> "
               "[--component overlap|one_electron|two_electron] "
               "[--k K] [--seed S] [--count N] [--step h] "
               "[--two-electron-mode auto|exact|ri] "
               "[--spin-adapted] [--spin-multiplicity mult] [--ms-twice 2Ms]"
               " [--profile]\n";
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
    if (name == "--component") {
      if (value == "overlap") {
        opt.component = Component::Overlap;
      } else if (value == "one_electron") {
        opt.component = Component::OneElectron;
      } else if (value == "two_electron") {
        opt.component = Component::TwoElectron;
      } else {
        throw std::invalid_argument("invalid component: " + value);
      }
      continue;
    }
    if (name == "--k") {
      opt.k = std::stoi(value);
      continue;
    }
    if (name == "--seed") {
      opt.seed = std::stoi(value);
      continue;
    }
    if (name == "--count") {
      opt.count = std::stoi(value);
      continue;
    }
    if (name == "--step") {
      opt.step = std::stod(value);
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

  if (opt.k <= 0) {
    throw std::invalid_argument("--k must be positive");
  }
  if (opt.count <= 0) {
    throw std::invalid_argument("--count must be positive");
  }
  if (opt.step <= 0.0) {
    throw std::invalid_argument("--step must be positive");
  }
  return opt;
}

void print_profile(
    const xmvb::pfaffian_vbscf::PfPairProfile& profile,
    const std::string& prefix,
    bool grad_path) {
  if (!grad_path) {
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
    std::cout << prefix << ".forward_avg_samples_per_pair = "
              << samples_per_pair << '\n';
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
    return;
  }

  const double other_dt = std::max(
      0.0,
      profile.grad_pair_total_dt -
          profile.grad_source_dt -
          profile.grad_same_spin_dt -
          profile.grad_mix_dt -
          profile.grad_sso_dt);
  const double samples_per_pair =
      (profile.grad_pair_calls > 0)
          ? static_cast<double>(profile.grad_sample_calls) /
                static_cast<double>(profile.grad_pair_calls)
          : 0.0;
  const double mix_frac =
      (profile.grad_pair_total_dt > 0.0)
          ? profile.grad_mix_dt / profile.grad_pair_total_dt
          : 0.0;
  const double non_mix_frac =
      (profile.grad_pair_total_dt > 0.0)
          ? (profile.grad_pair_total_dt - profile.grad_mix_dt) /
                profile.grad_pair_total_dt
          : 0.0;
  const double mix_only_speedup_ceiling =
      (non_mix_frac > 0.0) ? 1.0 / non_mix_frac : 0.0;

  std::cout << prefix << ".grad_pair_calls = " << profile.grad_pair_calls
            << '\n';
  std::cout << prefix << ".grad_sample_calls = " << profile.grad_sample_calls
            << '\n';
  std::cout << prefix << ".grad_avg_samples_per_pair = " << samples_per_pair
            << '\n';
  std::cout << prefix << ".grad_pair_total_dt = " << profile.grad_pair_total_dt
            << '\n';
  std::cout << prefix << ".grad_source_dt = " << profile.grad_source_dt << '\n';
  std::cout << prefix << ".grad_same_spin_dt = " << profile.grad_same_spin_dt
            << '\n';
  std::cout << prefix << ".grad_mix_dt = " << profile.grad_mix_dt << '\n';
  std::cout << prefix << ".grad_sso_dt = " << profile.grad_sso_dt << '\n';
  std::cout << prefix << ".grad_other_dt = " << other_dt << '\n';
  std::cout << prefix << ".grad_mix_fraction = " << mix_frac << '\n';
  std::cout << prefix << ".grad_mix_only_speedup_ceiling = "
            << mix_only_speedup_ceiling << '\n';
}

/**
 * @brief Evaluates the total energy from explicit active-space matrices.
 */
double eval_total_energy(
    const xmvb::pfaffian_vbscf::PfBasisData& basis,
    const xmvb::pfaffian_vbscf::PfActiveGradResult& baseline,
    const std::vector<double>& sso,
    const std::vector<double>& hho,
    const std::vector<double>& ggo,
    double e_nuc) {
  xmvb::pfaffian_vbscf::PfActiveSpaceData act;
  act.n_active_orbitals = basis.n_active_orbitals;
  act.n_alpha = basis.n_alpha;
  act.n_beta = basis.n_beta;
  act.sso = sso;
  act.hho = hho;
  act.ggo = ggo;

  xmvb::pfaffian_vbscf::PfScfEval eval;
  return eval.eval_active_space(act, basis, e_nuc, baseline.scf_result.e_ref).e_tot;
}

double eval_spin_adapted_total_energy(
    const xmvb::pfaffian_vbscf::PfSpinAdaptedBasisData& basis,
    const xmvb::pfaffian_vbscf::PfActiveGradResult& baseline,
    const std::vector<double>& sso,
    const std::vector<double>& hho,
    const std::vector<double>& ggo,
    double e_nuc) {
  xmvb::pfaffian_vbscf::PfActiveSpaceData act;
  act.n_active_orbitals = basis.primitive_basis.n_active_orbitals;
  act.n_alpha = basis.primitive_basis.n_alpha;
  act.n_beta = basis.primitive_basis.n_beta;
  act.sso = sso;
  act.hho = hho;
  act.ggo = ggo;

  xmvb::pfaffian_vbscf::PfSpinAdaptedScfEval eval;
  return eval.eval_active_space(act, basis, e_nuc, baseline.scf_result.e_ref).e_tot;
}

/**
 * @brief Returns the selected gradient vector.
 */
const std::vector<double>& component_grad(
    const xmvb::pfaffian_vbscf::PfActiveGradResult& result,
    Component component) {
  switch (component) {
    case Component::Overlap:
      return result.sso_grad;
    case Component::OneElectron:
      return result.hho_grad;
    case Component::TwoElectron:
      return result.ggo_grad;
  }
  throw std::invalid_argument("unknown component");
}

/**
 * @brief Returns a user-facing component name.
 */
const char* component_name(Component component) {
  switch (component) {
    case Component::Overlap:
      return "overlap";
    case Component::OneElectron:
      return "one_electron";
    case Component::TwoElectron:
      return "two_electron";
  }
  return "unknown";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opt = parse_args(argc, argv);
    if (opt.two_electron_mode == xmvb::vb::PfTwoElectronMode::ResolutionOfIdentity) {
      throw std::invalid_argument(
          "check_pf_active_grad does not support RI finite differences; "
          "use check_pf_orbital_grad or an RI-specific active-space checker instead");
    }
    auto load = xmvb::vb::load_cpp_vb_input_with_timings(opt.input_path);
    load.input.pf_two_electron_mode = opt.two_electron_mode;
    if (load.input.structure_data.alpha_det.empty() ||
        load.input.structure_data.beta_det.empty()) {
      throw std::runtime_error("determinant list is empty");
    }
    xmvb::pfaffian_vbscf::PfBasisFactoryOptions basis_options;
    basis_options.n_states = opt.k;
    basis_options.seed = opt.seed;
    xmvb::pfaffian_vbscf::PfActiveGradResult result;
    bool use_spin_adapted = opt.spin_adapted;
    xmvb::pfaffian_vbscf::PfBasisData basis;
    xmvb::pfaffian_vbscf::PfSpinAdaptedBasisData spin_basis;
    if (use_spin_adapted) {
      xmvb::pfaffian_vbscf::PfSpinAdaptedBasisFactoryOptions spin_options;
      spin_options.n_structures = opt.k;
      spin_options.seed = opt.seed;
      spin_options.target_spin_multiplicity = opt.spin_multiplicity;
      spin_options.ms_twice = opt.ms_twice;
      spin_basis =
          xmvb::pfaffian_vbscf::build_structure_pf_spin_adapted_basis(
              load.input,
              load.raw_structure_data,
              spin_options);

      xmvb::pfaffian_vbscf::PfSpinAdaptedActiveGradEval eval;
      result = eval.eval(load.input, spin_basis, load.nuclear_repulsion_energy);
      basis = spin_basis.primitive_basis;
    } else {
      basis =
          xmvb::pfaffian_vbscf::build_structure_pf_basis(
              load.input,
              load.raw_structure_data,
              basis_options);

      xmvb::pfaffian_vbscf::PfActiveGradEval eval;
      result = eval.eval(load.input, basis, load.nuclear_repulsion_energy);
    }
    const auto& grad = component_grad(result, opt.component);

    std::vector<std::pair<double, int>> ranked;
    ranked.reserve(grad.size());
    for (std::size_t i = 0; i < grad.size(); ++i) {
      ranked.emplace_back(std::abs(grad[i]), static_cast<int>(i));
    }
    std::sort(
        ranked.begin(),
        ranked.end(),
        [](const auto& lhs, const auto& rhs) {
          if (lhs.first != rhs.first) {
            return lhs.first > rhs.first;
          }
          return lhs.second < rhs.second;
        });

    const int n_report = std::min(opt.count, static_cast<int>(ranked.size()));
    std::cout << std::setprecision(12);
    std::cout << "component = " << component_name(opt.component) << '\n';
    std::cout << "spin_adapted = " << (use_spin_adapted ? "true" : "false")
              << '\n';
    if (use_spin_adapted) {
      std::cout << "spin_multiplicity = " << spin_basis.spin_multiplicity << '\n';
      std::cout << "ms_twice = " << spin_basis.ms_twice << '\n';
      std::cout << "primitive_k = " << spin_basis.primitive_basis.n_states << '\n';
      std::cout << "k = " << spin_basis.n_states << '\n';
    } else {
      std::cout << "k = " << basis.n_states << '\n';
    }
    std::cout << "seed = " << opt.seed << '\n';
    std::cout << "two_electron_mode = "
              << two_electron_mode_name(opt.two_electron_mode) << '\n';
    std::cout << "initial_total_energy = " << result.scf_result.e_tot << '\n';
    std::cout << "finite_difference_step = " << opt.step << '\n';
    std::cout << "reported_entries = " << n_report << '\n';
    if (opt.profile) {
      std::cout << "total_dt = " << result.total_dt << '\n';
      std::cout << "orbital_prepare_dt = " << result.orbital_prepare_dt << '\n';
      std::cout << "ao_h1e_build_dt = " << result.ao_h1e_build_dt << '\n';
      std::cout << "active_h1e_build_dt = " << result.active_h1e_build_dt << '\n';
      std::cout << "active_2e_build_dt = " << result.active_2e_build_dt << '\n';
      std::cout << "matrix_dt = " << result.matrix_dt << '\n';
      std::cout << "adj_dt = " << result.adj_dt << '\n';
      print_profile(result.scf_result.mats.pair_profile, "forward_pair_profile", false);
      print_profile(result.pair_grad_profile, "grad_pair_profile", true);
    }

    for (int i = 0; i < n_report; ++i) {
      const int idx = ranked[xmvb::to_size(i)].second;
      std::vector<double> plus_one = result.act_h1e_result.h1e_act;
      std::vector<double> minus_one = result.act_h1e_result.h1e_act;
      std::vector<double> plus_two =
          result.act_eri_result.packed_active_two_electron_integrals;
      std::vector<double> minus_two =
          result.act_eri_result.packed_active_two_electron_integrals;
      std::vector<double> plus_overlap = result.sso;
      std::vector<double> minus_overlap = result.sso;

      switch (opt.component) {
        case Component::Overlap:
          plus_overlap[xmvb::to_size(idx)] += opt.step;
          minus_overlap[xmvb::to_size(idx)] -= opt.step;
          break;
        case Component::OneElectron:
          plus_one[xmvb::to_size(idx)] += opt.step;
          minus_one[xmvb::to_size(idx)] -= opt.step;
          break;
        case Component::TwoElectron:
          plus_two[xmvb::to_size(idx)] += opt.step;
          minus_two[xmvb::to_size(idx)] -= opt.step;
          break;
      }

      const double plus_e = use_spin_adapted
          ? eval_spin_adapted_total_energy(
                spin_basis,
                result,
                plus_overlap,
                plus_one,
                plus_two,
                load.nuclear_repulsion_energy)
          : eval_total_energy(
                basis,
                result,
                plus_overlap,
                plus_one,
                plus_two,
                load.nuclear_repulsion_energy);
      const double minus_e = use_spin_adapted
          ? eval_spin_adapted_total_energy(
                spin_basis,
                result,
                minus_overlap,
                minus_one,
                minus_two,
                load.nuclear_repulsion_energy)
          : eval_total_energy(
                basis,
                result,
                minus_overlap,
                minus_one,
                minus_two,
                load.nuclear_repulsion_energy);
      const double fd = (plus_e - minus_e) / (2.0 * opt.step);
      const double analytic = grad[xmvb::to_size(idx)];
      const double abs_err = std::abs(analytic - fd);
      const double rel_err = abs_err / std::max(1.0, std::abs(fd));

      std::cout << "entry[" << i << "]"
                << " index=" << idx
                << " analytic=" << analytic
                << " fd=" << fd
                << " abs_error=" << abs_err
                << " rel_error=" << rel_err
                << '\n';
    }

    return 0;
  } catch (const std::exception& err) {
    std::cerr << err.what() << '\n';
    return 1;
  }
}
