#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pfaffian_vbscf/scf/pf_basis_factory.hpp"
#include "pfaffian_vbscf/scf/pf_orbital_grad_eval.hpp"
#include "pfaffian_vbscf/scf/pf_spin_adapted_basis_factory.hpp"
#include "pfaffian_vbscf/scf/pf_spin_adapted_scf_eval.hpp"
#include "pfaffian_vbscf/scf/pf_scf_eval.hpp"
#include "runtime/cpp_vb_input_loader.hpp"

namespace {

struct Options {
  std::string input_path;
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

int get_sparse_coefficient_count(
    const xmvb::vb::OrbitalPreparationInput& input,
    int orbital_index) {
  const int n_basis_functions = input.n_basis_functions;
  const int explicit_count =
      input.orbital_basis_counts[orbital_index];
  if (explicit_count > 1) {
    return explicit_count;
  }

  int count = 0;
  while (count < n_basis_functions) {
    const int basis_index =
        input.orbital_basis_index_table[
            orbital_index * n_basis_functions + count];
    if (basis_index == 0) {
      break;
    }
    ++count;
  }
  return count;
}

std::vector<int> collect_differentiable_parameter_indices(
    const xmvb::vb::OrbitalPreparationInput& input) {
  std::vector<int> out;
  for (int orbital = 0; orbital < input.n_orbitals; ++orbital) {
    const int count = get_sparse_coefficient_count(input, orbital);
    for (int coef = 0; coef < count; ++coef) {
      out.push_back(orbital * input.n_basis_functions + coef);
    }
  }
  return out;
}

/**
 * @brief Prints tool usage.
 */
void print_usage() {
  std::cerr << "usage: check_pf_orbital_grad <input.xmi> "
               "[--k K] [--seed S] [--count N] [--step h] "
               "[--two-electron-mode auto|exact|ri] "
               "[--spin-adapted] [--spin-multiplicity mult] [--ms-twice 2Ms] "
               "[--profile]\n";
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
  if (opt.count < 0) {
    throw std::invalid_argument("--count must be non-negative");
  }
  if (opt.step <= 0.0) {
    throw std::invalid_argument("--step must be positive");
  }
  return opt;
}

double eval_total_energy(
    const xmvb::vb::CppVbInput& input,
    const xmvb::pfaffian_vbscf::PfBasisData& basis,
    double e_nuc) {
  xmvb::pfaffian_vbscf::PfScfEval eval;
  return eval.eval(input, basis, e_nuc).e_tot;
}

double eval_spin_adapted_total_energy(
    const xmvb::vb::CppVbInput& input,
    const xmvb::pfaffian_vbscf::PfSpinAdaptedBasisData& basis,
    double e_nuc) {
  xmvb::pfaffian_vbscf::PfSpinAdaptedScfEval eval;
  return eval.eval(input, basis, e_nuc).e_tot;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opt = parse_args(argc, argv);
    auto load = xmvb::vb::load_cpp_vb_input_with_timings(opt.input_path);
    load.input.pf_two_electron_mode = opt.two_electron_mode;
    if (load.input.structure_data.alpha_det.empty() ||
        load.input.structure_data.beta_det.empty()) {
      throw std::runtime_error("determinant list is empty");
    }
    xmvb::pfaffian_vbscf::PfBasisFactoryOptions basis_options;
    basis_options.n_states = opt.k;
    basis_options.seed = opt.seed;
    xmvb::pfaffian_vbscf::PfOrbitalGradEval eval;
    xmvb::pfaffian_vbscf::PfOrbitalGradResult result;
    xmvb::pfaffian_vbscf::PfBasisData basis;
    xmvb::pfaffian_vbscf::PfSpinAdaptedBasisData spin_basis;
    bool use_spin_adapted = opt.spin_adapted;
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
      result = eval.eval(load.input, spin_basis, load.nuclear_repulsion_energy);
      basis = spin_basis.primitive_basis;
    } else {
      basis =
          xmvb::pfaffian_vbscf::build_structure_pf_basis(
              load.input,
              load.raw_structure_data,
              basis_options);
      result = eval.eval(load.input, basis, load.nuclear_repulsion_energy);
    }
    const auto& grad = result.sparse_orbital_energy_gradient;
    const auto differentiable =
        collect_differentiable_parameter_indices(load.input.orbital_preparation_input);

    std::vector<std::pair<double, int>> ranked;
    ranked.reserve(differentiable.size());
    for (const int idx : differentiable) {
      ranked.emplace_back(std::abs(grad[idx]), idx);
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
    std::cout << "reported_parameters = " << n_report << '\n';
    if (opt.profile) {
      std::cout << "total_dt = " << result.total_dt << '\n';
      std::cout << "forward_wall_time_seconds = "
                << result.forward_wall_time_seconds << '\n';
      std::cout << "active_space_grad_dt = " << result.active_space_grad_dt << '\n';
      std::cout << "backprop_wall_time_seconds = "
                << result.backprop_wall_time_seconds << '\n';
      std::cout << "matrix_backprop_dt = " << result.matrix_backprop_dt << '\n';
      std::cout << "two_electron_backprop_dt = "
                << result.two_electron_backprop_dt << '\n';
      std::cout << "ao_h1e_backprop_dt = " << result.ao_h1e_backprop_dt << '\n';
      std::cout << "orbital_backprop_dt = " << result.orbital_backprop_dt << '\n';
    }

    for (int i = 0; i < n_report; ++i) {
      const int param_idx = ranked[i].second;
      xmvb::vb::CppVbInput plus_input = load.input;
      xmvb::vb::CppVbInput minus_input = load.input;
      plus_input.orbital_preparation_input.orbital_value_table[param_idx] +=
          opt.step;
      minus_input.orbital_preparation_input.orbital_value_table[param_idx] -=
          opt.step;

      const double plus_e = use_spin_adapted
          ? eval_spin_adapted_total_energy(
                plus_input,
                spin_basis,
                load.nuclear_repulsion_energy)
          : eval_total_energy(
                plus_input,
                basis,
                load.nuclear_repulsion_energy);
      const double minus_e = use_spin_adapted
          ? eval_spin_adapted_total_energy(
                minus_input,
                spin_basis,
                load.nuclear_repulsion_energy)
          : eval_total_energy(
                minus_input,
                basis,
                load.nuclear_repulsion_energy);
      const double fd = (plus_e - minus_e) / (2.0 * opt.step);
      const double analytic = grad[param_idx];
      const double abs_err = std::abs(analytic - fd);
      const double rel_err = abs_err / std::max(1.0, std::abs(fd));

      std::cout << "parameter[" << i << "]"
                << " index=" << param_idx
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
