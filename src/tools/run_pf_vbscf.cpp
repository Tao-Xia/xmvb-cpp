#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "pfaffian_vbscf/scf/pf_basis_factory.hpp"
#include "pfaffian_vbscf/scf/pf_orbital_optimizer.hpp"
#include "pfaffian_vbscf/scf/pf_spin_adapted_basis_factory.hpp"
#include "runtime/cpp_vb_input_loader.hpp"

namespace {

struct Options {
  std::string input_path;
  int k = 0;
  int seed = 20260328;
  double pairing_noise = 0.0;
  int max_iterations = 50;
  double gradient_tolerance = 2.0e-3;
  double energy_tolerance = 1.0e-7;
  double initial_step_size = 1.0;
  double minimum_step_size = 1.0e-7;
  double armijo_constant = 1.0e-4;
  int history_size = 100;
  bool verbose = true;
  bool profile_optimizer = false;
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

void print_usage() {
  std::cerr << "usage: run_pf_vbscf <input.xmi> [--k K] [--seed S]"
               " [--pairing-noise x] [--max-iterations N]"
               " [--gradient-tolerance g] [--energy-tolerance e]"
               " [--initial-step-size a] [--minimum-step-size a_min]"
               " [--armijo c] [--history-size m]"
               " [--two-electron-mode auto|exact|ri]"
               " [--profile-optimizer]"
               " [--spin-adapted] [--spin-multiplicity mult] [--ms-twice 2Ms]"
               " [--quiet]\n"
               "  --k 0 uses all selected raw VB structures.\n";
}

std::string join_ints(const std::vector<int>& values) {
  std::ostringstream stream;
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index > 0) {
      stream << ',';
    }
    stream << values[index];
  }
  return stream.str();
}

Options parse_args(int argc, char** argv) {
  if (argc < 2) {
    print_usage();
    throw std::invalid_argument("invalid argument count");
  }

  Options options;
  options.input_path = argv[1];
  for (int index = 2; index < argc; ++index) {
    const std::string name = argv[index];
    if (name == "--quiet") {
      options.verbose = false;
      continue;
    }
    if (name == "--profile-optimizer") {
      options.profile_optimizer = true;
      continue;
    }
    if (name == "--spin-adapted") {
      options.spin_adapted = true;
      continue;
    }
    if (index + 1 >= argc) {
      print_usage();
      throw std::invalid_argument("missing value for argument: " + name);
    }

    const std::string value = argv[++index];
    if (name == "--k") {
      options.k = std::stoi(value);
      continue;
    }
    if (name == "--seed") {
      options.seed = std::stoi(value);
      continue;
    }
    if (name == "--pairing-noise") {
      options.pairing_noise = std::stod(value);
      continue;
    }
    if (name == "--max-iterations") {
      options.max_iterations = std::stoi(value);
      continue;
    }
    if (name == "--gradient-tolerance") {
      options.gradient_tolerance = std::stod(value);
      continue;
    }
    if (name == "--energy-tolerance") {
      options.energy_tolerance = std::stod(value);
      continue;
    }
    if (name == "--initial-step-size") {
      options.initial_step_size = std::stod(value);
      continue;
    }
    if (name == "--minimum-step-size") {
      options.minimum_step_size = std::stod(value);
      continue;
    }
    if (name == "--armijo") {
      options.armijo_constant = std::stod(value);
      continue;
    }
    if (name == "--history-size") {
      options.history_size = std::stoi(value);
      continue;
    }
    if (name == "--two-electron-mode") {
      options.two_electron_mode = parse_two_electron_mode(value);
      continue;
    }
    if (name == "--spin-multiplicity") {
      options.spin_multiplicity = std::stoi(value);
      continue;
    }
    if (name == "--ms-twice") {
      options.ms_twice = std::stoi(value);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + name);
  }

  if (options.k < 0 || options.max_iterations <= 0 || options.history_size <= 0) {
    throw std::invalid_argument("integer options are invalid");
  }
  if (options.pairing_noise < 0.0 ||
      options.gradient_tolerance <= 0.0 ||
      options.energy_tolerance <= 0.0 ||
      options.initial_step_size <= 0.0 ||
      options.minimum_step_size <= 0.0 ||
      options.armijo_constant <= 0.0 ||
      options.armijo_constant >= 1.0) {
    throw std::invalid_argument("floating-point optimizer controls are invalid");
  }
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_args(argc, argv);
    auto load =
        xmvb::vb::load_cpp_vb_input_with_timings(options.input_path);
    load.input.pf_two_electron_mode = options.two_electron_mode;

    xmvb::pfaffian_vbscf::PfScfOptimizerOptions optimizer_options;
    optimizer_options.max_iterations = options.max_iterations;
    optimizer_options.gradient_tolerance = options.gradient_tolerance;
    optimizer_options.energy_tolerance = options.energy_tolerance;
    optimizer_options.initial_step_size = options.initial_step_size;
    optimizer_options.minimum_step_size = options.minimum_step_size;
    optimizer_options.armijo_constant = options.armijo_constant;
    optimizer_options.history_size = options.history_size;
    optimizer_options.verbose = options.verbose;

    xmvb::pfaffian_vbscf::PfScfOptimizerResult result;
    int reported_k = 0;
    int primitive_k = 0;
    int n_alpha = 0;
    int n_beta = 0;
    int reported_spin_multiplicity = load.raw_structure_data.spin_multiplicity;
    int reported_ms_twice = std::numeric_limits<int>::min();

    if (options.spin_adapted) {
      xmvb::pfaffian_vbscf::PfSpinAdaptedBasisFactoryOptions basis_options;
      basis_options.n_structures = options.k;
      basis_options.seed = options.seed;
      basis_options.pairing_noise = options.pairing_noise;
      basis_options.target_spin_multiplicity = options.spin_multiplicity;
      basis_options.ms_twice = options.ms_twice;
      const auto basis =
          xmvb::pfaffian_vbscf::build_structure_pf_spin_adapted_basis(
              load.input,
              load.raw_structure_data,
              basis_options);

      xmvb::pfaffian_vbscf::PfScfOptimizer optimizer(
          basis,
          optimizer_options);
      result = optimizer.optimize(load.input, load.nuclear_repulsion_energy);
      reported_k = basis.n_states;
      primitive_k = basis.primitive_basis.n_states;
      n_alpha = basis.primitive_basis.n_alpha;
      n_beta = basis.primitive_basis.n_beta;
      reported_spin_multiplicity = basis.spin_multiplicity;
      reported_ms_twice = basis.ms_twice;
    } else {
      xmvb::pfaffian_vbscf::PfBasisFactoryOptions basis_options;
      basis_options.n_states = options.k;
      basis_options.seed = options.seed;
      basis_options.pairing_noise = options.pairing_noise;
      const auto basis =
          xmvb::pfaffian_vbscf::build_structure_pf_basis(
              load.input,
              load.raw_structure_data,
              basis_options);

      xmvb::pfaffian_vbscf::PfScfOptimizer optimizer(
          basis,
          optimizer_options);
      result = optimizer.optimize(load.input, load.nuclear_repulsion_energy);
      reported_k = basis.n_states;
      primitive_k = basis.n_states;
      n_alpha = basis.n_alpha;
      n_beta = basis.n_beta;
    }

    std::cout << std::setprecision(12);
    std::cout << "spin_adapted = "
              << (options.spin_adapted ? "true" : "false") << '\n';
    std::cout << "spin_multiplicity = "
              << reported_spin_multiplicity << '\n';
    if (options.spin_adapted) {
      std::cout << "ms_twice = " << reported_ms_twice << '\n';
      std::cout << "primitive_k = " << primitive_k << '\n';
    }
    std::cout << "requested_k = ";
    if (options.k == 0) {
      std::cout << "auto\n";
    } else {
      std::cout << options.k << '\n';
    }
    std::cout << "k = " << reported_k << '\n';
    std::cout << "seed = " << options.seed << '\n';
    std::cout << "pairing_noise = " << options.pairing_noise << '\n';
    std::cout << "two_electron_mode = "
              << two_electron_mode_name(options.two_electron_mode) << '\n';
    std::cout << "source_raw_structure_count = "
              << load.source_raw_structure_count << '\n';
    std::cout << "selected_raw_structure_count = "
              << load.raw_structure_data.n_structures << '\n';
    std::cout << "expanded_determinant_count = "
              << load.input.structure_data.alpha_det.size() << '\n';
    std::cout << "converged = " << (result.converged ? "true" : "false")
              << '\n';
    std::cout << "termination_reason = " << result.termination_reason << '\n';
    std::cout << "iterations = " << result.n_iterations << '\n';
    std::cout << "n_active_orbitals = "
              << load.input.orbital_preparation_input.n_active_orbitals << '\n';
    std::cout << "n_alpha = " << n_alpha << '\n';
    std::cout << "n_beta = " << n_beta << '\n';
    std::cout << "nuclear_repulsion_energy = "
              << load.nuclear_repulsion_energy << '\n';
    std::cout << "initial_total_energy = "
              << result.initial_total_energy << '\n';
    if (options.profile_optimizer) {
      std::cout << "initial_objective_eval_count = "
                << result.initial_objective_eval_count << '\n';
      std::cout << "total_objective_eval_count = "
                << result.total_objective_eval_count << '\n';
      std::cout << "objective_eval_count_history = "
                << join_ints(result.objective_eval_count_history) << '\n';
      std::cout << "primary_line_search_eval_count_history = "
                << join_ints(result.primary_line_search_eval_count_history) << '\n';
      std::cout << "fallback_line_search_eval_count_history = "
                << join_ints(result.fallback_line_search_eval_count_history) << '\n';
      std::cout << "fallback_used_history = "
                << join_ints(result.fallback_used_history) << '\n';
    }
    std::cout << "final_total_energy = "
              << result.final_total_energy << '\n';
    std::cout << "final_electronic_energy = "
              << result.scf_result.e_ele << '\n';
    std::cout << "final_reference_energy = "
              << result.scf_result.e_ref << '\n';
    std::cout << "final_gradient_inf_norm = "
              << result.final_gradient_inf_norm << '\n';
    std::cout << "final_gradient_l2_norm = "
              << result.final_gradient_l2_norm << '\n';
    std::cout << "final_avg_diag_s = "
              << result.scf_result.avg_diag_s << '\n';
    if (!result.scf_result.evals.empty()) {
      std::cout << "final_lowest_eval = "
                << result.scf_result.evals.front() << '\n';
    }
    std::cout << "input_total_wall_time_seconds = "
              << load.total_seconds << '\n';
    std::cout << "optimization_total_wall_time_seconds = "
              << result.total_wall_time_seconds << '\n';
    return result.converged ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
