#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "pfaffian_vbscf/scf/pf_basis_factory.hpp"
#include "pfaffian_vbscf/scf/pf_orbital_grad_eval.hpp"
#include "pfaffian_vbscf/scf/pf_spin_adapted_basis_factory.hpp"
#include "runtime/cpp_vb_input_loader.hpp"

namespace {

struct Options {
  std::string input_path;
  int k = 0;
  int seed = 20260328;
  int repeat = 20;
  bool spin_adapted = false;
  int spin_multiplicity = 0;
  int ms_twice = std::numeric_limits<int>::min();
};

void print_usage() {
  std::cerr << "usage: benchmark_pf_orbital_eval <input.xmi> "
               "[--k K] [--seed S] [--repeat N] "
               "[--spin-adapted] [--spin-multiplicity mult] [--ms-twice 2Ms]\n";
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
    if (name == "--repeat") {
      options.repeat = std::stoi(value);
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

  if (options.k < 0) {
    throw std::invalid_argument("--k must be >= 0");
  }
  if (options.repeat <= 0) {
    throw std::invalid_argument("--repeat must be positive");
  }
  return options;
}

void print_summary(
    bool spin_adapted,
    int k,
    int primitive_k,
    int n_alpha,
    int n_beta,
    int repeat,
    double reference_energy,
    double sum_total,
    double min_total,
    double max_total,
    double sum_forward,
    double sum_backprop,
    double sum_matrix_bp,
    double sum_two_electron_bp,
    double sum_ao_h1e_bp,
    double sum_orbital_bp,
    double sum_orbital_prepare,
    double sum_ao_h1e_build,
    double sum_active_h1e_build,
    double sum_active_2e_build,
    double sum_pf_matrix_forward,
    double sum_pf_adjoint,
    double loop_wall_time_seconds) {
  std::cout << std::setprecision(12);
  std::cout << "spin_adapted = " << (spin_adapted ? "true" : "false") << '\n';
  std::cout << "k = " << k << '\n';
  if (spin_adapted) {
    std::cout << "primitive_k = " << primitive_k << '\n';
  }
  std::cout << "n_alpha = " << n_alpha << '\n';
  std::cout << "n_beta = " << n_beta << '\n';
  std::cout << "repeat = " << repeat << '\n';
  std::cout << "reference_energy = " << reference_energy << '\n';
  std::cout << "mean_total_dt = " << (sum_total / repeat) << '\n';
  std::cout << "min_total_dt = " << min_total << '\n';
  std::cout << "max_total_dt = " << max_total << '\n';
  std::cout << "mean_forward_dt = " << (sum_forward / repeat) << '\n';
  std::cout << "mean_orbital_prepare_dt = "
            << (sum_orbital_prepare / repeat) << '\n';
  std::cout << "mean_ao_h1e_build_dt = "
            << (sum_ao_h1e_build / repeat) << '\n';
  std::cout << "mean_active_h1e_build_dt = "
            << (sum_active_h1e_build / repeat) << '\n';
  std::cout << "mean_active_2e_build_dt = "
            << (sum_active_2e_build / repeat) << '\n';
  std::cout << "mean_pf_matrix_forward_dt = "
            << (sum_pf_matrix_forward / repeat) << '\n';
  std::cout << "mean_pf_adjoint_dt = " << (sum_pf_adjoint / repeat) << '\n';
  std::cout << "mean_backprop_dt = " << (sum_backprop / repeat) << '\n';
  std::cout << "mean_matrix_backprop_dt = " << (sum_matrix_bp / repeat) << '\n';
  std::cout << "mean_two_electron_backprop_dt = "
            << (sum_two_electron_bp / repeat) << '\n';
  std::cout << "mean_ao_h1e_backprop_dt = " << (sum_ao_h1e_bp / repeat) << '\n';
  std::cout << "mean_orbital_backprop_dt = " << (sum_orbital_bp / repeat) << '\n';
  std::cout << "loop_wall_time_seconds = " << loop_wall_time_seconds << '\n';
}

template <typename Basis>
void accumulate_benchmark(
    const xmvb::vb::CppVbInput& input,
    const Basis& basis,
    double nuclear_repulsion_energy,
    int repeat,
    double* sum_total,
    double* sum_forward,
    double* sum_backprop,
    double* sum_matrix_bp,
    double* sum_two_electron_bp,
    double* sum_ao_h1e_bp,
    double* sum_orbital_bp,
    double* sum_orbital_prepare,
    double* sum_ao_h1e_build,
    double* sum_active_h1e_build,
    double* sum_active_2e_build,
    double* sum_pf_matrix_forward,
    double* sum_pf_adjoint,
    double* min_total,
    double* max_total,
    double* reference_energy,
    double* loop_wall_time_seconds) {
  xmvb::pfaffian_vbscf::PfOrbitalGradEval evaluator;
  if (sum_total == nullptr ||
      sum_forward == nullptr ||
      sum_backprop == nullptr ||
      sum_matrix_bp == nullptr ||
      sum_two_electron_bp == nullptr ||
      sum_ao_h1e_bp == nullptr ||
      sum_orbital_bp == nullptr ||
      sum_orbital_prepare == nullptr ||
      sum_ao_h1e_build == nullptr ||
      sum_active_h1e_build == nullptr ||
      sum_active_2e_build == nullptr ||
      sum_pf_matrix_forward == nullptr ||
      sum_pf_adjoint == nullptr ||
      min_total == nullptr ||
      max_total == nullptr ||
      reference_energy == nullptr ||
      loop_wall_time_seconds == nullptr) {
    throw std::invalid_argument("benchmark accumulators must not be null");
  }

  *sum_total = 0.0;
  *sum_forward = 0.0;
  *sum_backprop = 0.0;
  *sum_matrix_bp = 0.0;
  *sum_two_electron_bp = 0.0;
  *sum_ao_h1e_bp = 0.0;
  *sum_orbital_bp = 0.0;
  *sum_orbital_prepare = 0.0;
  *sum_ao_h1e_build = 0.0;
  *sum_active_h1e_build = 0.0;
  *sum_active_2e_build = 0.0;
  *sum_pf_matrix_forward = 0.0;
  *sum_pf_adjoint = 0.0;
  *min_total = std::numeric_limits<double>::infinity();
  *max_total = 0.0;
  *reference_energy = 0.0;

  const auto started_at = std::chrono::steady_clock::now();
  for (int repeat_index = 0; repeat_index < repeat; ++repeat_index) {
    const auto result = evaluator.eval(input, basis, nuclear_repulsion_energy);
    *reference_energy = result.scf_result.e_tot;
    *sum_total += result.total_dt;
    *sum_forward += result.forward_wall_time_seconds;
    *sum_backprop += result.backprop_wall_time_seconds;
    *sum_matrix_bp += result.matrix_backprop_dt;
    *sum_two_electron_bp += result.two_electron_backprop_dt;
    *sum_ao_h1e_bp += result.ao_h1e_backprop_dt;
    *sum_orbital_bp += result.orbital_backprop_dt;
    *sum_orbital_prepare += result.orbital_prepare_dt;
    *sum_ao_h1e_build += result.ao_h1e_build_dt;
    *sum_active_h1e_build += result.active_h1e_build_dt;
    *sum_active_2e_build += result.active_2e_build_dt;
    *sum_pf_matrix_forward += result.pf_matrix_forward_dt;
    *sum_pf_adjoint += result.pf_adjoint_dt;
    *min_total = std::min(*min_total, result.total_dt);
    *max_total = std::max(*max_total, result.total_dt);
  }
  *loop_wall_time_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started_at).count();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_args(argc, argv);
    const auto load =
        xmvb::vb::load_cpp_vb_input_with_timings(options.input_path);

    if (options.spin_adapted) {
      xmvb::pfaffian_vbscf::PfSpinAdaptedBasisFactoryOptions basis_options;
      basis_options.n_structures = options.k;
      basis_options.seed = options.seed;
      basis_options.target_spin_multiplicity = options.spin_multiplicity;
      basis_options.ms_twice = options.ms_twice;
      const auto basis =
          xmvb::pfaffian_vbscf::build_structure_pf_spin_adapted_basis(
              load.input,
              load.raw_structure_data,
              basis_options);
      double sum_total = 0.0;
      double sum_forward = 0.0;
      double sum_backprop = 0.0;
      double sum_matrix_bp = 0.0;
      double sum_two_electron_bp = 0.0;
      double sum_ao_h1e_bp = 0.0;
      double sum_orbital_bp = 0.0;
      double sum_orbital_prepare = 0.0;
      double sum_ao_h1e_build = 0.0;
      double sum_active_h1e_build = 0.0;
      double sum_active_2e_build = 0.0;
      double sum_pf_matrix_forward = 0.0;
      double sum_pf_adjoint = 0.0;
      double min_total = 0.0;
      double max_total = 0.0;
      double reference_energy = 0.0;
      double loop_wall_time_seconds = 0.0;
      accumulate_benchmark(
          load.input,
          basis,
          load.nuclear_repulsion_energy,
          options.repeat,
          &sum_total,
          &sum_forward,
          &sum_backprop,
          &sum_matrix_bp,
          &sum_two_electron_bp,
          &sum_ao_h1e_bp,
          &sum_orbital_bp,
          &sum_orbital_prepare,
          &sum_ao_h1e_build,
          &sum_active_h1e_build,
          &sum_active_2e_build,
          &sum_pf_matrix_forward,
          &sum_pf_adjoint,
          &min_total,
          &max_total,
          &reference_energy,
          &loop_wall_time_seconds);
      print_summary(
          true,
          basis.n_states,
          basis.primitive_basis.n_states,
          basis.primitive_basis.n_alpha,
          basis.primitive_basis.n_beta,
          options.repeat,
          reference_energy,
          sum_total,
          min_total,
          max_total,
          sum_forward,
          sum_backprop,
          sum_matrix_bp,
          sum_two_electron_bp,
          sum_ao_h1e_bp,
          sum_orbital_bp,
          sum_orbital_prepare,
          sum_ao_h1e_build,
          sum_active_h1e_build,
          sum_active_2e_build,
          sum_pf_matrix_forward,
          sum_pf_adjoint,
          loop_wall_time_seconds);
      return 0;
    }

    xmvb::pfaffian_vbscf::PfBasisFactoryOptions basis_options;
    basis_options.n_states = options.k;
    basis_options.seed = options.seed;
    const auto basis =
        xmvb::pfaffian_vbscf::build_structure_pf_basis(
            load.input,
            load.raw_structure_data,
            basis_options);
    double sum_total = 0.0;
    double sum_forward = 0.0;
    double sum_backprop = 0.0;
    double sum_matrix_bp = 0.0;
    double sum_two_electron_bp = 0.0;
    double sum_ao_h1e_bp = 0.0;
    double sum_orbital_bp = 0.0;
    double sum_orbital_prepare = 0.0;
    double sum_ao_h1e_build = 0.0;
    double sum_active_h1e_build = 0.0;
    double sum_active_2e_build = 0.0;
    double sum_pf_matrix_forward = 0.0;
    double sum_pf_adjoint = 0.0;
    double min_total = 0.0;
    double max_total = 0.0;
    double reference_energy = 0.0;
    double loop_wall_time_seconds = 0.0;
    accumulate_benchmark(
        load.input,
        basis,
        load.nuclear_repulsion_energy,
        options.repeat,
        &sum_total,
        &sum_forward,
        &sum_backprop,
        &sum_matrix_bp,
        &sum_two_electron_bp,
        &sum_ao_h1e_bp,
        &sum_orbital_bp,
        &sum_orbital_prepare,
        &sum_ao_h1e_build,
        &sum_active_h1e_build,
        &sum_active_2e_build,
        &sum_pf_matrix_forward,
        &sum_pf_adjoint,
        &min_total,
        &max_total,
        &reference_energy,
        &loop_wall_time_seconds);
    print_summary(
        false,
        basis.n_states,
        0,
        basis.n_alpha,
        basis.n_beta,
        options.repeat,
        reference_energy,
        sum_total,
        min_total,
        max_total,
        sum_forward,
        sum_backprop,
        sum_matrix_bp,
        sum_two_electron_bp,
        sum_ao_h1e_bp,
        sum_orbital_bp,
        sum_orbital_prepare,
        sum_ao_h1e_build,
        sum_active_h1e_build,
        sum_active_2e_build,
        sum_pf_matrix_forward,
        sum_pf_adjoint,
        loop_wall_time_seconds);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "benchmark_pf_orbital_eval failed: "
              << error.what() << '\n';
    return 1;
  }
}
